import argparse
import json
import os
import random
import time

import bitsandbytes as bnb
import datasets
import datasets.distributed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import transformers
import wandb
from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit, AdamW_AB
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from peft_pretraining import args_utils, training_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from tqdm import tqdm
from transformers import AdamW, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

wandb.login(key="9cd3afc6437b77dbdecdad14b9badbc9a51a95f7")
transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--ema_beta", type=float, default=0.0)

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(
    model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size
):
    _time = time.time()
    val_data = datasets.load_dataset(
        "c4", "en", split="validation", streaming=True
    )  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=global_rank, world_size=world_size
        )

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # for module_name, module in model.named_modules():
    for param_name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Our trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):
    temp_a_all = 0
    temp_b_all = 0
    temp_c_all = 0
    temp_d_all = 0
    temp_e_all = 0
    temp_f_all = 0
    temp_g_all = 0
    temp_i_all = 0
    temp_j_all = 0
    temp_x_all = 0
    temp_y_all = 0
    temp_z_all = 0
    # torch.cuda.synchronize()
    temp_a = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="galore-c4")
        # wandb.init(mode="disabled")

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=global_rank,
            world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )

    # Add lora config
    task_type = TaskType.CAUSAL_LM
    
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    
    print(task_type, target_modules)
    
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=args.rank,
        # scaling should be 1 in order to keep same gradient as same as W
        lora_alpha=args.rank,  # scaling =  lora_alpha / r = 1 y = (w +ba*scaling)X
        lora_dropout=0,
        target_modules=target_modules,
        init_lora_weights=True,  # B = 0 
    )
    
    lora_target_keys = ["base_layer", "lora_A", "lora_B"]

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    # Add Lora
    model = get_peft_model(model, lora_config)
    


    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    training_time = 0
    a = 0

    optimizer_spend_time = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )
        else:
            logger.warning(
                f"Did not find training state in {args.continue_from}, global step will start from zero"
            )
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    # print_trainable_parameters(model)
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    print(f"args.optimizer.lower(): {args.optimizer.lower()}")
    if "our_lora" in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        glore_params_names = []
        # target_modules_list = ["attn", "mlp"]
        target_modules_list = target_modules

        for module_name, module in model.named_modules():
            print(f"All weights in module: ", module_name)
  
    
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"enable {args.optimizer.lower()} for weights in module: ", module_name)
            galore_params.append(module.weight)
            glore_params_names.append(module_name)
            # print(f"add {module_name}")
        
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [
            p for p in model.parameters() if id(p) not in id_galore_params
        ]
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": args.rank,
                # "update_proj_gap": args.update_proj_gap,
                "scale": args.galore_scale,
                # "proj_type": args.proj_type,
                # "ema_beta": args.ema_beta,
                "lora_target_keys": lora_target_keys,
                "glore_params_names": glore_params_names
            },
        ]
    else:
        print("not expected")

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    if "our_lora" in args.optimizer.lower():
        logger.info(
            f"Total params with Our Lore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M"
        )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adamw_tf":
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        
    elif args.optimizer.lower() == "our_lora":
        optimizer = AdamW_AB(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            glore_params_names=glore_params_names,   # list of params names (detailed name)
            lora_target_keys= lora_target_keys       # ["base_layer", "lora_A", "lora_B"]
        )
    elif args.optimizer.lower() == "galore_adamw":
        optimizer = GaLoreAdamW(
            param_groups, #trainable_params
            lr=args.lr,
            weight_decay=args.weight_decay,
            galore_params_name=galore_params_name,
        )

    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
        )
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "galore_adamw8bit_per_layer":
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": args.rank,
                                "update_proj_gap": args.update_proj_gap * 2,
                                "scale": args.galore_scale,
                                "proj_type": args.proj_type,
                            }
                        ],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=args.lr, weight_decay=args.weight_decay
                    )

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    # torch.cuda.synchronize()
    temp_a_all += time.time() - temp_a
    # print(f">>> temp_aa {temp_a_all}")

    for batch_idx, batch in enumerate(dataloader):
        # torch.cuda.synchronize()
        record_training_time = time.time()
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step

        # add grad clipping
        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            A = time.time()
            optimizer.step()

            B = time.time() - A
            optimizer_spend_time += B

            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        training_time += time.time() - record_training_time
        temp_b = time.time()
        temp_e = time.time()
        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            if args.single_gpu:
                model.save_pretrained(current_model_directory, max_shard_size="100GB")
            else:
                model.module.save_pretrained(
                    current_model_directory, max_shard_size="100GB"
                )

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)
        # torch.cuda.synchronize()
        temp_e_all += time.time() - temp_e

        # evaluation
        temp_d = time.time()
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            # torch.cuda.synchronize()
            temp_f = time.time()
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
            )
            # torch.cuda.synchronize()
            temp_f_all += time.time() - temp_f
            temp_g = time.time()
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
            # torch.cuda.synchronize()
            temp_g_all += time.time() - temp_g

        # torch.cuda.synchronize()
        temp_j = time.time()
        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size
        # torch.cuda.synchronize()
        temp_j_all += time.time() - temp_j
        temp_i = time.time()
        if global_rank == 0:
            a += 1
            # torch.cuda.synchronize()
            temp_x = time.time()
            loss_ = loss.item()
            temp_y_all += time.time() - temp_x
            update_step_ = update_step
            tokens_seen_ = tokens_seen
            throughput_tokens_ = tokens_in_update / update_time
            throughput_examples_ = args.total_batch_size / update_time
            throughput_batches_ = batches_in_update / update_time
            temp_x_all += time.time() - temp_x
            # torch.cuda.synchronize()
            temp_z = time.time()
            wandb.log(
                {
                    "loss": loss_,
                    "lr": lr,
                    "update_step": update_step_,
                    "tokens_seen": tokens_seen_,
                    "throughput_tokens": throughput_tokens_,
                    "throughput_examples": throughput_examples_,
                    "throughput_batches": throughput_batches_,
                },
                step=global_step,
            )
            # torch.cuda.synchronize()
            temp_z_all += time.time() - temp_z
        # torch.cuda.synchronize()
        temp_i_all += time.time() - temp_i
        # print(f">>>>>>> {a} : {time.time() - temp_i}")
        update_time = time.time()
        temp_b_all += time.time() - temp_b
        temp_d_all += time.time() - temp_d
    # print(f">>> temp_b_all {temp_b_all}")
    # print(f">>> temp_d_all {temp_d_all}")
    # print(f">>> temp_e_all {temp_e_all}")
    # print(f">>> temp_f_all {temp_f_all}")
    # print(f">>> temp_g_all {temp_g_all}")
    # print(f">>> temp_i_all {temp_i_all}")
    # print(f">>> temp_j_all {temp_j_all}")
    # print(f">>> temp_x_all {temp_x_all}")
    # print(f">>> temp_y_all {temp_y_all}")
    # print(f">>> temp_z_all {temp_z_all}")
    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    # torch.cuda.synchronize()
    temp_c = time.time()
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)
        if args.single_gpu:
            model.save_pretrained(current_model_directory)
        else:
            model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
    )

    if global_rank == 0:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")



if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)