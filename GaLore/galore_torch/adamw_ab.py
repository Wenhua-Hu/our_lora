# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector

from collections import OrderedDict, defaultdict


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        glore_params_names=None,
        lora_target_keys=None
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.layers = defaultdict(lambda: defaultdict(dict))
        self.state_layers = defaultdict(dict)
        self.key_names = lora_target_keys
        self.glore_params_names = glore_params_names
        # print(f"dddd   {self.glore_params_names}")
        self.lora_layers = [key_name for key_name in self.key_names if "lora_" in key_name]
        base_layers = [key_name for key_name in self.key_names if "lora_" not in key_name]
        if len(base_layers) != 1:
            raise RuntimeError("Only allow one base layer !")
        else:
            self.base_layer = base_layers[0]

            
    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        # base_model.model.model.layers.0.self_attn.q_proj.base_layer  
        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default  
        # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if "rank" not in group:
                continue
                
            for p, p_name in zip(group["params"], self.glore_params_names):
                if p.grad is not None:
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                component_names = p_name.split(".")

                for key_name in self.key_names:
                    if key_name not in component_names:
                        continue
                    # given the index of key_name(e.g. base_layer) to back index to the position of block name and Linear name
                    idx_key_name = component_names.index(key_name)
                    block = int(component_names[idx_key_name - 3])
                    nn_name = component_names[idx_key_name - 1]
                    self.layers[block][nn_name][key_name] = p
                    
        for group in self.param_groups:
            glore_params_name = None
            for i, p in enumerate(group["params"]):
                if "rank" in group:
                    glore_params_name = self.glore_params_names[i]
                    component_names = glore_params_name.split(".")
                    # print(component_names)
                    # print(self.base_layer)
                    if self.base_layer not in component_names:
                        continue
                    idx_base_layer = component_names.index(self.base_layer)
                    block = int(component_names[idx_base_layer - 3])
                    nn_name = component_names[idx_base_layer - 1]
                    
                    nn_data = self.layers[block][nn_name]
                    #############################################
                    #     method 1： scalable solution          #                                 
                    #############################################
                    
                    # p = nn_data[self.base_layer]
                    # p_lora_A_p, grad_lora_A = nn_data[lora_name].data, nn_data[lora_name].grad
                    # p_lora_B, grad_lora_B = nn_data["lora_B"].data, nn_data["lora_B"].grad
                    # grad = torch.zeros_like(p.data)
                    # e.g. grad = p_lora_B @ grad_lora_A + grad_lora_B @ p_lora_A_p
                    # for p_i in range(len(self.lora_layers)):
                    #     partial = None
                    #     # e.g. partial = grad_lora_B @ p_lora_A_p
                    #     for i in range(len(self.lora_layers)-1, -1, -1):
                    #         lora_layer_data = nn_data[self.lora_layers[i]].grad if i == j else nn_data[self.lora_layers[i]].data 
                    #         partial = lora_layer_data if partial is None else partial @ lora_layer_data
                    #     grad += partial
                    
                    ###############################################
                    #    method 2：hardcoded when only 2 matrices  #                                 
                    ###############################################
                    p_lora_A_p, grad_lora_A = nn_data["lora_A"].data, nn_data["lora_A"].grad
                    p_lora_B, grad_lora_B = nn_data["lora_B"].data, nn_data["lora_B"].grad
                    grad = p_lora_B @ grad_lora_A + grad_lora_B @ p_lora_A_p
                        
                else:
                    # print(f"*********************************4***********************************")  
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                            
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                    
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # scaling constant
                # self.scaling = self.lora_alpha / self.r
                
                # lora scaling ?
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss