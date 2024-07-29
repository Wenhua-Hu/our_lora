#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=adamw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:59:00
#SBATCH --output=../logs/time_adamw_original_60m_%A.out

# module purge
# module load 2023
# module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch
cd ../../GaLore
# Activate your environment
source activate galore14
# Run your code
echo "Running experiment on galore..."
START_TIME=`date`; echo ">>> START: $START_TIME"

# Check whether the GPU is available
srun python -uc "import torch; print('>>> GPU available?', torch.cuda.is_available())"
rank=128
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --galore_scale 0.25 \
    --rank $rank \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --proj_type std \
    --ema_beta 0.9  \
    --optimizer adamw_tf

# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"
