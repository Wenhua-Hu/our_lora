#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=glora
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:40:00
#SBATCH --output=../logs/glora_60m_%A.out

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
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank $rank \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer our_lora

# Calculate the duration on execution
END_TIME=`date`; echo ">>> END: $END_TIME"
time_elapsed=`date -ud@$(($(date -ud"$END_TIME" +%s)-$(date -ud"$START_TIME" +%s))) +%T`; echo ">>> Job takes: $time_elapsed"
