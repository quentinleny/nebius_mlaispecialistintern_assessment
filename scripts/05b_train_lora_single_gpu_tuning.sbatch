#!/bin/bash


# 05b_train_lora_single_gpu_tuning.sbatch

#SBATCH --job-name=train_lora_sweep
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_lora_single_gpu_tuning_%j.out

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"

srun \
  --container-image=nvcr.io/nvidia/pytorch:24.12-py3 \
  --container-mount-home \
  bash -lc '
    set -e

    export PYTHONUSERBASE=/home/qln/nebius-poc/hf_home
    export PATH=$PYTHONUSERBASE/bin:$PATH
    export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH
    export HF_HOME=/home/qln/nebius-poc/hf_cache
    export HF_DATASETS_CACHE=/home/qln/nebius-poc/hf_cache/datasets

    python /home/qln/nebius-poc/scripts/train_lora_on_mmlu.py \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --category high_school_statistics \
      --output_dir /home/qln/nebius-poc/results/qwen_0p5b_lora_lr5e5_dropout005_seed36 \
      --seed 36 \
      --num_train_epochs 12 \
      --learning_rate 5e-5 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --train_test_examples 160 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05
  '