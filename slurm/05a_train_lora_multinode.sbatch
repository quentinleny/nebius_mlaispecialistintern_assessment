#!/bin/bash


# 05a_train_lora_multinode.sbatch

#SBATCH --job-name=train_lora_mmlu
#SBATCH -N 2
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_lora_multinode_%j.out

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

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

    export NCCL_DEBUG=WARN
    export OMP_NUM_THREADS=8

    NODE_RANK=$SLURM_NODEID

    torchrun \
      --nnodes=2 \
      --nproc_per_node=2 \
      --node_rank=$NODE_RANK \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
      /home/qln/nebius-poc/scripts/train_lora_on_mmlu.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --category high_school_statistics \
        --output_dir /home/qln/nebius-poc/results/qwen_0p5b_lora_lr5e5_dropout005_seed36_multinode \
        --seed 36 \
        --num_train_epochs 12 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --train_test_examples 160 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05
  '
