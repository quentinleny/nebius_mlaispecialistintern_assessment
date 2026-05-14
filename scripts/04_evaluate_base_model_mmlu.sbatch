#!/bin/bash


# 04_evaluate_base_model_mmlu.sbatch

#SBATCH --job-name=eval_base_mmlu
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/evaluate_base_model_mmlu_%j.out

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

    python /home/qln/nebius-poc/scripts/evaluate_mmlu_accuracy.py \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --category high_school_statistics \
      --split test \
      --start_index 160 \
      --end_index 216 \
      --output_csv /home/qln/nebius-poc/results/base_model_mmlu_heldout_accuracy.csv
  '
