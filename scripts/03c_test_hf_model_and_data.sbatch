#!/bin/bash


# 03b_test_hf_model_and_data.sbatch

#SBATCH --job-name=test_hf_model_data
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_hf_model_and_data_%j.out

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
    export TRANSFORMERS_CACHE=/home/qln/nebius-poc/hf_cache
    export HF_DATASETS_CACHE=/home/qln/nebius-poc/hf_cache/datasets

    mkdir -p "$HF_HOME"

    python /home/qln/nebius-poc/scripts/test_hf_model_and_data.py
  '
