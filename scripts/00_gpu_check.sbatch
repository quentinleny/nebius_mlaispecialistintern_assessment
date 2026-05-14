#!/bin/bash


# 00_gpu_check.sbatch

#SBATCH --job-name=gpu_check
#SBATCH -N 2
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_check_%j.out

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"

echo "Hostnames:"
srun hostname

echo "GPU status:"
srun nvidia-smi

echo "CUDA visible devices:"
srun bash -lc 'echo "$(hostname): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
