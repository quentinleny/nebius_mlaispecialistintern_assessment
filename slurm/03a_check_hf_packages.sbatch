#!/bin/bash


# 03a_check_hf_packages.sbatch

#SBATCH --job-name=check_hf_packages
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=logs/check_hf_packages_%j.out

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"

srun \
  --container-image=nvcr.io/nvidia/pytorch:24.12-py3 \
  --container-mount-home \
  bash -lc '
    echo "Python:"
    python --version

    echo "Torch:"
    python - <<PY
import torch
print(torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

    echo "Package availability:"
    python - <<PY
packages = ["transformers", "datasets", "accelerate", "peft", "evaluate", "sentencepiece", "bitsandbytes"]
for p in packages:
    try:
        mod = __import__(p)
        print(p, "OK", getattr(mod, "__version__", "version_unknown"))
    except Exception as e:
        print(p, "MISSING", repr(e))
PY
  '
