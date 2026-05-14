#!/bin/bash


# 03a_install_hf_packages.sbatch

#SBATCH --job-name=install_hf_packages
#SBATCH -N 1
#SBATCH --partition=earlytalent
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/install_hf_packages_%j.out

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"

srun \
  --container-image=nvcr.io/nvidia/pytorch:24.12-py3 \
  --container-mount-home \
  bash -lc '
    set -e

    rm -rf /home/qln/nebius-poc/hf_home
    mkdir -p /home/qln/nebius-poc/hf_home

    export PYTHONUSERBASE=/home/qln/nebius-poc/hf_home
    export PATH=$PYTHONUSERBASE/bin:$PATH
    export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

    echo "Base torch check before install:"
    python - <<PY
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

    echo "Installing HF stack without replacing torch..."

    python -m pip install --user \
      "transformers==4.46.3" \
      "tokenizers==0.20.3" \
      "huggingface-hub==0.26.5" \
      "safetensors==0.4.5" \
      "datasets==3.2.0" \
      "evaluate==0.4.3" \
      "peft==0.14.0" \
      "accelerate==1.2.1" \
      "sentencepiece==0.2.0" \
      "protobuf<5" \
      "pyarrow==17.0.0" \
      "dill==0.3.8" \
      "multiprocess==0.70.16" \
      "fsspec==2024.9.0" \
      "xxhash==3.5.0" \
      --no-deps

    echo "Verifying installs..."

    python - <<PY
packages = ["torch", "transformers", "datasets", "accelerate", "peft", "evaluate", "sentencepiece", "pyarrow"]
for p in packages:
    mod = __import__(p)
    print(p, "OK", getattr(mod, "__version__", "version_unknown"))

import torch
print("cuda_available:", torch.cuda.is_available())
print("torch_version:", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

    echo "HF package install complete."
  '
