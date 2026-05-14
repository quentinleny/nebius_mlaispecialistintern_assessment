# Cluster Validation


#### Purpose

Validate that the slurm cluster is correctly configured for GPU-based distributed training/inference workloads before running the LLM PoC.


#### Environment

- Login host: login-0 / login-1
- Slurm partition: earlytalent
- Compute nodes used: worker-0, worker-1
- GPU allocation tested: 2 nodes x 2 GPUs per node = 4 GPUs total
- GPU type: NVIDIA H200
- Container runtime: Slurm Pyxis + Enroot
- Container image tested: nvcr.io/nvidia/pytorch:24.12-py3


#### Tests

### Test 1: Slurm GPU Allocation

Job: gpu_check


Command summary:

SBATCH -N 2
SBATCH --partition=earlytalent
SBATCH --gres=gpu:2
SBATCH --ntasks-per-node=1
srun hostname
srun nvidia-smi


Result:

- Slurm allocated both nodes: worker-0, worker-1
- Each node exposed 2 NVIDIA H200 GPUs
- CUDA_VISIBLE_DEVICES=0,1 on both nodes
- No GPU processes were running before workload launch


Status: PASSED



## Test 2: Containerized PyTorch CUDA Validation

Job: container_check


Result:

- Pyxis successfully imported the NVIDIA PyTorch container
- PyTorch version: 2.6.0a0+df5bbc09d1.nv24.12
- PyTorch CUDA available: True
- Container CUDA version: 12.6
- Each node saw 2 NVIDIA H200 GPUs from inside the container


Status: PASSED



### Test 3: Multi-node NCCL All-reduce

Job: nccl_check


Configuration:

- 2 nodes
- 2 GPUs per node
- 2 tasks per node
- WORLD_SIZE=4
- Backend: nccl


Validation logic:

Each rank initialized a CUDA tensor equal to rank + 1.

Expected all-reduce sum:

1 + 2 + 3 + 4 = 10

Observed result on all ranks:

all_reduce=10.0 expected=10.0


Ranks observed:

- rank=0, local_rank=0, host worker-0
- rank=1, local_rank=1, host worker-0
- rank=2, local_rank=0, host worker-1
- rank=3, local_rank=1, host worker-1


Slurm accounting:

JobID      JobName     State      Elapsed   ExitCode
223        nccl_check  COMPLETED  00:00:34  0:0


Status: PASSED



#### Validation Summary

The cluster successfully supports:

- 2-node Slurm GPU allocation
- 4-GPU H200 jobs under the earlytalent partition
- Pyxis/Enroot container execution
- NVIDIA PyTorch container execution
- PyTorch CUDA GPU visibility inside the container
- Multi-node NCCL communication across 4 distributed ranks


Overall status: PASSED



#### Implication for PoC

The environment is ready for distributed LLM fine-tuning and evaluation using a containerized PyTorch/Hugging Face workflow.
