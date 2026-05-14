# Nebius PoC Summary


## Objective

This PoC validates a Slurm-based GPU workflow for fine-tuning and serving an open-source LLM on Nebius-style infrastructure.

The workflow covers:

- Cluster/container validation
- Multi-node GPU communication validation
- LoRA fine-tuning of an open-source LLM
- Accuracy comparison against the base model
- Inference throughput optimization


## Environment

| Component | Value |
|---|---|
| Scheduler | Slurm |
| Container runtime | Enroot/Pyxis through `srun --container-image` |
| Container image | `nvcr.io/nvidia/pytorch:24.12-py3` |
| GPU | NVIDIA H200 |
| Nodes used | 2 |
| GPUs per node | 2 |
| Total GPUs used for multi-node training | 4 |
| Python | 3.12.3 |
| PyTorch | `2.6.0a0+df5bbc09d1.nv24.12` |
| CUDA available in PyTorch | Yes |


## Cluster Validation

Before model training, the cluster was validated with three checks.

| Check | Result |
|---|---|
| GPU visibility | H200 GPUs visible through `nvidia-smi` |
| Container launch | PyTorch container launched successfully |
| PyTorch CUDA | `torch.cuda.is_available() = True` |
| Multi-node NCCL | 4-rank all-reduce completed correctly |
| NCCL expected value | 10.0 |
| NCCL observed value | 10.0 on all ranks |

The validation confirmed that the containerized environment could access GPUs and run distributed communication across 2 nodes.


## Model and Dataset

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `cais/mmlu` |
| Category | `high_school_statistics` |
| Task | Multiple-choice answer selection |
| Fine-tuning method | LoRA |
| Final adapter | `results/qwen_0p5b_lora_lr5e5_dropout005_seed36` |


## Dataset Split

Official split sizes:

| Split | Examples |
|---|---:|
| dev | 5 |
| validation | 23 |
| test | 216 |

Custom split used for this PoC:

| Purpose | Data | Examples |
|---|---|---:|
| Training | `dev + validation + test[0:160]` | 188 |
| Heldout evaluation | `test[160:216]` | 56 |

The heldout set was not used during training.


## Evaluation Method

The model was evaluated using next-token multiple-choice scoring.

For each prompt, the model scored only the logits for the next answer token:

| Choice | Token ID |
|---|---:|
| A | 362 |
| B | 425 |
| C | 356 |
| D | 422 |

The predicted answer was the highest-logit choice among A, B, C, and D. This was chosen over free-form generation because it is deterministic, fast, and avoids parsing generated explanations.


## Final Results Overview

| Metric | Result |
|---|---:|
| Baseline heldout accuracy | 39.29% |
| Fine-tuned heldout accuracy | 55.36% |
| Absolute accuracy gain | 16.07 percentage points |
| Relative accuracy gain | 40.91% |
| Single-GPU training throughput | 30.68 samples/sec |
| Multi-node training throughput | 106.31 samples/sec |
| Multi-node training speedup | 3.47x |
| Multi-node scaling efficiency | 86.8% |
| Batch-1 inference throughput | 25.00 examples/sec |
| Optimized inference throughput | 843.19 examples/sec |
| Inference throughput speedup | 33.73x |
