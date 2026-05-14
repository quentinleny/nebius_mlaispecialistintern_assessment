# Nebius LLM Fine-Tuning PoC

This repository contains a Slurm-based proof of concept for validating a GPU cluster, fine-tuning an open-source LLM on MMLU, evaluating accuracy, and benchmarking inference throughput.

## Structure

| Path | Description |

|---|---|

| `slurm/` | Slurm job files for cluster checks, package setup, training, evaluation, and throughput benchmarking |

| `scripts/` | Python scripts used by the Slurm jobs |

| `results/markdown/` | Final documentation and summary reports |

| `results/*.csv` | Accuracy and throughput result CSVs |

| `logs/final/` | Curated final Slurm output logs |

## How to Reproduce

Start with:

`results/markdown/00_slurm_job_subs.md`

Then review:

1. `results/markdown/01_poc_summary.md`

2. `results/markdown/02_training_accuracy.md`

3. `results/markdown/03_throughput_and_caveats.md`

The Slurm jobs should be run from the repository root.
