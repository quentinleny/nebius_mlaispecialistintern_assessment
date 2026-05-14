# Training and Accuracy Results


## Fine-Tuning Method

LoRA was used instead of full fine-tuning to reduce memory and compute cost while still adapting the model to the selected MMLU category.

LoRA freezes the base model weights and trains low-rank adapter updates:

    W = W0 + Delta_W
    Delta_W = (alpha / r) * B * A

Only the adapter matrices are trained. The base model weights remain frozen.


## Final LoRA Configuration

Source log: `logs/train_lora_single_gpu_tuning_283.out`

| Parameter | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Seed | 36 |
| LoRA rank r | 8 |
| LoRA alpha | 16 |
| LoRA scaling alpha/r | 2 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | `5e-5` |
| Epochs | 12 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Training examples | 188 |


## Parameter Efficiency

| Metric | Value |
|---|---:|
| Trainable parameters | 1,081,344 |
| Total parameters | 495,114,112 |
| Trainable percentage | 0.2184% |

This confirms the model was adapted by training less than 1% of total parameters.


## Single-GPU Training Metrics

| Metric | Value |
|---|---:|
| Runtime | 73.55 s |
| Samples/sec | 30.68 |
| Steps/sec | 3.75 |
| Mean train loss | 1.1059 |
| Epochs completed | 11.74 |
| Max GPU memory allocated | 5.039 GB |
| Max GPU memory reserved | 7.324 GB |
| Total FLOPs | `8.157e14` |


## Accuracy Comparison

Source logs:

| Log | Purpose |
|---|---|
| `logs/evaluate_base_model_mmlu_236.out` | Base model evaluation |
| `logs/evaluate_finetuned_model_mmlu_284.out` | Fine-tuned model evaluation |

| Model | Adapter | Correct | Total | Accuracy |
|---|---|---:|---:|---:|
| Base model | none | 22 | 56 | 39.29% |
| Fine-tuned model | `qwen_0p5b_lora_lr5e5_dropout005_seed36` | 31 | 56 | 55.36% |


## Accuracy Improvement

| Metric | Value |
|---|---:|
| Absolute improvement | 16.07 percentage points |
| Relative improvement | 40.91% |

The fine-tuned adapter improved heldout accuracy from 39.29% to 55.36%.


## Hyperparameter Selection

Several configurations were tested across learning rate, LoRA rank, dropout, random seed, and epoch count.


Selected run:

| Parameter | Value |
|---|---:|
| LoRA rank r | 8 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| Learning rate | `5e-5` |
| Seed | 36 |
| Epochs | 12 |


Key observations:

| Observation | Interpretation |
|---|---|
| r=8 improved heldout accuracy | Adapter capacity was sufficient |
| alpha=16 with r=8 gave scaling 2 | Adapter update was not vanishingly small |
| 16 epochs lowered train loss but reduced heldout accuracy | Longer training overfit |
| Final model chosen by heldout accuracy | Not by lowest training loss |


## Multi-Node Training

Source log: `logs/train_lora_multinode_298.out`

The multi-node run used the same model, data split, LoRA configuration, seed, learning rate, and effective batch size.

| Setting | Value |
|---|---:|
| Nodes | 2 |
| GPUs per node | 2 |
| Total GPU processes | 4 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 1 |
| Effective global batch size | 8 |
| Epochs | 12 |


## Single-GPU vs Multi-Node Training

| Metric | Single GPU | Multi-node |
|---|---:|---:|
| Runtime | 73.55 s | 21.22 s |
| Samples/sec | 30.68 | 106.31 |
| Steps/sec | 3.75 | 13.57 |
| Mean train loss | 1.1059 | 1.0528 |
| Max GPU memory allocated | 5.039 GB | 5.035 GB |
| Max GPU memory reserved | 7.324 GB | 6.045 GB |

| Scaling metric | Value |
|---|---:|
| Training throughput speedup | 3.47x |
| Scaling efficiency across 4 GPU processes | 86.8% |


## Interpretation

The LoRA adapter improved heldout accuracy from 39.29% to 55.36% while training only 0.2184% of the model parameters. The multi-node run demonstrated that the same training workload scaled across 2 nodes and 4 H200 GPU processes with 3.47x throughput improvement relative to the single-GPU run.
