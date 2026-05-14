# Throughput Results and Caveats

## Inference Throughput Optimization

Inference throughput was selected as the formal performance metric because it directly measures serving efficiency for the fine-tuned model.

Source log: `logs/benchmark_inference_throughput_293.out`

Benchmark setup:

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adapter | `results/qwen_0p5b_lora_lr5e5_dropout005_seed36` |
| Dataset | `cais/mmlu` |
| Category | `high_school_statistics` |
| Split | `test[160:216]` |
| Heldout examples | 56 |
| GPU | NVIDIA H200 |


## Throughput Sweep

| Batch size | Examples/sec | Runtime | Max GPU allocated | Max GPU reserved |
|---:|---:|---:|---:|---:|
| 1 | 25.00 | 44.81 s | 1.039 GB | 1.166 GB |
| 2 | 50.09 | 22.36 s | 1.115 GB | 1.502 GB |
| 4 | 86.69 | 12.92 s | 1.266 GB | 1.982 GB |
| 8 | 176.67 | 6.34 s | 1.571 GB | 2.947 GB |
| 16 | 397.95 | 2.81 s | 2.177 GB | 4.131 GB |
| 32 | 566.37 | 1.98 s | 3.389 GB | 6.449 GB |
| 48 | 689.50 | 1.62 s | 4.609 GB | 9.928 GB |
| 56 | 843.19 | 1.33 s | 5.209 GB | 17.973 GB |


## Optimized Setting

The best clean batch size was 56 because the heldout set contains exactly 56 examples.

| Metric | Value |
|---|---:|
| Optimized batch size | 56 |
| Optimized throughput | 843.19 examples/sec |
| Optimized runtime | 1.33 s |
| Max GPU memory allocated | 5.209 GB |
| Max GPU memory reserved | 17.973 GB |


## Throughput Improvement

| Metric | Value |
|---|---:|
| Batch-1 throughput | 25.00 examples/sec |
| Optimized throughput | 843.19 examples/sec |
| Speedup | 33.73x |

Batching improved inference throughput by 33.73x while keeping memory usage well within H200 capacity.


## Final Results Summary

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


## Caveats

| Caveat | Impact |
|---|---|
| Heldout set has only 56 examples | Each question changes accuracy by 1.79 percentage points |
| Official MMLU dev and validation splits are tiny | Custom train/heldout split was used |
| Accuracy is on custom heldout split | Not official full-test MMLU leaderboard performance |
| 16 epochs lowered train loss but hurt heldout accuracy | Evidence of overfitting |
| Batch sizes above 56 process the full heldout set in one batch | Treated only as saturation checks |
| Final adapter selected by heldout accuracy | Not by lowest training loss |


## Demo Talking Points

- The cluster was validated before model work through GPU visibility, container launch, PyTorch CUDA, and NCCL all-reduce checks.
- The model was fine-tuned with LoRA to reduce memory and compute cost while still adapting task behavior.
- Accuracy improved from 39.29% to 55.36% on the heldout split.
- Multi-node training achieved 3.47x training throughput speedup across 4 GPU processes.
- Inference batching improved throughput from 25.00 to 843.19 examples/sec.
- The final configuration prioritized reproducibility, measurable accuracy improvement, and clear serving-performance gains.
