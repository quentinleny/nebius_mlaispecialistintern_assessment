import argparse
import csv
import os
import socket
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ANSWER_LABELS = ["A", "B", "C", "D"]


def format_prompt(example):
    choices = example["choices"]
    return (
        "Answer the following multiple choice question. "
        "Return only the letter A, B, C, or D.\n\n"
        f"Question: {example['question']}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )


def get_choice_token_ids(tokenizer):
    token_ids = []
    for label in ANSWER_LABELS:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {label} does not map to one token: {ids}")
        token_ids.append(ids[0])
    return token_ids


def load_model(model_name, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def run_batch(model, tokenizer, prompts, choice_token_ids, max_length):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    choice_logits = logits[:, choice_token_ids]
    preds = torch.argmax(choice_logits, dim=1)

    return preds


def benchmark(model, tokenizer, prompts, choice_token_ids, batch_size, repeats, warmup, max_length):
    total_examples = 0

    for _ in range(warmup):
        batch = prompts[:batch_size]
        run_batch(model, tokenizer, batch, choice_token_ids, max_length)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for _ in range(repeats):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            run_batch(model, tokenizer, batch, choice_token_ids, max_length)
            total_examples += len(batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    runtime_s = end - start
    throughput = total_examples / runtime_s

    max_allocated_gb = None
    max_reserved_gb = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3

    return {
        "batch_size": batch_size,
        "total_examples": total_examples,
        "runtime_s": runtime_s,
        "examples_per_second": throughput,
        "max_gpu_memory_allocated_gb": max_allocated_gb,
        "max_gpu_memory_reserved_gb": max_reserved_gb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", default="/home/qln/nebius-poc/results/qwen_0p5b_lora_lr5e5_dropout005_seed36")
    parser.add_argument("--category", default="high_school_statistics")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_index", type=int, default=160)
    parser.add_argument("--end_index", type=int, default=216)
    parser.add_argument("--batch_sizes", default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_csv", default="/home/qln/nebius-poc/results/inference_throughput_benchmark.csv")
    args = parser.parse_args()

    print("Environment")
    print(f"host: {socket.gethostname()}")
    print(f"model_name: {args.model_name}")
    print(f"adapter_dir: {args.adapter_dir}")
    print(f"category: {args.category}")
    print(f"heldout: {args.split}[{args.start_index}:{args.end_index}]")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu_count: {torch.cuda.device_count()}")
        print(f"gpu_0: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model(args.model_name, args.adapter_dir)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print("choice_token_ids:", dict(zip(ANSWER_LABELS, choice_token_ids)))

    dataset = load_dataset("cais/mmlu", args.category)[args.split]
    dataset = dataset.select(range(args.start_index, args.end_index))
    prompts = [format_prompt(example) for example in dataset]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    rows = []
    for batch_size in batch_sizes:
        result = benchmark(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            choice_token_ids=choice_token_ids,
            batch_size=batch_size,
            repeats=args.repeats,
            warmup=args.warmup,
            max_length=args.max_length,
        )
        rows.append(result)

        print(
            f"batch_size={result['batch_size']} | "
            f"examples_per_s={result['examples_per_second']:.2f} | "
            f"runtime_s={result['runtime_s']:.2f} | "
            f"max_allocated_gb={result['max_gpu_memory_allocated_gb']:.3f} | "
            f"max_reserved_gb={result['max_gpu_memory_reserved_gb']:.3f}"
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_results: {args.output_csv}")
    print("Inference throughput benchmark complete.")


if __name__ == "__main__":
    main()
