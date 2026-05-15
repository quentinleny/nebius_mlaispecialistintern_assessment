# train_lora_on_mmlu.py

import argparse
import os
import socket
import time

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, PrinterCallback, ProgressCallback, set_seed
from peft import LoraConfig, get_peft_model, TaskType

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


def format_answer(example):
    return " " + ANSWER_LABELS[int(example["answer"])]


def print_environment(args):
    print("Environment")
    print(f"host: {socket.gethostname()}")
    print(f"python_userbase: {os.environ.get('PYTHONUSERBASE')}")
    print(f"model_name: {args.model_name}")
    print(f"category: {args.category}")
    print(f"output_dir: {args.output_dir}")
    print(f"seed: {args.seed}")
    print(f"lora_r: {args.lora_r}")
    print(f"lora_alpha: {args.lora_alpha}")
    print(f"lora_dropout: {args.lora_dropout}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_train_epochs: {args.num_train_epochs}")
    print(f"per_device_train_batch_size: {args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"train_test_examples: {args.train_test_examples}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu_count: {torch.cuda.device_count()}")
        print(f"gpu_0: {torch.cuda.get_device_name(0)}")


def build_train_dataset(category, train_test_examples):
    dataset = load_dataset("cais/mmlu", category)

    test_train = dataset["test"].select(range(train_test_examples))

    train_dataset = concatenate_datasets([
        dataset["dev"],
        dataset["validation"],
        test_train,
    ])

    heldout_start = train_test_examples
    heldout_end = len(dataset["test"])

    if is_main_process():
        print("Dataset")
        print(dataset)
        print(f"dev_examples: {len(dataset['dev'])}")
        print(f"validation_examples: {len(dataset['validation'])}")
        print(f"test_examples_used_for_training: {len(test_train)}")
        print(f"total_train_examples: {len(train_dataset)}")
        print(f"heldout_test_start: {heldout_start}")
        print(f"heldout_test_end: {heldout_end}")
        print(f"heldout_test_examples: {heldout_end - heldout_start}")

        os.makedirs("results", exist_ok=True)
        with open("results/train_eval_split.txt", "w") as f:
            f.write(f"category={category}\n")
            f.write(f"train=dev + validation + test[0:{train_test_examples}]\n")
            f.write(f"heldout_eval=test[{heldout_start}:{heldout_end}]\n")
            f.write(f"heldout_start={heldout_start}\n")
            f.write(f"heldout_end={heldout_end}\n")

        print("formatted_training_example:")
        print(format_prompt(train_dataset[0]) + format_answer(train_dataset[0]))

    return train_dataset


def tokenize_training_example(example, tokenizer, max_length):
    prompt = format_prompt(example)
    answer = format_answer(example)
    full_text = prompt + answer + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))

    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_batch(features, tokenizer):
    max_len = max(len(x["input_ids"]) for x in features)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for x in features:
        pad_len = max_len - len(x["input_ids"])

        batch_input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        batch_attention_mask.append(x["attention_mask"] + [0] * pad_len)
        batch_labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def build_lora_model(model_name, lora_r, lora_alpha, lora_dropout):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()

    return model, tokenizer


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def print_gpu_memory(prefix):
    if not is_main_process():
        return

    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3

    print(f"{prefix}_max_gpu_memory_allocated_gb: {allocated_gb:.3f}")
    print(f"{prefix}_max_gpu_memory_reserved_gb: {reserved_gb:.3f}")


class CleanLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        pieces = []

        if "loss" in logs:
            pieces.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            pieces.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "grad_norm" in logs:
            pieces.append(f"grad_norm={logs['grad_norm']:.3f}")
        if "learning_rate" in logs:
            pieces.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            pieces.append(f"epoch={logs['epoch']:.2f}")

        if "train_runtime" in logs:
            pieces.append(f"runtime_s={logs['train_runtime']:.2f}")
        if "train_samples_per_second" in logs:
            pieces.append(f"samples_per_s={logs['train_samples_per_second']:.2f}")
        if "train_steps_per_second" in logs:
            pieces.append(f"steps_per_s={logs['train_steps_per_second']:.2f}")
        if "train_loss" in logs:
            pieces.append(f"train_loss={logs['train_loss']:.4f}")

        if pieces and is_main_process():
            print(" | ".join(pieces))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--category", default="high_school_statistics")
    parser.add_argument("--output_dir", default="/home/qln/nebius-poc/results/qwen_0p5b_lora_hs_statistics")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_test_examples", type=int, default=160)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if is_main_process():
        print_environment(args)

    raw_train = build_train_dataset(args.category, args.train_test_examples)
    model, tokenizer = build_lora_model(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout)

    tokenized_train = raw_train.map(
        lambda x: tokenize_training_example(x, tokenizer, args.max_length),
        remove_columns=raw_train.column_names,
    )

    if is_main_process():
        print(f"tokenized_train_examples: {len(tokenized_train)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=False,
        bf16=False,
        logging_steps=5,
        disable_tqdm=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=lambda features: collate_batch(features, tokenizer),
    )

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CleanLoggingCallback)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if is_main_process():
        print("Starting training")
    start = time.time()
    train_result = trainer.train()
    end = time.time()

    print_gpu_memory("training")
    if is_main_process():
        print("final_training_summary:")
        metrics = train_result.metrics

        if "train_runtime" in metrics:
            print(f"total_runtime_s: {metrics['train_runtime']:.2f}")
        if "train_samples_per_second" in metrics:
            print(f"train_samples_per_s: {metrics['train_samples_per_second']:.2f}")
        if "train_steps_per_second" in metrics:
            print(f"train_steps_per_s: {metrics['train_steps_per_second']:.2f}")
        if "train_loss" in metrics:
            print(f"mean_train_loss: {metrics['train_loss']:.4f}")
        if "epoch" in metrics:
            print(f"epochs_completed: {metrics['epoch']:.2f}")
        if "total_flos" in metrics:
            print(f"total_flos: {metrics['total_flos']:.3e}")

    if is_main_process():
        print("Saving LoRA adapter")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print(f"saved_adapter: {args.output_dir}")
        print(f"train_wall_seconds: {end - start:.2f}")
        print("LoRA training complete.")


if __name__ == "__main__":
    main()
