# evaluate_mmlu_accuracy.py

import argparse
import csv
import os
import socket

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


def predict_answer(model, tokenizer, prompt, choice_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    choice_logits = next_token_logits[choice_token_ids]
    pred_index = int(torch.argmax(choice_logits).item())

    return pred_index, choice_logits.detach().float().cpu().tolist()


def load_model_and_tokenizer(model_name, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    if adapter_dir:
        print(f"loading_adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        print("loading_adapter: none")
        model = base_model

    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--category", default="high_school_statistics")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_csv", default="results/mmlu_accuracy.csv")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    print("Environment")
    print(f"host: {socket.gethostname()}")
    print(f"python_userbase: {os.environ.get('PYTHONUSERBASE')}")
    print(f"model_name: {args.model_name}")
    print(f"adapter_dir: {args.adapter_dir if args.adapter_dir else 'none'}")
    print(f"category: {args.category}")
    print(f"split: {args.split}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu_count: {torch.cuda.device_count()}")
        print(f"gpu_0: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_dir)

    choice_token_ids = get_choice_token_ids(tokenizer)
    print("choice_token_ids:", dict(zip(ANSWER_LABELS, choice_token_ids)))

    dataset = load_dataset("cais/mmlu", args.category)[args.split]

    start_index = args.start_index
    end_index = args.end_index if args.end_index > 0 else len(dataset)
    end_index = min(end_index, len(dataset))
    dataset = dataset.select(range(start_index, end_index))

    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"start_index: {start_index}")
    print(f"end_index: {end_index}")

    print(f"num_examples: {len(dataset)}")

    rows = []
    num_correct = 0

    for i, example in enumerate(dataset):
        prompt = format_prompt(example)
        true_index = int(example["answer"])

        pred_index, choice_logits = predict_answer(
            model,
            tokenizer,
            prompt,
            choice_token_ids,
        )

        correct = int(pred_index == true_index)
        num_correct += correct

        rows.append({
            "index": i,
            "category": args.category,
            "split": args.split,
            "model_name": args.model_name,
            "adapter_dir": args.adapter_dir if args.adapter_dir else "none",
            "prediction": ANSWER_LABELS[pred_index],
            "answer": ANSWER_LABELS[true_index],
            "correct": correct,
            "logit_A": choice_logits[0],
            "logit_B": choice_logits[1],
            "logit_C": choice_logits[2],
            "logit_D": choice_logits[3],
            "question": example["question"],
        })

        if i < 5:
            print(
                f"example={i} pred={ANSWER_LABELS[pred_index]} "
                f"answer={ANSWER_LABELS[true_index]} correct={correct}"
            )

    accuracy = num_correct / len(dataset)

    print(f"correct: {num_correct}")
    print(f"total: {len(dataset)}")
    print(f"accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_results: {args.output_csv}")
    print("MMLU evaluation complete.")


if __name__ == "__main__":
    main()
