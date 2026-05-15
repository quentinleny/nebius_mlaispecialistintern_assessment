# test_hf_model_and_data.py

import os
import socket
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CATEGORIES = [
    "high_school_statistics",
    "high_school_computer_science",
    "college_computer_science",
    "machine_learning",
]

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


def main():
    print("Environment")
    print(f"host: {socket.gethostname()}")
    print(f"python_userbase: {os.environ.get('PYTHONUSERBASE')}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu_count: {torch.cuda.device_count()}")
        print(f"gpu_0: {torch.cuda.get_device_name(0)}")

    print("\nDataset inspection")
    loaded = {}

    for category in CATEGORIES:
        print(f"\nLoading category: {category}")
        dataset = load_dataset("cais/mmlu", category)
        loaded[category] = dataset

        print(dataset)
        for split in dataset:
            print(f"{category}/{split}: {len(dataset[split])}")

        first = dataset["test"][0]
        print("first_example_keys:", list(first.keys()))
        print("first_question:", first["question"])
        print("first_choices:", first["choices"])
        print("first_answer_index:", first["answer"])
        print("first_answer_label:", ANSWER_LABELS[int(first["answer"])])

    primary_category = "high_school_statistics"
    dataset = loaded[primary_category]

    print("\nModel loading")
    print(f"model_name: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    print("model_loaded: True")
    print(f"model_device: {next(model.parameters()).device}")

    print("\nGeneration examples")
    examples = dataset["test"].select(range(3))

    for i, example in enumerate(examples):
        prompt = format_prompt(example)
        correct = ANSWER_LABELS[int(example["answer"])]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print("\n--- Example", i, "---")
        print(prompt)
        print("model_output:", repr(decoded))
        print("correct_answer:", correct)

    print("\nHF model and data test complete.")


if __name__ == "__main__":
    main()
