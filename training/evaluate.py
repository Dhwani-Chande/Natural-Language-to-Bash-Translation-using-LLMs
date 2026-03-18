# ============================================================
# evaluate.py — Evaluation pipeline for fine-tuned models
#
# Usage:
#   python evaluate.py                        # evaluates all models
#   python evaluate.py --model llama          # evaluates Llama only
#   python evaluate.py --model qwen           # evaluates Qwen only
# ============================================================

import argparse
import csv

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from config import DATASET_NAME, SYSTEM_PROMPT


MODEL_PATHS = [
    "llama_1b_nl2sh_finetuned",
    "qwen_0.5b_nl2sh_finetuned",
]

SEMANTIC_THRESHOLD = 0.8


def generate_prediction(model, tokenizer, instruction: str) -> str:
    """Generate a Bash command for a given natural language instruction."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()


def exact_match_eval(model_path: str, test_dataset) -> str:
    """Run exact match evaluation and save results to CSV."""
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {model_path}")
    print(f"{'=' * 80}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    results = []
    correct = 0

    for example in tqdm(test_dataset, desc="Generating predictions"):
        prediction = generate_prediction(model, tokenizer, example["nl"])
        ground_truth = example["bash"]
        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        results.append({
            "prompt": example["nl"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": 1 if is_correct else 0,
        })

    accuracy = (correct / len(test_dataset)) * 100
    print(f"\n{'─' * 80}")
    print(f"📊 RESULTS: {model_path}")
    print(f"{'─' * 80}")
    print(f"Correct: {correct}/{len(test_dataset)}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    print(f"{'─' * 80}\n")

    # Save to CSV
    csv_filename = f"{model_path}_benchmark.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt", "ground_truth", "prediction", "correct"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Saved results to {csv_filename}")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return csv_filename


def semantic_eval(model_path: str):
    """Run semantic similarity evaluation on top of exact match results."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv(f"{model_path}_benchmark.csv")
    similarities = []
    correct_semantic = 0

    print(f"\n{'=' * 80}")
    print(f"Semantic Evaluation: {model_path}")
    print(f"{'=' * 80}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing similarities"):
        embeddings = embed_model.encode([row["prediction"], row["ground_truth"]])
        sim = (embeddings[0] @ embeddings[1]) / (
            (embeddings[0] @ embeddings[0]) ** 0.5
            * (embeddings[1] @ embeddings[1]) ** 0.5
        )
        similarities.append(sim)
        if sim >= SEMANTIC_THRESHOLD:
            correct_semantic += 1

    df["similarity"] = similarities
    df["semantic_correct"] = df["similarity"] >= SEMANTIC_THRESHOLD
    df.to_csv(f"{model_path}_semantic_eval.csv", index=False)

    exact_acc = (df["correct"].sum() / len(df)) * 100
    semantic_acc = (correct_semantic / len(df)) * 100
    avg_similarity = df["similarity"].mean()

    print(f"\nResults:")
    print(f"  Exact Match:    {exact_acc:.2f}%")
    print(f"  Semantic Match: {semantic_acc:.2f}% (threshold={SEMANTIC_THRESHOLD})")
    print(f"  Avg Similarity: {avg_similarity:.3f}")
    print(f"  Improvement:    +{semantic_acc - exact_acc:.2f}%")


def main(model_filter: str = None):
    """Run full evaluation pipeline."""
    test_dataset = load_dataset(DATASET_NAME, "test", split="train")
    print(f"Loaded {len(test_dataset)} test examples")

    paths = MODEL_PATHS
    if model_filter == "llama":
        paths = [p for p in paths if "llama" in p]
    elif model_filter == "qwen":
        paths = [p for p in paths if "qwen" in p]

    # Exact match
    for model_path in paths:
        exact_match_eval(model_path, test_dataset)

    # Semantic similarity
    for model_path in paths:
        semantic_eval(model_path)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NL→Bash models")
    parser.add_argument(
        "--model",
        choices=["llama", "qwen"],
        default=None,
        help="Which model to evaluate (default: both)",
    )
    args = parser.parse_args()
    main(model_filter=args.model)
