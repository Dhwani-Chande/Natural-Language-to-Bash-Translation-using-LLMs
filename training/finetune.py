# ============================================================
# finetune.py — Fine-tuning pipeline for NL → Bash models
#
# Usage:
#   python finetune.py                        # trains all models
#   python finetune.py --model llama          # trains Llama only
#   python finetune.py --model qwen           # trains Qwen only
# ============================================================

import gc
import os
import time
import argparse

import torch
import wandb
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import MODELS_TO_FINETUNE, TRAINING_ARGS, WANDB_PROJECT
from dataset import load_nl2bash_datasets, prepare_datasets


def load_model_and_tokenizer(model_id: str, is_llama: bool):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        clean_up_tokenization_spaces=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Llama requires a specific pad token
    if is_llama:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        print("Set pad_token for Llama model")

    print(f"Model loaded with {model.num_parameters():,} parameters")
    return model, tokenizer


def finetune_model(config: dict) -> str:
    """Fine-tune a single model and return the output path."""
    model_id = config["model_id"]
    output_name = config["output_name"]
    is_llama = config["is_llama"]

    print("\n" + "=" * 80)
    print(f"FINE-TUNING: {model_id}")
    print("=" * 80 + "\n")

    # Init W&B run
    wandb.init(project=WANDB_PROJECT, name=output_name, reinit=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, is_llama)

    # Load and prepare datasets
    train_dataset, test_dataset = load_nl2bash_datasets()
    final_train, final_test = prepare_datasets(train_dataset, test_dataset, tokenizer)

    # Training configuration
    model.train()
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{output_name}",
        eval_strategy="steps",
        eval_steps=TRAINING_ARGS["eval_steps"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        save_steps=TRAINING_ARGS["save_steps"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        learning_rate=TRAINING_ARGS["learning_rate"],
        max_grad_norm=TRAINING_ARGS["max_grad_norm"],
        weight_decay=TRAINING_ARGS["weight_decay"],
        seed=TRAINING_ARGS["seed"],
        bf16=TRAINING_ARGS["bf16"],
        save_total_limit=TRAINING_ARGS["save_total_limit"],
        report_to="wandb",
        log_level="info",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train,
        eval_dataset=final_test,
        processing_class=tokenizer,
    )

    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=False)

    print(f"\n💾 Saving to {output_name}/")
    trainer.save_model(output_name)
    tokenizer.save_pretrained(output_name)

    wandb.finish()
    print(f"✅ Fine-tuning complete: {model_id}")

    # Free memory before next model
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_name


def main(model_filter: str = None):
    """Run fine-tuning for all models (or a specific one)."""

    # Auth
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set. Private models may not load.")

    # Filter models if requested
    configs = MODELS_TO_FINETUNE
    if model_filter == "llama":
        configs = [c for c in configs if "llama" in c["model_id"].lower()]
    elif model_filter == "qwen":
        configs = [c for c in configs if "qwen" in c["model_id"].lower()]

    start_time = time.time()
    finetuned_models = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'#' * 80}")
        print(f"# MODEL {i}/{len(configs)}")
        print(f"{'#' * 80}\n")
        try:
            output_path = finetune_model(config)
            finetuned_models.append(output_path)
            print(f"\n✅ Completed {i}/{len(configs)} models")
        except Exception as e:
            print(f"\n❌ Error with {config['model_id']}: {e}")
            continue

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"⏱️  Total time: {total_time / 3600:.2f} hours")
    print(f"📁 Models saved: {finetuned_models}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NL→Bash models")
    parser.add_argument(
        "--model",
        choices=["llama", "qwen"],
        default=None,
        help="Which model to fine-tune (default: both)",
    )
    args = parser.parse_args()
    main(model_filter=args.model)
