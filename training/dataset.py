# ============================================================
# dataset.py — Data loading and preprocessing
# ============================================================

from datasets import load_dataset
from config import DATASET_NAME, SYSTEM_PROMPT, TRAINING_ARGS


def load_nl2bash_datasets():
    """Load train and test splits from HuggingFace."""
    train_dataset = load_dataset(DATASET_NAME, "train", split="train")
    test_dataset = load_dataset(DATASET_NAME, "test", split="train")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset


def apply_chat_template(row, tokenizer):
    """Format a dataset row into a chat prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["nl"]},
        {"role": "assistant", "content": row["bash"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    return {"prompt": prompt}


def tokenize_rows(row, tokenizer):
    """Tokenize a formatted prompt and create labels."""
    tokens = tokenizer(
        row["prompt"],
        padding="max_length",
        truncation=True,
        max_length=TRAINING_ARGS["max_length"],
    )
    # Mask padding tokens in labels so loss is not computed on them
    tokens["labels"] = [
        -100 if token == tokenizer.pad_token_id else token
        for token in tokens["input_ids"]
    ]
    return tokens


def prepare_datasets(train_dataset, test_dataset, tokenizer):
    """Apply chat template and tokenize both splits."""
    print("Formatting datasets...")
    fmt_train = train_dataset.map(lambda r: apply_chat_template(r, tokenizer))
    fmt_test = test_dataset.map(lambda r: apply_chat_template(r, tokenizer))

    print("Tokenizing datasets...")
    tok_train = fmt_train.map(lambda r: tokenize_rows(r, tokenizer))
    tok_test = fmt_test.map(lambda r: tokenize_rows(r, tokenizer))

    # Remove columns not needed by the model
    cols_to_remove = ["nl", "bash", "prompt"]
    final_train = tok_train.remove_columns(
        [c for c in cols_to_remove if c in tok_train.column_names]
    )
    final_test = tok_test.remove_columns(
        [c for c in cols_to_remove if c in tok_test.column_names]
    )

    print("Dataset prepared!")
    return final_train, final_test
