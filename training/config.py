# ============================================================
# config.py — Model and training configuration
# ============================================================

SYSTEM_PROMPT = (
    "Your task is to translate a natural language instruction to a Bash command. "
    "You will receive an instruction in English and output a Bash command that can "
    "be run in a Linux terminal."
)

DATASET_NAME = "westenfelder/NL2SH-ALFA"

MODELS_TO_FINETUNE = [
    {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "output_name": "llama_1b_nl2sh_finetuned",
        "is_llama": True,
        "batch_size": 15,
    },
    {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "output_name": "qwen_0.5b_nl2sh_finetuned",
        "is_llama": False,
        "batch_size": 15,
    },
]

TRAINING_ARGS = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 15,
    "per_device_eval_batch_size": 15,
    "gradient_accumulation_steps": 5,
    "learning_rate": 1e-5,
    "max_grad_norm": 2,
    "weight_decay": 0.01,
    "seed": 123,
    "bf16": True,
    "eval_steps": 1000,
    "logging_steps": 100,
    "save_steps": 5000,
    "save_total_limit": 2,
    "max_length": 150,   # max token length for input sequences
}

WANDB_PROJECT = "nl2sh"
