# Natural Language to Bash Translation using LLMs

Fine-tuned **Llama-3.2-1B** and **Qwen2.5-Coder-0.5B** on 40K natural language → Bash command pairs. Includes a comprehensive evaluation suite benchmarking 8 heuristics, plus a FastAPI deployment.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/dhwanichande29/nl-to-bash)
[![Live API](https://img.shields.io/badge/🔗%20API-Live-green)](https://dhwanichande29-nl-to-bash.hf.space/docs)
[![Live Demo](https://img.shields.io/badge/🚀%20Demo-Gradio%20Space-blue)](https://huggingface.co/spaces/dhwanichande29/nl-to-bash)

> 💤 Note: The API may take ~30 seconds to wake up on first visit due to inactivity sleep. Once running, expect ~10s latency on free CPU hardware.

---

## Models

| Model | Parameters | Type |
|---|---|---|
| meta-llama/Llama-3.2-1B-Instruct | 1.23B | General purpose |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 494M | Code-specialized |

Both models were fully fine-tuned (no LoRA) on an NVIDIA A100-SXM4-80GB in ~2.09 hours.

---

## Dataset

- **Source:** [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
- **Train:** 40,639 examples
- **Test:** 300 examples
- **Format:** Natural language instruction → Bash command pairs

---

## Results

### Model Comparison

| Model | Exact Match | Semantic Match (≥0.8) | Avg Similarity |
|---|---|---|---|
| Llama-3.2-1B | 11.00% | 57.00% | 0.766 |
| Qwen2.5-Coder-0.5B | **13.67%** | **60.33%** | **0.776** |

> Qwen2.5-Coder-0.5B outperforms Llama-3.2-1B on all metrics despite being less than half the size.

---

### Evaluation Across 8 Heuristics (Qwen2.5-Coder-0.5B)

| Heuristic | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| BLEU | 0.99 | 0.39 | 0.56 | 0.69 |
| NL2CMD | 0.98 | 0.20 | 0.33 | 0.60 |
| TF-IDF | 0.99 | 0.46 | 0.63 | 0.73 |
| Exec TF-IDF | 0.99 | 0.65 | 0.78 | 0.82 |
| MxBai Embed | 0.83 | 0.82 | 0.82 | 0.82 |
| **Exec MxBai Embed** | **0.96** | **0.83** | **0.89** | **0.90** |
| Llama3 Judge | 0.49 | 0.78 | 0.60 | 0.48 |
| Exec Llama3 Judge | 0.61 | 0.91 | 0.73 | 0.67 |

> Execution-aware metrics (`exec_*`) consistently outperform their text-match counterparts.
> `exec_mxbai_embed` achieves the best overall performance — **90% accuracy and 0.89 F1** — by accounting for functionally equivalent Bash commands rather than requiring character-perfect matches.

---

## Project Structure

```
├── notebooks/
│   ├── finetune.ipynb          # Fine-tuning pipeline with outputs
│   ├── feh_comparison.ipynb    # 8-heuristic evaluation and comparison
│   └── example.ipynb           # Example usage and inference
├── training/
│   ├── config.py               # Hyperparameters and model config
│   ├── dataset.py              # Data loading and preprocessing
│   ├── finetune.py             # Training script (CLI)
│   └── evaluate.py             # Evaluation script (CLI)
├── api/                        # FastAPI deployment
├── .github/workflows/          # CI/CD
├── Dockerfile                  # Local deployment
└── requirements.txt
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/Dhwani-Chande/Natural-Language-to-Bash-Translation-using-LLMs
cd Natural-Language-to-Bash-Translation-using-LLMs
pip install -r requirements.txt
```

### Run the API
```bash
cd api
uvicorn main:app --reload
```

### Example Request
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"query": "list all files in current directory"}'
```

### Expected Response
```json
{
  "instruction": "list all files in current directory",
  "bash_command": "ls -l",
  "confidence": 0.9,
  "latency_ms": 10754.27
}
```
> ⚠️ Latency is ~10s on free CPU hardware. For faster inference, run locally with a GPU.

---

## Dependencies

```
torch
transformers
datasets
accelerate
huggingface-hub
wandb
bitsandbytes
sentence-transformers
pandas
fastapi
uvicorn
```

---

## Training Details

- **Epochs:** 10
- **Batch size:** 15 per device (effective batch size: 75 with gradient accumulation)
- **Gradient accumulation steps:** 5
- **Precision:** bfloat16
- **Max token length:** 150
- **Optimizer:** AdamW (default HuggingFace Trainer)
- **Experiment tracking:** Weights & Biases (`nl2sh` project)

---

## Evaluation

| Heuristic | Description |
|---|---|
| BLEU | N-gram overlap with reference command |
| NL2CMD | Command structure similarity |
| TF-IDF | Token frequency-based similarity |
| Exec TF-IDF | TF-IDF applied to command execution output |
| MxBai Embed | Semantic similarity via `mxbai-embed-large` embeddings |
| **Exec MxBai Embed** | Embedding similarity on execution output *(best overall)* |
| Llama3 Judge | LLM-as-judge correctness scoring |
| Exec Llama3 Judge | LLM judge applied to execution output |

Execution-aware variants (`exec_*`) evaluate whether commands produce the **same output** rather than whether they look the same — a critical distinction since `ls` and `find . -type f` are functionally equivalent.

---

## Acknowledgements

- Dataset: [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
- Models: [Meta Llama](https://huggingface.co/meta-llama) · [Qwen](https://huggingface.co/Qwen)
- Experiment tracking: [Weights & Biases](https://wandb.ai)
