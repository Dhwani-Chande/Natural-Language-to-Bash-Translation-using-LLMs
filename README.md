# Natural Language to Bash Translation using LLMs

Fine-tuned **Llama-3.2-1B** and **Qwen2.5-Coder-0.5B** on 40K natural language → Bash command pairs. Includes an evaluation suite using exact match and semantic similarity, plus a FastAPI deployment.

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

| Model | Exact Match | Semantic Match (≥0.8) | Avg Similarity |
|---|---|---|---|
| Llama-3.2-1B | 11.00% | 57.00% | 0.766 |
| Qwen2.5-Coder-0.5B | **13.67%** | **60.33%** | **0.776** |

> Qwen2.5-Coder-0.5B outperforms Llama-3.2-1B on all metrics despite being less than half the size.
> Semantic similarity (via `all-MiniLM-L6-v2`) is a better indicator of real-world quality than exact match alone, since multiple Bash commands can be functionally equivalent.

---

## Project Structure

```
├── finetune.ipynb          # Fine-tuning pipeline for both models
├── feh_comparison.ipynb    # Evaluation and model comparison
├── example.ipynb           # Example usage and inference
├── api/                    # FastAPI deployment
└── .github/workflows/      # CI/CD
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
  "command": "find . -type f"
}
```

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

- **Exact Match** — character-perfect accuracy against ground truth
- **Semantic Match** — cosine similarity ≥ 0.8 using `all-MiniLM-L6-v2` embeddings

---

## Acknowledgements

- Dataset: [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
- Models: [Meta Llama](https://huggingface.co/meta-llama) · [Qwen](https://huggingface.co/Qwen)
- Experiment tracking: [Weights & Biases](https://wandb.ai)
