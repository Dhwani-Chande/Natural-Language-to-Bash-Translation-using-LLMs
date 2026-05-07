# NL → Bash Translator

> Fine-tuned Qwen2.5-Coder on 40K natural language → Bash pairs. Includes an 8-metric evaluation suite and a FastAPI deployment on Hugging Face Spaces.

**[🚀 Live Demo](https://dhwanichande29-nl-to-bash.hf.space)** &nbsp;·&nbsp; **[📄 Portfolio Page](https://dhwani-chande.github.io/Natural-Language-to-Bash-Translation-using-LLMs)**

---

## Overview

This project explores whether a compact, fine-tuned LLM can reliably translate plain English descriptions into correct Bash commands — a task requiring understanding of both user intent and shell syntax.

The model was trained using LoRA adapters for parameter-efficient fine-tuning, keeping it lightweight while achieving strong benchmark results across 8 evaluation metrics.

```
Input:  "find all .log files larger than 10MB and delete them"
Output: find / -name "*.log" -size +10M -delete
```

---

## Results

| Metric | Score |
|---|---|
| Exact Match | — |
| BLEU | — |
| Functional Correctness | — |
| Token F1 | — |
| Command Accuracy | — |

> Fill in your actual numbers from `feh_comparison.ipynb`

---

## Project Structure

```
├── finetune.ipynb          # Fine-tuning pipeline (LoRA + Qwen2.5-Coder)
├── feh_comparison.ipynb    # Evaluation suite — 8 metrics, model comparison
├── example.ipynb           # Inference examples
├── api/
│   ├── main.py             # FastAPI app
│   └── Dockerfile          # Container for HF Spaces deployment
└── .github/workflows/      # CI/CD
```

---

## Model & Training

| Detail | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B |
| Fine-tuning method | LoRA (PEFT) |
| Training dataset | 40,000 NL→Bash pairs |
| Framework | HuggingFace Transformers |

---

## API

The model is served via FastAPI and deployed on Hugging Face Spaces.

**Translate a command**
```bash
curl -X POST https://dhwanichande29-nl-to-bash.hf.space/translate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "list all files modified in the last 7 days"}'
```

**Response**
```json
{
  "bash_command": "find . -mtime -7 -type f -ls"
}
```

**Endpoints**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/translate` | Translate one instruction |
| `POST` | `/batch` | Translate multiple instructions |

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Dhwani-Chande/Natural-Language-to-Bash-Translation-using-LLMs
cd Natural-Language-to-Bash-Translation-using-LLMs

# Install dependencies
pip install -r requirements.txt

# Start the API
cd api
uvicorn main:app --reload
```

Or with Docker:

```bash
docker build -t nl-to-bash ./api
docker run -p 8000:8000 nl-to-bash
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `finetune.ipynb` | Full fine-tuning pipeline — data loading, LoRA config, training loop |
| `feh_comparison.ipynb` | Evaluation across 8 metrics, base vs fine-tuned comparison |
| `example.ipynb` | Inference examples and qualitative analysis |

---

## Tech Stack

- **Model:** Qwen2.5-Coder-1.5B
- **Fine-tuning:** LoRA via HuggingFace PEFT
- **API:** FastAPI + Uvicorn
- **Deployment:** Docker → Hugging Face Spaces
- **CI/CD:** GitHub Actions

---

## Author

**Dhwani Chande**

[![GitHub](https://img.shields.io/badge/GitHub-Dhwani--Chande-black?logo=github)](https://github.com/Dhwani-Chande)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-dhwanichande29-yellow?logo=huggingface)](https://huggingface.co/dhwanichande29)
