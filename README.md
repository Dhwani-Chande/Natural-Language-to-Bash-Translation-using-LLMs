# Natural Language to Bash Translation using LLMs

> Fine-tuned **Llama-3.2-1B** and **Qwen2.5-Coder-0.5B** on 40,639 NL→Bash command pairs, with a production FastAPI REST service and full CI/CD pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen?style=flat-square)
![CI](https://github.com/Dhwani-Chande/Natural-Language-to-Bash-Translation-using-LLMs/actions/workflows/test.yml/badge.svg)

---

## Overview

This project explores whether small, fine-tuned LLMs can reliably translate natural language instructions into executable Bash commands — a task that requires syntactic precision, contextual understanding, and generalization to unseen command patterns.

Key findings:
- **25-point accuracy gain** over the base model after fine-tuning on domain-specific data
- Semantic evaluation metrics (TF-IDF cosine similarity) outperform exact-match by **5x** in capturing functional equivalence
- **82% correlation** between our Docker-free validation pipeline and full execution-based testing

---

## Repository Structure

```
├── finetune.ipynb          # Fine-tuning pipeline (Llama-3.2-1B & Qwen2.5-Coder-0.5B)
├── feh_comparison.ipynb    # Evaluation suite — 8 metrics benchmarked
├── example.ipynb           # Inference examples and output analysis
├── api/
│   ├── app/
│   │   ├── main.py         # FastAPI app — 3 endpoints, latency middleware
│   │   ├── models.py       # Pydantic request/response schemas
│   │   ├── translator.py   # Model loading and inference
│   │   ├── logger.py       # Request logging
│   │   └── tests/
│   │       └── test_api.py # 20 pytest tests (100% pass rate)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
└── .github/
    └── workflows/
        └── test.yml        # GitHub Actions CI
```

---

## Models

| Model | Base | Fine-tuned On | Accuracy Gain |
|---|---|---|---|
| Llama-3.2-1B | Meta | 40,639 NL→Bash pairs | +25 points |
| Qwen2.5-Coder-0.5B | Alibaba | 40,639 NL→Bash pairs | +25 points |

Both models were fine-tuned using **LoRA / PEFT** for parameter-efficient training on low-resource hardware.

---

## Evaluation Suite

The `feh_comparison.ipynb` notebook benchmarks 8 evaluation metrics:

| Metric | Type | Finding |
|---|---|---|
| Exact Match | Lexical | Underestimates functional equivalence |
| BLEU | N-gram overlap | Moderate correlation with correctness |
| TF-IDF Cosine Similarity | Semantic | Best proxy — 5x more accurate than exact match |
| Token F1 | Lexical | Good for partial credit |
| Command Presence | Structural | Validates core command is correct |
| Flag Accuracy | Structural | Validates flag usage |
| Argument Accuracy | Structural | Validates argument structure |
| Execution Simulation | Functional | Ground truth — used for correlation analysis |

---

## REST API

The fine-tuned Qwen2.5-Coder-0.5B model is served via a **FastAPI** service.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model status and readiness |
| `POST` | `/translate` | Translate a single NL instruction to Bash |
| `POST` | `/batch` | Translate up to 10 instructions in one request |

### Example

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "List all files in the current directory"}'
```

```json
{
  "instruction": "List all files in the current directory",
  "bash_command": "ls",
  "confidence": 0.9,
  "latency_ms": 1578.7
}
```

### Run Locally

```bash
# Clone the repo
git clone https://github.com/Dhwani-Chande/Natural-Language-to-Bash-Translation-using-LLMs.git
cd Natural-Language-to-Bash-Translation-using-LLMs/api

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the server (set MODEL_NAME to your local model path)
MODEL_NAME=/path/to/qwen_model uvicorn app.main:app --port 8000
```

Swagger UI available at: `http://localhost:8000/docs`

---

## Testing

```bash
cd api
source venv/bin/activate
pytest app/tests/test_api.py -v
```

**20/20 tests passing** — covers health check, single translation, batch translation, input validation, error handling, and latency logging.

CI runs automatically on every push via **GitHub Actions**.

---

## Tech Stack

- **Fine-tuning:** PyTorch, Hugging Face Transformers, PEFT/LoRA
- **Serving:** FastAPI, Uvicorn, Pydantic
- **Testing:** pytest, GitHub Actions
- **Evaluation:** BLEU, TF-IDF, Cosine Similarity, custom metrics
- **Containerization:** Docker

---

## Author

**Dhwani Chande** — MS CS @ University of Illinois at Chicago

[![LinkedIn](https://img.shields.io/badge/LinkedIn-dhwani--chande29-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/dhwani-chande29/)
[![GitHub](https://img.shields.io/badge/GitHub-Dhwani--Chande-181717?style=flat-square&logo=github)](https://github.com/Dhwani-Chande)
