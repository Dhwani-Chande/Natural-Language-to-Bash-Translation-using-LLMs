# NL to Bash Translator API

A REST API that translates natural language instructions into Bash commands using a fine-tuned LLM (Llama-3.2-1B / Qwen2.5-Coder-0.5B).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model and API status |
| POST | `/translate` | Translate a single instruction |
| POST | `/batch` | Translate up to 10 instructions |

## Quickstart

### Local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Docker
```bash
docker build -t nl-bash-api .
docker run -p 8000:8000 nl-bash-api
```

### Custom Model
Set the `MODEL_NAME` environment variable to point to your fine-tuned checkpoint:
```bash
MODEL_NAME=your-hf-username/nl-to-bash-qwen uvicorn app.main:app --reload
# or with Docker:
docker run -e MODEL_NAME=your-hf-username/nl-to-bash-qwen -p 8000:8000 nl-bash-api
```

## Example Request

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Find all Python files modified in the last 7 days"}'
```

```json
{
  "instruction": "Find all Python files modified in the last 7 days",
  "bash_command": "find . -name '*.py' -mtime -7",
  "confidence": 0.85,
  "latency_ms": 143.2
}
```

## Running Tests
```bash
pytest app/tests/ -v
```

## Project Structure
```
bash_translator_api/
├── app/
│   ├── main.py          # FastAPI app, routes, middleware
│   ├── models.py        # Pydantic request/response schemas
│   ├── translator.py    # Model loading and inference
│   ├── logger.py        # Request logging
│   └── tests/
│       └── test_api.py  # pytest test suite (20 tests)
├── Dockerfile
├── requirements.txt
└── README.md
```
