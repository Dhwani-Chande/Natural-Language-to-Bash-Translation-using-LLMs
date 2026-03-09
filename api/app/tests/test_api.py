import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health Check ──────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "ready" in data

    def test_health_status_is_ok(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


# ── Translate Endpoint ────────────────────────────────────────
class TestTranslateEndpoint:
    def test_translate_returns_200(self):
        response = client.post("/translate", json={"instruction": "List all files sorted by size"})
        assert response.status_code == 200

    def test_translate_response_schema(self):
        response = client.post("/translate", json={"instruction": "List all files sorted by size"})
        data = response.json()
        assert "instruction" in data
        assert "bash_command" in data
        assert "confidence" in data

    def test_translate_echoes_instruction(self):
        instruction = "Show disk usage"
        response = client.post("/translate", json={"instruction": instruction})
        assert response.json()["instruction"] == instruction

    def test_translate_confidence_between_0_and_1(self):
        response = client.post("/translate", json={"instruction": "Delete all .log files"})
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_translate_empty_instruction_rejected(self):
        response = client.post("/translate", json={"instruction": "   "})
        assert response.status_code == 422

    def test_translate_too_short_instruction_rejected(self):
        response = client.post("/translate", json={"instruction": "hi"})
        assert response.status_code == 422

    def test_translate_too_long_instruction_rejected(self):
        response = client.post("/translate", json={"instruction": "x" * 501})
        assert response.status_code == 422

    def test_translate_custom_max_tokens(self):
        response = client.post("/translate", json={
            "instruction": "Find all Python files recursively",
            "max_new_tokens": 64
        })
        assert response.status_code == 200

    def test_translate_invalid_max_tokens_rejected(self):
        response = client.post("/translate", json={
            "instruction": "List files",
            "max_new_tokens": 9  # below minimum of 10
        })
        assert response.status_code == 422

    def test_translate_latency_header_present(self):
        response = client.post("/translate", json={"instruction": "List all files sorted by size"})
        assert "X-Latency-Ms" in response.headers


# ── Batch Endpoint ────────────────────────────────────────────
class TestBatchEndpoint:
    def test_batch_returns_200(self):
        response = client.post("/batch", json={
            "instructions": ["List files", "Show disk usage", "Delete temp files"]
        })
        assert response.status_code == 200

    def test_batch_response_count_matches(self):
        instructions = ["List files", "Show disk usage"]
        response = client.post("/batch", json={"instructions": instructions})
        assert response.json()["count"] == len(instructions)

    def test_batch_results_length_matches(self):
        instructions = ["List files", "Show disk usage", "Find .py files"]
        response = client.post("/batch", json={"instructions": instructions})
        assert len(response.json()["results"]) == len(instructions)

    def test_batch_exceeds_limit_rejected(self):
        response = client.post("/batch", json={
            "instructions": [f"instruction {i}" for i in range(11)]
        })
        assert response.status_code == 400

    def test_batch_empty_list_rejected(self):
        response = client.post("/batch", json={"instructions": []})
        assert response.status_code == 422

    def test_batch_total_latency_present(self):
        response = client.post("/batch", json={"instructions": ["List files"]})
        assert "total_latency_ms" in response.json()

    def test_batch_filters_whitespace_instructions(self):
        response = client.post("/batch", json={
            "instructions": ["List files", "   ", "Show disk usage"]
        })
        # whitespace-only entries are stripped, so count should be 2
        assert response.json()["count"] == 2
