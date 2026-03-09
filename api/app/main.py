from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import TranslateRequest, TranslateResponse, BatchRequest, BatchResponse, HealthResponse
from app.translator import BashTranslator
from app.logger import log_request
import time

app = FastAPI(
    title="NL to Bash Translator API",
    description="REST API for translating natural language commands to Bash using fine-tuned LLMs.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = BashTranslator()


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Latency-Ms"] = str(latency_ms)
    log_request(request.method, str(request.url), response.status_code, latency_ms)
    return response


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model=translator.model_name,
        ready=translator.is_ready()
    )


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
def translate(request: TranslateRequest):
    """
    Translate a natural language instruction into a Bash command.
    - **instruction**: Plain English description of the desired Bash command
    - **max_new_tokens**: Max tokens to generate (default: 128)
    """
    try:
        start = time.time()
        result = translator.translate(request.instruction, request.max_new_tokens)
        latency_ms = round((time.time() - start) * 1000, 2)
        return TranslateResponse(
            instruction=request.instruction,
            bash_command=result["bash_command"],
            confidence=result["confidence"],
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse, tags=["Translation"])
def batch_translate(request: BatchRequest):
    """
    Translate a batch of natural language instructions into Bash commands.
    - **instructions**: List of plain English instructions (max 10)
    """
    if len(request.instructions) > 10:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 instructions.")
    try:
        start = time.time()
        results = [
            translator.translate(instr, request.max_new_tokens)
            for instr in request.instructions
        ]
        total_latency_ms = round((time.time() - start) * 1000, 2)
        return BatchResponse(
            results=[
                TranslateResponse(
                    instruction=instr,
                    bash_command=r["bash_command"],
                    confidence=r["confidence"],
                    latency_ms=None
                )
                for instr, r in zip(request.instructions, results)
            ],
            total_latency_ms=total_latency_ms,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
