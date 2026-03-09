from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class TranslateRequest(BaseModel):
    instruction: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language instruction to translate to Bash.",
        examples=["List all files in the current directory sorted by size"]
    )
    max_new_tokens: int = Field(
        default=128,
        ge=10,
        le=512,
        description="Maximum number of tokens to generate."
    )

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Instruction cannot be empty or whitespace.")
        return v.strip()


class TranslateResponse(BaseModel):
    instruction: str
    bash_command: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds.")


class BatchRequest(BaseModel):
    instructions: List[str] = Field(
        ...,
        min_length=1,
        description="List of natural language instructions (max 10).",
        examples=[["List all .py files", "Delete files older than 7 days"]]
    )
    max_new_tokens: int = Field(default=128, ge=10, le=512)

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, v):
        cleaned = [i.strip() for i in v if i.strip()]
        if not cleaned:
            raise ValueError("Instructions list cannot be empty.")
        return cleaned


class BatchResponse(BaseModel):
    results: List[TranslateResponse]
    total_latency_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool
