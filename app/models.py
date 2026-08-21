"""
Pydantic models for all pipeline stages.
Every stage has typed input/output — nothing passes raw dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    FILTERED = "filtered"  # Guardrail blocked


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    SENTENCE_WINDOW = "sentence_window"
    METADATA_AWARE = "metadata_aware"


# ── STT Models ─────────────────────────────────────────────────────────────

class STTResult(BaseModel):
    """Output from the Speech-to-Text stage."""
    transcript: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    language: str = Field(description="Detected language code (e.g., hi-IN)")
    latency_ms: float = Field(description="STT processing time in milliseconds")
    request_id: Optional[str] = None


# ── Chunk Models ───────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk."""
    passage_id: Optional[str] = None
    query_id: Optional[str] = None
    is_selected: Optional[bool] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    # Sentence-window specific
    window_context: Optional[str] = None
    # Position tracking
    chunk_index: int = 0
    parent_passage_index: int = 0


class ChunkResult(BaseModel):
    """A single chunk produced by a chunking strategy."""
    chunk_id: str
    text: str
    strategy: ChunkStrategy
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


# ── Retrieval Models ──────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from vector retrieval with its score."""
    chunk: ChunkResult
    score: float
    # Expanded text for sentence-window strategy
    expanded_text: Optional[str] = None

    @property
    def display_text(self) -> str:
        """Return the best text to show — expanded if available."""
        return self.expanded_text or self.chunk.text


class RetrievalResult(BaseModel):
    """Output from the retrieval stage."""
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    query_text: str = ""
    latency_ms: float = 0.0


# ── Generation Models ─────────────────────────────────────────────────────

class GenerationResult(BaseModel):
    """Output from the LLM generation stage."""
    answer: str
    context_used: List[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    fallback_triggered: bool = False


# ── Guardrail Models ──────────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    """Output from a guardrail check."""
    passed: bool
    reason: str = ""
    guardrail_type: str = ""
    latency_ms: float = 0.0


# ── Stage Result (unified wrapper) ────────────────────────────────────────

class StageResult(BaseModel):
    """Unified result wrapper for any pipeline stage."""
    stage_name: str
    status: StageStatus
    latency_ms: float = 0.0
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


# ── Pipeline Response ─────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """Complete response returned to the client."""
    request_id: str
    answer: str
    language: str = ""
    # Stage-level details
    stt_result: Optional[STTResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    generation_result: Optional[GenerationResult] = None
    guardrail_results: List[GuardrailResult] = Field(default_factory=list)
    # Latency breakdown
    stt_latency_ms: Optional[float] = None  # Reported SEPARATELY
    pipeline_latency_ms: float = 0.0        # embed + retrieve + generate + guardrail
    stage_latencies: Dict[str, float] = Field(default_factory=dict)
    fallback_triggered: bool = False
    # Status
    status: StageStatus = StageStatus.SUCCESS
    error: Optional[str] = None


# ── API Request Models ────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Request body for text-based queries."""
    query: str
    strategy: ChunkStrategy = ChunkStrategy.FIXED
    top_k: int = 5


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    embedding_model_loaded: bool
    index_loaded: bool
    index_size: int = 0
    available_strategies: List[str] = Field(default_factory=list)
    sarvam_configured: bool = False
    groq_configured: bool = False
