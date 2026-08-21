"""
Configuration loader for the Voice-Enabled RAG system.
Reads from config.yaml and .env, validates with Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Sub-models for nested config sections ──────────────────────────────────

class FixedChunkingConfig(BaseModel):
    chunk_size: int = 256
    overlap_ratio: float = 0.20


class SemanticChunkingConfig(BaseModel):
    similarity_threshold: float = 0.5
    min_chunk_size: int = 50
    max_chunk_size: int = 512


class SentenceWindowConfig(BaseModel):
    anchor_sentences: int = 2
    window_sentences: int = 2


class MetadataAwareConfig(BaseModel):
    preserve_fields: List[str] = Field(
        default_factory=lambda: ["query_id", "passage_id", "is_selected", "source_lang", "target_lang"]
    )


class ChunkingConfig(BaseModel):
    default_strategy: str = "fixed"
    fixed: FixedChunkingConfig = Field(default_factory=FixedChunkingConfig)
    semantic: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)
    sentence_window: SentenceWindowConfig = Field(default_factory=SentenceWindowConfig)
    metadata_aware: MetadataAwareConfig = Field(default_factory=MetadataAwareConfig)


class DatasetConfig(BaseModel):
    name: str = "ai4bharat/MSMARCO-XI"
    languages: List[str] = Field(default_factory=lambda: ["hi"])
    max_passages: int = 100_000
    validation_sample: int = 100


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 256
    cache_size: int = 2000
    normalize: bool = True


class RetrievalConfig(BaseModel):
    index_type: str = "IndexFlatIP"
    top_k: int = 2
    score_threshold: float = 0.3


class GenerationConfig(BaseModel):
    provider: str = "groq"
    model: str = "groq/compound-mini"
    max_tokens: int = 70
    temperature: float = 0.1
    timeout_seconds: float = 0.10
    system_prompt: str = (
        "You are a helpful multilingual assistant. Answer the user's question using ONLY the "
        "provided context passages.\nIf the context does not contain enough information to answer, "
        'respond with "I don\'t have enough information to answer that."\n'
        "Keep answers concise (2-3 sentences). Respond in the same language as the user's question."
    )


class GuardrailsConfig(BaseModel):
    stt_confidence_threshold: float = 0.6
    grounding_overlap_threshold: float = 0.3
    off_topic_similarity_threshold: float = 0.25
    unsafe_content_enabled: bool = True
    grounding_check_enabled: bool = True
    off_topic_check_enabled: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class TracingConfig(BaseModel):
    enabled: bool = True
    log_file: str = "logs/requests.jsonl"


# ── Top-level application config ───────────────────────────────────────────

class AppConfig(BaseModel):
    """Complete application configuration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


# ── Environment-based secrets ──────────────────────────────────────────────

class Secrets(BaseSettings):
    """API keys loaded from .env file."""

    sarvam_api_key: str = ""
    groq_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ── Config loading ─────────────────────────────────────────────────────────

_config: Optional[AppConfig] = None
_secrets: Optional[Secrets] = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and cache the application configuration from YAML."""
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _config = AppConfig(**raw)
    else:
        _config = AppConfig()

    return _config


def load_secrets() -> Secrets:
    """Load and cache API key secrets from .env."""
    global _secrets
    if _secrets is not None:
        return _secrets
    _secrets = Secrets()
    return _secrets


def get_config() -> AppConfig:
    """Get the cached config (loads if needed)."""
    return load_config()


def get_secrets() -> Secrets:
    """Get the cached secrets (loads if needed)."""
    return load_secrets()
