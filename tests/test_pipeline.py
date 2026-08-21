"""
Tests for the RAG pipeline components.
Run with: pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.models import (
    ChunkMetadata,
    ChunkResult,
    ChunkStrategy,
    GenerationResult,
    GuardrailResult,
    PipelineResponse,
    RetrievalResult,
    RetrievedChunk,
    STTResult,
    StageStatus,
    TextQueryRequest,
)


# ── Config fixture ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_config():
    """Ensure config is loaded for all tests."""
    load_config()


# ── Pydantic Model Tests ──────────────────────────────────────

class TestModels:
    def test_stt_result(self):
        result = STTResult(
            transcript="hello world",
            confidence=0.9,
            language="en-IN",
            latency_ms=500.0,
        )
        assert result.transcript == "hello world"
        assert result.confidence == 0.9
        assert result.language == "en-IN"

    def test_chunk_result(self):
        chunk = ChunkResult(
            chunk_id="abc123",
            text="This is a test passage.",
            strategy=ChunkStrategy.FIXED,
            metadata=ChunkMetadata(chunk_index=0, parent_passage_index=0),
        )
        assert chunk.strategy == ChunkStrategy.FIXED
        assert chunk.chunk_id == "abc123"

    def test_pipeline_response(self):
        response = PipelineResponse(
            request_id="test-001",
            answer="Test answer",
            pipeline_latency_ms=150.0,
            status=StageStatus.SUCCESS,
        )
        assert response.status == StageStatus.SUCCESS
        assert response.pipeline_latency_ms == 150.0

    def test_text_query_request(self):
        req = TextQueryRequest(query="what is the capital?", strategy=ChunkStrategy.SEMANTIC)
        assert req.query == "what is the capital?"
        assert req.strategy == ChunkStrategy.SEMANTIC
        assert req.top_k == 5  # default


# ── Chunking Tests ─────────────────────────────────────────────

class TestChunking:
    def test_fixed_size_chunker_short_passage(self):
        from app.stages.chunking import FixedSizeChunker

        chunker = FixedSizeChunker(chunk_size=256, overlap_ratio=0.2)
        text = "This is a short test passage that fits in one chunk."
        chunks = chunker.chunk(text, passage_index=0)

        assert len(chunks) == 1
        assert chunks[0].strategy == ChunkStrategy.FIXED
        assert chunks[0].text == text.strip()

    def test_fixed_size_chunker_long_passage(self):
        from app.stages.chunking import FixedSizeChunker

        chunker = FixedSizeChunker(chunk_size=10, overlap_ratio=0.2)
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk(text, passage_index=0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.strategy == ChunkStrategy.FIXED
            words = chunk.text.split()
            assert len(words) <= 10

    def test_sentence_window_chunker(self):
        from app.stages.chunking import SentenceWindowChunker

        chunker = SentenceWindowChunker(anchor_sentences=1, window_sentences=1)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text, passage_index=0)

        assert len(chunks) > 0
        assert chunks[0].strategy == ChunkStrategy.SENTENCE_WINDOW
        # Window context should be wider than anchor text
        for chunk in chunks:
            if chunk.metadata.window_context:
                assert len(chunk.metadata.window_context) >= len(chunk.text)

    def test_metadata_aware_chunker(self):
        from app.stages.chunking import MetadataAwareChunker

        chunker = MetadataAwareChunker()
        text = "This is a passage about science."
        metadata = {
            "passage_id": "hi_42_3",
            "query_id": "hi_42",
            "is_selected": True,
            "source_lang": "eng_Latn",
            "target_lang": "hi",
        }
        chunks = chunker.chunk(text, passage_index=0, metadata=metadata)

        assert len(chunks) == 1
        assert chunks[0].strategy == ChunkStrategy.METADATA_AWARE
        assert chunks[0].metadata.passage_id == "hi_42_3"
        assert chunks[0].metadata.is_selected is True

    def test_sentence_split_hindi(self):
        from app.stages.chunking import _sentence_split

        hindi_text = "यह पहला वाक्य है। यह दूसरा वाक्य है। तीसरा वाक्य यहाँ है।"
        sentences = _sentence_split(hindi_text)
        assert len(sentences) == 3


# ── Guardrail Tests ────────────────────────────────────────────

class TestGuardrails:
    def test_stt_confidence_pass(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()
        stt = STTResult(transcript="test", confidence=0.9, language="en", latency_ms=100)
        result = engine.check_stt_confidence(stt)
        assert result.passed is True

    def test_stt_confidence_fail(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()
        stt = STTResult(transcript="t", confidence=0.2, language="en", latency_ms=100)
        result = engine.check_stt_confidence(stt)
        assert result.passed is False
        assert "confidence" in result.reason.lower()

    def test_unsafe_input_clean(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()
        result = engine.check_unsafe_input("What is the capital of India?")
        assert result.passed is True

    def test_unsafe_input_pii_email(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()
        result = engine.check_unsafe_input("Send to user@example.com please")
        assert result.passed is False

    def test_grounding_check_pass(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()

        chunks = [
            RetrievedChunk(
                chunk=ChunkResult(
                    chunk_id="1",
                    text="The capital of India is New Delhi. It is located in northern India.",
                    strategy=ChunkStrategy.FIXED,
                ),
                score=0.9,
            )
        ]
        result = engine.check_grounding("New Delhi is the capital of India.", chunks)
        assert result.passed is True

    def test_grounding_check_fail(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()

        chunks = [
            RetrievedChunk(
                chunk=ChunkResult(
                    chunk_id="1",
                    text="The weather today is sunny and warm.",
                    strategy=ChunkStrategy.FIXED,
                ),
                score=0.9,
            )
        ]
        result = engine.check_grounding(
            "Quantum computing uses qubits for parallel processing of complex algorithms.",
            chunks,
        )
        assert result.passed is False

    def test_off_topic_without_domain_embeddings(self):
        from app.stages.guardrails import GuardrailEngine

        engine = GuardrailEngine()
        query_emb = np.random.randn(384).astype(np.float32)
        result = engine.check_off_topic(query_emb)
        # Should pass (skip) when no domain embeddings are loaded
        assert result.passed is True


# ── Tracing Tests ──────────────────────────────────────────────

class TestTracing:
    def test_request_trace_timing(self):
        import time
        from app.tracing import RequestTrace

        trace = RequestTrace(request_id="test-123")

        with trace.stage("test_stage"):
            time.sleep(0.01)  # 10ms

        latency = trace.get_stage_latency("test_stage")
        assert latency > 5  # Should be at least ~10ms
        assert trace.request_id == "test-123"

    def test_request_trace_multiple_stages(self):
        from app.tracing import RequestTrace

        trace = RequestTrace()

        with trace.stage("stage1"):
            pass
        with trace.stage("stage2"):
            pass

        latencies = trace.get_all_latencies()
        assert "stage1" in latencies
        assert "stage2" in latencies

    def test_request_trace_error(self):
        from app.tracing import RequestTrace

        trace = RequestTrace()

        with pytest.raises(ValueError):
            with trace.stage("failing_stage"):
                raise ValueError("test error")

        assert trace.stages["failing_stage"]["status"] == "error"
        assert len(trace.errors) == 1


# ── Embedder Tests (requires model download) ──────────────────

class TestEmbedder:
    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "1") == "1",
        reason="Skipping model tests (set SKIP_MODEL_TESTS=0 to run)",
    )
    def test_embedding_dimensions(self):
        from app.stages.embedder import EmbeddingModel

        model = EmbeddingModel()
        model.load()

        embedding = model.encode_query("test query")
        assert embedding.shape == (384,)

    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "1") == "1",
        reason="Skipping model tests (set SKIP_MODEL_TESTS=0 to run)",
    )
    def test_embedding_cache(self):
        from app.stages.embedder import EmbeddingModel

        model = EmbeddingModel()
        model.load()

        # First call — should cache
        emb1 = model.encode_query("test query")
        stats1 = model.cache_stats

        # Second call — should hit cache
        emb2 = model.encode_query("test query")
        stats2 = model.cache_stats

        np.testing.assert_array_equal(emb1, emb2)
        assert stats2["cache_size"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
