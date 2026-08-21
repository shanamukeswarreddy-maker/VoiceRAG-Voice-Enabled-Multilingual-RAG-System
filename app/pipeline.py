"""
RAG Pipeline Orchestrator.
Connects all stages: STT → Guardrail Pre-Check → Embed → Retrieve → Generate → Guardrail Post-Check.
Each stage has typed input/output and structured error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.config import get_config
from app.models import (
    ChunkStrategy,
    GenerationResult,
    GuardrailResult,
    PipelineResponse,
    RetrievalResult,
    STTResult,
    StageStatus,
)
from app.stages.embedder import get_embedding_model
from app.stages.generator import get_generator
from app.stages.guardrails import get_guardrail_engine
from app.stages.retriever import get_retriever
from app.stages.stt import STTProvider, create_stt_provider
from app.tracing import RequestTrace

logger = logging.getLogger("rag.pipeline")


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline with per-stage tracing and error handling.

    Latency budget:
    - STT: reported SEPARATELY (network-bound, typically 500-2000ms)
    - Pipeline (embed + retrieve + generate + guardrail): target <200ms
    """

    def __init__(self, stt_provider: Optional[STTProvider] = None):
        self.stt = stt_provider or create_stt_provider()
        self.embedder = get_embedding_model()
        self.retriever = get_retriever()
        self.generator = get_generator()
        self.guardrails = get_guardrail_engine()
        self.config = get_config()

    async def process_voice(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        strategy: ChunkStrategy = ChunkStrategy.FIXED,
        top_k: int = 5,
    ) -> PipelineResponse:
        """
        Full pipeline: audio → STT → RAG → response.
        STT latency is measured and reported separately.
        """
        trace = RequestTrace()

        # ── Stage: STT (measured separately) ──────────────────────────
        stt_result: Optional[STTResult] = None
        try:
            with trace.stage("stt"):
                stt_result = await self.stt.transcribe(audio_bytes, filename)
        except Exception as e:
            logger.error(f"[{trace.request_id}] STT failed: {e}")
            return PipelineResponse(
                request_id=trace.request_id,
                answer="Speech recognition failed. Please try again.",
                status=StageStatus.ERROR,
                error=f"STT error: {str(e)}",
                stt_latency_ms=trace.get_stage_latency("stt"),
                stage_latencies=trace.get_all_latencies(),
            )

        # ── STT Confidence Guardrail ──────────────────────────────────
        confidence_check = self.guardrails.check_stt_confidence(stt_result)
        if not confidence_check.passed:
            return PipelineResponse(
                request_id=trace.request_id,
                answer=confidence_check.reason,
                stt_result=stt_result,
                guardrail_results=[confidence_check],
                status=StageStatus.FILTERED,
                stt_latency_ms=stt_result.latency_ms,
                stage_latencies=trace.get_all_latencies(),
            )

        # ── Continue with text pipeline ───────────────────────────────
        response = await self.process_text(
            query=stt_result.transcript,
            strategy=strategy,
            top_k=top_k,
            trace=trace,
        )

        # Attach STT info
        response.stt_result = stt_result
        response.stt_latency_ms = stt_result.latency_ms
        response.language = stt_result.language

        trace.log()
        return response

    async def process_text(
        self,
        query: str,
        strategy: ChunkStrategy = ChunkStrategy.FIXED,
        top_k: int = 5,
        trace: Optional[RequestTrace] = None,
    ) -> PipelineResponse:
        """
        Text-only pipeline: query → RAG → response.
        Latency target: <200ms for embed + retrieve + generate + guardrail.
        """
        if trace is None:
            trace = RequestTrace()

        guardrail_results = []
        pipeline_start = time.perf_counter()

        # ── Check for Conversational Small Talk / Greetings ───────────
        clean_query = query.strip().lower()
        greeting_patterns = [
            r"^(hey|hi|hello|greetings|good\s*(morning|afternoon|evening)|how\s*are\s*you|how\s*do\s*you\s*do|are\s*you\s*fine)",
            r"^(नमस्ते|नमस्कार|आप\s*कैसे\s*हैं|क्या\s*हाल\s*है|हेलो)",
        ]
        import re
        is_greeting = any(re.search(p, clean_query) for p in greeting_patterns)
        
        if is_greeting:
            is_hindi = any("\u0900" <= c <= "\u097F" for c in query)
            greeting_ans = (
                "नमस्ते! मैं कुशल हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ?"
                if is_hindi
                else "Hello! I am doing great, thank you for asking. How can I help you today?"
            )
            return PipelineResponse(
                request_id=trace.request_id,
                answer=greeting_ans,
                guardrail_results=guardrail_results,
                status=StageStatus.SUCCESS,
                pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                stage_latencies=trace.get_all_latencies(),
            )

        # ── Pre-Guardrails (unsafe input + off-topic) ────────────────
        try:
            with trace.stage("guardrail_pre"):
                unsafe_check = self.guardrails.check_unsafe_input(query)
                guardrail_results.append(unsafe_check)

                if not unsafe_check.passed:
                    logger.warning(f"[{trace.request_id}] Query failed unsafe input filter")
                    return PipelineResponse(
                        request_id=trace.request_id,
                        answer=unsafe_check.reason,
                        guardrail_results=guardrail_results,
                        status=StageStatus.FILTERED,
                        pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                        stage_latencies=trace.get_all_latencies(),
                    )
        except Exception as e:
            logger.error(f"[{trace.request_id}] Pre-guardrail error: {e}", exc_info=True)

        # ── Stage: Embedding ──────────────────────────────────────────
        try:
            with trace.stage("embedding"):
                query_embedding = self.embedder.encode_query(query)
        except Exception as e:
            logger.error(f"[{trace.request_id}] Embedding error: {e}", exc_info=True)
            return PipelineResponse(
                request_id=trace.request_id,
                answer="Failed to process your query. Please try again.",
                status=StageStatus.ERROR,
                error=f"Embedding error: {str(e)}",
                pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                stage_latencies=trace.get_all_latencies(),
            )

        # ── Off-topic check (uses query embedding) ───────────────────
        try:
            with trace.stage("guardrail_offtopic"):
                off_topic_check = self.guardrails.check_off_topic(query_embedding)
                guardrail_results.append(off_topic_check)

                if not off_topic_check.passed:
                    logger.warning(f"[{trace.request_id}] Query flagged as off-topic")
                    return PipelineResponse(
                        request_id=trace.request_id,
                        answer=off_topic_check.reason,
                        guardrail_results=guardrail_results,
                        status=StageStatus.FILTERED,
                        pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                        stage_latencies=trace.get_all_latencies(),
                    )
        except Exception as e:
            logger.error(f"[{trace.request_id}] Off-topic check error: {e}", exc_info=True)

        # ── Stage: Retrieval ──────────────────────────────────────────
        retrieval_result: Optional[RetrievalResult] = None
        try:
            strategy_str = strategy.value if hasattr(strategy, "value") else str(strategy)
            with trace.stage("retrieval"):
                retrieval_result = self.retriever.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    strategy_filter=strategy_str,
                )
                retrieval_result.query_text = query
        except Exception as e:
            logger.error(f"[{trace.request_id}] Retrieval error: {e}", exc_info=True)
            return PipelineResponse(
                request_id=trace.request_id,
                answer="Failed to search the knowledge base. Please try again.",
                status=StageStatus.ERROR,
                error=f"Retrieval error: {str(e)}",
                pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                stage_latencies=trace.get_all_latencies(),
            )

        if not retrieval_result.chunks:
            return PipelineResponse(
                request_id=trace.request_id,
                answer="I couldn't find relevant information to answer your question.",
                retrieval_result=retrieval_result,
                guardrail_results=guardrail_results,
                status=StageStatus.SUCCESS,
                pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                stage_latencies=trace.get_all_latencies(),
            )

        # ── Stage: Generation ─────────────────────────────────────────
        generation_result: Optional[GenerationResult] = None
        try:
            with trace.stage("generation"):
                generation_result = await self.generator.generate(
                    query=query,
                    retrieved_chunks=retrieval_result.chunks,
                )
        except Exception as e:
            logger.error(f"[{trace.request_id}] Generation error: {e}", exc_info=True)
            return PipelineResponse(
                request_id=trace.request_id,
                answer="Failed to generate an answer. Please try again.",
                retrieval_result=retrieval_result,
                status=StageStatus.ERROR,
                error=f"Generation error: {str(e)}",
                pipeline_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
                stage_latencies=trace.get_all_latencies(),
            )

        # ── Post-Guardrail: Grounding Check ───────────────────────────
        try:
            with trace.stage("guardrail_post"):
                grounding_check = self.guardrails.check_grounding(
                    answer=generation_result.answer,
                    retrieved_chunks=retrieval_result.chunks,
                )
                guardrail_results.append(grounding_check)

                if not grounding_check.passed:
                    generation_result.answer = grounding_check.reason
        except Exception as e:
            logger.error(f"[{trace.request_id}] Grounding check error: {e}", exc_info=True)

        # ── Build final response ──────────────────────────────────────
        pipeline_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        fallback_triggered = generation_result.fallback_triggered if generation_result else False

        response = PipelineResponse(
            request_id=trace.request_id,
            answer=generation_result.answer,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            guardrail_results=guardrail_results,
            pipeline_latency_ms=pipeline_ms,
            stage_latencies=trace.get_all_latencies(),
            fallback_triggered=fallback_triggered,
            status=StageStatus.SUCCESS,
        )

        logger.info(
            f"[{trace.request_id}] Pipeline completed in {pipeline_ms:.1f}ms | "
            f"stages={trace.get_all_latencies()}"
        )

        trace.log()
        return response


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
