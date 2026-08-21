"""
LLM answer generation using Groq API (OpenAI-compatible).
Uses llama-3.1-8b-instant with persistent httpx connection pooling and hard 170ms timeout fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

import httpx
from groq import AsyncGroq

from app.config import get_config, get_secrets
from app.models import GenerationResult, RetrievedChunk

logger = logging.getLogger("rag.generator")


def fallback_response(retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Deterministic fallback response built directly from top retrieved chunk.
    Zero latency, no LLM call.
    """
    if not retrieved_chunks:
        return "I don't have enough information to answer that based on the provided context."

    top_text = retrieved_chunks[0].display_text.strip()
    # Format a concise 1-2 sentence excerpt
    sentences = [s.strip() for s in top_text.replace("\n", " ").split(".") if s.strip()]
    if sentences:
        fallback_text = ". ".join(sentences[:2])
        if not fallback_text.endswith("."):
            fallback_text += "."
        return fallback_text
    return top_text[:200] + ("..." if len(top_text) > 200 else "")


class LLMGenerator:
    """
    Answer generator using Groq's LPU-accelerated inference.
    Uses persistent httpx AsyncClient connection pooling for ultra-low latency.
    """

    def __init__(self):
        self.config = get_config().generation
        secrets = get_secrets()
        self._client: Optional[AsyncGroq] = None

        # Socket-level timeout limits set strictly at or below budget
        timeout_budget = float(getattr(self.config, "timeout_seconds", 0.10))
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=0.05, read=timeout_budget, write=0.05, pool=0.05),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

        if secrets.groq_api_key:
            self._client = AsyncGroq(
                api_key=secrets.groq_api_key,
                http_client=self._http_client,
                max_retries=0,  # Zero SDK retries — instant fallback on timeout/error!
            )
            # Pre-warm DNS resolution to prevent blocking getaddrinfo on event loop
            try:
                import socket
                socket.getaddrinfo("api.groq.com", 443)
            except Exception:
                pass
        else:
            logger.warning("GROQ_API_KEY not set — generation will use fallback responses")

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build a concise context string from top-2 retrieved chunks, trimmed to 150 chars max."""
        context_parts = []
        for i, rc in enumerate(chunks[:2], 1):  # Top 2 chunks max for sub-100ms latency
            text = rc.display_text[:150].strip()  # Truncate to 150 chars max per chunk
            context_parts.append(f"[{i}] {text}")
        return "\n".join(context_parts)

    def _build_messages(self, query: str, context: str) -> list:
        """Build minimal chat messages for ultra-fast generation."""
        return [
            {"role": "system", "content": "Answer in 1 concise sentence (under 15 words) based on context."},
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {query}\nAnswer:",
            },
        ]

    async def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate an answer from query and retrieved context with low latency budget.
        """
        timeout_budget = float(getattr(self.config, "timeout_seconds", 4.0))
        logger.info(f"[GENERATOR_EXEC] Entering generate() | timeout_budget={timeout_budget}s ({timeout_budget*1000:.0f}ms) | query='{query[:30]}...'")


        if not self._client:
            answer = fallback_response(retrieved_chunks)
            return GenerationResult(
                answer=answer,
                context_used=[rc.display_text for rc in retrieved_chunks],
                model=self.config.model,
                latency_ms=0.0,
                fallback_triggered=True,
            )

        context = self._build_context(retrieved_chunks)
        messages = self._build_messages(query, context)

        start = time.perf_counter()

        async def _call_groq():
            return await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
            )

        task = asyncio.create_task(_call_groq())
        fallback_triggered = False

        try:
            # Non-blocking wait_for: task is automatically cancelled by wait_for on timeout
            response = await asyncio.wait_for(task, timeout=timeout_budget)
            answer = response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            if not task.done():
                task.cancel()
            t_fb_start = time.perf_counter()
            answer = fallback_response(retrieved_chunks)
            fb_ms = (time.perf_counter() - t_fb_start) * 1000

            logger.warning(
                f"[FALLBACK_TIMING] LLM timed out after {timeout_budget*1000:.0f}ms | "
                f"fallback_formatting={fb_ms:.3f}ms"
            )
            fallback_triggered = True
        except Exception as e:
            if not task.done():
                task.cancel()
            t_fb_start = time.perf_counter()
            answer = fallback_response(retrieved_chunks)
            fb_ms = (time.perf_counter() - t_fb_start) * 1000

            logger.error(
                f"[FALLBACK_TIMING] LLM failed: {e} | "
                f"fallback_formatting={fb_ms:.3f}ms"
            )
            fallback_triggered = True

        elapsed_ms = (time.perf_counter() - start) * 1000
        context_used = [rc.display_text for rc in retrieved_chunks]

        logger.info(
            f"Generation completed in {elapsed_ms:.1f}ms | "
            f"model={self.config.model} | fallback={fallback_triggered} | answer_len={len(answer)}"
        )

        return GenerationResult(
            answer=answer,
            context_used=context_used,
            model=self.config.model,
            latency_ms=round(elapsed_ms, 2),
            fallback_triggered=fallback_triggered,
        )


# ── Singleton ──────────────────────────────────────────────────────────────

_generator: Optional[LLMGenerator] = None


def get_generator() -> LLMGenerator:
    """Get or create the singleton generator."""
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator
