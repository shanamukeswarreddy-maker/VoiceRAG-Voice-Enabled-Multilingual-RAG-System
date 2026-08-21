"""
Guardrails: safety checks before and after retrieval + generation.

1. Low STT confidence → ask for clarification
2. Unsafe input filter → block harmful content
3. Off-topic detection → check if query is within MSMARCO domain
4. Grounding check → verify answer is supported by retrieved context
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Set

import numpy as np

from app.config import get_config
from app.models import GuardrailResult, RetrievedChunk, STTResult

logger = logging.getLogger("rag.guardrails")

# ── Unsafe content patterns ───────────────────────────────────────────────

# Basic profanity / harmful content patterns (multilingual)
UNSAFE_PATTERNS = [
    # English patterns
    r"\b(kill|murder|suicide|bomb|attack|weapon|drug|porn)\b",
    # PII patterns
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone numbers
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit cards
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
]

COMPILED_UNSAFE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


class GuardrailEngine:
    """Runs all guardrail checks."""

    def __init__(self):
        self.config = get_config().guardrails
        self._domain_embeddings: Optional[np.ndarray] = None

    def set_domain_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Set representative domain embeddings for off-topic detection.
        Called during startup with a sample of MSMARCO embeddings.
        """
        self._domain_embeddings = embeddings

    # ── 1. STT Confidence Check ────────────────────────────────────────────

    def check_stt_confidence(self, stt_result: STTResult) -> GuardrailResult:
        """Check if STT confidence is above threshold."""
        start = time.perf_counter()

        passed = stt_result.confidence >= self.config.stt_confidence_threshold
        reason = ""
        if not passed:
            reason = (
                f"Speech recognition confidence ({stt_result.confidence:.2f}) is below "
                f"threshold ({self.config.stt_confidence_threshold}). "
                "Please speak more clearly or try again."
            )

        return GuardrailResult(
            passed=passed,
            reason=reason,
            guardrail_type="stt_confidence",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ── 2. Unsafe Input Filter ─────────────────────────────────────────────

    def check_unsafe_input(self, text: str) -> GuardrailResult:
        """Check for profanity, PII, and harmful content."""
        start = time.perf_counter()

        if not self.config.unsafe_content_enabled:
            return GuardrailResult(
                passed=True,
                guardrail_type="unsafe_input",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        for pattern in COMPILED_UNSAFE:
            match = pattern.search(text)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason=(
                        "Your query contains content that I cannot process. "
                        "Please rephrase your question."
                    ),
                    guardrail_type="unsafe_input",
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

        return GuardrailResult(
            passed=True,
            guardrail_type="unsafe_input",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ── 3. Off-Topic Detection ─────────────────────────────────────────────

    def check_off_topic(self, query_embedding: np.ndarray) -> GuardrailResult:
        """
        Check if the query is within the MSMARCO domain.
        Computes max cosine similarity against representative domain embeddings.
        """
        start = time.perf_counter()

        if not self.config.off_topic_check_enabled:
            return GuardrailResult(
                passed=True,
                guardrail_type="off_topic",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        if self._domain_embeddings is None or len(self._domain_embeddings) == 0:
            # No domain embeddings loaded — skip check
            return GuardrailResult(
                passed=True,
                reason="Domain embeddings not loaded, skipping off-topic check",
                guardrail_type="off_topic",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # Compute cosine similarity (embeddings are already L2 normalized)
        query = query_embedding.reshape(1, -1)
        similarities = np.dot(query, self._domain_embeddings.T)[0]
        max_sim = float(np.max(similarities))

        passed = max_sim >= self.config.off_topic_similarity_threshold

        reason = ""
        if not passed:
            reason = (
                "This question appears to be outside my knowledge area. "
                "I can help with general knowledge questions from the MS MARCO dataset."
            )

        return GuardrailResult(
            passed=passed,
            reason=reason,
            guardrail_type="off_topic",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # ── 4. Grounding Check ─────────────────────────────────────────────────

    def check_grounding(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GuardrailResult:
        """
        Verify the answer is supported by retrieved context.
        Uses Jaccard similarity of word sets.
        """
        start = time.perf_counter()

        if not self.config.grounding_check_enabled:
            return GuardrailResult(
                passed=True,
                guardrail_type="grounding",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        if not retrieved_chunks or not answer:
            return GuardrailResult(
                passed=False,
                reason="No context available to verify the answer.",
                guardrail_type="grounding",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # Build word set from all retrieved context
        context_text = " ".join(rc.display_text for rc in retrieved_chunks)
        context_words = self._normalize_words(context_text)
        answer_words = self._normalize_words(answer)

        if not answer_words:
            return GuardrailResult(
                passed=True,
                guardrail_type="grounding",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # Jaccard-like overlap: what fraction of answer words appear in context
        overlap = answer_words & context_words
        overlap_ratio = len(overlap) / len(answer_words)

        passed = overlap_ratio >= self.config.grounding_overlap_threshold
        reason = ""
        if not passed:
            reason = "I don't have enough verified information to answer that based on the provided context."

        return GuardrailResult(
            passed=passed,
            reason=reason,
            guardrail_type="grounding",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    @staticmethod
    def _normalize_words(text: str) -> Set[str]:
        """Normalize text into a set of lowercase words, filtering stopwords."""
        # Very small stopword set to avoid filtering meaningful Indic words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "of", "in", "to", "for",
            "with", "on", "at", "by", "from", "and", "or", "not", "no", "but",
            "if", "this", "that", "it", "i", "you", "he", "she", "we", "they",
            # Hindi stopwords
            "का", "के", "में", "है", "और", "से", "को", "पर", "यह", "एक",
            "की", "ने", "हो", "कि", "था", "हैं", "इस", "वह", "जो", "भी",
        }
        words = set(text.lower().split())
        return words - stopwords


# ── Singleton ──────────────────────────────────────────────────────────────

_guardrail_engine: Optional[GuardrailEngine] = None


def get_guardrail_engine() -> GuardrailEngine:
    """Get or create the singleton guardrail engine."""
    global _guardrail_engine
    if _guardrail_engine is None:
        _guardrail_engine = GuardrailEngine()
    return _guardrail_engine
