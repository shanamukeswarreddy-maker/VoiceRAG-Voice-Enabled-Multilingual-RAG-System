"""
Speech-to-Text adapter with a swappable provider interface.
Default implementation: Sarvam AI (saaras:v3).
"""

from __future__ import annotations

import abc
import io
import logging
import time
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_config, get_secrets
from app.models import STTResult

logger = logging.getLogger("rag.stt")


class STTProvider(abc.ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abc.abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> STTResult:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file content.
            filename: Original filename (used to infer format).

        Returns:
            STTResult with transcript, confidence, language, and latency.
        """
        ...


class SarvamSTTProvider(STTProvider):
    """
    Sarvam AI STT provider using the REST API.
    Retries once with exponential backoff on failure.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self):
        secrets = get_secrets()
        self.api_key = secrets.sarvam_api_key
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not set — STT will fail on real requests")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _call_api(self, audio_bytes: bytes, filename: str) -> dict:
        """Make the API call with retry logic."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Determine content type from filename
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
            content_type_map = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "flac": "audio/flac",
                "ogg": "audio/ogg",
                "webm": "audio/webm",
                "m4a": "audio/mp4",
            }
            content_type = content_type_map.get(ext, "audio/wav")

            response = await client.post(
                self.API_URL,
                headers={"api-subscription-key": self.api_key},
                files={"file": (filename, io.BytesIO(audio_bytes), content_type)},
                data={"model": "saaras:v3", "language_code": "unknown"},
            )
            response.raise_for_status()
            return response.json()

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> STTResult:
        """Transcribe audio using Sarvam AI."""
        start = time.perf_counter()

        try:
            result = await self._call_api(audio_bytes, filename)
            elapsed_ms = (time.perf_counter() - start) * 1000

            transcript = result.get("transcript", "")
            language = result.get("language_code", "unknown")
            request_id = result.get("request_id", "")

            # Sarvam doesn't always return an explicit confidence score.
            # Heuristic: if transcript is non-empty and reasonably long, assume high confidence.
            # If a language_probability field exists, use it.
            confidence = result.get("language_probability", None)
            if confidence is None:
                # Heuristic based on transcript quality
                if len(transcript.strip()) > 5:
                    confidence = 0.85
                elif len(transcript.strip()) > 0:
                    confidence = 0.6
                else:
                    confidence = 0.1

            logger.info(
                f"STT completed in {elapsed_ms:.1f}ms | lang={language} | "
                f"confidence={confidence:.2f} | transcript_len={len(transcript)}"
            )

            return STTResult(
                transcript=transcript,
                confidence=confidence,
                language=language,
                latency_ms=round(elapsed_ms, 2),
                request_id=request_id,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"STT failed after {elapsed_ms:.1f}ms: {e}")
            raise


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing without API keys."""

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> STTResult:
        return STTResult(
            transcript="यह एक परीक्षण प्रश्न है",  # "This is a test question" in Hindi
            confidence=0.95,
            language="hi-IN",
            latency_ms=50.0,
            request_id="mock-001",
        )


def create_stt_provider(provider_type: str = "sarvam") -> STTProvider:
    """Factory function to create the appropriate STT provider."""
    providers = {
        "sarvam": SarvamSTTProvider,
        "mock": MockSTTProvider,
    }
    provider_cls = providers.get(provider_type, SarvamSTTProvider)
    return provider_cls()
