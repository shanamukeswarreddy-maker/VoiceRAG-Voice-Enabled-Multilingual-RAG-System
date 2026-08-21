"""
Embedding model wrapper with LRU cache for repeated queries.
Uses paraphrase-multilingual-MiniLM-L6-v2 for fast Indic-language embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import unicodedata
from functools import partial
from typing import List, Optional

# Pin thread counts BEFORE PyTorch/MKL initialise their pools
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

# Lock PyTorch threadpools globally at import time
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already set

from cachetools import LRUCache

from app.config import get_config

logger = logging.getLogger("rag.embedder")


class EmbeddingModel:
    """
    Wraps a sentence-transformer model with:
    - Pre-warming at startup
    - Explicit device logging (CPU vs CUDA)
    - LRU cache for query embeddings with hit/miss logging
    - Batch encoding for index building
    """

    def __init__(self):
        self.config = get_config().embedding
        self._model = None
        self._cache: LRUCache = LRUCache(maxsize=self.config.cache_size)
        self._loaded = False
        self._load_timestamp: Optional[float] = None

    def load(self) -> None:
        """Load and pre-warm the embedding model. Call at startup."""
        if self._loaded:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self.config.model_name}")
        start = time.perf_counter()
        self._load_timestamp = time.time()

        # Load SentenceTransformer model
        self._model = SentenceTransformer(self.config.model_name)

        # Explicitly log device (CPU vs CUDA)
        device = getattr(self._model, "device", torch.device("cpu"))
        logger.info(f"Embedding model loaded on device: {device} at timestamp {self._load_timestamp:.2f}")

        # Pre-warm PyTorch model with multilingual sample queries of varying lengths
        warmup_queries = [
            "warmup",
            "boren scholarship essay examples",
            "why are earthquakes caused",
            "what bills has donald trump signed into law",
            "what does laches mean in legal terms",
            "भारत की राजधानी क्या है",
            "what is the definition of artificial intelligence and machine learning",
        ]
        with torch.inference_mode():
            _ = self._model.encode(
                warmup_queries,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Embedding model loaded and warmed in {elapsed:.0f}ms")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def clear_cache(self) -> None:
        """Clear the query embedding LRU cache."""
        self._cache.clear()
        logger.info("Query embedding cache cleared")

    def _cache_key(self, text: str) -> str:
        """Generate a consistent cache key using NFC Unicode normalization, lowercasing, and stripping."""
        normalized = unicodedata.normalize("NFC", text.strip().lower())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query with LRU caching and explicit hit/miss logging.
        Returns a 1D numpy array of shape (dimension,).
        """
        if not self._loaded:
            self.load()

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info(f"Query embedding cache HIT | key={key[:8]} | query='{text[:30]}'")
            return cached

        logger.info(f"Query embedding cache MISS | key={key[:8]} | query='{text[:30]}'")

        with torch.inference_mode():
            embedding = self._model.encode(
                [text],
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

        embedding = np.array(embedding, dtype=np.float32)
        self._cache[key] = embedding
        return embedding


    async def aembed_query(self, text: str) -> np.ndarray:
        """Async wrapper that runs encode_query in a thread executor to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self.encode_query, text))

    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts for index building.
        Returns a 2D numpy array of shape (n, dimension).
        """
        if not self._loaded:
            self.load()

        with torch.inference_mode():
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.config.normalize,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
            )
        return np.array(embeddings, dtype=np.float32)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode individual sentences (used by SemanticChunker).
        """
        if not self._loaded:
            self.load()

        with torch.inference_mode():
            return self._model.encode(
                sentences,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
            )

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def cache_stats(self) -> dict:
        return {
            "cache_size": len(self._cache),
            "cache_maxsize": self._cache.maxsize,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
