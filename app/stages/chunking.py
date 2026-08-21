"""
Four chunking strategies for MSMARCO-XI passages.
All produce ChunkResult objects tagged with a `strategy` field.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import get_config
from app.models import ChunkMetadata, ChunkResult, ChunkStrategy

logger = logging.getLogger("rag.chunking")


# ── Utilities ──────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    """Split text into sentences using regex. Handles Hindi/English."""
    # Split on period, question mark, exclamation, or Devanagari purna viram (।)
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _word_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer (works across scripts)."""
    return text.split()


def _make_chunk_id(strategy: str, passage_idx: int, chunk_idx: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{strategy}:{passage_idx}:{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Abstract Chunker ──────────────────────────────────────────────────────

class Chunker(abc.ABC):
    """Base class for all chunking strategies."""

    @abc.abstractmethod
    def chunk(
        self,
        text: str,
        passage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """
        Split a passage into chunks.

        Args:
            text: The passage text to chunk.
            passage_index: Index of the source passage (for ID generation).
            metadata: Additional metadata from the dataset.

        Returns:
            List of ChunkResult objects.
        """
        ...


# ── 1. Fixed-Size Chunker ─────────────────────────────────────────────────

class FixedSizeChunker(Chunker):
    """
    Baseline chunking: fixed token count with configurable overlap.
    Default: 256 tokens, 20% overlap.
    """

    def __init__(self, chunk_size: int = 256, overlap_ratio: float = 0.20):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)

    def chunk(
        self,
        text: str,
        passage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        metadata = metadata or {}
        words = _word_tokenize(text)

        if len(words) <= self.chunk_size:
            # Passage fits in one chunk
            return [
                ChunkResult(
                    chunk_id=_make_chunk_id("fixed", passage_index, 0),
                    text=text.strip(),
                    strategy=ChunkStrategy.FIXED,
                    metadata=ChunkMetadata(
                        chunk_index=0,
                        parent_passage_index=passage_index,
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            ]

        chunks = []
        step = self.chunk_size - self.overlap
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + self.chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append(
                ChunkResult(
                    chunk_id=_make_chunk_id("fixed", passage_index, i),
                    text=chunk_text,
                    strategy=ChunkStrategy.FIXED,
                    metadata=ChunkMetadata(
                        chunk_index=i,
                        parent_passage_index=passage_index,
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            )
        return chunks


# ── 2. Semantic Chunker ───────────────────────────────────────────────────

class SemanticChunker(Chunker):
    """
    Splits on embedding-similarity breakpoints between sentences.
    Detects topic shifts rather than cutting mid-thought.
    Requires an embedding function to be set before use.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 50,
        max_chunk_size: int = 512,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._embed_fn = None  # Set via set_embed_fn()

    def set_embed_fn(self, fn):
        """Set the embedding function: fn(List[str]) -> np.ndarray."""
        self._embed_fn = fn

    def chunk(
        self,
        text: str,
        passage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        metadata = metadata or {}
        sentences = _sentence_split(text)

        if len(sentences) <= 3 or len(text.split()) <= 150:
            return [
                ChunkResult(
                    chunk_id=_make_chunk_id("semantic", passage_index, 0),
                    text=text.strip(),
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata=ChunkMetadata(
                        chunk_index=0,
                        parent_passage_index=passage_index,
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            ]

        # If no embedding function, fall back to fixed-size
        if self._embed_fn is None:
            logger.warning("SemanticChunker: no embed_fn set, falling back to single chunk")
            return [
                ChunkResult(
                    chunk_id=_make_chunk_id("semantic", passage_index, 0),
                    text=text.strip(),
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata=ChunkMetadata(
                        chunk_index=0,
                        parent_passage_index=passage_index,
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            ]

        # Embed all sentences
        embeddings = self._embed_fn(sentences)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))

        # Build chunks from split points
        chunks = []
        for idx, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
            chunk_text = " ".join(sentences[start:end])
            words = _word_tokenize(chunk_text)

            # Enforce min/max sizes
            if len(words) < self.min_chunk_size and idx > 0 and chunks:
                # Merge with previous chunk
                prev = chunks[-1]
                chunks[-1] = ChunkResult(
                    chunk_id=prev.chunk_id,
                    text=prev.text + " " + chunk_text,
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata=prev.metadata,
                )
                continue

            chunks.append(
                ChunkResult(
                    chunk_id=_make_chunk_id("semantic", passage_index, idx),
                    text=chunk_text,
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata=ChunkMetadata(
                        chunk_index=idx,
                        parent_passage_index=passage_index,
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            )

        return chunks if chunks else [
            ChunkResult(
                chunk_id=_make_chunk_id("semantic", passage_index, 0),
                text=text.strip(),
                strategy=ChunkStrategy.SEMANTIC,
                metadata=ChunkMetadata(chunk_index=0, parent_passage_index=passage_index),
            )
        ]


# ── 3. Sentence-Window Chunker ────────────────────────────────────────────

class SentenceWindowChunker(Chunker):
    """
    Small anchor chunk (1-2 sentences) for embedding precision.
    Surrounding context window stored as metadata, expanded at retrieval time.
    """

    def __init__(self, anchor_sentences: int = 2, window_sentences: int = 2):
        self.anchor_sentences = anchor_sentences
        self.window_sentences = window_sentences

    def chunk(
        self,
        text: str,
        passage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        metadata = metadata or {}
        sentences = _sentence_split(text)

        if not sentences:
            return []

        chunks = []
        step = self.anchor_sentences

        for i in range(0, len(sentences), step):
            # Anchor: the core sentences for embedding
            anchor_end = min(i + self.anchor_sentences, len(sentences))
            anchor_text = " ".join(sentences[i:anchor_end])

            # Window: surrounding context for retrieval expansion
            window_start = max(0, i - self.window_sentences)
            window_end = min(len(sentences), anchor_end + self.window_sentences)
            window_text = " ".join(sentences[window_start:window_end])

            chunk_idx = i // step
            chunks.append(
                ChunkResult(
                    chunk_id=_make_chunk_id("sentence_window", passage_index, chunk_idx),
                    text=anchor_text,  # Embed only the anchor
                    strategy=ChunkStrategy.SENTENCE_WINDOW,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_idx,
                        parent_passage_index=passage_index,
                        window_context=window_text,  # Expanded at retrieval
                        **{k: v for k, v in metadata.items() if k in ChunkMetadata.model_fields},
                    ),
                )
            )

        return chunks


# ── 4. Metadata-Aware Chunker ─────────────────────────────────────────────

class MetadataAwareChunker(Chunker):
    """
    Preserves MSMARCO-XI structured fields intact.
    Passages are kept whole (no splitting) with rich metadata for filtering/boosting.
    """

    def chunk(
        self,
        text: str,
        passage_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        metadata = metadata or {}

        return [
            ChunkResult(
                chunk_id=_make_chunk_id("metadata_aware", passage_index, 0),
                text=text.strip(),
                strategy=ChunkStrategy.METADATA_AWARE,
                metadata=ChunkMetadata(
                    chunk_index=0,
                    parent_passage_index=passage_index,
                    passage_id=metadata.get("passage_id"),
                    query_id=metadata.get("query_id"),
                    is_selected=metadata.get("is_selected"),
                    source_lang=metadata.get("source_lang"),
                    target_lang=metadata.get("target_lang"),
                ),
            )
        ]


# ── Factory ────────────────────────────────────────────────────────────────

def create_chunker(strategy: str) -> Chunker:
    """Create a chunker from a strategy name."""
    config = get_config()

    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=config.chunking.fixed.chunk_size,
            overlap_ratio=config.chunking.fixed.overlap_ratio,
        )
    elif strategy == "semantic":
        return SemanticChunker(
            similarity_threshold=config.chunking.semantic.similarity_threshold,
            min_chunk_size=config.chunking.semantic.min_chunk_size,
            max_chunk_size=config.chunking.semantic.max_chunk_size,
        )
    elif strategy == "sentence_window":
        return SentenceWindowChunker(
            anchor_sentences=config.chunking.sentence_window.anchor_sentences,
            window_sentences=config.chunking.sentence_window.window_sentences,
        )
    elif strategy == "metadata_aware":
        return MetadataAwareChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
