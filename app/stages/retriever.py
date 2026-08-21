"""
FAISS-based vector retriever.
Loads index and chunk metadata into memory at startup for zero cold-start.
Maintains per-strategy FAISS sub-indices for pre-search strategy filtering.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.config import get_config, PROJECT_ROOT
from app.models import ChunkMetadata, ChunkResult, ChunkStrategy, RetrievalResult, RetrievedChunk

logger = logging.getLogger("rag.retriever")


class FAISSRetriever:
    """
    In-memory FAISS retriever with:
    - Pre-loaded per-strategy sub-indices at startup
    - True pre-search strategy filtering (never search all and discard)
    - Sentence-window context expansion
    """

    def __init__(self):
        self.config = get_config().retrieval
        self._global_index: Optional[faiss.Index] = None
        self._global_chunks: List[ChunkResult] = []
        # Per-strategy partitioned sub-indices for zero-overhead pre-search filtering
        self._strategy_faiss: Dict[str, faiss.Index] = {}
        self._strategy_chunks: Dict[str, List[ChunkResult]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def index_size(self) -> int:
        return len(self._global_chunks)

    @property
    def available_strategies(self) -> List[str]:
        return list(self._strategy_faiss.keys())

    def load(self, index_dir: Optional[str] = None) -> None:
        """Load FAISS index and chunk metadata from disk, building per-strategy sub-indices."""
        if self._loaded:
            return

        if index_dir is None:
            index_dir = str(PROJECT_ROOT / "data" / "indices")

        index_path = Path(index_dir) / "faiss_index.bin"
        metadata_path = Path(index_dir) / "chunks_metadata.json"

        dim = self.config.get_config().embedding.dimension if hasattr(self.config, "get_config") else 384

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Index files not found at {index_dir} — retriever will be empty")
            self._global_index = faiss.IndexFlatIP(dim)
            self._global_chunks = []
            self._loaded = True
            return

        logger.info(f"Loading FAISS index from {index_path}")
        start = time.perf_counter()

        self._global_index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)

        self._global_chunks = [ChunkResult(**c) for c in chunks_raw]

        # Extract vectors for each strategy to build per-strategy sub-indices
        # PRE-SEARCH FILTERING: Search is restricted to ONLY the selected strategy's sub-index
        num_vectors = self._global_index.ntotal
        dimension = self._global_index.d

        # Reconstruct all vectors from global index if possible
        all_vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
        for i in range(num_vectors):
            all_vectors[i] = self._global_index.reconstruct(i)

        self._strategy_faiss = {}
        self._strategy_chunks = {}

        strategy_vector_map: Dict[str, List[np.ndarray]] = {}

        for i, chunk in enumerate(self._global_chunks):
            strat = chunk.strategy.value
            if strat not in self._strategy_chunks:
                self._strategy_chunks[strat] = []
                strategy_vector_map[strat] = []

            self._strategy_chunks[strat].append(chunk)
            strategy_vector_map[strat].append(all_vectors[i])

        for strat, vec_list in strategy_vector_map.items():
            sub_idx = faiss.IndexFlatIP(dimension)
            vec_arr = np.array(vec_list, dtype=np.float32)
            sub_idx.add(vec_arr)
            self._strategy_faiss[strat] = sub_idx

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Index loaded & partitioned: {self._global_index.ntotal} vectors, "
            f"strategies={list(self._strategy_faiss.keys())} in {elapsed:.0f}ms"
        )
        self._loaded = True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        strategy_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Search the strategy-partitioned FAISS index.

        PRE-SEARCH FILTERING ENFORCED:
        Retrieval filters to the given strategy's sub-index BEFORE similarity search,
        never searching all strategies and discarding, avoiding 4x query-time overhead.
        """
        if not self._loaded:
            self.load()

        top_k = top_k or self.config.top_k
        start = time.perf_counter()

        # Reshape for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Select target index and chunk list based on strategy filter
        target_strategy = strategy_filter or "semantic"
        index_to_search = self._strategy_faiss.get(target_strategy, self._global_index)
        chunks_to_search = self._strategy_chunks.get(target_strategy, self._global_chunks)

        if index_to_search is None or index_to_search.ntotal == 0:
            return RetrievalResult(chunks=[], latency_ms=0.0)

        search_k = min(top_k, index_to_search.ntotal)
        scores, indices = index_to_search.search(query, search_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks_to_search):
                continue

            chunk = chunks_to_search[idx]

            # Expand sentence-window chunks to window_context at retrieval time
            expanded_text = None
            if chunk.strategy == ChunkStrategy.SENTENCE_WINDOW and chunk.metadata.window_context:
                expanded_text = chunk.metadata.window_context

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    expanded_text=expanded_text,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            chunks=results,
            latency_ms=round(elapsed_ms, 2),
        )


# ── Singleton ──────────────────────────────────────────────────────────────

_retriever: Optional[FAISSRetriever] = None


def get_retriever() -> FAISSRetriever:
    """Get or create the singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
