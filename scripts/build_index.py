"""
Build FAISS indices from preprocessed passages using all chunking strategies.
Embeds chunks and saves the index + metadata for each strategy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.models import ChunkStrategy
from app.stages.chunking import (
    FixedSizeChunker,
    MetadataAwareChunker,
    SemanticChunker,
    SentenceWindowChunker,
    create_chunker,
)
from app.stages.embedder import get_embedding_model


def build_index():
    """Run all chunking strategies, embed, and build FAISS indices."""
    config = load_config()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    index_dir = PROJECT_ROOT / "data" / "indices"
    chunks_dir = PROJECT_ROOT / "data" / "chunks"

    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load passages
    passages_path = raw_dir / "passages.json"
    if not passages_path.exists():
        print("ERROR: passages.json not found. Run download_dataset.py first.")
        sys.exit(1)

    print("Loading passages...")
    with open(passages_path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} passages")

    # Load embedding model
    print("\nLoading embedding model...")
    embedder = get_embedding_model()
    embedder.load()

    # Define chunking strategies
    strategies = {
        "fixed": create_chunker("fixed"),
        "semantic": create_chunker("semantic"),
        "sentence_window": create_chunker("sentence_window"),
        "metadata_aware": create_chunker("metadata_aware"),
    }

    # Set up SemanticChunker with the embedding function
    if isinstance(strategies["semantic"], SemanticChunker):
        strategies["semantic"].set_embed_fn(embedder.encode_sentences)

    # Process all strategies
    all_chunks = []
    strategy_stats = {}

    for strategy_name, chunker in strategies.items():
        print(f"\n{'='*60}")
        print(f"Chunking strategy: {strategy_name}")
        print(f"{'='*60}")

        strategy_chunks = []
        start = time.perf_counter()

        for i, passage in enumerate(tqdm(passages, desc=f"Chunking ({strategy_name})")):
            metadata = {
                "passage_id": passage.get("passage_id"),
                "query_id": passage.get("query_id"),
                "is_selected": passage.get("is_selected"),
                "source_lang": passage.get("source_lang"),
                "target_lang": passage.get("target_lang"),
            }
            chunks = chunker.chunk(
                text=passage["text"],
                passage_index=i,
                metadata=metadata,
            )
            strategy_chunks.extend(chunks)

        elapsed = time.perf_counter() - start
        strategy_stats[strategy_name] = {
            "chunk_count": len(strategy_chunks),
            "time_seconds": round(elapsed, 2),
            "avg_chunk_length": round(
                sum(len(c.text.split()) for c in strategy_chunks) / max(len(strategy_chunks), 1), 1
            ),
        }
        print(f"  Chunks: {len(strategy_chunks)}, Time: {elapsed:.1f}s")

        # Save strategy chunks
        chunks_path = chunks_dir / f"chunks_{strategy_name}.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in strategy_chunks], f, ensure_ascii=False)

        all_chunks.extend(strategy_chunks)

    print(f"\n{'='*60}")
    print(f"Total chunks across all strategies: {len(all_chunks)}")
    print(f"{'='*60}")

    # Embed all chunks
    print("\nEmbedding all chunks...")
    chunk_texts = [c.text for c in all_chunks]

    # Batch embed
    start = time.perf_counter()
    embeddings = embedder.encode_batch(chunk_texts, show_progress=True)
    embed_time = time.perf_counter() - start
    print(f"Embedded {len(embeddings)} chunks in {embed_time:.1f}s "
          f"({len(embeddings)/embed_time:.0f} chunks/sec)")

    # Build combined FAISS index
    print("\nBuilding FAISS index...")
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index
    faiss_path = index_dir / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index saved: {faiss_path} ({index.ntotal} vectors)")

    # Save chunk metadata
    metadata_path = index_dir / "chunks_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, ensure_ascii=False)
    print(f"Chunk metadata saved: {metadata_path}")

    # Save domain embeddings sample for off-topic detection
    # Take a random sample of 500 embeddings as "domain representatives"
    sample_size = min(500, len(embeddings))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(embeddings), sample_size, replace=False)
    domain_embeddings = embeddings[sample_indices]

    domain_path = index_dir / "domain_embeddings.npy"
    np.save(str(domain_path), domain_embeddings)
    print(f"Domain embeddings saved: {domain_path} ({sample_size} samples)")

    # Print summary
    print(f"\n{'='*60}")
    print("BUILD COMPLETE — Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Chunks':>10} {'Avg Words':>10} {'Time (s)':>10}")
    print("-" * 50)
    for name, stats in strategy_stats.items():
        print(f"{name:<20} {stats['chunk_count']:>10,} {stats['avg_chunk_length']:>10.1f} {stats['time_seconds']:>10.1f}")

    index_size_mb = os.path.getsize(faiss_path) / (1024 * 1024)
    print(f"\nFAISS index size: {index_size_mb:.1f} MB")
    print(f"Embedding dimension: {dimension}")
    print(f"Total vectors: {index.ntotal}")


if __name__ == "__main__":
    build_index()
