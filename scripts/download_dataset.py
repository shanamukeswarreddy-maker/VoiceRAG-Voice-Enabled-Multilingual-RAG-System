"""
Download and preprocess the MSMARCO-XI dataset (ai4bharat/MSMARCO-XI).
Extracts authentic Indic (Hindi) & English passage pairs with structured metadata.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config


def get_parquet_path() -> str:
    """Get path to hintrain.parquet (downloading from HF if needed)."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--ai4bharat--MSMARCO-XI"
    if cache_dir.exists():
        parquets = list(cache_dir.glob("**/train/hintrain.parquet"))
        if parquets:
            return str(parquets[0])

    print("Downloading hintrain.parquet from HuggingFace Hub...")
    return hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename="train/hintrain.parquet",
        repo_type="dataset",
    )


def _to_list(val):
    """Safely convert numpy array or list to python list."""
    if val is None:
        return []
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def download_dataset():
    """Extract authentic passages and queries from MSMARCO-XI parquet."""
    config = load_config()
    max_passages = config.dataset.max_passages  # default 25000

    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = get_parquet_path()
    print(f"Loading MSMARCO-XI dataset from: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    print(f"Total dataset examples in parquet: {pf.metadata.num_rows}")

    all_passages = []
    all_queries = []
    passage_count = 0
    batch_size = 500

    for batch in pf.iter_batches(batch_size=batch_size):
        if passage_count >= max_passages:
            break

        df = batch.to_pandas()
        for idx, row in df.iterrows():
            if passage_count >= max_passages:
                break

            q_id = str(row.get("query_id") or f"hi_{idx}")
            hi_query = str(row.get("query") or "").strip()
            eng_query = str(row.get("Eng_Query") or "").strip()
            hi_answer = str(row.get("Answer") or "").strip()
            eng_answer = str(row.get("Eng_Answer") or "").strip()
            query_type = str(row.get("query_type") or "DESCRIPTION")

            # Primary query choice
            primary_query = hi_query if hi_query else eng_query
            if not primary_query:
                continue

            all_queries.append({
                "query_id": q_id,
                "query": primary_query,
                "eng_query": eng_query,
                "query_type": query_type,
                "answers": [hi_answer] if hi_answer else ([eng_answer] if eng_answer else []),
                "well_formed_answers": [hi_answer] if hi_answer else [],
            })

            passages_dict = row.get("passages")
            if not isinstance(passages_dict, dict):
                continue

            hi_passages = _to_list(passages_dict.get("Translated_passages"))
            eng_passages = _to_list(passages_dict.get("English_passages"))
            is_selected_list = _to_list(passages_dict.get("is_selected"))

            # Process translated passages (Hindi)
            for p_idx, ptext in enumerate(hi_passages):
                if not ptext or not str(ptext).strip():
                    continue
                is_selected = int(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else 0

                all_passages.append({
                    "passage_id": f"{q_id}_hi_{p_idx}",
                    "query_id": q_id,
                    "text": str(ptext).strip(),
                    "is_selected": is_selected,
                    "source_lang": "eng_Latn",
                    "target_lang": "hi",
                })
                passage_count += 1

            # Process original English passages for cross-lingual support
            for p_idx, ptext in enumerate(eng_passages):
                if not ptext or not str(ptext).strip():
                    continue
                is_selected = int(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else 0

                all_passages.append({
                    "passage_id": f"{q_id}_en_{p_idx}",
                    "query_id": q_id,
                    "text": str(ptext).strip(),
                    "is_selected": is_selected,
                    "source_lang": "eng_Latn",
                    "target_lang": "eng_Latn",
                })
                passage_count += 1

        print(f"  Processed batch: {len(all_passages)} passages, {len(all_queries)} queries", flush=True)

    # Save passages
    passages_path = output_dir / "passages.json"
    print(f"\nSaving {len(all_passages)} passages to {passages_path}")
    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(all_passages, f, ensure_ascii=False)

    # Save queries
    queries_path = output_dir / "queries.json"
    print(f"Saving {len(all_queries)} queries to {queries_path}")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False)

    # Save benchmark sample
    random.seed(42)
    sample_size = min(config.dataset.validation_sample, len(all_queries))
    sample_queries = random.sample(all_queries, sample_size)

    sample_path = output_dir / "benchmark_queries.json"
    print(f"Saving {len(sample_queries)} benchmark queries to {sample_path}")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_queries, f, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"MSMARCO-XI Dataset Preprocessing Complete!")
    print(f"  Passages: {len(all_passages)}")
    print(f"  Queries: {len(all_queries)}")
    print(f"  Benchmark sample: {len(sample_queries)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    download_dataset()
