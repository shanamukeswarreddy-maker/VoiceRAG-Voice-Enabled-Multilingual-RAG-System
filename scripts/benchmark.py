"""
Latency benchmark harness.
Runs warm-up phase (5-10 discarded queries) then 100 benchmark queries across all 4 strategies.
Reports P50/P70/P100 per-stage latencies, flags suspect <5ms fast-paths/failures,
and outputs markdown reports and matplotlib charts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def run_benchmark():
    """Run the latency benchmark on sampled queries."""
    from app.config import load_config, load_secrets
    from app.models import ChunkStrategy, StageStatus
    from app.stages.embedder import get_embedding_model
    from app.stages.guardrails import get_guardrail_engine
    from app.stages.retriever import get_retriever
    from app.pipeline import RAGPipeline

    config = load_config()
    load_secrets()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark queries
    queries_path = PROJECT_ROOT / "data" / "raw" / "benchmark_queries.json"
    if not queries_path.exists():
        print("ERROR: benchmark_queries.json not found. Run download_dataset.py first.")
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} benchmark queries")

    # Initialize components
    print("\n" + "=" * 60)
    print("INITIALIZING BENCHMARK ENVIRONMENT")
    print("=" * 60)
    embedder = get_embedding_model()
    embedder.load()

    retriever = get_retriever()
    retriever.load()

    # Load domain embeddings for off-topic detection
    domain_path = PROJECT_ROOT / "data" / "indices" / "domain_embeddings.npy"
    if domain_path.exists():
        domain_emb = np.load(str(domain_path))
        get_guardrail_engine().set_domain_embeddings(domain_emb)

    pipeline = RAGPipeline()
    strategies = retriever.available_strategies or ["fixed", "semantic", "sentence_window", "metadata_aware"]

    # ── WARM-UP PHASE (5-10 discarded requests to eliminate cold-start) ────
    print("\n" + "=" * 60)
    print("WARM-UP PHASE: Running 8 discarded queries to eliminate cold-start contamination...")
    print("=" * 60)
    warmup_queries = queries[:8]
    for wq in warmup_queries:
        try:
            _ = await pipeline.process_text(query=wq.get("query", "warmup"), strategy=ChunkStrategy.SEMANTIC)
        except Exception as e:
            print(f"Warm-up query notice: {e}")
    print("Warm-up complete. All JIT caches & connections primed.")

    # Run benchmark for each strategy
    all_results = {}
    all_fast_paths = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Benchmarking strategy: {strategy}")
        print(f"{'='*60}")

        # Clear query embedding cache before each strategy run for realistic, un-cached query measurement
        embedder.clear_cache()

        latencies = {
            "embedding": [],
            "retrieval": [],
            "generation": [],
            "guardrail_pre": [],
            "guardrail_post": [],
            "pipeline_total": [],
            "fallback_triggered": [],
        }
        fast_paths = []

        try:
            chunk_strategy = ChunkStrategy(strategy)
        except ValueError:
            chunk_strategy = ChunkStrategy.SEMANTIC

        for i, q in enumerate(queries):
            query_text = q.get("query", "")
            if not query_text:
                continue

            wall_start = time.perf_counter()
            try:
                response = await pipeline.process_text(
                    query=query_text,
                    strategy=chunk_strategy,
                    top_k=config.retrieval.top_k,
                )
                wall_ms = round((time.perf_counter() - wall_start) * 1000, 2)

                # Check for fast-paths or errors (<5ms or status != SUCCESS)
                if wall_ms < 5.0 or response.status in [StageStatus.FILTERED, StageStatus.ERROR]:
                    fast_paths.append({
                        "query_index": i,
                        "query": query_text,
                        "status": response.status,
                        "latency_ms": wall_ms,
                        "answer": response.answer[:50],
                    })
                    # Do not include zero-latency short-circuits in core pipeline metrics
                    continue

                # Record fallback triggered boolean (1 for True, 0 for False)
                fallback_flag = getattr(response, "fallback_triggered", False)
                latencies["fallback_triggered"].append(1 if fallback_flag else 0)

                # Include valid full pipeline execution timings
                for stage_name in latencies:
                    if stage_name == "pipeline_total":
                        latencies[stage_name].append(wall_ms)
                    elif stage_name in response.stage_latencies:
                        latencies[stage_name].append(response.stage_latencies[stage_name])

                if (i + 1) % 25 == 0:
                    avg_pipeline = np.mean(latencies["pipeline_total"][-25:]) if latencies["pipeline_total"] else 0.0
                    print(f"  Query {i+1}/{len(queries)} | Avg pipeline wall latency: {avg_pipeline:.1f}ms")

            except Exception as e:
                print(f"  Query {i+1} ERROR: {e}")
                continue

        all_results[strategy] = latencies
        all_fast_paths[strategy] = fast_paths
        fb_rate = (sum(latencies["fallback_triggered"]) / max(len(latencies["fallback_triggered"]), 1)) * 100
        print(f"Strategy {strategy} benchmarked: {len(latencies['pipeline_total'])} queries processed | "
              f"Fallback rate: {fb_rate:.1f}% | {len(fast_paths)} fast-path/filtered queries logged.")

    # Generate reports and charts
    generate_report(all_results, all_fast_paths, results_dir)
    generate_strategy_comparison(all_results, results_dir)
    generate_charts(all_results, results_dir)

    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to {results_dir}")
    print(f"{'='*60}")


def percentile(data: list, p: float) -> float:
    """Compute percentile from a list of values."""
    if not data:
        return 0.0
    return float(np.percentile(data, p))


def generate_report(all_results: dict, all_fast_paths: dict, results_dir: Path):
    """Generate the latency report markdown."""
    report_lines = [
        "# Latency Benchmark Report",
        "",
        "> **Note**: STT latency (Sarvam API) is measured and reported separately from pipeline numbers.",
        "> Pipeline = embedding + retrieval + generation + guardrail checks.",
        "> Warm-up phase ran 8 discarded queries prior to measurement.",
        "",
    ]

    for strategy, latencies in all_results.items():
        report_lines.append(f"## Strategy: `{strategy}`")
        report_lines.append("")
        valid_count = len(latencies['pipeline_total'])
        fast_count = len(all_fast_paths.get(strategy, []))
        fb_list = latencies.get("fallback_triggered", [])
        fb_count = sum(fb_list)
        fb_pct = (fb_count / len(fb_list) * 100) if fb_list else 0.0

        report_lines.append(f"Full pipeline queries measured: {valid_count} (Fast-paths / filtered: {fast_count})")
        report_lines.append(f"**Fallback rate (timeout/error cutoff)**: {fb_count}/{valid_count} queries ({fb_pct:.1f}%)")
        report_lines.append("")

        # Build table
        report_lines.append("| Stage | P50 (ms) | P70 (ms) | P100 / Max (ms) | Mean (ms) |")
        report_lines.append("|-------|----------|----------|------------------|-----------|")

        for stage_name in ["embedding", "retrieval", "generation", "guardrail_pre", "guardrail_post", "pipeline_total"]:
            data = latencies.get(stage_name, [])
            if not data:
                continue
            p50 = percentile(data, 50)
            p70 = percentile(data, 70)
            p100 = percentile(data, 100)
            mean = np.mean(data)
            display_name = stage_name.replace("_", " ").title()
            report_lines.append(
                f"| {display_name} | {p50:.1f} | {p70:.1f} | {p100:.1f} | {mean:.1f} |"
            )

        report_lines.append("")
        pipeline_data = latencies.get("pipeline_total", [])
        if pipeline_data:
            under_200 = sum(1 for x in pipeline_data if x < 200)
            pct = (under_200 / len(pipeline_data)) * 100
            report_lines.append(
                f"**{under_200}/{len(pipeline_data)} queries ({pct:.1f}%) completed under 200ms target**"
            )
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    report_path = results_dir / "latency_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved: {report_path}")


def generate_strategy_comparison(all_results: dict, results_dir: Path):
    """Generate strategy comparison markdown table."""
    lines = [
        "# Strategy Latency & Performance Comparison",
        "",
        "| Strategy | Default | P50 (ms) | P70 (ms) | P100 (ms) | Under 200ms % | Fallback Rate % |",
        "|----------|---------|----------|----------|-----------|---------------|-----------------|",
    ]

    for strategy, latencies in all_results.items():
        data = latencies.get("pipeline_total", [])
        if not data:
            continue
        p50 = percentile(data, 50)
        p70 = percentile(data, 70)
        p100 = percentile(data, 100)
        under_200 = sum(1 for x in data if x < 200)
        pct = (under_200 / len(data)) * 100
        fb_list = latencies.get("fallback_triggered", [])
        fb_rate = (sum(fb_list) / max(len(fb_list), 1)) * 100
        is_default = "YES" if strategy == "semantic" else "No"
        lines.append(f"| `{strategy}` | {is_default} | {p50:.1f} | {p70:.1f} | {p100:.1f} | {pct:.1f}% | {fb_rate:.1f}% |")

    comp_path = results_dir / "strategy_comparison.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Comparison saved: {comp_path}")


def generate_charts(all_results: dict, results_dir: Path):
    """Generate latency distribution charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("VoiceRAG Pipeline Latency Benchmark", fontsize=14, fontweight="bold")

        # Chart 1: Pipeline total by strategy
        ax1 = axes[0]
        strategy_names = []
        strategy_data = []
        for strategy, latencies in all_results.items():
            data = latencies.get("pipeline_total", [])
            if data:
                strategy_names.append(strategy)
                strategy_data.append(data)

        if strategy_data:
            bp = ax1.boxplot(strategy_data, labels=strategy_names, patch_artist=True)
            colors = ["#4ecdc4", "#ff6b6b", "#ffe66d", "#a8e6cf"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.axhline(y=200, color="red", linestyle="--", alpha=0.7, label="200ms target")
            ax1.set_ylabel("Latency (ms)")
            ax1.set_title("Pipeline Total Latency by Strategy")
            ax1.legend()

        # Chart 2: Per-stage breakdown (semantic strategy)
        ax2 = axes[1]
        target_strat = "semantic" if "semantic" in all_results else (list(all_results.keys())[0] if all_results else None)
        if target_strat:
            latencies = all_results[target_strat]
            stage_names = []
            stage_data = []
            for stage in ["embedding", "retrieval", "generation", "guardrail_pre", "guardrail_post"]:
                data = latencies.get(stage, [])
                if data:
                    stage_names.append(stage.replace("_", "\n"))
                    stage_data.append(data)

            if stage_data:
                bp2 = ax2.boxplot(stage_data, labels=stage_names, patch_artist=True)
                stage_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
                for patch, color in zip(bp2["boxes"], stage_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax2.set_ylabel("Latency (ms)")
                ax2.set_title(f"Per-Stage Breakdown ({target_strat})")

        plt.tight_layout()
        chart_path = results_dir / "latency_chart.png"
        plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved: {chart_path}")

    except ImportError:
        print("WARNING: matplotlib not installed, skipping charts")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
