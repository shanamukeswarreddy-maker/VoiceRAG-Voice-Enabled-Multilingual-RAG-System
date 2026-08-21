import httpx
import json

url = "http://127.0.0.1:8000/api/query/text"

test_cases = [
    ("What is photosynthesis and how does it work?", "fixed"),
    ("भारत की राजधानी क्या है?", "semantic"),
    ("What causes earthquakes and how are they measured?", "sentence_window"),
    ("सौर मंडल का सबसे बड़ा ग्रह कौन सा है?", "metadata_aware"),
]

print("=" * 70)
print("LIVE VOICE-RAG PIPELINE RESPONSE TIME MEASUREMENTS")
print("=" * 70)

for query, strategy in test_cases:
    payload = {"query": query, "strategy": strategy, "top_k": 5}
    response = httpx.post(url, json=payload, timeout=10.0).json()
    
    total_ms = response.get("pipeline_latency_ms", 0.0)
    stages = response.get("stage_latencies", {})
    status = response.get("status")
    
    query_safe = query.encode("ascii", "replace").decode("ascii") if not query.isascii() else query
    print(f"\nQuery: {query_safe}")
    print(f"Strategy: {strategy}")
    print(f"Total Pipeline Latency: {total_ms:.2f} ms {'[PASS <200ms]' if total_ms < 200 else '[FAIL]'}")
    print("Stage Breakdown:")
    for stage, ms in stages.items():
        print(f"  • {stage:<15}: {ms:.2f} ms")

print("\n" + "=" * 70)
