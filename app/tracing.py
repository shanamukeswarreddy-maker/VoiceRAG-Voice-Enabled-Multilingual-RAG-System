"""
Request-level tracing and per-stage timing.
Every request gets a unique ID and structured timing logs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_config, PROJECT_ROOT

logger = logging.getLogger("rag.tracing")


class RequestTrace:
    """
    Accumulates per-stage timings and metadata for a single request.

    Usage:
        trace = RequestTrace()
        with trace.stage("embedding"):
            embedding = model.encode(query)
        with trace.stage("retrieval"):
            results = index.search(embedding)
        trace.log()
    """

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())[:12]
        self.start_time = time.perf_counter()
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.errors: List[Dict[str, str]] = []
        self._current_stage: Optional[str] = None

    @contextmanager
    def stage(self, name: str):
        """Context manager that times a pipeline stage."""
        self._current_stage = name
        stage_start = time.perf_counter()
        stage_data: Dict[str, Any] = {"status": "running"}

        try:
            yield stage_data
            stage_data["status"] = "success"
        except Exception as e:
            stage_data["status"] = "error"
            stage_data["error"] = str(e)
            self.errors.append({"stage": name, "error": str(e)})
            raise
        finally:
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            stage_data["latency_ms"] = round(elapsed_ms, 2)
            self.stages[name] = stage_data
            self._current_stage = None

    def get_stage_latency(self, name: str) -> float:
        """Get latency for a specific stage in milliseconds."""
        return self.stages.get(name, {}).get("latency_ms", 0.0)

    def get_pipeline_latency(self) -> float:
        """Get total pipeline latency (excluding STT)."""
        pipeline_stages = ["embedding", "retrieval", "generation", "guardrail_pre", "guardrail_offtopic", "guardrail_post"]
        return sum(self.get_stage_latency(s) for s in pipeline_stages)

    def get_total_latency(self) -> float:
        """Get total elapsed time from trace creation."""
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def get_all_latencies(self) -> Dict[str, float]:
        """Get a dict of all stage latencies."""
        return {name: data.get("latency_ms", 0.0) for name, data in self.stages.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trace to a dict for logging."""
        return {
            "request_id": self.request_id,
            "total_ms": self.get_total_latency(),
            "pipeline_ms": self.get_pipeline_latency(),
            "stages": self.stages,
            "errors": self.errors,
        }

    def log(self) -> None:
        """Write the trace to the log file and logger."""
        config = get_config()
        trace_data = self.to_dict()
        logger.info(f"Request {self.request_id} | pipeline={trace_data['pipeline_ms']:.1f}ms | "
                     f"stages={json.dumps(self.get_all_latencies())}")

        if config.tracing.enabled:
            log_path = PROJECT_ROOT / config.tracing.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data) + "\n")
