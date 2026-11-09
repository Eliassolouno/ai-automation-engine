"""
In-Memory & JSONL Telemetry Tracer conforming to TelemetryPort.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone


class InMemoryTracer:
    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self.token_costs: Dict[str, int] = {}

    def record_step(self, run_id: str, step_data: Dict[str, Any]) -> None:
        self.traces.append({
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step_data
        })

    def record_cost(self, run_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        total = prompt_tokens + completion_tokens
        self.token_costs[run_id] = self.token_costs.get(run_id, 0) + total
