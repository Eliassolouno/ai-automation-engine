"""
Kernel Ports: Abstract contracts that isolate core agent reasoning from providers and tools.
Following strict Hexagonal Architecture / Ports & Adapters.
"""

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
from aae.types import ToolCall, ToolResponse, RunBudget


@runtime_checkable
class LLMPort(Protocol):
    """Port for interacting with language models without vendor lock-in."""
    async def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        budget: Optional[RunBudget] = None
    ) -> Dict[str, Any]:
        ...


@runtime_checkable
class ToolExecutorPort(Protocol):
    """Port for executing verified tools and returning normalized outcomes."""
    async def execute(self, tool_call: ToolCall) -> ToolResponse:
        ...

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        ...


@runtime_checkable
class TelemetryPort(Protocol):
    """Port for tracing agent steps, tokens, and errors."""
    def record_step(self, run_id: str, step_data: Dict[str, Any]) -> None:
        ...

    def record_cost(self, run_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        ...
