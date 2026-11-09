"""
Tool Executor implementing ToolExecutorPort.
Handles validation, timeout bounds, and output normalization.
"""

import time
import inspect
import asyncio
from typing import List, Dict, Any
from aae.types import ToolCall, ToolResponse
from aae.tools.spec import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.registry.get_all_schemas()

    async def execute(self, tool_call: ToolCall) -> ToolResponse:
        start_time = time.monotonic()
        spec = self.registry.get(tool_call.tool_name)

        if not spec:
            duration = (time.monotonic() - start_time) * 1000
            return ToolResponse(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Error: Tool '{tool_call.tool_name}' is not registered.",
                is_error=True,
                execution_time_ms=round(duration, 2)
            )

        try:
            handler = spec.handler
            args = tool_call.arguments

            if inspect.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(**args), timeout=30.0)
            else:
                result = await asyncio.to_thread(handler, **args)

            duration = (time.monotonic() - start_time) * 1000
            return ToolResponse(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=result,
                is_error=False,
                execution_time_ms=round(duration, 2)
            )
        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000
            return ToolResponse(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=f"Execution error in {tool_call.tool_name}: {str(e)}",
                is_error=True,
                execution_time_ms=round(duration, 2)
            )
