"""
Edge-Case Unit Tests for AI Automation Engine (AAE).
Covers:
1. aae.kernel.loop (empty budgets, budget exhaustion, execution errors, timeout simulation).
2. aae.tools.executor (unregistered tools, malformed tool arguments, execution failures).
3. aae.gateway.router (fallback routing, empty prompt handlers).
4. aae.kernel.budget & aae.telemetry.tracing edge conditions.
"""

import asyncio
from unittest.mock import patch
import pytest

from aae.types import RunBudget, ExecutionStatus, ToolCall
from aae.kernel.loop import AgentLoop
from aae.kernel.budget import BudgetEnforcer, BudgetExceededError
from aae.tools.spec import ToolRegistry
from aae.tools.executor import ToolExecutor
from aae.gateway.router import MockModelGateway
from aae.telemetry.tracing import InMemoryTracer


# ============================================================================
# 1. Tests for aae.kernel.loop
# ============================================================================

class TestAgentLoopEdgeCases:

    @pytest.mark.asyncio
    async def test_loop_with_empty_budget_none(self):
        """When budget is None, AgentLoop should instantiate default RunBudget and succeed."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        llm = MockModelGateway(canned_responses=[
            {"content": "Completed with default budget.", "tool_calls": [], "tokens_used": 50}
        ])
        loop = AgentLoop(llm=llm, tools=executor, telemetry=None)

        result = await loop.run("Task with default budget", budget=None)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_answer == "Completed with default budget."
        assert len(result.steps) == 1
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_loop_with_zero_max_steps_budget(self):
        """When max_steps is 0, the execution loop ends immediately with step limit message."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        llm = MockModelGateway()
        loop = AgentLoop(llm=llm, tools=executor)

        zero_budget = RunBudget(max_steps=0)
        result = await loop.run("Task with zero steps", budget=zero_budget)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_answer == "Max step limit reached without explicit completion."
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_loop_max_step_limit_reached_without_completion(self):
        """When steps reach max_steps while model continues issuing tool calls, it stops cleanly."""
        registry = ToolRegistry()

        @registry.register(name="noop", description="No operation")
        def noop():
            return "done"

        executor = ToolExecutor(registry)
        # 3 canned responses all requesting tool calls, but budget is only 2 steps
        canned = [
            {
                "content": f"Step {i}",
                "tool_calls": [{"id": f"call_{i}", "name": "noop", "arguments": {}}],
                "tokens_used": 20
            }
            for i in range(1, 4)
        ]
        llm = MockModelGateway(canned_responses=canned)
        tracer = InMemoryTracer()
        loop = AgentLoop(llm=llm, tools=executor, telemetry=tracer)

        budget = RunBudget(max_steps=2)
        result = await loop.run("Perpetual task", budget=budget)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_answer == "Max step limit reached without explicit completion."
        assert len(result.steps) == 2
        assert len(tracer.traces) == 2

    @pytest.mark.asyncio
    async def test_loop_budget_exhaustion_tokens(self):
        """When token consumption exceeds max_tokens, execution fails with BudgetExceededError."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        canned = [
            {
                "content": "Expensive turn",
                "tool_calls": [{"id": "c1", "name": "dummy", "arguments": {}}],
                "tokens_used": 5000
            }
        ]
        llm = MockModelGateway(canned_responses=canned)
        loop = AgentLoop(llm=llm, tools=executor)

        tight_token_budget = RunBudget(max_steps=5, max_tokens=1000)
        result = await loop.run("Expensive task", budget=tight_token_budget)

        assert result.status == ExecutionStatus.FAILED
        assert "Aborted:" in result.final_answer
        assert "Token budget exceeded" in result.final_answer

    @pytest.mark.asyncio
    async def test_loop_timeout_simulation(self):
        """Simulate wallclock timeout triggering BudgetExceededError."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        llm = MockModelGateway(canned_responses=[
            {"content": "Step 1", "tool_calls": [], "tokens_used": 10}
        ])
        loop = AgentLoop(llm=llm, tools=executor)

        budget = RunBudget(max_steps=5, max_wallclock_seconds=1.0)

        # Mock time.monotonic so the second check exceeds 1.0 second
        start_time = 1000.0
        time_sequence = [start_time, start_time, start_time + 5.0, start_time + 5.0]

        with patch("time.monotonic", side_effect=time_sequence):
            result = await loop.run("Timeout task", budget=budget)

        assert result.status == ExecutionStatus.FAILED
        assert "Aborted:" in result.final_answer
        assert "Execution timeout" in result.final_answer

    @pytest.mark.asyncio
    async def test_loop_execution_error_llm_failure(self):
        """When LLM port raises an unhandled exception, loop returns FAILED status gracefully."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        class FaultyLLM:
            async def generate_response(self, *args, **kwargs):
                raise ConnectionResetError("Remote inference provider connection severed")

        loop = AgentLoop(llm=FaultyLLM(), tools=executor)
        result = await loop.run("Task doomed to fail")

        assert result.status == ExecutionStatus.FAILED
        assert "Error: Remote inference provider connection severed" in result.final_answer
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_loop_handles_tool_execution_failure_in_step(self):
        """When a tool execution produces an error, loop captures it and carries forward."""
        registry = ToolRegistry()

        @registry.register(name="crashing_tool", description="Always fails")
        def crashing_tool():
            raise RuntimeError("Database connection unreachable")

        executor = ToolExecutor(registry)
        canned = [
            {
                "content": "Trying crashing tool",
                "tool_calls": [{"id": "call_err", "name": "crashing_tool", "arguments": {}}],
                "tokens_used": 50
            },
            {
                "content": "Observed failure and recovered with explanation.",
                "tool_calls": [],
                "tokens_used": 30
            }
        ]
        llm = MockModelGateway(canned_responses=canned)
        loop = AgentLoop(llm=llm, tools=executor)

        result = await loop.run("Execute resiliently")

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_answer == "Observed failure and recovered with explanation."
        assert len(result.steps) == 2
        first_step_resp = result.steps[0].tool_responses[0]
        assert first_step_resp.is_error is True
        assert "Database connection unreachable" in str(first_step_resp.output)


# ============================================================================
# 2. Tests for aae.tools.executor
# ============================================================================

class TestToolExecutorEdgeCases:

    @pytest.mark.asyncio
    async def test_unregistered_tool(self):
        """Executing a tool that is not in the registry returns an error ToolResponse."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        call = ToolCall(call_id="call_unreg_1", tool_name="ghost_tool", arguments={"val": 42})
        resp = await executor.execute(call)

        assert resp.call_id == "call_unreg_1"
        assert resp.tool_name == "ghost_tool"
        assert resp.is_error is True
        assert "Error: Tool 'ghost_tool' is not registered." in resp.output
        assert resp.execution_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_malformed_arguments_missing_required(self):
        """Calling a registered tool with missing required positional arguments returns is_error=True."""
        registry = ToolRegistry()

        @registry.register(name="strict_calculator", description="Multiplies two numbers")
        def strict_calculator(x: int, y: int) -> int:
            return x * y

        executor = ToolExecutor(registry)
        # Provide only 'x', omitting required 'y'
        call = ToolCall(call_id="call_missing_arg", tool_name="strict_calculator", arguments={"x": 10})
        resp = await executor.execute(call)

        assert resp.is_error is True
        assert "Execution error in strict_calculator:" in resp.output
        assert "missing 1 required positional argument" in resp.output

    @pytest.mark.asyncio
    async def test_malformed_arguments_unexpected_extra(self):
        """Calling a registered tool with unexpected keyword arguments returns is_error=True."""
        registry = ToolRegistry()

        @registry.register(name="echo_tool", description="Echoes text")
        def echo_tool(text: str) -> str:
            return text

        executor = ToolExecutor(registry)
        call = ToolCall(
            call_id="call_unexpected_arg",
            tool_name="echo_tool",
            arguments={"text": "hello", "unknown_param": 123}
        )
        resp = await executor.execute(call)

        assert resp.is_error is True
        assert "Execution error in echo_tool:" in resp.output
        assert "unexpected keyword argument" in resp.output

    @pytest.mark.asyncio
    async def test_execution_failure_sync_tool_raises(self):
        """When a sync tool raises an exception during execution, executor captures it as error."""
        registry = ToolRegistry()

        @registry.register(name="faulty_sync", description="Raises ValueError")
        def faulty_sync(data: str):
            raise ValueError(f"Corrupted payload: {data}")

        executor = ToolExecutor(registry)
        call = ToolCall(call_id="c_fail_sync", tool_name="faulty_sync", arguments={"data": "test_bytes"})
        resp = await executor.execute(call)

        assert resp.is_error is True
        assert "Corrupted payload: test_bytes" in resp.output

    @pytest.mark.asyncio
    async def test_async_tool_success_and_failure(self):
        """Async tools are awaited properly and errors are captured."""
        registry = ToolRegistry()

        @registry.register(name="async_multiply", description="Asynchronously multiplies numbers")
        async def async_multiply(a: int, b: int) -> int:
            await asyncio.sleep(0.001)
            return a * b

        @registry.register(name="async_faulty", description="Asynchronously raises an exception")
        async def async_faulty():
            await asyncio.sleep(0.001)
            raise ZeroDivisionError("Division by zero in async task")

        executor = ToolExecutor(registry)

        # 1. Success case
        success_call = ToolCall(call_id="c_async_ok", tool_name="async_multiply", arguments={"a": 6, "b": 7})
        resp_ok = await executor.execute(success_call)
        assert resp_ok.is_error is False
        assert resp_ok.output == 42

        # 2. Failure case
        fail_call = ToolCall(call_id="c_async_fail", tool_name="async_faulty", arguments={})
        resp_fail = await executor.execute(fail_call)
        assert resp_fail.is_error is True
        assert "Division by zero in async task" in resp_fail.output

    @pytest.mark.asyncio
    async def test_async_tool_timeout(self):
        """Simulate async tool timeout triggering TimeoutError."""
        registry = ToolRegistry()

        @registry.register(name="slow_async", description="Simulates a long running async tool")
        async def slow_async():
            pass

        executor = ToolExecutor(registry)
        call = ToolCall(call_id="c_async_timeout", tool_name="slow_async", arguments={})

        async def fake_wait_for(coro, timeout=30.0):
            coro.close()
            raise asyncio.TimeoutError("Operation timed out")

        with patch("asyncio.wait_for", side_effect=fake_wait_for):
            resp = await executor.execute(call)

        assert resp.is_error is True
        assert "Operation timed out" in resp.output

    def test_executor_get_tool_schemas(self):
        """ToolExecutor.get_tool_schemas retrieves registered tool schemas correctly."""
        registry = ToolRegistry()

        @registry.register(name="sample_tool", description="Sample description")
        def sample_tool(param: str) -> str:
            return param

        executor = ToolExecutor(registry)
        schemas = executor.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "sample_tool"
        assert schemas[0]["function"]["description"] == "Sample description"


# ============================================================================
# 3. Tests for aae.gateway.router
# ============================================================================

class TestGatewayRouterEdgeCases:

    @pytest.mark.asyncio
    async def test_fallback_routing_with_no_canned_responses(self):
        """When initialized without canned responses, gateway returns default fallback response."""
        gateway = MockModelGateway(canned_responses=[])
        response = await gateway.generate_response(
            system_prompt="You are an automated code assistant.",
            messages=[{"role": "user", "content": "Hello world"}]
        )

        assert "Resolved step with system prompt: You are an automated" in response["content"]
        assert response["tool_calls"] == []
        assert response["tokens_used"] == 150

    @pytest.mark.asyncio
    async def test_fallback_routing_after_canned_responses_exhausted(self):
        """When canned responses run out, subsequent calls transition to fallback response."""
        canned = [
            {"content": "First response", "tool_calls": [], "tokens_used": 25}
        ]
        gateway = MockModelGateway(canned_responses=canned)

        resp1 = await gateway.generate_response(system_prompt="System", messages=[])
        assert resp1["content"] == "First response"
        assert resp1["tokens_used"] == 25

        resp2 = await gateway.generate_response(system_prompt="System", messages=[])
        assert "Resolved step with system prompt: System..." in resp2["content"]
        assert resp2["tokens_used"] == 150

    @pytest.mark.asyncio
    async def test_empty_prompt_handlers(self):
        """Gateway handles empty system prompt, empty messages, None tools schema, and None budget."""
        gateway = MockModelGateway()

        # Empty system prompt: slice [:20] handles empty string without IndexError
        resp_empty_prompt = await gateway.generate_response(
            system_prompt="",
            messages=[],
            tools_schema=None,
            budget=None
        )
        assert resp_empty_prompt["content"] == "Resolved step with system prompt: ..."
        assert resp_empty_prompt["tool_calls"] == []
        assert resp_empty_prompt["tokens_used"] == 150

        # Short prompt under 20 chars
        resp_short_prompt = await gateway.generate_response(
            system_prompt="Short",
            messages=[]
        )
        assert "Resolved step with system prompt: Short..." in resp_short_prompt["content"]


# ============================================================================
# 4. Tests for aae.kernel.budget & aae.telemetry.tracing edge conditions
# ============================================================================

class TestBudgetAndTelemetryEdgeCases:

    def test_budget_enforcer_step_exhaustion(self):
        """BudgetEnforcer raises BudgetExceededError when current_step exceeds max_steps."""
        budget = RunBudget(max_steps=3)
        enforcer = BudgetEnforcer(budget)

        enforcer.check_and_update(1)
        enforcer.check_and_update(2)
        enforcer.check_and_update(3)

        with pytest.raises(BudgetExceededError, match="Agent step limit exceeded: 4 > 3"):
            enforcer.check_and_update(4)

    def test_budget_enforcer_token_exhaustion(self):
        """BudgetEnforcer raises BudgetExceededError when consumed_tokens exceeds max_tokens."""
        budget = RunBudget(max_steps=10, max_tokens=100)
        enforcer = BudgetEnforcer(budget)

        enforcer.check_and_update(1, tokens_used=80)
        assert budget.consumed_tokens == 80

        with pytest.raises(BudgetExceededError, match="Token budget exceeded: 120 > 100"):
            enforcer.check_and_update(2, tokens_used=40)

    def test_budget_enforcer_timeout_exhaustion(self):
        """BudgetEnforcer raises BudgetExceededError when elapsed wallclock exceeds limit."""
        budget = RunBudget(max_wallclock_seconds=2.0)

        with patch("time.monotonic", side_effect=[1000.0, 1005.0]):
            enforcer = BudgetEnforcer(budget)
            with pytest.raises(BudgetExceededError, match="Execution timeout"):
                enforcer.check_and_update(1)

    def test_in_memory_tracer_record_cost(self):
        """InMemoryTracer accurately accumulates prompt and completion tokens per run_id."""
        tracer = InMemoryTracer()
        tracer.record_cost("run_alpha", prompt_tokens=100, completion_tokens=50)
        assert tracer.token_costs["run_alpha"] == 150

        tracer.record_cost("run_alpha", prompt_tokens=25, completion_tokens=25)
        assert tracer.token_costs["run_alpha"] == 200

        tracer.record_cost("run_beta", prompt_tokens=300, completion_tokens=100)
        assert tracer.token_costs["run_beta"] == 400
