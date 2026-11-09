"""
Self-contained test suite using standard library unittest for immediate CI execution.
"""

import unittest
import asyncio
from aae.types import RunBudget, ExecutionStatus
from aae.kernel.loop import AgentLoop
from aae.kernel.budget import BudgetEnforcer, BudgetExceededError
from aae.tools.spec import ToolRegistry
from aae.tools.executor import ToolExecutor
from aae.gateway.router import MockModelGateway
from aae.telemetry.tracing import InMemoryTracer


class TestAgentRuntime(unittest.TestCase):
    def test_agent_loop_execution(self):
        async def _test():
            registry = ToolRegistry()

            @registry.register(name="add_numbers", description="Adds two integers")
            def add(a: int, b: int) -> int:
                return a + b

            executor = ToolExecutor(registry)
            canned = [
                {
                    "content": "I should add 5 and 7 together.",
                    "tool_calls": [{"id": "call_1", "name": "add_numbers", "arguments": {"a": 5, "b": 7}}],
                    "tokens_used": 100
                },
                {
                    "content": "The result is 12.",
                    "tool_calls": [],
                    "tokens_used": 50
                }
            ]
            llm = MockModelGateway(canned_responses=canned)
            tracer = InMemoryTracer()
            loop = AgentLoop(llm=llm, tools=executor, telemetry=tracer)

            result = await loop.run("Please add 5 and 7.")

            self.assertEqual(result.status, ExecutionStatus.COMPLETED)
            self.assertEqual(result.final_answer, "The result is 12.")
            self.assertEqual(len(result.steps), 2)
            self.assertEqual(result.steps[0].tool_calls[0].tool_name, "add_numbers")
            self.assertEqual(result.steps[0].tool_responses[0].output, 12)
            self.assertEqual(len(tracer.traces), 1)

        asyncio.run(_test())

    def test_budget_enforcer(self):
        budget = RunBudget(max_steps=2, max_tokens=500)
        enforcer = BudgetEnforcer(budget)

        enforcer.check_and_update(1, tokens_used=200)
        enforcer.check_and_update(2, tokens_used=200)

        with self.assertRaises(BudgetExceededError):
            enforcer.check_and_update(3, tokens_used=50)


if __name__ == "__main__":
    unittest.main()
