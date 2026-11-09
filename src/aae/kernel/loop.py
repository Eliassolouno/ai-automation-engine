"""
Agent Execution Loop: Think -> Act -> Observe.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from aae.types import RunResult, ExecutionStatus, AgentStep, ToolCall, RunBudget
from aae.kernel.ports import LLMPort, ToolExecutorPort, TelemetryPort
from aae.kernel.budget import BudgetEnforcer, BudgetExceededError

logger = logging.getLogger("AAE.Kernel.Loop")


class AgentLoop:
    def __init__(
        self,
        llm: LLMPort,
        tools: ToolExecutorPort,
        telemetry: Optional[TelemetryPort] = None
    ):
        self.llm = llm
        self.tools = tools
        self.telemetry = telemetry

    async def run(
        self,
        task: str,
        system_prompt: str = "You are an autonomous AI specialist solving complex tasks with tools.",
        budget: Optional[RunBudget] = None
    ) -> RunResult:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        active_budget = budget or RunBudget()
        enforcer = BudgetEnforcer(active_budget)
        start_mono = time.monotonic()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": task}
        ]
        steps: List[AgentStep] = []

        logger.info(f"Starting execution run [{run_id}] for task: {task[:60]}...")

        current_step_num = 1
        final_answer: Optional[str] = None
        status = ExecutionStatus.RUNNING

        try:
            while current_step_num <= active_budget.max_steps:
                enforcer.check_and_update(current_step_num)

                # 1. THINK: Query LLM Port with available tool schemas
                schemas = self.tools.get_tool_schemas()
                llm_response = await self.llm.generate_response(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools_schema=schemas,
                    budget=active_budget
                )

                content = llm_response.get("content", "")
                raw_calls = llm_response.get("tool_calls", [])
                tokens_in_turn = llm_response.get("tokens_used", 0)
                enforcer.check_and_update(current_step_num, tokens_used=tokens_in_turn)

                # If no tool calls, model provided the final answer
                if not raw_calls:
                    final_answer = content
                    status = ExecutionStatus.COMPLETED
                    step = AgentStep(
                        step_number=current_step_num,
                        thought=content,
                        tool_calls=[],
                        tool_responses=[]
                    )
                    steps.append(step)
                    break

                # 2. ACT: Execute tool calls
                step_calls: List[ToolCall] = []
                step_responses = []

                for rc in raw_calls:
                    call = ToolCall(
                        call_id=rc.get("id", f"call_{uuid.uuid4().hex[:6]}"),
                        tool_name=rc["name"],
                        arguments=rc.get("arguments", {})
                    )
                    step_calls.append(call)
                    resp = await self.tools.execute(call)
                    step_responses.append(resp)

                step = AgentStep(
                    step_number=current_step_num,
                    thought=content,
                    tool_calls=step_calls,
                    tool_responses=step_responses
                )
                steps.append(step)

                if self.telemetry:
                    self.telemetry.record_step(run_id, step.model_dump())

                # 3. OBSERVE: Append assistant action and tool observations to message history
                messages.append({"role": "assistant", "content": content, "tool_calls": raw_calls})
                for resp in step_responses:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": resp.call_id,
                        "name": resp.tool_name,
                        "content": str(resp.output)
                    })

                current_step_num += 1

            if not final_answer and status == ExecutionStatus.RUNNING:
                status = ExecutionStatus.COMPLETED
                final_answer = "Max step limit reached without explicit completion."

        except BudgetExceededError as bee:
            logger.error(f"Run [{run_id}] aborted: {bee}")
            status = ExecutionStatus.FAILED
            final_answer = f"Aborted: {bee}"
        except Exception as e:
            logger.exception(f"Run [{run_id}] failed with unexpected error: {e}")
            status = ExecutionStatus.FAILED
            final_answer = f"Error: {e}"

        duration_ms = (time.monotonic() - start_mono) * 1000
        return RunResult(
            run_id=run_id,
            task_id=task_id,
            status=status,
            final_answer=final_answer,
            steps=steps,
            total_duration_ms=round(duration_ms, 2)
        )
