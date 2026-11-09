"""
Budget Guardrails and Resource Limiter.
Enforces step limits, token constraints, and execution timeouts to prevent infinite loops.
"""

import time
from aae.types import RunBudget


class BudgetExceededError(RuntimeError):
    """Raised when an agent execution exceeds its allocated budget."""
    pass


class BudgetEnforcer:
    def __init__(self, budget: RunBudget):
        self.budget = budget
        self.start_time = time.monotonic()

    def check_and_update(self, current_step: int, tokens_used: int = 0) -> None:
        elapsed = time.monotonic() - self.start_time
        self.budget.consumed_seconds = round(elapsed, 3)
        self.budget.consumed_tokens += tokens_used

        if current_step > self.budget.max_steps:
            raise BudgetExceededError(
                f"Agent step limit exceeded: {current_step} > {self.budget.max_steps}"
            )

        if self.budget.consumed_tokens > self.budget.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.budget.consumed_tokens} > {self.budget.max_tokens}"
            )

        if elapsed > self.budget.max_wallclock_seconds:
            raise BudgetExceededError(
                f"Execution timeout: {elapsed:.2f}s > {self.budget.max_wallclock_seconds}s"
            )
