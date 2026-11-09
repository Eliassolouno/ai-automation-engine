"""
Core Domain Types and Pydantic Contracts for AAE.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    call_id: str
    tool_name: str
    output: Any
    is_error: bool = False
    execution_time_ms: float = 0.0


class AgentStep(BaseModel):
    step_number: int
    thought: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_responses: List[ToolResponse] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunBudget(BaseModel):
    max_steps: int = 15
    max_tokens: int = 32_000
    max_wallclock_seconds: float = 60.0
    consumed_tokens: int = 0
    consumed_seconds: float = 0.0


class RunResult(BaseModel):
    run_id: str
    task_id: str
    status: ExecutionStatus
    final_answer: Optional[str] = None
    steps: List[AgentStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
