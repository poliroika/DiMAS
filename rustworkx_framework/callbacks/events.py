"""Unified event types for callback system.

All events inherit from BaseEvent and provide structured data
for callback handlers to process.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

__all__ = [
    "EventType",
    "BaseEvent",
    "RunStartEvent",
    "RunEndEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "AgentErrorEvent",
    "AgentRetryEvent",
    "TokenEvent",
    "PlanCreatedEvent",
    "ReplanEvent",
    "PruneEvent",
    "FallbackEvent",
    "ParallelStartEvent",
    "ParallelEndEvent",
    "MemoryReadEvent",
    "MemoryWriteEvent",
    "BudgetWarningEvent",
    "BudgetExceededEvent",
]


class EventType(str, Enum):
    """Types of callback events."""

    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    AGENT_RETRY = "agent_retry"

    # Token streaming
    TOKEN = "token"

    # Planning
    PLAN_CREATED = "plan_created"
    REPLAN = "replan"

    # Adaptive execution
    PRUNE = "prune"
    FALLBACK = "fallback"

    # Parallel execution
    PARALLEL_START = "parallel_start"
    PARALLEL_END = "parallel_end"

    # Memory
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"

    # Budget
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"


class BaseEvent(BaseModel):
    """Base event with common fields."""

    model_config = {"arbitrary_types_allowed": True}

    event_type: EventType
    run_id: UUID
    parent_run_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "run_id": str(self.run_id),
            "parent_run_id": str(self.parent_run_id) if self.parent_run_id else None,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }


class RunStartEvent(BaseEvent):
    """Emitted when execution run starts."""

    event_type: EventType = EventType.RUN_START
    query: str = ""
    num_agents: int = 0
    execution_order: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class RunEndEvent(BaseEvent):
    """Emitted when execution run ends."""

    event_type: EventType = EventType.RUN_END
    output: str = ""
    success: bool = True
    error: str | None = None
    total_tokens: int = 0
    total_time_ms: float = 0.0
    executed_agents: list[str] = Field(default_factory=list)


class AgentStartEvent(BaseEvent):
    """Emitted when an agent starts processing."""

    event_type: EventType = EventType.AGENT_START
    agent_id: str = ""
    agent_name: str = ""
    step_index: int = 0
    prompt: str = ""
    predecessors: list[str] = Field(default_factory=list)


class AgentEndEvent(BaseEvent):
    """Emitted when an agent completes processing."""

    event_type: EventType = EventType.AGENT_END
    agent_id: str = ""
    agent_name: str = ""
    step_index: int = 0
    output: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0
    is_final: bool = False


class AgentErrorEvent(BaseEvent):
    """Emitted when an agent encounters an error."""

    event_type: EventType = EventType.AGENT_ERROR
    agent_id: str = ""
    error_type: str = ""
    error_message: str = ""
    will_retry: bool = False
    attempt: int = 0
    max_attempts: int = 0


class AgentRetryEvent(BaseEvent):
    """Emitted when an agent is being retried."""

    event_type: EventType = EventType.AGENT_RETRY
    agent_id: str = ""
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: float = 0.0
    error: str = ""


class TokenEvent(BaseEvent):
    """Emitted for each token during streaming LLM output."""

    event_type: EventType = EventType.TOKEN
    agent_id: str = ""
    token: str = ""
    token_index: int = 0
    is_first: bool = False
    is_last: bool = False


class PlanCreatedEvent(BaseEvent):
    """Emitted when execution plan is created."""

    event_type: EventType = EventType.PLAN_CREATED
    num_steps: int = 0
    execution_order: list[str] = Field(default_factory=list)


class ReplanEvent(BaseEvent):
    """Emitted when execution plan is updated."""

    event_type: EventType = EventType.REPLAN
    reason: str = ""
    old_remaining: list[str] = Field(default_factory=list)
    new_remaining: list[str] = Field(default_factory=list)
    replan_count: int = 0


class PruneEvent(BaseEvent):
    """Emitted when an agent is pruned from execution."""

    event_type: EventType = EventType.PRUNE
    agent_id: str = ""
    reason: str = ""


class FallbackEvent(BaseEvent):
    """Emitted when fallback agent is activated."""

    event_type: EventType = EventType.FALLBACK
    failed_agent_id: str = ""
    fallback_agent_id: str = ""
    reason: str = ""


class ParallelStartEvent(BaseEvent):
    """Emitted when parallel execution group starts."""

    event_type: EventType = EventType.PARALLEL_START
    agent_ids: list[str] = Field(default_factory=list)
    group_index: int = 0


class ParallelEndEvent(BaseEvent):
    """Emitted when parallel execution group completes."""

    event_type: EventType = EventType.PARALLEL_END
    agent_ids: list[str] = Field(default_factory=list)
    group_index: int = 0
    successful: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class MemoryReadEvent(BaseEvent):
    """Emitted when agent reads from memory."""

    event_type: EventType = EventType.MEMORY_READ
    agent_id: str = ""
    entries_count: int = 0
    keys: list[str] = Field(default_factory=list)


class MemoryWriteEvent(BaseEvent):
    """Emitted when agent writes to memory."""

    event_type: EventType = EventType.MEMORY_WRITE
    agent_id: str = ""
    key: str = ""
    value_size: int = 0


class BudgetWarningEvent(BaseEvent):
    """Emitted when budget threshold is approached."""

    event_type: EventType = EventType.BUDGET_WARNING
    budget_type: str = ""
    current: float = 0.0
    limit: float = 0.0
    ratio: float = 0.0


class BudgetExceededEvent(BaseEvent):
    """Emitted when budget is exceeded."""

    event_type: EventType = EventType.BUDGET_EXCEEDED
    budget_type: str = ""
    current: float = 0.0
    limit: float = 0.0
    action_taken: str = ""
