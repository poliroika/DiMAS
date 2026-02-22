"""
Типизированные ошибки и результаты выполнения.

Обеспечивает явную типизацию ошибок вместо строк,
структурированные результаты и политики обработки.
"""

import builtins
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "AgentNotFoundError",
    "BudgetExceededError",
    "ErrorAction",
    "ErrorPolicy",
    "ExecutionError",
    "ExecutionMetrics",
    "RetryExhaustedError",
    "StepExecutionResult",
    "TimeoutError",
    "ValidationError",
]


class ExecutionError(Exception):
    """Базовая ошибка выполнения шага/агента с метаданными."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        step_index: int | None = None,
        cause: Exception | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.step_index = step_index
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать ошибку в словарь для логирования/ответа."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "agent_id": self.agent_id,
            "step_index": self.step_index,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }


class TimeoutError(ExecutionError, builtins.TimeoutError):  # noqa: A001
    """Ошибка таймаута агента."""

    def __init__(
        self,
        agent_id: str,
        timeout_seconds: float,
        step_index: int | None = None,
    ):
        super().__init__(
            f"Agent '{agent_id}' timed out after {timeout_seconds}s",
            agent_id=agent_id,
            step_index=step_index,
            recoverable=True,
        )
        self.timeout_seconds = timeout_seconds


class RetryExhaustedError(ExecutionError):
    """Ошибка исчерпания ретраев."""

    def __init__(
        self,
        agent_id: str,
        attempts: int,
        last_error: Exception | None = None,
        step_index: int | None = None,
    ):
        super().__init__(
            f"Agent '{agent_id}' failed after {attempts} attempts",
            agent_id=agent_id,
            step_index=step_index,
            cause=last_error,
            recoverable=False,
        )
        self.attempts = attempts
        self.last_error = last_error


class BudgetExceededError(ExecutionError):
    """Ошибка превышения бюджета."""

    def __init__(
        self,
        budget_type: str,
        limit: float,
        used: float,
        agent_id: str | None = None,
    ):
        super().__init__(
            f"{budget_type.capitalize()} budget exceeded: {used}/{limit}",
            agent_id=agent_id,
            recoverable=False,
        )
        self.budget_type = budget_type
        self.limit = limit
        self.used = used


class AgentNotFoundError(ExecutionError):
    """Ошибка отсутствия агента в графе."""

    def __init__(self, agent_id: str):
        super().__init__(
            f"Agent '{agent_id}' not found in graph",
            agent_id=agent_id,
            recoverable=False,
        )


class ValidationError(ExecutionError):
    """Ошибка валидации входных данных/параметров."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
    ):
        super().__init__(message, recoverable=False)
        self.field = field
        self.value = value


class ErrorAction(str, Enum):
    SKIP = "skip"
    RETRY = "retry"
    PRUNE = "prune"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    ABORT = "abort"


class ErrorPolicy(BaseModel):
    """Политика реакций на разные типы ошибок."""

    on_timeout: ErrorAction = ErrorAction.RETRY
    on_retry_exhausted: ErrorAction = ErrorAction.PRUNE
    on_budget_exceeded: ErrorAction = ErrorAction.ABORT
    on_agent_not_found: ErrorAction = ErrorAction.SKIP
    on_validation_error: ErrorAction = ErrorAction.ABORT
    on_unknown_error: ErrorAction = ErrorAction.SKIP

    max_skipped_agents: int = 5
    abort_on_critical_path: bool = True

    def get_action(self, error: "ExecutionError") -> ErrorAction:
        """Вернуть действие согласно типу ошибки."""
        if isinstance(error, TimeoutError):
            return self.on_timeout
        if isinstance(error, RetryExhaustedError):
            return self.on_retry_exhausted
        if isinstance(error, BudgetExceededError):
            return self.on_budget_exceeded
        if isinstance(error, AgentNotFoundError):
            return self.on_agent_not_found
        if isinstance(error, ValidationError):
            return self.on_validation_error
        return self.on_unknown_error


class ExecutionMetrics(BaseModel):
    """Сводные метрики по токенам/запросам/агентам/времени."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    start_time: datetime | None = None
    end_time: datetime | None = None
    latency_ms: float = 0.0

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0

    total_agents: int = 0
    executed_agents: int = 0
    skipped_agents: int = 0
    failed_agents: int = 0

    @property
    def success_rate(self) -> float:
        """Доля успешных запросов."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def duration_seconds(self) -> float:
        """Полная длительность выполнения в секундах."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def add_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool,
        latency_ms: float,
        retried: bool = False,
    ) -> None:
        """Добавить метрики единичного запроса/шага."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_requests += 1
        self.latency_ms += latency_ms

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if retried:
            self.retried_requests += 1

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать метрики в словарь."""
        return {
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "retried": self.retried_requests,
                "success_rate": self.success_rate,
            },
            "agents": {
                "total": self.total_agents,
                "executed": self.executed_agents,
                "skipped": self.skipped_agents,
                "failed": self.failed_agents,
            },
            "timing": {
                "duration_seconds": self.duration_seconds,
                "total_latency_ms": self.latency_ms,
                "avg_latency_ms": self.latency_ms / max(1, self.total_requests),
            },
        }


class StepExecutionResult(BaseModel):
    """Результат выполнения шага/агента с метриками и статусами."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    success: bool
    response: str | None = None
    error: "ExecutionError | None" = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1

    quality_score: float = 1.0

    skipped: bool = False
    fallback_used: bool = False

    @property
    def tokens_used(self) -> int:
        """Суммарные токены для шага."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать результат шага."""
        return {
            "agent_id": self.agent_id,
            "success": self.success,
            "response_length": len(self.response) if self.response else 0,
            "error": self.error.to_dict() if self.error else None,
            "metrics": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "latency_ms": self.latency_ms,
                "attempts": self.attempts,
                "quality_score": self.quality_score,
            },
            "status": {
                "skipped": self.skipped,
                "fallback_used": self.fallback_used,
            },
        }
