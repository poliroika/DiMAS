"""Metrics callback handler for aggregating execution metrics."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ..base import BaseCallbackHandler

__all__ = ["MetricsCallbackHandler"]


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    Aggregates metrics from execution events.

    Collects:
    - Total tokens used
    - Total duration
    - Agent execution counts and times
    - Error counts
    - Retry counts
    """

    def __init__(self):
        self._run_start_time: datetime | None = None
        self._total_tokens: int = 0
        self._total_duration_ms: float = 0.0
        self._agent_tokens: dict[str, int] = {}
        self._agent_durations: dict[str, float] = {}
        self._agent_calls: dict[str, int] = {}
        self._errors: list[dict[str, Any]] = []
        self._retries: int = 0
        self._runs_completed: int = 0
        self._runs_failed: int = 0
        self._budget_warnings: int = 0
        # Tool metrics
        self._tool_calls: dict[str, int] = {}  # tool_name.action -> count
        self._tool_durations: dict[str, float] = {}  # tool_name.action -> total ms
        self._tool_errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_duration_ms(self) -> float:
        return self._total_duration_ms

    @property
    def runs_completed(self) -> int:
        return self._runs_completed

    @property
    def runs_failed(self) -> int:
        return self._runs_failed

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics."""
        return {
            "total_tokens": self._total_tokens,
            "total_duration_ms": self._total_duration_ms,
            "agent_tokens": dict(self._agent_tokens),
            "agent_durations": dict(self._agent_durations),
            "agent_calls": dict(self._agent_calls),
            "errors_count": len(self._errors),
            "errors": self._errors[-10:],  # last 10 errors
            "retries": self._retries,
            "runs_completed": self._runs_completed,
            "runs_failed": self._runs_failed,
            "budget_warnings": self._budget_warnings,
            "avg_tokens_per_agent": (
                self._total_tokens / sum(self._agent_calls.values()) if self._agent_calls else 0.0
            ),
            "tool_calls": dict(self._tool_calls),
            "tool_durations": dict(self._tool_durations),
            "tool_errors": self._tool_errors,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._run_start_time = None
        self._total_tokens = 0
        self._total_duration_ms = 0.0
        self._agent_tokens.clear()
        self._agent_durations.clear()
        self._agent_calls.clear()
        self._errors.clear()
        self._retries = 0
        self._runs_completed = 0
        self._runs_failed = 0
        self._budget_warnings = 0
        self._tool_calls.clear()
        self._tool_durations.clear()
        self._tool_errors = 0

    # === Run lifecycle ===

    def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, query, num_agents, execution_order, parent_run_id, tags, metadata, kwargs
        self._run_start_time = datetime.now(UTC)

    def on_run_end(
        self,
        *,
        run_id: UUID,
        output: str,
        success: bool = True,
        error: BaseException | None = None,
        total_tokens: int = 0,
        total_time_ms: float = 0.0,
        executed_agents: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, output, error, executed_agents, parent_run_id, kwargs
        if success:
            self._runs_completed += 1
        else:
            self._runs_failed += 1

        # Use provided totals if available, otherwise use accumulated
        if total_tokens > 0:
            self._total_tokens = total_tokens
        if total_time_ms > 0:
            self._total_duration_ms = total_time_ms

    # === Agent lifecycle ===

    def on_agent_end(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        output: str,
        agent_name: str = "",
        step_index: int = 0,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, output, agent_name, step_index, is_final, parent_run_id, kwargs
        self._total_tokens += tokens_used
        self._total_duration_ms += duration_ms

        self._agent_tokens[agent_id] = self._agent_tokens.get(agent_id, 0) + tokens_used
        self._agent_durations[agent_id] = self._agent_durations.get(agent_id, 0.0) + duration_ms
        self._agent_calls[agent_id] = self._agent_calls.get(agent_id, 0) + 1

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        agent_id: str,
        error_type: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        self._errors.append(
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "error_type": error_type or type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    # === Retry ===

    def on_retry(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, agent_id, attempt, max_attempts, delay_ms, error, parent_run_id, kwargs
        self._retries += 1

    # === Budget ===

    def on_budget_warning(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del run_id, budget_type, current, limit, ratio, parent_run_id, kwargs
        self._budget_warnings += 1

    # === Tool lifecycle ===

    def on_tool_end(
        self,
        *,
        run_id: UUID,
        agent_id: str = "",
        tool_name: str,
        action: str = "",
        success: bool = True,
        output_size: int = 0,
        duration_ms: float = 0.0,
        result_summary: str = "",
        **kwargs: Any,
    ) -> None:
        _ = (run_id, agent_id, success, output_size, result_summary, kwargs)  # Unused but required by interface
        key = f"{tool_name}.{action}" if action else tool_name
        self._tool_calls[key] = self._tool_calls.get(key, 0) + 1
        self._tool_durations[key] = self._tool_durations.get(key, 0.0) + duration_ms

    def on_tool_error(
        self,
        *,
        run_id: UUID,
        _agent_id: str = "",
        tool_name: str = "",
        action: str = "",
        error_type: str = "",
        error_message: str = "",
        **_kwargs: Any,
    ) -> None:
        self._tool_errors += 1
        self._errors.append(
            {
                "run_id": str(run_id),
                "tool_name": tool_name,
                "action": action,
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
