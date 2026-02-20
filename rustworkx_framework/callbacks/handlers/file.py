"""File callback handler for writing events to JSON lines file."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ..base import BaseCallbackHandler

__all__ = ["FileCallbackHandler"]


class FileCallbackHandler(BaseCallbackHandler):
    """
    Writes events to a JSON lines file.

    Each event is written as a single JSON line.
    """

    def __init__(
        self,
        file_path: str | Path,
        append: bool = True,
        flush_every: int = 1,
    ):
        self.file_path = Path(file_path)
        self.append = append
        self.flush_every = flush_every
        self._file = None
        self._event_count = 0
        self._open_file()

    def _open_file(self) -> None:
        mode = "a" if self.append else "w"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.file_path, mode, encoding="utf-8")

    def _write_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self._file is None:
            return

        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        self._file.write(json.dumps(event, default=str) + "\n")
        self._event_count += 1

        if self._event_count % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    # === Run lifecycle ===

    def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "run_start",
            {
                "run_id": str(run_id),
                "query": query,
                "num_agents": num_agents,
                "execution_order": execution_order or [],
            },
        )

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
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "run_end",
            {
                "run_id": str(run_id),
                "success": success,
                "error": str(error) if error else None,
                "total_tokens": total_tokens,
                "total_time_ms": total_time_ms,
                "executed_agents": executed_agents or [],
            },
        )

    # === Agent lifecycle ===

    def on_agent_start(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        agent_name: str = "",
        step_index: int = 0,
        prompt: str = "",
        predecessors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "agent_start",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "step_index": step_index,
                "predecessors": predecessors or [],
            },
        )

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
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "agent_end",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "step_index": step_index,
                "tokens_used": tokens_used,
                "duration_ms": duration_ms,
                "is_final": is_final,
                "output_length": len(output),
            },
        )

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        agent_id: str,
        error_type: str = "",
        will_retry: bool = False,
        attempt: int = 0,
        max_attempts: int = 0,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "agent_error",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "error_type": error_type or type(error).__name__,
                "error_message": str(error),
                "will_retry": will_retry,
                "attempt": attempt,
                "max_attempts": max_attempts,
            },
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
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "retry",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_ms": delay_ms,
                "error": error,
            },
        )

    # === Token streaming ===

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        agent_id: str,
        token_index: int = 0,
        is_first: bool = False,
        is_last: bool = False,
        **kwargs: Any,
    ) -> None:
        # Don't log individual tokens by default (too verbose)
        if is_first or is_last:
            self._write_event(
                "token",
                {
                    "run_id": str(run_id),
                    "agent_id": agent_id,
                    "token_index": token_index,
                    "is_first": is_first,
                    "is_last": is_last,
                },
            )

    # === Planning ===

    def on_plan_created(
        self,
        *,
        run_id: UUID,
        num_steps: int,
        execution_order: list[str],
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "plan_created",
            {
                "run_id": str(run_id),
                "num_steps": num_steps,
                "execution_order": execution_order,
            },
        )

    def on_topology_changed(
        self,
        *,
        run_id: UUID,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        change_count: int = 0,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "topology_changed",
            {
                "run_id": str(run_id),
                "reason": reason,
                "old_remaining": old_remaining,
                "new_remaining": new_remaining,
                "change_count": change_count,
            },
        )

    # === Pruning/Fallback ===

    def on_prune(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "prune",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "reason": reason,
            },
        )

    def on_fallback(
        self,
        *,
        run_id: UUID,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "fallback",
            {
                "run_id": str(run_id),
                "failed_agent_id": failed_agent_id,
                "fallback_agent_id": fallback_agent_id,
                "reason": reason,
            },
        )

    # === Parallel execution ===

    def on_parallel_start(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "parallel_start",
            {
                "run_id": str(run_id),
                "agent_ids": agent_ids,
                "group_index": group_index,
            },
        )

    def on_parallel_end(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "parallel_end",
            {
                "run_id": str(run_id),
                "agent_ids": agent_ids,
                "group_index": group_index,
                "successful": successful or [],
                "failed": failed or [],
            },
        )

    # === Memory ===

    def on_memory_read(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "memory_read",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "entries_count": entries_count,
            },
        )

    def on_memory_write(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        key: str,
        value_size: int = 0,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "memory_write",
            {
                "run_id": str(run_id),
                "agent_id": agent_id,
                "key": key,
                "value_size": value_size,
            },
        )

    # === Budget ===

    def on_budget_warning(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "budget_warning",
            {
                "run_id": str(run_id),
                "budget_type": budget_type,
                "current": current,
                "limit": limit,
                "ratio": ratio,
            },
        )

    def on_budget_exceeded(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        **kwargs: Any,
    ) -> None:
        self._write_event(
            "budget_exceeded",
            {
                "run_id": str(run_id),
                "budget_type": budget_type,
                "current": current,
                "limit": limit,
                "action_taken": action_taken,
            },
        )
