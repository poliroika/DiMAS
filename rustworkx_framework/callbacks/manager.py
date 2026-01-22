"""Callback manager (like LangChain CallbackManager).

Manages multiple callback handlers and dispatches events to them.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

from .base import AsyncCallbackHandler, BaseCallbackHandler

# Type alias for handlers
Handler = BaseCallbackHandler | AsyncCallbackHandler

__all__ = [
    "CallbackManager",
    "AsyncCallbackManager",
]

_logger = logging.getLogger(__name__)


class CallbackManager:
    """Manages callback handlers and dispatches events.

    Like LangChain's CallbackManager, supports:
    - Multiple handlers
    - Inheritable handlers/tags/metadata
    - Parent run tracking
    - Error handling per handler
    """

    def __init__(
        self,
        handlers: list[Handler] | None = None,
        inheritable_handlers: list[Handler] | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ):
        self.handlers: list[Handler] = list(handlers or [])
        self.inheritable_handlers: list[Handler] = list(inheritable_handlers or [])
        self.parent_run_id = parent_run_id
        self.tags: list[str] = list(tags or [])
        self.inheritable_tags: list[str] = list(inheritable_tags or [])
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.inheritable_metadata: dict[str, Any] = dict(inheritable_metadata or {})

    @property
    def is_async(self) -> bool:
        return False

    @classmethod
    def configure(
        cls,
        handlers: list[Handler] | None = None,
        inheritable_handlers: list[Handler] | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> "CallbackManager":
        """Create configured callback manager."""
        return cls(
            handlers=handlers,
            inheritable_handlers=inheritable_handlers,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
        )

    def copy(self) -> "CallbackManager":
        """Create a copy of this manager."""
        return CallbackManager(
            handlers=list(self.handlers),
            inheritable_handlers=list(self.inheritable_handlers),
            parent_run_id=self.parent_run_id,
            tags=list(self.tags),
            inheritable_tags=list(self.inheritable_tags),
            metadata=dict(self.metadata),
            inheritable_metadata=dict(self.inheritable_metadata),
        )

    def merge(self, other: "CallbackManager") -> "CallbackManager":
        """Merge another manager into a new manager."""
        return CallbackManager(
            handlers=self.handlers + other.handlers,
            inheritable_handlers=self.inheritable_handlers + other.inheritable_handlers,
            parent_run_id=other.parent_run_id or self.parent_run_id,
            tags=self.tags + other.tags,
            inheritable_tags=self.inheritable_tags + other.inheritable_tags,
            metadata={**self.metadata, **other.metadata},
            inheritable_metadata={**self.inheritable_metadata, **other.inheritable_metadata},
        )

    def add_handler(self, handler: Handler, inherit: bool = False) -> None:
        """Add a callback handler."""
        self.handlers.append(handler)
        if inherit:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: Handler) -> None:
        """Remove a callback handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def set_handlers(self, handlers: list[Handler]) -> None:
        """Replace all handlers."""
        self.handlers = list(handlers)

    def add_tags(self, tags: list[str], inherit: bool = False) -> None:
        """Add tags."""
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags."""
        self.tags = [t for t in self.tags if t not in tags]
        self.inheritable_tags = [t for t in self.inheritable_tags if t not in tags]

    def add_metadata(self, metadata: dict[str, Any], inherit: bool = False) -> None:
        """Add metadata."""
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def get_child(self, parent_run_id: UUID) -> "CallbackManager":
        """Get a child callback manager with inherited settings."""
        return CallbackManager(
            handlers=list(self.inheritable_handlers),
            inheritable_handlers=list(self.inheritable_handlers),
            parent_run_id=parent_run_id,
            tags=list(self.inheritable_tags),
            inheritable_tags=list(self.inheritable_tags),
            metadata=dict(self.inheritable_metadata),
            inheritable_metadata=dict(self.inheritable_metadata),
        )

    def _handle_error(self, handler: Handler, method: str, error: Exception) -> None:
        """Handle error in callback handler."""
        if handler.raise_error:
            raise error
        _logger.warning(f"Error in callback handler {handler.__class__.__name__}.{method}: {error}")

    # === Run lifecycle ===

    def on_run_start(
        self,
        run_id: UUID | None = None,
        *,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        **kwargs: Any,
    ) -> UUID:
        """Notify handlers of run start. Returns run_id."""
        run_id = run_id or uuid4()
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            try:
                handler.on_run_start(
                    run_id=run_id,
                    query=query,
                    num_agents=num_agents,
                    execution_order=execution_order or [],
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_run_start", e)
        return run_id

    def on_run_end(
        self,
        run_id: UUID,
        *,
        output: str,
        success: bool = True,
        error: BaseException | None = None,
        total_tokens: int = 0,
        total_time_ms: float = 0.0,
        executed_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of run end."""
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            try:
                handler.on_run_end(
                    run_id=run_id,
                    output=output,
                    success=success,
                    error=error,
                    total_tokens=total_tokens,
                    total_time_ms=total_time_ms,
                    executed_agents=executed_agents or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_run_end", e)

    # === Agent lifecycle ===

    def on_agent_start(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        agent_name: str = "",
        step_index: int = 0,
        prompt: str = "",
        predecessors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent start."""
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            try:
                handler.on_agent_start(
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    step_index=step_index,
                    prompt=prompt,
                    predecessors=predecessors or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_agent_start", e)

    def on_agent_end(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        output: str,
        agent_name: str = "",
        step_index: int = 0,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent end."""
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            try:
                handler.on_agent_end(
                    run_id=run_id,
                    agent_id=agent_id,
                    output=output,
                    agent_name=agent_name,
                    step_index=step_index,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    is_final=is_final,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_agent_end", e)

    def on_agent_error(
        self,
        run_id: UUID,
        error: BaseException,
        *,
        agent_id: str,
        error_type: str = "",
        will_retry: bool = False,
        attempt: int = 0,
        max_attempts: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent error."""
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            try:
                handler.on_agent_error(
                    error,
                    run_id=run_id,
                    agent_id=agent_id,
                    error_type=error_type,
                    will_retry=will_retry,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_agent_error", e)

    # === Retry ===

    def on_retry(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of retry."""
        for handler in self.handlers:
            if handler.ignore_retry:
                continue
            try:
                handler.on_retry(
                    run_id=run_id,
                    agent_id=agent_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_ms=delay_ms,
                    error=error,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_retry", e)

    # === Token streaming ===

    def on_llm_new_token(
        self,
        run_id: UUID,
        token: str,
        *,
        agent_id: str,
        token_index: int = 0,
        is_first: bool = False,
        is_last: bool = False,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of new token."""
        for handler in self.handlers:
            if handler.ignore_llm:
                continue
            try:
                handler.on_llm_new_token(
                    token,
                    run_id=run_id,
                    agent_id=agent_id,
                    token_index=token_index,
                    is_first=is_first,
                    is_last=is_last,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_llm_new_token", e)

    # === Planning ===

    def on_plan_created(
        self,
        run_id: UUID,
        *,
        num_steps: int,
        execution_order: list[str],
        **kwargs: Any,
    ) -> None:
        """Notify handlers of plan creation."""
        for handler in self.handlers:
            try:
                handler.on_plan_created(
                    run_id=run_id,
                    num_steps=num_steps,
                    execution_order=execution_order,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_plan_created", e)

    def on_replan(
        self,
        run_id: UUID,
        *,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        replan_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of replan."""
        for handler in self.handlers:
            try:
                handler.on_replan(
                    run_id=run_id,
                    reason=reason,
                    old_remaining=old_remaining,
                    new_remaining=new_remaining,
                    replan_count=replan_count,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_replan", e)

    # === Pruning/Fallback ===

    def on_prune(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of prune."""
        for handler in self.handlers:
            try:
                handler.on_prune(
                    run_id=run_id,
                    agent_id=agent_id,
                    reason=reason,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_prune", e)

    def on_fallback(
        self,
        run_id: UUID,
        *,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of fallback."""
        for handler in self.handlers:
            try:
                handler.on_fallback(
                    run_id=run_id,
                    failed_agent_id=failed_agent_id,
                    fallback_agent_id=fallback_agent_id,
                    reason=reason,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_fallback", e)

    # === Parallel execution ===

    def on_parallel_start(
        self,
        run_id: UUID,
        *,
        agent_ids: list[str],
        group_index: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of parallel start."""
        for handler in self.handlers:
            try:
                handler.on_parallel_start(
                    run_id=run_id,
                    agent_ids=agent_ids,
                    group_index=group_index,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_parallel_start", e)

    def on_parallel_end(
        self,
        run_id: UUID,
        *,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of parallel end."""
        for handler in self.handlers:
            try:
                handler.on_parallel_end(
                    run_id=run_id,
                    agent_ids=agent_ids,
                    group_index=group_index,
                    successful=successful or [],
                    failed=failed or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_parallel_end", e)

    # === Memory ===

    def on_memory_read(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of memory read."""
        for handler in self.handlers:
            if handler.ignore_memory:
                continue
            try:
                handler.on_memory_read(
                    run_id=run_id,
                    agent_id=agent_id,
                    entries_count=entries_count,
                    keys=keys or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_memory_read", e)

    def on_memory_write(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        key: str,
        value_size: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of memory write."""
        for handler in self.handlers:
            if handler.ignore_memory:
                continue
            try:
                handler.on_memory_write(
                    run_id=run_id,
                    agent_id=agent_id,
                    key=key,
                    value_size=value_size,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_memory_write", e)

    # === Budget ===

    def on_budget_warning(
        self,
        run_id: UUID,
        *,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of budget warning."""
        for handler in self.handlers:
            if handler.ignore_budget:
                continue
            try:
                handler.on_budget_warning(
                    run_id=run_id,
                    budget_type=budget_type,
                    current=current,
                    limit=limit,
                    ratio=ratio,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_budget_warning", e)

    def on_budget_exceeded(
        self,
        run_id: UUID,
        *,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of budget exceeded."""
        for handler in self.handlers:
            if handler.ignore_budget:
                continue
            try:
                handler.on_budget_exceeded(
                    run_id=run_id,
                    budget_type=budget_type,
                    current=current,
                    limit=limit,
                    action_taken=action_taken,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            except Exception as e:
                self._handle_error(handler, "on_budget_exceeded", e)


class AsyncCallbackManager:
    """Async version of CallbackManager."""

    def __init__(
        self,
        handlers: list[Handler] | None = None,
        inheritable_handlers: list[Handler] | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ):
        self.handlers: list[Handler] = list(handlers or [])
        self.inheritable_handlers: list[Handler] = list(inheritable_handlers or [])
        self.parent_run_id = parent_run_id
        self.tags: list[str] = list(tags or [])
        self.inheritable_tags: list[str] = list(inheritable_tags or [])
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.inheritable_metadata: dict[str, Any] = dict(inheritable_metadata or {})

    @property
    def is_async(self) -> bool:
        return True

    @classmethod
    def configure(
        cls,
        handlers: list[Handler] | None = None,
        inheritable_handlers: list[Handler] | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> "AsyncCallbackManager":
        """Create configured async callback manager."""
        return cls(
            handlers=handlers,
            inheritable_handlers=inheritable_handlers,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
        )

    def copy(self) -> "AsyncCallbackManager":
        """Create a copy of this manager."""
        return AsyncCallbackManager(
            handlers=list(self.handlers),
            inheritable_handlers=list(self.inheritable_handlers),
            parent_run_id=self.parent_run_id,
            tags=list(self.tags),
            inheritable_tags=list(self.inheritable_tags),
            metadata=dict(self.metadata),
            inheritable_metadata=dict(self.inheritable_metadata),
        )

    def merge(self, other: "AsyncCallbackManager") -> "AsyncCallbackManager":
        """Merge another manager into a new manager."""
        return AsyncCallbackManager(
            handlers=self.handlers + other.handlers,
            inheritable_handlers=self.inheritable_handlers + other.inheritable_handlers,
            parent_run_id=other.parent_run_id or self.parent_run_id,
            tags=self.tags + other.tags,
            inheritable_tags=self.inheritable_tags + other.inheritable_tags,
            metadata={**self.metadata, **other.metadata},
            inheritable_metadata={**self.inheritable_metadata, **other.inheritable_metadata},
        )

    def add_handler(self, handler: Handler, inherit: bool = False) -> None:
        """Add a callback handler."""
        self.handlers.append(handler)
        if inherit:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: Handler) -> None:
        """Remove a callback handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def set_handlers(self, handlers: list[Handler]) -> None:
        """Replace all handlers."""
        self.handlers = list(handlers)

    def add_tags(self, tags: list[str], inherit: bool = False) -> None:
        """Add tags."""
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags."""
        self.tags = [t for t in self.tags if t not in tags]
        self.inheritable_tags = [t for t in self.inheritable_tags if t not in tags]

    def add_metadata(self, metadata: dict[str, Any], inherit: bool = False) -> None:
        """Add metadata."""
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def get_child(self, parent_run_id: UUID) -> "AsyncCallbackManager":
        """Get a child callback manager with inherited settings."""
        return AsyncCallbackManager(
            handlers=list(self.inheritable_handlers),
            inheritable_handlers=list(self.inheritable_handlers),
            parent_run_id=parent_run_id,
            tags=list(self.inheritable_tags),
            inheritable_tags=list(self.inheritable_tags),
            metadata=dict(self.inheritable_metadata),
            inheritable_metadata=dict(self.inheritable_metadata),
        )

    def _handle_error(self, handler: Handler, method: str, error: Exception) -> None:
        """Handle error in callback handler."""
        if handler.raise_error:
            raise error
        _logger.warning(f"Error in callback handler {handler.__class__.__name__}.{method}: {error}")

    async def _run_handler(
        self,
        handler: Handler,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run a handler method (sync or async)."""
        method = getattr(handler, method_name)
        try:
            if isinstance(handler, AsyncCallbackHandler):
                await method(*args, **kwargs)
            else:
                method(*args, **kwargs)
        except Exception as e:
            self._handle_error(handler, method_name, e)

    # === Run lifecycle ===

    async def on_run_start(
        self,
        run_id: UUID | None = None,
        *,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        **kwargs: Any,
    ) -> UUID:
        """Notify handlers of run start. Returns run_id."""
        run_id = run_id or uuid4()
        tasks = []
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_run_start",
                    run_id=run_id,
                    query=query,
                    num_agents=num_agents,
                    execution_order=execution_order or [],
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)
        return run_id

    async def on_run_end(
        self,
        run_id: UUID,
        *,
        output: str,
        success: bool = True,
        error: BaseException | None = None,
        total_tokens: int = 0,
        total_time_ms: float = 0.0,
        executed_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of run end."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_run_end",
                    run_id=run_id,
                    output=output,
                    success=success,
                    error=error,
                    total_tokens=total_tokens,
                    total_time_ms=total_time_ms,
                    executed_agents=executed_agents or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Agent lifecycle ===

    async def on_agent_start(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        agent_name: str = "",
        step_index: int = 0,
        prompt: str = "",
        predecessors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent start."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_agent_start",
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    step_index=step_index,
                    prompt=prompt,
                    predecessors=predecessors or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_agent_end(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        output: str,
        agent_name: str = "",
        step_index: int = 0,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent end."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_agent_end",
                    run_id=run_id,
                    agent_id=agent_id,
                    output=output,
                    agent_name=agent_name,
                    step_index=step_index,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    is_final=is_final,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_agent_error(
        self,
        run_id: UUID,
        error: BaseException,
        *,
        agent_id: str,
        error_type: str = "",
        will_retry: bool = False,
        attempt: int = 0,
        max_attempts: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of agent error."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_agent:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_agent_error",
                    error,
                    run_id=run_id,
                    agent_id=agent_id,
                    error_type=error_type,
                    will_retry=will_retry,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Retry ===

    async def on_retry(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of retry."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_retry:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_retry",
                    run_id=run_id,
                    agent_id=agent_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_ms=delay_ms,
                    error=error,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Token streaming ===

    async def on_llm_new_token(
        self,
        run_id: UUID,
        token: str,
        *,
        agent_id: str,
        token_index: int = 0,
        is_first: bool = False,
        is_last: bool = False,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of new token."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_llm:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_llm_new_token",
                    token,
                    run_id=run_id,
                    agent_id=agent_id,
                    token_index=token_index,
                    is_first=is_first,
                    is_last=is_last,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Planning ===

    async def on_plan_created(
        self,
        run_id: UUID,
        *,
        num_steps: int,
        execution_order: list[str],
        **kwargs: Any,
    ) -> None:
        """Notify handlers of plan creation."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_plan_created",
                    run_id=run_id,
                    num_steps=num_steps,
                    execution_order=execution_order,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_replan(
        self,
        run_id: UUID,
        *,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        replan_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of replan."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_replan",
                    run_id=run_id,
                    reason=reason,
                    old_remaining=old_remaining,
                    new_remaining=new_remaining,
                    replan_count=replan_count,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Pruning/Fallback ===

    async def on_prune(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of prune."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_prune",
                    run_id=run_id,
                    agent_id=agent_id,
                    reason=reason,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_fallback(
        self,
        run_id: UUID,
        *,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of fallback."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_fallback",
                    run_id=run_id,
                    failed_agent_id=failed_agent_id,
                    fallback_agent_id=fallback_agent_id,
                    reason=reason,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Parallel execution ===

    async def on_parallel_start(
        self,
        run_id: UUID,
        *,
        agent_ids: list[str],
        group_index: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of parallel start."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_parallel_start",
                    run_id=run_id,
                    agent_ids=agent_ids,
                    group_index=group_index,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_parallel_end(
        self,
        run_id: UUID,
        *,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of parallel end."""
        tasks = []
        for handler in self.handlers:
            tasks.append(
                self._run_handler(
                    handler,
                    "on_parallel_end",
                    run_id=run_id,
                    agent_ids=agent_ids,
                    group_index=group_index,
                    successful=successful or [],
                    failed=failed or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Memory ===

    async def on_memory_read(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of memory read."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_memory:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_memory_read",
                    run_id=run_id,
                    agent_id=agent_id,
                    entries_count=entries_count,
                    keys=keys or [],
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_memory_write(
        self,
        run_id: UUID,
        *,
        agent_id: str,
        key: str,
        value_size: int = 0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of memory write."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_memory:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_memory_write",
                    run_id=run_id,
                    agent_id=agent_id,
                    key=key,
                    value_size=value_size,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    # === Budget ===

    async def on_budget_warning(
        self,
        run_id: UUID,
        *,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Notify handlers of budget warning."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_budget:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_budget_warning",
                    run_id=run_id,
                    budget_type=budget_type,
                    current=current,
                    limit=limit,
                    ratio=ratio,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_budget_exceeded(
        self,
        run_id: UUID,
        *,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        **kwargs: Any,
    ) -> None:
        """Notify handlers of budget exceeded."""
        tasks = []
        for handler in self.handlers:
            if handler.ignore_budget:
                continue
            tasks.append(
                self._run_handler(
                    handler,
                    "on_budget_exceeded",
                    run_id=run_id,
                    budget_type=budget_type,
                    current=current,
                    limit=limit,
                    action_taken=action_taken,
                    parent_run_id=self.parent_run_id,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)
