"""Base callback handlers (like LangChain BaseCallbackHandler).

Provides abstract base classes for sync and async callback handlers.
"""

from typing import Any
from uuid import UUID

__all__ = [
    "BaseCallbackHandler",
    "AsyncCallbackHandler",
    "CallbackHandlerMixin",
]


class CallbackHandlerMixin:
    """Base mixin for callback handlers providing common attributes."""

    raise_error: bool = False
    run_inline: bool = True

    # Ignore flags
    ignore_agent: bool = False
    ignore_retry: bool = False
    ignore_budget: bool = False
    ignore_memory: bool = False
    ignore_llm: bool = False

    @property
    def is_async(self) -> bool:
        """Whether this handler is async."""
        return False


class BaseCallbackHandler(CallbackHandlerMixin):
    """Base callback handler for sync operations.

    Subclass this to create custom handlers. Override only the methods
    you need - all methods have default no-op implementations.

    Attributes:
        raise_error: If True, exceptions in handlers will propagate.
        run_inline: If True, handlers run in the same thread.
        ignore_agent: If True, skip agent lifecycle events.
        ignore_retry: If True, skip retry events.
        ignore_budget: If True, skip budget events.
        ignore_memory: If True, skip memory events.
        ignore_llm: If True, skip token streaming events.
    """

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
        """Called when execution run starts."""
        pass

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
        """Called when execution run ends."""
        pass

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
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent starts processing."""
        pass

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
        """Called when an agent completes processing."""
        pass

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
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent encounters an error."""
        pass

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
        """Called when an agent is being retried."""
        pass

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
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called for each token during streaming LLM output."""
        pass

    # === Planning ===

    def on_plan_created(
        self,
        *,
        run_id: UUID,
        num_steps: int,
        execution_order: list[str],
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when execution plan is created."""
        pass

    def on_replan(
        self,
        *,
        run_id: UUID,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        replan_count: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when execution plan is updated."""
        pass

    # === Pruning/Fallback ===

    def on_prune(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        reason: str,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent is pruned from execution."""
        pass

    def on_fallback(
        self,
        *,
        run_id: UUID,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when fallback agent is activated."""
        pass

    # === Parallel execution ===

    def on_parallel_start(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when parallel execution group starts."""
        pass

    def on_parallel_end(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when parallel execution group completes."""
        pass

    # === Memory ===

    def on_memory_read(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent reads from memory."""
        pass

    def on_memory_write(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        key: str,
        value_size: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent writes to memory."""
        pass

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
        """Called when budget threshold is approached."""
        pass

    def on_budget_exceeded(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when budget is exceeded."""
        pass


class AsyncCallbackHandler(CallbackHandlerMixin):
    """Async callback handler for async operations.

    All methods are async versions of BaseCallbackHandler methods.
    """

    @property
    def is_async(self) -> bool:
        return True

    # === Run lifecycle ===

    async def on_run_start(
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
        pass

    async def on_run_end(
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
        pass

    # === Agent lifecycle ===

    async def on_agent_start(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        agent_name: str = "",
        step_index: int = 0,
        prompt: str = "",
        predecessors: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_agent_end(
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
        pass

    async def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        agent_id: str,
        error_type: str = "",
        will_retry: bool = False,
        attempt: int = 0,
        max_attempts: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Retry ===

    async def on_retry(
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
        pass

    # === Token streaming ===

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        agent_id: str,
        token_index: int = 0,
        is_first: bool = False,
        is_last: bool = False,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Planning ===

    async def on_plan_created(
        self,
        *,
        run_id: UUID,
        num_steps: int,
        execution_order: list[str],
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_replan(
        self,
        *,
        run_id: UUID,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        replan_count: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Pruning/Fallback ===

    async def on_prune(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        reason: str,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_fallback(
        self,
        *,
        run_id: UUID,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Parallel execution ===

    async def on_parallel_start(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_parallel_end(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Memory ===

    async def on_memory_read(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def on_memory_write(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        key: str,
        value_size: int = 0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    # === Budget ===

    async def on_budget_warning(
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
        pass

    async def on_budget_exceeded(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass
