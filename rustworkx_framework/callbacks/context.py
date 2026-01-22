"""Context managers for callback handling.

Provides thread-safe context management for callbacks.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from .base import AsyncCallbackHandler, BaseCallbackHandler
from .handlers.metrics import MetricsCallbackHandler
from .manager import AsyncCallbackManager, CallbackManager

# Type alias for handlers
Handler = BaseCallbackHandler | AsyncCallbackHandler

__all__ = [
    "get_callback_manager",
    "set_callback_manager",
    "trace_as_callback",
    "collect_metrics",
]

# Context variable for current callback manager
_current_callback_manager: ContextVar[CallbackManager | None] = ContextVar(
    "current_callback_manager", default=None
)


def get_callback_manager() -> CallbackManager | None:
    """Get the current callback manager from context."""
    return _current_callback_manager.get()


def set_callback_manager(manager: CallbackManager | None) -> None:
    """Set the current callback manager in context."""
    _current_callback_manager.set(manager)


@contextmanager
def trace_as_callback(
    handlers: list[Handler] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    parent_run_id: UUID | None = None,
) -> Generator[CallbackManager, None, None]:
    """Context manager for tracing with callbacks.

    Example:
        from rustworkx_framework.callbacks import trace_as_callback, StdoutCallbackHandler

        with trace_as_callback(handlers=[StdoutCallbackHandler()]) as manager:
            runner.run_round(graph)
    """
    manager = CallbackManager.configure(
        handlers=handlers,
        tags=tags,
        metadata=metadata,
    )
    manager.parent_run_id = parent_run_id

    token = _current_callback_manager.set(manager)
    try:
        yield manager
    finally:
        _current_callback_manager.reset(token)


@contextmanager
def collect_metrics() -> Generator[MetricsCallbackHandler, None, None]:
    """Context manager for collecting metrics.

    Example:
        from rustworkx_framework.callbacks import collect_metrics

        with collect_metrics() as metrics:
            runner.run_round(graph)

        print(f"Total tokens: {metrics.total_tokens}")
        print(metrics.get_metrics())
    """
    handler = MetricsCallbackHandler()
    manager = CallbackManager.configure(handlers=[handler])

    token = _current_callback_manager.set(manager)
    try:
        yield handler
    finally:
        _current_callback_manager.reset(token)


def configure_callbacks(
    handlers: list[Handler] | None = None,
    inheritable_handlers: list[Handler] | None = None,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> CallbackManager:
    """Create a configured callback manager.

    Like LangChain's CallbackManager.configure().

    Example:
        manager = configure_callbacks(
            handlers=[StdoutCallbackHandler()],
            tags=["production"],
            metadata={"user_id": "123"},
        )
    """
    return CallbackManager.configure(
        handlers=handlers,
        inheritable_handlers=inheritable_handlers,
        tags=tags,
        inheritable_tags=inheritable_tags,
        metadata=metadata,
        inheritable_metadata=inheritable_metadata,
    )


def configure_async_callbacks(
    handlers: list[Handler] | None = None,
    inheritable_handlers: list[Handler] | None = None,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> AsyncCallbackManager:
    """Create a configured async callback manager.

    Example:
        manager = configure_async_callbacks(
            handlers=[MyAsyncHandler()],
        )
    """
    return AsyncCallbackManager(
        handlers=handlers,
        inheritable_handlers=inheritable_handlers,
        tags=tags,
        inheritable_tags=inheritable_tags,
        metadata=metadata,
        inheritable_metadata=inheritable_metadata,
    )
