"""Callback system for execution monitoring.

Provides LangChain-like callback handling for monitoring and logging
execution events.

Example (basic usage):
    from rustworkx_framework import MACPRunner
    from rustworkx_framework.callbacks import StdoutCallbackHandler

    runner = MACPRunner(
        llm_caller=my_llm,
        callbacks=[StdoutCallbackHandler()]
    )
    result = runner.run_round(graph)

Example (context manager):
    from rustworkx_framework.callbacks import trace_as_callback, MetricsCallbackHandler

    with trace_as_callback(handlers=[MetricsCallbackHandler()]) as manager:
        runner.run_round(graph)

    # Get metrics from handler
    metrics_handler = manager.handlers[0]
    print(metrics_handler.get_metrics())

Example (custom handler):
    from rustworkx_framework.callbacks import BaseCallbackHandler

    class MyHandler(BaseCallbackHandler):
        def on_agent_end(self, *, agent_id, output, tokens_used, **kwargs):
            print(f"{agent_id} used {tokens_used} tokens")

    runner = MACPRunner(callbacks=[MyHandler()])
"""

from .base import AsyncCallbackHandler, BaseCallbackHandler, CallbackHandlerMixin
from .context import (
    collect_metrics,
    configure_async_callbacks,
    configure_callbacks,
    get_callback_manager,
    set_callback_manager,
    trace_as_callback,
)
from .events import (
    AgentEndEvent,
    AgentErrorEvent,
    AgentRetryEvent,
    AgentStartEvent,
    BaseEvent,
    BudgetExceededEvent,
    BudgetWarningEvent,
    EventType,
    FallbackEvent,
    MemoryReadEvent,
    MemoryWriteEvent,
    ParallelEndEvent,
    ParallelStartEvent,
    PlanCreatedEvent,
    PruneEvent,
    ReplanEvent,
    RunEndEvent,
    RunStartEvent,
    TokenEvent,
)
from .handlers import (
    FileCallbackHandler,
    MetricsCallbackHandler,
    StdoutCallbackHandler,
)
from .manager import AsyncCallbackManager, CallbackManager

# Type alias for handlers
Handler = BaseCallbackHandler | AsyncCallbackHandler

__all__ = [
    # Base handlers
    "CallbackHandlerMixin",
    "BaseCallbackHandler",
    "AsyncCallbackHandler",
    "Handler",
    # Managers
    "CallbackManager",
    "AsyncCallbackManager",
    # Built-in handlers
    "StdoutCallbackHandler",
    "MetricsCallbackHandler",
    "FileCallbackHandler",
    # Context
    "trace_as_callback",
    "collect_metrics",
    "get_callback_manager",
    "set_callback_manager",
    "configure_callbacks",
    "configure_async_callbacks",
    # Events
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
