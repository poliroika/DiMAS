"""
Streaming execution support for real-time output.

Provides LangGraph-like streaming capabilities with:
- Typed stream events for all execution phases
- Sync and async generator interfaces
- Token-level streaming when LLM supports it
- Event callbacks for external integrations

Example (sync streaming):
    from rustworkx_framework.execution import MACPRunner, RunnerConfig

    runner = MACPRunner(llm_caller=my_llm)

    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            print(f"{event.agent_id}: {event.content}")
        elif event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

Example (async streaming):
    async for event in runner.astream(graph):
        match event.event_type:
            case StreamEventType.AGENT_START:
                print(f"Agent {event.agent_id} started...")
            case StreamEventType.AGENT_OUTPUT:
                print(f"Output: {event.content}")
            case StreamEventType.RUN_END:
                print(f"Completed in {event.total_time:.2f}s")

Example (with streaming LLM callback):
    async def streaming_llm(prompt: str) -> AsyncIterator[str]:
        async for chunk in openai_stream(prompt):
            yield chunk

    runner = MACPRunner(
        async_llm_caller=streaming_llm,
        config=RunnerConfig(enable_token_streaming=True)
    )

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)
"""

from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AgentErrorEvent",
    "AgentOutputEvent",
    "AgentStartEvent",
    "AnyStreamEvent",  # Discriminated union type
    "AsyncStreamCallback",
    "BudgetExceededEvent",
    "BudgetWarningEvent",
    "FallbackEvent",
    "MemoryReadEvent",
    "MemoryWriteEvent",
    "ParallelEndEvent",
    "ParallelStartEvent",
    "PruneEvent",
    "TopologyChangedEvent",
    "RunEndEvent",
    # Specific events
    "RunStartEvent",
    # Utilities
    "StreamBuffer",
    # Callback types
    "StreamCallback",
    "StreamEvent",
    # Event types
    "StreamEventType",
    "TokenEvent",
    "aprint_stream",
    "astream_to_string",
    "format_event",
    "print_stream",
    "stream_to_string",
]


class StreamEventType(str, Enum):
    """Types of streaming events."""

    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_OUTPUT = "agent_output"
    AGENT_ERROR = "agent_error"

    # Token-level streaming
    TOKEN = "token"  # noqa: S105  # This is not a password, it's an event type identifier

    # Adaptive execution events
    TOPOLOGY_CHANGED = "topology_changed"
    PRUNE = "prune"
    FALLBACK = "fallback"

    # Parallel execution
    PARALLEL_START = "parallel_start"
    PARALLEL_END = "parallel_end"

    # Memory events
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"

    # Budget events
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"


class StreamEvent(BaseModel):
    """Base streaming event with common fields."""

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "metadata": self.metadata,
        }

    class Config:
        arbitrary_types_allowed = True


class RunStartEvent(StreamEvent):
    """Emitted when execution run starts."""

    event_type: Literal["run_start"] = "run_start"
    query: str = ""
    num_agents: int = 0
    execution_order: list[str] = Field(default_factory=list)
    config_summary: dict[str, Any] = Field(default_factory=dict)


class RunEndEvent(StreamEvent):
    """Emitted when execution run completes."""

    event_type: Literal["run_end"] = "run_end"
    success: bool = True
    final_answer: str = ""
    final_agent_id: str = ""
    total_tokens: int = 0
    total_time: float = 0.0
    executed_agents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentStartEvent(StreamEvent):
    """Emitted when an agent starts processing."""

    event_type: Literal["agent_start"] = "agent_start"
    agent_id: str = ""
    agent_name: str = ""
    step_index: int = 0
    predecessors: list[str] = Field(default_factory=list)
    prompt_preview: str = ""  # First N chars of prompt


class AgentOutputEvent(StreamEvent):
    """Emitted when an agent produces output."""

    event_type: Literal["agent_output"] = "agent_output"
    agent_id: str = ""
    agent_name: str = ""
    content: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0
    is_final: bool = False  # True if this is the final agent


class AgentErrorEvent(StreamEvent):
    """Emitted when an agent encounters an error."""

    event_type: Literal["agent_error"] = "agent_error"
    agent_id: str = ""
    error_type: str = ""
    error_message: str = ""
    will_retry: bool = False
    attempt: int = 0
    max_attempts: int = 0


class TokenEvent(StreamEvent):
    """Emitted for each token during streaming LLM output."""

    event_type: Literal["token"] = "token"
    agent_id: str = ""
    token: str = ""
    token_index: int = 0
    is_first: bool = False
    is_last: bool = False


class TopologyChangedEvent(StreamEvent):
    """Emitted when execution plan is modified by topology hooks."""

    event_type: Literal["topology_changed"] = "topology_changed"
    reason: str = ""
    old_remaining: list[str] = Field(default_factory=list)
    new_remaining: list[str] = Field(default_factory=list)
    change_count: int = 0


class PruneEvent(StreamEvent):
    """Emitted when an agent is pruned from execution."""

    event_type: Literal["prune"] = "prune"
    agent_id: str = ""
    reason: str = ""


class FallbackEvent(StreamEvent):
    """Emitted when fallback agent is activated."""

    event_type: Literal["fallback"] = "fallback"
    failed_agent_id: str = ""
    fallback_agent_id: str = ""
    attempt: int = 0


class ParallelStartEvent(StreamEvent):
    """Emitted when parallel execution group starts."""

    event_type: Literal["parallel_start"] = "parallel_start"
    agent_ids: list[str] = Field(default_factory=list)
    group_index: int = 0


class ParallelEndEvent(StreamEvent):
    """Emitted when parallel execution group completes."""

    event_type: Literal["parallel_end"] = "parallel_end"
    agent_ids: list[str] = Field(default_factory=list)
    group_index: int = 0
    successful: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class MemoryWriteEvent(StreamEvent):
    """Emitted when agent writes to memory."""

    event_type: Literal["memory_write"] = "memory_write"
    agent_id: str = ""
    key: str = ""
    value_preview: str = ""


class MemoryReadEvent(StreamEvent):
    """Emitted when agent reads from memory."""

    event_type: Literal["memory_read"] = "memory_read"
    agent_id: str = ""
    entries_count: int = 0


class BudgetWarningEvent(StreamEvent):
    """Emitted when budget threshold is approached."""

    event_type: Literal["budget_warning"] = "budget_warning"
    budget_type: str = ""  # tokens, time, requests
    current: float = 0.0
    limit: float = 0.0
    ratio: float = 0.0


class BudgetExceededEvent(StreamEvent):
    """Emitted when budget is exceeded."""

    event_type: Literal["budget_exceeded"] = "budget_exceeded"
    budget_type: str = ""
    current: float = 0.0
    limit: float = 0.0


# Discriminated union type for all stream events
# This allows type checkers to narrow the type based on event_type
AnyStreamEvent = Annotated[
    RunStartEvent
    | RunEndEvent
    | AgentStartEvent
    | AgentOutputEvent
    | AgentErrorEvent
    | TokenEvent
    | TopologyChangedEvent
    | PruneEvent
    | FallbackEvent
    | ParallelStartEvent
    | ParallelEndEvent
    | MemoryWriteEvent
    | MemoryReadEvent
    | BudgetWarningEvent
    | BudgetExceededEvent,
    Field(discriminator="event_type"),
]

# Callback type aliases
StreamCallback = Callable[[AnyStreamEvent], None]
AsyncStreamCallback = Callable[[AnyStreamEvent], Any]  # Can be async


class StreamBuffer:
    """
    Buffer for collecting stream events and building final output.

    Example:
        buffer = StreamBuffer()

        for event in runner.stream(graph):
            buffer.add(event)
            if event.event_type == StreamEventType.TOKEN:
                print(event.token, end="")

        result = buffer.get_final_output()
        all_events = buffer.events

    """

    def __init__(self):
        self._events: list[StreamEvent] = []
        self._agent_outputs: dict[str, str] = {}
        self._current_tokens: dict[str, list[str]] = {}
        self._final_answer: str = ""
        self._final_agent_id: str = ""

    def add(self, event: StreamEvent) -> None:
        """Add event to buffer and update state."""
        self._events.append(event)

        if isinstance(event, TokenEvent):
            if event.agent_id not in self._current_tokens:
                self._current_tokens[event.agent_id] = []
            self._current_tokens[event.agent_id].append(event.token)

            if event.is_last:
                self._agent_outputs[event.agent_id] = "".join(self._current_tokens[event.agent_id])
                self._current_tokens[event.agent_id] = []

        elif isinstance(event, AgentOutputEvent):
            self._agent_outputs[event.agent_id] = event.content
            if event.is_final:
                self._final_answer = event.content
                self._final_agent_id = event.agent_id

        elif isinstance(event, RunEndEvent):
            self._final_answer = event.final_answer
            self._final_agent_id = event.final_agent_id

    @property
    def events(self) -> list[StreamEvent]:
        """All collected events."""
        return self._events

    @property
    def agent_outputs(self) -> dict[str, str]:
        """Map of agent_id to their final outputs."""
        return self._agent_outputs

    @property
    def final_answer(self) -> str:
        """Final answer from the last agent."""
        return self._final_answer

    @property
    def final_agent_id(self) -> str:
        """ID of the agent that produced final answer."""
        return self._final_agent_id

    def get_output_for(self, agent_id: str) -> str:
        """Get output for specific agent."""
        # Check completed outputs first
        if agent_id in self._agent_outputs:
            return self._agent_outputs[agent_id]
        # Check in-progress token streams
        if agent_id in self._current_tokens:
            return "".join(self._current_tokens[agent_id])
        return ""

    def clear(self) -> None:
        """Clear all buffered data."""
        self._events.clear()
        self._agent_outputs.clear()
        self._current_tokens.clear()
        self._final_answer = ""
        self._final_agent_id = ""


def stream_to_string(stream: Iterator[StreamEvent]) -> str:
    """
    Consume stream and return final answer.

    Example:
        answer = stream_to_string(runner.stream(graph))

    """
    buffer = StreamBuffer()
    for event in stream:
        buffer.add(event)
    return buffer.final_answer


async def astream_to_string(stream: AsyncIterator[StreamEvent]) -> str:
    """
    Consume async stream and return final answer.

    Example:
        answer = await astream_to_string(runner.astream(graph))

    """
    buffer = StreamBuffer()
    async for event in stream:
        buffer.add(event)
    return buffer.final_answer


def format_event(event: StreamEvent, *, verbose: bool = False) -> str:
    """
    Format event for display/logging.

    Args:
        event: Stream event to format
        verbose: Include full details if True

    Returns:
        Formatted string representation

    """
    timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]

    if isinstance(event, RunStartEvent):
        return f"[{timestamp}] 🚀 Run started: {event.num_agents} agents"

    if isinstance(event, RunEndEvent):
        status = "✅" if event.success else "❌"
        return f"[{timestamp}] {status} Run completed in {event.total_time:.2f}s ({event.total_tokens} tokens)"

    if isinstance(event, AgentStartEvent):
        name = event.agent_name or event.agent_id
        return f"[{timestamp}] ▶️  {name} started (step {event.step_index})"

    if isinstance(event, AgentOutputEvent):
        name = event.agent_name or event.agent_id
        # Maximum length for content preview
        max_preview_length = 100
        if len(event.content) > max_preview_length:
            preview = event.content[:max_preview_length] + "..."
        else:
            preview = event.content
        if verbose:
            return f"[{timestamp}] 💬 {name}: {event.content}"
        return f"[{timestamp}] 💬 {name}: {preview}"

    if isinstance(event, AgentErrorEvent):
        retry = f" (retry {event.attempt}/{event.max_attempts})" if event.will_retry else ""
        return f"[{timestamp}] ⚠️  {event.agent_id} error: {event.error_message}{retry}"

    if isinstance(event, TokenEvent):
        return event.token  # Just the token for streaming display

    if isinstance(event, TopologyChangedEvent):
        return f"[{timestamp}] 🔄 Topology changed #{event.change_count}: {event.reason}"

    if isinstance(event, PruneEvent):
        return f"[{timestamp}] ✂️  Pruned {event.agent_id}: {event.reason}"

    if isinstance(event, FallbackEvent):
        return f"[{timestamp}] 🔀 Fallback: {event.failed_agent_id} → {event.fallback_agent_id}"

    if isinstance(event, ParallelStartEvent):
        agents = ", ".join(event.agent_ids)
        return f"[{timestamp}] ⚡ Parallel group {event.group_index}: [{agents}]"

    if isinstance(event, ParallelEndEvent):
        success_count = len(event.successful)
        total = len(event.agent_ids)
        return f"[{timestamp}] ⚡ Parallel group {event.group_index} done: {success_count}/{total} succeeded"

    if isinstance(event, BudgetWarningEvent):
        return f"[{timestamp}] ⚠️  Budget warning: {event.budget_type} at {event.ratio:.0%}"

    if isinstance(event, BudgetExceededEvent):
        return f"[{timestamp}] 🛑 Budget exceeded: {event.budget_type}"

    return f"[{timestamp}] {event.event_type}"


def print_stream(
    stream: Iterator[StreamEvent],
    *,
    show_tokens: bool = True,
    verbose: bool = False,
) -> str:
    """
    Print stream events and return final answer.

    Args:
        stream: Event stream to consume
        show_tokens: Print individual tokens inline
        verbose: Show full event details

    Returns:
        Final answer string

    Example:
        answer = print_stream(runner.stream(graph))

    """
    buffer = StreamBuffer()
    current_agent: str | None = None

    for event in stream:
        buffer.add(event)

        if isinstance(event, TokenEvent):
            if show_tokens:
                if current_agent != event.agent_id:
                    if current_agent is not None:
                        pass  # New line for new agent
                    current_agent = event.agent_id
                if event.is_last:
                    pass  # End line after last token

        elif isinstance(event, AgentOutputEvent):
            if not show_tokens:  # Only print if not already streaming tokens
                pass

        elif isinstance(event, (RunStartEvent, RunEndEvent, AgentStartEvent, AgentErrorEvent)):
            if current_agent is not None:
                current_agent = None

        elif verbose:
            pass

    return buffer.final_answer


async def aprint_stream(
    stream: AsyncIterator[StreamEvent],
    *,
    show_tokens: bool = True,
    verbose: bool = False,
) -> str:
    """
    Async version of print_stream.

    Args:
        stream: Async event stream to consume
        show_tokens: Print individual tokens inline
        verbose: Show full event details

    Returns:
        Final answer string

    """
    buffer = StreamBuffer()
    current_agent: str | None = None

    async for event in stream:
        buffer.add(event)

        if isinstance(event, TokenEvent):
            if show_tokens:
                if current_agent != event.agent_id:
                    if current_agent is not None:
                        pass
                    current_agent = event.agent_id
                if event.is_last:
                    pass

        elif isinstance(event, AgentOutputEvent):
            if not show_tokens:
                pass

        elif isinstance(event, (RunStartEvent, RunEndEvent, AgentStartEvent, AgentErrorEvent)):
            if current_agent is not None:
                current_agent = None

        elif verbose:
            pass

    return buffer.final_answer
