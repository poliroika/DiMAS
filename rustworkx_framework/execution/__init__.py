"""
Execution components for running agents on a graph.

Provides both simple sequential execution and advanced adaptive execution
with dynamic topology, pruning, fallback and parallel execution.

Features:
- Typed errors and error policies
- Budget tracking (tokens, requests, time)
- Structured execution logging
- Parallel execution with retries
- Integrated agent memory (working/long-term)
- Shared memory pool between agents
- **LangGraph-like streaming execution** (stream/astream)
- **Multi-model support** (per-agent LLM configuration)

Example (simple batch):
    from rustworkx_framework.execution import MACPRunner

    runner = MACPRunner(llm_caller=my_llm)
    result = runner.run_round(graph)

Example (streaming - like LangGraph's stream()):
    from rustworkx_framework.execution import (
        MACPRunner, StreamEventType, format_event
    )

    runner = MACPRunner(llm_caller=my_llm)

    # Sync streaming
    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            print(f"{event.agent_id}: {event.content}")
        elif event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

    # Async streaming
    async for event in runner.astream(graph):
        print(format_event(event))

Example (token-level streaming):
    from rustworkx_framework.execution import MACPRunner, RunnerConfig

    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(
        streaming_llm_caller=my_streaming_llm,  # yields tokens
        config=config
    )

    for event in runner.stream(graph):
        if event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

Example (with budgets and logging):
    from rustworkx_framework.execution import (
        MACPRunner, RunnerConfig, BudgetConfig, ErrorPolicy
    )

    config = RunnerConfig(
        adaptive=True,
        budget_config=BudgetConfig(
            total_token_limit=10000,
            max_prompt_length=4000,
        ),
        enable_logging=True,
    )
    runner = MACPRunner(llm_caller=my_llm, config=config)
    result = runner.run_round(graph)
    print(result.metrics.to_dict())

Example (with memory):
    from rustworkx_framework.execution import (
        MACPRunner, RunnerConfig, MemoryConfig
    )

    config = RunnerConfig(
        enable_memory=True,
        memory_config=MemoryConfig(
            working_max_entries=10,
            long_term_max_entries=50,
        ),
        memory_context_limit=3,  # include last 3 entries in prompt
    )
    runner = MACPRunner(llm_caller=my_llm, config=config)
    result = runner.run_round(graph)

    # Access agent memory after execution
    agent_memory = runner.get_agent_memory("agent_id")

Example (streaming with buffer):
    from rustworkx_framework.execution import (
        MACPRunner, StreamBuffer, stream_to_string
    )

    # Collect all events and get final answer
    buffer = StreamBuffer()
    for event in runner.stream(graph):
        buffer.add(event)
        # process event...

    print(f"Final: {buffer.final_answer}")
    print(f"Agents: {list(buffer.agent_outputs.keys())}")

    # Or use helper function
    answer = stream_to_string(runner.stream(graph))

Example (with tools - function calling):
    from rustworkx_framework.execution import MACPRunner, RunnerConfig
    from rustworkx_framework.tools import ToolRegistry, ShellTool, FunctionTool

    # Create tool registry
    registry = ToolRegistry()
    registry.register(ShellTool(timeout=10))

    # Register custom functions
    func_tool = FunctionTool()

    @func_tool.register
    def calculate(expression: str) -> str:
        \"\"\"Evaluate a mathematical expression.\"\"\"
        return str(eval(expression))

    registry.register(func_tool)

    # Enable tools in config
    config = RunnerConfig(enable_tools=True)
    runner = MACPRunner(llm_caller=my_llm, tool_registry=registry, config=config)

    result = runner.run_round(graph)
"""

# Re-export callbacks for convenience
from ..callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    BaseCallbackHandler,
    CallbackManager,
    FileCallbackHandler,
    MetricsCallbackHandler,
    StdoutCallbackHandler,
    collect_metrics,
    trace_as_callback,
)
from .budget import (
    Budget,
    BudgetConfig,
    BudgetTracker,
    NodeBudget,
)
from .errors import (
    AgentNotFoundError,
    BudgetExceededError,
    ErrorAction,
    # Error policy
    ErrorPolicy,
    # Error types
    ExecutionError,
    ExecutionMetrics,
    RetryExhaustedError,
    # Result types
    StepExecutionResult,
    TimeoutError,
    ValidationError,
)
from .runner import (
    AgentMemory,
    # Dynamic topology
    EarlyStopCondition,
    HiddenState,
    # Multi-model support
    LLMCallerFactory,
    MACPResult,
    MACPRunner,
    MemoryConfig,
    RunnerConfig,
    SharedMemoryPool,
    StepContext,
    TopologyAction,
    create_openai_caller,
)
from .scheduler import (
    # Adaptive scheduling
    AdaptiveScheduler,
    # Conditional routing
    ConditionContext,
    ConditionEvaluator,
    EdgeCondition,
    ExecutionPlan,
    ExecutionStep,
    PruningConfig,
    RoutingPolicy,
    StepResult,
    # Core functions
    build_execution_order,
    extract_agent_adjacency,
    filter_reachable_agents,
    get_incoming_agents,
    get_outgoing_agents,
    get_parallel_groups,
)
from .streaming import (
    AgentErrorEvent,
    AgentOutputEvent,
    AgentStartEvent,
    AsyncStreamCallback,
    BudgetExceededEvent,
    BudgetWarningEvent,
    FallbackEvent,
    MemoryReadEvent,
    MemoryWriteEvent,
    ParallelEndEvent,
    ParallelStartEvent,
    PruneEvent,
    TopologyChangedEvent,
    RunEndEvent,
    # Specific events
    RunStartEvent,
    # Utilities
    StreamBuffer,
    # Callback types
    StreamCallback,
    StreamEvent,
    # Event types
    StreamEventType,
    TokenEvent,
    aprint_stream,
    astream_to_string,
    format_event,
    print_stream,
    stream_to_string,
)

__all__ = [
    # Adaptive scheduling
    "AdaptiveScheduler",
    "AgentErrorEvent",
    # Memory
    "AgentMemory",
    "AgentNotFoundError",
    "AgentOutputEvent",
    "AgentStartEvent",
    "AsyncCallbackHandler",
    "AsyncCallbackManager",
    "AsyncStreamCallback",
    # Callbacks (LangChain-like)
    "BaseCallbackHandler",
    # Budget
    "Budget",
    "BudgetConfig",
    "BudgetExceededError",
    "BudgetExceededEvent",
    "BudgetTracker",
    "BudgetWarningEvent",
    "CallbackManager",
    # Conditional routing
    "ConditionContext",
    "ConditionEvaluator",
    # Dynamic topology
    "EarlyStopCondition",
    "EdgeCondition",
    "ErrorAction",
    "ErrorPolicy",
    # Errors
    "ExecutionError",
    "ExecutionMetrics",
    "ExecutionPlan",
    "ExecutionStep",
    "FallbackEvent",
    "FileCallbackHandler",
    "HiddenState",
    # Multi-model support
    "LLMCallerFactory",
    "MACPResult",
    # Runner
    "MACPRunner",
    "MemoryConfig",
    "MemoryReadEvent",
    "MemoryWriteEvent",
    "MetricsCallbackHandler",
    "NodeBudget",
    "ParallelEndEvent",
    "ParallelStartEvent",
    "PruneEvent",
    "PruningConfig",
    "TopologyChangedEvent",
    "RetryExhaustedError",
    "RoutingPolicy",
    "RunEndEvent",
    "RunStartEvent",
    "RunnerConfig",
    "SharedMemoryPool",
    "StdoutCallbackHandler",
    "StepContext",
    "StepExecutionResult",
    "StepResult",
    "StreamBuffer",
    "StreamCallback",
    "StreamEvent",
    # Streaming
    "StreamEventType",
    "TimeoutError",
    "TokenEvent",
    "TopologyAction",
    "ValidationError",
    "aprint_stream",
    "astream_to_string",
    # Core scheduling functions
    "build_execution_order",
    "collect_metrics",
    "create_openai_caller",
    "extract_agent_adjacency",
    "filter_reachable_agents",
    "format_event",
    "get_incoming_agents",
    "get_outgoing_agents",
    "get_parallel_groups",
    "print_stream",
    "stream_to_string",
    "trace_as_callback",
]
