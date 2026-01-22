"""
RustworkX Agent Framework

Фреймворк для создания мульти-агентных систем на основе графов,
решающий проблемы LangGraph:

1. Динамическая топология - изменение графа в runtime
2. Децентрализованная память - локальное состояние агентов
3. First-class graph - полный доступ к структуре графа
4. Альтернативные представления - поддержка эмбеддингов
5. Стратифицированная память - working/long-term с TTL
6. Скрытые каналы - hidden_state/embeddings между агентами
7. **Мультимодельность** - каждый агент может использовать свой LLM

Пример использования (простой):

    from rustworkx_framework import RoleGraph, AgentProfile, MACPRunner
    from rustworkx_framework.builder import build_property_graph

    agents = [
        AgentProfile(identifier="solver", display_name="Solver"),
        AgentProfile(identifier="checker", display_name="Checker"),
    ]

    graph = build_property_graph(
        agents,
        workflow_edges=[("solver", "checker")],
        query="Solve the problem",
    )

    runner = MACPRunner(llm_caller=my_llm_function)
    result = runner.run_round(graph)

Пример использования (мультимодельность через GraphBuilder):

    from rustworkx_framework.builder import GraphBuilder

    builder = GraphBuilder()

    # Агент с GPT-4
    builder.add_agent(
        "solver",
        persona="Expert problem solver",
        llm_backbone="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="$OPENAI_API_KEY",
        temperature=0.7,
        max_tokens=2000
    )

    # Агент с локальной моделью Ollama
    builder.add_agent(
        "analyzer",
        persona="Data analyzer",
        llm_backbone="llama3:70b",
        base_url="http://localhost:11434/v1",
        temperature=0.0
    )

    # Агент с Claude
    builder.add_agent(
        "reviewer",
        persona="Code reviewer",
        llm_backbone="claude-3-opus-20240229",
        base_url="https://api.anthropic.com",
        api_key="$ANTHROPIC_API_KEY"
    )

    builder.add_workflow_edge("solver", "analyzer")
    builder.add_workflow_edge("analyzer", "reviewer")
    builder.add_task(query="Solve and review the problem")
    builder.connect_task_to_agents()

    graph = builder.build()

    # Runner автоматически создаст callers для каждого агента
    from rustworkx_framework import LLMCallerFactory
    factory = LLMCallerFactory.create_openai_factory()
    runner = MACPRunner(llm_factory=factory)
    result = runner.run_round(graph)

Пример использования (мультимодельность через словарь callers):

    from rustworkx_framework import MACPRunner, create_openai_caller

    # Создаём callers для разных моделей
    gpt4_caller = create_openai_caller(model="gpt-4", api_key="...")
    gpt4o_mini_caller = create_openai_caller(model="gpt-4o-mini", api_key="...")

    runner = MACPRunner(
        llm_caller=gpt4_caller,  # default caller
        llm_callers={
            "analyzer": gpt4o_mini_caller,  # использует более дешёвую модель
        }
    )
    result = runner.run_round(graph)

Пример использования (адаптивный с памятью):

    from rustworkx_framework import MACPRunner, RunnerConfig, RoutingPolicy, PruningConfig
    from rustworkx_framework.utils import AgentMemory, MemoryConfig, SharedMemoryPool

    config = RunnerConfig(
        adaptive=True,
        routing_policy=RoutingPolicy.WEIGHTED_TOPO,
        pruning_config=PruningConfig(token_budget=10000),
        enable_hidden_channels=True,
    )
    runner = MACPRunner(llm_caller=my_llm, config=config)
    result = runner.run_round_with_hidden(graph)
"""

__version__ = "0.1.0"

from rustworkx_framework.builder import build_property_graph
from rustworkx_framework.callbacks import (
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
from rustworkx_framework.config import FrameworkSettings, logger, setup_logging
from rustworkx_framework.core import (
    AgentLLMConfig,
    AgentProfile,
    LLMConfig,
    NodeEncoder,
    RoleGraph,
    TaskNode,
)
from rustworkx_framework.execution import (
    AdaptiveScheduler,
    # Dynamic topology
    EarlyStopCondition,
    ExecutionPlan,
    HiddenState,
    # Multi-model support
    LLMCallerFactory,
    MACPResult,
    # Runner
    MACPRunner,
    PruningConfig,
    RoutingPolicy,
    RunnerConfig,
    StepContext,
    StepResult,
    TopologyAction,
    # Scheduling
    build_execution_order,
    create_openai_caller,
    extract_agent_adjacency,
    filter_reachable_agents,
)

__all__ = [
    # Core
    "RoleGraph",
    "AgentProfile",
    "AgentLLMConfig",
    "LLMConfig",
    "TaskNode",
    "NodeEncoder",
    # Execution
    "MACPRunner",
    "MACPResult",
    "RunnerConfig",
    "HiddenState",
    "build_execution_order",
    "extract_agent_adjacency",
    "filter_reachable_agents",
    # Multi-model support
    "LLMCallerFactory",
    "create_openai_caller",
    # Adaptive
    "AdaptiveScheduler",
    "ExecutionPlan",
    "PruningConfig",
    "RoutingPolicy",
    "StepResult",
    # Dynamic topology
    "EarlyStopCondition",
    "StepContext",
    "TopologyAction",
    # Builder
    "build_property_graph",
    # Config
    "FrameworkSettings",
    "logger",
    "setup_logging",
    # Callbacks (LangChain-like)
    "BaseCallbackHandler",
    "AsyncCallbackHandler",
    "CallbackManager",
    "AsyncCallbackManager",
    "StdoutCallbackHandler",
    "MetricsCallbackHandler",
    "FileCallbackHandler",
    "trace_as_callback",
    "collect_metrics",
]
