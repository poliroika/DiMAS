"""MACPRunner — исполнитель Multi-Agent Communication Protocol.

Поддерживает как простое последовательное выполнение, так и адаптивный режим
с перепланировкой, pruning, fallback и параллельным выполнением.

Также поддерживает:
- Стратифицированную память (working/long-term)
- Скрытые каналы (hidden_state, embeddings)
- Протокол обмена с разделением видимых/скрытых данных
- Типизированные ошибки и политики обработки
- Бюджеты токенов/запросов на уровне графа и узлов
- Структурированное логирование событий выполнения
- **Мультимодельность**: каждый агент может использовать свой LLM

Мультимодельность:
    MACPRunner теперь поддерживает три способа задания LLM:

    1. Единый llm_caller для всех агентов (как раньше)
    2. Словарь llm_callers: dict[agent_id, Callable] для разных моделей
    3. LLMCallerFactory, которая создаёт caller на основе AgentLLMConfig

    Приоритет: agent-specific caller > factory > default caller
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import datetime
from typing import Any, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field

from ..callbacks import (
    CallbackManager,
    Handler,
    get_callback_manager,
)
from ..core.agent import AgentLLMConfig
from ..utils.memory import AgentMemory, MemoryConfig, SharedMemoryPool
from .budget import BudgetConfig, BudgetTracker
from .errors import (
    ErrorPolicy,
    ExecutionError,
    ExecutionMetrics,
)
from .scheduler import (
    AdaptiveScheduler,
    ConditionContext,
    ExecutionPlan,
    PruningConfig,
    RoutingPolicy,
    StepResult,
    build_execution_order,
    extract_agent_adjacency,
    filter_reachable_agents,
    get_incoming_agents,
)
from .streaming import (
    AgentErrorEvent,
    AgentOutputEvent,
    AgentStartEvent,
    AnyStreamEvent,
    AsyncStreamCallback,
    FallbackEvent,
    ParallelEndEvent,
    ParallelStartEvent,
    PruneEvent,
    ReplanEvent,
    RunEndEvent,
    RunStartEvent,
    StreamCallback,
    StreamEvent,
    StreamEventType,
    TokenEvent,
)

__all__ = [
    "MACPRunner",
    "MACPResult",
    "RunnerConfig",
    "HiddenState",
    "ErrorPolicy",
    "BudgetConfig",
    "ExecutionMetrics",
    "MemoryConfig",
    "SharedMemoryPool",
    "AgentMemory",
    # Multi-model support
    "LLMCallerFactory",
    "LLMCallerProtocol",
    "create_openai_caller",
    # Streaming types
    "StreamEvent",
    "StreamEventType",
    "StreamCallback",
    "AsyncStreamCallback",
    # Dynamic topology
    "StepContext",
    "TopologyAction",
    "TopologyHook",
    "AsyncTopologyHook",
    "EarlyStopCondition",
]


# Type aliases for LLM callers
LLMCallerProtocol = Callable[[str], str]
AsyncLLMCallerProtocol = Callable[[str], Awaitable[str]]


class LLMCallerFactory:
    """Фабрика для создания LLM callers на основе конфигурации агента.

    Поддерживает:
    - OpenAI-совместимые API (OpenAI, Azure, Ollama, vLLM, LiteLLM)
    - Кеширование созданных callers по (base_url, api_key, model_name)
    - Fallback на default caller если конфигурация не задана

    Example:
        factory = LLMCallerFactory(
            default_caller=my_default_caller,
            default_config=LLMConfig(model_name="gpt-4", base_url="...")
        )

        # Автоматически создаст caller для агента
        caller = factory.get_caller(agent.get_llm_config())
        response = caller(prompt)

        # Или с OpenAI
        factory = LLMCallerFactory.create_openai_factory(
            default_api_key="sk-...",
            default_model="gpt-4"
        )
    """

    def __init__(
        self,
        default_caller: LLMCallerProtocol | None = None,
        default_async_caller: AsyncLLMCallerProtocol | None = None,
        default_config: AgentLLMConfig | None = None,
        caller_builder: Callable[[AgentLLMConfig], LLMCallerProtocol] | None = None,
        async_caller_builder: Callable[[AgentLLMConfig], AsyncLLMCallerProtocol] | None = None,
    ):
        """Создать фабрику LLM callers.

        Args:
            default_caller: Caller по умолчанию (для агентов без кастомной конфигурации).
            default_async_caller: Async caller по умолчанию.
            default_config: Конфигурация по умолчанию (merge с agent config).
            caller_builder: Функция для создания sync caller из конфигурации.
            async_caller_builder: Функция для создания async caller из конфигурации.
        """
        self.default_caller = default_caller
        self.default_async_caller = default_async_caller
        self.default_config = default_config
        self.caller_builder = caller_builder
        self.async_caller_builder = async_caller_builder

        # Cache callers by config hash
        self._caller_cache: dict[str, LLMCallerProtocol] = {}
        self._async_caller_cache: dict[str, AsyncLLMCallerProtocol] = {}

    def _config_key(self, config: AgentLLMConfig) -> str:
        """Создать ключ кеша для конфигурации."""
        return f"{config.base_url}|{config.model_name}|{config.api_key}"

    def get_caller(
        self,
        config: AgentLLMConfig | None = None,
        agent_id: str | None = None,
    ) -> LLMCallerProtocol | None:
        """Получить sync caller для данной конфигурации.

        Args:
            config: LLM конфигурация агента.
            agent_id: ID агента (для логирования).

        Returns:
            LLM caller или None если не удалось создать.
        """
        if config is None or not config.is_configured():
            return self.default_caller

        # Merge with default config
        if self.default_config:
            effective_config = config
            # Use agent config values, fallback to default
            effective_config = AgentLLMConfig(
                model_name=config.model_name or self.default_config.model_name,
                base_url=config.base_url or self.default_config.base_url,
                api_key=config.api_key or self.default_config.api_key,
                max_tokens=config.max_tokens
                if config.max_tokens is not None
                else self.default_config.max_tokens,
                temperature=config.temperature
                if config.temperature is not None
                else self.default_config.temperature,
                timeout=config.timeout
                if config.timeout is not None
                else self.default_config.timeout,
                top_p=config.top_p if config.top_p is not None else self.default_config.top_p,
                stop_sequences=config.stop_sequences or self.default_config.stop_sequences,
                extra_params={**self.default_config.extra_params, **config.extra_params},
            )
            config = effective_config

        cache_key = self._config_key(config)

        if cache_key in self._caller_cache:
            return self._caller_cache[cache_key]

        if self.caller_builder:
            caller = self.caller_builder(config)
            self._caller_cache[cache_key] = caller
            return caller

        return self.default_caller

    def get_async_caller(
        self,
        config: AgentLLMConfig | None = None,
        agent_id: str | None = None,
    ) -> AsyncLLMCallerProtocol | None:
        """Получить async caller для данной конфигурации."""
        if config is None or not config.is_configured():
            return self.default_async_caller

        # Merge with default config
        if self.default_config:
            effective_config = AgentLLMConfig(
                model_name=config.model_name or self.default_config.model_name,
                base_url=config.base_url or self.default_config.base_url,
                api_key=config.api_key or self.default_config.api_key,
                max_tokens=config.max_tokens
                if config.max_tokens is not None
                else self.default_config.max_tokens,
                temperature=config.temperature
                if config.temperature is not None
                else self.default_config.temperature,
                timeout=config.timeout
                if config.timeout is not None
                else self.default_config.timeout,
                top_p=config.top_p if config.top_p is not None else self.default_config.top_p,
                stop_sequences=config.stop_sequences or self.default_config.stop_sequences,
                extra_params={**self.default_config.extra_params, **config.extra_params},
            )
            config = effective_config

        cache_key = self._config_key(config)

        if cache_key in self._async_caller_cache:
            return self._async_caller_cache[cache_key]

        if self.async_caller_builder:
            caller = self.async_caller_builder(config)
            self._async_caller_cache[cache_key] = caller
            return caller

        return self.default_async_caller

    @classmethod
    def create_openai_factory(
        cls,
        default_api_key: str | None = None,
        default_model: str = "gpt-4",
        default_base_url: str = "https://api.openai.com/v1",
        default_temperature: float = 0.7,
        default_max_tokens: int = 2000,
    ) -> "LLMCallerFactory":
        """Создать фабрику для OpenAI-совместимых API.

        Автоматически создаёт callers используя openai library.

        Args:
            default_api_key: API ключ по умолчанию (или $OPENAI_API_KEY).
            default_model: Модель по умолчанию.
            default_base_url: URL API по умолчанию.
            default_temperature: Температура по умолчанию.
            default_max_tokens: Максимум токенов по умолчанию.

        Example:
            factory = LLMCallerFactory.create_openai_factory(
                default_api_key="$OPENAI_API_KEY"
            )
            runner = MACPRunner(llm_factory=factory)
        """
        import os

        # Resolve default API key
        if default_api_key and default_api_key.startswith("$"):
            default_api_key = os.environ.get(default_api_key[1:])

        default_config = AgentLLMConfig(
            model_name=default_model,
            base_url=default_base_url,
            api_key=default_api_key,
            temperature=default_temperature,
            max_tokens=default_max_tokens,
        )

        def build_sync_caller(config: AgentLLMConfig) -> LLMCallerProtocol:
            return _create_openai_caller_from_config(config)

        def build_async_caller(config: AgentLLMConfig) -> AsyncLLMCallerProtocol:
            return _create_async_openai_caller_from_config(config)

        return cls(
            default_config=default_config,
            caller_builder=build_sync_caller,
            async_caller_builder=build_async_caller,
        )


def _create_openai_caller_from_config(config: AgentLLMConfig) -> LLMCallerProtocol:
    """Создать OpenAI-совместимый sync caller из конфигурации."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    api_key = config.resolve_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout or 60.0,
    )

    gen_params = config.to_generation_params()
    model = config.model_name or "gpt-4"

    def caller(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **gen_params,
        )
        return response.choices[0].message.content or ""

    return caller


def _create_async_openai_caller_from_config(config: AgentLLMConfig) -> AsyncLLMCallerProtocol:
    """Создать OpenAI-совместимый async caller из конфигурации."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    api_key = config.resolve_api_key()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout or 60.0,
    )

    gen_params = config.to_generation_params()
    model = config.model_name or "gpt-4"

    async def caller(prompt: str) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **gen_params,
        )
        return response.choices[0].message.content or ""

    return caller


def create_openai_caller(
    api_key: str | None = None,
    model: str = "gpt-4",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> LLMCallerProtocol:
    """Создать простой OpenAI caller (convenience function).

    Example:
        caller = create_openai_caller(api_key="sk-...", model="gpt-4")
        runner = MACPRunner(llm_caller=caller)
    """
    config = AgentLLMConfig(
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _create_openai_caller_from_config(config)


class HiddenState(BaseModel):
    """Скрытое состояние/эмбеддинги агента, передаваемые по скрытым каналам."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# ДИНАМИЧЕСКАЯ ТОПОЛОГИЯ (runtime graph modification)
# =============================================================================


class StepContext(BaseModel):
    """Контекст текущего шага выполнения для принятия решений о модификации графа.

    Передаётся в hooks для динамической модификации топологии во время выполнения.

    Attributes:
        agent_id: ID текущего агента.
        response: Ответ агента (если уже получен).
        messages: Все ответы агентов на данный момент.
        step_result: Результат текущего шага.
        execution_order: Порядок выполнения до текущего момента.
        remaining_agents: Агенты, которые ещё не выполнены.
        query: Исходный запрос.
        total_tokens: Использовано токенов.
        metadata: Произвольные метаданные.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    response: str | None = None
    messages: dict[str, str] = Field(default_factory=dict)
    step_result: StepResult | None = None
    execution_order: list[str] = Field(default_factory=list)
    remaining_agents: list[str] = Field(default_factory=list)
    query: str = ""
    total_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopologyAction(BaseModel):
    """Действие для модификации топологии графа во время выполнения.

    Возвращается из hooks для динамического изменения графа.
    """

    # Ранняя остановка
    early_stop: bool = False
    early_stop_reason: str | None = None

    # Добавление/удаление рёбер
    add_edges: list[tuple[str, str, float]] = Field(default_factory=list)  # (src, tgt, weight)
    remove_edges: list[tuple[str, str]] = Field(default_factory=list)

    # Пропуск агентов
    skip_agents: list[str] = Field(default_factory=list)

    # Принудительное выполнение агентов (даже если не в плане)
    force_agents: list[str] = Field(default_factory=list)

    # Изменить конечного агента
    new_end_agent: str | None = None

    # Триггер перепланирования
    trigger_replan: bool = False


# Type alias для hooks
TopologyHook = Callable[[StepContext, Any], TopologyAction | None]
AsyncTopologyHook = Callable[[StepContext, Any], Awaitable[TopologyAction | None]]


class EarlyStopCondition:
    """Условие для ранней остановки выполнения.

    Позволяет прекратить выполнение графа на основе произвольного условия.
    Условие — это любая функция (ctx: StepContext) -> bool.

    Attributes:
        condition: Функция проверки условия.
        reason: Причина остановки (для логирования/отладки).
        after_agents: Список агентов, после которых проверять условие.
        min_agents_executed: Минимум агентов, которые должны выполниться до проверки.

    Example:
        # Произвольное условие — любая логика
        stop_condition = EarlyStopCondition(
            condition=lambda ctx: my_complex_check(ctx.messages, ctx.metadata),
            reason="Custom condition met"
        )

        # Условие на основе метрик
        stop_condition = EarlyStopCondition(
            condition=lambda ctx: ctx.metadata.get("quality_score", 0) > 0.9,
            reason="Quality threshold reached"
        )

        # Остановка после достижения лимита токенов
        stop_condition = EarlyStopCondition(
            condition=lambda ctx: ctx.total_tokens > 5000,
            reason="Token limit reached"
        )

        runner = MACPRunner(
            llm_caller=my_llm,
            config=RunnerConfig(early_stop_conditions=[stop_condition])
        )
    """

    def __init__(
        self,
        condition: Callable[[StepContext], bool],
        reason: str = "Early stop condition met",
        after_agents: list[str] | None = None,
        min_agents_executed: int = 0,
    ):
        """Создать условие ранней остановки.

        Args:
            condition: Произвольная функция проверки условия (ctx -> bool).
            reason: Причина остановки (для логирования).
            after_agents: Список агентов, после которых проверять условие
                         (если None — проверять после каждого).
            min_agents_executed: Минимальное количество агентов, которые должны
                                выполниться перед проверкой условия.
        """
        self.condition = condition
        self.reason = reason
        self.after_agents = after_agents
        self.min_agents_executed = min_agents_executed

    def should_stop(self, ctx: StepContext) -> tuple[bool, str]:
        """Проверить, нужно ли остановиться."""
        # Проверить минимум выполненных агентов
        if len(ctx.execution_order) < self.min_agents_executed:
            return False, ""

        # Проверить, подходит ли текущий агент
        if self.after_agents and ctx.agent_id not in self.after_agents:
            return False, ""

        try:
            if self.condition(ctx):
                return True, self.reason
        except Exception:
            pass

        return False, ""

    # =========================================================================
    # ФАБРИЧНЫЕ МЕТОДЫ для типовых условий
    # =========================================================================

    @classmethod
    def on_keyword(
        cls,
        keyword: str,
        reason: str | None = None,
        case_sensitive: bool = False,
        in_last_response: bool = True,
    ) -> "EarlyStopCondition":
        """Остановить если в ответе есть ключевое слово.

        Args:
            keyword: Ключевое слово для поиска.
            reason: Причина остановки.
            case_sensitive: Учитывать регистр.
            in_last_response: Искать только в последнем ответе (иначе во всех).

        Example:
            stop = EarlyStopCondition.on_keyword("FINAL ANSWER")
        """

        def check(ctx: StepContext) -> bool:
            if in_last_response:
                text = ctx.response or ""
            else:
                text = " ".join(ctx.messages.values())

            if case_sensitive:
                return keyword in text
            return keyword.lower() in text.lower()

        return cls(condition=check, reason=reason or f"Keyword '{keyword}' found")

    @classmethod
    def on_token_limit(
        cls,
        max_tokens: int,
        reason: str | None = None,
    ) -> "EarlyStopCondition":
        """Остановить при достижении лимита токенов.

        Args:
            max_tokens: Максимум токенов.
            reason: Причина остановки.

        Example:
            stop = EarlyStopCondition.on_token_limit(5000)
        """
        return cls(
            condition=lambda ctx: ctx.total_tokens >= max_tokens,
            reason=reason or f"Token limit {max_tokens} reached",
        )

    @classmethod
    def on_agent_count(
        cls,
        max_agents: int,
        reason: str | None = None,
    ) -> "EarlyStopCondition":
        """Остановить после выполнения N агентов.

        Args:
            max_agents: Максимум агентов.
            reason: Причина остановки.

        Example:
            stop = EarlyStopCondition.on_agent_count(3)
        """
        return cls(
            condition=lambda ctx: len(ctx.execution_order) >= max_agents,
            reason=reason or f"Agent count limit {max_agents} reached",
        )

    @classmethod
    def on_metadata(
        cls,
        key: str,
        value: Any = None,
        comparator: Callable[[Any, Any], bool] | None = None,
        reason: str | None = None,
    ) -> "EarlyStopCondition":
        """Остановить на основе значения в metadata.

        Args:
            key: Ключ в metadata.
            value: Ожидаемое значение (если None — проверяется только наличие).
            comparator: Функция сравнения (default: ==).
            reason: Причина остановки.

        Example:
            # Остановить если quality > 0.9
            stop = EarlyStopCondition.on_metadata(
                "quality", 0.9,
                comparator=lambda v, threshold: v > threshold
            )
        """

        def check(ctx: StepContext) -> bool:
            if key not in ctx.metadata:
                return False
            actual = ctx.metadata[key]
            if value is None:
                return True
            if comparator:
                return comparator(actual, value)
            return actual == value

        return cls(condition=check, reason=reason or f"Metadata condition met: {key}")

    @classmethod
    def on_custom(
        cls,
        condition: Callable[[StepContext], bool],
        reason: str = "Custom condition met",
        **kwargs,
    ) -> "EarlyStopCondition":
        """Создать условие с произвольной функцией (алиас конструктора).

        Args:
            condition: Произвольная функция проверки.
            reason: Причина остановки.
            **kwargs: Дополнительные параметры (after_agents, min_agents_executed).

        Example:
            stop = EarlyStopCondition.on_custom(
                lambda ctx: my_rl_agent.should_stop(ctx.messages),
                reason="RL agent decided to stop"
            )
        """
        return cls(condition=condition, reason=reason, **kwargs)

    @classmethod
    def combine_any(
        cls,
        conditions: list["EarlyStopCondition"],
        reason: str = "One of conditions met",
    ) -> "EarlyStopCondition":
        """Объединить условия через OR (остановить если хотя бы одно выполнено).

        Args:
            conditions: Список условий.
            reason: Причина остановки.

        Example:
            stop = EarlyStopCondition.combine_any([
                EarlyStopCondition.on_keyword("DONE"),
                EarlyStopCondition.on_token_limit(10000),
            ])
        """

        def check(ctx: StepContext) -> bool:
            for cond in conditions:
                should_stop, _ = cond.should_stop(ctx)
                if should_stop:
                    return True
            return False

        return cls(condition=check, reason=reason)

    @classmethod
    def combine_all(
        cls,
        conditions: list["EarlyStopCondition"],
        reason: str = "All conditions met",
    ) -> "EarlyStopCondition":
        """Объединить условия через AND (остановить если все выполнены).

        Args:
            conditions: Список условий.
            reason: Причина остановки.

        Example:
            stop = EarlyStopCondition.combine_all([
                EarlyStopCondition.on_keyword("answer"),
                EarlyStopCondition.on_metadata("confidence", 0.8, lambda v, t: v > t),
            ])
        """

        def check(ctx: StepContext) -> bool:
            for cond in conditions:
                should_stop, _ = cond.should_stop(ctx)
                if not should_stop:
                    return False
            return True

        return cls(condition=check, reason=reason)


class MACPResult(NamedTuple):
    """Результат выполнения MACP с сообщениями, метриками и состояниями."""

    messages: dict[str, str]
    final_answer: str
    final_agent_id: str
    execution_order: list[str]
    agent_states: dict[str, list[dict[str, Any]]] | None = None
    step_results: dict[str, StepResult] | None = None
    total_tokens: int = 0
    total_time: float = 0.0
    replanning_count: int = 0
    fallback_count: int = 0
    pruned_agents: list[str] | None = None
    errors: list[ExecutionError] | None = None
    hidden_states: dict[str, HiddenState] | None = None
    metrics: ExecutionMetrics | None = None
    budget_summary: dict[str, Any] | None = None
    # Dynamic topology
    early_stopped: bool = False
    early_stop_reason: str | None = None
    topology_modifications: int = 0  # Количество модификаций топологии


class RunnerConfig(BaseModel):
    """Настройки раннера: таймауты, адаптивность, параллель, бюджеты и логирование."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timeout: float = 60.0
    adaptive: bool = False
    enable_replanning: bool = True
    replan_on_error_only: bool = False
    enable_parallel: bool = True
    max_parallel_size: int = 5
    max_retries: int = 2
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    update_states: bool = True

    routing_policy: RoutingPolicy = RoutingPolicy.TOPOLOGICAL
    pruning_config: PruningConfig | None = None

    enable_hidden_channels: bool = False
    hidden_combine_strategy: str = "mean"
    pass_embeddings: bool = True

    error_policy: ErrorPolicy = Field(default_factory=ErrorPolicy)

    budget_config: BudgetConfig | None = None

    callbacks: list[Handler] = Field(default_factory=list)

    # Memory integration
    enable_memory: bool = False
    memory_config: MemoryConfig | None = None
    memory_context_limit: int = 5  # сколько записей из памяти включать в промпт

    # Streaming configuration
    enable_token_streaming: bool = False  # Enable token-level streaming if LLM supports it
    stream_callbacks: list[StreamCallback] = Field(default_factory=list)
    async_stream_callbacks: list[AsyncStreamCallback] = Field(default_factory=list)
    prompt_preview_length: int = 100  # How many chars of prompt to include in events

    # Task query broadcast mode
    broadcast_task_to_all: bool = True  # True: task query передаётся всем агентам
    # False: только агентам, напрямую соединённым с task нодой

    # Dynamic topology (изменение графа во время выполнения)
    enable_dynamic_topology: bool = False  # Включить поддержку динамической топологии
    topology_hooks: list[Any] = Field(default_factory=list)  # TopologyHook callbacks
    async_topology_hooks: list[Any] = Field(default_factory=list)  # AsyncTopologyHook callbacks
    early_stop_conditions: list[Any] = Field(default_factory=list)  # EarlyStopCondition list


class MACPRunner:
    """Исполнитель протокола MACP для RoleGraph с sync/async и адаптивным режимом.

    Supports three execution modes:
    - Batch: run_round() / arun_round() - returns complete result
    - Streaming: stream() / astream() - yields events during execution

    Multi-model support (мультимодельность):
        Каждый агент может использовать свой LLM. Три способа задать модели:

        1. Единый llm_caller — для всех агентов (как раньше)
        2. llm_callers: dict[agent_id, Callable] — разные модели для разных агентов
        3. llm_factory: LLMCallerFactory — динамическое создание callers

        Приоритет: llm_callers[agent_id] > factory > default llm_caller

    Streaming Example:
        runner = MACPRunner(llm_caller=my_llm)

        # Sync streaming
        for event in runner.stream(graph):
            if event.event_type == StreamEventType.AGENT_OUTPUT:
                print(f"{event.agent_id}: {event.content}")

        # Async streaming
        async for event in runner.astream(graph):
            print(event)

    Multi-model Example:
        # Способ 1: Разные callers для разных агентов
        runner = MACPRunner(
            llm_caller=default_caller,
            llm_callers={
                "solver": gpt4_caller,
                "reviewer": claude_caller,
                "analyzer": local_llama_caller,
            }
        )

        # Способ 2: Фабрика (автоматически создаёт callers из LLM конфигурации агентов)
        factory = LLMCallerFactory.create_openai_factory(default_api_key="...")
        runner = MACPRunner(llm_factory=factory)

        # Агенты с llm_config автоматически получат свои callers:
        builder.add_agent("solver", llm_backbone="gpt-4", temperature=0.7)
        builder.add_agent("analyzer", llm_backbone="gpt-4o-mini", temperature=0.0)
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str] | None = None,
        async_llm_caller: Callable[[str], Awaitable[str]] | None = None,
        streaming_llm_caller: Callable[[str], Iterator[str]] | None = None,
        async_streaming_llm_caller: Callable[[str], AsyncIterator[str]] | None = None,
        token_counter: Callable[[str], int] | None = None,
        config: RunnerConfig | None = None,
        timeout: int = 60,
        memory_pool: SharedMemoryPool | None = None,
        # Multi-model support
        llm_callers: dict[str, Callable[[str], str]] | None = None,
        async_llm_callers: dict[str, Callable[[str], Awaitable[str]]] | None = None,
        llm_factory: LLMCallerFactory | None = None,
    ):
        """Создать раннер MACP с поддержкой мультимодельности.

        Args:
            llm_caller: Синхронный вызов LLM по умолчанию (возвращает полный ответ).
            async_llm_caller: Асинхронный вызов LLM по умолчанию (возвращает полный ответ).
            streaming_llm_caller: Синхронный streaming LLM (yield токены).
            async_streaming_llm_caller: Асинхронный streaming LLM (async yield токены).
            token_counter: Функция оценки токенов в тексте.
            config: Конфигурация раннера (иначе создаётся с указанным timeout).
            timeout: Таймаут по умолчанию (сек), если в config не указан.
            memory_pool: Внешний SharedMemoryPool (если None — создаётся автоматически).

            # Multi-model support (мультимодельность):
            llm_callers: Dict agent_id -> sync caller. Имеет наивысший приоритет.
            async_llm_callers: Dict agent_id -> async caller.
            llm_factory: Фабрика для создания callers на основе LLM конфигурации агентов.
                        Используется если для агента нет явного caller в llm_callers.

        Example:
            # Мультимодельность через словарь callers
            runner = MACPRunner(
                llm_caller=default_gpt4_caller,
                llm_callers={
                    "analyzer": create_openai_caller(model="gpt-4o-mini"),
                    "expert": create_openai_caller(model="gpt-4-turbo"),
                }
            )

            # Мультимодельность через фабрику
            factory = LLMCallerFactory.create_openai_factory()
            runner = MACPRunner(llm_factory=factory)
        """
        self.llm_caller = llm_caller
        self.async_llm_caller = async_llm_caller
        self.streaming_llm_caller = streaming_llm_caller
        self.async_streaming_llm_caller = async_streaming_llm_caller
        self.token_counter = token_counter or self._default_token_counter
        self.config = config or RunnerConfig(timeout=float(timeout))

        # Multi-model support
        self.llm_callers = llm_callers or {}
        self.async_llm_callers = async_llm_callers or {}
        self.llm_factory = llm_factory

        self._scheduler = (
            AdaptiveScheduler(
                policy=self.config.routing_policy,
                pruning_config=self.config.pruning_config,
            )
            if self.config.adaptive
            else None
        )

        self._callback_manager: CallbackManager | None = None
        self._budget_tracker: BudgetTracker | None = None
        self._metrics: ExecutionMetrics | None = None

        # Memory integration
        self._memory_pool: SharedMemoryPool | None = memory_pool
        self._agent_memories: dict[str, AgentMemory] = {}

    def _init_run(
        self,
        graph_name: str | None = None,
        num_agents: int = 0,
        query: str = "",
        execution_order: list[str] | None = None,
        callbacks: list[Handler] | None = None,
    ) -> uuid.UUID:
        """Initialize callbacks, budgets and metrics before running. Returns run_id."""
        # Merge config callbacks with per-run callbacks and context callbacks
        all_callbacks = list(self.config.callbacks)
        if callbacks:
            all_callbacks.extend(callbacks)

        # Check for context callback manager
        context_manager = get_callback_manager()
        if context_manager:
            all_callbacks.extend(context_manager.handlers)

        self._callback_manager = CallbackManager.configure(handlers=all_callbacks)

        if self.config.budget_config:
            self._budget_tracker = BudgetTracker(self.config.budget_config)
            self._budget_tracker.start()
        else:
            self._budget_tracker = None

        self._metrics = ExecutionMetrics(
            start_time=datetime.now(),
            total_agents=num_agents,
        )

        run_id = self._callback_manager.on_run_start(
            query=query,
            num_agents=num_agents,
            execution_order=execution_order or [],
        )
        return run_id

    def _init_memory(self, agent_ids: list[str]) -> None:
        """Инициализировать память для агентов перед запуском."""
        if not self.config.enable_memory:
            return

        if self._memory_pool is None:
            self._memory_pool = SharedMemoryPool()

        mem_config = self.config.memory_config or MemoryConfig()

        for agent_id in agent_ids:
            if agent_id not in self._agent_memories:
                memory = AgentMemory(agent_id, mem_config)
                self._agent_memories[agent_id] = memory
                self._memory_pool.register(memory)

    def _get_memory_context(self, agent_id: str) -> list[dict[str, Any]]:
        """Получить последние записи из памяти агента для контекста."""
        if not self.config.enable_memory or agent_id not in self._agent_memories:
            return []

        memory = self._agent_memories[agent_id]
        return memory.get_messages(limit=self.config.memory_context_limit)

    def _save_to_memory(
        self,
        agent_id: str,
        response: str,
        incoming_ids: list[str] | None = None,
    ) -> None:
        """Сохранить ответ агента в его память и расшарить с соседями."""
        if not self.config.enable_memory or agent_id not in self._agent_memories:
            return

        memory = self._agent_memories[agent_id]
        entry = memory.add_message(role="assistant", content=response)

        # Шарим с входящими агентами (соседями по графу)
        if self._memory_pool and incoming_ids:
            self._memory_pool.share(agent_id, entry, to_agents=incoming_ids)

    def get_agent_memory(self, agent_id: str) -> AgentMemory | None:
        """Получить память агента по id (для внешнего доступа)."""
        return self._agent_memories.get(agent_id)

    @property
    def memory_pool(self) -> SharedMemoryPool | None:
        """Доступ к SharedMemoryPool."""
        return self._memory_pool

    # =========================================================================
    # DYNAMIC TOPOLOGY METHODS
    # =========================================================================

    def _check_early_stop(
        self,
        agent_id: str,
        response: str | None,
        messages: dict[str, str],
        execution_order: list[str],
        remaining_agents: list[str],
        query: str,
        total_tokens: int,
    ) -> tuple[bool, str]:
        """Проверить условия ранней остановки.

        Returns:
            (should_stop, reason)
        """
        if not self.config.early_stop_conditions:
            return False, ""

        ctx = StepContext(
            agent_id=agent_id,
            response=response,
            messages=messages,
            execution_order=execution_order,
            remaining_agents=remaining_agents,
            query=query,
            total_tokens=total_tokens,
        )

        for condition in self.config.early_stop_conditions:
            if isinstance(condition, EarlyStopCondition):
                should_stop, reason = condition.should_stop(ctx)
                if should_stop:
                    return True, reason

        return False, ""

    def _apply_topology_hooks(
        self,
        agent_id: str,
        response: str | None,
        step_result: StepResult | None,
        messages: dict[str, str],
        execution_order: list[str],
        remaining_agents: list[str],
        query: str,
        total_tokens: int,
        role_graph: Any,
    ) -> TopologyAction | None:
        """Применить sync topology hooks и собрать действия.

        Returns:
            Объединённое TopologyAction или None.
        """
        if not self.config.enable_dynamic_topology or not self.config.topology_hooks:
            return None

        ctx = StepContext(
            agent_id=agent_id,
            response=response,
            step_result=step_result,
            messages=messages,
            execution_order=execution_order,
            remaining_agents=remaining_agents,
            query=query,
            total_tokens=total_tokens,
        )

        combined_action = TopologyAction()

        for hook in self.config.topology_hooks:
            try:
                action = hook(ctx, role_graph)
                if action is not None:
                    combined_action = self._merge_topology_actions(combined_action, action)
            except Exception:
                pass  # Игнорируем ошибки в hooks

        # Проверить, есть ли хоть какие-то действия
        if (
            combined_action.early_stop
            or combined_action.add_edges
            or combined_action.remove_edges
            or combined_action.skip_agents
            or combined_action.force_agents
            or combined_action.new_end_agent
            or combined_action.trigger_replan
        ):
            return combined_action

        return None

    async def _apply_async_topology_hooks(
        self,
        agent_id: str,
        response: str | None,
        step_result: StepResult | None,
        messages: dict[str, str],
        execution_order: list[str],
        remaining_agents: list[str],
        query: str,
        total_tokens: int,
        role_graph: Any,
    ) -> TopologyAction | None:
        """Применить async topology hooks и собрать действия."""
        if not self.config.enable_dynamic_topology or not self.config.async_topology_hooks:
            return None

        ctx = StepContext(
            agent_id=agent_id,
            response=response,
            step_result=step_result,
            messages=messages,
            execution_order=execution_order,
            remaining_agents=remaining_agents,
            query=query,
            total_tokens=total_tokens,
        )

        combined_action = TopologyAction()

        for hook in self.config.async_topology_hooks:
            try:
                action = await hook(ctx, role_graph)
                if action is not None:
                    combined_action = self._merge_topology_actions(combined_action, action)
            except Exception:
                pass

        if (
            combined_action.early_stop
            or combined_action.add_edges
            or combined_action.remove_edges
            or combined_action.skip_agents
            or combined_action.force_agents
            or combined_action.new_end_agent
            or combined_action.trigger_replan
        ):
            return combined_action

        return None

    def _merge_topology_actions(
        self,
        base: TopologyAction,
        new: TopologyAction,
    ) -> TopologyAction:
        """Объединить два TopologyAction."""
        return TopologyAction(
            early_stop=base.early_stop or new.early_stop,
            early_stop_reason=new.early_stop_reason or base.early_stop_reason,
            add_edges=base.add_edges + new.add_edges,
            remove_edges=base.remove_edges + new.remove_edges,
            skip_agents=list(set(base.skip_agents + new.skip_agents)),
            force_agents=list(set(base.force_agents + new.force_agents)),
            new_end_agent=new.new_end_agent or base.new_end_agent,
            trigger_replan=base.trigger_replan or new.trigger_replan,
        )

    def _apply_graph_modifications(
        self,
        role_graph: Any,
        action: TopologyAction,
    ) -> int:
        """Применить модификации к графу и вернуть количество изменений."""
        modifications = 0

        # Удалить рёбра
        for src, tgt in action.remove_edges:
            if role_graph.remove_edge(src, tgt):
                modifications += 1

        # Добавить рёбра
        for src, tgt, weight in action.add_edges:
            if role_graph.add_edge(src, tgt, weight):
                modifications += 1

        return modifications

    def _finalize_run(
        self,
        run_id: uuid.UUID,
        success: bool,
        executed_agents: int,
        final_answer: str = "",
        error: BaseException | None = None,
        executed_agent_ids: list[str] | None = None,
    ) -> None:
        """Finalize metrics and notify callbacks after execution."""
        if self._metrics:
            self._metrics.end_time = datetime.now()
            self._metrics.executed_agents = executed_agents

        if self._callback_manager:
            self._callback_manager.on_run_end(
                run_id=run_id,
                output=final_answer,
                success=success,
                error=error,
                total_tokens=self._metrics.total_tokens if self._metrics else 0,
                total_time_ms=self._metrics.duration_seconds * 1000 if self._metrics else 0,
                executed_agents=executed_agent_ids or [],
            )

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Простая оценка токенов: 4/3 от количества слов."""
        return len(text.split()) * 4 // 3

    def _get_caller_for_agent(
        self,
        agent_id: str,
        agent: Any,
    ) -> Callable[[str], str] | None:
        """Получить sync LLM caller для конкретного агента.

        Приоритет:
        1. llm_callers[agent_id] — явно заданный caller для агента
        2. llm_factory.get_caller(agent.llm_config) — из фабрики по конфигурации
        3. self.llm_caller — caller по умолчанию

        Returns:
            LLM caller или None если ни один не доступен.
        """
        # 1. Check explicit per-agent caller
        if agent_id in self.llm_callers:
            return self.llm_callers[agent_id]

        # 2. Try factory if agent has LLM config
        if self.llm_factory and hasattr(agent, "get_llm_config"):
            llm_config = agent.get_llm_config()
            if llm_config and llm_config.is_configured():
                caller = self.llm_factory.get_caller(llm_config, agent_id)
                if caller:
                    return caller
        elif self.llm_factory and hasattr(agent, "llm_config") and agent.llm_config:
            caller = self.llm_factory.get_caller(agent.llm_config, agent_id)
            if caller:
                return caller

        # 3. Fallback to default caller
        return self.llm_caller

    def _get_async_caller_for_agent(
        self,
        agent_id: str,
        agent: Any,
    ) -> Callable[[str], Awaitable[str]] | None:
        """Получить async LLM caller для конкретного агента.

        Приоритет:
        1. async_llm_callers[agent_id] — явно заданный async caller
        2. llm_factory.get_async_caller(agent.llm_config) — из фабрики
        3. self.async_llm_caller — async caller по умолчанию
        """
        # 1. Check explicit per-agent async caller
        if agent_id in self.async_llm_callers:
            return self.async_llm_callers[agent_id]

        # 2. Try factory if agent has LLM config
        if self.llm_factory and hasattr(agent, "get_llm_config"):
            llm_config = agent.get_llm_config()
            if llm_config and llm_config.is_configured():
                caller = self.llm_factory.get_async_caller(llm_config, agent_id)
                if caller:
                    return caller
        elif self.llm_factory and hasattr(agent, "llm_config") and agent.llm_config:
            caller = self.llm_factory.get_async_caller(agent.llm_config, agent_id)
            if caller:
                return caller

        # 3. Fallback to default async caller
        return self.async_llm_caller

    def _has_any_caller(self) -> bool:
        """Проверить, есть ли хотя бы один LLM caller."""
        return bool(
            self.llm_caller
            or self.llm_callers
            or (
                self.llm_factory
                and (self.llm_factory.default_caller or self.llm_factory.caller_builder)
            )
        )

    def _has_any_async_caller(self) -> bool:
        """Проверить, есть ли хотя бы один async LLM caller."""
        return bool(
            self.async_llm_caller
            or self.async_llm_callers
            or (
                self.llm_factory
                and (self.llm_factory.default_async_caller or self.llm_factory.async_caller_builder)
            )
        )

    def run_round(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        start_agent_id: str | None = None,
        update_states: bool | None = None,
        filter_unreachable: bool = False,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Run a synchronous round (simple or adaptive strategy).

        Args:
            role_graph: Role graph to execute.
            final_agent_id: ID of final agent (overrides role_graph.end_node).
            start_agent_id: ID of start agent (overrides role_graph.start_node).
            update_states: Whether to update agent states.
            filter_unreachable: Exclude isolated nodes from execution.
            callbacks: Per-run callback handlers (merged with config.callbacks).

        Returns:
            MACPResult with execution results.
        """
        if not self._has_any_caller():
            raise ValueError(
                "llm_caller, llm_callers, or llm_factory is required for synchronous execution"
            )

        # Get start/end from params or graph
        effective_start = start_agent_id or getattr(role_graph, "start_node", None)
        effective_end = final_agent_id or getattr(role_graph, "end_node", None)

        if self.config.adaptive:
            return self._run_adaptive(
                role_graph,
                effective_end,
                effective_start,
                update_states,
                filter_unreachable,
                callbacks,
            )
        else:
            return self._run_simple(
                role_graph,
                effective_end,
                effective_start,
                update_states,
                filter_unreachable,
                callbacks,
            )

    async def arun_round(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        start_agent_id: str | None = None,
        update_states: bool | None = None,
        filter_unreachable: bool = False,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Run an async round (simple or adaptive strategy).

        Args:
            role_graph: Role graph to execute.
            final_agent_id: ID of final agent (overrides role_graph.end_node).
            start_agent_id: ID of start agent (overrides role_graph.start_node).
            update_states: Whether to update agent states.
            filter_unreachable: Exclude isolated nodes from execution.
            callbacks: Per-run callback handlers (merged with config.callbacks).

        Returns:
            MACPResult with execution results.
        """
        if not self._has_any_async_caller():
            raise ValueError(
                "async_llm_caller, async_llm_callers, or llm_factory is required for async execution"
            )

        # Get start/end from params or graph
        effective_start = start_agent_id or getattr(role_graph, "start_node", None)
        effective_end = final_agent_id or getattr(role_graph, "end_node", None)

        if self.config.adaptive:
            return await self._arun_adaptive(
                role_graph,
                effective_end,
                effective_start,
                update_states,
                filter_unreachable,
                callbacks,
            )
        else:
            return await self._arun_simple(
                role_graph,
                effective_end,
                effective_start,
                update_states,
                filter_unreachable,
                callbacks,
            )

    def _run_simple(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        start_agent_id: str | None,
        update_states: bool | None,
        filter_unreachable: bool = True,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Sequential execution in topological order without adaptation.

        Supports multi-model: each agent uses its own LLM caller.
        Supports filtering of isolated nodes to save tokens.
        """
        if not self._has_any_caller():
            raise ValueError(
                "llm_caller, llm_callers, or llm_factory is required for synchronous execution"
            )

        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            return MACPResult(
                messages={},
                final_answer="",
                final_agent_id="",
                execution_order=[],
            )

        # Filter isolated nodes
        excluded_agents: list[str] = []
        effective_agent_ids = agent_ids
        effective_a = a_agents

        if filter_unreachable and (start_agent_id is not None or final_agent_id is not None):
            relevant, excluded_agents = filter_reachable_agents(
                a_agents, agent_ids, start_agent_id, final_agent_id, threshold=0.5
            )
            if relevant and len(relevant) < len(agent_ids):
                indices = [agent_ids.index(aid) for aid in relevant]
                indices_t = torch.tensor(indices, dtype=torch.long)
                effective_a = a_agents[indices_t][:, indices_t]
                effective_agent_ids = relevant

        exec_order = build_execution_order(
            effective_a, effective_agent_ids, role_graph.role_sequence
        )

        # Initialize memory
        self._init_memory(effective_agent_ids)

        # Initialize callbacks
        query = role_graph.query or ""
        run_id = self._init_run(
            graph_name=getattr(role_graph, "name", None),
            num_agents=len(effective_agent_ids),
            query=query,
            execution_order=exec_order,
            callbacks=callbacks,
        )

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        task_connected = self._get_task_connected_agents(role_graph)

        messages: dict[str, str] = {}
        total_tokens = 0
        actual_exec_order: list[str] = []
        early_stopped = False
        early_stop_reason: str | None = None
        topology_modifications = 0
        skipped_by_hooks: set[str] = set()
        run_error: BaseException | None = None

        # Get disabled nodes from graph
        disabled_nodes: set[str] = getattr(role_graph, "disabled_nodes", set())

        try:
            for step_idx, agent_id in enumerate(exec_order):
                # Check if agent was skipped by hooks
                if agent_id in skipped_by_hooks:
                    continue

                # Check if node is disabled
                if agent_id in disabled_nodes:
                    if agent_id not in excluded_agents:
                        excluded_agents.append(agent_id)
                    continue

                agent = agent_lookup.get(agent_id)
                if agent is None:
                    continue

                incoming_ids = get_incoming_agents(agent_id, effective_a, effective_agent_ids)
                incoming_messages = {aid: messages[aid] for aid in incoming_ids if aid in messages}

                include_query = self._should_include_query(agent_id, task_connected)
                memory_context = self._get_memory_context(agent_id)
                prompt = self._build_prompt(
                    agent, query, incoming_messages, agent_names, memory_context, include_query
                )

                # Notify callbacks of agent start
                if self._callback_manager:
                    self._callback_manager.on_agent_start(
                        run_id=run_id,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, agent_id),
                        step_index=step_idx,
                        prompt=prompt[: self.config.prompt_preview_length]
                        if hasattr(self.config, "prompt_preview_length")
                        else prompt[:100],
                        predecessors=incoming_ids,
                    )

                agent_start_time = time.time()
                try:
                    # Get caller for this specific agent (multi-model support)
                    caller = self._get_caller_for_agent(agent_id, agent)
                    if caller is None:
                        error_msg = f"No LLM caller available for agent {agent_id}"
                        messages[agent_id] = f"[Error: {error_msg}]"
                        actual_exec_order.append(agent_id)
                        if self._callback_manager:
                            self._callback_manager.on_agent_error(
                                run_id=run_id,
                                error=ValueError(error_msg),
                                agent_id=agent_id,
                                error_type="NoCallerError",
                            )
                        continue

                    response = caller(prompt)
                    agent_tokens = self.token_counter(prompt) + self.token_counter(response)
                    agent_duration_ms = (time.time() - agent_start_time) * 1000

                    messages[agent_id] = response
                    total_tokens += agent_tokens
                    self._save_to_memory(agent_id, response, incoming_ids)
                    actual_exec_order.append(agent_id)

                    # Notify callbacks of agent end
                    if self._callback_manager:
                        is_final = agent_id == final_agent_id or (
                            final_agent_id is None and agent_id == exec_order[-1]
                        )
                        self._callback_manager.on_agent_end(
                            run_id=run_id,
                            agent_id=agent_id,
                            output=response,
                            agent_name=agent_names.get(agent_id, agent_id),
                            step_index=step_idx,
                            tokens_used=agent_tokens,
                            duration_ms=agent_duration_ms,
                            is_final=is_final,
                        )

                except Exception as e:
                    messages[agent_id] = f"[Error: {e}]"
                    actual_exec_order.append(agent_id)
                    if self._callback_manager:
                        self._callback_manager.on_agent_error(
                            run_id=run_id,
                            error=e,
                            agent_id=agent_id,
                            error_type=type(e).__name__,
                        )

                # Check early stopping
                remaining = [
                    a for a in exec_order if a not in messages and a not in skipped_by_hooks
                ]
                should_stop, reason = self._check_early_stop(
                    agent_id,
                    messages.get(agent_id),
                    messages,
                    actual_exec_order,
                    remaining,
                    query,
                    total_tokens,
                )
                if should_stop:
                    early_stopped = True
                    early_stop_reason = reason
                    break

                # Apply topology hooks
                if self.config.enable_dynamic_topology:
                    action = self._apply_topology_hooks(
                        agent_id,
                        messages.get(agent_id),
                        None,
                        messages,
                        actual_exec_order,
                        remaining,
                        query,
                        total_tokens,
                        role_graph,
                    )
                    if action is not None:
                        if action.early_stop:
                            early_stopped = True
                            early_stop_reason = action.early_stop_reason
                            break
                        if action.skip_agents:
                            skipped_by_hooks.update(action.skip_agents)
                        topology_modifications += self._apply_graph_modifications(
                            role_graph, action
                        )

        except Exception as e:
            run_error = e

        final_id = self._determine_final_agent(final_agent_id, actual_exec_order, messages)
        final_answer = messages.get(final_id, "")

        # Finalize callbacks
        self._finalize_run(
            run_id=run_id,
            success=run_error is None,
            executed_agents=len(actual_exec_order),
            final_answer=final_answer,
            error=run_error,
            executed_agent_ids=actual_exec_order,
        )

        if run_error:
            raise run_error

        do_update = update_states if update_states is not None else self.config.update_states
        agent_states = self._build_agent_states(messages, agent_lookup) if do_update else None

        return MACPResult(
            messages=messages,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            execution_order=actual_exec_order,
            agent_states=agent_states,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            pruned_agents=excluded_agents if excluded_agents else None,
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
            topology_modifications=topology_modifications,
        )

    async def _arun_simple(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        start_agent_id: str | None,
        update_states: bool | None,
        filter_unreachable: bool = True,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Async sequential execution without adaptation.

        Supports multi-model: each agent uses its own LLM caller.
        Supports filtering of isolated nodes to save tokens.
        """
        if not self._has_any_async_caller():
            raise ValueError(
                "async_llm_caller, async_llm_callers, or llm_factory is required for async execution"
            )

        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            return MACPResult(
                messages={},
                final_answer="",
                final_agent_id="",
                execution_order=[],
            )

        # Filter isolated nodes
        excluded_agents: list[str] = []
        effective_agent_ids = agent_ids
        effective_a = a_agents

        if filter_unreachable and (start_agent_id is not None or final_agent_id is not None):
            relevant, excluded_agents = filter_reachable_agents(
                a_agents, agent_ids, start_agent_id, final_agent_id, threshold=0.5
            )
            if relevant and len(relevant) < len(agent_ids):
                indices = [agent_ids.index(aid) for aid in relevant]
                indices_t = torch.tensor(indices, dtype=torch.long)
                effective_a = a_agents[indices_t][:, indices_t]
                effective_agent_ids = relevant

        exec_order = build_execution_order(
            effective_a, effective_agent_ids, role_graph.role_sequence
        )

        # Initialize memory
        self._init_memory(effective_agent_ids)

        # Initialize callbacks
        query = role_graph.query or ""
        run_id = self._init_run(
            graph_name=getattr(role_graph, "name", None),
            num_agents=len(effective_agent_ids),
            query=query,
            execution_order=exec_order,
            callbacks=callbacks,
        )

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        task_connected = self._get_task_connected_agents(role_graph)

        messages: dict[str, str] = {}
        total_tokens = 0
        actual_exec_order: list[str] = []
        early_stopped = False
        early_stop_reason: str | None = None
        topology_modifications = 0
        skipped_by_hooks: set[str] = set()
        run_error: BaseException | None = None

        # Get disabled nodes from graph
        disabled_nodes: set[str] = getattr(role_graph, "disabled_nodes", set())

        try:
            for step_idx, agent_id in enumerate(exec_order):
                # Check if agent was skipped by hooks
                if agent_id in skipped_by_hooks:
                    continue

                # Check if node is disabled
                if agent_id in disabled_nodes:
                    if agent_id not in excluded_agents:
                        excluded_agents.append(agent_id)
                    continue

                agent = agent_lookup.get(agent_id)
                if agent is None:
                    continue

                incoming_ids = get_incoming_agents(agent_id, effective_a, effective_agent_ids)
                incoming_messages = {aid: messages[aid] for aid in incoming_ids if aid in messages}

                include_query = self._should_include_query(agent_id, task_connected)
                memory_context = self._get_memory_context(agent_id)
                prompt = self._build_prompt(
                    agent, query, incoming_messages, agent_names, memory_context, include_query
                )

                # Notify callbacks of agent start
                if self._callback_manager:
                    self._callback_manager.on_agent_start(
                        run_id=run_id,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, agent_id),
                        step_index=step_idx,
                        prompt=prompt[:100],
                        predecessors=incoming_ids,
                    )

                agent_start_time = time.time()
                try:
                    # Get async caller for this specific agent (multi-model support)
                    async_caller = self._get_async_caller_for_agent(agent_id, agent)
                    if async_caller is None:
                        error_msg = f"No async LLM caller available for agent {agent_id}"
                        messages[agent_id] = f"[Error: {error_msg}]"
                        actual_exec_order.append(agent_id)
                        if self._callback_manager:
                            self._callback_manager.on_agent_error(
                                run_id=run_id,
                                error=ValueError(error_msg),
                                agent_id=agent_id,
                                error_type="NoCallerError",
                            )
                        continue

                    response = await asyncio.wait_for(
                        async_caller(prompt),
                        timeout=self.config.timeout,
                    )
                    agent_tokens = self.token_counter(prompt) + self.token_counter(response)
                    agent_duration_ms = (time.time() - agent_start_time) * 1000

                    messages[agent_id] = response
                    total_tokens += agent_tokens
                    self._save_to_memory(agent_id, response, incoming_ids)
                    actual_exec_order.append(agent_id)

                    # Notify callbacks of agent end
                    if self._callback_manager:
                        is_final = agent_id == final_agent_id or (
                            final_agent_id is None and agent_id == exec_order[-1]
                        )
                        self._callback_manager.on_agent_end(
                            run_id=run_id,
                            agent_id=agent_id,
                            output=response,
                            agent_name=agent_names.get(agent_id, agent_id),
                            step_index=step_idx,
                            tokens_used=agent_tokens,
                            duration_ms=agent_duration_ms,
                            is_final=is_final,
                        )

                except Exception as e:
                    messages[agent_id] = f"[Error: {e}]"
                    actual_exec_order.append(agent_id)
                    if self._callback_manager:
                        self._callback_manager.on_agent_error(
                            run_id=run_id,
                            error=e,
                            agent_id=agent_id,
                            error_type=type(e).__name__,
                        )

                # Check early stopping
                remaining = [
                    a for a in exec_order if a not in messages and a not in skipped_by_hooks
                ]
                should_stop, reason = self._check_early_stop(
                    agent_id,
                    messages.get(agent_id),
                    messages,
                    actual_exec_order,
                    remaining,
                    query,
                    total_tokens,
                )
                if should_stop:
                    early_stopped = True
                    early_stop_reason = reason
                    break

                # Apply async topology hooks
                if self.config.enable_dynamic_topology:
                    action = await self._apply_async_topology_hooks(
                        agent_id,
                        messages.get(agent_id),
                        None,
                        messages,
                        actual_exec_order,
                        remaining,
                        query,
                        total_tokens,
                        role_graph,
                    )
                    if action is not None:
                        if action.early_stop:
                            early_stopped = True
                            early_stop_reason = action.early_stop_reason
                            break
                        if action.skip_agents:
                            skipped_by_hooks.update(action.skip_agents)
                        topology_modifications += self._apply_graph_modifications(
                            role_graph, action
                        )

        except Exception as e:
            run_error = e

        final_id = self._determine_final_agent(final_agent_id, actual_exec_order, messages)
        final_answer = messages.get(final_id, "")

        # Finalize callbacks
        self._finalize_run(
            run_id=run_id,
            success=run_error is None,
            executed_agents=len(actual_exec_order),
            final_answer=final_answer,
            error=run_error,
            executed_agent_ids=actual_exec_order,
        )

        if run_error:
            raise run_error

        do_update = update_states if update_states is not None else self.config.update_states
        agent_states = self._build_agent_states(messages, agent_lookup) if do_update else None

        return MACPResult(
            messages=messages,
            final_answer=final_answer,
            final_agent_id=final_id,
            execution_order=actual_exec_order,
            agent_states=agent_states,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            pruned_agents=excluded_agents if excluded_agents else None,
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
            topology_modifications=topology_modifications,
        )

    def _run_adaptive(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        start_agent_id: str | None,
        update_states: bool | None,
        filter_unreachable: bool = True,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Adaptive sync execution with replanning and fallback.

        Supports multi-model: each agent uses its own LLM caller.
        Supports filtering of isolated nodes to save tokens.
        """
        # Note: callbacks parameter accepted for API compatibility,
        # full callback integration pending
        if not self._has_any_caller():
            raise ValueError(
                "llm_caller, llm_callers, or llm_factory is required for synchronous execution"
            )
        if self._scheduler is None:
            raise ValueError("Scheduler not initialized for adaptive mode")

        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            return MACPResult(
                messages={},
                final_answer="",
                final_agent_id="",
                execution_order=[],
            )

        # Инициализация памяти
        self._init_memory(agent_ids)

        p_matrix = self._extract_p_matrix(role_graph, task_idx)
        query = role_graph.query or ""

        # Получить условия из графа для conditional routing
        edge_conditions = self._get_edge_conditions(role_graph)

        # Начальный контекст для условий
        condition_ctx = ConditionContext(
            source_agent="",
            target_agent="",
            messages={},
            step_results={},
            query=query,
        )

        plan = self._scheduler.build_plan(
            a_agents,
            agent_ids,
            p_matrix,
            start_agent=start_agent_id,
            end_agent=final_agent_id,
            edge_conditions=edge_conditions,
            condition_context=condition_ctx,
            filter_unreachable=filter_unreachable,
        )

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)

        messages: dict[str, str] = {}
        step_results: dict[str, StepResult] = {}
        execution_order: list[str] = []
        fallback_attempts: dict[str, int] = {}
        replanning_count = 0
        fallback_count = 0
        pruned_agents: list[str] = []
        errors: list[ExecutionError] = []

        while not plan.is_complete:
            step = plan.get_current_step()
            if step is None:
                break

            should_prune, reason = self._scheduler.should_prune(
                step, plan, step_results.get(execution_order[-1]) if execution_order else None
            )

            if should_prune:
                plan.mark_skipped(step.agent_id)
                pruned_agents.append(step.agent_id)
                errors.append(
                    ExecutionError(
                        message=f"Pruned: {reason}",
                        agent_id=step.agent_id,
                        recoverable=False,
                    )
                )
                continue

            result = self._execute_step(step, messages, agent_lookup, agent_names, query)

            step_results[step.agent_id] = result
            execution_order.append(step.agent_id)

            if result.success:
                messages[step.agent_id] = result.response or ""
                plan.mark_completed(step.agent_id, result.tokens_used)
                self._save_to_memory(step.agent_id, result.response or "", step.predecessors)
            else:
                plan.mark_failed(step.agent_id)
                errors.append(
                    ExecutionError(
                        message=result.error or "Unknown error",
                        agent_id=step.agent_id,
                        recoverable=True,
                    )
                )

                attempts = fallback_attempts.get(step.agent_id, 0)
                if self._scheduler.should_use_fallback(step, result, attempts):
                    for fb_agent in step.fallback_agents:
                        if fb_agent not in plan.completed and fb_agent not in plan.failed:
                            plan.insert_fallback(fb_agent, plan.current_index - 1)
                            fallback_count += 1
                            break
                    fallback_attempts[step.agent_id] = attempts + 1

            if self.config.enable_replanning:
                should_replan = not self.config.replan_on_error_only or not result.success
                if should_replan and not plan.is_complete:
                    plan = self._scheduler.replan(plan, a_agents, agent_ids, p_matrix, step_results)
                    replanning_count += 1

        final_id = self._determine_final_agent(final_agent_id, execution_order, messages)

        do_update = update_states if update_states is not None else self.config.update_states
        agent_states = self._build_agent_states(messages, agent_lookup) if do_update else None

        return MACPResult(
            messages=messages,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            execution_order=execution_order,
            agent_states=agent_states,
            step_results=step_results,
            total_tokens=plan.tokens_used,
            total_time=time.time() - start_time,
            replanning_count=replanning_count,
            fallback_count=fallback_count,
            pruned_agents=pruned_agents,
            errors=errors if errors else None,
        )

    async def _arun_adaptive(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        start_agent_id: str | None,
        update_states: bool | None,
        filter_unreachable: bool = True,
        callbacks: list[Handler] | None = None,
    ) -> MACPResult:
        """Adaptive async execution with parallelism and replanning.

        Supports multi-model: each agent uses its own LLM caller.
        Supports filtering of isolated nodes to save tokens.
        """
        # Note: callbacks parameter accepted for API compatibility,
        # full callback integration pending
        if not self._has_any_async_caller():
            raise ValueError(
                "async_llm_caller, async_llm_callers, or llm_factory is required for async execution"
            )
        if self._scheduler is None:
            raise ValueError("Scheduler not initialized for adaptive mode")

        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            return MACPResult(
                messages={},
                final_answer="",
                final_agent_id="",
                execution_order=[],
            )

        # Инициализация памяти
        self._init_memory(agent_ids)

        p_matrix = self._extract_p_matrix(role_graph, task_idx)
        query = role_graph.query or ""

        # Получить условия из графа для conditional routing
        edge_conditions = self._get_edge_conditions(role_graph)

        # Контекст для условий
        condition_ctx = ConditionContext(
            source_agent="",
            target_agent="",
            messages={},
            step_results={},
            query=query,
        )

        plan = self._scheduler.build_plan(
            a_agents,
            agent_ids,
            p_matrix,
            start_agent=start_agent_id,
            end_agent=final_agent_id,
            edge_conditions=edge_conditions,
            condition_context=condition_ctx,
            filter_unreachable=filter_unreachable,
        )

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)

        messages: dict[str, str] = {}
        step_results: dict[str, StepResult] = {}
        execution_order: list[str] = []
        fallback_attempts: dict[str, int] = {}
        replanning_count = 0
        fallback_count = 0
        pruned_agents: list[str] = []
        errors: list[ExecutionError] = []

        while not plan.is_complete:
            parallel_group = self._get_parallel_group(plan, messages.keys())

            if not parallel_group:
                break

            valid_steps = []
            for step in parallel_group:
                should_prune, reason = self._scheduler.should_prune(step, plan, None)
                if should_prune:
                    plan.mark_skipped(step.agent_id)
                    pruned_agents.append(step.agent_id)
                    errors.append(
                        ExecutionError(
                            message=f"Pruned: {reason}",
                            agent_id=step.agent_id,
                            recoverable=False,
                        )
                    )
                else:
                    valid_steps.append(step)

            if not valid_steps:
                continue

            if self.config.enable_parallel and len(valid_steps) > 1:
                results = await self._execute_parallel(
                    valid_steps, messages, agent_lookup, agent_names, query
                )
            else:
                results = []
                for step in valid_steps:
                    r = await self._execute_step_async(
                        step, messages, agent_lookup, agent_names, query
                    )
                    results.append((step, r))

            for step, result in results:
                step_results[step.agent_id] = result
                execution_order.append(step.agent_id)

                if result.success:
                    messages[step.agent_id] = result.response or ""
                    plan.mark_completed(step.agent_id, result.tokens_used)
                    self._save_to_memory(step.agent_id, result.response or "", step.predecessors)
                else:
                    plan.mark_failed(step.agent_id)
                    errors.append(
                        ExecutionError(
                            message=result.error or "Unknown error",
                            agent_id=step.agent_id,
                            recoverable=True,
                        )
                    )

                    attempts = fallback_attempts.get(step.agent_id, 0)
                    if self._scheduler.should_use_fallback(step, result, attempts):
                        for fb_agent in step.fallback_agents:
                            if fb_agent not in plan.completed and fb_agent not in plan.failed:
                                plan.insert_fallback(fb_agent, plan.current_index - 1)
                                fallback_count += 1
                                break
                        fallback_attempts[step.agent_id] = attempts + 1

            if self.config.enable_replanning:
                any_failed = any(not r.success for _, r in results)
                should_replan = not self.config.replan_on_error_only or any_failed
                if should_replan and not plan.is_complete:
                    plan = self._scheduler.replan(plan, a_agents, agent_ids, p_matrix, step_results)
                    replanning_count += 1

        final_id = self._determine_final_agent(final_agent_id, execution_order, messages)

        do_update = update_states if update_states is not None else self.config.update_states
        agent_states = self._build_agent_states(messages, agent_lookup) if do_update else None

        return MACPResult(
            messages=messages,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            execution_order=execution_order,
            agent_states=agent_states,
            step_results=step_results,
            total_tokens=plan.tokens_used,
            total_time=time.time() - start_time,
            replanning_count=replanning_count,
            fallback_count=fallback_count,
            pruned_agents=pruned_agents,
            errors=errors if errors else None,
        )

    def _get_task_index(self, role_graph: Any) -> int:
        """Получить rustworkx-индекс узла задачи или поднять ошибку."""
        if role_graph.task_node is None:
            raise ValueError("RoleGraph has no task_node set")

        task_idx = role_graph.get_node_index(role_graph.task_node)
        if task_idx is None:
            raise ValueError(f"Task node '{role_graph.task_node}' not found")
        return task_idx

    def _get_agent_ids(
        self,
        role_graph: Any,
        task_idx: int,
    ) -> tuple[list[str], dict[str, int]]:
        """Вернуть список agent_ids (без task) и map id->adjacency index."""
        agent_ids = []
        id_to_idx = {}

        adj_idx = 0
        for agent in role_graph.agents:
            graph_idx = role_graph.get_node_index(agent.identifier)
            if graph_idx == task_idx:
                continue

            agent_ids.append(agent.identifier)
            id_to_idx[agent.identifier] = adj_idx
            adj_idx += 1

        return agent_ids, id_to_idx

    def _extract_p_matrix(self, role_graph: Any, task_idx: int) -> torch.Tensor | None:
        """Вернуть вероятностную матрицу без строки/столбца задачи."""
        if role_graph.p_matrix is None:
            return None

        n_nodes = role_graph.p_matrix.shape[0]
        mask = torch.ones(n_nodes, dtype=torch.bool)
        mask[task_idx] = False
        return role_graph.p_matrix[mask][:, mask]

    def _get_edge_conditions(self, role_graph: Any) -> dict[tuple[str, str], Any]:
        """Получить все условия для рёбер из графа."""
        if hasattr(role_graph, "get_all_edge_conditions"):
            return role_graph.get_all_edge_conditions()
        # Fallback: проверить отдельные атрибуты
        conditions: dict[tuple[str, str], Any] = {}
        if hasattr(role_graph, "edge_condition_names"):
            conditions.update(role_graph.edge_condition_names)
        if hasattr(role_graph, "edge_conditions"):
            conditions.update(role_graph.edge_conditions)
        return conditions

    def _build_agent_names(self, role_graph: Any) -> dict[str, str]:
        """Отобразить id -> display_name/role для построения prompt."""
        return {
            a.identifier: a.display_name or getattr(a, "role", a.identifier)
            for a in role_graph.agents
        }

    def _get_task_connected_agents(self, role_graph: Any) -> set[str]:
        """Получить множество агентов, напрямую соединённых с task нодой."""
        if role_graph.task_node is None:
            return set()

        task_idx = role_graph.get_node_index(role_graph.task_node)
        if task_idx is None or role_graph.A_com is None:
            return set()

        connected = set()
        for agent in role_graph.agents:
            agent_idx = role_graph.get_node_index(agent.identifier)
            if agent_idx is not None and agent_idx != task_idx:
                if role_graph.A_com[task_idx, agent_idx] > 0:
                    connected.add(agent.identifier)
        return connected

    def _should_include_query(self, agent_id: str, task_connected: set[str]) -> bool:
        """Определить, включать ли query в промпт агента."""
        if self.config.broadcast_task_to_all:
            return True
        return agent_id in task_connected

    def _build_prompt(
        self,
        agent: Any,
        query: str,
        incoming_messages: dict[str, str],
        agent_names: dict[str, str],
        memory_context: list[dict[str, Any]] | None = None,
        include_query: bool = True,
    ) -> str:
        """Сформировать подсказку агенту с персоной/описанием, памятью и сообщениями.

        Args:
            include_query: Включать ли task query в промпт.
                          Контролируется через config.broadcast_task_to_all.
        """
        parts = []
        if hasattr(agent, "persona") and agent.persona:
            parts.append(f"You are {agent.persona}.")
        elif hasattr(agent, "role") and agent.role:
            parts.append(f"You are a {agent.role}.")

        if hasattr(agent, "description") and agent.description:
            parts.append(agent.description.strip())

        if hasattr(agent, "tools") and agent.tools:
            parts.append(f"Available tools: {', '.join(agent.tools)}")

        system_prompt = "\n\n".join(parts) if parts else "You are a helpful assistant."

        user_parts = []

        # Task query добавляется только если include_query=True
        if include_query and query:
            user_parts.append(f"Task: {query}")

        # Включить контекст из памяти
        if memory_context:
            user_parts.append("\nPrevious context:")
            for msg in memory_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                user_parts.append(f"[{role}]: {content}")

        if incoming_messages:
            user_parts.append("\nMessages from other agents:")
            for sender_id, message in incoming_messages.items():
                sender_name = agent_names.get(sender_id, sender_id)
                user_parts.append(f"\n[{sender_name}]:\n{message}")
        user_parts.append("\nProvide your response:")

        return f"{system_prompt}\n\n{''.join(user_parts)}"

    def _execute_step(
        self,
        step: Any,
        messages: dict[str, str],
        agent_lookup: dict[str, Any],
        agent_names: dict[str, str],
        query: str,
    ) -> StepResult:
        """Выполнить шаг синхронно с ретраями и подсчётом токенов.

        Поддерживает мультимодельность: использует caller для конкретного агента.
        """
        agent = agent_lookup.get(step.agent_id)
        if agent is None:
            return StepResult(
                agent_id=step.agent_id,
                success=False,
                error=f"Agent '{step.agent_id}' not found",
            )

        # Get caller for this specific agent (multi-model support)
        caller = self._get_caller_for_agent(step.agent_id, agent)
        if caller is None:
            return StepResult(
                agent_id=step.agent_id,
                success=False,
                error=f"No LLM caller available for agent '{step.agent_id}'",
            )

        incoming = {p: messages[p] for p in step.predecessors if p in messages}
        memory_context = self._get_memory_context(step.agent_id)
        prompt = self._build_prompt(agent, query, incoming, agent_names, memory_context)

        last_error = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                response = caller(prompt)
                tokens = self.token_counter(prompt) + self.token_counter(response)

                quality = 1.0
                if self._scheduler and self._scheduler.pruning.quality_scorer:
                    quality = self._scheduler.pruning.quality_scorer(response)

                return StepResult(
                    agent_id=step.agent_id,
                    success=True,
                    response=response,
                    tokens_used=tokens,
                    quality_score=quality,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff

        return StepResult(
            agent_id=step.agent_id,
            success=False,
            error=last_error,
        )

    async def _execute_step_async(
        self,
        step: Any,
        messages: dict[str, str],
        agent_lookup: dict[str, Any],
        agent_names: dict[str, str],
        query: str,
    ) -> StepResult:
        """Выполнить шаг асинхронно с ретраями и таймаутом.

        Поддерживает мультимодельность: использует async caller для конкретного агента.
        """
        agent = agent_lookup.get(step.agent_id)
        if agent is None:
            return StepResult(
                agent_id=step.agent_id,
                success=False,
                error=f"Agent '{step.agent_id}' not found",
            )

        # Get async caller for this specific agent (multi-model support)
        async_caller = self._get_async_caller_for_agent(step.agent_id, agent)
        if async_caller is None:
            return StepResult(
                agent_id=step.agent_id,
                success=False,
                error=f"No async LLM caller available for agent '{step.agent_id}'",
            )

        incoming = {p: messages[p] for p in step.predecessors if p in messages}
        memory_context = self._get_memory_context(step.agent_id)
        prompt = self._build_prompt(agent, query, incoming, agent_names, memory_context)

        last_error = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    async_caller(prompt),
                    timeout=self.config.timeout,
                )
                tokens = self.token_counter(prompt) + self.token_counter(response)

                quality = 1.0
                if self._scheduler and self._scheduler.pruning.quality_scorer:
                    quality = self._scheduler.pruning.quality_scorer(response)

                return StepResult(
                    agent_id=step.agent_id,
                    success=True,
                    response=response,
                    tokens_used=tokens,
                    quality_score=quality,
                )
            except TimeoutError:
                last_error = f"Timeout after {self.config.timeout}s"
            except Exception as e:
                last_error = str(e)

            if attempt < self.config.max_retries:
                await asyncio.sleep(delay)
                delay *= self.config.retry_backoff

        return StepResult(
            agent_id=step.agent_id,
            success=False,
            error=last_error,
        )

    async def _execute_parallel(
        self,
        steps: list[Any],
        messages: dict[str, str],
        agent_lookup: dict[str, Any],
        agent_names: dict[str, str],
        query: str,
    ) -> list[tuple[Any, StepResult]]:
        """Асинхронно выполнить группу шагов параллельно."""
        tasks = [
            self._execute_step_async(step, messages, agent_lookup, agent_names, query)
            for step in steps
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                sr = StepResult(
                    agent_id=step.agent_id,
                    success=False,
                    error=str(result),
                )
            else:
                sr = result
            output.append((step, sr))

        return output

    def _get_parallel_group(
        self,
        plan: ExecutionPlan,
        completed_agents: Any,
    ) -> list[Any]:
        """Вернуть группу шагов, готовых к параллельному запуску."""
        completed = set(completed_agents)
        group: list[Any] = []

        for step in plan.remaining_steps:
            if step.agent_id in plan.completed or step.agent_id in plan.skipped:
                continue

            predecessors_done = all(p in completed or p in plan.skipped for p in step.predecessors)

            if predecessors_done:
                group.append(step)
                if len(group) >= self.config.max_parallel_size:
                    break

        return group

    def _determine_final_agent(
        self,
        requested: str | None,
        exec_order: list[str],
        messages: dict[str, str],
    ) -> str:
        """Выбрать финального агента: запрошенный или последний в порядке."""
        if requested and requested in messages:
            return requested
        if exec_order:
            return exec_order[-1]
        return ""

    def _build_agent_states(
        self,
        messages: dict[str, str],
        agent_lookup: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Сформировать новые состояния агентов, дополнив историю ответами."""
        states: dict[str, list[dict[str, Any]]] = {}
        for agent_id, response in messages.items():
            agent = agent_lookup.get(agent_id)
            if agent is not None:
                new_state = list(getattr(agent, "state", []))
                new_state.append({"role": "assistant", "content": response})
                states[agent_id] = new_state
        return states

    def _collect_hidden_states(
        self,
        agent_lookup: dict[str, Any],
    ) -> dict[str, HiddenState]:
        """Собрать текущие hidden_state/embedding агентов в словарь."""
        hidden_states: dict[str, HiddenState] = {}
        for agent_id, agent in agent_lookup.items():
            hs = HiddenState()
            if hasattr(agent, "hidden_state") and agent.hidden_state is not None:
                hs.tensor = agent.hidden_state
            if hasattr(agent, "embedding") and agent.embedding is not None:
                hs.embedding = agent.embedding
            if hs.tensor is not None or hs.embedding is not None:
                hidden_states[agent_id] = hs
        return hidden_states

    def _combine_hidden_states(
        self,
        states: list[HiddenState],
    ) -> HiddenState | None:
        """Комбинировать список скрытых состояний по стратегии hidden_combine_strategy."""
        if not states:
            return None

        tensors = [s.tensor for s in states if s.tensor is not None]
        embeddings = [s.embedding for s in states if s.embedding is not None]

        combined = HiddenState()

        if tensors:
            combined.tensor = self._combine_tensors(tensors)

        if embeddings:
            combined.embedding = self._combine_tensors(embeddings)

        return combined if (combined.tensor is not None or combined.embedding is not None) else None

    def _combine_tensors(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Комбинировать список тензоров согласно стратегии (mean/sum/concat/attention)."""
        if len(tensors) == 1:
            return tensors[0]

        stacked = torch.stack(tensors)

        if self.config.hidden_combine_strategy == "mean":
            return stacked.mean(dim=0)
        elif self.config.hidden_combine_strategy == "sum":
            return stacked.sum(dim=0)
        elif self.config.hidden_combine_strategy == "concat":
            return torch.cat(tensors, dim=-1)
        elif self.config.hidden_combine_strategy == "attention":
            weights = torch.softmax(torch.ones(len(tensors)), dim=0)
            return (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
        else:
            return stacked.mean(dim=0)

    def _get_incoming_hidden(
        self,
        agent_id: str,
        incoming_ids: list[str],
        hidden_states: dict[str, HiddenState],
    ) -> HiddenState | None:
        """Получить объединённое скрытое состояние предшественников."""
        if not self.config.enable_hidden_channels:
            return None

        incoming_states = [hidden_states[aid] for aid in incoming_ids if aid in hidden_states]

        return self._combine_hidden_states(incoming_states)

    def _update_agent_hidden_state(
        self,
        agent: Any,
        response: str,
        incoming_hidden: HiddenState | None,
        hidden_encoder: Any | None = None,
    ) -> HiddenState:
        """Обновить hidden_state агента по ответу и входящему hidden."""
        new_hidden = HiddenState()

        if hasattr(agent, "embedding") and agent.embedding is not None:
            new_hidden.embedding = agent.embedding

        if hidden_encoder is not None:
            try:
                encoded = hidden_encoder.encode([response])
                if isinstance(encoded, torch.Tensor) and encoded.numel() > 0:
                    new_hidden.tensor = encoded[0]
            except Exception:
                pass

        if new_hidden.tensor is None and incoming_hidden is not None:
            new_hidden.tensor = incoming_hidden.tensor

        new_hidden.metadata = {
            "last_response_length": len(response),
            "has_incoming": incoming_hidden is not None,
        }

        return new_hidden

    def run_round_with_hidden(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        hidden_encoder: Any | None = None,
    ) -> MACPResult:
        """Синхронный раунд с передачей скрытых состояний между агентами.

        Поддерживает мультимодельность: каждый агент использует свой LLM caller.
        """
        if not self._has_any_caller():
            raise ValueError("llm_caller, llm_callers, or llm_factory is required")

        original_hidden_setting = self.config.enable_hidden_channels
        self.config.enable_hidden_channels = True

        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            self.config.enable_hidden_channels = original_hidden_setting
            return MACPResult(
                messages={},
                final_answer="",
                final_agent_id="",
                execution_order=[],
            )

        exec_order = build_execution_order(a_agents, agent_ids, role_graph.role_sequence)

        # Инициализация памяти
        self._init_memory(agent_ids)

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)

        hidden_states = self._collect_hidden_states(agent_lookup)

        messages: dict[str, str] = {}
        query = role_graph.query or ""
        total_tokens = 0

        for agent_id in exec_order:
            agent = agent_lookup.get(agent_id)
            if agent is None:
                continue

            incoming_ids = get_incoming_agents(agent_id, a_agents, agent_ids)
            incoming_messages = {aid: messages[aid] for aid in incoming_ids if aid in messages}

            incoming_hidden = self._get_incoming_hidden(agent_id, incoming_ids, hidden_states)

            memory_context = self._get_memory_context(agent_id)
            prompt = self._build_prompt(
                agent, query, incoming_messages, agent_names, memory_context
            )

            if incoming_hidden and incoming_hidden.metadata:
                context_hint = self._format_hidden_context(incoming_hidden)
                if context_hint:
                    prompt = f"{prompt}\n\n[Context: {context_hint}]"

            try:
                # Get caller for this specific agent (multi-model support)
                caller = self._get_caller_for_agent(agent_id, agent)
                if caller is None:
                    messages[agent_id] = f"[Error: No LLM caller available for agent {agent_id}]"
                    continue

                response = caller(prompt)
                messages[agent_id] = response
                total_tokens += self.token_counter(prompt) + self.token_counter(response)
                self._save_to_memory(agent_id, response, incoming_ids)

                hidden_states[agent_id] = self._update_agent_hidden_state(
                    agent, response, incoming_hidden, hidden_encoder
                )
            except Exception as e:
                messages[agent_id] = f"[Error: {e}]"

        final_id = self._determine_final_agent(final_agent_id, exec_order, messages)
        agent_states = self._build_agent_states(messages, agent_lookup)

        self.config.enable_hidden_channels = original_hidden_setting

        return MACPResult(
            messages=messages,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            execution_order=exec_order,
            agent_states=agent_states,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            hidden_states=hidden_states,
        )

    def _format_hidden_context(self, hidden: HiddenState) -> str:
        """Форматировать метаданные скрытого состояния для включения в подсказку."""
        parts = []
        if hidden.metadata:
            if "last_response_length" in hidden.metadata:
                parts.append(f"previous response length: {hidden.metadata['last_response_length']}")
        return ", ".join(parts) if parts else ""

    # =========================================================================
    # STREAMING EXECUTION METHODS
    # =========================================================================

    def stream(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        update_states: bool | None = None,
    ) -> Iterator[AnyStreamEvent]:
        """Stream execution events for real-time output.

        Yields events as agents are executed, allowing real-time monitoring
        and display of intermediate results.

        Args:
            role_graph: The RoleGraph to execute
            final_agent_id: Override which agent produces final answer
            update_states: Whether to update agent states after execution

        Yields:
            StreamEvent instances for each execution phase

        Example:
            for event in runner.stream(graph):
                if event.event_type == StreamEventType.AGENT_OUTPUT:
                    print(f"{event.agent_id}: {event.content}")
                elif event.event_type == StreamEventType.TOKEN:
                    print(event.token, end="", flush=True)
        """
        if not self._has_any_caller() and self.streaming_llm_caller is None:
            raise ValueError(
                "llm_caller, llm_callers, llm_factory, or streaming_llm_caller required for streaming"
            )

        if self.config.adaptive:
            yield from self._stream_adaptive(role_graph, final_agent_id, update_states)
        else:
            yield from self._stream_simple(role_graph, final_agent_id, update_states)

    async def astream(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        update_states: bool | None = None,
    ) -> AsyncIterator[AnyStreamEvent]:
        """Async streaming execution for real-time output.

        Async version of stream() for use in async contexts.

        Args:
            role_graph: The RoleGraph to execute
            final_agent_id: Override which agent produces final answer
            update_states: Whether to update agent states after execution

        Yields:
            StreamEvent instances for each execution phase

        Example:
            async for event in runner.astream(graph):
                match event.event_type:
                    case StreamEventType.AGENT_START:
                        print(f"Agent {event.agent_id} started")
                    case StreamEventType.AGENT_OUTPUT:
                        print(f"Output: {event.content}")
        """
        if not self._has_any_async_caller() and self.async_streaming_llm_caller is None:
            raise ValueError(
                "async_llm_caller, async_llm_callers, llm_factory, or async_streaming_llm_caller required"
            )

        if self.config.adaptive:
            async for event in self._astream_adaptive(role_graph, final_agent_id, update_states):
                yield event
        else:
            async for event in self._astream_simple(role_graph, final_agent_id, update_states):
                yield event

    def _stream_simple(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        update_states: bool | None,
    ) -> Iterator[StreamEvent]:
        """Simple sequential streaming execution."""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            yield RunEndEvent(
                run_id=run_id,
                success=True,
                final_answer="",
                final_agent_id="",
                total_time=time.time() - start_time,
            )
            return

        exec_order = build_execution_order(a_agents, agent_ids, role_graph.role_sequence)
        self._init_memory(agent_ids)

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        task_connected = self._get_task_connected_agents(role_graph)
        query = role_graph.query or ""

        # Emit run start
        yield RunStartEvent(
            run_id=run_id,
            query=query,
            num_agents=len(exec_order),
            execution_order=exec_order,
            config_summary={
                "adaptive": False,
                "timeout": self.config.timeout,
                "enable_memory": self.config.enable_memory,
                "broadcast_task_to_all": self.config.broadcast_task_to_all,
            },
        )

        messages: dict[str, str] = {}
        total_tokens = 0
        errors: list[str] = []

        for step_idx, agent_id in enumerate(exec_order):
            agent = agent_lookup.get(agent_id)
            if agent is None:
                continue

            agent_name = agent_names.get(agent_id, agent_id)
            incoming_ids = get_incoming_agents(agent_id, a_agents, agent_ids)
            incoming_messages = {aid: messages[aid] for aid in incoming_ids if aid in messages}

            include_query = self._should_include_query(agent_id, task_connected)
            memory_context = self._get_memory_context(agent_id)
            prompt = self._build_prompt(
                agent, query, incoming_messages, agent_names, memory_context, include_query
            )

            # Emit agent start
            yield AgentStartEvent(
                run_id=run_id,
                agent_id=agent_id,
                agent_name=agent_name,
                step_index=step_idx,
                predecessors=incoming_ids,
                prompt_preview=prompt[: self.config.prompt_preview_length],
            )

            step_start = time.time()

            try:
                # Get caller for this specific agent (multi-model support)
                caller = self._get_caller_for_agent(agent_id, agent)
                if caller is None:
                    raise ValueError(f"No LLM caller available for agent {agent_id}")

                # Use streaming LLM if available and enabled
                if self.streaming_llm_caller and self.config.enable_token_streaming:
                    response_parts: list[str] = []
                    token_idx = 0

                    for token in self.streaming_llm_caller(prompt):
                        response_parts.append(token)
                        yield TokenEvent(
                            run_id=run_id,
                            agent_id=agent_id,
                            token=token,
                            token_index=token_idx,
                            is_first=(token_idx == 0),
                            is_last=False,
                        )
                        token_idx += 1

                    # Mark last token
                    if response_parts:
                        yield TokenEvent(
                            run_id=run_id,
                            agent_id=agent_id,
                            token="",
                            token_index=token_idx,
                            is_first=False,
                            is_last=True,
                        )

                    response = "".join(response_parts)
                else:
                    # Use regular LLM caller for this agent
                    response = caller(prompt)

                messages[agent_id] = response
                tokens = self.token_counter(prompt) + self.token_counter(response)
                total_tokens += tokens
                self._save_to_memory(agent_id, response, incoming_ids)

                is_final = (step_idx == len(exec_order) - 1) or (agent_id == final_agent_id)

                yield AgentOutputEvent(
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    content=response,
                    tokens_used=tokens,
                    duration_ms=(time.time() - step_start) * 1000,
                    is_final=is_final,
                )

            except Exception as e:
                error_msg = str(e)
                errors.append(f"{agent_id}: {error_msg}")
                messages[agent_id] = f"[Error: {e}]"

                yield AgentErrorEvent(
                    run_id=run_id,
                    agent_id=agent_id,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    will_retry=False,
                )

        final_id = self._determine_final_agent(final_agent_id, exec_order, messages)

        yield RunEndEvent(
            run_id=run_id,
            success=len(errors) == 0,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            executed_agents=list(messages.keys()),
            errors=errors,
        )

    async def _astream_simple(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        update_states: bool | None,
    ) -> AsyncIterator[StreamEvent]:
        """Async simple sequential streaming execution."""
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            yield RunEndEvent(
                run_id=run_id,
                success=True,
                final_answer="",
                final_agent_id="",
                total_time=time.time() - start_time,
            )
            return

        exec_order = build_execution_order(a_agents, agent_ids, role_graph.role_sequence)
        self._init_memory(agent_ids)

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        query = role_graph.query or ""

        yield RunStartEvent(
            run_id=run_id,
            query=query,
            num_agents=len(exec_order),
            execution_order=exec_order,
        )

        messages: dict[str, str] = {}
        total_tokens = 0
        errors: list[str] = []

        for step_idx, agent_id in enumerate(exec_order):
            agent = agent_lookup.get(agent_id)
            if agent is None:
                continue

            agent_name = agent_names.get(agent_id, agent_id)
            incoming_ids = get_incoming_agents(agent_id, a_agents, agent_ids)
            incoming_messages = {aid: messages[aid] for aid in incoming_ids if aid in messages}

            memory_context = self._get_memory_context(agent_id)
            prompt = self._build_prompt(
                agent, query, incoming_messages, agent_names, memory_context
            )

            yield AgentStartEvent(
                run_id=run_id,
                agent_id=agent_id,
                agent_name=agent_name,
                step_index=step_idx,
                predecessors=incoming_ids,
                prompt_preview=prompt[: self.config.prompt_preview_length],
            )

            step_start = time.time()

            try:
                # Use async streaming LLM if available
                if self.async_streaming_llm_caller and self.config.enable_token_streaming:
                    response_parts: list[str] = []
                    token_idx = 0

                    async for token in self.async_streaming_llm_caller(prompt):
                        response_parts.append(token)
                        yield TokenEvent(
                            run_id=run_id,
                            agent_id=agent_id,
                            token=token,
                            token_index=token_idx,
                            is_first=(token_idx == 0),
                            is_last=False,
                        )
                        token_idx += 1

                    if response_parts:
                        yield TokenEvent(
                            run_id=run_id,
                            agent_id=agent_id,
                            token="",
                            token_index=token_idx,
                            is_first=False,
                            is_last=True,
                        )

                    response = "".join(response_parts)
                else:
                    response = await asyncio.wait_for(
                        self.async_llm_caller(prompt),  # type: ignore
                        timeout=self.config.timeout,
                    )

                messages[agent_id] = response
                tokens = self.token_counter(prompt) + self.token_counter(response)
                total_tokens += tokens
                self._save_to_memory(agent_id, response, incoming_ids)

                is_final = (step_idx == len(exec_order) - 1) or (agent_id == final_agent_id)

                yield AgentOutputEvent(
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    content=response,
                    tokens_used=tokens,
                    duration_ms=(time.time() - step_start) * 1000,
                    is_final=is_final,
                )

            except Exception as e:
                error_msg = str(e)
                errors.append(f"{agent_id}: {error_msg}")
                messages[agent_id] = f"[Error: {e}]"

                yield AgentErrorEvent(
                    run_id=run_id,
                    agent_id=agent_id,
                    error_type=type(e).__name__,
                    error_message=error_msg,
                )

        final_id = self._determine_final_agent(final_agent_id, exec_order, messages)

        yield RunEndEvent(
            run_id=run_id,
            success=len(errors) == 0,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            executed_agents=list(messages.keys()),
            errors=errors,
        )

    def _stream_adaptive(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        update_states: bool | None,
    ) -> Iterator[StreamEvent]:
        """Adaptive streaming execution with replanning, pruning, and fallback."""
        if self._scheduler is None:
            raise ValueError("Scheduler not initialized for adaptive mode")

        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            yield RunEndEvent(run_id=run_id, success=True, total_time=0)
            return

        self._init_memory(agent_ids)
        p_matrix = self._extract_p_matrix(role_graph, task_idx)
        plan = self._scheduler.build_plan(a_agents, agent_ids, p_matrix, end_agent=final_agent_id)

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        query = role_graph.query or ""

        yield RunStartEvent(
            run_id=run_id,
            query=query,
            num_agents=len(agent_ids),
            execution_order=plan.execution_order,
            config_summary={"adaptive": True, "policy": self.config.routing_policy.value},
        )

        messages: dict[str, str] = {}
        step_results: dict[str, StepResult] = {}
        execution_order: list[str] = []
        fallback_attempts: dict[str, int] = {}
        replanning_count = 0
        errors: list[str] = []
        total_tokens = 0
        step_idx = 0

        while not plan.is_complete:
            step = plan.get_current_step()
            if step is None:
                break

            # Check for pruning
            should_prune, reason = self._scheduler.should_prune(
                step, plan, step_results.get(execution_order[-1]) if execution_order else None
            )

            if should_prune:
                plan.mark_skipped(step.agent_id)

                yield PruneEvent(
                    run_id=run_id,
                    agent_id=step.agent_id,
                    reason=reason,
                )
                continue

            agent = agent_lookup.get(step.agent_id)
            if agent is None:
                continue

            agent_name = agent_names.get(step.agent_id, step.agent_id)
            incoming = {p: messages[p] for p in step.predecessors if p in messages}
            memory_context = self._get_memory_context(step.agent_id)
            prompt = self._build_prompt(agent, query, incoming, agent_names, memory_context)

            yield AgentStartEvent(
                run_id=run_id,
                agent_id=step.agent_id,
                agent_name=agent_name,
                step_index=step_idx,
                predecessors=step.predecessors,
                prompt_preview=prompt[: self.config.prompt_preview_length],
            )

            step_start = time.time()
            result = self._execute_step(step, messages, agent_lookup, agent_names, query)
            step_results[step.agent_id] = result
            execution_order.append(step.agent_id)

            if result.success:
                messages[step.agent_id] = result.response or ""
                plan.mark_completed(step.agent_id, result.tokens_used)
                total_tokens += result.tokens_used
                self._save_to_memory(step.agent_id, result.response or "", step.predecessors)

                yield AgentOutputEvent(
                    run_id=run_id,
                    agent_id=step.agent_id,
                    agent_name=agent_name,
                    content=result.response or "",
                    tokens_used=result.tokens_used,
                    duration_ms=(time.time() - step_start) * 1000,
                )
            else:
                plan.mark_failed(step.agent_id)
                errors.append(f"{step.agent_id}: {result.error}")

                yield AgentErrorEvent(
                    run_id=run_id,
                    agent_id=step.agent_id,
                    error_type="ExecutionError",
                    error_message=result.error or "Unknown error",
                )

                # Handle fallback
                attempts = fallback_attempts.get(step.agent_id, 0)
                if self._scheduler.should_use_fallback(step, result, attempts):
                    for fb_agent in step.fallback_agents:
                        if fb_agent not in plan.completed and fb_agent not in plan.failed:
                            plan.insert_fallback(fb_agent, plan.current_index - 1)

                            yield FallbackEvent(
                                run_id=run_id,
                                failed_agent_id=step.agent_id,
                                fallback_agent_id=fb_agent,
                                attempt=attempts + 1,
                            )
                            break
                    fallback_attempts[step.agent_id] = attempts + 1

            # Handle replanning
            if self.config.enable_replanning and not plan.is_complete:
                should_replan = not self.config.replan_on_error_only or not result.success
                if should_replan:
                    old_remaining = [s.agent_id for s in plan.remaining_steps]
                    plan = self._scheduler.replan(plan, a_agents, agent_ids, p_matrix, step_results)
                    new_remaining = [s.agent_id for s in plan.remaining_steps]
                    replanning_count += 1

                    if old_remaining != new_remaining:
                        yield ReplanEvent(
                            run_id=run_id,
                            reason="Quality/error based replanning",
                            old_remaining=old_remaining,
                            new_remaining=new_remaining,
                            replan_count=replanning_count,
                        )

            step_idx += 1

        final_id = self._determine_final_agent(final_agent_id, execution_order, messages)

        yield RunEndEvent(
            run_id=run_id,
            success=len(errors) == 0,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            executed_agents=execution_order,
            errors=errors,
        )

    async def _astream_adaptive(
        self,
        role_graph: Any,
        final_agent_id: str | None,
        update_states: bool | None,
    ) -> AsyncIterator[StreamEvent]:
        """Async adaptive streaming with parallel execution support."""
        if self._scheduler is None:
            raise ValueError("Scheduler not initialized for adaptive mode")

        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        task_idx = self._get_task_index(role_graph)
        a_agents = extract_agent_adjacency(role_graph.A_com, task_idx)
        agent_ids, _ = self._get_agent_ids(role_graph, task_idx)

        if not agent_ids:
            yield RunEndEvent(run_id=run_id, success=True, total_time=0)
            return

        self._init_memory(agent_ids)
        p_matrix = self._extract_p_matrix(role_graph, task_idx)
        plan = self._scheduler.build_plan(a_agents, agent_ids, p_matrix, end_agent=final_agent_id)

        agent_lookup = {a.identifier: a for a in role_graph.agents}
        agent_names = self._build_agent_names(role_graph)
        query = role_graph.query or ""

        yield RunStartEvent(
            run_id=run_id,
            query=query,
            num_agents=len(agent_ids),
            execution_order=plan.execution_order,
            config_summary={
                "adaptive": True,
                "parallel": self.config.enable_parallel,
                "policy": self.config.routing_policy.value,
            },
        )

        messages: dict[str, str] = {}
        step_results: dict[str, StepResult] = {}
        execution_order: list[str] = []
        fallback_attempts: dict[str, int] = {}
        replanning_count = 0
        errors: list[str] = []
        total_tokens = 0
        group_idx = 0

        while not plan.is_complete:
            parallel_group = self._get_parallel_group(plan, messages.keys())
            if not parallel_group:
                break

            # Filter pruned steps
            valid_steps = []
            for step in parallel_group:
                should_prune, reason = self._scheduler.should_prune(step, plan, None)
                if should_prune:
                    plan.mark_skipped(step.agent_id)
                    yield PruneEvent(run_id=run_id, agent_id=step.agent_id, reason=reason)
                else:
                    valid_steps.append(step)

            if not valid_steps:
                continue

            # Emit parallel start event
            if self.config.enable_parallel and len(valid_steps) > 1:
                yield ParallelStartEvent(
                    run_id=run_id,
                    agent_ids=[s.agent_id for s in valid_steps],
                    group_index=group_idx,
                )

            # Emit agent start events
            for step in valid_steps:
                agent_name = agent_names.get(step.agent_id, step.agent_id)
                incoming = {p: messages[p] for p in step.predecessors if p in messages}
                agent = agent_lookup.get(step.agent_id)
                if agent:
                    memory_context = self._get_memory_context(step.agent_id)
                    prompt = self._build_prompt(agent, query, incoming, agent_names, memory_context)
                    yield AgentStartEvent(
                        run_id=run_id,
                        agent_id=step.agent_id,
                        agent_name=agent_name,
                        predecessors=step.predecessors,
                        prompt_preview=prompt[: self.config.prompt_preview_length],
                    )

            # Execute steps (parallel or sequential)
            if self.config.enable_parallel and len(valid_steps) > 1:
                results = await self._execute_parallel(
                    valid_steps, messages, agent_lookup, agent_names, query
                )
            else:
                results = []
                for step in valid_steps:
                    r = await self._execute_step_async(
                        step, messages, agent_lookup, agent_names, query
                    )
                    results.append((step, r))

            # Process results and emit events
            successful: list[str] = []
            failed: list[str] = []

            for step, result in results:
                step_results[step.agent_id] = result
                execution_order.append(step.agent_id)
                agent_name = agent_names.get(step.agent_id, step.agent_id)

                if result.success:
                    messages[step.agent_id] = result.response or ""
                    plan.mark_completed(step.agent_id, result.tokens_used)
                    total_tokens += result.tokens_used
                    self._save_to_memory(step.agent_id, result.response or "", step.predecessors)
                    successful.append(step.agent_id)

                    yield AgentOutputEvent(
                        run_id=run_id,
                        agent_id=step.agent_id,
                        agent_name=agent_name,
                        content=result.response or "",
                        tokens_used=result.tokens_used,
                    )
                else:
                    plan.mark_failed(step.agent_id)
                    errors.append(f"{step.agent_id}: {result.error}")
                    failed.append(step.agent_id)

                    yield AgentErrorEvent(
                        run_id=run_id,
                        agent_id=step.agent_id,
                        error_message=result.error or "Unknown error",
                    )

                    # Handle fallback
                    attempts = fallback_attempts.get(step.agent_id, 0)
                    if self._scheduler.should_use_fallback(step, result, attempts):
                        for fb_agent in step.fallback_agents:
                            if fb_agent not in plan.completed and fb_agent not in plan.failed:
                                plan.insert_fallback(fb_agent, plan.current_index - 1)
                                yield FallbackEvent(
                                    run_id=run_id,
                                    failed_agent_id=step.agent_id,
                                    fallback_agent_id=fb_agent,
                                    attempt=attempts + 1,
                                )
                                break
                        fallback_attempts[step.agent_id] = attempts + 1

            # Emit parallel end event
            if self.config.enable_parallel and len(valid_steps) > 1:
                yield ParallelEndEvent(
                    run_id=run_id,
                    agent_ids=[s.agent_id for s in valid_steps],
                    group_index=group_idx,
                    successful=successful,
                    failed=failed,
                )

            # Handle replanning
            if self.config.enable_replanning and not plan.is_complete:
                any_failed = any(not r.success for _, r in results)
                should_replan = not self.config.replan_on_error_only or any_failed
                if should_replan:
                    old_remaining = [s.agent_id for s in plan.remaining_steps]
                    plan = self._scheduler.replan(plan, a_agents, agent_ids, p_matrix, step_results)
                    new_remaining = [s.agent_id for s in plan.remaining_steps]
                    replanning_count += 1

                    if old_remaining != new_remaining:
                        yield ReplanEvent(
                            run_id=run_id,
                            reason="Adaptive replanning",
                            old_remaining=old_remaining,
                            new_remaining=new_remaining,
                            replan_count=replanning_count,
                        )

            group_idx += 1

        final_id = self._determine_final_agent(final_agent_id, execution_order, messages)

        yield RunEndEvent(
            run_id=run_id,
            success=len(errors) == 0,
            final_answer=messages.get(final_id, ""),
            final_agent_id=final_id,
            total_tokens=total_tokens,
            total_time=time.time() - start_time,
            executed_agents=execution_order,
            errors=errors,
        )

    def stream_to_result(
        self,
        role_graph: Any,
        final_agent_id: str | None = None,
        update_states: bool | None = None,
    ) -> tuple[Iterator[StreamEvent], MACPResult]:
        """Stream execution and also return final MACPResult.

        Useful when you want both streaming display and complete result.

        Returns:
            Tuple of (event iterator, MACPResult)

        Example:
            stream, result_future = runner.stream_to_result(graph)
            for event in stream:
                print(event)
            result = result_future  # Available after stream exhausted
        """
        events: list[StreamEvent] = []
        messages: dict[str, str] = {}
        final_answer = ""
        final_agent = ""
        execution_order: list[str] = []
        total_tokens = 0
        total_time = 0.0
        errors_list: list[str] = []

        def collecting_stream() -> Iterator[StreamEvent]:
            nonlocal final_answer, final_agent, total_tokens, total_time, errors_list

            for event in self.stream(role_graph, final_agent_id, update_states):
                events.append(event)

                if isinstance(event, AgentOutputEvent):
                    messages[event.agent_id] = event.content
                    execution_order.append(event.agent_id)

                elif isinstance(event, RunEndEvent):
                    final_answer = event.final_answer
                    final_agent = event.final_agent_id
                    total_tokens = event.total_tokens
                    total_time = event.total_time
                    errors_list = event.errors

                yield event

        stream = collecting_stream()

        # Create a lazy result that becomes valid after stream is exhausted
        class LazyResult:
            def __init__(self, runner: MACPRunner):
                self._runner = runner
                self._result: MACPResult | None = None

            def __getattr__(self, name: str) -> Any:
                if self._result is None:
                    self._result = MACPResult(
                        messages=messages,
                        final_answer=final_answer,
                        final_agent_id=final_agent,
                        execution_order=execution_order,
                        total_tokens=total_tokens,
                        total_time=total_time,
                        errors=[
                            ExecutionError(message=e, agent_id="", recoverable=False)
                            for e in errors_list
                        ]
                        if errors_list
                        else None,
                    )
                return getattr(self._result, name)

        return stream, LazyResult(self)  # type: ignore
