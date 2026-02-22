"""
Построение графа агентов из профилей.

Поддерживает:
- Валидацию топологии (циклы, дубликаты, направленность)
- Кастомные атрибуты узлов и рёбер
- Расширяемые схемы с версионированием
- Экспорт в PyG формат с произвольными признаками
"""

import itertools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import rustworkx as rx
import torch

from rustworkx_framework.core.schema import (
    SCHEMA_VERSION,
    AgentNodeSchema,
    BaseEdgeSchema,
    BaseNodeSchema,
    CostMetrics,
    EdgeType,
    GraphSchema,
    LLMConfig,
    NodeType,
    SchemaValidator,
    TaskNodeSchema,
    ValidationResult,
    WorkflowEdgeSchema,
)

__all__ = [
    "BuilderConfig",
    "GraphBuilder",
    "build_from_adjacency",
    "build_from_schema",
    "build_property_graph",
    "default_edges",
    "default_sequence",
]

_TASK_NODE_ID = "__task__"

if TYPE_CHECKING:
    from rustworkx_framework.core.graph import RoleGraph


class BuilderConfig:
    """
    Настройки построителя графа.

    Контролирует проверку схемы, допустимость циклов/дубликатов и ширину
    признаков, а также поведение по умолчанию для весов и подключения
    виртуального узла задачи.
    """

    def __init__(
        self,
        validate: bool = True,
        check_cycles: bool = True,
        check_duplicates: bool = True,
        allow_self_loops: bool = False,
        node_feature_names: list[str] | None = None,
        edge_feature_names: list[str] | None = None,
        default_edge_dim: int | None = None,
        weight_fn: Callable[[str, str, dict], float] | None = None,
        default_weight: float = 1.0,
        include_task_node: bool = True,
        task_edge_weight: float = 1.0,
    ):
        """
        Создать конфигурацию построителя.

        Args:
            validate: Запускать ли валидацию схемы перед сборкой графа.
            check_cycles: Проверять ли наличие циклов при валидации.
            check_duplicates: Проверять ли дубликаты узлов и рёбер.
            allow_self_loops: Разрешать ли самосвязи в графе.
            node_feature_names: Имена признаков узлов, попадающие в схему.
            edge_feature_names: Имена признаков рёбер, попадающие в схему.
            default_edge_dim: Размерность вектора признаков рёбер по умолчанию.
            weight_fn: Пользовательская функция для вычисления веса ребра.
            default_weight: Вес ребра, если weight_fn не задана.
            include_task_node: Добавлять ли виртуальный узел задачи.
            task_edge_weight: Вес рёбер между задачей и агентами.

        """
        self.validate = validate
        self.check_cycles = check_cycles
        self.check_duplicates = check_duplicates
        self.allow_self_loops = allow_self_loops

        self.node_feature_names = node_feature_names or []
        self.edge_feature_names = edge_feature_names or []
        self.default_edge_dim = default_edge_dim

        self.weight_fn = weight_fn
        self.default_weight = default_weight

        self.include_task_node = include_task_node
        self.task_edge_weight = task_edge_weight


class GraphBuilder:
    """
    Удобный интерфейс для поэтапного построения `GraphSchema`.

    Поддерживает условную маршрутизацию (conditional routing) через
    методы add_conditional_edge и add_conditional_edges.

    Поддерживает явное указание start/end нод для оптимизации выполнения.

    Example:
        builder = GraphBuilder()
        builder.add_agent("solver", description="Solves problems")
        builder.add_agent("reviewer", description="Reviews solutions")
        builder.add_agent("finalizer", description="Finalizes answer")

        # Безусловные рёбра
        builder.add_workflow_edge("solver", "reviewer")

        # Условное ребро: переход к finalizer только если reviewer успешен
        builder.add_conditional_edge(
            "reviewer", "finalizer",
            condition=lambda ctx: "approved" in ctx.get_last_response().lower()
        )

        # Установить явные границы выполнения
        builder.set_start_node("solver")
        builder.set_end_node("finalizer")

        graph = builder.build()

    """

    def __init__(self, config: BuilderConfig | None = None):
        """Инициализировать построитель с указанной конфигурацией."""
        self.config = config or BuilderConfig()
        self._schema = GraphSchema(
            schema_version=SCHEMA_VERSION,
            node_feature_names=self.config.node_feature_names,
            edge_feature_names=self.config.edge_feature_names,
        )
        self._validator = SchemaValidator(
            check_cycles=self.config.check_cycles,
            check_duplicates=self.config.check_duplicates,
        )
        # Callable условия хранятся отдельно (не сериализуются в схему)
        self._edge_conditions: dict[tuple[str, str], Callable] = {}
        # Явные границы выполнения графа
        self._start_node: str | None = None
        self._end_node: str | None = None
        # Профили агентов (для передачи tool объектов)
        self._agent_profiles: dict[str, Any] = {}

    def add_agent(
        self,
        agent_id: str,
        display_name: str | None = None,
        persona: str = "",
        description: str = "",
        embedding: list[float] | None = None,
        trust_score: float = 1.0,
        llm_config: LLMConfig | None = None,
        tools: list[str] | None = None,
        input_schema: Any | None = None,
        output_schema: Any | None = None,
        llm_backbone: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить узел-агента в схему с опциональной LLM конфигурацией и валидацией.

        Args:
            agent_id: Уникальный идентификатор агента.
            display_name: Отображаемое имя (по умолчанию равно id).
            persona: Краткое позиционирование/роль.
            description: Текстовое описание возможностей.
            embedding: Вектор признаков агента (если уже вычислен).
            trust_score: Базовая оценка доверия агенту.
            llm_config: Готовый объект LLMConfig (переопределяет отдельные параметры).
            tools: Список инструментов агента.
            input_schema: Pydantic модель или JSON Schema для валидации входящих данных.
            output_schema: Pydantic модель или JSON Schema для валидации ответа агента.
            llm_backbone: Имя модели (например, "gpt-4", "claude-3-opus").
            base_url: URL API endpoint (например, "https://api.openai.com/v1").
            api_key: API ключ или ссылка на переменную окружения ("$OPENAI_API_KEY").
            max_tokens: Максимальное количество токенов в ответе.
            temperature: Температура генерации (0.0-2.0).
            timeout: Таймаут запроса в секундах.
            top_p: Top-p (nucleus) sampling parameter.
            stop_sequences: Последовательности для остановки генерации.
            **metadata: Произвольные дополнительные поля.

        Example:
            # Агент с OpenAI GPT-4
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

            # Агент с готовым LLMConfig
            config = LLMConfig(model_name="claude-3-opus", ...)
            builder.add_agent("reviewer", llm_config=config)

            # Агент с валидацией ввода/вывода
            from pydantic import BaseModel

            class SolverInput(BaseModel):
                question: str
                context: str | None = None

            class SolverOutput(BaseModel):
                answer: str
                confidence: float

            builder.add_agent(
                "solver",
                persona="Math solver",
                input_schema=SolverInput,
                output_schema=SolverOutput
            )

        """
        # If llm_config provided, use it as fallback
        effective_llm_backbone = llm_backbone
        effective_base_url = base_url
        effective_api_key = api_key
        effective_max_tokens = max_tokens
        effective_temperature = temperature
        effective_timeout = timeout
        effective_top_p = top_p
        effective_stop_sequences = stop_sequences

        if llm_config:
            effective_llm_backbone = llm_backbone or llm_config.model_name
            effective_base_url = base_url or llm_config.base_url
            effective_api_key = api_key or llm_config.api_key
            effective_max_tokens = max_tokens if max_tokens is not None else llm_config.max_tokens
            effective_temperature = temperature if temperature is not None else llm_config.temperature
            effective_timeout = timeout if timeout is not None else llm_config.timeout
            effective_top_p = top_p if top_p is not None else llm_config.top_p
            effective_stop_sequences = stop_sequences or llm_config.stop_sequences

        node = AgentNodeSchema(
            id=agent_id,
            display_name=display_name or agent_id,
            persona=persona,
            description=description,
            embedding=embedding,
            trust_score=trust_score,
            llm_backbone=effective_llm_backbone,
            base_url=effective_base_url,
            api_key=effective_api_key,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            timeout=effective_timeout,
            top_p=effective_top_p,
            stop_sequences=effective_stop_sequences,
            tools=tools or [],
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
        )
        self._schema.add_node(node)
        return self

    def add_agent_profile(
        self,
        profile: Any,  # AgentProfile
        trust_score: float = 1.0,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить агента из готового AgentProfile объекта.

        Args:
            profile: AgentProfile объект.
            trust_score: Базовая оценка доверия агенту.
            **metadata: Дополнительные поля.

        Example:
            from rustworkx_framework.core.agent import AgentProfile
            from rustworkx_framework.tools import CodeInterpreterTool

            agent = AgentProfile(
                agent_id="coder",
                display_name="Coder Agent",
                persona="a Python programmer",
                tools=[CodeInterpreterTool()],
            )

            builder = GraphBuilder()
            builder.add_agent_profile(agent)

        """
        # Получаем имена tools (строки или имена из объектов)
        tool_names = []
        if hasattr(profile, "get_tool_names"):
            tool_names = profile.get_tool_names()
        elif hasattr(profile, "tools"):
            tool_names = profile.tools

        # Получаем LLM config если есть
        llm_config = None
        if hasattr(profile, "llm_config") and profile.llm_config:
            llm_config = profile.llm_config

        node = AgentNodeSchema(
            id=profile.agent_id,
            display_name=profile.display_name,
            persona=getattr(profile, "persona", ""),
            description=getattr(profile, "description", ""),
            embedding=None,  # Будет вычислен позже если нужно
            trust_score=trust_score,
            llm_backbone=getattr(profile, "llm_backbone", None),
            base_url=llm_config.base_url if llm_config else None,
            api_key=llm_config.api_key if llm_config else None,
            max_tokens=llm_config.max_tokens if llm_config else None,
            temperature=llm_config.temperature if llm_config else None,
            timeout=llm_config.timeout if llm_config else None,
            top_p=llm_config.top_p if llm_config else None,
            stop_sequences=llm_config.stop_sequences if llm_config else None,
            tools=tool_names,
            input_schema=getattr(profile, "input_schema", None),
            output_schema=getattr(profile, "output_schema", None),
            metadata=metadata,
        )
        self._schema.add_node(node)

        # Сохраняем оригинальный profile для передачи tools объектов в runner
        self._agent_profiles[profile.agent_id] = profile

        return self

    def add_task(
        self,
        task_id: str = _TASK_NODE_ID,
        query: str = "",
        description: str = "",
        embedding: list[float] | None = None,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить виртуальный узел задачи.

        Args:
            task_id: Идентификатор узла задачи.
            query: Запрос/формулировка задачи.
            description: Дополнительное описание контекста.
            embedding: Вектор признаков задачи.
            **metadata: Дополнительные произвольные поля.

        """
        node = TaskNodeSchema(
            id=task_id,
            query=query,
            description=description,
            embedding=embedding,
            metadata=metadata,
        )
        self._schema.add_node(node)
        return self

    def add_node(
        self,
        node_id: str,
        node_type: NodeType = NodeType.CUSTOM,
        **kwargs,
    ) -> "GraphBuilder":
        """Добавить произвольный узел указанного типа."""
        node = BaseNodeSchema(
            id=node_id,
            type=node_type,
            **kwargs,
        )
        self._schema.add_node(node)
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.WORKFLOW,
        weight: float | None = None,
        probability: float = 1.0,
        cost: CostMetrics | None = None,
        attr: list[float] | None = None,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить базовое ребро между узлами.

        Args:
            source: Источник ребра.
            target: Приёмник ребра.
            edge_type: Тип ребра (workflow/task/...).
            weight: Вес ребра; если None, будет вычислен weight_fn или default.
            probability: Вероятность передачи/активации.
            cost: Метрики стоимости перехода.
            attr: Вектор признаков ребра.
            **metadata: Дополнительные метаданные, записываемые в схему.

        """
        if weight is None:
            if self.config.weight_fn:
                weight = self.config.weight_fn(source, target, metadata)
            else:
                weight = self.config.default_weight

        if source == target and not self.config.allow_self_loops:
            msg = f"Self-loops not allowed: {source} -> {target}"
            raise ValueError(msg)

        edge = BaseEdgeSchema(
            source=source,
            target=target,
            type=edge_type,
            weight=weight,
            probability=probability,
            cost=cost or CostMetrics(),
            attr=attr,
            metadata=metadata,
        )
        self._schema.add_edge(edge)
        return self

    def add_workflow_edge(
        self,
        source: str,
        target: str,
        weight: float | None = None,
        condition: str | None = None,
        priority: int = 0,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить ребро бизнес-логики (workflow).

        Args:
            source: Узел-источник.
            target: Узел-приёмник.
            weight: Вес, если None — рассчитывается как для обычного ребра.
            condition: Условие активации (например, фильтр маршрута).
            priority: Приоритет исполнения в рамках workflow.
            **metadata: Дополнительные атрибуты схемы.

        """
        if weight is None:
            if self.config.weight_fn:
                weight = self.config.weight_fn(source, target, metadata)
            else:
                weight = self.config.default_weight

        edge = WorkflowEdgeSchema(
            source=source,
            target=target,
            weight=weight,
            condition=condition,
            priority=priority,
            metadata=metadata,
        )
        self._schema.add_edge(edge)
        return self

    def add_conditional_edge(
        self,
        source: str,
        target: str,
        condition: Callable | str,
        weight: float | None = None,
        priority: int = 0,
        **metadata,
    ) -> "GraphBuilder":
        """
        Добавить условное ребро (conditional edge).

        Условие вычисляется в runtime при построении плана выполнения.
        Если условие не выполнено, ребро игнорируется.

        Args:
            source: Узел-источник.
            target: Узел-приёмник.
            condition: Условие активации:
                - Callable[[ConditionContext], bool] — функция проверки
                - str — имя зарегистрированного условия или выражение
            weight: Вес ребра (если None — default_weight).
            priority: Приоритет ребра.
            **metadata: Дополнительные атрибуты.

        Returns:
            self для chaining.

        Example:
            # Callable condition
            builder.add_conditional_edge(
                "solver", "reviewer",
                condition=lambda ctx: ctx.source_succeeded()
            )

            # String condition (встроенное)
            builder.add_conditional_edge(
                "reviewer", "finalizer",
                condition="source_success"
            )

            # String condition (проверка содержимого)
            builder.add_conditional_edge(
                "classifier", "math_agent",
                condition="contains:math"
            )

        """
        if weight is None:
            if self.config.weight_fn:
                weight = self.config.weight_fn(source, target, metadata)
            else:
                weight = self.config.default_weight

        # Строковое условие сохраняется в схеме
        condition_str = condition if isinstance(condition, str) else None

        edge = WorkflowEdgeSchema(
            source=source,
            target=target,
            weight=weight,
            condition=condition_str,
            priority=priority,
            is_conditional=True,
            metadata=metadata,
        )
        self._schema.add_edge(edge)

        # Callable условие сохраняется отдельно
        if callable(condition):
            self._edge_conditions[(source, target)] = condition

        return self

    def add_conditional_edges(
        self,
        source: str,
        path_map: dict[str, Callable | str | None],
        default: str | None = None,
        weight: float | None = None,
    ) -> "GraphBuilder":
        """
        Добавить несколько условных рёбер от одного источника (router pattern).

        Аналог LangGraph add_conditional_edges — выбор следующего узла
        на основе условий.

        Args:
            source: Узел-источник.
            path_map: Словарь {target: condition}.
                      condition=None означает безусловный переход.
            default: Узел по умолчанию, если ни одно условие не выполнено.
            weight: Вес для всех рёбер.

        Returns:
            self для chaining.

        Example:
            # Router: классификатор направляет к разным агентам
            builder.add_conditional_edges(
                "classifier",
                {
                    "math_agent": lambda ctx: "math" in ctx.get_last_response(),
                    "code_agent": lambda ctx: "code" in ctx.get_last_response(),
                    "general_agent": None,  # fallback
                },
                default="general_agent"
            )

        """
        for target, condition in path_map.items():
            if condition is None:
                # Безусловное ребро
                self.add_workflow_edge(source, target, weight=weight)
            else:
                self.add_conditional_edge(source, target, condition, weight=weight)

        # Если есть default и он ещё не добавлен
        if default and default not in path_map:
            self.add_workflow_edge(source, default, weight=weight)

        return self

    @property
    def edge_conditions(self) -> dict[tuple[str, str], Callable]:
        """Получить все callable условия для рёбер."""
        return self._edge_conditions.copy()

    def set_start_node(self, node_id: str) -> "GraphBuilder":
        """
        Установить стартовую ноду для выполнения.

        Стартовая нода — это точка входа в граф. Ноды, недостижимые из
        стартовой ноды, будут исключены из выполнения.

        Args:
            node_id: ID ноды (должна быть добавлена до вызова build()).

        Returns:
            self для chaining.

        Example:
            builder.add_agent("input_agent", ...)
            builder.set_start_node("input_agent")

        """
        self._start_node = node_id
        return self

    def set_end_node(self, node_id: str) -> "GraphBuilder":
        """
        Установить конечную ноду для выполнения.

        Конечная нода — это точка выхода из графа. Ноды, из которых
        невозможно достичь конечной ноды, будут исключены из выполнения.

        Args:
            node_id: ID ноды (должна быть добавлена до вызова build()).

        Returns:
            self для chaining.

        Example:
            builder.add_agent("output_agent", ...)
            builder.set_end_node("output_agent")

        """
        self._end_node = node_id
        return self

    def set_execution_bounds(
        self,
        start_node: str | None,
        end_node: str | None,
    ) -> "GraphBuilder":
        """
        Установить границы выполнения (стартовую и конечную ноды).

        Args:
            start_node: ID стартовой ноды (None для автоопределения).
            end_node: ID конечной ноды (None для автоопределения).

        Returns:
            self для chaining.

        Example:
            builder.set_execution_bounds("input", "output")

        """
        self._start_node = start_node
        self._end_node = end_node
        return self

    @property
    def start_node(self) -> str | None:
        """Получить ID стартовой ноды."""
        return self._start_node

    @property
    def end_node(self) -> str | None:
        """Получить ID конечной ноды."""
        return self._end_node

    def connect_task_to_agents(
        self,
        task_id: str = _TASK_NODE_ID,
        agent_ids: list[str] | None = None,
        bidirectional: bool = True,
    ) -> "GraphBuilder":
        """
        Соединить узел задачи с агентами рёбрами контекста/обновлений.

        Args:
            task_id: Идентификатор узла задачи.
            agent_ids: Список агентов; если None, используются все агенты схемы.
            bidirectional: Добавлять ли обратные рёбра от агента к задаче.

        """
        if agent_ids is None:
            agent_ids = [node_id for node_id, node in self._schema.nodes.items() if node.type == NodeType.AGENT]

        for agent_id in agent_ids:
            self.add_edge(
                task_id,
                agent_id,
                edge_type=EdgeType.TASK_CONTEXT,
                weight=self.config.task_edge_weight,
            )

            if bidirectional:
                self.add_edge(
                    agent_id,
                    task_id,
                    edge_type=EdgeType.TASK_UPDATE,
                    weight=self.config.task_edge_weight,
                )

        return self

    def from_edges(
        self,
        edges: Sequence[tuple[str, str]],
        weight: float | None = None,
    ) -> "GraphBuilder":
        """Добавить набор workflow-рёбер из списка пар (source, target)."""
        for source, target in edges:
            self.add_workflow_edge(source, target, weight=weight)
        return self

    def validate(self) -> ValidationResult:
        """Проверить схему на корректность и вернуть результат валидации."""
        return self._validator.validate(self._schema)

    def build(self) -> "RoleGraph":
        """Собрать `RoleGraph` из текущей схемы, опционально провалидировав её."""
        if self.config.validate:
            result = self.validate()
            if not result.valid:
                msg = f"Schema validation failed: {result.errors}"
                raise ValueError(msg)

        return build_from_schema(
            self._schema,
            edge_conditions=self._edge_conditions if self._edge_conditions else None,
            start_node=self._start_node,
            end_node=self._end_node,
        )

    @property
    def schema(self) -> GraphSchema:
        """Текущая схема графа (для чтения/расширения вне билдера)."""
        return self._schema


def build_from_schema(
    schema: GraphSchema,
    edge_conditions: dict[tuple[str, str], Callable] | None = None,
    start_node: str | None = None,
    end_node: str | None = None,
) -> "RoleGraph":
    """
    Сконструировать `RoleGraph` из готовой схемы `GraphSchema`.

    Создаёт внутренний `rx.PyDiGraph`, переносит данные узлов/рёбер,
    восстанавливает матрицы смежности и вероятностей, и формирует
    `RoleGraph` с соответствующими объектами агентов и задач.

    LLM конфигурация из AgentNodeSchema автоматически переносится
    в AgentProfile.llm_config для поддержки мультимодельности.

    Args:
        schema: Схема графа.
        edge_conditions: Callable условия для рёбер.
        start_node: ID стартовой ноды для выполнения.
        end_node: ID конечной ноды для выполнения.

    """
    from rustworkx_framework.core.agent import AgentLLMConfig, AgentProfile, TaskNode
    from rustworkx_framework.core.graph import RoleGraph

    graph = rx.PyDiGraph()
    idx_map = {}

    agents = []
    task_node = None
    task_node_id = None

    for node_id, node_schema in schema.nodes.items():
        idx_map[node_id] = graph.add_node(
            {
                "id": node_id,
                "type": node_schema.type.value,
                "schema": node_schema.model_dump(),
            }
        )

        embedding = None
        if node_schema.embedding:
            embedding = torch.tensor(node_schema.embedding, dtype=torch.float32)

        if node_schema.type == NodeType.TASK:
            task_schema = node_schema
            task_node = TaskNode(
                agent_id=node_id,
                query=getattr(task_schema, "query", ""),
                description=getattr(task_schema, "description", ""),
                embedding=embedding,
            )
            task_node_id = node_id
            agents.append(task_node)
        else:
            agent_schema = node_schema

            # Build LLM config from schema if any LLM params are set
            llm_config = None
            if isinstance(agent_schema, AgentNodeSchema) and agent_schema.has_llm_config():
                llm_config = AgentLLMConfig(
                    model_name=agent_schema.llm_backbone,
                    base_url=agent_schema.base_url,
                    api_key=agent_schema.api_key,
                    max_tokens=agent_schema.max_tokens,
                    temperature=agent_schema.temperature,
                    timeout=agent_schema.timeout,
                    top_p=agent_schema.top_p,
                    stop_sequences=agent_schema.stop_sequences,
                )

            agent = AgentProfile(
                agent_id=node_id,
                display_name=node_schema.display_name or node_id,
                persona=getattr(agent_schema, "persona", ""),
                description=getattr(agent_schema, "description", ""),
                llm_backbone=getattr(agent_schema, "llm_backbone", None),
                llm_config=llm_config,
                tools=getattr(agent_schema, "tools", []),
                embedding=embedding,
                input_schema=getattr(agent_schema, "input_schema", None),
                output_schema=getattr(agent_schema, "output_schema", None),
            )
            agents.append(agent)

    for edge_schema in schema.edges:
        if edge_schema.source in idx_map and edge_schema.target in idx_map:
            edge_data = {
                "type": edge_schema.type.value,
                "weight": edge_schema.weight,
                "probability": edge_schema.probability,
                "attr": edge_schema.to_attr_tensor(),
                "schema": edge_schema.model_dump(exclude={"attr"}),
            }
            graph.add_edge(
                idx_map[edge_schema.source],
                idx_map[edge_schema.target],
                edge_data,
            )

    n = graph.num_nodes()
    a_com = torch.zeros((n, n), dtype=torch.float32)
    p_matrix = torch.zeros((n, n), dtype=torch.float32)

    for edge_idx in graph.edge_indices():
        s, t = graph.get_edge_endpoints_by_index(edge_idx)
        d = graph.get_edge_data_by_index(edge_idx)
        a_com[s, t] = d.get("weight", 1.0)
        p_matrix[s, t] = d.get("probability", 1.0)

    connections = {a.agent_id: [] for a in agents}
    edge_condition_names: dict[tuple[str, str], str] = {}

    for edge_schema in schema.edges:
        if edge_schema.source in connections and edge_schema.target not in connections[edge_schema.source]:
            connections[edge_schema.source].append(edge_schema.target)

        # Извлечь строковые условия из WorkflowEdgeSchema
        if hasattr(edge_schema, "condition") and edge_schema.condition and isinstance(edge_schema.condition, str):
            edge_condition_names[(edge_schema.source, edge_schema.target)] = edge_schema.condition

    return RoleGraph(
        agents=agents,
        node_ids=[a.agent_id for a in agents],
        role_connections=connections,
        task_node=task_node_id,
        query=getattr(task_node, "query", "") if task_node else "",
        graph=graph,
        A_com=a_com,
        p_matrix=p_matrix,
        edge_conditions=edge_conditions or {},
        edge_condition_names=edge_condition_names,
        start_node=start_node,
        end_node=end_node,
    )


def default_sequence(roles: Sequence[str], anchor: str) -> list[str]:
    """Сформировать порядок ролей, начиная с anchor (если он присутствует)."""
    ordered = [anchor] if anchor in roles else []
    ordered.extend(role for role in roles if role != anchor)
    return ordered


def default_edges(sequence: Sequence[str]) -> list[tuple[str, str]]:
    """Построить цепочку рёбер из последовательности узлов (s -> t по порядку)."""
    return list(itertools.pairwise(sequence))


def build_property_graph(
    agents: Sequence[Any],
    workflow_edges: Sequence[tuple[str, str]],
    *,
    query: str = "",
    answer: str = "",
    anchor: str | None = None,
    encoder: Any | None = None,
    include_task_node: bool = True,
    config: BuilderConfig | None = None,
    node_features: dict[str, dict[str, Any]] | None = None,
    edge_features: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> "RoleGraph":
    """
    Построить `RoleGraph` с заданной топологией workflow.

    Args:
        agents: Список агентов (`AgentProfile`).
        workflow_edges: Рёбра рабочего процесса [(source, target), ...].
        query: Текст задачи, который кладётся в узел задачи.
        answer: Известный ответ (добавляется в итоговый граф).
        anchor: ID якорного агента — будет первым в порядке и может повысить вес рёбер.
        encoder: `NodeEncoder` для автоматического вычисления эмбеддингов.
        include_task_node: Добавить ли виртуальный узел задачи и связать его с агентами.
        config: Конфигурация построителя (если не передана — создаётся новая).
        node_features: Доп. признаки узлов `{node_id: {feature: value}}`.
        edge_features: Доп. признаки рёбер `{(src, tgt): {feature: value}}`.

    Returns:
        Готовый `RoleGraph` с заполненными матрицами смежности и вероятностей.

    """
    config = config or BuilderConfig(include_task_node=include_task_node)
    builder = GraphBuilder(config)

    if encoder is not None:
        from rustworkx_framework.core.agent import TaskNode

        texts = [a.to_text() for a in agents]
        if include_task_node:
            task_tmpl = TaskNode(query=query)
            texts.append(task_tmpl.to_text())

        embs = encoder.encode(texts)
        emb_list = [embs[i].cpu().tolist() for i in range(len(agents))]
        task_emb = embs[-1].cpu().tolist() if include_task_node else None
    else:
        emb_list = [None] * len(agents)
        task_emb = None

    if anchor is None and agents:
        anchor = agents[0].agent_id

    for i, agent in enumerate(agents):
        extra = (node_features or {}).get(agent.agent_id, {})
        trust = 1.0 + (0.1 if agent.agent_id == anchor else 0)

        builder.add_agent(
            agent_id=agent.agent_id,
            display_name=agent.display_name,
            persona=getattr(agent, "persona", ""),
            description=getattr(agent, "description", ""),
            embedding=emb_list[i],
            trust_score=min(trust, 1.0),
            llm_backbone=getattr(agent, "llm_backbone", None),
            tools=getattr(agent, "tools", []),
            **extra,
        )

    if include_task_node:
        task_extra = (node_features or {}).get(_TASK_NODE_ID, {})
        builder.add_task(
            task_id=_TASK_NODE_ID,
            query=query,
            embedding=task_emb,
            **task_extra,
        )

    pos = {a.agent_id: i for i, a in enumerate(agents)}

    for s, t in workflow_edges:
        if s not in pos or t not in pos:
            continue

        w = 1.0
        if anchor and s == anchor:
            w += 0.1
        if pos.get(s, 0) <= pos.get(t, 0):
            w += 0.05

        extra = (edge_features or {}).get((s, t), {})

        builder.add_workflow_edge(
            source=s,
            target=t,
            weight=round(w, 3),
            **extra,
        )

    if include_task_node:
        builder.connect_task_to_agents(_TASK_NODE_ID, bidirectional=True)

    rg = builder.build()

    if answer:
        object.__setattr__(rg, "answer", answer)

    return rg


def build_from_adjacency(
    agents: Sequence[Any],
    adjacency: torch.Tensor,
    *,
    query: str = "",
    answer: str = "",
    threshold: float = 0.5,
    config: BuilderConfig | None = None,
) -> "RoleGraph":
    """
    Построить `RoleGraph`, исходя из матрицы смежности.

    Args:
        agents: Список агентов (`AgentProfile`).
        adjacency: Квадратная матрица весов/связности (torch.Tensor).
        query: Текст задачи.
        answer: Известный ответ.
        threshold: Порог, выше которого связь считается существующей.
        config: Конфигурация построителя (узел задачи отключён).

    """
    n = len(agents)
    if tuple(adjacency.shape) != (n, n):
        msg = f"Adjacency shape {tuple(adjacency.shape)} doesn't match {n} agents"
        raise ValueError(msg)

    edges = [
        (agents[i].agent_id, agents[j].agent_id)
        for i in range(n)
        for j in range(n)
        if i != j and adjacency[i, j] > threshold
    ]

    config = config or BuilderConfig(include_task_node=False)
    config.include_task_node = False

    return build_property_graph(
        agents,
        edges,
        query=query,
        answer=answer,
        config=config,
    )
