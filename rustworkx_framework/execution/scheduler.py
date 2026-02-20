"""
Планировщик выполнения агентов по топологии графа.

Поддерживает как простой топологический порядок, так и адаптивные политики
с учётом весов, pruning и перепланировкой.

Также поддерживает условную маршрутизацию (conditional routing) —
аналог conditional edges в LangGraph.
"""

import heapq

from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple

import rustworkx as rx
import torch
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AdaptiveScheduler",
    # Conditional routing
    "ConditionContext",
    "ConditionEvaluator",
    "EdgeCondition",
    "ExecutionPlan",
    "ExecutionStep",
    "PruningConfig",
    "RoutingPolicy",
    "StepResult",
    "build_execution_order",
    "extract_agent_adjacency",
    "filter_reachable_agents",
    "get_incoming_agents",
    "get_outgoing_agents",
    "get_parallel_groups",
]


# =============================================================================
# CONDITIONAL ROUTING (аналог LangGraph conditional edges)
# =============================================================================


class ConditionContext(BaseModel):
    """
    Контекст для вычисления условий маршрутизации.

    Передаётся в condition-функции для принятия решений о маршруте.

    Attributes:
        source_agent: ID агента-источника ребра.
        target_agent: ID агента-приёмника ребра.
        messages: Словарь ответов агентов {agent_id: response}.
        step_results: Результаты выполнения шагов {agent_id: StepResult}.
        state: Произвольное пользовательское состояние.
        query: Текущий запрос/задача.
        metadata: Дополнительные метаданные.

    Example:
        def my_condition(ctx: ConditionContext) -> bool:
            # Переходим к reviewer только если solver дал ответ
            if "solver" in ctx.messages:
                return "error" not in ctx.messages["solver"].lower()
            return False

        builder.add_conditional_edge("solver", "reviewer", condition=my_condition)

    """

    source_agent: str
    target_agent: str
    messages: dict[str, str] = Field(default_factory=dict)
    step_results: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    query: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_last_response(self) -> str | None:
        """Получить последний ответ от source_agent."""
        return self.messages.get(self.source_agent)

    def source_succeeded(self) -> bool:
        """Успешно ли выполнился source_agent."""
        result = self.step_results.get(self.source_agent)
        if result is None:
            return self.source_agent in self.messages
        return getattr(result, "success", True)

    def has_keyword(self, keyword: str, *, in_source: bool = True) -> bool:
        """Проверить наличие ключевого слова в ответе."""
        agent = self.source_agent if in_source else self.target_agent
        msg = self.messages.get(agent, "")
        return keyword.lower() in msg.lower()

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Получить значение из state."""
        return self.state.get(key, default)


# Type alias for condition functions
EdgeCondition = Callable[[ConditionContext], bool]


class ConditionEvaluator:
    """
    Evaluator для условий на рёбрах графа.

    Поддерживает:
    - Callable conditions (функции)
    - String conditions (простые выражения)
    - Composition (AND/OR)

    Example:
        evaluator = ConditionEvaluator()

        # Зарегистрировать именованные условия
        evaluator.register("has_error", lambda ctx: "error" in ctx.get_last_response() or "")
        evaluator.register("high_quality", lambda ctx: ctx.step_results.get(ctx.source_agent, {}).quality_score > 0.8)

        # Использовать по имени
        if evaluator.evaluate("has_error", context):
            ...

    """

    def __init__(self) -> None:
        self._named_conditions: dict[str, EdgeCondition] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Регистрация встроенных условий."""
        # Всегда True/False
        self._named_conditions["always"] = lambda _ctx: True
        self._named_conditions["never"] = lambda _ctx: False

        # Проверка успешности источника
        self._named_conditions["source_success"] = lambda ctx: ctx.source_succeeded()
        self._named_conditions["source_failed"] = lambda ctx: not ctx.source_succeeded()

        # Проверка наличия ответа
        self._named_conditions["has_response"] = lambda ctx: ctx.get_last_response() is not None

    def register(self, name: str, condition: EdgeCondition) -> None:
        """Зарегистрировать именованное условие."""
        self._named_conditions[name] = condition

    def unregister(self, name: str) -> bool:
        """Удалить именованное условие."""
        if name in self._named_conditions:
            del self._named_conditions[name]
            return True
        return False

    def get(self, name: str) -> EdgeCondition | None:
        """Получить условие по имени."""
        return self._named_conditions.get(name)

    def evaluate(
        self,
        condition: EdgeCondition | str | None,
        context: ConditionContext,
    ) -> bool:
        """
        Вычислить условие.

        Args:
            condition: Callable, имя условия или None (= always True).
            context: Контекст для вычисления.

        Returns:
            True если условие выполнено, False иначе.

        """
        if condition is None:
            return True

        if callable(condition):
            try:
                return bool(condition(context))
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                # При ошибке условие считается невыполненным
                return False

        if isinstance(condition, str):
            # Попытка найти именованное условие
            named = self._named_conditions.get(condition)
            if named is not None:
                return self.evaluate(named, context)

            # Простые строковые выражения
            return self._evaluate_string_condition(condition, context)

        return True

    def _evaluate_string_condition(self, expr: str, context: ConditionContext) -> bool:
        """
        Вычислить простое строковое условие.

        Поддерживает:
        - "contains:keyword" — проверка наличия слова в ответе
        - "state:key=value" — проверка значения в state
        - "not:condition" — отрицание
        """
        expr = expr.strip()

        # Отрицание
        if expr.startswith("not:"):
            inner = expr[4:]
            return not self._evaluate_string_condition(inner, context)

        # Проверка содержимого
        if expr.startswith("contains:"):
            keyword = expr[9:]
            return context.has_keyword(keyword)

        # Проверка state
        if expr.startswith("state:"):
            kv = expr[6:]
            if "=" in kv:
                key, value = kv.split("=", 1)
                return str(context.get_state_value(key.strip())) == value.strip()
            return context.get_state_value(kv.strip()) is not None

        # По умолчанию — поиск по имени
        named = self._named_conditions.get(expr)
        if named is not None:
            return self.evaluate(named, context)

        return True

    def compose_and(self, *conditions: EdgeCondition | str) -> EdgeCondition:
        """Создать условие AND из нескольких."""

        def composed(ctx: ConditionContext) -> bool:
            return all(self.evaluate(c, ctx) for c in conditions)

        return composed

    def compose_or(self, *conditions: EdgeCondition | str) -> EdgeCondition:
        """Создать условие OR из нескольких."""

        def composed(ctx: ConditionContext) -> bool:
            return any(self.evaluate(c, ctx) for c in conditions)

        return composed


# Global evaluator instance (can be replaced)
_default_evaluator = ConditionEvaluator()


class RoutingPolicy(str, Enum):
    TOPOLOGICAL = "topological"
    WEIGHTED_TOPO = "weighted_topo"
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    K_SHORTEST = "k_shortest"


class PruningConfig(BaseModel):
    """Параметры отсечения (pruning) и fallback для планировщика."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    min_weight_threshold: float = 0.1
    min_probability_threshold: float = 0.05
    max_consecutive_errors: int = 3
    skip_on_predecessor_failure: bool = True
    token_budget: int | None = None
    quality_scorer: Callable[[str], float] | None = None
    min_quality_threshold: float = 0.3
    enable_fallback: bool = True
    max_fallback_attempts: int = 2


class StepResult(NamedTuple):
    """Результат выполнения шага для планировщика."""

    agent_id: str
    success: bool
    response: str | None = None
    tokens_used: int = 0
    quality_score: float = 1.0
    error: str | None = None
    fallback_used: bool = False


class ExecutionStep(BaseModel):
    """Шаг плана: агент, предшественники и метаданные веса/вероятности."""

    agent_id: str
    predecessors: list[str]
    weight: float = 1.0
    probability: float = 1.0
    fallback_agents: list[str] = Field(default_factory=list)
    is_optional: bool = False
    priority: int = 0


class ExecutionPlan(BaseModel):
    """
    Последовательность шагов с состояниями выполнения и токенами.

    Поддерживает условные циклы: агенты могут выполняться повторно
    при срабатывании условных рёбер (например, review_failed -> mathematician).
    """

    steps: list[ExecutionStep] = Field(default_factory=list)
    completed: set[str] = Field(default_factory=set)
    failed: set[str] = Field(default_factory=set)
    skipped: set[str] = Field(default_factory=set)
    tokens_used: int = 0
    current_index: int = 0

    # Поддержка условных циклов
    iteration_count: dict[str, int] = Field(default_factory=dict)
    max_iterations: int = Field(default=5)  # Защита от бесконечных циклов
    end_agent: str | None = Field(default=None)  # Конечный агент (для завершения)

    # Агенты, пропущенные из-за невыполненных условий
    condition_skipped: set[str] = Field(default_factory=set)

    @property
    def remaining_steps(self) -> list[ExecutionStep]:
        """Шаги, ещё не выполненные и не пропущенные, начиная с current_index."""
        return [
            step
            for step in self.steps[self.current_index :]
            if step.agent_id not in self.skipped and step.agent_id not in self.condition_skipped
        ]

    @property
    def is_complete(self) -> bool:
        """True, если дошли до конца списка шагов."""
        return self.current_index >= len(self.steps)

    @property
    def execution_order(self) -> list[str]:
        """Текущий порядок агентов в плане."""
        return [step.agent_id for step in self.steps]

    def mark_completed(self, agent_id: str, tokens: int = 0) -> None:
        """
        Пометить шаг выполненным и прибавить токены.

        Добавляет в completed set и инкрементирует счётчик итераций.
        Циклы обрабатываются через insert_conditional_step, который добавляет
        новый шаг в конец плана — completed set не мешает повторному выполнению.
        """
        self.completed.add(agent_id)
        self.iteration_count[agent_id] = self.iteration_count.get(agent_id, 0) + 1
        self.tokens_used += tokens
        self.advance()

    def mark_failed(self, agent_id: str) -> None:
        """Пометить шаг проваленным."""
        self.failed.add(agent_id)
        self.advance()

    def mark_skipped(self, agent_id: str) -> None:
        """Пометить шаг пропущенным."""
        self.skipped.add(agent_id)
        self.advance()

    def advance(self) -> None:
        """Сдвинуть current_index на следующий шаг."""
        self.current_index += 1

    def get_current_step(self) -> ExecutionStep | None:
        """Текущий шаг или None, если план завершён."""
        if self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    def insert_fallback(self, fallback_agent_id: str, after_index: int) -> None:
        """Вставить fallback-агента после указанного индекса."""
        if fallback_agent_id in self.skipped:
            return

        fallback_step = ExecutionStep(
            agent_id=fallback_agent_id,
            predecessors=[],
            is_optional=True,
            priority=-1,
        )
        self.steps.insert(after_index + 1, fallback_step)

    def insert_conditional_step(
        self,
        agent_id: str,
        predecessors: list[str] | None = None,
    ) -> bool:
        """
        Вставить агента для условного перехода (цикл).

        Args:
            agent_id: ID агента для добавления.
            predecessors: Предшественники (обычно текущий завершённый агент).

        Returns:
            True если шаг добавлен, False если превышен лимит итераций.

        """
        current_iterations = self.iteration_count.get(agent_id, 0)
        if current_iterations >= self.max_iterations:
            return False

        if agent_id in self.skipped:
            return False

        step = ExecutionStep(
            agent_id=agent_id,
            predecessors=predecessors or [],
            is_optional=False,
            priority=0,
        )
        self.steps.append(step)
        return True

    def can_iterate(self, agent_id: str) -> bool:
        """Проверить, можно ли ещё раз выполнить агента."""
        return self.iteration_count.get(agent_id, 0) < self.max_iterations


def extract_agent_adjacency(
    a_com: torch.Tensor,
    task_idx: int,
) -> torch.Tensor:
    """Убрать строку/столбец задачи из матрицы смежности агентов."""
    n_nodes = a_com.shape[0]
    mask = torch.ones(n_nodes, dtype=torch.bool)
    mask[task_idx] = False
    return a_com[mask][:, mask]


def get_incoming_agents(
    agent_id: str,
    a_agents: torch.Tensor,
    agent_ids: list[str],
    threshold: float = 0.5,
) -> list[str]:
    """ID предшественников агента по матрице > threshold."""
    if agent_id not in agent_ids:
        return []

    agent_idx = agent_ids.index(agent_id)
    incoming: list[str] = []

    for i, aid in enumerate(agent_ids):
        if a_agents[i, agent_idx].item() > threshold:
            incoming.append(aid)

    return incoming


def get_outgoing_agents(
    agent_id: str,
    a_agents: torch.Tensor,
    agent_ids: list[str],
    threshold: float = 0.5,
) -> list[str]:
    """ID последователей агента по матрице > threshold."""
    if agent_id not in agent_ids:
        return []

    agent_idx = agent_ids.index(agent_id)
    outgoing: list[str] = []

    for j, aid in enumerate(agent_ids):
        if a_agents[agent_idx, j].item() > threshold:
            outgoing.append(aid)

    return outgoing


def filter_reachable_agents(
    a_agents: torch.Tensor,
    agent_ids: list[str],
    start_agent: str | None = None,
    end_agent: str | None = None,
    threshold: float = 0.5,
) -> tuple[list[str], list[str]]:
    """
    Отфильтровать агентов, оставив только тех, кто на пути от start к end.

    Это ключевая функция для оптимизации: исключает изолированные ноды и субграфы,
    которые не влияют на результат, тем самым экономя токены и вызовы LLM.

    Args:
        a_agents: Матрица смежности агентов.
        agent_ids: Список ID всех агентов.
        start_agent: ID стартового агента (None = первый без входящих рёбер).
        end_agent: ID конечного агента (None = последний без исходящих рёбер).
        threshold: Минимальный вес ребра.

    Returns:
        Tuple из:
        - Список релевантных agent_ids (на пути start->end)
        - Список исключённых agent_ids (изолированные)

    Example:
        relevant, excluded = filter_reachable_agents(
            a_agents, agent_ids,
            start_agent="input",
            end_agent="output"
        )
        # relevant содержит только агентов на пути input->output
        # excluded содержит агентов, которые не нужны для получения результата

    """
    num_agents = len(agent_ids)
    if num_agents == 0:
        return [], []

    # Определить стартовый агент
    effective_start = start_agent
    if effective_start is None:
        # Первый агент без входящих рёбер
        in_degree = torch.sum((a_agents > threshold).int(), dim=0)
        for i, aid in enumerate(agent_ids):
            if in_degree[i].item() == 0:
                effective_start = aid
                break
        if effective_start is None:
            effective_start = agent_ids[0]

    # Определить конечный агент
    effective_end = end_agent
    if effective_end is None:
        # Последний агент без исходящих рёбер
        out_degree = torch.sum((a_agents > threshold).int(), dim=1)
        for i in range(num_agents - 1, -1, -1):
            if out_degree[i].item() == 0:
                effective_end = agent_ids[i]
                break
        if effective_end is None:
            effective_end = agent_ids[-1]

    # BFS вперёд от start — найти все достижимые из start
    reachable_from_start: set[str] = set()
    if effective_start in agent_ids:
        queue = [effective_start]
        reachable_from_start.add(effective_start)
        while queue:
            current = queue.pop(0)
            current_idx = agent_ids.index(current)
            for j, aid in enumerate(agent_ids):
                if aid not in reachable_from_start and a_agents[current_idx, j].item() > threshold:
                    reachable_from_start.add(aid)
                    queue.append(aid)

    # BFS назад от end — найти все ноды, из которых достижим end
    reaching_end: set[str] = set()
    if effective_end in agent_ids:
        queue = [effective_end]
        reaching_end.add(effective_end)
        while queue:
            current = queue.pop(0)
            current_idx = agent_ids.index(current)
            for i, aid in enumerate(agent_ids):
                if aid not in reaching_end and a_agents[i, current_idx].item() > threshold:
                    reaching_end.add(aid)
                    queue.append(aid)

    # Пересечение — ноды на путях от start к end
    relevant_set = reachable_from_start & reaching_end
    relevant = [aid for aid in agent_ids if aid in relevant_set]
    excluded = [aid for aid in agent_ids if aid not in relevant_set]

    return relevant, excluded


def build_execution_order(
    a_agents: torch.Tensor,
    agent_ids: list[str],
    fallback_order: list[str] | None = None,
    threshold: float = 0.5,
    start_agent: str | None = None,
) -> list[str]:
    """
    Построить порядок выполнения: topo, SCC + fallback сортировка.

    Args:
        a_agents: Матрица весов связей.
        agent_ids: Список ID агентов.
        fallback_order: Порядок для сортировки внутри SCC.
        threshold: Порог веса для включения ребра.
        start_agent: Стартовый агент (будет первым в результате).

    """
    num_agents = a_agents.shape[0]
    if num_agents != len(agent_ids):
        msg = f"a_agents size {num_agents} != agent_ids length {len(agent_ids)}"
        raise ValueError(msg)

    if num_agents == 0:
        return []

    graph = rx.PyDiGraph()
    node_indices = [graph.add_node(aid) for aid in agent_ids]

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j and a_agents[i, j] > threshold:
                graph.add_edge(node_indices[i], node_indices[j], None)

    try:
        topo_order = rx.topological_sort(graph)
        return [agent_ids[node_indices.index(idx)] for idx in topo_order]
    except rx.DAGHasCycle:
        pass

    sccs = list(rx.strongly_connected_components(graph))

    scc_map: dict[int, int] = {}
    for scc_idx, scc in enumerate(sccs):
        for node_idx in scc:
            scc_map[node_idx] = scc_idx

    scc_graph = rx.PyDiGraph()
    scc_nodes = [scc_graph.add_node(i) for i in range(len(sccs))]
    scc_edges_seen: set[tuple[int, int]] = set()

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j and a_agents[i, j] > threshold:
                src_scc = scc_map[node_indices[i]]
                dst_scc = scc_map[node_indices[j]]
                if src_scc != dst_scc and (src_scc, dst_scc) not in scc_edges_seen:
                    scc_graph.add_edge(scc_nodes[src_scc], scc_nodes[dst_scc], None)
                    scc_edges_seen.add((src_scc, dst_scc))

    try:
        scc_order = rx.topological_sort(scc_graph)
    except rx.DAGHasCycle:
        scc_order = list(range(len(sccs)))

    fallback = fallback_order or agent_ids
    fallback_rank = {aid: i for i, aid in enumerate(fallback)}

    result: list[str] = []
    for scc_idx in scc_order:
        scc = sccs[scc_idx]
        scc_agents: list[str] = []
        for node_idx in scc:
            agent_idx = node_indices.index(node_idx)
            scc_agents.append(agent_ids[agent_idx])

        # Сортируем по fallback_rank, но start_agent всегда первый в своём SCC
        def sort_key(a: str) -> tuple[int, int]:
            is_start = 0 if a == start_agent else 1
            return (is_start, fallback_rank.get(a, len(fallback)))

        scc_agents.sort(key=sort_key)
        result.extend(scc_agents)

    return result


def get_parallel_groups(
    a_agents: torch.Tensor,
    agent_ids: list[str],
    threshold: float = 0.5,
) -> list[list[str]]:
    """Разбить на группы узлов, которые можно выполнять параллельно."""
    num_agents = a_agents.shape[0]
    if num_agents == 0:
        return []

    in_degree = torch.sum((a_agents > threshold).int(), dim=0)
    remaining_in = in_degree.clone()
    executed = torch.zeros(num_agents, dtype=torch.bool)
    groups: list[list[str]] = []

    while not torch.all(executed):
        ready: list[str] = []
        ready = [agent_ids[i] for i in range(num_agents) if not executed[i] and remaining_in[i] == 0]

        if not ready:
            for i in range(num_agents):
                if not executed[i]:
                    ready.append(agent_ids[i])
                    break

        groups.append(ready)

        for aid in ready:
            i = agent_ids.index(aid)
            executed[i] = True
            for j in range(num_agents):
                if a_agents[i, j].item() > threshold:
                    remaining_in[j] = max(0, remaining_in[j] - 1)

    return groups


class AdaptiveScheduler:
    """
    Планировщик с разными политиками маршрутизации.

    Поддерживает условную маршрутизацию (conditional routing) через
    edge_conditions — словарь условий для каждого ребра.
    Оценка условий выполняется в runtime через topology pipeline в MACPRunner.

    Example:
        scheduler = AdaptiveScheduler(policy=RoutingPolicy.TOPOLOGICAL)

        # Условия на рёбра
        conditions = {
            ("solver", "reviewer"): lambda ctx: "error" not in ctx.messages.get("solver", ""),
            ("reviewer", "finalizer"): "source_success",  # встроенное условие
        }

        plan = scheduler.build_plan(
            a_agents, agent_ids, p_matrix,
            edge_conditions=conditions,
            condition_context=ConditionContext(...)
        )

    """

    def __init__(
        self,
        policy: RoutingPolicy = RoutingPolicy.TOPOLOGICAL,
        pruning_config: PruningConfig | None = None,
        beam_width: int = 3,
        k_paths: int = 3,
        condition_evaluator: ConditionEvaluator | None = None,
    ):
        self.policy = policy
        self.pruning = pruning_config or PruningConfig()
        self.beam_width = beam_width
        self.k_paths = k_paths
        self.condition_evaluator = condition_evaluator or _default_evaluator
        self._last_edge_conditions: dict[tuple[str, str], EdgeCondition | str] = {}

    def build_plan(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        p_matrix: torch.Tensor | None = None,
        start_agent: str | None = None,
        end_agent: str | None = None,
        edge_conditions: dict[tuple[str, str], EdgeCondition | str] | None = None,
        condition_context: ConditionContext | None = None,
        *,
        filter_unreachable: bool = True,
    ) -> ExecutionPlan:
        """
        Собрать ExecutionPlan согласно политике маршрутизации.

        Args:
            a_agents: Матрица весов связей между агентами.
            agent_ids: Список ID агентов.
            p_matrix: Матрица вероятностей (опционально).
            start_agent: Стартовый агент.
            end_agent: Конечный агент.
            edge_conditions: Словарь условий {(source, target): condition}.
            condition_context: Контекст для вычисления условий.
            filter_unreachable: Исключить ли изолированные ноды из плана.
                               Экономит токены, исключая ноды не на пути start->end.

        Returns:
            ExecutionPlan с шагами выполнения.

        """
        del condition_context  # Reserved for future conditional logic

        if a_agents.size == 0 or not agent_ids:
            return ExecutionPlan()

        # Сохраняем edge_conditions для runtime-оценки в topology pipeline
        if edge_conditions:
            self._last_edge_conditions = edge_conditions
        else:
            self._last_edge_conditions = {}

        # Все агенты включаются в начальный план по топологическому порядку.
        # Условные переходы обрабатываются в runtime через topology pipeline.
        effective_a = a_agents.clone()

        # Отфильтровать изолированные ноды для оптимизации
        effective_agent_ids = agent_ids
        effective_p = p_matrix
        excluded_agents: list[str] = []

        if filter_unreachable and (start_agent is not None or end_agent is not None):
            relevant, excluded_agents = filter_reachable_agents(
                effective_a, agent_ids, start_agent, end_agent, self.pruning.min_weight_threshold
            )

            if relevant and len(relevant) < len(agent_ids):
                # Построить подматрицы только для релевантных агентов
                indices = [agent_ids.index(aid) for aid in relevant]
                indices_t = torch.tensor(indices, dtype=torch.long)
                effective_a = effective_a[indices_t][:, indices_t]
                effective_agent_ids = relevant

                if p_matrix is not None:
                    effective_p = p_matrix[indices_t][:, indices_t]

        # Определить start/end в рамках отфильтрованного списка
        effective_start = start_agent if start_agent in effective_agent_ids else None
        effective_end = end_agent if end_agent in effective_agent_ids else None

        if self.policy == RoutingPolicy.GREEDY:
            order = self._greedy_order(effective_a, effective_agent_ids, effective_p, effective_start, effective_end)
        elif self.policy == RoutingPolicy.BEAM_SEARCH:
            order = self._beam_search_order(
                effective_a, effective_agent_ids, effective_p, effective_start, effective_end
            )
        elif self.policy == RoutingPolicy.K_SHORTEST:
            order = self._k_shortest_order(
                effective_a, effective_agent_ids, effective_p, effective_start, effective_end
            )
        elif self.policy == RoutingPolicy.WEIGHTED_TOPO:
            order = self._weighted_topological_order(
                effective_a, effective_agent_ids, effective_p, effective_start, effective_end
            )
        else:
            order = build_execution_order(effective_a, effective_agent_ids, start_agent=effective_start)

        steps: list[ExecutionStep] = []
        # Множество агентов, которые уже добавлены в план (идут раньше в order)
        agents_before: set[str] = set()

        for agent_id in order:
            idx = effective_agent_ids.index(agent_id)
            all_incoming = get_incoming_agents(
                agent_id, effective_a, effective_agent_ids, self.pruning.min_weight_threshold
            )
            # Для циклических графов: predecessors только из тех, кто идёт РАНЬШЕ в order.
            # Это предотвращает deadlock, когда агент ждёт агента который идёт после него.
            predecessors = [p for p in all_incoming if p in agents_before]
            agents_before.add(agent_id)

            weight = self._compute_step_weight(idx, predecessors, effective_a, effective_agent_ids)
            prob = (
                self._compute_step_probability(idx, predecessors, effective_p, effective_agent_ids)
                if effective_p is not None
                else 1.0
            )
            fallbacks = self._find_fallback_agents(agent_id, effective_agent_ids, effective_a, order)

            step = ExecutionStep(
                agent_id=agent_id,
                predecessors=predecessors,
                weight=weight,
                probability=prob,
                fallback_agents=fallbacks,
            )
            steps.append(step)

        plan = ExecutionPlan(steps=steps, end_agent=end_agent)

        # Пометить исключённые агенты как skipped
        for excluded in excluded_agents:
            plan.skipped.add(excluded)

        return plan

    def _apply_conditions(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        edge_conditions: dict[tuple[str, str], EdgeCondition | str],
        context: ConditionContext | None,
    ) -> torch.Tensor:
        """
        Применить условия к матрице весов, обнуляя рёбра с невыполненными условиями.

        Args:
            a_agents: Матрица весов (будет изменена).
            agent_ids: Список ID агентов.
            edge_conditions: Словарь условий.
            context: Базовый контекст (source/target будут подставлены).

        Returns:
            Модифицированная матрица с обнулёнными рёбрами.

        """
        for (source, target), condition in edge_conditions.items():
            if source not in agent_ids or target not in agent_ids:
                continue

            src_idx = agent_ids.index(source)
            tgt_idx = agent_ids.index(target)

            # Если вес уже 0, пропускаем
            if a_agents[src_idx, tgt_idx] == 0:
                continue

            # Создать контекст для этого ребра
            edge_ctx = ConditionContext(
                source_agent=source,
                target_agent=target,
                messages=context.messages if context else {},
                step_results=context.step_results if context else {},
                state=context.state if context else {},
                query=context.query if context else "",
                metadata=context.metadata if context else {},
            )

            # Вычислить условие
            if not self.condition_evaluator.evaluate(condition, edge_ctx):
                a_agents[src_idx, tgt_idx] = 0.0

        return a_agents

    def evaluate_edge_condition(
        self,
        source: str,
        target: str,
        condition: EdgeCondition | str | None,
        context: ConditionContext,
    ) -> bool:
        """
        Вычислить условие для конкретного ребра.

        Удобный метод для проверки одного условия.
        """
        if condition is None:
            return True

        edge_ctx = ConditionContext(
            source_agent=source,
            target_agent=target,
            messages=context.messages,
            step_results=context.step_results,
            state=context.state,
            query=context.query,
            metadata=context.metadata,
        )
        return self.condition_evaluator.evaluate(condition, edge_ctx)

    def should_prune(
        self,
        step: ExecutionStep,
        plan: ExecutionPlan,
        last_result: StepResult | None = None,
    ) -> tuple[bool, str]:
        """Решить, нужно ли отсечь шаг по весу/вероятности/ошибкам/бюджету."""
        del last_result  # Reserved for future quality-based pruning
        if step.weight < self.pruning.min_weight_threshold:
            return True, f"weight {step.weight:.3f} < threshold {self.pruning.min_weight_threshold}"

        if step.probability < self.pruning.min_probability_threshold:
            return (
                True,
                f"probability {step.probability:.3f} < threshold {self.pruning.min_probability_threshold}",
            )

        if self.pruning.token_budget is not None and plan.tokens_used >= self.pruning.token_budget:
            return (
                True,
                f"token budget exhausted ({plan.tokens_used}/{self.pruning.token_budget})",
            )

        consecutive_errors = self._count_consecutive_errors(plan)
        if consecutive_errors >= self.pruning.max_consecutive_errors:
            return True, f"too many consecutive errors ({consecutive_errors})"

        if self.pruning.skip_on_predecessor_failure:
            failed_predecessors = [p for p in step.predecessors if p in plan.failed]
            has_no_fallback = not step.fallback_agents or not self.pruning.enable_fallback
            if failed_predecessors and not step.is_optional and has_no_fallback:
                return True, f"predecessors failed: {failed_predecessors}"

        return False, ""

    def should_use_fallback(
        self,
        step: ExecutionStep,
        result: StepResult,
        fallback_attempts: int,
    ) -> bool:
        """Нужно ли активировать fallback-агента для шага."""
        if not self.pruning.enable_fallback:
            return False
        if fallback_attempts >= self.pruning.max_fallback_attempts:
            return False
        if not step.fallback_agents:
            return False
        if not result.success:
            return True
        return result.quality_score < self.pruning.min_quality_threshold


    def _weighted_topological_order(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        p_matrix: torch.Tensor | None = None,
        start_agent: str | None = None,
        end_agent: str | None = None,
    ) -> list[str]:
        """
        Сортировка с приоритетами по сумме весов исходящих/входящих.

        При наличии циклов использует build_execution_order для корректной
        обработки SCC (strongly connected components) и start_agent.
        """
        del p_matrix, end_agent  # Reserved for future probability-based and end-agent routing

        num_agents = len(agent_ids)
        if num_agents == 0:
            return []

        graph = rx.PyDiGraph()
        node_indices = [graph.add_node(aid) for aid in agent_ids]

        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and a_agents[i, j].item() > self.pruning.min_weight_threshold:
                    graph.add_edge(node_indices[i], node_indices[j], a_agents[i, j].item())

        try:
            topo_order = rx.topological_sort(graph)
            result = [agent_ids[node_indices.index(idx)] for idx in topo_order]
            # Если есть start_agent, убедимся что он первый
            if start_agent and start_agent in result:
                result.remove(start_agent)
                result.insert(0, start_agent)
        except rx.DAGHasCycle:
            # Граф содержит циклы — используем SCC-based алгоритм
            result = build_execution_order(
                a_agents,
                agent_ids,
                start_agent=start_agent,
                threshold=self.pruning.min_weight_threshold,
            )

        return result

    def _greedy_order(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        p_matrix: torch.Tensor | None = None,
        start_agent: str | None = None,
        end_agent: str | None = None,
    ) -> list[str]:
        """Жадный выбор следующего узла по сумме исходящих весов."""
        num_agents = len(agent_ids)
        if num_agents == 0:
            return []

        combined = a_agents * p_matrix if p_matrix is not None else a_agents
        visited: set[int] = set()
        order: list[str] = []

        if start_agent and start_agent in agent_ids:
            current_set: set[int] = {agent_ids.index(start_agent)}
        else:
            in_degree = torch.sum((a_agents > self.pruning.min_weight_threshold).int(), dim=0)
            current_set = set(torch.where(in_degree == 0)[0].tolist())
            if not current_set:
                current_set = {0}

        while len(visited) < num_agents:
            best_idx: int | None = None
            best_score = float("-inf")

            for idx in current_set:
                if idx in visited:
                    continue
                score = torch.sum(combined[idx, :]).item()
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                for i in range(num_agents):
                    if i not in visited:
                        order.append(agent_ids[i])
                        visited.add(i)
                break

            order.append(agent_ids[best_idx])
            visited.add(best_idx)

            for j in range(num_agents):
                if combined[best_idx, j].item() > self.pruning.min_weight_threshold:
                    current_set.add(j)

            if end_agent and agent_ids[best_idx] == end_agent:
                break

        return order

    def _beam_search_order(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        p_matrix: torch.Tensor | None = None,
        start_agent: str | None = None,
        end_agent: str | None = None,
    ) -> list[str]:
        """Beam-search по путям с максимальным весом."""
        num_agents = len(agent_ids)
        if num_agents == 0:
            return []

        combined = a_agents * p_matrix if p_matrix is not None else a_agents

        if start_agent and start_agent in agent_ids:
            start_indices = [agent_ids.index(start_agent)]
        else:
            in_degree = torch.sum((a_agents > self.pruning.min_weight_threshold).int(), dim=0)
            start_indices = torch.where(in_degree == 0)[0].tolist()
            if not start_indices:
                start_indices = [0]

        beam: list[tuple[float, list[int]]] = [(0.0, [idx]) for idx in start_indices]
        heapq.heapify(beam)

        best_path: list[int] = []
        best_score = float("-inf")

        while beam:
            neg_score, path = heapq.heappop(beam)
            score = -neg_score

            if len(path) == num_agents:
                if score > best_score:
                    best_score = score
                    best_path = path
                continue

            last_idx = path[-1]
            visited = set(path)

            if end_agent and agent_ids[last_idx] == end_agent:
                remaining = [i for i in range(num_agents) if i not in visited]
                full_path = path + remaining
                if score > best_score:
                    best_score = score
                    best_path = full_path
                continue

            candidates: list[tuple[float, list[int]]] = []
            for j in range(num_agents):
                if j not in visited:
                    edge_weight = combined[last_idx, j].item()
                    new_score = score + edge_weight
                    candidates.append((new_score, [*path, j]))

            candidates.sort(key=lambda x: -x[0])
            for new_score, new_path in candidates[: self.beam_width]:
                heapq.heappush(beam, (-new_score, new_path))

            if len(beam) > self.beam_width * num_agents:
                beam = heapq.nsmallest(self.beam_width, beam)
                heapq.heapify(beam)

        if not best_path:
            return agent_ids

        return [agent_ids[idx] for idx in best_path]

    def _k_shortest_order(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        p_matrix: torch.Tensor | None = None,
        start_agent: str | None = None,
        end_agent: str | None = None,
    ) -> list[str]:
        """Порядок на основе кратчайшего пути (по обратным весам) или topo."""
        num_agents = len(agent_ids)
        if num_agents == 0:
            return []

        graph = rx.PyDiGraph()
        node_indices = [graph.add_node(aid) for aid in agent_ids]

        combined = a_agents * p_matrix if p_matrix is not None else a_agents

        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and a_agents[i, j].item() > self.pruning.min_weight_threshold:
                    weight = 1.0 / (combined[i, j].item() + 1e-6)
                    graph.add_edge(node_indices[i], node_indices[j], weight)

        start_idx = agent_ids.index(start_agent) if start_agent and start_agent in agent_ids else 0
        end_idx = agent_ids.index(end_agent) if end_agent and end_agent in agent_ids else num_agents - 1

        try:
            paths = rx.dijkstra_shortest_paths(
                graph,
                node_indices[start_idx],
                weight_fn=lambda e: e,
            )

            if node_indices[end_idx] in paths:
                path_indices = paths[node_indices[end_idx]]
                order = [agent_ids[node_indices.index(idx)] for idx in path_indices]
                [aid for aid in agent_ids if aid not in order]
        except (ValueError, KeyError, IndexError, RuntimeError):
            # If path finding fails, fall back to topological order
            order = []

        if order:
            return order + [aid for aid in agent_ids if aid not in order]
        return self._weighted_topological_order(a_agents, agent_ids, p_matrix)

    def _compute_step_weight(
        self,
        idx: int,
        predecessors: list[str],
        a_agents: torch.Tensor,
        agent_ids: list[str],
    ) -> float:
        """Средний вес входящих рёбер для шага."""
        if not predecessors:
            return 1.0
        weights = torch.tensor([a_agents[agent_ids.index(p), idx].item() for p in predecessors])
        return float(torch.mean(weights).item()) if len(weights) > 0 else 1.0

    def _compute_step_probability(
        self,
        idx: int,
        predecessors: list[str],
        p_matrix: torch.Tensor,
        agent_ids: list[str],
    ) -> float:
        """Произведение вероятностей предшественников для шага."""
        if not predecessors:
            return 1.0
        probs = torch.tensor([p_matrix[agent_ids.index(p), idx].item() for p in predecessors])
        return float(torch.prod(probs).item()) if len(probs) > 0 else 1.0

    def _find_fallback_agents(
        self,
        agent_id: str,
        agent_ids: list[str],
        a_agents: torch.Tensor,
        current_order: list[str],
    ) -> list[str]:
        """Найти похожих агентов по паттерну входящих рёбер для fallback."""
        if not self.pruning.enable_fallback:
            return []

        idx = agent_ids.index(agent_id)
        fallbacks: list[str] = []
        in_pattern = a_agents[:, idx]

        for i, aid in enumerate(agent_ids):
            if aid == agent_id or aid in current_order:
                continue

            other_in = a_agents[:, i]
            dot = torch.dot(in_pattern, other_in).item()
            norm1 = torch.norm(in_pattern).item()
            norm2 = torch.norm(other_in).item()

            if norm1 > 0 and norm2 > 0:
                similarity = dot / (norm1 * norm2)
                # Threshold for considering agents as similar fallback candidates
                similarity_threshold = 0.5
                if similarity > similarity_threshold:
                    fallbacks.append(aid)

        return fallbacks[: self.pruning.max_fallback_attempts]

    def _count_consecutive_errors(self, plan: ExecutionPlan) -> int:
        """Сколько подряд шагов перед текущим закончились ошибкой."""
        count = 0
        for step in reversed(plan.steps[: plan.current_index]):
            if step.agent_id in plan.failed:
                count += 1
            else:
                break
        return count

    def _adjust_weights_from_results(
        self,
        a_agents: torch.Tensor,
        agent_ids: list[str],
        results: dict[str, StepResult],
    ) -> torch.Tensor:
        """Ослабить веса исходящих рёбер неуспешных или низкокачественных агентов."""
        for agent_id, result in results.items():
            if agent_id not in agent_ids:
                continue
            idx = agent_ids.index(agent_id)
            if not result.success:
                a_agents[idx, :] *= 0.5
            elif result.quality_score < 1.0:
                a_agents[idx, :] *= result.quality_score
        return a_agents

    def _adjust_probabilities_from_results(
        self,
        p_matrix: torch.Tensor,
        agent_ids: list[str],
        results: dict[str, StepResult],
    ) -> torch.Tensor:
        """Ослабить вероятности исходящих рёбер для неуспешных/низкокачественных агентов."""
        for agent_id, result in results.items():
            if agent_id not in agent_ids:
                continue
            idx = agent_ids.index(agent_id)
            if not result.success:
                p_matrix[idx, :] *= 0.3
            elif result.quality_score < 1.0:
                p_matrix[idx, :] *= result.quality_score
        return p_matrix
