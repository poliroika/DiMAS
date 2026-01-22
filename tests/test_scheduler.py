"""Тесты для execution/scheduler.py — планировщик выполнения."""

import pytest
import torch

from rustworkx_framework.execution.scheduler import (
    AdaptiveScheduler,
    ExecutionPlan,
    ExecutionStep,
    PruningConfig,
    RoutingPolicy,
    StepResult,
    build_execution_order,
    extract_agent_adjacency,
    get_incoming_agents,
    get_outgoing_agents,
    get_parallel_groups,
)


def create_adjacency_matrix(nodes, edges, weights=None):
    """Создать матрицу смежности."""
    n = len(nodes)
    adjacency = torch.zeros((n, n), dtype=torch.float32)

    for i, (src, tgt) in enumerate(edges):
        w = weights[i] if weights and i < len(weights) else 1.0
        src_idx = nodes.index(src)
        tgt_idx = nodes.index(tgt)
        adjacency[src_idx, tgt_idx] = w

    return adjacency


class TestBuildExecutionOrder:
    """Тесты топологической сортировки."""

    def test_linear_graph(self):
        """Линейный граф: a -> b -> c."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c")],
        )

        order = build_execution_order(a_agents, nodes)

        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_graph(self):
        """Ромбовидный граф: a -> b, c -> d."""
        nodes = ["a", "b", "c", "d"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )

        order = build_execution_order(a_agents, nodes)

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cyclic_graph_fallback(self):
        """Граф с циклом использует SCC fallback."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c"), ("c", "a")],
        )

        order = build_execution_order(a_agents, nodes)

        # Should return some order despite cycle
        assert len(order) == 3
        assert set(order) == {"a", "b", "c"}

    def test_disconnected_components(self):
        """Граф с несвязными компонентами."""
        nodes = ["a", "b", "x", "y"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("x", "y")],
        )

        order = build_execution_order(a_agents, nodes)

        assert len(order) == 4
        assert order.index("a") < order.index("b")
        assert order.index("x") < order.index("y")


class TestExtractAgentAdjacency:
    """Тесты извлечения матрицы смежности."""

    def test_basic_adjacency(self):
        """Базовое извлечение."""
        # Создаём матрицу с task node в индексе 0
        a_com = torch.zeros((4, 4), dtype=torch.float32)
        a_com[1, 2] = 0.5  # a -> b
        a_com[2, 3] = 0.8  # b -> c

        adj = extract_agent_adjacency(a_com, task_idx=0)

        assert tuple(adj.shape) == (3, 3)
        assert abs(adj[0, 1].item() - 0.5) < 1e-6  # a -> b
        assert abs(adj[1, 2].item() - 0.8) < 1e-6  # b -> c

    def test_empty_graph(self):
        """Граф с только task node."""
        a_com = torch.zeros((1, 1), dtype=torch.float32)

        adj = extract_agent_adjacency(a_com, task_idx=0)

        assert tuple(adj.shape) == (0, 0)


class TestParallelGroups:
    """Тесты параллельных групп."""

    def test_linear_no_parallel(self):
        """Линейный граф без параллельных групп."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c")],
        )

        groups = get_parallel_groups(a_agents, nodes)

        # Each step is sequential
        assert all(len(g) == 1 for g in groups)

    def test_diamond_has_parallel(self):
        """Ромбовидный граф имеет параллельную группу."""
        nodes = ["a", "b", "c", "d"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )

        groups = get_parallel_groups(a_agents, nodes)

        # b and c should be in parallel
        parallel_group = [g for g in groups if len(g) > 1]
        assert len(parallel_group) == 1
        assert set(parallel_group[0]) == {"b", "c"}

    def test_wide_parallel(self):
        """Широкий параллелизм: a -> b, c, d, e."""
        nodes = ["a", "b", "c", "d", "e"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")],
        )

        groups = get_parallel_groups(a_agents, nodes)

        # All b, c, d, e should be parallel
        parallel_groups = [g for g in groups if len(g) > 1]
        assert len(parallel_groups) == 1
        assert set(parallel_groups[0]) == {"b", "c", "d", "e"}


class TestIncomingOutgoing:
    """Тесты получения входящих/исходящих агентов."""

    def test_get_incoming(self):
        """Получение входящих агентов."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "c"), ("b", "c")],
            weights=[0.8, 0.6],
        )

        incoming = get_incoming_agents("c", a_agents, nodes)

        assert set(incoming) == {"a", "b"}

    def test_get_incoming_with_threshold(self):
        """Получение входящих агентов с порогом веса."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "c"), ("b", "c")],
            weights=[0.8, 0.3],
        )

        incoming = get_incoming_agents("c", a_agents, nodes, threshold=0.5)

        assert "a" in incoming
        assert "b" not in incoming

    def test_get_outgoing(self):
        """Получение исходящих агентов."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c")],
        )

        outgoing = get_outgoing_agents("a", a_agents, nodes)

        assert set(outgoing) == {"b", "c"}

    def test_no_incoming(self):
        """Узел без входящих."""
        nodes = ["a", "b"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b")],
        )

        incoming = get_incoming_agents("a", a_agents, nodes)

        assert incoming == []


class TestAdaptiveScheduler:
    """Тесты адаптивного планировщика."""

    def test_topological_policy(self):
        """Политика TOPOLOGICAL."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c")],
        )

        scheduler = AdaptiveScheduler(policy=RoutingPolicy.TOPOLOGICAL)
        plan = scheduler.build_plan(a_agents, nodes)

        assert len(plan.steps) == 3
        step_ids = [s.agent_id for s in plan.steps]
        assert step_ids.index("a") < step_ids.index("b")

    def test_weighted_topo_policy(self):
        """Политика WEIGHTED_TOPO."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            weights=[0.9, 0.3, 0.8],
        )

        scheduler = AdaptiveScheduler(policy=RoutingPolicy.WEIGHTED_TOPO)
        plan = scheduler.build_plan(a_agents, nodes)

        # Should respect both topology and weights
        assert len(plan.steps) == 3

    def test_pruning_config(self):
        """Конфигурация pruning."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c")],
            weights=[0.9, 0.05],  # c has low weight
        )

        config = PruningConfig(min_weight_threshold=0.1)
        scheduler = AdaptiveScheduler(
            policy=RoutingPolicy.WEIGHTED_TOPO,
            pruning_config=config,
        )
        plan = scheduler.build_plan(a_agents, nodes)

        # c might be marked as optional due to low weight
        c_step = next((s for s in plan.steps if s.agent_id == "c"), None)
        assert c_step is not None

    def test_replan(self):
        """Перепланировка."""
        nodes = ["a", "b", "c"]
        a_agents = create_adjacency_matrix(
            nodes=nodes,
            edges=[("a", "b"), ("b", "c")],
        )

        scheduler = AdaptiveScheduler()
        plan = scheduler.build_plan(a_agents, nodes)

        # Simulate failure of b
        step_results = {
            "b": StepResult(
                agent_id="b",
                success=False,
                error="timeout",
            )
        }

        new_plan = scheduler.replan(plan, a_agents, nodes, step_results=step_results)

        assert new_plan is not None


class TestExecutionPlan:
    """Тесты плана выполнения."""

    def test_plan_creation(self):
        """Создание плана."""
        steps = [
            ExecutionStep(agent_id="a", predecessors=[]),
            ExecutionStep(agent_id="b", predecessors=["a"]),
        ]

        plan = ExecutionPlan(steps=steps)

        assert len(plan.steps) == 2
        assert plan.current_index == 0

    def test_plan_iteration(self):
        """Итерация по плану."""
        steps = [
            ExecutionStep(agent_id="a", predecessors=[]),
            ExecutionStep(agent_id="b", predecessors=["a"]),
        ]

        plan = ExecutionPlan(steps=steps)

        step1 = plan.get_current_step()
        assert step1 is not None and step1.agent_id == "a"

        plan.mark_completed("a")

        step2 = plan.get_current_step()
        assert step2 is not None and step2.agent_id == "b"

    def test_plan_skip_step(self):
        """Пропуск шага."""
        steps = [
            ExecutionStep(agent_id="a", predecessors=[]),
            ExecutionStep(agent_id="b", predecessors=["a"]),
            ExecutionStep(agent_id="c", predecessors=["b"]),
        ]

        plan = ExecutionPlan(steps=steps)

        plan.mark_completed("a")
        plan.mark_skipped("b")  # Skip b

        step = plan.get_current_step()
        assert step is not None and step.agent_id == "c"

    def test_plan_is_complete(self):
        """Проверка завершённости плана."""
        steps = [
            ExecutionStep(agent_id="a", predecessors=[]),
        ]

        plan = ExecutionPlan(steps=steps)

        assert not plan.is_complete

        plan.mark_completed("a")

        assert plan.is_complete


class TestStepResult:
    """Тесты результата шага."""

    def test_success_result(self):
        """Успешный результат."""
        result = StepResult(
            agent_id="test",
            success=True,
            response="done",
            tokens_used=100,
        )

        assert result.success
        assert result.agent_id == "test"
        assert result.tokens_used == 100

    def test_failure_result(self):
        """Результат с ошибкой."""
        result = StepResult(
            agent_id="test",
            success=False,
            error="timeout",
        )

        assert not result.success
        assert result.error == "timeout"

    def test_fallback_used(self):
        """Результат с использованием fallback."""
        result = StepResult(
            agent_id="fallback_agent",
            success=True,
            fallback_used=True,
        )

        assert result.fallback_used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
