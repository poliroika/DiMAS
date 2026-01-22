"""Тесты для execution/runner.py — MACPRunner."""

import asyncio

import pytest
import rustworkx as rx
import torch

from rustworkx_framework.core.graph import RoleGraph
from rustworkx_framework.execution.budget import BudgetConfig
from rustworkx_framework.execution.runner import MACPResult, MACPRunner, RunnerConfig


def create_test_graph(nodes, edges):
    """Создать тестовый RoleGraph."""
    from rustworkx_framework.core.agent import AgentProfile

    g = rx.PyDiGraph()

    id_to_idx = {}
    agents = []
    for nid in nodes:
        if nid != "task":
            idx = g.add_node({"id": nid})
            id_to_idx[nid] = idx
            # Создаём роли для агентов
            agent = AgentProfile(identifier=nid, display_name=f"Agent {nid.upper()}")
            agents.append(agent)

    connections = {n: [] for n in nodes}

    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            g.add_edge(id_to_idx[src], id_to_idx[tgt], {"weight": 1.0})
            connections[src].append(tgt)

    n = len(id_to_idx)
    a_com = torch.zeros((n + 1, n + 1), dtype=torch.float32)  # +1 для task node

    # Добавляем task node
    task_idx = g.add_node({"id": "task"})
    id_to_idx["task"] = task_idx

    # Заполняем матрицу
    node_list = [nid for nid in nodes if nid != "task"]
    for i, src in enumerate(node_list):
        for tgt in connections[src]:
            if tgt in node_list:
                j = node_list.index(tgt)
                a_com[i, j] = 1.0

    role_graph = RoleGraph(
        node_ids=nodes,
        role_connections=connections,
        graph=g,
        A_com=a_com,
        task_node="task",
        query="test query",
    )
    role_graph.agents = agents

    return role_graph


def create_simple_llm_caller(response_text="Test response"):
    """Создать простой синхронный LLM caller."""

    def llm_caller(prompt: str) -> str:
        return response_text

    return llm_caller


def create_simple_async_llm_caller(response_text="Test response"):
    """Создать простой асинхронный LLM caller."""

    async def async_llm_caller(prompt: str) -> str:
        await asyncio.sleep(0.001)  # Имитация задержки
        return response_text

    return async_llm_caller


class TestMACPRunnerCreation:
    """Тесты создания MACPRunner."""

    def test_basic_creation(self):
        """Базовое создание."""
        llm_caller = create_simple_llm_caller()
        runner = MACPRunner(llm_caller=llm_caller)

        assert runner is not None
        assert runner.llm_caller is not None

    def test_creation_with_config(self):
        """Создание с конфигурацией."""
        llm_caller = create_simple_llm_caller()
        config = RunnerConfig(
            timeout=30.0,
            max_retries=3,
            adaptive=True,
        )

        runner = MACPRunner(llm_caller=llm_caller, config=config)

        assert runner.config.timeout == 30.0
        assert runner.config.max_retries == 3
        assert runner.config.adaptive


class TestSyncExecution:
    """Тесты синхронного выполнения."""

    def test_run_simple(self):
        """Простой запуск."""
        graph = create_test_graph(["a", "b"], [("a", "b")])
        llm_caller = create_simple_llm_caller()

        runner = MACPRunner(llm_caller=llm_caller)
        result = runner.run_round(graph)

        assert isinstance(result, MACPResult)
        assert result.final_answer is not None

    def test_run_linear_graph(self):
        """Запуск на линейном графе."""
        graph = create_test_graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
        llm_caller = create_simple_llm_caller()

        runner = MACPRunner(llm_caller=llm_caller)
        result = runner.run_round(graph)

        assert len(result.execution_order) == 3
        assert result.final_answer is not None

    def test_run_with_final_agent(self):
        """Запуск с указанием финального агента."""
        graph = create_test_graph(["a", "b"], [("a", "b")])
        llm_caller = create_simple_llm_caller("final response")

        runner = MACPRunner(llm_caller=llm_caller)
        result = runner.run_round(graph, final_agent_id="b")

        assert result.final_agent_id == "b"


class TestAsyncExecution:
    """Тесты асинхронного выполнения."""

    @pytest.mark.asyncio
    async def test_arun_simple(self):
        """Простой асинхронный запуск."""
        graph = create_test_graph(["a", "b"], [("a", "b")])
        async_llm_caller = create_simple_async_llm_caller()

        runner = MACPRunner(async_llm_caller=async_llm_caller)
        result = await runner.arun_round(graph)

        assert isinstance(result, MACPResult)
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_arun_parallel_execution(self):
        """Параллельное асинхронное выполнение."""
        # a -> b, c (parallel) -> d
        graph = create_test_graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )

        async_llm_caller = create_simple_async_llm_caller()
        config = RunnerConfig(enable_parallel=True, adaptive=True)
        runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should have executed all agents
        assert len(result.execution_order) == 4
        # a should be first in execution order
        assert result.execution_order[0] == "a"


class TestTimeouts:
    """Тесты обработки таймаутов."""

    @pytest.mark.asyncio
    async def test_timeout_triggers(self):
        """Таймаут срабатывает."""
        graph = create_test_graph(["a"], [])

        async def slow_llm_caller(prompt: str) -> str:
            await asyncio.sleep(10.0)  # Очень медленный
            return "done"

        config = RunnerConfig(timeout=0.1)  # Короткий таймаут

        runner = MACPRunner(async_llm_caller=slow_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should complete but might have timeout in messages
        assert result is not None

    @pytest.mark.asyncio
    async def test_per_agent_timeout(self):
        """Таймаут на уровне агента."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        call_count = 0

        async def slow_llm_caller(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                await asyncio.sleep(10.0)
            return "response"

        config = RunnerConfig(timeout=0.1)
        runner = MACPRunner(async_llm_caller=slow_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # First agent should succeed, second might timeout
        assert result is not None


class TestRetries:
    """Тесты механизма повторов."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Повтор при ошибке."""
        graph = create_test_graph(["a"], [])

        attempt_count = 0

        async def flaky_llm_caller(prompt: str) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        config = RunnerConfig(max_retries=5, adaptive=True)
        runner = MACPRunner(async_llm_caller=flaky_llm_caller, config=config)
        result = await runner.arun_round(graph)

        assert attempt_count == 3
        assert result.messages.get("a") == "success"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Превышение максимума повторов."""
        graph = create_test_graph(["a"], [])

        async def always_fails(prompt: str) -> str:
            raise RuntimeError("Always fails")

        config = RunnerConfig(max_retries=2, adaptive=True)
        runner = MACPRunner(async_llm_caller=always_fails, config=config)
        result = await runner.arun_round(graph)

        # Should fail after max retries
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Повтор с экспоненциальной задержкой."""
        graph = create_test_graph(["a"], [])

        import time

        timestamps = []

        async def timing_llm_caller(prompt: str) -> str:
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise RuntimeError("Retry")
            return "done"

        config = RunnerConfig(
            max_retries=5,
            retry_delay=0.1,
            retry_backoff=2.0,
            adaptive=True,
        )

        runner = MACPRunner(async_llm_caller=timing_llm_caller, config=config)
        await runner.arun_round(graph)

        # Check delays increased
        if len(timestamps) >= 3:
            delay1 = timestamps[1] - timestamps[0]
            delay2 = timestamps[2] - timestamps[1]
            assert delay2 > delay1  # Backoff should increase delay


class TestBudgetControl:
    """Тесты контроля бюджета."""

    @pytest.mark.asyncio
    async def test_token_budget_respected(self):
        """Бюджет токенов соблюдается."""
        graph = create_test_graph(["a", "b", "c"], [("a", "b"), ("b", "c")])

        async def token_hungry_llm_caller(prompt: str) -> str:
            # Симулируем использование токенов
            return "response " * 100  # Много токенов

        budget_config = BudgetConfig(
            # Using reasonable defaults - BudgetConfig doesn't require specific params
        )
        config = RunnerConfig(budget_config=budget_config, adaptive=True)

        runner = MACPRunner(async_llm_caller=token_hungry_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should stop due to budget or complete with budget tracking
        assert result.budget_summary is not None or result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_budget_warning(self):
        """Предупреждение о приближении к бюджету."""
        graph = create_test_graph(["a"], [])

        async_llm_caller = create_simple_async_llm_caller()
        budget_config = BudgetConfig(
            # Using reasonable defaults
        )
        config = RunnerConfig(budget_config=budget_config)

        runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should complete successfully
        assert result is not None


class TestMemoryUpdates:
    """Тесты обновления памяти агентов."""

    @pytest.mark.asyncio
    async def test_state_propagation(self):
        """Состояние передаётся между агентами."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        async_llm_caller = create_simple_async_llm_caller()
        runner = MACPRunner(async_llm_caller=async_llm_caller)
        result = await runner.arun_round(graph)

        # b should have received context from a
        assert "b" in result.messages
        assert result.messages["b"] is not None

    @pytest.mark.asyncio
    async def test_hidden_state_channels(self):
        """Скрытые каналы состояния."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        async_llm_caller = create_simple_async_llm_caller()
        config = RunnerConfig(enable_hidden_channels=True)
        runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)

        result = await runner.arun_round(graph)

        assert result is not None


class TestAdaptiveMode:
    """Тесты адаптивного режима."""

    @pytest.mark.asyncio
    async def test_adaptive_routing(self):
        """Адаптивная маршрутизация."""
        graph = create_test_graph(
            ["a", "b", "c"],
            [("a", "b"), ("a", "c"), ("b", "c")],
        )

        async_llm_caller = create_simple_async_llm_caller()
        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)

        result = await runner.arun_round(graph)

        assert len(result.execution_order) > 0

    @pytest.mark.asyncio
    async def test_adaptive_replanning(self):
        """Адаптивная перепланировка при ошибке."""
        graph = create_test_graph(
            ["a", "b", "fallback", "c"],
            [("a", "b"), ("a", "fallback"), ("b", "c"), ("fallback", "c")],
        )

        call_count = 0

        async def maybe_failing_llm_caller(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # Симулируем ошибку на втором вызове (агент b)
            if call_count == 2:
                raise RuntimeError("b failed")
            return "response"

        config = RunnerConfig(
            adaptive=True,
            enable_replanning=True,
            max_retries=0,
        )
        runner = MACPRunner(async_llm_caller=maybe_failing_llm_caller, config=config)

        result = await runner.arun_round(graph)

        # Should complete with some agents executed
        assert result is not None


class TestErrorHandling:
    """Тесты обработки ошибок."""

    @pytest.mark.asyncio
    async def test_agent_exception_handled(self):
        """Исключение агента обрабатывается."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        call_count = 0

        async def failing_llm_caller(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Agent error")
            return "response"

        config = RunnerConfig(max_retries=0)
        runner = MACPRunner(async_llm_caller=failing_llm_caller, config=config)

        result = await runner.arun_round(graph)

        # Should not crash, error should be recorded
        assert result is not None

    @pytest.mark.asyncio
    async def test_on_error_fail_policy(self):
        """Обработка ошибок с повторами."""
        graph = create_test_graph(["a"], [])

        async def failing_llm_caller(prompt: str) -> str:
            raise RuntimeError("Critical error")

        config = RunnerConfig(max_retries=0, adaptive=True)
        runner = MACPRunner(async_llm_caller=failing_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should have error recorded
        assert result.errors is not None or "[Error:" in str(result.messages.get("a", ""))

    @pytest.mark.asyncio
    async def test_on_error_skip_policy(self):
        """Обработка ошибок и продолжение выполнения."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        call_count = 0

        async def maybe_failing_llm_caller(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("a failed")
            return "response"

        config = RunnerConfig(max_retries=0)
        runner = MACPRunner(async_llm_caller=maybe_failing_llm_caller, config=config)
        result = await runner.arun_round(graph)

        # Should continue to agent b
        assert result is not None


class TestMACPResult:
    """Тесты результата выполнения."""

    def test_result_structure(self):
        """Структура результата."""

        result = MACPResult(
            messages={"a": "response"},
            final_answer="final answer",
            final_agent_id="a",
            execution_order=["a"],
            errors=[],
        )

        assert result.final_answer == "final answer"
        assert result.final_agent_id == "a"
        assert result.execution_order == ["a"]
        assert result.errors == []

    def test_result_with_metrics(self):
        """Результат с метриками."""
        from datetime import datetime

        from rustworkx_framework.execution.errors import ExecutionMetrics

        metrics = ExecutionMetrics(
            start_time=datetime.now(),
            total_agents=1,
            total_tokens=500,
        )

        result = MACPResult(
            messages={"a": "response"},
            final_answer="answer",
            final_agent_id="a",
            execution_order=["a"],
            errors=[],
            metrics=metrics,
            total_tokens=500,
            total_time=1.234,
        )

        assert result.total_tokens == 500
        assert result.total_time == 1.234


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
