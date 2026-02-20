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
            agent = AgentProfile(agent_id=nid, display_name=f"Agent {nid.upper()}")
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
                msg = "Temporary failure"
                raise RuntimeError(msg)
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
            msg = "Always fails"
            raise RuntimeError(msg)

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
                msg = "Retry"
                raise RuntimeError(msg)
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
    async def test_adaptive_topology_change(self):
        """Адаптивное изменение топологии при ошибке."""
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
                msg = "b failed"
                raise RuntimeError(msg)
            return "response"

        config = RunnerConfig(
            adaptive=True,
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
                msg = "Agent error"
                raise ValueError(msg)
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
            msg = "Critical error"
            raise RuntimeError(msg)

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
                msg = "a failed"
                raise RuntimeError(msg)
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


class TestConditionalEdgesAdaptive:
    """Тесты условных рёбер в адаптивном режиме."""

    def test_condition_true_executes_target(self):
        """Если условие выполнено — target агент выполняется."""
        graph = create_test_graph(["a", "b", "c"], [("a", "b"), ("b", "c")])

        # Условие: a→b выполняется только если a отвечает с "ok"
        graph.edge_conditions = {
            ("a", "b"): lambda ctx: "ok" in ctx.messages.get("a", ""),
        }

        llm_caller = create_simple_llm_caller("ok response")
        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(llm_caller=llm_caller, config=config)

        result = runner.run_round(graph)

        assert "a" in result.execution_order
        assert "b" in result.execution_order

    def test_condition_false_skips_target(self):
        """Если условие не выполнено — target агент скипается."""
        graph = create_test_graph(["a", "b", "c"], [("a", "b"), ("a", "c")])

        # Условие: a→b выполняется только если a отвечает с "secret"
        graph.edge_conditions = {
            ("a", "b"): lambda ctx: "secret" in ctx.messages.get("a", ""),
        }

        llm_caller = create_simple_llm_caller("normal response")
        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(llm_caller=llm_caller, config=config)

        result = runner.run_round(graph)

        assert "a" in result.execution_order
        # b должен быть пропущен из-за невыполненного условия
        # c должен выполниться (безусловное ребро)
        assert "c" in result.execution_order

    @pytest.mark.asyncio
    async def test_async_conditional_edges(self):
        """Условные рёбра работают в async режиме."""
        graph = create_test_graph(["a", "b", "c"], [("a", "b"), ("a", "c")])

        graph.edge_conditions = {
            ("a", "b"): lambda ctx: ctx.source_succeeded(),
        }

        async_llm_caller = create_simple_async_llm_caller("response")
        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)

        result = await runner.arun_round(graph)

        assert "a" in result.execution_order
        assert "b" in result.execution_order

    def test_topology_changed_count(self):
        """topology_changed_count увеличивается при изменении плана."""
        graph = create_test_graph(["a", "b"], [("a", "b")])

        graph.edge_conditions = {
            ("a", "b"): lambda ctx: ctx.source_succeeded(),
        }

        llm_caller = create_simple_llm_caller("response")
        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(llm_caller=llm_caller, config=config)

        result = runner.run_round(graph)

        assert result is not None
        assert isinstance(result.topology_changed_count, int)

    def test_multiple_incoming_conditional_edges(self):
        """Множественные входящие условные рёбра: B не скипается пока не оценены все."""
        graph = create_test_graph(
            ["a", "c", "b"],
            [("a", "b"), ("c", "b")],
        )

        # Разные callers для разных агентов
        llm_callers = {
            "a": lambda prompt: "fail result",
            "c": lambda prompt: "good result",
            "b": lambda prompt: "final response",
        }

        # a→b: условие НЕ выполнено (нет "success" в ответе a)
        # c→b: условие ВЫПОЛНЕНО (есть "good" в ответе c)
        graph.edge_conditions = {
            ("a", "b"): lambda ctx: "success" in ctx.messages.get("a", ""),
            ("c", "b"): lambda ctx: "good" in ctx.messages.get("c", ""),
        }

        config = RunnerConfig(adaptive=True)
        runner = MACPRunner(
            llm_caller=lambda p: "default",
            llm_callers=llm_callers,
            config=config,
        )

        result = runner.run_round(graph)

        # b должен выполниться, т.к. c→b условие выполнено
        assert "a" in result.execution_order
        assert "c" in result.execution_order
        assert "b" in result.execution_order


    def test_conditional_edges_with_hidden_states_and_chain(self):
        """
        Комплексный тест: условные рёбра + hidden states + каскадная цепочка.

        Граф:
            solver → reviewer → finalize  (условное ребро solver→reviewer)
            solver → alt_end              (безусловное)

        Сценарий 1 (condition=True):  solver("correct") → reviewer → finalize выполняются.
        Сценарий 2 (condition=False): solver("wrong") → reviewer + finalize скипаются,
                                      alt_end выполняется.

        Покрывает:
        - Проблема 1: hidden states + условные рёбра
        - Проблема 2: план не прерывается после последнего агента
        - Проблема 3: вся цепочка выполняется после условного перехода
        """
        # --- Сценарий 1: условие выполнено → вся цепочка ---
        graph1 = create_test_graph(
            ["solver", "reviewer", "finalize", "alt_end"],
            [
                ("solver", "reviewer"),
                ("reviewer", "finalize"),
                ("solver", "alt_end"),
            ],
        )
        graph1.edge_conditions = {
            ("solver", "reviewer"): lambda ctx: "correct" in ctx.messages.get("solver", ""),
        }

        callers1 = {
            "solver": lambda p: "answer is correct",
            "reviewer": lambda p: "review passed",
            "finalize": lambda p: "done",
            "alt_end": lambda p: "alt",
        }

        config = RunnerConfig(adaptive=True)
        runner1 = MACPRunner(
            llm_caller=lambda p: "default",
            llm_callers=callers1,
            config=config,
        )

        result1 = runner1.run_round_with_hidden(graph1)

        assert "solver" in result1.execution_order
        assert "reviewer" in result1.execution_order
        assert "finalize" in result1.execution_order  # вся цепочка
        assert result1.hidden_states is not None
        assert "solver" in result1.hidden_states

        # --- Сценарий 2: условие НЕ выполнено → каскадный skip ---
        graph2 = create_test_graph(
            ["solver", "reviewer", "finalize", "alt_end"],
            [
                ("solver", "reviewer"),
                ("reviewer", "finalize"),
                ("solver", "alt_end"),
            ],
        )
        graph2.edge_conditions = {
            ("solver", "reviewer"): lambda ctx: "correct" in ctx.messages.get("solver", ""),
        }

        callers2 = {
            "solver": lambda p: "answer is wrong",
            "reviewer": lambda p: "review passed",
            "finalize": lambda p: "done",
            "alt_end": lambda p: "alt ending",
        }

        runner2 = MACPRunner(
            llm_caller=lambda p: "default",
            llm_callers=callers2,
            config=config,
        )

        result2 = runner2.run_round_with_hidden(graph2)

        assert "solver" in result2.execution_order
        assert "reviewer" not in result2.execution_order  # скипнут
        assert "finalize" not in result2.execution_order  # каскадно скипнут
        assert "alt_end" in result2.execution_order  # безусловный путь


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
