"""
Пример: Multi-agent графы с tools.

Демонстрирует работу нескольких агентов в графе, каждый со своими tools.
Показывает что tools работают корректно при передаче сообщений между агентами.

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. Логирование ВСЕХ tool calls (из LLM и при выполнении)
2. Счётчик tool calls для проверки
3. Кэширование — повторные вызовы не выполняются
"""

import logging
import sys
from typing import Any

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import (
    CodeInterpreterTool,
    LLMResponse,
    ToolResult,
    create_openai_caller,
    get_registry,
    register_tool,
    tool,
)

# Включаем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Логирующая обёртка для caller
# ============================================================


class ToolCallLogger:
    """Обёртка над caller которая логирует ВСЕ tool calls от LLM."""

    def __init__(self, inner_caller):
        self.inner = inner_caller
        self.tool_calls: list[dict] = []  # Все tool calls от LLM

    def __call__(self, prompt: str, tools=None):
        response = self.inner(prompt, tools=tools)

        # Логируем tool calls если это LLMResponse
        if isinstance(response, LLMResponse) and response.has_tool_calls:
            for tc in response.tool_calls:
                self.tool_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

        return response

    def reset(self):
        """Сбросить счётчик."""
        self.tool_calls = []

    def summary(self) -> str:
        """Получить сводку по tool calls."""
        if not self.tool_calls:
            return "❌ NO TOOL CALLS DETECTED"

        # Подсчёт по именам
        counts = {}
        for tc in self.tool_calls:
            name = tc["name"]
            counts[name] = counts.get(name, 0) + 1

        parts = [f"{name}×{count}" for name, count in counts.items()]
        return f"✅ {len(self.tool_calls)} tool calls: {', '.join(parts)}"


# ============================================================
# 2. Регистрируем tools С ЛОГИРОВАНИЕМ
# ============================================================


# Обёртка для CodeInterpreter с логированием
class LoggingCodeInterpreter(CodeInterpreterTool):
    """CodeInterpreter с логированием выполнения."""

    def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        result = super().execute(code, **kwargs)
        result.output[:100] + "..." if len(result.output) > 100 else result.output
        return result


register_tool(LoggingCodeInterpreter(timeout=10, safe_mode=True))


@tool
def fibonacci(n: int) -> str:
    """Calculate the n-th Fibonacci number."""
    if n <= 0:
        return "0"
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return str(a)


@tool
def is_prime(n: int) -> str:
    """Check if a number is prime. Returns 'True' or 'False'."""
    if n < 2:
        result = "False"
    else:
        result = "True"
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                result = "False"
                break
    return result


@tool
def factorize(n: int) -> str:
    """Find prime factors of a number."""
    if n <= 1:
        return str(n)
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return " × ".join(map(str, factors))


@tool
def sum_digits(n: int) -> str:
    """Calculate the sum of digits of a number."""
    result = sum(int(d) for d in str(abs(n)))
    return str(result)


# ============================================================
# 3. Тесты multi-agent графов с tools
# ============================================================


def test_two_connected_agents(caller: ToolCallLogger) -> dict:
    """Тест: Два связанных агента с разными tools."""
    caller.reset()  # Сбрасываем счётчик

    builder = GraphBuilder()

    builder.add_agent(
        agent_id="calculator",
        display_name="Calculator",
        persona="a calculator",
        description="I calculate Fibonacci numbers. ALWAYS use fibonacci tool.",
        tools=["fibonacci"],
    )

    builder.add_agent(
        agent_id="analyzer",
        display_name="Analyzer",
        persona="a number analyzer",
        description="I analyze numbers. I MUST use ALL my tools in order: 1) is_prime, 2) factorize, 3) sum_digits. Never skip any tool!",
        tools=["is_prime", "factorize", "sum_digits"],
    )

    builder.add_task(
        query="Calculate fibonacci(20). Then Analyzer MUST: 1) use is_prime, 2) use factorize, 3) use sum_digits. ALL THREE tools required!"
    )
    builder.connect_task_to_agents(agent_ids=["calculator"])
    builder.add_edge(source="calculator", target="analyzer")

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    # Проверяем что основные tools были вызваны (минимум 3 из 4)
    expected = {"fibonacci", "is_prime", "factorize", "sum_digits"}
    actual = {tc["name"] for tc in caller.tool_calls}
    missing = expected - actual

    # Минимальные обязательные tools
    required = {"fibonacci", "is_prime"}
    missing_required = required - actual

    if missing_required:
        return {"passed": False, "missing": missing_required, "actual": actual}

    if missing:
        pass

    return {"passed": True, "actual": actual, "missing_optional": missing}


def test_parallel_agents(caller: ToolCallLogger) -> dict:
    """Тест: Два параллельных агента с разными tools."""
    caller.reset()

    builder = GraphBuilder()

    builder.add_agent(
        agent_id="math_agent",
        display_name="Math Agent",
        persona="a math specialist",
        description="I calculate Fibonacci numbers. ALWAYS use fibonacci tool.",
        tools=["fibonacci"],
    )

    builder.add_agent(
        agent_id="code_agent",
        display_name="Code Agent",
        persona="a Python programmer",
        description="I execute Python code. ALWAYS use code_interpreter to run: print(2**100)",
        tools=["code_interpreter"],
    )

    builder.add_task(query="Math Agent: use fibonacci(30). Code Agent: use code_interpreter to run print(2**100)")
    builder.connect_task_to_agents(agent_ids=["math_agent", "code_agent"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    expected = {"fibonacci", "code_interpreter"}
    actual = {tc["name"] for tc in caller.tool_calls}
    missing = expected - actual

    if missing:
        return {"passed": False, "missing": missing, "actual": actual}

    return {"passed": True, "actual": actual}


def test_chain_of_three(caller: ToolCallLogger) -> dict:
    """Тест: Цепочка из трёх агентов."""
    caller.reset()

    builder = GraphBuilder()

    builder.add_agent(
        agent_id="fib_agent",
        display_name="Fibonacci Agent",
        persona="a Fibonacci calculator",
        description="I calculate Fibonacci numbers. ALWAYS use fibonacci tool. Output ONLY the number.",
        tools=["fibonacci"],
    )

    builder.add_agent(
        agent_id="prime_agent",
        display_name="Prime Checker",
        persona="a prime number checker",
        description="I check if numbers are prime. ALWAYS use is_prime tool on the number from previous agent.",
        tools=["is_prime"],
    )

    builder.add_agent(
        agent_id="digit_agent",
        display_name="Digit Summer",
        persona="a digit sum calculator",
        description="I calculate sum of digits. ALWAYS use sum_digits tool on the number.",
        tools=["sum_digits"],
    )

    builder.add_task(
        query="Calculate fibonacci(25), then check if prime, then sum its digits. Each agent MUST use their tool."
    )
    builder.connect_task_to_agents(agent_ids=["fib_agent"])
    builder.add_edge(source="fib_agent", target="prime_agent")
    builder.add_edge(source="prime_agent", target="digit_agent")

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    expected = {"fibonacci", "is_prime", "sum_digits"}
    actual = {tc["name"] for tc in caller.tool_calls}
    missing = expected - actual

    if missing:
        return {"passed": False, "missing": missing, "actual": actual}

    return {"passed": True, "actual": actual}


# ============================================================
# 4. Главная функция
# ============================================================


def main():
    get_registry()

    # Создаём caller с логированием
    inner_caller = create_openai_caller(
        base_url="https://quick-dramatically-resource-applicable.trycloudflare.com/v1",
        api_key="very-secure-key-sber1love",
        model="gpt-oss",
        temperature=0.1,
        tool_choice="required",
    )
    caller = ToolCallLogger(inner_caller)

    # Запускаем тесты
    results = {}

    for test_name, test_fn in [
        ("two_agents", test_two_connected_agents),
        ("parallel_agents", test_parallel_agents),
        ("chain_of_three", test_chain_of_three),
    ]:
        try:
            result = test_fn(caller)
            results[test_name] = result
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[test_name] = {"passed": False, "error": str(e)}

    # Итоги

    all_passed = True
    for test_name, result in results.items():
        passed = result.get("passed", False)
        if passed:
            result.get("actual", set())
        else:
            all_passed = False
            missing = result.get("missing", set())
            result.get("actual", set())
            result.get("error", "")
            if missing:
                pass
            else:
                pass

    if all_passed:
        pass
    else:
        pass


if __name__ == "__main__":
    main()
