"""
Пример: Агент с tools.

Показывает как использовать все доступные инструменты:
- shell: выполнение команд
- code_interpreter: выполнение Python кода
- file_search: поиск файлов
- кастомные функции через @tool

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. Логирование ВСЕХ tool calls
2. Проверка что tools реально вызываются
3. Один caller для всех случаев
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
    FileSearchTool,
    LLMResponse,
    ShellTool,
    ToolResult,
    create_openai_caller,
    get_registry,
    tool,
)

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
        self.tool_calls: list[dict] = []

    def __call__(self, prompt: str, tools=None):
        response = self.inner(prompt, tools=tools)

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
        self.tool_calls = []

    def summary(self) -> str:
        if not self.tool_calls:
            return "❌ NO TOOL CALLS DETECTED"
        counts = {}
        for tc in self.tool_calls:
            name = tc["name"]
            counts[name] = counts.get(name, 0) + 1
        parts = [f"{name}×{count}" for name, count in counts.items()]
        return f"✅ {len(self.tool_calls)} tool calls: {', '.join(parts)}"


# ============================================================
# 2. Обёртки для встроенных tools с логированием
# ============================================================


class LoggingShellTool(ShellTool):
    """ShellTool с логированием."""

    def execute(self, command: str = "", **kwargs: Any) -> ToolResult:
        result = super().execute(command, **kwargs)
        result.output[:100] + "..." if len(result.output) > 100 else result.output
        return result


class LoggingCodeInterpreter(CodeInterpreterTool):
    """CodeInterpreter с логированием."""

    def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        result = super().execute(code, **kwargs)
        result.output[:100] + "..." if len(result.output) > 100 else result.output
        return result


class LoggingFileSearch(FileSearchTool):
    """FileSearchTool с логированием."""

    def execute(
        self,
        pattern: str = "*",
        query: str = "",
        read_file: str = "",
        directory: str = "",
        regex: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        result = super().execute(
            pattern=pattern,
            query=query,
            read_file=read_file,
            directory=directory,
            regex=regex,
            **kwargs,
        )
        result.output[:100] + "..." if len(result.output) > 100 else result.output
        return result


# ============================================================
# 3. Регистрируем tools
# ============================================================


def setup_tools():
    """Настроить все доступные tools."""
    registry = get_registry()

    # Встроенные tools с логированием
    registry.register(LoggingShellTool(timeout=10))
    registry.register(LoggingCodeInterpreter(timeout=10, safe_mode=True))
    registry.register(LoggingFileSearch(base_directory=".", max_results=10))

    # Кастомные функции
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
        """Check if a number is prime."""
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
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        import math

        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e}
        try:
            return str(eval(expression, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# 4. Тесты
# ============================================================


def test_custom_functions(caller: ToolCallLogger) -> dict:
    """Тест кастомных функций: fibonacci, is_prime, calculate."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="math",
        display_name="Math Agent",
        persona="a helpful math assistant",
        description="I solve math problems. ALWAYS use fibonacci, is_prime, and calculate tools.",
        tools=["fibonacci", "is_prime", "calculate"],
    )
    builder.add_task(query="Use fibonacci(10), then is_prime(17), then calculate('2**10')")
    builder.connect_task_to_agents(agent_ids=["math"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    expected = {"fibonacci", "is_prime", "calculate"}
    actual = {tc["name"] for tc in caller.tool_calls}
    missing = expected - actual

    if missing:
        return {"passed": False, "missing": missing, "actual": actual}

    return {"passed": True, "actual": actual}


def test_code_interpreter(caller: ToolCallLogger) -> dict:
    """Тест code_interpreter."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="coder",
        display_name="Python Coder",
        persona="a Python programmer",
        description="I execute Python code. ALWAYS use code_interpreter to run: print(2**100)",
        tools=["code_interpreter"],
    )
    builder.add_task(query="Use code_interpreter to run: print(2**100)")
    builder.connect_task_to_agents(agent_ids=["coder"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    expected = {"code_interpreter"}
    actual = {tc["name"] for tc in caller.tool_calls}

    if "code_interpreter" not in actual:
        return {"passed": False, "missing": expected, "actual": actual}

    return {"passed": True, "actual": actual}


def test_shell(caller: ToolCallLogger) -> dict:
    """Тест shell tool."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="sysadmin",
        display_name="System Admin",
        persona="a system administrator",
        description="I execute shell commands. ALWAYS use shell tool.",
        tools=["shell"],
    )
    builder.add_task(query="Use shell tool to run: echo Hello")
    builder.connect_task_to_agents(agent_ids=["sysadmin"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    if "shell" not in {tc["name"] for tc in caller.tool_calls}:
        return {"passed": False, "missing": {"shell"}, "actual": set()}

    return {"passed": True, "actual": {"shell"}}


def test_file_search(caller: ToolCallLogger) -> dict:
    """Тест file_search tool."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="searcher",
        display_name="File Searcher",
        persona="a file search specialist",
        description="I search for files. ALWAYS use file_search tool.",
        tools=["file_search"],
    )
    builder.add_task(query="Use file_search to find Python files (pattern='*.py')")
    builder.connect_task_to_agents(agent_ids=["searcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    runner.run_round(graph)

    if "file_search" not in {tc["name"] for tc in caller.tool_calls}:
        return {"passed": False, "missing": {"file_search"}, "actual": set()}

    return {"passed": True, "actual": {"file_search"}}


# ============================================================
# 5. Главная функция
# ============================================================


def main():
    setup_tools()

    inner_caller = create_openai_caller(
        base_url="https://quick-dramatically-resource-applicable.trycloudflare.com/v1",
        api_key="very-secure-key-sber1love",
        model="gpt-oss",
        temperature=0.1,
        tool_choice="required",
    )
    caller = ToolCallLogger(inner_caller)

    results = {}

    for test_name, test_fn in [
        ("custom_functions", test_custom_functions),
        ("code_interpreter", test_code_interpreter),
        ("shell", test_shell),
        ("file_search", test_file_search),
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
