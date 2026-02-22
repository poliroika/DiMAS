"""
Agent with tools — built-in and custom.

Demonstrates:
  1. Custom tools via the ``@tool`` decorator (fibonacci, is_prime, calculate)
  2. Built-in ``code_interpreter`` tool
  3. Built-in ``shell`` tool
  4. Built-in ``file_search`` tool

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.agent_with_tools_example
"""

import math
import os

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import (
    CodeInterpreterTool,
    FileSearchTool,
    ShellTool,
    create_openai_caller,
    get_registry,
    tool,
)

# ── Constants ───────────────────────────────────────────────────────────────────

MIN_PRIME_NUMBER = 2

# ── Custom tools ─────────────────────────────────────────────────────────────


@tool
def fibonacci(n: int) -> str:
    """Calculate the n-th Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return str(a)


@tool
def is_prime(n: int) -> str:
    """Check if a number is prime. Returns 'True' or 'False'."""
    if n < MIN_PRIME_NUMBER:
        return "False"
    for i in range(MIN_PRIME_NUMBER, math.isqrt(n) + 1):
        if n % i == 0:
            return "False"
    return "True"


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe math expression (sqrt, sin, cos, pi, e)."""
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as exc:
        return f"Error: {exc}"


# ── Shared helpers ───────────────────────────────────────────────────────────


def _setup_tools() -> None:
    """Register built-in tools once."""
    registry = get_registry()
    registry.register(ShellTool(timeout=10))
    registry.register(CodeInterpreterTool(timeout=10, safe_mode=True))
    registry.register(FileSearchTool(base_directory=".", max_results=10))


def _create_llm():
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


def _header(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── Example 1: Custom math tools ────────────────────────────────────────────


def example_custom_math_tools():
    """Agent uses fibonacci, is_prime, and calculate tools."""
    _header("1 · Custom Math Tools")

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="math_agent",
        display_name="Math Agent",
        persona="a helpful math assistant",
        description="Solves math problems using available tools.",
        tools=["fibonacci", "is_prime", "calculate"],
    )
    builder.add_task(query="Calculate fibonacci(10), check if 17 is prime, and compute 2**10")
    builder.connect_task_to_agents(agent_ids=["math_agent"])

    graph = builder.build()
    result = MACPRunner(llm_caller=_create_llm()).run_round(graph)

    print(f"  Task   : {graph.query}")
    print(f"  Answer : {result.final_answer}")


# ── Example 2: Code interpreter ─────────────────────────────────────────────


def example_code_interpreter():
    """Agent runs Python code via the code_interpreter tool."""
    _header("2 · Code Interpreter")

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="coder",
        display_name="Python Coder",
        persona="a Python programmer",
        description="Executes Python code to solve problems.",
        tools=["code_interpreter"],
    )
    builder.add_task(query="Use code_interpreter to calculate 2**100")
    builder.connect_task_to_agents(agent_ids=["coder"])

    graph = builder.build()
    result = MACPRunner(llm_caller=_create_llm()).run_round(graph)

    print(f"  Task   : {graph.query}")
    print(f"  Answer : {result.final_answer}")


# ── Example 3: Shell tool ───────────────────────────────────────────────────


def example_shell_tool():
    """Agent executes a shell command."""
    _header("3 · Shell Tool")

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="sysadmin",
        display_name="System Admin",
        persona="a system administrator",
        description="Executes shell commands.",
        tools=["shell"],
    )
    builder.add_task(query="Use shell to run: echo 'Hello from shell'")
    builder.connect_task_to_agents(agent_ids=["sysadmin"])

    graph = builder.build()
    result = MACPRunner(llm_caller=_create_llm()).run_round(graph)

    print(f"  Task   : {graph.query}")
    print(f"  Answer : {result.final_answer}")


# ── Example 4: File search ──────────────────────────────────────────────────


def example_file_search():
    """Agent searches for files matching a pattern."""
    _header("4 · File Search")

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="searcher",
        display_name="File Searcher",
        persona="a file search specialist",
        description="Searches for files by pattern.",
        tools=["file_search"],
    )
    builder.add_task(query="Find Python files (pattern='*.py')")
    builder.connect_task_to_agents(agent_ids=["searcher"])

    graph = builder.build()
    result = MACPRunner(llm_caller=_create_llm()).run_round(graph)

    print(f"  Task   : {graph.query}")
    print(f"  Answer : {result.final_answer[:200]}…")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    _setup_tools()

    example_custom_math_tools()
    example_code_interpreter()
    example_shell_tool()
    example_file_search()

    print(f"\n{'=' * 60}")
    print("All tool examples completed ✅")


if __name__ == "__main__":
    main()
