"""
Example: Agent with tools.

Demonstrates how to use built-in and custom tools with agents:
- shell: execute shell commands
- code_interpreter: run Python code in a sandbox
- file_search: search and read files
- custom functions registered via @tool decorator

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run with:
    python -m examples.agent_with_tools_example
"""

import math
import os
import sys

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import (
    CodeInterpreterTool,
    FileSearchTool,
    ShellTool,
    create_openai_caller,
    get_registry,
    tool,
)

# Setup framework logging
setup_logging(level="INFO")


# ── Register custom tools ───────────────────────────────────────────────────────


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
        return "False"
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return "False"
    return "True"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a safe math expression.

    Supported names: sqrt, sin, cos, pi, e.
    Examples: 'sqrt(16)', '2**10', 'sin(pi/2)'.
    """
    allowed = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


def setup_tools():
    """Register built-in tools and custom @tool functions."""
    registry = get_registry()
    registry.register(ShellTool(timeout=10))
    registry.register(CodeInterpreterTool(timeout=10, safe_mode=True))
    registry.register(FileSearchTool(base_directory=".", max_results=10))


def create_llm():
    """Create LLM caller from environment variables."""
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


# ── Example 1: Custom math tools ────────────────────────────────────────────────


def example_custom_math_tools():
    """Example: Agent using custom fibonacci, is_prime, and calculate tools."""
    print("\n" + "─" * 60)
    print("Example 1: Custom Math Tools")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="math_agent",
        display_name="Math Agent",
        persona="a helpful math assistant",
        description="I solve math problems using available tools.",
        tools=["fibonacci", "is_prime", "calculate"],
    )
    builder.add_task(
        query="Calculate fibonacci(10), check if 17 is prime, and calculate 2**10"
    )
    builder.connect_task_to_agents(agent_ids=["math_agent"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 2: Code interpreter ─────────────────────────────────────────────────


def example_code_interpreter():
    """Example: Agent using code_interpreter to run Python code."""
    print("\n" + "─" * 60)
    print("Example 2: Code Interpreter")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="coder",
        display_name="Python Coder",
        persona="a Python programmer",
        description="I execute Python code to solve problems.",
        tools=["code_interpreter"],
    )
    builder.add_task(query="Use code_interpreter to calculate 2**100")
    builder.connect_task_to_agents(agent_ids=["coder"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 3: Shell tool ───────────────────────────────────────────────────────


def example_shell_tool():
    """Example: Agent using shell tool to execute commands."""
    print("\n" + "─" * 60)
    print("Example 3: Shell Tool")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="sysadmin",
        display_name="System Admin",
        persona="a system administrator",
        description="I execute shell commands.",
        tools=["shell"],
    )
    builder.add_task(query="Use shell to run: echo 'Hello from shell'")
    builder.connect_task_to_agents(agent_ids=["sysadmin"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 4: File search ──────────────────────────────────────────────────────


def example_file_search():
    """Example: Agent using file_search to find files."""
    print("\n" + "─" * 60)
    print("Example 4: File Search")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="searcher",
        display_name="File Searcher",
        persona="a file search specialist",
        description="I search for files.",
        tools=["file_search"],
    )
    builder.add_task(query="Find Python files (pattern='*.py')")
    builder.connect_task_to_agents(agent_ids=["searcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer[:200]}...")


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    """Run all examples."""
    setup_tools()

    try:
        example_custom_math_tools()
        example_code_interpreter()
        example_shell_tool()
        example_file_search()
        print("\n" + "=" * 60)
        print("All examples completed ✅")
    except Exception as e:
        logger.exception("Error running examples: %s", e)
        raise


if __name__ == "__main__":
    main()
