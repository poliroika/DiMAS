"""
Example: Multi-agent graphs with tools.

Demonstrates several agents each equipped with their own tools:
- Two connected agents: Calculator (fibonacci) → Analyzer (is_prime, factorize, sum_digits)
- Two parallel agents: Math Agent (fibonacci) + Code Agent (code_interpreter)
- Chain of three: fibonacci → is_prime → sum_digits

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run with:
    python -m examples.multi_agent_tools_example
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
    create_openai_caller,
    get_registry,
    register_tool,
    tool,
)

# Setup framework logging
setup_logging(level="INFO")


# ── Register tools ──────────────────────────────────────────────────────────────


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
def factorize(n: int) -> str:
    """Return the prime factorisation of n (e.g. '2 × 3 × 5')."""
    if n <= 1:
        return str(n)
    factors: list[int] = []
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
    """Return the sum of all decimal digits of n."""
    return str(sum(int(d) for d in str(abs(n))))


register_tool(CodeInterpreterTool(timeout=10, safe_mode=True))


def create_llm():
    """Create LLM caller from environment variables."""
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


# ── Example 1: Two connected agents ─────────────────────────────────────────────


def example_two_connected_agents():
    """Two agents in series: Calculator → Analyzer."""
    print("\n" + "─" * 60)
    print("Example 1: Two Connected Agents")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="calculator",
        display_name="Calculator",
        persona="a calculator",
        description="I calculate Fibonacci numbers.",
        tools=["fibonacci"],
    )
    builder.add_agent(
        agent_id="analyzer",
        display_name="Analyzer",
        persona="a number analyzer",
        description="I analyze numbers using is_prime, factorize, and sum_digits.",
        tools=["is_prime", "factorize", "sum_digits"],
    )
    builder.add_task(query="Calculate fibonacci(20), then analyze the result.")
    builder.connect_task_to_agents(agent_ids=["calculator"])
    builder.add_edge(source="calculator", target="analyzer")

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 2: Parallel agents ──────────────────────────────────────────────────


def example_parallel_agents():
    """Two agents that receive the same task in parallel."""
    print("\n" + "─" * 60)
    print("Example 2: Parallel Agents")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="math_agent",
        display_name="Math Agent",
        persona="a math specialist",
        description="I calculate Fibonacci numbers.",
        tools=["fibonacci"],
    )
    builder.add_agent(
        agent_id="code_agent",
        display_name="Code Agent",
        persona="a Python programmer",
        description="I execute Python code.",
        tools=["code_interpreter"],
    )
    builder.add_task(
        query="Math Agent: calculate fibonacci(30). Code Agent: calculate 2**100"
    )
    builder.connect_task_to_agents(agent_ids=["math_agent", "code_agent"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 3: Chain of three ───────────────────────────────────────────────────


def example_chain_of_three():
    """Chain of three agents: fibonacci → is_prime → sum_digits."""
    print("\n" + "─" * 60)
    print("Example 3: Chain of Three Agents")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="fib_agent",
        display_name="Fibonacci Agent",
        persona="a Fibonacci calculator",
        description="I calculate Fibonacci numbers. Output ONLY the number.",
        tools=["fibonacci"],
    )
    builder.add_agent(
        agent_id="prime_agent",
        display_name="Prime Checker",
        persona="a prime number checker",
        description="I check if numbers are prime.",
        tools=["is_prime"],
    )
    builder.add_agent(
        agent_id="digit_agent",
        display_name="Digit Summer",
        persona="a digit sum calculator",
        description="I calculate the sum of digits.",
        tools=["sum_digits"],
    )
    builder.add_task(
        query="Calculate fibonacci(25), check if prime, then sum its digits."
    )
    builder.connect_task_to_agents(agent_ids=["fib_agent"])
    builder.add_edge(source="fib_agent", target="prime_agent")
    builder.add_edge(source="prime_agent", target="digit_agent")

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    """Run all examples."""
    get_registry()

    try:
        example_two_connected_agents()
        example_parallel_agents()
        example_chain_of_three()
        print("\n" + "=" * 60)
        print("All examples completed ✅")
    except Exception as e:
        logger.exception("Error running examples: %s", e)
        raise


if __name__ == "__main__":
    main()
