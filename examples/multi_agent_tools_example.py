"""
Multi-agent graphs with tools.

Demonstrates several agents each equipped with their own tools:
  1. Two connected agents: Calculator (fibonacci) → Analyzer (is_prime, factorize, sum_digits)
  2. Two parallel agents: Math Agent (fibonacci) + Code Agent (code_interpreter)
  3. Chain of three: fibonacci → is_prime → sum_digits

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.multi_agent_tools_example
"""

import math
import os

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import (
    CodeInterpreterTool,
    create_openai_caller,
    register_tool,
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
    """Check if a number is prime."""
    if n < MIN_PRIME_NUMBER:
        return "False"
    for i in range(MIN_PRIME_NUMBER, math.isqrt(n) + 1):
        if n % i == 0:
            return "False"
    return "True"


@tool
def factorize(n: int) -> str:
    """Return the prime factorisation of n (e.g. '2 x 3 x 5')."""
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_llm():
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


def _header(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── Example 1: Two connected agents ─────────────────────────────────────────


def example_two_connected():
    """Calculator → Analyzer pipeline."""
    _header("1 · Two Connected Agents")

    builder = GraphBuilder()
    builder.add_agent("calculator", "Calculator", "a calculator", "Calculates Fibonacci numbers.", tools=["fibonacci"])
    builder.add_agent(
        "analyzer",
        "Analyzer",
        "a number analyzer",
        "Analyses numbers using is_prime, factorize, and sum_digits.",
        tools=["is_prime", "factorize", "sum_digits"],
    )
    builder.add_task(query="Calculate fibonacci(20), then analyse the result.")
    builder.connect_task_to_agents(agent_ids=["calculator"])
    builder.add_edge(source="calculator", target="analyzer")

    result = MACPRunner(llm_caller=_create_llm()).run_round(builder.build())
    print(f"  Result: {result.final_answer}")


# ── Example 2: Parallel agents ──────────────────────────────────────────────


def example_parallel():
    """Two agents receive the same task in parallel."""
    _header("2 · Parallel Agents")

    builder = GraphBuilder()
    builder.add_agent(
        "math_agent", "Math Agent", "a math specialist", "Calculates Fibonacci numbers.", tools=["fibonacci"]
    )
    builder.add_agent(
        "code_agent", "Code Agent", "a Python programmer", "Executes Python code.", tools=["code_interpreter"]
    )
    builder.add_task(query="Math Agent: fibonacci(30). Code Agent: 2**100")
    builder.connect_task_to_agents(agent_ids=["math_agent", "code_agent"])

    result = MACPRunner(llm_caller=_create_llm()).run_round(builder.build())
    print(f"  Result: {result.final_answer}")


# ── Example 3: Chain of three ───────────────────────────────────────────────


def example_chain():
    """Fibonacci → is_prime → sum_digits chain."""
    _header("3 · Chain of Three Agents")

    builder = GraphBuilder()
    builder.add_agent(
        "fib_agent",
        "Fibonacci Agent",
        "a Fibonacci calculator",
        "Calculates Fibonacci numbers. Output ONLY the number.",
        tools=["fibonacci"],
    )
    builder.add_agent(
        "prime_agent", "Prime Checker", "a prime checker", "Checks if numbers are prime.", tools=["is_prime"]
    )
    builder.add_agent(
        "digit_agent", "Digit Summer", "a digit sum calculator", "Calculates the sum of digits.", tools=["sum_digits"]
    )
    builder.add_task(query="Calculate fibonacci(25), check if prime, then sum its digits.")
    builder.connect_task_to_agents(agent_ids=["fib_agent"])
    builder.add_edge(source="fib_agent", target="prime_agent")
    builder.add_edge(source="prime_agent", target="digit_agent")

    result = MACPRunner(llm_caller=_create_llm()).run_round(builder.build())
    print(f"  Result: {result.final_answer}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    example_two_connected()
    example_parallel()
    example_chain()

    print(f"\n{'=' * 60}")
    print("All multi-agent tool examples completed ✅")


if __name__ == "__main__":
    main()
