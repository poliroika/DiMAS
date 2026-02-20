"""
Benchmark: MECE vs LangGraph.

Runs three equivalent workflows on both frameworks and compares
execution time and token usage:

  Test 1 — Single agent (Math Solver)
  Test 2 — Chain of 3 agents (Analyzer -> Solver -> Formatter)
  Test 3 — Fan-in: 2 parallel agents -> Aggregator
            (LangGraph runs them sequentially due to its fan-in limitations)

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run with:
    python examples/benchmark_vs_langgraph.py
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

# ── LLM configuration ────────────────────────────────────────────────────────
DEFAULT_API_KEY  = ""  # set LLM_API_KEY env var
DEFAULT_BASE_URL = ""  # set LLM_BASE_URL env var
DEFAULT_MODEL    = ""  # set LLM_MODEL env var

import os  # noqa: E402
DEFAULT_API_KEY  = os.getenv("LLM_API_KEY",  "your-api-key")
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL    = os.getenv("LLM_MODEL",    "gpt-4o-mini")

MATH_PROBLEM = "Solve: 3x^2 - 7x + 2 = 0. Find both roots."


# ── Shared LLM client with token tracking ────────────────────────────────────

class LLMClient:
    """OpenAI-compatible client that tracks token usage and call history."""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.total_tokens = 0
        self.call_count = 0
        self.history: list[dict] = []

    def reset(self):
        """Reset counters between benchmark runs."""
        self.total_tokens = 0
        self.call_count = 0
        self.history = []

    def call(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM and accumulate token / timing stats."""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - start

        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        self.total_tokens += tokens
        self.call_count += 1
        self.history.append({
            "system":   system_prompt[:200],
            "user":     user_message[:200],
                "response": content,
            "tokens":   tokens,
            "time":     elapsed,
        })
        return content


# ── Test 1: Single agent ──────────────────────────────────────────────────────

def test_single_agent_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: one Math Solver node."""
    try:
        from typing import Any, TypedDict, cast
        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        output: str

    def solver_node(state: State) -> dict:
        result = llm.call(
            "You are a math solver. Solve problems step by step and give a precise answer.",
            state["input"],
        )
        return {"output": result}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(cast("Any", State))
    graph.add_node("solver", solver_node)
    graph.set_entry_point("solver")
    graph.set_finish_point("solver")
    app = graph.compile()

    result = app.invoke({"input": problem, "output": ""})
    elapsed = time.perf_counter() - start

    return {
        "framework": "langgraph",
        "test": "single_agent",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
        "history": llm.history.copy(),
    }


def test_single_agent_mece(llm: LLMClient, problem: str) -> dict:
    """MECE: one Math Solver agent."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")
    builder.add_agent(
        agent_id="solver",
        display_name="Math Solver",
        persona="mathematician",
        description="You are a math solver. Solve problems step by step and give a precise answer.",
    )
    builder.connect_task_to_agents(agent_ids=["solver"], bidirectional=False)
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=120.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="solver"):
        if event.event_type == StreamEventType.AGENT_OUTPUT and hasattr(event, "content"):
            output = event.content

    elapsed = time.perf_counter() - start

    return {
        "framework": "mece",
        "test": "single_agent",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
        "history": llm.history.copy(),
    }


# ── Test 2: Chain of 3 agents ─────────────────────────────────────────────────

def test_chain_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: Analyzer -> Solver -> Formatter chain."""
    try:
        from typing import Any, TypedDict, cast
        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        step1: str
        step2: str
        output: str

    def analyzer(state: State) -> dict:
        return {"step1": llm.call(
            "You are a problem analyst. Identify the equation type and solution method.",
            f"Problem: {state['input']}",
        )}

    def solver(state: State) -> dict:
        return {"step2": llm.call(
            "You are an equation solver. Solve the problem step by step.",
            f"Problem: {state['input']}\nAnalysis: {state['step1']}",
        )}

    def formatter(state: State) -> dict:
        return {"output": llm.call(
            "You are a result formatter. Present the final answer clearly and concisely.",
            f"Solution: {state['step2']}",
        )}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(cast("Any", State))
    graph.add_node("analyzer",  analyzer)
    graph.add_node("solver",    solver)
    graph.add_node("formatter", formatter)
    graph.add_edge("analyzer", "solver")
    graph.add_edge("solver", "formatter")
    graph.set_entry_point("analyzer")
    graph.set_finish_point("formatter")
    app = graph.compile()

    result = app.invoke({"input": problem, "step1": "", "step2": "", "output": ""})
    elapsed = time.perf_counter() - start

    return {
        "framework": "langgraph",
        "test": "chain_3",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
        "history": llm.history.copy(),
    }


def test_chain_mece(llm: LLMClient, problem: str) -> dict:
    """MECE: Analyzer -> Solver -> Formatter chain."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")
    builder.add_agent(
        "analyzer",  "Analyzer",  "analyst",
        "You are a problem analyst. Identify the equation type and solution method.",
    )
    builder.add_agent(
        "solver",    "Solver",    "solver",
        "You are an equation solver. Solve the problem step by step.",
    )
    builder.add_agent(
        "formatter", "Formatter", "formatter",
        "You are a result formatter. Present the final answer clearly and concisely.",
    )
    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver",   "formatter")
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=180.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="formatter"):
        if event.event_type == StreamEventType.AGENT_OUTPUT and hasattr(event, "content"):
            output = event.content

    elapsed = time.perf_counter() - start

    return {
        "framework": "mece",
        "test": "chain_3",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
        "history": llm.history.copy(),
    }


# ── Test 3: Fan-in (2 -> 1) ───────────────────────────────────────────────────

def test_fanin_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: Algebraic + Vieta -> Aggregator (sequential due to LangGraph limitations)."""
    try:
        from typing import Any, TypedDict, cast
        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        method1: str
        method2: str
        output: str

    def algebraic(state: State) -> dict:
        return {"method1": llm.call(
            "Solve the quadratic equation using the discriminant formula.",
            f"Problem: {state['input']}",
        )}

    def vieta(state: State) -> dict:
        return {"method2": llm.call(
            "Solve the quadratic equation using Vieta's formulas.",
            f"Problem: {state['input']}",
        )}

    def aggregator(state: State) -> dict:
        return {"output": llm.call(
            "Compare two solutions and give the definitive correct answer.",
            f"Method 1 (discriminant): {state['method1']}\n\nMethod 2 (Vieta): {state['method2']}",
        )}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(cast("Any", State))
    graph.add_node("algebraic",  algebraic)
    graph.add_node("vieta",      vieta)
    graph.add_node("aggregator", aggregator)
    # LangGraph runs them sequentially (no native fan-in parallelism)
    graph.add_edge("algebraic", "vieta")
    graph.add_edge("vieta", "aggregator")
    graph.set_entry_point("algebraic")
    graph.set_finish_point("aggregator")
    app = graph.compile()

    result = app.invoke({"input": problem, "method1": "", "method2": "", "output": ""})
    elapsed = time.perf_counter() - start

    return {
        "framework": "langgraph",
        "test": "fan_in",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
        "history": llm.history.copy(),
    }


def test_fanin_mece(llm: LLMClient, problem: str) -> dict:
    """MECE: Algebraic + Vieta run in parallel, then feed into Aggregator."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")
    builder.add_agent(
        "algebraic",  "Algebraic Solver", "algebraist",
        "Solve the quadratic equation using the discriminant formula.",
    )
    builder.add_agent(
        "vieta",      "Vieta Solver",     "vietaist",
        "Solve the quadratic equation using Vieta's formulas.",
    )
    builder.add_agent(
        "aggregator", "Aggregator",       "aggregator",
        "Compare two solutions and give the definitive correct answer.",
    )
    builder.connect_task_to_agents(agent_ids=["algebraic", "vieta"], bidirectional=False)
    builder.add_workflow_edge("algebraic", "aggregator")
    builder.add_workflow_edge("vieta",     "aggregator")
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=180.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="aggregator"):
        if event.event_type == StreamEventType.AGENT_OUTPUT and hasattr(event, "content"):
            output = event.content

    elapsed = time.perf_counter() - start

    return {
        "framework": "mece",
        "test": "fan_in",
        "time": elapsed,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
        "history": llm.history.copy(),
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark() -> dict:
    """Run all benchmark pairs and save results to JSON."""
    llm = LLMClient()
    results: list[dict] = []

    tests = [
        ("Single Agent",    test_single_agent_langgraph, test_single_agent_mece),
        ("Chain (3 agents)",test_chain_langgraph,        test_chain_mece),
        ("Fan-in (2->1)",   test_fanin_langgraph,        test_fanin_mece),
    ]

    print(f"Problem: {MATH_PROBLEM}\nModel  : {DEFAULT_MODEL}\n")
    print(f"{'Test':<20} {'Framework':<12} {'Time':>8} {'Tokens':>8} {'Calls':>6}")
    print("-" * 60)

    for name, lg_test, mece_test in tests:
        lg_result = lg_test(llm, MATH_PROBLEM)
        if "error" in lg_result:
            print(f"  {name:<20} LangGraph skipped: {lg_result['error']}")
        else:
            print(
                f"  {name:<20} {'langgraph':<12} "
                f"{lg_result['time']:>7.2f}s "
                f"{lg_result['tokens']:>8} "
                f"{lg_result['calls']:>6}"
            )
        results.append(lg_result)

        mece_result = mece_test(llm, MATH_PROBLEM)
        print(
            f"  {'':20} {'mece':<12} "
            f"{mece_result['time']:>7.2f}s "
            f"{mece_result['tokens']:>8} "
            f"{mece_result['calls']:>6}"
        )
        results.append(mece_result)

        # Time and token comparison
        if "error" not in lg_result:
            time_diff_pct  = (lg_result["time"]   - mece_result["time"])   / lg_result["time"]   * 100
            token_diff_pct = (lg_result["tokens"] - mece_result["tokens"]) / max(lg_result["tokens"], 1) * 100
            print(f"  {'':20} {'MECE vs LG':<12} time {time_diff_pct:+.1f}%  tokens {token_diff_pct:+.1f}%")
        print()

    # ── Averages ─────────────────────────────────────────────────────────────
    mece_results = [r for r in results if r.get("framework") == "mece"]
    lg_results   = [r for r in results if r.get("framework") == "langgraph" and "error" not in r]

    if lg_results and mece_results:
        avg_mece_time   = sum(r["time"]   for r in mece_results) / len(mece_results)
        avg_lg_time     = sum(r["time"]   for r in lg_results)   / len(lg_results)
        avg_mece_tokens = sum(r["tokens"] for r in mece_results) / len(mece_results)
        avg_lg_tokens   = sum(r["tokens"] for r in lg_results)   / len(lg_results)

        print("Averages:")
        print(f"  MECE     — time: {avg_mece_time:.2f}s  tokens: {avg_mece_tokens:.0f}")
        print(f"  LangGraph — time: {avg_lg_time:.2f}s  tokens: {avg_lg_tokens:.0f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "problem": MATH_PROBLEM,
        "model": DEFAULT_MODEL,
        "results": results,
    }

    log_dir = Path(__file__).parent / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved -> {log_file}")

    return output


if __name__ == "__main__":
    run_benchmark()
