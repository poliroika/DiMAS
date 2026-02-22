"""
Benchmark: MECE vs LangGraph.

Runs three equivalent workflows on both frameworks and compares
execution time and token usage:

  Test 1 — Single agent (Math Solver)
  Test 2 — Chain of 3 agents (Analyzer → Solver → Formatter)
  Test 3 — Fan-in: 2 parallel agents → Aggregator

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.benchmark_vs_langgraph
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from rustworkx_framework.builder import BuilderConfig, GraphBuilder
from rustworkx_framework.execution import MACPRunner, RunnerConfig, StreamEventType

# ── LLM configuration ────────────────────────────────────────────────────────

API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

MATH_PROBLEM = "Solve: 3x^2 - 7x + 2 = 0. Find both roots."


# ── Shared LLM client with token tracking ─────────────────────────────────────


class LLMClient:
    """OpenAI-compatible client that tracks token usage."""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.total_tokens = 0
        self.call_count = 0

    def reset(self):
        self.total_tokens = 0
        self.call_count = 0

    def call(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        self.total_tokens += tokens
        self.call_count += 1
        return content


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mece_llm_caller(llm: LLMClient):
    """Return a simple callable that wraps ``llm.call``."""

    def caller(prompt: str) -> str:
        return llm.call("", prompt)

    return caller


def _last_agent_output(llm: LLMClient, graph, final_id: str) -> str:
    """Stream a MECE graph and return the last agent output."""
    runner = MACPRunner(
        llm_caller=_mece_llm_caller(llm),
        config=RunnerConfig(timeout=180.0),
    )
    output = ""
    for event in runner.stream(graph, final_agent_id=final_id):
        if event.event_type == StreamEventType.AGENT_OUTPUT and hasattr(event, "content"):
            output = event.content
    return output


# ── Test 1: Single agent ──────────────────────────────────────────────────────


def _single_agent_langgraph(llm: LLMClient, problem: str) -> dict:
    try:
        from typing import Any, TypedDict, cast

        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        output: str

    def solver_node(state: State) -> dict:
        return {
            "output": llm.call(
                "You are a math solver. Solve problems step by step.",
                state["input"],
            )
        }

    llm.reset()
    t0 = time.perf_counter()

    g = StateGraph(cast("Any", State))
    g.add_node("solver", solver_node)
    g.set_entry_point("solver")
    g.set_finish_point("solver")
    result = g.compile().invoke({"input": problem, "output": ""})

    return {
        "framework": "langgraph",
        "test": "single_agent",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
    }


def _single_agent_mece(llm: LLMClient, problem: str) -> dict:
    llm.reset()
    t0 = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task(query=problem, description="Math problem")
    builder.add_agent(
        agent_id="solver",
        display_name="Math Solver",
        persona="mathematician",
        description="Solve problems step by step and give a precise answer.",
    )
    builder.connect_task_to_agents(agent_ids=["solver"], bidirectional=False)
    graph = builder.build()

    output = _last_agent_output(llm, graph, "solver")

    return {
        "framework": "mece",
        "test": "single_agent",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
    }


# ── Test 2: Chain of 3 agents ─────────────────────────────────────────────────


def _chain_langgraph(llm: LLMClient, problem: str) -> dict:
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

    def analyzer(s: State) -> dict:
        return {"step1": llm.call("Identify the equation type and method.", f"Problem: {s['input']}")}

    def solver(s: State) -> dict:
        return {"step2": llm.call("Solve step by step.", f"Problem: {s['input']}\nAnalysis: {s['step1']}")}

    def formatter(s: State) -> dict:
        return {"output": llm.call("Present the answer clearly.", f"Solution: {s['step2']}")}

    llm.reset()
    t0 = time.perf_counter()

    g = StateGraph(cast("Any", State))
    for name, fn in [("analyzer", analyzer), ("solver", solver), ("formatter", formatter)]:
        g.add_node(name, fn)
    g.add_edge("analyzer", "solver")
    g.add_edge("solver", "formatter")
    g.set_entry_point("analyzer")
    g.set_finish_point("formatter")
    result = g.compile().invoke({"input": problem, "step1": "", "step2": "", "output": ""})

    return {
        "framework": "langgraph",
        "test": "chain_3",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
    }


def _chain_mece(llm: LLMClient, problem: str) -> dict:
    llm.reset()
    t0 = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task(query=problem, description="Math problem")
    builder.add_agent("analyzer", "Analyzer", "analyst", "Identify the equation type and solution method.")
    builder.add_agent("solver", "Solver", "solver", "Solve the problem step by step.")
    builder.add_agent("formatter", "Formatter", "formatter", "Present the final answer clearly and concisely.")
    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver", "formatter")
    graph = builder.build()

    output = _last_agent_output(llm, graph, "formatter")

    return {
        "framework": "mece",
        "test": "chain_3",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
    }


# ── Test 3: Fan-in (2 → 1) ───────────────────────────────────────────────────


def _fanin_langgraph(llm: LLMClient, problem: str) -> dict:
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

    def algebraic(s: State) -> dict:
        return {"method1": llm.call("Solve using the discriminant formula.", f"Problem: {s['input']}")}

    def vieta(s: State) -> dict:
        return {"method2": llm.call("Solve using Vieta's formulas.", f"Problem: {s['input']}")}

    def aggregator(s: State) -> dict:
        return {
            "output": llm.call(
                "Compare two solutions and give the definitive answer.",
                f"Method 1: {s['method1']}\n\nMethod 2: {s['method2']}",
            )
        }

    llm.reset()
    t0 = time.perf_counter()

    g = StateGraph(cast("Any", State))
    for name, fn in [("algebraic", algebraic), ("vieta", vieta), ("aggregator", aggregator)]:
        g.add_node(name, fn)
    g.add_edge("algebraic", "vieta")
    g.add_edge("vieta", "aggregator")
    g.set_entry_point("algebraic")
    g.set_finish_point("aggregator")
    result = g.compile().invoke({"input": problem, "method1": "", "method2": "", "output": ""})

    return {
        "framework": "langgraph",
        "test": "fan_in",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": result["output"],
    }


def _fanin_mece(llm: LLMClient, problem: str) -> dict:
    llm.reset()
    t0 = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task(query=problem, description="Math problem")
    builder.add_agent("algebraic", "Algebraic Solver", "algebraist", "Solve using the discriminant formula.")
    builder.add_agent("vieta", "Vieta Solver", "vieta", "Solve using Vieta's formulas.")
    builder.add_agent("aggregator", "Aggregator", "aggregator", "Compare two solutions and give the definitive answer.")
    builder.connect_task_to_agents(agent_ids=["algebraic", "vieta"], bidirectional=False)
    builder.add_workflow_edge("algebraic", "aggregator")
    builder.add_workflow_edge("vieta", "aggregator")
    graph = builder.build()

    output = _last_agent_output(llm, graph, "aggregator")

    return {
        "framework": "mece",
        "test": "fan_in",
        "time": time.perf_counter() - t0,
        "tokens": llm.total_tokens,
        "calls": llm.call_count,
        "output": output,
    }


# ── Main benchmark ────────────────────────────────────────────────────────────


def run_benchmark() -> dict:
    """Run all benchmark pairs and print a comparison table."""
    llm = LLMClient()
    results: list[dict] = []

    tests = [
        ("Single Agent", _single_agent_langgraph, _single_agent_mece),
        ("Chain (3 agents)", _chain_langgraph, _chain_mece),
        ("Fan-in (2→1)", _fanin_langgraph, _fanin_mece),
    ]

    print(f"Problem : {MATH_PROBLEM}")
    print(f"Model   : {MODEL}\n")
    print(f"{'Test':<20} {'Framework':<12} {'Time':>8} {'Tokens':>8} {'Calls':>6}")
    print("─" * 60)

    for name, lg_fn, mece_fn in tests:
        lg = lg_fn(llm, MATH_PROBLEM)
        if "error" in lg:
            print(f"  {name:<20} LangGraph skipped: {lg['error']}")
        else:
            print(f"  {name:<20} {'langgraph':<12} {lg['time']:>7.2f}s {lg['tokens']:>8} {lg['calls']:>6}")
        results.append(lg)

        mece = mece_fn(llm, MATH_PROBLEM)
        print(f"  {'':20} {'mece':<12} {mece['time']:>7.2f}s {mece['tokens']:>8} {mece['calls']:>6}")
        results.append(mece)

        if "error" not in lg:
            dt = (lg["time"] - mece["time"]) / lg["time"] * 100
            dk = (lg["tokens"] - mece["tokens"]) / max(lg["tokens"], 1) * 100
            print(f"  {'':20} {'delta':<12} time {dt:+.1f}%  tokens {dk:+.1f}%")
        print()

    # Save results
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "problem": MATH_PROBLEM,
        "model": MODEL,
        "results": results,
    }
    log_dir = Path("benchmark_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Results saved → {log_file}")

    return output


if __name__ == "__main__":
    run_benchmark()
