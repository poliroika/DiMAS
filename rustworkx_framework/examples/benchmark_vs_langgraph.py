"""Benchmark: MECE vs LangGraph comparison.

Tests:
1. Single agent (Math Solver)
2. Chain of 3 agents (A → B → C)
3. Fan-in: 2 agents → aggregator
"""

import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ============== CONFIG ==============
DEFAULT_API_KEY = "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"
DEFAULT_BASE_URL = "https://advisory-locked-summaries-steady.trycloudflare.com/v1"
DEFAULT_MODEL = "./models/Qwen3-Next-80B-A3B-Instruct"

MATH_PROBLEM = "Реши уравнение: 3x² - 7x + 2 = 0. Найди оба корня."


# ============== SHARED LLM CLIENT WITH TOKEN COUNTING ==============
class LLMClient:
    """Unified LLM client with token tracking."""

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
        self.total_tokens = 0
        self.call_count = 0
        self.history = []

    def call(self, system_prompt: str, user_message: str) -> str:
        """Call LLM and track tokens."""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - start

        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        self.total_tokens += tokens
        self.call_count += 1
        self.history.append(
            {
                "system": system_prompt[:200],
                "user": user_message[:200],
                "response": content,
                "tokens": tokens,
                "time": elapsed,
            }
        )
        return content


# ============== TEST 1: SINGLE AGENT ==============
def test_single_agent_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: single agent."""
    try:
        from typing import TypedDict

        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        output: str

    def solver_node(state: State) -> dict:
        result = llm.call(
            "Ты математический решатель. Решай задачи пошагово и давай точный ответ.",
            state["input"],
        )
        return {"output": result}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(State)
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
    """MECE: single agent."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    # Build graph
    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")
    builder.add_agent(
        agent_id="solver",
        display_name="Math Solver",
        persona="математик",
        description="Ты математический решатель. Решай задачи пошагово и давай точный ответ.",
    )
    builder.connect_task_to_agents(agent_ids=["solver"], bidirectional=False)
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)  # System prompt is in the description

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=120.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="solver"):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
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


# ============== TEST 2: CHAIN OF 3 AGENTS ==============
def test_chain_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: chain A → B → C."""
    try:
        from typing import TypedDict

        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        step1: str
        step2: str
        output: str

    def analyzer(state: State) -> dict:
        result = llm.call(
            "Ты анализатор задач. Проанализируй задачу и определи тип уравнения и метод решения.",
            f"Задача: {state['input']}",
        )
        return {"step1": result}

    def solver(state: State) -> dict:
        result = llm.call(
            "Ты решатель уравнений. Реши задачу пошагово используя указанный метод.",
            f"Задача: {state['input']}\nАнализ: {state['step1']}",
        )
        return {"step2": result}

    def formatter(state: State) -> dict:
        result = llm.call(
            "Ты форматтер ответов. Оформи финальный ответ кратко и понятно.",
            f"Решение: {state['step2']}",
        )
        return {"output": result}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(State)
    graph.add_node("analyzer", analyzer)
    graph.add_node("solver", solver)
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
    """MECE: chain A → B → C."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")

    builder.add_agent(
        "analyzer",
        "Analyzer",
        "анализатор",
        "Ты анализатор задач. Проанализируй задачу и определи тип уравнения и метод решения.",
    )
    builder.add_agent(
        "solver",
        "Solver",
        "решатель",
        "Ты решатель уравнений. Реши задачу пошагово используя указанный метод.",
    )
    builder.add_agent(
        "formatter",
        "Formatter",
        "форматтер",
        "Ты форматтер ответов. Оформи финальный ответ кратко и понятно.",
    )

    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver", "formatter")
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=180.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="formatter"):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
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


# ============== TEST 3: FAN-IN (2 → 1) ==============
def test_fanin_langgraph(llm: LLMClient, problem: str) -> dict:
    """LangGraph: 2 parallel agents → aggregator (sequential due to LangGraph limitations)."""
    try:
        from typing import TypedDict

        from langgraph.graph import StateGraph
    except ImportError:
        return {"error": "langgraph not installed"}

    class State(TypedDict):
        input: str
        method1: str
        method2: str
        output: str

    def algebraic_solver(state: State) -> dict:
        result = llm.call(
            "Реши уравнение алгебраически через дискриминант.", f"Задача: {state['input']}"
        )
        return {"method1": result}

    def vieta_solver(state: State) -> dict:
        result = llm.call("Реши уравнение используя теорему Виета.", f"Задача: {state['input']}")
        return {"method2": result}

    def aggregator(state: State) -> dict:
        result = llm.call(
            "Сравни два решения и дай финальный верный ответ.",
            f"Метод 1 (дискриминант): {state['method1']}\n\nМетод 2 (Виета): {state['method2']}",
        )
        return {"output": result}

    llm.reset()
    start = time.perf_counter()

    graph = StateGraph(State)
    graph.add_node("algebraic", algebraic_solver)
    graph.add_node("vieta", vieta_solver)
    graph.add_node("aggregator", aggregator)
    # Sequential: algebraic → vieta → aggregator (LangGraph limitation for fan-in)
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
    """MECE: 2 parallel agents → aggregator."""
    from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
    from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
    from rustworkx_framework.execution.streaming import StreamEventType

    llm.reset()
    start = time.perf_counter()

    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))
    builder.add_task("__task__", query=problem, description="Math problem")

    builder.add_agent(
        "algebraic",
        "Algebraic Solver",
        "алгебраист",
        "Реши уравнение алгебраически через дискриминант.",
    )
    builder.add_agent("vieta", "Vieta Solver", "виетист", "Реши уравнение используя теорему Виета.")
    builder.add_agent(
        "aggregator", "Aggregator", "агрегатор", "Сравни два решения и дай финальный верный ответ."
    )

    builder.connect_task_to_agents(agent_ids=["algebraic", "vieta"], bidirectional=False)
    builder.add_workflow_edge("algebraic", "aggregator")
    builder.add_workflow_edge("vieta", "aggregator")
    graph = builder.build()

    def llm_caller(prompt: str) -> str:
        return llm.call("", prompt)

    runner = MACPRunner(llm_caller=llm_caller, config=RunnerConfig(timeout=180.0))

    output = ""
    for event in runner.stream(graph, final_agent_id="aggregator"):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
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


# ============== MAIN BENCHMARK ==============
def run_benchmark():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("BENCHMARK: MECE vs LangGraph")
    print("=" * 60)

    llm = LLMClient()
    results = []

    tests = [
        ("Single Agent", test_single_agent_langgraph, test_single_agent_mece),
        ("Chain (3 agents)", test_chain_langgraph, test_chain_mece),
        ("Fan-in (2→1)", test_fanin_langgraph, test_fanin_mece),
    ]

    for name, lg_test, mece_test in tests:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print("=" * 60)

        print("\n🔵 Running LangGraph...")
        lg_result = lg_test(llm, MATH_PROBLEM)
        if "error" not in lg_result:
            print(
                f"   Time: {lg_result['time']:.2f}s, Tokens: {lg_result['tokens']}, Calls: {lg_result['calls']}"
            )
        else:
            print(f"   ⚠️  {lg_result['error']}")
        results.append(lg_result)

        print("\n🟢 Running MECE...")
        mece_result = mece_test(llm, MATH_PROBLEM)
        print(
            f"   Time: {mece_result['time']:.2f}s, Tokens: {mece_result['tokens']}, Calls: {mece_result['calls']}"
        )
        results.append(mece_result)

        # Comparison
        if "error" not in lg_result:
            time_diff = ((lg_result["time"] - mece_result["time"]) / lg_result["time"]) * 100
            token_diff = (
                (lg_result["tokens"] - mece_result["tokens"]) / max(lg_result["tokens"], 1)
            ) * 100
            print("\n📊 Comparison:")
            print(
                f"   Time: MECE is {abs(time_diff):.1f}% {'faster' if time_diff > 0 else 'slower'}"
            )
            print(
                f"   Tokens: MECE uses {abs(token_diff):.1f}% {'fewer' if token_diff > 0 else 'more'}"
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mece_results = [r for r in results if r.get("framework") == "mece"]
    lg_results = [r for r in results if r.get("framework") == "langgraph" and "error" not in r]

    if lg_results:
        avg_mece_time = sum(r["time"] for r in mece_results) / len(mece_results)
        avg_lg_time = sum(r["time"] for r in lg_results) / len(lg_results)
        avg_mece_tokens = sum(r["tokens"] for r in mece_results) / len(mece_results)
        avg_lg_tokens = sum(r["tokens"] for r in lg_results) / len(lg_results)

        print(f"\nAverage Time:   MECE={avg_mece_time:.2f}s  LangGraph={avg_lg_time:.2f}s")
        print(f"Average Tokens: MECE={avg_mece_tokens:.0f}    LangGraph={avg_lg_tokens:.0f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "problem": MATH_PROBLEM,
        "model": DEFAULT_MODEL,
        "results": results,
    }

    log_dir = Path(__file__).parent / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📁 Results saved to: {log_file}")
    return output


if __name__ == "__main__":
    run_benchmark()
