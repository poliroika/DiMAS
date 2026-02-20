"""
Example: Early stopping in a three-agent chain.

Demonstrates:
- Chain: Analyzer -> Solver -> Validator
- After Solver runs, a custom condition checks whether the answer is correct
- If correct, the Validator is skipped (early stop)
- Full dialogue is saved to JSON

Use case: save tokens when an intermediate agent already produces the right answer.

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run with:
    python -m examples.early_stop_example
"""

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.execution.runner import (
    EarlyStopCondition,
    MACPRunner,
    RunnerConfig,
    StepContext,
)
from rustworkx_framework.tools import create_openai_caller

# Setup framework logging
setup_logging(level="INFO")

# The equation we want to solve: 2x + 5 = 13  =>  x = 4
EQUATION = "2x + 5 = 13"
CORRECT_ANSWER = 4


# ── Graph construction ────────────────────────────────────────────────────────

def build_three_agent_graph():
    """Build the chain: Analyzer -> Solver -> Validator."""
    config = BuilderConfig(include_task_node=True, validate=True)
    builder = GraphBuilder(config)

    builder.add_task(
        task_id="__task__",
        query=f"Solve the equation: {EQUATION}",
        description="A linear equation to solve",
    )

    builder.add_agent(
        agent_id="analyzer",
        display_name="Analyzer",
        persona="a mathematical analyst",
        description=(
            "Analyse the mathematical problem and write a detailed solution plan. "
            "Do NOT solve it yourself -- only outline the steps."
        ),
    )

    builder.add_agent(
        agent_id="solver",
        display_name="Solver",
        persona="a mathematics solver",
        description=(
            "Solve the equation following the plan from the previous agent. "
            "Perform every step and ALWAYS output the final answer in the format: "
            '"FINAL_ANSWER: x = <value>".'
        ),
    )

    builder.add_agent(
        agent_id="validator",
        display_name="Validator",
        persona="a checking mathematician",
        description=(
            "Verify the solution by substituting the value back into the equation. "
            "Confirm whether it is correct."
        ),
    )

    # Chain: task -> analyzer -> solver -> validator
    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver", "validator")

    builder.set_start_node("analyzer")
    builder.set_end_node("validator")

    return builder.build()


# ── Early-stop condition ──────────────────────────────────────────────────────

def create_early_stop_condition() -> EarlyStopCondition:
    """Return a condition that stops after Solver if the answer is correct."""

    def check_answer_correct(ctx: StepContext) -> bool:
        """Return True when Solver produces the correct answer (x = 4)."""
        if ctx.agent_id != "solver":
            return False

        response = ctx.response or ""
        if "FINAL_ANSWER" not in response:
            return False

        final_part = response.split("FINAL_ANSWER")[-1]

        # Accept several answer formats: x = 4, x = (4), x = \(4\), etc.
        patterns = [
            r"x\s*=\s*(\d+)",
            r"x\s*=\s*\\?\(\s*(\d+)\s*\\?\)",
            r":\s*x\s*=\s*(\d+)",
        ]

        for pattern in patterns:
            m = re.search(pattern, final_part)
            if m:
                return int(m.group(1)) == CORRECT_ANSWER

            return False

    return EarlyStopCondition.on_custom(
        condition=check_answer_correct,
        reason="Solver produced the correct answer; Validator is not needed",
        min_agents_executed=2,  # at least Analyzer + Solver must have run
    )


# ── Execution ─────────────────────────────────────────────────────────────────

def run_with_early_stop() -> dict:
    """Run the chain with early stopping and save the log to JSON."""
    graph = build_three_agent_graph()
    llm_caller = create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.7,
    )
    early_stop = create_early_stop_condition()

    runner = MACPRunner(
        llm_caller=llm_caller,
        config=RunnerConfig(
            timeout=60.0,
            adaptive=False,
            update_states=True,
            broadcast_task_to_all=False,
            early_stop_conditions=[early_stop],
        ),
    )

    print(f"Task: {graph.query}")
    print(f"Agents: {[a.agent_id for a in graph.agents]}\n")

    result = runner.run_round(graph, final_agent_id="validator")

    # ── Display agent outputs ─────────────────────────────────────────────────
    all_agents = ["analyzer", "solver", "validator"]
    for agent_id in result.execution_order:
        msg = result.messages.get(agent_id, "")
        print(f"[{agent_id}]")
        print(msg[:400] + ("..." if len(msg) > 400 else ""))
        print()

    skipped = [a for a in all_agents if a not in result.execution_order]

    if result.early_stopped:
        print(f"Early stop triggered: {result.early_stop_reason}")
        if skipped:
            print(f"Skipped agents: {skipped}")
    else:
        print("No early stop — all agents executed.")

    print(f"\nFinal answer : {result.final_answer}")
    print(f"Total tokens : {result.total_tokens}")
    print(f"Total time   : {result.total_time:.2f}s")

    # ── Persist communication log ─────────────────────────────────────────────
    communication_log = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": "Early Stopping Example",
        "task": graph.query,
        "graph_structure": {
            "nodes": [a.agent_id for a in graph.agents],
            "edges": [("analyzer", "solver"), ("solver", "validator")],
            "start_node": graph.start_node,
            "end_node": graph.end_node,
        },
        "execution": {
            "execution_order": result.execution_order,
            "early_stopped": result.early_stopped,
            "early_stop_reason": result.early_stop_reason,
            "total_agents": len(graph.agents),
            "executed_agents": len(result.execution_order),
            "skipped_agents": skipped,
        },
        "messages": result.messages,
        "final_answer": result.final_answer,
        "final_agent_id": result.final_agent_id,
        "metrics": {
            "total_tokens": result.total_tokens,
            "total_time": result.total_time,
            "tokens_saved_estimate": len(skipped) * 500,
        },
    }

    log_path = Path(__file__).parent / "early_stop_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(communication_log, f, ensure_ascii=False, indent=2)
    print(f"\nLog saved -> {log_path}")

    return communication_log


def run_comparison():
    """Compare early-stop vs full execution token counts."""
    print("=" * 50)
    print("Scenario: WITH early stop")
    print("=" * 50)
    result = run_with_early_stop()

    executed = len(result["execution"]["execution_order"])
    total = result["execution"]["total_agents"]
    saved = total - executed

    print(f"\n  Agents executed : {executed} / {total}")
    print(f"  Agents skipped  : {saved}")
    print(f"  Estimated tokens saved: ~{result['metrics']['tokens_saved_estimate']}")


if __name__ == "__main__":
    # run_comparison()  # uncomment to see the side-by-side comparison
    run_with_early_stop()
