"""
Early stopping in a three-agent chain.

Demonstrates:
  - Chain topology: Analyzer → Solver → Validator
  - A custom condition checks the Solver's answer after it runs
  - If correct, the Validator is skipped (early stop) to save tokens

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.early_stop_example
"""

import os
import re

from rustworkx_framework.builder import BuilderConfig, GraphBuilder
from rustworkx_framework.execution import (
    EarlyStopCondition,
    MACPRunner,
    RunnerConfig,
    StepContext,
)
from rustworkx_framework.tools import create_openai_caller

# ── Constants ───────────────────────────────────────────────────────────────────

# The equation we want to solve: 2x + 5 = 13  →  x = 4
EQUATION = "2x + 5 = 13"
CORRECT_ANSWER = 4
MESSAGE_PREVIEW_LENGTH = 400


# ── Graph construction ────────────────────────────────────────────────────────


def _build_graph():
    """Build the chain: Task → Analyzer → Solver → Validator."""
    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))

    builder.add_task(
        query=f"Solve the equation: {EQUATION}",
        description="A linear equation to solve",
    )

    builder.add_agent(
        agent_id="analyzer",
        display_name="Analyzer",
        persona="a mathematical analyst",
        description=(
            "Analyse the problem and write a detailed solution plan. Do NOT solve it yourself — only outline the steps."
        ),
    )
    builder.add_agent(
        agent_id="solver",
        display_name="Solver",
        persona="a mathematics solver",
        description=(
            "Solve the equation following the plan from the previous agent. "
            'Always output the final answer as: "FINAL_ANSWER: x = <value>".'
        ),
    )
    builder.add_agent(
        agent_id="validator",
        display_name="Validator",
        persona="a checking mathematician",
        description=(
            "Verify the solution by substituting the value back into the equation. Confirm whether it is correct."
        ),
    )

    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver", "validator")
    builder.set_start_node("analyzer")
    builder.set_end_node("validator")

    return builder.build()


# ── Early-stop condition ──────────────────────────────────────────────────────


def _make_early_stop() -> EarlyStopCondition:
    """Stop after Solver if the answer is correct (x = 4)."""

    def _check(ctx: StepContext) -> bool:
        if ctx.agent_id != "solver":
            return False
        response = ctx.response or ""
        if "FINAL_ANSWER" not in response:
            return False

        tail = response.split("FINAL_ANSWER")[-1]
        for pattern in (r"x\s*=\s*(\d+)", r":\s*x\s*=\s*(\d+)"):
            m = re.search(pattern, tail)
            if m and int(m.group(1)) == CORRECT_ANSWER:
                return True
        return False

    return EarlyStopCondition.on_custom(
        condition=_check,
        reason="Solver produced the correct answer; Validator is not needed",
        min_agents_executed=2,
    )


# ── Execution ─────────────────────────────────────────────────────────────────


def main():
    graph = _build_graph()

    llm_caller = create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.7,
    )

    runner = MACPRunner(
        llm_caller=llm_caller,
        config=RunnerConfig(
            timeout=60.0,
            adaptive=False,
            update_states=True,
            broadcast_task_to_all=False,
            early_stop_conditions=[_make_early_stop()],
        ),
    )

    print(f"Task   : {graph.query}")
    print(f"Agents : {[a.agent_id for a in graph.agents]}\n")

    result = runner.run_round(graph, final_agent_id="validator")

    # Show agent outputs
    for agent_id in result.execution_order:
        msg = result.messages.get(agent_id, "")
        print(f"[{agent_id}]")
        print(msg[:MESSAGE_PREVIEW_LENGTH] + ("…" if len(msg) > MESSAGE_PREVIEW_LENGTH else ""))
        print()

    all_agents = ["analyzer", "solver", "validator"]
    skipped = [a for a in all_agents if a not in result.execution_order]

    if result.early_stopped:
        print(f"⚡ Early stop: {result.early_stop_reason}")
        if skipped:
            print(f"   Skipped   : {skipped}")
    else:
        print("No early stop — all agents executed.")

    print(f"\nFinal answer : {result.final_answer}")
    print(f"Total tokens : {result.total_tokens}")
    print(f"Total time   : {result.total_time:.2f}s")


if __name__ == "__main__":
    main()
