"""
Agent chain — Task → Math Researcher → Math Solver.

Demonstrates:
  - Building a two-agent sequential chain
  - Streaming execution to capture per-agent prompts and responses
  - Saving the communication log to JSON

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.math_chain_example
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from rustworkx_framework.builder import BuilderConfig, GraphBuilder
from rustworkx_framework.execution import MACPRunner, RunnerConfig, StreamEventType
from rustworkx_framework.tools import create_openai_caller

# ── Graph construction ────────────────────────────────────────────────────────


def _build_graph():
    """Build: Task → Math Researcher → Math Solver."""
    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True))

    builder.add_task(
        query="Solve the equation: 2x - 3x² = 1",
        description="Mathematical problem to solve",
    )

    builder.add_agent(
        agent_id="math_researcher",
        display_name="Math Researcher",
        persona="a mathematical researcher",
        description=(
            "Outline the steps required to solve the problem but do NOT compute the final answer — only the plan."
        ),
    )
    builder.add_agent(
        agent_id="math_solver",
        display_name="Math Solver",
        persona="a mathematics solver",
        description="Follow the plan and output the CORRECT ANSWER.",
    )

    builder.connect_task_to_agents(agent_ids=["math_researcher"], bidirectional=False)
    builder.add_workflow_edge("math_researcher", "math_solver")

    return builder.build()


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
            timeout=120.0,
            adaptive=False,
            update_states=True,
            broadcast_task_to_all=False,
        ),
    )

    node_data: dict[str, dict] = {}
    final_answer = ""
    final_agent = ""
    total_tokens = 0
    total_time = 0.0
    execution_order: list[str] = []

    print(f"Task: {graph.query}")
    print("─" * 50)

    for event in runner.stream(graph, final_agent_id="math_solver"):
        etype = event.event_type

        if etype == StreamEventType.AGENT_START:
            aid = getattr(event, "agent_id", "")
            name = getattr(event, "agent_name", aid)
            print(f"  ▶ {name} starting…")
            node_data[aid] = {"agent_name": name, "response": ""}

        elif etype == StreamEventType.AGENT_OUTPUT:
            aid = getattr(event, "agent_id", "")
            content = getattr(event, "content", "")
            if aid in node_data:
                node_data[aid]["response"] = content
            execution_order.append(aid)
            print(f"  ✓ {node_data.get(aid, {}).get('agent_name', aid)}: {content[:120]}…")

        elif etype == StreamEventType.AGENT_ERROR:
            aid = getattr(event, "agent_id", "")
            err = getattr(event, "error_message", "unknown")
            execution_order.append(aid)
            print(f"  ✗ {aid}: ERROR — {err}")

        elif etype == StreamEventType.RUN_END:
            final_answer = getattr(event, "final_answer", "")
            final_agent = getattr(event, "final_agent_id", "")
            total_tokens = getattr(event, "total_tokens", 0)
            total_time = getattr(event, "total_time", 0.0)

    print("─" * 50)
    print(f"Final answer (from '{final_agent}'):")
    print(final_answer)
    print(f"\nTokens: {total_tokens}  |  Time: {total_time:.2f}s")

    # Save log
    log = {
        "timestamp": datetime.now(UTC).isoformat(),
        "task": graph.query,
        "execution_order": execution_order,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "nodes": node_data,
        "final_answer": final_answer,
        "final_agent": final_agent,
    }
    log_path = Path(__file__).parent / "math_chain_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
