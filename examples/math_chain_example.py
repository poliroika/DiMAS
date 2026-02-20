"""
Example: Agent chain — Task → Math Researcher → Math Solver.

Demonstrates:
- Building a two-agent sequential chain
- Passing messages between nodes
- Streaming execution to capture per-agent prompts and responses
- Saving the full communication log to JSON

Configure your LLM via environment variables:
    LLM_API_KEY   — API key
    LLM_BASE_URL  — OpenAI-compatible endpoint URL
    LLM_MODEL     — model name / path

Run with:
    python -m examples.math_chain_example
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
from rustworkx_framework.execution.streaming import StreamEventType
from rustworkx_framework.tools import create_openai_caller

# Setup framework logging
setup_logging(level="INFO")


def build_math_chain_graph():
    """Build the graph: Task → Math Researcher → Math Solver."""
    config = BuilderConfig(include_task_node=True, validate=True)
    builder = GraphBuilder(config)

    builder.add_task(
        task_id="__task__",
        query="Solve the equation: 2x - 3x² = 1",
        description="Mathematical problem to solve",
    )

    # Agent 1: plans the solution steps but does NOT compute the answer
    builder.add_agent(
        agent_id="math_researcher",
        display_name="Math Researcher",
        persona="a mathematical researcher",
        description=(
            "You outline the steps required to solve the mathematical problem "
            "but do NOT write the final answer — only the solution plan. "
            "Also restate the original equation that needs to be solved."
        ),
    )

    # Agent 2: follows the plan and produces the final answer
    builder.add_agent(
        agent_id="math_solver",
        display_name="Math Solver",
        persona="a mathematics solver",
        description="You solve the mathematical problem following the plan and output the CORRECT ANSWER.",
    )

    # Flow: task → researcher → solver
    builder.connect_task_to_agents(agent_ids=["math_researcher"], bidirectional=False)
    builder.add_workflow_edge("math_researcher", "math_solver")

    return builder.build()


def run_and_log() -> dict:
    """Execute the chain and write the full communication log to JSON."""
    graph = build_math_chain_graph()
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
            prompt_preview_length=10_000,
            broadcast_task_to_all=False,
        ),
    )

    node_data: dict[str, dict] = {}
    final_answer = ""
    final_agent = ""
    total_tokens = 0
    total_time = 0.0
    execution_order: list[str] = []

    print(f"\nTask: {graph.query}")
    print("─" * 50)

    for event in runner.stream(graph, final_agent_id="math_solver"):
        if event.event_type == StreamEventType.AGENT_START:
            if hasattr(event, "agent_id") and hasattr(event, "agent_name"):
                agent_id = str(event.agent_id)
                print(f"  ▶ {event.agent_name} starting…")
                node_data[agent_id] = {
                    "agent_name": str(event.agent_name),
                    "predecessors": getattr(event, "predecessors", []),
                    "input_prompt": getattr(event, "prompt_preview", ""),
                    "response": "",
                }

        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            if hasattr(event, "agent_id") and hasattr(event, "content"):
                agent_id = str(event.agent_id)
                content = str(event.content)
                if agent_id in node_data:
                    node_data[agent_id]["response"] = content
                    node_data[agent_id]["tokens_used"] = getattr(event, "tokens_used", 0)
                execution_order.append(agent_id)
                print(f"  ✓ {node_data[agent_id]['agent_name']}: {content[:120]}…")

        elif event.event_type == StreamEventType.AGENT_ERROR:
            if hasattr(event, "agent_id") and hasattr(event, "error_message"):
                agent_id = str(event.agent_id)
                error_msg = str(event.error_message)
                if agent_id in node_data:
                    node_data[agent_id]["response"] = f"[Error: {error_msg}]"
                    node_data[agent_id]["error"] = error_msg
                execution_order.append(agent_id)
                print(f"  ✗ {agent_id}: ERROR — {error_msg}")

        elif event.event_type == StreamEventType.RUN_END and hasattr(event, "final_answer"):
            final_answer = event.final_answer
            final_agent = getattr(event, "final_agent_id", "")
            total_tokens = getattr(event, "total_tokens", 0)
            total_time = getattr(event, "total_time", 0.0)

    print("─" * 50)
    print(f"Final answer (from '{final_agent}'):")
    print(final_answer)
    print(f"\nTotal tokens: {total_tokens}  |  Total time: {total_time:.2f}s")

    communication_log = {
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
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(communication_log, f, ensure_ascii=False, indent=2)
    print(f"\nLog saved → {log_path}")

    return communication_log


if __name__ == "__main__":
    run_and_log()
