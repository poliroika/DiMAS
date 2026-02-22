"""
Star topology with medical agents.

Demonstrates:
  - Parallel execution of 5 specialist agents
  - Aggregation by a central GP agent (star topology)
  - Saving the full dialogue history to JSON

Topology:
    Orthopaedist ───┐
    Ophthalmologist ┤
    Cardiologist ───┼──→ General Practitioner
    Neurologist ────┤
    Dermatologist ──┘

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.medical_star_topology
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import LLMCallerFactory, MACPRunner, RunnerConfig

# ── Constants ───────────────────────────────────────────────────────────────────

SPECIALIST_MESSAGE_PREVIEW_LENGTH = 600
FINAL_ANSWER_PREVIEW_LENGTH = 1200

# ── Configuration ─────────────────────────────────────────────────────────────

LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", "your-api-key"),
    "base_url": os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
    "model_name": os.getenv("LLM_MODEL", "gpt-4o-mini"),
}

SPECIALISTS = ["orthopedist", "ophthalmologist", "cardiologist", "neurologist", "dermatologist"]

SPECIALIST_DISPLAY = {
    "orthopedist": "Orthopaedist",
    "ophthalmologist": "Ophthalmologist",
    "cardiologist": "Cardiologist",
    "neurologist": "Neurologist",
    "dermatologist": "Dermatologist",
}

PATIENT_CASE = """\
Patient: Male, 45 years old

Complaints:
- Right knee pain when walking (3 weeks)
- Periodic headaches
- Blurred distance vision
- Elevated blood pressure (150/95)
- Dry skin on hands and elbows
- General fatigue

Medical history:
- Office job (sedentary)
- Runs 3 times per week
- Work-related stress for the past 2 months
- Family history: father has hypertension
"""


# ── Graph construction ────────────────────────────────────────────────────────


def _build_graph():
    """Build a star-topology graph: 5 specialists → GP."""
    builder = GraphBuilder()
    cfg = LLM_CONFIG

    specialist_defs = [
        (
            "orthopedist",
            "An experienced orthopaedic surgeon.",
            "Analyse the musculoskeletal system. Focus on joint/muscle pain and physical activity impact. "
            "Provide a concise 3–5 sentence conclusion.",
        ),
        (
            "ophthalmologist",
            "A qualified ophthalmologist.",
            "Evaluate vision and potential eye problems. Provide a concise 3–5 sentence conclusion.",
        ),
        (
            "cardiologist",
            "A cardiovascular disease specialist.",
            "Analyse the cardiovascular system. Pay attention to blood pressure, heredity, and risk factors. "
            "Provide a concise 3–5 sentence conclusion.",
        ),
        (
            "neurologist",
            "A neurologist specialising in nervous system disorders.",
            "Evaluate neurological symptoms. Analyse headaches, stress, and nervous system state. "
            "Provide a concise 3–5 sentence conclusion.",
        ),
        (
            "dermatologist",
            "A dermatologist with expertise in skin conditions.",
            "Analyse the skin condition. Note dryness, potential causes, and links to general health. "
            "Provide a concise 3–5 sentence conclusion.",
        ),
    ]

    for agent_id, persona, description in specialist_defs:
        builder.add_agent(
            agent_id,
            display_name=SPECIALIST_DISPLAY[agent_id],
            persona=persona,
            description=description,
            llm_backbone=cfg["model_name"],
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            temperature=0.3,
            max_tokens=500,
        )

    builder.add_agent(
        "general_practitioner",
        display_name="General Practitioner",
        persona="An experienced GP coordinating a team of specialists.",
        description=(
            "Analyse all specialist conclusions.\n"
            "1. Identify connections between symptoms\n"
            "2. Formulate a diagnosis or hypotheses\n"
            "3. Give comprehensive treatment recommendations\n\n"
            "Structure: ANALYSIS → DIAGNOSIS → RECOMMENDATIONS"
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.2,
        max_tokens=1500,
    )

    for spec in SPECIALISTS:
        builder.add_workflow_edge(spec, "general_practitioner")

    builder.add_task(query=PATIENT_CASE)
    builder.connect_task_to_agents(agent_ids=SPECIALISTS)

    return builder.build()


# ── Display ───────────────────────────────────────────────────────────────────


def _print_results(result) -> None:
    print("\n" + "=" * 60)
    print("SPECIALIST CONSULTATIONS")
    print("=" * 60)

    for sid, name in SPECIALIST_DISPLAY.items():
        if sid in result.messages:
            print(f"\n  {name}")
            print("  " + "─" * 40)
            text = result.messages[sid]
            print(f"  {text[:SPECIALIST_MESSAGE_PREVIEW_LENGTH]}")
            if len(text) > SPECIALIST_MESSAGE_PREVIEW_LENGTH:
                print("  … (truncated)")

    print("\n" + "=" * 60)
    print("GENERAL PRACTITIONER — FINAL DIAGNOSIS")
    print("=" * 60)
    text = result.final_answer
    print(text[:FINAL_ANSWER_PREVIEW_LENGTH])
    if len(text) > FINAL_ANSWER_PREVIEW_LENGTH:
        print("… (truncated)")

    print(f"\nExecution order : {result.execution_order}")
    print(f"Total time      : {result.total_time:.2f}s")
    print(f"Total tokens    : {result.total_tokens}")


# ── Persistence ───────────────────────────────────────────────────────────────


def _save_history(result) -> Path:
    history = {
        "timestamp": datetime.now(UTC).isoformat(),
        "topology": "star",
        "patient_case": PATIENT_CASE,
        "execution_order": result.execution_order,
        "total_time": result.total_time,
        "total_tokens": result.total_tokens,
        "specialists": {
            sid: {"display_name": SPECIALIST_DISPLAY.get(sid, sid), "response": result.messages.get(sid, "")}
            for sid in SPECIALISTS
            if sid in result.messages
        },
        "final_diagnosis": result.final_answer,
    }
    path = Path(__file__).parent / "medical_dialogue_history.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Medical Multi-Agent Consultation (Star Topology)")
    print("=" * 60)
    print(f"\n{PATIENT_CASE}")

    graph = _build_graph()
    print(f"Agents: {[a.agent_id for a in graph.agents]}")

    runner = MACPRunner(
        llm_factory=LLMCallerFactory.create_openai_factory(),
        config=RunnerConfig(
            timeout=120.0,
            adaptive=True,
            enable_parallel=True,
            max_parallel_size=5,
            broadcast_task_to_all=True,
        ),
    )

    print("\nStarting parallel specialist consultation…")
    result = runner.run_round(graph, final_agent_id="general_practitioner")

    _print_results(result)

    path = _save_history(result)
    print(f"\nDialogue saved → {path}")
    print("✅ All specialists executed successfully!")


if __name__ == "__main__":
    main()
