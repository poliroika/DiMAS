"""
Example: Star topology with medical agents.

Demonstrates:
- Parallel execution of 5 specialist agents
- Aggregation by a central GP agent (star topology)
- Saving the full dialogue history to JSON
- Topology: all specialists → general practitioner

Structure:
    Orthopaedist ───┐
    Ophthalmologist ┤
    Cardiologist ───┼──→ General Practitioner (final diagnosis)
    Neurologist ────┤
    Dermatologist ──┘

Configure your LLM via environment variables:
    LLM_API_KEY   — API key
    LLM_BASE_URL  — OpenAI-compatible endpoint URL
    LLM_MODEL     — model name / path

Run with:
    python examples/medical_star_topology.py
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.core.graph import RoleGraph
from rustworkx_framework.execution import LLMCallerFactory, MACPRunner, RunnerConfig

# Setup framework logging
setup_logging(level="INFO")

# ── LLM configuration (override via env vars) ───────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

DEFAULT_LLM_CONFIG = {
    "api_key": LLM_API_KEY,
    "base_url": LLM_BASE_URL,
    "model_name": LLM_MODEL,
}

# Prompt-level I/O log (populated by the logging caller factory)
AGENT_IO_LOG: dict[str, dict] = {}

# Patient case used across all agents
PATIENT_CASE = """
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

SPECIALISTS = ["orthopedist", "ophthalmologist", "cardiologist", "neurologist", "dermatologist"]

SPECIALIST_DISPLAY = {
    "orthopedist":    "🦴 Orthopaedist",
    "ophthalmologist":"👁️  Ophthalmologist",
    "cardiologist":   "❤️  Cardiologist",
    "neurologist":    "🧠 Neurologist",
    "dermatologist":  "🧴 Dermatologist",
}


# ── Logging caller factory ───────────────────────────────────────────────────

def create_logging_caller_factory():
    """Wrap the default OpenAI factory to capture every prompt and response."""
    base_factory = LLMCallerFactory.create_openai_factory()

    class LoggingFactory:
        """Delegates to *base_factory* while recording I/O per agent."""

        def __init__(self, base):
            self.base_factory = base
            self.default_caller = base.default_caller
            self.default_async_caller = base.default_async_caller
            self.default_config = base.default_config
            self.caller_builder = base.caller_builder
            self.async_caller_builder = base.async_caller_builder

        def get_caller(self, config=None, agent_id=None):
            base_caller = self.base_factory.get_caller(config, agent_id)
            if base_caller is None:
                return None

            def logging_caller(prompt: str) -> str:
                if agent_id:
                    AGENT_IO_LOG.setdefault(agent_id, {})
                    AGENT_IO_LOG[agent_id]["input_prompt"] = prompt
                    AGENT_IO_LOG[agent_id]["input_length"] = len(prompt)

                response = base_caller(prompt)

                if agent_id:
                    AGENT_IO_LOG[agent_id]["output_response"] = response
                    AGENT_IO_LOG[agent_id]["output_length"] = len(response)

                return response

            return logging_caller

        def get_async_caller(self, config=None, agent_id=None):
            return self.base_factory.get_async_caller(config, agent_id)

        def get_streaming_caller(self, config=None, agent_id=None):
            return self.base_factory.get_streaming_caller(config, agent_id)

        def get_async_streaming_caller(self, config=None, agent_id=None):
            return self.base_factory.get_async_streaming_caller(config, agent_id)

    return LoggingFactory(base_factory)


# ── Graph construction ───────────────────────────────────────────────────────

def create_medical_graph() -> RoleGraph:
    """Build a star-topology graph with 5 specialists feeding into a GP."""
    builder = GraphBuilder()

    cfg = DEFAULT_LLM_CONFIG

    # 1. Orthopaedist
    builder.add_agent(
        "orthopedist",
        display_name="Orthopaedist",
        persona="An experienced orthopaedic surgeon with 15 years of practice.",
        description=(
            "Analyse the patient's musculoskeletal system. "
            "Focus on joint and muscle pain, and the impact of physical activity. "
            "Provide a concise 3–5 sentence conclusion with observations and recommendations."
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 2. Ophthalmologist
    builder.add_agent(
        "ophthalmologist",
        display_name="Ophthalmologist",
        persona="A qualified ophthalmologist.",
        description=(
            "Evaluate the patient's vision and potential eye problems. "
            "Analyse vision-related symptoms. "
            "Provide a concise 3–5 sentence conclusion with observations and recommendations."
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 3. Cardiologist
    builder.add_agent(
        "cardiologist",
        display_name="Cardiologist",
        persona="A cardiovascular disease specialist.",
        description=(
            "Analyse the patient's cardiovascular system. "
            "Pay special attention to blood pressure, heredity, and risk factors. "
            "Provide a concise 3–5 sentence conclusion with observations and recommendations."
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 4. Neurologist
    builder.add_agent(
        "neurologist",
        display_name="Neurologist",
        persona="A neurologist specialising in nervous system disorders.",
        description=(
            "Evaluate the patient's neurological symptoms. "
            "Analyse headaches, their connection to stress, and overall nervous system state. "
            "Provide a concise 3–5 sentence conclusion with observations and recommendations."
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 5. Dermatologist
    builder.add_agent(
        "dermatologist",
        display_name="Dermatologist",
        persona="A dermatologist with expertise in skin conditions.",
        description=(
            "Analyse the patient's skin condition. "
            "Note dryness symptoms, potential causes, and links to general health. "
            "Provide a concise 3–5 sentence conclusion with observations and recommendations."
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 6. General Practitioner — aggregates all specialist reports
    builder.add_agent(
        "general_practitioner",
        display_name="General Practitioner",
        persona="An experienced GP coordinating a team of specialists.",
        description=(
            "You have received reports from all specialists. Your task:\n"
            "1. Analyse all specialist conclusions\n"
            "2. Identify connections between different symptoms\n"
            "3. Formulate a general or provisional diagnosis\n"
            "4. Give comprehensive treatment recommendations\n\n"
            "Structure your answer:\n"
            "- ANALYSIS: brief analysis of specialist reports\n"
            "- DIAGNOSIS: main diagnosis or hypotheses\n"
            "- RECOMMENDATIONS: specific recommendations for the patient\n"
        ),
        llm_backbone=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.2,
        max_tokens=1500,
    )

    # Star topology: every specialist → GP
    for specialist in SPECIALISTS:
        builder.add_workflow_edge(specialist, "general_practitioner")

    builder.add_task(query=PATIENT_CASE, answer="")
    builder.connect_task_to_agents(agent_ids=SPECIALISTS)

    return builder.build()


# ── Result display ───────────────────────────────────────────────────────────

def print_formatted_results(result) -> None:
    """Print specialist conclusions and the GP's final diagnosis."""
    print("\n" + "=" * 60)
    print("SPECIALIST CONSULTATIONS")
    print("=" * 60)

    for specialist_id, display_name in SPECIALIST_DISPLAY.items():
        if specialist_id in result.messages:
            print(f"\n{display_name}")
            print("─" * 40)
            print(result.messages[specialist_id][:600])
            if len(result.messages[specialist_id]) > 600:
                print("  … (truncated)")

    print("\n" + "=" * 60)
    print("🏥 GENERAL PRACTITIONER — FINAL DIAGNOSIS")
    print("=" * 60)
    print(result.final_answer[:1200])
    if len(result.final_answer) > 1200:
        print("  … (truncated)")

    print(f"\nExecution order : {result.execution_order}")
    print(f"Total time      : {result.total_time:.2f}s")
    print(f"Total tokens    : {result.total_tokens}")
    if result.pruned_agents:
        print(f"Pruned agents   : {result.pruned_agents}")


# ── Dialogue history persistence ─────────────────────────────────────────────

def save_dialogue_history(result, output_path: str = "medical_dialogue_history.json") -> Path:
    """Save the full consultation dialogue to a JSON file."""
    timestamp = datetime.now(UTC).isoformat()

    history: dict = {
        "metadata": {
            "timestamp": timestamp,
            "topology": "star",
            "total_agents": len(result.execution_order),
            "execution_time_seconds": result.total_time,
            "total_tokens": result.total_tokens,
        },
        "patient_case": PATIENT_CASE,
        "specialists_consultation": {},
        "final_diagnosis": {
            "agent": result.final_agent_id,
            "output": {"response": result.final_answer, "length": len(result.final_answer)},
        },
        "execution_flow": {
            "execution_order": result.execution_order,
            "parallel_groups": [],
        },
        "metrics": {
            "total_time": result.total_time,
            "total_tokens": result.total_tokens,
            "topology_changed_count": result.topology_changed_count,
            "fallback_count": result.fallback_count,
        },
    }

    for specialist_id in SPECIALISTS:
        if specialist_id not in result.messages:
            continue
        entry: dict = {
            "display_name": SPECIALIST_DISPLAY.get(specialist_id, specialist_id),
                "output": {
                    "response": result.messages[specialist_id],
                    "length": len(result.messages[specialist_id]),
                },
            "execution_index": (
                result.execution_order.index(specialist_id)
                if specialist_id in result.execution_order
                else -1
            ),
            }
            if specialist_id in AGENT_IO_LOG:
            io = AGENT_IO_LOG[specialist_id]
            entry["input"] = {
                "prompt": io.get("input_prompt", ""),
                "length": io.get("input_length", 0),
                }
        history["specialists_consultation"][specialist_id] = entry

    if result.final_agent_id in AGENT_IO_LOG:
        io = AGENT_IO_LOG[result.final_agent_id]
        history["final_diagnosis"]["input"] = {
            "prompt": io.get("input_prompt", ""),
            "length": io.get("input_length", 0),
        }

    if result.step_results:
        history["execution_flow"]["parallel_execution_info"] = [
                {
                    "agent": agent_id,
                "start_time": getattr(step, "start_time", None),
                "end_time": getattr(step, "end_time", None),
                }
            for agent_id, step in result.step_results.items()
        ]

    if result.agent_states:
        history["agent_states"] = result.agent_states

    output_file = Path(__file__).parent / output_path
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return output_file


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    """Run the medical star-topology consultation example."""
    print("=" * 60)
    print("Medical Multi-Agent Consultation (Star Topology)")
    print("=" * 60)
    print("\nPatient case:")
    print(PATIENT_CASE)

    # Build graph
    start_time = time.time()
    graph = create_medical_graph()
    print(f"Graph built in {time.time() - start_time:.2f}s")
    print(f"Agents: {[a.agent_id for a in graph.agents]}")

    # Runner with parallel execution
    runner_config = RunnerConfig(
        timeout=120.0,
        adaptive=True,
        enable_parallel=True,
        max_parallel_size=5,
        broadcast_task_to_all=True,
    )

    AGENT_IO_LOG.clear()
    factory = create_logging_caller_factory()
    runner = MACPRunner(llm_factory=factory, config=runner_config)

    print("\nStarting parallel specialist consultation…")
    execution_start = time.time()

    try:
        result = runner.run_round(graph, final_agent_id="general_practitioner")
    except Exception as e:
        logger.exception("Error running consultation: %s", e)
        return 1

    print(f"Consultation completed in {time.time() - execution_start:.2f}s")

    print_formatted_results(result)

    # Verify all specialists were executed
    specialists_in_order = [s for s in result.execution_order if s in SPECIALISTS]
    if len(specialists_in_order) != len(SPECIALISTS):
        raise ValueError(
            f"Expected all {len(SPECIALISTS)} specialists to run, "
            f"but only {len(specialists_in_order)} did: {specialists_in_order}"
        )
    print("\n✅ All specialists executed successfully!")

    # Persist dialogue
    output_file = save_dialogue_history(result)
    print(f"Dialogue saved → {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
