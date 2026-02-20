"""
Example: Multi-model setup — two agents backed by different LLMs.

Agent 1 (Doctor): a stronger model that suggests several immunity-boosting foods.
Agent 2 (Organiser): a lighter model that picks the single best option.

Demonstrates:
- Per-agent LLM configuration (different endpoints, models, temperatures)
- LLMCallerFactory dispatching callers by agent
- Sequential workflow: Doctor → Organiser

Configure your models via environment variables:
    DOCTOR_API_KEY / DOCTOR_BASE_URL / DOCTOR_MODEL
    ORGANIZER_API_KEY / ORGANIZER_BASE_URL / ORGANIZER_MODEL

Run with:
    python -m examples.multi_model_example
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.execution import LLMCallerFactory, MACPRunner

# Setup framework logging
setup_logging(level="INFO")

# ── Model configurations ─────────────────────────────────────────────────────
DOCTOR_CONFIG = {
    "api_key":    os.getenv("DOCTOR_API_KEY",    "your-doctor-api-key"),
    "base_url":   os.getenv("DOCTOR_BASE_URL",   "http://localhost:8000/v1"),
    "model_name": os.getenv("DOCTOR_MODEL",      "gpt-4o"),
}

ORGANIZER_CONFIG = {
    "api_key":    os.getenv("ORGANIZER_API_KEY",  "your-organizer-api-key"),
    "base_url":   os.getenv("ORGANIZER_BASE_URL", "http://localhost:8001/v1"),
    "model_name": os.getenv("ORGANIZER_MODEL",    "gpt-4o-mini"),
}


def main():
    """Run the two-model, two-agent example."""

    # ── Step 1: Build the agent graph ────────────────────────────────────────
    builder = GraphBuilder()

    # Agent 1: Doctor (stronger / more creative model)
    builder.add_agent(
        "doctor",
        display_name="Nutrition Doctor",
        persona="You are an experienced nutritional doctor.",
        description=(
            "Provide professional dietary recommendations. "
            "Suggest 3–5 different foods that boost immunity, "
            "with a brief explanation of the benefit of each."
        ),
        llm_backbone=DOCTOR_CONFIG["model_name"],
        base_url=DOCTOR_CONFIG["base_url"],
        api_key=DOCTOR_CONFIG["api_key"],
        temperature=0.7,
        max_tokens=1000,
    )

    # Agent 2: Organiser (lighter / more deterministic model)
    builder.add_agent(
        "organizer",
        display_name="Organiser",
        persona="You are a practical organiser.",
        description=(
            "Your task is to choose THE SINGLE best option from those proposed. "
            "Analyse all suggestions and pick the most optimal one by benefit, "
            "availability, and ease of preparation. "
            "Answer briefly: 'Best choice: [food] — [1–2 sentence justification]'"
        ),
        llm_backbone=ORGANIZER_CONFIG["model_name"],
        base_url=ORGANIZER_CONFIG["base_url"],
        api_key=ORGANIZER_CONFIG["api_key"],
        temperature=0.1,
        max_tokens=200,
    )

    # Workflow: doctor → organizer
    builder.add_workflow_edge("doctor", "organizer")

    builder.add_task(query="What food is best for boosting immunity?")
    builder.connect_task_to_agents()

    graph = builder.build()

    print("Agents and their models:")
    for agent in graph.agents:
        if hasattr(agent, "llm_config") and agent.llm_config:
            model = agent.llm_config.get("model_name", "?")
            print(f"  {agent.agent_id:<12} → {model}")

    # ── Step 2: Create runner with per-agent LLM factory ─────────────────────
    # LLMCallerFactory dispatches the right caller to each agent automatically
    # based on its llm_config field.
    factory = LLMCallerFactory.create_openai_factory()
    runner = MACPRunner(llm_factory=factory)

    # ── Step 3: Run ───────────────────────────────────────────────────────────
    print("\nRunning consultation…")
    try:
        result = runner.run_round(graph, final_agent_id="organizer")
    except Exception as e:
        logger.exception("Error running consultation: %s", e)
        raise

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

        if "doctor" in result.messages:
        print("\n🩺 Doctor's recommendations:")
        print(result.messages["doctor"])

        if "organizer" in result.messages:
        print("\n📋 Organiser's pick:")
        print(result.messages["organizer"])

    print(f"\nTotal tokens : {result.total_tokens}")
    print(f"Total time   : {result.total_time:.2f}s")
    print(f"Final answer : {result.final_answer}")


if __name__ == "__main__":
    main()
