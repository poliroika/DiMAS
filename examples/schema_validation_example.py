"""
Input / output schema validation via Pydantic — no LLM required.

Demonstrates:
  1. Defining Pydantic models for agent input and output
  2. Validating correct and incorrect input data
  3. Validating LLM responses (correct vs missing required fields)
  4. Embedding JSON Schema strings in prompts
  5. Using plain JSON Schema dicts instead of Pydantic

Run:
    python -m examples.schema_validation_example
"""

import json

from pydantic import BaseModel, Field

from rustworkx_framework.builder import GraphBuilder

# ── Pydantic schemas ─────────────────────────────────────────────────────────


class MathProblemInput(BaseModel):
    """Input for the math solver agent."""

    question: str = Field(..., description="Mathematical question to solve")
    context: str | None = Field(None, description="Additional context or constraints")
    difficulty: int = Field(1, ge=1, le=10, description="Difficulty level 1–10")


class MathSolutionOutput(BaseModel):
    """Output produced by the math solver agent."""

    answer: str = Field(..., description="The final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0–1.0")
    explanation: str | None = Field(None, description="Step-by-step explanation")
    steps: list[str] = Field(default_factory=list, description="Solution steps")


class ReviewInput(BaseModel):
    solution: str
    original_question: str


class ReviewOutput(BaseModel):
    is_correct: bool
    feedback: str
    confidence: float


# ── Graph factory ────────────────────────────────────────────────────────────


def _create_pipeline():
    """Solver → Reviewer pipeline with schema validation."""
    builder = GraphBuilder()
    builder.add_agent(
        "solver",
        display_name="Math Solver",
        persona="Expert mathematician who solves problems step by step",
        description="Solves mathematical problems with detailed explanations",
        input_schema=MathProblemInput,
        output_schema=MathSolutionOutput,
        llm_backbone="gpt-4",
        temperature=0.0,
        tools=["calculator"],
    )
    builder.add_agent(
        "reviewer",
        display_name="Solution Reviewer",
        persona="Critical thinker who validates mathematical solutions",
        description="Reviews and validates mathematical solutions",
        input_schema=ReviewInput,
        output_schema=ReviewOutput,
        llm_backbone="gpt-4o-mini",
        temperature=0.0,
    )
    builder.add_workflow_edge("solver", "reviewer")
    return builder.build()


# ── Examples ─────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n── {title} ──")


def example_valid_input():
    _header("1 · Valid input")
    graph = _create_pipeline()
    r = graph.validate_agent_input(
        "solver",
        {
            "question": "Solve: x² + 5x + 6 = 0",
            "context": "Find both solutions",
            "difficulty": 3,
        },
    )
    print(f"  {'✅ Valid' if r.valid else f'❌ Errors: {r.errors}'}")


def example_invalid_input():
    _header("2 · Invalid input (missing field + wrong type)")
    graph = _create_pipeline()
    r = graph.validate_agent_input(
        "solver",
        {
            "context": "Some context",
            "difficulty": "hard",  # should be int; 'question' is missing
        },
    )
    if r.valid:
        print("  ✅ Unexpectedly valid")
    else:
        print(f"  ❌ {len(r.errors)} error(s):")
        for e in r.errors:
            print(f"     • {e}")


def example_valid_output():
    _header("3 · Valid LLM output")
    graph = _create_pipeline()
    r = graph.validate_agent_output(
        "solver",
        json.dumps(
            {
                "answer": "x₁ = −2, x₂ = −3",
                "confidence": 0.95,
                "explanation": "Factoring: (x+2)(x+3) = 0",
                "steps": ["Factor the equation", "Apply zero product property", "Solve for x"],
            }
        ),
    )
    print(f"  {'✅ Valid' if r.valid else f'❌ Errors: {r.errors}'}")


def example_invalid_output():
    _header("4 · Invalid LLM output (missing 'confidence')")
    graph = _create_pipeline()
    r = graph.validate_agent_output(
        "solver",
        json.dumps(
            {
                "answer": "x = −2 or x = −3",
                "explanation": "Solved it!",
            }
        ),
    )
    if r.valid:
        print("  ✅ Unexpectedly valid")
    else:
        print(f"  ❌ Errors: {r.errors}")
        print("  → Strategy: retry or fall back to a safe default.")


def example_schema_in_prompt():
    _header("5 · JSON Schema embedded in prompt")
    graph = _create_pipeline()
    in_schema = graph.get_input_schema_json("solver")
    out_schema = graph.get_output_schema_json("solver")

    prompt = (
        "You are a math solver.\n\n"
        f"Input format:\n{json.dumps(in_schema, indent=2)}\n\n"
        f"Output format:\n{json.dumps(out_schema, indent=2)}\n\n"
        "Now solve: {{question}}"
    )
    print(f"  Prompt preview (300 chars):\n  {prompt[:300]}…")


def example_dict_schema():
    _header("6 · Plain dict schema (no Pydantic)")
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}, "score": {"type": "number"}},
        "required": ["result", "score"],
    }
    builder = GraphBuilder()
    builder.add_agent("simple_solver", output_schema=schema)
    graph = builder.build()

    r1 = graph.validate_agent_output("simple_solver", {"result": "42", "score": 0.9})
    r2 = graph.validate_agent_output("simple_solver", {"result": "42", "score": "high"})
    print(f"  Valid data   → valid={r1.valid}")
    print(f"  Invalid data → valid={r2.valid}  errors={r2.errors}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    example_valid_input()
    example_invalid_input()
    example_valid_output()
    example_invalid_output()
    example_schema_in_prompt()
    example_dict_schema()
    print("\nAll schema examples completed ✅")


if __name__ == "__main__":
    main()
