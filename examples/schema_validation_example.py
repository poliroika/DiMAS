"""
Example: Input / output schema validation via Pydantic.

Shows:
1. Defining Pydantic models for agent input and output
2. Building a graph with schema-aware agents
3. Validating correct and incorrect input data
4. Validating LLM responses (correct vs missing required fields)
5. Getting JSON Schema strings to embed in prompts
6. Using plain JSON Schema dicts instead of Pydantic

Run with:
    python -m examples.schema_validation_example
"""

import json

from pydantic import BaseModel, Field

from rustworkx_framework.builder import GraphBuilder


# =============================================================================
# 1. Pydantic schema definitions
# =============================================================================


class MathProblemInput(BaseModel):
    """Input for the math solver agent."""

    question: str = Field(..., description="Mathematical question to solve")
    context: str | None = Field(None, description="Additional context or constraints")
    difficulty: int = Field(1, ge=1, le=10, description="Difficulty level 1–10")


class MathSolutionOutput(BaseModel):
    """Output produced by the math solver agent."""

    answer: str = Field(..., description="The final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0–1.0")
    explanation: str | None = Field(None, description="Step-by-step explanation")
    steps: list[str] = Field(default_factory=list, description="Solution steps")


class ReviewInput(BaseModel):
    """Input for the solution reviewer agent."""

    solution: str
    original_question: str


class ReviewOutput(BaseModel):
    """Output produced by the solution reviewer agent."""

    is_correct: bool
    feedback: str
    confidence: float


# =============================================================================
# 2. Graph factory
# =============================================================================


def create_math_pipeline():
    """Create a solver → reviewer pipeline with schema validation."""
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


# =============================================================================
# 3. Examples
# =============================================================================


def example_1_valid_input():
    """Validate a correctly structured input dict."""
    print("\n── Example 1: valid input ──")
    graph = create_math_pipeline()

    input_data = {
        "question": "Solve the equation: x² + 5x + 6 = 0",
        "context": "Find both solutions",
        "difficulty": 3,
    }

    result = graph.validate_agent_input("solver", input_data)
    if result.valid:
        print("  ✅ Input is valid")
    else:
        print(f"  ❌ Validation errors: {result.errors}")


def example_2_invalid_input():
    """Validate an input dict with a missing required field and a wrong type."""
    print("\n── Example 2: invalid input ──")
    graph = create_math_pipeline()

    invalid_data = {
        "context": "Some context",
        "difficulty": "hard",  # should be int
        # 'question' is required but missing
    }

    result = graph.validate_agent_input("solver", invalid_data)
    if result.valid:
        print("  ✅ Input unexpectedly valid")
    else:
        print(f"  ❌ Found {len(result.errors)} validation error(s):")
        for error in result.errors:
            print(f"     • {error}")


def example_3_validate_output():
    """Validate a well-formed LLM response JSON."""
    print("\n── Example 3: valid LLM output ──")
    graph = create_math_pipeline()

    llm_response = json.dumps({
        "answer": "x₁ = −2, x₂ = −3",
        "confidence": 0.95,
        "explanation": "Factoring: (x+2)(x+3) = 0",
        "steps": ["Factor the equation", "Apply zero product property", "Solve for x"],
    })

    result = graph.validate_agent_output("solver", llm_response)
    if result.valid:
        print("  ✅ Output is valid")
    else:
        print(f"  ❌ Output errors: {result.errors}")


def example_4_handle_invalid_llm_response():
    """Handle a response that is missing the required 'confidence' field."""
    print("\n── Example 4: invalid LLM output (missing required field) ──")
    graph = create_math_pipeline()

    bad_response = json.dumps({
        "answer": "x = −2 or x = −3",
        # 'confidence' is required but absent
        "explanation": "Solved it!",
    })

    result = graph.validate_agent_output("solver", bad_response)
    if result.valid:
        print("  ✅ Output unexpectedly valid")
    else:
        print(f"  ❌ Output errors: {result.errors}")
        print("  Strategy: fall back to a safe default response or ask the LLM to retry.")


def example_5_json_schema_for_prompts():
    """Embed the agent's JSON Schema directly in a prompt string."""
    print("\n── Example 5: JSON Schema in prompt ──")
    graph = create_math_pipeline()

    input_schema  = graph.get_input_schema_json("solver")
    output_schema = graph.get_output_schema_json("solver")

    prompt = (
        "You are a math solver.\n\n"
        f"You will receive input in this format:\n{json.dumps(input_schema, indent=2)}\n\n"
        f"You MUST respond in the following JSON format:\n{json.dumps(output_schema, indent=2)}\n\n"
        "Now solve: {{question}}"
    )

    print("  Prompt template (first 300 chars):")
    print(f"  {prompt[:300]}…")


def example_6_json_schema_dict():
    """Use a plain dict as the output schema instead of a Pydantic model."""
    print("\n── Example 6: plain dict schema ──")

    output_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "score":  {"type": "number"},
        },
        "required": ["result", "score"],
    }

    builder = GraphBuilder()
    builder.add_agent("simple_solver", output_schema=output_schema)
    graph = builder.build()

    valid_data = {"result": "42", "score": 0.9}
    r = graph.validate_agent_output("simple_solver", valid_data)
    print(f"  Valid data   → valid={r.valid}")

    invalid_data = {"result": "42", "score": "high"}  # wrong type
    r = graph.validate_agent_output("simple_solver", invalid_data)
    print(f"  Invalid data → valid={r.valid}  errors={r.errors}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    example_1_valid_input()
    example_2_invalid_input()
    example_3_validate_output()
    example_4_handle_invalid_llm_response()
    example_5_json_schema_for_prompts()
    example_6_json_schema_dict()
    print("\nAll schema examples completed ✅")