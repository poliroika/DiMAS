"""Пример валидации input/output данных агентов через Pydantic схемы.

Показывает:
1. Создание агентов с input/output схемами
2. Валидация данных через RoleGraph
3. Обработка невалидных ответов LLM
4. Использование JSON Schema для промптов
5. Альтернатива: JSON Schema напрямую (без Pydantic)
"""

import json

from pydantic import BaseModel, Field

from rustworkx_framework.builder import GraphBuilder

# =============================================================================
# 1. Определяем Pydantic схемы для валидации
# =============================================================================


class MathProblemInput(BaseModel):
    """Входные данные для решения математической задачи."""

    question: str = Field(..., description="Mathematical question to solve")
    context: str | None = Field(None, description="Additional context or constraints")
    difficulty: int = Field(1, ge=1, le=10, description="Difficulty level 1-10")


class MathSolutionOutput(BaseModel):
    """Выходные данные — решение математической задачи."""

    answer: str = Field(..., description="The final answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    explanation: str | None = Field(None, description="Step-by-step explanation")
    steps: list[str] = Field(default_factory=list, description="Solution steps")


class ReviewInput(BaseModel):
    """Входные данные для проверки решения."""

    solution: str
    original_question: str


class ReviewOutput(BaseModel):
    """Результат проверки решения."""

    is_correct: bool
    feedback: str
    confidence: float


# =============================================================================
# 2. Создаём граф с агентами, имеющими схемы валидации
# =============================================================================


def create_math_pipeline():
    """Создать pipeline для решения математических задач с валидацией."""
    builder = GraphBuilder()

    # Агент-решатель с Pydantic схемами
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

    # Агент-проверяльщик
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

    # Workflow: solver -> reviewer
    builder.add_workflow_edge("solver", "reviewer")

    return builder.build()


# =============================================================================
# 3. Пример использования валидации
# =============================================================================


def example_1_valid_input():
    """Пример 1: Валидация корректных входных данных."""
    print("=" * 80)
    print("Пример 1: Валидация корректных входных данных")
    print("=" * 80)

    graph = create_math_pipeline()

    # Валидация входных данных для solver
    input_data = {
        "question": "Solve the equation: x^2 + 5x + 6 = 0",
        "context": "Find both solutions",
        "difficulty": 3,
    }

    result = graph.validate_agent_input("solver", input_data)

    if result.valid:
        print("✅ Input is VALID")
        print(f"Validated data: {json.dumps(result.validated_data, indent=2)}")
    else:
        print("❌ Input is INVALID")
        print(f"Errors: {result.errors}")

    print()


def example_2_invalid_input():
    """Пример 2: Валидация некорректных входных данных."""
    print("=" * 80)
    print("Пример 2: Валидация некорректных входных данных")
    print("=" * 80)

    graph = create_math_pipeline()

    # Невалидные данные: missing required field, wrong type
    invalid_data = {
        "context": "Some context",
        "difficulty": "hard",  # Should be int, not str
    }

    result = graph.validate_agent_input("solver", invalid_data)

    if result.valid:
        print("✅ Input is VALID")
    else:
        print("❌ Input is INVALID")
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    print()


def example_3_validate_output():
    """Пример 3: Валидация выходных данных (ответа LLM)."""
    print("=" * 80)
    print("Пример 3: Валидация выходных данных (ответа LLM)")
    print("=" * 80)

    graph = create_math_pipeline()

    # Симулируем корректный ответ LLM в JSON формате
    llm_response = json.dumps(
        {
            "answer": "x1 = -2, x2 = -3",
            "confidence": 0.95,
            "explanation": "Factoring: (x+2)(x+3) = 0",
            "steps": ["Factor the equation", "Apply zero product property", "Solve for x"],
        }
    )

    result = graph.validate_agent_output("solver", llm_response)

    if result.valid:
        print("✅ Output is VALID")
        print("Parsed data:")
        print(json.dumps(result.validated_data, indent=2))
    else:
        print("❌ Output is INVALID")
        print(f"Errors: {result.errors}")

    print()


def example_4_handle_invalid_llm_response():
    """Пример 4: Обработка некорректного ответа LLM."""
    print("=" * 80)
    print("Пример 4: Обработка некорректного ответа LLM")
    print("=" * 80)

    graph = create_math_pipeline()

    # Симулируем некорректный ответ LLM (missing required fields)
    bad_response = json.dumps(
        {
            "answer": "x = -2 or x = -3",
            # Missing: confidence (required field)
            "explanation": "Solved it!",
        }
    )

    result = graph.validate_agent_output("solver", bad_response)

    if result.valid:
        parsed = result.validated_data
        print(f"✅ Valid response: {parsed['answer']}")
    else:
        print("❌ Invalid LLM response!")
        print(f"Errors: {result.errors}")
        print()
        print("Handling strategy:")
        print("  1. Retry with stricter prompt")
        print("  2. Use fallback values")
        print("  3. Raise error and skip agent")
        print()

        # Стратегия: Fallback на дефолтные значения
        fallback_data = {
            "answer": bad_response,
            "confidence": 0.5,  # Low confidence due to format error
            "explanation": "LLM failed to format correctly",
            "steps": [],
        }
        print(f"Using fallback: {json.dumps(fallback_data, indent=2)}")

    print()


def example_5_json_schema_for_prompts():
    """Пример 5: Получение JSON Schema для инструкций LLM."""
    print("=" * 80)
    print("Пример 5: Получение JSON Schema для инструкций LLM")
    print("=" * 80)

    graph = create_math_pipeline()

    # Получить JSON Schema
    input_schema = graph.get_input_schema_json("solver")
    output_schema = graph.get_output_schema_json("solver")

    print("Input Schema:")
    print(json.dumps(input_schema, indent=2))
    print()

    print("Output Schema:")
    print(json.dumps(output_schema, indent=2))
    print()

    # Использование в промпте
    prompt_template = f"""You are a math solver.

You will receive input in this format:
{json.dumps(input_schema, indent=2)}

You MUST respond in the following JSON format:
{json.dumps(output_schema, indent=2)}

Now solve: {{question}}
"""

    print("Generated prompt template:")
    print(prompt_template)
    print()


def example_6_json_schema_dict():
    """Пример 6: Использование JSON Schema dict вместо Pydantic."""
    print("=" * 80)
    print("Пример 6: JSON Schema dict (без Pydantic)")
    print("=" * 80)

    # Определяем схемы как обычные словари
    output_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["result", "score"],
    }

    builder = GraphBuilder()
    builder.add_agent(
        "simple_solver",
        output_schema=output_schema,  # JSON Schema dict
    )
    graph = builder.build()

    # Валидация работает через базовую проверку типов
    valid_data = {"result": "42", "score": 0.9}
    result = graph.validate_agent_output("simple_solver", valid_data)
    print(f"Valid data: {result.valid}")

    invalid_data = {"result": "42", "score": "high"}  # Wrong type
    result = graph.validate_agent_output("simple_solver", invalid_data)
    print(f"Invalid data (wrong type): {result.valid}, errors: {result.errors}")

    print()


# =============================================================================
# Запуск всех примеров
# =============================================================================

if __name__ == "__main__":
    print()
    print("🔍 Примеры валидации input/output схем")
    print("=" * 80)
    print()

    example_1_valid_input()
    example_2_invalid_input()
    example_3_validate_output()
    example_4_handle_invalid_llm_response()
    example_5_json_schema_for_prompts()
    example_6_json_schema_dict()

    print("=" * 80)
    print("✅ Все примеры выполнены!")
    print("=" * 80)
