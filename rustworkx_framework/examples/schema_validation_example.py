"""
Пример валидации input/output данных агентов через Pydantic схемы.

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
    graph = create_math_pipeline()

    # Валидация входных данных для solver
    input_data = {
        "question": "Solve the equation: x^2 + 5x + 6 = 0",
        "context": "Find both solutions",
        "difficulty": 3,
    }

    result = graph.validate_agent_input("solver", input_data)

    if result.valid:
        pass
    else:
        pass


def example_2_invalid_input():
    """Пример 2: Валидация некорректных входных данных."""
    graph = create_math_pipeline()

    # Невалидные данные: missing required field, wrong type
    invalid_data = {
        "context": "Some context",
        "difficulty": "hard",  # Should be int, not str
    }

    result = graph.validate_agent_input("solver", invalid_data)

    if result.valid:
        pass
    else:
        for _error in result.errors:
            pass


def example_3_validate_output():
    """Пример 3: Валидация выходных данных (ответа LLM)."""
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
        pass
    else:
        pass


def example_4_handle_invalid_llm_response():
    """Пример 4: Обработка некорректного ответа LLM."""
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
        pass
    else:
        # Стратегия: Fallback на дефолтные значения
        pass


def example_5_json_schema_for_prompts():
    """Пример 5: Получение JSON Schema для инструкций LLM."""
    graph = create_math_pipeline()

    # Получить JSON Schema
    input_schema = graph.get_input_schema_json("solver")
    output_schema = graph.get_output_schema_json("solver")

    # Использование в промпте
    f"""You are a math solver.

You will receive input in this format:
{json.dumps(input_schema, indent=2)}

You MUST respond in the following JSON format:
{json.dumps(output_schema, indent=2)}

Now solve: {{question}}
"""


def example_6_json_schema_dict():
    """Пример 6: Использование JSON Schema dict вместо Pydantic."""
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
    graph.validate_agent_output("simple_solver", valid_data)

    invalid_data = {"result": "42", "score": "high"}  # Wrong type
    graph.validate_agent_output("simple_solver", invalid_data)


# =============================================================================
# Запуск всех примеров
# =============================================================================

if __name__ == "__main__":
    example_1_valid_input()
    example_2_invalid_input()
    example_3_validate_output()
    example_4_handle_invalid_llm_response()
    example_5_json_schema_for_prompts()
    example_6_json_schema_dict()
