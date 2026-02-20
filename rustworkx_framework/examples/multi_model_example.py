"""
Пример мультимодельности: два агента с разными LLM моделями.

Агент 1 (врач-консультант): Сильная модель Qwen3-Next-80B
  - Расписывает несколько вариантов еды для иммунитета

Агент 2 (организатор): Слабая модель GigaChat-Lightning
  - Выбирает один лучший вариант из предложенных

Демонстрирует:
- Разные модели для разных агентов
- Разные API endpoints (cloudflare tunnels)
- Разные промпты и температуры
"""

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import LLMCallerFactory, MACPRunner

# Конфигурация моделей
DOCTOR_CONFIG = {
    "api_key": "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",
    "base_url": "https://advisory-locked-summaries-steady.trycloudflare.com/v1",
    "model_name": "./models/Qwen3-Next-80B-A3B-Instruct",
}

ORGANIZER_CONFIG = {
    "api_key": "gigachat-lightning-strong-secure-key",
    "base_url": "https://keyword-cameras-homework-analyze.trycloudflare.com/v1",
    "model_name": "GigaChat-Lightning",
}


def main():
    """Запустить пример с двумя агентами на разных моделях."""
    # Шаг 1: Построить граф с агентами

    builder = GraphBuilder()

    # Агент 1: Врач-консультант (сильная модель)
    builder.add_agent(
        "doctor",
        display_name="Врач-консультант",
        persona="Вы опытный врач-диетолог",
        description=(
            "Вы предоставляете профессиональные рекомендации по питанию. "
            "Предложите 3-5 различных вариантов еды, полезных для иммунитета, "
            "с кратким объяснением пользы каждого продукта."
        ),
        llm_backbone=DOCTOR_CONFIG["model_name"],
        base_url=DOCTOR_CONFIG["base_url"],
        api_key=DOCTOR_CONFIG["api_key"],
        temperature=0.7,  # Более креативный, разные варианты
        max_tokens=1000,
    )

    # Агент 2: Организатор (слабая модель)
    builder.add_agent(
        "organizer",
        display_name="Организатор",
        persona="Вы практичный организатор",
        description=(
            "Ваша задача - выбрать ОДИН самый лучший вариант из предложенных. "
            "Проанализируйте все варианты и выберите наиболее оптимальный по соотношению "
            "пользы, доступности и простоты приготовления. "
            "Ответьте кратко: 'Лучший выбор: [название продукта] - [краткое обоснование в 1-2 предложениях]'"
        ),
        llm_backbone=ORGANIZER_CONFIG["model_name"],
        base_url=ORGANIZER_CONFIG["base_url"],
        api_key=ORGANIZER_CONFIG["api_key"],
        temperature=0.1,  # Более детерминированный выбор
        max_tokens=200,
    )

    # Workflow: врач -> организатор
    builder.add_workflow_edge("doctor", "organizer")

    # Добавить задачу
    builder.add_task(query="Какая еда лучше всего подходит для укрепления иммунитета?")
    builder.connect_task_to_agents()

    graph = builder.build()

    for agent in graph.agents:
        if hasattr(agent, "llm_config") and agent.llm_config:
            pass

    # Шаг 2: Создать runner с фабрикой LLM

    # Фабрика автоматически создаст callers для каждого агента
    # на основе их llm_config
    factory = LLMCallerFactory.create_openai_factory()

    runner = MACPRunner(llm_factory=factory)

    # Шаг 3: Запустить выполнение

    try:
        result = runner.run_round(graph, final_agent_id="organizer")

        # Показать результаты

        # Ответ врача
        if "doctor" in result.messages:
            pass

        # Ответ организатора
        if "organizer" in result.messages:
            pass

        # Метрики

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
