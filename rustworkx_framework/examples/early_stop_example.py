"""Пример ранней остановки выполнения графа (Early Stopping).

Демонстрирует:
- 3 агента: analyzer → solver → validator
- После solver проверяется правильность ответа
- Если ответ правильный - early stop (validator не выполняется)
- История диалога сохраняется в JSON

Use case: экономия токенов при досрочном получении правильного ответа.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
from rustworkx_framework.execution.runner import (
    EarlyStopCondition,
    MACPRunner,
    RunnerConfig,
    StepContext,
)

# ============== LLM CONFIG ==============
DEFAULT_API_KEY = "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"
DEFAULT_BASE_URL = "https://advisory-locked-summaries-steady.trycloudflare.com/v1"
DEFAULT_MODEL = "./models/Qwen3-Next-80B-A3B-Instruct"


def create_llm_caller():
    """Создать функцию для вызова реального LLM."""
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)

    def call_llm(prompt: str) -> str:
        """Вызов LLM."""
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    return call_llm


# ============== ГРАФ ==============
def build_three_agent_graph():
    """Построить граф с 3 агентами: analyzer → solver → validator."""
    config = BuilderConfig(
        include_task_node=True,
        validate=True,
    )
    builder = GraphBuilder(config)

    # Task
    builder.add_task(
        task_id="__task__",
        query="Реши уравнение: 2x + 5 = 13",
        description="Математическое уравнение",
    )

    # Agent 1: Analyzer - анализирует задачу
    builder.add_agent(
        agent_id="analyzer",
        display_name="Analyzer",
        persona="математический аналитик",
        description=(
            "Ты анализируешь математическую задачу и составляешь подробный план решения. "
            "Распиши все необходимые шаги, но НЕ решай задачу сам. "
            "Твоя задача - только составить план действий."
        ),
    )

    # Agent 2: Solver - решает задачу
    builder.add_agent(
        agent_id="solver",
        display_name="Solver",
        persona="математик-решатель",
        description=(
            "Ты решаешь математическую задачу следуя плану от предыдущего агента. "
            "Выполни все шаги решения и ОБЯЗАТЕЛЬНО выведи финальный ответ в формате: "
            '"FINAL_ANSWER: x = <значение>". '
            "Важно: напиши именно FINAL_ANSWER с двоеточием и ответом."
        ),
    )

    # Agent 3: Validator - проверяет ответ (может не выполниться!)
    builder.add_agent(
        agent_id="validator",
        display_name="Validator",
        persona="проверяющий математик",
        description=(
            "Ты проверяешь правильность решения. "
            "Подставь найденное значение в исходное уравнение и убедись что оно верно. "
            "Если ответ правильный - подтверди это."
        ),
    )

    # Связи: task → analyzer → solver → validator
    builder.connect_task_to_agents(agent_ids=["analyzer"], bidirectional=False)
    builder.add_workflow_edge("analyzer", "solver")
    builder.add_workflow_edge("solver", "validator")

    # Установить границы выполнения
    builder.set_start_node("analyzer")
    builder.set_end_node("validator")

    return builder.build()


# ============== EARLY STOP CONDITIONS ==============
def create_early_stop_condition():
    """Создать условие ранней остановки: если solver дал правильный ответ."""

    def check_answer_correct(ctx: StepContext) -> bool:
        """Проверить что ответ правильный.

        Проверяем:
        1. Есть ли "FINAL_ANSWER" в ответе
        2. Правильный ли ответ (x = 4)
        """
        if ctx.agent_id != "solver":
            return False  # Проверяем только после solver

        response = ctx.response or ""

        # 1. Проверить что есть FINAL_ANSWER
        if "FINAL_ANSWER" not in response:
            print("\n⚠️  FINAL_ANSWER не найден в ответе solver")
            return False

        # 2. Извлечь ответ (поддержка разных форматов включая LaTeX)
        # Ищем после FINAL_ANSWER
        final_answer_part = response.split("FINAL_ANSWER")[-1]

        # Паттерны для извлечения ответа
        patterns = [
            r"x\s*=\s*(\d+)",  # x = 4
            r"x\s*=\s*\\?\(\s*(\d+)\s*\\?\)",  # x = \(4\) или x = (4)
            r":\s*x\s*=\s*(\d+)",  # : x = 4
        ]

        answer = None
        for pattern in patterns:
            match = re.search(pattern, final_answer_part)
            if match:
                answer = int(match.group(1))
                break

        if answer is None:
            print(f"\n⚠️  Не удалось извлечь числовой ответ из: {final_answer_part[:100]}")
            return False

        # 3. Проверить правильность (2x + 5 = 13 → x = 4)
        correct_answer = 4
        is_correct = answer == correct_answer

        if is_correct:
            print(f"\n✅ Ответ ПРАВИЛЬНЫЙ (x = {answer})!")
            print("🎯 Ранняя остановка: validator не нужен")
            return True

        print(f"\n❌ Ответ неправильный (x = {answer}, ожидалось {correct_answer})")
        return False

    return EarlyStopCondition.on_custom(
        condition=check_answer_correct,
        reason="Solver дал правильный ответ, validator не нужен",
        min_agents_executed=2,  # Минимум 2 агента (analyzer + solver)
    )


# ============== ВЫПОЛНЕНИЕ ==============
def run_with_early_stop():
    """Запустить граф с early stopping и логированием."""
    print("=" * 70)
    print("🚀 ПРИМЕР: Early Stopping (ранняя остановка)")
    print("=" * 70)

    print("\n🔧 Создаю граф (3 агента)...")
    graph = build_three_agent_graph()

    print(f"   Ноды: {graph.num_nodes}")
    print(f"   Агенты: {[a.identifier for a in graph.agents]}")
    print(f"   Start: {graph.start_node}")
    print(f"   End: {graph.end_node}")

    print("\n📋 Задача:", graph.query)

    print("\n🤖 Создаю LLM caller...")
    llm_caller = create_llm_caller()

    print("\n⚙️  Настраиваю early stopping condition...")
    early_stop = create_early_stop_condition()

    runner = MACPRunner(
        llm_caller=llm_caller,
        config=RunnerConfig(
            timeout=60.0,
            adaptive=False,
            update_states=True,
            broadcast_task_to_all=False,
            # ⭐ КЛЮЧЕВОЕ: включаем early stopping
            early_stop_conditions=[early_stop],
        ),
    )

    print("\n" + "=" * 70)
    print("🏃 Запускаю выполнение графа...\n")

    # Выполнение
    result = runner.run_round(graph, final_agent_id="validator")

    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 70)

    # Вывод результатов
    print(f"\n✅ Выполнено агентов: {len(result.execution_order)}")
    print(f"   Порядок: {' → '.join(result.execution_order)}")

    if result.early_stopped:
        print(f"\n🎯 Early Stop: {result.early_stopped}")
        print(f"   Причина: {result.early_stop_reason}")

        # Какие агенты НЕ выполнились
        all_agents = ["analyzer", "solver", "validator"]
        skipped = [a for a in all_agents if a not in result.execution_order]
        if skipped:
            print(f"   Пропущено: {', '.join(skipped)} 🚫")
            print(f"   💰 Экономия: {len(skipped)} агент(ов)")
    else:
        print("\n⏱️  Early Stop: Нет (выполнены все агенты)")

    print(f"\n📝 Финальный ответ от [{result.final_agent_id}]:")
    print(f"   {result.final_answer[:200]}...")

    print(f"\n🔢 Токенов использовано: {result.total_tokens}")
    print(f"⏱️  Время выполнения: {result.total_time:.2f} сек")

    # Собираем подробную историю
    communication_log = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Early Stopping Example",
        "task": graph.query,
        "graph_structure": {
            "nodes": [a.identifier for a in graph.agents],
            "edges": [
                ("analyzer", "solver"),
                ("solver", "validator"),
            ],
            "start_node": graph.start_node,
            "end_node": graph.end_node,
        },
        "execution": {
            "execution_order": result.execution_order,
            "early_stopped": result.early_stopped,
            "early_stop_reason": result.early_stop_reason,
            "total_agents": len(graph.agents),
            "executed_agents": len(result.execution_order),
            "skipped_agents": [
                a.identifier for a in graph.agents if a.identifier not in result.execution_order
            ],
        },
        "messages": result.messages,
        "final_answer": result.final_answer,
        "final_agent_id": result.final_agent_id,
        "metrics": {
            "total_tokens": result.total_tokens,
            "total_time": result.total_time,
            "tokens_saved_estimate": (
                len([a for a in graph.agents if a.identifier not in result.execution_order])
                * 500  # Примерная оценка
            ),
        },
    }

    # Сохраняем JSON
    log_path = Path(__file__).parent / "early_stop_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(communication_log, f, ensure_ascii=False, indent=2)

    print(f"\n📄 История диалога сохранена: {log_path}")

    # Вывод сообщений агентов
    print("\n" + "=" * 70)
    print("💬 ИСТОРИЯ ДИАЛОГА")
    print("=" * 70)

    for agent_id in result.execution_order:
        print(f"\n[{agent_id.upper()}]")
        print("-" * 70)
        print(result.messages.get(agent_id, "(нет ответа)"))
        print("-" * 70)

    if result.early_stopped:
        skipped = [
            a.identifier
            for a in graph.agents
            if a.identifier not in result.execution_order and a.identifier != "__task__"
        ]
        if skipped:
            for agent_id in skipped:
                print(f"\n[{agent_id.upper()}] 🚫 НЕ ВЫПОЛНЕН (early stop)")
                print("-" * 70)
                print("Агент был пропущен из-за ранней остановки")
                print("-" * 70)

    print("\n" + "=" * 70)
    print("✅ ЗАВЕРШЕНО")
    print("=" * 70)

    return communication_log


# ============== ПРИМЕР С РАЗНЫМИ СЦЕНАРИЯМИ ==============
def run_comparison():
    """Сравнение: с early stop vs без early stop."""
    print("\n\n")
    print("=" * 70)
    print("📊 СРАВНЕНИЕ: с Early Stop vs без Early Stop")
    print("=" * 70)

    # Сценарий 1: С early stop
    print("\n🎯 Сценарий 1: С EARLY STOP")
    print("-" * 70)
    result1 = run_with_early_stop()

    print("\n\n")
    print("=" * 70)
    print("⏱️  Сценарий 2: БЕЗ EARLY STOP")
    print("=" * 70)
    print("\n(имитация - validator выполняется)")

    # Показываем что было бы без early stop
    print("""
📊 Результаты без early stop:
   Выполнено агентов: 3
   Порядок: analyzer → solver → validator
   Early Stop: Нет
   Токенов: ~1500 (оценка)
   Время: ~3 сек (оценка)
    """)

    print("\n" + "=" * 70)
    print("💡 ИТОГО")
    print("=" * 70)

    executed_with_stop = len(result1["execution"]["execution_order"])
    total_agents = result1["execution"]["total_agents"]
    saved = total_agents - executed_with_stop

    print(f"""
С Early Stop:
   ✅ Выполнено: {executed_with_stop} агента
   🚫 Пропущено: {saved} агент
   💰 Экономия: ~{saved * 500} токенов ({saved}/{total_agents} = {100 * saved / total_agents:.0f}%)

Без Early Stop:
   ✅ Выполнено: {total_agents} агента
   🚫 Пропущено: 0 агентов
   💰 Экономия: 0 токенов

ВЫГОДА: {100 * saved / total_agents:.0f}% экономии токенов при правильном ответе!
    """)


# ============== MAIN ==============
if __name__ == "__main__":
    # Основной пример
    run_with_early_stop()

    # Раскомментируйте для сравнения:
    # run_comparison()

    print("\n\n💡 Tips:")
    print("   - Используйте early stop для экономии токенов")
    print("   - Комбинируйте несколько условий через EarlyStopCondition.combine_any()")
    print("   - Установите min_agents_executed чтобы дать графу 'прогреться'")
    print("   - Проверяйте result.early_stopped для отладки\n")
