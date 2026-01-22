"""Пример цепочки агентов: Task → Math Researcher → Math Solver

Демонстрирует передачу сообщений между нодами и логирование коммуникации.
"""

import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from rustworkx_framework.builder.graph_builder import BuilderConfig, GraphBuilder
from rustworkx_framework.execution.runner import MACPRunner, RunnerConfig
from rustworkx_framework.execution.streaming import StreamEventType

# ============== LLM CONFIG ==============
API_KEY = "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"
BASE_URL = "https://advisory-locked-summaries-steady.trycloudflare.com/v1"
MODEL = "./models/Qwen3-Next-80B-A3B-Instruct"

# Хранилище промптов для логирования
prompts_log: dict[str, str] = {}


def create_llm_caller():
    """Создать функцию для вызова LLM с логированием промптов."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    return call_llm


def build_math_chain_graph():
    """Построить граф: Task → Math Researcher → Math Solver."""
    config = BuilderConfig(
        include_task_node=True,
        validate=True,
    )
    builder = GraphBuilder(config)

    # Добавляем task ноду с задачей
    builder.add_task(
        task_id="__task__",
        query="Реши уравнение: 2x - 3x² = 1",
        description="Математическая задача для решения",
    )

    # Math Researcher - расписывает шаги решения
    builder.add_agent(
        agent_id="math_researcher",
        display_name="Math Researcher",
        persona="математический исследователь",
        description=(
            "Ты расписываешь необходимые действия для решения математической задачи, "
            "но НЕ пишешь конечный ответ. Только план решения. Также пиши начальное уравнение которое нужно решить"
        ),
    )

    # Math Solver - решает и даёт ответ
    builder.add_agent(
        agent_id="math_solver",
        display_name="Math Solver",
        persona="математик-решатель",
        description=(
            "Ты решаешь математическую задачу согласно плану и выводишь ПРАВИЛЬНЫЙ ОТВЕТ."
        ),
    )

    # Соединяем ноды: task → researcher → solver
    builder.connect_task_to_agents(agent_ids=["math_researcher"], bidirectional=False)
    builder.add_workflow_edge("math_researcher", "math_solver")

    return builder.build()


def run_and_log():
    """Запустить граф и записать всю коммуникацию в файл."""
    print("🔧 Создаю граф...")
    graph = build_math_chain_graph()

    print(f"   Ноды: {graph.num_nodes}")
    print(f"   Рёбра: {graph.num_edges}")
    print(f"   Агенты: {[a.identifier for a in graph.agents]}")

    print("\n🤖 Создаю LLM caller...")
    llm_caller = create_llm_caller()

    runner = MACPRunner(
        llm_caller=llm_caller,
        config=RunnerConfig(
            timeout=120.0,
            adaptive=False,
            update_states=True,
            prompt_preview_length=10000,  # Полный промпт
            broadcast_task_to_all=False,  # Task передаётся только агентам соединённым с task нодой
        ),
    )

    print("\n🚀 Запускаю выполнение графа...\n")
    print("=" * 60)

    # Используем streaming для получения промптов
    node_data: dict[str, dict] = {}
    final_answer = ""
    final_agent = ""
    total_tokens = 0
    total_time = 0.0
    execution_order = []

    for event in runner.stream(graph, final_agent_id="math_solver"):
        if event.event_type == StreamEventType.AGENT_START:
            # Сохраняем входные данные ноды
            node_data[event.agent_id] = {
                "agent_name": event.agent_name,
                "predecessors": event.predecessors,
                "input_prompt": event.prompt_preview,
                "response": "",
            }
            print(f"\n📥 [{event.agent_id.upper()}] INPUT:")
            print("-" * 40)
            print(
                event.prompt_preview[:500] + "..."
                if len(event.prompt_preview) > 500
                else event.prompt_preview
            )
            print("-" * 40)

        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            # Сохраняем выход ноды
            if event.agent_id in node_data:
                node_data[event.agent_id]["response"] = event.content
                node_data[event.agent_id]["tokens_used"] = event.tokens_used
            execution_order.append(event.agent_id)

            print(f"\n📤 [{event.agent_id.upper()}] OUTPUT:")
            print("-" * 40)
            print(event.content)
            print("-" * 40)

        elif event.event_type == StreamEventType.AGENT_ERROR:
            if event.agent_id in node_data:
                node_data[event.agent_id]["response"] = f"[Error: {event.error_message}]"
                node_data[event.agent_id]["error"] = event.error_message
            execution_order.append(event.agent_id)

        elif event.event_type == StreamEventType.RUN_END:
            final_answer = event.final_answer
            final_agent = event.final_agent_id
            total_tokens = event.total_tokens
            total_time = event.total_time

    # Собираем лог коммуникации
    communication_log = {
        "timestamp": datetime.now().isoformat(),
        "task": graph.query,
        "execution_order": execution_order,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "nodes": node_data,
        "final_answer": final_answer,
        "final_agent": final_agent,
    }

    print("\n" + "=" * 60)
    print(f"\n✅ Финальный ответ от [{final_agent}]:")
    print(final_answer)

    # Сохраняем лог
    log_path = Path(__file__).parent / "math_chain_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(communication_log, f, ensure_ascii=False, indent=2)

    print(f"\n📝 Лог коммуникации сохранён: {log_path}")

    return communication_log


if __name__ == "__main__":
    run_and_log()
