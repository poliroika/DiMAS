# 🚀 Быстрый старт — RustworkX Agent Framework

Это краткое руководство поможет вам начать работу с фреймворком за 5 минут.

## Установка

```bash
pip install rustworkx>=0.13 pydantic>=2.0 pydantic-settings>=2.0 torch>=2.0 loguru>=0.7
pip install sentence-transformers>=2.0  # опционально, для эмбеддингов
```

## Шаг 1: Создайте агентов

```python
from rustworkx_framework import AgentProfile

# Каждый агент имеет уникальный identifier и описание своей роли
# Эмбеддинги и состояние хранятся внутри AgentProfile (децентрализованно)
agents = [
    AgentProfile(
        identifier="researcher",
        display_name="Исследователь",
        description="Ищет информацию и собирает факты",
        tools=["search", "browse"],
    ),
    AgentProfile(
        identifier="analyst",
        display_name="Аналитик",
        description="Анализирует данные и делает выводы",
        tools=["calculate", "compare"],
    ),
    AgentProfile(
        identifier="writer",
        display_name="Писатель",
        description="Формулирует финальный ответ",
    ),
]
```

## Шаг 2: Постройте граф

```python
from rustworkx_framework.builder import build_property_graph

# Определяем связи: researcher -> analyst -> writer
workflow_edges = [
    ("researcher", "analyst"),
    ("analyst", "writer"),
]

# Строим граф с задачей
graph = build_property_graph(
    agents,
    workflow_edges=workflow_edges,
    query="Какие технологии будут важны в 2025 году?",
)

print(f"Граф: {graph.num_nodes} узлов, {graph.num_edges} рёбер")
```

## Шаг 3: Настройте LLM

```python
# Пример с OpenAI
import openai

def my_llm_caller(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Пример с локальным Ollama
import requests

def ollama_caller(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return response.json()["response"]
```

## Шаг 4: Запустите выполнение

```python
from rustworkx_framework import MACPRunner

runner = MACPRunner(llm_caller=my_llm_caller)
result = runner.run_round(graph)

print("=" * 50)
print(f"Порядок выполнения: {result.execution_order}")
print(f"Использовано токенов: {result.total_tokens}")
print(f"Время: {result.total_time:.2f} сек")
print("=" * 50)
print(f"\n📝 Финальный ответ:\n{result.final_answer}")
```

## Шаг 5: Streaming (опционально)

```python
from rustworkx_framework.execution import StreamEventType

# Получайте результаты в реальном времени
for event in runner.stream(graph):
    if event.event_type == StreamEventType.AGENT_START:
        print(f"\n🤖 {event.agent_name} начал работу...")
    elif event.event_type == StreamEventType.AGENT_OUTPUT:
        print(f"✅ {event.agent_name}: {event.content[:100]}...")
    elif event.event_type == StreamEventType.RUN_END:
        print(f"\n🏁 Завершено за {event.total_time:.2f} сек")
```

---

## Полезные паттерны

### Параллельная обработка

```python
# Несколько агентов работают параллельно
edges = [
    ("planner", "researcher_1"),
    ("planner", "researcher_2"),
    ("researcher_1", "synthesizer"),
    ("researcher_2", "synthesizer"),
]

from rustworkx_framework.execution import RunnerConfig

config = RunnerConfig(enable_parallel=True, max_parallel_size=3)
runner = MACPRunner(llm_caller=my_llm, config=config)
```

### Динамическое изменение графа

```python
# Добавить нового агента на лету
new_agent = AgentProfile(identifier="fact_checker", display_name="Fact Checker")
graph.add_node(new_agent, connections_to=["writer"])
graph.add_edge("analyst", "fact_checker", weight=0.8)
```

### Асинхронное выполнение

```python
async def async_llm(prompt: str) -> str:
    # Ваш async LLM вызов
    return await call_llm_async(prompt)

runner = MACPRunner(async_llm_caller=async_llm)
result = await runner.arun_round(graph)
```

---

## Следующие шаги

📚 Прочитайте [полную документацию](DOCUMENTATION.md) для:
- Настройки памяти и эмбеддингов агентов (хранятся внутри `AgentProfile`)
- GNN-маршрутизации
- Адаптивного выполнения с pruning и fallback
- Интеграции с PyTorch Geometric
- Конфигурации через переменные окружения

💡 Изучите примеры в папке `rustworkx_framework/examples/`:
- `basic_usage.py` — базовые операции
- `gnn_routing.py` — GNN-маршрутизация
- `streaming_example.py` — streaming выполнение
