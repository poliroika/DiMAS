# RustworkX Agent Framework — Полная Документация

<p align="center">
  <strong>Современный фреймворк для мультиагентных систем на основе графов</strong>
</p>

<p align="center">
  <em>Гибкая и производительная альтернатива LangGraph с динамической топологией, децентрализованной памятью и полным доступом к графовым структурам</em>
</p>

---

## 📋 Содержание

- [Введение](#введение)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Ключевые концепции](#ключевые-концепции)
- [Основные компоненты](#основные-компоненты)
  - [RoleGraph](#rolegraph)
  - [AgentProfile](#agentprofile)
  - [TaskNode](#tasknode)
  - [NodeEncoder](#nodeencoder)
  - [MACPRunner](#macprunner)
  - [Планировщик (Scheduler)](#планировщик-scheduler)
  - [Система памяти](#система-памяти)
  - [Streaming API](#streaming-api)
  - [Бюджет токенов](#бюджет-токенов-budget-system)
  - [Обработка ошибок](#обработка-ошибок-error-handling)
  - [Алгоритмы графа](#алгоритмы-графа-graph-algorithms)
  - [Отслеживание метрик](#отслеживание-метрик-metrics-tracker)
  - [Визуализация](#визуализация-visualization)
  - [Схемы графа](#схемы-графа-schema-system)
  - [Builder API](#builder-api-подробно)
  - [Система событий](#система-событий-event-system)
  - [Callback-система (LangChain-подобная)](#callback-система)
  - [Хранилище состояний](#хранилище-состояний-state-storage)
  - [Асинхронные утилиты](#асинхронные-утилиты-async-utils)
  - [Условная маршрутизация](#условная-маршрутизация-conditional-routing)
- [Продвинутые возможности](#продвинутые-возможности)
  - [Оптимизация выполнения и экономия токенов](#оптимизация-выполнения-и-экономия-токенов)
  - [Мультимодельная поддержка](#мультимодельная-поддержка-multi-model-support)
  - [Динамическая топология](#динамическая-топология)
  - [GNN-маршрутизация](#gnn-маршрутизация)
  - [Скрытые каналы](#скрытые-каналы)
  - [Адаптивное выполнение](#адаптивное-выполнение)
- [Конфигурация](#конфигурация)
- [Примеры использования](#примеры-использования)
- [API Reference](#api-reference)
- [FAQ](#faq)

---

## Введение

**RustworkX Agent Framework** (MECE) — это фреймворк для построения мультиагентных систем, использующий библиотеку `rustworkx` для высокопроизводительных графовых операций. Он решает ключевые ограничения существующих решений, таких как LangGraph:

### Почему MECE лучше LangGraph?

| Особенность | LangGraph | MECE Framework |
|-------------|-----------|----------------|
| **Топология** | Фиксированная | **Динамическая** (изменение в runtime через hooks) |
| **Оптимизация токенов** | Минимальная | **Автоматическая** (фильтрация изолированных нод, disabled nodes, early stopping) |
| **Память** | Централизованная | Децентрализованная (локальное состояние агентов) |
| **Граф** | Скрытый от разработчика | First-class citizen (полный доступ) |
| **Представления** | Только текст | Текст + эмбеддинги + скрытые состояния |
| **Типизация и валидация** | Минимальная | **Полная Pydantic валидация** (типобезопасность) |
| **Схемы данных** | Неформализованные | **Pydantic BaseModel** (автовалидация, сериализация) |
| **Мультимодельность** | Ограниченная | Полная поддержка разных LLM для каждого агента |
| **Параллелизм** | Ограниченный | Полная поддержка async/parallel |
| **Интеграция с ML** | Нет | PyTorch Geometric, GNN-маршрутизация, RL-hooks |
| **Сериализация** | Ручная | **Автоматическая** (Pydantic `.model_dump()`) |
| **Runtime адаптация** | Нет | **Topology hooks, early stopping, disabled nodes** |
| **Callbacks** | BaseCallbackHandler | **Полная совместимость** (те же методы: on_run_start, on_agent_end, etc.) |

---

## Установка

### Требования
- Python 3.12+
- PyTorch 2.0+
- **Pydantic 2.0+** (обязательно — фреймворк полностью основан на Pydantic)

### Через pip (из исходников)

```bash
git clone https://github.com/yourusername/rustworkx-agent-framework.git
cd rustworkx-agent-framework
pip install -e .
```

### Зависимости

```bash
# Основные (обязательные)
pip install rustworkx>=0.13 pydantic>=2.0 pydantic-settings>=2.0 torch>=2.0 loguru>=0.7

# Для эмбеддингов (опционально)
pip install sentence-transformers>=2.0

# Для GNN-маршрутизации (опционально)
pip install torch-geometric>=2.0

# Для визуализации (опционально)
pip install rich>=13.0 graphviz>=0.20
```

### Установка всех опциональных зависимостей

```bash
pip install -e ".[all]"
```

### Важно: Pydantic 2.0+

MECE Framework **требует Pydantic 2.0+** и несовместим с Pydantic 1.x. Все модели (`AgentProfile`, `TaskNode`, схемы, конфигурации) используют Pydantic v2 API:
- `.model_dump()` вместо `.dict()`
- `.model_validate()` вместо `.parse_obj()`
- `.model_dump_json()` вместо `.json()`

Если у вас установлен Pydantic 1.x:
```bash
pip install --upgrade "pydantic>=2.0"
```

---

## Быстрый старт

### Минимальный пример

```python
from rustworkx_framework import RoleGraph, AgentProfile, MACPRunner
from rustworkx_framework.builder import build_property_graph

# 1. Определяем агентов
agents = [
    AgentProfile(
        identifier="solver",
        display_name="Math Solver",
        description="Решает математические задачи шаг за шагом",
        tools=["calculator"],
    ),
    AgentProfile(
        identifier="checker",
        display_name="Answer Checker",
        description="Проверяет корректность решений",
    ),
]

# 2. Определяем связи между агентами
workflow_edges = [("solver", "checker")]

# 3. Строим граф
graph = build_property_graph(
    agents,
    workflow_edges=workflow_edges,
    query="Сколько будет 25 × 17?",
)

# 4. Определяем функцию вызова LLM
def my_llm_caller(prompt: str) -> str:
    # Интегрируйте ваш LLM здесь (OpenAI, Anthropic, локальный и т.д.)
    return call_your_llm(prompt)

# 5. Запускаем выполнение
runner = MACPRunner(llm_caller=my_llm_caller)
result = runner.run_round(graph)

# 6. Получаем результат
print(f"Ответ: {result.final_answer}")
print(f"Порядок выполнения: {result.execution_order}")
print(f"Использовано токенов: {result.total_tokens}")
```

### Быстрый старт: С мониторингом (Callbacks)

```python
from rustworkx_framework import MACPRunner, RunnerConfig
from rustworkx_framework.callbacks import (
    StdoutCallbackHandler,
    MetricsCallbackHandler,
    collect_metrics,
)

# 1. Добавляем callback handlers
config = RunnerConfig(
    callbacks=[
        StdoutCallbackHandler(show_outputs=True),  # Вывод в консоль
        MetricsCallbackHandler(),                  # Сбор метрик
    ]
)

runner = MACPRunner(llm_caller=my_llm_caller, config=config)
result = runner.run_round(graph)

# 2. Или используем context manager
with collect_metrics() as metrics:
    result = runner.run_round(graph)

    print(f"Всего токенов: {metrics.total_tokens}")
    print(f"Время выполнения: {metrics.total_duration_ms}ms")
    print(f"Вызовов агентов: {metrics.get_metrics()['agent_calls']}")
```

### Быстрый старт: Мультимодельность (разные LLM для каждого агента)

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner, LLMCallerFactory

# 1. Создаём билдер и добавляем агентов с разными моделями
builder = GraphBuilder()

# Агент 1: Сильная модель для сложного анализа
builder.add_agent(
    identifier="analyst",
    display_name="Senior Analyst",
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=2000,
)

# Агент 2: Слабая модель для форматирования
builder.add_agent(
    identifier="formatter",
    display_name="Report Formatter",
    llm_backbone="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.3,
    max_tokens=500,
)

# 2. Определяем связи
builder.add_workflow_edge("analyst", "formatter")

# 3. Строим граф
graph = builder.build()

# 4. Создаём фабрику LLM (автоматически создаст callers для каждого агента)
factory = LLMCallerFactory.create_openai_factory()

# 5. Запускаем выполнение
runner = MACPRunner(llm_factory=factory)
result = runner.run_round(graph, query="Проанализируй продажи за Q4")

# 6. Получаем результат
print(f"Финальный ответ: {result.final_answer}")
print(f"Экономия: используем gpt-4 только для анализа, gpt-4o-mini для форматирования")
```

### Быстрый старт: Оптимизация токенов и динамическая топология

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import (
    MACPRunner, RunnerConfig, EarlyStopCondition, TopologyAction
)

# 1. Создаём граф с явными границами
builder = GraphBuilder()
builder.add_agent("input", persona="Input processor")
builder.add_agent("solver", persona="Problem solver")
builder.add_agent("checker", persona="Solution checker")
builder.add_agent("expert", persona="Expert reviewer (expensive)")
builder.add_agent("output", persona="Output formatter")
builder.add_agent("optional", persona="Optional analyzer")

builder.add_workflow_edge("input", "solver")
builder.add_workflow_edge("solver", "checker")
builder.add_workflow_edge("checker", "output")
# expert подключается динамически при необходимости

# Установить границы (для фильтрации изолированных нод)
builder.set_start_node("input")
builder.set_end_node("output")

builder.add_task(query="Solve the problem")
builder.connect_task_to_agents()

graph = builder.build()

# 2. Деактивировать опциональные ноды
graph.disable("optional")  # Не выполнится, экономия токенов

# 3. Hook для адаптации топологии
def adaptive_hook(ctx, graph):
    # Если checker нашёл ошибку — добавить expert
    if ctx.agent_id == "checker" and "ERROR" in (ctx.response or ""):
        return TopologyAction(
            add_edges=[("checker", "expert", 1.0), ("expert", "output", 1.0)],
            trigger_replan=True
        )

    # Если solver уверен — пропустить checker
    if ctx.agent_id == "solver" and "CONFIDENT" in (ctx.response or ""):
        return TopologyAction(skip_agents=["checker"])

    return None

# 4. Настроить runner с оптимизацией
config = RunnerConfig(
    adaptive=True,
    enable_dynamic_topology=True,
    topology_hooks=[adaptive_hook],
    early_stop_conditions=[
        EarlyStopCondition.on_keyword("FINAL_ANSWER"),
        EarlyStopCondition.on_token_limit(5000),
    ],
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# 5. Выполнить с фильтрацией изолированных нод
result = runner.run_round(
    graph,
    filter_unreachable=True  # Исключить ноды не на пути input->output
)

# 6. Результат
print(f"Executed: {result.execution_order}")
print(f"Pruned: {result.pruned_agents}")          # optional + изолированные
print(f"Early stopped: {result.early_stopped}")
print(f"Topology mods: {result.topology_modifications}")  # expert был добавлен?
print(f"Tokens: {result.total_tokens}")
```

---

## Ключевые концепции

### Pydantic-ориентированная архитектура

MECE Framework **полностью основан на Pydantic** для типобезопасности, валидации и сериализации данных. Все ключевые модели наследуются от `pydantic.BaseModel`:

#### Основные Pydantic модели в фреймворке

| Модель | Назначение | Особенности |
|--------|-----------|-------------|
| `AgentProfile` | Профиль агента | `frozen=True` (иммутабельный), `arbitrary_types_allowed` для torch.Tensor |
| `AgentLLMConfig` | LLM конфигурация агента | Валидация параметров модели, разрешение env vars |
| `TaskNode` | Узел задачи | Хранение query и контекста задачи |
| `GraphSchema` | Схема всего графа | Nodes (dict), edges (list), метаданные |
| `AgentNodeSchema` | Схема узла-агента | LLM config, tools, метрики, эмбеддинги |
| `TaskNodeSchema` | Схема узла-задачи | Query, статус, deadline |
| `BaseEdgeSchema` | Базовая схема ребра | Weight, probability, cost metrics |
| `WorkflowEdgeSchema` | Workflow ребро | Условия, приоритет, трансформации |
| `CostMetrics` | Метрики стоимости | Токены, latency, trust, reliability |
| `LLMConfig` | Полная LLM конфигурация | Model name, base URL, API key, параметры генерации |
| `VisualizationStyle` | Стили визуализации | Настройки цветов, форм, показа элементов |
| `NodeStyle` | Стиль узла | Shape, colors, icon |
| `EdgeStyle` | Стиль ребра | Line style, arrow, colors |
| `ValidationResult` | Результат валидации | Errors, warnings |
| `FeatureConfig` | Конфигурация GNN | Размерности признаков |
| `TrainingConfig` | Конфигурация обучения | Learning rate, epochs, optimizer |

#### Преимущества Pydantic в MECE

1. **Автоматическая валидация типов**
   ```python
   # Pydantic автоматически проверяет типы
   agent = AgentProfile(
       identifier="test",           # str - OK
       display_name="Test Agent",   # str - OK
       tools=["search", "calc"],    # list[str] - OK
   )

   # Ошибка валидации при неверном типе
   agent = AgentProfile(identifier=123)  # ❌ ValidationError: identifier должен быть str
   ```

2. **Значения по умолчанию**
   ```python
   # Pydantic заполняет поля значениями по умолчанию
   agent = AgentProfile(identifier="test", display_name="Test")
   print(agent.tools)  # [] (пустой список по умолчанию)
   print(agent.persona)  # "" (пустая строка по умолчанию)
   ```

3. **Автоматическое преобразование типов**
   ```python
   # Pydantic validators автоматически преобразуют типы
   schema = AgentNodeSchema(
       id="test",
       embedding=torch.tensor([0.1, 0.2, 0.3])  # torch.Tensor → list[float]
   )
   print(type(schema.embedding))  # <class 'list'>
   ```

4. **Вложенные модели**
   ```python
   # Pydantic валидирует вложенные модели
   agent = AgentProfile(
       identifier="test",
       display_name="Test",
       llm_config=AgentLLMConfig(  # Вложенная Pydantic модель
           model_name="gpt-4",
           temperature=0.7,
       )
   )
   ```

5. **Сериализация и десериализация**
   ```python
   # Pydantic встроенные методы
   data = agent.model_dump()  # → dict
   json_str = agent.model_dump_json(indent=2)  # → JSON string

   # Загрузка из dict/JSON
   loaded = AgentProfile.model_validate(data)
   loaded_json = AgentProfile.model_validate_json(json_str)
   ```

6. **Иммутабельность**
   ```python
   # frozen=True для AgentProfile
   agent = AgentProfile(identifier="test", display_name="Test")
   agent.identifier = "new_id"  # ❌ ValidationError: frozen model

   # Используйте copy методы для изменений
   updated = agent.model_copy(update={"display_name": "New Name"})
   ```

7. **Расширяемость**
   ```python
   # extra="allow" позволяет произвольные поля
   schema = GraphSchema(
       name="MyGraph",
       custom_field="custom_value",  # Дополнительное поле
       another_field=123,             # Ещё одно
   )
   ```

### Декларативная типизация

Благодаря Pydantic, все типы декларативны и проверяются статически (mypy, pyright) и динамически (во время выполнения):

```python
from rustworkx_framework.core import AgentProfile
from rustworkx_framework.core.schema import AgentNodeSchema, LLMConfig

# Статическая типизация (IDE автодополнение)
agent: AgentProfile = AgentProfile(...)
config: LLMConfig = LLMConfig(...)
schema: AgentNodeSchema = AgentNodeSchema(...)

# Динамическая валидация (runtime)
try:
    bad_agent = AgentProfile(identifier=None)  # ❌ None вместо str
except ValidationError as e:
    print(e.errors())  # Подробная информация об ошибке
```

---

### Децентрализованное хранение данных

В отличие от централизованных архитектур, MECE использует **децентрализованный** подход:
- **Эмбеддинги** хранятся внутри `AgentProfile.embedding`
- **Скрытые состояния** хранятся внутри `AgentProfile.hidden_state`
- **Локальная память** хранится внутри `AgentProfile.state`
- `RoleGraph.embeddings` — это accessor, который собирает эмбеддинги всех агентов в один тензор

Это позволяет каждому агенту владеть своими представлениями и обеспечивает независимость узлов.

### Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                       RoleGraph                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Agent   │──│  Agent   │──│  Agent   │──│  Agent   │        │
│  │ Profile  │  │ Profile  │  │ Profile  │  │ Profile  │        │
│  │(embedding│  │(embedding│  │(embedding│  │(embedding│        │
│  │  state)  │  │  state)  │  │  state)  │  │  state)  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│       ↑             ↑             ↑             ↑              │
│       └─────────────┴─────────────┴─────────────┘              │
│                    Матрица смежности (A_com)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MACPRunner                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Scheduler  │  │   Memory    │  │   Budget    │             │
│  │             │  │    Pool     │  │   Tracker   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   MACPResult    │
                    │  • messages     │
                    │  • final_answer │
                    │  • metrics      │
                    └─────────────────┘
```

### Поток данных

1. **Создание агентов** → `AgentProfile` описывает роль, возможности и инструменты
2. **Построение графа** → `build_property_graph` создаёт `RoleGraph` с топологией
3. **Планирование** → `Scheduler` определяет порядок выполнения
4. **Выполнение** → `MACPRunner` последовательно/параллельно запускает агентов
5. **Результат** → `MACPResult` содержит ответы всех агентов и метрики

---

## Основные компоненты

### RoleGraph

`RoleGraph` — центральная структура данных, представляющая граф агентов.

```python
from rustworkx_framework.core import RoleGraph

# === Свойства графа ===
graph.num_nodes        # Количество узлов
graph.num_edges        # Количество рёбер
graph.agents           # Список AgentProfile объектов
graph.node_ids         # Список идентификаторов узлов ["agent1", "agent2", ...]
graph.role_sequence    # Порядок ролей (legacy)
graph.A_com            # Матрица смежности (torch.Tensor, N x N)
graph.edge_index       # Индекс рёбер в формате PyG (torch.Tensor, 2 x E)
graph.edge_attr        # Атрибуты рёбер (torch.Tensor, E x feature_dim)
graph.embeddings       # Accessor: собирает эмбеддинги из агентов в тензор (N x dim)
graph.graph            # Внутренний rustworkx.PyDiGraph объект
graph.task_node        # TaskNode если включён, иначе None
graph.query            # Запрос задачи (строка)

# === Методы работы с узлами ===
# Добавление узла
graph.add_node(
    agent,                        # AgentProfile
    connections_to=["other"],     # Список ID для исходящих рёбер
    connections_from=["prev"],    # Список ID для входящих рёбер
    weight=1.0,                   # Вес рёбер по умолчанию
)

# Удаление узла с политикой миграции
graph.remove_node(
    "agent_id",
    policy=StateMigrationPolicy.ARCHIVE,  # DISCARD, COPY, ARCHIVE
)

# Замена узла
graph.replace_node(
    old_node_id="old",
    new_agent=new_agent_profile,
    policy=StateMigrationPolicy.COPY,     # Копировать состояние
    keep_connections=True,                # Сохранить рёбра
)

# Получение агента
agent = graph.get_agent_by_id("agent_id")

# Получение индекса узла в матрице
idx = graph.get_node_index("agent_id")  # -> int

# Проверка существования
if "agent_id" in graph.node_ids:
    ...

# === Методы работы с рёбрами ===
# Добавление ребра
graph.add_edge(
    source="agent1",
    target="agent2",
    weight=0.8,
    edge_type="workflow",          # Тип ребра (опционально)
    metadata={"priority": 1},      # Дополнительные данные
)

# Удаление ребра
graph.remove_edge("agent1", "agent2")

# Обновление веса ребра
graph.update_edge_weight("agent1", "agent2", new_weight=0.9)

# Получение соседей
out_neighbors = graph.get_neighbors("agent_id", direction="out")   # Исходящие
in_neighbors = graph.get_neighbors("agent_id", direction="in")     # Входящие
all_neighbors = graph.get_neighbors("agent_id", direction="both")  # Все

# Проверка существования ребра
has_edge = graph.has_edge("agent1", "agent2")

# Получение веса ребра
weight = graph.get_edge_weight("agent1", "agent2")

# === Границы выполнения (start/end nodes) ===
# Установить стартовую и конечную ноды для оптимизации
graph.set_start_node("input_agent")
graph.set_end_node("output_agent")

# Или обе сразу
graph.set_execution_bounds("input_agent", "output_agent")

# Проверить границы
print(f"Start: {graph.start_node}, End: {graph.end_node}")

# === Неактивные ноды (disabled nodes) ===
# Деактивировать ноды (они останутся в графе, но не выполнятся)
graph.disable("agent1")              # Одна нода
graph.disable(["agent2", "agent3"])  # Несколько нод

# Активировать обратно
graph.enable("agent1")               # Одна нода
graph.enable(["agent2", "agent3"])   # Несколько
graph.enable()                       # Все деактивированные

# Проверить статус
graph.is_enabled("agent1")           # -> bool
graph.get_enabled()                  # -> ["agent1", ...]
graph.get_disabled()                 # -> ["agent2", ...]

# Использование: экономия токенов на основе алгоритмов
if rl_model.predict(graph_state) < threshold:
    graph.disable("expensive_agent")

# === Анализ достижимости ===
# Получить ноды, достижимые из start_node
reachable = graph.get_reachable_from("input_agent")

# Получить ноды, из которых достижим end_node
reaching = graph.get_nodes_reaching("output_agent")

# Получить релевантные ноды (на пути start -> end)
relevant = graph.get_relevant_nodes()
# Автоматически использует graph.start_node и graph.end_node

# Получить изолированные ноды (не на пути start -> end)
isolated = graph.get_isolated_nodes()

# Оптимизированный порядок выполнения (без изолированных)
order = graph.get_optimized_execution_order()

# === Условные рёбра ===
# Добавление ребра с условием
from rustworkx_framework.execution.scheduler import ConditionContext

def condition_func(context: ConditionContext) -> bool:
    return context.state.get("quality") > 0.8

graph.add_conditional_edge(
    source="writer",
    target="editor",
    condition=condition_func,
    weight=0.9,
)

# === Динамическое обновление топологии ===
# Полное обновление матрицы смежности
graph.update_communication(
    a_new,                    # Новая матрица смежности (torch.Tensor)
    s_tilde=scores,          # Матрица оценок качества (опционально)
    p_matrix=probabilities,  # Матрица вероятностей переходов (опционально)
)

# === Конвертация и экспорт ===
# Сериализация в словарь
data = graph.to_dict()
# {
#   "agents": [...],
#   "adjacency": [[...]],
#   "query": "...",
#   "task_node": {...},
# }

# Конвертация в PyTorch Geometric Data
pyg_data = graph.to_pyg_data()
# Data(x=node_features, edge_index=edges, edge_attr=weights)

# Извлечение подграфа
subgraph = graph.subgraph(["agent1", "agent2", "agent3"])

# Копирование графа
graph_copy = graph.copy()

# === Проверка целостности ===
# Проверить консистентность внутренних структур
graph.verify_integrity(raise_on_error=True)

# Быстрая проверка
is_valid = graph.is_consistent()

# === Анализ графа ===
# Проверка на DAG (направленный ациклический граф)
is_dag = graph.is_dag()

# Получение топологического порядка (если DAG)
if graph.is_dag():
    topo_order = graph.topological_sort()

# === Обновление агентов ===
# Обновить embedding агента
agent = graph.get_agent_by_id("solver")
agent = agent.with_embedding(new_embedding)
graph.update_agent("solver", agent)

# Обновить состояние агента
agent = agent.append_state({"role": "assistant", "content": "Response"})
graph.update_agent("solver", agent)

# === Batch операции ===
# Обновить несколько агентов
updates = {
    "agent1": updated_agent1,
    "agent2": updated_agent2,
}
graph.batch_update_agents(updates)

# Добавить несколько рёбер
edges = [
    ("a", "b", 0.8),
    ("b", "c", 0.9),
    ("c", "d", 0.7),
]
graph.batch_add_edges(edges)
```

#### Политики миграции состояния

При удалении или замене узла можно указать политику миграции:

```python
from rustworkx_framework.core.graph import StateMigrationPolicy

# DISCARD — состояние удаляется
graph.remove_node("agent_id", policy=StateMigrationPolicy.DISCARD)

# COPY — состояние копируется в новый узел
graph.replace_node("old_id", new_agent, policy=StateMigrationPolicy.COPY)

# ARCHIVE — состояние сохраняется во внешнее хранилище
graph.remove_node("agent_id", policy=StateMigrationPolicy.ARCHIVE)
```

---

### AgentProfile

`AgentProfile` — **иммутабельная Pydantic модель** (`BaseModel` с `frozen=True`) профиля агента с описанием, инструментами, состоянием и LLM конфигурацией.

> **Важно**:
> - `AgentProfile` наследуется от `pydantic.BaseModel`, что обеспечивает **автоматическую валидацию типов** и **типобезопасность**
> - Эмбеддинги и скрытые состояния хранятся **на уровне агента**, а не на уровне графа
> - Поддержка **мультимодельности** — каждый агент может иметь свою LLM конфигурацию
> - Иммутабельность (`frozen=True`) — методы возвращают новые объекты

#### Структура AgentProfile (Pydantic модель)

| Поле | Тип | Описание |
|------|-----|----------|
| `identifier` | `str` | Уникальный идентификатор агента (обязательное) |
| `display_name` | `str` | Отображаемое имя агента (обязательное) |
| `persona` | `str` | Роль/персона агента (например, "Expert analyst") |
| `description` | `str` | Текстовое описание возможностей агента |
| `llm_backbone` | `str \| None` | Идентификатор LLM модели (legacy, использовать `llm_config`) |
| `llm_config` | `AgentLLMConfig \| None` | **Pydantic модель** конфигурации LLM для агента |
| `tools` | `list[str]` | Список доступных инструментов |
| `raw` | `Mapping[str, Any]` | Произвольные дополнительные данные |
| `embedding` | `torch.Tensor \| None` | Векторное представление агента (arbitrary_types_allowed) |
| `state` | `list[dict[str, Any]]` | Локальное состояние/история сообщений |
| `hidden_state` | `torch.Tensor \| None` | Скрытое состояние для передачи между агентами |

#### AgentLLMConfig (Pydantic модель)

```python
from rustworkx_framework.core.agent import AgentLLMConfig

# AgentLLMConfig - Pydantic модель для LLM конфигурации
llm_config = AgentLLMConfig(
    model_name="gpt-4",                         # Имя модели
    base_url="https://api.openai.com/v1",      # API endpoint
    api_key="$OPENAI_API_KEY",                 # Ключ (или $ENV_VAR)
    max_tokens=2000,                            # Макс. токенов
    temperature=0.7,                            # Температура
    timeout=60.0,                               # Таймаут в секундах
    top_p=0.9,                                  # Top-p sampling
    stop_sequences=["END", "STOP"],             # Стоп-последовательности
    extra_params={"frequency_penalty": 0.5},    # Доп. параметры
)

# Методы AgentLLMConfig
api_key = llm_config.resolve_api_key()      # Разрешить $ENV_VAR
is_set = llm_config.is_configured()         # Проверить наличие конфигурации
params = llm_config.to_generation_params()  # Собрать параметры для LLM
```

#### Создание и работа с AgentProfile

```python
from rustworkx_framework.core import AgentProfile
from rustworkx_framework.core.agent import AgentLLMConfig

# 1. Базовое создание (Pydantic валидирует типы)
agent = AgentProfile(
    identifier="analyzer",           # Уникальный ID (str, обязательный)
    display_name="Data Analyzer",    # Отображаемое имя (str, обязательный)
    persona="Expert data analyst",   # Роль/персона (str, default="")
    description="Analyzes data and produces insights",  # Описание (str, default="")
    tools=["python", "sql"],         # Доступные инструменты (list[str], default=[])
)

# 2. Создание с LLM конфигурацией (Pydantic модель)
llm_config = AgentLLMConfig(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",  # Будет разрешён из окружения
    temperature=0.7,
    max_tokens=2000,
)

agent = AgentProfile(
    identifier="researcher",
    display_name="Researcher",
    llm_config=llm_config,  # Pydantic валидирует вложенную модель
    tools=["web_search"],
)

# 3. Работа с состоянием (иммутабельно — возвращает НОВЫЙ объект)
agent = agent.append_state({"role": "user", "content": "Hello!"})
agent = agent.with_state([{"role": "system", "content": "You are helpful"}])
agent = agent.clear_state()

# 4. Работа с эмбеддингами (arbitrary_types_allowed для torch.Tensor)
import torch

embedding = torch.randn(384)
agent = agent.with_embedding(embedding)

hidden_state = torch.randn(768)
agent = agent.with_hidden_state(hidden_state)

# 5. Работа с LLM конфигурацией
agent = agent.with_llm_config(llm_config)

# Получить имя модели агента (приоритет: llm_config.model_name → llm_backbone)
model_name = agent.get_model_name()  # "gpt-4"

# Проверить, есть ли кастомная LLM конфигурация
if agent.has_custom_llm():
    print(f"Agent uses custom LLM: {agent.llm_config.model_name}")
    print(f"Base URL: {agent.llm_config.base_url}")
    print(f"Generation params: {agent.llm_config.to_generation_params()}")

# 6. Сериализация (Pydantic методы)
# Для кодировщика (текст)
text = agent.to_text()

# Для сохранения (dict, включает llm_config)
data = agent.to_dict()

# Pydantic методы сериализации
agent_dict = agent.model_dump()  # Dict[str, Any]
agent_json = agent.model_dump_json(indent=2)  # JSON string

# 7. Десериализация (Pydantic методы)
loaded_agent = AgentProfile.model_validate(agent_dict)
loaded_from_json = AgentProfile.model_validate_json(agent_json)
```

#### Пример: Создание агентов с разными LLM

```python
from rustworkx_framework.core import AgentProfile
from rustworkx_framework.core.agent import AgentLLMConfig

# Агент 1: Сильная модель для анализа
analyst = AgentProfile(
    identifier="analyst",
    display_name="Senior Analyst",
    persona="Expert data analyst with 10 years experience",
    description="Performs deep analysis of complex data",
    llm_config=AgentLLMConfig(
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="$OPENAI_API_KEY",
        temperature=0.0,  # Детерминированность для анализа
        max_tokens=2000,
    ),
    tools=["python", "sql", "visualization"],
)

# Агент 2: Слабая модель для форматирования
formatter = AgentProfile(
    identifier="formatter",
    display_name="Report Formatter",
    persona="Technical writer",
    description="Formats analysis results into readable reports",
    llm_config=AgentLLMConfig(
        model_name="gpt-4o-mini",  # Дешевле для простых задач
        base_url="https://api.openai.com/v1",
        api_key="$OPENAI_API_KEY",
        temperature=0.3,
        max_tokens=500,
    ),
    tools=["markdown", "latex"],
)

# Агент 3: Локальная модель
local_agent = AgentProfile(
    identifier="local_llm",
    display_name="Local Assistant",
    llm_config=AgentLLMConfig(
        model_name="llama3:70b",
        base_url="http://localhost:11434/v1",  # Ollama
        temperature=0.5,
    ),
)
```

#### Преимущества Pydantic валидации

1. **Автоматическая проверка типов** при создании объектов
2. **Значения по умолчанию** для опциональных полей
3. **Иммутабельность** (`frozen=True`) предотвращает случайные изменения
4. **Вложенные модели** (AgentLLMConfig валидируется автоматически)
5. **Сериализация/десериализация** через `.model_dump()` и `.model_validate()`
6. **Поддержка произвольных типов** (`arbitrary_types_allowed`) для torch.Tensor

---

### TaskNode

`TaskNode` — **иммутабельная Pydantic модель** (`BaseModel` с `frozen=True`) виртуального узла задачи, который хранит формулировку запроса и может быть связан со всеми агентами.

> **Важно**: `TaskNode` наследуется от `pydantic.BaseModel`, обеспечивая автоматическую валидацию типов и иммутабельность (как и `AgentProfile`).

#### Структура TaskNode (Pydantic модель)

| Поле | Тип | Описание |
|------|-----|----------|
| `identifier` (`id`) | `str` | Идентификатор узла задачи (по умолчанию `__task__`) |
| `type` | `str` | Тип узла (`"task"`, автоматически) |
| `query` | `str` | Формулировка задачи/запроса |
| `description` | `str` | Дополнительное описание контекста |
| `embedding` | `torch.Tensor \| None` | Эмбеддинг задачи (arbitrary_types_allowed) |
| `display_name` | `str` | Отображаемое имя (по умолчанию `"Task"`) |
| `persona` | `str` | Персона/роль задачи (по умолчанию пусто) |
| `llm_backbone` | `str \| None` | Идентификатор модели, если нужен |
| `tools` | `list[str]` | Инструменты, доступные узлу задачи (default=[]) |
| `state` | `list[dict[str, Any]]` | Локальное состояние/история сообщений задачи (default=[]) |

```python
from rustworkx_framework.core import TaskNode

# Pydantic валидирует типы при создании
task = TaskNode(
    identifier="__task__",          # можно переопределить (str)
    query="Сформулируй план исследования рынка",  # обязательный (str)
    description="Задача для всей команды агентов",  # опциональный (str, default="")
)

# Эмбеддинг задачи (опционально, arbitrary_types_allowed для torch.Tensor)
import torch
task_embedding = torch.randn(384)
task = task.with_embedding(task_embedding)

# TaskNode иммутабельный (frozen=True), используйте copy методы
updated_task = task.model_copy(update={"description": "New description"})

# Pydantic сериализация
task_dict = task.model_dump()
task_json = task.model_dump_json(indent=2)

# Десериализация
loaded = TaskNode.model_validate(task_dict)
```

> При использовании `build_property_graph(..., include_task_node=True)` узел задачи создаётся автоматически и соединяется с агентами рёбрами контекста/обновлений.

#### Методы TaskNode (иммутабельные)

```python
# Работа с эмбеддингом (возвращает новый объект)
task = task.with_embedding(embedding_tensor)

# Работа с состоянием (возвращает новый объект)
task = task.append_state({"role": "system", "content": "Context"})
task = task.with_state([{"role": "user", "content": "Query"}])
task = task.clear_state()

# Преобразование в текст
task_text = task.to_text()  # Для кодировщика

# Преобразование в dict
task_data = task.to_dict()  # Для сохранения
```

---

### NodeEncoder

`NodeEncoder` преобразует текстовые описания агентов в векторные представления.

```python
from rustworkx_framework.core import NodeEncoder

# Использование sentence-transformers (рекомендуется)
encoder = NodeEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    normalize_embeddings=True,
)

# Использование hash-fallback (быстрый, не требует моделей)
encoder = NodeEncoder(model_name="hash:256")

# Кодирование текстов
texts = [agent.to_text() for agent in agents]
embeddings = encoder.encode(texts)  # torch.Tensor (N x dim)

# Получение размерности
dim = encoder.embedding_dim
```

---

### MACPRunner

`MACPRunner` — исполнитель Multi-Agent Communication Protocol.

```python
from rustworkx_framework.execution import MACPRunner, RunnerConfig

# Базовая настройка (один LLM для всех агентов)
runner = MACPRunner(
    llm_caller=sync_llm_function,           # Синхронный LLM
    async_llm_caller=async_llm_function,    # Асинхронный LLM
    token_counter=my_token_counter,         # Подсчёт токенов
)

# Мультимодельная настройка (разные LLM для разных агентов)
from rustworkx_framework.execution import LLMCallerFactory, create_openai_caller

# Способ 1: Использовать фабрику (рекомендуется)
factory = LLMCallerFactory.create_openai_factory(
    default_model="gpt-4o-mini",
    default_base_url="https://api.openai.com/v1",
)
runner = MACPRunner(llm_factory=factory)

# Способ 2: Словарь callers для каждого агента
runner = MACPRunner(
    llm_callers={
        "analyst": create_openai_caller(model="gpt-4", temperature=0.0),
        "writer": create_openai_caller(model="gpt-4o-mini", temperature=0.7),
    },
    async_llm_callers={
        "analyst": create_openai_caller(model="gpt-4", is_async=True),
        "writer": create_openai_caller(model="gpt-4o-mini", is_async=True),
    },
)

# Способ 3: Комбинированный (фабрика + переопределение для конкретных агентов)
runner = MACPRunner(
    llm_factory=factory,                          # Default для всех
    llm_callers={"critical_agent": specialized_caller},  # Переопределить для critical_agent
)

# Расширенная настройка
config = RunnerConfig(
    timeout=60.0,                    # Таймаут на агента
    adaptive=True,                   # Адаптивный режим
    enable_replanning=True,          # Перепланирование
    enable_parallel=True,            # Параллельное выполнение
    max_parallel_size=5,             # Макс. параллельных агентов
    max_retries=2,                   # Повторы при ошибках
    update_states=True,              # Обновлять состояния агентов
    enable_memory=True,              # Включить память
    callbacks=[StdoutCallbackHandler()],  # Callbacks для логирования
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# Синхронное выполнение
result = runner.run_round(graph)

# С явными границами выполнения и фильтрацией
result = runner.run_round(
    graph,
    start_agent_id="input",          # Стартовый агент (переопределяет graph.start_node)
    final_agent_id="output",         # Конечный агент (переопределяет graph.end_node)
    filter_unreachable=True,         # Исключить изолированные ноды (экономия токенов)
    update_states=True,              # Обновлять состояния агентов
)

# Асинхронное выполнение
result = await runner.arun_round(
    graph,
    start_agent_id="input",
    final_agent_id="output",
    filter_unreachable=True,
)

# Выполнение со скрытыми каналами
result = runner.run_round_with_hidden(graph, hidden_encoder=encoder)
```

#### RunnerConfig (Полная спецификация)

```python
from rustworkx_framework.execution import RunnerConfig, RoutingPolicy, PruningConfig

config = RunnerConfig(
    # === Базовые параметры ===
    timeout=60.0,                        # Таймаут на агента (сек)
    max_retries=3,                       # Макс. попыток при ошибках
    update_states=True,                  # Обновлять AgentProfile.state

    # === Адаптивный режим ===
    adaptive=True,                       # Включить адаптивную маршрутизацию
    routing_policy=RoutingPolicy.WEIGHTED_TOPO,  # Политика маршрутизации
    enable_replanning=True,              # Перепланирование в runtime
    replan_on_error_only=False,          # Перепланировать только при ошибках
    replan_interval=5,                   # Перепланировать каждые N шагов

    # === Параллельное выполнение ===
    enable_parallel=True,                # Параллельное выполнение групп
    max_parallel_size=5,                 # Макс. агентов в параллельной группе
    parallel_timeout_factor=1.5,         # Множитель таймаута для группы

    # === Отсечение (Pruning) ===
    pruning_config=PruningConfig(
        min_weight_threshold=0.1,        # Мин. вес ребра
        min_probability_threshold=0.05,  # Мин. вероятность перехода
        max_consecutive_errors=3,        # Макс. ошибок подряд
        token_budget=10000,              # Бюджет токенов для pruning
        enable_fallback=True,            # Использовать fallback-агентов
        max_fallback_attempts=2,         # Макс. попыток fallback
        quality_scorer=None,             # Функция оценки качества
        min_quality_threshold=0.5,       # Мин. качество для продолжения
    ),

    # === Бюджет ===
    budget_config={
        "total_token_limit": 50000,
        "per_node_token_limit": 2000,
        "max_prompt_length": 4000,
        "max_response_length": 2000,
        "warning_threshold": 0.8,
        "time_limit_seconds": 600,
        "request_limit": 100,
    },
    enable_budget_tracking=True,

    # === Память ===
    enable_memory=True,                  # Включить систему памяти
    memory_config=MemoryConfig(
        working_max_entries=20,
        long_term_max_entries=100,
        working_default_ttl=3600.0,
        auto_compress=True,
        promote_after_accesses=3,
    ),
    memory_context_limit=5,              # Записей памяти в промпт
    enable_shared_memory=True,           # Шаринг памяти между агентами

    # === Скрытые каналы ===
    enable_hidden_channels=True,         # Передача hidden_state
    hidden_combine_strategy="mean",      # mean, sum, concat, attention
    pass_embeddings=True,                # Передавать эмбеддинги

    # === Передача Task Query ===
    broadcast_task_to_all=True,          # True: task query передаётся всем агентам
                                         # False: только агентам, соединённым с task нодой

    # === Динамическая топология (Runtime Modification) ===
    enable_dynamic_topology=True,        # Включить изменение графа во время выполнения
    topology_hooks=[my_hook_func],       # Sync hooks для модификации топологии
    async_topology_hooks=[async_hook],   # Async hooks для модификации
    early_stop_conditions=[              # Условия ранней остановки
        EarlyStopCondition.on_keyword("FINAL ANSWER"),
        EarlyStopCondition.on_token_limit(10000),
        EarlyStopCondition.on_custom(lambda ctx: my_logic(ctx)),
    ],

    # === Callbacks (мониторинг и логирование) ===
    callbacks=[                          # Callback handlers
        StdoutCallbackHandler(           # Вывод в консоль
            show_prompts=False,
            show_outputs=True,
        ),
        MetricsCallbackHandler(),        # Агрегация метрик
        FileCallbackHandler("run.jsonl"), # Лог в файл
    ]

    # === Обработка ошибок ===
    error_policy=ErrorPolicy(
        timeout=ErrorAction.SKIP,
        retry_exhausted=ErrorAction.FALLBACK,
        budget_exceeded=ErrorAction.FAIL,
        validation_error=ErrorAction.RETRY,
        max_retries=3,
        retry_delay=1.0,
        exponential_backoff=True,
    ),
    fail_fast=False,                     # Прекратить при первой ошибке

    # === Streaming ===
    enable_streaming=False,              # Включить streaming режим
    stream_tokens=True,                  # Стримить токены LLM
    stream_intermediate_steps=True,      # Стримить промежуточные шаги
)
```

#### Результат выполнения (MACPResult)

```python
result.messages          # Dict[agent_id -> response]
result.final_answer      # Ответ финального агента
result.final_agent_id    # ID финального агента
result.execution_order   # Порядок выполнения
result.agent_states      # Обновлённые состояния агентов
result.total_tokens      # Общее количество токенов
result.total_time        # Время выполнения (сек)
result.replanning_count  # Количество перепланирований
result.fallback_count    # Количество fallback-ов
result.pruned_agents     # Отсечённые агенты (включая disabled и isolated)
result.errors            # Список ошибок
result.hidden_states     # Скрытые состояния агентов
result.metrics           # ExecutionMetrics с детальной статистикой
# Новые поля (динамическая топология)
result.early_stopped           # bool: была ли ранняя остановка
result.early_stop_reason       # str: причина ранней остановки
result.topology_modifications  # int: количество модификаций топологии
```

---

### Планировщик (Scheduler)

Планировщик определяет порядок выполнения агентов.

```python
from rustworkx_framework.execution import (
    build_execution_order,
    get_parallel_groups,
    AdaptiveScheduler,
    RoutingPolicy,
    PruningConfig,
)

# Простой топологический порядок
order = build_execution_order(graph.A_com, agent_ids)

# Группы для параллельного выполнения
groups = get_parallel_groups(graph.A_com, agent_ids)
# Результат: [["a", "b"], ["c"], ["d", "e"]]

# Адаптивный планировщик
scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.WEIGHTED_TOPO,  # Политика маршрутизации
    pruning_config=PruningConfig(
        min_weight_threshold=0.1,        # Мин. вес ребра
        min_probability_threshold=0.05,  # Мин. вероятность
        max_consecutive_errors=3,        # Макс. ошибок подряд
        token_budget=10000,              # Бюджет токенов
        enable_fallback=True,            # Включить fallback
        max_fallback_attempts=2,         # Макс. попыток fallback
    ),
    beam_width=3,                        # Ширина beam search
)

# Построение плана
plan = scheduler.build_plan(
    a_agents,           # Матрица смежности агентов
    agent_ids,          # Список ID
    p_matrix=probs,     # Матрица вероятностей
    end_agent="final",  # Конечный агент
)

# Работа с планом
step = plan.get_current_step()
plan.mark_completed("agent_id", tokens=100)
plan.mark_failed("agent_id")
plan.mark_skipped("agent_id")
```

#### Политики маршрутизации (подробно)

```python
from rustworkx_framework.execution import RoutingPolicy, AdaptiveScheduler

# ========== 1. TOPOLOGICAL (Топологическая сортировка) ==========
# Описание: Классическая топологическая сортировка для DAG
# Использование: Простые пайплайны без адаптивности
# Сложность: O(V + E)

scheduler = AdaptiveScheduler(policy=RoutingPolicy.TOPOLOGICAL)
plan = scheduler.build_plan(adjacency, agent_ids)

# Пример:
#   A → B → C → D
# Порядок: [A, B, C, D]

# ========== 2. WEIGHTED_TOPO (Топологическая с весами) ==========
# Описание: Топологическая сортировка с приоритетом по весам рёбер
# Использование: Когда нужно учитывать важность связей
# Сложность: O(V + E log V)

scheduler = AdaptiveScheduler(policy=RoutingPolicy.WEIGHTED_TOPO)
plan = scheduler.build_plan(adjacency, agent_ids)

# Пример:
#       ┌─(0.9)→ B ─┐
#   A ──┤          ├→ D
#       └─(0.3)→ C ─┘
# Порядок: [A, B, C, D]  (B выполнится раньше C из-за веса 0.9 > 0.3)

# ========== 3. GREEDY (Жадный выбор) ==========
# Описание: На каждом шаге выбирается агент с максимальным весом ребра
# Использование: Оптимизация по качеству связей
# Сложность: O(V²)

scheduler = AdaptiveScheduler(policy=RoutingPolicy.GREEDY)
plan = scheduler.build_plan(
    adjacency,
    agent_ids,
    start_node="coordinator",
    end_node="final",
)

# Пример:
#   Start → A(0.9) → B(0.8) → End
#   Start → C(0.5) → D(0.7) → End
# Выбирается: Start → A → B → End (суммарный вес выше)

# ========== 4. BEAM_SEARCH (Поиск лучом) ==========
# Описание: Поддерживает beam_width лучших путей, выбирает оптимальный
# Использование: Баланс между качеством и скоростью
# Сложность: O(V * beam_width * E)

scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.BEAM_SEARCH,
    beam_width=3,  # Держать 3 лучших пути
)

plan = scheduler.build_plan(
    adjacency,
    agent_ids,
    p_matrix=probability_matrix,  # Вероятности переходов
)

# Пример с beam_width=2:
#   Start ─┬→ A(0.8) ─┬→ B(0.9) → End  [путь 1: 0.72]
#          │          └→ C(0.6) → End  [путь 2: 0.48]
#          └→ D(0.7) ─→ E(0.8) → End   [путь 3: 0.56]
# Beam держит пути 1 и 3, отбрасывает путь 2
# Итоговый выбор: путь 1

# ========== 5. K_SHORTEST (K кратчайших путей) ==========
# Описание: Находит K кратчайших путей, выбирает лучший по критерию
# Использование: Когда нужны альтернативные маршруты
# Сложность: O(K * (V + E) log V)

scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.K_SHORTEST,
    k_paths=5,  # Найти 5 кратчайших путей
)

plan = scheduler.build_plan(
    adjacency,
    agent_ids,
    start_node="input",
    end_node="output",
    path_metric=PathMetric.WEIGHTED,  # HOP_COUNT, WEIGHTED, RELIABILITY
)

# Пример:
# Найденные пути:
#   1. input → A → B → output  (cost=3, hops=3)
#   2. input → C → output      (cost=4, hops=2)
#   3. input → A → D → output  (cost=5, hops=3)
#   4. input → E → F → output  (cost=6, hops=3)
#   5. input → G → output      (cost=7, hops=2)
# Выбор на основе metric: путь 1 (минимальная стоимость)

# ========== 6. GNN_BASED (На основе GNN) ==========
# Описание: Использует обученную GNN для предсказания оптимального пути
# Использование: Адаптивная маршрутизация на основе истории
# Требует: Обученную GNN модель

from rustworkx_framework.core.gnn import GNNRouterInference

scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.GNN_BASED,
    gnn_router=gnn_inference,     # GNNRouterInference объект
    gnn_threshold=0.7,            # Мин. confidence для использования GNN
)

# При confidence < threshold используется fallback политика
scheduler.set_fallback_policy(RoutingPolicy.WEIGHTED_TOPO)

plan = scheduler.build_plan(
    adjacency,
    agent_ids,
    metrics_tracker=tracker,  # Для GNN признаков
)

# ========== Сравнение политик ==========

# | Политика       | Адаптивность | Сложность | Качество | Use Case                    |
# |----------------|--------------|-----------|----------|-----------------------------|
# | TOPOLOGICAL    | Нет          | O(V+E)    | ⭐       | Простые пайплайны          |
# | WEIGHTED_TOPO  | Низкая       | O(V+E·logV)| ⭐⭐     | Пайплайны с приоритетами   |
# | GREEDY         | Средняя      | O(V²)     | ⭐⭐⭐    | Оптимизация по весам       |
# | BEAM_SEARCH    | Высокая      | O(V·k·E)  | ⭐⭐⭐⭐   | Баланс качества и скорости |
# | K_SHORTEST     | Высокая      | O(K·V·logV)| ⭐⭐⭐⭐   | Поиск альтернатив          |
# | GNN_BASED      | Очень высокая| O(GNN)    | ⭐⭐⭐⭐⭐  | Обученные системы          |

# ========== Выбор политики в зависимости от задачи ==========

# Простой линейный пайплайн
config = RunnerConfig(routing_policy=RoutingPolicy.TOPOLOGICAL)

# Граф с разными приоритетами агентов
config = RunnerConfig(routing_policy=RoutingPolicy.WEIGHTED_TOPO)

# Оптимизация качества маршрута
config = RunnerConfig(routing_policy=RoutingPolicy.GREEDY)

# Баланс между исследованием и эксплуатацией
config = RunnerConfig(
    routing_policy=RoutingPolicy.BEAM_SEARCH,
    adaptive=True,
)
scheduler = AdaptiveScheduler(policy=RoutingPolicy.BEAM_SEARCH, beam_width=3)

# Нужны запасные варианты
config = RunnerConfig(routing_policy=RoutingPolicy.K_SHORTEST)
scheduler = AdaptiveScheduler(policy=RoutingPolicy.K_SHORTEST, k_paths=3)

# Продвинутая система с обучением
config = RunnerConfig(routing_policy=RoutingPolicy.GNN_BASED)
scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.GNN_BASED,
    gnn_router=trained_router,
)
```

---

### Система памяти (Memory System)

Стратифицированная память с **working** и **long-term** уровнями, поддержкой TTL, тегов, приоритетов и автоматического сжатия.

#### Архитектура памяти

```
┌─────────────────────────────────────────────────────────────┐
│                       AgentMemory                           │
│  ┌────────────────────┐     ┌──────────────────────┐       │
│  │  Working Memory    │     │  Long-term Memory    │       │
│  │  (TTL: 1 час)      │     │  (TTL: ∞)            │       │
│  │  Max: 20 entries   │     │  Max: 100 entries    │       │
│  │                    │     │                      │       │
│  │  - Recent messages │────▶│  - Important facts   │       │
│  │  - Temp context    │     │  - Key insights      │       │
│  │  - Active tasks    │     │  - Historical data   │       │
│  └────────────────────┘     └──────────────────────┘       │
│         ▲                            ▲                      │
│         │ promotion                  │                      │
│         │ (after N accesses)         │                      │
│         └────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
         │
         │ sharing
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SharedMemoryPool                           │
│  Обмен памятью между агентами                               │
│  - Broadcast: один → все                                    │
│  - Share: один → выбранные                                  │
│  - Query: поиск по тегам                                    │
└─────────────────────────────────────────────────────────────┘
```

---

#### Базовое использование AgentMemory

```python
from rustworkx_framework.utils.memory import (
    AgentMemory,
    MemoryConfig,
    MemoryLevel,
    MemoryEntry,
)

# 1. Конфигурация памяти
config = MemoryConfig(
    # Working memory (кратковременная)
    working_max_entries=20,         # Макс. записей
    working_default_ttl=3600.0,     # TTL: 1 час

    # Long-term memory (долговременная)
    long_term_max_entries=100,      # Макс. записей
    long_term_default_ttl=None,     # Бессрочно

    # Автоматическое управление
    auto_compress=True,             # Автосжатие при превышении лимита
    compress_strategy="truncate",   # truncate, summarize
    promote_after_accesses=3,       # Продвижение в long-term после N доступов

    # Приоритизация
    use_priority=True,              # Учитывать приоритеты при очистке
    priority_weight=0.3,            # Вес приоритета vs recency
)

# 2. Создание памяти агента
memory = AgentMemory("researcher", config)

# 3. Добавление записей
# 3.1. Добавление сообщений (самый простой способ)
memory.add_message(role="user", content="Analyze the dataset")
memory.add_message(role="assistant", content="I will analyze it")

# 3.2. Добавление с параметрами
memory.add(
    content={"type": "insight", "text": "Pattern detected in data"},
    level=MemoryLevel.WORKING,      # WORKING или LONG_TERM
    priority=5,                     # 0-10 (чем выше, тем важнее)
    tags={"insight", "data"},       # Теги для поиска
    ttl=7200.0,                     # Custom TTL (2 часа)
    metadata={"source": "analysis", "confidence": 0.95},
)

# 3.3. Добавление напрямую в long-term
memory.add(
    content="Critical finding: correlation coefficient = 0.87",
    level=MemoryLevel.LONG_TERM,
    priority=10,
    tags={"critical", "finding"},
)

# 4. Получение записей
# 4.1. Получить последние сообщения
messages = memory.get_messages(limit=5)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

# 4.2. Получить из working memory
working_entries = memory.get(level=MemoryLevel.WORKING, limit=10)
for entry in working_entries:
    print(f"[{entry.priority}] {entry.content}")

# 4.3. Получить из long-term
longterm_entries = memory.get(level=MemoryLevel.LONG_TERM)

# 4.4. Поиск по тегам
insights = memory.search_by_tags({"insight"}, level=MemoryLevel.WORKING)
critical = memory.search_by_tags({"critical"}, level=MemoryLevel.LONG_TERM)

# 4.5. Получить все записи
all_entries = memory.get_all()

# 5. Управление памятью
# 5.1. Удалить запись
memory.remove(entry_key)

# 5.2. Очистить уровень
memory.clear(level=MemoryLevel.WORKING)

# 5.3. Принудительное сжатие
memory.compress(level=MemoryLevel.WORKING)

# 5.4. Продвинуть запись в long-term
memory.promote(entry_key)

# 5.5. Обновить запись
memory.update(entry_key, new_content={"updated": "data"})

# 6. Статистика
stats = memory.get_stats()
print(f"Working: {stats['working_count']}/{stats['working_max']}")
print(f"Long-term: {stats['longterm_count']}/{stats['longterm_max']}")
print(f"Total accesses: {stats['total_accesses']}")
print(f"Promotions: {stats['promotion_count']}")
```

---

#### SharedMemoryPool — обмен памятью между агентами

```python
from rustworkx_framework.utils.memory import SharedMemoryPool

# 1. Создание пула
pool = SharedMemoryPool(max_shared_entries=1000)

# 2. Регистрация агентов
memory_a = AgentMemory("agent_a", config)
memory_b = AgentMemory("agent_b", config)
memory_c = AgentMemory("agent_c", config)

pool.register(memory_a)
pool.register(memory_b)
pool.register(memory_c)

# 3. Broadcast — отправить всем
pool.broadcast(
    from_agent="agent_a",
    entry={
        "content": "Important discovery: X correlates with Y",
        "priority": 8,
        "tags": {"discovery", "shared"},
    },
)

# Все агенты получат эту запись в working memory

# 4. Share — отправить конкретным агентам
pool.share(
    from_agent="agent_a",
    entry={"content": "Secret info", "priority": 9},
    to_agents=["agent_b", "agent_c"],
)

# Только agent_b и agent_c получат запись

# 5. Query — запросить информацию из пула
results = pool.query(
    tags={"discovery"},
    min_priority=5,
    limit=10,
)

for result in results:
    print(f"From {result['source_agent']}: {result['content']}")

# 6. Подписка на обновления (callback)
def on_shared_entry(entry, from_agent, to_agents):
    print(f"{from_agent} shared: {entry['content']}")

pool.subscribe("agent_b", on_shared_entry)

# 7. Удаление из пула
pool.unregister("agent_c")

# 8. Очистка пула
pool.clear()
```

---

#### Сжатие памяти (Compression)

```python
from rustworkx_framework.utils.memory import (
    TruncateCompressor,
    SummaryCompressor,
)

# 1. Truncate — простое обрезание старых записей
compressor = TruncateCompressor(keep_ratio=0.5)  # Оставить 50%

memory = AgentMemory("agent", config)
memory.set_compressor(compressor)

# При превышении лимита автоматически удалятся 50% старых записей

# 2. Summary — суммаризация с помощью LLM
def summarize_llm(entries: list[MemoryEntry]) -> str:
    texts = [e.content for e in entries]
    combined = "\n".join(texts)
    return my_llm(f"Summarize these entries: {combined}")

compressor = SummaryCompressor(
    summarizer=summarize_llm,
    chunk_size=10,  # Суммаризировать по 10 записей
)

memory.set_compressor(compressor)

# При сжатии 10 записей заменяются на 1 суммаризированную

# 3. Кастомный компрессор
from rustworkx_framework.utils.memory import MemoryCompressor

class SmartCompressor(MemoryCompressor):
    def compress(self, entries: list[MemoryEntry], target_count: int) -> list[MemoryEntry]:
        # Удалить низкоприоритетные и старые записи
        sorted_entries = sorted(
            entries,
            key=lambda e: (e.priority, e.timestamp),
            reverse=True,
        )
        return sorted_entries[:target_count]

memory.set_compressor(SmartCompressor())
```

---

#### Интеграция памяти с Runner

```python
from rustworkx_framework.execution import MACPRunner, RunnerConfig

# 1. Конфигурация с памятью
config = RunnerConfig(
    enable_memory=True,
    memory_config=MemoryConfig(
        working_max_entries=20,
        long_term_max_entries=100,
        auto_compress=True,
        promote_after_accesses=3,
    ),
    memory_context_limit=5,      # Сколько записей включать в промпт
    enable_shared_memory=True,   # Включить SharedMemoryPool
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# 2. Выполнение — память автоматически обновляется
result1 = runner.run_round(graph)

# 3. Доступ к памяти агента
memory = runner.get_agent_memory("researcher")

entries = memory.get_messages(limit=10)
print(f"Researcher memory: {entries}")

# 4. Ручное добавление в память
runner.add_to_memory(
    "researcher",
    content="External knowledge: XYZ",
    level=MemoryLevel.LONG_TERM,
    priority=8,
)

# 5. Второй раунд — агенты помнят контекст
graph.query = "Continue analysis from previous round"
result2 = runner.run_round(graph)

# 6. Экспорт памяти
memory_export = runner.export_memories()
# {
#   "agent_a": {"working": [...], "long_term": [...]},
#   "agent_b": {"working": [...], "long_term": [...]},
# }

# 7. Импорт памяти (восстановление состояния)
runner.import_memories(memory_export)

# 8. Очистка памяти всех агентов
runner.clear_all_memories()
```

---

#### Продвинутое использование: Семантический поиск в памяти

```python
from rustworkx_framework.utils.memory import SemanticMemoryIndex
from rustworkx_framework.core import NodeEncoder

# 1. Создание семантического индекса
encoder = NodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

semantic_index = SemanticMemoryIndex(encoder)

# 2. Добавление записей в индекс
memory = AgentMemory("agent", config)

for entry in memory.get_all():
    semantic_index.add(entry.key, entry.content, entry.tags)

# 3. Семантический поиск
query = "findings about correlation"
results = semantic_index.search(
    query,
    top_k=5,
    min_similarity=0.7,
    filter_tags={"finding"},
)

for result in results:
    print(f"[{result['similarity']:.3f}] {result['content']}")

# 4. Интеграция с AgentMemory
memory.enable_semantic_search(encoder)

# Теперь можно искать семантически
results = memory.semantic_search(
    query="data patterns",
    top_k=3,
    level=MemoryLevel.LONG_TERM,
)
```

---

#### Практический пример: Multi-round conversation с памятью

```python
# Создание графа с памятью
agents = [
    AgentProfile(identifier="analyzer", display_name="Data Analyzer"),
    AgentProfile(identifier="reporter", display_name="Report Writer"),
]

graph = build_property_graph(
    agents,
    workflow_edges=[("analyzer", "reporter")],
    query="Analyze dataset.csv",
)

# Конфигурация с памятью
config = RunnerConfig(
    enable_memory=True,
    memory_config=MemoryConfig(
        working_max_entries=15,
        long_term_max_entries=50,
        auto_compress=True,
        promote_after_accesses=2,
    ),
    memory_context_limit=5,
    enable_shared_memory=True,
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# Round 1: Начальный анализ
graph.query = "Analyze the dataset and find key patterns"
result1 = runner.run_round(graph)

print(f"Round 1 answer: {result1.final_answer}")

# Analyzer сохранил находки в память
analyzer_memory = runner.get_agent_memory("analyzer")
print(f"Analyzer memory entries: {len(analyzer_memory.get_all())}")

# Round 2: Углублённый анализ (агенты помнят предыдущий раунд)
graph.query = "Based on previous findings, analyze correlations"
result2 = runner.run_round(graph)

print(f"Round 2 answer: {result2.final_answer}")

# Round 3: Генерация отчёта
graph.query = "Generate final report summarizing all findings"
result3 = runner.run_round(graph)

print(f"Round 3 answer: {result3.final_answer}")

# Reporter использовал накопленную память для полного отчёта
reporter_memory = runner.get_agent_memory("reporter")

# Экспорт всей истории
history = {
    "round_1": result1.to_dict(),
    "round_2": result2.to_dict(),
    "round_3": result3.to_dict(),
    "memories": runner.export_memories(),
}

import json
with open("conversation_history.json", "w") as f:
    json.dump(history, f, indent=2)
```

---

### Streaming API

LangGraph-like streaming для real-time вывода.

```python
from rustworkx_framework.execution import (
    MACPRunner,
    StreamEventType,
    StreamBuffer,
    format_event,
    print_stream,
)

runner = MACPRunner(llm_caller=my_llm)

# Синхронный streaming
for event in runner.stream(graph):
    if event.event_type == StreamEventType.AGENT_OUTPUT:
        print(f"{event.agent_id}: {event.content}")
    elif event.event_type == StreamEventType.TOKEN:
        print(event.token, end="", flush=True)

# Асинхронный streaming
async for event in runner.astream(graph):
    print(format_event(event))

# Использование буфера
buffer = StreamBuffer()
for event in runner.stream(graph):
    buffer.add(event)
    # ... обработка события

print(f"Итоговый ответ: {buffer.final_answer}")
print(f"Выходы агентов: {buffer.agent_outputs}")

# Удобная печать
answer = print_stream(runner.stream(graph), show_tokens=True)
```

#### Типы событий (полная спецификация)

```python
from rustworkx_framework.execution.streaming import StreamEventType, StreamEvent

# === Жизненный цикл выполнения ===
StreamEventType.RUN_START
# Поля: run_id, query, num_agents, config

StreamEventType.RUN_END
# Поля: run_id, success, total_time, total_tokens, execution_order, final_answer

# === События агентов ===
StreamEventType.AGENT_START
# Поля: agent_id, step_index, predecessors, prompt_preview

StreamEventType.AGENT_OUTPUT
# Поля: agent_id, step_index, content, tokens_used, latency_ms

StreamEventType.AGENT_ERROR
# Поля: agent_id, step_index, error_type, error_message, will_retry

# === Streaming токенов ===
StreamEventType.TOKEN
# Поля: agent_id, token (str), token_index

# === Адаптивное выполнение ===
StreamEventType.REPLAN
# Поля: reason, old_plan, new_plan, remaining_steps

StreamEventType.PRUNE
# Поля: agent_id, reason (low_weight/low_probability/budget/quality)

StreamEventType.FALLBACK
# Поля: original_agent, fallback_agent, reason, attempt

# === Параллельное выполнение ===
StreamEventType.PARALLEL_START
# Поля: group_agents (list), group_index

StreamEventType.PARALLEL_END
# Поля: group_agents, completed_count, failed_count, duration_ms

# === Бюджет ===
StreamEventType.BUDGET_WARNING
# Поля: budget_type (tokens/requests/time), current, limit, ratio

StreamEventType.BUDGET_EXCEEDED
# Поля: budget_type, current, limit, action_taken

# === Память ===
StreamEventType.MEMORY_WRITE
# Поля: agent_id, memory_level (working/long_term), entry_key

StreamEventType.MEMORY_READ
# Поля: agent_id, memory_level, entry_key, found

StreamEventType.MEMORY_PROMOTED
# Поля: agent_id, entry_key, from_level, to_level

# === Метрики ===
StreamEventType.METRICS_UPDATE
# Поля: agent_id, metrics (dict с reliability, latency, quality, cost)

# Пример обработки всех типов событий
for event in runner.stream(graph):
    match event.event_type:
        case StreamEventType.RUN_START:
            print(f"Starting run {event.run_id} with {event.num_agents} agents")

        case StreamEventType.AGENT_START:
            print(f"Agent {event.agent_id} starting (step {event.step_index})")

        case StreamEventType.AGENT_OUTPUT:
            print(f"Agent {event.agent_id}: {event.content[:100]}...")
            print(f"  Tokens: {event.tokens_used}, Latency: {event.latency_ms}ms")

        case StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

        case StreamEventType.REPLAN:
            print(f"⟳ Replanning: {event.reason}")
            print(f"  New plan: {event.new_plan}")

        case StreamEventType.PRUNE:
            print(f"✂ Pruned {event.agent_id}: {event.reason}")

        case StreamEventType.FALLBACK:
            print(f"⤷ Fallback: {event.original_agent} → {event.fallback_agent}")

        case StreamEventType.PARALLEL_START:
            print(f"⫸ Starting parallel group: {event.group_agents}")

        case StreamEventType.PARALLEL_END:
            print(f"⫷ Parallel group done: {event.completed_count}/{len(event.group_agents)}")

        case StreamEventType.BUDGET_WARNING:
            print(f"⚠ Budget warning: {event.budget_type} at {event.ratio:.1%}")

        case StreamEventType.BUDGET_EXCEEDED:
            print(f"❌ Budget exceeded: {event.budget_type}")

        case StreamEventType.RUN_END:
            print(f"✓ Execution completed in {event.total_time:.2f}s")
            print(f"  Total tokens: {event.total_tokens}")
            print(f"  Final answer: {event.final_answer[:100]}...")
```

---

## Продвинутые возможности

### Оптимизация выполнения и экономия токенов

Фреймворк предоставляет несколько механизмов для оптимизации выполнения и экономии токенов:

#### 1. Фильтрация изолированных нод

Автоматическое исключение нод, не лежащих на пути от start к end:

```python
# Установить границы выполнения
graph.set_execution_bounds("input", "output")

# При выполнении фильтровать изолированные ноды
result = runner.run_round(
    graph,
    filter_unreachable=True  # Исключить ноды не на пути input->output
)

# Ноды, не связанные с путём input->output, не выполнятся
print(f"Исключено агентов: {len(result.pruned_agents or [])}")
```

**Пример:**

```python
builder = GraphBuilder()
builder.add_agent("a1")
builder.add_agent("a2")
builder.add_agent("a3")
builder.add_agent("isolated")  # Не связана с a1->a3

builder.add_workflow_edge("a1", "a2")
builder.add_workflow_edge("a2", "a3")
builder.set_execution_bounds("a1", "a3")

graph = builder.build()

# Анализ достижимости
relevant = graph.get_relevant_nodes()    # {"a1", "a2", "a3"}
isolated = graph.get_isolated_nodes()    # {"isolated"}

result = runner.run_round(graph, filter_unreachable=True)
# "isolated" не выполнится → экономия токенов
```

#### 2. Деактивация нод (Disabled Nodes)

Временная деактивация нод без удаления из графа:

```python
# Деактивировать на основе метрик/RL
if quality_score < threshold:
    graph.disable("expensive_agent")

# Или несколько
graph.disable(["agent1", "agent2"])

# Проверить
if graph.is_enabled("agent1"):
    ...

# Активировать обратно
graph.enable("agent1")
graph.enable()  # Все

result = runner.run_round(graph)
# Деактивированные ноды в result.pruned_agents
```

**Use case: RL-управление**

```python
# RL-агент решает какие ноды деактивировать
for agent_id in graph.node_ids:
    rl_score = rl_model.predict(graph_state, agent_id)
    if rl_score < 0.3:
        graph.disable(agent_id)

result = runner.run_round(graph)
```

#### 3. Ранняя остановка (Early Stopping)

Остановка выполнения при достижении условия:

```python
from rustworkx_framework import EarlyStopCondition, RunnerConfig

# По ключевому слову
stop1 = EarlyStopCondition.on_keyword("FINAL ANSWER")

# По лимиту токенов
stop2 = EarlyStopCondition.on_token_limit(5000)

# По количеству агентов
stop3 = EarlyStopCondition.on_agent_count(3)

# По metadata (для RL/метрик)
stop4 = EarlyStopCondition.on_metadata(
    "quality", 0.95,
    comparator=lambda v, t: v > t
)

# Произвольная логика
stop5 = EarlyStopCondition.on_custom(
    lambda ctx: my_evaluator.is_done(ctx.messages),
    reason="Evaluator decided task is done",
    min_agents_executed=2  # Минимум 2 агента до проверки
)

# Комбинация (OR)
stop_any = EarlyStopCondition.combine_any([
    EarlyStopCondition.on_keyword("DONE"),
    EarlyStopCondition.on_token_limit(10000),
])

config = RunnerConfig(
    early_stop_conditions=[stop1, stop2, stop5]
)
runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(graph)

if result.early_stopped:
    print(f"Причина: {result.early_stop_reason}")
    saved = len(graph.node_ids) - len(result.execution_order)
    print(f"Сэкономлено агентов: {saved}")
```

#### 4. Runtime топология (Topology Hooks)

Изменение графа **во время выполнения** на основе промежуточных результатов:

```python
from rustworkx_framework import TopologyAction, StepContext

def adaptive_topology(ctx: StepContext, graph) -> TopologyAction:
    """Hook вызывается после каждого агента."""

    # ctx.agent_id — текущий агент
    # ctx.response — его ответ
    # ctx.messages — все ответы
    # ctx.execution_order — порядок выполнения
    # ctx.remaining_agents — оставшиеся
    # ctx.total_tokens — использовано токенов

    # Добавить ребро если нужна проверка
    if "uncertain" in (ctx.response or "").lower():
        return TopologyAction(
            add_edges=[(ctx.agent_id, "reviewer", 1.0)],
            trigger_replan=True
        )

    # Удалить ребро
    if confident:
        return TopologyAction(
            remove_edges=[("agent1", "checker")]
        )

    # Пропустить агентов
    if ctx.total_tokens > 8000:
        return TopologyAction(
            skip_agents=["expensive_agent"]
        )

    # Ранняя остановка
    if "DONE" in (ctx.response or ""):
        return TopologyAction(
            early_stop=True,
            early_stop_reason="Task completed"
        )

    return None

config = RunnerConfig(
    enable_dynamic_topology=True,
    topology_hooks=[adaptive_topology]
)
```

#### 5. Комбинированная оптимизация

Все механизмы вместе для максимальной оптимизации:

```python
from rustworkx_framework import (
    GraphBuilder, MACPRunner, RunnerConfig,
    EarlyStopCondition, TopologyAction, StepContext
)

# Построить граф
builder = GraphBuilder()
builder.add_agent("input")
builder.add_agent("solver")
builder.add_agent("checker")
builder.add_agent("expert")      # Дорогой агент
builder.add_agent("formatter")
builder.add_agent("optional")    # Опциональный

builder.add_workflow_edge("input", "solver")
builder.add_workflow_edge("solver", "checker")
builder.add_workflow_edge("checker", "formatter")

# Установить границы
builder.set_execution_bounds("input", "formatter")

graph = builder.build()

# Деактивировать опциональные ноды
graph.disable("optional")

# Hooks для адаптации
def smart_topology(ctx: StepContext, graph) -> TopologyAction:
    # Если solver уверен — пропустить checker
    if ctx.agent_id == "solver" and ctx.metadata.get("confidence", 0) > 0.95:
        return TopologyAction(skip_agents=["checker"])

    # Если checker нашёл проблему — добавить expert
    if ctx.agent_id == "checker" and "ERROR" in (ctx.response or ""):
        return TopologyAction(
            add_edges=[("checker", "expert", 1.0), ("expert", "formatter", 1.0)],
            trigger_replan=True
        )

    return None

# Настроить runner с оптимизацией
config = RunnerConfig(
    adaptive=True,
    enable_dynamic_topology=True,
    topology_hooks=[smart_topology],
    early_stop_conditions=[
        EarlyStopCondition.on_keyword("FINAL_ANSWER"),
        EarlyStopCondition.on_token_limit(10000),
    ],
    pruning_config=PruningConfig(token_budget=15000),
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(
    graph,
    filter_unreachable=True  # Исключить isolated ноды
)

# Анализ оптимизации
print(f"Выполнено агентов: {len(result.execution_order)}")
print(f"Исключено: {len(result.pruned_agents or [])}")
print(f"Early stopped: {result.early_stopped}")
print(f"Модификаций: {result.topology_modifications}")
print(f"Токенов: {result.total_tokens}")
```

---

### Мультимодельная поддержка (Multi-Model Support)

Каждый агент в графе может использовать свою собственную LLM модель с индивидуальными настройками. Это позволяет:
- **Оптимизировать затраты** — использовать дорогие модели только для сложных задач
- **Балансировать производительность** — быстрые модели для простых операций
- **Специализировать агентов** — модели, обученные на конкретных доменах
- **Гибридные решения** — комбинировать облачные и локальные модели

#### Архитектура мультимодельности

```
┌─────────────────────────────────────────────────────────────┐
│                       TASK NODE                             │
│              "Проанализировать рынок"                       │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐   ┌───────────────┐
│  ANALYST      │   │  COORDINATOR  │
│               │──▶│               │
│ GPT-4         │   │ GPT-4o-mini   │
│ temp: 0.0     │   │ temp: 0.3     │
│ tokens: 4000  │   │ tokens: 1000  │
└───────────────┘   └───────────────┘
```

---

#### Ключевые компоненты

**1. LLMConfig** — конфигурация LLM для агента

```python
from rustworkx_framework.core.schema import LLMConfig

llm_config = LLMConfig(
    model_name="gpt-4",                    # Имя модели
    base_url="https://api.openai.com/v1", # API endpoint
    api_key="$OPENAI_API_KEY",             # Ключ (или $ENV_VAR)
    max_tokens=2000,                       # Макс. токенов в ответе
    temperature=0.7,                       # Температура генерации
    timeout=60.0,                          # Таймаут запроса
    top_p=0.9,                             # Nucleus sampling
    stop_sequences=["END"],                # Стоп-последовательности
)

# Проверка конфигурации
if llm_config.is_configured():
    params = llm_config.to_generation_params()
    print(f"Generation params: {params}")

# Слияние конфигураций (fallback)
default_config = LLMConfig(model_name="gpt-4o-mini", temperature=0.5)
final_config = llm_config.merge_with(default_config)
```

**2. AgentLLMConfig** — иммутабельная конфигурация для AgentProfile

```python
from rustworkx_framework.core.agent import AgentLLMConfig

agent_llm_config = AgentLLMConfig(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2000,
)

# Конвертация в LLMConfig
llm_config = agent_llm_config.to_llm_config()
```

**3. LLMCallerFactory** — фабрика для создания LLM callers

```python
from rustworkx_framework.execution import LLMCallerFactory

# Создание фабрики для OpenAI-совместимых API
factory = LLMCallerFactory.create_openai_factory(
    default_model="gpt-4o-mini",
    default_base_url="https://api.openai.com/v1",
    default_api_key="sk-...",
    default_temperature=0.7,
    default_max_tokens=2000,
)

# Фабрика автоматически создаёт callers на основе AgentLLMConfig
# При использовании с MACPRunner
```

**4. create_openai_caller** — утилита для создания caller

```python
from rustworkx_framework.execution import create_openai_caller

# Синхронный caller
caller = create_openai_caller(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2000,
    is_async=False,
    is_streaming=False,
)

response = caller("What is 2+2?")

# Асинхронный caller
async_caller = create_openai_caller(
    model="gpt-4",
    is_async=True,
)

response = await async_caller("What is 2+2?")

# Streaming caller
streaming_caller = create_openai_caller(
    model="gpt-4",
    is_streaming=True,
)

for token in streaming_caller("What is 2+2?"):
    print(token, end="", flush=True)
```

---

#### Способы конфигурации мультимодельности

##### Способ 1: Через GraphBuilder (рекомендуется)

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner, LLMCallerFactory

builder = GraphBuilder()

# Агент 1: Сильная модель для анализа
builder.add_agent(
    identifier="analyst",
    display_name="Senior Analyst",
    persona="Expert data analyst with deep domain knowledge",
    llm_backbone="gpt-4",               # Или model_name
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.0,                    # Строгий анализ
    max_tokens=4000,
    timeout=120.0,
)

# Агент 2: Слабая модель для форматирования
builder.add_agent(
    identifier="formatter",
    display_name="Report Formatter",
    persona="Formats data into readable reports",
    llm_backbone="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.3,
    max_tokens=1000,
    timeout=30.0,
)

# Агент 3: Локальная модель для конфиденциальных данных
builder.add_agent(
    identifier="privacy_checker",
    display_name="Privacy Checker",
    llm_backbone="llama3:70b",
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="not-needed",
    temperature=0.1,
    max_tokens=500,
)

builder.add_workflow_edge("analyst", "formatter")
builder.add_workflow_edge("analyst", "privacy_checker")

graph = builder.build()

# Фабрика автоматически создаст callers для каждого агента
factory = LLMCallerFactory.create_openai_factory()

runner = MACPRunner(llm_factory=factory)
result = runner.run_round(graph, query="Analyze Q4 sales data")

print(f"Final answer: {result.final_answer}")
```

##### Способ 2: Явное указание LLMConfig

```python
from rustworkx_framework.core.schema import LLMConfig

# Предопределённые конфигурации
gpt4_config = LLMConfig(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.7,
    max_tokens=2000,
)

gpt4_mini_config = LLMConfig(
    model_name="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.5,
    max_tokens=1000,
)

builder = GraphBuilder()
builder.add_agent(
    "researcher",
    display_name="Researcher",
    llm_config=gpt4_config,  # Передать готовую конфигурацию
)
builder.add_agent(
    "writer",
    display_name="Writer",
    llm_config=gpt4_mini_config,
)

graph = builder.build()
```

##### Способ 3: Словарь llm_callers

```python
from rustworkx_framework.execution import create_openai_caller

# Создать callers вручную
callers = {
    "analyst": create_openai_caller(
        model="gpt-4",
        temperature=0.0,
        max_tokens=4000,
    ),
    "formatter": create_openai_caller(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
    ),
    "privacy_checker": create_openai_caller(
        model="llama3:70b",
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
    ),
}

# Передать напрямую в runner
runner = MACPRunner(llm_callers=callers)
result = runner.run_round(graph)
```

##### Способ 4: Комбинированный подход

```python
# Использовать фабрику как default, но переопределить для некоторых агентов
factory = LLMCallerFactory.create_openai_factory(
    default_model="gpt-4o-mini",  # По умолчанию
)

# Создать кастомный caller для специфического агента
specialized_caller = create_openai_caller(
    model="gpt-4",
    temperature=0.0,
    max_tokens=4000,
)

runner = MACPRunner(
    llm_factory=factory,                    # Для всех агентов
    llm_callers={"analyst": specialized_caller},  # Переопределить для analyst
)
```

---

#### Приоритеты разрешения LLM caller

```
1. llm_callers[agent_id]       ← Явно указанный caller
        ↓
2. llm_factory.get_caller()    ← Фабрика создаёт на основе agent.llm_config
        ↓
3. llm_caller                  ← Default caller для всех агентов
        ↓
4. Exception                   ← Ошибка: не указан ни один caller
```

---

#### Примеры использования

##### Пример 1: Оптимизация затрат

```python
# Дешёвая модель для рутинных операций, дорогая — для сложных

builder = GraphBuilder()

# 5 простых аналитиков (дешёвая модель)
for i in range(5):
    builder.add_agent(
        f"analyst_{i}",
        display_name=f"Junior Analyst {i}",
        llm_backbone="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500,
    )
    builder.add_workflow_edge(f"analyst_{i}", "senior")

# 1 старший аналитик (дорогая модель)
builder.add_agent(
    "senior",
    display_name="Senior Analyst",
    llm_backbone="gpt-4",
    temperature=0.7,
    max_tokens=4000,
)

graph = builder.build()

# Экономия: ~80% токенов используют дешёвую модель
```

##### Пример 2: Гибридное решение (облако + локальная модель)

```python
builder = GraphBuilder()

# Публичные данные → облачная модель
builder.add_agent(
    "public_analyzer",
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
)

# Конфиденциальные данные → локальная модель
builder.add_agent(
    "private_analyzer",
    llm_backbone="llama3:70b",
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
)

# Агрегатор → дешёвая облачная модель
builder.add_agent(
    "aggregator",
    llm_backbone="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
)

builder.add_workflow_edge("public_analyzer", "aggregator")
builder.add_workflow_edge("private_analyzer", "aggregator")

graph = builder.build()
```

##### Пример 3: Специализированные модели

```python
builder = GraphBuilder()

# Медицинский эксперт → модель, обученная на медицинских данных
builder.add_agent(
    "medical_expert",
    llm_backbone="medical-llm-v2",
    base_url="https://medical-api.example.com/v1",
    api_key="$MEDICAL_API_KEY",
    temperature=0.0,  # Строгие медицинские рекомендации
)

# Юридический эксперт → модель, обученная на юридических текстах
builder.add_agent(
    "legal_expert",
    llm_backbone="legal-llm-v3",
    base_url="https://legal-api.example.com/v1",
    api_key="$LEGAL_API_KEY",
    temperature=0.0,
)

# Координатор → общая модель
builder.add_agent(
    "coordinator",
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.5,
)

builder.add_workflow_edge("medical_expert", "coordinator")
builder.add_workflow_edge("legal_expert", "coordinator")

graph = builder.build()
```

##### Пример 4: Разные температуры для разных стилей

```python
builder = GraphBuilder()

# Креативный писатель (высокая температура)
builder.add_agent(
    "creative_writer",
    llm_backbone="gpt-4",
    temperature=0.9,  # Креативность
    max_tokens=2000,
)

# Строгий редактор (низкая температура)
builder.add_agent(
    "strict_editor",
    llm_backbone="gpt-4",
    temperature=0.1,  # Точность
    max_tokens=1500,
)

# Финальный форматёр (средняя температура)
builder.add_agent(
    "formatter",
    llm_backbone="gpt-4o-mini",
    temperature=0.5,  # Баланс
    max_tokens=1000,
)

builder.add_workflow_edge("creative_writer", "strict_editor")
builder.add_workflow_edge("strict_editor", "formatter")

graph = builder.build()
```

---

#### Поддерживаемые провайдеры

Фреймворк поддерживает **любые OpenAI-совместимые API**:

| Провайдер | Base URL | Примечания |
|-----------|----------|------------|
| **OpenAI** | `https://api.openai.com/v1` | GPT-4, GPT-4o-mini, GPT-3.5-turbo |
| **Anthropic** | через wrapper | Claude (требует адаптер) |
| **Ollama** | `http://localhost:11434/v1` | Локальные модели (llama3, mistral, etc.) |
| **vLLM** | custom | Self-hosted модели |
| **LiteLLM** | custom | Unified API для всех провайдеров |
| **Azure OpenAI** | `https://<resource>.openai.azure.com/` | Azure-hosted модели |
| **GigaChat** | custom | Модели Сбера |
| **Cloudflare Tunnels** | custom | Через cloudflare tunnels |

```python
# Примеры разных провайдеров

# OpenAI
builder.add_agent("agent1", llm_backbone="gpt-4",
                  base_url="https://api.openai.com/v1")

# Ollama (локально)
builder.add_agent("agent2", llm_backbone="llama3:70b",
                  base_url="http://localhost:11434/v1")

# Azure OpenAI
builder.add_agent("agent3", llm_backbone="gpt-4",
                  base_url="https://myresource.openai.azure.com/")

# GigaChat
builder.add_agent("agent4", llm_backbone="GigaChat-Lightning",
                  base_url="https://gigachat-api.trycloudflare.com/v1")

# vLLM
builder.add_agent("agent5", llm_backbone="./models/Qwen3-80B",
                  base_url="https://my-vllm-server.com/v1")
```

---

#### Асинхронная и Streaming поддержка

```python
from rustworkx_framework.execution import create_openai_caller

# Асинхронный caller для каждого агента
async_callers = {
    "agent1": create_openai_caller(model="gpt-4", is_async=True),
    "agent2": create_openai_caller(model="gpt-4o-mini", is_async=True),
}

runner = MACPRunner(async_llm_callers=async_callers)
result = await runner.arun_round(graph)

# Streaming callers
streaming_callers = {
    "agent1": create_openai_caller(model="gpt-4", is_streaming=True),
    "agent2": create_openai_caller(model="gpt-4o-mini", is_streaming=True),
}

runner = MACPRunner(streaming_llm_callers=streaming_callers)

for event in runner.stream(graph):
    if event.event_type == StreamEventType.TOKEN:
        print(f"[{event.agent_id}] {event.token}", end="")
```

---

#### Обработка API ключей

```python
# 1. Прямое указание
builder.add_agent("agent", api_key="sk-...")

# 2. Из переменной окружения (рекомендуется)
builder.add_agent("agent", api_key="$OPENAI_API_KEY")

# При парсинге автоматически резолвится в os.getenv("OPENAI_API_KEY")

# 3. Из файла
import os
os.environ["OPENAI_API_KEY"] = open("keys/openai.key").read().strip()
builder.add_agent("agent", api_key="$OPENAI_API_KEY")
```

---

#### Мониторинг мультимодельного выполнения

```python
from rustworkx_framework.core.metrics import MetricsTracker

tracker = MetricsTracker()

runner = MACPRunner(
    llm_factory=factory,
    metrics_tracker=tracker,
)

result = runner.run_round(graph)

# Анализ по моделям
for agent_id in graph.node_ids:
    agent = graph.get_agent_by_id(agent_id)
    model = agent.llm_config.model_name if agent.llm_config else "default"

    metrics = tracker.get_node_metrics(agent_id)

    print(f"\n{agent_id} ({model}):")
    print(f"  Latency: {metrics.avg_latency_ms:.0f}ms")
    print(f"  Tokens: {metrics.total_cost_tokens}")
    print(f"  Reliability: {metrics.reliability:.2%}")
```

---

#### Обратная совместимость

Старый код **продолжает работать** без изменений:

```python
# Старый способ (один LLM для всех агентов)
runner = MACPRunner(llm_caller=my_llm)
result = runner.run_round(graph)
# ✅ Работает как раньше

# Новый способ (мультимодельность)
runner = MACPRunner(llm_factory=factory)
result = runner.run_round(graph)
# ✅ Использует индивидуальные модели для каждого агента
```

---

### Динамическая топология

#### Статическое изменение графа

Изменение структуры графа до выполнения:

```python
# Добавление нового агента
new_agent = AgentProfile(identifier="expert", display_name="Expert")
graph.add_node(new_agent, connections_to=["checker"])

# Изменение связей
graph.add_edge("solver", "expert", weight=0.9)
graph.remove_edge("solver", "checker")

# Деактивация нод (без удаления)
graph.disable("expensive_agent")  # Не выполнится, но останется в графе

# Полное обновление топологии из матрицы
import torch

new_adjacency = torch.tensor([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
], dtype=torch.float32)

graph.update_communication(
    new_adjacency,
    s_tilde=score_matrix,      # Оценки качества связей
    p_matrix=probability_matrix # Вероятности переходов
)
```

#### Runtime модификация (во время выполнения)

Новый мощный функционал для изменения графа **во время выполнения раунда** на основе промежуточных результатов:

##### Ранняя остановка (Early Stopping)

```python
from rustworkx_framework import EarlyStopCondition, RunnerConfig

# 1. По ключевому слову в ответе
stop_on_answer = EarlyStopCondition.on_keyword(
    "FINAL ANSWER",
    reason="Answer found"
)

# 2. По лимиту токенов
stop_on_tokens = EarlyStopCondition.on_token_limit(
    max_tokens=5000,
    reason="Token budget exceeded"
)

# 3. По количеству выполненных агентов
stop_on_count = EarlyStopCondition.on_agent_count(
    max_agents=5,
    reason="Sufficient agents executed"
)

# 4. По значению в metadata (для RL, метрик)
stop_on_quality = EarlyStopCondition.on_metadata(
    "quality_score",
    0.95,
    comparator=lambda v, threshold: v > threshold,
    reason="Quality threshold reached"
)

# 5. Произвольное условие
stop_custom = EarlyStopCondition.on_custom(
    condition=lambda ctx: my_rl_agent.should_stop(ctx.messages),
    reason="RL agent decided to stop",
    min_agents_executed=2  # Минимум 2 агента до проверки
)

# 6. Комбинация условий (OR)
stop_any = EarlyStopCondition.combine_any([
    EarlyStopCondition.on_keyword("DONE"),
    EarlyStopCondition.on_token_limit(10000),
    stop_on_quality,
])

# 7. Комбинация условий (AND)
stop_all = EarlyStopCondition.combine_all([
    EarlyStopCondition.on_keyword("answer"),
    stop_on_quality,
])

# Использование
config = RunnerConfig(
    early_stop_conditions=[stop_on_answer, stop_on_tokens]
)
runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(graph)

if result.early_stopped:
    print(f"Остановлено: {result.early_stop_reason}")
    print(f"Сэкономлено: {len(graph.node_ids) - len(result.execution_order)} агентов")
```

##### Topology Hooks (модификация графа на лету)

```python
from rustworkx_framework import TopologyAction, StepContext, RunnerConfig

def my_topology_hook(ctx: StepContext, graph) -> TopologyAction:
    """Вызывается после каждого шага выполнения.

    StepContext содержит:
        - agent_id: текущий агент
        - response: его ответ
        - messages: все ответы до сих пор
        - execution_order: порядок выполнения
        - remaining_agents: оставшиеся агенты
        - total_tokens: использовано токенов
        - metadata: произвольные данные
    """

    # 1. Ранняя остановка на основе кастомной логики
    if "TASK_COMPLETE" in (ctx.response or ""):
        return TopologyAction(
            early_stop=True,
            early_stop_reason="Task marked as complete"
        )

    # 2. Добавить ребро если качество низкое
    if ctx.metadata.get("quality", 1.0) < 0.5:
        return TopologyAction(
            add_edges=[
                (ctx.agent_id, "reviewer_agent", 1.0),
            ],
            trigger_replan=True  # Перепланировать оставшиеся шаги
        )

    # 3. Удалить ребро
    if some_condition:
        return TopologyAction(
            remove_edges=[
                ("agent1", "agent2"),
            ]
        )

    # 4. Пропустить следующих агентов
    if ctx.total_tokens > 8000:
        return TopologyAction(
            skip_agents=["expensive_agent1", "expensive_agent2"]
        )

    # 5. Принудительно выполнить агентов
    if needs_expert_review:
        return TopologyAction(
            force_agents=["expert_reviewer"]
        )

    # 6. Изменить конечного агента
    if early_finish:
        return TopologyAction(
            new_end_agent="quick_finalizer"
        )

    return None  # Без изменений

# Async hook для интеграции с RL, API и т.д.
async def rl_topology_hook(ctx: StepContext, graph) -> TopologyAction:
    """Async hook для сложной логики."""
    # Можно вызывать async API, RL модели
    decision = await my_rl_agent.get_topology_decision(
        messages=ctx.messages,
        graph_state=graph.to_dict()
    )

    if decision.add_connection:
        return TopologyAction(
            add_edges=[(decision.from_node, decision.to_node, decision.weight)]
        )

    return None

# Использование
config = RunnerConfig(
    enable_dynamic_topology=True,
    topology_hooks=[my_topology_hook],
    async_topology_hooks=[rl_topology_hook],
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(graph)

print(f"Модификаций топологии: {result.topology_modifications}")
```

##### Пример: RL-управляемая топология

```python
import torch
from your_rl_agent import RLAgent

class TopologyRL:
    def __init__(self):
        self.rl_agent = RLAgent()

    def should_stop(self, ctx: StepContext) -> bool:
        """Решение RL-агента о ранней остановке."""
        state = self.encode_state(ctx)
        action = self.rl_agent.predict(state)
        return action == "STOP"

    def get_topology_action(self, ctx: StepContext) -> TopologyAction | None:
        """RL-агент решает как изменить топологию."""
        state = self.encode_state(ctx)
        action = self.rl_agent.predict(state)

        if action == "ADD_REVIEWER":
            return TopologyAction(
                add_edges=[(ctx.agent_id, "reviewer", 1.0)],
                trigger_replan=True
            )
        elif action == "SKIP_EXPENSIVE":
            return TopologyAction(
                skip_agents=["expensive_model"]
            )

        return None

    def encode_state(self, ctx: StepContext) -> torch.Tensor:
        # Закодировать состояние для RL
        return torch.tensor([
            len(ctx.messages),
            ctx.total_tokens,
            len(ctx.remaining_agents),
        ])

# Использование
rl_controller = TopologyRL()

config = RunnerConfig(
    enable_dynamic_topology=True,
    early_stop_conditions=[
        EarlyStopCondition.on_custom(
            rl_controller.should_stop,
            reason="RL decided to stop"
        )
    ],
    topology_hooks=[rl_controller.get_topology_action],
)
```

##### Полный пример: Адаптивная система

```python
from rustworkx_framework import (
    GraphBuilder, MACPRunner, RunnerConfig,
    EarlyStopCondition, TopologyAction, StepContext
)

# Построить граф
builder = GraphBuilder()
builder.add_agent("input", persona="Input processor")
builder.add_agent("solver", persona="Problem solver")
builder.add_agent("checker", persona="Solution checker")
builder.add_agent("expensive_expert", persona="Expert (expensive)")
builder.add_agent("output", persona="Output formatter")

builder.add_workflow_edge("input", "solver")
builder.add_workflow_edge("solver", "checker")
builder.add_workflow_edge("checker", "output")
# expensive_expert подключается динамически

builder.set_start_node("input")
builder.set_end_node("output")
builder.add_task(query="Solve the complex problem")
builder.connect_task_to_agents()

graph = builder.build()

# Hooks для адаптации
def adaptive_hook(ctx: StepContext, graph) -> TopologyAction:
    # Если checker нашёл проблему — добавить expert
    if ctx.agent_id == "checker" and "ERROR" in (ctx.response or ""):
        return TopologyAction(
            add_edges=[("checker", "expensive_expert", 1.0),
                      ("expensive_expert", "output", 1.0)],
            trigger_replan=True
        )

    # Если solver дал хороший ответ — пропустить checker
    if ctx.agent_id == "solver" and ctx.metadata.get("confidence", 0) > 0.95:
        return TopologyAction(
            skip_agents=["checker"],
            reason="High confidence, skipping validation"
        )

    return None

# Настроить runner
config = RunnerConfig(
    adaptive=True,
    enable_dynamic_topology=True,
    topology_hooks=[adaptive_hook],
    early_stop_conditions=[
        EarlyStopCondition.on_keyword("FINAL_ANSWER"),
        EarlyStopCondition.on_token_limit(10000),
    ],
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(
    graph,
    filter_unreachable=True  # Исключить изолированные ноды
)

# Результат
print(f"Executed: {result.execution_order}")
print(f"Early stopped: {result.early_stopped}")
print(f"Topology mods: {result.topology_modifications}")
print(f"Tokens saved: calculated from pruned_agents")
```

---

### GNN-маршрутизация (Graph Neural Networks для Routing)

Использование графовых нейросетей для **обучаемой** оптимальной маршрутизации на основе истории выполнения.

#### Обзор GNN моделей

| Модель | Описание | Когда использовать |
|--------|----------|-------------------|
| **GCN** (Graph Convolutional Network) | Классическая свёрточная сеть для графов | Гомогенные графы, простые задачи |
| **GAT** (Graph Attention Network) | Использует механизм внимания | Важность связей различна |
| **GraphSAGE** | Сэмплирование соседей для больших графов | Большие графы, индуктивное обучение |
| **GIN** (Graph Isomorphism Network) | Максимально выразительная архитектура | Сложные паттерны, малые графы |

---

#### Полный пример: Обучение GNN для маршрутизации

```python
from rustworkx_framework.core.gnn import (
    create_gnn_router,
    GNNTrainer,
    GNNRouterInference,
    GNNModelType,
    TrainingConfig,
    FeatureConfig,
    RoutingStrategy,
    DefaultFeatureGenerator,
)
from rustworkx_framework.core.metrics import MetricsTracker
import torch
from torch_geometric.data import Data

# ========== ШАГ 1: Сбор данных выполнения ==========
tracker = MetricsTracker()

# Выполнить несколько раундов для накопления метрик
for i in range(100):
    result = runner.run_round(graph)

    # Записать метрики каждого агента
    for agent_id in result.execution_order:
        response = result.messages[agent_id]
        tracker.record_node_execution(
            node_id=agent_id,
            success=True,
            latency_ms=response["latency"],
            cost_tokens=response["tokens"],
            quality=evaluate_quality(response["content"]),
        )

    # Записать метрики рёбер
    for i, agent_id in enumerate(result.execution_order[:-1]):
        next_agent = result.execution_order[i + 1]
        tracker.record_edge_traversal(
            source=agent_id,
            target=next_agent,
            weight=graph.get_edge_weight(agent_id, next_agent),
            success=True,
            latency_ms=50,
        )

# ========== ШАГ 2: Генерация признаков ==========
feature_config = FeatureConfig(
    include_degree=True,           # Степени вершин
    include_centrality=True,       # Центральность (betweenness, closeness)
    include_embeddings=True,       # Эмбеддинги агентов
    include_metrics=True,          # Метрики производительности
    include_structural=True,       # Структурные признаки (clustering coef)
    normalize=True,                # Нормализация признаков
)

feature_gen = DefaultFeatureGenerator(config=feature_config)

node_features = feature_gen.generate_node_features(
    graph,
    graph.node_ids,
    tracker,
)  # Shape: (num_nodes, feature_dim)

edge_features = feature_gen.generate_edge_features(
    graph,
    tracker,
)  # Shape: (num_edges, edge_feature_dim)

print(f"Node features shape: {node_features.shape}")
print(f"Edge features shape: {edge_features.shape}")

# ========== ШАГ 3: Подготовка датасета ==========
# Создание PyTorch Geometric Data объектов

train_data_list = []
val_data_list = []

for sample in dataset:  # Ваш датасет с историей выполнения
    data = Data(
        x=sample['node_features'],          # Node features
        edge_index=sample['edge_index'],    # Edge connections (2, E)
        edge_attr=sample['edge_features'],  # Edge features
        y=sample['labels'],                 # Labels (optimal next node, quality score, etc.)
    )

    if sample['is_train']:
        train_data_list.append(data)
    else:
        val_data_list.append(data)

# ========== ШАГ 4: Конфигурация обучения ==========
training_config = TrainingConfig(
    # Гиперпараметры
    learning_rate=1e-3,
    hidden_dim=64,
    num_layers=3,
    dropout=0.2,

    # Обучение
    epochs=100,
    batch_size=32,
    patience=10,              # Early stopping

    # Задача
    task="node_classification",  # или "link_prediction", "graph_regression"
    num_classes=2,               # Для классификации

    # Оптимизация
    optimizer="adam",            # adam, sgd, adamw
    weight_decay=1e-5,
    scheduler="reduce_on_plateau",  # step, cosine, reduce_on_plateau

    # Устройство
    device="cuda" if torch.cuda.is_available() else "cpu",

    # Логирование
    log_interval=10,
    save_best=True,
)

# ========== ШАГ 5: Создание модели ==========

# 5.1. GCN (Graph Convolutional Network)
model_gcn = create_gnn_router(
    model_type=GNNModelType.GCN,
    in_channels=node_features.shape[1],
    out_channels=training_config.num_classes,
    config=training_config,
)

# 5.2. GAT (Graph Attention Network)
model_gat = create_gnn_router(
    model_type=GNNModelType.GAT,
    in_channels=node_features.shape[1],
    out_channels=training_config.num_classes,
    config=training_config,
    heads=4,              # Количество attention heads
    concat=True,          # Конкатенировать heads или усреднять
)

# 5.3. GraphSAGE
model_sage = create_gnn_router(
    model_type=GNNModelType.GraphSAGE,
    in_channels=node_features.shape[1],
    out_channels=training_config.num_classes,
    config=training_config,
    aggr="mean",          # mean, max, lstm
)

# 5.4. GIN (Graph Isomorphism Network)
model_gin = create_gnn_router(
    model_type=GNNModelType.GIN,
    in_channels=node_features.shape[1],
    out_channels=training_config.num_classes,
    config=training_config,
    train_eps=True,       # Обучаемый параметр epsilon
)

# ========== ШАГ 6: Обучение ==========
trainer = GNNTrainer(model_gat, training_config)

training_result = trainer.train(
    train_data_list,
    val_data_list,
    verbose=True,
)

print(f"Best validation accuracy: {training_result['best_val_acc']:.3f}")
print(f"Best epoch: {training_result['best_epoch']}")
print(f"Training time: {training_result['training_time']:.2f}s")

# Сохранение модели
trainer.save("gnn_router.pt")

# Загрузка модели
trainer.load("gnn_router.pt")

# ========== ШАГ 7: Инференс ==========
router = GNNRouterInference(
    model=model_gat,
    feature_generator=feature_gen,
)

# 7.1. Предсказание следующего узла (node selection)
prediction = router.predict(
    graph,
    source="coordinator",
    candidates=["researcher", "analyst", "writer"],
    metrics_tracker=tracker,
    strategy=RoutingStrategy.ARGMAX,  # ARGMAX, TOP_K, SAMPLING, THRESHOLD
)

print(f"Recommended nodes: {prediction.recommended_nodes}")
print(f"Scores: {prediction.scores}")
print(f"Confidence: {prediction.confidence:.3f}")

# 7.2. Предсказание с топ-K
prediction_topk = router.predict(
    graph,
    source="coordinator",
    candidates=["a", "b", "c", "d"],
    strategy=RoutingStrategy.TOP_K,
    k=2,  # Вернуть 2 лучших
)

print(f"Top 2: {prediction_topk.recommended_nodes}")

# 7.3. Сэмплирование с вероятностями
prediction_sample = router.predict(
    graph,
    source="coordinator",
    candidates=candidates,
    strategy=RoutingStrategy.SAMPLING,
    temperature=0.8,  # Температура для сэмплирования
)

# 7.4. Пороговая фильтрация
prediction_threshold = router.predict(
    graph,
    source="coordinator",
    candidates=candidates,
    strategy=RoutingStrategy.THRESHOLD,
    threshold=0.7,  # Только узлы с вероятностью > 0.7
)

# ========== ШАГ 8: Интеграция с AdaptiveScheduler ==========
from rustworkx_framework.execution import AdaptiveScheduler, RoutingPolicy

scheduler = AdaptiveScheduler(
    policy=RoutingPolicy.GNN_BASED,
    gnn_router=router,
    gnn_threshold=0.6,          # Мин. confidence для использования GNN
    fallback_policy=RoutingPolicy.WEIGHTED_TOPO,  # Fallback при низком confidence
)

plan = scheduler.build_plan(
    graph.A_com,
    graph.node_ids,
    metrics_tracker=tracker,
)

# ========== ШАГ 9: Мониторинг и дообучение ==========
# Собрать новые данные после деплоя
new_data = []
for i in range(20):
    result = runner.run_round(graph)
    # ... запись данных ...
    new_data.append(create_data_sample(result))

# Дообучить модель (fine-tuning)
trainer.fine_tune(
    new_data,
    epochs=10,
    learning_rate=1e-4,
)

trainer.save("gnn_router_finetuned.pt")

# ========== Оценка качества ==========
from rustworkx_framework.core.gnn import evaluate_router

metrics = evaluate_router(
    router,
    test_data_list,
    metrics=["accuracy", "f1", "precision", "recall"],
)

print(f"Test accuracy: {metrics['accuracy']:.3f}")
print(f"F1 score: {metrics['f1']:.3f}")
```

---

#### Сравнение GNN моделей

```python
# Эксперимент: сравнить производительность разных моделей

models = {
    "GCN": create_gnn_router(GNNModelType.GCN, in_channels, out_channels, config),
    "GAT": create_gnn_router(GNNModelType.GAT, in_channels, out_channels, config),
    "GraphSAGE": create_gnn_router(GNNModelType.GraphSAGE, in_channels, out_channels, config),
    "GIN": create_gnn_router(GNNModelType.GIN, in_channels, out_channels, config),
}

results = {}

for name, model in models.items():
    trainer = GNNTrainer(model, training_config)
    result = trainer.train(train_data, val_data)
    results[name] = result

# Сравнение
import pandas as pd

df = pd.DataFrame([
    {
        "Model": name,
        "Val Acc": res["best_val_acc"],
        "Train Time": res["training_time"],
        "Params": sum(p.numel() for p in models[name].parameters()),
    }
    for name, res in results.items()
])

print(df)

# Вывод:
# | Model     | Val Acc | Train Time | Params  |
# |-----------|---------|------------|---------|
# | GCN       | 0.853   | 12.5s      | 45123   |
# | GAT       | 0.891   | 18.3s      | 67891   |
# | GraphSAGE | 0.874   | 15.2s      | 52341   |
# | GIN       | 0.867   | 14.8s      | 48976   |
```

---

#### Использование в продакшене

```python
# Загрузить обученную модель
router = GNNRouterInference.load("gnn_router.pt", feature_gen)

# Интегрировать с runner
config = RunnerConfig(
    adaptive=True,
    routing_policy=RoutingPolicy.GNN_BASED,
)

runner = MACPRunner(
    llm_caller=my_llm,
    config=config,
    gnn_router=router,
    metrics_tracker=tracker,
)

# Выполнение с GNN-маршрутизацией
result = runner.run_round(graph)

# Мониторинг предсказаний GNN
print(f"GNN predictions used: {result.gnn_prediction_count}")
print(f"Fallback to heuristic: {result.fallback_to_heuristic_count}")
```

---

### Скрытые каналы (Hidden Channels)

Скрытые каналы позволяют передавать **неявную информацию** между агентами в виде векторных представлений, минуя текстовые промпты. Это особенно полезно для:
- Передачи контекстуальной информации без увеличения длины промпта
- Сохранения семантических эмбеддингов для downstream задач
- Реализации attention-механизмов между агентами
- Интеграции с GNN для предсказания следующих шагов

#### Архитектура скрытых каналов

```
┌─────────────┐     hidden_state     ┌─────────────┐
│   Agent A   │ ──────────────────>  │   Agent B   │
│ (embedding) │     embedding        │ (receives   │
└─────────────┘                      │  combined)  │
                                     └─────────────┘
```

Каждый агент владеет своими:
- **`embedding`** — векторное представление описания агента
- **`hidden_state`** — скрытое состояние, обновляемое после выполнения

Runner комбинирует `hidden_state` и `embedding` от предшественников и передаёт агенту.

#### Использование скрытых каналов

```python
from rustworkx_framework.execution import RunnerConfig, MACPRunner, HiddenState
from rustworkx_framework.core import NodeEncoder

# 1. Создание энкодера для генерации эмбеддингов
encoder = NodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Конфигурация со скрытыми каналами
config = RunnerConfig(
    enable_hidden_channels=True,
    hidden_combine_strategy="mean",  # Стратегия объединения
    pass_embeddings=True,            # Передавать embeddings тоже
    hidden_dim=384,                  # Размерность скрытых состояний
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# 3. Вычислить эмбеддинги агентов
texts = [agent.to_text() for agent in graph.agents]
embeddings = encoder.encode(texts)

for agent, emb in zip(graph.agents, embeddings):
    agent = agent.with_embedding(emb)
    graph.update_agent(agent.identifier, agent)

# 4. Выполнение со скрытыми каналами
result = runner.run_round_with_hidden(
    graph,
    hidden_encoder=encoder,  # Для создания hidden_state из ответов
)

# 5. Доступ к скрытым состояниям после выполнения
for agent_id, hidden in result.hidden_states.items():
    print(f"{agent_id}:")
    print(f"  Hidden state: {hidden.tensor.shape}")      # (hidden_dim,)
    print(f"  Embedding: {hidden.embedding.shape}")      # (embedding_dim,)
    print(f"  Combined: {hidden.combined.shape}")        # (hidden_dim + embedding_dim,)

# 6. Использование hidden states для downstream задач
hidden_states_matrix = torch.stack([
    result.hidden_states[aid].tensor for aid in graph.node_ids
])  # Shape: (num_agents, hidden_dim)

# Например, для кластеризации агентов по семантике
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(hidden_states_matrix.cpu().numpy())
```

#### Стратегии комбинирования (combine_strategy)

Когда у агента несколько предшественников, их скрытые состояния комбинируются:

```python
# 1. "mean" — среднее (по умолчанию)
# hidden_combined = mean([h1, h2, h3])
config.hidden_combine_strategy = "mean"

# 2. "sum" — сумма
# hidden_combined = h1 + h2 + h3
config.hidden_combine_strategy = "sum"

# 3. "concat" — конкатенация
# hidden_combined = concat([h1, h2, h3])  # размерность увеличивается
config.hidden_combine_strategy = "concat"

# 4. "attention" — взвешенное внимание (веса из матрицы смежности)
# hidden_combined = w1*h1 + w2*h2 + w3*h3, где wi = edge_weight(i -> current)
config.hidden_combine_strategy = "attention"

# 5. "max" — поэлементный максимум
# hidden_combined = max(h1, h2, h3)
config.hidden_combine_strategy = "max"
```

#### Продвинутое использование: Кастомная обработка hidden states

```python
from rustworkx_framework.utils.memory import HiddenChannel

# Создание кастомного HiddenChannel
channel = HiddenChannel(
    node_id="agent_id",
    hidden_dim=384,
)

# Установка hidden state
import torch
channel.set_hidden(torch.randn(384))
channel.set_embedding(torch.randn(384))

# Получение combined representation
combined = channel.get_combined(strategy="attention", edge_weights=torch.tensor([0.8, 0.2]))

# Сброс
channel.reset()

# Интеграция с памятью агента
from rustworkx_framework.utils.memory import AgentMemory

memory = AgentMemory("agent_id")
memory.hidden_state = torch.randn(384)
memory.embedding = torch.randn(384)

# Получить для передачи следующему агенту
hidden_to_pass = memory.hidden_state
embedding_to_pass = memory.embedding
```

#### Использование с GNN

```python
from rustworkx_framework.core.gnn import GNNRouterInference, DefaultFeatureGenerator

# 1. Скрытые состояния как признаки для GNN
feature_gen = DefaultFeatureGenerator()

# Использовать hidden states как часть признаков
node_features = feature_gen.generate_node_features(
    graph,
    graph.node_ids,
    metrics_tracker,
    include_hidden_states=True,  # Добавить hidden_state в признаки
)

# 2. GNN предсказывает следующего агента на основе hidden states
router = GNNRouterInference(model, feature_gen)

prediction = router.predict(
    graph,
    source="current_agent",
    candidates=["next1", "next2"],
    metrics_tracker=tracker,
    hidden_states=result.hidden_states,  # Передать текущие hidden states
)

# 3. Обновить граф на основе предсказаний GNN
if prediction.confidence > 0.8:
    next_agent = prediction.recommended_nodes[0]
    graph.add_edge("current_agent", next_agent, weight=prediction.confidence)
```

#### Пример: Multi-hop reasoning с hidden channels

```python
# Задача: multi-hop reasoning, где каждый агент накапливает контекст

agents = [
    AgentProfile(identifier="reader", display_name="Document Reader"),
    AgentProfile(identifier="analyzer", display_name="Analyzer"),
    AgentProfile(identifier="reasoner", display_name="Reasoner"),
    AgentProfile(identifier="answerer", display_name="Final Answerer"),
]

edges = [
    ("reader", "analyzer"),
    ("analyzer", "reasoner"),
    ("reasoner", "answerer"),
]

graph = build_property_graph(agents, edges, query="Complex question")

# Включаем hidden channels для передачи контекста
config = RunnerConfig(
    enable_hidden_channels=True,
    hidden_combine_strategy="attention",
    pass_embeddings=True,
)

encoder = NodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
runner = MACPRunner(llm_caller=my_llm, config=config)

result = runner.run_round_with_hidden(graph, hidden_encoder=encoder)

# После каждого шага hidden_state содержит "накопленный контекст"
# answerer получает взвешенную комбинацию всех предыдущих hidden states
```

---

### Адаптивное выполнение

Полный контроль над адаптивным выполнением:

```python
from rustworkx_framework.execution import (
    MACPRunner,
    RunnerConfig,
    RoutingPolicy,
    PruningConfig,
    BudgetConfig,
    ErrorPolicy,
)

config = RunnerConfig(
    adaptive=True,
    enable_replanning=True,
    replan_on_error_only=False,
    enable_parallel=True,
    max_parallel_size=5,

    routing_policy=RoutingPolicy.BEAM_SEARCH,

    pruning_config=PruningConfig(
        min_weight_threshold=0.1,
        token_budget=10000,
        enable_fallback=True,
        max_fallback_attempts=2,
        quality_scorer=lambda response: evaluate_quality(response),
        min_quality_threshold=0.5,
    ),

    budget_config=BudgetConfig(
        total_token_limit=50000,
        max_prompt_length=4000,
        per_node_token_limit=2000,
    ),

    error_policy=ErrorPolicy(
        on_timeout="skip",     # skip, retry, fail
        on_error="fallback",   # skip, retry, fallback, fail
        max_retries=3,
    ),
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(graph)

print(f"Перепланирований: {result.replanning_count}")
print(f"Fallback-ов: {result.fallback_count}")
print(f"Отсечённых агентов: {result.pruned_agents}")
```

---

## Конфигурация

### Переменные окружения

```bash
# API ключ (обязательно)
export RWXF_API_KEY="sk-your-api-key"
# или через файл
export RWXF_API_KEY_FILE=/secure/rwxf.key

# URL LLM сервиса
export RWXF_BASE_URL="https://api.openai.com/v1"

# Модели
export RWXF_MODEL_NAME="gpt-4o-mini"
export RWXF_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Логирование
export RWXF_LOG_LEVEL="INFO"
export RWXF_LOG_FILE="./logs/framework.log"

# Сетевые настройки
export RWXF_DEFAULT_TIMEOUT=60
export RWXF_MAX_RETRIES=3
```

### Программная конфигурация

```python
from rustworkx_framework.config import FrameworkSettings, load_settings

# Загрузка из окружения
settings = FrameworkSettings()

# Загрузка из .env файла
settings = load_settings(".env")

# Доступ к настройкам
api_key = settings.resolved_api_key
model = settings.model_name
timeout = settings.default_timeout
```

---

## Примеры использования

### Пример 1: Простой Pipeline

```python
from rustworkx_framework import AgentProfile, MACPRunner
from rustworkx_framework.builder import build_property_graph

agents = [
    AgentProfile(identifier="researcher", display_name="Researcher"),
    AgentProfile(identifier="writer", display_name="Writer"),
    AgentProfile(identifier="editor", display_name="Editor"),
]

graph = build_property_graph(
    agents,
    workflow_edges=[("researcher", "writer"), ("writer", "editor")],
    query="Напиши статью о квантовых компьютерах",
)

runner = MACPRunner(llm_caller=my_llm)
result = runner.run_round(graph)

print(result.final_answer)
```

### Пример 2: Параллельная обработка

```python
# Агенты работают параллельно, затем результаты агрегируются
agents = [
    AgentProfile(identifier="analyst_1", display_name="Financial Analyst"),
    AgentProfile(identifier="analyst_2", display_name="Market Analyst"),
    AgentProfile(identifier="analyst_3", display_name="Risk Analyst"),
    AgentProfile(identifier="aggregator", display_name="Report Aggregator"),
]

edges = [
    ("analyst_1", "aggregator"),
    ("analyst_2", "aggregator"),
    ("analyst_3", "aggregator"),
]

graph = build_property_graph(agents, workflow_edges=edges, query="Analyze company X")

config = RunnerConfig(
    enable_parallel=True,
    max_parallel_size=3,
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = await runner.arun_round(graph)
```

### Пример 3: Streaming с callback

```python
def on_event(event):
    if event.event_type == StreamEventType.AGENT_OUTPUT:
        save_to_db(event.agent_id, event.content)
        notify_frontend(event)

runner = MACPRunner(llm_caller=my_llm)

for event in runner.stream(graph):
    on_event(event)

    if event.event_type == StreamEventType.TOKEN:
        yield event.token  # Для SSE или WebSocket
```

### Пример 4: Работа с памятью

```python
from rustworkx_framework.execution import MACPRunner, RunnerConfig, MemoryConfig

config = RunnerConfig(
    enable_memory=True,
    memory_config=MemoryConfig(
        working_max_entries=20,
        long_term_max_entries=100,
    ),
    memory_context_limit=5,  # Включать 5 последних записей в промпт
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# Первый раунд
result1 = runner.run_round(graph)

# Второй раунд — агенты помнят контекст
graph.query = "Продолжи предыдущую задачу"
result2 = runner.run_round(graph)

# Доступ к памяти агента
agent_memory = runner.get_agent_memory("solver")
entries = agent_memory.get_messages()
```

### Пример 5: Визуализация графа

```python
from rustworkx_framework.core import AgentProfile
from rustworkx_framework.core.visualization import (
    GraphVisualizer,
    VisualizationStyle,
    MermaidDirection,
    NodeStyle,
    NodeShape,
    # Convenience functions
    to_mermaid,
    to_ascii,
    to_dot,
    print_graph,
    render_to_image,
)
from rustworkx_framework.builder import build_property_graph

# Создаём граф
agents = [
    AgentProfile(
        identifier="input",
        display_name="Input Handler",
        tools=["api_reader"],
    ),
    AgentProfile(
        identifier="processor",
        display_name="Data Processor",
        tools=["pandas", "torch"],
    ),
    AgentProfile(
        identifier="output",
        display_name="Output Formatter",
        tools=["json", "csv"],
    ),
]

graph = build_property_graph(
    agents,
    workflow_edges=[("input", "processor"), ("processor", "output")],
    query="Process data pipeline",
    include_task_node=True,
)

# Способ 1: Быстрая визуализация (convenience functions)
print("=== MERMAID ===")
mermaid = to_mermaid(graph, direction=MermaidDirection.LEFT_RIGHT)
print(mermaid)

print("\n=== ASCII ===")
ascii_art = to_ascii(graph, show_edges=True)
print(ascii_art)

print("\n=== COLORED (если установлен Rich) ===")
print_graph(graph, format="auto")  # Автоматически выберет colored или ascii

# Способ 2: Продвинутая визуализация с кастомными стилями (Pydantic модели)
# Создаём стиль (Pydantic модель с валидацией)
custom_style = VisualizationStyle(
    direction=MermaidDirection.LEFT_RIGHT,
    agent_style=NodeStyle(
        shape=NodeShape.ROUND,
        fill_color="#e3f2fd",
        stroke_color="#1976d2",
        icon="🤖",
    ),
    task_style=NodeStyle(
        shape=NodeShape.DIAMOND,
        fill_color="#fff3e0",
        stroke_color="#f57c00",
        icon="📋",
    ),
    show_weights=True,
    show_tools=True,
    max_label_length=30,
)

# Создаём визуализатор с кастомным стилем
viz = GraphVisualizer(graph, custom_style)

# Mermaid с заголовком
mermaid_styled = viz.to_mermaid(title="Data Pipeline")
print("\n=== STYLED MERMAID ===")
print(mermaid_styled)

# Сохранение в файлы
viz.save_mermaid("pipeline.md", title="Data Pipeline")  # Markdown с ```mermaid```
viz.save_dot("pipeline.dot", graph_name="DataPipeline")

# Рендеринг в изображения (требует системный Graphviz)
try:
    render_to_image(graph, "pipeline.png", format="png", dpi=150, style=custom_style)
    render_to_image(graph, "pipeline.svg", format="svg", style=custom_style)
    print("\n✅ Изображения созданы: pipeline.png, pipeline.svg")
except Exception as e:
    print(f"\n⚠️  Рендеринг изображений не удался: {e}")
    print("   Установите системный Graphviz для рендеринга изображений")

# Матрица смежности (текстовое представление)
print("\n=== ADJACENCY MATRIX ===")
matrix = viz.to_adjacency_matrix(show_labels=True)
print(matrix)

# Rich Console вывод с деревом и таблицами
print("\n=== RICH CONSOLE ===")
viz.print_colored()
```

### Пример 6: Условная маршрутизация

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution.scheduler import ConditionContext

# Определяем условия
def is_high_quality(context: ConditionContext) -> bool:
    return context.state.get("quality", 0) > 0.8

def needs_review(context: ConditionContext) -> bool:
    return context.state.get("word_count", 0) > 1000

# Строим граф с условными рёбрами
builder = GraphBuilder()
builder.add_agent(identifier="writer", display_name="Content Writer")
builder.add_agent(identifier="editor", display_name="Quick Editor")
builder.add_agent(identifier="reviewer", display_name="Senior Reviewer")
builder.add_agent(identifier="publisher", display_name="Publisher")

# Условные переходы
builder.add_conditional_edge("writer", "editor", condition=is_high_quality)
builder.add_conditional_edge("writer", "reviewer", condition=needs_review)
builder.add_workflow_edge("editor", "publisher")
builder.add_workflow_edge("reviewer", "publisher")

graph = builder.build()

# Запуск
runner = MACPRunner(llm_caller=my_llm)
result = runner.run_round(graph)
```

### Пример 7: Мониторинг с событиями

```python
from rustworkx_framework.core.events import (
    global_event_bus,
    EventType,
    MetricsEventHandler,
)

# Настройка обработчиков событий
bus = global_event_bus()
metrics_handler = MetricsEventHandler()

# Подписка на события
bus.subscribe(None, metrics_handler)  # Слушать все события

@bus.subscribe(EventType.STEP_COMPLETED)
def on_step_completed(event):
    print(f"✅ {event.agent_id} завершён за {event.duration_ms:.0f}ms")

@bus.subscribe(EventType.BUDGET_WARNING)
def on_budget_warning(event):
    print(f"⚠️  Бюджет {event.budget_type}: {event.ratio:.1%}")

# Запуск с мониторингом
runner = MACPRunner(llm_caller=my_llm)
result = runner.run_round(graph)

# Получение агрегированных метрик
metrics = metrics_handler.get_metrics()
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Errors: {metrics['errors_count']}")
print(f"Avg step duration: {metrics['avg_step_duration_ms']:.1f}ms")
```

### Пример 8: GNN-маршрутизация с обучением

```python
from rustworkx_framework.core.gnn import (
    create_gnn_router,
    GNNTrainer,
    GNNRouterInference,
    GNNModelType,
    TrainingConfig,
    DefaultFeatureGenerator,
)
from rustworkx_framework.core.metrics import MetricsTracker
import torch

# Сбор данных выполнения для обучения
tracker = MetricsTracker()

# ... выполнить несколько раундов с разными запросами ...
for i in range(100):
    result = runner.run_round(graph)
    # Записать метрики
    for agent_id, response in result.messages.items():
        tracker.record_node_execution(
            node_id=agent_id,
            success=True,
            latency_ms=response["latency"],
            cost_tokens=response["tokens"],
            quality=evaluate_quality(response["content"]),
        )

# Генерация признаков
feature_gen = DefaultFeatureGenerator()
node_features = feature_gen.generate_node_features(
    graph,
    graph.node_ids,
    tracker,
)

# Создание датасета
# ... подготовка train_data, val_data в формате PyG Data ...

# Обучение модели
config = TrainingConfig(
    learning_rate=1e-3,
    hidden_dim=64,
    num_layers=2,
    epochs=50,
    task="node_classification",
)

model = create_gnn_router(
    model_type=GNNModelType.GAT,
    in_channels=node_features.shape[1],
    out_channels=2,
    config=config,
)

trainer = GNNTrainer(model, config)
result = trainer.train(train_data, val_data)

print(f"Best validation accuracy: {result['best_val_acc']:.3f}")
trainer.save("gnn_router.pt")

# Использование обученной модели для маршрутизации
router = GNNRouterInference(model, feature_gen)

prediction = router.predict(
    graph,
    source="coordinator",
    candidates=["agent1", "agent2", "agent3"],
    metrics_tracker=tracker,
)

print(f"Recommended: {prediction.recommended_nodes[0]}")
print(f"Confidence: {prediction.confidence:.3f}")
```

### Пример 9: Адаптивное выполнение с бюджетом

```python
from rustworkx_framework.execution import (
    MACPRunner,
    RunnerConfig,
    RoutingPolicy,
    PruningConfig,
)
from rustworkx_framework.execution.budget import Budget

# Настройка адаптивного выполнения
config = RunnerConfig(
    adaptive=True,
    enable_replanning=True,
    enable_parallel=True,
    max_parallel_size=3,

    routing_policy=RoutingPolicy.WEIGHTED_TOPO,

    pruning_config=PruningConfig(
        min_weight_threshold=0.1,
        token_budget=5000,
        enable_fallback=True,
        max_fallback_attempts=2,
    ),

    budget_config={
        "total_token_limit": 10000,
        "per_node_token_limit": 2000,
        "max_prompt_length": 3000,
        "warning_threshold": 0.8,
    },

    timeout=60.0,
    max_retries=2,
)

runner = MACPRunner(llm_caller=my_llm, config=config)

# Выполнение
try:
    result = runner.run_round(graph)

    print(f"Executed agents: {len(result.execution_order)}")
    print(f"Pruned agents: {result.pruned_agents}")
    print(f"Replanning count: {result.replanning_count}")
    print(f"Fallback count: {result.fallback_count}")
    print(f"Total tokens: {result.total_tokens}")

except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
except ExecutionError as e:
    print(f"Execution failed: {e}")
```

### Пример 10: Анализ графа с алгоритмами

```python
from rustworkx_framework.core.algorithms import (
    GraphAlgorithms,
    CentralityType,
    PathMetric,
)

# Создаём сложный граф
algo = GraphAlgorithms(graph)

# Поиск критических узлов
centrality = algo.centrality(CentralityType.BETWEENNESS, normalized=True)
print(f"Most critical agents: {centrality.top_nodes[:3]}")

# Поиск альтернативных путей
paths = algo.k_shortest_paths(
    source="input",
    target="output",
    k=3,
    metric=PathMetric.WEIGHTED,
)

print(f"Found {len(paths)} alternative paths:")
for i, path in enumerate(paths, 1):
    print(f"  Path {i}: {' -> '.join(path.nodes)} (cost: {path.cost:.2f})")

# Обнаружение сообществ
communities = algo.detect_communities(algorithm="louvain")
print(f"Communities found: {len(communities.communities)}")
for i, community in enumerate(communities.communities):
    print(f"  Community {i}: {community}")

# Проверка на циклы
cycles = algo.find_cycles(max_length=5)
if cycles.has_cycles:
    print(f"⚠️  Graph has {len(cycles.cycles)} cycles!")
else:
    print("✓ Graph is acyclic (DAG)")
```

### Пример 11: Мультимодельная система с оптимизацией затрат

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner, LLMCallerFactory

# Создание графа с разными моделями для разных задач
builder = GraphBuilder()

# Этап 1: Сбор данных (5 параллельных агентов, дешёвая модель)
for i in range(5):
    builder.add_agent(
        f"collector_{i}",
        display_name=f"Data Collector {i}",
        persona="Collects and formats raw data",
        llm_backbone="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="$OPENAI_API_KEY",
        temperature=0.2,
        max_tokens=500,
    )
    builder.add_workflow_edge(f"collector_{i}", "analyst")

# Этап 2: Глубокий анализ (1 агент, сильная модель)
builder.add_agent(
    "analyst",
    display_name="Senior Data Analyst",
    persona="Expert analyst with deep statistical knowledge",
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=4000,
)
builder.add_workflow_edge("analyst", "privacy_checker")

# Этап 3: Проверка конфиденциальности (локальная модель)
builder.add_agent(
    "privacy_checker",
    display_name="Privacy Compliance Checker",
    persona="Ensures data privacy and compliance",
    llm_backbone="llama3:70b",
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
    temperature=0.0,
    max_tokens=1000,
)
builder.add_workflow_edge("privacy_checker", "reporter")

# Этап 4: Генерация отчёта (дешёвая модель)
builder.add_agent(
    "reporter",
    display_name="Report Generator",
    persona="Formats analysis into readable reports",
    llm_backbone="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.5,
    max_tokens=2000,
)

builder.set_task(
    query="Analyze Q4 sales data and generate compliance report",
    description="Full pipeline from data collection to final report",
)

graph = builder.build()

# Вывод конфигурации
print("=== Multi-Model Pipeline Configuration ===\n")
for agent in graph.agents:
    if hasattr(agent, 'llm_config') and agent.llm_config:
        config = agent.llm_config
        print(f"{agent.display_name}:")
        print(f"  Model: {config.model_name}")
        print(f"  Endpoint: {config.base_url}")
        print(f"  Temp: {config.temperature}, Max tokens: {config.max_tokens}")
        print()

# Создание фабрики и runner
factory = LLMCallerFactory.create_openai_factory()

config = RunnerConfig(
    enable_parallel=True,
    max_parallel_size=5,  # Collectors работают параллельно
    timeout=120.0,
    callbacks=[StdoutCallbackHandler()],  # Мониторинг выполнения
)

runner = MACPRunner(
    llm_factory=factory,
    config=config,
)

# Выполнение
print("=== Executing Multi-Model Pipeline ===\n")
result = runner.run_round(graph)

print(f"\n=== Results ===")
print(f"Execution order: {' → '.join(result.execution_order)}")
print(f"Total time: {result.total_time:.2f}s")
print(f"Total tokens: {result.total_tokens}")
print(f"\nFinal report:\n{result.final_answer}")

# Анализ затрат по моделям
from collections import defaultdict

costs_by_model = defaultdict(int)
for agent_id in result.execution_order:
    agent = graph.get_agent_by_id(agent_id)
    model = agent.llm_config.model_name if agent.llm_config else "default"
    tokens = result.messages.get(agent_id, {}).get("tokens", 0)
    costs_by_model[model] += tokens

print(f"\n=== Token Usage by Model ===")
for model, tokens in costs_by_model.items():
    print(f"{model}: {tokens} tokens")

# Расчёт экономии
# gpt-4: $30/$60 per 1M tokens (input/output)
# gpt-4o-mini: $0.15/$0.60 per 1M tokens
# llama3 (local): $0

gpt4_tokens = costs_by_model.get("gpt-4", 0)
mini_tokens = costs_by_model.get("gpt-4o-mini", 0)
llama_tokens = costs_by_model.get("llama3:70b", 0)

actual_cost = (gpt4_tokens * 45 / 1_000_000) + (mini_tokens * 0.375 / 1_000_000)
if_all_gpt4_cost = (gpt4_tokens + mini_tokens + llama_tokens) * 45 / 1_000_000

print(f"\n=== Cost Analysis ===")
print(f"Actual cost: ${actual_cost:.4f}")
print(f"Cost if all GPT-4: ${if_all_gpt4_cost:.4f}")
print(f"Savings: ${if_all_gpt4_cost - actual_cost:.4f} ({((1 - actual_cost/if_all_gpt4_cost)*100):.1f}%)")
```

---

### Бюджет токенов (Budget System)

Управление ресурсами выполнения (токены, запросы, время).

```python
from rustworkx_framework.execution.budget import (
    Budget,
    NodeBudget,
    BudgetTracker,
)

# Определение бюджетов
global_budget = Budget(
    total_tokens=50000,       # Общий лимит токенов
    total_requests=100,        # Общий лимит запросов
    time_seconds=600,          # Общий лимит времени (10 мин)
    max_prompt_tokens=4000,    # Макс. токенов в промпте
    max_response_tokens=2000,  # Макс. токенов в ответе
)

# Бюджет на узел
node_budget = NodeBudget(
    node_id="solver",
    tokens=2000,
    requests=10,
    time_seconds=60,
)

# Трекер бюджета
tracker = BudgetTracker(
    global_budget=global_budget,
    node_budgets={"solver": node_budget},
    warning_threshold=0.8,  # Предупреждение при 80%
)

# Проверка доступности
if tracker.is_available("solver", tokens=100):
    # Записать использование
    tracker.record_usage(
        node_id="solver",
        tokens=100,
        requests=1,
        duration=1.5,
    )

# Проверка превышения
if tracker.is_exceeded():
    print(f"Бюджет превышен: {tracker.exceeded_types()}")

# Предупреждения
warnings = tracker.get_warnings()
for w in warnings:
    print(f"Предупреждение: {w['type']} на {w['ratio']:.1%}")

# Усечение промпта/ответа при превышении
prompt = "очень длинный промпт..."
truncated = tracker.truncate_prompt(prompt, max_tokens=4000)

# Сброс
tracker.reset()
```

#### Интеграция с RunnerConfig

```python
from rustworkx_framework.execution import RunnerConfig

config = RunnerConfig(
    budget_config={
        "total_token_limit": 50000,
        "per_node_token_limit": 2000,
        "max_prompt_length": 4000,
        "warning_threshold": 0.8,
    },
    enable_budget_tracking=True,
)
```

---

### Обработка ошибок (Error Handling)

Структурированные исключения и политики обработки ошибок.

```python
from rustworkx_framework.execution.errors import (
    ExecutionError,
    TimeoutError,
    RetryExhaustedError,
    BudgetExceededError,
    AgentNotFoundError,
    ValidationError,
    ErrorPolicy,
    ErrorAction,
    ExecutionMetrics,
)

# Политика обработки ошибок
error_policy = ErrorPolicy(
    timeout=ErrorAction.SKIP,            # skip, retry, fallback, fail
    retry_exhausted=ErrorAction.FALLBACK,
    budget_exceeded=ErrorAction.FAIL,
    validation_error=ErrorAction.RETRY,
    max_retries=3,
    retry_delay=1.0,                     # Задержка между попытками (сек)
    exponential_backoff=True,
)

# Применение в конфигурации
config = RunnerConfig(
    error_policy=error_policy,
    max_retries=3,
    timeout=60.0,
)

# Обработка ошибок
try:
    result = runner.run_round(graph)
except TimeoutError as e:
    print(f"Таймаут: {e}")
except RetryExhaustedError as e:
    print(f"Исчерпаны попытки: {e}")
except BudgetExceededError as e:
    print(f"Превышен бюджет: {e}")
except ExecutionError as e:
    print(f"Ошибка выполнения: {e}")
    # Доступ к метрикам
    metrics: ExecutionMetrics = e.metrics
    print(f"Попыток: {metrics.retry_count}")
    print(f"Fallback-ов: {metrics.fallback_count}")

# Получение метрик из результата
if result.errors:
    for error in result.errors:
        print(f"{error['agent_id']}: {error['type']} - {error['message']}")
```

---

### Алгоритмы графа (Graph Algorithms)

Сервисный слой для анализа графа с использованием алгоритмов `rustworkx`.

```python
from rustworkx_framework.core.algorithms import (
    GraphAlgorithms,
    CentralityType,
    PathMetric,
    SubgraphFilter,
)

algo = GraphAlgorithms(graph)

# K кратчайших путей
paths = algo.k_shortest_paths(
    source="researcher",
    target="writer",
    k=3,
    metric=PathMetric.HOP_COUNT,  # HOP_COUNT, WEIGHTED, RELIABILITY
    edge_weights=None,             # или custom weights
)
for i, path in enumerate(paths):
    print(f"Путь {i+1}: {path.nodes} (cost={path.cost:.2f})")

# Центральность узлов
centrality = algo.centrality(
    centrality_type=CentralityType.BETWEENNESS,  # DEGREE, BETWEENNESS, CLOSENESS, EIGENVECTOR, PAGERANK
    normalized=True,
)
print(f"Самый центральный узел: {centrality.top_nodes[0]}")
print(f"Оценки: {centrality.scores}")

# Поиск сообществ
communities = algo.detect_communities(algorithm="louvain")  # louvain, label_propagation
print(f"Найдено сообществ: {len(communities.communities)}")
print(f"Модулярность: {communities.modularity:.3f}")

# Поиск циклов
cycles = algo.find_cycles(max_length=5)
if cycles.has_cycles:
    print(f"Найдено циклов: {len(cycles.cycles)}")
    for cycle in cycles.cycles:
        print(f"  {cycle}")

# Фильтрация подграфа
subgraph_filter = SubgraphFilter(
    include_node_ids=["a", "b", "c"],
    min_edge_weight=0.5,
    max_hop_distance=2,
    from_node="a",
)
subgraph = algo.filter_subgraph(subgraph_filter)
print(f"Узлов в подграфе: {len(subgraph.node_ids)}")

# Анализ достижимости
reachable = algo.get_reachable_nodes("start", max_distance=3)
print(f"Достижимые узлы: {reachable}")

# Топологический порядок
if algo.is_dag():
    topo_order = algo.topological_sort()
    print(f"Топологический порядок: {topo_order}")
```

---

### Отслеживание метрик (Metrics Tracker)

Сбор и агрегация метрик производительности узлов и рёбер.

```python
from rustworkx_framework.core.metrics import (
    MetricsTracker,
    NodeMetrics,
    EdgeMetrics,
    MetricAggregator,
    ExponentialMovingAverage,
    SlidingWindowAverage,
)

tracker = MetricsTracker()

# Запись метрик узла
tracker.record_node_execution(
    node_id="solver",
    success=True,
    latency_ms=150,
    cost_tokens=200,
    quality=0.95,
)

# Запись метрик ребра
tracker.record_edge_traversal(
    source="solver",
    target="checker",
    weight=0.9,
    success=True,
    latency_ms=50,
)

# Получение метрик узла
metrics: NodeMetrics = tracker.get_node_metrics("solver")
print(f"Reliability: {metrics.reliability:.3f}")
print(f"Avg latency: {metrics.avg_latency_ms:.1f}ms")
print(f"Total cost: {metrics.total_cost_tokens}")
print(f"Avg quality: {metrics.avg_quality:.3f}")
print(f"Executions: {metrics.execution_count}")

# Получение метрик ребра
edge_metrics: EdgeMetrics = tracker.get_edge_metrics("solver", "checker")
print(f"Edge reliability: {edge_metrics.reliability:.3f}")
print(f"Traversals: {edge_metrics.traversal_count}")

# Снимок всех метрик
snapshot = tracker.snapshot()
print(f"Timestamp: {snapshot.timestamp}")
print(f"Node metrics: {snapshot.node_metrics}")
print(f"Edge metrics: {snapshot.edge_metrics}")

# История метрик (если включено)
tracker = MetricsTracker(keep_history=True, history_window=100)
# ... записи ...
history = tracker.get_history(node_id="solver")
for snapshot in history.snapshots:
    print(f"{snapshot.timestamp}: {snapshot.metrics}")

# Кастомные агрегаторы
ema = ExponentialMovingAverage(alpha=0.1)
tracker.set_aggregator("solver", "latency", ema)

swa = SlidingWindowAverage(window_size=10)
tracker.set_aggregator("checker", "quality", swa)

# Экспорт метрик
data = tracker.to_dict()
tracker.save("metrics.json")

# Загрузка метрик
tracker = MetricsTracker.load("metrics.json")
```

---

### Визуализация (Visualization)

Инструменты для визуализации графов в различных форматах. Все стили визуализации основаны на **Pydantic моделях** для валидации и типобезопасности.

#### Основные классы

```python
from rustworkx_framework.core.visualization import (
    GraphVisualizer,
    VisualizationStyle,
    MermaidDirection,
    NodeShape,
    NodeStyle,
    EdgeStyle,
    # Convenience functions
    to_mermaid,
    to_ascii,
    to_dot,
    print_graph,
    render_to_image,
    show_graph_interactive,
)
```

#### 1. Быстрое использование (Convenience Functions)

```python
# Простой Mermaid
mermaid_code = to_mermaid(graph, direction=MermaidDirection.LEFT_RIGHT)
print(mermaid_code)

# Простой ASCII
ascii_art = to_ascii(graph, show_edges=True)
print(ascii_art)

# Простой DOT
dot_code = to_dot(graph, graph_name="MyGraph")
print(dot_code)

# Печать в консоль (автоматически выберет Rich или ASCII)
print_graph(graph, format="auto")  # "auto", "colored", "ascii", "mermaid"

# Рендер в изображение (требует системный Graphviz)
render_to_image(graph, "output.png", format="png", dpi=300)
render_to_image(graph, "output.svg", format="svg")

# Интерактивный просмотр (открывает в системном просмотрщике)
show_graph_interactive(graph, graph_name="MyWorkflow")
```

#### 2. Продвинутое использование (GraphVisualizer с кастомными стилями)

**VisualizationStyle**, **NodeStyle**, **EdgeStyle** — это Pydantic модели с валидацией полей.

```python
# Создаём кастомные стили узлов (Pydantic модели)
agent_style = NodeStyle(
    shape=NodeShape.ROUND,      # RECTANGLE, ROUND, STADIUM, CIRCLE, DIAMOND, etc.
    fill_color="#e3f2fd",       # Цвет заливки
    stroke_color="#1976d2",     # Цвет границы
    text_color="#000000",       # Цвет текста
    icon="🤖",                  # Emoji иконка
)

task_style = NodeStyle(
    shape=NodeShape.DIAMOND,
    fill_color="#fff3e0",
    stroke_color="#f57c00",
    icon="📋",
)

# Стили рёбер (Pydantic модели)
workflow_edge = EdgeStyle(
    line_style="solid",         # solid, dashed, dotted
    arrow_head="normal",        # normal, none, diamond
    color="#1976d2",
    label_color="#333333",
)

task_edge = EdgeStyle(
    line_style="dashed",
    color="#f57c00",
)

# Общий стиль визуализации (Pydantic модель)
style = VisualizationStyle(
    direction=MermaidDirection.LEFT_RIGHT,  # TOP_BOTTOM, BOTTOM_TOP, LEFT_RIGHT, RIGHT_LEFT
    agent_style=agent_style,
    task_style=task_style,
    workflow_edge_style=workflow_edge,
    task_edge_style=task_edge,
    show_weights=True,          # Показывать веса рёбер
    show_probabilities=False,   # Показывать вероятности
    show_tools=True,            # Показывать инструменты агентов
    show_descriptions=False,    # Показывать описания
    max_label_length=30,        # Максимальная длина меток
)

# Создаём визуализатор с кастомным стилем
viz = GraphVisualizer(graph, style)

# Mermaid диаграммы
mermaid = viz.to_mermaid(
    direction=MermaidDirection.TOP_BOTTOM,  # Можно переопределить из style
    title="Agent Workflow",                 # Заголовок диаграммы
)
print(mermaid)

# Сохранить Mermaid в файл
viz.save_mermaid("graph.md", title="My Workflow")  # Автоматически обернёт в ```mermaid```
viz.save_mermaid("graph.mmd", title="My Workflow")  # Чистый .mmd без обёртки

# ASCII art для терминала
ascii_art = viz.to_ascii(
    show_edges=True,
    box_width=20,
)
print(ascii_art)

# Graphviz DOT
dot = viz.to_dot(
    graph_name="AgentGraph",
    rankdir="LR",  # TB, LR, BT, RL
)
viz.save_dot("graph.dot", graph_name="AgentGraph")

# Рендеринг в изображение (требует установленный Graphviz)
viz.render_image(
    "output.png",
    format="png",     # png, svg, pdf, jpg
    dpi=300,          # Для растровых форматов
    graph_name="MyGraph",
)

# Интерактивный просмотр
viz.show_interactive(graph_name="MyGraph")  # Откроет в системном просмотрщике

# Матрица смежности (текстовое представление)
matrix = viz.to_adjacency_matrix(show_labels=True)
print(matrix)
```

#### 3. Цветной вывод в терминал (Rich Console)

```python
# Автоматический цветной вывод (если Rich установлен)
print_graph(graph, format="colored")

# Или напрямую через визуализатор
viz = GraphVisualizer(graph)
viz.print_colored()  # Красивый вывод с деревом, таблицами и цветами
```

#### 4. Пример полной настройки

```python
from rustworkx_framework.core.visualization import (
    GraphVisualizer,
    VisualizationStyle,
    NodeStyle,
    EdgeStyle,
    NodeShape,
    MermaidDirection,
)

# Полностью настроенный стиль
custom_style = VisualizationStyle(
    direction=MermaidDirection.LEFT_RIGHT,
    agent_style=NodeStyle(
        shape=NodeShape.ROUND,
        fill_color="#bbdefb",
        stroke_color="#0d47a1",
        icon="🤖",
    ),
    task_style=NodeStyle(
        shape=NodeShape.DIAMOND,
        fill_color="#ffe0b2",
        stroke_color="#e65100",
        icon="📋",
    ),
    workflow_edge_style=EdgeStyle(
        line_style="solid",
        color="#1976d2",
    ),
    task_edge_style=EdgeStyle(
        line_style="dashed",
        color="#f57c00",
    ),
    show_weights=True,
    show_tools=True,
    max_label_length=40,
)

viz = GraphVisualizer(graph, custom_style)

# Создаём все форматы
viz.save_mermaid("docs/graph.md", title="Workflow")
viz.save_dot("docs/graph.dot")
viz.render_image("docs/graph.png", format="png", dpi=150)
viz.render_image("docs/graph.svg", format="svg")

print(viz.to_ascii())
```

#### 5. Установка Graphviz для рендеринга изображений

Для `render_image()` и `render_to_image()` требуется:
1. Python библиотека: `pip install graphviz`
2. Системный Graphviz:
   - Ubuntu/Debian: `sudo apt install graphviz`
   - macOS: `brew install graphviz`
   - Windows: `winget install graphviz` или https://graphviz.org/download/

---

### Схемы графа (Schema System)

Полная система **Pydantic-схем** для типобезопасной валидации, сериализации и миграции графовых данных. Все схемы наследуются от `pydantic.BaseModel` и обеспечивают автоматическую валидацию типов, значений по умолчанию и преобразования данных.

#### Основные классы схем

```python
from rustworkx_framework.core.schema import (
    # Версионирование
    SCHEMA_VERSION,
    SchemaVersion,
    # Типы узлов и рёбер
    NodeType,
    EdgeType,
    # Схемы узлов (Pydantic BaseModel)
    BaseNodeSchema,
    AgentNodeSchema,
    TaskNodeSchema,
    # Схемы рёбер (Pydantic BaseModel)
    BaseEdgeSchema,
    WorkflowEdgeSchema,
    CostMetrics,
    # Схема графа (Pydantic BaseModel)
    GraphSchema,
    # LLM конфигурация (Pydantic BaseModel)
    LLMConfig,
    # Валидация (Pydantic BaseModel)
    ValidationResult,
    SchemaValidator,
    # Миграции
    SchemaMigration,
    MigrationRegistry,
    migrate_schema,
)
```

#### 1. Создание схем узлов (Pydantic модели)

```python
# Агент с полной LLM конфигурацией
agent_node = AgentNodeSchema(
    id="solver",
    type=NodeType.AGENT,
    display_name="Math Solver",
    persona="You are an expert mathematician",
    description="Solves complex math problems step by step",
    tools=["calculator", "wolfram_alpha"],
    # LLM конфигурация (Pydantic модель)
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=2000,
    # Метрики и состояние
    trust_score=0.95,
    quality_score=0.9,
    success_rate=1.0,
    total_calls=0,
    total_tokens_used=0,
    # Pydantic валидирует embedding автоматически
    embedding=[0.1, 0.2, 0.3],  # Может быть list или torch.Tensor
    embedding_dim=3,  # Автоматически заполняется если None
    # Метаданные (произвольные данные)
    metadata={"priority": "high", "category": "math"},
    tags={"solver", "math", "primary"},
)

# Задача
task_node = TaskNodeSchema(
    id="main_task",
    type=NodeType.TASK,
    query="Solve: x^2 + 5x + 6 = 0",
    description="Main mathematical task",
    expected_output="Two solutions: x1, x2",
    max_iterations=10,
    status="pending",  # pending, running, completed, failed
)

# Извлечение LLM конфигурации из агента
llm_config: LLMConfig = agent_node.get_llm_config()
print(f"Model: {llm_config.model_name}")
print(f"Configured: {llm_config.is_configured()}")
print(f"Generation params: {llm_config.to_generation_params()}")

# Проверка наличия LLM конфигурации
if agent_node.has_llm_config():
    print("Agent has LLM configuration")
```

#### 2. Создание схем рёбер (Pydantic модели)

```python
# Базовое ребро с cost metrics (Pydantic модель)
edge = BaseEdgeSchema(
    source="solver",
    target="checker",
    type=EdgeType.WORKFLOW,
    weight=1.0,
    probability=0.95,
    bidirectional=False,
    # Cost metrics (Pydantic модель)
    cost=CostMetrics(
        estimated_tokens=500,
        actual_tokens=None,
        latency_ms=150.0,
        timeout_ms=5000.0,
        trust=0.9,
        reliability=0.95,
        cost_usd=0.01,
        custom={"priority": 1.0},
    ),
    # Pydantic валидирует attr автоматически
    attr=[1.0, 0.95, 0.9],  # Может быть list или torch.Tensor
    attr_dim=3,  # Автоматически заполняется если None
    metadata={"route": "primary"},
)

# Workflow ребро с условной маршрутизацией
conditional_edge = WorkflowEdgeSchema(
    source="solver",
    target="checker",
    type=EdgeType.WORKFLOW,
    weight=0.9,
    probability=1.0,
    # Условная маршрутизация
    condition="source_success",  # Имя встроенного или зарегистрированного условия
    priority=1,                  # Приоритет (выше = раньше проверяется)
    transform="extract_answer",  # Опциональное преобразование данных
    is_conditional=True,         # Автоматически установится если condition задан
)

# Получение признаков ребра
feature_vector = edge.get_feature_vector(feature_names=["trust", "reliability"])
print(f"Features: {feature_vector}")

# Преобразование в torch.Tensor
attr_tensor = edge.to_attr_tensor()
print(f"Attr tensor: {attr_tensor}")
```

#### 3. Полная схема графа (Pydantic модель)

```python
from datetime import datetime

# GraphSchema - главная Pydantic модель
schema = GraphSchema(
    schema_version=SCHEMA_VERSION,  # "2.0.0"
    name="Math Pipeline",
    description="A workflow for solving mathematical problems",
    created_at=datetime.now(),
    updated_at=datetime.now(),
    # nodes - это dict[str, BaseNodeSchema], не list!
    nodes={
        "solver": AgentNodeSchema(
            id="solver",
            display_name="Math Solver",
            description="Solves math problems",
            tools=["calculator"],
            llm_backbone="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="$OPENAI_API_KEY",
        ),
        "checker": AgentNodeSchema(
            id="checker",
            display_name="Answer Checker",
            description="Validates solutions",
            llm_backbone="gpt-4o-mini",
        ),
        "__task__": TaskNodeSchema(
            id="__task__",
            query="Solve: x^2 + 5x + 6 = 0",
        ),
    },
    edges=[
        WorkflowEdgeSchema(
            source="solver",
            target="checker",
            weight=0.9,
            type=EdgeType.WORKFLOW,
        ),
    ],
    # Имена признаков для feature extraction
    node_feature_names=["trust_score", "quality_score"],
    edge_feature_names=["trust", "reliability"],
    # Метаданные
    metadata={
        "created_by": "user@example.com",
        "purpose": "math_pipeline",
        "version": "1.0",
    },
)

# Добавление узлов и рёбер
new_agent = AgentNodeSchema(
    id="reviewer",
    display_name="Reviewer",
)
schema.add_node(new_agent)

new_edge = BaseEdgeSchema(
    source="checker",
    target="reviewer",
)
schema.add_edge(new_edge)

# Получение узлов и рёбер
solver_node = schema.get_node("solver")
edges_from_solver = schema.get_edges(source="solver")
edges_to_checker = schema.get_edges(target="checker")

# Вычисление размерностей признаков
schema.compute_feature_dims()
print(f"Node feature dim: {schema.node_feature_dim}")
print(f"Edge feature dim: {schema.edge_feature_dim}")
```

#### 4. Сериализация и валидация (Pydantic)

```python
# Сериализация (Pydantic методы)
schema_dict = schema.model_dump()  # Dict[str, Any]
schema_json = schema.model_dump_json(indent=2)  # JSON string

# Или специальный метод
schema_data = schema.to_dict()

# Десериализация (Pydantic методы)
loaded_schema = GraphSchema.model_validate(schema_dict)
loaded_from_json = GraphSchema.model_validate_json(schema_json)

# Валидация схемы (возвращает ValidationResult - Pydantic модель)
validator = SchemaValidator(
    check_cycles=True,
    check_duplicates=True,
    check_orphans=True,
    check_connectivity=False,
)
result: ValidationResult = validator.validate(schema)

if result.valid:
    print("✓ Schema is valid")
else:
    print("✗ Validation errors:")
    for error in result.errors:
        print(f"  - {error}")

if result.warnings:
    print("⚠ Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

#### 5. Миграция схем между версиями

```python
# Автоматическая миграция устаревших данных
old_data = {
    "schema_version": "1.0.0",
    "agents": [  # Старый формат (список agents)
        {"identifier": "solver", "display_name": "Solver"},
    ],
    "edges": [
        {"source": "solver", "target": "checker"},
    ],
}

# Миграция до текущей версии (2.0.0)
migrated_data = migrate_schema(old_data)
print(f"Migrated to version: {migrated_data['schema_version']}")

# Создание собственной миграции
from rustworkx_framework.core.schema import SchemaMigration, register_migration

class MyCustomMigration(SchemaMigration):
    from_version = "1.5.0"
    to_version = "2.0.0"

    def migrate(self, data: dict) -> dict:
        # Ваша логика миграции
        data["new_field"] = "default_value"
        return data

# Регистрация миграции
register_migration(MyCustomMigration())
```

#### 6. Версионирование

```python
# Проверка версии схемы
current_version = SchemaVersion.parse(SCHEMA_VERSION)  # "2.0.0"
print(f"Current: {current_version}")

old_version = SchemaVersion.parse("1.5.0")
print(f"Compatible: {current_version.is_compatible(old_version)}")  # False (разные мажорные версии)
print(f"Newer: {current_version > old_version}")  # True
```

#### Преимущества Pydantic схем

1. **Автоматическая валидация типов** - Pydantic проверяет типы при создании объектов
2. **Значения по умолчанию** - автоматическое заполнение полей
3. **Преобразование типов** - автоматическое преобразование (torch.Tensor → list)
4. **Сериализация/десериализация** - встроенные методы `.model_dump()`, `.model_validate()`
5. **Расширяемость** - `extra="allow"` позволяет добавлять произвольные поля
6. **Иммутабельность** - `frozen=True` для неизменяемых моделей
7. **Документация** - автогенерация JSON Schema

---

#### 7. Валидация input/output данных агентов

**Новинка:** Каждый агент может иметь **input_schema** и **output_schema** для валидации входящих данных и ответов. Это позволяет:
- 🔒 Гарантировать корректность данных
- 📝 Автоматически парсить структурированные ответы
- 🚫 Отлавливать некорректные ответы LLM
- 📋 Генерировать JSON Schema для промптов

##### Импорты

```python
from pydantic import BaseModel
from rustworkx_framework.core.schema import (
    AgentNodeSchema,
    SchemaValidationResult,  # Результат валидации
)
from rustworkx_framework.builder import GraphBuilder
```

##### 7.1. Создание агента с Pydantic схемами

```python
# Определяем схемы ввода/вывода как Pydantic модели
class SolverInput(BaseModel):
    question: str
    context: str | None = None
    difficulty: int = 1

class SolverOutput(BaseModel):
    answer: str
    confidence: float  # 0.0 - 1.0
    explanation: str | None = None

# Создаём агента с валидацией
builder = GraphBuilder()
builder.add_agent(
    "solver",
    display_name="Math Solver",
    persona="Expert mathematician",
    description="Solves mathematical problems",
    # Схемы для валидации
    input_schema=SolverInput,
    output_schema=SolverOutput,
    # LLM конфигурация
    llm_backbone="gpt-4",
    temperature=0.0,
)

graph = builder.build()
```

##### 7.2. Использование JSON Schema (без Pydantic)

Можно передать обычный словарь с JSON Schema:

```python
# JSON Schema напрямую (без Pydantic моделей)
input_schema = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "context": {"type": "string"},
    },
    "required": ["question"]
}

output_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"]
}

builder.add_agent(
    "solver",
    input_schema=input_schema,   # JSON Schema dict
    output_schema=output_schema, # JSON Schema dict
)
```

##### 7.3. Валидация через RoleGraph

```python
# Проверка наличия схем
has_input = graph.has_input_schema("solver")   # True
has_output = graph.has_output_schema("solver") # True

# Валидация входных данных
result: SchemaValidationResult = graph.validate_agent_input(
    "solver",
    {"question": "Solve x^2 + 5x + 6 = 0"}
)

if result.valid:
    print("✅ Input is valid")
    print(f"Validated data: {result.validated_data}")
else:
    print("❌ Input validation failed")
    print(f"Errors: {result.errors}")

# Валидация выходных данных (JSON строка или dict)
response = '{"answer": "x1=-2, x2=-3", "confidence": 0.95}'
result = graph.validate_agent_output("solver", response)

if result.valid:
    parsed = result.validated_data
    print(f"Answer: {parsed['answer']}")
    print(f"Confidence: {parsed['confidence']}")
else:
    print(f"Invalid output: {result.errors}")
    # Можно выбросить исключение
    result.raise_if_invalid()  # -> ValueError
```

##### 7.4. Получение JSON Schema для промптов

```python
# Получить JSON Schema для инструкций LLM
input_schema_json = graph.get_input_schema_json("solver")
output_schema_json = graph.get_output_schema_json("solver")

# Использовать в промпте
prompt = f"""You are a math solver.

INPUT FORMAT:
{json.dumps(input_schema_json, indent=2)}

You MUST respond in the following JSON format:
{json.dumps(output_schema_json, indent=2)}

Now solve: {{question}}
"""
```

##### 7.5. Валидация напрямую через AgentNodeSchema

```python
# Создание агента с схемами
agent = AgentNodeSchema(
    id="solver",
    display_name="Math Solver",
    input_schema=SolverInput,
    output_schema=SolverOutput,
)

# Валидация
result = agent.validate_input({"question": "2+2=?"})
print(f"Valid: {result.valid}")

result = agent.validate_output('{"answer": "4", "confidence": 0.99}')
print(f"Valid: {result.valid}, data: {result.validated_data}")

# Проверка наличия схем
if agent.has_input_schema():
    print("Agent has input schema")
if agent.has_output_schema():
    print("Agent has output schema")
```

##### 7.6. Обработка невалидных ответов LLM

```python
# Сценарий: LLM отвечает не в том формате
response = llm_call(prompt)
result = graph.validate_agent_output("solver", response)

if not result.valid:
    # Вариант 1: Retry с более строгим промптом
    retry_prompt = f"{prompt}\n\n⚠️ IMPORTANT: You MUST respond with valid JSON!"
    response = llm_call(retry_prompt)
    result = graph.validate_agent_output("solver", response)

    if not result.valid:
        # Вариант 2: Fallback на дефолтные значения
        parsed = {
            "answer": response,
            "confidence": 0.5,
            "explanation": "LLM failed to format correctly"
        }
    else:
        parsed = result.validated_data
else:
    parsed = result.validated_data

print(f"Final answer: {parsed['answer']}")
```

##### 7.7. SchemaValidationResult API

```python
class SchemaValidationResult(BaseModel):
    """Результат валидации данных по схеме."""

    valid: bool                              # True если данные валидны
    schema_type: str                         # "input" или "output"
    errors: list[str]                        # Список ошибок валидации
    warnings: list[str]                      # Список предупреждений
    validated_data: dict[str, Any] | None    # Провалидированные данные
    message: str                             # Дополнительное сообщение

# Методы
result.raise_if_invalid()  # Выбросить ValueError если невалидно
```

##### 7.8. Поддержка сериализации

При сохранении графа:
- **Pydantic модели** (`input_schema`/`output_schema`) **НЕ** сериализуются (exclude=True)
- **JSON Schema** (`input_schema_json`/`output_schema_json`) **сериализуются**

```python
# При создании агента с Pydantic моделью
agent = AgentNodeSchema(
    id="solver",
    input_schema=SolverInput,   # Не сериализуется
    output_schema=SolverOutput,  # Не сериализуется
)

# Автоматически извлекается JSON Schema
print(agent.input_schema_json)   # {'type': 'object', 'properties': {...}}
print(agent.output_schema_json)  # {'type': 'object', 'properties': {...}}

# При десериализации графа из JSON
# Pydantic модели будут потеряны, но JSON Schema останется
# Валидация будет работать через базовую проверку типов
```

##### Когда использовать input/output схемы?

| Сценарий | Рекомендация |
|----------|--------------|
| **Структурированные данные** | ✅ Используй Pydantic схемы |
| **JSON ответы LLM** | ✅ Обязательно! Парсинг и валидация |
| **Свободный текст** | ❌ Не нужно |
| **API интеграция** | ✅ Гарантия корректности данных |
| **Отладка** | ✅ Быстрое выявление проблем |

##### Влияние на производительность

- ✅ **Валидация не расходует токены** — это чистый Python-код
- ⚠️ **Инструкции в промпте расходуют токены** — добавление JSON Schema в промпт увеличивает расход
- ⚡ **Валидация быстрая** — Pydantic оптимизирован для скорости

##### FAQ по валидации

**Q: Это обязательно?**
A: Нет, полностью опционально. Если схемы не заданы — валидация пропускается.

**Q: Что если LLM не может ответить в формате?**
A: `validate_output()` вернёт `valid=False` + ошибки. Решение: retry/fallback/игнорировать.

**Q: Можно передать просто JSON Schema?**
A: Да! Передай dict с JSON Schema вместо Pydantic модели.

**Q: Увеличивается расход токенов?**
A: Валидация не расходует токены. Но если добавишь JSON Schema в промпт — да, расход увеличится.

---

### Builder API (Подробно)

Различные способы построения графов.

#### 1. build_property_graph (Быстрое построение)

```python
from rustworkx_framework.builder import build_property_graph

graph = build_property_graph(
    agents=[agent1, agent2, agent3],
    workflow_edges=[("agent1", "agent2"), ("agent2", "agent3")],
    context_edges=[("agent1", "agent3")],  # Дополнительные связи
    query="Solve this task",
    include_task_node=True,               # Добавить узел задачи
    task_node_id="__task__",              # ID узла задачи
    connect_task_to_all=False,            # Соединить задачу со всеми агентами
    edge_weights=None,                    # Custom веса рёбер
    default_weight=1.0,                   # Вес по умолчанию
    bidirectional=False,                  # Двунаправленные рёбра
    encoder=None,                         # NodeEncoder для эмбеддингов
    compute_embeddings=False,             # Вычислить эмбеддинги сразу
)
```

#### 2. GraphBuilder (Fluent API)

```python
from rustworkx_framework.builder import GraphBuilder

builder = GraphBuilder()

# Добавление агентов (базовое)
builder.add_agent(
    identifier="researcher",
    display_name="Researcher",
    description="Does research",
    tools=["search", "read"],
)

# Добавление агента с мультимодельной конфигурацией
builder.add_agent(
    identifier="analyst",
    display_name="Senior Analyst",
    persona="Expert data analyst",
    # LLM конфигурация
    llm_backbone="gpt-4",              # Имя модели
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",         # Или $ENV_VAR
    temperature=0.7,
    max_tokens=2000,
    timeout=60.0,
    top_p=0.9,
    stop_sequences=["END", "STOP"],
)

# Или через LLMConfig объект
from rustworkx_framework.core.schema import LLMConfig

llm_config = LLMConfig(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.7,
    max_tokens=2000,
)

builder.add_agent(
    identifier="writer",
    display_name="Writer",
    llm_config=llm_config,  # Передать готовую конфигурацию
)

# Добавление рёбер
builder.add_workflow_edge("researcher", "writer", weight=0.9)
builder.add_context_edge("researcher", "writer", weight=0.5)

# Добавление задачи
builder.set_task(query="Write a report", description="Main task")

# Условные рёбра
def quality_check(state: dict) -> bool:
    return state.get("quality_score", 0) > 0.8

builder.add_conditional_edge(
    source="writer",
    target="editor",
    condition=quality_check,
    weight=0.9,
)

# Установить границы выполнения (новое!)
builder.set_start_node("researcher")    # Стартовая нода
builder.set_end_node("writer")          # Конечная нода
# Или обе сразу:
builder.set_execution_bounds("researcher", "writer")

# Построение графа
graph = builder.build(compute_embeddings=True, encoder=my_encoder)

# Валидация перед построением
is_valid, errors = builder.validate()
if not is_valid:
    print(f"Ошибки: {errors}")
```

#### 3. build_from_adjacency (Из матрицы)

```python
from rustworkx_framework.builder import build_from_adjacency
import torch

adjacency = torch.tensor([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
], dtype=torch.float32)

graph = build_from_adjacency(
    adjacency_matrix=adjacency,
    agents=[agent1, agent2, agent3],
    query="Task",
    threshold=0.1,  # Игнорировать рёбра с весом < threshold
)
```

#### 4. build_from_schema (Из схемы)

```python
from rustworkx_framework.builder import build_from_schema

graph = build_from_schema(
    schema=my_schema,
    compute_embeddings=True,
    encoder=my_encoder,
    validate=True,  # Валидация перед построением
)
```

---

### Система событий (Event System)

Подписка на события для мониторинга и отладки.

```python
from rustworkx_framework.core.events import (
    EventBus,
    global_event_bus,
    EventType,
    LoggingEventHandler,
    MetricsEventHandler,
    on_event,
    # События
    NodeAddedEvent,
    EdgeAddedEvent,
    StepCompletedEvent,
    BudgetWarningEvent,
)

# Получение глобальной шины событий
bus = global_event_bus()

# 1. Подписка через обработчик
logging_handler = LoggingEventHandler(
    log_level="INFO",
    include_metadata=True,
)
bus.subscribe(EventType.STEP_COMPLETED, logging_handler)

# 2. Подписка через функцию
def on_step_completed(event):
    if isinstance(event, StepCompletedEvent):
        print(f"Agent {event.agent_id} completed: {event.tokens_used} tokens")

bus.subscribe(EventType.STEP_COMPLETED, on_step_completed)

# 3. Подписка через декоратор
@on_event(EventType.BUDGET_WARNING)
def handle_budget_warning(event: BudgetWarningEvent):
    print(f"⚠️  Budget warning: {event.budget_type} at {event.ratio:.1%}")

# 4. Глобальная подписка (на все события)
@on_event(None)
def handle_all_events(event):
    print(f"Event: {event.event_type.value}")

# Отключение обработки событий
bus.disable()

# Включение
bus.enable()

# Очистка всех обработчиков
bus.clear()

# Агрегация метрик через события
metrics_handler = MetricsEventHandler()
bus.subscribe(None, metrics_handler)

# После выполнения
metrics = metrics_handler.get_metrics()
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Errors: {metrics['errors_count']}")
print(f"Budget warnings: {metrics['budget_warnings']}")
```

---

### Callback-система

Мониторинг и логирование выполнения через callback handlers

#### Основные концепции

- **`BaseCallbackHandler`** — базовый класс для создания callback-обработчиков
- **`AsyncCallbackHandler`** — async версия для асинхронных операций
- **`CallbackManager`** — менеджер, который управляет и вызывает handlers
- **Встроенные handlers** — StdoutCallbackHandler, MetricsCallbackHandler, FileCallbackHandler

#### Быстрый старт

```python
from rustworkx_framework import MACPRunner
from rustworkx_framework.callbacks import (
    StdoutCallbackHandler,
    MetricsCallbackHandler,
    FileCallbackHandler,
)

# 1. Callbacks через RunnerConfig
from rustworkx_framework.execution import RunnerConfig

config = RunnerConfig(
    callbacks=[
        StdoutCallbackHandler(show_outputs=True),
        MetricsCallbackHandler(),
    ]
)

runner = MACPRunner(llm_caller=my_llm, config=config)
result = runner.run_round(graph)

# 2. Per-run callbacks (переопределяют config)
result = runner.run_round(
    graph,
    callbacks=[FileCallbackHandler("execution_log.jsonl")]
)
```

#### Context Manager

```python
from rustworkx_framework.callbacks import collect_metrics, trace_as_callback

# 1. Сбор метрик
with collect_metrics() as metrics:
    runner.run_round(graph)

    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total duration: {metrics.total_duration_ms}ms")
    print(f"Runs completed: {metrics.runs_completed}")
    print(f"Runs failed: {metrics.runs_failed}")

    # Полная статистика
    all_metrics = metrics.get_metrics()
    print(f"Agent calls: {all_metrics['agent_calls']}")
    print(f"Errors: {all_metrics['errors_count']}")

# 2. Трассировка с произвольными handlers
from rustworkx_framework.callbacks import StdoutCallbackHandler

with trace_as_callback(handlers=[StdoutCallbackHandler()]) as manager:
    runner.run_round(graph)
    # Callbacks автоматически применяются к этому запуску
```

#### Создание своего CallbackHandler

```python
from rustworkx_framework.callbacks import BaseCallbackHandler
from uuid import UUID

class MySlackAlertHandler(BaseCallbackHandler):
    """Отправляет алерты в Slack при ошибках."""

    def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        num_agents: int = 0,
        **kwargs,
    ) -> None:
        send_slack(f"🚀 Started run {run_id}: {num_agents} agents")

    def on_agent_end(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        output: str,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> None:
        print(f"✅ Agent {agent_id}: {tokens_used} tokens, {duration_ms:.0f}ms")

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        agent_id: str,
        **kwargs,
    ) -> None:
        send_slack_alert(
            f"❌ Agent {agent_id} failed in run {run_id}: {error}",
            severity="high"
        )

    def on_run_end(
        self,
        *,
        run_id: UUID,
        output: str,
        success: bool = True,
        total_tokens: int = 0,
        **kwargs,
    ) -> None:
        if not success:
            send_slack_alert(f"🛑 Run {run_id} failed!")
        else:
            send_slack(f"✅ Run {run_id} completed: {total_tokens} tokens")

# Использование
runner = MACPRunner(
    llm_caller=my_llm,
    config=RunnerConfig(callbacks=[MySlackAlertHandler()])
)
```

#### Async Callbacks

```python
from rustworkx_framework.callbacks import AsyncCallbackHandler
import aiohttp

class AsyncWebhookHandler(AsyncCallbackHandler):
    """Асинхронно отправляет webhook при событиях."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        **kwargs,
    ) -> None:
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.webhook_url,
                json={"event": "run_start", "run_id": str(run_id), "query": query}
            )

    async def on_agent_end(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        output: str,
        tokens_used: int = 0,
        **kwargs,
    ) -> None:
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.webhook_url,
                json={
                    "event": "agent_end",
                    "run_id": str(run_id),
                    "agent_id": agent_id,
                    "tokens": tokens_used,
                }
            )

# Использование с async runner
runner = MACPRunner(
    async_llm_caller=my_async_llm,
    config=RunnerConfig(callbacks=[AsyncWebhookHandler("https://api.example.com/webhook")])
)

result = await runner.arun_round(graph)
```

#### Встроенные Handlers

##### 1. StdoutCallbackHandler — вывод в консоль

```python
from rustworkx_framework.callbacks import StdoutCallbackHandler

handler = StdoutCallbackHandler(
    color=True,                  # Цветной вывод
    show_prompts=False,          # Показывать промпты
    show_outputs=True,           # Показывать ответы агентов
    truncate_length=200,         # Длина обрезки текста
)

runner = MACPRunner(
    llm_caller=my_llm,
    config=RunnerConfig(callbacks=[handler])
)

# Вывод:
# 🚀 Run started: 5 agents
#    Order: researcher → analyst → writer → editor → publisher
#   ▶️  [0] Researcher started
#   ✅ [0] Researcher completed: 150 tokens, 1200ms
#      Output: Market analysis shows strong growth...
#   ▶️  [1] Analyst started
#   ✅ [1] Analyst completed: 200 tokens, 1500ms [FINAL]
# ✅ Run completed: 350 tokens, 2700ms
```

##### 2. MetricsCallbackHandler — агрегация метрик

```python
from rustworkx_framework.callbacks import MetricsCallbackHandler

metrics_handler = MetricsCallbackHandler()

runner = MACPRunner(
    llm_caller=my_llm,
    config=RunnerConfig(callbacks=[metrics_handler])
)

result = runner.run_round(graph)

# Получение метрик
metrics = metrics_handler.get_metrics()

print(f"Total tokens: {metrics['total_tokens']}")
print(f"Total duration: {metrics['total_duration_ms']}ms")
print(f"Agent calls: {metrics['agent_calls']}")  # {'researcher': 1, 'writer': 1, ...}
print(f"Agent tokens: {metrics['agent_tokens']}")  # {'researcher': 150, ...}
print(f"Errors: {metrics['errors_count']}")
print(f"Retries: {metrics['retries']}")
print(f"Budget warnings: {metrics['budget_warnings']}")
print(f"Runs completed: {metrics['runs_completed']}")

# Средние значения
print(f"Avg tokens per agent: {metrics['avg_tokens_per_agent']}")

# Последние 10 ошибок
for error in metrics['errors']:
    print(f"Error in {error['agent_id']}: {error['error_message']}")

# Сброс метрик
metrics_handler.reset()
```

##### 3. FileCallbackHandler — запись в JSON Lines файл

```python
from rustworkx_framework.callbacks import FileCallbackHandler

handler = FileCallbackHandler(
    file_path="execution_log.jsonl",
    append=True,           # Дописывать или перезаписывать
    flush_every=1,         # Flush после каждого события
)

runner = MACPRunner(
    llm_caller=my_llm,
    config=RunnerConfig(callbacks=[handler])
)

result = runner.run_round(graph)

# Закрыть файл вручную (или автоматически через __del__)
handler.close()

# Формат файла (JSON Lines):
# {"event_type": "run_start", "timestamp": "2024-...", "run_id": "...", "query": "...", "num_agents": 5}
# {"event_type": "agent_start", "timestamp": "...", "run_id": "...", "agent_id": "researcher", ...}
# {"event_type": "agent_end", "timestamp": "...", "run_id": "...", "agent_id": "researcher", "tokens_used": 150, ...}
```

#### Доступные callback-методы

| Метод | Описание | Параметры |
|-------|----------|-----------|
| `on_run_start` | Начало выполнения | `run_id`, `query`, `num_agents`, `execution_order` |
| `on_run_end` | Конец выполнения | `run_id`, `output`, `success`, `error`, `total_tokens`, `total_time_ms`, `executed_agents` |
| `on_agent_start` | Агент начал работу | `run_id`, `agent_id`, `agent_name`, `step_index`, `prompt`, `predecessors` |
| `on_agent_end` | Агент завершил работу | `run_id`, `agent_id`, `output`, `tokens_used`, `duration_ms`, `is_final` |
| `on_agent_error` | Ошибка агента | `error`, `run_id`, `agent_id`, `error_type`, `will_retry`, `attempt` |
| `on_retry` | Повторная попытка | `run_id`, `agent_id`, `attempt`, `max_attempts`, `delay_ms`, `error` |
| `on_llm_new_token` | Новый токен (streaming) | `token`, `run_id`, `agent_id`, `token_index`, `is_first`, `is_last` |
| `on_plan_created` | План создан | `run_id`, `num_steps`, `execution_order` |
| `on_replan` | Перепланирование | `run_id`, `reason`, `old_remaining`, `new_remaining`, `replan_count` |
| `on_prune` | Агент обрезан | `run_id`, `agent_id`, `reason` |
| `on_fallback` | Fallback активирован | `run_id`, `failed_agent_id`, `fallback_agent_id`, `reason` |
| `on_parallel_start` | Начало параллельной группы | `run_id`, `agent_ids`, `group_index` |
| `on_parallel_end` | Конец параллельной группы | `run_id`, `agent_ids`, `successful`, `failed` |
| `on_memory_read` | Чтение из памяти | `run_id`, `agent_id`, `entries_count`, `keys` |
| `on_memory_write` | Запись в память | `run_id`, `agent_id`, `key`, `value_size` |
| `on_budget_warning` | Предупреждение бюджета | `run_id`, `budget_type`, `current`, `limit`, `ratio` |
| `on_budget_exceeded` | Бюджет превышен | `run_id`, `budget_type`, `current`, `limit`, `action_taken` |

#### Ignore флаги

Можно отключить определённые типы событий:

```python
class MyMinimalHandler(BaseCallbackHandler):
    # Игнорируем большинство событий
    ignore_llm = True       # Не вызывать on_llm_new_token
    ignore_retry = True     # Не вызывать on_retry
    ignore_budget = True    # Не вызывать on_budget_*
    ignore_memory = True    # Не вызывать on_memory_*

    # Обрабатываем только ошибки
    def on_agent_error(self, error, *, run_id, agent_id, **kwargs):
        log_critical_error(agent_id, error)
```

#### Комбинирование handlers

```python
from rustworkx_framework.callbacks import (
    StdoutCallbackHandler,
    MetricsCallbackHandler,
    FileCallbackHandler,
)

# Можно использовать несколько handlers одновременно
runner = MACPRunner(
    llm_caller=my_llm,
    config=RunnerConfig(callbacks=[
        StdoutCallbackHandler(show_outputs=False),  # Только статус в консоль
        MetricsCallbackHandler(),                   # Сбор метрик
        FileCallbackHandler("debug.jsonl"),         # Полный лог в файл
        MySlackAlertHandler(),                      # Алерты в Slack
    ])
)
```

---

### Хранилище состояний (State Storage)

Персистентное хранение состояний узлов.

```python
from rustworkx_framework.utils.state_storage import (
    InMemoryStateStorage,
    FileStateStorage,
)

# 1. In-memory хранилище
storage = InMemoryStateStorage()

storage.save("agent_id", {"messages": [...], "context": {...}})
state = storage.load("agent_id")
storage.delete("agent_id")

all_keys = storage.keys()
storage.clear()

# 2. Файловое хранилище
storage = FileStateStorage(directory="./agent_states")

storage.save("researcher", {
    "messages": [{"role": "user", "content": "Hello"}],
    "iteration": 5,
})

state = storage.load("researcher")
if state:
    print(f"Iteration: {state['iteration']}")

storage.delete("researcher")

# Получить все сохранённые ID
all_agent_ids = storage.keys()

# Очистить все состояния
storage.clear()
```

---

### Асинхронные утилиты (Async Utils)

Вспомогательные функции для асинхронного выполнения.

```python
from rustworkx_framework.utils.async_utils import (
    run_sync,
    gather_with_concurrency,
    timeout_wrapper,
)

# 1. Синхронный запуск корутины
async def my_async_function():
    return "result"

result = run_sync(my_async_function(), context="my_context")

# 2. Параллельное выполнение с ограничением
async def fetch_data(agent_id: str):
    # ... асинхронный вызов ...
    return response

async def main():
    tasks = [fetch_data(f"agent_{i}") for i in range(20)]

    # Выполнить не более 5 одновременно
    results = await gather_with_concurrency(5, *tasks)
    return results

# 3. Таймауты
async def slow_operation():
    await asyncio.sleep(10)
    return "done"

async def main():
    try:
        result = await timeout_wrapper(
            slow_operation(),
            timeout=5.0,
            error_message="Operation took too long",
        )
    except TimeoutError as e:
        print(f"Timeout: {e}")
```

---

### Условная маршрутизация (Conditional Routing)

Динамический выбор следующего агента на основе условий.

```python
from rustworkx_framework.core.graph import ConditionalEdge
from rustworkx_framework.execution.scheduler import ConditionContext, ConditionEvaluator

# 1. Определение условных рёбер
def quality_above_threshold(context: ConditionContext) -> bool:
    """Перейти к editor только если качество > 0.8"""
    quality = context.state.get("quality_score", 0)
    return quality > 0.8

def has_errors(context: ConditionContext) -> bool:
    """Перейти к fixer если есть ошибки"""
    return "errors" in context.state and len(context.state["errors"]) > 0

# Добавление условных рёбер в граф
graph.add_conditional_edge(
    source="writer",
    targets={
        "editor": quality_above_threshold,
        "fixer": has_errors,
    },
    default="reviewer",  # Fallback если ни одно условие не выполнено
)

# 2. Использование в билдере
from rustworkx_framework.builder import GraphBuilder

builder = GraphBuilder()
builder.add_agent(identifier="writer", display_name="Writer")
builder.add_agent(identifier="editor", display_name="Editor")
builder.add_agent(identifier="fixer", display_name="Fixer")

builder.add_conditional_edge(
    source="writer",
    target="editor",
    condition=quality_above_threshold,
    weight=0.9,
)
builder.add_conditional_edge(
    source="writer",
    target="fixer",
    condition=has_errors,
    weight=0.7,
)

graph = builder.build()

# 3. Оценка условий в runtime
evaluator = ConditionEvaluator()

context = ConditionContext(
    current_node="writer",
    state={"quality_score": 0.85, "errors": []},
    history=["researcher", "writer"],
    metadata={"iteration": 1},
)

# Оценить одно условие
if evaluator.evaluate(quality_above_threshold, context):
    next_node = "editor"

# Оценить все условия для узла
next_nodes = evaluator.evaluate_all(graph, "writer", context)
print(f"Next nodes: {next_nodes}")
```

---

## API Reference

### Основные классы

| Класс | Описание | Pydantic |
|-------|----------|----------|
| `RoleGraph` | Граф ролей/агентов с матрицами смежности | ❌ |
| `AgentProfile` | **Pydantic BaseModel** — Иммутабельный профиль агента | ✅ |
| `TaskNode` | **Pydantic BaseModel** — Виртуальный узел задачи | ✅ |
| `NodeEncoder` | Кодировщик текста в эмбеддинги | ❌ |
| `MACPRunner` | Исполнитель протокола MACP | ❌ |
| `AdaptiveScheduler` | Адаптивный планировщик | ❌ |
| `LLMCallerFactory` | Фабрика для создания LLM callers (мультимодельность) | ❌ |
| `LLMConfig` | **Pydantic BaseModel** — Конфигурация LLM для схем | ✅ |
| `AgentLLMConfig` | **Pydantic BaseModel** — LLM конфигурация для AgentProfile | ✅ |
| `AgentMemory` | Менеджер памяти агента | ❌ |
| `SharedMemoryPool` | Пул шаренной памяти | ❌ |
| `BudgetTracker` | Трекер бюджета токенов/запросов | ❌ |
| `MetricsTracker` | Трекер метрик производительности | ❌ |
| `GraphVisualizer` | Визуализация графов | ❌ |
| `BaseCallbackHandler` | Базовый callback handler | ❌ |
| `AsyncCallbackHandler` | Async callback handler | ❌ |
| `CallbackManager` | Менеджер callback handlers | ❌ |
| `AsyncCallbackManager` | Async менеджер callbacks | ❌ |
| `StdoutCallbackHandler` | Вывод событий в консоль | ❌ |
| `MetricsCallbackHandler` | Агрегация метрик выполнения | ❌ |
| `FileCallbackHandler` | Запись событий в JSON Lines файл | ❌ |
| `EventBus` | Шина событий для мониторинга графа | ❌ |
| `EarlyStopCondition` | Условие ранней остановки выполнения | ❌ |
| `StepContext` | **Pydantic BaseModel** — Контекст шага для hooks | ✅ |
| `TopologyAction` | **Pydantic BaseModel** — Действие модификации топологии | ✅ |

### Схемы (Pydantic BaseModel)

| Класс схемы | Описание | Использование |
|-------------|----------|---------------|
| `GraphSchema` | **Pydantic** — Полная схема графа | Валидация, сериализация, миграция |
| `BaseNodeSchema` | **Pydantic** — Базовая схема узла | Родительский класс для узлов |
| `AgentNodeSchema` | **Pydantic** — Схема узла-агента | LLM config, tools, метрики, эмбеддинги |
| `TaskNodeSchema` | **Pydantic** — Схема узла-задачи | Query, статус, deadline |
| `BaseEdgeSchema` | **Pydantic** — Базовая схема ребра | Weight, probability, cost |
| `WorkflowEdgeSchema` | **Pydantic** — Workflow ребро | Условия, приоритет, трансформации |
| `CostMetrics` | **Pydantic** — Метрики стоимости | Токены, latency, trust, reliability |
| `ValidationResult` | **Pydantic** — Результат валидации | Errors, warnings |

### Визуализация (Pydantic BaseModel)

| Класс | Описание | Использование |
|-------|----------|---------------|
| `VisualizationStyle` | **Pydantic** — Общий стиль визуализации | Настройка цветов, форм, показа элементов |
| `NodeStyle` | **Pydantic** — Стиль узла | Shape, fill_color, stroke_color, icon |
| `EdgeStyle` | **Pydantic** — Стиль ребра | Line style, arrow, colors |
| `NodeShape` | Enum — Формы узлов | RECTANGLE, ROUND, STADIUM, CIRCLE, DIAMOND, etc. |
| `MermaidDirection` | Enum — Направление графа | TOP_BOTTOM, LEFT_RIGHT, etc. |

### GNN (Pydantic BaseModel)

| Класс | Описание | Использование |
|-------|----------|---------------|
| `FeatureConfig` | **Pydantic** — Конфигурация признаков | Node/edge feature dimensions |
| `TrainingConfig` | **Pydantic** — Конфигурация обучения | Learning rate, epochs, optimizer |

### Функции построения графа

| Функция | Описание |
|---------|----------|
| `build_property_graph()` | Основной билдер графа |
| `build_from_schema()` | Построение из GraphSchema |
| `build_from_adjacency()` | Построение из матрицы смежности |
| `GraphBuilder` | Fluent-билдер графа с поддержкой мультимодельности |

### Функции для мультимодельности

| Функция | Описание |
|---------|----------|
| `create_openai_caller()` | Создание OpenAI-совместимого LLM caller |
| `LLMCallerFactory.create_openai_factory()` | Создание фабрики для автоматической генерации callers |
| `LLMConfig.merge_with()` | Слияние конфигураций LLM (fallback) |
| `AgentProfile.with_llm_config()` | Установка LLM конфигурации для агента |
| `AgentProfile.has_custom_llm()` | Проверка наличия кастомной LLM конфигурации |

### Функции планирования

| Функция | Описание |
|---------|----------|
| `build_execution_order()` | Топологический порядок |
| `get_parallel_groups()` | Группы для параллельного выполнения |
| `extract_agent_adjacency()` | Извлечение матрицы агентов |
| `get_incoming_agents()` | Предшественники агента |
| `get_outgoing_agents()` | Последователи агента |

### Конфигурационные классы

| Класс | Описание |
|-------|----------|
| `RunnerConfig` | Конфигурация MACPRunner |
| `LLMConfig` | Конфигурация LLM для агента (мультимодельность) |
| `AgentLLMConfig` | Иммутабельная LLM конфигурация для AgentProfile |
| `RoutingPolicy` | Политики маршрутизации |
| `PruningConfig` | Конфигурация отсечения агентов |
| `MemoryConfig` | Конфигурация системы памяти |
| `TrainingConfig` | Конфигурация обучения GNN |
| `ErrorPolicy` | Политики обработки ошибок |
| `FrameworkSettings` | Глобальные настройки фреймворка |

---

## FAQ

### Почему Pydantic? Какие преимущества это даёт?

MECE Framework полностью построен на **Pydantic 2.0+** для обеспечения типобезопасности, автоматической валидации и удобной сериализации. Основные преимущества:

1. **Автоматическая валидация типов** — ошибки обнаруживаются при создании объектов, а не в runtime
2. **Декларативная типизация** — IDE автодополнение, статическая проверка (mypy, pyright)
3. **Автоматическая сериализация** — `.model_dump()`, `.model_dump_json()` работают из коробки
4. **Значения по умолчанию** — не нужно писать бойлерплейт код
5. **Вложенные модели** — автоматическая валидация вложенных структур
6. **Миграции** — безопасное обновление схем между версиями
7. **Иммутабельность** — `frozen=True` предотвращает случайные изменения

```python
from rustworkx_framework.core import AgentProfile
from pydantic import ValidationError

# ✅ Правильное использование - Pydantic валидирует
agent = AgentProfile(
    identifier="test",
    display_name="Test Agent",
    tools=["tool1", "tool2"],
)

# ❌ Неправильное - Pydantic выбросит ValidationError
try:
    bad_agent = AgentProfile(
        identifier=123,  # Должен быть str, не int
        display_name="Test",
    )
except ValidationError as e:
    print(e.errors())  # Подробная информация об ошибке

# Автоматическая сериализация (Pydantic v2 API)
data = agent.model_dump()  # → dict
json_str = agent.model_dump_json(indent=2)  # → JSON string

# Автоматическая десериализация
loaded = AgentProfile.model_validate(data)
from_json = AgentProfile.model_validate_json(json_str)
```

### Какая версия Pydantic нужна? Совместим ли с Pydantic 1.x?

**MECE Framework требует Pydantic 2.0+ и несовместим с Pydantic 1.x.**

Основные различия в API:
- Pydantic 1.x: `.dict()`, `.parse_obj()`, `.json()`
- Pydantic 2.x: `.model_dump()`, `.model_validate()`, `.model_dump_json()`

Если у вас Pydantic 1.x:
```bash
pip install --upgrade "pydantic>=2.0"
```

Проверка версии:
```python
import pydantic
print(pydantic.VERSION)  # Должно быть >= 2.0.0
```

### Как использовать разные модели для разных агентов?

```python
from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner, LLMCallerFactory

# Способ 1: Через GraphBuilder (рекомендуется)
builder = GraphBuilder()

builder.add_agent(
    "analyst",
    llm_backbone="gpt-4",                 # Сильная модель
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=4000,
)

builder.add_agent(
    "formatter",
    llm_backbone="gpt-4o-mini",           # Слабая модель
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
    temperature=0.3,
    max_tokens=1000,
)

builder.add_workflow_edge("analyst", "formatter")
graph = builder.build()

# Фабрика автоматически создаст callers
factory = LLMCallerFactory.create_openai_factory()
runner = MACPRunner(llm_factory=factory)

result = runner.run_round(graph)
```

### Как интегрировать с OpenAI?

```python
import openai

# Способ 1: Простая интеграция (один LLM для всех)
def openai_caller(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

runner = MACPRunner(llm_caller=openai_caller)

# Способ 2: Мультимодельная интеграция (рекомендуется)
from rustworkx_framework.execution import create_openai_caller

# Автоматически использует openai SDK
runner = MACPRunner(
    llm_factory=LLMCallerFactory.create_openai_factory(
        default_api_key="sk-...",
        default_base_url="https://api.openai.com/v1",
    )
)
```

### Как использовать с локальными моделями (Ollama)?

```python
import requests

def ollama_caller(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt, "stream": False},
    )
    return response.json()["response"]

runner = MACPRunner(llm_caller=ollama_caller)
```

### Как добавить кастомные инструменты?

Инструменты — это просто строки, которые включаются в промпт агента:

```python
agent = AgentProfile(
    identifier="code_executor",
    display_name="Code Executor",
    tools=["python_execute", "file_read", "file_write"],
)
```

Логика инструментов реализуется в вашем LLM-вызове.

### Как визуализировать граф? Какие форматы поддерживаются?

MECE Framework предоставляет мощную систему визуализации с **Pydantic-стилями** и поддержкой множества форматов:

**Поддерживаемые форматы:**
1. **Mermaid** — для GitHub/документации
2. **ASCII art** — для терминала
3. **Graphviz DOT** — для профессиональной визуализации
4. **Rich Console** — цветной вывод в терминал
5. **PNG/SVG/PDF** — рендеринг изображений (требует системный Graphviz)

```python
from rustworkx_framework.core.visualization import (
    GraphVisualizer,
    VisualizationStyle,
    NodeStyle,
    NodeShape,
    MermaidDirection,
    # Convenience functions
    to_mermaid,
    to_ascii,
    print_graph,
    render_to_image,
)

# Быстрая визуализация (convenience functions)
print(to_mermaid(graph, direction=MermaidDirection.LEFT_RIGHT))
print(to_ascii(graph, show_edges=True))
print_graph(graph, format="auto")  # Автоматически выберет colored/ascii

# Продвинутая с кастомными стилями (Pydantic модели)
style = VisualizationStyle(
    direction=MermaidDirection.LEFT_RIGHT,
    agent_style=NodeStyle(
        shape=NodeShape.ROUND,
        fill_color="#e3f2fd",
        stroke_color="#1976d2",
        icon="🤖",
    ),
    show_weights=True,
    show_tools=True,
)

viz = GraphVisualizer(graph, style)
viz.save_mermaid("graph.md", title="My Workflow")
viz.save_dot("graph.dot")

# Рендеринг изображений (требует: pip install graphviz + системный graphviz)
try:
    render_to_image(graph, "output.png", format="png", dpi=150, style=style)
    render_to_image(graph, "output.svg", format="svg", style=style)
    print("✅ Изображения созданы")
except Exception as e:
    print(f"⚠️  Установите системный Graphviz: {e}")
    # Ubuntu: sudo apt install graphviz
    # macOS: brew install graphviz
```

**Установка Graphviz для рендеринга изображений:**
```bash
# Python библиотека
pip install graphviz

# Системный Graphviz
# Ubuntu/Debian:
sudo apt install graphviz

# macOS:
brew install graphviz

# Windows:
winget install graphviz
```

### Как сохранить и загрузить граф?

```python
import json

# Сохранение
data = graph.to_dict()
with open("graph.json", "w") as f:
    json.dump(data, f)

# Загрузка
with open("graph.json", "r") as f:
    data = json.load(f)
graph = RoleGraph.from_dict(data)
```

**Сохранение через Pydantic схемы (рекомендуется):**
```python
from rustworkx_framework.core.schema import GraphSchema

# Создание схемы из графа
schema = GraphSchema(
    name="MyGraph",
    nodes={agent.identifier: AgentNodeSchema.from_profile(agent) for agent in graph.agents},
    edges=[BaseEdgeSchema.from_edge(e) for e in graph.edges],
)

# Сохранение (Pydantic автосериализация)
schema_json = schema.model_dump_json(indent=2)
with open("graph_schema.json", "w") as f:
    f.write(schema_json)

# Загрузка (Pydantic автовалидация)
with open("graph_schema.json", "r") as f:
    loaded_schema = GraphSchema.model_validate_json(f.read())

# Построение графа из схемы
from rustworkx_framework.builder import build_from_schema
graph = build_from_schema(loaded_schema)
```

### Как обрабатывать ошибки агентов?

```python
from rustworkx_framework.execution import RunnerConfig, ErrorPolicy

config = RunnerConfig(
    error_policy=ErrorPolicy(
        on_error="fallback",  # skip, retry, fallback, fail
        max_retries=3,
    ),
    pruning_config=PruningConfig(
        enable_fallback=True,
        max_fallback_attempts=2,
    ),
)

result = runner.run_round(graph)

if result.errors:
    for error in result.errors:
        print(f"Ошибка в {error.agent_id}: {error.message}")
```

### Как отслеживать производительность агентов?

```python
from rustworkx_framework.core.metrics import MetricsTracker

tracker = MetricsTracker()

# Интеграция с runner
runner = MACPRunner(llm_caller=my_llm, metrics_tracker=tracker)
result = runner.run_round(graph)

# Получение метрик
for agent_id in graph.node_ids:
    metrics = tracker.get_node_metrics(agent_id)
    print(f"{agent_id}:")
    print(f"  Reliability: {metrics.reliability:.2%}")
    print(f"  Avg latency: {metrics.avg_latency_ms:.0f}ms")
    print(f"  Quality: {metrics.avg_quality:.2f}")

# Сохранение метрик
tracker.save("metrics.json")
```

### Как использовать динамическую топологию?

```python
# Изменение графа в runtime
graph.add_node(new_agent, connections_to=["existing_agent"])
graph.add_edge("agent1", "new_agent", weight=0.8)

# Удаление неэффективных агентов
if metrics.get_node_metrics("slow_agent").avg_latency_ms > 5000:
    graph.remove_node("slow_agent", policy=StateMigrationPolicy.DISCARD)

# Обновление весов на основе производительности
new_weights = compute_weights_from_metrics(tracker)
graph.update_communication(new_weights)
```

### Как интегрировать с LangChain?

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4")

def langchain_caller(prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

runner = MACPRunner(llm_caller=langchain_caller)
result = runner.run_round(graph)
```

### Как реализовать human-in-the-loop?

```python
from rustworkx_framework.execution import StreamEventType

def human_approval(agent_id: str, response: str) -> bool:
    print(f"\n{agent_id} ответил: {response}")
    approval = input("Одобрить? (y/n): ")
    return approval.lower() == 'y'

def stream_with_approval(graph):
    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            if not human_approval(event.agent_id, event.content):
                # Перезапустить агента с feedback
                feedback = input("Ваш feedback: ")
                # ... логика перезапуска ...
        yield event
```

### Как использовать граф с несколькими задачами?

```python
# Вариант 1: Последовательно
queries = ["Task 1", "Task 2", "Task 3"]

for query in queries:
    graph.query = query
    result = runner.run_round(graph)
    print(f"{query}: {result.final_answer}")

# Вариант 2: Параллельно (async)
async def process_queries(queries):
    tasks = []
    for query in queries:
        graph_copy = copy.deepcopy(graph)
        graph_copy.query = query
        tasks.append(runner.arun_round(graph_copy))

    results = await asyncio.gather(*tasks)
    return results
```

### Как комбинировать облачные и локальные модели?

```python
from rustworkx_framework.builder import GraphBuilder

builder = GraphBuilder()

# Облачная модель для публичных данных
builder.add_agent(
    "public_analyzer",
    llm_backbone="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="$OPENAI_API_KEY",
)

# Локальная модель (Ollama) для конфиденциальных данных
builder.add_agent(
    "private_analyzer",
    llm_backbone="llama3:70b",
    base_url="http://localhost:11434/v1",
    api_key="not-needed",  # Ollama не требует API key
)

builder.add_workflow_edge("public_analyzer", "private_analyzer")
graph = builder.build()

factory = LLMCallerFactory.create_openai_factory()
runner = MACPRunner(llm_factory=factory)
```

### Как оптимизировать затраты на LLM с мультимодельностью?

```python
# Стратегия: дешёвые модели для рутины, дорогие — для сложных задач

builder = GraphBuilder()

# Шаг 1-3: Простые операции → дешёвая модель
for i in range(3):
    builder.add_agent(
        f"processor_{i}",
        llm_backbone="gpt-4o-mini",  # $0.15/$0.60 per 1M tokens
        max_tokens=500,
    )

# Шаг 4: Сложный анализ → дорогая модель
builder.add_agent(
    "analyst",
    llm_backbone="gpt-4",            # $30/$60 per 1M tokens
    max_tokens=2000,
)

# Шаг 5: Финальное форматирование → дешёвая модель
builder.add_agent(
    "formatter",
    llm_backbone="gpt-4o-mini",
    max_tokens=500,
)

# Экономия: ~70-80% от стоимости использования gpt-4 для всех шагов
```

### Как использовать API ключи безопасно?

```python
# ❌ НЕ ДЕЛАЙТЕ ТАК (хардкод ключей)
builder.add_agent("agent", api_key="sk-1234567890...")

# ✅ ПРАВИЛЬНО: Использовать переменные окружения
import os

# Способ 1: Загрузить из .env файла
from dotenv import load_dotenv
load_dotenv()

builder.add_agent("agent", api_key="$OPENAI_API_KEY")

# Способ 2: Явно установить переменную
os.environ["OPENAI_API_KEY"] = open("keys/openai.key").read().strip()
builder.add_agent("agent", api_key="$OPENAI_API_KEY")

# Способ 3: Использовать фабрику с default ключом
factory = LLMCallerFactory.create_openai_factory(
    default_api_key=os.getenv("OPENAI_API_KEY"),
)
```

### Как настроить логирование?

```python
from rustworkx_framework.config import setup_logging

# Настройка глобального логирования
setup_logging(
    level="DEBUG",
    log_file="framework.log",
    rotation="500 MB",
    retention="10 days",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)

# Использование в коде
from rustworkx_framework.config import logger

logger.info("Starting execution")
logger.debug(f"Graph has {graph.num_nodes} nodes")
logger.error("Failed to execute agent", exc_info=True)
```

### Как экспортировать граф для анализа?

```python
# 1. JSON сериализация
import json

graph_data = graph.to_dict()
with open("graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)

# 2. PyTorch Geometric формат
pyg_data = graph.to_pyg_data()
torch.save(pyg_data, "graph.pt")

# 3. NetworkX формат (если нужен)
import networkx as nx

G = nx.DiGraph()
for node_id in graph.node_ids:
    G.add_node(node_id, **graph.get_agent_by_id(node_id).to_dict())

for i, j in zip(*graph.edge_index):
    src = graph.node_ids[i]
    tgt = graph.node_ids[j]
    G.add_edge(src, tgt, weight=graph.A_com[i, j])

nx.write_gexf(G, "graph.gexf")

# 4. CSV экспорт
import pandas as pd

# Nodes
nodes_df = pd.DataFrame([
    {"id": agent.identifier, "name": agent.display_name, "tools": ",".join(agent.tools)}
    for agent in graph.agents
])
nodes_df.to_csv("nodes.csv", index=False)

# Edges
edges = []
for i in range(graph.num_nodes):
    for j in range(graph.num_nodes):
        if graph.A_com[i, j] > 0:
            edges.append({
                "source": graph.node_ids[i],
                "target": graph.node_ids[j],
                "weight": graph.A_com[i, j],
            })
edges_df = pd.DataFrame(edges)
edges_df.to_csv("edges.csv", index=False)
```

### Как тестировать агентов?

```python
import pytest
from unittest.mock import Mock

def test_agent_execution():
    # Мокаем LLM
    mock_llm = Mock(return_value="Mocked response")

    # Создаём граф
    agents = [AgentProfile(identifier="test", display_name="Test Agent")]
    graph = build_property_graph(agents, [], query="Test query")

    # Запускаем
    runner = MACPRunner(llm_caller=mock_llm)
    result = runner.run_round(graph)

    # Проверки
    assert result.final_answer == "Mocked response"
    assert len(result.execution_order) == 1
    assert result.total_tokens >= 0
    mock_llm.assert_called_once()

def test_error_handling():
    # Мокаем LLM с ошибкой
    mock_llm = Mock(side_effect=Exception("LLM error"))

    graph = build_property_graph([agent], [], query="Test")

    config = RunnerConfig(
        max_retries=2,
        error_policy=ErrorPolicy(on_error=ErrorAction.SKIP),
    )
    runner = MACPRunner(llm_caller=mock_llm, config=config)

    result = runner.run_round(graph)

    assert len(result.errors) > 0
    assert result.final_answer is None

def test_parallel_execution():
    agents = [
        AgentProfile(identifier=f"agent_{i}", display_name=f"Agent {i}")
        for i in range(3)
    ]
    edges = [("agent_0", "agent_1"), ("agent_0", "agent_2")]
    graph = build_property_graph(agents, edges, query="Test")

    config = RunnerConfig(enable_parallel=True, max_parallel_size=2)
    runner = MACPRunner(llm_caller=mock_llm, config=config)

    result = runner.run_round(graph)

    assert len(result.execution_order) == 3
```

### Как масштабировать на большие графы?

```python
# 1. Используйте pruning для отсечения неэффективных путей
config = RunnerConfig(
    pruning_config=PruningConfig(
        min_weight_threshold=0.2,
        min_probability_threshold=0.1,
        token_budget=5000,
    ),
)

# 2. Используйте параллельное выполнение
config.enable_parallel = True
config.max_parallel_size = 10

# 3. Используйте beam search для ограничения путей
config.routing_policy = RoutingPolicy.BEAM_SEARCH
scheduler = AdaptiveScheduler(policy=RoutingPolicy.BEAM_SEARCH, beam_width=5)

# 4. Используйте фильтрацию подграфа
from rustworkx_framework.core.algorithms import GraphAlgorithms, SubgraphFilter

algo = GraphAlgorithms(graph)
subgraph = algo.filter_subgraph(SubgraphFilter(
    max_hop_distance=3,
    from_node="start",
    min_edge_weight=0.3,
))

# 5. Используйте async для параллельных запросов
async def process_large_graph(graph):
    results = await runner.arun_round(graph)
    return results
```

---

## Лицензия

СБЕР?

---

## Поддержка

- GitHub Issues: [github.com/yourusername/rustworkx-agent-framework/issues](https://github.com/yourusername/rustworkx-agent-framework/issues)
- Документация: [github.com/yourusername/rustworkx-agent-framework#readme](https://github.com/yourusername/rustworkx-agent-framework#DOCUMENTATION)

---

<p align="center">
  Создано с ❤️ для сообщества разработчиков мультиагентных систем
</p>
