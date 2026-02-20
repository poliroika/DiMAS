# RustworkX Agent Framework

Современный фреймворк для создания мультиагентных систем на основе rustworkx - гибкая и производительная альтернатива LangGraph с динамической топологией, децентрализованной памятью и полным доступом к графовым структурам.

## Почему RustworkX Agent Framework лучше LangGraph

### 1. Динамическая топология
В отличие от LangGraph, где топология фиксирована, наш фреймворк позволяет динамически изменять структуру графа агентов прямо во время выполнения через:
- `RoleGraph.update_communication()` - динамическое обновление рёбер
- Прямой доступ к `rx.PyDiGraph` для добавления/удаления узлов и рёбер

### 2. Децентрализованная память
В отличие от централизованной архитектуры LangGraph, мы реализуем децентрализованный подход:
- `AgentProfile.state` - локальное состояние каждого агента
- Возможность сохранения/восстановления состояний отдельных вершин
- Отделение состояния агента от общего состояния графа

### 3. Граф как First-Class Citizen
LangGraph скрывает графовые структуры от разработчика. Мы предоставляем полный контроль:
- Полный доступ к матрице смежности (`A_com`, `edge_index`)
- Возможность добавлять данные на вершины/рёбра (`edge_attr`, node data)
- Конвертация в PyTorch Geometric для графовых нейросетей

### 4. Альтернативные методы передачи информации
Поддержка не только текстовых сообщений:
- Эмбеддинги хранятся внутри каждого агента (`AgentProfile.embedding`), кодируются через `NodeEncoder`
- Скрытые состояния агентов (`AgentProfile.hidden_state`) для передачи между агентами
- Атрибуты на рёбрах для весов/типов связей
- Готовность к токенам и скрытым представлениям

## Структура файлов

```
rustworkx_framework/
├── README.md                    # Этот файл
├── core/
│   ├── __init__.py
│   ├── graph.py                 # RoleGraph - основной граф агентов
│   ├── agent.py                 # AgentProfile, TaskNode - модели агентов
│   └── encoder.py               # NodeEncoder - кодирование описаний
├── execution/
│   ├── __init__.py
│   ├── scheduler.py             # Планировщик выполнения (топологическая сортировка, SCC)
│   └── runner.py                # MACPRunner - запуск агентов по графу
├── builder/
│   ├── __init__.py
│   └── graph_builder.py         # Построение графа из профилей агентов
├── config/
│   ├── __init__.py
│   ├── settings.py              # Конфигурация через pydantic-settings
│   └── logging.py               # Логирование через loguru
└── utils/
    ├── __init__.py
    └── async_utils.py           # Утилиты для async/sync
```

## Ключевые концепции

### RoleGraph
Основная структура данных - направленный граф на основе rustworkx с:
- Списком агентов (эмбеддинги хранятся внутри каждого `AgentProfile`)
- Матрицей смежности для быстрого доступа
- Атрибутами на рёбрах (вес, тип, дополнительные данные)
- Методами конвертации в разные форматы
- Accessor `embeddings` для сбора эмбеддингов всех агентов в тензор

### AgentProfile
Профиль агента с:
- Идентификатором и описанием
- Списком доступных инструментов
- Эмбеддингом (`embedding`) и скрытым состоянием (`hidden_state`) — хранятся внутри агента
- Локальным состоянием (`state`) — децентрализованная память

### Execution Scheduler
Планировщик выполнения:
- Топологическая сортировка для DAG
- SCC-обработка для графов с циклами
- Поддержка параллельного выполнения независимых агентов

### MACPRunner
Исполнитель Multi-Agent Communication Protocol:
- Выполнение агентов по порядку из графа
- Передача сообщений между связанными агентами
- Асинхронная и синхронная версии

## Зависимости

```
rustworkx>=0.13
pydantic>=2.0
pydantic-settings>=2.0
torch>=2.0
loguru>=0.7
sentence-transformers>=2.0  # опционально для эмбеддингов
```

### Совместимость и версии

- Минимально поддерживаемый Python: **3.12**.
- Тестовая матрица совместимости библиотек:

| rustworkx | torch | transformers | Статус |
| --- | --- | --- | --- |
| 0.13.x | 2.2.x | 4.40.x | ✅ проверено с GNN примером и runner |
| 0.13.x | 2.1.x | 4.38.x | ⚠️ допускается, без ускоренных GNN-фич |
| 0.13.x | 2.0.x | 4.36.x | ⚠️ только базовые алгоритмы графа |

При обновлении rustworkx или torch требуется подтвердить совместимость маршрутизации
и сериализации `RoleGraph`.

## Безопасная конфигурация

Для загрузки настроек используйте `FrameworkSettings`, поддерживающий строгую валидацию
и безопасные ключи через файл:

```bash
export RWXF_API_KEY="sk-..."
export RWXF_BASE_URL="https://api.provider.example"
```

или

```bash
echo "sk-from-vault" > /secure/rwxf.key
export RWXF_API_KEY_FILE=/secure/rwxf.key
```

```python
from rustworkx_framework.config import FrameworkSettings

settings = FrameworkSettings()
llm_key = settings.resolved_api_key  # явная ошибка, если ключа нет
```

Невалидные или пустые ключи блокируют запуск без молчаливого fallback. Поддерживаются
настройки таймаутов, ретраев и параметров логирования через `RWXF_*` переменные.

## Пример использования

```python
from rustworkx_framework.core import RoleGraph, AgentProfile
from rustworkx_framework.execution import MACPRunner, build_execution_order
from rustworkx_framework.builder import build_property_graph

# Создание агентов
agents = [
    AgentProfile(agent_id="solver", display_name="Math Solver", ...),
    AgentProfile(agent_id="checker", display_name="Checker", ...),
]

# Построение графа
graph = build_property_graph(agents, edges=[("solver", "checker")])

# Динамическое изменение топологии
graph.graph.add_edge(node_a, node_b, {"weight": 1.0})

# Получение порядка выполнения
order = build_execution_order(graph.A_com, agent_ids)

# Запуск агентов
runner = MACPRunner(settings)
result = runner.run_round(graph)

# Доступ к графовым данным
adjacency = graph.A_com  # torch.Tensor матрица смежности
edge_index = graph.edge_index  # формат PyG (torch.Tensor)
pyg_data = graph.to_pyg_data()  # PyTorch Geometric Data объект
```

### Расширенные маршруты и память

```python
graph.update_communication(A_com, S_tilde=scores, p_matrix=probabilities)

# Стратифицированная память с активным hidden_state
agent = AgentProfile(...)
agent = agent.with_hidden_state(hidden_state_tensor)
graph.add_node(agent, connections_to=["checker"])

# Онлайновая перепланировка: пересчитать порядок после изменения весов
order = build_execution_order(graph.A_com, graph.role_sequence)
runner.run_round(graph, execution_order=order)
```

Для GNN маршрутизации используйте пример `examples/gnn_routing.py`: он готовит
`edge_index/edge_attr`, запускает PyTorch Geometric модель и обновляет `RoleGraph`
на основе её вывода.

## Расширение для новых методов передачи

Для добавления альтернативных методов передачи информации (токены, скрытые представления):

1. Расширить `AgentProfile` новыми полями для хранения представлений
2. Модифицировать `MACPRunner._format_user_prompt()` для передачи не только текста
3. Добавить атрибуты на рёбра для хранения промежуточных представлений
