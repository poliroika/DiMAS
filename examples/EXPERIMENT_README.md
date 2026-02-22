# GNN Routing Experiment — Инструкция по запуску

## Быстрый старт

### 1. Установка зависимостей

```powershell
# Перейдите в директорию проекта
cd C:\Users\poliroika\Documents\MECE

# Установите необходимые зависимости
python -m pip install torch_geometric semver
```

### 2. Запуск эксперимента

```powershell
# Запустите эксперимент
python -m examples.gnn_routing_experiment
```

Эксперимент выполнит 8 шагов:
- Построение графа агентов
- Сбор реальных метрик (300 задач)
- Анализ графа
- Визуализация ДО обучения
- Обучение GNN моделей (GCN, GAT, GraphSAGE)
- Сравнение стратегий маршрутизации
- Визуализация ПОСЛЕ обучения
- Генерация JSON-лога

### 3. Просмотр результатов

После завершения откройте в браузере файлы из `examples/experiment_output/`:

```powershell
# Откройте папку с результатами
explorer examples\experiment_output
```

**Рекомендуемые файлы для просмотра:**
- `graph_before.html` — интерактивный граф ДО обучения
- `graph_after.html` — интерактивный граф ПОСЛЕ обучения
- `training_curves.html` — кривые обучения
- `routing_comparison.html` — сравнение стратегий
- `experiment_log.json` — полный структурированный лог

### 4. Инструкции по созданию видео/веб-ресурса

```powershell
# Показать инструкции
python -m examples.gnn_routing_experiment --help-video
```

## Ожидаемое время выполнения

- Сбор метрик: ~1-2 секунды
- Обучение GNN: ~15-20 секунд (3 модели)
- Сравнение стратегий: ~2-3 секунды
- **Итого: ~20-30 секунд**

## Структура результатов

```
examples/experiment_output/
├── experiment_log.json          # Полный лог эксперимента (35 KB)
├── gnn_router_model.pt          # Обученная модель (80 KB)
├── graph_before.html            # Граф ДО обучения (6.7 KB)
├── graph_after.html             # Граф ПОСЛЕ обучения (6.7 KB)
├── training_curves.html         # Кривые обучения (4.9 KB)
├── routing_comparison.html      # Сравнение стратегий (3.3 KB)
├── graph_before.md / .dot       # Статичные форматы
├── graph_after.md / .dot       # Статичные форматы
└── animation_frames/            # Кадры для анимации
```

## Устранение проблем

### Ошибка: `ModuleNotFoundError: No module named 'torch_geometric'`
```powershell
python -m pip install torch_geometric
```

### Ошибка: `ModuleNotFoundError: No module named 'semver'`
```powershell
python -m pip install semver
```

### Ошибка: Unicode в консоли Windows
Скрипт автоматически исправляет кодировку. Если проблемы остаются:
```powershell
$env:PYTHONIOENCODING="utf-8"
python -m examples.gnn_routing_experiment
```

## Для публикации результатов

1. **Веб-ресурс**: Загрузите HTML-файлы на GitHub Pages или любой статический хостинг
2. **Видео**: Следуйте инструкциям из `--help-video`
3. **JSON-лог**: Используйте `experiment_log.json` для воспроизведения результатов
