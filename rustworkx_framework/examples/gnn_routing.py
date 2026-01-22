"""Пример пайплайна GNN-маршрутизации.

Демонстрирует:
1. Создание графа агентов
2. Генерацию признаков узлов/рёбер
3. Подготовку данных для обучения
4. Обучение модели маршрутизации
5. Сохранение/загрузка весов
6. Онлайн-инференс для выбора маршрутов

Требования:
    pip install torch torch_geometric rustworkx

Запуск:
    python examples/gnn_routing.py
"""

import random
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from rustworkx_framework.core.algorithms import CentralityType, GraphAlgorithms
from rustworkx_framework.core.gnn import (
    DefaultFeatureGenerator,
    GNNModelType,
    GNNRouterInference,
    GNNTrainer,
    RoutingStrategy,
    TrainingConfig,
    create_gnn_router,
)
from rustworkx_framework.core.graph import RoleGraph
from rustworkx_framework.core.metrics import MetricsTracker

DEPS_AVAILABLE = True


def create_demo_graph() -> RoleGraph:
    import rustworkx as rx

    graph = rx.PyDiGraph()

    agents = [
        {"id": "coordinator", "role": "coordinator", "capability": "planning"},
        {"id": "researcher", "role": "researcher", "capability": "search"},
        {"id": "analyst", "role": "analyst", "capability": "analysis"},
        {"id": "writer", "role": "writer", "capability": "writing"},
        {"id": "reviewer", "role": "reviewer", "capability": "review"},
        {"id": "expert_1", "role": "expert", "capability": "domain_knowledge"},
        {"id": "expert_2", "role": "expert", "capability": "domain_knowledge"},
        {"id": "aggregator", "role": "aggregator", "capability": "synthesis"},
    ]

    node_indices = {}
    for agent in agents:
        idx = graph.add_node(agent)
        node_indices[agent["id"]] = idx

    edges = [
        ("coordinator", "researcher", {"weight": 0.9, "latency": 50, "reliability": 0.95}),
        ("coordinator", "analyst", {"weight": 0.8, "latency": 60, "reliability": 0.90}),
        ("researcher", "expert_1", {"weight": 0.7, "latency": 100, "reliability": 0.85}),
        ("researcher", "expert_2", {"weight": 0.6, "latency": 120, "reliability": 0.80}),
        ("analyst", "expert_1", {"weight": 0.75, "latency": 80, "reliability": 0.88}),
        ("analyst", "writer", {"weight": 0.85, "latency": 70, "reliability": 0.92}),
        ("expert_1", "aggregator", {"weight": 0.8, "latency": 90, "reliability": 0.87}),
        ("expert_2", "aggregator", {"weight": 0.7, "latency": 110, "reliability": 0.82}),
        ("writer", "reviewer", {"weight": 0.95, "latency": 40, "reliability": 0.98}),
        ("aggregator", "reviewer", {"weight": 0.9, "latency": 55, "reliability": 0.93}),
        (
            "reviewer",
            "coordinator",
            {"weight": 0.5, "latency": 30, "reliability": 0.99},
        ),
    ]

    for src, tgt, data in edges:
        graph.add_edge(node_indices[src], node_indices[tgt], data)

    role_graph = RoleGraph(
        node_ids=[a["id"] for a in agents],
        role_connections={a["id"]: [] for a in agents},
        graph=graph,
    )

    return role_graph


def simulate_execution_history(
    graph: RoleGraph,
    tracker: MetricsTracker,
    num_runs: int = 100,
) -> None:
    print(f"\n📊 Симуляция {num_runs} запусков...")

    for run in range(num_runs):
        current = "coordinator"
        visited = [current]

        for _ in range(5):
            successors = []
            for idx in graph.graph.node_indices():
                data = graph.graph.get_node_data(idx)
                if data["id"] == current:
                    for succ_idx in graph.graph.successors(idx):
                        succ_data = graph.graph.get_node_data(succ_idx)
                        successors.append(succ_data["id"])
                    break

            if not successors:
                break

            next_agent = random.choice(successors)

            success = random.random() > 0.1
            latency = random.gauss(100, 30)
            tokens = random.randint(100, 500)
            quality = random.gauss(0.8, 0.1) if success else 0.0

            tracker.record_node_execution(
                node_id=current,
                success=success,
                latency_ms=max(0, latency),
                cost_tokens=tokens,
                quality=max(0, min(1, quality)),
            )

            tracker.record_edge_transition(
                source_id=current,
                target_id=next_agent,
                success=success,
                latency_ms=random.gauss(10, 5),
            )

            visited.append(next_agent)
            current = next_agent

    print(f"   ✓ Собраны метрики для {len(tracker.get_all_node_metrics())} узлов")


def prepare_training_data(
    graph: RoleGraph,
    tracker: MetricsTracker,
    num_samples: int = 200,
) -> tuple[list[Any], list[Any]]:
    if not DEPS_AVAILABLE:
        print("⚠️  PyTorch/PyG не доступны - пропуск подготовки данных")
        return [], []

    print(f"\n📦 Подготовка {num_samples} обучающих примеров...")

    feature_gen = DefaultFeatureGenerator()
    node_ids = graph.node_ids

    node_features = feature_gen.generate_node_features(graph, node_ids, tracker)

    edge_index = graph.edge_index

    node_scores = tracker.get_node_scores()

    median_score = (
        torch.median(torch.tensor(list(node_scores.values()))).item() if node_scores else 0.5
    )
    labels = []
    for nid in node_ids:
        score = node_scores.get(nid, 0.5)
        labels.append(1 if score >= median_score else 0)

    train_data = []
    val_data = []

    for i in range(num_samples):
        noise = torch.randn(*node_features.shape) * 0.1
        noisy_features = node_features + noise

        data = Data(
            x=torch.tensor(noisy_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
        )

        if i < int(num_samples * 0.8):
            train_data.append(data)
        else:
            val_data.append(data)

    print(f"   ✓ Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"   ✓ Node features: {node_features.shape}")

    return train_data, val_data


def train_router_model(
    train_data: list[Any],
    val_data: list[Any],
    in_channels: int,
    model_path: Path,
) -> Any:
    if not DEPS_AVAILABLE or not train_data:
        print("⚠️  Обучение пропущено (нет зависимостей или данных)")
        return None

    print("\n🧠 Обучение модели...")

    config = TrainingConfig(
        learning_rate=1e-3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        epochs=50,
        batch_size=16,
        patience=10,
        task="node_classification",
        num_classes=2,
    )

    model = create_gnn_router(
        model_type=GNNModelType.GCN,
        in_channels=in_channels,
        out_channels=config.num_classes,
        config=config,
    )

    print(f"   Model: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = GNNTrainer(model, config)
    result = trainer.train(train_data, val_data, verbose=True)

    print(f"\n   ✓ Best epoch: {result.best_epoch}")
    print(f"   ✓ Best val loss: {result.best_val_loss:.4f}")

    trainer.save(model_path)
    print(f"   ✓ Модель сохранена: {model_path}")

    return model


def demo_online_inference(
    graph: RoleGraph,
    model: Any,
    tracker: MetricsTracker,
) -> None:
    if not DEPS_AVAILABLE or model is None:
        print("⚠️  Инференс пропущен")
        return

    print("\n🔮 Онлайн-инференс...")

    router = GNNRouterInference(model, DefaultFeatureGenerator())

    print("\n   Сценарий 1: После 'coordinator' выбрать следующего")
    pred = router.predict(
        graph,
        source="coordinator",
        candidates=["researcher", "analyst"],
        metrics_tracker=tracker,
        strategy=RoutingStrategy.ARGMAX,
    )
    print(f"   → Рекомендация: {pred.recommended_nodes}")
    print(f"   → Confidence: {pred.confidence:.3f}")
    print(f"   → Scores: {pred.scores}")

    print("\n   Сценарий 2: Top-3 агента для задачи")
    pred = router.predict(
        graph,
        metrics_tracker=tracker,
        strategy=RoutingStrategy.TOP_K,
        top_k=3,
    )
    print(f"   → Top-3: {pred.recommended_nodes}")

    print("\n   Сценарий 3: Агенты с confidence > 0.1")
    pred = router.predict(
        graph,
        metrics_tracker=tracker,
        strategy=RoutingStrategy.THRESHOLD,
        threshold=0.1,
    )
    print(f"   → Подходящие: {pred.recommended_nodes}")


def demo_graph_algorithms(graph: RoleGraph) -> None:
    print("\n📐 Графовые алгоритмы...")

    alg = GraphAlgorithms(graph)

    print("\n   K-shortest paths (coordinator → reviewer):")
    paths = alg.k_shortest_paths("coordinator", "reviewer", k=3)
    for i, path in enumerate(paths, 1):
        print(f"   {i}. {' → '.join(path.nodes)} (weight: {path.total_weight:.2f})")

    print("\n   PageRank централность (top-3):")
    pr = alg.compute_centrality(CentralityType.PAGERANK)
    for node, score in pr.top_k(3):
        print(f"   • {node}: {score:.4f}")

    print("\n   Betweenness централность (top-3):")
    bc = alg.compute_centrality(CentralityType.BETWEENNESS)
    for node, score in bc.top_k(3):
        print(f"   • {node}: {score:.4f}")

    print("\n   Обнаружение сообществ:")
    communities = alg.detect_communities()
    print(f"   Найдено {communities.num_communities} сообществ:")
    for i, comm in enumerate(communities.communities):
        print(f"   {i + 1}. {comm}")

    print("\n   Детекция циклов:")
    cycles = alg.detect_cycles(max_length=5)
    if cycles:
        for cycle in cycles[:3]:
            print(f"   • {' → '.join(cycle.nodes)} → {cycle.nodes[0]}")
    else:
        print("   Циклов не найдено")

    print(f"\n   Граф является DAG: {alg.is_dag()}")


def demo_adaptive_routing_integration(
    graph: RoleGraph,
    tracker: MetricsTracker,
) -> None:
    print("\n🔄 Интеграция с адаптивной маршрутизацией...")

    weights = tracker.get_routing_weights()
    print(f"\n   Веса рёбер для маршрутизации ({len(weights)} рёбер):")
    for (src, tgt), w in sorted(weights.items(), key=lambda x: x[1])[:5]:
        print(f"   • {src} → {tgt}: {w:.4f}")

    print("\n   Рекомендации по pruning:")
    suggestions = tracker.suggest_pruning(
        node_reliability_threshold=0.5,
        max_latency_ms=200,
    )
    if suggestions["prune_nodes"]:
        print(f"   • Исключить узлы: {suggestions['prune_nodes']}")
    if suggestions["slow_nodes"]:
        print(f"   • Медленные узлы: {suggestions['slow_nodes']}")
    if suggestions["prune_edges"]:
        print(f"   • Исключить рёбра: {suggestions['prune_edges']}")
    if not any(suggestions.values()):
        print("   • Все узлы и рёбра в норме")

    print("\n   Экспорт метрик в JSON:")
    metrics_dict = tracker.to_dict()
    print(f"   • Узлов с метриками: {len(metrics_dict['nodes'])}")
    print(f"   • Рёбер с метриками: {len(metrics_dict['edges'])}")
    print(f"   • Глобальная avg_latency: {metrics_dict['global']['avg_latency_ms']:.1f}ms")


def main():
    print("=" * 60)
    print("🚀 MECE Framework: GNN Routing Pipeline Example")
    print("=" * 60)

    model_path = Path(__file__).parent / "gnn_router_model.pt"

    graph = create_demo_graph()
    print(f"\n✓ Граф создан: {graph.num_nodes} узлов, {graph.num_edges} рёбер")

    tracker = MetricsTracker()

    simulate_execution_history(graph, tracker, num_runs=100)

    demo_graph_algorithms(graph)

    train_data, val_data = prepare_training_data(graph, tracker, num_samples=200)

    in_channels = train_data[0].x.shape[1] if train_data else 4
    model = train_router_model(train_data, val_data, in_channels, model_path)

    demo_online_inference(graph, model, tracker)

    demo_adaptive_routing_integration(graph, tracker)

    print("\n" + "=" * 60)
    print("✅ Пример завершён!")
    print("=" * 60)

    if model_path.exists():
        print(f"\n📁 Сохранённая модель: {model_path}")
        print("   Загрузка: GNNRouterInference.load(path)")


if __name__ == "__main__":
    main()
