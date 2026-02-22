"""
GNN-based routing pipeline — no LLM required.

Demonstrates:
  1. Building an 8-node directed agent graph
  2. Simulating execution history and collecting metrics
  3. Graph algorithms (k-shortest paths, PageRank, betweenness, communities)
  4. Preparing training data (train / val split)
  5. Training a GCN routing model
  6. Online inference (ARGMAX / TOP_K / THRESHOLD)
  7. Adaptive routing suggestions based on observed metrics

Requirements:
    pip install torch torch_geometric rustworkx

Run:
    python -m examples.gnn_routing
"""

import random
from pathlib import Path
from typing import Any

import rustworkx as rx
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

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "gnn_router_model.pt"

# Simulation parameters
_FAIL_RATE = 0.1
_LATENCY = (100.0, 30.0)  # mean, std
_TOKENS = (100, 500)  # min, max
_QUALITY = (0.8, 0.1)  # mean, std
_EDGE_LATENCY = (10.0, 5.0)  # mean, std


# ── Helpers ──────────────────────────────────────────────────────────────────


def _header(step: int, total: int, title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Step {step}/{total} — {title}")
    print("─" * 50)


# ── 1. Graph construction ───────────────────────────────────────────────────


def create_demo_graph() -> RoleGraph:
    """Build an 8-node directed agent graph with weighted edges."""
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

    idx: dict[str, int] = {}
    for a in agents:
        idx[a["id"]] = graph.add_node(a)

    edges = [
        ("coordinator", "researcher", 0.9, 50, 0.95),
        ("coordinator", "analyst", 0.8, 60, 0.90),
        ("researcher", "expert_1", 0.7, 100, 0.85),
        ("researcher", "expert_2", 0.6, 120, 0.80),
        ("analyst", "expert_1", 0.75, 80, 0.88),
        ("analyst", "writer", 0.85, 70, 0.92),
        ("expert_1", "aggregator", 0.8, 90, 0.87),
        ("expert_2", "aggregator", 0.7, 110, 0.82),
        ("writer", "reviewer", 0.95, 40, 0.98),
        ("aggregator", "reviewer", 0.9, 55, 0.93),
        ("reviewer", "coordinator", 0.5, 30, 0.99),
    ]
    for src, tgt, w, lat, rel in edges:
        graph.add_edge(idx[src], idx[tgt], {"weight": w, "latency": lat, "reliability": rel})

    return RoleGraph(
        node_ids=[a["id"] for a in agents],
        role_connections={a["id"]: [] for a in agents},
        graph=graph,
    )


# ── 2. Simulation ───────────────────────────────────────────────────────────


def simulate_history(graph: RoleGraph, tracker: MetricsTracker, runs: int = 100) -> None:
    """Random-walk simulation to populate the metrics tracker."""
    for _ in range(runs):
        current = "coordinator"
        for _ in range(5):
            successors: list[str] = []
            for i in graph.graph.node_indices():
                if graph.graph.get_node_data(i)["id"] == current:
                    successors = [graph.graph.get_node_data(s)["id"] for s in graph.graph.successors(i)]
                    break
            if not successors:
                break

            nxt = random.choice(successors)
            ok = random.random() > _FAIL_RATE

            tracker.record_node_execution(
                node_id=current,
                success=ok,
                latency_ms=max(0.0, random.gauss(*_LATENCY)),
                cost_tokens=random.randint(*_TOKENS),
                quality=max(0.0, min(1.0, random.gauss(*_QUALITY))) if ok else 0.0,
            )
            tracker.record_edge_transition(
                source_id=current,
                target_id=nxt,
                success=ok,
                latency_ms=random.gauss(*_EDGE_LATENCY),
            )
            current = nxt


# ── 3. Graph algorithms ─────────────────────────────────────────────────────


def demo_algorithms(graph: RoleGraph) -> None:
    alg = GraphAlgorithms(graph)

    print("  k-Shortest paths (coordinator → reviewer, k=3):")
    for i, path in enumerate(alg.k_shortest_paths("coordinator", "reviewer", k=3), 1):
        print(f"    {i}. {' → '.join(path)}")

    print("\n  PageRank — top 3:")
    for node, score in alg.compute_centrality(CentralityType.PAGERANK).top_k(3):
        print(f"    {node:<15} {score:.4f}")

    print("\n  Betweenness — top 3:")
    for node, score in alg.compute_centrality(CentralityType.BETWEENNESS).top_k(3):
        print(f"    {node:<15} {score:.4f}")

    communities = alg.detect_communities()
    print(f"\n  Communities: {len(communities.communities)}")
    for i, c in enumerate(communities.communities):
        print(f"    {i + 1}: {c}")

    cycles = alg.detect_cycles(max_length=5)
    if cycles:
        print("\n  Cycles (up to 3):")
        for cyc in cycles[:3]:
            print(f"    {cyc}")
    else:
        print("\n  No cycles detected.")


# ── 4. Training data ────────────────────────────────────────────────────────


def prepare_data(
    graph: RoleGraph,
    tracker: MetricsTracker,
    n: int = 200,
) -> tuple[list[Any], list[Any]]:
    """Generate augmented training samples with Gaussian noise."""
    feat_gen = DefaultFeatureGenerator()
    node_ids = graph.node_ids
    features = feat_gen.generate_node_features(graph, node_ids, tracker)
    edge_index = graph.edge_index
    scores = tracker.get_node_scores()

    median = torch.median(torch.tensor(list(scores.values()))).item() if scores else 0.5
    labels = [1 if scores.get(nid, 0.5) >= median else 0 for nid in node_ids]

    split = int(n * 0.8)
    train, val = [], []
    for i in range(n):
        d = Data(
            x=(features + torch.randn_like(features) * 0.1).to(torch.float32),
            edge_index=edge_index.clone().to(torch.long),
            y=torch.tensor(labels, dtype=torch.long),
        )
        (train if i < split else val).append(d)

    print(f"  Train: {len(train)}  Val: {len(val)}  Features: {features.shape[1]}")
    return train, val


# ── 5. Training ─────────────────────────────────────────────────────────────


def train_model(train_data: list[Any], val_data: list[Any], in_ch: int) -> Any:
    cfg = TrainingConfig(
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
    model = create_gnn_router(GNNModelType.GCN, in_ch, cfg.num_classes, cfg)
    trainer = GNNTrainer(model, cfg)
    trainer.train(train_data, val_data, verbose=True)
    trainer.save(MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")
    return model


# ── 6. Inference ────────────────────────────────────────────────────────────


def demo_inference(graph: RoleGraph, model: Any, tracker: MetricsTracker) -> None:
    router = GNNRouterInference(model, DefaultFeatureGenerator())

    for strategy, kwargs, label in [
        (RoutingStrategy.ARGMAX, {"candidates": ["researcher", "analyst"]}, "ARGMAX"),
        (RoutingStrategy.TOP_K, {"top_k": 3}, "TOP_K (3)"),
        (RoutingStrategy.THRESHOLD, {"threshold": 0.1}, "THRESHOLD (≥0.1)"),
    ]:
        result = router.predict(
            graph,
            source="coordinator",
            metrics_tracker=tracker,
            strategy=strategy,
            **kwargs,
        )
        print(f"  {label:<20} → {result}")


# ── 7. Adaptive suggestions ────────────────────────────────────────────────


def demo_adaptive(graph: RoleGraph, tracker: MetricsTracker) -> None:
    weights = tracker.get_routing_weights()
    print("  Bottom-5 edge weights:")
    for (s, t), w in sorted(weights.items(), key=lambda x: x[1])[:5]:
        tag = "(in graph)" if s in graph.node_ids and t in graph.node_ids else "(stale)"
        print(f"    {s} → {t}  w={w:.4f}  {tag}")

    suggestions = tracker.suggest_pruning(node_reliability_threshold=0.5, max_latency_ms=200)
    if suggestions["prune_nodes"]:
        print(f"\n  Prune nodes: {suggestions['prune_nodes']}")
    if suggestions["slow_nodes"]:
        print(f"  Slow nodes : {suggestions['slow_nodes']}")
    if suggestions["prune_edges"]:
        print(f"  Prune edges: {suggestions['prune_edges']}")
    if not any(suggestions.values()):
        print("\n  Graph is healthy — no pruning suggested.")


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    _header(1, 6, "Build agent graph")
    graph = create_demo_graph()
    print(f"  Nodes: {graph.node_ids}")

    _header(2, 6, "Simulate execution history (100 runs)")
    tracker = MetricsTracker()
    simulate_history(graph, tracker, runs=100)
    for node, score in tracker.get_node_scores().items():
        print(f"  {node:<15} score={score:.4f}")

    _header(3, 6, "Graph algorithms")
    demo_algorithms(graph)

    _header(4, 6, "Prepare training data")
    train_data, val_data = prepare_data(graph, tracker)

    _header(5, 6, "Train GNN routing model")
    in_ch = train_data[0].x.shape[1] if train_data else 4
    model = train_model(train_data, val_data, in_ch)

    _header(6, 6, "Online inference & adaptive routing")
    demo_inference(graph, model, tracker)
    demo_adaptive(graph, tracker)

    if MODEL_PATH.exists():
        print(f"\n  Model size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
