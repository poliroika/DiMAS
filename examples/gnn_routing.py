"""
Example: GNN-based routing pipeline.

Demonstrates:
1. Building an agent graph (RoleGraph)
2. Generating node / edge features
3. Simulating execution history and collecting metrics
4. Preparing training data (train / val split)
5. Training a GNN routing model (GCN)
6. Saving and loading model weights
7. Online inference — choosing the next agent via ARGMAX / TOP_K / THRESHOLD
8. Graph algorithms — k-shortest paths, PageRank, betweenness, community detection
9. Adaptive routing suggestions based on observed metrics

Requirements:
    pip install torch torch_geometric rustworkx

Run with:
    python examples/gnn_routing.py
"""

import random
import sys
from pathlib import Path
from typing import Any

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import torch
from torch_geometric.data import Data

from rustworkx_framework.config import logger, setup_logging
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

# Setup framework logging
setup_logging(level="INFO")

# ── Simulation constants ────────────────────────────────────────────────────
SUCCESS_THRESHOLD = 0.1   # execution fails if random() <= this value
LATENCY_MEAN = 100.0      # mean node latency in ms
LATENCY_STD = 30.0        # std-dev of node latency
TOKEN_MIN = 100           # minimum tokens per execution
TOKEN_MAX = 500           # maximum tokens per execution
QUALITY_MEAN = 0.8        # mean quality score on success
QUALITY_STD = 0.1         # std-dev of quality score
EDGE_LATENCY_MEAN = 10.0  # mean edge transition latency in ms
EDGE_LATENCY_STD = 5.0    # std-dev of edge transition latency

DEPS_AVAILABLE = True


# ── Graph construction ──────────────────────────────────────────────────────

def create_demo_graph() -> RoleGraph:
    """Build an 8-node directed agent graph with weighted edges."""
    import rustworkx as rx

    graph = rx.PyDiGraph()

    agents = [
        {"id": "coordinator", "role": "coordinator",  "capability": "planning"},
        {"id": "researcher",  "role": "researcher",   "capability": "search"},
        {"id": "analyst",     "role": "analyst",      "capability": "analysis"},
        {"id": "writer",      "role": "writer",       "capability": "writing"},
        {"id": "reviewer",    "role": "reviewer",     "capability": "review"},
        {"id": "expert_1",    "role": "expert",       "capability": "domain_knowledge"},
        {"id": "expert_2",    "role": "expert",       "capability": "domain_knowledge"},
        {"id": "aggregator",  "role": "aggregator",   "capability": "synthesis"},
    ]

    node_indices: dict[str, int] = {}
    for agent in agents:
        node_indices[agent["id"]] = graph.add_node(agent)

    edges = [
        ("coordinator", "researcher", {"weight": 0.9, "latency": 50,  "reliability": 0.95}),
        ("coordinator", "analyst",    {"weight": 0.8, "latency": 60,  "reliability": 0.90}),
        ("researcher",  "expert_1",   {"weight": 0.7, "latency": 100, "reliability": 0.85}),
        ("researcher",  "expert_2",   {"weight": 0.6, "latency": 120, "reliability": 0.80}),
        ("analyst",     "expert_1",   {"weight": 0.75,"latency": 80,  "reliability": 0.88}),
        ("analyst",     "writer",     {"weight": 0.85,"latency": 70,  "reliability": 0.92}),
        ("expert_1",    "aggregator", {"weight": 0.8, "latency": 90,  "reliability": 0.87}),
        ("expert_2",    "aggregator", {"weight": 0.7, "latency": 110, "reliability": 0.82}),
        ("writer",      "reviewer",   {"weight": 0.95,"latency": 40,  "reliability": 0.98}),
        ("aggregator",  "reviewer",   {"weight": 0.9, "latency": 55,  "reliability": 0.93}),
        ("reviewer",    "coordinator",{"weight": 0.5, "latency": 30,  "reliability": 0.99}),
    ]

    for src, tgt, data in edges:
        graph.add_edge(node_indices[src], node_indices[tgt], data)

    return RoleGraph(
        node_ids=[a["id"] for a in agents],
        role_connections={a["id"]: [] for a in agents},
        graph=graph,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def simulate_execution_history(
    graph: RoleGraph,
    tracker: MetricsTracker,
    num_runs: int = 100,
) -> None:
    """Simulate `num_runs` random walks through the graph to populate metrics."""
    for _run in range(num_runs):
        current = "coordinator"

        for _ in range(5):
            # Collect successors of the current node
            successors: list[str] = []
            for idx in graph.graph.node_indices():
                data = graph.graph.get_node_data(idx)
                if data["id"] == current:
                    for succ_idx in graph.graph.successors(idx):
                        succ_data = graph.graph.get_node_data(succ_idx)
                        successors.append(succ_data["id"])
                    break

            if not successors:
                break

            next_agent = random.choice(successors)  # noqa: S311

            success = random.random() > SUCCESS_THRESHOLD  # noqa: S311
            latency = random.gauss(LATENCY_MEAN, LATENCY_STD)  # noqa: S311
            tokens = random.randint(TOKEN_MIN, TOKEN_MAX)  # noqa: S311
            quality = random.gauss(QUALITY_MEAN, QUALITY_STD) if success else 0.0  # noqa: S311

            tracker.record_node_execution(
                node_id=current,
                success=success,
                latency_ms=max(0.0, latency),
                cost_tokens=tokens,
                quality=max(0.0, min(1.0, quality)),
            )
            tracker.record_edge_transition(
                source_id=current,
                target_id=next_agent,
                success=success,
                latency_ms=random.gauss(EDGE_LATENCY_MEAN, EDGE_LATENCY_STD),  # noqa: S311
            )

            current = next_agent


# ── Training data preparation ───────────────────────────────────────────────

def prepare_training_data(
    graph: RoleGraph,
    tracker: MetricsTracker,
    num_samples: int = 200,
) -> tuple[list[Any], list[Any]]:
    """Generate augmented training samples by adding Gaussian noise to node features."""
    if not DEPS_AVAILABLE:
        return [], []

    feature_gen = DefaultFeatureGenerator()
    node_ids = graph.node_ids
    node_features = feature_gen.generate_node_features(graph, node_ids, tracker)
    edge_index = graph.edge_index
    node_scores = tracker.get_node_scores()

    # Binary label: 1 if the node's score is at or above the median
    median_score = (
        torch.median(torch.tensor(list(node_scores.values()))).item()
        if node_scores
        else 0.5
    )
    labels = [1 if node_scores.get(nid, 0.5) >= median_score else 0 for nid in node_ids]

    train_data: list[Any] = []
    val_data: list[Any] = []
    split = int(num_samples * 0.8)

    for i in range(num_samples):
        noise = torch.randn_like(node_features) * 0.1
        data = Data(
            x=(node_features + noise).clone().to(dtype=torch.float32),
            edge_index=edge_index.clone().to(dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
        )
        (train_data if i < split else val_data).append(data)

    print(f"  Training samples : {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Feature dimension : {node_features.shape[1]}")
    return train_data, val_data


# ── Model training ──────────────────────────────────────────────────────────

def train_router_model(
    train_data: list[Any],
    val_data: list[Any],
    in_channels: int,
    model_path: Path,
) -> Any:
    """Train a 2-class GCN node classifier and save weights to *model_path*."""
    if not DEPS_AVAILABLE or not train_data:
        return None

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

    trainer = GNNTrainer(model, config)
    trainer.train(train_data, val_data, verbose=True)
    trainer.save(model_path)
    print(f"  Model saved → {model_path}")
    return model


# ── Online inference ────────────────────────────────────────────────────────

def demo_online_inference(
    graph: RoleGraph,
    model: Any,
    tracker: MetricsTracker,
) -> None:
    """Show three routing strategies: ARGMAX, TOP_K, and THRESHOLD."""
    if not DEPS_AVAILABLE or model is None:
        return

    router = GNNRouterInference(model, DefaultFeatureGenerator())

    print("\n  Strategy: ARGMAX (pick highest-scoring candidate)")
    result = router.predict(
        graph,
        source="coordinator",
        candidates=["researcher", "analyst"],
        metrics_tracker=tracker,
        strategy=RoutingStrategy.ARGMAX,
    )
    print(f"    → selected: {result}")

    print("\n  Strategy: TOP_K (top-3 nodes globally)")
    result = router.predict(
        graph,
        metrics_tracker=tracker,
        strategy=RoutingStrategy.TOP_K,
        top_k=3,
    )
    print(f"    → selected: {result}")

    print("\n  Strategy: THRESHOLD (score ≥ 0.1)")
    result = router.predict(
        graph,
        metrics_tracker=tracker,
        strategy=RoutingStrategy.THRESHOLD,
        threshold=0.1,
    )
    print(f"    → selected: {result}")


# ── Graph algorithms ────────────────────────────────────────────────────────

def demo_graph_algorithms(graph: RoleGraph) -> None:
    """Show k-shortest paths, PageRank, betweenness centrality and communities."""
    alg = GraphAlgorithms(graph)

    print("\n  k-Shortest paths (coordinator → reviewer, k=3):")
    paths = alg.k_shortest_paths("coordinator", "reviewer", k=3)
    for i, path in enumerate(paths, 1):
        print(f"    {i}. {' → '.join(path)}")

    print("\n  PageRank — top-3 nodes:")
    pr = alg.compute_centrality(CentralityType.PAGERANK)
    for node, score in pr.top_k(3):
        print(f"    {node:<15} {score:.4f}")

    print("\n  Betweenness centrality — top-3 nodes:")
    bc = alg.compute_centrality(CentralityType.BETWEENNESS)
    for node, score in bc.top_k(3):
        print(f"    {node:<15} {score:.4f}")

    communities = alg.detect_communities()
    print(f"\n  Communities detected: {len(communities.communities)}")
    for i, comm in enumerate(communities.communities):
        print(f"    community {i + 1}: {comm}")

    cycles = alg.detect_cycles(max_length=5)
    if cycles:
        print(f"\n  Cycles found (showing up to 3):")
        for cycle in cycles[:3]:
            print(f"    {cycle}")
    else:
        print("\n  No cycles detected.")


# ── Adaptive routing suggestions ────────────────────────────────────────────

def demo_adaptive_routing_integration(
    graph: RoleGraph,
    tracker: MetricsTracker,
) -> None:
    """Print routing weight suggestions and pruning recommendations."""
    weights = tracker.get_routing_weights()

    print("\n  Bottom-5 edge weights (candidates for pruning):")
    for (src, tgt), w in sorted(weights.items(), key=lambda x: x[1])[:5]:
        in_graph = src in graph.node_ids and tgt in graph.node_ids
        print(f"    {src} → {tgt}  weight={w:.4f}  {'(in graph)' if in_graph else '(stale)'}")

    suggestions = tracker.suggest_pruning(
        node_reliability_threshold=0.5,
        max_latency_ms=200,
    )

    if suggestions["prune_nodes"]:
        print(f"\n  Nodes to prune (low reliability): {suggestions['prune_nodes']}")
    if suggestions["slow_nodes"]:
        print(f"  Slow nodes (high latency): {suggestions['slow_nodes']}")
    if suggestions["prune_edges"]:
        print(f"  Edges to prune: {suggestions['prune_edges']}")
    if not any(suggestions.values()):
        print("\n  Graph looks healthy — no pruning suggested.")

    snapshot = tracker.to_dict()
    print(f"\n  Metrics snapshot keys: {list(snapshot.keys())}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    model_path = Path(__file__).parent / "gnn_router_model.pt"

    print("─" * 50)
    print("Step 1 / 6 — Build agent graph")
    graph = create_demo_graph()
    print(f"  Nodes: {graph.node_ids}")

    print("\n─" * 50)
    print("Step 2 / 6 — Simulate execution history (100 runs)")
    tracker = MetricsTracker()
    simulate_execution_history(graph, tracker, num_runs=100)
    scores = tracker.get_node_scores()
    for node, score in scores.items():
        print(f"  {node:<15} score={score:.4f}")

    print("\n─" * 50)
    print("Step 3 / 6 — Graph algorithms")
    demo_graph_algorithms(graph)

    print("\n─" * 50)
    print("Step 4 / 6 — Prepare training data (200 augmented samples)")
    train_data, val_data = prepare_training_data(graph, tracker, num_samples=200)

    print("\n─" * 50)
    print("Step 5 / 6 — Train GNN routing model")
    in_channels = train_data[0].x.shape[1] if train_data else 4
    model = train_router_model(train_data, val_data, in_channels, model_path)

    print("\n─" * 50)
    print("Step 6 / 6 — Online inference & adaptive routing")
    demo_online_inference(graph, model, tracker)
    demo_adaptive_routing_integration(graph, tracker)

    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        print(f"\n  Saved model size: {size_kb:.1f} KB")

    print("\nDone ✅")


if __name__ == "__main__":
    main()
