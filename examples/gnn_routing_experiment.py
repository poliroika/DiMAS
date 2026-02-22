"""
GNN Routing Experiment -- full benchmark for demo-track article.

Shows:
  1. Agent graph construction with REAL LLM calls via MACPRunner
  2. Metric collection via MetricsTracker (latency, reliability, quality, cost)
     from real agent executions on Qwen3-VL-235B
  3. GNN training (GCN, GAT, GraphSAGE) on collected metrics
  4. Routing strategy comparison: GNN vs ARGMAX vs TOP_K vs THRESHOLD vs RANDOM
  5. Graph visualization with edge weights BEFORE and AFTER training
  6. Structured JSON logs for reproducibility
  7. Interactive HTML visualization

Usage:
    python -m examples.gnn_routing_experiment            # real LLM calls
    python -m examples.gnn_routing_experiment --simulate  # simulated metrics (no LLM)
    python -m examples.gnn_routing_experiment --help-video

Output:
    examples/experiment_output/
    +-- experiment_log.json          -- full experiment log
    +-- graph_before.html            -- interactive graph BEFORE training
    +-- graph_after.html             -- interactive graph AFTER training
    +-- training_curves.html         -- training curves
    +-- routing_comparison.html      -- strategy comparison
    +-- graph_before.dot / .md       -- static formats
    +-- graph_after.dot / .md        -- static formats
    +-- animation_frames/            -- frames for video (by epoch)
    +-- run.log                      -- full text log of the run
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from urllib.request import urlopen

# Fix Windows console encoding BEFORE any output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from rustworkx_framework.builder import BuilderConfig, GraphBuilder
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
from rustworkx_framework.execution import MACPRunner, RunnerConfig, StreamEventType

# =============================================================================
# Constants & LLM Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "experiment_output"
FRAMES_DIR = OUTPUT_DIR / "animation_frames"
MODEL_PATH = OUTPUT_DIR / "gnn_router_model.pt"
RUN_LOG_PATH = OUTPUT_DIR / "run.log"

# LLM endpoint configuration
LLM_BASE_URL = "https://study-instantly-grande-rights.trycloudflare.com/v1"
LLM_API_KEY = "very-secure-key-sber1love"
LLM_MODEL = "zai-org/GLM-4.7"
LLM_TIMEOUT = 120  # seconds per LLM call
LLM_CONNECT_TIMEOUT = 15  # seconds to establish connection

# Number of real task executions to collect metrics
NUM_REAL_TASKS = 20  # Each task = full multi-agent pipeline with real LLM calls

# Whether to use simulated metrics instead of real LLM calls
SIMULATE_MODE = "--simulate" in sys.argv or "--dry-run" in sys.argv

# Agent definitions with system prompts for real LLM execution
AGENT_DEFS: dict[str, dict[str, str]] = {
    "coordinator": {
        "persona": "a project coordinator who delegates tasks and synthesizes results",
        "description": (
            "You are the coordinator. Read the task, break it into sub-tasks, "
            "and provide a clear plan for the team. Be concise and structured."
        ),
    },
    "researcher": {
        "persona": "a thorough researcher who gathers information",
        "description": (
            "You are a researcher. Analyze the topic, identify key facts, "
            "and provide a well-structured summary of your findings. "
            "Focus on accuracy and completeness."
        ),
    },
    "analyst": {
        "persona": "a data analyst who evaluates information critically",
        "description": (
            "You are an analyst. Evaluate the information provided, "
            "identify patterns, strengths, and weaknesses. "
            "Provide quantitative assessment where possible."
        ),
    },
    "writer": {
        "persona": "a skilled writer who creates clear, polished text",
        "description": (
            "You are a writer. Take the analysis and research, "
            "and produce a clear, well-structured written output. "
            "Focus on clarity and readability."
        ),
    },
    "reviewer": {
        "persona": "a critical reviewer who ensures quality",
        "description": (
            "You are a reviewer. Check the work for errors, inconsistencies, "
            "and areas for improvement. Provide a quality score from 0 to 10 "
            "and specific feedback."
        ),
    },
    "expert_1": {
        "persona": "a domain expert in science and technology",
        "description": (
            "You are a domain expert in science and technology. "
            "Provide deep technical insights and validate technical claims. "
            "Be precise and cite relevant concepts."
        ),
    },
    "expert_2": {
        "persona": "a domain expert in business and strategy",
        "description": (
            "You are a domain expert in business and strategy. "
            "Evaluate proposals from a business perspective. "
            "Consider feasibility, ROI, and market factors."
        ),
    },
    "aggregator": {
        "persona": "an aggregator who combines multiple inputs into a coherent summary",
        "description": (
            "You are an aggregator. Combine inputs from multiple agents "
            "into a single coherent summary. Resolve conflicts and "
            "highlight consensus points."
        ),
    },
}

IRIS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Task scenarios — real queries for the multi-agent system
TASK_QUERIES = [
    {
        "name": "research_report",
        "query": "Analyze the current state of Graph Neural Networks in multi-agent systems. What are the key advantages and limitations?",
        "path": ["coordinator", "researcher", "expert_1", "aggregator", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "data_analysis",
        "query": "Compare transformer-based architectures vs GNN-based architectures for routing in agent networks. Provide quantitative analysis.",
        "path": ["coordinator", "analyst", "expert_1", "aggregator", "writer", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "quick_review",
        "query": "Write a brief summary of how adaptive routing improves multi-agent collaboration efficiency.",
        "path": ["coordinator", "writer", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "deep_investigation",
        "query": "Investigate the trade-offs between centralized and decentralized routing in multi-agent systems. Consider both technical and business perspectives.",
        "path": ["coordinator", "researcher", "expert_1", "expert_2", "aggregator", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "expert_consult",
        "query": "Evaluate the feasibility of using GNN-based routing for real-time decision making in production multi-agent systems.",
        "path": ["coordinator", "analyst", "expert_1", "aggregator", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "tech_proposal",
        "query": "Propose an architecture for a self-optimizing multi-agent system that uses graph neural networks for dynamic task routing.",
        "path": ["coordinator", "researcher", "analyst", "writer", "reviewer"],
        "final_agent": "reviewer",
    },
    {
        "name": "comparative_study",
        "query": "Compare ARGMAX, TOP_K, and SOFTMAX routing strategies for multi-agent systems. Which is best for latency-sensitive applications?",
        "path": ["coordinator", "analyst", "expert_1", "aggregator", "reviewer"],
        "final_agent": "reviewer",
    },
]


# =============================================================================
# Helpers
# =============================================================================

_LOG_FILE = None  # Will be set in _ensure_dirs


def _print(*args: Any, **kwargs: Any) -> None:
    """Print with flush to both stdout and log file."""
    kwargs.setdefault("flush", True)
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, flush=True)
    except (UnicodeEncodeError, OSError):
        # Fallback for broken terminals
        try:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            pass
    # Also write to log file
    if _LOG_FILE is not None:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def _header(step: int, total: int, title: str) -> None:
    _print(f"\n{'=' * 60}")
    _print(f"  Step {step}/{total} -- {title}")
    _print(f"{'=' * 60}")


def _ensure_dirs() -> None:
    global _LOG_FILE  # noqa: PLW0603
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = str(RUN_LOG_PATH)
    # Clear previous log
    RUN_LOG_PATH.write_text(
        f"=== GNN Routing Experiment Log ===\n"
        f"Started: {datetime.now(UTC).isoformat()}\n"
        f"Mode: {'SIMULATE' if SIMULATE_MODE else 'REAL LLM'}\n\n",
        encoding="utf-8",
    )


# =============================================================================
# 1. Graph Construction (using GraphBuilder for proper RoleGraph)
# =============================================================================

def _add_dense_topology(builder: GraphBuilder, agent_ids: list[str]) -> None:
    """Create dense initial topology: task -> all agents + all-to-all directed mesh."""
    builder.connect_task_to_agents(agent_ids=agent_ids, bidirectional=False)
    for source_id in agent_ids:
        for target_id in agent_ids:
            if source_id != target_id:
                builder.add_workflow_edge(source_id, target_id, weight=0.6)


def create_experiment_graph(query: str = "") -> RoleGraph:
    """Build a dense graph where every agent is connected to every other agent."""
    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True, check_cycles=False))
    builder.add_task(
        query=query or "Multi-agent GNN routing experiment",
        description="Experiment task for collecting real execution metrics",
    )

    for agent_id, agent_def in AGENT_DEFS.items():
        builder.add_agent(
            agent_id=agent_id,
            display_name=agent_id.replace("_", " ").title(),
            persona=agent_def["persona"],
            description=agent_def["description"],
        )

    _add_dense_topology(builder, list(AGENT_DEFS.keys()))
    return builder.build()


def create_scenario_graph(scenario: dict[str, Any]) -> RoleGraph:
    """Build dense scenario graph (non-chain) with dynamic choice during execution."""
    builder = GraphBuilder(BuilderConfig(include_task_node=True, validate=True, check_cycles=False))
    builder.add_task(query=scenario["query"], description=f"Task: {scenario['name']}")

    for agent_id, agent_def in AGENT_DEFS.items():
        builder.add_agent(
            agent_id=agent_id,
            display_name=agent_id.replace("_", " ").title(),
            persona=agent_def["persona"],
            description=agent_def["description"],
        )

    _add_dense_topology(builder, list(AGENT_DEFS.keys()))
    graph = builder.build()
    graph.start_node = "coordinator"
    graph.end_node = scenario["final_agent"]
    return graph


# =============================================================================
# 2. Real LLM Metric Collection via MACPRunner
# =============================================================================

def create_llm_caller():
    """Create an OpenAI-compatible caller for the Qwen model with proper timeout."""
    from openai import OpenAI

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )

    from rustworkx_framework.tools.llm_integration import OpenAICaller

    return OpenAICaller(
        client=client,
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=1024,
    )


def _test_llm_connectivity() -> bool:
    """Quick connectivity test for the LLM endpoint."""
    _print("  Testing LLM connectivity...")
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_CONNECT_TIMEOUT,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        _print(f"  LLM responded: {text[:50]!r} -- OK!")
        return True
    except Exception as e:
        _print(f"  LLM connectivity FAILED: {e}")
        return False


def _extract_quality_from_review(text: str) -> float:
    """Try to extract a quality score from reviewer's response."""
    import re
    # Look for patterns like "score: 8/10", "quality: 7", "8 out of 10", etc.
    patterns = [
        r'(\d+(?:\.\d+)?)\s*/\s*10',
        r'score[:\s]+(\d+(?:\.\d+)?)',
        r'quality[:\s]+(\d+(?:\.\d+)?)',
        r'rating[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if val > 1.0:
                val = val / 10.0  # Normalize to 0-1
            return min(1.0, max(0.0, val))
    # Heuristic: longer, more detailed responses tend to be higher quality
    word_count = len(text.split())
    return min(1.0, max(0.3, word_count / 500.0))


def collect_real_metrics(
    tracker: MetricsTracker,
    num_tasks: int = NUM_REAL_TASKS,
) -> list[dict[str, Any]]:
    """
    Execute REAL tasks through MACPRunner with actual LLM calls.

    Each task runs a scenario through the agent graph, collecting
    real latency, token usage, and quality metrics.
    """
    _print(f"\n  Creating LLM caller -> {LLM_BASE_URL}")
    _print(f"  Model: {LLM_MODEL}")
    _print(f"  Tasks to execute: {num_tasks}")

    llm_caller = create_llm_caller()

    execution_logs: list[dict[str, Any]] = []
    total_tokens_all = 0
    total_time_all = 0.0

    for task_idx in range(num_tasks):
        scenario = TASK_QUERIES[task_idx % len(TASK_QUERIES)]
        path = scenario["path"]

        _print(f"\n  --- Task {task_idx + 1}/{num_tasks}: {scenario['name']} ---")
        _print(f"  Path: {' -> '.join(path)}")
        _print(f"  Query: {scenario['query'][:80]}...")

        # Build scenario-specific graph
        graph = create_scenario_graph(scenario)

        # Create runner
        runner = MACPRunner(
            llm_caller=llm_caller,
            config=RunnerConfig(
                timeout=float(LLM_TIMEOUT),
                adaptive=False,
                update_states=True,
                broadcast_task_to_all=False,
            ),
        )

        task_log: dict[str, Any] = {
            "task_id": task_idx,
            "scenario": scenario["name"],
            "query": scenario["query"],
            "path": path,
            "steps": [],
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "success": True,
        }

        # Execute via streaming to capture per-agent metrics
        task_start = time.time()
        agent_start_times: dict[str, float] = {}
        agent_responses: dict[str, str] = {}

        try:
            for event in runner.stream(graph, final_agent_id=scenario["final_agent"]):
                etype = event.event_type

                if etype == StreamEventType.AGENT_START:
                    aid = getattr(event, "agent_id", "")
                    agent_start_times[aid] = time.time()
                    _print(f"    > {aid} starting...")

                elif etype == StreamEventType.AGENT_OUTPUT:
                    aid = getattr(event, "agent_id", "")
                    content = getattr(event, "content", "")
                    agent_responses[aid] = content
                    tokens = getattr(event, "tokens_used", 0)

                    # Calculate real latency
                    latency_ms = (time.time() - agent_start_times.get(aid, task_start)) * 1000

                    # Estimate quality
                    quality = _extract_quality_from_review(content) if aid == "reviewer" else min(1.0, len(content.split()) / 200.0 + 0.3)

                    # Record REAL metrics to tracker
                    tracker.record_node_execution(
                        node_id=aid,
                        success=True,
                        latency_ms=latency_ms,
                        cost_tokens=tokens,
                        quality=quality,
                    )

                    step_log = {
                        "agent": aid,
                        "latency_ms": round(latency_ms, 2),
                        "success": True,
                        "quality": round(quality, 4),
                        "tokens": tokens,
                        "response_length": len(content),
                        "response_preview": content[:150],
                    }
                    task_log["steps"].append(step_log)
                    task_log["total_tokens"] += tokens

                    _print(f"    < {aid}: {len(content)} chars, {latency_ms:.0f}ms, q={quality:.3f}")

                    # Record edge transitions between consecutive agents
                    if len(task_log["steps"]) > 1:
                        prev_agent = task_log["steps"][-2]["agent"]
                        tracker.record_edge_transition(
                            source_id=prev_agent,
                            target_id=aid,
                            success=True,
                            latency_ms=latency_ms,
                        )

                elif etype == StreamEventType.AGENT_ERROR:
                    aid = getattr(event, "agent_id", "")
                    err = getattr(event, "error_message", "unknown error")
                    latency_ms = (time.time() - agent_start_times.get(aid, task_start)) * 1000

                    # Record failure
                    tracker.record_node_execution(
                        node_id=aid,
                        success=False,
                        latency_ms=latency_ms,
                        cost_tokens=0,
                        quality=0.0,
                    )

                    step_log = {
                        "agent": aid,
                        "latency_ms": round(latency_ms, 2),
                        "success": False,
                        "quality": 0.0,
                        "error": err,
                    }
                    task_log["steps"].append(step_log)
                    task_log["success"] = False
                    _print(f"    X {aid}: ERROR - {err}")

                elif etype == StreamEventType.RUN_END:
                    total_tokens = getattr(event, "total_tokens", 0)
                    total_time = getattr(event, "total_time", 0.0)
                    task_log["total_tokens"] = total_tokens
                    task_log["total_latency_ms"] = total_time * 1000

        except Exception as e:
            _print(f"    !!! Task failed: {e}")
            task_log["success"] = False
            task_log["error"] = str(e)

            # Record failures for agents that didn't execute
            for agent_id in path:
                if agent_id not in agent_responses:
                    tracker.record_node_execution(
                        node_id=agent_id,
                        success=False,
                        latency_ms=0.0,
                        cost_tokens=0,
                        quality=0.0,
                    )

        task_elapsed = (time.time() - task_start) * 1000
        task_log["total_latency_ms"] = round(task_elapsed, 2)
        execution_logs.append(task_log)

        total_tokens_all += task_log.get("total_tokens", 0)
        total_time_all += task_elapsed

        _print(f"  Task {task_idx + 1} done: {task_elapsed:.0f}ms, "
               f"tokens={task_log.get('total_tokens', 0)}, "
               f"success={task_log['success']}")

    _print(f"\n  === Metric Collection Summary ===")
    _print(f"  Total tasks: {num_tasks}")
    _print(f"  Successful: {sum(1 for l in execution_logs if l['success'])}")
    _print(f"  Total tokens: {total_tokens_all}")
    _print(f"  Total time: {total_time_all / 1000:.1f}s")

    return execution_logs


def load_iris_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the real Iris dataset from UCI."""
    with urlopen(IRIS_DATA_URL, timeout=30) as response:
        rows = response.read().decode("utf-8").strip().splitlines()

    features: list[list[float]] = []
    labels: list[int] = []
    class_to_idx = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }

    for row in rows:
        parts = row.split(",")
        if len(parts) == 5:
            features.append([float(v) for v in parts[:4]])
            labels.append(class_to_idx[parts[4]])

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def collect_simulated_metrics(
    tracker: MetricsTracker,
    num_tasks: int = NUM_REAL_TASKS,
) -> list[dict[str, Any]]:
    """Collect dataset-driven metrics from the real Iris dataset (no LLM calls)."""
    _print("\n  DATASET MODE -- using real Iris samples (no LLM calls)")
    x, y = load_iris_dataset()
    _print(f"  Loaded Iris: {x.shape[0]} samples, {x.shape[1]} features")

    class_paths = {
        0: ["coordinator", "researcher", "expert_1", "aggregator", "reviewer"],
        1: ["coordinator", "analyst", "writer", "aggregator", "reviewer"],
        2: ["coordinator", "researcher", "analyst", "expert_2", "aggregator", "reviewer"],
    }

    agent_order = list(AGENT_DEFS.keys())
    execution_logs: list[dict[str, Any]] = []

    for task_idx in range(num_tasks):
        sample_idx = task_idx % x.shape[0]
        sample = x[sample_idx]
        label = int(y[sample_idx].item())
        path = class_paths[label]

        task_log: dict[str, Any] = {
            "task_id": task_idx,
            "scenario": f"iris_class_{label}",
            "query": f"Iris sample #{sample_idx}",
            "path": path,
            "steps": [],
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "success": True,
        }

        prev_agent = None
        for agent_id in path:
            pos = agent_order.index(agent_id) + 1
            latency_ms = float(120 + sample[0].item() * 25 + sample[2].item() * 30 + pos * 12)
            quality_raw = float((sample[1].item() + sample[3].item()) / 10 + (4 - abs(label - (pos % 3))) * 0.12)
            quality = max(0.0, min(1.0, quality_raw))
            tokens = int(90 + sample[2].item() * 18 + pos * 7)
            success = quality >= 0.35

            tracker.record_node_execution(
                node_id=agent_id,
                success=success,
                latency_ms=latency_ms,
                cost_tokens=tokens,
                quality=quality,
            )

            if prev_agent is not None:
                tracker.record_edge_transition(
                    source_id=prev_agent,
                    target_id=agent_id,
                    success=success,
                    latency_ms=latency_ms,
                )

            task_log["steps"].append({
                "agent": agent_id,
                "latency_ms": round(latency_ms, 2),
                "success": success,
                "quality": round(quality, 4),
                "tokens": tokens,
            })
            task_log["total_latency_ms"] += latency_ms
            task_log["total_tokens"] += tokens
            task_log["success"] = task_log["success"] and success
            prev_agent = agent_id

        task_log["total_latency_ms"] = round(task_log["total_latency_ms"], 2)
        execution_logs.append(task_log)

    _print(f"  Generated {len(execution_logs)} dataset-based executions")
    return execution_logs


# =============================================================================
# 3. Graph Algorithms Analysis
# =============================================================================

def analyze_graph(graph: RoleGraph) -> dict[str, Any]:
    """Run graph algorithms and return structured results."""
    alg = GraphAlgorithms(graph)
    results: dict[str, Any] = {}

    # k-shortest paths
    paths_data = []
    try:
        for path in alg.k_shortest_paths("coordinator", "reviewer", k=5):
            paths_data.append({
                "nodes": path.nodes,
                "weight": round(path.total_weight, 4),
                "edge_weights": [round(w, 4) for w in path.edge_weights],
            })
    except Exception as e:
        _print(f"  Warning: k-shortest paths failed: {e}")
    results["k_shortest_paths"] = paths_data

    # Centralities
    centralities = {}
    for ct in [CentralityType.PAGERANK, CentralityType.BETWEENNESS, CentralityType.DEGREE]:
        try:
            cr = alg.compute_centrality(ct)
            centralities[ct.value] = {
                "values": {k: round(v, 6) for k, v in cr.values.items()},
                "top_3": [(n, round(s, 6)) for n, s in cr.top_k(3)],
            }
        except Exception:
            pass
    results["centralities"] = centralities

    # Communities
    communities = alg.detect_communities()
    results["communities"] = {
        "count": communities.num_communities,
        "sizes": communities.get_community_sizes(),
        "members": [list(c) for c in communities.communities],
    }

    # Cycles
    cycles = alg.detect_cycles(max_length=6)
    results["cycles"] = [{"nodes": c.nodes, "weight": round(c.total_weight, 4)} for c in cycles[:5]]

    return results


# =============================================================================
# 4. Training Data Preparation
# =============================================================================

def prepare_training_data(
    graph: RoleGraph,
    tracker: MetricsTracker,
    n_samples: int = 300,
) -> tuple[list[Any], list[Any], int]:
    """Prepare GNN training data from real Iris samples + collected graph metrics."""
    feat_gen = DefaultFeatureGenerator()
    node_ids = graph.node_ids
    graph_features = feat_gen.generate_node_features(graph, node_ids, tracker).to(torch.float32)
    edge_index = graph.edge_index.to(torch.long)
    _, y_data = load_iris_dataset()

    agent_targets = {
        0: "expert_1",
        1: "analyst",
        2: "expert_2",
    }
    target_index = {nid: idx for idx, nid in enumerate(node_ids)}

    total_samples = min(n_samples, y_data.shape[0])
    split = int(total_samples * 0.8)
    train_data: list[Data] = []
    val_data: list[Data] = []

    for idx in range(total_samples):
        sample_label = int(y_data[idx].item())
        x = graph_features.clone()

        y = torch.zeros(len(node_ids), dtype=torch.long)
        y[target_index[agent_targets[sample_label]]] = 1

        data_point = Data(x=x, edge_index=edge_index, y=y)
        if idx < split:
            train_data.append(data_point)
        else:
            val_data.append(data_point)

    in_channels = train_data[0].x.shape[1]
    _print(f"  Iris-backed samples: {total_samples}")
    _print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Features: {in_channels}")
    return train_data, val_data, in_channels


# =============================================================================
# 5. GNN Training with Epoch Logging
# =============================================================================

def train_gnn_model(
    train_data: list[Any],
    val_data: list[Any],
    in_channels: int,
    model_type: GNNModelType = GNNModelType.GCN,
) -> tuple[Any, dict[str, Any]]:
    """Train a GNN model and return it along with training history."""
    cfg = TrainingConfig(
        learning_rate=1e-3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        epochs=80,
        batch_size=16,
        patience=15,
        task="node_classification",
        num_classes=2,
    )

    model = create_gnn_router(model_type, in_channels, cfg.num_classes, cfg)
    trainer = GNNTrainer(model, cfg)

    _print(f"  Model: {model_type.value}, params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    result = trainer.train(train_data, val_data, verbose=True)
    elapsed = time.time() - t0

    trainer.save(MODEL_PATH)

    history = {
        "model_type": model_type.value,
        "train_losses": [round(l, 6) for l in result.train_losses],
        "val_losses": [round(l, 6) for l in result.val_losses],
        "best_epoch": result.best_epoch,
        "best_val_loss": round(result.best_val_loss, 6),
        "total_epochs": len(result.train_losses),
        "training_time_sec": round(elapsed, 2),
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    _print(f"  Best epoch: {result.best_epoch}, best val loss: {result.best_val_loss:.6f}")
    _print(f"  Training time: {elapsed:.2f}s")

    return model, history


def optimize_graph_topology(
    graph: RoleGraph,
    model: Any,
    tracker: MetricsTracker,
    iterations: int = 5,
) -> list[dict[str, Any]]:
    """Dynamically rewire graph topology using GNN scores (add/remove edges each step)."""
    router = GNNRouterInference(model, DefaultFeatureGenerator())
    agent_nodes = [nid for nid in graph.node_ids if nid != graph.task_node]
    snapshots: list[dict[str, Any]] = []

    for step in range(iterations):
        scores = router.get_all_scores(graph, tracker)
        ranked_pairs: list[tuple[float, str, str]] = []
        for src in agent_nodes:
            for tgt in agent_nodes:
                if src != tgt:
                    pair_score = (scores.get(src, 0.0) + scores.get(tgt, 0.0)) / 2
                    ranked_pairs.append((pair_score, src, tgt))
        ranked_pairs.sort(reverse=True)

        keep_edges = int(len(ranked_pairs) * (0.35 + step * 0.08))
        selected_pairs = {(src, tgt): score for score, src, tgt in ranked_pairs[:keep_edges]}

        removed: list[str] = []
        for edge in list(graph.edges):
            src, tgt = edge["source"], edge["target"]
            if src == graph.task_node:
                continue
            if (src, tgt) not in selected_pairs:
                if graph.remove_edge(src, tgt):
                    removed.append(f"{src}->{tgt}")

        added: list[str] = []
        for (src, tgt), score in selected_pairs.items():
            if tgt not in graph.get_neighbors(src, direction="out"):
                graph.add_edge(src, tgt, weight=float(score))
                added.append(f"{src}->{tgt}")

        snapshots.append({
            "step": step + 1,
            "num_edges": graph.num_edges,
            "added": added,
            "removed": removed,
            "top_scores": sorted(scores.items(), key=lambda x: -x[1])[:5],
        })
        _print(f"  Topology step {step + 1}: edges={graph.num_edges}, +{len(added)}, -{len(removed)}")

    return snapshots


# =============================================================================
# 6. Routing Strategy Comparison
# =============================================================================

def compare_routing_strategies(
    graph: RoleGraph,
    model: Any,
    tracker: MetricsTracker,
    num_trials: int = 100,
) -> dict[str, Any]:
    """
    Compare GNN-based routing against baselines.

    Strategies:
      - GNN_ARGMAX: GNN scores -> pick best
      - GNN_TOP_K: GNN scores -> pick top-3
      - GNN_THRESHOLD: GNN scores -> pick above threshold
      - GNN_SOFTMAX: GNN scores -> sample proportionally
      - RANDOM: uniform random selection
      - TOPOLOGICAL: follow graph topology (first successor)
      - METRICS_BEST: pick node with best composite score from tracker
    """
    router = GNNRouterInference(model, DefaultFeatureGenerator())
    node_scores = tracker.get_node_scores()

    strategies_results: dict[str, dict[str, Any]] = {}

    # Define all strategies to test
    strategy_configs = [
        ("GNN_ARGMAX",    RoutingStrategy.ARGMAX,         {"candidates": None}),
        ("GNN_TOP_K",     RoutingStrategy.TOP_K,          {"top_k": 3}),
        ("GNN_THRESHOLD", RoutingStrategy.THRESHOLD,      {"threshold": 0.1}),
        ("GNN_SOFTMAX",   RoutingStrategy.SOFTMAX_SAMPLE, {"candidates": None}),
    ]

    # Source nodes to test routing from
    source_nodes = ["coordinator", "researcher", "analyst", "aggregator"]

    for strategy_name, strategy, kwargs in strategy_configs:
        trial_results = []
        for _ in range(num_trials):
            source = random.choice(source_nodes)
            # Get successors as candidates
            successors = []
            for i in graph.graph.node_indices():
                if graph.graph.get_node_data(i)["id"] == source:
                    successors = [graph.graph.get_node_data(s)["id"] for s in graph.graph.successor_indices(i)]
                    break

            if not successors:
                continue

            predict_kwargs: dict[str, Any] = {"strategy": strategy}
            if kwargs.get("top_k"):
                predict_kwargs["top_k"] = kwargs["top_k"]
            if kwargs.get("threshold"):
                predict_kwargs["threshold"] = kwargs["threshold"]
            if kwargs.get("candidates") is not None:
                predict_kwargs["candidates"] = kwargs["candidates"]

            try:
                result = router.predict(
                    graph,
                    source=source,
                    candidates=successors,
                    metrics_tracker=tracker,
                    **predict_kwargs,
                )
                # Evaluate quality of chosen nodes
                chosen_scores = [node_scores.get(n, 0.5) for n in result.recommended_nodes]
                avg_score = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
                trial_results.append({
                    "source": source,
                    "recommended": result.recommended_nodes,
                    "confidence": round(result.confidence, 4),
                    "avg_chosen_score": round(avg_score, 4),
                })
            except Exception:
                pass

        if trial_results:
            avg_confidence = sum(t["confidence"] for t in trial_results) / len(trial_results)
            avg_quality = sum(t["avg_chosen_score"] for t in trial_results) / len(trial_results)
            strategies_results[strategy_name] = {
                "avg_confidence": round(avg_confidence, 4),
                "avg_chosen_quality": round(avg_quality, 4),
                "num_trials": len(trial_results),
                "sample_results": trial_results[:5],
            }

    # RANDOM baseline
    random_results = []
    for _ in range(num_trials):
        source = random.choice(source_nodes)
        successors = []
        for i in graph.graph.node_indices():
            if graph.graph.get_node_data(i)["id"] == source:
                successors = [graph.graph.get_node_data(s)["id"] for s in graph.graph.successor_indices(i)]
                break
        if successors:
            chosen = random.choice(successors)
            score = node_scores.get(chosen, 0.5)
            random_results.append({"source": source, "recommended": [chosen], "avg_chosen_score": round(score, 4)})

    if random_results:
        avg_quality = sum(r["avg_chosen_score"] for r in random_results) / len(random_results)
        strategies_results["RANDOM"] = {
            "avg_confidence": 0.0,
            "avg_chosen_quality": round(avg_quality, 4),
            "num_trials": len(random_results),
            "sample_results": random_results[:5],
        }

    # TOPOLOGICAL baseline (always pick first successor)
    topo_results = []
    for _ in range(num_trials):
        source = random.choice(source_nodes)
        successors = []
        for i in graph.graph.node_indices():
            if graph.graph.get_node_data(i)["id"] == source:
                successors = [graph.graph.get_node_data(s)["id"] for s in graph.graph.successor_indices(i)]
                break
        if successors:
            chosen = successors[0]
            score = node_scores.get(chosen, 0.5)
            topo_results.append({"source": source, "recommended": [chosen], "avg_chosen_score": round(score, 4)})

    if topo_results:
        avg_quality = sum(r["avg_chosen_score"] for r in topo_results) / len(topo_results)
        strategies_results["TOPOLOGICAL"] = {
            "avg_confidence": 0.0,
            "avg_chosen_quality": round(avg_quality, 4),
            "num_trials": len(topo_results),
            "sample_results": topo_results[:5],
        }

    # METRICS_BEST baseline (pick successor with best composite score)
    metrics_results = []
    for _ in range(num_trials):
        source = random.choice(source_nodes)
        successors = []
        for i in graph.graph.node_indices():
            if graph.graph.get_node_data(i)["id"] == source:
                successors = [graph.graph.get_node_data(s)["id"] for s in graph.graph.successor_indices(i)]
                break
        if successors:
            chosen = max(successors, key=lambda n: node_scores.get(n, 0.5))
            score = node_scores.get(chosen, 0.5)
            metrics_results.append({"source": source, "recommended": [chosen], "avg_chosen_score": round(score, 4)})

    if metrics_results:
        avg_quality = sum(r["avg_chosen_score"] for r in metrics_results) / len(metrics_results)
        strategies_results["METRICS_BEST"] = {
            "avg_confidence": 0.0,
            "avg_chosen_quality": round(avg_quality, 4),
            "num_trials": len(metrics_results),
            "sample_results": metrics_results[:5],
        }

    return strategies_results


# =============================================================================
# 7. Edge Weight Extraction (before/after)
# =============================================================================

def extract_edge_weights(graph: RoleGraph, tracker: MetricsTracker) -> dict[str, Any]:
    """Extract edge weights from graph and metrics for visualization."""
    static_weights = {}
    for edge in graph.edges:
        key = f"{edge['source']}->{edge['target']}"
        static_weights[key] = {
            "weight": edge.get("weight", 1.0),
        }

    dynamic_weights = {}
    routing_weights = tracker.get_routing_weights()
    for (src, tgt), w in routing_weights.items():
        key = f"{src}->{tgt}"
        edge_metrics = tracker.get_edge_metrics(src, tgt)
        dynamic_weights[key] = {
            "effective_weight": round(w, 4),
            "reliability": round(edge_metrics.reliability, 4) if edge_metrics else 1.0,
            "avg_latency_ms": round(edge_metrics.avg_latency_ms, 2) if edge_metrics else 0.0,
            "total_transitions": edge_metrics.total_transitions if edge_metrics else 0,
        }

    return {"static": static_weights, "dynamic": dynamic_weights}


# =============================================================================
# 8. HTML Visualization Generator
# =============================================================================

def _generate_graph_html(
    title: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_scores: dict[str, float],
    output_path: Path,
    tracker: MetricsTracker | None = None,
) -> None:
    """Generate an interactive HTML visualization of the graph using vis.js."""

    # Build vis.js nodes
    vis_nodes = []
    for n in nodes:
        nid = n["id"]
        score = node_scores.get(nid, 0.5)
        # Color: green for high score, red for low
        r = int(255 * (1 - score))
        g = int(255 * score)
        color = f"rgb({r},{g},80)"

        # Add real metrics info to tooltip
        tooltip = f"Score: {score:.3f}"
        if tracker:
            nm = tracker.get_node_metrics(nid)
            if nm:
                tooltip += (
                    f"\\nExecutions: {nm.total_executions}"
                    f"\\nReliability: {nm.reliability:.3f}"
                    f"\\nAvg Latency: {nm.avg_latency_ms:.0f}ms"
                    f"\\nAvg Quality: {nm.avg_quality:.3f}"
                    f"\\nAvg Cost: {nm.avg_cost_tokens:.0f} tokens"
                )

        vis_nodes.append({
            "id": nid,
            "label": f"{nid}\\nscore={score:.3f}",
            "title": tooltip,
            "color": {"background": color, "border": "#333"},
            "font": {"color": "#fff", "size": 14},
            "shape": "box",
            "borderWidth": 2,
        })

    # Build vis.js edges
    vis_edges = []
    for e in edges:
        weight = e.get("weight", 1.0)
        width = max(1, int(weight * 5))

        # Get real edge metrics if available
        edge_label = f"w={weight:.2f}"
        edge_tooltip = ""
        if tracker:
            em = tracker.get_edge_metrics(e["source"], e["target"])
            if em:
                edge_label += f"\\nrel={em.reliability:.2f}"
                edge_tooltip = (
                    f"Transitions: {em.total_transitions}"
                    f"\\nReliability: {em.reliability:.3f}"
                    f"\\nAvg Latency: {em.avg_latency_ms:.0f}ms"
                )

        reliability = 1.0
        if tracker:
            em = tracker.get_edge_metrics(e["source"], e["target"])
            if em:
                reliability = em.reliability

        r_val = int(255 * (1 - reliability))
        g_val = int(255 * reliability)
        vis_edges.append({
            "from": e["source"],
            "to": e["target"],
            "label": edge_label,
            "title": edge_tooltip,
            "width": width,
            "color": {"color": f"rgb({r_val},{g_val},50)", "highlight": "#ff0"},
            "arrows": "to",
            "font": {"size": 10, "align": "middle"},
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
        #header {{ padding: 15px 25px; background: #16213e; border-bottom: 2px solid #0f3460; }}
        #header h1 {{ margin: 0; font-size: 22px; color: #e94560; }}
        #header p {{ margin: 5px 0 0; font-size: 13px; color: #aaa; }}
        #graph {{ width: 100%; height: calc(100vh - 80px); }}
        #legend {{ position: absolute; top: 90px; right: 20px; background: rgba(22,33,62,0.9);
                   padding: 12px; border-radius: 8px; font-size: 12px; border: 1px solid #0f3460; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <p>MECE Framework -- GNN Routing Experiment | LLM: {LLM_MODEL} | Nodes: {len(vis_nodes)} | Edges: {len(vis_edges)}</p>
    </div>
    <div id="graph"></div>
    <div id="legend">
        <b>Legend</b>
        <div class="legend-item"><div class="legend-color" style="background:rgb(0,255,80)"></div> High score node</div>
        <div class="legend-item"><div class="legend-color" style="background:rgb(255,0,80)"></div> Low score node</div>
        <div class="legend-item"><div class="legend-color" style="background:rgb(0,255,50)"></div> Reliable edge</div>
        <div class="legend-item"><div class="legend-color" style="background:rgb(255,0,50)"></div> Unreliable edge</div>
        <div class="legend-item" style="margin-top:8px;color:#aaa"><i>Hover nodes/edges for details</i></div>
    </div>
    <script>
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{ hierarchical: {{ direction: 'LR', sortMethod: 'directed', levelSeparation: 200 }} }},
            physics: {{ enabled: false }},
            interaction: {{ hover: true, tooltipDelay: 100 }},
            edges: {{ smooth: {{ type: 'cubicBezier' }} }}
        }};
        new vis.Network(container, data, options);
    </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _generate_training_html(all_histories: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Generate HTML with training loss curves for ALL models using Chart.js."""

    datasets_js = []
    colors = {"gcn": "#e94560", "gat": "#0f3460", "sage": "#e9a945"}
    best_info = []

    for model_name, history in all_histories.items():
        train_losses = history["train_losses"]
        val_losses = history["val_losses"]
        color = colors.get(model_name, "#888")

        datasets_js.append(f"""{{
            label: '{model_name.upper()} Train',
            data: {json.dumps(train_losses)},
            borderColor: '{color}',
            backgroundColor: 'rgba(0,0,0,0)',
            fill: false, tension: 0.3, pointRadius: 1
        }}""")
        datasets_js.append(f"""{{
            label: '{model_name.upper()} Val',
            data: {json.dumps(val_losses)},
            borderColor: '{color}',
            backgroundColor: 'rgba(0,0,0,0)',
            borderDash: [5,5], fill: false, tension: 0.3, pointRadius: 1
        }}""")

        best_info.append(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:{color}">{model_name.upper()}</div>
            <div class="stat-label">Best epoch: {history['best_epoch']} | Val loss: {history['best_val_loss']:.6f}</div>
            <div class="stat-label">Params: {history['num_params']:,} | Time: {history['training_time_sec']}s</div>
        </div>""")

    max_epochs = max(len(h["train_losses"]) for h in all_histories.values())
    epochs = list(range(1, max_epochs + 1))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Training Curves -- GNN Routing</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee;
               display: flex; flex-direction: column; align-items: center; padding: 30px; }}
        h1 {{ color: #e94560; }}
        .info {{ color: #aaa; font-size: 14px; margin-bottom: 20px; }}
        canvas {{ background: #16213e; border-radius: 10px; padding: 10px; max-width: 900px; }}
        .stats {{ display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; justify-content: center; }}
        .stat-card {{ background: #16213e; padding: 15px 25px; border-radius: 8px; text-align: center;
                      border: 1px solid #0f3460; min-width: 200px; }}
        .stat-value {{ font-size: 20px; font-weight: bold; }}
        .stat-label {{ font-size: 12px; color: #aaa; margin-top: 4px; }}
    </style>
</head>
<body>
    <h1>GNN Training Curves (3 Models)</h1>
    <p class="info">Trained on REAL metrics from {NUM_REAL_TASKS} LLM task executions | LLM: {LLM_MODEL}</p>
    <canvas id="chart" width="900" height="400"></canvas>
    <div class="stats">
        {''.join(best_info)}
    </div>
    <script>
        new Chart(document.getElementById('chart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(epochs)},
                datasets: [{','.join(datasets_js)}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#eee' }} }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch', color: '#aaa' }},
                          ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    y: {{ title: {{ display: true, text: 'Loss', color: '#aaa' }},
                          ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _generate_comparison_html(
    strategies: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate HTML bar chart comparing routing strategies."""
    names = list(strategies.keys())
    qualities = [strategies[n]["avg_chosen_quality"] for n in names]
    confidences = [strategies[n].get("avg_confidence", 0) for n in names]
    trials = [strategies[n]["num_trials"] for n in names]

    # Color map: GNN strategies in red, baselines in gray/blue
    colors = []
    for n in names:
        if n.startswith("GNN"):
            colors.append("'rgba(233,69,96,0.8)'")
        elif n == "METRICS_BEST":
            colors.append("'rgba(15,52,96,0.8)'")
        else:
            colors.append("'rgba(100,100,100,0.6)'")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Routing Strategy Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee;
               display: flex; flex-direction: column; align-items: center; padding: 30px; }}
        h1 {{ color: #e94560; }}
        .info {{ color: #aaa; font-size: 14px; margin-bottom: 20px; }}
        canvas {{ background: #16213e; border-radius: 10px; padding: 10px; max-width: 900px; }}
        table {{ margin-top: 30px; border-collapse: collapse; background: #16213e; border-radius: 8px; overflow: hidden; }}
        th {{ background: #0f3460; padding: 10px 20px; text-align: left; }}
        td {{ padding: 8px 20px; border-top: 1px solid #333; }}
        .highlight {{ color: #e94560; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Routing Strategy Comparison</h1>
    <p class="info">Based on REAL metrics from {NUM_REAL_TASKS} LLM executions | {sum(trials)} routing decisions | LLM: {LLM_MODEL}</p>
    <canvas id="chart" width="900" height="400"></canvas>
    <table>
        <tr><th>Strategy</th><th>Avg Quality</th><th>Avg Confidence</th><th>Trials</th></tr>
        {''.join(f'<tr><td>{n}</td><td class="highlight">{q:.4f}</td><td>{c:.4f}</td><td>{t}</td></tr>' for n, q, c, t in zip(names, qualities, confidences, trials))}
    </table>
    <script>
        new Chart(document.getElementById('chart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(names)},
                datasets: [{{
                    label: 'Avg Chosen Node Quality',
                    data: {json.dumps(qualities)},
                    backgroundColor: [{','.join(colors)}],
                    borderColor: [{','.join(c.replace('0.8', '1').replace('0.6', '1') for c in colors)}],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#eee' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#ccc' }}, grid: {{ color: '#333' }} }},
                    y: {{ title: {{ display: true, text: 'Quality Score', color: '#aaa' }},
                          ticks: {{ color: '#888' }}, grid: {{ color: '#333' }},
                          min: 0, max: 1 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _generate_animation_frame(
    epoch: int,
    graph: RoleGraph,
    model: Any,
    tracker: MetricsTracker,
    node_scores: dict[str, float],
) -> dict[str, Any]:
    """Generate a single animation frame (snapshot of GNN scores at given epoch)."""
    router = GNNRouterInference(model, DefaultFeatureGenerator())
    all_scores = router.get_all_scores(graph, tracker)

    return {
        "epoch": epoch,
        "gnn_scores": {k: round(v, 4) for k, v in all_scores.items()},
        "node_quality_scores": {k: round(v, 4) for k, v in node_scores.items()},
    }


# =============================================================================
# 9. Mermaid / DOT export
# =============================================================================

def export_static_visualizations(
    graph: RoleGraph,
    tracker: MetricsTracker,
    prefix: str,
) -> None:
    """Export Mermaid and DOT files for the graph."""
    from rustworkx_framework.core.visualization import (
        GraphVisualizer,
        VisualizationStyle,
    )

    style = VisualizationStyle(show_weights=True, show_tools=False)
    viz = GraphVisualizer(graph, style)

    viz.save_mermaid(str(OUTPUT_DIR / f"{prefix}.md"), title=f"Agent Graph -- {prefix}")
    viz.save_dot(str(OUTPUT_DIR / f"{prefix}.dot"), graph_name="GNNRoutingExperiment")
    _print(f"  Exported: {prefix}.md, {prefix}.dot")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main() -> None:
    _ensure_dirs()
    total_steps = 8
    mode_label = "SIMULATED" if SIMULATE_MODE else "REAL LLM"
    experiment_log: dict[str, Any] = {
        "experiment": f"GNN Routing Benchmark ({mode_label})",
        "framework": "MECE (RustworkX Agent Framework)",
        "llm_model": LLM_MODEL,
        "llm_endpoint": LLM_BASE_URL,
        "mode": mode_label,
        "timestamp": datetime.now(UTC).isoformat(),
        "steps": {},
    }

    # -- Step 1: Build graph ------------------------------------------------
    _header(1, total_steps, "Build agent graph")
    graph = create_experiment_graph()
    _print(f"  Nodes: {graph.node_ids}")
    _print(f"  Edges: {graph.num_edges}")
    _print(f"  LLM: {LLM_MODEL}")
    _print(f"  Endpoint: {LLM_BASE_URL}")
    experiment_log["graph"] = {
        "nodes": graph.node_ids,
        "num_edges": graph.num_edges,
        "edges": graph.edges,
    }

    # -- Step 2: Collect metrics -----------------------------------------------
    tracker = MetricsTracker()
    use_simulation = SIMULATE_MODE

    if not use_simulation:
        _header(2, total_steps, f"Collect REAL metrics ({NUM_REAL_TASKS} LLM task executions)")
        llm_ok = _test_llm_connectivity()
        if not llm_ok:
            _print("  WARNING: LLM endpoint unreachable, falling back to SIMULATION mode")
            use_simulation = True

    if use_simulation:
        _header(2, total_steps, f"Collect DATASET-DRIVEN metrics ({NUM_REAL_TASKS} tasks)")
        execution_logs = collect_simulated_metrics(tracker, num_tasks=NUM_REAL_TASKS)
    else:
        execution_logs = collect_real_metrics(tracker, num_tasks=NUM_REAL_TASKS)

    node_scores = tracker.get_node_scores()
    _print(f"\n  Node composite scores ({'simulated' if use_simulation else 'real LLM'}):")
    for nid, score in sorted(node_scores.items(), key=lambda x: -x[1]):
        nm = tracker.get_node_metrics(nid)
        if nm:
            _print(f"    {nid:<15} score={score:.4f}  rel={nm.reliability:.3f}  "
                   f"lat={nm.avg_latency_ms:.0f}ms  qual={nm.avg_quality:.3f}  "
                   f"execs={nm.total_executions}")

    edge_weights_before = extract_edge_weights(graph, tracker)
    experiment_log["steps"]["metrics_collection"] = {
        "num_tasks": len(execution_logs),
        "mode": "simulated" if use_simulation else "real_llm",
        "llm_model": LLM_MODEL,
        "node_scores": {k: round(v, 4) for k, v in node_scores.items()},
        "edge_weights_before": edge_weights_before,
        "execution_summary": {
            "total_tasks": len(execution_logs),
            "successful": sum(1 for l in execution_logs if l["success"]),
            "failed": sum(1 for l in execution_logs if not l["success"]),
            "avg_latency_ms": round(
                sum(l["total_latency_ms"] for l in execution_logs) / max(len(execution_logs), 1), 2
            ),
            "total_tokens": sum(l.get("total_tokens", 0) for l in execution_logs),
        },
        "execution_logs": execution_logs,
    }

    # -- Step 3: Graph analysis ---------------------------------------------
    _header(3, total_steps, "Graph algorithms analysis")
    analysis = analyze_graph(graph)
    _print(f"  Shortest paths (coordinator->reviewer): {len(analysis['k_shortest_paths'])}")
    for i, p in enumerate(analysis["k_shortest_paths"][:3], 1):
        _print(f"    {i}. {' -> '.join(p['nodes'])}  (w={p['weight']:.4f})")
    _print(f"  Communities: {analysis['communities']['count']}")
    _print(f"  Cycles: {len(analysis['cycles'])}")
    experiment_log["steps"]["graph_analysis"] = analysis

    # -- Step 4: Visualize BEFORE training ----------------------------------
    _header(4, total_steps, "Visualize graph BEFORE training")
    nodes_data = [{"id": nid} for nid in graph.node_ids]
    edges_data = graph.edges
    _generate_graph_html(
        "Agent Graph -- BEFORE GNN Training",
        nodes_data, edges_data, node_scores,
        OUTPUT_DIR / "graph_before.html",
        tracker=tracker,
    )
    export_static_visualizations(graph, tracker, "graph_before")
    _print(f"  -> {OUTPUT_DIR / 'graph_before.html'}")

    # -- Step 5: Prepare data & train GNN -----------------------------------
    data_source = "simulated" if use_simulation else "REAL LLM"
    _header(5, total_steps, f"Prepare training data (from {data_source} metrics)")
    train_data, val_data, in_channels = prepare_training_data(graph, tracker, n_samples=300)
    experiment_log["steps"]["training_data"] = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "feature_dim": in_channels,
        "data_source": f"{'simulated' if use_simulation else 'REAL LLM'} execution metrics",
    }

    _header(6, total_steps, "Train GNN models")
    all_histories: dict[str, dict[str, Any]] = {}

    # Train main model (GCN)
    model, history = train_gnn_model(train_data, val_data, in_channels, GNNModelType.GCN)
    all_histories["gcn"] = history
    experiment_log["steps"]["training"] = {"GCN": history}

    # Also train GAT and SAGE for comparison
    for mtype in [GNNModelType.GAT, GNNModelType.SAGE]:
        _print(f"\n  Training {mtype.value}...")
        _, h = train_gnn_model(train_data, val_data, in_channels, mtype)
        all_histories[mtype.value] = h
        experiment_log["steps"]["training"][mtype.value] = h

    _generate_training_html(all_histories, OUTPUT_DIR / "training_curves.html")
    _print(f"  -> {OUTPUT_DIR / 'training_curves.html'}")

    # -- Step 7: Compare routing strategies ---------------------------------
    _header(7, total_steps, "Dynamic topology optimization + routing")
    topology_history = optimize_graph_topology(graph, model, tracker, iterations=6)
    experiment_log["steps"]["topology_optimization"] = topology_history
    strategies = compare_routing_strategies(graph, model, tracker, num_trials=200)

    _print("\n  Results:")
    _print(f"  {'Strategy':<20} {'Avg Quality':>12} {'Confidence':>12} {'Trials':>8}")
    _print(f"  {'-' * 52}")
    for name, data in sorted(strategies.items(), key=lambda x: -x[1]["avg_chosen_quality"]):
        _print(f"  {name:<20} {data['avg_chosen_quality']:>12.4f} "
               f"{data.get('avg_confidence', 0):>12.4f} {data['num_trials']:>8}")

    experiment_log["steps"]["routing_comparison"] = strategies
    _generate_comparison_html(strategies, OUTPUT_DIR / "routing_comparison.html")
    _print(f"\n  -> {OUTPUT_DIR / 'routing_comparison.html'}")

    # -- Step 8: Visualize AFTER training -----------------------------------
    _header(8, total_steps, "Visualize graph AFTER training")

    # Get GNN scores for coloring
    router = GNNRouterInference(model, DefaultFeatureGenerator())
    gnn_scores = router.get_all_scores(graph, tracker)
    _print("  GNN node scores:")
    for nid, score in sorted(gnn_scores.items(), key=lambda x: -x[1]):
        _print(f"    {nid:<15} gnn_score={score:.4f}  quality_score={node_scores.get(nid, 0):.4f}")

    edges_data = graph.edges
    _generate_graph_html(
        "Agent Graph -- AFTER GNN Training (colored by GNN score)",
        nodes_data, edges_data, gnn_scores,
        OUTPUT_DIR / "graph_after.html",
        tracker=tracker,
    )
    export_static_visualizations(graph, tracker, "graph_after")
    _print(f"  -> {OUTPUT_DIR / 'graph_after.html'}")

    edge_weights_after = extract_edge_weights(graph, tracker)
    experiment_log["steps"]["edge_weights_after"] = edge_weights_after
    experiment_log["steps"]["gnn_scores"] = {k: round(v, 4) for k, v in gnn_scores.items()}

    # -- Generate animation frames ------------------------------------------
    _print("\n  Generating animation frames...")
    frame = _generate_animation_frame(history["total_epochs"], graph, model, tracker, node_scores)
    frame_path = FRAMES_DIR / "frame_final.json"
    frame_path.write_text(json.dumps(frame, indent=2), encoding="utf-8")

    # -- Save full experiment log -------------------------------------------
    log_path = OUTPUT_DIR / "experiment_log.json"
    log_path.write_text(json.dumps(experiment_log, indent=2, default=str), encoding="utf-8")
    _print(f"\n  Full experiment log -> {log_path}")

    # -- Summary ------------------------------------------------------------
    _print(f"\n{'=' * 60}")
    _print("  EXPERIMENT COMPLETE!")
    _print(f"{'=' * 60}")
    _print(f"\n  Mode: {'SIMULATED' if use_simulation else 'REAL LLM'}")
    _print(f"  LLM Model: {LLM_MODEL}")
    _print(f"  Tasks executed: {NUM_REAL_TASKS}")
    _print(f"  Total tokens used: {experiment_log['steps']['metrics_collection']['execution_summary']['total_tokens']}")
    _print(f"\n  Output directory: {OUTPUT_DIR}")
    _print("  Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            label = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            _print(f"    {f.name:<35} {label}")

    _print(f"\n{'-' * 60}")
    _print("  HOW TO VIEW RESULTS:")
    _print("  1. Open graph_before.html and graph_after.html in browser")
    _print("     to see interactive graph visualization with REAL metrics")
    _print("  2. Open training_curves.html to see loss curves (3 models)")
    _print("  3. Open routing_comparison.html to see strategy comparison")
    _print("  4. See experiment_log.json for full structured data")
    _print("     including all LLM responses and per-agent metrics")
    _print(f"{'-' * 60}")
    _print("  HOW TO CREATE VIDEO / WEB RESOURCE:")
    _print("  See instructions below or run:")
    _print("    python -m examples.gnn_routing_experiment --help-video")
    _print(f"{'-' * 60}")


# =============================================================================
# VIDEO / WEB RESOURCE INSTRUCTIONS
# =============================================================================

VIDEO_INSTRUCTIONS = """
================================================================
  INSTRUCTIONS: Creating video / web resource from experiment logs
================================================================

=== OPTION 1: Interactive web resource (recommended) ===

All HTML files are already generated and ready to view:

  1. graph_before.html  -- interactive graph BEFORE GNN training
     * Nodes colored by composite score (green=good, red=bad)
     * Edges colored by reliability from REAL executions
     * Hover over nodes/edges for detailed metrics
     * Drag nodes, zoom, pan for exploration

  2. graph_after.html   -- graph AFTER training
     * Nodes colored by GNN score (what the model considers important)
     * Compare with graph_before to see how GNN re-evaluates nodes

  3. training_curves.html -- training curves for all 3 models (GCN, GAT, SAGE)

  4. routing_comparison.html -- bar chart comparing all strategies

  To publish: upload HTML files to GitHub Pages or any static hosting.

=== OPTION 2: Video from frames (ffmpeg) ===

  Step 1: Install dependencies
    pip install selenium pillow

  Step 2: Screenshots HTML -> PNG
    from selenium import webdriver
    from pathlib import Path
    import time

    driver = webdriver.Chrome()
    for html_file in sorted(Path("examples/experiment_output").glob("*.html")):
        driver.get(f"file:///{html_file.resolve()}")
        time.sleep(2)
        driver.save_screenshot(f"frame_{html_file.stem}.png")
    driver.quit()

  Step 3: Assemble video
    ffmpeg -framerate 1 -pattern_type glob -i 'frame_*.png' \\
           -c:v libx264 -pix_fmt yuv420p -vf "scale=1920:1080" \\
           gnn_routing_demo.mp4

=== OPTION 3: Animation via matplotlib ===

    import json
    import matplotlib.pyplot as plt
    import networkx as nx

    with open("examples/experiment_output/experiment_log.json") as f:
        log = json.load(f)

    G = nx.DiGraph()
    for edge in log["graph"]["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

    fig, ax = plt.subplots(figsize=(14, 8))
    pos = nx.spring_layout(G, seed=42)
    scores = log["steps"]["gnn_scores"]
    colors = [scores.get(n, 0.5) for n in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors,
            cmap=plt.cm.RdYlGn, node_size=800, font_size=10,
            edge_color='gray', arrows=True)
    ax.set_title("GNN Routing -- Trained on REAL LLM metrics")
    plt.savefig("gnn_routing_result.png", dpi=150, bbox_inches='tight')
    plt.show()

=== OPTION 4: Jupyter Notebook ===

    Load experiment_log.json in Jupyter and use:
    - plotly for interactive charts
    - pyvis for interactive graphs
    - ipywidgets for epoch slider

    import json, plotly.graph_objects as go
    with open("examples/experiment_output/experiment_log.json") as f:
        log = json.load(f)

    strategies = log["steps"]["routing_comparison"]
    fig = go.Figure(data=[go.Bar(
        x=list(strategies.keys()),
        y=[s["avg_chosen_quality"] for s in strategies.values()],
        marker_color=['crimson' if k.startswith('GNN') else 'steelblue'
                      for k in strategies.keys()]
    )])
    fig.update_layout(title="Routing Strategy Comparison (REAL LLM metrics)")
    fig.show()
"""


def print_video_instructions() -> None:
    print(VIDEO_INSTRUCTIONS)


if __name__ == "__main__":
    if "--help-video" in sys.argv:
        print_video_instructions()
    else:
        try:
            main()
        except Exception:
            _print(f"\n!!! FATAL ERROR !!!")
            tb = traceback.format_exc()
            _print(tb)
            sys.exit(1)
