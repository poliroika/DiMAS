"""Pytest fixtures and configuration."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rustworkx as rx


@pytest.fixture
def empty_digraph():
    """Пустой directed graph."""
    return rx.PyDiGraph()


@pytest.fixture
def simple_digraph():
    """Простой directed graph с 3 узлами."""
    g = rx.PyDiGraph()
    g.add_node({"id": "a", "role": "agent"})
    g.add_node({"id": "b", "role": "agent"})
    g.add_node({"id": "c", "role": "agent"})
    g.add_edge(0, 1, {"weight": 0.8})
    g.add_edge(1, 2, {"weight": 0.9})
    return g


@pytest.fixture
def cyclic_digraph():
    """Граф с циклом."""
    g = rx.PyDiGraph()
    g.add_node({"id": "a"})
    g.add_node({"id": "b"})
    g.add_node({"id": "c"})
    g.add_edge(0, 1, {"weight": 1.0})
    g.add_edge(1, 2, {"weight": 1.0})
    g.add_edge(2, 0, {"weight": 1.0})  # Cycle
    return g


@pytest.fixture
def parallel_digraph():
    """Граф с параллельными ветками."""
    g = rx.PyDiGraph()
    # Task -> A, B (parallel) -> C
    g.add_node({"id": "task", "role": "task"})
    g.add_node({"id": "a", "role": "agent"})
    g.add_node({"id": "b", "role": "agent"})
    g.add_node({"id": "c", "role": "agent"})
    g.add_edge(0, 1, {"weight": 0.5})
    g.add_edge(0, 2, {"weight": 0.5})
    g.add_edge(1, 3, {"weight": 0.8})
    g.add_edge(2, 3, {"weight": 0.8})
    return g


@pytest.fixture
def sample_agent_configs():
    """Примеры конфигураций агентов."""
    return [
        {
            "identifier": "coordinator",
            "role_name": "Coordinator",
            "persona": "Manages workflow",
            "state": {},
        },
        {
            "identifier": "researcher",
            "role_name": "Researcher",
            "persona": "Finds information",
            "state": {},
        },
        {
            "identifier": "writer",
            "role_name": "Writer",
            "persona": "Creates content",
            "state": {},
        },
    ]


@pytest.fixture
def sample_connections():
    """Примеры связей между агентами."""
    return {
        "coordinator": ["researcher", "writer"],
        "researcher": ["writer"],
        "writer": [],
    }


@pytest.fixture
def mock_agent_callable():
    """Mock callable для агента."""

    async def agent_fn(query: str, context: dict) -> str:
        return f"Response to: {query}"

    return agent_fn


@pytest.fixture
def mock_failing_agent():
    """Mock агент, который всегда падает."""

    async def failing_agent(query: str, context: dict) -> str:
        raise RuntimeError("Agent failure")

    return failing_agent


@pytest.fixture
def mock_slow_agent():
    """Mock медленный агент."""
    import asyncio

    async def slow_agent(query: str, context: dict) -> str:
        await asyncio.sleep(5.0)  # Намеренно медленный
        return "Slow response"

    return slow_agent


def create_role_graph(node_ids, connections):
    """Helper для создания RoleGraph."""
    from rustworkx_framework.core.agent import AgentProfile
    from rustworkx_framework.core.graph import RoleGraph

    g = rx.PyDiGraph()

    id_to_idx = {}
    agents = []
    for nid in node_ids:
        idx = g.add_node({"id": nid})
        id_to_idx[nid] = idx
        agent = AgentProfile(identifier=nid, display_name=f"Agent {nid.upper()}")
        agents.append(agent)

    for src, targets in connections.items():
        for tgt in targets:
            if src in id_to_idx and tgt in id_to_idx:
                g.add_edge(id_to_idx[src], id_to_idx[tgt], {"weight": 1.0})

    n = len(node_ids)
    a_com = torch.zeros((n, n), dtype=torch.float32)
    for src, targets in connections.items():
        i = node_ids.index(src)
        for tgt in targets:
            if tgt in node_ids:
                j = node_ids.index(tgt)
                a_com[i, j] = 1.0

    graph = RoleGraph(
        node_ids=node_ids,
        role_connections=connections,
        graph=g,
        A_com=a_com,
    )
    graph.agents = agents
    return graph
