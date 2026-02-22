"""
Basic framework usage — no LLM required.

Demonstrates the core building blocks of MECE:
  1. Building a property graph from AgentProfile objects
  2. Dynamic topology changes (add / remove edges, update adjacency)
  3. Topological execution order and parallel groups
  4. Decentralised agent state management
  5. Encoding agent profiles into embeddings
  6. Converting to PyG Data for GNN training
  7. Extracting subgraphs

Run:
    python -m examples.basic_usage
"""

import torch

from rustworkx_framework.builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.core.encoder import NodeEncoder
from rustworkx_framework.execution import build_execution_order, get_parallel_groups

# ── Helpers ──────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")


# ── 1. Building a property graph ────────────────────────────────────────────


def example_basic_graph():
    """Create a three-agent graph with a shared task node."""
    agents = [
        AgentProfile(
            agent_id="math_solver",
            display_name="Math Solver",
            description="Solves mathematical problems step by step",
            tools=["calculator"],
        ),
        AgentProfile(
            agent_id="code_writer",
            display_name="Code Writer",
            description="Writes Python code to solve problems",
            tools=["python", "code_execution"],
        ),
        AgentProfile(
            agent_id="checker",
            display_name="Answer Checker",
            description="Verifies the correctness of solutions",
        ),
    ]

    edges = [("math_solver", "checker"), ("code_writer", "checker")]

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        query="What is 25 * 17?",
        answer="425",
        include_task_node=True,
    )

    print(f"  Agents : {[a.agent_id for a in agents]}")
    print(f"  Edges  : {edges}")
    print(f"  Query  : {graph.query}")
    return graph


# ── 2. Dynamic topology ─────────────────────────────────────────────────────


def example_dynamic_topology():
    """Add / remove edges and push a new adjacency matrix at runtime."""
    graph = example_basic_graph()

    graph.add_edge("math_solver", "code_writer", weight=0.8)
    print("  + edge math_solver → code_writer (weight 0.8)")

    graph.remove_edge("math_solver", "code_writer")
    print("  − edge math_solver → code_writer removed")

    new_adj = torch.tensor(
        [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0]],
        dtype=torch.float32,
    )
    graph.update_communication(new_adj)
    print(f"  Adjacency matrix updated — shape {new_adj.shape}")
    return graph


# ── 3. Execution order ──────────────────────────────────────────────────────


def example_execution_order():
    """Compute topological order and parallel groups for a diamond DAG."""
    agents = [
        AgentProfile(agent_id="a", display_name="Agent A"),
        AgentProfile(agent_id="b", display_name="Agent B"),
        AgentProfile(agent_id="c", display_name="Agent C"),
        AgentProfile(agent_id="d", display_name="Agent D"),
    ]
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]

    graph = build_property_graph(agents, workflow_edges=edges, include_task_node=False)
    ids = [a.agent_id for a in agents]

    order = build_execution_order(graph.A_com, ids)
    groups = get_parallel_groups(graph.A_com, ids)

    print(f"  Order  : {order}")
    print(f"  Groups : {groups}")
    print("  → b and c can run in parallel before d")
    return graph


# ── 4. Decentralised state ──────────────────────────────────────────────────


def example_decentralised_state():
    """Immutable append / clear on an AgentProfile's local state."""
    agent = AgentProfile(
        agent_id="assistant",
        display_name="Assistant",
        state=[{"role": "system", "content": "You are a helpful assistant."}],
    )

    agent = agent.append_state({"role": "user", "content": "Hello!"})
    agent = agent.append_state({"role": "assistant", "content": "Hi there!"})
    print(f"  After 2 turns : {len(agent.state)} messages")

    cleared = agent.clear_state()
    print(f"  After clear   : {len(cleared.state)} messages")
    return agent


# ── 5. Embeddings ───────────────────────────────────────────────────────────


def example_embeddings():
    """Encode agent descriptions into fixed-size hash embeddings."""
    encoder = NodeEncoder(model_name="hash:128")

    agents = [
        AgentProfile(
            agent_id="solver",
            display_name="Math Solver",
            description="Expert in mathematics and calculations",
        ),
        AgentProfile(
            agent_id="writer",
            display_name="Code Writer",
            description="Expert in Python programming",
        ),
    ]

    embeddings = encoder.encode([a.to_text() for a in agents])
    print(f"  Shape: {embeddings.shape}")  # (2, 128)

    agents_emb = [a.with_embedding(embeddings[i]) for i, a in enumerate(agents)]

    return build_property_graph(
        agents_emb,
        workflow_edges=[("solver", "writer")],
        include_task_node=False,
    )


# ── 6. PyG conversion ──────────────────────────────────────────────────────


def example_pyg_conversion():
    """Convert a property graph to a PyTorch Geometric Data object."""
    graph = example_embeddings()
    data = graph.to_pyg_data()
    print(f"  Node features : {data.x.shape}")
    print(f"  Edge index    : {data.edge_index.shape}")
    return data


# ── 7. Subgraph extraction ─────────────────────────────────────────────────


def example_subgraph():
    """Extract a subgraph containing only a subset of agents."""
    graph = example_basic_graph()
    sub = graph.subgraph(["math_solver", "checker"])
    print(f"  Original : {[a.agent_id for a in graph.agents]}")
    print(f"  Subgraph : {[a.agent_id for a in sub.agents]}")
    return sub


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    examples = [
        ("1. Basic graph", example_basic_graph),
        ("2. Dynamic topology", example_dynamic_topology),
        ("3. Execution order", example_execution_order),
        ("4. Decentralised state", example_decentralised_state),
        ("5. Embeddings", example_embeddings),
        ("6. PyG conversion", example_pyg_conversion),
        ("7. Subgraph extraction", example_subgraph),
    ]

    for title, fn in examples:
        _header(title)
        fn()

    print("\nAll examples completed ✅")


if __name__ == "__main__":
    main()
