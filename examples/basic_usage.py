"""
Example: Basic framework usage.

Shows the fundamental building blocks of the rustworkx_framework:
1. Building a property graph from AgentProfile objects
2. Dynamic topology changes (add / remove edges, update adjacency matrix)
3. Computing topological execution order and parallel groups
4. Decentralised agent state management
5. Encoding agent profiles into embeddings
6. Converting to PyG Data for GNN training
7. Extracting subgraphs

Run with:
    python -m examples.basic_usage
"""

import torch

from rustworkx_framework.builder.graph_builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.core.encoder import NodeEncoder
from rustworkx_framework.execution.scheduler import build_execution_order, get_parallel_groups


# ── 1. Building a property graph ─────────────────────────────────────────────

def example_basic_graph():
    """Create a simple three-agent graph with a shared task."""
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
            tools=[],
        ),
    ]

    edges = [
        ("math_solver", "checker"),
        ("code_writer", "checker"),
    ]

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        query="What is 25 * 17?",
        answer="425",
        include_task_node=True,
    )

    print(f"  Agents  : {[a.agent_id for a in agents]}")
    print(f"  Edges   : {edges}")
    print(f"  Query   : {graph.query}")
    return graph


# ── 2. Dynamic topology ───────────────────────────────────────────────────────

def example_dynamic_topology():
    """Add and remove edges, then push a new adjacency matrix."""
    graph = example_basic_graph()

    graph.add_edge("math_solver", "code_writer", weight=0.8)
    print("  Added edge math_solver → code_writer (weight=0.8)")

    graph.remove_edge("math_solver", "code_writer")
    print("  Removed edge math_solver → code_writer")

    # Replace the communication matrix with a new one
    new_a = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    graph.update_communication(new_a)
    print(f"  Updated adjacency matrix shape: {new_a.shape}")
    return graph


# ── 3. Execution order ────────────────────────────────────────────────────────

def example_execution_order():
    """Compute topological order and parallel groups for a diamond graph."""
    agents = [
        AgentProfile(agent_id="a", display_name="Agent A"),
        AgentProfile(agent_id="b", display_name="Agent B"),
        AgentProfile(agent_id="c", display_name="Agent C"),
        AgentProfile(agent_id="d", display_name="Agent D"),
    ]
    # Diamond: a → b, a → c, b → d, c → d
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]

    graph = build_property_graph(agents, workflow_edges=edges, include_task_node=False)
    agent_ids = [a.agent_id for a in agents]

    order = build_execution_order(graph.A_com, agent_ids)
    groups = get_parallel_groups(graph.A_com, agent_ids)

    print(f"  Execution order   : {order}")
    print(f"  Parallel groups   : {groups}")
    print(f"  (b and c can run in parallel before d)")
    return graph


# ── 4. Decentralised state ────────────────────────────────────────────────────

def example_decentralized_state():
    """Show immutable state append / clear on an AgentProfile."""
    agent = AgentProfile(
        agent_id="assistant",
        display_name="Assistant",
        state=[{"role": "system", "content": "You are a helpful assistant."}],
    )

    agent = agent.append_state({"role": "user", "content": "Hello!"})
    agent = agent.append_state({"role": "assistant", "content": "Hi there!"})
    print(f"  State after 2 turns : {len(agent.state)} messages")

    cleared = agent.clear_state()
    print(f"  State after clear   : {len(cleared.state)} messages")
    return agent


# ── 5. Embeddings ─────────────────────────────────────────────────────────────

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

    texts = [a.to_text() for a in agents]
    embeddings = encoder.encode(texts)
    print(f"  Embedding shape: {embeddings.shape}")  # (2, 128)

    agents_with_emb = [a.with_embedding(embeddings[i]) for i, a in enumerate(agents)]

    return build_property_graph(
        agents_with_emb,
        workflow_edges=[("solver", "writer")],
        include_task_node=False,
    )


# ── 6. PyG conversion ─────────────────────────────────────────────────────────

def example_pyg_conversion():
    """Convert a property graph to a PyTorch Geometric Data object."""
    graph = example_embeddings()
    pyg_data = graph.to_pyg_data()
    print(f"  PyG node features : {pyg_data.x.shape}")
    print(f"  PyG edge index    : {pyg_data.edge_index.shape}")
    return pyg_data


# ── 7. Subgraph ───────────────────────────────────────────────────────────────

def example_subgraph():
    """Extract a subgraph containing only a subset of agents."""
    graph = example_basic_graph()
    sub = graph.subgraph(["math_solver", "checker"])
    print(f"  Original agents : {[a.agent_id for a in graph.agents]}")
    print(f"  Subgraph agents : {[a.agent_id for a in sub.agents]}")
    return sub


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    sections = [
        ("1. Basic graph",         example_basic_graph),
        ("2. Dynamic topology",    example_dynamic_topology),
        ("3. Execution order",     example_execution_order),
        ("4. Decentralised state", example_decentralized_state),
        ("5. Embeddings",          example_embeddings),
        ("6. PyG conversion",      example_pyg_conversion),
        ("7. Subgraph",            example_subgraph),
    ]

    for title, fn in sections:
        print(f"\n{'─' * 45}")
        print(f"  {title}")
        print("─" * 45)
        fn()

    print("\nAll examples completed ✅")


if __name__ == "__main__":
    main()
