"""Базовый пример использования фреймворка."""

import torch

from rustworkx_framework.builder.graph_builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.core.encoder import NodeEncoder
from rustworkx_framework.execution.scheduler import build_execution_order, get_parallel_groups


def example_basic_graph():
    agents = [
        AgentProfile(
            identifier="math_solver",
            display_name="Math Solver",
            description="Solves mathematical problems step by step",
            tools=["calculator"],
        ),
        AgentProfile(
            identifier="code_writer",
            display_name="Code Writer",
            description="Writes Python code to solve problems",
            tools=["python", "code_execution"],
        ),
        AgentProfile(
            identifier="checker",
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

    print("=== Basic Graph ===")
    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.num_edges}")
    print(f"Agents: {[a.identifier for a in graph.agents]}")
    print(f"Adjacency matrix shape: {graph.A_com.shape}")

    return graph


def example_dynamic_topology():
    graph = example_basic_graph()

    print("\n=== Dynamic Topology ===")
    print(f"Initial edges: {graph.num_edges}")

    success = graph.add_edge("math_solver", "code_writer", weight=0.8)
    print(f"Added edge math_solver -> code_writer: {success}")
    print(f"Edges after add: {graph.num_edges}")

    success = graph.remove_edge("math_solver", "code_writer")
    print(f"Removed edge: {success}")
    print(f"Edges after remove: {graph.num_edges}")

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
    print(f"Edges after matrix update: {graph.num_edges}")

    return graph


def example_execution_order():
    agents = [
        AgentProfile(identifier="a", display_name="Agent A"),
        AgentProfile(identifier="b", display_name="Agent B"),
        AgentProfile(identifier="c", display_name="Agent C"),
        AgentProfile(identifier="d", display_name="Agent D"),
    ]

    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        include_task_node=False,
    )

    print("\n=== Execution Order ===")

    agent_ids = [a.identifier for a in agents]
    order = build_execution_order(graph.A_com, agent_ids)
    print(f"Execution order: {order}")

    groups = get_parallel_groups(graph.A_com, agent_ids)
    print(f"Parallel groups: {groups}")

    return graph


def example_decentralized_state():
    print("\n=== Decentralized State ===")

    agent = AgentProfile(
        identifier="assistant",
        display_name="Assistant",
        state=[{"role": "system", "content": "You are a helpful assistant."}],
    )

    print(f"Initial state: {agent.state}")

    agent = agent.append_state({"role": "user", "content": "Hello!"})
    print(f"After user message: {len(agent.state)} messages")

    agent = agent.append_state({"role": "assistant", "content": "Hi there!"})
    print(f"After assistant response: {len(agent.state)} messages")

    clean_agent = agent.clear_state()
    print(f"After clear: {len(clean_agent.state)} messages")
    print(f"Original still has: {len(agent.state)} messages")

    return agent


def example_embeddings():
    print("\n=== Embeddings ===")

    encoder = NodeEncoder(model_name="hash:128")

    agents = [
        AgentProfile(
            identifier="solver",
            display_name="Math Solver",
            description="Expert in mathematics and calculations",
        ),
        AgentProfile(
            identifier="writer",
            display_name="Code Writer",
            description="Expert in Python programming",
        ),
    ]

    texts = [a.to_text() for a in agents]
    embeddings = encoder.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dim: {encoder.embedding_dim}")

    agents_with_emb = [a.with_embedding(embeddings[i]) for i, a in enumerate(agents)]

    graph = build_property_graph(
        agents_with_emb,
        workflow_edges=[("solver", "writer")],
        include_task_node=False,
    )

    print(f"Graph embeddings shape: {graph.embeddings.shape}")

    return graph


def example_pyg_conversion():
    print("\n=== PyG Conversion ===")

    graph = example_embeddings()

    pyg_data = graph.to_pyg_data()
    print("PyG Data:")
    print(f"  x (node features): {pyg_data.x.shape}")
    print(f"  edge_index: {pyg_data.edge_index.shape}")
    print(f"  edge_attr: {pyg_data.edge_attr.shape}")
    print(f"  num_nodes: {pyg_data.num_nodes}")


def example_subgraph():
    print("\n=== Subgraph ===")

    graph = example_basic_graph()

    subgraph = graph.subgraph(["math_solver", "checker"])

    print(f"Original graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Subgraph: {subgraph.num_nodes} nodes, {subgraph.num_edges} edges")
    print(f"Subgraph agents: {[a.identifier for a in subgraph.agents if hasattr(a, 'identifier')]}")

    return subgraph


def main():
    example_basic_graph()
    example_dynamic_topology()
    example_execution_order()
    example_decentralized_state()
    example_embeddings()
    example_pyg_conversion()
    example_subgraph()

    print("\n=== All examples completed! ===")


if __name__ == "__main__":
    main()
