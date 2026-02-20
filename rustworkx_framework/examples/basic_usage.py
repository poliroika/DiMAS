"""Базовый пример использования фреймворка."""

import torch

from rustworkx_framework.builder.graph_builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.core.encoder import NodeEncoder
from rustworkx_framework.execution.scheduler import build_execution_order, get_parallel_groups


def example_basic_graph():
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

    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="What is 25 * 17?",
        answer="425",
        include_task_node=True,
    )


def example_dynamic_topology():
    graph = example_basic_graph()

    graph.add_edge("math_solver", "code_writer", weight=0.8)

    graph.remove_edge("math_solver", "code_writer")

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

    return graph


def example_execution_order():
    agents = [
        AgentProfile(agent_id="a", display_name="Agent A"),
        AgentProfile(agent_id="b", display_name="Agent B"),
        AgentProfile(agent_id="c", display_name="Agent C"),
        AgentProfile(agent_id="d", display_name="Agent D"),
    ]

    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        include_task_node=False,
    )

    agent_ids = [a.agent_id for a in agents]
    build_execution_order(graph.A_com, agent_ids)

    get_parallel_groups(graph.A_com, agent_ids)

    return graph


def example_decentralized_state():
    agent = AgentProfile(
        agent_id="assistant",
        display_name="Assistant",
        state=[{"role": "system", "content": "You are a helpful assistant."}],
    )

    agent = agent.append_state({"role": "user", "content": "Hello!"})

    agent = agent.append_state({"role": "assistant", "content": "Hi there!"})

    agent.clear_state()

    return agent


def example_embeddings():
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

    agents_with_emb = [a.with_embedding(embeddings[i]) for i, a in enumerate(agents)]

    return build_property_graph(
        agents_with_emb,
        workflow_edges=[("solver", "writer")],
        include_task_node=False,
    )


def example_pyg_conversion():
    graph = example_embeddings()

    graph.to_pyg_data()


def example_subgraph():
    graph = example_basic_graph()

    return graph.subgraph(["math_solver", "checker"])


def main():
    example_basic_graph()
    example_dynamic_topology()
    example_execution_order()
    example_decentralized_state()
    example_embeddings()
    example_pyg_conversion()
    example_subgraph()


if __name__ == "__main__":
    main()
