"""Тесты для core/graph.py — RoleGraph и динамические операции."""

import json
import tempfile

import pytest
import rustworkx as rx
import torch

from rustworkx_framework.core.graph import (
    GraphIntegrityError,
    RoleGraph,
    StateMigrationPolicy,
)
from rustworkx_framework.utils.state_storage import FileStateStorage, InMemoryStateStorage


class TestRoleGraphCreation:
    def test_empty_graph(self):
        graph = RoleGraph()
        assert graph.num_nodes == 0
        assert graph.num_edges == 0
        assert graph.node_ids == []

    def test_graph_with_nodes(self):
        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
        )

        assert graph.num_nodes == 2
        assert "a" in graph.node_ids
        assert "b" in graph.node_ids

    def test_graph_with_edges(self):
        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {"weight": 0.5})

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": ["b"], "b": []},
            graph=g,
        )

        assert graph.num_edges == 1
        edges = graph.edges
        assert len(edges) == 1
        assert edges[0]["source"] == "a"
        assert edges[0]["target"] == "b"


class TestAddNode:
    def test_add_node_basic(self):
        from rustworkx_framework.core.agent import AgentProfile

        graph = RoleGraph()
        agent = AgentProfile(identifier="new_agent", display_name="Agent")

        result = graph.add_node(agent)

        assert result is True
        assert "new_agent" in graph.node_ids
        assert graph.num_nodes == 1

    def test_add_node_with_connections(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            role_connections={"a": []},
            graph=g,
            A_com=torch.zeros((1, 1), dtype=torch.float32),
        )
        graph.agents = [agent_a]

        agent_b = AgentProfile(identifier="b", display_name="Agent B")
        graph.add_node(agent_b, connections_to=["a"])

        assert "b" in graph.node_ids
        assert graph.A_com.shape == (2, 2)

    def test_add_duplicate_node_raises(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
        )
        graph.agents = [agent_a]

        agent_a_dup = AgentProfile(identifier="a", display_name="Agent A")
        result = graph.add_node(agent_a_dup)

        # add_node возвращает False для дубликатов
        assert result is False

    def test_add_node_expands_matrices(self):
        from rustworkx_framework.core.agent import AgentProfile

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            A_com=torch.tensor([[0, 1], [0, 0]], dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b]
        original_shape = graph.A_com.shape

        agent_c = AgentProfile(identifier="c", display_name="Agent C")
        graph.add_node(agent_c)

        assert graph.A_com.shape[0] == original_shape[0] + 1
        assert graph.A_com.shape[1] == original_shape[1] + 1


class TestRemoveNode:
    def test_remove_node_basic(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
            A_com=torch.zeros((2, 2), dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b]

        graph.remove_node("b")

        assert "b" not in graph.node_ids
        assert graph.num_nodes == 1

    def test_remove_nonexistent_node_raises(self):
        graph = RoleGraph()

        # remove_node returns None for nonexistent nodes, doesn't raise
        result = graph.remove_node("nonexistent")
        assert result is None

    def test_remove_node_shrinks_matrices(self):
        from rustworkx_framework.core.agent import AgentProfile

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")
        agent_c = AgentProfile(identifier="c", display_name="Agent C")

        graph = RoleGraph(
            node_ids=["a", "b", "c"],
            A_com=torch.eye(3, dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b, agent_c]

        graph.remove_node("b")

        assert graph.A_com.shape == (2, 2)
        assert len(graph.node_ids) == 2

    def test_remove_node_with_discard_policy(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a", "state": {"data": "important"}})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
        )
        graph.agents = [agent_a]

        graph.remove_node("a", policy=StateMigrationPolicy.DISCARD)

        assert "a" not in graph.node_ids

    def test_remove_node_with_archive_policy(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a", "state": {"data": "important"}})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        storage = InMemoryStateStorage()
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
            state_storage=storage,
        )
        graph.agents = [agent_a]

        graph.remove_node("a", policy=StateMigrationPolicy.ARCHIVE)

        assert "a" not in graph.node_ids
        archived = storage.load("a")
        assert archived is not None


class TestReplaceNode:
    def test_replace_node_basic(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "old", "role": "agent"})

        old_agent = AgentProfile(identifier="old", display_name="Old Agent")
        graph = RoleGraph(
            node_ids=["old"],
            graph=g,
        )
        graph.agents = [old_agent]

        new_agent = AgentProfile(identifier="new", display_name="New Agent")
        graph.replace_node("old", new_agent, StateMigrationPolicy.COPY)

        assert "old" not in graph.node_ids
        assert "new" in graph.node_ids

    def test_replace_preserves_connections(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_node({"id": "c"})
        g.add_edge(0, 1, {"weight": 0.5})
        g.add_edge(1, 2, {"weight": 0.8})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")
        agent_c = AgentProfile(identifier="c", display_name="Agent C")

        graph = RoleGraph(
            node_ids=["a", "b", "c"],
            role_connections={"a": ["b"], "b": ["c"], "c": []},
            graph=g,
        )
        graph.agents = [agent_a, agent_b, agent_c]

        agent_b_new = AgentProfile(identifier="b_new", display_name="Agent B New")
        graph.replace_node("b", agent_b_new, StateMigrationPolicy.COPY)

        assert "b_new" in graph.node_ids
        assert "b_new" in graph.role_connections

    def test_replace_with_copy_policy(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "old", "state": {"key": "value"}})

        old_agent = AgentProfile(identifier="old", display_name="Old Agent")
        graph = RoleGraph(
            node_ids=["old"],
            graph=g,
        )
        graph.agents = [old_agent]

        new_agent = AgentProfile(identifier="new", display_name="New Agent")
        graph.replace_node(
            "old",
            new_agent,
            policy=StateMigrationPolicy.COPY,
        )

        assert "new" in graph.node_ids
        assert "old" not in graph.node_ids


class TestIntegrity:
    def test_verify_integrity_valid(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": ["b"], "b": []},
            graph=g,
            A_com=torch.tensor([[0, 1], [0, 0]], dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b]

        graph.verify_integrity()

    def test_is_consistent_true(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
            A_com=torch.zeros((1, 1), dtype=torch.float32),
        )
        graph.agents = [agent_a]

        assert graph.is_consistent()

    def test_verify_integrity_mismatched_counts(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
        )
        graph.agents = [agent_a]

        with pytest.raises(GraphIntegrityError):
            graph.verify_integrity()


class TestSerialization:
    def test_model_dump_basic(self):
        from rustworkx_framework.core.agent import AgentProfile

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": ["b"], "b": []},
            query="test query",
        )
        graph.agents = [agent_a, agent_b]

        data = graph.model_dump(exclude={"graph"})

        assert "node_ids" in data
        assert "role_connections" in data
        assert data["query"] == "test query"

    def test_model_dump_excludes_graph(self):
        g = rx.PyDiGraph()
        graph = RoleGraph(graph=g)

        data = graph.model_dump(exclude={"graph"})

        assert "graph" not in data

    def test_json_roundtrip(self):
        from rustworkx_framework.core.agent import AgentProfile

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": ["b"], "b": []},
            query="test",
            answer="response",
        )
        graph.agents = [agent_a, agent_b]

        json_str = graph.model_dump_json(exclude={"graph", "A_com", "S_tilde", "p_matrix"})
        data = json.loads(json_str)

        assert data["node_ids"] == ["a", "b"]
        assert data["query"] == "test"


class TestEdgeOperations:
    def test_add_edge(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": [], "b": []},
            graph=g,
            A_com=torch.zeros((2, 2), dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b]

        graph.add_edge("a", "b", weight=0.7)

        assert graph.num_edges == 1
        # Role connections are updated automatically in add_edge
        # Check via A_com instead
        assert graph.A_com[0, 1] == 0.7

    def test_remove_edge(self):
        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {"weight": 0.5})

        graph = RoleGraph(
            node_ids=["a", "b"],
            role_connections={"a": ["b"], "b": []},
            graph=g,
        )

        graph.remove_edge("a", "b")

        assert graph.num_edges == 0

    def test_update_edge_weight(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {"weight": 0.5})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
            A_com=torch.tensor([[0, 0.5], [0, 0]], dtype=torch.float32),
        )
        graph.agents = [agent_a, agent_b]

        # Remove and re-add edge with new weight
        graph.remove_edge("a", "b")
        graph.add_edge("a", "b", weight=0.9)

        # Check via edges property
        edges = graph.edges
        ab_edge = [e for e in edges if e["source"] == "a" and e["target"] == "b"]
        assert len(ab_edge) == 1
        assert ab_edge[0]["weight"] == 0.9


class TestPyGExport:
    def test_edge_index_empty(self):
        graph = RoleGraph()

        ei = graph.edge_index

        assert ei.shape == (2, 0)

    def test_edge_index_with_edges(self):
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
        )
        graph.agents = [agent_a, agent_b]

        ei = graph.edge_index

        assert ei.shape[0] == 2
        assert ei.shape[1] == 1
        assert ei[0, 0] == 0
        assert ei[1, 0] == 1

    def test_to_pyg_data(self):
        pytest.importorskip("torch_geometric", reason="torch_geometric not installed")
        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {"weight": 0.5})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
        )
        graph.agents = [agent_a, agent_b]

        data = graph.to_pyg_data()
        assert data is not None
        assert hasattr(data, "edge_index")

    def test_to_pyg_data_with_custom_features(self):
        pytest.importorskip("torch_geometric", reason="torch_geometric not installed")
        import torch

        from rustworkx_framework.core.agent import AgentProfile

        g = rx.PyDiGraph()
        g.add_node({"id": "a"})
        g.add_node({"id": "b"})
        g.add_edge(0, 1, {})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        agent_b = AgentProfile(identifier="b", display_name="Agent B")

        graph = RoleGraph(
            node_ids=["a", "b"],
            graph=g,
        )
        graph.agents = [agent_a, agent_b]

        node_features = {"custom": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)}
        edge_features = {"custom": torch.tensor([[0.5, 0.5]], dtype=torch.float32)}

        data = graph.to_pyg_data(
            node_features=node_features,
            edge_features=edge_features,
        )
        assert data.x.shape[0] == 2
        assert data.edge_attr.shape[0] == 1


class TestStateStorage:
    def test_in_memory_storage(self):
        storage = InMemoryStateStorage()

        storage.save("node1", {"key": "value"})

        assert storage.load("node1") == {"key": "value"}
        assert storage.load("nonexistent") is None

        storage.delete("node1")
        assert storage.load("node1") is None

    def test_file_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileStateStorage(tmpdir)

            storage.save("node1", {"key": "value"})

            assert storage.load("node1") == {"key": "value"}

            storage.delete("node1")
            assert storage.load("node1") is None

    def test_graph_with_storage(self):
        from rustworkx_framework.core.agent import AgentProfile

        storage = InMemoryStateStorage()

        g = rx.PyDiGraph()
        g.add_node({"id": "a", "state": {"data": 123}})

        agent_a = AgentProfile(identifier="a", display_name="Agent A")
        graph = RoleGraph(
            node_ids=["a"],
            graph=g,
            state_storage=storage,
        )
        graph.agents = [agent_a]

        graph.remove_node("a", policy=StateMigrationPolicy.ARCHIVE)

        archived = storage.load("a")
        assert archived is not None
        assert isinstance(archived.get("state", []), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
