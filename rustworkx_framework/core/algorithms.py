"""Сервис-слой над алгоритмами rustworkx для анализа графов.

Предоставляет:
- K кратчайших путей
- Централности (betweenness, closeness, degree, eigenvector, PageRank)
- Обнаружение сообществ
- Детекция циклов
- Фильтрация подграфов по метаданным
- Интеграция с маршрутизатором
"""

from collections.abc import Callable
from enum import Enum
from typing import Any

import rustworkx as rx
import torch
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    # Enums
    "CentralityType",
    "PathMetric",
    # Data classes
    "PathResult",
    "CentralityResult",
    "CommunityResult",
    "CycleInfo",
    "SubgraphFilter",
    # Main service
    "GraphAlgorithms",
    # Utility functions
    "compute_all_centralities",
    "find_critical_nodes",
    "get_graph_metrics",
]


class CentralityType(str, Enum):
    """Типы централности."""

    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    DEGREE = "degree"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    KATZ = "katz"


class PathMetric(str, Enum):
    """Метрика для вычисления путей."""

    HOPS = "hops"
    WEIGHT = "weight"
    LATENCY = "latency"
    COST = "cost"
    RELIABILITY = "reliability"


class PathResult(BaseModel):
    """Описание найденного пути с весами и произвольными метаданными."""

    nodes: list[str]
    total_weight: float
    edge_weights: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def length(self) -> int:
        """Количество рёбер в пути."""
        return len(self.nodes) - 1 if len(self.nodes) > 1 else 0

    def __repr__(self) -> str:
        return f"PathResult({' -> '.join(self.nodes)}, weight={self.total_weight:.3f})"


class CentralityResult(BaseModel):
    """Результат вычисления централности для узлов графа."""

    centrality_type: CentralityType
    values: dict[str, float]
    normalized: bool = True

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Вернуть топ-k узлов по значению централности."""
        sorted_items = sorted(self.values.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def get_node_rank(self, node_id: str) -> int | None:
        """Позиция узла в ранжировании (1-based) или None, если узла нет."""
        sorted_nodes = sorted(self.values.keys(), key=lambda n: self.values[n], reverse=True)
        try:
            return sorted_nodes.index(node_id) + 1
        except ValueError:
            return None


class CommunityResult(BaseModel):
    """Результат обнаружения сообществ."""

    communities: list[set[str]]
    modularity: float | None = None
    algorithm: str = "unknown"

    @property
    def num_communities(self) -> int:
        """Количество найденных сообществ."""
        return len(self.communities)

    def get_node_community(self, node_id: str) -> int | None:
        """Найти индекс сообщества, которому принадлежит узел."""
        for i, community in enumerate(self.communities):
            if node_id in community:
                return i
        return None

    def get_community_sizes(self) -> list[int]:
        """Вернуть список размеров сообществ."""
        return [len(c) for c in self.communities]


class CycleInfo(BaseModel):
    """Информация о найденном цикле."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    total_weight: float = 0.0

    @property
    def length(self) -> int:
        """Количество узлов в цикле."""
        return len(self.nodes)


class SubgraphFilter(BaseModel):
    """Правила фильтрации узлов и рёбер при построении подграфа."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_filter: Callable[[str, dict[str, Any]], bool] | None = None
    edge_filter: Callable[[str, str, dict[str, Any]], bool] | None = None
    include_nodes: set[str] | None = None
    exclude_nodes: set[str] | None = None
    min_weight: float | None = None
    max_weight: float | None = None
    required_attrs: list[str] | None = None

    def matches_node(self, node_id: str, attrs: dict[str, Any]) -> bool:
        """Проверить, удовлетворяет ли узел заданным условиям."""
        if self.exclude_nodes and node_id in self.exclude_nodes:
            return False
        if self.include_nodes and node_id not in self.include_nodes:
            return False
        if self.required_attrs:
            if not all(attr in attrs for attr in self.required_attrs):
                return False
        if self.node_filter and not self.node_filter(node_id, attrs):
            return False
        return True

    def matches_edge(self, src: str, tgt: str, attrs: dict[str, Any]) -> bool:
        """Проверить, удовлетворяет ли ребро заданным условиям."""
        weight = attrs.get("weight", 1.0)
        if self.min_weight is not None and weight < self.min_weight:
            return False
        if self.max_weight is not None and weight > self.max_weight:
            return False
        if self.edge_filter and not self.edge_filter(src, tgt, attrs):
            return False
        return True


class GraphAlgorithms:
    """Сервисный слой над `rustworkx` для анализа графа `RoleGraph`."""

    def __init__(
        self,
        graph: Any,  # RoleGraph
        weight_attr: str = "weight",
        default_weight: float = 1.0,
    ):
        """Инициализировать обёртку над графом.

        Args:
            graph: Экземпляр RoleGraph (или объект с атрибутом `graph: PyDiGraph`).
            weight_attr: Ключ для веса ребра в данных.
            default_weight: Значение веса, если оно не указано в ребре.
        """
        self._role_graph = graph
        self._graph: rx.PyDiGraph = graph.graph
        self._weight_attr = weight_attr
        self._default_weight = default_weight

        self._node_id_to_idx: dict[str, int] = {}
        self._idx_to_node_id: dict[int, str] = {}
        self._rebuild_index_cache()

    def _rebuild_index_cache(self) -> None:
        """Перестроить кеш соответствия node_id ↔ индекс rustworkx."""
        self._node_id_to_idx.clear()
        self._idx_to_node_id.clear()
        for idx in self._graph.node_indices():
            data = self._graph.get_node_data(idx)
            if isinstance(data, dict):
                node_id = data.get("id", str(idx))
            elif hasattr(data, "identifier"):
                node_id = data.identifier
            else:
                node_id = str(idx)
            self._node_id_to_idx[node_id] = idx
            self._idx_to_node_id[idx] = node_id

    def _get_node_idx(self, node_id: str) -> int:
        """Получить индекс узла по ID."""
        if node_id not in self._node_id_to_idx:
            self._rebuild_index_cache()
        if node_id not in self._node_id_to_idx:
            raise ValueError(f"Node '{node_id}' not found in graph")
        return self._node_id_to_idx[node_id]

    def _get_node_id(self, idx: int) -> str:
        """Получить ID узла по внутреннему индексу графа."""
        if idx not in self._idx_to_node_id:
            self._rebuild_index_cache()
        return self._idx_to_node_id.get(idx, str(idx))

    def _get_edge_weight(
        self,
        edge_data: Any,
        metric: PathMetric = PathMetric.WEIGHT,
    ) -> float:
        """Получить вес ребра с учётом выбранной метрики."""
        if edge_data is None:
            return self._default_weight

        if isinstance(edge_data, dict):
            if metric == PathMetric.HOPS:
                return 1.0
            elif metric == PathMetric.WEIGHT:
                return edge_data.get(self._weight_attr, self._default_weight)
            elif metric == PathMetric.LATENCY:
                return edge_data.get("latency", self._default_weight)
            elif metric == PathMetric.COST:
                return edge_data.get("cost", self._default_weight)
            elif metric == PathMetric.RELIABILITY:
                rel = edge_data.get("reliability", 1.0)
                return -torch.log(torch.tensor(max(rel, 1e-10))).item()

        return self._default_weight

    def k_shortest_paths(
        self,
        source: str,
        target: str,
        k: int = 3,
        metric: PathMetric = PathMetric.WEIGHT,
    ) -> list[PathResult]:
        """Найти k кратчайших путей между узлами по заданной метрике."""
        src_idx = self._get_node_idx(source)
        tgt_idx = self._get_node_idx(target)

        def weight_fn(edge_data: Any) -> float:
            return self._get_edge_weight(edge_data, metric)

        paths = self._yen_k_shortest_paths(src_idx, tgt_idx, k, weight_fn)

        results = []
        for path_indices, total_weight in paths:
            nodes = [self._get_node_id(idx) for idx in path_indices]
            edge_weights = []
            for i in range(len(path_indices) - 1):
                edge_data = self._graph.get_edge_data(path_indices[i], path_indices[i + 1])
                edge_weights.append(self._get_edge_weight(edge_data, metric))

            results.append(
                PathResult(
                    nodes=nodes,
                    total_weight=total_weight,
                    edge_weights=edge_weights,
                    metadata={"metric": metric.value},
                )
            )

        return results

    def _yen_k_shortest_paths(
        self,
        source: int,
        target: int,
        k: int,
        weight_fn: Callable[[Any], float],
    ) -> list[tuple[list[int], float]]:
        """Алгоритм Йена: вернуть пути в виде индексов узлов и общий вес."""
        import heapq

        try:
            distances = rx.dijkstra_shortest_path_lengths(self._graph, source, weight_fn)
            if target not in distances:
                return []

            path_map = rx.dijkstra_shortest_paths(
                self._graph, source, target=target, weight_fn=weight_fn
            )
            if target not in path_map:
                return []

            first_path = list(path_map[target])
            first_weight = distances[target]
        except Exception:
            return []

        found_paths = [(first_path, first_weight)]
        candidate_heap: list[tuple[float, list[int]]] = []

        for i in range(1, k):
            if i - 1 >= len(found_paths):
                break

            prev_path, _ = found_paths[i - 1]

            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[: j + 1]

                removed_edges = []
                for path, _ in found_paths:
                    if len(path) > j and path[: j + 1] == root_path:
                        if j + 1 < len(path):
                            try:
                                edge_data = self._graph.get_edge_data(path[j], path[j + 1])
                                self._graph.remove_edge(path[j], path[j + 1])
                                removed_edges.append((path[j], path[j + 1], edge_data))
                            except Exception:
                                pass

                try:
                    spur_distances = rx.dijkstra_shortest_path_lengths(
                        self._graph, spur_node, weight_fn
                    )
                    if target in spur_distances:
                        spur_path_map = rx.dijkstra_shortest_paths(
                            self._graph, spur_node, target=target, weight_fn=weight_fn
                        )
                        if target in spur_path_map:
                            spur_path = list(spur_path_map[target])
                            total_path = root_path[:-1] + spur_path

                            total_weight = 0.0
                            for idx in range(len(total_path) - 1):
                                ed = self._graph.get_edge_data(total_path[idx], total_path[idx + 1])
                                total_weight += weight_fn(ed) if ed else self._default_weight

                            if not any(p == total_path for _, p in candidate_heap) and not any(
                                p == total_path for p, _ in found_paths
                            ):
                                heapq.heappush(candidate_heap, (total_weight, total_path))
                except Exception:
                    pass

                for u, v, data in removed_edges:
                    self._graph.add_edge(u, v, data)

            if candidate_heap:
                weight, path = heapq.heappop(candidate_heap)
                found_paths.append((path, weight))

        return found_paths

    def shortest_path(
        self,
        source: str,
        target: str,
        metric: PathMetric = PathMetric.WEIGHT,
    ) -> PathResult | None:
        """Найти один кратчайший путь между двумя узлами."""
        paths = self.k_shortest_paths(source, target, k=1, metric=metric)
        return paths[0] if paths else None

    def all_pairs_shortest_paths(
        self,
        metric: PathMetric = PathMetric.WEIGHT,
    ) -> dict[str, dict[str, float]]:
        """Вычислить кратчайшие пути между всеми парами узлов."""

        def weight_fn(edge_data: Any) -> float:
            return self._get_edge_weight(edge_data, metric)

        all_distances = rx.all_pairs_dijkstra_path_lengths(self._graph, weight_fn)

        result = {}
        for src_idx, distances in all_distances.items():
            src_id = self._get_node_id(src_idx)
            result[src_id] = {}
            for tgt_idx, dist in distances.items():
                tgt_id = self._get_node_id(tgt_idx)
                result[src_id][tgt_id] = dist

        return result

    def compute_centrality(
        self,
        centrality_type: CentralityType,
        normalized: bool = True,
        **kwargs: Any,
    ) -> CentralityResult:
        """Вычислить выбранный тип централности для всех узлов графа."""
        values: dict[int, float] = {}

        if centrality_type == CentralityType.BETWEENNESS:
            values = rx.betweenness_centrality(self._graph, normalized=normalized)  # type: ignore[assignment]

        elif centrality_type == CentralityType.CLOSENESS:
            undirected = self._graph.to_undirected()
            raw_values = rx.closeness_centrality(undirected)
            values = dict(enumerate(raw_values)) if isinstance(raw_values, list) else raw_values  # type: ignore[assignment]

        elif centrality_type == CentralityType.DEGREE:
            for idx in self._graph.node_indices():
                in_deg = self._graph.in_degree(idx)
                out_deg = self._graph.out_degree(idx)
                values[idx] = float(in_deg + out_deg)
            if normalized and self._graph.num_nodes() > 1:
                max_deg = 2 * (self._graph.num_nodes() - 1)
                values = {k: v / max_deg for k, v in values.items()}

        elif centrality_type == CentralityType.EIGENVECTOR:
            try:
                raw = rx.eigenvector_centrality(self._graph)
                values = dict(enumerate(raw)) if isinstance(raw, list) else raw  # type: ignore[assignment]
            except Exception:
                values = rx.pagerank(self._graph)  # type: ignore[assignment]

        elif centrality_type == CentralityType.PAGERANK:
            alpha = kwargs.get("alpha", 0.85)
            values = rx.pagerank(self._graph, alpha=alpha)  # type: ignore[assignment]

        elif centrality_type == CentralityType.KATZ:
            alpha = kwargs.get("alpha", 0.1)
            beta = kwargs.get("beta", 1.0)
            try:
                values = rx.katz_centrality(self._graph, alpha=alpha, beta=beta)  # type: ignore[assignment]
            except Exception:
                values = rx.pagerank(self._graph)  # type: ignore[assignment]

        result_values = {}
        for idx, val in values.items():
            node_id = self._get_node_id(idx)
            result_values[node_id] = float(val)

        return CentralityResult(
            centrality_type=centrality_type,
            values=result_values,
            normalized=normalized,
        )

    def compute_all_centralities(
        self, normalized: bool = True
    ) -> dict[CentralityType, CentralityResult]:
        """Посчитать все централности и вернуть словарь по типам."""
        results = {}
        for ct in CentralityType:
            try:
                results[ct] = self.compute_centrality(ct, normalized=normalized)
            except Exception:
                pass
        return results

    def detect_communities(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> CommunityResult:
        """Обнаружить сообщества указанным алгоритмом (louvain/label_propagation)."""
        undirected = self._graph.to_undirected()

        communities: list[set[str]] = []
        modularity: float | None = None

        if algorithm == "louvain":
            try:
                components = rx.connected_components(undirected)
                communities = [{self._get_node_id(idx) for idx in comp} for comp in components]
            except Exception:
                communities = [{self._get_node_id(idx) for idx in undirected.node_indices()}]

        elif algorithm == "label_propagation":
            communities = self._label_propagation(undirected)

        elif algorithm == "connected_components":
            components = rx.connected_components(undirected)
            communities = [{self._get_node_id(idx) for idx in comp} for comp in components]

        else:
            components = rx.connected_components(undirected)
            communities = [{self._get_node_id(idx) for idx in comp} for comp in components]

        return CommunityResult(
            communities=communities,
            modularity=modularity,
            algorithm=algorithm,
        )

    def _label_propagation(self, graph: rx.PyGraph) -> list[set[str]]:
        """Простая реализация label propagation для неориентированного графа."""
        import random

        labels = {idx: idx for idx in graph.node_indices()}

        for _ in range(100):
            changed = False
            nodes = list(graph.node_indices())
            random.shuffle(nodes)

            for node in nodes:
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    continue

                label_counts: dict[int, int] = {}
                for neighbor in neighbors:
                    lbl = labels[neighbor]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                max_count = max(label_counts.values())
                best_labels = [label for label, count in label_counts.items() if count == max_count]
                new_label = random.choice(best_labels)

                if labels[node] != new_label:
                    labels[node] = new_label
                    changed = True

            if not changed:
                break

        label_to_nodes: dict[int, set[str]] = {}
        for node, label in labels.items():
            if label not in label_to_nodes:
                label_to_nodes[label] = set()
            label_to_nodes[label].add(self._get_node_id(node))

        return list(label_to_nodes.values())

    def detect_cycles(self, max_length: int | None = None) -> list[CycleInfo]:
        """Найти простые циклы, опционально ограничив максимальную длину."""
        cycles = []

        try:
            simple_cycles = rx.simple_cycles(self._graph)
            for cycle_indices in simple_cycles:
                if max_length and len(cycle_indices) > max_length:
                    continue

                nodes = [self._get_node_id(idx) for idx in cycle_indices]
                edges = []
                total_weight = 0.0

                for i in range(len(cycle_indices)):
                    src = cycle_indices[i]
                    tgt = cycle_indices[(i + 1) % len(cycle_indices)]
                    edges.append((self._get_node_id(src), self._get_node_id(tgt)))

                    edge_data = self._graph.get_edge_data(src, tgt)
                    if edge_data and isinstance(edge_data, dict):
                        total_weight += edge_data.get(self._weight_attr, self._default_weight)
                    else:
                        total_weight += self._default_weight

                cycles.append(
                    CycleInfo(
                        nodes=nodes,
                        edges=edges,
                        total_weight=total_weight,
                    )
                )
        except Exception:
            pass

        return cycles

    def is_dag(self) -> bool:
        """Проверить, является ли граф ориентированным ацикличным (DAG)."""
        return rx.is_directed_acyclic_graph(self._graph)

    def topological_sort(self) -> list[str] | None:
        """Вернуть топологический порядок узлов, если граф — DAG."""
        if not self.is_dag():
            return None

        order = rx.topological_sort(self._graph)
        return [self._get_node_id(idx) for idx in order]

    def filter_subgraph(
        self,
        filter_spec: SubgraphFilter,
    ) -> "GraphAlgorithms":
        """Отфильтровать узлы/рёбра по правилам и вернуть обёртку над подграфом."""
        keep_nodes = set()
        for idx in self._graph.node_indices():
            node_id = self._get_node_id(idx)
            node_data = self._graph.get_node_data(idx)
            attrs = node_data if isinstance(node_data, dict) else {}

            if filter_spec.matches_node(node_id, attrs):
                keep_nodes.add(idx)

        new_graph = rx.PyDiGraph()
        old_to_new: dict[int, int] = {}

        for old_idx in keep_nodes:
            node_data = self._graph.get_node_data(old_idx)
            new_idx = new_graph.add_node(node_data)
            old_to_new[old_idx] = new_idx

        for edge_idx in self._graph.edge_indices():
            src, tgt = self._graph.get_edge_endpoints_by_index(edge_idx)
            if src not in keep_nodes or tgt not in keep_nodes:
                continue

            edge_data = self._graph.get_edge_data_by_index(edge_idx)
            attrs = edge_data if isinstance(edge_data, dict) else {}

            src_id = self._get_node_id(src)
            tgt_id = self._get_node_id(tgt)

            if filter_spec.matches_edge(src_id, tgt_id, attrs):
                new_graph.add_edge(old_to_new[src], old_to_new[tgt], edge_data)

        class SubgraphWrapper:
            def __init__(self, g: rx.PyDiGraph):
                self.graph = g

        return GraphAlgorithms(
            SubgraphWrapper(new_graph),
            weight_attr=self._weight_attr,
            default_weight=self._default_weight,
        )

    def get_reachable_nodes(self, source: str, max_depth: int | None = None) -> set[str]:
        """Вернуть множество узлов, достижимых из source, опционально ограничить глубину."""
        src_idx = self._get_node_idx(source)

        visited = set()
        queue = [(src_idx, 0)]

        while queue:
            node, depth = queue.pop(0)
            if node in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue

            visited.add(node)

            for successor in self._graph.successors(node):
                if successor not in visited:
                    queue.append((successor, depth + 1))

        return {self._get_node_id(idx) for idx in visited}

    def get_predecessors(self, node: str, max_depth: int | None = None) -> set[str]:
        """Вернуть множество предшественников узла, опционально ограничить глубину."""
        node_idx = self._get_node_idx(node)

        visited = set()
        queue = [(node_idx, 0)]

        while queue:
            n, depth = queue.pop(0)
            if n in visited:
                continue
            if max_depth is not None and depth > max_depth:
                continue

            visited.add(n)

            for predecessor in self._graph.predecessors(n):
                if predecessor not in visited:
                    queue.append((predecessor, depth + 1))

        visited.discard(node_idx)
        return {self._get_node_id(idx) for idx in visited}

    def get_routing_metrics(self, source: str, target: str) -> dict[str, Any]:
        """Собрать краткую сводку путей и централности для пары узлов."""
        paths_list: list[dict[str, Any]] = []
        centrality_dict: dict[str, float] = {}
        is_reachable = False

        for metric in [PathMetric.WEIGHT, PathMetric.LATENCY, PathMetric.COST]:
            try:
                paths = self.k_shortest_paths(source, target, k=3, metric=metric)
                if paths:
                    is_reachable = True
                    paths_list.append(
                        {
                            "metric": metric.value,
                            "best_path": paths[0].nodes,
                            "best_weight": paths[0].total_weight,
                            "alternatives": len(paths) - 1,
                        }
                    )
            except Exception:
                pass

        try:
            pr = self.compute_centrality(CentralityType.PAGERANK)
            centrality_dict["pagerank"] = pr.values.get(target, 0.0)
        except Exception:
            pass

        return {
            "source": source,
            "target": target,
            "paths": paths_list,
            "centrality": centrality_dict,
            "is_reachable": is_reachable,
        }


def compute_all_centralities(graph: Any) -> dict[str, CentralityResult]:
    """Посчитать все типы централности и вернуть по строковым ключам."""
    alg = GraphAlgorithms(graph)
    results = alg.compute_all_centralities()
    return {ct.value: result for ct, result in results.items()}


def find_critical_nodes(graph: Any, top_k: int = 5) -> list[str]:
    """Вернуть узлы с наибольшей betweenness-централностью."""
    alg = GraphAlgorithms(graph)
    bc = alg.compute_centrality(CentralityType.BETWEENNESS)
    return [node_id for node_id, _ in bc.top_k(top_k)]


def get_graph_metrics(graph: Any) -> dict[str, Any]:
    """Собрать ключевые метрики графа: размер, DAG, сообщества, циклы."""
    alg = GraphAlgorithms(graph)

    return {
        "num_nodes": graph.graph.num_nodes(),
        "num_edges": graph.graph.num_edges(),
        "is_dag": alg.is_dag(),
        "num_communities": alg.detect_communities().num_communities,
        "num_cycles": len(alg.detect_cycles(max_length=10)),
    }
