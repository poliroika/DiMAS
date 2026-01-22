"""Интеграция Graph Neural Networks для маршрутизации.

Предоставляет:
- Генераторы признаков узлов и рёбер
- Обёртки моделей (GCN, GAT, GraphSAGE)
- Утилиты обучения и инференса
- Сохранение/загрузка весов
- Применение к online-решениям маршрутизации
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import rustworkx as rx
import torch
import torch.nn as nn
import torch.nn.functional as functional
from pydantic import BaseModel, Field
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

TORCH_AVAILABLE = True
PYG_AVAILABLE = True


__all__ = [
    # Enums
    "GNNModelType",
    "RoutingStrategy",
    # Data classes
    "FeatureConfig",
    "TrainingConfig",
    "RoutingPrediction",
    # Feature generation
    "FeatureGenerator",
    "DefaultFeatureGenerator",
    # Models
    "BaseGNNModel",
    "GCNRouter",
    "GATRouter",
    "GraphSAGERouter",
    # Training
    "GNNTrainer",
    # Inference
    "GNNRouterInference",
    # Factory
    "create_gnn_router",
]


class GNNModelType(str, Enum):
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"


class RoutingStrategy(str, Enum):
    ARGMAX = "argmax"
    SOFTMAX_SAMPLE = "softmax_sample"
    TOP_K = "top_k"
    THRESHOLD = "threshold"


class FeatureConfig(BaseModel):
    node_feature_dim: int = 64
    edge_feature_dim: int = 16
    embedding_dim: int = 128

    use_embeddings: bool = True
    use_metrics: bool = True
    use_centrality: bool = True
    use_structural: bool = True

    normalize_features: bool = True
    clip_outliers: bool = True
    outlier_std: float = 3.0


class TrainingConfig(BaseModel):
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    gat_heads: int = 4
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    use_batch_norm: bool = True
    use_residual: bool = True
    task: Literal["node_classification", "edge_prediction", "path_ranking"] = "node_classification"
    num_classes: int = 2


class RoutingPrediction(BaseModel):
    recommended_nodes: list[str]
    scores: dict[str, float]
    confidence: float
    strategy: RoutingStrategy
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureGenerator(ABC):
    @abstractmethod
    def generate_node_features(
        self,
        graph: Any,
        node_ids: list[str],
        metrics_tracker: Any | None = None,
    ) -> torch.Tensor:
        """Сформировать матрицу признаков для списка узлов."""
        pass

    @abstractmethod
    def generate_edge_features(
        self,
        graph: Any,
        edges: list[tuple[str, str]],
        metrics_tracker: Any | None = None,
    ) -> torch.Tensor:
        """Сформировать матрицу признаков для списка рёбер."""
        pass


class DefaultFeatureGenerator(FeatureGenerator):
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def generate_node_features(
        self,
        graph: Any,
        node_ids: list[str],
        metrics_tracker: Any | None = None,
    ) -> torch.Tensor:
        """Собрать признаки узлов: эмбеддинги, метрики, структура, централность."""
        features_list = []
        if self.config.use_embeddings:
            emb_features = self._get_embedding_features(graph, node_ids)
            if emb_features is not None:
                features_list.append(emb_features)
        if self.config.use_metrics and metrics_tracker is not None:
            metric_features = metrics_tracker.get_node_features(node_ids)
            features_list.append(metric_features)
        if self.config.use_structural:
            struct_features = self._get_structural_features(graph, node_ids)
            features_list.append(struct_features)
        if self.config.use_centrality:
            centr_features = self._get_centrality_features(graph, node_ids)
            features_list.append(centr_features)
        if not features_list:
            return torch.eye(len(node_ids), dtype=torch.float32)

        combined = torch.cat(features_list, dim=1)
        if self.config.normalize_features:
            combined = self._normalize(combined)

        return combined.to(torch.float32)

    def generate_edge_features(
        self,
        graph: Any,
        edges: list[tuple[str, str]],
        metrics_tracker: Any | None = None,
    ) -> torch.Tensor:
        """Собрать признаки рёбер: вес, метрики надёжности/задержки/трафика."""
        features = []

        for src, tgt in edges:
            edge_feat = []
            weight = self._get_edge_weight(graph, src, tgt)
            edge_feat.append(weight)
            if metrics_tracker is not None:
                edge_metrics = metrics_tracker.get_edge_metrics(src, tgt)
                if edge_metrics:
                    edge_feat.extend(
                        [
                            edge_metrics.reliability,
                            edge_metrics.avg_latency_ms / 1000.0,
                            edge_metrics.avg_data_volume / 1000.0,
                        ]
                    )
                else:
                    edge_feat.extend([1.0, 0.0, 0.0])
            else:
                edge_feat.extend([1.0, 0.0, 0.0])

            features.append(edge_feat)

        result = torch.tensor(features, dtype=torch.float32)

        if self.config.normalize_features:
            result = self._normalize(result)

        return result

    def _get_embedding_features(self, graph: Any, node_ids: list[str]) -> torch.Tensor | None:
        """Вернуть матрицу эмбеддингов узлов, если они доступны в графе."""
        if not hasattr(graph, "embeddings"):
            return None

        embeddings = graph.embeddings
        if embeddings is None or embeddings.numel() == 0:
            return None

        if isinstance(embeddings, torch.Tensor):
            return embeddings.to(torch.float32)
        return torch.tensor(embeddings, dtype=torch.float32)

    def _get_structural_features(self, graph: Any, node_ids: list[str]) -> torch.Tensor:
        """Подсчитать нормализованные входящие/исходящие степени для узлов."""
        features = []
        rx_graph = graph.graph
        id_to_idx = {}
        for idx in rx_graph.node_indices():
            data = rx_graph.get_node_data(idx)
            if isinstance(data, dict):
                nid = data.get("id", str(idx))
            elif hasattr(data, "identifier"):
                nid = data.identifier
            else:
                nid = str(idx)
            id_to_idx[nid] = idx

        num_nodes = rx_graph.num_nodes()

        for node_id in node_ids:
            idx = id_to_idx.get(node_id)
            if idx is not None:
                in_deg = rx_graph.in_degree(idx) / max(num_nodes - 1, 1)
                out_deg = rx_graph.out_degree(idx) / max(num_nodes - 1, 1)
                features.append([in_deg, out_deg])
            else:
                features.append([0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32)

    def _get_centrality_features(self, graph: Any, node_ids: list[str]) -> torch.Tensor:
        """Вычислить простую централность (pagerank) для узлов."""
        try:
            pagerank = rx.pagerank(graph.graph)
            id_to_idx = {}
            for idx in graph.graph.node_indices():
                data = graph.graph.get_node_data(idx)
                if isinstance(data, dict):
                    nid = data.get("id", str(idx))
                elif hasattr(data, "identifier"):
                    nid = data.identifier
                else:
                    nid = str(idx)
                id_to_idx[nid] = idx

            features = []
            for node_id in node_ids:
                idx = id_to_idx.get(node_id)
                if idx is not None and idx in pagerank:
                    features.append([pagerank[idx]])
                else:
                    features.append([0.0])

            return torch.tensor(features, dtype=torch.float32)
        except Exception:
            return torch.zeros((len(node_ids), 1), dtype=torch.float32)

    def _get_edge_weight(self, graph: Any, src: str, tgt: str) -> float:
        """Извлечь вес ребра графа или вернуть 1.0 по умолчанию."""
        try:
            rx_graph = graph.graph
            src_idx = tgt_idx = None

            for idx in rx_graph.node_indices():
                data = rx_graph.get_node_data(idx)
                if isinstance(data, dict):
                    nid = data.get("id", str(idx))
                elif hasattr(data, "identifier"):
                    nid = data.identifier
                else:
                    nid = str(idx)

                if nid == src:
                    src_idx = idx
                if nid == tgt:
                    tgt_idx = idx

            if src_idx is not None and tgt_idx is not None:
                edge_data = rx_graph.get_edge_data(src_idx, tgt_idx)
                if isinstance(edge_data, dict):
                    return edge_data.get("weight", 1.0)
            return 1.0
        except Exception:
            return 1.0

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Нормализовать признаки по столбцам и обрезать выбросы."""
        if features.numel() == 0:
            return features
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)

        normalized = (features - mean) / std
        if self.config.clip_outliers:
            normalized = torch.clamp(
                normalized,
                -self.config.outlier_std,
                self.config.outlier_std,
            )

        return normalized


if TORCH_AVAILABLE and PYG_AVAILABLE:

    class BaseGNNModel(nn.Module):
        """Базовая архитектура GNN с проекцией входа/выхода и residual-слоями."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 3,
            dropout: float = 0.1,
            use_batch_norm: bool = True,
            use_residual: bool = True,
        ):
            super().__init__()
            self.in_channels = in_channels  # type: ignore[misc]
            self.hidden_channels = hidden_channels  # type: ignore[misc]
            self.out_channels = out_channels  # type: ignore[misc]
            self.num_layers = num_layers  # type: ignore[misc]
            self.dropout = dropout  # type: ignore[misc]
            self.use_batch_norm = use_batch_norm  # type: ignore[misc]
            self.use_residual = use_residual  # type: ignore[misc]

            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.input_proj = nn.Linear(in_channels, hidden_channels)

            self.output_proj = nn.Linear(hidden_channels, out_channels)

        def reset_parameters(self):
            """Сбросить параметры всех слоёв модели."""
            for conv in self.convs:
                conv.reset_parameters()
            for bn in self.batch_norms:
                bn.reset_parameters()
            self.input_proj.reset_parameters()
            self.output_proj.reset_parameters()

        def forward(
            self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Прямой проход: свёртки по графу с Dropout, BatchNorm и residual."""
            x = self.input_proj(x)
            x = functional.relu(x)
            for i, conv in enumerate(self.convs):
                x_prev = x
                x = conv(x, edge_index)

                if self.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)

                x = functional.relu(x)
                x = functional.dropout(x, p=self.dropout, training=self.training)

                if self.use_residual and x.shape == x_prev.shape:
                    x = x + x_prev
            x = self.output_proj(x)

            return x

    class GCNRouter(BaseGNNModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))

            for _ in range(self.num_layers - 1):
                self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))

    class GATRouter(BaseGNNModel):
        def __init__(self, heads: int = 4, **kwargs):
            super().__init__(**kwargs)
            self.heads = heads  # type: ignore[misc]

            self.convs.append(
                GATConv(
                    self.hidden_channels,
                    self.hidden_channels // heads,
                    heads=heads,
                    dropout=self.dropout,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))

            for _ in range(self.num_layers - 1):
                self.convs.append(
                    GATConv(
                        self.hidden_channels,
                        self.hidden_channels // heads,
                        heads=heads,
                        dropout=self.dropout,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))

    class GraphSAGERouter(BaseGNNModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))

            for _ in range(self.num_layers - 1):
                self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))


class TrainingResult(BaseModel):
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    metrics: dict[str, float] = Field(default_factory=dict)


class GNNTrainer:
    """Утилита обучения GNN моделей для задач маршрутизации."""

    def __init__(
        self,
        model: Any,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ):
        """Подготовить трейнер, оптимизатор и устройство."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GNNTrainer")

        self.config = config or TrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._best_model_state = None
        self._history: TrainingResult | None = None

    def train(
        self,
        train_data: list[Any],  # list[Data]
        val_data: list[Any] | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        """Обучить модель на train/val датасете с ранней остановкой."""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for training")

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size) if val_data else None

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            if val_loader:
                val_loss = self._eval_epoch(val_loader)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                val_str = f", val_loss={val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}{val_str}")

        if self._best_model_state:
            self.model.load_state_dict(self._best_model_state)

        self._history = TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        )

        return self._history

    def _train_epoch(self, loader: Any) -> float:
        """Один эпох обучения, вернуть средний loss."""
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))

            if self.config.task == "node_classification":
                loss = functional.cross_entropy(out, batch.y)
            elif self.config.task == "path_ranking":
                loss = self._compute_ranking_loss(out, batch)
            else:
                loss = functional.mse_loss(out.squeeze(), batch.y.float())

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_epoch(self, loader: Any) -> float:
        """Оценка модели на валидационном датасете, вернуть средний loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))

                if self.config.task == "node_classification":
                    loss = functional.cross_entropy(out, batch.y)
                else:
                    loss = functional.mse_loss(out.squeeze(), batch.y.float())

                total_loss += loss.item()

        return total_loss / len(loader)

    def _compute_ranking_loss(self, out: Any, batch: Any) -> Any:
        """Простейший MSE loss для ранжирования путей."""
        scores = out.squeeze()
        targets = batch.y.float()

        return functional.mse_loss(scores, targets)

    def save(self, path: str | Path) -> None:
        """Сохранить веса модели и историю обучения в checkpoint."""
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "history": self._history,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Загрузить checkpoint и восстановить модель/конфигурацию."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        if "history" in checkpoint:
            self._history = checkpoint["history"]


class GNNRouterInference:
    """Инференс-модуль: генерирует признаки, запускает GNN и выбирает маршруты."""

    def __init__(
        self,
        model: Any,
        feature_generator: FeatureGenerator | None = None,
        device: str | None = None,
    ):
        """Подготовить модель для инференса на заданном устройстве."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GNNRouterInference")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_generator = feature_generator or DefaultFeatureGenerator()

    def predict(
        self,
        graph: Any,  # RoleGraph
        source: str | None = None,
        candidates: list[str] | None = None,
        metrics_tracker: Any | None = None,
        strategy: RoutingStrategy = RoutingStrategy.ARGMAX,
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> RoutingPrediction:
        """Предсказать следующие узлы маршрутизации согласно выбранной стратегии."""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for inference")

        node_ids = graph.node_ids if hasattr(graph, "node_ids") else []
        if not node_ids:
            node_ids = [str(i) for i in range(graph.num_nodes)]

        node_features = self.feature_generator.generate_node_features(
            graph, node_ids, metrics_tracker
        )

        edge_index = graph.edge_index

        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(x, edge_index_tensor)

        scores = functional.softmax(out.squeeze(-1) if out.dim() > 1 else out, dim=0)
        scores = scores.cpu()

        node_scores = {node_ids[i]: float(scores[i].item()) for i in range(len(node_ids))}

        if candidates:
            node_scores = {k: v for k, v in node_scores.items() if k in candidates}

        if source and source in node_scores:
            del node_scores[source]

        recommended = self._apply_strategy(node_scores, strategy, top_k, threshold)

        if recommended:
            confidence = torch.mean(torch.tensor([node_scores[n] for n in recommended])).item()
        else:
            confidence = 0.0

        return RoutingPrediction(
            recommended_nodes=recommended,
            scores=node_scores,
            confidence=float(confidence),
            strategy=strategy,
            metadata={"source": source, "num_candidates": len(node_scores)},
        )

    def _apply_strategy(
        self,
        scores: dict[str, float],
        strategy: RoutingStrategy,
        top_k: int,
        threshold: float,
    ) -> list[str]:
        """Выбрать рекомендуемые узлы согласно заданной стратегии отбора."""
        if not scores:
            return []

        if strategy == RoutingStrategy.ARGMAX:
            best_node = max(scores.keys(), key=lambda k: scores[k])
            return [best_node]

        elif strategy == RoutingStrategy.TOP_K:
            sorted_nodes = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            return sorted_nodes[:top_k]

        elif strategy == RoutingStrategy.THRESHOLD:
            return [n for n, s in scores.items() if s >= threshold]

        elif strategy == RoutingStrategy.SOFTMAX_SAMPLE:
            nodes = list(scores.keys())
            probs = torch.tensor([scores[n] for n in nodes])
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            return [nodes[idx]]

        return []

    def get_all_scores(
        self,
        graph: Any,
        metrics_tracker: Any | None = None,
    ) -> dict[str, float]:
        """Получить оценки для всех узлов графа."""
        prediction = self.predict(
            graph,
            metrics_tracker=metrics_tracker,
            strategy=RoutingStrategy.TOP_K,
            top_k=graph.num_nodes,
        )
        return prediction.scores

    @classmethod
    def load(
        cls, path: str | Path, feature_generator: FeatureGenerator | None = None
    ) -> "GNNRouterInference":
        """Загрузить модель и создать инстанс инференса из checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint.get("config", TrainingConfig())

        model = GCNRouter(
            in_channels=64,
            hidden_channels=config.hidden_dim,
            out_channels=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, feature_generator)


def create_gnn_router(
    model_type: GNNModelType,
    in_channels: int,
    out_channels: int,
    config: TrainingConfig | None = None,
) -> Any:
    """Фабрика: создать GNN роутер нужного типа (GCN/GAT/SAGE)."""
    if not TORCH_AVAILABLE or not PYG_AVAILABLE:
        raise ImportError("PyTorch and PyTorch Geometric are required")

    config = config or TrainingConfig()

    kwargs = {
        "in_channels": in_channels,
        "hidden_channels": config.hidden_dim,
        "out_channels": out_channels,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "use_batch_norm": config.use_batch_norm,
        "use_residual": config.use_residual,
    }

    if model_type == GNNModelType.GCN:
        return GCNRouter(**kwargs)
    elif model_type == GNNModelType.GAT:
        return GATRouter(heads=int(config.gat_heads), **kwargs)
    elif model_type == GNNModelType.SAGE:
        return GraphSAGERouter(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if not TORCH_AVAILABLE or not PYG_AVAILABLE:

    class BaseGNNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric are required for GNN models")

    class GCNRouter(BaseGNNModel):
        pass

    class GATRouter(BaseGNNModel):
        pass

    class GraphSAGERouter(BaseGNNModel):
        pass
