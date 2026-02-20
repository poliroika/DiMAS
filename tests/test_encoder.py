"""Тесты для core/encoder.py — NodeEncoder."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rustworkx_framework.core.encoder import NodeEncoder


class TestNodeEncoderCreation:
    """Тесты создания NodeEncoder."""

    def test_default_creation(self):
        """Создание с параметрами по умолчанию."""
        encoder = NodeEncoder()

        assert encoder is not None
        assert encoder.fallback_dim > 0

    def test_creation_with_model(self):
        """Создание с указанием модели."""
        encoder = NodeEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

        assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_creation_with_fallback_dim(self):
        """Создание с указанием fallback размерности."""
        encoder = NodeEncoder(fallback_dim=128)

        assert encoder.fallback_dim == 128


class TestHashEmbeddings:
    """Тесты hash-эмбеддингов (fallback)."""

    def test_hash_embedding_deterministic(self):
        """Hash-эмбеддинг детерминирован."""
        encoder = NodeEncoder(model_name="hash:64")

        text = "test agent"
        emb1 = encoder.encode([text])
        emb2 = encoder.encode([text])

        assert torch.allclose(emb1, emb2)

    def test_hash_embedding_different_texts(self):
        """Разные тексты дают разные эмбеддинги."""
        encoder = NodeEncoder(model_name="hash:64")

        embs = encoder.encode(["agent one", "agent two"])

        assert not torch.allclose(embs[0], embs[1])

    def test_hash_embedding_dimension(self):
        """Размерность hash-эмбеддинга."""
        encoder = NodeEncoder(model_name="hash:128")

        embs = encoder.encode(["test"])

        assert embs.shape == (1, 128)

    def test_hash_embedding_normalized(self):
        """Hash-эмбеддинг нормализован."""
        encoder = NodeEncoder(model_name="hash:64")

        embs = encoder.encode(["test"])
        norm = torch.norm(embs[0]).item()

        assert abs(norm - 1.0) < 0.01  # Близко к 1

    def test_hash_embedding_empty_string(self):
        """Hash-эмбеддинг для пустой строки."""
        encoder = NodeEncoder(model_name="hash:64")

        embs = encoder.encode([""])

        assert embs.shape == (1, 64)
        assert not torch.isnan(embs).any()


class TestSentenceTransformerEmbeddings:
    """Тесты sentence-transformer эмбеддингов."""

    def test_encode_single_text(self):
        """Кодирование одного текста."""
        encoder = NodeEncoder()

        embs = encoder.encode(["Test agent description"])

        assert isinstance(embs, torch.Tensor)
        assert embs.dim() == 2
        assert embs.shape[0] == 1
        assert embs.shape[1] > 0

    def test_encode_batch(self):
        """Кодирование батча текстов."""
        encoder = NodeEncoder()

        texts = ["Agent one", "Agent two", "Agent three"]
        embs = encoder.encode(texts)

        assert isinstance(embs, torch.Tensor)
        assert embs.shape[0] == 3

    def test_encode_empty_batch(self):
        """Кодирование пустого батча."""
        encoder = NodeEncoder()

        embs = encoder.encode([])

        assert embs.shape[0] == 0

    def test_fallback_when_st_unavailable(self):
        """Fallback на hash когда ST недоступен."""
        encoder = NodeEncoder(model_name="hash:64")

        embs = encoder.encode(["test"])

        assert embs.shape == (1, 64)


class TestAgentProfileEncoding:
    """Тесты кодирования профилей агентов."""

    def test_encode_agent_profile(self):
        """Кодирование профиля агента."""
        from rustworkx_framework.core.agent import AgentProfile

        encoder = NodeEncoder()

        profile = AgentProfile(
            agent_id="test_agent",
            display_name="Researcher",
            persona="Finds and analyzes information",
        )

        embs = encoder.encode([profile.to_text()])

        assert isinstance(embs, torch.Tensor)
        assert embs.dim() == 2
        assert embs.shape[0] == 1

    def test_encode_minimal_profile(self):
        """Кодирование минимального профиля."""
        from rustworkx_framework.core.agent import AgentProfile

        encoder = NodeEncoder()

        profile = AgentProfile(agent_id="minimal", display_name="minimal")

        embs = encoder.encode([profile.to_text()])

        assert isinstance(embs, torch.Tensor)

    def test_encode_profiles_batch(self):
        """Кодирование батча профилей."""
        from rustworkx_framework.core.agent import AgentProfile

        encoder = NodeEncoder()

        profiles = [
            AgentProfile(agent_id="a", display_name="Role A"),
            AgentProfile(agent_id="b", display_name="Role B"),
        ]

        texts = [p.to_text() for p in profiles]
        embs = encoder.encode(texts)

        assert embs.shape[0] == 2


class TestConsistency:
    """Тесты консистентности энкодера."""

    def test_same_input_same_output(self):
        """Одинаковый вход — одинаковый выход."""
        encoder = NodeEncoder()

        text = "consistent input"
        emb1 = encoder.encode([text])
        emb2 = encoder.encode([text])

        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_similar_texts_close_embeddings(self):
        """Похожие тексты имеют близкие эмбеддинги."""
        encoder = NodeEncoder()

        embs = encoder.encode(
            [
                "This is a researcher agent",
                "This is a research agent",
                "This is a completely different unrelated text about cats",
            ]
        )

        # Cosine similarity
        sim_12 = torch.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()
        sim_13 = torch.cosine_similarity(embs[0].unsqueeze(0), embs[2].unsqueeze(0)).item()

        # Similar texts should have higher similarity
        assert sim_12 > sim_13

    def test_dimension_consistency(self):
        """Консистентность размерности."""
        encoder = NodeEncoder()

        texts = ["short", "medium length text", "a very long text " * 100]

        dims = set()
        embs = encoder.encode(texts)
        for i in range(len(texts)):
            dims.add(embs[i].shape[0])

        # All should have same dimension
        assert len(dims) == 1


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_unicode_text(self):
        """Unicode текст."""
        encoder = NodeEncoder()

        embs = encoder.encode(["Тестовый агент с юникодом 日本語"])

        assert isinstance(embs, torch.Tensor)
        assert not torch.isnan(embs).any()

    def test_special_characters(self):
        """Специальные символы."""
        encoder = NodeEncoder()

        embs = encoder.encode(["Agent with special chars: !@#$%^&*()"])

        assert isinstance(embs, torch.Tensor)
        assert not torch.isnan(embs).any()

    def test_very_long_text(self):
        """Очень длинный текст."""
        encoder = NodeEncoder()

        long_text = "word " * 10000
        embs = encoder.encode([long_text])

        assert isinstance(embs, torch.Tensor)
        assert not torch.isnan(embs).any()

    def test_whitespace_only(self):
        """Только пробелы."""
        encoder = NodeEncoder()

        embs = encoder.encode(["   \t\n   "])

        assert isinstance(embs, torch.Tensor)

    def test_numbers_only(self):
        """Только числа."""
        encoder = NodeEncoder()

        embs = encoder.encode(["12345 67890"])

        assert isinstance(embs, torch.Tensor)


class TestGraphIntegration:
    """Тесты интеграции с графом."""

    def test_encode_graph_agents(self):
        """Кодирование агентов графа."""
        from rustworkx_framework.core.agent import AgentProfile

        encoder = NodeEncoder()

        agents = [
            AgentProfile(
                agent_id="coordinator",
                display_name="Coordinator",
                persona="Manages workflow",
            ),
            AgentProfile(agent_id="researcher", display_name="Researcher", persona="Finds information"),
            AgentProfile(agent_id="writer", display_name="Writer", persona="Creates content"),
        ]

        texts = [a.to_text() for a in agents]
        embeddings = encoder.encode(texts)

        assert embeddings.shape[0] == 3
        # All unique agents should have different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[1], embeddings[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
