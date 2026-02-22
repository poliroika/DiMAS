"""NodeEncoder для преобразования описаний агентов в эмбеддинги."""

import hashlib
import importlib
import importlib.util
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

__all__ = ["NodeEncoder"]

_TOKEN_RE = re.compile(r"[\w']+")
_HASH_PROVIDER = "hash"
_HASH_PREFIX = f"{_HASH_PROVIDER}:"
_SENTENCE_TRANSFORMERS_PREFIXES = ("sentence-transformers/", "sentence-transformers:")


def _tokenize(text: str) -> list[str]:
    """Разбить текст на токены (слова и числа) в нижнем регистре."""
    return _TOKEN_RE.findall(text.lower())


class NodeEncoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    fallback_dim: int = 384
    _model: SentenceTransformer | None = PrivateAttr(default=None)
    _provider: str = PrivateAttr(default="sentence-transformers")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Проверить имя модели."""
        if v == _HASH_PROVIDER or v.startswith(_HASH_PREFIX):
            if v.startswith(_HASH_PREFIX):
                _, _, raw_dim = v.partition(":")
                if not raw_dim.isdigit():
                    msg = f"Hash embedding dimension must be numeric, got {raw_dim!r}"
                    raise ValueError(msg)
                if int(raw_dim) < 1:
                    msg = f"Hash embedding dimension must be positive, got {raw_dim}"
                    raise ValueError(msg)
            return v

        if any(v.startswith(prefix) for prefix in _SENTENCE_TRANSFORMERS_PREFIXES):
            parts = v.split("/", 1) if "/" in v else v.split(":", 1)
            parts_expected = 2
            if len(parts) == parts_expected and parts[1].strip():
                return v
            msg = f"SentenceTransformer specification '{v}' is missing the model identifier"
            raise ValueError(msg)

        msg = "Unsupported embedding model. Expected 'sentence-transformers/<model>' or 'hash[:<dim>]'"
        raise ValueError(msg)

    def model_post_init(self, __context, /) -> None:
        """Определить провайдера (hash или sentence-transformers) и fallback dim."""
        if self.model_name == _HASH_PROVIDER or self.model_name.startswith(_HASH_PREFIX):
            self._provider = _HASH_PROVIDER
            if self.model_name.startswith(_HASH_PREFIX):
                _, _, raw_dim = self.model_name.partition(":")
                self.fallback_dim = max(int(raw_dim), 32)
        else:
            self._provider = "sentence-transformers"

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Закодировать список текстов в эмбеддинги."""
        cleaned = [text.strip() if isinstance(text, str) else "" for text in texts]
        if not cleaned:
            return torch.zeros((0, 0), dtype=torch.float32)

        if self._provider == _HASH_PROVIDER:
            return self._hash_fallback(cleaned)

        model = self._load_model()
        if model is None:
            return self._hash_fallback(cleaned)

        embeddings = model.encode(
            cleaned,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.to(dtype=torch.float32)

    def _load_model(self) -> Any:
        """Лениво загрузить SentenceTransformer, если он доступен."""
        if self._provider != "sentence-transformers":
            return None

        if self._model is not None:
            return self._model

        if importlib.util.find_spec("sentence_transformers") is None:
            self._provider = _HASH_PROVIDER
            return None

        module = importlib.import_module("sentence_transformers")
        self._model = module.SentenceTransformer(self.model_name)

        return self._model

    def _hash_fallback(self, texts: Sequence[str]) -> torch.Tensor:
        """Построить нормализованные bag-of-words эмбеддинги с помощью hash trick."""
        dimension = max(self.fallback_dim, 32)
        matrix = torch.zeros((len(texts), dimension), dtype=torch.float32)

        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=32).digest()
                index = int.from_bytes(digest[:8], byteorder="big", signed=False) % dimension
                matrix[row, index] += 1.0

            norm = torch.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm

        return matrix

    @property
    def embedding_dim(self) -> int:
        """Размерность генерируемых эмбеддингов для выбранного провайдера."""
        if self._provider == _HASH_PROVIDER:
            return self.fallback_dim

        model = self._load_model()
        if model is not None:
            return model.get_sentence_embedding_dimension()

        return self.fallback_dim
