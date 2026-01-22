"""Расширяемые схемы узлов и рёбер графа.

Поддерживает:
- Pydantic-валидацию атрибутов
- Кастомные поля пользователя
- Версионирование схем
- Миграции между версиями
- Стоимостные метрики (токены, доверие, latency)
"""

import builtins
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    # Schema version
    "SCHEMA_VERSION",
    "SchemaVersion",
    # LLM Configuration
    "LLMConfig",
    # Node schemas
    "NodeType",
    "BaseNodeSchema",
    "AgentNodeSchema",
    "TaskNodeSchema",
    # Edge schemas
    "EdgeType",
    "BaseEdgeSchema",
    "WorkflowEdgeSchema",
    "CostMetrics",
    # Graph schema
    "GraphSchema",
    # Migration
    "SchemaMigration",
    "MigrationRegistry",
    # Validation
    "ValidationResult",
    "SchemaValidator",
    "SchemaValidationResult",  # for input/output validation
]

# Текущая версия схемы
SCHEMA_VERSION = "2.0.0"


class SchemaVersion(BaseModel):
    major: int = 2
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """Разобрать строку версии в объект SchemaVersion."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: "SchemaVersion") -> bool:
        """Совместимость по мажорной версии."""
        return self.major == other.major

    def __lt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)


class NodeType(str, Enum):
    AGENT = "agent"
    TASK = "task"
    SUBGRAPH = "subgraph"
    TOOL = "tool"
    CUSTOM = "custom"


class LLMConfig(BaseModel):
    """Конфигурация LLM для отдельного агента.

    Позволяет каждому агенту использовать свой LLM с отдельными настройками:
    - Разные провайдеры (OpenAI, Anthropic, локальные модели)
    - Разные модели (gpt-4, claude-3, llama-3)
    - Разные параметры генерации (temperature, max_tokens)

    Example:
        # OpenAI GPT-4
        config1 = LLMConfig(
            model_name="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="$OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=2000
        )

        # Local Ollama model
        config2 = LLMConfig(
            model_name="llama3:70b",
            base_url="http://localhost:11434/v1",
            temperature=0.0
        )

        # Anthropic Claude
        config3 = LLMConfig(
            model_name="claude-3-opus-20240229",
            base_url="https://api.anthropic.com",
            api_key="$ANTHROPIC_API_KEY"
        )
    """

    model_config = {"extra": "allow"}

    # Model identification
    model_name: str | None = None  # e.g., "gpt-4", "claude-3-opus", "llama3:70b"
    base_url: str | None = None  # API endpoint URL
    api_key: str | None = None  # API key or env var reference (e.g., "$OPENAI_API_KEY")

    # Generation parameters
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None

    # Additional provider-specific options
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Разрешить API ключ из переменной окружения если указан как $VAR."""
        import os

        if self.api_key and self.api_key.startswith("$"):
            env_var = self.api_key[1:]
            return os.environ.get(env_var)
        return self.api_key

    def merge_with(self, other: "LLMConfig | None") -> "LLMConfig":
        """Объединить с другой конфигурацией (self имеет приоритет)."""
        if other is None:
            return self
        return LLMConfig(
            model_name=self.model_name or other.model_name,
            base_url=self.base_url or other.base_url,
            api_key=self.api_key or other.api_key,
            max_tokens=self.max_tokens if self.max_tokens is not None else other.max_tokens,
            temperature=self.temperature if self.temperature is not None else other.temperature,
            timeout=self.timeout if self.timeout is not None else other.timeout,
            top_p=self.top_p if self.top_p is not None else other.top_p,
            stop_sequences=self.stop_sequences or other.stop_sequences,
            extra_params={**other.extra_params, **self.extra_params},
        )

    def to_generation_params(self) -> dict[str, Any]:
        """Собрать параметры генерации для передачи в LLM caller."""
        params = {}
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        params.update(self.extra_params)
        return params

    def is_configured(self) -> bool:
        """Проверить, задана ли минимальная конфигурация для LLM."""
        return bool(self.model_name or self.base_url)


class BaseNodeSchema(BaseModel):
    """Базовая схема узла с эмбеддингами и пользовательскими метаданными."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    id: str
    type: NodeType = NodeType.AGENT
    display_name: str | None = None

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    embedding: list[float] | None = None
    embedding_dim: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)

    schema_version: str = SCHEMA_VERSION

    @field_validator("embedding", mode="before")
    @classmethod
    def convert_embedding(cls, v: Any) -> list[float] | None:
        """Привести embedding к списку float, если он задан."""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, torch.Tensor):
            return v.cpu().tolist()
        return v

    @model_validator(mode="after")
    def set_embedding_dim(self) -> "BaseNodeSchema":
        """Автозаполнить embedding_dim по длине embedding."""
        if self.embedding is not None and self.embedding_dim is None:
            object.__setattr__(self, "embedding_dim", len(self.embedding))
        return self

    def to_tensor_embedding(self) -> torch.Tensor | None:
        """Вернуть embedding как torch.Tensor или None."""
        if self.embedding is None:
            return None
        return torch.tensor(self.embedding, dtype=torch.float32)

    def get_feature_vector(self, feature_names: list[str] | None = None) -> torch.Tensor:
        """Собрать вектор признаков из embedding и выбранных метаданных."""
        features = []

        if self.embedding:
            features.extend(self.embedding)

        if feature_names:
            for name in feature_names:
                value = self.metadata.get(name, 0.0)
                if isinstance(value, (int, float)):
                    features.append(float(value))

        return (
            torch.tensor(features, dtype=torch.float32)
            if features
            else torch.zeros(0, dtype=torch.float32)
        )


class SchemaValidationResult(BaseModel):
    """Результат валидации данных по схеме.

    Attributes:
        valid: True если данные соответствуют схеме.
        schema_type: Тип схемы ('input' или 'output').
        errors: Список ошибок валидации.
        warnings: Список предупреждений.
        validated_data: Валидированные данные (если успешно).
        message: Дополнительное сообщение.
    """

    valid: bool = True
    schema_type: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_data: dict[str, Any] | None = None
    message: str = ""

    def raise_if_invalid(self) -> None:
        """Выбросить исключение если данные невалидны."""
        if not self.valid:
            raise ValueError(
                f"Schema validation failed ({self.schema_type}): {'; '.join(self.errors)}"
            )


class AgentNodeSchema(BaseNodeSchema):
    """Схема узла-агента с LLM конфигурацией и валидацией ввода/вывода.

    Поддерживает мультимодельность — каждый агент может использовать
    свой LLM с отдельными настройками (base_url, api_key, model_name и т.д.).

    Поддерживает валидацию данных через Pydantic-схемы:
    - input_schema: Pydantic модель или JSON Schema для валидации входящих данных
    - output_schema: Pydantic модель или JSON Schema для валидации ответа агента

    Example:
        from pydantic import BaseModel

        class SolverInput(BaseModel):
            question: str
            context: str | None = None

        class SolverOutput(BaseModel):
            answer: str
            confidence: float

        agent = AgentNodeSchema(
            id="solver",
            input_schema=SolverInput,
            output_schema=SolverOutput,
        )

        # Валидация
        result = agent.validate_input({"question": "2+2=?"})
        result = agent.validate_output('{"answer": "4", "confidence": 0.99}')
    """

    type: NodeType = NodeType.AGENT

    persona: str = ""
    description: str = ""

    # LLM Configuration - per-agent model settings
    llm_backbone: str | None = None  # model name (e.g., "gpt-4", "claude-3-opus")
    base_url: str | None = None  # API base URL (e.g., "https://api.openai.com/v1")
    api_key: str | None = None  # API key (or env var reference like "$OPENAI_API_KEY")

    # LLM Generation parameters
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None

    tools: list[str] = Field(default_factory=list)

    # Input/Output Schema for validation
    # Can be: Type[BaseModel], JSON Schema dict, or None
    input_schema: builtins.type[BaseModel] | dict[str, Any] | None = Field(
        default=None, exclude=True, repr=False
    )
    output_schema: builtins.type[BaseModel] | dict[str, Any] | None = Field(
        default=None, exclude=True, repr=False
    )

    # JSON Schema representations (for serialization)
    input_schema_json: dict[str, Any] | None = Field(default=None, repr=False)
    output_schema_json: dict[str, Any] | None = Field(default=None, repr=False)

    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    total_calls: int = 0
    total_tokens_used: int = 0
    avg_latency_ms: float = 0.0

    @model_validator(mode="after")
    def extract_json_schemas(self) -> "AgentNodeSchema":
        """Автоматически извлечь JSON Schema из Pydantic моделей."""
        if self.input_schema is not None and self.input_schema_json is None:
            if isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
                object.__setattr__(self, "input_schema_json", self.input_schema.model_json_schema())
            elif isinstance(self.input_schema, dict):
                object.__setattr__(self, "input_schema_json", self.input_schema)

        if self.output_schema is not None and self.output_schema_json is None:
            if isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
                object.__setattr__(
                    self, "output_schema_json", self.output_schema.model_json_schema()
                )
            elif isinstance(self.output_schema, dict):
                object.__setattr__(self, "output_schema_json", self.output_schema)

        return self

    def validate_input(self, data: dict[str, Any] | str) -> "SchemaValidationResult":
        """Валидировать входящие данные по input_schema.

        Args:
            data: Данные для валидации (dict или JSON-строка).

        Returns:
            SchemaValidationResult с результатом валидации.
        """
        # Приоритет: Pydantic модель > JSON Schema
        schema = self.input_schema or self.input_schema_json
        return self._validate_data(data, schema, "input")

    def validate_output(self, data: dict[str, Any] | str) -> "SchemaValidationResult":
        """Валидировать ответ агента по output_schema.

        Args:
            data: Данные для валидации (dict или JSON-строка).

        Returns:
            SchemaValidationResult с результатом валидации.
        """
        # Приоритет: Pydantic модель > JSON Schema
        schema = self.output_schema or self.output_schema_json
        return self._validate_data(data, schema, "output")

    def _validate_data(
        self,
        data: dict[str, Any] | str,
        schema: builtins.type[BaseModel] | dict[str, Any] | None,
        schema_type: str,
    ) -> "SchemaValidationResult":
        """Внутренний метод валидации данных."""
        import json

        if schema is None:
            return SchemaValidationResult(
                valid=True,
                schema_type=schema_type,
                message="No schema defined, validation skipped",
            )

        # Парсим JSON-строку если нужно
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                return SchemaValidationResult(
                    valid=False,
                    schema_type=schema_type,
                    errors=[f"Invalid JSON: {e}"],
                )

        # Валидация через Pydantic модель
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                validated = schema.model_validate(data)
                return SchemaValidationResult(
                    valid=True,
                    schema_type=schema_type,
                    validated_data=validated.model_dump(),
                )
            except Exception as e:
                return SchemaValidationResult(
                    valid=False,
                    schema_type=schema_type,
                    errors=[str(e)],
                )

        # Валидация через JSON Schema (без jsonschema библиотеки — базовая проверка)
        if isinstance(schema, dict):
            return self._validate_json_schema(data, schema, schema_type)

        return SchemaValidationResult(
            valid=False,
            schema_type=schema_type,
            errors=["Unknown schema type"],
        )

    def _validate_json_schema(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        schema_type: str,
    ) -> "SchemaValidationResult":
        """Базовая валидация JSON Schema (без внешних зависимостей)."""
        errors = []

        # Проверка обязательных полей
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Проверка типов (базовая)
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                prop_schema = properties[field]
                expected_type = prop_schema.get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(
                        f"Field '{field}': expected {expected_type}, got {type(value).__name__}"
                    )

        if errors:
            return SchemaValidationResult(valid=False, schema_type=schema_type, errors=errors)

        return SchemaValidationResult(valid=True, schema_type=schema_type, validated_data=data)

    def _check_type(self, value: Any, expected: str) -> bool:
        """Проверить соответствие типа JSON Schema."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Неизвестный тип — пропускаем
        return isinstance(value, expected_types)

    def has_input_schema(self) -> bool:
        """Проверить, задана ли input схема."""
        return self.input_schema is not None or self.input_schema_json is not None

    def has_output_schema(self) -> bool:
        """Проверить, задана ли output схема."""
        return self.output_schema is not None or self.output_schema_json is not None

    def get_llm_config(self) -> "LLMConfig":
        """Извлечь LLM конфигурацию из схемы агента."""
        return LLMConfig(
            model_name=self.llm_backbone,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences,
        )

    def has_llm_config(self) -> bool:
        """Проверить, задана ли LLM конфигурация для агента."""
        return any(
            [
                self.llm_backbone,
                self.base_url,
                self.api_key,
                self.max_tokens is not None,
                self.temperature is not None,
            ]
        )


class TaskNodeSchema(BaseNodeSchema):
    type: NodeType = NodeType.TASK

    query: str = ""
    description: str = ""
    expected_output: str | None = None

    max_iterations: int | None = None
    deadline: datetime | None = None

    answer: str | None = None
    status: str = "pending"  # pending, running, completed, failed


class EdgeType(str, Enum):
    WORKFLOW = "workflow"
    TASK_CONTEXT = "task_context"
    TASK_UPDATE = "task_update"
    DEPENDENCY = "dependency"
    FEEDBACK = "feedback"
    FALLBACK = "fallback"
    CUSTOM = "custom"


class CostMetrics(BaseModel):
    model_config = {"extra": "allow"}

    estimated_tokens: int | None = None
    actual_tokens: int | None = None

    latency_ms: float | None = None
    timeout_ms: float | None = None

    trust: float = Field(default=1.0, ge=0.0, le=1.0)
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)

    cost_usd: float | None = None

    custom: dict[str, float] = Field(default_factory=dict)


class BaseEdgeSchema(BaseModel):
    """Базовая схема ребра с весами, вероятностями и пользовательскими признаками."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    source: str
    target: str
    type: EdgeType = EdgeType.WORKFLOW

    weight: float = Field(default=1.0, ge=0.0)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)

    bidirectional: bool = False

    cost: CostMetrics = Field(default_factory=CostMetrics)

    embedding: list[float] | None = None

    attr: list[float] | None = None
    attr_dim: int | None = None

    created_at: datetime = Field(default_factory=datetime.now)

    metadata: dict[str, Any] = Field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    @field_validator("embedding", "attr", mode="before")
    @classmethod
    def convert_array(cls, v: Any) -> list[float] | None:
        """Привести массивные поля к списку float, если они заданы."""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, torch.Tensor):
            return v.cpu().tolist()
        return v

    @model_validator(mode="after")
    def set_attr_dim(self) -> "BaseEdgeSchema":
        """Автозаполнить attr_dim, если attr задан."""
        if self.attr is not None and self.attr_dim is None:
            object.__setattr__(self, "attr_dim", len(self.attr))
        return self

    def to_attr_tensor(self) -> torch.Tensor:
        """Вернуть признаки ребра в виде torch.Tensor."""
        if self.attr is not None:
            return torch.tensor(self.attr, dtype=torch.float32)
        return self._build_default_attr()

    def _build_default_attr(self) -> torch.Tensor:
        """Построить атрибуты по умолчанию (weight, probability, trust, типы)."""
        attr = [
            self.weight,
            self.probability,
            self.cost.trust,
            1.0 if self.type == EdgeType.WORKFLOW else 0.0,
            1.0 if self.type == EdgeType.TASK_CONTEXT else 0.0,
            1.0 if self.type == EdgeType.TASK_UPDATE else 0.0,
            1.0 if self.type == EdgeType.FEEDBACK else 0.0,
            1.0 if self.type == EdgeType.FALLBACK else 0.0,
        ]
        return torch.tensor(attr, dtype=torch.float32)

    def get_feature_vector(self, feature_names: list[str] | None = None) -> torch.Tensor:
        """Собрать вектор признаков из базовых полей, embedding и выбранных имён."""
        features = [self.weight, self.probability, self.cost.trust, self.cost.reliability]

        if self.embedding:
            features.extend(self.embedding)

        if feature_names:
            for name in feature_names:
                if hasattr(self.cost, name):
                    value = getattr(self.cost, name)
                elif name in self.metadata:
                    value = self.metadata[name]
                elif name in self.cost.custom:
                    value = self.cost.custom[name]
                else:
                    value = 0.0
                if isinstance(value, (int, float)):
                    features.append(float(value))

        return torch.tensor(features, dtype=torch.float32)


class WorkflowEdgeSchema(BaseEdgeSchema):
    """Схема workflow-ребра с условиями маршрутизации.

    Attributes:
        condition: Строковое условие или имя зарегистрированного условия.
                   Используется ConditionEvaluator для вычисления.
        priority: Приоритет ребра (выше = раньше).
        transform: Опциональное преобразование данных при переходе.
        is_conditional: True если ребро условное (для быстрой проверки).

    Example:
        edge = WorkflowEdgeSchema(
            source="solver",
            target="reviewer",
            condition="source_success",  # встроенное условие
            priority=1,
        )
    """

    type: EdgeType = EdgeType.WORKFLOW
    condition: str | None = None
    priority: int = 0
    transform: str | None = None
    is_conditional: bool = False

    @model_validator(mode="after")
    def set_is_conditional(self) -> "WorkflowEdgeSchema":
        """Автоматически установить is_conditional если condition задан."""
        if self.condition is not None:
            object.__setattr__(self, "is_conditional", True)
        return self


class GraphSchema(BaseModel):
    """Схема всего графа: узлы, рёбра, имена признаков и метаданные."""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    schema_version: str = SCHEMA_VERSION

    name: str | None = None
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    nodes: dict[str, BaseNodeSchema] = Field(default_factory=dict)
    edges: list[BaseEdgeSchema] = Field(default_factory=list)

    node_feature_names: list[str] = Field(default_factory=list)
    edge_feature_names: list[str] = Field(default_factory=list)

    node_feature_dim: int | None = None
    edge_feature_dim: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_node(self, node: BaseNodeSchema) -> None:
        """Добавить узел в схему и обновить метку времени."""
        self.nodes[node.id] = node
        self.updated_at = datetime.now()

    def add_edge(self, edge: BaseEdgeSchema) -> None:
        """Добавить ребро в схему и обновить метку времени."""
        self.edges.append(edge)
        self.updated_at = datetime.now()

    def get_node(self, node_id: str) -> BaseNodeSchema | None:
        """Вернуть узел по ID или None."""
        return self.nodes.get(node_id)

    def get_edges(
        self, source: str | None = None, target: str | None = None
    ) -> list[BaseEdgeSchema]:
        """Отфильтровать рёбра по источнику/приёмнику."""
        result = self.edges
        if source is not None:
            result = [e for e in result if e.source == source]
        if target is not None:
            result = [e for e in result if e.target == target]
        return result

    def compute_feature_dims(self) -> None:
        """Определить размерности признаков узлов и рёбер по текущим данным."""
        if self.nodes:
            sample_node = next(iter(self.nodes.values()))
            features = sample_node.get_feature_vector(self.node_feature_names)
            self.node_feature_dim = len(features) if features.size > 0 else 0

        if self.edges:
            features = self.edges[0].get_feature_vector(self.edge_feature_names)
            self.edge_feature_dim = len(features) if features.size > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать схему в словарь."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "nodes": {k: v.model_dump() for k, v in self.nodes.items()},
            "edges": [e.model_dump() for e in self.edges],
            "node_feature_names": self.node_feature_names,
            "edge_feature_names": self.edge_feature_names,
            "metadata": self.metadata,
        }


class SchemaMigration[T: BaseModel](ABC):
    from_version: ClassVar[str]
    to_version: ClassVar[str]

    @abstractmethod
    def migrate(self, data: dict[str, Any]) -> dict[str, Any]: ...

    def can_migrate(self, version: str) -> bool:
        """Проверить, применима ли миграция к указанной версии."""
        return version == self.from_version


class MigrationRegistry:
    def __init__(self) -> None:
        self._migrations: dict[str, list[SchemaMigration]] = {}

    def register(self, migration: SchemaMigration) -> None:
        """Зарегистрировать миграцию для её исходной версии."""
        key = migration.from_version
        if key not in self._migrations:
            self._migrations[key] = []
        self._migrations[key].append(migration)

    def migrate_to_latest(self, data: dict[str, Any], current_version: str) -> dict[str, Any]:
        """Применить цепочку миграций до актуальной версии схемы."""
        version = current_version

        while version != SCHEMA_VERSION:
            if version not in self._migrations:
                break

            for migration in self._migrations[version]:
                data = migration.migrate(data)
                version = migration.to_version
                break

        data["schema_version"] = SCHEMA_VERSION
        return data

    def get_migration_path(self, from_version: str, to_version: str) -> list[SchemaMigration]:
        """Вернуть последовательность миграций от версии A к B, если известна."""
        path = []
        version = from_version

        while version != to_version:
            if version not in self._migrations:
                break

            migration = self._migrations[version][0]
            path.append(migration)
            version = migration.to_version

        return path


_migration_registry = MigrationRegistry()


def register_migration(migration: SchemaMigration) -> None:
    """Зарегистрировать миграцию в глобальном реестре."""
    _migration_registry.register(migration)


def migrate_schema(data: dict[str, Any]) -> dict[str, Any]:
    """Применить миграции к данным схемы до актуальной версии."""
    version = data.get("schema_version", "1.0.0")
    if version == SCHEMA_VERSION:
        return data
    return _migration_registry.migrate_to_latest(data, version)


class MigrationV1ToV2(SchemaMigration):
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Миграция из схемы v1: перенос agents -> nodes и нормализация edges."""
        if "agents" in data and "nodes" not in data:
            nodes = {}
            for agent_data in data.get("agents", []):
                node_id = agent_data.get("identifier", agent_data.get("id"))
                nodes[node_id] = {
                    "id": node_id,
                    "type": "agent",
                    "display_name": agent_data.get("display_name"),
                    "persona": agent_data.get("persona", ""),
                    "description": agent_data.get("description", ""),
                    "embedding": agent_data.get("embedding"),
                    "metadata": {},
                    "schema_version": self.to_version,
                }
            data["nodes"] = nodes

        if "edges" in data:
            new_edges = []
            for edge in data["edges"]:
                new_edge = {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "type": edge.get("type", "workflow"),
                    "weight": edge.get("weight", 1.0),
                    "probability": 1.0,
                    "cost": {"trust": 1.0, "reliability": 1.0},
                    "schema_version": self.to_version,
                }
                new_edges.append(new_edge)
            data["edges"] = new_edges

        return data


register_migration(MigrationV1ToV2())


class ValidationResult(BaseModel):
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Добавить ошибку и пометить результат как невалидный."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Добавить предупреждение без изменения валидности."""
        self.warnings.append(message)


class SchemaValidator:
    def __init__(
        self,
        check_cycles: bool = True,
        check_duplicates: bool = True,
        check_orphans: bool = True,
        check_connectivity: bool = False,
    ):
        """Настройки валидации схемы (циклы, дубликаты, изолированные узлы)."""
        self.check_cycles = check_cycles
        self.check_duplicates = check_duplicates
        self.check_orphans = check_orphans
        self.check_connectivity = check_connectivity

    def validate(self, schema: GraphSchema) -> ValidationResult:
        """Провалидировать схему и вернуть результат с ошибками/предупреждениями."""
        result = ValidationResult()

        self._validate_nodes(schema, result)

        self._validate_edges(schema, result)

        if self.check_cycles:
            self._check_cycles(schema, result)

        if self.check_connectivity:
            self._check_connectivity(schema, result)

        return result

    def _validate_nodes(self, schema: GraphSchema, result: ValidationResult) -> None:
        """Проверить уникальность и корректность узлов."""
        seen_ids = set()

        for node_id, node in schema.nodes.items():
            if self.check_duplicates and node_id in seen_ids:
                result.add_error(f"Duplicate node ID: {node_id}")
            seen_ids.add(node_id)

            if node.id != node_id:
                result.add_error(f"Node ID mismatch: key={node_id}, node.id={node.id}")

            if node.embedding and node.embedding_dim:
                if len(node.embedding) != node.embedding_dim:
                    result.add_warning(
                        f"Node {node_id}: embedding length {len(node.embedding)} != embedding_dim {node.embedding_dim}"
                    )

    def _validate_edges(self, schema: GraphSchema, result: ValidationResult) -> None:
        """Проверить корректность рёбер и базовые ограничения."""
        seen_edges = set()
        node_ids = set(schema.nodes.keys())

        for i, edge in enumerate(schema.edges):
            if edge.source not in node_ids:
                result.add_error(f"Edge {i}: source '{edge.source}' not found")
            if edge.target not in node_ids:
                result.add_error(f"Edge {i}: target '{edge.target}' not found")

            if edge.source == edge.target:
                result.add_warning(f"Edge {i}: self-loop on '{edge.source}'")

            if self.check_duplicates:
                edge_key = (edge.source, edge.target, edge.type)
                if edge_key in seen_edges:
                    result.add_warning(f"Edge {i}: duplicate edge {edge_key}")
                seen_edges.add(edge_key)
            if edge.weight < 0:
                result.add_error(f"Edge {i}: negative weight {edge.weight}")
            if edge.probability < 0 or edge.probability > 1:
                result.add_error(f"Edge {i}: invalid probability {edge.probability}")

    def _check_cycles(self, schema: GraphSchema, result: ValidationResult) -> None:
        """Проверить наличие циклов через топологическую сортировку."""
        import rustworkx as rx

        graph = rx.PyDiGraph()
        node_indices = {}

        for node_id in schema.nodes:
            node_indices[node_id] = graph.add_node(node_id)

        for edge in schema.edges:
            if edge.source in node_indices and edge.target in node_indices:
                graph.add_edge(node_indices[edge.source], node_indices[edge.target], None)

        try:
            rx.topological_sort(graph)
        except rx.DAGHasCycle:
            sccs = list(rx.strongly_connected_components(graph))
            cycles = [scc for scc in sccs if len(scc) > 1]
            if cycles:
                cycle_nodes = [
                    [schema.nodes[list(schema.nodes.keys())[idx]].id for idx in scc]
                    for scc in cycles
                ]
                result.add_warning(f"Graph contains cycles: {cycle_nodes}")

    def _check_connectivity(self, schema: GraphSchema, result: ValidationResult) -> None:
        """Проверить связность и выделить изолированные узлы."""
        if len(schema.nodes) <= 1:
            return

        reachable = set()
        edge_map = {}
        for edge in schema.edges:
            if edge.source not in edge_map:
                edge_map[edge.source] = []
            edge_map[edge.source].append(edge.target)

        start = next(iter(schema.nodes.keys()))
        queue = [start]
        reachable.add(start)

        while queue:
            node = queue.pop(0)
            for neighbor in edge_map.get(node, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        if self.check_orphans:
            orphans = set(schema.nodes.keys()) - reachable
            if orphans:
                result.add_warning(f"Orphan nodes (not reachable): {orphans}")
