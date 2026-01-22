from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

__all__ = ["AgentProfile", "TaskNode", "AgentLLMConfig"]


class AgentLLMConfig(BaseModel):
    """LLM конфигурация для AgentProfile (легковесная копия LLMConfig).

    Позволяет задать персональные настройки LLM для агента:
    - model_name: имя модели (gpt-4, claude-3-opus, llama3:70b)
    - base_url: URL API endpoint
    - api_key: ключ API (или ссылка на переменную окружения $VAR)
    - Параметры генерации: max_tokens, temperature, timeout, top_p
    """

    model_config = {"extra": "allow"}

    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Разрешить API ключ из переменной окружения."""
        import os

        if self.api_key and self.api_key.startswith("$"):
            return os.environ.get(self.api_key[1:])
        return self.api_key

    def is_configured(self) -> bool:
        """Проверить, задана ли конфигурация."""
        return bool(self.model_name or self.base_url)

    def to_generation_params(self) -> dict[str, Any]:
        """Собрать параметры генерации для LLM."""
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


class AgentProfile(BaseModel):
    """Неподвижный профиль агента с описанием, инструментами, эмбеддингами и LLM конфигурацией.

    Поддерживает мультимодельность — каждый агент может использовать свой LLM.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    identifier: str
    display_name: str
    persona: str = ""
    description: str = ""

    # LLM Configuration
    llm_backbone: str | None = None  # model name (legacy, use llm_config.model_name)
    llm_config: AgentLLMConfig | None = Field(default=None, repr=False)

    tools: list[str] = Field(default_factory=list)
    raw: Mapping[str, Any] = Field(default_factory=dict)
    embedding: torch.Tensor | None = Field(default=None, repr=False)
    state: list[dict[str, Any]] = Field(default_factory=list)
    hidden_state: torch.Tensor | None = Field(default=None, repr=False)

    def to_text(self) -> str:
        """Сериализовать профиль в текст для кодировщика."""
        parts = [self.display_name or self.identifier]
        if self.persona and self.persona != self.description:
            parts.append(self.persona)
        if self.description:
            parts.append(self.description)
        if self.tools:
            parts.append("Tools: " + ", ".join(self.tools))
        model_name = self.get_model_name()
        if model_name:
            parts.append(f"LLM Backbone: {model_name}")
        return "\n".join(p.strip() for p in parts if p.strip())

    def get_model_name(self) -> str | None:
        """Получить имя модели (из llm_config или llm_backbone)."""
        if self.llm_config and self.llm_config.model_name:
            return self.llm_config.model_name
        return self.llm_backbone

    def get_llm_config(self) -> AgentLLMConfig:
        """Получить эффективную LLM конфигурацию агента.

        Возвращает llm_config если задана, иначе создаёт из llm_backbone.
        """
        if self.llm_config:
            return self.llm_config
        return AgentLLMConfig(model_name=self.llm_backbone)

    def has_custom_llm(self) -> bool:
        """Проверить, задана ли кастомная LLM конфигурация."""
        return self.llm_config is not None and self.llm_config.is_configured()

    def with_llm_config(self, llm_config: AgentLLMConfig) -> "AgentProfile":
        """Вернуть копию профиля с заданной LLM конфигурацией."""
        return self.model_copy(update={"llm_config": llm_config})

    def with_embedding(self, embedding: torch.Tensor) -> "AgentProfile":
        """Вернуть копию профиля с обновлённым эмбеддингом."""
        return self.model_copy(update={"embedding": embedding})

    def with_state(self, state: list[dict[str, Any]]) -> "AgentProfile":
        """Вернуть копию профиля с новым состоянием."""
        return self.model_copy(update={"state": state})

    def append_state(self, message: dict[str, Any]) -> "AgentProfile":
        """Вернуть копию с добавленным сообщением в историю состояния."""
        new_state = list(self.state) + [message]
        return self.model_copy(update={"state": new_state})

    def with_hidden_state(self, hidden_state: torch.Tensor) -> "AgentProfile":
        """Вернуть копию профиля с обновлённым скрытым состоянием."""
        return self.model_copy(update={"hidden_state": hidden_state})

    def clear_state(self) -> "AgentProfile":
        """Вернуть копию с очищенным локальным состоянием."""
        return self.model_copy(update={"state": []})

    @property
    def role(self) -> str:
        """Синоним идентификатора роли агента."""
        return self.identifier

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать профиль в сериализуемый dict."""
        result = {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "persona": self.persona,
            "description": self.description,
            "llm_backbone": self.llm_backbone,
            "tools": list(self.tools),
            "state": list(self.state),
            "embedding": self.embedding.cpu().tolist() if self.embedding is not None else None,
        }
        if self.llm_config:
            result["llm_config"] = self.llm_config.model_dump()
        return result


class TaskNode(BaseModel):
    """Виртуальный узел задачи, соединяющий всех агентов."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    identifier: str = Field(default="__task__", alias="id")
    type: str = Field(default="task")
    query: str
    description: str = Field(
        default="Virtual task node that encodes the problem statement and connects to all agents."
    )
    embedding: torch.Tensor | None = Field(default=None, repr=False)

    display_name: str = Field(default="Task")
    persona: str = Field(default="")
    llm_backbone: str | None = Field(default=None)
    tools: list[str] = Field(default_factory=list)
    state: list[dict[str, Any]] = Field(default_factory=list)

    def with_embedding(self, embedding: torch.Tensor) -> "TaskNode":
        """Вернуть копию узла задачи с заданным эмбеддингом."""
        return self.model_copy(update={"embedding": embedding})

    def to_text(self) -> str:
        """Сериализовать задачу в текстовое описание."""
        parts = []
        if self.description:
            parts.append(self.description.strip())
        query_text = self.query.strip() if self.query.strip() else "(unspecified)"
        parts.append(f"Task: {query_text}")
        return "\n".join(p for p in parts if p)


def extract_agent_profiles(agents_data: Mapping[str, Any]) -> list[AgentProfile]:
    """Собрать уникальные `AgentProfile` из словаря со списком агентов."""
    seen: dict[str, AgentProfile] = {}

    entries = agents_data.get("agents", []) if isinstance(agents_data, dict) else []

    for entry in entries:
        agent_dict = entry.get("agent") if isinstance(entry, dict) else entry
        if not isinstance(agent_dict, dict):
            continue

        identifier = _extract_text(agent_dict.get("role") or agent_dict.get("name"))
        if not identifier or identifier in seen:
            continue

        profile = AgentProfile(
            identifier=identifier,
            display_name=_extract_text(agent_dict.get("name")) or identifier,
            persona=_extract_text(agent_dict.get("persona")),
            description=_extract_text(agent_dict.get("description")),
            llm_backbone=_extract_llm_backbone(agent_dict),
            tools=_extract_tools(agent_dict),
            raw=agent_dict,
        )
        seen[identifier] = profile

    return list(seen.values())


def _extract_text(value: Any) -> str:
    """Вернуть очищенный текст, если значение строковое, иначе пустую строку."""
    return value.strip() if isinstance(value, str) else ""


def _extract_tools(agent_dict: Mapping[str, Any]) -> list[str]:
    """Извлечь список уникальных инструментов из описания агента."""
    tools = agent_dict.get("tools")
    if not isinstance(tools, (list, tuple, set)):
        return []

    result: list[str] = []
    for entry in tools:
        if isinstance(entry, str):
            value = entry.strip()
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("tool") or entry.get("id")
            value = name.strip() if isinstance(name, str) else ""
        else:
            value = ""
        if value and value not in result:
            result.append(value)
    return result


def _extract_llm_backbone(agent_dict: Mapping[str, Any]) -> str | None:
    """Извлечь идентификатор LLM из разных возможных полей описания."""
    candidate = agent_dict.get("llm") or agent_dict.get("model") or agent_dict.get("llm_backbone")

    if isinstance(candidate, dict):
        for key in ("model", "name", "type"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    return None
