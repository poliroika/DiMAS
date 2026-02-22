from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..tools.base import BaseTool

__all__ = ["AgentLLMConfig", "AgentProfile", "TaskNode"]


class AgentLLMConfig(BaseModel):
    """
    LLM конфигурация для AgentProfile.

    Позволяет задать персональные настройки LLM для агента:
    - model_name: имя модели (gpt-4, claude-3-opus, llama3:70b)
    - base_url: URL API endpoint
    - api_key: ключ API (или ссылка на переменную окружения $VAR)
    """

    model_config = ConfigDict(extra="allow")

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
    """
    Профиль агента с описанием, инструментами и LLM конфигурацией.

    Если у агента есть tools - они ВСЕГДА используются при каждом вызове LLM.
    Tools можно указать как строки (имена из registry) или как BaseTool объекты.

    Example:
        from rustworkx_framework.core.agent import AgentProfile
        from rustworkx_framework.tools import CodeInterpreterTool, tool

        # Регистрируем кастомный tool
        @tool
        def fibonacci(n: int) -> str:
            '''Calculate n-th Fibonacci number.'''
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return str(a)

        # Создаём агента с tools
        agent = AgentProfile(
            agent_id="math",
            display_name="Math Agent",
            persona="a helpful math assistant",
            tools=["fibonacci", "code_interpreter"],
        )

        # Или передаём объекты напрямую
        agent = AgentProfile(
            agent_id="coder",
            display_name="Coder",
            persona="a Python programmer",
            tools=[CodeInterpreterTool()],
        )

    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent_id: str
    display_name: str
    persona: str = ""
    description: str = ""

    # LLM Configuration
    llm_backbone: str | None = None
    llm_config: AgentLLMConfig | None = Field(default=None, repr=False)

    # Tools - список имён (str) или объектов (BaseTool)
    # При использовании tools ВСЕГДА вызываются через native function calling
    tools: list[Any] = Field(default_factory=list)

    raw: Mapping[str, Any] = Field(default_factory=dict)
    embedding: torch.Tensor | None = Field(default=None, repr=False)
    state: list[dict[str, Any]] = Field(default_factory=list)
    hidden_state: torch.Tensor | None = Field(default=None, repr=False)

    # Input/Output Schema for validation
    input_schema: Any | None = Field(default=None, repr=False)
    output_schema: Any | None = Field(default=None, repr=False)

    def get_tool_names(self) -> list[str]:
        """
        Получить имена tools агента.

        Возвращает:
            Список имён tools (для строк - как есть, для dict - из "name",
            для BaseTool - tool.name).
        """
        from ..tools.base import BaseTool

        names = []
        for t in self.tools:
            if isinstance(t, str):
                names.append(t)
            elif isinstance(t, dict):
                name = t.get("name") or t.get("tool") or t.get("id")
                if isinstance(name, str):
                    names.append(name)
            elif isinstance(t, BaseTool):
                names.append(t.name)
        return names

    def get_tool_objects(self) -> list["BaseTool"]:
        """
        Получить объекты tools агента.

        Поддерживает три формата:
        - str: ищет в глобальном registry по имени
        - dict: создаёт tool из конфига через фабрику
          (например ``{"name": "web_search", "use_selenium": True}``)
        - BaseTool: возвращает как есть

        Example:
            agent = AgentProfile(
                agent_id="browser",
                display_name="Browser Agent",
                tools=[
                    "shell",                                    # по имени из registry
                    {"name": "web_search", "use_selenium": True},  # dict-конфиг
                    WebSearchTool(use_selenium=True),           # объект напрямую
                ],
            )

        """
        from ..tools.base import BaseTool, create_tool_from_config, get_registry

        registry = get_registry()
        tools: list[BaseTool] = []
        for t in self.tools:
            if isinstance(t, str):
                tool_obj = registry.get(t)
                if tool_obj:
                    tools.append(tool_obj)
            elif isinstance(t, dict):
                tool_obj = create_tool_from_config(t)
                if tool_obj:
                    tools.append(tool_obj)
            elif isinstance(t, BaseTool):
                tools.append(t)
        return tools

    def has_tools(self) -> bool:
        """Проверить, есть ли у агента tools."""
        return len(self.tools) > 0

    def to_text(self) -> str:
        """Сериализовать профиль в текст для кодировщика."""
        parts = [self.display_name or self.agent_id]
        if self.persona and self.persona != self.description:
            parts.append(self.persona)
        if self.description:
            parts.append(self.description)
        if self.tools:
            tool_names = self.get_tool_names()
            parts.append("Tools: " + ", ".join(tool_names))
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
        """Получить эффективную LLM конфигурацию агента."""
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
        new_state = [*list(self.state), message]
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
        return self.agent_id

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать профиль в сериализуемый dict."""
        result = {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "persona": self.persona,
            "description": self.description,
            "llm_backbone": self.llm_backbone,
            "tools": self.get_tool_names(),
            "state": list(self.state),
            "embedding": self.embedding.cpu().tolist() if self.embedding is not None else None,
        }
        if self.llm_config:
            result["llm_config"] = self.llm_config.model_dump()
        if self.input_schema:
            result["input_schema"] = self.input_schema
        if self.output_schema:
            result["output_schema"] = self.output_schema
        return result


class TaskNode(BaseModel):
    """Виртуальный узел задачи, соединяющий всех агентов."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent_id: str = Field(default="__task__", alias="id")
    type: str = Field(default="task")
    query: str
    description: str = Field(default="Virtual task node that encodes the problem statement and connects to all agents.")
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

        agent_id = _extract_text(agent_dict.get("role") or agent_dict.get("name"))
        if not agent_id or agent_id in seen:
            continue

        profile = AgentProfile(
            agent_id=agent_id,
            display_name=_extract_text(agent_dict.get("name")) or agent_id,
            persona=_extract_text(agent_dict.get("persona")),
            description=_extract_text(agent_dict.get("description")),
            llm_backbone=_extract_llm_backbone(agent_dict),
            tools=_extract_tools(agent_dict),
            raw=agent_dict,
        )
        seen[agent_id] = profile

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
