"""
Базовые классы для инструментов агентов.

Tools используются через Native Function Calling (OpenAI/Anthropic API).
Если агент имеет tools - он ВСЕГДА их использует при каждом вызове.
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Запрос на вызов инструмента."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def parse_from_response(cls, response: str) -> list["ToolCall"]:
        r"""
        Парсить вызовы инструментов из ответа LLM.

        Поддерживает два формата:
        1. XML-like теги: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Markdown code blocks: ```tool_call\n{"name": "...", "arguments": {...}}\n```

        Args:
            response: Текст ответа от LLM

        Returns:
            Список ToolCall объектов

        """
        calls: list[ToolCall] = []

        # Паттерн для XML-like тегов
        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        xml_matches = re.findall(xml_pattern, response, re.DOTALL)

        for match in xml_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "name" in data:
                    calls.append(cls(name=data["name"], arguments=data.get("arguments", {})))
            except (json.JSONDecodeError, ValueError):
                # Пропускаем невалидный JSON
                pass

        # Паттерн для markdown code blocks
        code_block_pattern = r"```tool_call\s*\n(\{.*?\})\s*\n```"
        code_matches = re.findall(code_block_pattern, response, re.DOTALL)

        for match in code_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "name" in data:
                    calls.append(cls(name=data["name"], arguments=data.get("arguments", {})))
            except (json.JSONDecodeError, ValueError):
                # Пропускаем невалидный JSON
                pass

        return calls


class ToolResult(BaseModel):
    """Результат выполнения инструмента."""

    tool_name: str
    success: bool = True
    output: str = ""
    error: str | None = None

    def to_message(self) -> str:
        """Форматировать результат для вставки в промпт."""
        if self.success:
            return f'<tool_result name="{self.tool_name}">\n{self.output}\n</tool_result>'
        return f'<tool_error name="{self.tool_name}">\n{self.error}\n</tool_error>'


class BaseTool(ABC):
    """
    Абстрактный базовый класс для инструментов.

    Все инструменты должны наследоваться от этого класса и реализовать
    методы name, description и execute.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя инструмента."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Описание инструмента для LLM."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema параметров инструмента."""
        return {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Выполнить инструмент с заданными аргументами."""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Сериализовать инструмент в OpenAI function calling формат."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """
    Реестр инструментов для агентов.

    Example:
        from rustworkx_framework.tools import get_registry, CodeInterpreterTool

        # Получить глобальный реестр
        registry = get_registry()
        registry.register(CodeInterpreterTool())

        # Или создать свой
        my_registry = ToolRegistry()
        my_registry.register(ShellTool())

    """

    def __init__(self):
        """Инициализировать пустой реестр."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """
        Зарегистрировать инструмент.

        Args:
            tool: Экземпляр инструмента.

        Returns:
            self для цепочки вызовов.

        """
        self._tools[tool.name] = tool
        return self

    def function(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """
        Декоратор для регистрации функции как инструмента.

        Example:
            @registry.function
            def my_tool(arg: str) -> str:
                \"\"\"Описание инструмента.\"\"\"
                return arg.upper()

        """

        def decorator(f: Callable) -> Callable:
            tool_name = name or getattr(f, "__name__", "unnamed")
            tool_desc = description or getattr(f, "__doc__", None) or f"Function {tool_name}"

            from .function_calling import FunctionWrapper

            wrapper = FunctionWrapper(
                func=f,
                tool_name=tool_name,
                tool_description=tool_desc,
            )
            self._tools[tool_name] = wrapper
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def get(self, name: str) -> BaseTool | None:
        """Получить инструмент по имени."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Проверить, зарегистрирован ли инструмент."""
        return name in self._tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Выполнить вызов инструмента."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=f"Tool '{call.name}' not found",
            )

        try:
            return tool.execute(**call.arguments)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return ToolResult(
                tool_name=call.name,
                success=False,
                error=str(e),
            )

    def execute_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Выполнить несколько вызовов инструментов."""
        return [self.execute(call) for call in calls]

    def list_tools(self) -> list[str]:
        """Получить список имён зарегистрированных инструментов."""
        return list(self._tools.keys())

    def get_tools(self, tool_names: list[str] | None = None) -> list[BaseTool]:
        """
        Получить инструменты по именам.

        Args:
            tool_names: Список имён инструментов (None = все).

        Returns:
            Список BaseTool объектов.

        """
        if tool_names is None:
            return list(self._tools.values())
        return [self._tools[name] for name in tool_names if name in self._tools]

    def to_openai_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Получить schemas в формате OpenAI function calling API.

        Args:
            tool_names: Список имён инструментов (None = все).

        Returns:
            Список schemas в формате OpenAI tools API.

        """
        tools = self.get_tools(tool_names)
        return [tool.to_openai_schema() for tool in tools]

    def get_tools_for_agent(self, tool_names: list[str]) -> list[BaseTool]:
        """
        Получить инструменты для агента по именам.

        Args:
            tool_names: Список имён инструментов.

        Returns:
            Список BaseTool объектов (только существующие).

        """
        return [self._tools[name] for name in tool_names if name in self._tools]

    def to_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Получить упрощённые schemas инструментов.

        Args:
            tool_names: Список имён инструментов (None = все).

        Returns:
            Список schemas в упрощённом формате.

        """
        tools = self.get_tools(tool_names)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in tools
        ]

    def format_tools_prompt(self, tool_names: list[str] | None = None) -> str:
        """
        Сформировать текстовый промпт с описанием инструментов.

        Args:
            tool_names: Список имён инструментов (None = все).

        Returns:
            Строка с описанием инструментов для промпта.

        """
        tools = self.get_tools(tool_names)
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"\n- {tool.name}: {tool.description}")
            params = tool.parameters_schema.get("properties", {})
            if params:
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}")

        lines.append("\nTo use a tool, respond with:")
        lines.append("<tool_call>")
        lines.append('{"name": "tool_name", "arguments": {...}}')
        lines.append("</tool_call>")

        return "\n".join(lines)


# Глобальный реестр инструментов
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """
    Получить глобальный реестр инструментов.

    Создаёт реестр при первом вызове (синглтон).

    Example:
        from rustworkx_framework.tools import get_registry, ShellTool

        registry = get_registry()
        registry.register(ShellTool())

    """
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> BaseTool:
    """
    Зарегистрировать инструмент в глобальном реестре.

    Example:
        from rustworkx_framework.tools import register_tool, ShellTool

        register_tool(ShellTool())  # Теперь доступен глобально

    """
    get_registry().register(tool)
    return tool


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """
    Декоратор для регистрации функции как инструмента в глобальном реестре.

    Example:
        from rustworkx_framework.tools import tool

        @tool
        def fibonacci(n: int) -> str:
            '''Calculate the n-th Fibonacci number.'''
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return str(a)

        # Теперь 'fibonacci' доступен глобально через get_registry()

    """
    return get_registry().function(func, name=name, description=description)


# ============================================================
# Tool factory — создание tools из dict-конфига
# ============================================================

# Реестр фабрик: tool_name -> callable(config_dict) -> BaseTool
_tool_factories: dict[str, Callable[..., BaseTool]] = {}


def register_tool_factory(tool_name: str, factory: Callable[..., BaseTool]) -> None:
    """
    Зарегистрировать фабрику для создания tool по имени из конфига.

    Args:
        tool_name: Имя инструмента (например, "web_search").
        factory: Функция, принимающая **kwargs и возвращающая BaseTool.

    Example:
        register_tool_factory("web_search", lambda **kw: WebSearchTool(**kw))

    """
    _tool_factories[tool_name] = factory


def create_tool_from_config(config: dict[str, Any]) -> BaseTool | None:
    """
    Создать tool из dict-конфига.

    Конфиг должен содержать ключ "name" (имя tool).
    Остальные ключи передаются как параметры конструктора.

    Args:
        config: Словарь с настройками, например:
            {"name": "web_search", "use_selenium": True}

    Returns:
        BaseTool или None если фабрика не найдена.

    Example:
        tool = create_tool_from_config({
            "name": "web_search",
            "use_selenium": True,
            "selenium_config": {"headless": True, "browser": "chrome"},
        })

    """
    name = config.get("name") or config.get("tool") or config.get("id")
    if not name:
        return None

    factory = _tool_factories.get(name)
    if factory is None:
        return None

    # Убираем ключи идентификации, оставляем только параметры
    params = {k: v for k, v in config.items() if k not in ("name", "tool", "id")}
    return factory(**params)
