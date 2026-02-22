"""
Function calling tool — вызов пользовательских функций.

Позволяет агентам вызывать зарегистрированные Python функции.
Поддерживает автоматическое извлечение схемы из type hints и docstrings.
"""

import inspect
from collections.abc import Callable
from typing import Any, Self, get_type_hints

from .base import BaseTool, ToolResult


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Преобразовать Python тип в JSON Schema тип."""
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }

    # Обработка Optional, Union и других typing конструкций
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # List[X], dict[X, Y], etc.
        if origin is list:
            args = getattr(python_type, "__args__", ())
            if args:
                return {"type": "array", "items": _python_type_to_json_schema(args[0])}
            return {"type": "array"}
        if origin is dict:
            return {"type": "object"}
        # Union, Optional
        return {"type": "string"}  # fallback

    return type_mapping.get(python_type, {"type": "string"})


def _extract_parameters_schema(func: Callable) -> dict[str, Any]:
    """Извлечь JSON Schema параметров из функции."""
    try:
        hints = get_type_hints(func)
    except (NameError, AttributeError, TypeError):
        hints = {}

    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        if param_type is inspect.Parameter.empty:
            param_type = str

        prop = _python_type_to_json_schema(param_type)

        # Попробовать извлечь описание из docstring
        # (простая эвристика: ищем param_name: description)
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


class FunctionWrapper(BaseTool):
    """
    Обёртка для Python функции как BaseTool.

    Автоматически извлекает имя, описание и схему параметров из функции.
    """

    def __init__(
        self,
        func: Callable,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """
        Создать обёртку для функции.

        Args:
            func: Python функция.
            tool_name: Имя инструмента (по умолчанию имя функции).
            tool_description: Описание (по умолчанию docstring).

        """
        self._func = func
        self._name = tool_name or getattr(func, "__name__", "unnamed")
        self._description = tool_description or getattr(func, "__doc__", None) or f"Function {self._name}"
        self._parameters_schema = _extract_parameters_schema(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters_schema

    def execute(self, **kwargs: Any) -> ToolResult:
        """Выполнить функцию с заданными аргументами."""
        try:
            result = self._func(**kwargs)
            output = str(result) if result is not None else "(no output)"
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
            )
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Execution error: {e}",
            )


class FunctionTool(BaseTool):
    """
    Менеджер пользовательских функций как инструмент.

    Позволяет регистрировать несколько функций и вызывать их по имени.
    Это meta-tool, который управляет коллекцией функций.

    Example:
        tool = FunctionTool()

        @tool.register
        def calculate(expression: str) -> str:
            \"\"\"Вычислить математическое выражение.\"\"\"
            return str(eval(expression))

        @tool.register
        def uppercase(text: str) -> str:
            \"\"\"Преобразовать текст в верхний регистр.\"\"\"
            return text.upper()

        # Вызов конкретной функции
        result = tool.execute(function="calculate", expression="2 + 2")

    """

    def __init__(self):
        """Создать пустой FunctionTool."""
        self._functions: dict[str, Callable] = {}
        self._wrappers: dict[str, FunctionWrapper] = {}

    @property
    def name(self) -> str:
        return "function_calling"

    @property
    def description(self) -> str:
        func_list = ", ".join(self._functions.keys()) if self._functions else "none"
        return f"Call a registered function. Available functions: {func_list}"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "function": {
                    "type": "string",
                    "description": "Name of the function to call",
                    "enum": list(self._functions.keys()) if self._functions else [],
                },
            },
            "required": ["function"],
            "additionalProperties": True,  # Allow function-specific args
        }

    def register(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """
        Зарегистрировать функцию. Может использоваться как декоратор.

        Args:
            func: Функция для регистрации.
            name: Имя функции (по умолчанию func.__name__).
            description: Описание (по умолчанию docstring).

        Returns:
            Оригинальная функция.

        Example:
            @tool.register
            def my_function(arg: str) -> str:
                return arg

            @tool.register(name="custom_name")
            def another(x: int) -> int:
                return x * 2

        """

        def decorator(f: Callable) -> Callable:
            func_name = name or getattr(f, "__name__", "unnamed")
            self._functions[func_name] = f
            self._wrappers[func_name] = FunctionWrapper(
                func=f,
                tool_name=func_name,
                tool_description=description or f.__doc__,
            )
            return f

        if func is not None:
            return decorator(func)
        return decorator

    def add_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> Self:
        """
        Добавить функцию (не-декораторный способ).

        Args:
            func: Функция для добавления.
            name: Имя функции.
            description: Описание функции.

        Returns:
            self для цепочки вызовов.

        """
        self.register(func, name=name, description=description)
        return self

    def get_function(self, name: str) -> Callable | None:
        """Получить функцию по имени."""
        return self._functions.get(name)

    def list_functions(self) -> list[str]:
        """Получить список имён зарегистрированных функций."""
        return list(self._functions.keys())

    def execute(self, function: str = "", **kwargs: Any) -> ToolResult:
        """
        Выполнить зарегистрированную функцию.

        Args:
            function: Имя функции для вызова.
            **kwargs: Аргументы функции.

        Returns:
            ToolResult с результатом.

        """
        if not function:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No function name provided",
            )

        wrapper = self._wrappers.get(function)
        if wrapper is None:
            available = ", ".join(self._functions.keys()) or "none"
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Function '{function}' not found. Available: {available}",
            )

        # Вызов через wrapper (который уже BaseTool)
        return wrapper.execute(**kwargs)
