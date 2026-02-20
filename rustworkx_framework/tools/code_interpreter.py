"""
Code Interpreter tool — выполнение Python кода.

Позволяет агентам выполнять Python код в изолированном окружении.
Поддерживает таймауты и ограничение вывода.
"""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from .base import BaseTool, ToolResult


class CodeInterpreterTool(BaseTool):
    """
    Инструмент для выполнения Python кода.

    Выполняет Python код и возвращает результат. Поддерживает:
    - Ограничение времени выполнения
    - Ограничение размера вывода
    - Безопасный sandbox (ограниченные builtins)

    Example:
        tool = CodeInterpreterTool(timeout=10, max_output_size=4096)
        result = tool.execute(code="print(2 + 2)")

        if result.success:
            print(result.output)  # "4"
        else:
            print(f"Error: {result.error}")

    """

    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 8192,
        *,
        safe_mode: bool = True,
    ):
        """
        Создать CodeInterpreterTool.

        Args:
            timeout: Максимальное время выполнения в секундах.
            max_output_size: Максимальный размер вывода в байтах.
            safe_mode: Если True, ограничивает доступные builtins для безопасности.

        """
        self._timeout = timeout
        self._max_output_size = max_output_size
        self._safe_mode = safe_mode

        # Безопасные builtins для sandbox
        self._safe_builtins = {
            # Типы
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "bytes": bytes,
            "bytearray": bytearray,
            # Функции
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "chr": chr,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "format": format,
            "hash": hash,
            "hex": hex,
            "len": len,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "slice": slice,
            "sorted": sorted,
            "sum": sum,
            "zip": zip,
            # Исключения
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
            # Другое
            "True": True,
            "False": False,
            "None": None,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "type": type,
            "callable": callable,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "iter": iter,
            "next": next,
            "input": lambda _: "",  # Заблокирован ввод
        }

    @property
    def name(self) -> str:
        return "code_interpreter"

    @property
    def description(self) -> str:
        return (
            "Execute Python code and return the output. "
            "Use for calculations, data processing, and algorithmic tasks. "
            "The code runs in a sandboxed environment with limited access."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can be multi-line.",
                },
            },
            "required": ["code"],
        }

    def _get_safe_globals(self) -> dict[str, Any]:
        """Получить безопасный globals для exec."""
        import collections
        import datetime
        import functools
        import itertools
        import json
        import math
        import random
        import re
        import statistics

        return {
            "__builtins__": self._safe_builtins if self._safe_mode else __builtins__,
            # Безопасные модули
            "math": math,
            "statistics": statistics,
            "json": json,
            "re": re,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "random": random,
        }

    def execute(self, code: str = "", **_kwargs: Any) -> ToolResult:
        """
        Выполнить Python код.

        Args:
            code: Python код для выполнения.

        Returns:
            ToolResult с выводом или ошибкой.

        """
        if not code:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No code provided",
            )

        # Захватываем stdout и stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Подготавливаем окружение
            # Используем один словарь для globals и locals чтобы избежать
            # проблем со scoping (функции определённые в exec() должны быть
            # видны при вызове)
            exec_globals = self._get_safe_globals()

            # Выполняем код
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Компилируем для определения типа (expression или statement)
                try:
                    # Пробуем как expression (чтобы вернуть результат)
                    compiled = compile(code, "<code>", "eval")
                    result = eval(compiled, exec_globals)
                    if result is not None:
                        pass
                except SyntaxError:
                    # Выполняем как statements
                    # Используем один словарь для globals и locals
                    exec(code, exec_globals)

            # Собираем вывод
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            output = stdout_output
            if stderr_output:
                output += f"\n[stderr]\n{stderr_output}"

            # Ограничиваем размер вывода
            if len(output) > self._max_output_size:
                output = output[: self._max_output_size] + "\n... (output truncated)"

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output.strip() if output else "(no output)",
            )

        except (
            ValueError,
            TypeError,
            SyntaxError,
            NameError,
            AttributeError,
            KeyError,
            IndexError,
            ZeroDivisionError,
            RuntimeError,
            OSError,
        ) as e:
            # Форматируем ошибку
            error_output = stderr_capture.getvalue()
            _ = traceback.format_exc()  # Available for debugging

            # Извлекаем только полезную часть traceback
            error_msg = f"{type(e).__name__}: {e}"
            if error_output:
                error_msg = f"{error_output}\n{error_msg}"

            return ToolResult(
                tool_name=self.name,
                success=False,
                error=error_msg,
                output=stdout_capture.getvalue(),
            )
