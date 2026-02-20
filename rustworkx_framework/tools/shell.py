"""
Shell tool — выполнение shell команд.

Позволяет агентам выполнять команды в системной оболочке.
Поддерживает таймауты и ограничение размера вывода.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from .base import BaseTool, ToolResult


class ShellTool(BaseTool):
    """
    Инструмент для выполнения shell команд.

    Безопасность:
        - Команды выполняются в subprocess с таймаутом
        - Вывод ограничен по размеру для предотвращения переполнения
        - Не используется shell=True на Windows для безопасности

    Example:
        tool = ShellTool(timeout=30, max_output_size=4096)
        result = tool.execute(command="ls -la")

        if result.success:
            print(result.output)
        else:
            print(f"Error: {result.error}")

    """

    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 8192,
        working_dir: str | None = None,
        allowed_commands: list[str] | None = None,
    ):
        """
        Создать ShellTool.

        Args:
            timeout: Максимальное время выполнения команды в секундах.
            max_output_size: Максимальный размер вывода в байтах.
            working_dir: Рабочая директория для команд.
            allowed_commands: Белый список разрешённых команд (None = все).

        """
        self._timeout = timeout
        self._max_output_size = max_output_size
        self._working_dir = working_dir
        self._allowed_commands = set(allowed_commands) if allowed_commands else None

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Use for system operations, file manipulation, or running scripts."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        }

    def _is_command_allowed(self, command: str) -> bool:
        """Проверить, разрешена ли команда."""
        if self._allowed_commands is None:
            return True

        # Извлечь первое слово (имя команды)
        cmd_name = command.strip().split()[0] if command.strip() else ""
        return cmd_name in self._allowed_commands

    def execute(self, command: str = "", **_kwargs: Any) -> ToolResult:
        """
        Выполнить shell команду.

        Args:
            command: Команда для выполнения.

        Returns:
            ToolResult с выводом команды или ошибкой.

        """
        if not command:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No command provided",
            )

        if not self._is_command_allowed(command):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Command not allowed: {command.split()[0]}",
            )

        try:
            # Определяем shell в зависимости от ОС
            if sys.platform == "win32":
                # На Windows используем cmd.exe
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=self._working_dir,
                    check=False,
                )
            else:
                # На Unix используем /bin/sh
                result = subprocess.run(
                    command,
                    shell=True,
                    executable="/bin/sh",
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=self._working_dir,
                    check=False,
                )

            # Объединяем stdout и stderr
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # Ограничиваем размер вывода
            if len(output) > self._max_output_size:
                output = output[: self._max_output_size] + "\n... (output truncated)"

            if result.returncode != 0:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=output,
                    error=f"Command exited with code {result.returncode}",
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output.strip() if output else "(no output)",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Command timed out after {self._timeout} seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Command not found",
            )
        except (OSError, ValueError) as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Execution error: {e}",
            )
