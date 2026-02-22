"""
Интеграция tools с LLM через native function calling.

Этот модуль предоставляет:
- LLMResponse и LLMToolCall для парсинга ответов от LLM
- OpenAICaller — ЕДИНЫЙ caller для всех случаев (с tools и без)
- Парсеры ответов для OpenAI и Anthropic

Использование:
    # Один caller для всего
    caller = create_openai_caller(api_key="...", model="gpt-4")

    # Без tools — возвращает str
    response = caller("Hello!")

    # С tools — возвращает LLMResponse
    response = caller("Calculate fib(10)", tools=[...])
    if response.has_tool_calls:
        for tc in response.tool_calls:
            print(tc.name, tc.arguments)
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .base import ToolCall


@dataclass
class LLMToolCall:
    """
    Структурированный tool call от LLM.

    Представляет вызов инструмента, возвращённый LLM через native function calling.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_tool_call(self) -> ToolCall:
        """Преобразовать в ToolCall для выполнения."""
        return ToolCall(name=self.name, arguments=self.arguments)


@dataclass
class LLMResponse:
    """
    Ответ от LLM с поддержкой tool calls.

    Attributes:
        content: Текстовый контент ответа.
        tool_calls: Список вызовов инструментов (если LLM их запросил).
        raw_response: Оригинальный ответ от API (для отладки).

    """

    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Есть ли вызовы инструментов."""
        return len(self.tool_calls) > 0

    def get_tool_calls(self) -> list[ToolCall]:
        """Получить ToolCall объекты для выполнения."""
        return [tc.to_tool_call() for tc in self.tool_calls]


def parse_openai_response(response: Any) -> LLMResponse:
    """
    Парсить ответ OpenAI API в LLMResponse.

    Поддерживает как новый формат (tool_calls), так и legacy (function_call).

    Args:
        response: Ответ от OpenAI ChatCompletion API.

    Returns:
        LLMResponse с распарсенными данными.

    """
    message = response.choices[0].message

    tool_calls = []

    # Новый формат: tool_calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(
                LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    # Legacy формат: function_call
    elif hasattr(message, "function_call") and message.function_call:
        fc = message.function_call
        try:
            args = json.loads(fc.arguments) if fc.arguments else {}
        except json.JSONDecodeError:
            args = {}

        tool_calls.append(
            LLMToolCall(
                id="legacy_call",
                name=fc.name,
                arguments=args,
            )
        )

    return LLMResponse(
        content=message.content or "",
        tool_calls=tool_calls,
        raw_response=response,
    )


def parse_anthropic_response(response: Any) -> LLMResponse:
    """
    Парсить ответ Anthropic API в LLMResponse.

    Args:
        response: Ответ от Anthropic Messages API.

    Returns:
        LLMResponse с распарсенными данными.

    """
    tool_calls = []
    content_parts = []

    for block in response.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                LLMToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )

    return LLMResponse(
        content="\n".join(content_parts),
        tool_calls=tool_calls,
        raw_response=response,
    )


class OpenAICaller:
    """
    ЕДИНЫЙ LLM caller для OpenAI — работает и с tools и без.

    Это РЕКОМЕНДУЕМЫЙ способ создания caller для агентов.

    - Без tools: возвращает str (как обычный caller)
    - С tools: возвращает LLMResponse с tool_calls

    Example:
        from openai import OpenAI

        client = OpenAI(api_key="...")
        caller = OpenAICaller(client, model="gpt-4")

        # Без tools — обычный текстовый ответ
        response = caller("Hello!")  # -> str

        # С tools — LLMResponse с tool_calls
        response = caller("Calculate fib(15)", tools=[...])  # -> LLMResponse
        if response.has_tool_calls:
            for tc in response.tool_calls:
                print(f"Call {tc.name} with {tc.arguments}")

    """

    def __init__(
        self,
        client: Any,  # OpenAI client
        model: str = "gpt-4",
        temperature: float = 0.1,  # Низкая temperature для детерминизма
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        tool_choice: str = "required",  # "required" = обязательно, "auto" = опционально
    ):
        """
        Создать универсальный OpenAI caller.

        Args:
            client: Экземпляр OpenAI клиента.
            model: Название модели.
            temperature: Температура генерации (по умолчанию 0.1 для детерминизма).
            max_tokens: Максимум токенов в ответе.
            system_prompt: Системный промпт (опционально).
            tool_choice: Политика использования tools:
                - "required": LLM ОБЯЗАН вызвать tool (по умолчанию)
                - "auto": LLM сам решает использовать ли tools

        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.tool_choice = tool_choice

    def __call__(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | LLMResponse:
        """
        Вызвать OpenAI API.

        Args:
            prompt: Промпт пользователя.
            tools: Tools в OpenAI формате (опционально).

        Returns:
            - str: если tools не переданы
            - LLMResponse: если tools переданы

        """
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = self.tool_choice

        response = self.client.chat.completions.create(**kwargs)

        # Если tools были переданы — возвращаем LLMResponse
        if tools:
            return parse_openai_response(response)

        # Если без tools — возвращаем просто строку
        return response.choices[0].message.content or ""


# Alias для обратной совместимости
OpenAIToolsCaller = OpenAICaller


def create_openai_caller(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4",
    temperature: float = 0.1,  # Низкая temperature по умолчанию
    max_tokens: int = 2048,
    system_prompt: str | None = None,
    tool_choice: str = "required",
) -> OpenAICaller:
    """
    Создать универсальный OpenAI caller.

    Это РЕКОМЕНДУЕМЫЙ способ создания caller для агентов.
    Работает как с tools, так и без них.

    Args:
        api_key: API ключ OpenAI (или из переменной окружения).
        base_url: Базовый URL (для совместимых API).
        model: Название модели.
        temperature: Температура генерации (по умолчанию 0.1 для детерминизма).
        max_tokens: Максимум токенов.
        system_prompt: Системный промпт.
        tool_choice: Политика использования tools:
            - "required": LLM ОБЯЗАН вызвать tool (по умолчанию)
            - "auto": LLM сам решает использовать ли tools

    Returns:
        OpenAICaller готовый к использованию.

    Example:
        # Один caller для всех агентов
        caller = create_openai_caller(
            api_key="sk-...",
            model="gpt-4",
        )

        # Без tools — обычный текст
        response = caller("Hello!")  # -> str

        # С tools — LLMResponse
        response = caller("Calculate fib(10)", tools=[...])
        if response.has_tool_calls:
            ...

    """
    try:
        from openai import OpenAI
    except ImportError as e:
        msg = "openai package required: pip install openai"
        raise ImportError(msg) from e

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    return OpenAICaller(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        tool_choice=tool_choice,
    )


# Alias для обратной совместимости
create_openai_tools_caller = create_openai_caller
