"""
Инструменты (tools) для агентов.

Если агент имеет tools - он ВСЕГДА использует их при каждом вызове LLM.
Tools передаются через API (параметр `tools`), не в тексте промпта.

Поддерживаемые инструменты:
- shell: Выполнение shell команд
- code_interpreter: Выполнение Python кода в sandbox
- file_search: Поиск файлов и их содержимого
- web_search: Поиск информации в интернете (DuckDuckGo, Serper и др.)
- Любые пользовательские функции через @tool декоратор

Пример использования:
    from rustworkx_framework.tools import tool, get_registry, CodeInterpreterTool
    from rustworkx_framework.core.agent import AgentProfile
    from rustworkx_framework.execution import MACPRunner

    # 1. Регистрируем tools (глобально или через registry)
    @tool
    def fibonacci(n: int) -> str:
        '''Calculate n-th Fibonacci number.'''
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return str(a)

    # 2. Создаём агента с tools
    agent = AgentProfile(
        agent_id="math",
        display_name="Math Agent",
        persona="a helpful math assistant",
        tools=["fibonacci", "code_interpreter"],  # <-- tools здесь!
    )

    # 3. Запускаем через runner - tools используются автоматически
    runner = MACPRunner(llm_caller=my_caller)
    result = runner.run_round(graph)
"""

from .base import (
    BaseTool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    get_registry,
    register_tool,
    tool,
)
from .code_interpreter import CodeInterpreterTool
from .file_search import FileSearchTool
from .function_calling import FunctionTool, FunctionWrapper
from .llm_integration import (
    LLMResponse,
    LLMToolCall,
    # Новый единый caller (рекомендуется)
    OpenAICaller,
    # Aliases для обратной совместимости
    OpenAIToolsCaller,
    create_openai_caller,
    create_openai_tools_caller,
    parse_anthropic_response,
    parse_openai_response,
)
from .shell import ShellTool
from .web_search import (
    DuckDuckGoProvider,
    SearchProvider,
    SeleniumFetcher,
    SerperProvider,
    TavilyProvider,
    URLFetcher,
    WebSearchTool,
)

__all__ = [
    # Base classes
    "BaseTool",
    "CodeInterpreterTool",
    "DuckDuckGoProvider",
    "FileSearchTool",
    "FunctionTool",
    "FunctionWrapper",
    # Native function calling (рекомендуется)
    "LLMResponse",
    "LLMToolCall",
    "OpenAICaller",  # Единый caller — работает и с tools и без
    # Backward compatibility aliases
    "OpenAIToolsCaller",  # = OpenAICaller
    # Web search providers and utilities
    "SearchProvider",
    "SeleniumFetcher",
    "SerperProvider",
    # Built-in tools
    "ShellTool",
    "TavilyProvider",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "URLFetcher",
    "WebSearchTool",
    "create_openai_caller",  # Рекомендуемый способ создания caller
    "create_openai_tools_caller",  # = create_openai_caller
    # Global registry helpers
    "get_registry",
    "parse_anthropic_response",
    "parse_openai_response",
    "register_tool",
    "tool",
]
