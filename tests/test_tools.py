"""Тесты для модуля tools."""

from rustworkx_framework.tools import (
    DuckDuckGoProvider,
    FunctionTool,
    SearchProvider,
    ShellTool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
)


class TestToolCall:
    """Тесты для ToolCall."""

    def test_parse_xml_format(self):
        """Парсинг tool_call в XML формате."""
        response = """
Some text before.
<tool_call>
{"name": "test_tool", "arguments": {"arg1": "value1"}}
</tool_call>
Some text after.
"""
        calls = ToolCall.parse_from_response(response)
        assert len(calls) == 1
        assert calls[0].name == "test_tool"
        assert calls[0].arguments == {"arg1": "value1"}

    def test_parse_code_block_format(self):
        """Парсинг tool_call в формате code block."""
        response = """
```tool_call
{"name": "another_tool", "arguments": {"x": 42}}
```
"""
        calls = ToolCall.parse_from_response(response)
        assert len(calls) == 1
        assert calls[0].name == "another_tool"
        assert calls[0].arguments == {"x": 42}

    def test_parse_multiple_calls(self):
        """Парсинг нескольких tool_call."""
        response = """
<tool_call>
{"name": "tool1", "arguments": {}}
</tool_call>
<tool_call>
{"name": "tool2", "arguments": {"a": 1}}
</tool_call>
"""
        calls = ToolCall.parse_from_response(response)
        assert len(calls) == 2
        assert calls[0].name == "tool1"
        assert calls[1].name == "tool2"

    def test_parse_no_calls(self):
        """Ответ без tool_call."""
        response = "Just a regular response without any tools."
        calls = ToolCall.parse_from_response(response)
        assert len(calls) == 0

    def test_parse_invalid_json(self):
        """Невалидный JSON игнорируется."""
        response = """
<tool_call>
{invalid json here}
</tool_call>
"""
        calls = ToolCall.parse_from_response(response)
        assert len(calls) == 0


class TestToolResult:
    """Тесты для ToolResult."""

    def test_success_message(self):
        """Форматирование успешного результата."""
        result = ToolResult(tool_name="test", success=True, output="Hello")
        msg = result.to_message()
        assert '<tool_result name="test">' in msg
        assert "Hello" in msg
        assert "</tool_result>" in msg

    def test_error_message(self):
        """Форматирование ошибки."""
        result = ToolResult(tool_name="test", success=False, error="Something went wrong")
        msg = result.to_message()
        assert '<tool_error name="test">' in msg
        assert "Something went wrong" in msg
        assert "</tool_error>" in msg


class TestShellTool:
    """Тесты для ShellTool."""

    def test_name_and_description(self):
        """Проверка имени и описания."""
        tool = ShellTool()
        assert tool.name == "shell"
        assert "shell command" in tool.description.lower()

    def test_execute_echo(self):
        """Выполнение простой команды echo."""
        tool = ShellTool(timeout=5)
        result = tool.execute(command="echo Hello")
        assert result.success is True
        assert "Hello" in result.output

    def test_execute_no_command(self):
        """Ошибка при отсутствии команды."""
        tool = ShellTool()
        result = tool.execute()
        assert result.success is False
        assert result.error is not None
        assert "No command" in result.error

    def test_allowed_commands(self):
        """Белый список команд."""
        tool = ShellTool(allowed_commands=["echo"])

        # Разрешённая команда
        result = tool.execute(command="echo test")
        assert result.success is True

        # Запрещённая команда
        result = tool.execute(command="rm -rf /")
        assert result.success is False
        assert result.error is not None
        assert "not allowed" in result.error

    def test_parameters_schema(self):
        """Схема параметров."""
        tool = ShellTool()
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "command" in schema["properties"]


class TestFunctionTool:
    """Тесты для FunctionTool."""

    def test_register_decorator(self):
        """Регистрация через декоратор."""
        tool = FunctionTool()

        @tool.register
        def my_func(x: int) -> int:
            """Double the input."""
            return x * 2

        assert "my_func" in tool.list_functions()

    def test_register_with_custom_name(self):
        """Регистрация с кастомным именем."""
        tool = FunctionTool()

        @tool.register(name="custom_name")
        def some_func():
            pass

        assert "custom_name" in tool.list_functions()
        assert "some_func" not in tool.list_functions()

    def test_execute_function(self):
        """Выполнение зарегистрированной функции."""
        tool = FunctionTool()

        @tool.register
        def add(a: int, b: int) -> int:
            return a + b

        result = tool.execute(function="add", a=2, b=3)
        assert result.success is True
        assert result.output == "5"

    def test_execute_unknown_function(self):
        """Ошибка при вызове незарегистрированной функции."""
        tool = FunctionTool()
        result = tool.execute(function="unknown")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    def test_execute_no_function_name(self):
        """Ошибка при отсутствии имени функции."""
        tool = FunctionTool()
        result = tool.execute()
        assert result.success is False
        assert result.error is not None
        assert "No function name" in result.error


class TestToolRegistry:
    """Тесты для ToolRegistry."""

    def test_register_and_get(self):
        """Регистрация и получение инструмента."""
        registry = ToolRegistry()
        tool = ShellTool()
        registry.register(tool)

        assert registry.has("shell")
        assert registry.get("shell") is tool

    def test_list_tools(self):
        """Список зарегистрированных инструментов."""
        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(FunctionTool())

        tools = registry.list_tools()
        assert "shell" in tools
        assert "function_calling" in tools

    def test_execute(self):
        """Выполнение инструмента через реестр."""
        registry = ToolRegistry()
        registry.register(ShellTool(timeout=5))

        call = ToolCall(name="shell", arguments={"command": "echo test"})
        result = registry.execute(call)
        assert result.success is True
        assert "test" in result.output

    def test_execute_unknown_tool(self):
        """Ошибка при вызове незарегистрированного инструмента."""
        registry = ToolRegistry()
        call = ToolCall(name="unknown", arguments={})
        result = registry.execute(call)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error

    def test_execute_all(self):
        """Выполнение нескольких вызовов."""
        registry = ToolRegistry()
        registry.register(ShellTool(timeout=5))

        calls = [
            ToolCall(name="shell", arguments={"command": "echo first"}),
            ToolCall(name="shell", arguments={"command": "echo second"}),
        ]
        results = registry.execute_all(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_function_decorator(self):
        """Регистрация функции через декоратор реестра."""
        registry = ToolRegistry()

        @registry.function
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert registry.has("greet")
        result = registry.execute(ToolCall(name="greet", arguments={"name": "World"}))
        assert result.success is True
        assert result.output == "Hello, World!"

    def test_format_tools_prompt(self):
        """Форматирование промпта с инструментами."""
        registry = ToolRegistry()
        registry.register(ShellTool())

        prompt = registry.format_tools_prompt(["shell"])
        assert "Available tools:" in prompt
        assert "shell" in prompt
        assert "<tool_call>" in prompt

    def test_get_tools_for_agent(self):
        """Получение инструментов для агента."""
        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(FunctionTool())

        # Агент с обоими инструментами
        tools = registry.get_tools_for_agent(["shell", "function_calling"])
        assert len(tools) == 2

        # Агент только с shell
        tools = registry.get_tools_for_agent(["shell"])
        assert len(tools) == 1
        assert tools[0].name == "shell"

        # Агент с несуществующим инструментом
        tools = registry.get_tools_for_agent(["unknown"])
        assert len(tools) == 0

    def test_to_schemas(self):
        """Сериализация в JSON Schema."""
        registry = ToolRegistry()
        registry.register(ShellTool())

        schemas = registry.to_schemas(["shell"])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "shell"
        assert "description" in schemas[0]
        assert "parameters" in schemas[0]


class TestWebSearchTool:
    """Тесты для WebSearchTool."""

    def test_name_and_description(self):
        """Проверка имени и описания."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert "web" in tool.description.lower()

    def test_parameters_schema(self):
        """Схема параметров."""
        tool = WebSearchTool()
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "url" in schema["properties"]
        assert "fetch_content" in schema["properties"]

    def test_execute_no_query(self):
        """Ошибка при отсутствии запроса."""
        tool = WebSearchTool()
        result = tool.execute()
        assert result.success is False
        assert result.error is not None
        assert "No" in result.error and "provided" in result.error

    def test_execute_empty_query(self):
        """Ошибка при пустом запросе."""
        tool = WebSearchTool()
        result = tool.execute(query="")
        assert result.success is False
        assert result.error is not None
        assert "No" in result.error and "provided" in result.error

    def test_execute_with_mock_provider(self):
        """Выполнение с mock провайдером."""

        class MockProvider(SearchProvider):
            def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                return [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "snippet": f"This is a result for: {query}",
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "snippet": "Another test result snippet",
                    },
                ]

        tool = WebSearchTool(provider=MockProvider())
        result = tool.execute(query="test query")

        assert result.success is True
        assert "Test Result 1" in result.output
        assert "Test Result 2" in result.output
        assert "https://example.com/1" in result.output
        assert "test query" in result.output

    def test_execute_with_empty_results(self):
        """Обработка пустых результатов."""

        class EmptyProvider(SearchProvider):
            def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                return []

        tool = WebSearchTool(provider=EmptyProvider())
        result = tool.execute(query="no results query")

        assert result.success is True
        assert "No results found" in result.output

    def test_max_results_limit(self):
        """Ограничение количества результатов."""

        class ManyResultsProvider(SearchProvider):
            def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                # Возвращаем столько результатов, сколько запрошено
                return [
                    {
                        "title": f"Result {i}",
                        "url": f"https://example.com/{i}",
                        "snippet": f"Snippet {i}",
                    }
                    for i in range(max_results)
                ]

        tool = WebSearchTool(provider=ManyResultsProvider(), max_results=3)
        result = tool.execute(query="test")

        assert result.success is True
        assert "Found 3 result(s)" in result.output

    def test_fetch_content_with_mock(self):
        """Тест fetch_content с mock содержимым."""

        class ContentProvider(SearchProvider):
            def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                return [
                    {
                        "title": "Page with Content",
                        "url": "https://example.com/page",
                        "snippet": "Short snippet",
                        "content": "This is the full page content that was fetched from the URL.",
                    }
                ]

        tool = WebSearchTool(provider=ContentProvider(), fetch_content=True)
        result = tool.execute(query="test")

        assert result.success is True
        assert "Page with Content" in result.output
        assert "full page content" in result.output

    def test_url_parameter(self):
        """Тест параметра url для чтения конкретной страницы."""
        tool = WebSearchTool()
        # URL несуществующий, но проверяем что параметр обрабатывается
        result = tool.execute(url="https://nonexistent.invalid/page")
        # Должна быть ошибка сети, но не ошибка валидации
        assert result.success is False
        assert result.error is not None
        assert "Failed to fetch" in result.error or "error" in result.error.lower()

    def test_to_openai_schema(self):
        """Сериализация в OpenAI формат."""
        tool = WebSearchTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_registry_integration(self):
        """Интеграция с ToolRegistry."""

        class MockProvider(SearchProvider):
            def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                return [{"title": "Mock", "url": "https://mock.com", "snippet": "Mock result"}]

        registry = ToolRegistry()
        registry.register(WebSearchTool(provider=MockProvider()))

        assert registry.has("web_search")

        call = ToolCall(name="web_search", arguments={"query": "test"})
        result = registry.execute(call)

        assert result.success is True
        assert "Mock" in result.output


class TestDuckDuckGoProvider:
    """Тесты для DuckDuckGoProvider."""

    def test_initialization(self):
        """Инициализация провайдера."""
        provider = DuckDuckGoProvider(timeout=5)
        assert provider._timeout == 5

    def test_search_returns_list(self):
        """Метод search возвращает список."""
        provider = DuckDuckGoProvider(timeout=5)
        # Не делаем реальный запрос в unit-тестах,
        # просто проверяем что метод существует и возвращает нужный тип
        results = provider.search("python", max_results=3)
        assert isinstance(results, list)
