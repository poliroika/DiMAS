"""
Пример: Агент с web_search tool.

Показывает как использовать WebSearchTool для поиска информации в интернете
и чтения содержимого веб-страниц (как в LangGraph с Tavily).

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. WebSearchTool может искать И читать страницы
2. Параметр fetch_content=True автоматически скачивает содержимое
3. Параметр url позволяет читать конкретную страницу
4. Поддержка DuckDuckGo (бесплатно), Serper (Google), Tavily
5. Поддержка Selenium для рендеринга JavaScript (SPA, динамический контент)

ЗАПУСК:
    python -m rustworkx_framework.examples.web_search_example
"""

import logging
import sys
from typing import Any

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.callbacks import (
    BaseCallbackHandler,
    CallbackManager,
    set_callback_manager,
)
from rustworkx_framework.tools import (
    LLMResponse,
    ToolResult,
    WebSearchTool,
    create_openai_caller,
    get_registry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Логирующая обёртка для caller
# ============================================================


class ToolCallLogger:
    """Обёртка над caller которая логирует ВСЕ tool calls от LLM."""

    def __init__(self, inner_caller):
        self.inner = inner_caller
        self.tool_calls: list[dict] = []

    def __call__(self, prompt: str, tools=None):
        response = self.inner(prompt, tools=tools)

        if isinstance(response, LLMResponse) and response.has_tool_calls:
            for tc in response.tool_calls:
                self.tool_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )

        return response

    def reset(self):
        self.tool_calls = []

    def summary(self) -> str:
        if not self.tool_calls:
            return "❌ NO TOOL CALLS DETECTED"
        counts = {}
        for tc in self.tool_calls:
            name = tc["name"]
            counts[name] = counts.get(name, 0) + 1
        parts = [f"{name}×{count}" for name, count in counts.items()]
        return f"✅ {len(self.tool_calls)} tool calls: {', '.join(parts)}"


# ============================================================
# 2. WebSearchTool с логированием
# ============================================================


class LoggingWebSearchTool(WebSearchTool):
    """WebSearchTool с логированием вызовов."""

    def execute(
        self,
        query: str = "",
        url: str = "",
        fetch_content: bool | None = None,
        max_results: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if url:
            pass
        else:
            pass

        result = super().execute(query=query, url=url, fetch_content=fetch_content, max_results=max_results, **kwargs)

        if result.success:
            lines = result.output.split("\n")
            preview = "\n".join(lines[:8])
            if len(lines) > 8:
                preview += "\n    ... (more content)"
        else:
            pass

        return result


# ============================================================
# 3. Настройка tools
# ============================================================


def setup_tools():
    """Настроить WebSearchTool."""
    registry = get_registry()

    # WebSearchTool с возможностью скачивания контента
    registry.register(
        LoggingWebSearchTool(
            max_results=3,
            max_content_length=2000,
            fetch_content=False,  # По умолчанию быстрый режим
            timeout=15,
        )
    )


# ============================================================
# 4. Демонстрация прямого использования
# ============================================================


def demo_direct_usage():
    """Демонстрация прямого использования WebSearchTool без LLM."""
    # Режим 1: Только поиск (быстро)
    tool = WebSearchTool(max_results=3, fetch_content=False)
    result = tool.execute(query="Python programming")
    if result.success:
        pass
    else:
        pass

    # Режим 2: Поиск + скачивание содержимого
    tool = WebSearchTool(max_results=2, fetch_content=True, max_content_length=1000)
    result = tool.execute(query="Python asyncio")
    if result.success:
        # Показываем превью
        pass
    else:
        pass

    # Режим 3: Чтение конкретного URL
    tool = WebSearchTool()
    result = tool.execute(url="https://httpbin.org/html")
    if result.success:
        pass
    else:
        pass


class ToolCallbackHandler(BaseCallbackHandler):
    """Callback handler для логирования tool events."""

    def on_tool_start(self, *, tool_name: str, action: str = "", arguments=None, **kwargs):
        args_str = str(arguments)[:100] if arguments else ""
        logger.info("[TOOL START] %s.%s(%s)", tool_name, action, args_str)

    def on_tool_end(
        self, *, tool_name: str, action: str = "", success: bool = True,
        output_size: int = 0, duration_ms: float = 0.0, result_summary: str = "", **kwargs,
    ):
        status = "✅" if success else "❌"
        logger.info(
            "[TOOL END] %s %s.%s — %s (%.0fms, %d bytes)",
            status, tool_name, action, result_summary, duration_ms, output_size,
        )

    def on_tool_error(
        self, *, tool_name: str, action: str = "",
        error_type: str = "", error_message: str = "", **kwargs,
    ):
        logger.error("[TOOL ERROR] %s.%s — %s: %s", tool_name, action, error_type, error_message)


def demo_selenium_usage():
    """
    Демонстрация использования WebSearchTool с Selenium.

    Требует: pip install selenium webdriver-manager

    Показывает:
    - Чтение страниц через Selenium (JS рендеринг)
    - Клик по элементам
    - Заполнение форм
    - Извлечение ссылок
    - Выполнение JavaScript
    - Рекурсивный обход сайта (crawl)
    - Все события логируются через callback-систему
    """
    try:
        import selenium  # noqa: F401
    except ImportError:
        logger.info("Selenium not installed, skipping Selenium demo.")
        logger.info("Install with: pip install selenium webdriver-manager")
        return

    # Настраиваем callback manager для логирования tool events
    cb_manager = CallbackManager(handlers=[ToolCallbackHandler()])
    set_callback_manager(cb_manager)

    logger.info("=== Selenium Demo (with callbacks) ===")

    # Создаём tool с Selenium
    tool = WebSearchTool(
        use_selenium=True,
        fetch_content=True,
        max_results=2,
        max_content_length=1500,
        selenium_config={
            "headless": True,
            "browser": "chrome",
            "extra_wait": 2.0,
        },
        callback_manager=cb_manager,
    )

    with tool:
        # 1. Чтение страницы через Selenium (рендерит JavaScript)
        logger.info("\n--- 1. Fetch URL ---")
        result = tool.execute(url="https://httpbin.org/html")
        if result.success:
            logger.info("Fetched %d chars", len(result.output))

        # 2. Извлечение ссылок со страницы
        logger.info("\n--- 2. Extract Links ---")
        result = tool.execute(
            action="extract_links",
            url="https://httpbin.org/",
        )
        if result.success:
            logger.info("Links result:\n%s", result.output[:500])

        # 3. Клик по элементу
        logger.info("\n--- 3. Click Element ---")
        result = tool.execute(action="click", selector="a[href]")
        if result.success:
            logger.info("Click result: %s", result.output)
        else:
            logger.warning("Click failed (expected on some pages): %s", result.error)

        # 4. Выполнение JavaScript
        logger.info("\n--- 4. Execute JavaScript ---")
        result = tool.execute(
            action="execute_js",
            js_code="return document.title + ' | Links: ' + document.querySelectorAll('a').length",
        )
        if result.success:
            logger.info("JS result: %s", result.output)

        # 5. Заполнение формы (на примере поисковой формы)
        logger.info("\n--- 5. Fill Form ---")
        # Сначала открываем страницу с формой
        tool.execute(url="https://httpbin.org/forms/post")
        result = tool.execute(
            action="fill",
            selector="input[name='custname']",
            value="Test User",
        )
        if result.success:
            logger.info("Fill result: %s", result.output)

        # 6. Получить содержимое текущей страницы
        logger.info("\n--- 6. Get Current Page Content ---")
        result = tool.execute(action="get_content")
        if result.success:
            logger.info("Current page content: %d chars", len(result.output))

        # 7. Рекурсивный обход сайта (crawl)
        logger.info("\n--- 7. Crawl Website ---")
        result = tool.execute(
            action="crawl",
            url="https://httpbin.org/",
            max_pages=3,
            max_depth=1,
        )
        if result.success:
            logger.info("Crawl result:\n%s", result.output[:800])

    # Сбрасываем callback manager
    set_callback_manager(None)
    logger.info("=== Selenium Demo Complete ===")


# ============================================================
# 5. Тесты с LLM
# ============================================================


def test_web_search(caller: ToolCallLogger) -> dict:
    """Тест web_search tool."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="researcher",
        display_name="Web Researcher",
        persona="a research assistant that searches the web for information",
        description="I search the web for current information. ALWAYS use web_search tool.",
        tools=["web_search"],
    )
    builder.add_task(
        query="Use web_search to find information about Python asyncio. Search for 'Python asyncio tutorial'"
    )
    builder.connect_task_to_agents(agent_ids=["researcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    _ = runner.run_round(graph)

    if "web_search" not in {tc["name"] for tc in caller.tool_calls}:
        return {"passed": False, "missing": {"web_search"}, "actual": set()}

    return {"passed": True, "actual": {"web_search"}}


def test_web_search_with_fetch(caller: ToolCallLogger) -> dict:
    """Тест web_search с fetch_content."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="deep_researcher",
        display_name="Deep Researcher",
        persona="a thorough researcher who reads full page content",
        description=(
            "I search the web AND read full page content. "
            "ALWAYS use web_search with fetch_content=true to get detailed information."
        ),
        tools=["web_search"],
    )
    builder.add_task(
        query="Search for 'FastAPI tutorial' and read the full content of the pages. "
        "Use fetch_content=true to get detailed information."
    )
    builder.connect_task_to_agents(agent_ids=["deep_researcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    _ = runner.run_round(graph)

    actual = {tc["name"] for tc in caller.tool_calls}
    if "web_search" not in actual:
        return {"passed": False, "missing": {"web_search"}, "actual": actual}

    # Проверяем что был fetch_content
    has_fetch = any(
        tc.get("arguments", {}).get("fetch_content") is True for tc in caller.tool_calls if tc["name"] == "web_search"
    )
    if has_fetch:
        pass
    else:
        pass

    return {"passed": True, "actual": actual}


def test_read_url(caller: ToolCallLogger) -> dict:
    """Тест чтения конкретного URL."""
    caller.reset()

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="reader",
        display_name="Page Reader",
        persona="a web page reader",
        description=("I read specific web pages. Use web_search with url parameter to read a specific page."),
        tools=["web_search"],
    )
    builder.add_task(
        query="Read the content of this URL: https://httpbin.org/html. Use web_search with the url parameter."
    )
    builder.connect_task_to_agents(agent_ids=["reader"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=caller)
    _ = runner.run_round(graph)

    actual = {tc["name"] for tc in caller.tool_calls}
    if "web_search" not in actual:
        return {"passed": False, "missing": {"web_search"}, "actual": actual}

    # Проверяем что был url параметр
    has_url = any(tc.get("arguments", {}).get("url") for tc in caller.tool_calls if tc["name"] == "web_search")
    if has_url:
        pass
    else:
        pass

    return {"passed": True, "actual": actual}


# ============================================================
# 6. Главная функция
# ============================================================


def main():
    # Сначала демо прямого использования (без LLM)
    demo_direct_usage()

    # Демо Selenium (если установлен)
    demo_selenium_usage()

    # Настройка tools
    setup_tools()

    # Создаём caller (замените URL на ваш LLM endpoint)
    inner_caller = create_openai_caller(
        base_url="https://quick-dramatically-resource-applicable.trycloudflare.com/v1",
        api_key="very-secure-key-sber1love",
        model="gpt-oss",
        temperature=0.1,
        tool_choice="required",
    )
    caller = ToolCallLogger(inner_caller)

    results = {}

    for test_name, test_fn in [
        ("search_basic", test_web_search),
        ("search_with_fetch", test_web_search_with_fetch),
        ("read_url", test_read_url),
    ]:
        try:
            result = test_fn(caller)
            results[test_name] = result
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[test_name] = {"passed": False, "error": str(e)}

    # Итоги

    all_passed = True
    for test_name, result in results.items():
        passed = result.get("passed", False)
        if passed:
            result.get("actual", set())
        else:
            all_passed = False
            missing = result.get("missing", set())
            result.get("error", "")
            if missing:
                pass
            else:
                pass

    if all_passed:
        pass
    else:
        pass


if __name__ == "__main__":
    main()
