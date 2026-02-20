"""
Example: Agent with web_search tool.

Demonstrates the WebSearchTool for searching the internet and reading web pages.

Modes supported:
1. Quick search (fetch_content=False)  — titles and snippets
2. Deep search (fetch_content=True)   — full page content
3. Direct URL read (url parameter)    — fetch specific page

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run with:
    python -m examples.web_search_example
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import (
    WebSearchTool,
    create_openai_caller,
    get_registry,
)

# Setup framework logging
setup_logging(level="INFO")


def setup_tools():
    """Register the WebSearchTool in the global registry."""
    registry = get_registry()
    registry.register(
        WebSearchTool(
            max_results=3,
            max_content_length=2000,
            fetch_content=False,
            timeout=15,
        )
    )


def create_llm():
    """Create LLM caller from environment variables."""
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


# ── Example 1: Direct tool usage (no LLM) ───────────────────────────────────────


def example_direct_usage():
    """Show the three WebSearchTool modes without any LLM."""
    print("\n" + "─" * 60)
    print("Example 1: Direct WebSearchTool Usage (no LLM)")
    print("─" * 60)

    # Quick search
    print("\n1. Quick search (titles and snippets):")
    tool = WebSearchTool(max_results=3, fetch_content=False)
    result = tool.execute(query="Python programming")
    if result.success:
        print(result.output[:400])
    else:
        print(f"  Search failed: {result.error}")

    # Deep search with content
    print("\n2. Deep search (with full page content):")
    tool = WebSearchTool(max_results=2, fetch_content=True, max_content_length=1000)
    result = tool.execute(query="Python asyncio")
    if result.success:
        print(result.output[:400])
    else:
        print(f"  Search failed: {result.error}")

    # Direct URL read
    print("\n3. Read specific URL:")
    tool = WebSearchTool()
    result = tool.execute(url="https://httpbin.org/html")
    if result.success:
        print(result.output[:300])
    else:
        print(f"  Fetch failed: {result.error}")


# ── Example 2: Agent with web search ────────────────────────────────────────────


def example_agent_web_search():
    """Example: Agent using web_search tool to find information."""
    print("\n" + "─" * 60)
    print("Example 2: Agent with Web Search")
    print("─" * 60)

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="researcher",
        display_name="Web Researcher",
        persona="a research assistant",
        description="I search the web for current information.",
        tools=["web_search"],
    )
    builder.add_task(
        query="Search for 'Python asyncio tutorial' and summarize the key concepts"
    )
    builder.connect_task_to_agents(agent_ids=["researcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Example 3: Agent with deep search ───────────────────────────────────────────


def example_agent_deep_search():
    """Example: Agent using web_search with fetch_content=true."""
    print("\n" + "─" * 60)
    print("Example 3: Agent with Deep Search")
    print("─" * 60)

    # Register tool with fetch_content enabled
    registry = get_registry()
    registry.register(
        WebSearchTool(
            max_results=2,
            fetch_content=True,
            max_content_length=2000,
            timeout=15,
        )
    )

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="deep_researcher",
        display_name="Deep Researcher",
        persona="a thorough researcher",
        description="I search the web and read full page content.",
        tools=["web_search"],
    )
    builder.add_task(query="Search for 'FastAPI tutorial' and read the full content")
    builder.connect_task_to_agents(agent_ids=["deep_researcher"])

    graph = builder.build()
    runner = MACPRunner(llm_caller=create_llm())
    result = runner.run_round(graph)

    print(f"\nTask: {graph.query}")
    print(f"Result: {result.final_answer}")


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    """Run all examples."""
    setup_tools()

    try:
        example_direct_usage()
        example_agent_web_search()
        example_agent_deep_search()
        print("\n" + "=" * 60)
        print("All examples completed ✅")
    except Exception as e:
        logger.exception("Error running examples: %s", e)
        raise


if __name__ == "__main__":
    main()
