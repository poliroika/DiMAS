"""
Agent with web_search tool.

Demonstrates the WebSearchTool for searching the internet and reading web pages:
  1. Direct tool usage (no LLM) — quick search, deep search, URL read
  2. Agent with web search
  3. Agent with deep search (fetch_content=True)

Configure your LLM via environment variables:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Run:
    python -m examples.web_search_example
"""

import os

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.execution import MACPRunner
from rustworkx_framework.tools import WebSearchTool, create_openai_caller, get_registry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_tools() -> None:
    get_registry().register(WebSearchTool(max_results=3, max_content_length=2000, fetch_content=False, timeout=15))


def _create_llm():
    return create_openai_caller(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "your-api-key"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


def _header(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── Example 1: Direct tool usage ────────────────────────────────────────────


def example_direct_usage():
    """Three WebSearchTool modes without any LLM."""
    _header("1 · Direct WebSearchTool Usage (no LLM)")

    # Quick search
    print("\n  a) Quick search (titles + snippets):")
    result = WebSearchTool(max_results=3, fetch_content=False).execute(query="Python programming")
    print(f"  {result.output[:400]}" if result.success else f"  Failed: {result.error}")

    # Deep search
    print("\n  b) Deep search (full page content):")
    result = WebSearchTool(max_results=2, fetch_content=True, max_content_length=1000).execute(query="Python asyncio")
    print(f"  {result.output[:400]}" if result.success else f"  Failed: {result.error}")

    # Direct URL
    print("\n  c) Read specific URL:")
    result = WebSearchTool().execute(url="https://httpbin.org/html")
    print(f"  {result.output[:300]}" if result.success else f"  Failed: {result.error}")


# ── Example 2: Agent with web search ────────────────────────────────────────


def example_agent_search():
    """Agent searches the web and summarises results."""
    _header("2 · Agent with Web Search")

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="researcher",
        display_name="Web Researcher",
        persona="a research assistant",
        description="Searches the web for current information.",
        tools=["web_search"],
    )
    builder.add_task(query="Search for 'Python asyncio tutorial' and summarise the key concepts")
    builder.connect_task_to_agents(agent_ids=["researcher"])

    result = MACPRunner(llm_caller=_create_llm()).run_round(builder.build())
    print(f"  Result: {result.final_answer}")


# ── Example 3: Agent with deep search ───────────────────────────────────────


def example_agent_deep_search():
    """Agent searches with fetch_content=True to read full pages."""
    _header("3 · Agent with Deep Search")

    # Re-register with fetch_content enabled
    get_registry().register(WebSearchTool(max_results=2, fetch_content=True, max_content_length=2000, timeout=15))

    builder = GraphBuilder()
    builder.add_agent(
        agent_id="deep_researcher",
        display_name="Deep Researcher",
        persona="a thorough researcher",
        description="Searches the web and reads full page content.",
        tools=["web_search"],
    )
    builder.add_task(query="Search for 'FastAPI tutorial' and read the full content")
    builder.connect_task_to_agents(agent_ids=["deep_researcher"])

    result = MACPRunner(llm_caller=_create_llm()).run_round(builder.build())
    print(f"  Result: {result.final_answer}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    _setup_tools()

    example_direct_usage()
    example_agent_search()
    example_agent_deep_search()

    print(f"\n{'=' * 60}")
    print("All web search examples completed ✅")


if __name__ == "__main__":
    main()
