"""
Streaming execution example.

Demonstrates LangGraph-style streaming with real-time output:
1. Synchronous streaming — iterate over events
2. Synchronous streaming with a StreamBuffer to collect results
3. Token-level streaming — word-by-word output
4. Async streaming
5. Async token-level streaming
6. print_stream() helper — handles formatting automatically
7. Adaptive streaming — topology may change during execution

The examples use a local mock LLM so no API key is needed.

Run with:
    python -m examples.streaming_example
"""

import asyncio
import sys

# Fix Windows console encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from rustworkx_framework.builder import build_property_graph
from rustworkx_framework.config import logger, setup_logging
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.execution import (
    MACPRunner,
    RunnerConfig,
    StreamBuffer,
    StreamEventType,
    format_event,
    print_stream,
)

# Setup framework logging
setup_logging(level="INFO")


# ── Sample graph ──────────────────────────────────────────────────────────────

def create_example_graph():
    """Create a three-agent pipeline: Researcher -> Writer -> Editor."""
    agents = [
        AgentProfile(
            agent_id="researcher",
            display_name="Research Specialist",
            persona="You are an AI researcher who explains technical concepts clearly.",
            description="Gather and synthesise information about the topic.",
        ),
        AgentProfile(
            agent_id="writer",
            display_name="Content Writer",
            persona="You are a skilled writer who creates engaging content.",
            description="Transform research into readable content.",
        ),
        AgentProfile(
            agent_id="editor",
            display_name="Editor",
            persona="You are an editor who ensures clarity and quality.",
            description="Review and polish the final content.",
        ),
    ]

    edges = [("researcher", "writer"), ("writer", "editor")]

    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="Explain how AI works in simple terms",
        include_task_node=True,
    )


# ── Mock LLM callables ────────────────────────────────────────────────────────

def mock_llm(prompt: str) -> str:
    """Return a canned response based on which agent's turn it is."""
    if "researcher" in prompt.lower() or "research" in prompt.lower():
        return (
            "AI (Artificial Intelligence) works by using mathematical models "
            "trained on large amounts of data to recognise patterns and make predictions. "
            "Key components include neural networks, machine learning algorithms, "
            "and training data. Modern AI systems like ChatGPT use transformer "
            "architectures with billions of parameters."
        )
    if "writer" in prompt.lower() or "content" in prompt.lower():
        return (
            "## How AI Works: A Simple Guide\n\n"
            "Imagine teaching a child to recognise cats by showing them thousands "
            "of cat pictures. AI works similarly -- it learns from examples!\n\n"
            "**The Basics:**\n"
            "- AI uses 'neural networks' inspired by the human brain\n"
            "- It learns patterns from massive amounts of data\n"
            "- After training, it can make predictions on new data\n\n"
            "Modern AI assistants like ChatGPT have seen billions of text examples, "
            "helping them understand and generate human-like responses."
        )
    return (
        "# How AI Works: A Simple Guide\n\n"
        "AI learns from examples, just like humans! By analysing patterns in data, "
        "AI systems can recognise images, understand language, and even have "
        "conversations. The key is training on lots of examples until patterns emerge.\n\n"
        "Content reviewed and polished for clarity."
    )


def mock_streaming_llm(prompt: str):
    """Yield words one by one to simulate token-level streaming."""
    for i, word in enumerate(mock_llm(prompt).split(" ")):
        yield word + (" " if i < len(mock_llm(prompt).split(" ")) - 1 else "")


async def mock_async_llm(prompt: str) -> str:
    """Async version of mock_llm with a small simulated latency."""
    await asyncio.sleep(0.05)
    return mock_llm(prompt)


async def mock_async_streaming_llm(prompt: str):
    """Yield words asynchronously to simulate async token streaming."""
    words = mock_llm(prompt).split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(0.01)
        yield word + (" " if i < len(words) - 1 else "")


# ── Streaming examples ────────────────────────────────────────────────────────

def example_sync_streaming():
    """1. Iterate over raw stream events and print each one."""
    print("\n-- 1. Synchronous streaming --")
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    for event in runner.stream(graph):
        formatted = format_event(event)
        if formatted:
            print(f"  {formatted}")


def example_sync_streaming_with_buffer():
    """2. Collect events in a StreamBuffer; only print AGENT_OUTPUT."""
    print("\n-- 2. Streaming with buffer --")
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)
    buffer = StreamBuffer()

    for event in runner.stream(graph):
        buffer.add(event)
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            agent = getattr(event, "agent_name", "?")
            content = getattr(event, "content", "")
            print(f"  [{agent}] {content[:80]}...")

    print(f"  Total buffered events: {len(buffer.events)}")


def example_token_streaming():
    """3. Token-level (word-by-word) streaming."""
    print("\n-- 3. Token streaming --")
    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(streaming_llm_caller=mock_streaming_llm, config=config)

    current_agent = None
    token_count = 0

    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                print(f"  ({token_count} tokens)")
            current_agent = getattr(event, "agent_name", "unknown")
            token_count = 0
            print(f"  [{current_agent}] ", end="", flush=True)

        elif event.event_type == StreamEventType.TOKEN:
            token = getattr(event, "token", "")
            print(token, end="", flush=True)
            token_count += 1

        elif event.event_type == StreamEventType.RUN_END:
            print(f"\n  Run finished. Final: {getattr(event, 'final_answer', '')[:60]}...")


async def example_async_streaming():
    """4. Async streaming — await each event."""
    print("\n-- 4. Async streaming --")
    graph = create_example_graph()
    runner = MACPRunner(async_llm_caller=mock_async_llm)

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            print(f"  Starting: {getattr(event, 'agent_name', '?')}")
        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            content = getattr(event, "content", "")
            print(f"  Output  : {content[:80]}...")
        elif event.event_type == StreamEventType.RUN_END:
            print(f"  Done. Final answer: {getattr(event, 'final_answer', '')[:60]}...")


async def example_async_token_streaming():
    """5. Async token-level streaming."""
    print("\n-- 5. Async token streaming --")
    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(async_streaming_llm_caller=mock_async_streaming_llm, config=config)

    current_agent = None

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                print()
            current_agent = getattr(event, "agent_name", "unknown")
            print(f"  [{current_agent}] ", end="", flush=True)

        elif event.event_type == StreamEventType.TOKEN:
            print(getattr(event, "token", ""), end="", flush=True)

        elif event.event_type == StreamEventType.RUN_END:
            print(f"\n  Done.")


def example_print_stream_helper():
    """6. Use print_stream() -- it handles all formatting and returns the answer."""
    print("\n-- 6. print_stream() helper --")
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    final_answer = print_stream(runner.stream(graph), show_tokens=False, verbose=True)
    print(f"  Returned answer length: {len(final_answer or '')}")


def example_adaptive_streaming():
    """7. Adaptive execution: topology may be adjusted during the run."""
    print("\n-- 7. Adaptive streaming --")
    graph = create_example_graph()
    config = RunnerConfig(adaptive=True)
    runner = MACPRunner(llm_caller=mock_llm, config=config)

    for event in runner.stream(graph):
        formatted = format_event(event)
        if formatted:
            print(f"  {formatted}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Run all streaming examples."""
    example_sync_streaming()
    example_sync_streaming_with_buffer()
    example_token_streaming()
    example_print_stream_helper()
    example_adaptive_streaming()

    asyncio.run(example_async_streaming())
    asyncio.run(example_async_token_streaming())

    print("\nAll streaming examples completed.")


if __name__ == "__main__":
    main()
