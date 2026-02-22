"""
Streaming execution — no LLM required (uses a local mock).

Demonstrates LangGraph-style streaming with real-time output:
  1. Synchronous streaming — iterate over events
  2. Streaming with a StreamBuffer to collect results
  3. Token-level streaming — word-by-word output
  4. Async streaming
  5. Async token-level streaming
  6. print_stream() helper — handles formatting automatically
  7. Adaptive streaming — topology may change during execution

Run:
    python -m examples.streaming_example
"""

import asyncio

from rustworkx_framework.builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.execution import (
    MACPRunner,
    RunnerConfig,
    StreamBuffer,
    StreamEventType,
    format_event,
    print_stream,
)

# ── Sample graph ──────────────────────────────────────────────────────────────


def _create_graph():
    """Three-agent pipeline: Researcher → Writer → Editor."""
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
    return build_property_graph(
        agents,
        workflow_edges=[("researcher", "writer"), ("writer", "editor")],
        query="Explain how AI works in simple terms",
        include_task_node=True,
    )


# ── Mock LLM callables ───────────────────────────────────────────────────────


def _mock_llm(prompt: str) -> str:
    """Return a canned response based on which agent is running."""
    p = prompt.lower()
    if "researcher" in p or "research" in p:
        return (
            "AI works by using mathematical models trained on large amounts of data "
            "to recognise patterns and make predictions. Key components include "
            "neural networks, machine learning algorithms, and training data. "
            "Modern AI systems like ChatGPT use transformer architectures."
        )
    if "writer" in p or "content" in p:
        return (
            "## How AI Works: A Simple Guide\n\n"
            "Imagine teaching a child to recognise cats by showing thousands of pictures. "
            "AI works similarly — it learns from examples!\n\n"
            "**The Basics:**\n"
            "- Neural networks inspired by the human brain\n"
            "- Learns patterns from massive data\n"
            "- Makes predictions on new data"
        )
    return (
        "# How AI Works: A Simple Guide\n\n"
        "AI learns from examples, just like humans! By analysing patterns in data, "
        "AI systems can recognise images, understand language, and have conversations.\n\n"
        "Content reviewed and polished for clarity."
    )


def _mock_streaming_llm(prompt: str):
    """Yield words one by one to simulate token-level streaming."""
    words = _mock_llm(prompt).split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


async def _mock_async_llm(prompt: str) -> str:
    await asyncio.sleep(0.05)
    return _mock_llm(prompt)


async def _mock_async_streaming_llm(prompt: str):
    words = _mock_llm(prompt).split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(0.01)
        yield word + (" " if i < len(words) - 1 else "")


# ── Examples ──────────────────────────────────────────────────────────────────


def example_sync_streaming():
    """1. Iterate over raw stream events."""
    print("\n── 1 · Synchronous streaming ──")
    runner = MACPRunner(llm_caller=_mock_llm)
    for event in runner.stream(_create_graph()):
        text = format_event(event)
        if text:
            print(f"  {text}")


def example_buffer():
    """2. Collect events in a StreamBuffer; print only AGENT_OUTPUT."""
    print("\n── 2 · Streaming with buffer ──")
    runner = MACPRunner(llm_caller=_mock_llm)
    buf = StreamBuffer()
    for event in runner.stream(_create_graph()):
        buf.add(event)
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            name = getattr(event, "agent_name", "?")
            content = getattr(event, "content", "")
            print(f"  [{name}] {content[:80]}…")
    print(f"  Buffered events: {len(buf.events)}")


def example_token_streaming():
    """3. Token-level (word-by-word) streaming."""
    print("\n── 3 · Token streaming ──")
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(streaming_llm_caller=_mock_streaming_llm, config=config)

    current_agent = None
    tokens = 0
    for event in runner.stream(_create_graph()):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                print(f"  ({tokens} tokens)")
            current_agent = getattr(event, "agent_name", "?")
            tokens = 0
            print(f"  [{current_agent}] ", end="", flush=True)
        elif event.event_type == StreamEventType.TOKEN:
            print(getattr(event, "token", ""), end="", flush=True)
            tokens += 1
        elif event.event_type == StreamEventType.RUN_END:
            answer = getattr(event, "final_answer", "")
            print(f"\n  Done. Answer: {answer[:60]}…")


async def example_async_streaming():
    """4. Async streaming."""
    print("\n── 4 · Async streaming ──")
    runner = MACPRunner(async_llm_caller=_mock_async_llm)
    async for event in runner.astream(_create_graph()):
        if event.event_type == StreamEventType.AGENT_START:
            print(f"  Starting: {getattr(event, 'agent_name', '?')}")
        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            print(f"  Output  : {getattr(event, 'content', '')[:80]}…")
        elif event.event_type == StreamEventType.RUN_END:
            print(f"  Done. Answer: {getattr(event, 'final_answer', '')[:60]}…")


async def example_async_token_streaming():
    """5. Async token-level streaming."""
    print("\n── 5 · Async token streaming ──")
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(async_streaming_llm_caller=_mock_async_streaming_llm, config=config)

    current = None
    async for event in runner.astream(_create_graph()):
        if event.event_type == StreamEventType.AGENT_START:
            if current:
                print()
            current = getattr(event, "agent_name", "?")
            print(f"  [{current}] ", end="", flush=True)
        elif event.event_type == StreamEventType.TOKEN:
            print(getattr(event, "token", ""), end="", flush=True)
        elif event.event_type == StreamEventType.RUN_END:
            print("\n  Done.")


def example_print_stream():
    """6. print_stream() helper — handles formatting automatically."""
    print("\n── 6 · print_stream() helper ──")
    runner = MACPRunner(llm_caller=_mock_llm)
    answer = print_stream(runner.stream(_create_graph()), show_tokens=False, verbose=True)
    print(f"  Returned answer length: {len(answer or '')}")


def example_adaptive():
    """7. Adaptive execution — topology may change during the run."""
    print("\n── 7 · Adaptive streaming ──")
    runner = MACPRunner(llm_caller=_mock_llm, config=RunnerConfig(adaptive=True))
    for event in runner.stream(_create_graph()):
        text = format_event(event)
        if text:
            print(f"  {text}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    example_sync_streaming()
    example_buffer()
    example_token_streaming()
    example_print_stream()
    example_adaptive()

    asyncio.run(example_async_streaming())
    asyncio.run(example_async_token_streaming())

    print("\nAll streaming examples completed ✅")


if __name__ == "__main__":
    main()
