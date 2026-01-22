"""Streaming execution example.

Demonstrates LangGraph-like streaming capabilities with real-time output.

Run with: uv run python -m rustworkx_framework.examples.streaming_example
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


def create_example_graph():
    """Create a simple multi-agent graph for demonstration."""
    agents = [
        AgentProfile(
            identifier="researcher",
            display_name="Research Specialist",
            persona="You are an AI researcher who explains technical concepts clearly.",
            description="Gather and synthesize information about the topic.",
        ),
        AgentProfile(
            identifier="writer",
            display_name="Content Writer",
            persona="You are a skilled writer who creates engaging content.",
            description="Transform research into readable content.",
        ),
        AgentProfile(
            identifier="editor",
            display_name="Editor",
            persona="You are an editor who ensures clarity and quality.",
            description="Review and polish the final content.",
        ),
    ]

    # Define communication flow: researcher -> writer -> editor
    edges = [
        ("researcher", "writer"),
        ("writer", "editor"),
    ]

    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="Explain how AI works in simple terms",
        include_task_node=True,
    )


def mock_llm(prompt: str) -> str:
    """Mock LLM for demonstration purposes."""
    if "researcher" in prompt.lower() or "research" in prompt.lower():
        return (
            "AI (Artificial Intelligence) works by using mathematical models "
            "trained on large amounts of data to recognize patterns and make predictions. "
            "Key components include neural networks, machine learning algorithms, "
            "and training data. Modern AI systems like ChatGPT use transformer "
            "architectures with billions of parameters."
        )
    elif "writer" in prompt.lower() or "content" in prompt.lower():
        return (
            "## How AI Works: A Simple Guide\n\n"
            "Imagine teaching a child to recognize cats by showing them thousands "
            "of cat pictures. AI works similarly - it learns from examples!\n\n"
            "**The Basics:**\n"
            "- AI uses 'neural networks' inspired by the human brain\n"
            "- It learns patterns from massive amounts of data\n"
            "- After training, it can make predictions on new data\n\n"
            "Modern AI assistants like ChatGPT have seen billions of text examples, "
            "helping them understand and generate human-like responses."
        )
    else:
        return (
            "# How AI Works: A Simple Guide\n\n"
            "AI learns from examples, just like humans! By analyzing patterns in data, "
            "AI systems can recognize images, understand language, and even have "
            "conversations. The key is training on lots of examples until patterns emerge.\n\n"
            "✅ Content reviewed and polished for clarity."
        )


def mock_streaming_llm(prompt: str):
    """Mock streaming LLM that yields tokens one by one."""
    response = mock_llm(prompt)
    # Simulate token-by-token streaming
    words = response.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


async def mock_async_llm(prompt: str) -> str:
    """Mock async LLM."""
    await asyncio.sleep(0.1)  # Simulate API latency
    return mock_llm(prompt)


async def mock_async_streaming_llm(prompt: str):
    """Mock async streaming LLM."""
    response = mock_llm(prompt)
    words = response.split(" ")
    for i, word in enumerate(words):
        await asyncio.sleep(0.02)  # Simulate streaming delay
        yield word + (" " if i < len(words) - 1 else "")


def example_sync_streaming():
    """Example: Synchronous streaming execution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Synchronous Streaming")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    print("Streaming events:\n")
    for event in runner.stream(graph):
        # Format and print each event
        print(format_event(event, verbose=False))


def example_sync_streaming_with_buffer():
    """Example: Streaming with buffer for collecting results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Streaming with Buffer")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    buffer = StreamBuffer()

    for event in runner.stream(graph):
        buffer.add(event)

        # Only show agent outputs
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            print(f"📝 {event.agent_name}: {event.content[:80]}...")
            print()

    print("\n--- Final Answer ---")
    print(buffer.final_answer)
    print(f"\nFinal agent: {buffer.final_agent_id}")


def example_token_streaming():
    """Example: Token-level streaming."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Token-Level Streaming")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(streaming_llm_caller=mock_streaming_llm, config=config)

    current_agent = None
    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                print("\n")
            current_agent = event.agent_name
            print(f"\n🤖 {event.agent_name}: ", end="", flush=True)

        elif event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

        elif event.event_type == StreamEventType.RUN_END:
            print(f"\n\n✅ Completed in {event.total_time:.2f}s")


async def example_async_streaming():
    """Example: Async streaming execution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Async Streaming")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    runner = MACPRunner(async_llm_caller=mock_async_llm)

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            print(f"▶️  Starting {event.agent_name}...")
        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            print(f"✓  {event.agent_name} completed ({event.tokens_used} tokens)")
        elif event.event_type == StreamEventType.RUN_END:
            print(f"\n🏁 All done! Total: {event.total_tokens} tokens, {event.total_time:.2f}s")


async def example_async_token_streaming():
    """Example: Async token-level streaming."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Async Token Streaming")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(async_streaming_llm_caller=mock_async_streaming_llm, config=config)

    print("Streaming tokens in real-time:\n")
    current_agent = None

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                print("\n")
            current_agent = event.agent_name
            print(f"\n🤖 {event.agent_name}: ", end="", flush=True)

        elif event.event_type == StreamEventType.TOKEN:
            print(event.token, end="", flush=True)

        elif event.event_type == StreamEventType.RUN_END:
            print("\n\n✅ Streaming complete!")


def example_print_stream_helper():
    """Example: Using print_stream helper."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Using print_stream Helper")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    # print_stream handles all formatting and returns final answer
    final_answer = print_stream(runner.stream(graph), show_tokens=False, verbose=True)
    print(f"\n--- Returned final answer length: {len(final_answer)} chars ---")


def example_adaptive_streaming():
    """Example: Adaptive execution with streaming."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Adaptive Streaming")
    print("=" * 60 + "\n")

    graph = create_example_graph()
    config = RunnerConfig(
        adaptive=True,
        enable_replanning=True,
    )
    runner = MACPRunner(llm_caller=mock_llm, config=config)

    for event in runner.stream(graph):
        formatted = format_event(event)
        if formatted:  # Skip empty formats
            print(formatted)


def main():
    """Run all streaming examples."""
    print("🔄 STREAMING EXECUTION EXAMPLES")
    print("=" * 60)
    print("These examples demonstrate LangGraph-like streaming capabilities")
    print("for real-time monitoring of multi-agent execution.\n")

    # Sync examples
    example_sync_streaming()
    example_sync_streaming_with_buffer()
    example_token_streaming()
    example_print_stream_helper()
    example_adaptive_streaming()

    # Async examples
    asyncio.run(example_async_streaming())
    asyncio.run(example_async_token_streaming())

    print("\n" + "=" * 60)
    print("✅ All streaming examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
