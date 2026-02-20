"""
Streaming execution example.

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
            agent_id="researcher",
            display_name="Research Specialist",
            persona="You are an AI researcher who explains technical concepts clearly.",
            description="Gather and synthesize information about the topic.",
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
    if "writer" in prompt.lower() or "content" in prompt.lower():
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
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    for _event in runner.stream(graph):
        # Format and print each event
        pass


def example_sync_streaming_with_buffer():
    """Example: Streaming with buffer for collecting results."""
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    buffer = StreamBuffer()

    for event in runner.stream(graph):
        buffer.add(event)

        # Only show agent outputs
        if event.event_type == StreamEventType.AGENT_OUTPUT:
            pass


def example_token_streaming():
    """Example: Token-level streaming."""
    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(streaming_llm_caller=mock_streaming_llm, config=config)

    current_agent = None
    for event in runner.stream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                pass
            current_agent = getattr(event, "agent_name", "")

        elif event.event_type in (StreamEventType.TOKEN, StreamEventType.RUN_END):
            pass


async def example_async_streaming():
    """Example: Async streaming execution."""
    graph = create_example_graph()
    runner = MACPRunner(async_llm_caller=mock_async_llm)

    async for event in runner.astream(graph):
        if (
            event.event_type in (StreamEventType.AGENT_START, StreamEventType.AGENT_OUTPUT)
            or event.event_type == StreamEventType.RUN_END
        ):
            pass


async def example_async_token_streaming():
    """Example: Async token-level streaming."""
    graph = create_example_graph()
    config = RunnerConfig(enable_token_streaming=True)
    runner = MACPRunner(async_streaming_llm_caller=mock_async_streaming_llm, config=config)

    current_agent = None

    async for event in runner.astream(graph):
        if event.event_type == StreamEventType.AGENT_START:
            if current_agent:
                pass
            current_agent = getattr(event, "agent_name", "")

        elif event.event_type in (StreamEventType.TOKEN, StreamEventType.RUN_END):
            pass


def example_print_stream_helper():
    """Example: Using print_stream helper."""
    graph = create_example_graph()
    runner = MACPRunner(llm_caller=mock_llm)

    # print_stream handles all formatting and returns final answer
    print_stream(runner.stream(graph), show_tokens=False, verbose=True)


def example_adaptive_streaming():
    """Example: Adaptive execution with streaming."""
    graph = create_example_graph()
    config = RunnerConfig(
        adaptive=True,
    )
    runner = MACPRunner(llm_caller=mock_llm, config=config)

    for event in runner.stream(graph):
        formatted = format_event(event)
        if formatted:  # Skip empty formats
            pass


def main():
    """Run all streaming examples."""
    # Sync examples
    example_sync_streaming()
    example_sync_streaming_with_buffer()
    example_token_streaming()
    example_print_stream_helper()
    example_adaptive_streaming()

    # Async examples
    asyncio.run(example_async_streaming())
    asyncio.run(example_async_token_streaming())


if __name__ == "__main__":
    main()
