"""
Example: Graph visualisation.

Demonstrates several ways to render a RoleGraph:
- Mermaid (Markdown / GitHub-friendly)
- ASCII art (terminal-friendly)
- Graphviz DOT (for external tools)
- Rich coloured output (optional)
- Adjacency matrix
- Saving Mermaid / DOT to files
- Rendering PNG / SVG / PDF images (requires graphviz system package)

Run with:
    python -m examples.visualization_example
"""

import contextlib
import shutil
from pathlib import Path

from rustworkx_framework.builder import build_property_graph
from rustworkx_framework.core.agent import AgentProfile
from rustworkx_framework.core.visualization import (
    GraphVisualizer,
    MermaidDirection,
    VisualizationStyle,
    print_graph,
    render_to_image,
    to_ascii,
    to_dot,
    to_mermaid,
)

OUTPUT_DIR = Path(__file__).parent / "visualization_output"


def get_output_path(filename: str) -> Path:
    """Return an absolute path inside the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


# ── Sample graphs ─────────────────────────────────────────────────────────────

def create_sample_graph():
    """Build a four-agent graph with a parallel branch."""
    agents = [
        AgentProfile(
            agent_id="researcher",
            display_name="Researcher",
            description="Gathers and synthesises information from various sources.",
            persona="You are a thorough researcher who finds relevant information.",
            tools=["web_search", "document_reader"],
        ),
        AgentProfile(
            agent_id="analyzer",
            display_name="Data Analyzer",
            description="Analyses data and provides insights.",
            persona="You are an analytical expert who finds patterns.",
            tools=["statistics", "visualization"],
        ),
        AgentProfile(
            agent_id="writer",
            display_name="Technical Writer",
            description="Writes clear and concise documentation.",
            persona="You are a skilled technical writer.",
            tools=["formatter", "spell_checker"],
        ),
        AgentProfile(
            agent_id="reviewer",
            display_name="Quality Reviewer",
            description="Reviews and ensures quality of output.",
            persona="You ensure high quality standards.",
            tools=["grammar_check"],
        ),
    ]

    # researcher -> analyzer -> writer -> reviewer
    #           \-> writer  (parallel branch)
    edges = [
        ("researcher", "analyzer"),
        ("researcher", "writer"),
        ("analyzer", "writer"),
        ("writer", "reviewer"),
    ]

    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="Analyse the impact of AI on software development",
        include_task_node=True,
    )


# ── Demo functions ────────────────────────────────────────────────────────────

def demo_mermaid():
    """Print the graph in Mermaid format (top-bottom and left-right)."""
    print("\n-- Mermaid (top-bottom) --")
    graph = create_sample_graph()
    mermaid_tb = to_mermaid(graph, direction=MermaidDirection.TOP_BOTTOM)
    print(mermaid_tb)

    print("\n-- Mermaid (left-right, titled) --")
    mermaid_lr = to_mermaid(graph, direction=MermaidDirection.LEFT_RIGHT, title="Agent Workflow")
    print(mermaid_lr[:400] + ("..." if len(mermaid_lr) > 400 else ""))


def demo_ascii():
    """Print the graph as ASCII art with and without edge labels."""
    print("\n-- ASCII (with edges) --")
    ascii_with = to_ascii(graph=create_sample_graph(), show_edges=True)
    print(ascii_with)

    print("\n-- ASCII (nodes only) --")
    ascii_no_edges = to_ascii(graph=create_sample_graph(), show_edges=False)
    print(ascii_no_edges)


def demo_dot():
    """Print the graph as a Graphviz DOT string."""
    print("\n-- Graphviz DOT --")
    dot = to_dot(create_sample_graph(), graph_name="AgentWorkflow")
    print(dot[:500] + ("..." if len(dot) > 500 else ""))


def demo_colored():
    """Print a coloured graph using Rich (falls back to ASCII if Rich is absent)."""
    print("\n-- Coloured output --")
    graph = create_sample_graph()
    try:
        import rich  # noqa: F401
        print_graph(graph, format="colored")
    except ImportError:
        print("  (rich not installed — using ASCII fallback)")
        print_graph(graph, format="ascii")


def demo_adjacency_matrix():
    """Print the adjacency matrix for the graph."""
    print("\n-- Adjacency matrix --")
    graph = create_sample_graph()
    viz = GraphVisualizer(graph)
    matrix = viz.to_adjacency_matrix()
    print(matrix)


def demo_save_files():
    """Save Mermaid and DOT representations to disk."""
    print("\n-- Saving files --")
    graph = create_sample_graph()
    viz = GraphVisualizer(graph)

    mermaid_path = get_output_path("agent_graph.md")
    viz.save_mermaid(str(mermaid_path), title="Agent Workflow Example")
    print(f"  Mermaid saved -> {mermaid_path}")

    # Show first 300 characters of the saved Mermaid file
    with mermaid_path.open(encoding="utf-8") as f:
        content = f.read()
    print(f"  Preview: {content[:300]}{'...' if len(content) > 300 else ''}")

    dot_path = get_output_path("agent_graph.dot")
    viz.save_dot(str(dot_path), graph_name="AgentWorkflow")
    print(f"  DOT saved    -> {dot_path}")


def demo_render_images():
    """Render PNG, SVG, and PDF images (requires graphviz system package)."""
    print("\n-- Rendering images --")
    graph = create_sample_graph()

    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("  graphviz Python package not installed — skipping image rendering.")
        print("  Install with: pip install graphviz")
        return

    if not shutil.which("dot"):
        print("  Graphviz system package not found — skipping image rendering.")
        print("  Install: https://graphviz.org/download/")
        return

    for fmt, dpi in [("png", 150), ("svg", None), ("pdf", None)]:
        path = get_output_path(f"agent_graph.{fmt}")
        kwargs = {"dpi": dpi} if dpi else {}
    with contextlib.suppress(Exception):
            render_to_image(graph, str(path), format=fmt, **kwargs)
            print(f"  {fmt.upper()} saved -> {path}  ({path.stat().st_size} bytes)")


def demo_custom_styled_image():
    """Render a PNG with custom node colours and shapes."""
    print("\n-- Custom styled image --")
    graph = create_sample_graph()

    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("  graphviz not installed — skipping.")
        return

        if not shutil.which("dot"):
        print("  Graphviz system binary not found — skipping.")
            return

        from rustworkx_framework.core.visualization import NodeShape, NodeStyle

        style = VisualizationStyle(
            direction=MermaidDirection.LEFT_RIGHT,
            show_weights=True,
            show_tools=True,
            max_label_length=30,
            agent_style=NodeStyle(
                shape=NodeShape.ROUND,
            fill_color="#bbdefb",
            stroke_color="#0d47a1",
            icon="robot",
            ),
            task_style=NodeStyle(
                shape=NodeShape.DIAMOND,
            fill_color="#ffe0b2",
            stroke_color="#e65100",
            icon="task",
            ),
        )

        styled_path = get_output_path("agent_graph_styled.png")
        with contextlib.suppress(Exception):
            viz = GraphVisualizer(graph, style)
        viz.render_image(str(styled_path), format="png", dpi=150)
        if styled_path.exists():
            print(f"  Styled PNG saved -> {styled_path}")


def demo_simple_graph():
    """Visualise the smallest possible graph: 2 agents."""
    print("\n-- Simple 2-agent graph --")
    agents = [
        AgentProfile(agent_id="solver",  display_name="Problem Solver",  description="Solves problems", tools=["calculator"]),
        AgentProfile(agent_id="checker", display_name="Solution Checker", description="Verifies solutions"),
    ]
    graph = build_property_graph(
        agents,
        workflow_edges=[("solver", "checker")],
        query="Calculate 2 + 2",
        include_task_node=True,
    )
    print(to_ascii(graph, show_edges=True))


def demo_complex_graph():
    """Visualise a graph with parallel branches."""
    print("\n-- Complex graph with parallel branches --")
    agents = [
        AgentProfile(agent_id="coordinator",  display_name="Coordinator"),
        AgentProfile(agent_id="researcher_a", display_name="Researcher A"),
        AgentProfile(agent_id="researcher_b", display_name="Researcher B"),
        AgentProfile(agent_id="analyst",      display_name="Analyst"),
        AgentProfile(agent_id="synthesizer",  display_name="Synthesizer"),
    ]
    edges = [
        ("coordinator",  "researcher_a"),
        ("coordinator",  "researcher_b"),
        ("researcher_a", "analyst"),
        ("researcher_b", "analyst"),
        ("analyst",      "synthesizer"),
    ]
    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        query="Research and synthesise findings",
        include_task_node=True,
    )
    print(to_ascii(graph, show_edges=True))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Run all visualisation demos."""
    demo_simple_graph()
    demo_mermaid()
    demo_ascii()
    demo_dot()
    demo_adjacency_matrix()
    demo_colored()
    demo_complex_graph()
    demo_save_files()
    demo_render_images()
    demo_custom_styled_image()

    # Report generated files
    if OUTPUT_DIR.exists():
        files = sorted(OUTPUT_DIR.glob("agent_graph*"))
        if files:
            print(f"\nGenerated files in {OUTPUT_DIR}:")
            for f in files:
                size = f.stat().st_size
                label = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                print(f"  {f.name:<35} {label}")

    print("\nAll visualisation examples completed.")


if __name__ == "__main__":
    main()
