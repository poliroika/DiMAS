"""
Graph visualisation — no LLM required.

Demonstrates several ways to render a RoleGraph:
  - Mermaid (Markdown / GitHub-friendly)
  - ASCII art (terminal-friendly)
  - Graphviz DOT (for external tools)
  - Rich coloured output (optional)
  - Adjacency matrix
  - Saving Mermaid / DOT to files
  - Rendering PNG / SVG / PDF images (requires graphviz system package)

Run:
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

# ── Constants ───────────────────────────────────────────────────────────────────

MERMAID_PREVIEW_LENGTH = 400
DOT_PREVIEW_LENGTH = 500
BYTES_PER_KB = 1024

OUTPUT_DIR = Path(__file__).parent / "visualization_output"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _header(title: str) -> None:
    print(f"\n── {title} ──")


# ── Sample graphs ─────────────────────────────────────────────────────────────


def _sample_graph():
    """Four-agent graph with a parallel branch."""
    agents = [
        AgentProfile(
            agent_id="researcher",
            display_name="Researcher",
            description="Gathers and synthesises information.",
            persona="A thorough researcher.",
            tools=["web_search", "document_reader"],
        ),
        AgentProfile(
            agent_id="analyzer",
            display_name="Data Analyzer",
            description="Analyses data and provides insights.",
            persona="An analytical expert.",
            tools=["statistics", "visualization"],
        ),
        AgentProfile(
            agent_id="writer",
            display_name="Technical Writer",
            description="Writes clear documentation.",
            persona="A skilled technical writer.",
            tools=["formatter", "spell_checker"],
        ),
        AgentProfile(
            agent_id="reviewer",
            display_name="Quality Reviewer",
            description="Reviews and ensures quality.",
            persona="Ensures high quality.",
            tools=["grammar_check"],
        ),
    ]
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


def _simple_graph():
    """Minimal 2-agent graph."""
    agents = [
        AgentProfile(
            agent_id="solver", display_name="Problem Solver", description="Solves problems", tools=["calculator"]
        ),
        AgentProfile(agent_id="checker", display_name="Solution Checker", description="Verifies solutions"),
    ]
    return build_property_graph(
        agents,
        workflow_edges=[("solver", "checker")],
        query="Calculate 2 + 2",
        include_task_node=True,
    )


def _complex_graph():
    """Graph with parallel branches."""
    agents = [
        AgentProfile(agent_id="coordinator", display_name="Coordinator"),
        AgentProfile(agent_id="researcher_a", display_name="Researcher A"),
        AgentProfile(agent_id="researcher_b", display_name="Researcher B"),
        AgentProfile(agent_id="analyst", display_name="Analyst"),
        AgentProfile(agent_id="synthesizer", display_name="Synthesizer"),
    ]
    edges = [
        ("coordinator", "researcher_a"),
        ("coordinator", "researcher_b"),
        ("researcher_a", "analyst"),
        ("researcher_b", "analyst"),
        ("analyst", "synthesizer"),
    ]
    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="Research and synthesise findings",
        include_task_node=True,
    )


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_simple_graph():
    _header("Simple 2-agent graph")
    print(to_ascii(_simple_graph(), show_edges=True))


def demo_mermaid():
    _header("Mermaid (top-bottom)")
    print(to_mermaid(_sample_graph(), direction=MermaidDirection.TOP_BOTTOM))

    _header("Mermaid (left-right, titled)")
    text = to_mermaid(_sample_graph(), direction=MermaidDirection.LEFT_RIGHT, title="Agent Workflow")
    print(text[:MERMAID_PREVIEW_LENGTH] + ("…" if len(text) > MERMAID_PREVIEW_LENGTH else ""))


def demo_ascii():
    _header("ASCII (with edges)")
    print(to_ascii(_sample_graph(), show_edges=True))

    _header("ASCII (nodes only)")
    print(to_ascii(_sample_graph(), show_edges=False))


def demo_dot():
    _header("Graphviz DOT")
    dot = to_dot(_sample_graph(), graph_name="AgentWorkflow")
    print(dot[:DOT_PREVIEW_LENGTH] + ("…" if len(dot) > DOT_PREVIEW_LENGTH else ""))


def demo_colored():
    _header("Coloured output")
    try:
        import rich  # noqa: F401

        print_graph(_sample_graph(), format="colored")
    except ImportError:
        print("  (rich not installed — ASCII fallback)")
        print_graph(_sample_graph(), format="ascii")


def demo_adjacency_matrix():
    _header("Adjacency matrix")
    print(GraphVisualizer(_sample_graph()).to_adjacency_matrix())


def demo_complex_graph():
    _header("Complex graph with parallel branches")
    print(to_ascii(_complex_graph(), show_edges=True))


def demo_save_files():
    _header("Saving files")
    out = _ensure_output_dir()
    viz = GraphVisualizer(_sample_graph())

    mermaid_path = out / "agent_graph.md"
    viz.save_mermaid(str(mermaid_path), title="Agent Workflow Example")
    print(f"  Mermaid → {mermaid_path}")

    dot_path = out / "agent_graph.dot"
    viz.save_dot(str(dot_path), graph_name="AgentWorkflow")
    print(f"  DOT     → {dot_path}")


def demo_render_images():
    _header("Rendering images")
    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("  graphviz Python package not installed — skipping.")
        return

    if not shutil.which("dot"):
        print("  Graphviz system binary not found — skipping.")
        return

    out = _ensure_output_dir()
    graph = _sample_graph()
    for fmt, dpi in [("png", 150), ("svg", None), ("pdf", None)]:
        path = out / f"agent_graph.{fmt}"
        kwargs = {"dpi": dpi} if dpi else {}
        with contextlib.suppress(Exception):
            render_to_image(graph, str(path), format=fmt, **kwargs)
            print(f"  {fmt.upper()} → {path}  ({path.stat().st_size} bytes)")


def demo_custom_styled_image():
    _header("Custom styled image")
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

    out = _ensure_output_dir()
    path = out / "agent_graph_styled.png"
    with contextlib.suppress(Exception):
        viz = GraphVisualizer(_sample_graph(), style)
        viz.render_image(str(path), format="png", dpi=150)
        if path.exists():
            print(f"  Styled PNG → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
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

    if OUTPUT_DIR.exists():
        files = sorted(OUTPUT_DIR.glob("agent_graph*"))
        if files:
            print(f"\nGenerated files in {OUTPUT_DIR}:")
            for f in files:
                size = f.stat().st_size
                label = f"{size / BYTES_PER_KB:.1f} KB" if size > BYTES_PER_KB else f"{size} B"
                print(f"  {f.name:<35} {label}")

    print("\nAll visualisation examples completed ✅")


if __name__ == "__main__":
    main()
