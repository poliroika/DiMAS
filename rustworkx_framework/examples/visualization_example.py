"""
Пример визуализации графов агентов.

Демонстрирует различные способы визуализации RoleGraph:
- Mermaid (для документации и GitHub)
- ASCII art (для терминала)
- Graphviz DOT (для внешних инструментов)
- Rich Console (цветной вывод)

Запуск:
    python -m rustworkx_framework.examples.visualization_example
"""

import contextlib
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

# Директория для сохранения выходных файлов
OUTPUT_DIR = Path(__file__).parent / "visualization_output"


def get_output_path(filename: str) -> str:
    """Получить путь для сохранения файла в директории с примерами."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / filename)


def create_sample_graph():
    """Создать пример графа с агентами."""
    agents = [
        AgentProfile(
            agent_id="researcher",
            display_name="Researcher",
            description="Gathers and synthesizes information from various sources.",
            persona="You are a thorough researcher who finds relevant information.",
            tools=["web_search", "document_reader"],
        ),
        AgentProfile(
            agent_id="analyzer",
            display_name="Data Analyzer",
            description="Analyzes data and provides insights.",
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

    # Создаём граф: researcher -> analyzer -> writer -> reviewer
    #                          \-> writer (параллельно)
    edges = [
        ("researcher", "analyzer"),
        ("researcher", "writer"),  # Параллельная ветка
        ("analyzer", "writer"),
        ("writer", "reviewer"),
    ]

    return build_property_graph(
        agents,
        workflow_edges=edges,
        query="Analyze the impact of AI on software development",
        include_task_node=True,
    )


def demo_mermaid():
    """Демонстрация Mermaid формата."""
    graph = create_sample_graph()

    # Простой вывод
    to_mermaid(graph, direction=MermaidDirection.TOP_BOTTOM)

    # Left-Right направление
    to_mermaid(
        graph,
        direction=MermaidDirection.LEFT_RIGHT,
        title="Agent Workflow",
    )

    # С кастомным стилем
    style = VisualizationStyle(
        direction=MermaidDirection.TOP_BOTTOM,
        show_weights=True,
        show_tools=True,
    )
    GraphVisualizer(graph, style)


def demo_ascii():
    """Демонстрация ASCII формата."""
    graph = create_sample_graph()

    to_ascii(graph, show_edges=True)

    to_ascii(graph, show_edges=False)


def demo_dot():
    """Демонстрация Graphviz DOT формата."""
    graph = create_sample_graph()

    to_dot(graph, graph_name="AgentWorkflow")


def demo_colored():
    """Демонстрация цветного вывода (Rich)."""
    graph = create_sample_graph()

    try:
        from rich.console import Console  # noqa: F401

        print_graph(graph, format="colored")
    except ImportError:
        print_graph(graph, format="ascii")


def demo_adjacency_matrix():
    """Демонстрация матрицы смежности."""
    graph = create_sample_graph()
    viz = GraphVisualizer(graph)

    viz.to_adjacency_matrix()


def demo_save_files():
    """Демонстрация сохранения в файлы."""
    graph = create_sample_graph()
    viz = GraphVisualizer(graph)

    # Сохраняем Mermaid
    mermaid_path = get_output_path("agent_graph.md")
    viz.save_mermaid(mermaid_path, title="Agent Workflow Example")

    # Сохраняем DOT
    dot_path = get_output_path("agent_graph.dot")
    viz.save_dot(dot_path, graph_name="AgentWorkflow")

    # Показываем содержимое Mermaid файла
    with open(mermaid_path):
        pass


def demo_render_images():
    """Демонстрация рендеринга изображений."""
    graph = create_sample_graph()

    # Проверяем наличие Python библиотеки
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return

    # Проверяем наличие системного Graphviz
    import shutil

    if not shutil.which("dot"):
        return

    # PNG изображение
    png_path = get_output_path("agent_graph.png")
    with contextlib.suppress(Exception):
        render_to_image(graph, png_path, format="png", dpi=150)

    # SVG изображение (векторное, масштабируется без потери качества)
    svg_path = get_output_path("agent_graph.svg")
    with contextlib.suppress(Exception):
        render_to_image(graph, svg_path, format="svg")

    # PDF изображение
    pdf_path = get_output_path("agent_graph.pdf")
    with contextlib.suppress(Exception):
        render_to_image(graph, pdf_path, format="pdf")

    # Опционально: показать интерактивно
    # show_graph_interactive(graph)  # Откроет в системном просмотрщике


def demo_custom_styled_image():
    """Демонстрация рендеринга с кастомным стилем."""
    graph = create_sample_graph()

    try:
        import shutil

        import graphviz  # noqa: F401

        if not shutil.which("dot"):
            return

        # Создаём граф с кастомным стилем
        from rustworkx_framework.core.visualization import NodeShape, NodeStyle

        style = VisualizationStyle(
            direction=MermaidDirection.LEFT_RIGHT,
            show_weights=True,
            show_tools=True,
            max_label_length=30,
            agent_style=NodeStyle(
                shape=NodeShape.ROUND,
                fill_color="#bbdefb",  # Светло-синий
                stroke_color="#0d47a1",  # Тёмно-синий
                icon="🤖",
            ),
            task_style=NodeStyle(
                shape=NodeShape.DIAMOND,
                fill_color="#ffe0b2",  # Светло-оранжевый
                stroke_color="#e65100",  # Тёмно-оранжевый
                icon="📋",
            ),
        )

        styled_path = get_output_path("agent_graph_styled.png")
        try:
            viz = GraphVisualizer(graph, style)
            viz.render_image(styled_path, format="png", dpi=150)
        except Exception:
            pass

    except ImportError:
        pass


def demo_simple_graph():
    """Демонстрация на простом графе."""
    # Минимальный граф
    agents = [
        AgentProfile(
            agent_id="solver",
            display_name="Problem Solver",
            description="Solves problems",
            tools=["calculator"],
        ),
        AgentProfile(
            agent_id="checker",
            display_name="Solution Checker",
            description="Verifies solutions",
        ),
    ]

    build_property_graph(
        agents,
        workflow_edges=[("solver", "checker")],
        query="Calculate 2 + 2",
        include_task_node=True,
    )


def demo_complex_graph():
    """Демонстрация на сложном графе с параллельными ветками."""
    # Сложный граф с параллельными путями
    agents = [
        AgentProfile(agent_id="coordinator", display_name="Coordinator"),
        AgentProfile(agent_id="researcher_a", display_name="Researcher A"),
        AgentProfile(agent_id="researcher_b", display_name="Researcher B"),
        AgentProfile(agent_id="analyst", display_name="Analyst"),
        AgentProfile(agent_id="synthesizer", display_name="Synthesizer"),
    ]

    # Параллельные ветки: coordinator -> (researcher_a, researcher_b) -> analyst -> synthesizer
    edges = [
        ("coordinator", "researcher_a"),
        ("coordinator", "researcher_b"),
        ("researcher_a", "analyst"),
        ("researcher_b", "analyst"),
        ("analyst", "synthesizer"),
    ]

    build_property_graph(
        agents,
        workflow_edges=edges,
        query="Research and synthesize findings",
        include_task_node=True,
    )


def main():
    """Запустить все демонстрации."""
    # Простой граф
    demo_simple_graph()

    # Основной пример
    demo_mermaid()
    demo_ascii()
    demo_dot()

    # Матрица смежности
    demo_adjacency_matrix()

    # Цветной вывод
    demo_colored()

    # Сложный граф
    demo_complex_graph()

    # Сохранение файлов
    demo_save_files()

    # 🎨 НОВОЕ: Рендеринг изображений
    demo_render_images()
    demo_custom_styled_image()

    # Проверяем какие файлы реально созданы
    if OUTPUT_DIR.exists():
        created_files = list(OUTPUT_DIR.glob("agent_graph*"))
        if created_files:
            for f in sorted(created_files):
                size = f.stat().st_size
                f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
        else:
            pass


if __name__ == "__main__":
    main()
