"""Пример визуализации графов агентов.

Демонстрирует различные способы визуализации RoleGraph:
- Mermaid (для документации и GitHub)
- ASCII art (для терминала)
- Graphviz DOT (для внешних инструментов)
- Rich Console (цветной вывод)

Запуск:
    python -m rustworkx_framework.examples.visualization_example
"""

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
            identifier="researcher",
            display_name="Researcher",
            description="Gathers and synthesizes information from various sources.",
            persona="You are a thorough researcher who finds relevant information.",
            tools=["web_search", "document_reader"],
        ),
        AgentProfile(
            identifier="analyzer",
            display_name="Data Analyzer",
            description="Analyzes data and provides insights.",
            persona="You are an analytical expert who finds patterns.",
            tools=["statistics", "visualization"],
        ),
        AgentProfile(
            identifier="writer",
            display_name="Technical Writer",
            description="Writes clear and concise documentation.",
            persona="You are a skilled technical writer.",
            tools=["formatter", "spell_checker"],
        ),
        AgentProfile(
            identifier="reviewer",
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

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        query="Analyze the impact of AI on software development",
        include_task_node=True,
    )

    return graph


def demo_mermaid():
    """Демонстрация Mermaid формата."""
    print("\n" + "=" * 60)
    print("📊 MERMAID FORMAT")
    print("=" * 60)

    graph = create_sample_graph()

    # Простой вывод
    print("\n--- Top-Bottom направление ---")
    mermaid = to_mermaid(graph, direction=MermaidDirection.TOP_BOTTOM)
    print(mermaid)

    # Left-Right направление
    print("\n--- Left-Right направление ---")
    mermaid_lr = to_mermaid(
        graph,
        direction=MermaidDirection.LEFT_RIGHT,
        title="Agent Workflow",
    )
    print(mermaid_lr)

    # С кастомным стилем
    print("\n--- С кастомным стилем (показ весов) ---")
    style = VisualizationStyle(
        direction=MermaidDirection.TOP_BOTTOM,
        show_weights=True,
        show_tools=True,
    )
    viz = GraphVisualizer(graph, style)
    print(viz.to_mermaid())


def demo_ascii():
    """Демонстрация ASCII формата."""
    print("\n" + "=" * 60)
    print("📝 ASCII FORMAT")
    print("=" * 60)

    graph = create_sample_graph()

    print("\n--- Полный вывод ---")
    ascii_art = to_ascii(graph, show_edges=True)
    print(ascii_art)

    print("\n--- Только узлы ---")
    ascii_nodes = to_ascii(graph, show_edges=False)
    print(ascii_nodes)


def demo_dot():
    """Демонстрация Graphviz DOT формата."""
    print("\n" + "=" * 60)
    print("🔵 GRAPHVIZ DOT FORMAT")
    print("=" * 60)

    graph = create_sample_graph()

    print("\n--- DOT код ---")
    dot = to_dot(graph, graph_name="AgentWorkflow")
    print(dot)

    print("\n💡 Tip: Сохраните в файл и используйте команду:")
    print("   dot -Tpng graph.dot -o graph.png")
    print("   dot -Tsvg graph.dot -o graph.svg")


def demo_colored():
    """Демонстрация цветного вывода (Rich)."""
    print("\n" + "=" * 60)
    print("🌈 COLORED OUTPUT (Rich)")
    print("=" * 60)

    graph = create_sample_graph()

    try:
        from rich.console import Console  # noqa: F401

        print("\n--- Rich Console Output ---")
        print_graph(graph, format="colored")
    except ImportError:
        print("\n⚠️  Rich не установлен. Установите: pip install rich")
        print("    Показываю ASCII fallback:")
        print_graph(graph, format="ascii")


def demo_adjacency_matrix():
    """Демонстрация матрицы смежности."""
    print("\n" + "=" * 60)
    print("📐 ADJACENCY MATRIX")
    print("=" * 60)

    graph = create_sample_graph()
    viz = GraphVisualizer(graph)

    print("\n--- Матрица смежности ---")
    matrix = viz.to_adjacency_matrix()
    print(matrix)


def demo_save_files():
    """Демонстрация сохранения в файлы."""
    print("\n" + "=" * 60)
    print("💾 SAVE TO FILES")
    print("=" * 60)

    graph = create_sample_graph()
    viz = GraphVisualizer(graph)

    # Сохраняем Mermaid
    mermaid_path = get_output_path("agent_graph.md")
    viz.save_mermaid(mermaid_path, title="Agent Workflow Example")
    print(f"\n✅ Mermaid saved to: {mermaid_path}")

    # Сохраняем DOT
    dot_path = get_output_path("agent_graph.dot")
    viz.save_dot(dot_path, graph_name="AgentWorkflow")
    print(f"✅ DOT saved to: {dot_path}")

    # Показываем содержимое Mermaid файла
    print(f"\n--- Content of {mermaid_path} ---")
    with open(mermaid_path) as f:
        print(f.read())


def demo_render_images():
    """Демонстрация рендеринга изображений."""
    print("\n" + "=" * 60)
    print("🖼️  RENDER TO IMAGES")
    print("=" * 60)

    graph = create_sample_graph()

    # Проверяем наличие Python библиотеки
    try:
        import graphviz  # noqa: F401
    except ImportError:
        print("\n⚠️  Python библиотека graphviz не установлена!")
        print("\n📦 Установка:")
        print("   uv add graphviz")
        print("   # или: pip install graphviz")
        return

    # Проверяем наличие системного Graphviz
    import shutil

    if not shutil.which("dot"):
        print("\n⚠️  Системный Graphviz не установлен!")
        print("\n📦 Установка системного Graphviz:")
        print("   Ubuntu/Debian:")
        print("      sudo apt install graphviz")
        print("\n   macOS:")
        print("      brew install graphviz")
        print("\n   Windows:")
        print("      winget install graphviz")
        print("      # или скачайте с https://graphviz.org/download/")
        print("\n   После установки перезапустите терминал!")
        return

    print("\n✅ Graphviz установлен (Python + система), создаём изображения...\n")

    # PNG изображение
    png_path = get_output_path("agent_graph.png")
    try:
        render_to_image(graph, png_path, format="png", dpi=150)
        print(f"✅ PNG создан: {png_path}")
    except Exception as e:
        print(f"❌ Ошибка при создании PNG: {e}")

    # SVG изображение (векторное, масштабируется без потери качества)
    svg_path = get_output_path("agent_graph.svg")
    try:
        render_to_image(graph, svg_path, format="svg")
        print(f"✅ SVG создан: {svg_path}")
    except Exception as e:
        print(f"❌ Ошибка при создании SVG: {e}")

    # PDF изображение
    pdf_path = get_output_path("agent_graph.pdf")
    try:
        render_to_image(graph, pdf_path, format="pdf")
        print(f"✅ PDF создан: {pdf_path}")
    except Exception as e:
        print(f"❌ Ошибка при создании PDF: {e}")

    print("\n💡 Tip: Используйте SVG для веб-страниц, PNG для документов")
    print("         PDF подходит для печати и профессиональной документации")

    # Опционально: показать интерактивно
    print("\n🔍 Хотите открыть граф интерактивно?")
    print("   Раскомментируйте строку show_graph_interactive() в коде")
    # show_graph_interactive(graph)  # Откроет в системном просмотрщике


def demo_custom_styled_image():
    """Демонстрация рендеринга с кастомным стилем."""
    print("\n" + "=" * 60)
    print("🎨 CUSTOM STYLED IMAGES")
    print("=" * 60)

    graph = create_sample_graph()

    try:
        import shutil

        import graphviz  # noqa: F401

        if not shutil.which("dot"):
            print("\n⚠️  Системный Graphviz не установлен. См. инструкции выше.")
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
            print(f"\n✅ Styled PNG создан: {styled_path}")
            print("   • Left-Right направление")
            print("   • Кастомные цвета")
            print("   • Показаны веса рёбер")
            print("   • Показаны инструменты агентов")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

    except ImportError:
        print("\n⚠️  Graphviz не установлен. См. инструкции выше.")


def demo_simple_graph():
    """Демонстрация на простом графе."""
    print("\n" + "=" * 60)
    print("🔷 SIMPLE 2-AGENT GRAPH")
    print("=" * 60)

    # Минимальный граф
    agents = [
        AgentProfile(
            identifier="solver",
            display_name="Problem Solver",
            description="Solves problems",
            tools=["calculator"],
        ),
        AgentProfile(
            identifier="checker",
            display_name="Solution Checker",
            description="Verifies solutions",
        ),
    ]

    graph = build_property_graph(
        agents,
        workflow_edges=[("solver", "checker")],
        query="Calculate 2 + 2",
        include_task_node=True,
    )

    print("\n--- Mermaid ---")
    print(to_mermaid(graph))

    print("\n--- ASCII ---")
    print(to_ascii(graph))


def demo_complex_graph():
    """Демонстрация на сложном графе с параллельными ветками."""
    print("\n" + "=" * 60)
    print("🔶 COMPLEX PARALLEL GRAPH")
    print("=" * 60)

    # Сложный граф с параллельными путями
    agents = [
        AgentProfile(identifier="coordinator", display_name="Coordinator"),
        AgentProfile(identifier="researcher_a", display_name="Researcher A"),
        AgentProfile(identifier="researcher_b", display_name="Researcher B"),
        AgentProfile(identifier="analyst", display_name="Analyst"),
        AgentProfile(identifier="synthesizer", display_name="Synthesizer"),
    ]

    # Параллельные ветки: coordinator -> (researcher_a, researcher_b) -> analyst -> synthesizer
    edges = [
        ("coordinator", "researcher_a"),
        ("coordinator", "researcher_b"),
        ("researcher_a", "analyst"),
        ("researcher_b", "analyst"),
        ("analyst", "synthesizer"),
    ]

    graph = build_property_graph(
        agents,
        workflow_edges=edges,
        query="Research and synthesize findings",
        include_task_node=True,
    )

    print("\n--- Mermaid (Left-Right) ---")
    print(to_mermaid(graph, direction=MermaidDirection.LEFT_RIGHT))

    print("\n--- ASCII ---")
    print(to_ascii(graph))


def main():
    """Запустить все демонстрации."""
    print("=" * 60)
    print("🎨 MECE Framework - Graph Visualization Examples")
    print("=" * 60)

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

    print("\n" + "=" * 60)
    print("✅ All visualization examples completed!")
    print("=" * 60)
    print(f"\n📁 Файлы созданы в: {OUTPUT_DIR.absolute()}")
    print("   - agent_graph.md (Mermaid)")
    print("   - agent_graph.dot (DOT)")

    # Проверяем какие файлы реально созданы
    if OUTPUT_DIR.exists():
        created_files = list(OUTPUT_DIR.glob("agent_graph*"))
        if created_files:
            print("\n   Созданные файлы:")
            for f in sorted(created_files):
                size = f.stat().st_size
                size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
                print(f"   ✓ {f.name} ({size_str})")
        else:
            print("\n   ⚠️  Изображения не созданы (требуется системный Graphviz)")

    print("\n💡 Для создания изображений установите системный Graphviz:")
    print("   Ubuntu/Debian: sudo apt install graphviz")
    print("   macOS: brew install graphviz")
    print("=" * 60)


if __name__ == "__main__":
    main()
