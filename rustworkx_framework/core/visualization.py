"""Визуализация графов агентов.

Поддерживает:
- Mermaid (для Markdown/GitHub/документации)
- ASCII art (для терминала)
- Graphviz DOT (для внешних инструментов)
- Rich Console (цветной вывод в терминал)

Использование:
    from rustworkx_framework.core.visualization import GraphVisualizer

    viz = GraphVisualizer(graph)
    print(viz.to_mermaid())
    print(viz.to_ascii())
    viz.print_colored()  # Rich console output
"""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

__all__ = [
    "GraphVisualizer",
    "VisualizationStyle",
    "NodeStyle",
    "EdgeStyle",
    "MermaidDirection",
    "to_mermaid",
    "to_ascii",
    "to_dot",
    "print_graph",
    "render_to_image",
    "show_graph_interactive",
]

if TYPE_CHECKING:
    from rustworkx_framework.core.graph import RoleGraph


class MermaidDirection(str, Enum):
    """Направление графа в Mermaid."""

    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class NodeShape(str, Enum):
    """Формы узлов в Mermaid."""

    RECTANGLE = "rect"
    ROUND = "round"
    STADIUM = "stadium"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallelogram"
    TRAPEZOID = "trapezoid"


class NodeStyle(BaseModel):
    """Стиль отображения узла."""

    shape: NodeShape = NodeShape.ROUND
    fill_color: str = "#e1f5fe"
    stroke_color: str = "#01579b"
    text_color: str = "#000000"
    icon: str = ""  # Emoji или символ


class EdgeStyle(BaseModel):
    """Стиль отображения ребра."""

    line_style: str = "solid"  # solid, dashed, dotted
    arrow_head: str = "normal"  # normal, none, diamond
    color: str = "#666666"
    label_color: str = "#333333"


class VisualizationStyle(BaseModel):
    """Общий стиль визуализации."""

    direction: MermaidDirection = MermaidDirection.TOP_BOTTOM
    agent_style: NodeStyle = Field(
        default_factory=lambda: NodeStyle(
            shape=NodeShape.ROUND,
            fill_color="#e3f2fd",
            stroke_color="#1976d2",
            icon="🤖",
        )
    )
    task_style: NodeStyle = Field(
        default_factory=lambda: NodeStyle(
            shape=NodeShape.DIAMOND,
            fill_color="#fff3e0",
            stroke_color="#f57c00",
            icon="📋",
        )
    )
    workflow_edge_style: EdgeStyle = Field(
        default_factory=lambda: EdgeStyle(
            line_style="solid",
            color="#1976d2",
        )
    )
    task_edge_style: EdgeStyle = Field(
        default_factory=lambda: EdgeStyle(
            line_style="dashed",
            color="#f57c00",
        )
    )
    show_weights: bool = False
    show_probabilities: bool = False
    show_tools: bool = True
    show_descriptions: bool = False
    max_label_length: int = 30


class GraphVisualizer:
    """Визуализатор RoleGraph в различных форматах."""

    def __init__(
        self,
        graph: "RoleGraph",
        style: VisualizationStyle | None = None,
    ):
        """Создать визуализатор для графа.

        Args:
            graph: RoleGraph для визуализации
            style: Стиль визуализации (по умолчанию создаётся новый)
        """
        self.graph = graph
        self.style = style or VisualizationStyle()

    def to_mermaid(
        self,
        direction: MermaidDirection | None = None,
        title: str | None = None,
    ) -> str:
        """Экспортировать граф в Mermaid формат.

        Args:
            direction: Направление графа (TB, LR, etc.)
            title: Заголовок диаграммы

        Returns:
            Mermaid-код диаграммы

        Example:
            ```mermaid
            flowchart TB
                researcher[🤖 Researcher]
                analyzer[🤖 Analyzer]
                researcher --> analyzer
            ```
        """
        direction = direction or self.style.direction
        lines = []

        # Заголовок
        if title:
            lines.append("---")
            lines.append(f"title: {title}")
            lines.append("---")

        lines.append(f"flowchart {direction.value}")

        # Узлы
        for agent in self.graph.agents:
            node_id = self._safe_id(agent.identifier)
            is_task = getattr(agent, "type", None) == "task"
            style = self.style.task_style if is_task else self.style.agent_style

            label = self._format_node_label(agent, style)

            if is_task:
                # Diamond shape for task: {label}
                lines.append(f"    {node_id}{{{label}}}")
            else:
                # Round rectangle for agents: (label)
                lines.append(f"    {node_id}({label})")

        lines.append("")

        # Рёбра
        edges_added = set()
        for edge in self.graph.edges:
            src = self._safe_id(edge.get("source", ""))
            tgt = self._safe_id(edge.get("target", ""))

            if not src or not tgt:
                continue

            edge_key = (src, tgt)
            if edge_key in edges_added:
                continue
            edges_added.add(edge_key)

            edge_type = edge.get("type", "workflow")
            weight = edge.get("weight", 1.0)

            # Определяем стиль линии
            if "task" in edge_type.lower():
                arrow = "-.->"  # dashed
            else:
                arrow = "-->"  # solid

            # Подпись ребра
            if self.style.show_weights and weight != 1.0:
                lines.append(f"    {src} {arrow}|w={weight:.2f}| {tgt}")
            else:
                lines.append(f"    {src} {arrow} {tgt}")

        # Стили
        lines.append("")
        lines.append("    %% Styles")

        # Стиль для агентов
        agent_ids = [
            self._safe_id(a.identifier)
            for a in self.graph.agents
            if getattr(a, "type", None) != "task"
        ]
        if agent_ids:
            s = self.style.agent_style
            lines.append(
                f"    classDef agent fill:{s.fill_color},stroke:{s.stroke_color},stroke-width:2px"
            )
            lines.append(f"    class {','.join(agent_ids)} agent")

        # Стиль для task узлов
        task_ids = [
            self._safe_id(a.identifier)
            for a in self.graph.agents
            if getattr(a, "type", None) == "task"
        ]
        if task_ids:
            s = self.style.task_style
            lines.append(
                f"    classDef task fill:{s.fill_color},stroke:{s.stroke_color},stroke-width:2px"
            )
            lines.append(f"    class {','.join(task_ids)} task")

        return "\n".join(lines)

    def to_ascii(
        self,
        show_edges: bool = True,
        box_width: int = 20,
    ) -> str:
        """Экспортировать граф в ASCII art.

        Args:
            show_edges: Показывать ли список рёбер
            box_width: Ширина блоков узлов

        Returns:
            ASCII-представление графа
        """
        lines = []

        # Заголовок
        title = f" Graph: {len(self.graph.agents)} nodes, {self.graph.num_edges} edges "
        border = "═" * (box_width + 4)
        lines.append(f"╔{border}╗")
        lines.append(f"║{title:^{box_width + 4}}║")
        lines.append(f"╠{border}╣")

        # Узлы
        for agent in self.graph.agents:
            is_task = getattr(agent, "type", None) == "task"
            icon = "📋" if is_task else "🤖"
            name = agent.display_name or agent.identifier

            # Обрезаем длинные имена
            if len(name) > box_width - 4:
                name = name[: box_width - 7] + "..."

            node_line = f"{icon} {name}"
            lines.append(f"║  {node_line:<{box_width + 2}}║")

            # Инструменты
            if self.style.show_tools and hasattr(agent, "tools") and agent.tools:
                tools_str = ", ".join(agent.tools[:3])
                if len(agent.tools) > 3:
                    tools_str += f" (+{len(agent.tools) - 3})"
                if len(tools_str) > box_width - 2:
                    tools_str = tools_str[: box_width - 5] + "..."
                lines.append(f"║    🔧 {tools_str:<{box_width}}║")

        lines.append(f"╠{border}╣")

        # Рёбра
        if show_edges:
            lines.append(f"║{'  Edges:':<{box_width + 4}}║")

            edges_shown = 0
            max_edges = 10

            for edge in self.graph.edges:
                if edges_shown >= max_edges:
                    remaining = len(self.graph.edges) - max_edges
                    lines.append(f"║    ... +{remaining} more{' ' * (box_width - 10)}║")
                    break

                src = edge.get("source", "?")
                tgt = edge.get("target", "?")
                edge_type = edge.get("type", "")

                # Сокращаем имена если нужно
                if len(src) > 8:
                    src = src[:6] + ".."
                if len(tgt) > 8:
                    tgt = tgt[:6] + ".."

                arrow = "⤳" if "task" in edge_type.lower() else "→"
                edge_str = f"{src} {arrow} {tgt}"
                lines.append(f"║    {edge_str:<{box_width}}║")
                edges_shown += 1

        lines.append(f"╚{border}╝")

        return "\n".join(lines)

    def to_dot(
        self,
        graph_name: str = "AgentGraph",
        rankdir: str = "TB",
    ) -> str:
        """Экспортировать граф в Graphviz DOT формат.

        Args:
            graph_name: Имя графа
            rankdir: Направление (TB, LR, BT, RL)

        Returns:
            DOT-код для Graphviz
        """
        lines = [
            f"digraph {graph_name} {{",
            f"    rankdir={rankdir};",
            '    node [fontname="Helvetica", fontsize=12];',
            '    edge [fontname="Helvetica", fontsize=10];',
            "",
        ]

        # Узлы
        for agent in self.graph.agents:
            node_id = self._safe_id(agent.identifier)
            is_task = getattr(agent, "type", None) == "task"

            label = agent.display_name or agent.identifier
            if self.style.show_tools and hasattr(agent, "tools") and agent.tools:
                tools = ", ".join(agent.tools[:3])
                label = f"{label}\\n[{tools}]"

            if is_task:
                style = self.style.task_style
                shape = "diamond"
            else:
                style = self.style.agent_style
                shape = "box"

            lines.append(
                f"    {node_id} ["
                f'label="{label}", '
                f"shape={shape}, "
                f"style=filled, "
                f'fillcolor="{style.fill_color}", '
                f'color="{style.stroke_color}"'
                f"];"
            )

        lines.append("")

        # Рёбра
        for edge in self.graph.edges:
            src = self._safe_id(edge.get("source", ""))
            tgt = self._safe_id(edge.get("target", ""))

            if not src or not tgt:
                continue

            edge_type = edge.get("type", "workflow")
            weight = edge.get("weight", 1.0)

            attrs = []
            if "task" in edge_type.lower():
                attrs.append("style=dashed")
                attrs.append(f'color="{self.style.task_edge_style.color}"')
            else:
                attrs.append(f'color="{self.style.workflow_edge_style.color}"')

            if self.style.show_weights and weight != 1.0:
                attrs.append(f'label="{weight:.2f}"')

            attr_str = ", ".join(attrs) if attrs else ""
            lines.append(f"    {src} -> {tgt} [{attr_str}];")

        lines.append("}")
        return "\n".join(lines)

    def to_adjacency_matrix(self, show_labels: bool = True) -> str:
        """Показать матрицу смежности в текстовом виде.

        Args:
            show_labels: Показывать ли метки узлов

        Returns:
            Текстовое представление матрицы
        """

        a_com = self.graph.A_com
        if a_com.size == 0:
            return "Empty adjacency matrix"

        lines = []
        n = a_com.shape[0]

        # Короткие метки
        labels = []
        for i, agent in enumerate(self.graph.agents[:n]):
            name = agent.identifier[:6]
            labels.append(name)

        # Заголовок
        if show_labels:
            header = "       " + " ".join(f"{label:>6}" for label in labels)
            lines.append(header)
            lines.append("       " + "-" * (7 * n))

        # Строки матрицы
        for i in range(n):
            row_label = f"{labels[i]:>6} |" if show_labels else ""
            row_values = " ".join(
                f"{a_com[i, j]:>6.2f}" if a_com[i, j] != 0 else "     ." for j in range(n)
            )
            lines.append(f"{row_label}{row_values}")

        return "\n".join(lines)

    def print_colored(self) -> None:
        """Вывести граф в консоль с цветами (требует rich)."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.tree import Tree
        except ImportError:
            # Fallback to ASCII if rich not available
            print(self.to_ascii())
            return

        console = Console()

        # Создаём дерево
        tree = Tree(
            f"[bold blue]🌐 Graph[/bold blue] "
            f"({len(self.graph.agents)} nodes, {self.graph.num_edges} edges)"
        )

        # Группируем агентов и задачи
        agents_branch = tree.add("[bold cyan]🤖 Agents[/bold cyan]")
        tasks_branch = tree.add("[bold yellow]📋 Tasks[/bold yellow]")

        for agent in self.graph.agents:
            is_task = getattr(agent, "type", None) == "task"
            branch = tasks_branch if is_task else agents_branch

            name = agent.display_name or agent.identifier
            node = branch.add(f"[bold]{name}[/bold] ({agent.identifier})")

            if hasattr(agent, "description") and agent.description:
                desc = agent.description[:60]
                if len(agent.description) > 60:
                    desc += "..."
                node.add(f"[dim]{desc}[/dim]")

            if hasattr(agent, "tools") and agent.tools:
                tools_str = ", ".join(agent.tools)
                node.add(f"[green]🔧 {tools_str}[/green]")

            # Показываем связи
            neighbors = self.graph.get_neighbors(agent.identifier, direction="out")
            if neighbors:
                conns = ", ".join(neighbors)
                node.add(f"[blue]→ {conns}[/blue]")

        console.print(tree)

        # Таблица рёбер
        if self.graph.num_edges > 0:
            console.print()
            table = Table(title="Edges", show_header=True)
            table.add_column("Source", style="cyan")
            table.add_column("Target", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Weight", style="magenta")

            for edge in self.graph.edges[:15]:  # Limit to 15 edges
                table.add_row(
                    str(edge.get("source", "")),
                    str(edge.get("target", "")),
                    str(edge.get("type", "workflow")),
                    f"{edge.get('weight', 1.0):.2f}",
                )

            if len(self.graph.edges) > 15:
                table.add_row("...", "...", "...", f"+{len(self.graph.edges) - 15} more")

            console.print(table)

    def save_mermaid(self, filepath: str, title: str | None = None) -> None:
        """Сохранить Mermaid-диаграмму в файл.

        Args:
            filepath: Путь к файлу (.md или .mmd)
            title: Заголовок диаграммы
        """
        content = self.to_mermaid(title=title)

        # Оборачиваем в markdown code block если .md файл
        if filepath.endswith(".md"):
            content = f"```mermaid\n{content}\n```"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def save_dot(self, filepath: str, graph_name: str = "AgentGraph") -> None:
        """Сохранить DOT-файл для Graphviz.

        Args:
            filepath: Путь к файлу (.dot или .gv)
            graph_name: Имя графа
        """
        content = self.to_dot(graph_name=graph_name)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def render_image(
        self,
        filepath: str,
        format: str = "png",
        dpi: int = 300,
        graph_name: str = "AgentGraph",
    ) -> None:
        """Отрендерить граф в изображение используя Graphviz.

        Args:
            filepath: Путь к выходному файлу (без расширения или с ним)
            format: Формат изображения ('png', 'svg', 'pdf', 'jpg')
            dpi: DPI для растровых форматов
            graph_name: Имя графа

        Raises:
            ImportError: Если graphviz не установлен
            RuntimeError: Если рендеринг не удался

        Example:
            viz = GraphVisualizer(graph)
            viz.render_image("my_graph", format="png")  # Creates my_graph.png
            viz.render_image("output.svg", format="svg")
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("Graphviz не установлен. Установите: pip install graphviz") from None

        # Получаем DOT код
        dot_source = self.to_dot(graph_name=graph_name)

        # Создаём Graphviz Source объект
        source = graphviz.Source(dot_source)

        # Удаляем расширение из filepath если оно есть
        if filepath.endswith(f".{format}"):
            filepath = filepath[: -len(format) - 1]

        try:
            # Рендерим изображение
            source.render(
                filename=filepath,
                format=format,
                cleanup=True,  # Удаляет промежуточный .dot файл
            )
        except Exception as e:
            raise RuntimeError(f"Не удалось отрендерить изображение: {e}") from e

    def show_interactive(self, graph_name: str = "AgentGraph") -> None:
        """Показать граф интерактивно в окне (используя Graphviz).

        Args:
            graph_name: Имя графа

        Raises:
            ImportError: Если graphviz не установлен

        Note:
            Требует установленного Graphviz с поддержкой GUI
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("Graphviz не установлен. Установите: pip install graphviz") from None

        dot_source = self.to_dot(graph_name=graph_name)
        source = graphviz.Source(dot_source)

        try:
            source.view(cleanup=True)
        except Exception as e:
            print(f"⚠️  Не удалось открыть интерактивный просмотр: {e}")
            print("    Убедитесь, что Graphviz установлен системно и поддерживает GUI")

    def _safe_id(self, identifier: str) -> str:
        """Преобразовать идентификатор в безопасный для Mermaid/DOT."""
        # Заменяем спецсимволы
        safe = identifier.replace("-", "_").replace(" ", "_").replace(".", "_")
        # Убираем двойные подчёркивания
        while "__" in safe:
            safe = safe.replace("__", "_")
        # Удаляем начальные/конечные подчёркивания
        safe = safe.strip("_")
        # Если начинается с цифры, добавляем префикс
        if safe and safe[0].isdigit():
            safe = "n_" + safe
        return safe or "unknown"

    def _format_node_label(self, agent: Any, style: NodeStyle) -> str:
        """Сформатировать метку узла."""
        name = agent.display_name or agent.identifier

        # Обрезаем длинные имена
        if len(name) > self.style.max_label_length:
            name = name[: self.style.max_label_length - 3] + "..."

        # Добавляем иконку
        if style.icon:
            name = f"{style.icon} {name}"

        # Добавляем инструменты
        if self.style.show_tools and hasattr(agent, "tools") and agent.tools:
            tools = agent.tools[:2]
            tools_str = ", ".join(tools)
            if len(agent.tools) > 2:
                tools_str += "..."
            name = f"{name}<br/>🔧 {tools_str}"

        return name


# ============================================================================
# Convenience functions
# ============================================================================


def to_mermaid(
    graph: "RoleGraph",
    direction: MermaidDirection = MermaidDirection.TOP_BOTTOM,
    title: str | None = None,
    style: VisualizationStyle | None = None,
) -> str:
    """Быстрый экспорт графа в Mermaid.

    Args:
        graph: RoleGraph для визуализации
        direction: Направление графа
        title: Заголовок диаграммы
        style: Стиль визуализации

    Returns:
        Mermaid-код

    Example:
        mermaid_code = to_mermaid(graph, direction=MermaidDirection.LR)
        print(mermaid_code)
    """
    viz = GraphVisualizer(graph, style)
    return viz.to_mermaid(direction=direction, title=title)


def to_ascii(
    graph: "RoleGraph",
    show_edges: bool = True,
    style: VisualizationStyle | None = None,
) -> str:
    """Быстрый экспорт графа в ASCII.

    Args:
        graph: RoleGraph для визуализации
        show_edges: Показывать ли рёбра
        style: Стиль визуализации

    Returns:
        ASCII-представление графа
    """
    viz = GraphVisualizer(graph, style)
    return viz.to_ascii(show_edges=show_edges)


def to_dot(
    graph: "RoleGraph",
    graph_name: str = "AgentGraph",
    style: VisualizationStyle | None = None,
) -> str:
    """Быстрый экспорт графа в Graphviz DOT.

    Args:
        graph: RoleGraph для визуализации
        graph_name: Имя графа
        style: Стиль визуализации

    Returns:
        DOT-код
    """
    viz = GraphVisualizer(graph, style)
    return viz.to_dot(graph_name=graph_name)


def print_graph(
    graph: "RoleGraph",
    format: str = "auto",
    style: VisualizationStyle | None = None,
) -> None:
    """Напечатать граф в консоль.

    Args:
        graph: RoleGraph для визуализации
        format: Формат вывода ("auto", "colored", "ascii", "mermaid")
        style: Стиль визуализации
    """
    viz = GraphVisualizer(graph, style)

    if format == "auto":
        # Пробуем rich, иначе ASCII
        try:
            from rich.console import Console  # noqa: F401

            viz.print_colored()
        except ImportError:
            print(viz.to_ascii())
    elif format == "colored":
        viz.print_colored()
    elif format == "ascii":
        print(viz.to_ascii())
    elif format == "mermaid":
        print(viz.to_mermaid())
    else:
        print(viz.to_ascii())


def render_to_image(
    graph: "RoleGraph",
    filepath: str,
    format: str = "png",
    dpi: int = 300,
    graph_name: str = "AgentGraph",
    style: VisualizationStyle | None = None,
) -> None:
    """Отрендерить граф в изображение.

    Args:
        graph: RoleGraph для визуализации
        filepath: Путь к выходному файлу
        format: Формат изображения ('png', 'svg', 'pdf', 'jpg')
        dpi: DPI для растровых форматов
        graph_name: Имя графа
        style: Стиль визуализации

    Raises:
        ImportError: Если graphviz не установлен

    Example:
        render_to_image(graph, "output.png", format="png")
        render_to_image(graph, "diagram", format="svg")
    """
    viz = GraphVisualizer(graph, style)
    viz.render_image(filepath, format=format, dpi=dpi, graph_name=graph_name)


def show_graph_interactive(
    graph: "RoleGraph",
    graph_name: str = "AgentGraph",
    style: VisualizationStyle | None = None,
) -> None:
    """Показать граф интерактивно.

    Args:
        graph: RoleGraph для визуализации
        graph_name: Имя графа
        style: Стиль визуализации

    Raises:
        ImportError: Если graphviz не установлен
    """
    viz = GraphVisualizer(graph, style)
    viz.show_interactive(graph_name=graph_name)
