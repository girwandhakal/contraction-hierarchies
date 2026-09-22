"""
Rendering grid searches, in the terminal and with matplotlib.

The ASCII renderer needs nothing beyond the standard library. The matplotlib
renderer draws the same picture for a saved figure: which cells the search
expanded, and the route it settled on.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.result import SearchResult
from ..problems.grid_problem import GridProblem

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

State = tuple[int, int]


def render_ascii(
    problem: GridProblem,
    result: SearchResult | None = None,
    show_expanded: bool = False,
) -> str:
    """
    Draw the grid as text.

    ``*`` marks the returned path; with ``show_expanded`` set, ``.`` marks
    cells the search expanded but did not use.
    """
    path_cells = set(result.path) if result else set()
    expanded_cells = set(result.expansion_order) if (result and show_expanded) else set()

    rows: list[str] = []
    for r in range(problem.rows):
        row: list[str] = []
        for c in range(problem.cols):
            cell = (r, c)
            token = problem.grid[r][c]

            if token in {"S", "G"}:
                row.append(token)
            elif cell in path_cells:
                row.append("*")
            elif cell in expanded_cells:
                row.append(".")
            else:
                row.append(token)
        rows.append(" ".join(row))

    return "\n".join(rows)


def plot_grid_search(
    problem: GridProblem,
    result: SearchResult,
    title: str | None = None,
    save_to: str | Path | None = None,
    show_expansion_order: bool = True,
) -> "Figure":
    """
    Plot a grid search: terrain, expanded cells, and the final path.

    Expanded cells are shaded by *when* they were expanded, so the order the
    search worked through the grid is visible at a glance.

    Requires the ``viz`` extra (matplotlib).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    terrain = np.zeros((problem.rows, problem.cols))
    obstacles = np.zeros((problem.rows, problem.cols), dtype=bool)

    for r in range(problem.rows):
        for c in range(problem.cols):
            token = problem.grid[r][c]
            if token == "#":
                obstacles[r, c] = True
                terrain[r, c] = np.nan
            elif token in {"S", "G"}:
                terrain[r, c] = 1.0
            else:
                terrain[r, c] = float(token)

    fig, ax = plt.subplots(figsize=(max(6.0, problem.cols * 0.5), max(4.0, problem.rows * 0.5)))

    ax.imshow(terrain, cmap="YlOrBr", vmin=0, vmax=9, alpha=0.65, interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(~obstacles, obstacles),
        cmap=ListedColormap(["#2b3947"]),
        interpolation="nearest",
    )

    if show_expansion_order and result.expansion_order:
        ys = [cell[0] for cell in result.expansion_order]
        xs = [cell[1] for cell in result.expansion_order]
        order = list(range(len(result.expansion_order)))
        scatter = ax.scatter(xs, ys, c=order, cmap="viridis", s=42, alpha=0.75, marker="s")
        fig.colorbar(scatter, ax=ax, label="expansion order", shrink=0.8)

    if result.found:
        ax.plot(
            [cell[1] for cell in result.path],
            [cell[0] for cell in result.path],
            color="#e11d48",
            linewidth=2.6,
            marker="o",
            markersize=4,
            label=f"path · cost {result.cost:g}",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.plot(problem.start[1], problem.start[0], marker="*", color="#0ea5e9", markersize=18)
    ax.plot(problem.goal[1], problem.goal[0], marker="*", color="#22c55e", markersize=18)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title
        or f"{result.nodes_expanded} nodes expanded · cost {result.cost:g}",
        fontsize=11,
    )

    fig.tight_layout()

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_grid_algorithms(
    problem: GridProblem,
    results: dict[str, SearchResult],
    save_to: str | Path | None = None,
) -> "Figure":
    """
    Plot several algorithms on one grid, side by side, for direct comparison.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(max(4.0, problem.cols * 0.36) * len(names), max(3.6, problem.rows * 0.42)),
        squeeze=False,
    )

    terrain = np.zeros((problem.rows, problem.cols))
    for r in range(problem.rows):
        for c in range(problem.cols):
            token = problem.grid[r][c]
            terrain[r, c] = np.nan if token == "#" else (1.0 if token in {"S", "G"} else float(token))

    for ax, name in zip(axes[0], names):
        result = results[name]
        ax.imshow(terrain, cmap="YlOrBr", vmin=0, vmax=9, alpha=0.6, interpolation="nearest")

        if result.expansion_order:
            ax.scatter(
                [cell[1] for cell in result.expansion_order],
                [cell[0] for cell in result.expansion_order],
                c=range(len(result.expansion_order)),
                cmap="viridis",
                s=26,
                alpha=0.7,
                marker="s",
            )

        if result.found:
            ax.plot(
                [cell[1] for cell in result.path],
                [cell[0] for cell in result.path],
                color="#e11d48",
                linewidth=2.2,
            )

        ax.plot(problem.start[1], problem.start[0], marker="*", color="#0ea5e9", markersize=14)
        ax.plot(problem.goal[1], problem.goal[0], marker="*", color="#22c55e", markersize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{name.upper()}\n{result.nodes_expanded} expanded · cost {result.cost:g}",
            fontsize=10,
        )

    fig.tight_layout()

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
