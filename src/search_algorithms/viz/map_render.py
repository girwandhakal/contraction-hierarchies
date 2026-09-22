"""
Static renderings of a route on a real street network.

Produces the figures that go in the README and ``docs/analysis.md``: a
search's expanded nodes dotted over the map with the chosen route drawn on
top, so the *shape* of each algorithm's exploration is visible side by side.

These are the colour figures from the original grid/OSM analysis. The
Contraction Hierarchies plots in
:mod:`search_algorithms.benchmark.run_ch_benchmark` are deliberately black
and white instead; this module predates that convention and is left as-is
rather than restyled, since its existing output is already published in
``docs/``.

Requires the ``viz`` and ``osm`` extras (matplotlib and OSMnx).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from ..core.result import SearchResult
from ..problems.osm_problem import OSMProblem

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

#: One colour per algorithm, so a route keeps the same colour across every
#: figure in the docs.
ALGORITHM_COLORS = {
    "bfs": "#f2994a",
    "ucs": "#9b8cff",
    "astar": "#34d399",
    "greedy": "#38bdf8",
    "weighted_astar": "#fb7185",
}


def plot_route(
    graph: nx.MultiDiGraph,
    problem: OSMProblem,
    result: SearchResult,
    title: str | None = None,
    color: str = "#34d399",
    show_expanded: bool = True,
    save_to: str | Path | None = None,
) -> "Figure":
    """
    Draw one algorithm's search over the street network.

    Expanded intersections are dotted in faintly; the chosen route is drawn
    boldly on top, so the search and its answer read as one picture.
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    fig, ax = ox.plot_graph(
        graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#2b3947",
        edge_linewidth=0.5,
        bgcolor="#0f1419",
    )

    if show_expanded and result.expansion_order:
        coords = [problem.latlon(node) for node in result.expansion_order]
        ax.scatter(
            [lon for _, lon in coords],
            [lat for lat, _ in coords],
            c=color,
            s=3.5,
            alpha=0.35,
            zorder=2,
            linewidths=0,
        )

    if result.found:
        coords = problem.path_latlon(result.path)
        ax.plot(
            [lon for _, lon in coords],
            [lat for lat, _ in coords],
            color=color,
            linewidth=2.6,
            zorder=3,
            solid_capstyle="round",
        )

        start_lat, start_lon = coords[0]
        goal_lat, goal_lon = coords[-1]
        ax.scatter([start_lon], [start_lat], c="#38bdf8", s=70, zorder=4, edgecolors="#0b1016")
        ax.scatter([goal_lon], [goal_lat], c="#fb7185", s=70, zorder=4, edgecolors="#0b1016")

    # osmnx resets the axes facecolor while plotting, so set it afterwards.
    ax.set_facecolor("#0f1419")
    ax.set_title(
        title
        or f"{result.nodes_expanded:,} nodes expanded · {result.cost:,.0f} m",
        color="#e6edf3",
        fontsize=11,
    )

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1419")

    return fig


def compare_routes(
    graph: nx.MultiDiGraph,
    start: int,
    goal: int,
    algorithms: dict[str, "object"],
    save_to: str | Path | None = None,
    margin: float = 0.18,
) -> "Figure":
    """
    Render several algorithms on the same start/goal pair, side by side.

    ``algorithms`` maps a display name to a search function, e.g.
    ``{"bfs": bfs, "ucs": ucs, "astar": astar}``.

    All panels are cropped to the same window - the area the widest search
    actually touched, plus ``margin`` - so the panels stay comparable and the
    search is not lost inside the whole city.
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    names = list(algorithms)

    # Run every algorithm first, so the shared viewport can be computed from
    # the union of what they explored.
    runs: dict[str, tuple[OSMProblem, object]] = {}
    lats: list[float] = []
    lons: list[float] = []

    for name in names:
        problem = OSMProblem(graph, start=start, goal=goal)
        result = algorithms[name](problem)
        runs[name] = (problem, result)

        for node in list(result.expansion_order) + list(result.path):
            lat, lon = problem.latlon(node)
            lats.append(lat)
            lons.append(lon)

    if not lats:  # pragma: no cover - only if every search failed instantly
        lats, lons = [0.0], [0.0]

    lat_pad = max((max(lats) - min(lats)) * margin, 0.002)
    lon_pad = max((max(lons) - min(lons)) * margin, 0.002)
    ylim = (min(lats) - lat_pad, max(lats) + lat_pad)
    xlim = (min(lons) - lon_pad, max(lons) + lon_pad)

    fig, axes = plt.subplots(
        1, len(names), figsize=(6.0 * len(names), 6.0), squeeze=False, facecolor="#0f1419"
    )

    for ax, name in zip(axes[0], names):
        problem, result = runs[name]
        color = ALGORITHM_COLORS.get(name, "#34d399")

        ox.plot_graph(
            graph,
            ax=ax,
            show=False,
            close=False,
            node_size=0,
            edge_color="#3a4b5c",
            edge_linewidth=0.5,
            bgcolor="#0f1419",
        )

        if result.expansion_order:
            coords = [problem.latlon(node) for node in result.expansion_order]
            ax.scatter(
                [lon for _, lon in coords],
                [lat for lat, _ in coords],
                c=color,
                s=7.0,
                alpha=0.45,
                zorder=2,
                linewidths=0,
            )

        if result.found:
            coords = problem.path_latlon(result.path)
            ax.plot(
                [lon for _, lon in coords],
                [lat for lat, _ in coords],
                color=color,
                linewidth=2.6,
                zorder=3,
                solid_capstyle="round",
            )
            ax.scatter(
                [coords[0][1]], [coords[0][0]],
                c="#38bdf8", s=90, zorder=4, edgecolors="#0b1016",
            )
            ax.scatter(
                [coords[-1][1]], [coords[-1][0]],
                c="#fb7185", s=90, zorder=4, edgecolors="#0b1016",
            )

        # osmnx resets the axes facecolor, so set it after plotting.
        ax.set_facecolor("#0f1419")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(
            f"{name.upper()}\n{result.nodes_expanded:,} expanded · {result.cost:,.0f} m",
            color="#e6edf3",
            fontsize=12,
            pad=10,
        )

    fig.tight_layout()

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1419")

    return fig
