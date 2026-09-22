"""
Generate the static figures used in the README and docs.

::

    python examples/make_figures.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from search_algorithms import GridProblem, astar, bfs, ucs  # noqa: E402
from search_algorithms.benchmark import benchmark_grids  # noqa: E402
from search_algorithms.benchmark.run_benchmark import plot_benchmark  # noqa: E402
from search_algorithms.viz.grid_render import compare_grid_algorithms  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "docs" / "images"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Grid comparison on the hardest of the three reference maps.
    problem = GridProblem.from_file(REPO_ROOT / "maps" / "map3.txt")
    results = {"bfs": bfs(problem), "ucs": ucs(problem), "astar": astar(problem)}
    compare_grid_algorithms(problem, results, save_to=IMAGES_DIR / "grid_comparison.png")
    print(f"wrote {IMAGES_DIR / 'grid_comparison.png'}")

    # 2. Benchmark curves, both heuristic regimes.
    for max_weight, name in ((1, "benchmark_uniform.png"), (9, "benchmark_weighted.png")):
        rows = benchmark_grids(
            sizes=[10, 20, 30, 40], trials=10, max_weight=max_weight
        )
        plot_benchmark(rows, IMAGES_DIR / name)
        print(f"wrote {IMAGES_DIR / name}")

    # 3. Route comparison on the real street network, if the graph is cached.
    try:
        from search_algorithms.problems.osm_problem import (
            DEFAULT_CACHE_DIR,
            OSMProblem,
            load_place_graph,
        )
        from search_algorithms.viz.map_render import compare_routes

        cache = DEFAULT_CACHE_DIR / "tuscaloosa_alabama_usa_drive.graphml"
        if not cache.exists():
            print("skipping map figure: run `python examples/build_graph.py` first")
            return

        graph = load_place_graph()
        start = OSMProblem.nearest_node(graph, 33.2083, -87.5504)
        goal = OSMProblem.nearest_node(graph, 33.2122, -87.5692)
        compare_routes(
            graph,
            start,
            goal,
            {"bfs": bfs, "ucs": ucs, "astar": astar},
            save_to=IMAGES_DIR / "route_comparison.png",
        )
        print(f"wrote {IMAGES_DIR / 'route_comparison.png'}")
    except ImportError as exc:
        print(f"skipping map figure ({exc})")


if __name__ == "__main__":
    main()
