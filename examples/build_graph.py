"""
Download and cache the Tuscaloosa street network, then sanity-check routing.

Run this once before anything that needs the real street graph - the
Contraction Hierarchies CLI/benchmark, the replanning demo, or the figures in
``examples/make_figures.py``. It populates ``data/osm_cache/`` so those tools
start instantly instead of waiting on the Overpass API.

Examples
--------
::

    python examples/build_graph.py
    python examples/build_graph.py --place "Northport, Alabama, USA"
    python examples/build_graph.py --refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_algorithms import astar, bfs, ucs  # noqa: E402
from search_algorithms.problems.osm_problem import (  # noqa: E402
    DEFAULT_PLACE,
    OSMProblem,
    load_place_graph,
)

# Two landmarks that sit inside the Tuscaloosa drive network, used as a
# smoke-test route: Bryant-Denny Stadium to Tuscaloosa Amphitheater.
BRYANT_DENNY = (33.2083, -87.5504)
AMPHITHEATER = (33.2122, -87.5692)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", default=DEFAULT_PLACE, help="Place to download.")
    parser.add_argument(
        "--network-type",
        default="drive",
        choices=["drive", "walk", "bike", "all"],
        help="Which street network to fetch.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even if a cached copy exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    graph = load_place_graph(
        place=args.place,
        network_type=args.network_type,
        refresh=args.refresh,
    )

    print(f"\nGraph for {args.place!r} ({args.network_type})")
    print(f"  Intersections (nodes): {graph.number_of_nodes():,}")
    print(f"  Road segments (edges): {graph.number_of_edges():,}")

    start = OSMProblem.nearest_node(graph, *BRYANT_DENNY)
    goal = OSMProblem.nearest_node(graph, *AMPHITHEATER)
    problem = OSMProblem(graph, start=start, goal=goal)

    print("\nSmoke-test route: Bryant-Denny Stadium -> Tuscaloosa Amphitheater")
    print(f"  start node {start} at {problem.latlon(start)}")
    print(f"  goal  node {goal} at {problem.latlon(goal)}")
    print(f"  straight-line distance: {problem.heuristic(start):,.0f} m\n")

    print(f"{'algorithm':<12}{'distance (m)':>14}{'steps':>8}{'expanded':>10}{'ms':>9}")
    print("-" * 53)
    for name, algorithm in (("bfs", bfs), ("ucs", ucs), ("astar", astar)):
        result = algorithm(OSMProblem(graph, start=start, goal=goal))
        if not result.found:
            print(f"{name:<12}{'no route':>14}")
            continue
        print(
            f"{name:<12}{result.cost:>14,.0f}{result.steps:>8}"
            f"{result.nodes_expanded:>10,}{result.elapsed_seconds * 1000:>9.1f}"
        )


if __name__ == "__main__":
    main()
