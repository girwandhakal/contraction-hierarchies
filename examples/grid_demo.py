"""
Run the search algorithms on a weighted grid map.

Prints the same figures tabulated in ``docs/analysis.md``, so the
implementation can be checked against the published reference results.

Examples
--------
Run every algorithm on a map::

    python examples/grid_demo.py --map maps/map1.txt --algorithm all

Run one algorithm and draw the path it found::

    python examples/grid_demo.py --map maps/map3.txt --algorithm astar --show-path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_algorithms import ALGORITHMS, GridProblem  # noqa: E402
from search_algorithms.core.result import SearchResult  # noqa: E402


def print_result(
    name: str,
    problem: GridProblem,
    result: SearchResult,
    show_path: bool,
) -> None:
    print(f"\n{name.upper()}")
    print("-" * max(len(name), 3))

    if not result.found:
        print("No path found.")
        print(f"Nodes expanded: {result.nodes_expanded}")
        return

    valid, message = problem.validate_path(result.path)
    recomputed = problem.path_cost(result.path)

    print(f"Path valid:      {valid} ({message})")
    print(f"Steps:           {result.steps}")
    print(f"Reported cost:   {result.cost:g}")
    print(f"Recomputed cost: {recomputed:g}")
    print(f"Nodes expanded:  {result.nodes_expanded}")
    print(f"Elapsed:         {result.elapsed_seconds * 1000:.2f} ms")

    if result.cost != recomputed:
        print("WARNING: reported cost does not match the path's actual cost.")

    if show_path:
        print("\nPath:")
        print(problem.render(result.path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="Path to a map file, e.g. maps/map1.txt")
    parser.add_argument(
        "--algorithm",
        choices=[*ALGORITHMS, "all", "classic"],
        default="classic",
        help="Which algorithm(s) to run. 'classic' is bfs/ucs/astar.",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print the grid with the returned path marked by *.",
    )
    args = parser.parse_args()

    problem = GridProblem.from_file(args.map)

    print("GRID")
    print("----")
    print(problem.render())
    print(f"\nStart: {problem.start}")
    print(f"Goal:  {problem.goal}")

    if args.algorithm == "all":
        names = list(ALGORITHMS)
    elif args.algorithm == "classic":
        names = ["bfs", "ucs", "astar"]
    else:
        names = [args.algorithm]

    for name in names:
        print_result(name, problem, ALGORITHMS[name](problem), args.show_path)


if __name__ == "__main__":
    main()
