"""
Benchmark harness.

Three hand-drawn maps can show a trend but not establish one. This module
runs each algorithm across as many randomly generated maps as you like, so
claims like "A* expands fewer nodes" rest on a distribution rather than a
handful of data points.

Run it as a script::

    python -m search_algorithms.benchmark.run_benchmark --trials 40
    python -m search_algorithms.benchmark.run_benchmark --sizes 10 20 40 --plot out.png
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..core.algorithms import ALGORITHMS, CLASSIC_ALGORITHMS
from ..problems.grid_problem import GridProblem


@dataclass
class BenchmarkRow:
    """One algorithm's result on one generated problem."""

    algorithm: str
    size: int
    obstacle_density: float
    max_weight: int
    trial: int
    found: bool
    cost: float
    steps: int
    nodes_expanded: int
    elapsed_ms: float


def random_grid(
    size: int,
    obstacle_density: float = 0.25,
    max_weight: int = 9,
    rng: random.Random | None = None,
    max_attempts: int = 200,
) -> GridProblem:
    """
    Generate a random solvable ``size`` x ``size`` weighted grid.

    Start is the top-left corner, goal the bottom-right. Grids whose goal is
    walled off are discarded and regenerated, so the returned problem always
    has a solution.
    """
    from ..core.algorithms import bfs

    rng = rng or random.Random()

    for _ in range(max_attempts):
        grid = [
            [
                "#"
                if rng.random() < obstacle_density
                else str(rng.randint(1, max_weight))
                for _ in range(size)
            ]
            for _ in range(size)
        ]
        grid[0][0] = "S"
        grid[size - 1][size - 1] = "G"

        problem = GridProblem(grid)
        if bfs(problem).found:
            return problem

    raise RuntimeError(
        f"Could not generate a solvable {size}x{size} grid at density "
        f"{obstacle_density} in {max_attempts} attempts; try a lower density."
    )


def benchmark_grids(
    sizes: Sequence[int] = (10, 20, 30, 40),
    trials: int = 10,
    obstacle_density: float = 0.25,
    max_weight: int = 9,
    algorithms: Iterable[str] = CLASSIC_ALGORITHMS,
    seed: int = 0,
) -> list[BenchmarkRow]:
    """
    Run each algorithm on randomly generated grids of each size.

    Every algorithm sees the identical set of problems - the RNG is reseeded
    per trial - so the comparison is paired rather than merely averaged.

    ``max_weight`` controls how expensive the priciest terrain can be, and it
    matters more than it looks. Manhattan distance assumes every step costs at
    least 1, so on grids where steps can cost up to 9 the heuristic
    underestimates the true remaining cost several-fold. It stays admissible -
    A* still returns the optimal path - but it stops being *informative*, and
    A* degenerates toward UCS. Set ``max_weight=1`` to see what the same
    heuristic is worth when it is accurate.
    """
    names = list(algorithms)
    rows: list[BenchmarkRow] = []

    for size in sizes:
        for trial in range(trials):
            # Same seed per (size, trial) => every algorithm gets the same map.
            problem = random_grid(
                size,
                obstacle_density=obstacle_density,
                max_weight=max_weight,
                rng=random.Random(seed + trial * 1000 + size),
            )

            for name in names:
                result = ALGORITHMS[name](problem)
                rows.append(
                    BenchmarkRow(
                        algorithm=name,
                        size=size,
                        obstacle_density=obstacle_density,
                        max_weight=max_weight,
                        trial=trial,
                        found=result.found,
                        cost=result.cost,
                        steps=result.steps,
                        nodes_expanded=result.nodes_expanded,
                        elapsed_ms=result.elapsed_seconds * 1000.0,
                    )
                )

    return rows


def summarize(rows: Sequence[BenchmarkRow]) -> str:
    """Render the benchmark rows as a readable per-size table."""
    sizes = sorted({row.size for row in rows})
    algorithms = sorted({row.algorithm for row in rows}, key=lambda n: list(ALGORITHMS).index(n))

    lines: list[str] = []
    header = (
        f"{'grid':>6}{'algorithm':>16}{'mean cost':>12}"
        f"{'mean steps':>12}{'mean expanded':>16}{'mean ms':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for size in sizes:
        for name in algorithms:
            subset = [r for r in rows if r.size == size and r.algorithm == name and r.found]
            if not subset:
                continue
            lines.append(
                f"{f'{size}x{size}':>6}{name:>16}"
                f"{statistics.mean(r.cost for r in subset):>12.1f}"
                f"{statistics.mean(r.steps for r in subset):>12.1f}"
                f"{statistics.mean(r.nodes_expanded for r in subset):>16.1f}"
                f"{statistics.mean(r.elapsed_ms for r in subset):>10.2f}"
            )
        lines.append("")

    # The headline comparison, stated as a ratio over all trials.
    ucs_rows = [r for r in rows if r.algorithm == "ucs" and r.found]
    astar_rows = [r for r in rows if r.algorithm == "astar" and r.found]
    if ucs_rows and astar_rows:
        ucs_mean = statistics.mean(r.nodes_expanded for r in ucs_rows)
        astar_mean = statistics.mean(r.nodes_expanded for r in astar_rows)
        if astar_mean:
            lines.append(
                f"Across every trial, A* expanded {ucs_mean / astar_mean:.2f}x "
                f"fewer nodes than UCS for the same optimal cost."
            )

        max_weights = {r.max_weight for r in rows}
        if max_weights and max(max_weights) > 1:
            lines.append(
                f"\nNote: terrain costs range 1-{max(max_weights)}, but Manhattan "
                "distance assumes a minimum step cost of 1, so it underestimates "
                f"the remaining cost by up to {max(max_weights)}x. The heuristic is "
                "still admissible (A* matches UCS's optimal cost above) but it is "
                "weakly informative, which is why the speed-up is modest. Re-run "
                "with --max-weight 1 to see the same heuristic when it is accurate."
            )

    return "\n".join(lines)


def plot_benchmark(rows: Sequence[BenchmarkRow], save_to: str | Path) -> None:
    """Plot mean nodes expanded and mean runtime against grid size."""
    import matplotlib.pyplot as plt

    sizes = sorted({row.size for row in rows})
    algorithms = sorted({row.algorithm for row in rows}, key=lambda n: list(ALGORITHMS).index(n))
    colors = {"bfs": "#f2994a", "ucs": "#9b8cff", "astar": "#34d399"}

    fig, (ax_nodes, ax_time) = plt.subplots(1, 2, figsize=(13, 5))

    for name in algorithms:
        means, times = [], []
        for size in sizes:
            subset = [r for r in rows if r.size == size and r.algorithm == name and r.found]
            means.append(statistics.mean(r.nodes_expanded for r in subset) if subset else 0)
            times.append(statistics.mean(r.elapsed_ms for r in subset) if subset else 0)

        color = colors.get(name)
        ax_nodes.plot(sizes, means, marker="o", label=name.upper(), color=color)
        ax_time.plot(sizes, times, marker="o", label=name.upper(), color=color)

    max_weight = max({r.max_weight for r in rows}, default=9)
    subtitle = f"terrain cost 1-{max_weight}"

    ax_nodes.set_xlabel("grid size (n x n)")
    ax_nodes.set_ylabel("mean nodes expanded")
    ax_nodes.set_title(f"Search effort grows with problem size\n({subtitle})")
    ax_nodes.legend()
    ax_nodes.grid(alpha=0.3)

    ax_time.set_xlabel("grid size (n x n)")
    ax_time.set_ylabel("mean runtime (ms)")
    ax_time.set_title("Wall-clock cost")
    ax_time.legend()
    ax_time.grid(alpha=0.3)

    fig.tight_layout()

    save_path = Path(save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument(
        "--max-weight",
        type=int,
        default=9,
        help=(
            "Highest terrain cost. Use 1 for a uniform-cost grid, where "
            "Manhattan distance is accurate and A* shows its full advantage."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(CLASSIC_ALGORITHMS),
        choices=list(ALGORITHMS),
    )
    parser.add_argument("--plot", type=str, default=None, help="Save a comparison plot here.")
    args = parser.parse_args()

    print(
        f"Benchmarking {', '.join(args.algorithms)} on {args.trials} random grid(s) "
        f"per size {args.sizes} at {args.density:.0%} obstacle density, "
        f"terrain cost 1-{args.max_weight}...\n"
    )

    rows = benchmark_grids(
        sizes=args.sizes,
        trials=args.trials,
        obstacle_density=args.density,
        max_weight=args.max_weight,
        algorithms=args.algorithms,
        seed=args.seed,
    )

    print(summarize(rows))

    if args.plot:
        plot_benchmark(rows, args.plot)
        print(f"\nPlot written to {args.plot}")


if __name__ == "__main__":
    main()
