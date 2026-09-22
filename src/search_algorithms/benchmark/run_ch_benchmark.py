"""
The Contraction Hierarchies benchmark: Experiments A-E.

Run it as a script::

    python -m search_algorithms.benchmark.run_ch_benchmark
    python -m search_algorithms.benchmark.run_ch_benchmark --sizes 1000 4000 16000
    python -m search_algorithms.benchmark.run_ch_benchmark --json docs/ch_results.json

What each experiment answers
----------------------------
A. How does query cost scale, CH versus Dijkstra and A*, as the graph grows?
B. What does preprocessing cost, and how does that scale?
C. After how many queries has preprocessing paid for itself?
D. Does the synthetic trend hold on a real street network?
E. How much of CH's advantage comes from the node *ordering*, rather than
   from merely having shortcuts?

Measurement discipline
----------------------
Everything reported here follows the rules the plan set out before any of it
was written, because a speedup number is only worth as much as the method
behind it:

* **Paired trials.** Every algorithm answers the identical set of
  ``(start, goal)`` pairs on the identical graph, in one process, back to
  back. Numbers are never compared across runs or machines.
* **Means with spread.** Each figure is a mean over many queries, reported
  with its standard deviation, so a single lucky measurement cannot pass for
  a trend.
* **A fair baseline.** Dijkstra and A* run through
  :func:`~search_algorithms.ch.build.dijkstra_on_ch_graph`, which is written
  in the same style as the CH query - same heap discipline, same hoisted
  lookups, no expansion-order bookkeeping. An unoptimised baseline would
  inflate CH's speedup for free.
* **Reachable pairs only.** Unreachable pairs make both algorithms scan their
  whole component and would measure connectivity, not routing.
* **Correctness checked inline.** Every timed query is compared against
  Dijkstra's cost as it runs, and a mismatch is reported rather than
  silently averaged in. A fast wrong answer is worse than no answer.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..ch.build import dijkstra_on_ch_graph
from ..ch.graph import CHGraph
from ..ch.ordering import OrderingStrategy
from ..ch.preprocess import DEFAULT_MAX_HOPS, build_ch
from ..ch.query import ch_query
from .synthetic_graph import SyntheticGraph, generate

#: Graph sizes for the scaling experiments. Preprocessing is superlinear in
#: pure Python, so these stop where a full run still finishes in minutes
#: rather than hours; ``--sizes`` overrides them. The real scaling claim
#: rests on the *shape* of the curve across these points, not on reaching any
#: particular size.
DEFAULT_SIZES: tuple[int, ...] = (1_000, 2_000, 4_000, 8_000, 16_000)

#: Queries per graph. Enough that the mean is stable and the standard
#: deviation is meaningful.
DEFAULT_QUERIES = 60


@dataclass
class QueryStats:
    """Timing and search effort for one algorithm on one graph."""

    algorithm: str
    queries: int
    mean_nodes_expanded: float
    stdev_nodes_expanded: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(
        cls,
        algorithm: str,
        expanded: Sequence[int],
        milliseconds: Sequence[float],
    ) -> "QueryStats":
        if not expanded:
            return cls(algorithm, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return cls(
            algorithm=algorithm,
            queries=len(expanded),
            mean_nodes_expanded=statistics.mean(expanded),
            stdev_nodes_expanded=(
                statistics.stdev(expanded) if len(expanded) > 1 else 0.0
            ),
            mean_ms=statistics.mean(milliseconds),
            stdev_ms=(
                statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0
            ),
            min_ms=min(milliseconds),
            max_ms=max(milliseconds),
        )


@dataclass
class SizeResult:
    """Everything measured at one graph size."""

    node_count: int
    edge_count: int
    preprocess_seconds: float
    shortcuts_added: int
    witness_searches: int
    core_size: int
    algorithms: dict[str, QueryStats] = field(default_factory=dict)
    cost_mismatches: int = 0

    @property
    def speedup_vs_dijkstra(self) -> float:
        """Mean query wall-clock speedup of CH over Dijkstra."""
        ch = self.algorithms.get("ch")
        dijkstra = self.algorithms.get("dijkstra")
        if not ch or not dijkstra or ch.mean_ms == 0:
            return 0.0
        return dijkstra.mean_ms / ch.mean_ms

    @property
    def node_reduction_vs_dijkstra(self) -> float:
        """How many fewer nodes CH expands than Dijkstra, as a ratio."""
        ch = self.algorithms.get("ch")
        dijkstra = self.algorithms.get("dijkstra")
        if not ch or not dijkstra or ch.mean_nodes_expanded == 0:
            return 0.0
        return dijkstra.mean_nodes_expanded / ch.mean_nodes_expanded

    def break_even_queries(self) -> float | None:
        """
        Queries after which building the hierarchy has paid for itself.

        Solves ``preprocess + n * ch_ms == n * dijkstra_ms``. Returns ``None``
        when CH is not actually faster per query, in which case there is no
        break-even point to report and saying so is the honest answer.
        """
        ch = self.algorithms.get("ch")
        dijkstra = self.algorithms.get("dijkstra")
        if not ch or not dijkstra:
            return None

        saved_per_query_ms = dijkstra.mean_ms - ch.mean_ms
        if saved_per_query_ms <= 0:
            return None

        return (self.preprocess_seconds * 1000.0) / saved_per_query_ms


def _reachable_pairs(
    graph: CHGraph,
    count: int,
    rng: random.Random,
    attempts_per_pair: int = 20,
) -> list[tuple[int, int]]:
    """
    Sample ``count`` connected ``(start, goal)`` pairs.

    Unreachable pairs are discarded rather than timed: both algorithms would
    exhaust their reachable component, which measures how fragmented the
    graph is rather than how fast the routing is.
    """
    node_count = graph.node_count
    pairs: list[tuple[int, int]] = []

    for _ in range(count):
        for _ in range(attempts_per_pair):
            start = rng.randrange(node_count)
            goal = rng.randrange(node_count)
            if start == goal:
                continue
            if dijkstra_on_ch_graph(graph, start, goal).found:
                pairs.append((start, goal))
                break

    return pairs


def _time_queries(
    baseline: CHGraph,
    prepared,
    pairs: Sequence[tuple[int, int]],
    heuristic_for: Callable[[int], Callable[[int], float]] | None,
) -> tuple[dict[str, QueryStats], int]:
    """
    Run every algorithm over the same pairs and collect per-query samples.

    Returns the stats plus the number of pairs where CH disagreed with
    Dijkstra on cost - which should always be zero, and is reported rather
    than asserted so a benchmark run surfaces the problem instead of
    crashing halfway through.
    """
    samples: dict[str, tuple[list[int], list[float]]] = {
        "dijkstra": ([], []),
        "astar": ([], []),
        "ch": ([], []),
    }
    mismatches = 0

    for start, goal in pairs:
        reference = dijkstra_on_ch_graph(baseline, start, goal)
        samples["dijkstra"][0].append(reference.nodes_expanded)
        samples["dijkstra"][1].append(reference.elapsed_seconds * 1000.0)

        if heuristic_for is not None:
            astar_result = dijkstra_on_ch_graph(
                baseline, start, goal, heuristic=heuristic_for(goal)
            )
            samples["astar"][0].append(astar_result.nodes_expanded)
            samples["astar"][1].append(astar_result.elapsed_seconds * 1000.0)
            if astar_result.found and abs(astar_result.cost - reference.cost) > 1e-6:
                mismatches += 1

        # unpack=False: the benchmark is timing the search, and every
        # algorithm here returns a path, so none of them is charged for
        # path reconstruction. Costs are identical either way, which
        # tests/test_ch.py asserts.
        ch_result = ch_query(prepared, start, goal, unpack=False)
        samples["ch"][0].append(ch_result.nodes_expanded)
        samples["ch"][1].append(ch_result.elapsed_seconds * 1000.0)

        if ch_result.found != reference.found:
            mismatches += 1
        elif reference.found and abs(ch_result.cost - reference.cost) > 1e-6:
            mismatches += 1

    stats = {
        name: QueryStats.from_samples(name, expanded, times)
        for name, (expanded, times) in samples.items()
        if expanded
    }
    return stats, mismatches


def benchmark_size(
    node_count: int,
    queries: int = DEFAULT_QUERIES,
    seed: int = 0,
    max_hops: int = DEFAULT_MAX_HOPS,
    verbose: bool = True,
) -> SizeResult:
    """Experiments A, B and C at a single graph size."""
    if verbose:
        print(f"\n[{node_count:,} nodes] generating graph...", flush=True)

    synthetic: SyntheticGraph = generate(node_count, seed=seed)
    baseline = synthetic.graph.copy()

    if verbose:
        print(
            f"  {synthetic.node_count:,} nodes, "
            f"{synthetic.edge_count:,} directed edges",
            flush=True,
        )
        print("  contracting...", flush=True)

    started = time.perf_counter()
    prepared = build_ch(synthetic.graph, max_hops=max_hops)
    preprocess_seconds = time.perf_counter() - started

    if verbose:
        print(
            f"  preprocessed in {preprocess_seconds:.2f}s, "
            f"{prepared.stats.shortcuts_added:,} shortcuts "
            f"({prepared.stats.edge_growth:.2f}x original edges)",
            flush=True,
        )
        print(f"  sampling {queries} reachable query pairs...", flush=True)

    rng = random.Random(seed + 1)
    pairs = _reachable_pairs(baseline, queries, rng)

    if verbose:
        print(f"  timing {len(pairs)} queries per algorithm...", flush=True)

    stats, mismatches = _time_queries(
        baseline, prepared, pairs, synthetic.heuristic_to
    )

    result = SizeResult(
        node_count=synthetic.node_count,
        edge_count=synthetic.edge_count,
        preprocess_seconds=preprocess_seconds,
        shortcuts_added=prepared.stats.shortcuts_added,
        witness_searches=prepared.stats.witness_searches,
        core_size=prepared.stats.core_size,
        algorithms=stats,
        cost_mismatches=mismatches,
    )

    if verbose:
        for name in ("dijkstra", "astar", "ch"):
            if name not in stats:
                continue
            entry = stats[name]
            print(
                f"    {name:<9} {entry.mean_nodes_expanded:9.0f} expanded  "
                f"{entry.mean_ms:8.3f} ms  (sd {entry.stdev_ms:.3f})",
                flush=True,
            )
        print(
            f"  CH vs Dijkstra: {result.speedup_vs_dijkstra:.1f}x faster, "
            f"{result.node_reduction_vs_dijkstra:.1f}x fewer nodes",
            flush=True,
        )
        if mismatches:
            print(f"  !! {mismatches} cost mismatches - results are NOT valid", flush=True)

    return result


def benchmark_ordering(
    node_count: int,
    queries: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Experiment E: how much does the node ordering matter?

    Contracts the same graph three ways and compares the resulting
    hierarchies. All three are *correct* - ``tests/test_ch.py`` asserts that
    - so any difference here is purely quality.
    """
    if verbose:
        print(f"\n[ordering ablation @ {node_count:,} nodes]", flush=True)

    synthetic = generate(node_count, seed=seed)
    baseline = synthetic.graph.copy()

    rng = random.Random(seed + 2)
    pairs = _reachable_pairs(baseline, queries, rng)

    results: dict[str, dict[str, float]] = {}

    for strategy in OrderingStrategy:
        prepared = build_ch(baseline.copy(), strategy=strategy, seed=seed)

        expanded: list[int] = []
        times: list[float] = []
        for start, goal in pairs:
            result = ch_query(prepared, start, goal, unpack=False)
            expanded.append(result.nodes_expanded)
            times.append(result.elapsed_seconds * 1000.0)

        results[strategy.value] = {
            "preprocess_seconds": prepared.stats.elapsed_seconds,
            "shortcuts_added": float(prepared.stats.shortcuts_added),
            "mean_nodes_expanded": statistics.mean(expanded),
            "mean_ms": statistics.mean(times),
        }

        if verbose:
            entry = results[strategy.value]
            print(
                f"  {strategy.value:<16} "
                f"{entry['shortcuts_added']:8,.0f} shortcuts  "
                f"{entry['mean_nodes_expanded']:7.0f} expanded  "
                f"{entry['mean_ms']:7.3f} ms  "
                f"(prep {entry['preprocess_seconds']:.2f}s)",
                flush=True,
            )

    return results


def benchmark_osm(
    queries: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> SizeResult | None:
    """
    Experiment D: the same comparison on the real Tuscaloosa street network.

    Returns ``None`` when the cached graph is unavailable, so a benchmark run
    on a fresh checkout degrades to the synthetic experiments instead of
    failing.
    """
    try:
        from ..problems.osm_problem import OSMProblem, load_place_graph
        from ..ch.build import from_osm_problem
    except ImportError:
        if verbose:
            print("\n[OSM] osmnx not installed; skipping Experiment D.", flush=True)
        return None

    try:
        street_graph = load_place_graph()
    except Exception as exc:  # pragma: no cover - depends on cache state
        if verbose:
            print(f"\n[OSM] street graph unavailable ({exc}); skipping.", flush=True)
        return None

    if verbose:
        print(
            f"\n[OSM] Tuscaloosa: {street_graph.number_of_nodes():,} intersections, "
            f"{street_graph.number_of_edges():,} road segments",
            flush=True,
        )
        print("  contracting...", flush=True)

    nodes = [int(node) for node in street_graph.nodes]
    problem = OSMProblem(street_graph, start=nodes[0], goal=nodes[-1])

    started = time.perf_counter()
    index = from_osm_problem(problem)
    preprocess_seconds = time.perf_counter() - started

    baseline = CHGraph(index.node_count)
    for dense in range(index.node_count):
        label = index.original_label(dense)
        for neighbor in problem.neighbors(label):
            baseline.add_edge(
                dense,
                index.dense_id(int(neighbor)),
                problem.edge_cost(label, neighbor),
            )

    if verbose:
        print(
            f"  preprocessed in {preprocess_seconds:.2f}s, "
            f"{index.prepared.stats.shortcuts_added:,} shortcuts",
            flush=True,
        )

    rng = random.Random(seed)
    pairs = _reachable_pairs(baseline, queries, rng)

    if verbose:
        print(f"  timing {len(pairs)} queries per algorithm...", flush=True)

    # No heuristic: the dense CH ids carry no coordinates, and building a
    # lat/lon heuristic over them would time coordinate lookups rather than
    # the search. A* on the real graph is covered in the README.
    stats, mismatches = _time_queries(baseline, index.prepared, pairs, None)

    result = SizeResult(
        node_count=index.node_count,
        edge_count=baseline.edge_count,
        preprocess_seconds=preprocess_seconds,
        shortcuts_added=index.prepared.stats.shortcuts_added,
        witness_searches=index.prepared.stats.witness_searches,
        core_size=index.prepared.stats.core_size,
        algorithms=stats,
        cost_mismatches=mismatches,
    )

    if verbose:
        for name in ("dijkstra", "ch"):
            if name not in stats:
                continue
            entry = stats[name]
            print(
                f"    {name:<9} {entry.mean_nodes_expanded:9.0f} expanded  "
                f"{entry.mean_ms:8.3f} ms  (sd {entry.stdev_ms:.3f})",
                flush=True,
            )
        print(
            f"  CH vs Dijkstra: {result.speedup_vs_dijkstra:.1f}x faster, "
            f"{result.node_reduction_vs_dijkstra:.1f}x fewer nodes",
            flush=True,
        )
        print(f"  cost mismatches: {mismatches} / {len(pairs)}", flush=True)

    return result


def summarize(results: Sequence[SizeResult]) -> str:
    """Render the scaling results as a table."""
    lines: list[str] = []

    header = (
        f"{'nodes':>8}{'prep s':>9}{'shortcuts':>11}"
        f"{'dij exp':>10}{'CH exp':>9}{'dij ms':>9}{'CH ms':>9}"
        f"{'speedup':>9}{'node gain':>11}{'break-even':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for result in results:
        dijkstra = result.algorithms.get("dijkstra")
        ch = result.algorithms.get("ch")
        if not dijkstra or not ch:
            continue

        break_even = result.break_even_queries()
        break_even_text = f"{break_even:,.0f}" if break_even else "never"

        lines.append(
            f"{result.node_count:>8,}"
            f"{result.preprocess_seconds:>9.2f}"
            f"{result.shortcuts_added:>11,}"
            f"{dijkstra.mean_nodes_expanded:>10.0f}"
            f"{ch.mean_nodes_expanded:>9.0f}"
            f"{dijkstra.mean_ms:>9.3f}"
            f"{ch.mean_ms:>9.3f}"
            f"{result.speedup_vs_dijkstra:>8.1f}x"
            f"{result.node_reduction_vs_dijkstra:>10.1f}x"
            f"{break_even_text:>12}"
        )

    mismatches = sum(result.cost_mismatches for result in results)
    lines.append("")
    if mismatches:
        lines.append(
            f"!! {mismatches} queries disagreed with Dijkstra. "
            "These results are not valid - fix correctness first."
        )
    else:
        lines.append(
            "Every timed query returned exactly Dijkstra's cost."
        )

    return "\n".join(lines)


def plot_results(results: Sequence[SizeResult], save_to: str | Path) -> None:
    """
    Plot query effort and preprocessing cost against graph size.

    Black and white only: series are distinguished by line style and marker,
    never colour, so the figures read identically in print, in a terminal
    screenshot and for a colour-blind reader.
    """
    import matplotlib.pyplot as plt

    sizes = [result.node_count for result in results]

    styles = {
        "dijkstra": ("-", "o", "Dijkstra"),
        "astar": ("--", "s", "A*"),
        "ch": (":", "^", "Contraction Hierarchies"),
    }

    fig, (ax_nodes, ax_time, ax_prep) = plt.subplots(1, 3, figsize=(16, 4.8))

    for name, (linestyle, marker, label) in styles.items():
        expanded = [
            result.algorithms[name].mean_nodes_expanded
            for result in results
            if name in result.algorithms
        ]
        times = [
            result.algorithms[name].mean_ms
            for result in results
            if name in result.algorithms
        ]
        if not expanded:
            continue

        present = [
            result.node_count for result in results if name in result.algorithms
        ]
        ax_nodes.plot(
            present, expanded, linestyle=linestyle, marker=marker,
            color="black", label=label, markerfacecolor="white",
        )
        ax_time.plot(
            present, times, linestyle=linestyle, marker=marker,
            color="black", label=label, markerfacecolor="white",
        )

    ax_nodes.set_xlabel("nodes in graph")
    ax_nodes.set_ylabel("mean nodes expanded per query")
    ax_nodes.set_title("Search effort")
    ax_nodes.set_yscale("log")
    ax_nodes.legend()
    ax_nodes.grid(alpha=0.3, linestyle=":")

    ax_time.set_xlabel("nodes in graph")
    ax_time.set_ylabel("mean query time (ms)")
    ax_time.set_title("Query wall-clock time")
    ax_time.set_yscale("log")
    ax_time.legend()
    ax_time.grid(alpha=0.3, linestyle=":")

    ax_prep.plot(
        sizes,
        [result.preprocess_seconds for result in results],
        linestyle="-", marker="o", color="black", markerfacecolor="white",
    )
    ax_prep.set_xlabel("nodes in graph")
    ax_prep.set_ylabel("preprocessing time (s)")
    ax_prep.set_title("Preprocessing cost (paid once)")
    ax_prep.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()

    path = Path(save_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_break_even(results: Sequence[SizeResult], save_to: str | Path) -> None:
    """
    Cumulative time against query count, for the largest graph measured.

    The crossing point is the engineering question CH actually poses: run
    fewer queries than this and plain Dijkstra is the better choice.
    """
    import matplotlib.pyplot as plt

    usable = [
        result
        for result in results
        if "ch" in result.algorithms and "dijkstra" in result.algorithms
    ]
    if not usable:
        return

    result = usable[-1]
    break_even = result.break_even_queries()
    ceiling = int(break_even * 2) if break_even else 1_000
    counts = list(range(0, max(ceiling, 10), max(1, ceiling // 200)))

    dijkstra_ms = result.algorithms["dijkstra"].mean_ms
    ch_ms = result.algorithms["ch"].mean_ms
    prep_ms = result.preprocess_seconds * 1000.0

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        counts, [count * dijkstra_ms for count in counts],
        linestyle="-", color="black", label="Dijkstra (no preprocessing)",
    )
    ax.plot(
        counts, [prep_ms + count * ch_ms for count in counts],
        linestyle=":", color="black", label="CH (preprocess once, then query)",
    )

    if break_even:
        ax.axvline(break_even, linestyle="--", color="black", alpha=0.5)
        ax.annotate(
            f"break-even\n{break_even:,.0f} queries",
            xy=(break_even, prep_ms + break_even * ch_ms),
            xytext=(break_even * 1.05, (prep_ms + break_even * ch_ms) * 0.55),
            fontsize=9,
        )

    ax.set_xlabel("queries answered")
    ax.set_ylabel("cumulative time (ms)")
    ax.set_title(
        f"When preprocessing pays for itself ({result.node_count:,}-node graph)"
    )
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    path = Path(save_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _platform_details() -> dict[str, str]:
    """
    Record where the numbers came from.

    A wall-clock speedup means nothing without the machine and interpreter
    that produced it, so this travels with the results rather than being
    remembered separately.
    """
    import platform

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
        help="Synthetic graph sizes for Experiments A-C.",
    )
    parser.add_argument(
        "--queries", type=int, default=DEFAULT_QUERIES,
        help="Queries timed per graph.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-hops", type=int, default=DEFAULT_MAX_HOPS, dest="max_hops",
        help="Witness-search hop limit.",
    )
    parser.add_argument(
        "--skip-osm", action="store_true",
        help="Skip Experiment D even if the cached street graph exists.",
    )
    parser.add_argument(
        "--skip-ordering", action="store_true",
        help="Skip the Experiment E ordering ablation.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Write the raw results here, for the analysis writeup.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory to save the figures in.",
    )
    args = parser.parse_args()

    platform_details = _platform_details()
    print("Contraction Hierarchies benchmark")
    print(
        f"  {platform_details['implementation']} {platform_details['python']} "
        f"on {platform_details['system']} ({platform_details['machine']})"
    )
    print(f"  {args.queries} queries per graph, seed {args.seed}")

    results = [
        benchmark_size(
            size,
            queries=args.queries,
            seed=args.seed,
            max_hops=args.max_hops,
        )
        for size in args.sizes
    ]

    print("\n" + "=" * 100)
    print("Experiments A-C: query speedup, preprocessing cost, break-even")
    print("=" * 100)
    print(summarize(results))

    ordering = None
    if not args.skip_ordering:
        ordering = benchmark_ordering(
            min(args.sizes), queries=max(20, args.queries // 2), seed=args.seed
        )

    osm = None
    if not args.skip_osm:
        osm = benchmark_osm(queries=200, verbose=True)

    if args.plot_dir:
        plot_results(results, Path(args.plot_dir) / "ch_scaling.png")
        plot_break_even(results, Path(args.plot_dir) / "ch_break_even.png")
        print(f"\nFigures written to {args.plot_dir}")

    if args.json:
        payload = {
            "platform": platform_details,
            "settings": {
                "sizes": args.sizes,
                "queries": args.queries,
                "seed": args.seed,
                "max_hops": args.max_hops,
            },
            "scaling": [
                {
                    **{
                        key: value
                        for key, value in asdict(result).items()
                        if key != "algorithms"
                    },
                    "algorithms": {
                        name: asdict(stats)
                        for name, stats in result.algorithms.items()
                    },
                    "speedup_vs_dijkstra": result.speedup_vs_dijkstra,
                    "node_reduction_vs_dijkstra": result.node_reduction_vs_dijkstra,
                    "break_even_queries": result.break_even_queries(),
                }
                for result in results
            ],
            "ordering_ablation": ordering,
            "osm": (
                {
                    **{
                        key: value
                        for key, value in asdict(osm).items()
                        if key != "algorithms"
                    },
                    "algorithms": {
                        name: asdict(stats) for name, stats in osm.algorithms.items()
                    },
                    "speedup_vs_dijkstra": osm.speedup_vs_dijkstra,
                    "node_reduction_vs_dijkstra": osm.node_reduction_vs_dijkstra,
                    "break_even_queries": osm.break_even_queries(),
                }
                if osm
                else None
            ),
        }

        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Raw results written to {path}")


if __name__ == "__main__":
    main()
