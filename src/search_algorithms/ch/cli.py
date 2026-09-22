"""
Command line access to Contraction Hierarchies.

Two subcommands, both deliberately verbose - the point of running these from
a terminal is to watch the computation happen, not to get a number after an
unexplained pause::

    python -m search_algorithms.ch.cli preprocess --graph synthetic --nodes 20000
    python -m search_algorithms.ch.cli preprocess --graph grid --map maps/map3.txt
    python -m search_algorithms.ch.cli preprocess --graph osm

    python -m search_algorithms.ch.cli query --graph synthetic --nodes 5000
    python -m search_algorithms.ch.cli query --graph grid --map maps/map3.txt
    python -m search_algorithms.ch.cli query --graph osm --start <id> --goal <id>

``preprocess`` reports contraction progress, then the preprocessing stats
Experiment B is built from. ``query`` runs Dijkstra, A* and the CH query on
one start/goal pair and prints each result as it completes, so the
difference in search effort is visible per-algorithm rather than only in a
summary table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..core.result import SearchResult
from .build import CHIndex, dijkstra_on_ch_graph, from_problem
from .graph import CHGraph, Node
from .ordering import OrderingStrategy
from .preprocess import DEFAULT_MAX_HOPS, ProgressEvent

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _progress_printer(every_seconds: float = 0.25):
    """
    A progress callback that prints at most a few times a second.

    Throttling by wall-clock rather than by node count keeps output readable
    whether the graph has 1,000 nodes or 500,000 - the callback itself already
    fires only every 1% of nodes, but on a small graph that is still a burst
    of lines in a few milliseconds.
    """
    state = {"last": 0.0}

    def report(event: ProgressEvent) -> None:
        now = time.perf_counter()

        if event.phase == "start":
            print(f"  contracting {event.total:,} nodes...", flush=True)
            state["last"] = now
            return

        if event.phase == "done":
            print(
                f"  contracted {event.contracted:,}/{event.total:,} nodes, "
                f"{event.shortcuts_added:,} shortcuts, "
                f"{event.elapsed_seconds:.2f}s",
                flush=True,
            )
            return

        if now - state["last"] < every_seconds:
            return
        state["last"] = now

        print(
            f"    {event.percent:5.1f}%  "
            f"{event.contracted:,}/{event.total:,} nodes  "
            f"{event.shortcuts_added:,} shortcuts  "
            f"{event.elapsed_seconds:6.2f}s",
            flush=True,
        )

    return report


def _build_index(args: argparse.Namespace, verbose: bool) -> tuple[CHIndex, CHGraph]:
    """
    Build a CH index for whichever graph the arguments name.

    Returns the index and an *uncontracted* copy of the same graph, which the
    baseline Dijkstra/A* runs need: querying the contracted graph with plain
    Dijkstra would let it use shortcuts, which is not the baseline anyone
    means by "plain Dijkstra".
    """
    on_progress = _progress_printer() if verbose else None
    strategy = OrderingStrategy(args.strategy)

    if args.graph == "grid":
        from ..problems.grid_problem import GridProblem

        map_path = Path(args.map)
        if not map_path.is_absolute():
            map_path = _REPO_ROOT / map_path
        problem = GridProblem.from_file(map_path)
        print(f"Grid {map_path.name}: {problem.rows}x{problem.cols}")

        index = from_problem(
            problem,
            strategy=strategy,
            max_hops=args.max_hops,
            on_progress=on_progress,
        )
        baseline = _uncontracted_copy(index, problem=problem)
        return index, baseline

    if args.graph == "synthetic":
        from ..benchmark.synthetic_graph import road_like_graph

        graph = road_like_graph(args.nodes, seed=args.seed)
        print(
            f"Synthetic road-like graph: {graph.node_count:,} nodes, "
            f"{graph.edge_count:,} directed edges"
        )

        from .preprocess import build_ch

        prepared = build_ch(
            graph,
            strategy=strategy,
            max_hops=args.max_hops,
            on_progress=on_progress,
            seed=args.seed,
        )
        labels = list(range(graph.node_count))
        index = CHIndex(prepared, {label: label for label in labels}, labels)
        return index, graph

    if args.graph == "osm":
        from ..problems.osm_problem import OSMProblem, load_place_graph
        from .build import from_osm_problem

        street_graph = load_place_graph(args.place)
        nodes = list(street_graph.nodes)
        problem = OSMProblem(street_graph, start=int(nodes[0]), goal=int(nodes[-1]))
        print(
            f"OSM {args.place}: {street_graph.number_of_nodes():,} intersections, "
            f"{street_graph.number_of_edges():,} road segments"
        )

        index = from_osm_problem(
            problem,
            strategy=strategy,
            max_hops=args.max_hops,
            on_progress=on_progress,
        )
        baseline = _uncontracted_copy(index, osm_problem=problem)
        return index, baseline

    raise ValueError(f"Unknown graph source {args.graph!r}.")


def _uncontracted_copy(
    index: CHIndex,
    problem=None,
    osm_problem=None,
) -> CHGraph:
    """
    Rebuild the original (shortcut-free) graph in the index's dense id space.

    Rebuilt from the source problem rather than by filtering shortcuts out of
    the contracted graph: contraction can *improve* an existing edge's cost
    when a shortcut is cheaper, so subtracting the shortcut table would not
    reliably recover the original weights.
    """
    graph = CHGraph(index.node_count)
    source = problem if problem is not None else osm_problem

    for dense in range(index.node_count):
        label = index.original_label(dense)
        for neighbor in source.neighbors(label):
            neighbor_dense = index.dense_id(neighbor)
            graph.add_edge(dense, neighbor_dense, source.edge_cost(label, neighbor))

    return graph


def _print_result(name: str, result: SearchResult, reference_cost: float | None) -> None:
    """One algorithm's outcome, with a correctness verdict when possible."""
    if not result.found:
        print(f"  {name:<22} no path  ({result.nodes_expanded:,} expanded)")
        return

    verdict = ""
    if reference_cost is not None:
        verdict = (
            "  optimal"
            if abs(result.cost - reference_cost) < 1e-6
            else f"  !! differs from Dijkstra by {result.cost - reference_cost:+.6f}"
        )

    print(
        f"  {name:<22} cost {result.cost:>14,.2f}  "
        f"{result.nodes_expanded:>9,} expanded  "
        f"{result.elapsed_seconds * 1000:>8.2f} ms"
        f"{verdict}"
    )


def _cmd_preprocess(args: argparse.Namespace) -> int:
    print(f"Preprocessing with {args.strategy} ordering, max_hops={args.max_hops}\n")

    index, baseline = _build_index(args, verbose=True)
    stats = index.prepared.stats

    print("\nPreprocessing stats")
    print(f"  nodes                  {stats.node_count:,}")
    print(f"  original edges         {stats.original_edge_count:,}")
    print(f"  shortcuts added        {stats.shortcuts_added:,}")
    print(f"  edge growth            {stats.edge_growth:.2f}x original edges")
    print(f"  witness searches       {stats.witness_searches:,}")
    print(f"  witness hits           {stats.witness_hits:,}")
    print(f"  priority re-evaluations {stats.priority_reevaluations:,}")
    print(f"  wall time              {stats.elapsed_seconds:.2f}s")
    return 0


def _pick_endpoints(
    args: argparse.Namespace,
    index: CHIndex,
    baseline: CHGraph,
) -> tuple[Node, Node]:
    """
    Resolve the start/goal arguments, defaulting to a genuinely long route.

    The default matters more than it looks. Taking the first and last node in
    index order gives an adjacent pair on a grid - a "route" of one step,
    which makes the comparison meaningless. Instead the farthest reachable
    node from an arbitrary start is used, found with one Dijkstra run, so the
    demo always shows a query worth measuring.
    """
    labels = index.labels

    if args.start is not None and args.goal is not None:
        # Labels are ints for OSM/synthetic graphs and tuples for grids; the
        # argument arrives as a string either way.
        converter = type(labels[0])
        if converter is tuple:
            raise ValueError(
                "--start/--goal are not supported for grid graphs; "
                "grid states are (row, col) tuples."
            )
        return converter(args.start), converter(args.goal)

    start = 0
    frontier = _farthest_from(baseline, start)
    return index.original_label(start), index.original_label(frontier)


def _farthest_from(graph: CHGraph, source: Node) -> Node:
    """The reachable node with the greatest shortest-path distance from ``source``."""
    import heapq

    inf = float("inf")
    dist: dict[Node, float] = {source: 0.0}
    frontier: list[tuple[float, Node]] = [(0.0, source)]
    best_node, best_cost = source, 0.0

    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost > dist.get(node, inf):
            continue
        if cost > best_cost:
            best_node, best_cost = node, cost

        for nxt, edge_cost in graph.successors(node).items():
            nxt_cost = cost + edge_cost
            if nxt_cost < dist.get(nxt, inf):
                dist[nxt] = nxt_cost
                heapq.heappush(frontier, (nxt_cost, nxt))

    return best_node


def _cmd_query(args: argparse.Namespace) -> int:
    index, baseline = _build_index(args, verbose=not args.quiet)

    start_label, goal_label = _pick_endpoints(args, index, baseline)
    start = index.dense_id(start_label)
    goal = index.dense_id(goal_label)

    print(f"\nQuery {start_label} -> {goal_label}\n")

    print("  running Dijkstra...", flush=True)
    dijkstra = dijkstra_on_ch_graph(baseline, start, goal)
    _print_result("Dijkstra", dijkstra, None)

    reference = dijkstra.cost if dijkstra.found else None

    print("  running CH query...", flush=True)
    ch = index.query(start_label, goal_label)
    _print_result("Contraction Hierarchies", ch, reference)

    if dijkstra.found and ch.found:
        print()
        # Phrased by direction rather than always claiming a win: on a tiny or
        # trivially short route CH can legitimately do more work, and printing
        # "0.5x fewer" would be nonsense.
        if ch.nodes_expanded:
            ratio = dijkstra.nodes_expanded / ch.nodes_expanded
            if ratio >= 1.0:
                print(f"  CH expanded {ratio:.1f}x fewer nodes than Dijkstra")
            else:
                print(
                    f"  CH expanded {1 / ratio:.1f}x MORE nodes than Dijkstra "
                    "- expected on a route this short"
                )
        if ch.elapsed_seconds > 0:
            speedup = dijkstra.elapsed_seconds / ch.elapsed_seconds
            comparison = (
                f"{speedup:.1f}x faster"
                if speedup >= 1.0
                else f"{1 / speedup:.1f}x slower"
            )
            print(
                f"  CH query was {comparison} "
                "(single query; see the benchmark for means)"
            )
        print(
            "\n  Note: preprocessing took "
            f"{index.prepared.stats.elapsed_seconds:.2f}s and is paid once, not "
            "per query - run_ch_benchmark.py computes the break-even point."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m search_algorithms.ch.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--graph",
            choices=("grid", "synthetic", "osm"),
            default="synthetic",
            help="Which graph to run on.",
        )
        sub.add_argument(
            "--map",
            default="maps/map3.txt",
            help="Grid map file, when --graph grid.",
        )
        sub.add_argument(
            "--nodes",
            type=int,
            default=5_000,
            help="Node count, when --graph synthetic.",
        )
        sub.add_argument(
            "--place",
            default="Tuscaloosa, Alabama, USA",
            help="Place query, when --graph osm.",
        )
        sub.add_argument(
            "--strategy",
            choices=[s.value for s in OrderingStrategy],
            default=OrderingStrategy.EDGE_DIFFERENCE.value,
            help="Node ordering heuristic.",
        )
        sub.add_argument(
            "--max-hops",
            type=int,
            default=DEFAULT_MAX_HOPS,
            dest="max_hops",
            help="Witness-search hop limit.",
        )
        sub.add_argument("--seed", type=int, default=0)

    preprocess = subparsers.add_parser(
        "preprocess", help="Contract a graph and report preprocessing cost."
    )
    add_common(preprocess)
    preprocess.set_defaults(func=_cmd_preprocess)

    query = subparsers.add_parser(
        "query", help="Compare Dijkstra and CH on one start/goal pair."
    )
    add_common(query)
    query.add_argument("--start", default=None, help="Start node label.")
    query.add_argument("--goal", default=None, help="Goal node label.")
    query.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress contraction progress output.",
    )
    query.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
