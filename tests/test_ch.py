"""
Correctness tests for Contraction Hierarchies.

CH is easy to implement *plausibly* and wrong: a witness search that accepts
one witness too many silently drops a necessary shortcut, and the result is a
query that is fast and returns paths that are a little too long. Nothing
crashes. The only way to know the implementation is right is to check its
answers against a reference on enough graphs that the awkward cases actually
occur.

So the contract asserted here is strict: **the cost CH returns is exactly the
cost Dijkstra returns**, and the path it returns is a real walk in the
original graph whose edge costs sum to that number. Both, on every graph shape
the library supports.

The tie-heavy fuzz test earns its place
---------------------------------------
An earlier version of this suite fuzzed with random float weights and passed
completely while the implementation had a real bug: the witness search
accepted a witness of *equal* cost. Random floats essentially never tie, so
the case never arose. Grid maps tie constantly, and `maps/map3.txt` returned
30 instead of 28.

That is why :func:`test_tie_heavy_graphs_match_dijkstra` uses small integer
weights and lattice structures - it makes ties the common case rather than a
rarity. Any future change to the witness search should be assumed wrong until
that test passes.
"""

from __future__ import annotations

import random

import pytest

from search_algorithms import GridProblem, astar
from search_algorithms.ch import (
    CHGraph,
    OrderingStrategy,
    build_ch,
    ch_query,
    from_adjacency,
    from_problem,
)
from search_algorithms.ch.build import dijkstra_on_ch_graph
from search_algorithms.ch.preprocess import ProgressEvent

MAPS = ("map1", "map2", "map3")

#: Core fractions exercised everywhere. 0.0 contracts the whole graph (a pure
#: hierarchy); the others leave an uncontracted core, where the query must
#: relax core-to-core edges regardless of rank. Both paths need coverage:
#: the core logic is a correctness-relevant special case, not a tuning knob.
CORE_FRACTIONS = (0.0, 0.005, 0.2)


# -- helpers ---------------------------------------------------------------


def adjacency_to_graph(adjacency: dict[int, list[tuple[int, float]]]) -> CHGraph:
    """The same adjacency map as an uncontracted :class:`CHGraph`."""
    graph = CHGraph(len(adjacency))
    for node, edges in adjacency.items():
        for neighbor, cost in edges:
            graph.add_edge(node, neighbor, cost)
    return graph


def walk_cost(graph: CHGraph, path: list[int]) -> float:
    """
    Sum the path's edge costs in ``graph``, or raise if an edge is missing.

    This is the check that catches broken shortcut unpacking: a CH path can
    report the right total while containing "edges" that only exist in the
    contracted graph.
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        cost = graph.edge_cost(a, b)
        if cost == float("inf"):
            raise AssertionError(f"Path uses non-existent edge {a} -> {b}: {path}")
        total += cost
    return total


def random_weighted_graph(
    rng: random.Random,
    node_count: int,
    max_weight: int | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """
    A connected random graph: a spanning path plus random extra edges.

    ``max_weight`` picks integer weights in ``1..max_weight`` (tie-heavy);
    ``None`` uses floats, which essentially never tie.
    """

    def weight() -> float:
        if max_weight is None:
            return round(rng.uniform(1.0, 10.0), 2)
        return float(rng.randint(1, max_weight))

    adjacency: dict[int, list[tuple[int, float]]] = {
        node: [] for node in range(node_count)
    }

    for node in range(node_count - 1):
        cost = weight()
        adjacency[node].append((node + 1, cost))
        adjacency[node + 1].append((node, cost))

    for _ in range(rng.randint(0, node_count * 2)):
        a, b = rng.randrange(node_count), rng.randrange(node_count)
        if a != b:
            adjacency[a].append((b, weight()))

    return adjacency


def random_lattice(
    rng: random.Random,
    rows: int,
    cols: int,
    max_weight: int,
) -> dict[int, list[tuple[int, float]]]:
    """A grid lattice with small integer weights - ties everywhere."""
    node_count = rows * cols
    adjacency: dict[int, list[tuple[int, float]]] = {
        node: [] for node in range(node_count)
    }

    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                cost = float(rng.randint(1, max_weight))
                adjacency[node].append((node + 1, cost))
                adjacency[node + 1].append((node, cost))
            if row + 1 < rows:
                below = node + cols
                cost = float(rng.randint(1, max_weight))
                adjacency[node].append((below, cost))
                adjacency[below].append((node, cost))

    return adjacency


def assert_matches_dijkstra(
    adjacency: dict[int, list[tuple[int, float]]],
    core_fraction: float = 0.0,
    strategy: OrderingStrategy = OrderingStrategy.EDGE_DIFFERENCE,
    max_hops: int | None = None,
) -> int:
    """
    Check every ``(start, goal)`` pair against Dijkstra. Returns pairs checked.

    Exhaustive over all pairs rather than sampled: these graphs are small, and
    the failures CH produces are pair-specific, so sampling would make the
    suite flaky instead of fast.
    """
    baseline = adjacency_to_graph(adjacency)
    kwargs = {} if max_hops is None else {"max_hops": max_hops}
    index = from_adjacency(
        adjacency,
        strategy=strategy,
        core_fraction=core_fraction,
        **kwargs,
    )

    node_count = len(adjacency)
    checked = 0

    for start in range(node_count):
        for goal in range(node_count):
            expected = dijkstra_on_ch_graph(baseline, start, goal)
            actual = index.query(start, goal)
            checked += 1

            assert actual.found == expected.found, (
                f"{start} -> {goal}: CH found={actual.found}, "
                f"Dijkstra found={expected.found}"
            )

            if not expected.found:
                continue

            assert actual.cost == pytest.approx(expected.cost, abs=1e-7), (
                f"{start} -> {goal}: CH cost {actual.cost}, "
                f"Dijkstra cost {expected.cost}"
            )
            assert actual.path[0] == start
            assert actual.path[-1] == goal
            assert walk_cost(baseline, actual.path) == pytest.approx(
                actual.cost, abs=1e-6
            )

    return checked


# -- grid maps -------------------------------------------------------------


@pytest.mark.parametrize("name", MAPS)
@pytest.mark.parametrize("core_fraction", CORE_FRACTIONS)
def test_grid_maps_match_astar(name: str, core_fraction: float) -> None:
    """CH returns A*'s exact optimal cost on the reference grid maps."""
    problem = GridProblem.from_file(f"maps/{name}.txt")
    expected = astar(problem)

    index = from_problem(problem, core_fraction=core_fraction)
    result = index.query(problem.start, problem.goal)

    assert result.found
    assert result.cost == pytest.approx(expected.cost)


@pytest.mark.parametrize("name", MAPS)
def test_grid_paths_are_walkable(name: str) -> None:
    """
    The unpacked path is a legal walk, not just a correct total.

    ``validate_path`` is the existing oracle every other algorithm in this
    repo is held to, so shortcut unpacking is held to it too.
    """
    problem = GridProblem.from_file(f"maps/{name}.txt")
    index = from_problem(problem)
    result = index.query(problem.start, problem.goal)

    valid, message = problem.validate_path(result.path)
    assert valid, message
    assert problem.path_cost(result.path) == pytest.approx(result.cost)


@pytest.mark.parametrize("strategy", list(OrderingStrategy))
def test_every_ordering_is_correct(strategy: OrderingStrategy) -> None:
    """
    Ordering changes hierarchy *quality*, never its answers.

    The baseline orderings exist for the ablation experiment; if one of them
    produced different costs, the ablation would be comparing correctness
    rather than efficiency.
    """
    problem = GridProblem.from_file("maps/map3.txt")
    expected = astar(problem).cost

    index = from_problem(problem, strategy=strategy)
    result = index.query(problem.start, problem.goal)

    assert result.cost == pytest.approx(expected)
    valid, message = problem.validate_path(result.path)
    assert valid, message


@pytest.mark.parametrize("max_hops", [1, 2, 3, 5, 20])
def test_hop_limit_never_breaks_correctness(max_hops: int) -> None:
    """
    A tighter witness search adds redundant shortcuts; it must not lose any.

    This is the test that pins down the direction of the hop limit's error.
    """
    problem = GridProblem.from_file("maps/map3.txt")
    expected = astar(problem).cost

    index = from_problem(problem, max_hops=max_hops)
    result = index.query(problem.start, problem.goal)

    assert result.cost == pytest.approx(expected)


# -- random graphs ---------------------------------------------------------


@pytest.mark.parametrize("core_fraction", CORE_FRACTIONS)
def test_random_float_graphs_match_dijkstra(core_fraction: float) -> None:
    """Exhaustive all-pairs agreement on random graphs with float weights."""
    checked = 0
    for trial in range(40):
        rng = random.Random(trial)
        adjacency = random_weighted_graph(rng, rng.randint(2, 12))
        checked += assert_matches_dijkstra(adjacency, core_fraction=core_fraction)

    assert checked > 1_000, "Fuzz coverage unexpectedly small."


@pytest.mark.parametrize("max_weight", [1, 2, 3])
def test_tie_heavy_graphs_match_dijkstra(max_weight: int) -> None:
    """
    Small integer weights, where equal-cost paths are everywhere.

    This is the regression test for the witness-search equality bug described
    in the module docstring. It fails loudly if a witness of merely equal cost
    is ever accepted again.
    """
    checked = 0
    for trial in range(30):
        rng = random.Random(trial * 31 + max_weight)

        if trial % 2:
            adjacency = random_weighted_graph(
                rng, rng.randint(2, 12), max_weight=max_weight
            )
        else:
            adjacency = random_lattice(
                rng, rng.randint(2, 5), rng.randint(2, 5), max_weight
            )

        checked += assert_matches_dijkstra(adjacency)

    assert checked > 1_000


def test_directed_graphs_match_dijkstra() -> None:
    """
    One-way edges: the case that motivates edge-based costs in the first place.

    A CH that quietly treats the graph as undirected still passes symmetric
    tests, so asymmetry needs its own.
    """
    rng = random.Random(99)
    for _ in range(20):
        node_count = rng.randint(3, 10)
        adjacency: dict[int, list[tuple[int, float]]] = {
            node: [] for node in range(node_count)
        }
        # A directed cycle guarantees reachability without symmetry.
        for node in range(node_count):
            adjacency[node].append(
                ((node + 1) % node_count, round(rng.uniform(1, 9), 2))
            )
        for _ in range(node_count):
            a, b = rng.randrange(node_count), rng.randrange(node_count)
            if a != b:
                adjacency[a].append((b, round(rng.uniform(1, 9), 2)))

        assert_matches_dijkstra(adjacency)


def test_unreachable_pairs_report_failure() -> None:
    """Two disconnected components: queries across them must fail cleanly."""
    adjacency = {
        0: [(1, 1.0)],
        1: [(0, 1.0)],
        2: [(3, 1.0)],
        3: [(2, 1.0)],
    }
    index = from_adjacency(adjacency)

    result = index.query(0, 3)
    assert not result.found
    assert result.cost == float("inf")
    assert result.path == []

    # The reachable pair still works.
    assert index.query(0, 1).cost == pytest.approx(1.0)


def test_self_query_is_zero_cost() -> None:
    """A query from a node to itself is a zero-cost, single-node path."""
    index = from_adjacency({0: [(1, 5.0)], 1: [(0, 5.0)]})
    result = index.query(1, 1)

    assert result.found
    assert result.cost == 0.0
    assert result.path == [1]


def test_single_node_graph() -> None:
    """Degenerate input: one node, no edges."""
    index = from_adjacency({0: []})
    result = index.query(0, 0)

    assert result.found
    assert result.cost == 0.0


# -- preprocessing invariants ----------------------------------------------


def test_ranks_are_a_permutation() -> None:
    """
    Every node gets exactly one rank, and the ranks are ``0..n-1``.

    If this breaks, the query's rank comparisons are meaningless and its
    correctness argument collapses - so it is checked structurally rather
    than inferred from query results.
    """
    rng = random.Random(5)
    adjacency = random_weighted_graph(rng, 40)
    index = from_adjacency(adjacency)
    prepared = index.prepared

    assert sorted(prepared.rank) == list(range(prepared.node_count))
    assert sorted(prepared.order) == list(range(prepared.node_count))
    for position, node in enumerate(prepared.order):
        assert prepared.rank[node] == position


def test_upward_edges_are_acyclic() -> None:
    """
    Following up-edges strictly increases rank, so the search cannot loop.

    This is what guarantees the query terminates without a visited set.
    """
    rng = random.Random(6)
    adjacency = random_weighted_graph(rng, 40)
    prepared = from_adjacency(adjacency, core_fraction=0.0).prepared

    for node in range(prepared.node_count):
        for neighbor, _ in prepared.up_forward[node]:
            assert prepared.rank[neighbor] > prepared.rank[node]
        for neighbor, _ in prepared.up_backward[node]:
            assert prepared.rank[neighbor] > prepared.rank[node]


def test_contraction_preserves_distances() -> None:
    """
    The augmented graph has the same shortest-path distances as the original.

    A stronger statement than "CH queries are right": it isolates contraction
    from the query, so a failure here points at shortcut insertion rather
    than at the bidirectional search.

    Note the graph is built through :func:`build_ch` directly rather than
    :func:`from_adjacency`. ``from_adjacency`` renumbers nodes into dense ids
    in discovery order, so its contracted graph is keyed differently from an
    adjacency-keyed baseline - comparing the two without translating is an
    easy way to write a test that fails on a correct implementation.
    """
    rng = random.Random(7)
    adjacency = random_weighted_graph(rng, 25)
    baseline = adjacency_to_graph(adjacency)
    contracted = build_ch(baseline.copy()).graph

    for start in range(len(adjacency)):
        for goal in range(len(adjacency)):
            before = dijkstra_on_ch_graph(baseline, start, goal)
            after = dijkstra_on_ch_graph(contracted, start, goal)

            assert before.found == after.found
            if before.found:
                assert after.cost == pytest.approx(before.cost, abs=1e-9)


def test_shortcuts_only_replace_real_two_edge_paths() -> None:
    """
    Every shortcut's cost equals the two-edge path it stands for.

    Catches a shortcut whose recorded ``via`` does not match its cost, which
    would make unpacking produce a path with the wrong total.
    """
    rng = random.Random(8)
    adjacency = random_weighted_graph(rng, 30)
    graph = build_ch(adjacency_to_graph(adjacency)).graph

    for (u, w), (via, cost) in graph.shortcuts.items():
        assert graph.edge_cost(u, via) + graph.edge_cost(via, w) == pytest.approx(
            cost, abs=1e-9
        )


def test_core_nodes_share_the_top_of_the_hierarchy() -> None:
    """A non-zero core leaves exactly that many nodes uncontracted."""
    rng = random.Random(9)
    adjacency = random_weighted_graph(rng, 100)
    prepared = from_adjacency(adjacency, core_fraction=0.1).prepared

    assert prepared.stats.core_size == 10
    core = [
        node
        for node in range(prepared.node_count)
        if prepared.rank[node] >= prepared.core_rank
    ]
    assert len(core) == 10


# -- reporting and plumbing ------------------------------------------------


def test_progress_callback_reports_completion() -> None:
    """
    Verbose output is driven by structured events, and they add up.

    The CLI's progress display is only as trustworthy as these events, and a
    callback that silently stopped firing would make a long preprocessing run
    look hung.
    """
    rng = random.Random(10)
    adjacency = random_weighted_graph(rng, 60)
    graph = adjacency_to_graph(adjacency)

    events: list[ProgressEvent] = []
    build_ch(graph, on_progress=events.append)

    assert events, "No progress events were emitted."
    assert events[0].phase == "start"
    assert events[-1].phase == "done"
    assert events[-1].contracted == graph.node_count
    assert events[-1].percent == pytest.approx(100.0)


def test_stats_are_populated() -> None:
    """Preprocessing reports the numbers the analysis depends on."""
    rng = random.Random(11)
    adjacency = random_weighted_graph(rng, 50)
    stats = from_adjacency(adjacency).prepared.stats

    assert stats.node_count == 50
    assert stats.original_edge_count > 0
    assert stats.shortcuts_added >= 0
    assert stats.witness_searches > 0
    assert stats.elapsed_seconds > 0.0


def test_unpacked_and_packed_costs_agree() -> None:
    """
    Skipping unpacking changes the path representation, never the cost.

    The benchmarks time queries with ``unpack=False``; if that path reported
    different costs, every timing number would be measuring something other
    than what the correctness tests check.
    """
    rng = random.Random(12)
    adjacency = random_weighted_graph(rng, 40)
    index = from_adjacency(adjacency)
    prepared = index.prepared

    for start in range(0, 40, 7):
        for goal in range(0, 40, 5):
            packed = ch_query(prepared, start, goal, unpack=False)
            unpacked = ch_query(prepared, start, goal, unpack=True)

            assert packed.found == unpacked.found
            if packed.found:
                assert packed.cost == pytest.approx(unpacked.cost)
                assert len(unpacked.path) >= len(packed.path)


def test_query_rejects_unknown_labels() -> None:
    """A label that is not in the index is an error, not a silent failure."""
    index = from_adjacency({0: [(1, 1.0)], 1: []})

    with pytest.raises(ValueError, match="not in the indexed graph"):
        index.query(0, 99)


def test_negative_edges_are_rejected() -> None:
    """
    Negative weights break Dijkstra, and therefore break CH.

    Failing at construction is much better than returning a wrong distance
    later, so the graph refuses them outright.
    """
    graph = CHGraph(2)
    with pytest.raises(ValueError, match="Negative edge cost"):
        graph.add_edge(0, 1, -1.0)
