"""
Contraction Hierarchies against the real Tuscaloosa street network.

The synthetic and grid tests in ``test_ch.py`` prove the algorithm is correct
on graphs chosen to stress it. This module proves it on the graph the project
actually claims results for: 4,630 real intersections with one-way streets,
dead ends, and disconnected fragments - none of which a generated graph
reproduces faithfully.

These tests need the cached OSM graph. Following the convention already used
by ``test_replanning.py``, the whole module skips when it has not been
downloaded, so a fresh checkout still runs the rest of the suite offline::

    python examples/build_graph.py
"""

from __future__ import annotations

import random

import pytest

from search_algorithms.ch.build import dijkstra_on_ch_graph, from_osm_problem
from search_algorithms.ch.graph import CHGraph
from search_algorithms.ch.query import ch_query

osmnx = pytest.importorskip("osmnx", reason="Needs the 'osm' extra.")

from search_algorithms.problems.osm_problem import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_PLACE,
    OSMProblem,
    load_place_graph,
)

#: How many random pairs the equivalence test checks. This is the number the
#: analysis quotes as its correctness evidence, so it is defined once here
#: rather than inlined.
QUERY_PAIRS = 200


def _cached_graph_exists() -> bool:
    cache = DEFAULT_CACHE_DIR / "tuscaloosa_alabama_usa_drive.graphml"
    return cache.exists()


pytestmark = pytest.mark.skipif(
    not _cached_graph_exists(),
    reason="Cached street graph not found; run examples/build_graph.py first.",
)


@pytest.fixture(scope="module")
def street_graph():
    return load_place_graph(DEFAULT_PLACE)


@pytest.fixture(scope="module")
def indexed(street_graph):
    """
    The CH index plus an uncontracted baseline in the same dense id space.

    Building the baseline from the problem rather than by stripping shortcuts
    out of the contracted graph matters: contraction can lower an existing
    edge's cost, so subtracting shortcuts would not recover the original
    weights.
    """
    nodes = [int(node) for node in street_graph.nodes]
    problem = OSMProblem(street_graph, start=nodes[0], goal=nodes[-1])
    index = from_osm_problem(problem)

    baseline = CHGraph(index.node_count)
    for dense in range(index.node_count):
        label = index.original_label(dense)
        for neighbor in problem.neighbors(label):
            baseline.add_edge(
                dense,
                index.dense_id(int(neighbor)),
                problem.edge_cost(label, neighbor),
            )

    return index, baseline, street_graph


def test_graph_is_the_expected_size(indexed) -> None:
    """Guards the figures the analysis quotes for the real network."""
    index, _, _ = indexed
    assert index.node_count == 4630


def test_random_pairs_match_dijkstra_exactly(indexed) -> None:
    """
    The headline correctness claim: CH is exact on the real network.

    Every reachable pair must return Dijkstra's cost to within float noise.
    This is the test behind the "200/200 pairs match" statement in the
    analysis - if the count here changes, that sentence has to change too.
    """
    index, baseline, _ = indexed
    rng = random.Random(42)

    checked = 0
    reachable = 0

    for _ in range(QUERY_PAIRS):
        start = rng.randrange(index.node_count)
        goal = rng.randrange(index.node_count)

        expected = dijkstra_on_ch_graph(baseline, start, goal)
        actual = ch_query(index.prepared, start, goal, unpack=False)
        checked += 1

        assert actual.found == expected.found, (
            f"{start} -> {goal}: CH found={actual.found}, "
            f"Dijkstra found={expected.found}"
        )

        if expected.found:
            reachable += 1
            assert actual.cost == pytest.approx(expected.cost, rel=1e-9)

    assert checked == QUERY_PAIRS
    # A graph where almost nothing is reachable would make the test vacuous.
    assert reachable > QUERY_PAIRS * 0.8


def test_unpacked_paths_are_drivable(indexed) -> None:
    """
    Unpacked routes are legal walks on the street network.

    ``validate_path`` checks each step is a real one-way-respecting move, so
    this catches shortcut unpacking that produces a correct total over an
    impossible route.
    """
    index, _, graph = indexed
    rng = random.Random(7)

    checked = 0
    for _ in range(25):
        start = index.original_label(rng.randrange(index.node_count))
        goal = index.original_label(rng.randrange(index.node_count))

        result = index.query(start, goal)
        if not result.found:
            continue

        problem = OSMProblem(graph, start=start, goal=goal)
        valid, message = problem.validate_path(result.path)
        assert valid, message
        assert problem.path_cost(result.path) == pytest.approx(result.cost, rel=1e-9)
        checked += 1

    assert checked > 0, "No reachable pairs were sampled."


def test_ch_expands_far_fewer_nodes(indexed) -> None:
    """
    CH's advantage on a real road network, asserted rather than assumed.

    The threshold is deliberately loose - this is a regression guard against
    the hierarchy silently degenerating, not a benchmark. The measured figure
    is roughly 25x; asserting 5x leaves room for tuning without letting a
    broken ordering pass.
    """
    index, baseline, _ = indexed
    rng = random.Random(11)

    dijkstra_total = 0
    ch_total = 0

    for _ in range(40):
        start = rng.randrange(index.node_count)
        goal = rng.randrange(index.node_count)

        expected = dijkstra_on_ch_graph(baseline, start, goal)
        if not expected.found:
            continue

        actual = ch_query(index.prepared, start, goal, unpack=False)
        dijkstra_total += expected.nodes_expanded
        ch_total += actual.nodes_expanded

    assert ch_total > 0
    assert dijkstra_total / ch_total > 5.0, (
        f"CH expanded {ch_total} nodes vs Dijkstra's {dijkstra_total}; "
        "the hierarchy may have degenerated."
    )


def test_one_way_streets_are_respected(indexed) -> None:
    """
    Asymmetric costs survive contraction.

    A CH that lost edge direction would still pass symmetric tests, so this
    checks a pair where the two directions genuinely differ - which on a real
    street network with one-way roads is common.
    """
    index, baseline, _ = indexed
    rng = random.Random(3)

    asymmetric_found = 0
    for _ in range(80):
        start = rng.randrange(index.node_count)
        goal = rng.randrange(index.node_count)

        there = dijkstra_on_ch_graph(baseline, start, goal)
        back = dijkstra_on_ch_graph(baseline, goal, start)
        if not (there.found and back.found):
            continue
        if there.cost == pytest.approx(back.cost, rel=1e-6):
            continue

        asymmetric_found += 1
        assert ch_query(index.prepared, start, goal, unpack=False).cost == (
            pytest.approx(there.cost, rel=1e-9)
        )
        assert ch_query(index.prepared, goal, start, unpack=False).cost == (
            pytest.approx(back.cost, rel=1e-9)
        )

    assert asymmetric_found > 0, "No asymmetric pair was sampled."
