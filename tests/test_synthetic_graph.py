"""
Tests for the synthetic road-like graph generator.

The generator is benchmark infrastructure, not a deliverable, but a broken
generator silently corrupts every number the benchmark produces - so its
structural promises are asserted rather than assumed.

The admissibility test earns its place
--------------------------------------
Arterial edges cost less per unit length than local streets, which means raw
straight-line distance *overestimates* the remaining cost and is therefore
not admissible. An earlier version of the generator handed A* that raw
distance, and A* duly returned routes up to 30% too long - while looking
perfectly healthy, because nothing was checking its answers. The benchmark's
inline cost check caught it; :func:`test_heuristic_is_admissible` is what
stops it coming back.
"""

from __future__ import annotations

import math

import pytest

from search_algorithms.benchmark.synthetic_graph import (
    ARTERIAL_SPEED_FACTOR,
    generate,
    road_like_graph,
)
from search_algorithms.ch.build import dijkstra_on_ch_graph

SIZES = (50, 200, 1_000)


@pytest.mark.parametrize("node_count", SIZES)
def test_graph_has_requested_size(node_count: int) -> None:
    synthetic = generate(node_count, seed=1)
    assert synthetic.node_count == node_count
    assert synthetic.edge_count > 0


@pytest.mark.parametrize("node_count", SIZES)
def test_graph_is_strongly_connected(node_count: int) -> None:
    """
    Every node reaches every other node.

    Disconnected fragments would make the benchmark quietly measure
    connectivity: an unreachable pair forces both algorithms to exhaust their
    component, which is a different quantity from routing speed.
    """
    synthetic = generate(node_count, seed=2)
    graph = synthetic.graph

    reachable = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in graph.successors(node):
            if neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)

    assert len(reachable) == node_count


@pytest.mark.parametrize("node_count", SIZES)
def test_degree_stays_road_like(node_count: int) -> None:
    """
    Mean degree resembles a street network and no node becomes a hub.

    This is the property that keeps preprocessing tractable. A generator that
    produced a few very high-degree nodes would make contraction consider
    O(degree^2) neighbour pairs at those nodes and the benchmark would end up
    measuring the generator's artefacts.
    """
    synthetic = generate(node_count, seed=3)
    graph = synthetic.graph

    degrees = [len(graph.successors(node)) for node in graph.nodes()]
    mean_degree = sum(degrees) / node_count

    assert 2.0 <= mean_degree <= 6.0, f"mean out-degree {mean_degree:.2f}"
    assert max(degrees) <= 12, f"max out-degree {max(degrees)}"


@pytest.mark.parametrize("node_count", SIZES)
def test_heuristic_is_admissible(node_count: int) -> None:
    """
    The heuristic never overestimates the true remaining cost.

    Admissibility is what makes A* optimal, so an inadmissible heuristic
    turns the A* baseline into a different (and worse) algorithm while
    looking fine. Checked directly against true distances rather than
    inferred from A*'s answers.
    """
    synthetic = generate(node_count, seed=4)
    graph = synthetic.graph

    goal = node_count - 1
    heuristic = synthetic.heuristic_to(goal)

    # True distances to the goal, via a reverse search from it.
    checked = 0
    for start in range(0, node_count, max(1, node_count // 25)):
        true_cost = dijkstra_on_ch_graph(graph, start, goal)
        if not true_cost.found:
            continue
        estimate = heuristic(start)
        assert estimate <= true_cost.cost + 1e-9, (
            f"h({start}) = {estimate} overestimates true cost {true_cost.cost}"
        )
        checked += 1

    assert checked > 0


def test_heuristic_accounts_for_arterial_speed() -> None:
    """
    The heuristic is scaled by exactly the arterial factor.

    Pins the relationship the admissibility argument depends on, so changing
    the arterial cost without revisiting the heuristic fails here rather than
    silently producing a wrong baseline.
    """
    synthetic = generate(100, seed=5)
    heuristic = synthetic.heuristic_to(99)

    raw = synthetic.straight_line(0, 99)
    assert heuristic(0) == pytest.approx(raw * ARTERIAL_SPEED_FACTOR)


def test_generation_is_deterministic() -> None:
    """
    The same seed produces the same graph.

    Benchmarks are only reproducible if this holds.
    """
    first = generate(300, seed=7)
    second = generate(300, seed=7)

    assert first.edge_count == second.edge_count
    assert first.xs == second.xs
    for node in range(first.node_count):
        assert first.graph.successors(node) == second.graph.successors(node)


def test_different_seeds_differ() -> None:
    """Different seeds give different graphs, so trials are independent."""
    first = generate(300, seed=1)
    second = generate(300, seed=2)
    assert first.xs != second.xs


def test_edges_are_symmetric_in_cost() -> None:
    """
    Local streets are two-way at equal cost.

    Directed asymmetry is the OSM graph's job; if it appeared here it would
    mean the generator had a bug, since nothing in it builds one-way edges.
    """
    synthetic = generate(200, seed=8)
    graph = synthetic.graph

    for node in graph.nodes():
        for neighbor, cost in graph.successors(node).items():
            assert graph.edge_cost(neighbor, node) == pytest.approx(cost)


def test_road_like_graph_returns_bare_graph() -> None:
    """The convenience wrapper used by the CLI."""
    graph = road_like_graph(150, seed=9)
    assert graph.node_count == 150


def test_rejects_degenerate_sizes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        generate(1)
