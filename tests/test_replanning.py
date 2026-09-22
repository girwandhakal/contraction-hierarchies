"""
Tests for dynamic replanning and the benchmark harness.

The replanning tests use a small hand-built graph that mimics an OSMnx graph
(nodes carrying ``x``/``y``, edges carrying ``length``), so they run fast and
offline without needing the cached Tuscaloosa download.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from search_algorithms import astar, ucs
from search_algorithms.benchmark import benchmark_grids, random_grid, summarize
from search_algorithms.core.heuristics import haversine

# OSMProblem itself only needs networkx; osmnx is imported lazily, and solely
# by the download and nearest-node helpers, neither of which these tests touch.
# The guard is here so the module still skips cleanly rather than erroring at
# collection time if that ever stops being true.
pytest.importorskip("networkx", reason="graph problems need networkx")

from search_algorithms.problems.osm_problem import OSMProblem  # noqa: E402
from search_algorithms.replanning import (  # noqa: E402
    plan_with_full_knowledge,
    simulate_with_closures,
)


#: Length declared on every lattice edge, in meters.
LATTICE_EDGE_METERS = 100.0

#: Latitude the lattice sits at (Tuscaloosa).
LATTICE_ORIGIN_LAT = 33.2
LATTICE_ORIGIN_LON = -87.6

#: Node spacing in degrees. One degree of latitude is ~111.32 km everywhere,
#: but one degree of longitude shrinks by cos(latitude) as you leave the
#: equator - at 33.2 deg it is only ~83% as wide. Scaling the longitude step
#: by 1/cos(lat) keeps both axes a true 100 m apart.
#:
#: This matters because the haversine heuristic reads these coordinates while
#: edge costs come from the declared lengths. If the two disagreed, a
#: perfectly admissible heuristic would look inadmissible and
#: ``test_heuristic_never_exceeds_true_cost`` would fail for the wrong reason.
LATTICE_LAT_STEP_DEG = LATTICE_EDGE_METERS / 111_320.0
LATTICE_LON_STEP_DEG = LATTICE_LAT_STEP_DEG / math.cos(math.radians(LATTICE_ORIGIN_LAT))


def build_test_graph() -> nx.MultiDiGraph:
    """
    A 3x3 lattice with OSM-style attributes.

    Nodes are numbered 0..8 left-to-right, top-to-bottom, spaced so that each
    edge really is about its declared 100 m long, with every edge traversable
    in both directions.
    """
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"

    for index in range(9):
        row, col = divmod(index, 3)
        graph.add_node(
            index,
            x=LATTICE_ORIGIN_LON + col * LATTICE_LON_STEP_DEG,
            y=LATTICE_ORIGIN_LAT + row * LATTICE_LAT_STEP_DEG,
        )

    def connect(a: int, b: int, length: float = LATTICE_EDGE_METERS) -> None:
        graph.add_edge(a, b, length=length)
        graph.add_edge(b, a, length=length)

    # Horizontal edges
    for row in range(3):
        connect(row * 3, row * 3 + 1)
        connect(row * 3 + 1, row * 3 + 2)
    # Vertical edges
    for col in range(3):
        connect(col, col + 3)
        connect(col + 3, col + 6)

    return graph


@pytest.fixture
def graph() -> nx.MultiDiGraph:
    return build_test_graph()


# --------------------------------------------------------------------------
# OSMProblem basics
# --------------------------------------------------------------------------


def test_finds_a_route_on_the_lattice(graph):
    result = astar(OSMProblem(graph, start=0, goal=8))
    assert result.found
    assert result.path[0] == 0
    assert result.path[-1] == 8
    assert result.cost == pytest.approx(400.0)  # four 100 m hops


def test_blocked_edges_are_not_traversable(graph):
    problem = OSMProblem(graph, start=0, goal=8, blocked_edges={(0, 1)})
    assert 1 not in problem.neighbors(0)
    assert 3 in problem.neighbors(0)


def test_blocking_everything_out_of_start_makes_it_unsolvable(graph):
    problem = OSMProblem(graph, start=0, goal=8, blocked_edges={(0, 1), (0, 3)})
    assert not astar(problem).found


def test_lattice_geometry_matches_declared_edge_lengths(graph):
    """
    Guard the fixture itself.

    If the node spacing and the declared edge lengths ever drift apart, the
    admissibility test below would fail for a reason that has nothing to do
    with the heuristic. Catch that here instead.
    """
    problem = OSMProblem(graph, start=0, goal=8)
    for a, b in ((0, 1), (0, 3), (4, 5), (4, 7)):
        declared = problem.edge_cost(a, b)
        actual = haversine(problem.latlon(a), problem.latlon(b))
        assert actual == pytest.approx(declared, rel=0.02), (
            f"edge {a}->{b} declares {declared} m but its coordinates are "
            f"{actual:.1f} m apart"
        )


def test_heuristic_is_zero_at_the_goal(graph):
    problem = OSMProblem(graph, start=0, goal=8)
    assert problem.heuristic(8) == pytest.approx(0.0)


def test_heuristic_never_exceeds_true_cost(graph):
    """Admissibility check: h(n) <= the real remaining cost, for every node."""
    for node in graph.nodes:
        problem = OSMProblem(graph, start=node, goal=8)
        actual = ucs(problem)
        if actual.found:
            assert problem.heuristic(node) <= actual.cost + 1e-9


def test_rejects_start_outside_the_graph(graph):
    with pytest.raises(ValueError, match="Start node"):
        OSMProblem(graph, start=999, goal=8)


def test_rejects_goal_outside_the_graph(graph):
    with pytest.raises(ValueError, match="Goal node"):
        OSMProblem(graph, start=0, goal=999)


# --------------------------------------------------------------------------
# Replanning
# --------------------------------------------------------------------------


def test_no_closures_means_no_replans(graph):
    outcome = simulate_with_closures(graph, 0, 8, closures=[])
    assert outcome.reached_goal
    assert outcome.replan_count == 0
    assert outcome.travelled_cost == pytest.approx(400.0)


def test_agent_replans_around_a_closure(graph):
    """Close the first step of the planned route and the agent must adapt."""
    planned = astar(OSMProblem(graph, start=0, goal=8)).path
    first_edge = (planned[0], planned[1])

    outcome = simulate_with_closures(graph, 0, 8, closures=[first_edge])

    assert outcome.reached_goal
    assert outcome.replan_count == 1
    assert outcome.travelled_path[-1] == 8
    # The closed edge must not appear in what was actually driven.
    driven = list(zip(outcome.travelled_path, outcome.travelled_path[1:]))
    assert first_edge not in driven


def test_travelled_path_is_contiguous(graph):
    """Every consecutive pair in the driven path is a real, open edge."""
    planned = astar(OSMProblem(graph, start=0, goal=8)).path
    outcome = simulate_with_closures(graph, 0, 8, closures=[(planned[0], planned[1])])

    for a, b in zip(outcome.travelled_path, outcome.travelled_path[1:]):
        assert graph.has_edge(a, b)


def test_unreachable_goal_is_reported(graph):
    """Fence the goal off completely and the agent should give up honestly."""
    closures = [(5, 8), (7, 8)]
    outcome = simulate_with_closures(graph, 0, 8, closures=closures)
    assert not outcome.reached_goal


def test_total_nodes_expanded_sums_every_search(graph):
    planned = astar(OSMProblem(graph, start=0, goal=8)).path
    outcome = simulate_with_closures(graph, 0, 8, closures=[(planned[0], planned[1])])

    expected = outcome.initial_result.nodes_expanded + sum(
        event.result.nodes_expanded for event in outcome.replans
    )
    assert outcome.total_nodes_expanded == expected


def test_omniscient_planner_is_never_worse(graph):
    """Knowing the closures up front cannot produce a longer drive."""
    planned = astar(OSMProblem(graph, start=0, goal=8)).path
    closures = [(planned[0], planned[1])]

    outcome = simulate_with_closures(graph, 0, 8, closures=closures)
    omniscient = plan_with_full_knowledge(graph, 0, 8, closures)

    assert omniscient.found
    assert omniscient.cost <= outcome.travelled_cost + 1e-9


def test_omniscient_plan_avoids_closures(graph):
    planned = astar(OSMProblem(graph, start=0, goal=8)).path
    closures = [(planned[0], planned[1])]

    result = plan_with_full_knowledge(graph, 0, 8, closures)
    edges = list(zip(result.path, result.path[1:]))
    assert closures[0] not in edges


# --------------------------------------------------------------------------
# Benchmark harness
# --------------------------------------------------------------------------


def test_random_grid_is_always_solvable():
    from search_algorithms import bfs

    import random as _random

    for seed in range(8):
        problem = random_grid(12, obstacle_density=0.25, rng=_random.Random(seed))
        assert bfs(problem).found


def test_random_grid_respects_max_weight():
    import random as _random

    problem = random_grid(10, obstacle_density=0.1, max_weight=1, rng=_random.Random(0))
    tokens = {token for row in problem.grid for token in row}
    assert tokens <= {"S", "G", "#", "1"}


def test_random_grid_gives_up_on_impossible_density():
    import random as _random

    with pytest.raises(RuntimeError, match="Could not generate"):
        random_grid(
            8,
            obstacle_density=0.99,
            rng=_random.Random(0),
            max_attempts=5,
        )


def test_benchmark_returns_a_row_per_algorithm_and_trial():
    rows = benchmark_grids(sizes=[8], trials=3, algorithms=["bfs", "ucs", "astar"])
    assert len(rows) == 3 * 3
    assert {row.algorithm for row in rows} == {"bfs", "ucs", "astar"}


def test_benchmark_gives_every_algorithm_the_same_problems():
    """UCS and A* must agree on cost trial-by-trial, proving the maps match."""
    rows = benchmark_grids(sizes=[10], trials=4, algorithms=["ucs", "astar"])

    by_trial: dict[int, dict[str, float]] = {}
    for row in rows:
        by_trial.setdefault(row.trial, {})[row.algorithm] = row.cost

    for costs in by_trial.values():
        assert costs["ucs"] == pytest.approx(costs["astar"])


def test_summarize_mentions_the_heuristic_caveat_on_weighted_grids():
    rows = benchmark_grids(sizes=[8], trials=2, max_weight=9)
    assert "admissible" in summarize(rows)


def test_summarize_omits_caveat_on_uniform_grids():
    rows = benchmark_grids(sizes=[8], trials=2, max_weight=1)
    assert "admissible" not in summarize(rows)
