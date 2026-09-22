"""
Correctness tests for the search algorithms on grid problems.

The headline test is :func:`test_matches_reference_results`, which pins the
exact figures published in ``docs/analysis.md``. If a refactor ever changes
tie-breaking or the node-counting convention, that test fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from search_algorithms import (
    ALGORITHMS,
    GridProblem,
    astar,
    bfs,
    greedy_best_first,
    ucs,
    weighted_astar,
)

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

OPTIMAL_ALGORITHMS = (ucs, astar)
ALL_ALGORITHMS = (bfs, ucs, astar, greedy_best_first, weighted_astar)


# --------------------------------------------------------------------------
# Regression against the published reference results
# --------------------------------------------------------------------------

#: (map, algorithm) -> (cost, steps, nodes_expanded), transcribed from
#: docs/analysis.md. These are the numbers the implementation must reproduce.
REFERENCE_RESULTS = {
    ("map1.txt", "bfs"): (11, 11, 25),
    ("map1.txt", "ucs"): (11, 11, 25),
    ("map1.txt", "astar"): (11, 11, 19),
    ("map2.txt", "bfs"): (57, 8, 15),
    ("map2.txt", "ucs"): (12, 12, 13),
    ("map2.txt", "astar"): (12, 12, 12),
    ("map3.txt", "bfs"): (28, 28, 336),
    ("map3.txt", "ucs"): (28, 28, 336),
    ("map3.txt", "astar"): (28, 28, 126),
}


@pytest.mark.parametrize(
    ("map_name", "algorithm_name", "expected"),
    [(m, a, v) for (m, a), v in REFERENCE_RESULTS.items()],
)
def test_matches_reference_results(map_name, algorithm_name, expected):
    """The implementation reproduces the published reference figures."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    result = ALGORITHMS[algorithm_name](problem)

    expected_cost, expected_steps, expected_expanded = expected
    assert result.cost == expected_cost
    assert result.steps == expected_steps
    assert result.nodes_expanded == expected_expanded


# --------------------------------------------------------------------------
# General correctness properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_returned_path_is_valid(map_name, algorithm):
    """Every algorithm returns a legal start-to-goal walk."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    result = algorithm(problem)

    assert result.found
    valid, message = problem.validate_path(result.path)
    assert valid, message


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_reported_cost_matches_path(map_name, algorithm):
    """The reported cost equals the true cost of the returned path."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    result = algorithm(problem)
    assert result.cost == pytest.approx(problem.path_cost(result.path))


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
def test_ucs_and_astar_agree_on_optimal_cost(map_name):
    """A* with an admissible heuristic finds the same optimal cost as UCS."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    assert astar(problem).cost == ucs(problem).cost


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
def test_astar_never_expands_more_than_ucs(map_name):
    """An informed search should never do more work than the uninformed one."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    assert astar(problem).nodes_expanded <= ucs(problem).nodes_expanded


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
def test_bfs_minimizes_steps(map_name):
    """BFS returns a path with the fewest moves, whatever it costs."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    bfs_steps = bfs(problem).steps
    for algorithm in OPTIMAL_ALGORITHMS:
        assert bfs_steps <= algorithm(problem).steps


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
def test_bfs_never_cheaper_than_ucs(map_name):
    """BFS ignores cost, so it can tie UCS but never beat it."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    assert bfs(problem).cost >= ucs(problem).cost


# --------------------------------------------------------------------------
# Expansion trace - the record of how each search explored
# --------------------------------------------------------------------------


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_expansion_order_length_matches_count(map_name, algorithm):
    """``nodes_expanded`` is exactly the length of the recorded trace."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    result = algorithm(problem)
    assert len(result.expansion_order) == result.nodes_expanded


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_expansion_order_has_no_duplicates(map_name, algorithm):
    """No state is expanded twice."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    order = algorithm(problem).expansion_order
    assert len(set(order)) == len(order)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_goal_is_not_counted_as_expanded(algorithm):
    """Search stops when the goal leaves the frontier, so it is never expanded."""
    problem = GridProblem.from_file(MAPS_DIR / "map1.txt")
    result = algorithm(problem)
    assert problem.goal not in result.expansion_order


@pytest.mark.parametrize("map_name", ["map1.txt", "map2.txt", "map3.txt"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_parent_pointers_reconstruct_the_path(map_name, algorithm):
    """Every state on the path (bar the start) has a parent in the trace."""
    problem = GridProblem.from_file(MAPS_DIR / map_name)
    result = algorithm(problem)

    assert result.parent_of[problem.start] is None
    for child, parent in zip(result.path[1:], result.path):
        assert result.parent_of[child] == parent


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_start_equals_goal_is_not_possible_but_adjacent_is_trivial(algorithm):
    """A one-move problem returns a two-state path costing one step."""
    problem = GridProblem.from_string("S G")
    result = algorithm(problem)

    assert result.path == [(0, 0), (0, 1)]
    assert result.cost == 1
    assert result.nodes_expanded == 1


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_unreachable_goal_reports_failure(algorithm):
    """A walled-off goal yields an empty path and infinite cost."""
    problem = GridProblem.from_string(
        """
        S 1 # 1 G
        1 1 # 1 1
        """
    )
    result = algorithm(problem)

    assert not result.found
    assert result.path == []
    assert result.cost == float("inf")
    assert result.nodes_expanded > 0


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_single_corridor_forces_one_path(algorithm):
    """With only one route available, every algorithm must agree on it."""
    problem = GridProblem.from_string(
        """
        S 1 1 1 G
        # # # # #
        """
    )
    result = algorithm(problem)
    assert result.path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_weighted_astar_is_bounded_suboptimal():
    """Weighted A* costs at most ``weight`` x optimal, and usually works less."""
    problem = GridProblem.from_file(MAPS_DIR / "map3.txt")
    optimal = astar(problem)
    relaxed = weighted_astar(problem, weight=3.0)

    assert relaxed.cost <= 3.0 * optimal.cost
    assert relaxed.nodes_expanded <= optimal.nodes_expanded


def test_weighted_astar_rejects_weight_below_one():
    problem = GridProblem.from_file(MAPS_DIR / "map1.txt")
    with pytest.raises(ValueError):
        weighted_astar(problem, weight=0.5)


# --------------------------------------------------------------------------
# Map parsing
# --------------------------------------------------------------------------


def test_rejects_map_without_exactly_one_start():
    with pytest.raises(ValueError, match="exactly one S"):
        GridProblem.from_string("S S G")


def test_rejects_map_without_exactly_one_goal():
    with pytest.raises(ValueError, match="exactly one G"):
        GridProblem.from_string("S 1 1")


def test_rejects_ragged_map():
    with pytest.raises(ValueError, match="rectangular"):
        GridProblem([["S", "1"], ["1", "1", "G"]])


def test_rejects_invalid_token():
    with pytest.raises(ValueError, match="Invalid token"):
        GridProblem.from_string("S X G")


def test_neighbors_follow_fixed_order():
    """Neighbour order is up, right, down, left - it fixes tie-breaking."""
    problem = GridProblem.from_string(
        """
        1 1 1
        1 S 1
        1 1 G
        """
    )
    assert problem.neighbors((1, 1)) == [(0, 1), (1, 2), (2, 1), (1, 0)]
