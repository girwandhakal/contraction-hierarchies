"""
Tests that the ``Problem`` abstraction really is problem-agnostic.

The point of these tests is to show the same unmodified algorithms solve a
plain weighted graph - no grid, no coordinates - and that the interface
behaves correctly on directed edges, which is what real one-way streets need.
"""

from __future__ import annotations

import math

import pytest

from search_algorithms import ALGORITHMS, GraphProblem, astar, bfs, ucs
from search_algorithms.core.heuristics import euclidean, haversine, manhattan

ALL_ALGORITHMS = list(ALGORITHMS.values())


# --------------------------------------------------------------------------
# A small textbook graph with a known optimal answer
# --------------------------------------------------------------------------
#
#        4        8
#   A ------ B ------ C
#   |        |        |
# 2 |      5 |        | 3
#   |        |        |
#   D ------ E ------ F
#        1        7
#
# Cheapest A -> F is A-D-E-B-C-F? No: A-D-E-F = 2 + 1 + 7 = 10.
#
TEXTBOOK_EDGES = [
    ("A", "B", 4.0),
    ("B", "C", 8.0),
    ("A", "D", 2.0),
    ("B", "E", 5.0),
    ("C", "F", 3.0),
    ("D", "E", 1.0),
    ("E", "F", 7.0),
]


def textbook_problem() -> GraphProblem:
    return GraphProblem.from_undirected(TEXTBOOK_EDGES, start="A", goal="F")


@pytest.mark.parametrize("algorithm", [ucs, astar])
def test_optimal_algorithms_find_cheapest_route(algorithm):
    """UCS and A* (zero heuristic) agree on the optimal cost of 10."""
    result = algorithm(textbook_problem())
    assert result.path == ["A", "D", "E", "F"]
    assert result.cost == pytest.approx(10.0)


def test_bfs_finds_fewest_hops_not_cheapest():
    """BFS takes the 2-hop route via B/C-ish shape rather than the cheap one."""
    result = bfs(textbook_problem())
    assert result.found
    # Every edge counts as one hop, so BFS cannot return more hops than UCS.
    assert result.steps <= ucs(textbook_problem()).steps


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_every_algorithm_solves_a_plain_graph(algorithm):
    """The grid algorithms run unmodified on a non-grid problem."""
    problem = textbook_problem()
    result = algorithm(problem)

    assert result.found
    valid, message = problem.validate_path(result.path)
    assert valid, message


def test_path_cost_agrees_with_reported_cost():
    problem = textbook_problem()
    result = ucs(problem)
    assert problem.path_cost(result.path) == pytest.approx(result.cost)


# --------------------------------------------------------------------------
# Directed edges (one-way streets)
# --------------------------------------------------------------------------


def test_directed_edges_are_respected():
    """A one-way edge cannot be traversed backwards."""
    problem = GraphProblem(
        adjacency={
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [],
        },
        start="A",
        goal="C",
    )
    assert ucs(problem).path == ["A", "B", "C"]

    # Reversing the request is impossible: no edges lead back.
    backwards = GraphProblem(
        adjacency={"A": [("B", 1.0)], "B": [("C", 1.0)], "C": []},
        start="C",
        goal="A",
    )
    assert not ucs(backwards).found


def test_asymmetric_costs_are_respected():
    """A -> B and B -> A may cost different amounts."""
    problem = GraphProblem(
        adjacency={"A": [("B", 1.0)], "B": [("A", 99.0)]},
        start="A",
        goal="B",
    )
    assert problem.edge_cost("A", "B") == 1.0
    assert problem.edge_cost("B", "A") == 99.0


def test_parallel_edges_collapse_to_cheapest():
    problem = GraphProblem(
        adjacency={"A": [("B", 5.0), ("B", 2.0)], "B": []},
        start="A",
        goal="B",
    )
    assert problem.edge_cost("A", "B") == 2.0


# --------------------------------------------------------------------------
# Heuristics wired through the interface
# --------------------------------------------------------------------------


def test_heuristic_makes_astar_goal_directed():
    """
    On a wide grid-like graph, a good heuristic should cut expansions.

    Nodes are laid out on a line so Euclidean distance to the goal is exact,
    which lets A* walk straight there while UCS fans out.
    """
    positions = {i: (float(i), 0.0) for i in range(30)}
    adjacency = {
        i: [(j, 1.0) for j in (i - 1, i + 1) if j in positions] for i in positions
    }

    goal = 29
    informed = GraphProblem(
        adjacency,
        start=15,
        goal=goal,
        heuristic=lambda n: euclidean(positions[n], positions[goal]),
    )
    uninformed = GraphProblem(adjacency, start=15, goal=goal)

    assert astar(informed).cost == ucs(uninformed).cost
    assert astar(informed).nodes_expanded < ucs(uninformed).nodes_expanded


def test_zero_heuristic_makes_astar_equal_ucs():
    """h(n) = 0 everywhere turns A* into UCS exactly."""
    problem = textbook_problem()  # no heuristic supplied -> zero heuristic
    assert astar(problem).nodes_expanded == ucs(problem).nodes_expanded
    assert astar(problem).expansion_order == ucs(problem).expansion_order


# --------------------------------------------------------------------------
# Construction validation
# --------------------------------------------------------------------------


def test_rejects_missing_start():
    with pytest.raises(ValueError, match="Start node"):
        GraphProblem({"A": []}, start="Z", goal="A")


def test_rejects_missing_goal():
    with pytest.raises(ValueError, match="Goal node"):
        GraphProblem({"A": []}, start="A", goal="Z")


def test_rejects_negative_costs():
    with pytest.raises(ValueError, match="Negative edge cost"):
        GraphProblem({"A": [("B", -1.0)], "B": []}, start="A", goal="B")


def test_unknown_edge_lookup_raises():
    problem = textbook_problem()
    with pytest.raises(ValueError, match="No edge"):
        problem.edge_cost("A", "C")


# --------------------------------------------------------------------------
# Heuristic functions themselves
# --------------------------------------------------------------------------


def test_manhattan_matches_hand_calculation():
    assert manhattan((0, 0), (3, 4)) == 7.0


def test_euclidean_matches_hand_calculation():
    assert euclidean((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_haversine_is_zero_for_identical_points():
    point = (33.2098, -87.5692)  # Tuscaloosa
    assert haversine(point, point) == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_known_distance():
    """Tuscaloosa to Birmingham is roughly 75 km as the crow flies."""
    tuscaloosa = (33.2098, -87.5692)
    birmingham = (33.5186, -86.8104)
    meters = haversine(tuscaloosa, birmingham)
    assert 70_000 < meters < 82_000


def test_haversine_is_symmetric():
    a, b = (33.2098, -87.5692), (33.5186, -86.8104)
    assert haversine(a, b) == pytest.approx(haversine(b, a))


def test_haversine_never_exceeds_half_circumference():
    antipodal = haversine((0.0, 0.0), (0.0, 180.0))
    assert antipodal == pytest.approx(math.pi * 6_371_008.8, rel=1e-6)
