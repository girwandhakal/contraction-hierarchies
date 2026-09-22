"""
Heuristic functions.

Each heuristic estimates the remaining cost from a state to the goal. A
heuristic is *admissible* if it never overestimates that cost, which is what
guarantees A* returns an optimal path.

Problems wire one of these into their ``heuristic`` method, baking in the goal
so the algorithms never need to know which heuristic is in play.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

#: Mean Earth radius in meters (WGS-84 mean radius), used by :func:`haversine`.
EARTH_RADIUS_METERS = 6_371_008.8


def manhattan(state: tuple[int, int], goal: tuple[int, int]) -> float:
    """
    Manhattan (L1) distance between two grid cells.

        h(n) = |row_n - row_goal| + |col_n - col_goal|

    Admissible on a 4-connected grid whose minimum step cost is at least 1,
    because reaching the goal needs at least this many moves and no move can
    cost less than 1.
    """
    return float(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))


def euclidean(state: tuple[float, float], goal: tuple[float, float]) -> float:
    """
    Straight-line (L2) distance between two points in the plane.

    Admissible whenever movement cost is at least proportional to Euclidean
    distance - notably on grids that allow diagonal moves, where Manhattan
    distance would overestimate.
    """
    return sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)


def chebyshev(state: tuple[int, int], goal: tuple[int, int]) -> float:
    """
    Chebyshev (L-infinity) distance: ``max(|dr|, |dc|)``.

    The right heuristic for an 8-connected grid where a diagonal move costs
    the same as a straight one.
    """
    return float(max(abs(state[0] - goal[0]), abs(state[1] - goal[1])))


def haversine(
    state: tuple[float, float],
    goal: tuple[float, float],
) -> float:
    """
    Great-circle distance in meters between two ``(latitude, longitude)``
    points given in decimal degrees.

    This is the heuristic for real road networks. It is admissible against
    edge costs measured in meters because no road between two points can be
    shorter than the straight line over the Earth's surface joining them.
    """
    lat1, lon1 = state
    lat2, lon2 = goal

    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_METERS * asin(sqrt(a))
