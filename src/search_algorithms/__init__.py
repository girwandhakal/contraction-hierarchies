"""
search-algorithms
=================

Classic graph search (BFS, UCS, A* and friends), generalized from a class
assignment on weighted grids into a library that routes over real
OpenStreetMap street networks - and shows you how each algorithm got there.

Quick start
-----------
::

    from search_algorithms import GridProblem, astar

    problem = GridProblem.from_file("maps/map1.txt")
    result = astar(problem)
    print(result.cost, result.nodes_expanded)

The optional OpenStreetMap layer lives in
:mod:`search_algorithms.problems.osm_problem` and is imported lazily, so the
core package has no heavy geospatial dependencies.

Contraction Hierarchies - preprocessing a static graph so queries get much
faster - lives in :mod:`search_algorithms.ch`::

    from search_algorithms.ch import from_problem

    index = from_problem(problem)
    result = index.query(problem.start, problem.goal)

It is imported on demand rather than here, keeping this module's import free
of preprocessing machinery that most callers never touch.
"""

from .core.algorithms import (
    ALGORITHMS,
    CLASSIC_ALGORITHMS,
    astar,
    bfs,
    greedy_best_first,
    reconstruct_path,
    ucs,
    weighted_astar,
)
from .core.heuristics import chebyshev, euclidean, haversine, manhattan
from .core.problem import Problem
from .core.result import SearchResult
from .problems.graph_problem import GraphProblem
from .problems.grid_problem import GridProblem

__version__ = "0.1.0"

__all__ = [
    # algorithms
    "bfs",
    "ucs",
    "astar",
    "weighted_astar",
    "greedy_best_first",
    "reconstruct_path",
    "ALGORITHMS",
    "CLASSIC_ALGORITHMS",
    # heuristics
    "manhattan",
    "euclidean",
    "chebyshev",
    "haversine",
    # core types
    "Problem",
    "SearchResult",
    # problems
    "GridProblem",
    "GraphProblem",
]
