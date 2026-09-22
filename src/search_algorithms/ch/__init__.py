"""
Contraction Hierarchies: preprocessing a graph so queries get much faster.

Plain Dijkstra and A* pay their full search cost on every query. When the
graph is *static* and the queries are *many* - a routing service answering
requests against a road network that changes monthly - that is the wrong
trade. Contraction Hierarchies pays a one-off preprocessing cost to build an
index, then answers each query by searching a tiny fraction of the graph.

Usage
-----
::

    from search_algorithms import GridProblem
    from search_algorithms.ch import from_problem

    problem = GridProblem.from_file("maps/map3.txt")
    index = from_problem(problem)
    result = index.query(problem.start, problem.goal)

    print(result.cost)            # identical to astar(problem).cost
    print(result.nodes_expanded)  # far fewer

The distance CH returns is *exactly* the optimal distance - this is not an
approximation technique. ``tests/test_ch.py`` asserts that against Dijkstra
and A* across grid maps, random graphs and the real Tuscaloosa street
network.

Module map
----------
``graph``
    ``CHGraph`` (the mutable graph being contracted) and ``PreparedCH`` (the
    immutable queryable hierarchy).
``ordering``
    Node importance: which node to contract next, and the baseline
    strategies the ordering ablation compares against.
``preprocess``
    The contraction loop and the witness search that decides which shortcuts
    are necessary.
``query``
    Bidirectional rank-restricted search plus shortcut unpacking.
``build``
    Adapters from ``Problem``/``OSMProblem``/adjacency maps, handling the
    translation between original node labels and the dense integer ids
    preprocessing needs.
``cli``
    ``python -m search_algorithms.ch.cli`` - verbose preprocessing and
    single-query comparison from the terminal.
"""

from .build import (
    CHIndex,
    dijkstra_on_ch_graph,
    from_adjacency,
    from_osm_problem,
    from_problem,
)
from .graph import CHGraph, PreparedCH, PreprocessStats
from .ordering import OrderingStrategy
from .preprocess import DEFAULT_MAX_HOPS, ProgressEvent, build_ch
from .query import ch_query, unpack_path

__all__ = [
    # graph types
    "CHGraph",
    "PreparedCH",
    "PreprocessStats",
    # preprocessing
    "build_ch",
    "OrderingStrategy",
    "ProgressEvent",
    "DEFAULT_MAX_HOPS",
    # querying
    "ch_query",
    "unpack_path",
    # adapters
    "CHIndex",
    "from_problem",
    "from_adjacency",
    "from_osm_problem",
    "dijkstra_on_ch_graph",
]
