"""Core search machinery: the problem interface, algorithms and heuristics."""

from .algorithms import (
    ALGORITHMS,
    CLASSIC_ALGORITHMS,
    astar,
    bfs,
    greedy_best_first,
    reconstruct_path,
    ucs,
    weighted_astar,
)
from .heuristics import chebyshev, euclidean, haversine, manhattan
from .problem import Problem
from .result import SearchResult

__all__ = [
    "Problem",
    "SearchResult",
    "bfs",
    "ucs",
    "astar",
    "weighted_astar",
    "greedy_best_first",
    "reconstruct_path",
    "ALGORITHMS",
    "CLASSIC_ALGORITHMS",
    "manhattan",
    "euclidean",
    "chebyshev",
    "haversine",
]
