"""
Concrete search problems.

``GridProblem`` and ``GraphProblem`` are dependency-free. ``OSMProblem`` needs
the optional ``osm`` extra (OSMnx), so import it directly from
:mod:`search_algorithms.problems.osm_problem` rather than from here.
"""

from .graph_problem import GraphProblem
from .grid_problem import GridProblem

__all__ = ["GridProblem", "GraphProblem"]
