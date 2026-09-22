"""
A search problem over an arbitrary weighted graph.

This class exists to prove the abstraction actually generalizes: the very same
``bfs``/``ucs``/``astar`` functions that solve grid mazes also solve a plain
adjacency list, with no changes to the algorithms.

It also underpins the OSM routing problem, which is just a very large weighted
graph with geographic coordinates attached to its nodes.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, Mapping, Sequence

from ..core.problem import Problem

Node = Hashable

#: An adjacency mapping: ``{node: [(neighbour, cost), ...]}``.
AdjacencyMap = Mapping[Node, Sequence[tuple[Node, float]]]


class GraphProblem(Problem):
    """
    Search over an explicit adjacency map.

    Parameters
    ----------
    adjacency:
        ``{node: [(neighbour, cost), ...]}``. Costs must be non-negative.
        The mapping is taken as directed; use :meth:`from_undirected` to build
        a symmetric graph from one-way declarations.
    start, goal:
        Endpoints of the search.
    heuristic:
        Optional ``h(node) -> float`` estimate of remaining cost. Defaults to
        the zero heuristic, which makes A* behave exactly like UCS.
    """

    def __init__(
        self,
        adjacency: AdjacencyMap,
        start: Node,
        goal: Node,
        heuristic: Callable[[Node], float] | None = None,
    ):
        if start not in adjacency:
            raise ValueError(f"Start node {start!r} is not in the graph.")
        if goal not in adjacency:
            raise ValueError(f"Goal node {goal!r} is not in the graph.")

        # Store as a dict of dicts so edge lookup is O(1) and parallel edges
        # collapse to the cheapest option.
        self._adjacency: dict[Node, dict[Node, float]] = {}
        for node, edges in adjacency.items():
            collapsed: dict[Node, float] = {}
            for neighbor, cost in edges:
                if cost < 0:
                    raise ValueError(
                        f"Negative edge cost {cost} on {node!r} -> {neighbor!r}; "
                        "these algorithms require non-negative costs."
                    )
                if neighbor not in collapsed or cost < collapsed[neighbor]:
                    collapsed[neighbor] = float(cost)
            self._adjacency[node] = collapsed

        # Guarantee every referenced node is a key, so neighbors() never fails.
        for edges in list(self._adjacency.values()):
            for neighbor in edges:
                self._adjacency.setdefault(neighbor, {})

        self.start: Node = start
        self.goal: Node = goal
        self._heuristic = heuristic

    @classmethod
    def from_undirected(
        cls,
        edges: Iterable[tuple[Node, Node, float]],
        start: Node,
        goal: Node,
        heuristic: Callable[[Node], float] | None = None,
    ) -> "GraphProblem":
        """Build a problem from ``(a, b, cost)`` triples, traversable both ways."""
        adjacency: dict[Node, list[tuple[Node, float]]] = {}
        for a, b, cost in edges:
            adjacency.setdefault(a, []).append((b, cost))
            adjacency.setdefault(b, []).append((a, cost))
        return cls(adjacency, start=start, goal=goal, heuristic=heuristic)

    def neighbors(self, state: Node) -> list[Node]:
        return list(self._adjacency.get(state, {}))

    def edge_cost(self, a: Node, b: Node) -> float:
        try:
            return self._adjacency[a][b]
        except KeyError:
            raise ValueError(f"No edge from {a!r} to {b!r}.") from None

    def heuristic(self, state: Node) -> float:
        if self._heuristic is None:
            return 0.0
        return self._heuristic(state)

    @property
    def node_count(self) -> int:
        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adjacency.values())
