"""
The search problem interface.

A naive weighted-grid implementation tends to hardwire two assumptions into
its search algorithms:

1. a state is always a ``(row, col)`` tuple, and
2. the cost of a move is a property of the *destination cell alone*
   (``step_cost(state)``).

Assumption 2 breaks on real road networks, where the cost of travelling
between two intersections belongs to the road connecting them, and where
A -> B need not cost the same as B -> A (one-way streets). So cost here is a
function of an **edge**: ``edge_cost(a, b)``.

Anything that satisfies this interface can be searched by every algorithm in
:mod:`search_algorithms.core.algorithms` without those algorithms knowing
whether they are walking a grid, an abstract graph, or the streets of
Tuscaloosa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, Iterable, TypeVar

State = TypeVar("State", bound=Hashable)


class Problem(ABC):
    """
    A weighted search problem over a graph of hashable states.

    Concrete subclasses must set ``start`` and ``goal`` and implement
    ``neighbors`` and ``edge_cost``. ``heuristic`` defaults to zero, which
    is admissible for any problem and makes A* degenerate to UCS - exactly
    the property explored in Question 3 of the original lab analysis.
    """

    start: State
    goal: State

    @abstractmethod
    def neighbors(self, state: State) -> list[State]:
        """Return the states reachable from ``state`` in one move."""
        raise NotImplementedError

    @abstractmethod
    def edge_cost(self, a: State, b: State) -> float:
        """
        Return the cost of traversing the edge from ``a`` to ``b``.

        ``b`` is assumed to be in ``neighbors(a)``. Implementations may raise
        if it is not.
        """
        raise NotImplementedError

    def is_goal(self, state: State) -> bool:
        """Return True if ``state`` satisfies the goal test."""
        return state == self.goal

    def heuristic(self, state: State) -> float:
        """
        Estimated cost from ``state`` to the goal.

        The default is the zero heuristic: admissible everywhere, and it
        turns A* into uniform-cost search. Subclasses override this with
        something informative (Manhattan distance on grids, great-circle
        distance on maps).
        """
        return 0.0

    def path_cost(self, path: Iterable[State]) -> float:
        """
        Total cost of traversing ``path``, summing the cost of each edge.

        An empty path costs infinity; a single-state path costs zero.
        """
        states = list(path)
        if not states:
            return float("inf")
        return float(
            sum(self.edge_cost(a, b) for a, b in zip(states, states[1:]))
        )

    def validate_path(self, path: Iterable[State]) -> tuple[bool, str]:
        """
        Check that ``path`` is a legal start-to-goal walk in this problem.

        Returns ``(is_valid, message)``. Useful as a test oracle: any
        algorithm's returned path should pass this.
        """
        states = list(path)

        if not states:
            return False, "Path is empty."
        if states[0] != self.start:
            return False, f"Path must start at {self.start!r}."
        if not self.is_goal(states[-1]):
            return False, f"Path must end at a goal state, not {states[-1]!r}."

        for current, nxt in zip(states, states[1:]):
            if nxt not in self.neighbors(current):
                return False, f"Invalid move from {current!r} to {nxt!r}."

        return True, "Path is valid."
