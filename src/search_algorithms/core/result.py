"""
Search result container shared by every algorithm in this package.

The core three fields (``path``, ``cost``, ``nodes_expanded``) are the
standard reporting contract for a search algorithm, kept fixed so results
stay directly comparable across problems and across algorithms.

Two more fields record the search itself, not just its answer:

``expansion_order``
    The states whose successors were generated, in the order they were
    expanded. Every algorithm already computed this implicitly; it is now
    recorded instead of discarded.

``parent_of``
    The parent-pointer map built during search. Combined with
    ``expansion_order`` it lets a client redraw the exact search tree the
    algorithm explored, not just the final path.

Both new fields default to empty, so any caller that only reads the original
three fields is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Hashable, TypeVar

State = TypeVar("State", bound=Hashable)


@dataclass
class SearchResult:
    """
    Standard result returned by every search algorithm.

    Attributes
    ----------
    path:
        Sequence of states from start through goal, including both endpoints.
        Empty if no solution exists.

    cost:
        Total traversal cost of ``path``. ``float("inf")`` if no solution
        exists.

    nodes_expanded:
        Number of states whose successors were generated.

        - The goal is NOT counted, because every algorithm here stops
          immediately when the goal is removed from the frontier.
        - Stale priority-queue entries discarded without generating
          successors are NOT counted.

        This is always equal to ``len(expansion_order)``.

    expansion_order:
        The expanded states in expansion order. Used to animate/replay a
        search; see module docstring.

    parent_of:
        Parent pointers discovered during the search (``state -> parent``).
        The start state maps to ``None``.

    elapsed_seconds:
        Wall-clock duration of the search, filled in by the algorithm.
    """

    path: list[State]
    cost: float
    nodes_expanded: int
    expansion_order: list[State] = field(default_factory=list)
    parent_of: dict[State, State | None] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def found(self) -> bool:
        """Return True if a solution path was found."""
        return bool(self.path)

    @property
    def steps(self) -> int:
        """Return the number of moves in the path."""
        return len(self.path) - 1 if self.path else 0

    @classmethod
    def failure(
        cls,
        nodes_expanded: int,
        expansion_order: list[State] | None = None,
        parent_of: dict[State, State | None] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> "SearchResult":
        """Convenience constructor for an unsuccessful search."""
        return cls(
            path=[],
            cost=inf,
            nodes_expanded=nodes_expanded,
            expansion_order=expansion_order if expansion_order is not None else [],
            parent_of=parent_of if parent_of is not None else {},
            elapsed_seconds=elapsed_seconds,
        )
