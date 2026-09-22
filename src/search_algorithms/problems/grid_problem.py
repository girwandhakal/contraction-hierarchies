"""
Weighted grid navigation, implemented against the generalized
:class:`Problem` interface.

Map format
----------
Whitespace-separated rectangular grid of tokens:

    S    start state
    G    goal state
    #    obstacle
    1-9  cost of entering that cell

Movement is four-directional in this fixed order: up, right, down, left.

Cost convention
---------------
Cost is defined per *cell entered*: entering ``G`` costs 1, entering a
numeric cell costs its value, and the start contributes 0. The generalized
interface asks for a cost per *edge*, so ``edge_cost(a, b)`` is defined as the
cost of entering ``b``. Those two formulations agree on every path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.heuristics import manhattan
from ..core.problem import Problem

State = tuple[int, int]


class GridProblem(Problem):
    """A four-neighbour weighted grid-navigation problem."""

    #: Move offsets in the fixed order: up, right, down, left.
    MOVES: tuple[tuple[int, int], ...] = (
        (-1, 0),  # up
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
    )

    def __init__(self, grid: list[list[str]]):
        if not grid or not grid[0]:
            raise ValueError("Grid must contain at least one cell.")

        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("Grid must be rectangular.")

        self.grid: tuple[tuple[str, ...], ...] = tuple(tuple(row) for row in grid)
        self.rows = len(self.grid)
        self.cols = width

        starts: list[State] = []
        goals: list[State] = []

        for r, row in enumerate(self.grid):
            for c, token in enumerate(row):
                if token == "S":
                    starts.append((r, c))
                elif token == "G":
                    goals.append((r, c))
                elif token == "#":
                    pass
                elif token.isdigit() and 1 <= int(token) <= 9:
                    pass
                else:
                    raise ValueError(
                        f"Invalid token {token!r} at row {r}, column {c}. "
                        "Use S, G, #, or an integer cost from 1 to 9."
                    )

        if len(starts) != 1:
            raise ValueError(f"Grid must contain exactly one S; found {len(starts)}.")
        if len(goals) != 1:
            raise ValueError(f"Grid must contain exactly one G; found {len(goals)}.")

        self.start: State = starts[0]
        self.goal: State = goals[0]

    @classmethod
    def from_file(cls, filename: str | Path) -> "GridProblem":
        """Load a whitespace-separated map file."""
        path = Path(filename)
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        return cls([line.split() for line in lines])

    @classmethod
    def from_string(cls, text: str) -> "GridProblem":
        """Build a problem from an inline map string - handy in tests."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return cls([line.split() for line in lines])

    def in_bounds(self, state: State) -> bool:
        """Return True if ``state`` lies inside the grid."""
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, state: State) -> bool:
        """Return True if ``state`` is in bounds and not an obstacle."""
        if not self.in_bounds(state):
            return False
        r, c = state
        return self.grid[r][c] != "#"

    def neighbors(self, state: State) -> list[State]:
        """
        Return passable neighbours in the fixed order up, right, down, left.

        The order matters: it determines tie-breaking, and therefore the exact
        node-expansion counts reported in the README's Results section.
        """
        r, c = state
        result: list[State] = []

        for dr, dc in self.MOVES:
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                result.append(nxt)

        return result

    def entry_cost(self, state: State) -> int:
        """
        Cost of entering ``state``: 0 for the start, 1 for the goal, and the
        displayed digit for numeric terrain.
        """
        if not self.in_bounds(state):
            raise ValueError(f"State {state} is outside the grid.")
        if not self.passable(state):
            raise ValueError(f"Obstacle state {state} has no traversal cost.")

        r, c = state
        token = self.grid[r][c]

        if token == "S":
            return 0
        if token == "G":
            return 1
        return int(token)

    def edge_cost(self, a: State, b: State) -> float:
        """Cost of moving from ``a`` into ``b``, i.e. the cost of entering ``b``."""
        return float(self.entry_cost(b))

    def heuristic(self, state: State) -> float:
        """Manhattan distance to the goal - admissible here since every step costs >= 1."""
        return manhattan(state, self.goal)

    def render(self, path: Iterable[State] | None = None) -> str:
        """
        Return a printable grid, marking intermediate path cells with ``*``.
        """
        path_states = set(path or [])
        rows: list[str] = []

        for r in range(self.rows):
            output_row: list[str] = []
            for c in range(self.cols):
                token = self.grid[r][c]
                if (r, c) in path_states and token not in {"S", "G"}:
                    output_row.append("*")
                else:
                    output_row.append(token)
            rows.append(" ".join(output_row))

        return "\n".join(rows)
