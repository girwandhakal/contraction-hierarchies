"""
Graph search algorithms.

BFS, UCS and A* (plus weighted A* and greedy best-first) implemented against
any :class:`~search_algorithms.core.problem.Problem`, not just a weighted
grid. Cost comes from ``problem.edge_cost(a, b)`` and the heuristic from
``problem.heuristic(state)``, so the same functions run unmodified on grids,
plain graphs and real street networks. Each algorithm also records the order
in which it expanded states, so a client can replay the search.

Node-expansion accounting follows a fixed convention:

- a state counts as expanded when its successors are generated;
- the goal is not counted, because search stops the moment the goal is
  removed from the frontier;
- stale priority-queue entries are skipped without being counted.

That convention is what makes the node-expansion counts in the README's
Results section directly comparable across algorithms and across problems.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from itertools import count
from typing import Callable, Hashable, TypeVar

from .problem import Problem
from .result import SearchResult

State = TypeVar("State", bound=Hashable)


def reconstruct_path(
    parent_of: dict[State, State | None],
    goal: State,
) -> list[State]:
    """
    Rebuild a path by walking parent pointers back from ``goal``.

    The returned path includes both the start state and ``goal``.
    """
    path: list[State] = []
    current: State | None = goal

    while current is not None:
        path.append(current)
        current = parent_of[current]

    path.reverse()
    return path


def bfs(problem: Problem) -> SearchResult:
    """
    Breadth-first search.

    Explores by increasing number of moves, ignoring edge costs entirely when
    choosing what to expand next. On a weighted problem it therefore returns
    the path with the fewest *steps*, which is often not the cheapest path -
    see the README's Results section (``maps/map2.txt``) for a concrete
    example.
    """
    started = time.perf_counter()

    frontier: deque[State] = deque([problem.start])
    # The parent map doubles as the discovered/visited set.
    parent_of: dict[State, State | None] = {problem.start: None}
    expansion_order: list[State] = []

    while frontier:
        current = frontier.popleft()

        if problem.is_goal(current):
            path = reconstruct_path(parent_of, current)
            return SearchResult(
                path=path,
                cost=problem.path_cost(path),
                nodes_expanded=len(expansion_order),
                expansion_order=expansion_order,
                parent_of=parent_of,
                elapsed_seconds=time.perf_counter() - started,
            )

        expansion_order.append(current)

        for neighbor in problem.neighbors(current):
            if neighbor not in parent_of:
                parent_of[neighbor] = current
                frontier.append(neighbor)

    return SearchResult.failure(
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        parent_of=parent_of,
        elapsed_seconds=time.perf_counter() - started,
    )


def _best_first(
    problem: Problem,
    priority: Callable[[float, State], float],
) -> SearchResult:
    """
    Shared best-first engine behind UCS, A*, weighted A* and greedy search.

    ``priority(g, state)`` maps a state's cost-so-far to its queue priority:

    - UCS:            ``g``
    - A*:             ``g + h(state)``
    - Weighted A*:    ``g + w * h(state)``
    - Greedy best-first: ``h(state)``

    Ties are broken by insertion order via a monotonic counter, so the search
    is deterministic. Stale entries - queue entries superseded by a cheaper
    path found later - are detected by comparing the entry's recorded ``g``
    against the best known ``g`` and skipped without counting as expansions.
    """
    started = time.perf_counter()

    counter = count()
    start = problem.start

    best_g: dict[State, float] = {start: 0.0}
    parent_of: dict[State, State | None] = {start: None}
    expansion_order: list[State] = []

    # Entries are (priority, tie_breaker, g_at_push, state).
    frontier: list[tuple[float, int, float, State]] = [
        (priority(0.0, start), next(counter), 0.0, start)
    ]

    while frontier:
        _, _, g_at_push, current = heapq.heappop(frontier)

        # Skip stale entries: a cheaper route to `current` was found after
        # this entry was pushed.
        if g_at_push > best_g[current]:
            continue

        if problem.is_goal(current):
            path = reconstruct_path(parent_of, current)
            return SearchResult(
                path=path,
                cost=best_g[current],
                nodes_expanded=len(expansion_order),
                expansion_order=expansion_order,
                parent_of=parent_of,
                elapsed_seconds=time.perf_counter() - started,
            )

        expansion_order.append(current)

        for neighbor in problem.neighbors(current):
            tentative_g = best_g[current] + problem.edge_cost(current, neighbor)

            if neighbor not in best_g or tentative_g < best_g[neighbor]:
                best_g[neighbor] = tentative_g
                parent_of[neighbor] = current
                heapq.heappush(
                    frontier,
                    (
                        priority(tentative_g, neighbor),
                        next(counter),
                        tentative_g,
                        neighbor,
                    ),
                )

    return SearchResult.failure(
        nodes_expanded=len(expansion_order),
        expansion_order=expansion_order,
        parent_of=parent_of,
        elapsed_seconds=time.perf_counter() - started,
    )


def ucs(problem: Problem) -> SearchResult:
    """
    Uniform-cost search: expand the cheapest-known state first.

    Optimal for non-negative edge costs, and it ignores the heuristic
    entirely, so it explores outward in every direction equally.
    """
    return _best_first(problem, lambda g, state: g)


def astar(problem: Problem) -> SearchResult:
    """
    A* search, using ``problem.heuristic`` for h(n).

    Priority is f(n) = g(n) + h(n). With an admissible heuristic this is
    optimal, and it expands no more states than UCS - usually far fewer,
    because h(n) pulls the search toward the goal.
    """
    return _best_first(problem, lambda g, state: g + problem.heuristic(state))


def weighted_astar(problem: Problem, weight: float = 1.5) -> SearchResult:
    """
    Weighted A*: f(n) = g(n) + ``weight`` * h(n).

    With ``weight > 1`` the heuristic may overestimate, so the result is no
    longer guaranteed optimal - but the search is typically much faster. The
    returned cost is bounded by ``weight`` times the optimal cost: see the
    README's "Follow-up questions" section for the bounded-suboptimality
    discussion.
    """
    if weight < 1.0:
        raise ValueError("weight must be >= 1.0")
    return _best_first(problem, lambda g, state: g + weight * problem.heuristic(state))


def greedy_best_first(problem: Problem) -> SearchResult:
    """
    Greedy best-first search: f(n) = h(n), ignoring cost-so-far.

    Fast and heavily goal-directed, but not optimal: it will happily commit to
    an expensive route that merely points the right way.
    """
    return _best_first(problem, lambda g, state: problem.heuristic(state))


#: Registry used by the CLIs and the benchmark harness so they all expose
#: exactly the same set of algorithms under the same names.
ALGORITHMS: dict[str, Callable[[Problem], SearchResult]] = {
    "bfs": bfs,
    "ucs": ucs,
    "astar": astar,
    "greedy": greedy_best_first,
    "weighted_astar": weighted_astar,
}

#: The three textbook algorithms, compared side by side throughout the
#: README's Results section.
CLASSIC_ALGORITHMS: tuple[str, ...] = ("bfs", "ucs", "astar")
