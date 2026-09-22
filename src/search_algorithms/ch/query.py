"""
Querying a contracted hierarchy.

The query runs two Dijkstra searches at once - forward from the start,
backward from the goal - and both only ever move *up* the hierarchy. The
forward search relaxes ``u -> v`` only when ``rank[v] > rank[u]``; the
backward search walks incoming edges under the same rule.

Why that is correct
-------------------
Contraction maintains this invariant: for every pair ``(s, t)``, the
augmented graph contains a shortest ``s``-``t`` path that is *bitonic* in
rank - it climbs to a single highest-ranked node (the "peak") and then
descends. The shortcuts are exactly what guarantee this: whenever
contracting ``v`` would have broken a shortest path ``u -> v -> w``, a
shortcut ``u -> w`` was inserted to carry that distance at a higher level.

So searching upward from both ends is not a heuristic restriction - it is
enough to find the peak, and the peak splits the optimal path into the two
halves the two searches find. This is why CH can ignore the vast majority of
the graph and still return *exactly* the same distance as Dijkstra, which the
test suite verifies rather than takes on faith.

Why both searches must run to completion-ish
--------------------------------------------
The first node settled in both directions is *not* necessarily on a shortest
path - the peak may be settled later, via a cheaper combination. So the
search keeps a running best meeting cost and only stops a direction once its
frontier minimum exceeds that best. Stopping at the first meeting is the most
common CH bug, and it produces paths that are slightly too long on a small
fraction of queries - the kind of error that hand-picked test cases miss and
randomised cross-checks against Dijkstra catch.
"""

from __future__ import annotations

import heapq
import time

from ..core.result import SearchResult
from .graph import Node, PreparedCH

#: Relative tolerance when comparing a CH distance against Dijkstra's. Both
#: sum the same floats, but in different orders, so exact equality is not a
#: reasonable expectation on long paths.
COST_TOLERANCE = 1e-9


def ch_query(
    prepared: PreparedCH,
    start: Node,
    goal: Node,
    unpack: bool = True,
) -> SearchResult:
    """
    Shortest path from ``start`` to ``goal`` over a prepared hierarchy.

    Parameters
    ----------
    prepared:
        Output of :func:`~search_algorithms.ch.preprocess.build_ch`.
    start, goal:
        Node ids in the prepared graph.
    unpack:
        When ``True`` the returned ``path`` is a full sequence of original
        nodes, with every shortcut expanded. When ``False`` the path is left
        as the hierarchy-level node sequence, which is cheaper and is what the
        timing benchmarks use when they only need the distance. The reported
        ``cost`` is identical either way.

    Returns
    -------
    A :class:`~search_algorithms.core.result.SearchResult`.
    ``nodes_expanded`` counts nodes settled across *both* directions, matching
    the convention in :mod:`search_algorithms.core.algorithms`: a node counts
    once its outgoing edges are relaxed, and the stale queue entries skipped
    without relaxation are not counted.
    """
    started = time.perf_counter()

    node_count = prepared.node_count
    if not 0 <= start < node_count:
        raise ValueError(f"Start node {start} out of range.")
    if not 0 <= goal < node_count:
        raise ValueError(f"Goal node {goal} out of range.")

    if start == goal:
        return SearchResult(
            path=[start],
            cost=0.0,
            nodes_expanded=0,
            elapsed_seconds=time.perf_counter() - started,
        )

    inf = float("inf")
    forward_dist: dict[Node, float] = {start: 0.0}
    backward_dist: dict[Node, float] = {goal: 0.0}
    forward_parent: dict[Node, Node | None] = {start: None}
    backward_parent: dict[Node, Node | None] = {goal: None}

    # Entries are (cost, node). No tie-breaker counter: node ids are
    # integers, so a tuple comparison falls through to comparing them and
    # never raises. Dropping the counter removes an itertools.count() call
    # per push, which is measurable on a search this short.
    forward_heap: list[tuple[float, Node]] = [(0.0, start)]
    backward_heap: list[tuple[float, Node]] = [(0.0, goal)]

    # Bound method lookups hoisted out of the loop: on a CH query the loop
    # body runs only a few hundred times, so interpreter overhead per
    # iteration is a large share of total query time - which is precisely
    # what the benchmark is measuring.
    heappush = heapq.heappush
    heappop = heapq.heappop
    # Pre-split upward adjacency, so the loop never examines a downward edge
    # it would only discard. See PreparedCH.finalize().
    up_forward = prepared.up_forward
    up_backward = prepared.up_backward

    best_cost = inf
    meeting: Node | None = None
    expanded = 0

    while forward_heap or backward_heap:
        # Interleave the two directions, always advancing whichever frontier
        # is currently cheaper. Balancing this way keeps the searched volume
        # small; alternating blindly would over-expand one side.
        forward_min = forward_heap[0][0] if forward_heap else inf
        backward_min = backward_heap[0][0] if backward_heap else inf

        # Neither direction can improve on the best meeting found: every
        # remaining path costs at least this frontier minimum, and any
        # complete path is the sum of two non-negative halves.
        if (forward_min if forward_min < backward_min else backward_min) >= best_cost:
            break

        if forward_min <= backward_min:
            cost, node = heappop(forward_heap)
            if cost > forward_dist.get(node, inf):
                continue
            expanded += 1

            other = backward_dist.get(node)
            if other is not None:
                total = cost + other
                if total < best_cost:
                    best_cost = total
                    meeting = node

            for nxt, edge_cost in up_forward[node]:
                nxt_cost = cost + edge_cost
                if nxt_cost < forward_dist.get(nxt, inf):
                    forward_dist[nxt] = nxt_cost
                    forward_parent[nxt] = node
                    heappush(forward_heap, (nxt_cost, nxt))
        else:
            cost, node = heappop(backward_heap)
            if cost > backward_dist.get(node, inf):
                continue
            expanded += 1

            other = forward_dist.get(node)
            if other is not None:
                total = cost + other
                if total < best_cost:
                    best_cost = total
                    meeting = node

            for prev, edge_cost in up_backward[node]:
                prev_cost = cost + edge_cost
                if prev_cost < backward_dist.get(prev, inf):
                    backward_dist[prev] = prev_cost
                    backward_parent[prev] = node
                    heappush(backward_heap, (prev_cost, prev))

    elapsed = time.perf_counter() - started

    if meeting is None or best_cost == inf:
        return SearchResult.failure(
            nodes_expanded=expanded,
            elapsed_seconds=elapsed,
        )

    path = _assemble_path(
        prepared,
        meeting,
        forward_parent,
        backward_parent,
        unpack=unpack,
    )

    return SearchResult(
        path=path,
        cost=best_cost,
        nodes_expanded=expanded,
        elapsed_seconds=elapsed,
    )


def _assemble_path(
    prepared: PreparedCH,
    meeting: Node,
    forward_parent: dict[Node, Node | None],
    backward_parent: dict[Node, Node | None],
    unpack: bool,
) -> list[Node]:
    """
    Stitch the two search trees together at ``meeting`` into one path.

    The forward tree gives ``start -> meeting`` by walking parents backward;
    the backward tree gives ``meeting -> goal`` by walking parents forward
    (its "parent" pointers already point toward the goal, because the
    backward search traversed incoming edges).
    """
    upward: list[Node] = []
    node: Node | None = meeting
    while node is not None:
        upward.append(node)
        node = forward_parent[node]
    upward.reverse()

    downward: list[Node] = []
    node = backward_parent[meeting]
    while node is not None:
        downward.append(node)
        node = backward_parent[node]

    hierarchy_path = upward + downward

    if not unpack:
        return hierarchy_path

    return unpack_path(prepared, hierarchy_path)


def unpack_path(prepared: PreparedCH, hierarchy_path: list[Node]) -> list[Node]:
    """
    Expand every shortcut in ``hierarchy_path`` into original edges.

    A shortcut ``u -> w`` stands for ``u -> via -> w``, and each of those
    halves may itself be a shortcut, so expansion recurses until only
    original edges remain. The result is a path that exists in the *input*
    graph, which is what makes it comparable to Dijkstra's output and
    checkable by ``Problem.validate_path``.

    Implemented with an explicit stack rather than recursion: shortcut chains
    on a large graph get deep enough to exceed Python's recursion limit.
    """
    if len(hierarchy_path) < 2:
        return list(hierarchy_path)

    shortcuts = prepared.graph.shortcuts
    result: list[Node] = [hierarchy_path[0]]

    for a, b in zip(hierarchy_path, hierarchy_path[1:]):
        # Stack of edges still to expand, in reverse order so the leftmost
        # edge is processed first.
        pending: list[tuple[Node, Node]] = [(a, b)]

        while pending:
            u, w = pending.pop()
            entry = shortcuts.get((u, w))

            if entry is None:
                # An original edge: nothing to expand.
                result.append(w)
                continue

            via, cost = entry

            # A shortcut is only recorded while it is the cheapest u -> w
            # edge, but a *later* contraction can add a cheaper real edge on
            # top of it. If the graph's current cost is cheaper than the
            # recorded shortcut, the edge being traversed is not this
            # shortcut, so expanding it would produce a path that is valid
            # but longer than the reported cost.
            if prepared.graph.edge_cost(u, w) < cost - COST_TOLERANCE:
                result.append(w)
                continue

            pending.append((via, w))
            pending.append((u, via))

    return result
