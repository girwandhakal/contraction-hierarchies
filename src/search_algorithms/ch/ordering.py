"""
Node importance: deciding *which* node to contract next.

This is the part of Contraction Hierarchies that actually determines whether
the technique works. Contracting nodes in a bad order still produces a
*correct* hierarchy - the query is correct for any total order - but it
produces a useless one, because the number of shortcuts explodes and the
augmented graph ends up denser than the original. Experiment E in the
analysis measures exactly that gap by running the same graph through a
deliberately naive order.

The heuristic implemented here is the standard teaching-grade one:

    priority(v) = edge_difference(v) + contracted_neighbours(v)

``edge_difference`` is the real signal. It simulates contracting ``v`` and
asks: how many shortcut edges would that add, minus how many edges would it
remove? A node whose neighbours are all mutually reachable without going
through it (a dead-end street, a node in the middle of a chain) has a
strongly negative edge difference - contracting it makes the graph smaller.
A motorway junction has a positive one, so it gets contracted late and ends
up high in the hierarchy, which is precisely the structure that makes queries
fast.

``contracted_neighbours`` is a uniformity term. Without it, contraction
happily eats one region of the graph completely before touching another,
which leaves a few survivors with enormous degree. Preferring nodes whose
neighbours are *not* yet contracted spreads the damage out.

What this deliberately does not do
----------------------------------
Production CH re-evaluates the priority of every *neighbour* of a contracted
node, since contracting ``v`` changes the local structure those priorities
were computed from. That is a documented simplification here (see the plan's
limitations section): this implementation only re-checks the priority of the
node it just popped, which catches the worst staleness for a fraction of the
cost. :func:`~search_algorithms.ch.preprocess.build_ch` reports how often
that re-check fired, so the cost of the simplification is measured rather
than assumed.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from .graph import CHGraph, Node


class OrderingStrategy(str, Enum):
    """
    Which node-ordering heuristic to contract with.

    ``EDGE_DIFFERENCE`` is the real one. The other two exist so Experiment E
    can quantify how much the ordering matters - they are baselines, not
    alternatives anyone should use.
    """

    #: edge difference + contracted-neighbour uniformity. The default.
    EDGE_DIFFERENCE = "edge_difference"

    #: Contract in ascending degree order, computed once and never updated.
    #: Plausible-looking and much worse, because it ignores whether
    #: contracting a node actually adds shortcuts.
    DEGREE = "degree"

    #: Contract in a fixed pseudo-random order. The control condition.
    RANDOM = "random"


def edge_difference(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
    count_shortcuts: Callable[[Node], int],
) -> int:
    """
    Shortcuts contracting ``v`` would add, minus edges it would remove.

    ``count_shortcuts`` is injected rather than computed here because
    simulating a contraction requires the witness search, which lives in
    :mod:`~search_algorithms.ch.preprocess` alongside the contraction itself.
    Keeping the arithmetic here and the graph search there is what stops this
    module from needing to know how witness searches work.
    """
    if contracted[v]:
        return 0

    removed = live_degree(graph, v, contracted)
    return count_shortcuts(v) - removed


def contracted_neighbour_count(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
) -> int:
    """How many of ``v``'s distinct neighbours are already contracted."""
    seen: set[Node] = set()
    for w in graph.successors(v):
        if contracted[w]:
            seen.add(w)
    for u in graph.predecessors(v):
        if contracted[u]:
            seen.add(u)
    return len(seen)


def live_degree(graph: CHGraph, v: Node, contracted: list[bool]) -> int:
    """
    Degree of ``v`` counting only edges to nodes that are still live.

    Edges to already-contracted nodes are not "removed" by contracting ``v``
    - they were already taken out of the working graph - so counting them
    would overstate the benefit of contracting ``v``.
    """
    total = 0
    for w in graph.successors(v):
        if not contracted[w]:
            total += 1
    for u in graph.predecessors(v):
        if not contracted[u]:
            total += 1
    return total


def priority(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
    count_shortcuts: Callable[[Node], int],
) -> int:
    """
    The composite importance score. Lower means "contract sooner".

    This is ``edge_difference(v) + contracted_neighbour_count(v)``: the
    structural cost of contracting ``v`` plus a penalty for contracting in an
    already-hollowed-out neighbourhood.
    """
    return edge_difference(graph, v, contracted, count_shortcuts) + (
        contracted_neighbour_count(graph, v, contracted)
    )


def static_order(
    graph: CHGraph,
    strategy: OrderingStrategy,
    seed: int = 0,
) -> list[Node]:
    """
    Produce a complete contraction order up front, without any simulation.

    Only meaningful for the baseline strategies in Experiment E;
    ``EDGE_DIFFERENCE`` cannot be precomputed this way because its priorities
    depend on contractions that have not happened yet.
    """
    nodes = list(graph.nodes())

    if strategy is OrderingStrategy.DEGREE:
        # Stable: ties broken by node id so the run is reproducible.
        return sorted(nodes, key=lambda v: (graph.degree(v), v))

    if strategy is OrderingStrategy.RANDOM:
        import random

        rng = random.Random(seed)
        rng.shuffle(nodes)
        return nodes

    raise ValueError(
        f"{strategy} is computed during contraction, not up front; "
        "call build_ch() instead."
    )
