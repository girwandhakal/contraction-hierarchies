"""
Synthetic road-like graphs, for measuring how CH scales.

Why synthetic graphs at all
---------------------------
The real Tuscaloosa network in this repo has 4,630 nodes. That is a genuine
road network and Experiment D validates against it, but it is far too small
to show what Contraction Hierarchies is actually known for: the speedup grows
with graph size, and the published 100-1000x figures come from continental
networks with millions of nodes. Downloading and caching a graph that size is
not reasonable for this repository, and it would not isolate the thing being
measured anyway.

So the scaling story uses generated graphs whose *topology* resembles a road
network, at sizes from a thousand to half a million nodes. What matters for CH
is not that the edges are real streets but that the graph has the structural
property road networks have: mostly local, near-planar connectivity with a
sparse set of long-range links. That structure is what gives a graph a small
"highway dimension", which is the formal property CH exploits. A uniformly
random graph (Erdős-Rényi) has no such structure, CH performs poorly on it,
and benchmarking against one would understate the technique for reasons that
have nothing to do with this implementation.

How the graphs are built
------------------------
1. Lay ``n`` nodes on a square lattice and jitter their positions, giving a
   planar street grid with uniform degree at most four.
2. Punch out a few edges, so the graph has dead ends like a real network
   rather than being perfectly regular.
3. Overlay "arterial" runs: straight lines of cheaper-per-distance edges
   spanning whole rows and columns. This is the layer that creates a
   *hierarchy* - without it there is no distinction between important and
   unimportant roads for CH to discover.

Edge costs are Euclidean distances, scaled down on arterials. That scaling
means raw straight-line distance is *not* admissible - see
:meth:`SyntheticGraph.heuristic_to`, which corrects for it so the A* baseline
is genuinely optimal rather than quietly returning slightly long routes.

Why not k-nearest-neighbours on random points
---------------------------------------------
That was the first approach here and it was a measurable mistake. A kNN graph
on uniform random points is not planar: "nearest neighbour" is not symmetric,
so asking for k=4 yields mean degree above five with a heavy tail of locally
dense nodes. Contraction considers O(degree^2) neighbour pairs, so those hubs
dominate preprocessing, accumulate shortcuts, and push it toward quadratic
time. The measurements then describe the generator rather than the algorithm.
A jittered lattice avoids all of that while modelling a street grid at least
as faithfully.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..ch.graph import CHGraph

#: Neighbours each node connects to in the local layer. Four gives a mean
#: undirected degree near real road networks (most intersections are 3-4 way).
DEFAULT_NEIGHBOURS = 4

#: Long-range edges as a fraction of node count. Road networks have very few
#: motorways relative to streets; 2% keeps the hierarchy realistic.
DEFAULT_ARTERIAL_FRACTION = 0.02

#: Cost of an arterial edge per unit of geometric length, relative to a local
#: street. Below 1.0 so arterials are worth routing onto - that preference is
#: what creates a hierarchy for CH to find.
#:
#: It also fixes the lower bound for an admissible heuristic: no route can
#: beat ``euclidean_distance * ARTERIAL_SPEED_FACTOR``. See
#: :meth:`SyntheticGraph.heuristic_to`.
ARTERIAL_SPEED_FACTOR = 0.55


@dataclass(frozen=True)
class SyntheticGraph:
    """
    A generated road-like graph plus the coordinates behind it.

    The coordinates are kept because the A* baseline needs a heuristic, and
    a straight-line distance heuristic requires knowing where nodes are.
    """

    graph: CHGraph
    xs: list[float]
    ys: list[float]

    @property
    def node_count(self) -> int:
        return self.graph.node_count

    @property
    def edge_count(self) -> int:
        return self.graph.edge_count

    def straight_line(self, a: int, b: int) -> float:
        """Raw Euclidean distance between two nodes."""
        return math.hypot(self.xs[a] - self.xs[b], self.ys[a] - self.ys[b])

    def heuristic_to(self, goal: int):
        """
        An admissible ``h(node)`` aimed at ``goal``, for the A* baseline.

        Straight-line distance alone is **not** admissible on these graphs,
        and assuming it was produced a measurably wrong A* baseline before
        this was corrected. Arterial edges cost ``ARTERIAL_SPEED_FACTOR``
        times their geometric length - that is the whole point of them, since
        a hierarchy needs roads that are cheap relative to distance - so a
        route covering Euclidean distance ``d`` can cost as little as
        ``d * ARTERIAL_SPEED_FACTOR``. Scaling the estimate by that factor
        makes it a true lower bound again.

        The same correction exists in real routers: when edge costs are
        travel *times*, the heuristic must divide distance by the maximum
        speed on the network, not by a typical speed.
        """
        gx, gy = self.xs[goal], self.ys[goal]
        xs, ys = self.xs, self.ys
        scale = ARTERIAL_SPEED_FACTOR

        def h(node: int) -> float:
            return math.hypot(xs[node] - gx, ys[node] - gy) * scale

        return h


def road_like_graph(
    node_count: int,
    neighbours: int = DEFAULT_NEIGHBOURS,
    arterial_fraction: float = DEFAULT_ARTERIAL_FRACTION,
    seed: int = 0,
) -> CHGraph:
    """Convenience wrapper returning just the graph."""
    return generate(
        node_count,
        neighbours=neighbours,
        arterial_fraction=arterial_fraction,
        seed=seed,
    ).graph


def generate(
    node_count: int,
    neighbours: int = DEFAULT_NEIGHBOURS,
    arterial_fraction: float = DEFAULT_ARTERIAL_FRACTION,
    seed: int = 0,
) -> SyntheticGraph:
    """
    Generate a connected, road-like graph with ``node_count`` nodes.

    Deterministic for a given ``(node_count, neighbours, arterial_fraction,
    seed)``, matching ``random_grid``'s convention in
    :mod:`~search_algorithms.benchmark.run_benchmark` so benchmark runs are
    reproducible.

    All edges are added in both directions: the local street layer of a real
    network is mostly bidirectional, and one-way streets would add asymmetry
    without changing what these graphs are measuring. ``OSMProblem`` covers
    the directed case in Experiment D.
    """
    if node_count < 2:
        raise ValueError("node_count must be at least 2.")
    if neighbours < 1:
        raise ValueError("neighbours must be at least 1.")

    rng = random.Random(seed)
    xs = [0.0] * node_count
    ys = [0.0] * node_count

    graph = CHGraph(node_count)

    # -- local layer: a perturbed grid -------------------------------------
    # Nodes are laid out on a rows x cols lattice and then jittered, rather
    # than scattered freely and joined by nearest-neighbour search.
    #
    # This matters more than it looks. A k-nearest-neighbour graph on uniform
    # random points is *not* planar: neighbour relations are not symmetric, so
    # asking for k=4 produces mean degree well above 4, with a heavy tail of
    # locally-dense nodes. Contraction then has to consider O(degree^2)
    # neighbour pairs at exactly those nodes, shortcuts pile onto them, and
    # preprocessing degrades toward quadratic - which measures the generator's
    # artefacts rather than CH.
    #
    # A perturbed lattice is genuinely planar, has uniform degree <= 4, and is
    # a fair model of a street grid. The jitter keeps edge costs from being
    # all-identical (which would make ties dominate and flatter the search).
    cols = max(2, int(math.isqrt(node_count)))
    rows = max(2, (node_count + cols - 1) // cols)

    # The lattice may overshoot; only the first `node_count` cells are used.
    def cell_index(row: int, col: int) -> int | None:
        index = row * cols + col
        return index if index < node_count else None

    jitter = 0.35 / cols
    for index in range(node_count):
        row, col = divmod(index, cols)
        xs[index] = (col + 0.5) / cols + rng.uniform(-jitter, jitter)
        ys[index] = (row + 0.5) / rows + rng.uniform(-jitter, jitter)

    for index in range(node_count):
        row, col = divmod(index, cols)

        # East and south only; the undirected helper supplies the reverse, so
        # every lattice edge is added exactly once.
        for drow, dcol in ((0, 1), (1, 0)):
            if dcol and col + 1 >= cols:
                continue
            other = cell_index(row + drow, col + dcol)
            if other is None:
                continue
            graph.add_undirected_edge(
                index,
                other,
                math.hypot(xs[index] - xs[other], ys[index] - ys[other]),
            )

    # A pure lattice has no dead ends, and real networks are full of them.
    # Removing a few edges also stops every shortest path from having many
    # equally cheap alternatives.
    removable = int(node_count * 0.05)
    for _ in range(removable):
        node = rng.randrange(node_count)
        outgoing = list(graph.successors(node))
        # Keep degree >= 2 so removal cannot disconnect the lattice.
        if len(outgoing) > 2:
            victim = outgoing[rng.randrange(len(outgoing))]
            if len(list(graph.successors(victim))) > 2:
                graph.remove_edge(node, victim)
                graph.remove_edge(victim, node)

    # -- arterial layer ----------------------------------------------------
    # Long-range links are what give a road network its hierarchy, but they
    # are *not* random pairs: a motorway connects places that are far apart
    # along one axis, and it runs roughly straight. Random long chords would
    # destroy the planarity the local layer just established and hand
    # contraction the same high-degree hubs the kNN layout did.
    #
    # So arterials are built as straight runs across the lattice: pick a row
    # (or column), then link every `stride`-th node along it with a
    # cheaper-per-distance edge.
    arterials = int(node_count * arterial_fraction)
    stride = 4
    built = 0
    attempts = 0
    while built < arterials and attempts < arterials * 8:
        attempts += 1
        if rng.random() < 0.5:
            row = rng.randrange(rows)
            line = [
                idx
                for col in range(0, cols, stride)
                if (idx := cell_index(row, col)) is not None
            ]
        else:
            col = rng.randrange(cols)
            line = [
                idx
                for row in range(0, rows, stride)
                if (idx := cell_index(row, col)) is not None
            ]

        for a, b in zip(line, line[1:]):
            distance = math.hypot(xs[a] - xs[b], ys[a] - ys[b])
            # Faster per unit distance than local streets: that speed
            # difference is what makes the arterial worth routing onto, and
            # therefore what creates a hierarchy for CH to exploit.
            graph.add_undirected_edge(a, b, distance * ARTERIAL_SPEED_FACTOR)
            built += 1

    return SyntheticGraph(graph=graph, xs=xs, ys=ys)
