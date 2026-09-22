"""
Turning a :class:`~search_algorithms.core.problem.Problem` into a CH graph.

The CH layer works on dense integer node ids (``0 .. n-1``) because
preprocessing indexes flat arrays by node - ``rank``, ``contracted``, the
witness searcher's scratch buffers. Real problems have other node types: OSM
ids are sparse 64-bit integers, grid states are ``(row, col)`` tuples, and
``GraphProblem`` accepts any hashable.

:class:`CHIndex` owns that translation. It holds the dense graph plus the
mapping in both directions, so a query can be issued with original node
labels and return a path in original node labels, with the integer ids never
leaking into calling code.

Enumerating the graph
---------------------
``Problem`` only offers ``neighbors(state)``, so the full node set is
discovered by traversal from a seed set. For ``GridProblem`` and
``GraphProblem`` that is exact. For ``OSMProblem`` the underlying networkx
graph is available and is used directly, which is both faster and guaranteed
complete - a traversal from one node would silently miss any component that
node cannot reach, and a city street graph downloaded by bounding box
routinely has small disconnected fragments.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, Mapping, Sequence

from ..core.problem import Problem
from .graph import CHGraph, Node
from .ordering import OrderingStrategy
from .preprocess import (
    DEFAULT_CORE_FRACTION,
    DEFAULT_MAX_HOPS,
    ProgressCallback,
    build_ch,
)
from .query import ch_query
from ..core.result import SearchResult


class CHIndex:
    """
    A prepared hierarchy plus the mapping back to original node labels.

    Build one with :func:`from_problem`, :func:`from_adjacency` or
    :func:`from_osm_problem`, then call :meth:`query` with original labels.
    """

    __slots__ = ("prepared", "_to_dense", "_to_original")

    def __init__(
        self,
        prepared,
        to_dense: Mapping[Hashable, Node],
        to_original: Sequence[Hashable],
    ):
        self.prepared = prepared
        self._to_dense = dict(to_dense)
        self._to_original = list(to_original)

    # -- label translation -------------------------------------------------

    def dense_id(self, label: Hashable) -> Node:
        """Dense id for an original node label."""
        try:
            return self._to_dense[label]
        except KeyError:
            raise ValueError(f"Node {label!r} is not in the indexed graph.") from None

    def original_label(self, node: Node) -> Hashable:
        """Original node label for a dense id."""
        return self._to_original[node]

    def __contains__(self, label: Hashable) -> bool:
        return label in self._to_dense

    @property
    def node_count(self) -> int:
        return self.prepared.node_count

    @property
    def labels(self) -> Sequence[Hashable]:
        """All original node labels, indexed by dense id."""
        return self._to_original

    # -- querying ----------------------------------------------------------

    def query(
        self,
        start: Hashable,
        goal: Hashable,
        unpack: bool = True,
    ) -> SearchResult:
        """
        Shortest path between two *original* node labels.

        The returned ``SearchResult.path`` is in original labels, so it can be
        handed straight to ``Problem.validate_path`` or ``Problem.path_cost``.
        """
        result = ch_query(
            self.prepared,
            self.dense_id(start),
            self.dense_id(goal),
            unpack=unpack,
        )

        if not result.path:
            return result

        return SearchResult(
            path=[self._to_original[node] for node in result.path],
            cost=result.cost,
            nodes_expanded=result.nodes_expanded,
            expansion_order=[],
            parent_of={},
            elapsed_seconds=result.elapsed_seconds,
        )


def _discover_nodes(
    problem: Problem,
    seeds: Iterable[Hashable],
) -> list[Hashable]:
    """
    Every state reachable from ``seeds`` via ``problem.neighbors``.

    Depth-first with an explicit stack. Insertion order is preserved so dense
    ids are deterministic for a given problem, which keeps benchmark runs
    reproducible.
    """
    discovered: dict[Hashable, None] = {}
    stack = list(seeds)

    for seed in stack:
        discovered.setdefault(seed, None)

    while stack:
        state = stack.pop()
        for neighbor in problem.neighbors(state):
            if neighbor not in discovered:
                discovered[neighbor] = None
                stack.append(neighbor)

    return list(discovered)


def from_problem(
    problem: Problem,
    strategy: OrderingStrategy = OrderingStrategy.EDGE_DIFFERENCE,
    max_hops: int = DEFAULT_MAX_HOPS,
    on_progress: ProgressCallback | None = None,
    extra_seeds: Iterable[Hashable] = (),
    seed: int = 0,
    core_fraction: float = DEFAULT_CORE_FRACTION,
) -> CHIndex:
    """
    Build a hierarchy from any :class:`Problem`.

    Nodes are discovered by traversing from ``problem.start``,
    ``problem.goal`` and any ``extra_seeds``. Anything unreachable from those
    is not in the index - which is correct for answering queries between
    them, and is why :func:`from_osm_problem` takes the whole networkx graph
    instead of traversing.
    """
    seeds = [problem.start, problem.goal, *extra_seeds]
    labels = _discover_nodes(problem, seeds)
    to_dense = {label: index for index, label in enumerate(labels)}

    graph = CHGraph(len(labels))
    for label in labels:
        u = to_dense[label]
        for neighbor in problem.neighbors(label):
            v = to_dense.get(neighbor)
            if v is None:  # pragma: no cover - discovery is exhaustive
                continue
            graph.add_edge(u, v, problem.edge_cost(label, neighbor))

    prepared = build_ch(
        graph,
        strategy=strategy,
        max_hops=max_hops,
        on_progress=on_progress,
        seed=seed,
        core_fraction=core_fraction,
    )
    return CHIndex(prepared, to_dense, labels)


def from_adjacency(
    adjacency: Mapping[Hashable, Iterable[tuple[Hashable, float]]],
    strategy: OrderingStrategy = OrderingStrategy.EDGE_DIFFERENCE,
    max_hops: int = DEFAULT_MAX_HOPS,
    on_progress: ProgressCallback | None = None,
    seed: int = 0,
    core_fraction: float = DEFAULT_CORE_FRACTION,
) -> CHIndex:
    """
    Build a hierarchy from ``{node: [(neighbour, cost), ...]}``.

    The path the synthetic benchmarks take: no ``Problem`` instance needed,
    and the adjacency map is already the shape the generator produces.
    """
    labels: dict[Hashable, None] = {}
    for node, edges in adjacency.items():
        labels.setdefault(node, None)
        for neighbor, _ in edges:
            labels.setdefault(neighbor, None)

    ordered = list(labels)
    to_dense = {label: index for index, label in enumerate(ordered)}

    graph = CHGraph(len(ordered))
    for node, edges in adjacency.items():
        u = to_dense[node]
        for neighbor, cost in edges:
            graph.add_edge(u, to_dense[neighbor], cost)

    prepared = build_ch(
        graph,
        strategy=strategy,
        max_hops=max_hops,
        on_progress=on_progress,
        seed=seed,
        core_fraction=core_fraction,
    )
    return CHIndex(prepared, to_dense, ordered)


def from_osm_problem(
    problem,
    strategy: OrderingStrategy = OrderingStrategy.EDGE_DIFFERENCE,
    max_hops: int = DEFAULT_MAX_HOPS,
    on_progress: ProgressCallback | None = None,
    seed: int = 0,
    core_fraction: float = DEFAULT_CORE_FRACTION,
) -> CHIndex:
    """
    Build a hierarchy from an :class:`OSMProblem`, using its whole graph.

    Unlike :func:`from_problem` this does not traverse from the start node,
    so disconnected fragments of the street network are still indexed. Edge
    costs and one-way restrictions come from the problem itself, so blocked
    edges and the multigraph's cheapest-parallel-edge rule are honoured
    exactly as a plain A* run would see them.
    """
    graph_nx = problem.graph
    labels = [int(node) for node in graph_nx.nodes]
    to_dense = {label: index for index, label in enumerate(labels)}

    graph = CHGraph(len(labels))
    for label in labels:
        u = to_dense[label]
        for neighbor in problem.neighbors(label):
            v = to_dense.get(int(neighbor))
            if v is None:  # pragma: no cover - neighbours are always nodes
                continue
            graph.add_edge(u, v, problem.edge_cost(label, neighbor))

    prepared = build_ch(
        graph,
        strategy=strategy,
        max_hops=max_hops,
        on_progress=on_progress,
        seed=seed,
        core_fraction=core_fraction,
    )
    return CHIndex(prepared, to_dense, labels)


#: Reference Dijkstra used by the benchmarks and correctness tests as the
#: ground truth CH is checked against. It runs on the *dense* CH graph rather
#: than through ``Problem``, so a CH-vs-Dijkstra timing comparison measures
#: the algorithms rather than the cost of two different graph representations.
def dijkstra_on_ch_graph(
    graph: CHGraph,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node], float] | None = None,
) -> SearchResult:
    """
    Plain Dijkstra (or A*, with ``heuristic``) over a :class:`CHGraph`.

    Only original edges are present when this is called on the pre-contraction
    graph; call it on a contracted graph and it will happily use shortcuts too,
    which is not what the baseline wants - so the benchmarks keep an
    uncontracted copy for this.

    Written in the same style as
    :func:`~search_algorithms.ch.query.ch_query` - same heap discipline, same
    hoisted lookups, ``expansion_order`` not recorded - so a timing comparison
    between them reflects the algorithms rather than how carefully each loop
    was written. That symmetry is a benchmarking requirement, not a
    micro-optimisation: an unoptimised baseline would inflate CH's speedup.
    """
    import heapq
    import time

    started = time.perf_counter()
    inf = float("inf")

    dist: dict[Node, float] = {start: 0.0}
    parent: dict[Node, Node | None] = {start: None}
    frontier: list[tuple[float, Node]] = [(0.0, start)]
    expanded = 0

    heappush = heapq.heappush
    heappop = heapq.heappop
    successors = graph.successors

    while frontier:
        priority, node = heappop(frontier)
        node_dist = dist.get(node, inf)

        # Recover g from f when running as A*.
        if heuristic is not None:
            if priority - heuristic(node) > node_dist + 1e-12:
                continue
        elif priority > node_dist:
            continue

        if node == goal:
            path: list[Node] = []
            cursor: Node | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return SearchResult(
                path=path,
                cost=node_dist,
                nodes_expanded=expanded,
                parent_of=parent,
                elapsed_seconds=time.perf_counter() - started,
            )

        expanded += 1

        for nxt, edge_cost in successors(node).items():
            nxt_dist = node_dist + edge_cost
            if nxt_dist < dist.get(nxt, inf):
                dist[nxt] = nxt_dist
                parent[nxt] = node
                key = nxt_dist if heuristic is None else nxt_dist + heuristic(nxt)
                heappush(frontier, (key, nxt))

    return SearchResult.failure(
        nodes_expanded=expanded,
        parent_of=parent,
        elapsed_seconds=time.perf_counter() - started,
    )
