"""
The graph representation Contraction Hierarchies preprocesses and queries.

Why this is not a :class:`~search_algorithms.core.problem.Problem`
-----------------------------------------------------------------
Every algorithm in :mod:`search_algorithms.core.algorithms` asks a
``Problem`` for ``neighbors(state)`` and ``edge_cost(a, b)`` on demand. That
interface is exactly right for searching a world you cannot precompute -
a grid, a live street graph, a state space generated as you walk it.

CH is the opposite situation. It *rewrites* the graph before any query runs:
it adds shortcut edges, assigns every node a rank, and then answers queries
by walking only the edges that lead to higher-ranked nodes. A query needs
random access to "the up-edges of ``v``" and "the down-edges of ``v``", plus
the ability to unpack a shortcut back into the two edges it replaced. None of
that fits behind ``neighbors()``, so the CH layer owns its own structure.

It also deliberately avoids ``networkx``. Preprocessing touches every node and
every pair of its neighbours, so on a 500k-node synthetic graph the per-edge
dictionary-of-dictionaries overhead networkx carries would dominate the
measurement we are trying to take. Plain dicts of dicts keyed by integer node
ids are what make the large-scale benchmarks finish in reasonable time.

Two graph types live here:

:class:`CHGraph`
    The mutable working graph that preprocessing contracts. It holds forward
    and backward adjacency (CH needs both: the backward search walks incoming
    edges) and records every shortcut it creates.

:class:`PreparedCH`
    The immutable result of preprocessing: the augmented graph plus the rank
    of every node. This is what :func:`~search_algorithms.ch.query.ch_query`
    consumes. Separating the two keeps "the graph being built" and "the graph
    being queried" from sharing mutable state, which is the kind of aliasing
    bug that makes a CH implementation return *almost* correct answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

#: CH node ids are plain integers. Callers with other node types (OSM ids,
#: grid ``(row, col)`` tuples, string labels) go through
#: :mod:`search_algorithms.ch.build`, which assigns dense integer ids and
#: keeps the mapping so results can be translated back.
Node = int

#: ``(u, w) -> (via, cost)``: the shortcut from ``u`` to ``w`` replaces the
#: two-edge path ``u -> via -> w``. ``via`` may itself be the endpoint of
#: further shortcuts, so unpacking is recursive.
ShortcutMap = dict[tuple[Node, Node], tuple[Node, float]]


class CHGraph:
    """
    A directed, non-negatively weighted graph that supports contraction.

    Parallel edges collapse to the cheapest one, which is safe for shortest
    paths and keeps the adjacency a plain ``dict[Node, dict[Node, float]]``.

    Parameters
    ----------
    node_count:
        Nodes are ``0 .. node_count - 1``. Isolated nodes are legal.
    """

    __slots__ = ("_out", "_in", "_shortcuts", "_node_count")

    def __init__(self, node_count: int):
        if node_count < 0:
            raise ValueError("node_count must be non-negative.")

        self._node_count = node_count
        self._out: list[dict[Node, float]] = [{} for _ in range(node_count)]
        self._in: list[dict[Node, float]] = [{} for _ in range(node_count)]
        self._shortcuts: ShortcutMap = {}

    # -- construction -----------------------------------------------------

    def add_edge(self, u: Node, v: Node, cost: float) -> None:
        """
        Add a directed edge ``u -> v``, keeping the cheapest of any parallels.

        Self-loops are dropped: they can never appear on a shortest path with
        non-negative costs, and letting them through would make the
        contraction loop consider ``u -> v -> u`` "paths".
        """
        if cost < 0:
            raise ValueError(
                f"Negative edge cost {cost} on {u} -> {v}; CH (like Dijkstra) "
                "requires non-negative costs."
            )
        self._check_node(u)
        self._check_node(v)

        if u == v:
            return

        cost = float(cost)
        existing = self._out[u].get(v)
        if existing is not None and existing <= cost:
            return

        self._out[u][v] = cost
        self._in[v][u] = cost

    def add_undirected_edge(self, u: Node, v: Node, cost: float) -> None:
        """Add ``u -> v`` and ``v -> u`` with the same cost."""
        self.add_edge(u, v, cost)
        self.add_edge(v, u, cost)

    def _check_node(self, node: Node) -> None:
        if not 0 <= node < self._node_count:
            raise ValueError(
                f"Node {node} is out of range for a graph with "
                f"{self._node_count} nodes."
            )

    # -- inspection -------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def edge_count(self) -> int:
        """Number of directed edges, shortcuts included."""
        return sum(len(edges) for edges in self._out)

    @property
    def shortcut_count(self) -> int:
        return len(self._shortcuts)

    @property
    def shortcuts(self) -> Mapping[tuple[Node, Node], tuple[Node, float]]:
        """The shortcut table: ``(u, w) -> (via, cost)``. Read-only view."""
        return self._shortcuts

    def nodes(self) -> Iterator[Node]:
        return iter(range(self._node_count))

    def successors(self, u: Node) -> Mapping[Node, float]:
        """Outgoing edges of ``u`` as ``{v: cost}``."""
        return self._out[u]

    def predecessors(self, u: Node) -> Mapping[Node, float]:
        """Incoming edges of ``u`` as ``{v: cost}``."""
        return self._in[u]

    def edge_cost(self, u: Node, v: Node) -> float:
        """Cost of ``u -> v``, or ``inf`` when the edge does not exist."""
        return self._out[u].get(v, float("inf"))

    def degree(self, u: Node) -> int:
        """Combined in- and out-degree, the quantity contraction removes."""
        return len(self._out[u]) + len(self._in[u])

    # -- mutation used by contraction --------------------------------------

    def add_shortcut(self, u: Node, w: Node, via: Node, cost: float) -> bool:
        """
        Add (or improve) the shortcut ``u -> w`` standing in for ``u -> via -> w``.

        Returns ``True`` if the graph changed. A shortcut is only recorded in
        the shortcut table when it is actually the cheapest ``u -> w`` edge;
        if a cheaper original edge already exists, the shortcut would never be
        traversed and recording it would corrupt unpacking - the unpacker
        would expand an edge whose real cost belongs to a different,
        non-shortcut edge.
        """
        cost = float(cost)
        existing = self._out[u].get(w)

        if existing is not None and existing <= cost:
            return False

        self._out[u][w] = cost
        self._in[w][u] = cost
        self._shortcuts[(u, w)] = (via, cost)
        return True

    def remove_edge(self, u: Node, v: Node) -> bool:
        """
        Remove the edge ``u -> v`` if present. Returns whether it existed.

        Used by the synthetic graph generator to punch dead ends into an
        otherwise perfectly regular lattice; contraction never removes edges.
        """
        if self._out[u].pop(v, None) is None:
            return False
        self._in[v].pop(u, None)
        self._shortcuts.pop((u, v), None)
        return True

    def copy(self) -> "CHGraph":
        """A deep-enough copy: adjacency dicts are cloned, costs are floats."""
        clone = CHGraph(self._node_count)
        clone._out = [dict(edges) for edges in self._out]
        clone._in = [dict(edges) for edges in self._in]
        clone._shortcuts = dict(self._shortcuts)
        return clone

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CHGraph(nodes={self._node_count}, edges={self.edge_count}, "
            f"shortcuts={self.shortcut_count})"
        )


@dataclass(frozen=True)
class PreprocessStats:
    """
    What preprocessing cost and produced.

    These are the numbers Experiment B in the analysis reports, so they are
    part of the returned value rather than something printed and discarded.

    Attributes
    ----------
    node_count, original_edge_count:
        Size of the input graph.
    shortcuts_added:
        Shortcut edges in the final hierarchy.
    witness_searches:
        How many witness searches ran. The dominant cost of preprocessing,
        and the quantity the hop limit trades against shortcut quality.
    witness_hits:
        Witness searches that found a witness path and therefore *avoided*
        adding a shortcut. ``witness_searches - witness_hits`` is the number
        of shortcut insertions attempted.
    priority_reevaluations:
        Times the lazy heap popped a node whose priority had gone stale and
        pushed it back. A proxy for how much the ordering drifted during
        contraction.
    elapsed_seconds:
        Wall-clock preprocessing time.
    core_size:
        Nodes left uncontracted at the top of the hierarchy. Zero means the
        graph was contracted completely.
    """

    node_count: int
    original_edge_count: int
    shortcuts_added: int
    witness_searches: int
    witness_hits: int
    priority_reevaluations: int
    elapsed_seconds: float
    core_size: int = 0

    @property
    def edge_growth(self) -> float:
        """Shortcuts added per original edge - the space cost of the index."""
        if self.original_edge_count == 0:
            return 0.0
        return self.shortcuts_added / self.original_edge_count


@dataclass(frozen=True)
class PreparedCH:
    """
    A contracted hierarchy, ready to answer queries.

    Attributes
    ----------
    graph:
        The augmented graph: every original edge plus every shortcut.
    rank:
        ``rank[v]`` is ``v``'s position in the contraction order. Contracted
        first means lowest rank. The query only ever relaxes an edge
        ``u -> v`` when ``rank[v] > rank[u]``.
    order:
        The contraction order itself, ``order[i]`` being the node of rank
        ``i``. Redundant with ``rank`` but convenient for reporting.
    core_rank:
        Ranks at or above this belong to the uncontracted top *core*. The
        rank restriction does not apply between two core nodes, because no
        shortcuts were built to bypass them - see
        :data:`~search_algorithms.ch.preprocess.DEFAULT_CORE_FRACTION`. When
        every node was contracted this equals ``node_count``, and the
        restriction applies everywhere.
    stats:
        What preprocessing cost; see :class:`PreprocessStats`.
    """

    graph: CHGraph
    rank: list[int]
    order: list[Node]
    core_rank: int = 1 << 62
    stats: PreprocessStats = field(
        default_factory=lambda: PreprocessStats(0, 0, 0, 0, 0, 0, 0.0)
    )

    #: Pre-split adjacency the query walks: ``up_forward[v]`` lists the edges
    #: out of ``v`` that climb the hierarchy, ``up_backward[v]`` the edges
    #: into ``v`` that do. Built once by :meth:`finalize`.
    #:
    #: This split is not a micro-optimisation, it is the difference between
    #: CH being fast and merely expanding fewer nodes. Contraction roughly
    #: triples average degree (shortcuts are edges too), so a query that
    #: walked the full adjacency and skipped downward edges would scan every
    #: one of them - paying the contracted graph's degree to enjoy the
    #: hierarchy's small search space, and cancelling most of the win.
    up_forward: list[list[tuple[Node, float]]] = field(default_factory=list)
    up_backward: list[list[tuple[Node, float]]] = field(default_factory=list)

    def in_core(self, node: Node) -> bool:
        """Whether ``node`` was left uncontracted in the top core."""
        return self.rank[node] >= self.core_rank

    def finalize(self) -> "PreparedCH":
        """
        Populate :attr:`up_forward` / :attr:`up_backward`. Returns ``self``.

        Core nodes keep every edge to another core node in both lists,
        regardless of rank, because the rank restriction does not hold among
        uncontracted nodes.
        """
        rank = self.rank
        core_rank = self.core_rank
        graph = self.graph

        forward: list[list[tuple[Node, float]]] = []
        backward: list[list[tuple[Node, float]]] = []

        for v in range(graph.node_count):
            v_rank = rank[v]
            v_core = v_rank >= core_rank

            forward.append(
                [
                    (w, cost)
                    for w, cost in graph.successors(v).items()
                    if rank[w] > v_rank or (v_core and rank[w] >= core_rank)
                ]
            )
            backward.append(
                [
                    (u, cost)
                    for u, cost in graph.predecessors(v).items()
                    if rank[u] > v_rank or (v_core and rank[u] >= core_rank)
                ]
            )

        object.__setattr__(self, "up_forward", forward)
        object.__setattr__(self, "up_backward", backward)
        return self

    @property
    def node_count(self) -> int:
        return self.graph.node_count

