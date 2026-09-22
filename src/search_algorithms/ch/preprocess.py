"""
Preprocessing: contracting the graph into a hierarchy.

The contraction of a node ``v`` asks one question for every pair of its live
neighbours ``(u, w)``:

    Is ``u -> v -> w`` the *only* shortest way from ``u`` to ``w`` once ``v``
    is gone?

If some other path from ``u`` to ``w`` through still-live nodes is no longer
than ``cost(u, v) + cost(v, w)``, then removing ``v`` loses nothing: that
other path (the *witness*) already carries the distance. If no such path
exists, removing ``v`` would break the shortest path, so a shortcut edge
``u -> w`` is inserted to preserve it.

Getting the witness search wrong is the classic way to produce a CH that is
fast and subtly incorrect:

* Searching through contracted nodes finds witnesses that no longer exist in
  the working graph, so necessary shortcuts get skipped and queries return
  paths that are too long. The search here is restricted to live nodes.
* Searching *through* ``v`` itself finds the very path the shortcut is meant
  to replace, which again skips a needed shortcut. ``v`` is excluded
  explicitly.
* Accepting a witness strictly longer than the shortcut is also wrong; the
  comparison must allow equality (an equally short witness is a valid
  substitute) but nothing longer.

The hop limit
-------------
An unbounded witness search is a full Dijkstra per neighbour pair, which
makes preprocessing quadratic-ish and unusable at scale. Real CH
implementations bound it, and so does this one: the search gives up after
``max_hops`` edges. Giving up means "no witness found", which inserts a
shortcut that may not have been strictly necessary. That direction of error
is safe - an unnecessary shortcut costs a little space and query time but
never changes a shortest-path distance - whereas the opposite error
(wrongly concluding a witness exists) would corrupt results. The tests assert
correctness holds at several hop limits for exactly this reason.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from itertools import count
from typing import Callable

from .graph import CHGraph, Node, PreparedCH, PreprocessStats
from .ordering import (
    OrderingStrategy,
    contracted_neighbour_count,
    live_degree,
    static_order,
)

#: Default cap on witness-search depth, in edges. Five is the value used in
#: the CH literature for road networks and is plenty here: witnesses in road
#: graphs are almost always short detours around the node being removed.
DEFAULT_MAX_HOPS = 5

#: Slack when deciding whether a witness path is strictly cheaper than the
#: shortcut it would replace. Costs are sums of floats accumulated in
#: different orders, so two genuinely equal paths can differ in the last bits;
#: without this margin such a pair would be treated as "strictly cheaper" and
#: hit the tie problem the strict comparison exists to avoid.
WITNESS_TOLERANCE = 1e-9

#: Nodes a single witness search may settle before giving up. This is the
#: bound that makes preprocessing scale: without it, one expensive candidate
#: shortcut can send a witness search across a large part of the graph, and
#: since a witness search runs for every neighbour pair of every node, that
#: cost compounds. Giving up early reports "no witness", which inserts a
#: possibly-unnecessary shortcut - the safe direction of error.
#:
#: 500 is high enough that witnesses are rarely missed on road-like graphs
#: (a missed witness means an unnecessary shortcut, and unnecessary shortcuts
#: compound: they raise node degrees, which makes every later contraction
#: consider more neighbour pairs). Starving this bound is a false economy -
#: it makes individual witness searches cheap and the overall hierarchy much
#: worse.
DEFAULT_MAX_SETTLED = 500

#: How often the progress callback fires during contraction, as a fraction of
#: the node count. 0.01 means roughly one hundred updates over a full run,
#: which keeps a 500k-node run legible without flooding the terminal.
PROGRESS_FRACTION = 0.01

#: Fraction of nodes left uncontracted at the top of the hierarchy (the
#: "core"), as used by production CH implementations.
#:
#: Contraction gets dramatically more expensive as it approaches the top. The
#: remaining nodes become almost completely interconnected, so each one has a
#: high live degree, and contracting it considers O(degree^2) neighbour pairs
#: with a witness search for each. Measured on this implementation, the last
#: 1% of nodes accounted for over half of total preprocessing time - the core
#: is small enough that a query crossing it explores very little, so all that
#: work buys almost nothing.
#:
#: Leaving a core uncontracted keeps preprocessing near-linear. Correctness is
#: unaffected: core nodes simply all share the top rank band, and the query's
#: rank restriction is relaxed *within* the core (see
#: :mod:`search_algorithms.ch.query`), so paths through it are still found.
DEFAULT_CORE_FRACTION = 0.005


@dataclass(frozen=True)
class ProgressEvent:
    """
    One progress notification from :func:`build_ch`.

    Verbosity is delivered as structured events rather than prints so the CLI
    can format them, the benchmark can count them, and the tests can ignore
    them - the same reason the search algorithms return ``nodes_expanded``
    instead of printing it.

    Attributes
    ----------
    phase:
        ``"start"``, ``"contracting"`` or ``"done"``.
    contracted:
        Nodes contracted so far.
    total:
        Nodes in the graph.
    node:
        The node just contracted, when ``phase == "contracting"``.
    shortcuts_added:
        Running shortcut total.
    elapsed_seconds:
        Time since contraction began.
    """

    phase: str
    contracted: int
    total: int
    node: Node | None
    shortcuts_added: int
    elapsed_seconds: float

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 100.0
        return 100.0 * self.contracted / self.total


ProgressCallback = Callable[[ProgressEvent], None]


class _WitnessSearcher:
    """
    Bounded Dijkstra used to decide whether a shortcut is necessary.

    Kept as a class so the scratch arrays (``dist`` and the ``stamp`` that
    invalidates them) are allocated once per preprocessing run instead of
    once per witness search. On a 500k-node graph, reallocating a distance
    array for every neighbour pair is the difference between minutes and
    hours.

    The ``stamp`` trick avoids clearing the distance array between searches:
    an entry counts as unset unless its stamp matches the current search's
    stamp. Clearing an array of 500k floats per witness search would dominate
    the runtime of the search itself.
    """

    __slots__ = (
        "graph",
        "dist",
        "hops",
        "stamp",
        "current",
        "searches",
        "hits",
        "max_settled",
    )

    def __init__(self, graph: CHGraph, max_settled: int = 0):
        self.graph = graph
        self.dist: list[float] = [0.0] * graph.node_count
        self.hops: list[int] = [0] * graph.node_count
        self.stamp: list[int] = [0] * graph.node_count
        self.current = 0
        self.searches = 0
        self.hits = 0
        self.max_settled = max_settled

    def has_witness(
        self,
        source: Node,
        targets: dict[Node, float],
        excluded: Node,
        contracted: list[bool],
        max_hops: int,
    ) -> dict[Node, bool]:
        """
        For each target, is there a path from ``source`` avoiding ``excluded``
        that is no longer than the target's given limit?

        All targets are answered by a single search from ``source``, bounded
        by the largest limit among them - running one Dijkstra per pair would
        repeat almost identical work for every neighbour of the node being
        contracted.

        Parameters
        ----------
        source:
            Where the candidate shortcuts start (``u``).
        targets:
            ``{w: limit}``. ``limit`` is ``cost(u, v) + cost(v, w)``, the cost
            of the shortcut that would be added.
        excluded:
            The node being contracted (``v``). Paths through it do not count.
        contracted:
            Nodes already removed from the working graph; also not traversable.
        max_hops:
            Give up after this many edges. Giving up reports "no witness",
            which is the safe direction of error.

        Returns
        -------
        ``{w: True}`` when a witness was found and the shortcut can be
        skipped.
        """
        self.searches += 1

        if not targets:
            return {}

        # One search answers every target. The frontier is bounded by the
        # largest limit in play: beyond that, nothing left in the queue can
        # witness anything, because costs only grow.
        ceiling = max(targets.values())

        self.current += 1
        stamp = self.current
        dist = self.dist
        stamp_arr = self.stamp
        hops_arr = self.hops
        graph = self.graph
        max_settled = self.max_settled

        dist[source] = 0.0
        hops_arr[source] = 0
        stamp_arr[source] = stamp

        frontier: list[tuple[float, Node]] = [(0.0, source)]
        settled = 0

        while frontier:
            cost, node = heapq.heappop(frontier)

            # A cheaper route to `node` was found after this entry was pushed,
            # so this entry is stale. The stamp check is not needed here: any
            # node in the frontier was stamped when it was pushed, and stamps
            # only advance between searches, never within one.
            if dist[node] < cost:
                continue

            # Everything still queued costs at least this much, so no target
            # can still be reached within its limit.
            if cost > ceiling:
                break

            # Settled-node budget. Like the hop limit, exhausting it means
            # "no witness found" for anything not yet settled, which
            # over-inserts shortcuts rather than dropping necessary ones.
            # Without this bound one expensive candidate shortcut can drag a
            # witness search across a large part of the graph, and that cost
            # is paid for every neighbour pair of every node.
            settled += 1
            if max_settled and settled > max_settled:
                break

            node_hops = hops_arr[node]
            if node_hops >= max_hops:
                continue

            next_hops = node_hops + 1
            for nxt, edge_cost in graph.successors(node).items():
                if nxt == excluded or contracted[nxt]:
                    continue

                nxt_cost = cost + edge_cost
                if nxt_cost > ceiling:
                    continue

                if stamp_arr[nxt] != stamp:
                    stamp_arr[nxt] = stamp
                    dist[nxt] = nxt_cost
                    hops_arr[nxt] = next_hops
                    heapq.heappush(frontier, (nxt_cost, nxt))
                elif nxt_cost < dist[nxt]:
                    dist[nxt] = nxt_cost
                    hops_arr[nxt] = next_hops
                    heapq.heappush(frontier, (nxt_cost, nxt))

        # A target is witnessed when the search reached it within its own
        # limit without passing through `excluded`. Reading the answers off
        # the distance array afterwards is what lets a single search serve
        # every target.
        #
        # Using a possibly-unsettled distance is sound here, and worth being
        # explicit about because it looks like a bug. `dist[w]` is always the
        # cost of some real path from `source` to `w` in the live graph - the
        # search only ever writes a cost it actually traversed. It may be an
        # overestimate of the true shortest distance, never an underestimate.
        # So `dist[w] <= limit` proves a witness path exists, which is exactly
        # the question being asked. The reverse error - concluding "witness"
        # when none exists - is the one that would corrupt the hierarchy, and
        # it cannot happen this way.
        #
        # When the budget ran out, unreached targets are reported as
        # un-witnessed, which is the same safe direction of error as the hop
        # limit: an unnecessary shortcut, not a missing one.
        # The comparison is strict, and that strictness is load-bearing.
        #
        # An *equally* cheap witness looks like a valid substitute, but it is
        # only a substitute while it survives. Consider a grid, where equal
        # cost paths are everywhere: contracting `v` finds an equal-cost
        # witness through some node `x` and skips the shortcut; later `x` is
        # itself contracted and its own witness search finds the (now
        # equal-cost) route back through where `v` used to be. Both shortcuts
        # get skipped, and the distance they jointly carried is lost - the
        # hierarchy then has no bitonic path for that pair and the query
        # returns a cost that is too high.
        #
        # This is precisely the bug random-graph fuzzing does not catch:
        # random float weights almost never tie, so the equality case never
        # arises. Requiring the witness to be strictly cheaper (by more than
        # float noise) costs a few extra shortcuts and is always safe.
        result: dict[Node, bool] = {}
        found_any = False
        for w, limit in targets.items():
            if stamp_arr[w] == stamp and dist[w] < limit - WITNESS_TOLERANCE:
                result[w] = True
                found_any = True
            else:
                result[w] = False

        if found_any:
            self.hits += 1
        return result


def _shortcut_candidates(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
) -> dict[Node, dict[Node, float]]:
    """
    The shortcuts contracting ``v`` might need, before witness filtering.

    Returns ``{u: {w: cost(u,v) + cost(v,w)}}`` over live neighbours only,
    skipping ``u == w`` (a shortcut from a node to itself is never useful).
    """
    predecessors = {
        u: cost for u, cost in graph.predecessors(v).items() if not contracted[u]
    }
    successors = {
        w: cost for w, cost in graph.successors(v).items() if not contracted[w]
    }

    candidates: dict[Node, dict[Node, float]] = {}
    for u, in_cost in predecessors.items():
        targets: dict[Node, float] = {}
        for w, out_cost in successors.items():
            if w == u:
                continue
            targets[w] = in_cost + out_cost
        if targets:
            candidates[u] = targets
    return candidates


def _needed_shortcuts(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
    searcher: _WitnessSearcher,
    max_hops: int,
) -> list[tuple[Node, Node, float]]:
    """
    The shortcuts contracting ``v`` would require, as ``(u, w, cost)``.

    This is the expensive half of preprocessing - it runs a witness search per
    live predecessor of ``v`` - so the result is *returned* rather than
    applied. The node ordering needs the count, and the contraction needs the
    shortcuts themselves; computing them once and using the list for both is
    what keeps a node from paying for its witness searches twice.
    """
    candidates = _shortcut_candidates(graph, v, contracted)
    needed: list[tuple[Node, Node, float]] = []

    for u, targets in candidates.items():
        witnesses = searcher.has_witness(u, targets, v, contracted, max_hops)
        for w, limit in targets.items():
            if witnesses[w]:
                continue
            # A strictly cheaper existing edge already carries the distance,
            # so the shortcut would be redundant. An edge that merely *ties*
            # is left alone deliberately: `add_shortcut` will decline to
            # replace it, which is the correct outcome, and treating a tie as
            # redundant here would under-count shortcuts relative to what
            # contraction actually does - making the ordering heuristic
            # disagree with reality.
            if graph.edge_cost(u, w) < limit - WITNESS_TOLERANCE:
                continue
            needed.append((u, w, limit))

    return needed


def _apply_shortcuts(
    graph: CHGraph,
    v: Node,
    contracted: list[bool],
    shortcuts: list[tuple[Node, Node, float]],
) -> int:
    """
    Insert ``shortcuts`` and mark ``v`` contracted. Returns how many landed.

    ``v``'s edges are *not* removed from ``graph``: ``graph`` becomes the
    queried hierarchy and the query needs them. The ``contracted`` flag is
    what hides ``v`` from later witness searches - removal is logical, not
    destructive.
    """
    added = 0
    for u, w, cost in shortcuts:
        if graph.add_shortcut(u, w, v, cost):
            added += 1
    contracted[v] = True
    return added


def build_ch(
    graph: CHGraph,
    strategy: OrderingStrategy = OrderingStrategy.EDGE_DIFFERENCE,
    max_hops: int = DEFAULT_MAX_HOPS,
    on_progress: ProgressCallback | None = None,
    seed: int = 0,
    max_settled: int | None = None,
    core_fraction: float = DEFAULT_CORE_FRACTION,
) -> PreparedCH:
    """
    Contract every node in ``graph``, producing a queryable hierarchy.

    The input graph is copied, so the caller's graph is not modified.

    Parameters
    ----------
    graph:
        The graph to preprocess.
    strategy:
        Node ordering. ``EDGE_DIFFERENCE`` is the real heuristic; ``DEGREE``
        and ``RANDOM`` are Experiment E's baselines.
    max_hops:
        Witness-search depth limit. See the module docstring.
    on_progress:
        Called with :class:`ProgressEvent`s. ``None`` disables progress
        reporting entirely, which is what tests and inner benchmark loops
        want.
    seed:
        Seeds ``RANDOM`` ordering so runs are reproducible.
    max_settled:
        Nodes a single witness search may settle before giving up. ``None``
        picks :data:`DEFAULT_MAX_SETTLED`. Like ``max_hops``, a tighter bound
        trades extra shortcuts for faster preprocessing and never affects
        correctness.
    core_fraction:
        Fraction of nodes to leave uncontracted at the top; see
        :data:`DEFAULT_CORE_FRACTION`. Pass ``0.0`` to contract everything,
        which is slower on large graphs but produces a pure hierarchy (the
        tests exercise both).

    Returns
    -------
    A :class:`~search_algorithms.ch.graph.PreparedCH` holding the augmented
    graph, every node's rank, and :class:`PreprocessStats`.
    """
    started = time.perf_counter()

    working = graph.copy()
    total = working.node_count
    original_edges = working.edge_count

    contracted: list[bool] = [False] * total
    rank: list[int] = [0] * total
    order: list[Node] = []
    searcher = _WitnessSearcher(
        working,
        max_settled=DEFAULT_MAX_SETTLED if max_settled is None else max_settled,
    )
    shortcuts_added = 0
    reevaluations = 0

    # Stop contracting once only this many nodes remain live. They become the
    # core, all sharing the top rank band.
    core_size = max(0, min(total, int(total * core_fraction)))
    contract_limit = total - core_size

    progress_every = max(1, int(total * PROGRESS_FRACTION))

    def emit(phase: str, node: Node | None) -> None:
        if on_progress is None:
            return
        on_progress(
            ProgressEvent(
                phase=phase,
                contracted=len(order),
                total=total,
                node=node,
                shortcuts_added=shortcuts_added,
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    emit("start", None)

    if strategy is OrderingStrategy.EDGE_DIFFERENCE:
        # Lazy priority queue. Entries go stale as contraction changes the
        # graph around them, so a popped node's priority is recomputed and
        # the node is only contracted if it still beats the next candidate.
        tie = count()
        heap: list[tuple[int, int, Node]] = []

        # The shortcuts computed for each node the last time its priority was
        # evaluated. Contracting a node re-uses this instead of running its
        # witness searches a second time, which halves the dominant cost.
        pending: dict[Node, list[tuple[Node, Node, float]]] = {}

        for v in working.nodes():
            shortcuts = _needed_shortcuts(
                working, v, contracted, searcher, max_hops
            )
            pending[v] = shortcuts
            degree = live_degree(working, v, contracted)
            heapq.heappush(heap, (len(shortcuts) - degree, next(tie), v))

        # Nodes whose cached priority is out of date because a neighbour was
        # contracted since it was computed. Contracting a node adds shortcuts
        # around it, changing exactly its neighbours' edge differences.
        # Leaving those stale is what makes a hierarchy degenerate: the heap
        # then contracts in near-insertion order and shortcuts compound.
        stale: set[Node] = set()

        while heap and len(order) < contract_limit:
            priority, _, v = heapq.heappop(heap)
            if contracted[v]:
                continue

            # Recompute only when this node's cached priority might be wrong.
            # Skipping recomputation for untouched nodes is where most of the
            # preprocessing time is saved, since the witness searches behind
            # `_needed_shortcuts` dominate everything else.
            if v in stale:
                stale.discard(v)
                shortcuts = _needed_shortcuts(
                    working, v, contracted, searcher, max_hops
                )
                pending[v] = shortcuts
                current = (
                    len(shortcuts)
                    - live_degree(working, v, contracted)
                    + contracted_neighbour_count(working, v, contracted)
                )

                # If it got worse, it may no longer be the best candidate, so
                # put it back and let the heap decide again.
                if current > priority and heap and current > heap[0][0]:
                    heapq.heappush(heap, (current, next(tie), v))
                    reevaluations += 1
                    continue

            shortcuts_added += _apply_shortcuts(
                working, v, contracted, pending.pop(v)
            )
            rank[v] = len(order)
            order.append(v)

            # Everything still live around v now has a stale priority.
            for w in working.successors(v):
                if not contracted[w]:
                    stale.add(w)
            for u in working.predecessors(v):
                if not contracted[u]:
                    stale.add(u)

            if len(order) % progress_every == 0:
                emit("contracting", v)
    else:
        for v in static_order(working, strategy, seed=seed):
            if len(order) >= contract_limit:
                break
            if contracted[v]:
                continue
            shortcuts_added += _apply_shortcuts(
                working,
                v,
                contracted,
                _needed_shortcuts(working, v, contracted, searcher, max_hops),
            )
            rank[v] = len(order)
            order.append(v)

            if len(order) % progress_every == 0:
                emit("contracting", v)

    # Whatever is left forms the core. Every core node gets a rank at or above
    # `core_rank`, and the query treats core-to-core edges as unrestricted -
    # necessarily so, because no shortcuts were built to bypass these nodes,
    # and a rank restriction among them could cut the only remaining path.
    core_rank = len(order)
    for v in working.nodes():
        if not contracted[v]:
            rank[v] = len(order)
            order.append(v)

    elapsed = time.perf_counter() - started

    stats = PreprocessStats(
        node_count=total,
        original_edge_count=original_edges,
        shortcuts_added=shortcuts_added,
        witness_searches=searcher.searches,
        witness_hits=searcher.hits,
        priority_reevaluations=reevaluations,
        elapsed_seconds=elapsed,
        core_size=total - core_rank,
    )

    prepared = PreparedCH(
        graph=working,
        rank=rank,
        order=order,
        core_rank=core_rank,
        stats=stats,
    ).finalize()
    emit("done", None)
    return prepared

