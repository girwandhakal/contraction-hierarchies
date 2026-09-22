"""
Dynamic replanning by repeated A*.

A route computed once assumes the world holds still. It does not: roads close,
wrecks happen, a street you planned to use turns out to be blocked. This module
simulates an agent that only discovers a blockage when it reaches it, and
replans from wherever it is standing.

The strategy here is deliberately the simple one - throw the old search away
and run A* again from the current position. That is easy to get right, and it
gives a concrete baseline to measure against: the demo reports how much total
work (nodes expanded) the repeated searches cost versus a single search by an
omniscient planner who knew about every closure from the start.

A smarter approach (D* Lite) reuses the previous search tree instead of
discarding it. That is the natural next step for this project and is noted as
future work in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Sequence

import networkx as nx

from ..core.algorithms import astar
from ..core.result import SearchResult
from ..problems.osm_problem import OSMProblem

Node = Hashable
Edge = tuple[Node, Node]


@dataclass
class ReplanEvent:
    """One replanning episode, recorded for reporting."""

    #: Node the agent had reached when it discovered the blockage.
    at_node: Node
    #: The edge it tried to traverse and found closed.
    blocked_edge: Edge
    #: The search it ran to recover.
    result: SearchResult


@dataclass
class ReplanOutcome:
    """
    Result of driving an agent from start to goal through a changing world.

    Attributes
    ----------
    reached_goal:
        Whether the agent arrived. False if a closure cut the goal off.
    travelled_path:
        The nodes the agent actually visited, in order.
    travelled_cost:
        Real cost of the route actually driven, including detours.
    initial_result:
        The optimistic first plan, made before any closure was known.
    replans:
        One entry per time the agent hit a closure and had to replan.
    """

    reached_goal: bool
    travelled_path: list[Node]
    travelled_cost: float
    initial_result: SearchResult
    replans: list[ReplanEvent] = field(default_factory=list)

    @property
    def replan_count(self) -> int:
        return len(self.replans)

    @property
    def total_nodes_expanded(self) -> int:
        """Total search work across the first plan and every replan."""
        return self.initial_result.nodes_expanded + sum(
            event.result.nodes_expanded for event in self.replans
        )

    @property
    def total_elapsed_seconds(self) -> float:
        return self.initial_result.elapsed_seconds + sum(
            event.result.elapsed_seconds for event in self.replans
        )


def simulate_with_closures(
    graph: nx.MultiDiGraph,
    start: Node,
    goal: Node,
    closures: Iterable[Edge],
    search: Callable[[OSMProblem], SearchResult] = astar,
    max_replans: int = 100,
) -> ReplanOutcome:
    """
    Drive an agent from ``start`` to ``goal``, replanning when it hits a closure.

    The agent is optimistic and myopic: it plans as though every road is open,
    then walks its plan. When the next step turns out to be closed it adds that
    edge to what it knows and replans from where it stands.

    Parameters
    ----------
    graph:
        The street network.
    start, goal:
        OSM node IDs.
    closures:
        Edges that are really closed. The agent does not know these up front;
        it learns each one by bumping into it.
    search:
        The search function used for planning and replanning.
    max_replans:
        Safety valve against pathological loops.

    Returns
    -------
    ReplanOutcome
    """
    truly_closed = {(u, v) for u, v in closures}
    # What the agent has discovered so far - starts empty.
    known_closed: set[Edge] = set()

    initial_problem = OSMProblem(graph, start=start, goal=goal)
    initial_result = search(initial_problem)

    outcome = ReplanOutcome(
        reached_goal=False,
        travelled_path=[start],
        travelled_cost=0.0,
        initial_result=initial_result,
    )

    if not initial_result.found:
        return outcome

    current = start
    plan: Sequence[Node] = initial_result.path

    while len(outcome.replans) <= max_replans:
        # Walk the current plan until the road runs out from under us.
        blocked_at: Edge | None = None

        for step_from, step_to in zip(plan, plan[1:]):
            if (step_from, step_to) in truly_closed:
                blocked_at = (step_from, step_to)
                break

            problem_for_cost = OSMProblem(graph, start=step_from, goal=goal)
            outcome.travelled_cost += problem_for_cost.edge_cost(step_from, step_to)
            outcome.travelled_path.append(step_to)
            current = step_to

        if blocked_at is None:
            outcome.reached_goal = current == goal
            return outcome

        # Learn the closure and plan again from where we stopped.
        known_closed.add(blocked_at)
        replan_problem = OSMProblem(
            graph, start=current, goal=goal, blocked_edges=known_closed
        )
        replan_result = search(replan_problem)
        outcome.replans.append(
            ReplanEvent(
                at_node=current,
                blocked_edge=blocked_at,
                result=replan_result,
            )
        )

        if not replan_result.found:
            # The closure cut the goal off entirely.
            return outcome

        plan = replan_result.path

    return outcome


def plan_with_full_knowledge(
    graph: nx.MultiDiGraph,
    start: Node,
    goal: Node,
    closures: Iterable[Edge],
    search: Callable[[OSMProblem], SearchResult] = astar,
) -> SearchResult:
    """
    Plan once, knowing every closure in advance.

    This is the omniscient baseline the repeated-replanning agent is measured
    against: the best any planner could do, and the yardstick for how much the
    agent's ignorance cost it.
    """
    problem = OSMProblem(graph, start=start, goal=goal, blocked_edges=closures)
    return search(problem)
