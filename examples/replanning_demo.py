"""
What happens when the road you planned to take is closed?

This demo plans a route across Tuscaloosa, then closes several roads along it.
The agent only finds out about a closure when it arrives at one, so it has to
replan from wherever it is standing - repeatedly, if it is unlucky.

Two costs are reported:

* the **detour cost**: how much longer the drive actually was compared with a
  planner that knew about every closure from the start;
* the **search cost**: how many more nodes were expanded in total, because the
  agent had to solve several search problems instead of one.

Run it with::

    python examples/replanning_demo.py
    python examples/replanning_demo.py --closures 5 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from search_algorithms import astar  # noqa: E402
from search_algorithms.problems.osm_problem import (  # noqa: E402
    DEFAULT_PLACE,
    OSMProblem,
    load_place_graph,
)
from search_algorithms.replanning import (  # noqa: E402
    plan_with_full_knowledge,
    simulate_with_closures,
)

BRYANT_DENNY = (33.2083, -87.5504)
AMPHITHEATER = (33.2122, -87.5692)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", default=DEFAULT_PLACE)
    parser.add_argument(
        "--closures",
        type=int,
        default=3,
        help="How many roads along the original route to close.",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    rng = random.Random(args.seed)

    graph = load_place_graph(place=args.place)
    start = OSMProblem.nearest_node(graph, *BRYANT_DENNY)
    goal = OSMProblem.nearest_node(graph, *AMPHITHEATER)

    # The route the agent would drive if nothing were closed.
    baseline = astar(OSMProblem(graph, start=start, goal=goal))
    if not baseline.found:
        print("No route between those points; nothing to demonstrate.")
        return

    print("=" * 66)
    print("DYNAMIC REPLANNING - Bryant-Denny Stadium to Tuscaloosa Amphitheater")
    print("=" * 66)
    print(f"\nUndisturbed route: {baseline.cost:,.0f} m over {baseline.steps} intersections")
    print(f"Found by expanding {baseline.nodes_expanded:,} nodes.\n")

    # Close roads on the interior of that route, so the agent is guaranteed to
    # run into them rather than being blocked before it sets off.
    candidates = list(zip(baseline.path, baseline.path[1:]))[1:-1]
    if not candidates:
        print("Route too short to close anything interesting.")
        return

    count = min(args.closures, len(candidates))
    closures = rng.sample(candidates, count)

    print(f"Closing {count} road segment(s) along that route:")
    reference = OSMProblem(graph, start=start, goal=goal)
    for u, v in closures:
        lat, lon = reference.latlon(u)
        print(f"  - segment {u} -> {v}  (near {lat:.5f}, {lon:.5f})")

    # The agent discovers closures the hard way.
    outcome = simulate_with_closures(graph, start, goal, closures, search=astar)

    # The omniscient planner knew about all of them up front.
    omniscient = plan_with_full_knowledge(graph, start, goal, closures, search=astar)

    print("\n" + "-" * 66)
    print("RESULTS")
    print("-" * 66)

    if not outcome.reached_goal:
        print("\nThe agent never reached the goal - the closures cut it off entirely.")
    else:
        print(f"\nReplans needed:        {outcome.replan_count}")
        print(f"Distance driven:       {outcome.travelled_cost:,.0f} m")
        if omniscient.found:
            print(f"Best possible:         {omniscient.cost:,.0f} m")
            detour = outcome.travelled_cost - omniscient.cost
            pct = (detour / omniscient.cost * 100) if omniscient.cost else 0.0
            print(f"Cost of not knowing:   {detour:,.0f} m  ({pct:+.1f}%)")

        print(f"\nTotal nodes expanded:  {outcome.total_nodes_expanded:,}"
              "   (first plan + every replan)")
        if omniscient.found:
            print(f"Omniscient planner:    {omniscient.nodes_expanded:,}")
            ratio = (
                outcome.total_nodes_expanded / omniscient.nodes_expanded
                if omniscient.nodes_expanded
                else float("inf")
            )
            print(f"Extra search work:     {ratio:.2f}x")
        print(f"Total search time:     {outcome.total_elapsed_seconds * 1000:.1f} ms")

    if outcome.replans:
        print("\nReplanning log:")
        print(f"  {'#':<4}{'at node':>12}{'blocked segment':>26}{'expanded':>11}")
        print("  " + "-" * 51)
        for i, event in enumerate(outcome.replans, start=1):
            segment = f"{event.blocked_edge[0]}->{event.blocked_edge[1]}"
            print(
                f"  {i:<4}{event.at_node:>12}{segment:>26}"
                f"{event.result.nodes_expanded:>11,}"
            )

    print(
        "\nTakeaway: replanning from scratch always finds a valid way through,"
        "\nbut it pays for its ignorance twice - in detour distance, and in"
        "\nrepeated search effort. Incremental replanning (D* Lite) reuses the"
        "\nprevious search tree to cut that second cost."
    )


if __name__ == "__main__":
    main()
