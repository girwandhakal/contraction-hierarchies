"""
Dynamic replanning: searching a world that changes while you travel through it.

Needs the optional ``osm`` extra, since it operates on OSMnx street graphs.
"""

from .repeated_astar import (
    ReplanEvent,
    ReplanOutcome,
    plan_with_full_knowledge,
    simulate_with_closures,
)

__all__ = [
    "ReplanEvent",
    "ReplanOutcome",
    "simulate_with_closures",
    "plan_with_full_knowledge",
]
