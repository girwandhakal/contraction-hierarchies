"""
Routing over a real street network from OpenStreetMap.

This is where a grid maze becomes an actual map. The graph comes from OSMnx:
nodes are intersections (each carrying a latitude/longitude), and edges are
road segments whose ``length`` attribute is their length in meters.

Two details make this a genuinely different problem from the grid:

* Edge costs are **directional**. A one-way street is an edge in one direction
  only, which is exactly why :class:`Problem` defines cost per edge rather
  than per destination cell.
* The heuristic is **great-circle distance** to the goal, in meters. It is
  admissible because no road between two intersections can be shorter than the
  straight line joining them.

Downloading a city graph is slow, so :func:`load_place_graph` caches the result
on disk and reuses it on subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import networkx as nx

from ..core.heuristics import haversine
from ..core.problem import Problem

logger = logging.getLogger(__name__)

Node = int

#: Default study area for this project: Tuscaloosa, Alabama, home of UA.
DEFAULT_PLACE = "Tuscaloosa, Alabama, USA"

#: Where downloaded graphs are cached (``data/osm_cache/`` at the repo root).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "osm_cache"


def _slugify(text: str) -> str:
    """Turn a place name into a safe filename fragment."""
    keep = [ch.lower() if ch.isalnum() else "_" for ch in text]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_place_graph(
    place: str = DEFAULT_PLACE,
    network_type: str = "drive",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> nx.MultiDiGraph:
    """
    Load the street network for ``place``, downloading it only if needed.

    The graph is cached as GraphML under ``cache_dir``. Delete that file (or
    pass ``refresh=True``) to re-download.

    Parameters
    ----------
    place:
        Any place query OSMnx/Nominatim understands, e.g.
        ``"Tuscaloosa, Alabama, USA"``.
    network_type:
        ``"drive"``, ``"walk"``, ``"bike"``, ... as accepted by OSMnx.
    refresh:
        Force a fresh download even if a cached copy exists.
    """
    import osmnx as ox

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_slugify(place)}_{network_type}.graphml"

    # OSMnx keeps its own cache of raw Overpass responses, and by default puts
    # it in ./cache relative to the working directory - which scatters an
    # 8 MB folder wherever the code happens to be run from. Keep it alongside
    # our own cache instead.
    ox.settings.cache_folder = str(cache_dir.parent / "overpass_cache")

    if cache_path.exists() and not refresh:
        logger.info("Loading cached street graph from %s", cache_path)
        return ox.load_graphml(cache_path)

    logger.info("Downloading street network for %r (this may take a minute)...", place)
    graph = ox.graph_from_place(place, network_type=network_type)
    ox.save_graphml(graph, cache_path)
    logger.info(
        "Cached %d nodes / %d edges to %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cache_path,
    )
    return graph


class OSMProblem(Problem):
    """
    A routing problem on an OSMnx street graph.

    Parameters
    ----------
    graph:
        An OSMnx-style graph whose nodes carry ``x``/``y`` attributes
        (longitude/latitude) and whose edges carry ``length`` in meters.
    start, goal:
        OSM node IDs. Use :meth:`nearest_node` to convert a clicked
        latitude/longitude into one.
    blocked_edges:
        Optional set of ``(u, v)`` pairs to treat as impassable. This is how
        the replanning demo simulates a road closure without mutating the
        shared graph.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        start: Node,
        goal: Node,
        blocked_edges: Iterable[tuple[Node, Node]] | None = None,
    ):
        if start not in graph:
            raise ValueError(f"Start node {start} is not in the graph.")
        if goal not in graph:
            raise ValueError(f"Goal node {goal} is not in the graph.")

        self.graph = graph
        self.start: Node = start
        self.goal: Node = goal
        self.blocked_edges: set[tuple[Node, Node]] = set(blocked_edges or ())

        goal_data = graph.nodes[goal]
        self._goal_latlon = (float(goal_data["y"]), float(goal_data["x"]))

    # -- geometry helpers -------------------------------------------------

    def latlon(self, node: Node) -> tuple[float, float]:
        """Return ``(latitude, longitude)`` for an OSM node."""
        data = self.graph.nodes[node]
        return (float(data["y"]), float(data["x"]))

    def path_latlon(self, path: Iterable[Node]) -> list[tuple[float, float]]:
        """Convert a node path into ``(lat, lon)`` points for drawing."""
        return [self.latlon(node) for node in path]

    @staticmethod
    def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> Node:
        """
        Snap a ``(lat, lon)`` point to the nearest node in ``graph``.

        OSMnx takes ``X=longitude, Y=latitude`` - the opposite order from the
        ``(lat, lon)`` convention used throughout this package.
        """
        import osmnx as ox

        return int(ox.distance.nearest_nodes(graph, X=lon, Y=lat))

    # -- Problem interface ------------------------------------------------

    def neighbors(self, state: Node) -> list[Node]:
        """Successors of ``state``, honouring one-way streets and blockages."""
        result: list[Node] = []
        for neighbor in self.graph.successors(state):
            if (state, neighbor) in self.blocked_edges:
                continue
            result.append(int(neighbor))
        return result

    def edge_cost(self, a: Node, b: Node) -> float:
        """
        Length in meters of the shortest road segment from ``a`` to ``b``.

        OSM graphs are multigraphs: two intersections can be joined by several
        distinct ways, so take the shortest.
        """
        edges = self.graph.get_edge_data(a, b)
        if not edges:
            raise ValueError(f"No edge from {a} to {b}.")
        return float(min(float(data.get("length", 0.0)) for data in edges.values()))

    def heuristic(self, state: Node) -> float:
        """Great-circle distance in meters from ``state`` to the goal."""
        return haversine(self.latlon(state), self._goal_latlon)
