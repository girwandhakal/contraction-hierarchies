"""
Rendering helpers.

``render_ascii`` is dependency-free. The plotting functions need the ``viz``
extra (matplotlib), and the map plots additionally need the ``osm`` extra, so
they are imported lazily from their own modules rather than re-exported here.
"""

from .grid_render import render_ascii

__all__ = ["render_ascii"]
