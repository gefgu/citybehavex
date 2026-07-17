"""Road/rail graph construction for citybehavex -- thin re-export of
``fastmob.network`` (migrated there since none of this is citybehavex-specific:
Overture Maps fetch/build/cache and nearest-node snapping apply to any
mobility-analysis project working with a road/rail network).
"""

from __future__ import annotations

from fastmob.network import (
    build_rail_graph,
    build_road_graph,
    fetch_rail_network,
    fetch_road_network,
    haversine_m_batch as haversine_m,
    snap_locations_to_graph,
)

__all__ = [
    "build_rail_graph",
    "build_road_graph",
    "fetch_rail_network",
    "fetch_road_network",
    "haversine_m",
    "snap_locations_to_graph",
]
