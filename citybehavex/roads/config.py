from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RoadNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overture_release: Optional[str] = None  # None -> reuse tessellation.overture_release
    nodes_output: str = "data/road_graph_nodes.parquet"
    edges_output: str = "data/road_graph_edges.parquet"
    snap_output: str = "data/road_graph_snap.parquet"
    snap_max_distance_m: float = 750.0
    max_leg_waypoints: int = 16
