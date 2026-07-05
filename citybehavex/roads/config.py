from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RoadNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overture_release: Optional[str] = None  # None -> reuse tessellation.overture_release
    nodes_output: str = "data/road_graph_nodes.parquet"
    edges_output: str = "data/road_graph_edges.parquet"
    snap_output: str = "data/road_graph_snap.parquet"
    snap_max_distance_m: float = 750.0
    max_leg_waypoints: int = 128


class RailNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    overture_release: Optional[str] = None  # None -> reuse tessellation.overture_release
    nodes_output: str = "data/rail_graph_nodes.parquet"
    edges_output: str = "data/rail_graph_edges.parquet"
    snap_output: str = "data/rail_graph_snap.parquet"
    snap_max_distance_m: float = 1500.0
    max_leg_waypoints: int = 128
    classes: list[str] = Field(
        default_factory=lambda: ["subway", "tram", "light_rail", "monorail", "standard_gauge"]
    )
    speed_kmh_by_class: dict[str, float] = Field(
        default_factory=lambda: {
            "subway": 35.0,
            "tram": 22.0,
            "light_rail": 30.0,
            "monorail": 28.0,
            "standard_gauge": 45.0,
        }
    )
    default_speed_kmh: float = 35.0

    @field_validator("classes")
    @classmethod
    def non_empty_classes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("classes must not be empty")
        return value
