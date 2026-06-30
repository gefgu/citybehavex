from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class TessellationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    min_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lon: Optional[float] = None
    max_lat: Optional[float] = None
    resolution: int = 10
    enrich_overture: bool = False
    overture_release: str = "2026-05-20.0"
    min_poi_count: int = 1
    poi_tessellation: bool = False
    output: str = "tessellation.parquet"
    relevance_column: str = "total_poi_count"

    @model_validator(mode="after")
    def validate_source(self) -> TessellationConfig:
        bbox_values = [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
        has_any_bbox = any(v is not None for v in bbox_values)
        has_full_bbox = all(v is not None for v in bbox_values)
        if has_any_bbox and not has_full_bbox:
            raise ValueError("bbox requires min_lon, min_lat, max_lon, and max_lat")
        return self
