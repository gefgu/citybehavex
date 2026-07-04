from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from .comparison import ALL_REPORT_SECTIONS


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    label: str = "observed"
    # Deprecated standalone metrics export; None = skip it.
    json_output: Optional[str] = None
    # Which report sections to compute; None (default) = run all of them.
    # Wasserstein/CPC summary metrics and the ECDF charts always run.
    sections: Optional[list[str]] = None
    # When True (default) and a cached road graph is available (road_network
    # section), jump_lengths/radius_of_gyration are recomputed as
    # road-network distance instead of straight-line Haversine, for both
    # synthetic and real trajectories. Road-network routing is far more
    # expensive per pair than skmob2's vectorized Haversine, so very large
    # real datasets (tens/hundreds of millions of rows) may want this off to
    # keep the live web comparison responsive.
    road_network_distance: bool = True

    @field_validator("sections")
    @classmethod
    def valid_sections(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return value
        unknown = set(value) - ALL_REPORT_SECTIONS
        if unknown:
            raise ValueError(
                f"Unknown comparison report section(s): {sorted(unknown)}. "
                f"Valid sections: {sorted(ALL_REPORT_SECTIONS)}"
            )
        return value
