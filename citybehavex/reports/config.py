from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .comparison import ALL_REPORT_SECTIONS


class NetworkValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    observed_enabled: bool = False
    synthetic_enabled: bool = True
    time_window: Literal["day"] = "day"
    location_mode: Literal["auto", "location_col", "h3"] = "auto"
    location_col: Optional[str] = None
    h3_resolution: int = 9
    random_seed: int = 42
    # The ablation table only reads synthetic_vs_observed/observed_vs_observed
    # (see aggregate_ablation_results.py's NETWORK_METRICS lookup), never
    # synthetic_vs_random/observed_vs_random -- building the degree-preserving
    # random null-model graph and its clustering/topological-overlap metrics
    # is ~56% of total network-validation cost on dense graphs (measured on
    # yjmob) for a comparison nothing downstream reads. Default True to keep
    # existing web-UI behavior (Charts.tsx renders these blocks); the
    # ablation CLI/scripts turn it off explicitly.
    random_baseline: bool = True
    # The observed-side co-presence graph and its metrics are identical
    # across every ablation variant/round for a given dataset, but were
    # recomputed from scratch on every single report call -- cache them
    # (keyed by comparison path + these params) so repeated calls reuse the
    # first computation instead of repeating it.
    cache_observed: bool = True


class TransportSpatialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    observed_enabled: bool = False
    synthetic_moving_path: Optional[str] = None
    uid_col: Optional[str] = None
    datetime_col: Optional[str] = None
    lat_col: Optional[str] = None
    lng_col: Optional[str] = None
    transport_col: Optional[str] = None
    mode_map: dict[str, str] = Field(default_factory=dict)


class EvaluationAdaptationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # auto = adapt only timestamp-only panel/ping-like observed data; force =
    # always run the observed side through the stay normalizer; off = preserve
    # historical raw-row behavior.
    mode: Literal["auto", "force", "off"] = "auto"
    # Preferred observed location column for collapsing consecutive panel rows.
    # When unset or missing, the evaluator falls back to H3 cells.
    location_col: Optional[str] = None
    h3_resolution: int = Field(default=10, ge=0, le=15)


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    label: str = "observed"
    time_use_path: Optional[str] = None
    time_use_label: str = "time-use"
    time_use_country: Optional[str] = None
    time_use_survey: Optional[int] = None
    time_use_weight_col: str = "propwt"
    # Real per-trip travel-time ground truth (a "Duration" column, in
    # minutes), used in place of the distance/CAR_SPEED_KMH proxy for the
    # trip_duration_min metric's observed side. None (default) keeps the
    # proxy -- most datasets have no real travel-time survey data.
    trip_duration_path: Optional[str] = None
    trip_duration_label: str = "observed"
    # Deprecated standalone metrics export; None = skip it.
    json_output: Optional[str] = None
    # Which optional comparison metric sections to compute; None (default) = run all of them.
    # Core Wasserstein summary metrics always run.
    sections: Optional[list[str]] = None
    # When True (default) and a cached road graph is available (road_network
    # section), jump_lengths/radius_of_gyration are recomputed as
    # road-network distance instead of straight-line Haversine, for both
    # synthetic and real trajectories. Road-network routing is far more
    # expensive per pair than fastmob's vectorized Haversine, so very large
    # real datasets (tens/hundreds of millions of rows) may want this off to
    # keep the live web comparison responsive.
    road_network_distance: bool = True
    # When set, both synthetic and observed coordinates are snapped to their
    # H3 cell centroid at this resolution before jump_lengths_km/
    # radius_of_gyration_km/visits_per_user are computed -- matching the
    # spatial granularity the simulation actually operates at (an H3-cell
    # tessellation, e.g. Shanghai's resolution-8 grid) instead of comparing
    # a coarse synthetic side against continuous real GPS/check-in
    # coordinates. Only meaningful for H3-tessellation scenarios; leave
    # unset (None, default) for POI-tessellation scenarios like GreaterParis,
    # where "cell" isn't the right unit.
    distance_h3_resolution: Optional[int] = Field(default=None, ge=0, le=15)
    evaluation_adaptation: EvaluationAdaptationConfig = Field(
        default_factory=EvaluationAdaptationConfig
    )
    network_validation: NetworkValidationConfig = Field(default_factory=NetworkValidationConfig)
    transport_spatial: TransportSpatialConfig = Field(default_factory=TransportSpatialConfig)

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
