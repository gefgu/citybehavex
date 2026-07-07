"""Single import surface for the reusable report-compute helpers.

The comparison report in ``citybehavex.reports.comparison`` already contains the
schema auto-detection and all the numeric feature extraction; the web backend
reuses those functions and only replaces the ``skmob_vis`` widget rendering with
JSON serialization. Centralizing the imports here keeps that coupling in one
place.
"""

from __future__ import annotations

from citybehavex.reports.comparison import (  # noqa: F401
    CAR_SPEED_KMH,
    ActivityVisitsResult,
    _ACTIVITY_CANDIDATES,
    _DATETIME_CANDIDATES,
    _DURATION_CANDIDATES,
    _END_TS_CANDIDATES,
    _LOCATION_CANDIDATES,
    _collapse_to_stays,
    _common_part_of_commuters,
    _compute_stvd_layers,
    _daily_location_lognormal_dataset,
    _diff_stvd_layers,
    _distance_frequency_dataset,
    _location_resolution,
    _mobility_law_visits,
    _motif_visits,
    _micro_activity_daily_usage_data,
    _prepare_activity_visits,
    _stvd_hourly_histogram,
    _truncated_powerlaw_dataset,
    _visits_for_comparison,
    detect_column,
    load_trajectory,
    waiting_times_minutes,
)
from citybehavex.profiles import PROFILE_METRICS, compute_profiles  # noqa: F401
