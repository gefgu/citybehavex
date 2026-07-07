"""Progressive comparison payload builders.

This package keeps the historical ``web.backend.app.payload`` import path while
splitting the implementation into focused modules.
"""

from __future__ import annotations

import sys
from types import ModuleType

from . import legacy as _legacy
from .legacy import *  # noqa: F401,F403 - compatibility surface for existing tests/callers
from .sections import (
    SECTION_BUILDERS,
    build_chart_base_payload,
    build_chart_section_payload,
    build_section_activity,
    build_section_distributions,
    build_section_metrics,
    build_section_micro_activity,
    build_section_mobility_laws,
    build_section_motifs,
    build_section_profiles,
    build_section_social_network,
    build_section_stvd,
    build_section_time_use,
)

__all__ = [
    name
    for name in globals()
    if not name.startswith("__")
]

for _name in (
    "_common_part_of_commuters",
    "_filter_df",
    "_load_social_network_sidecar",
    "_mobility_law_visits",
    "_special_day_filters",
    "visits_per_user_wasserstein_distance",
):
    globals()[_name] = getattr(_legacy, _name)
    if _name not in __all__:
        __all__.append(_name)


class _PayloadModule(ModuleType):
    def __setattr__(self, name: str, value):  # noqa: ANN001
        super().__setattr__(name, value)
        if hasattr(_legacy, name):
            setattr(_legacy, name, value)


sys.modules[__name__].__class__ = _PayloadModule
