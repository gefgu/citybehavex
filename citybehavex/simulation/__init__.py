from __future__ import annotations

from .config import SimulationConfig


def __getattr__(name: str):
    if name in {"CoreTiming", "simulate_agents"}:
        from . import core

        return getattr(core, name)
    if name == "load_or_build_tessellation":
        from . import tessellation_pipeline

        return getattr(tessellation_pipeline, name)
    if name == "maybe_build_diaries" or name == "simulation_dates":
        from . import diary_pipeline

        return getattr(diary_pipeline, name)
    if name == "maybe_build_profiles":
        from . import profile_pipeline

        return getattr(profile_pipeline, name)
    if name == "run_simulation":
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CoreTiming",
    "SimulationConfig",
    "load_or_build_tessellation",
    "maybe_build_diaries",
    "maybe_build_profiles",
    "run_simulation",
    "simulate_agents",
    "simulation_dates",
]
