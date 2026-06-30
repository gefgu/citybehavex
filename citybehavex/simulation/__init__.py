from __future__ import annotations

from .config import SimulationConfig


def __getattr__(name: str):
    if name in {"CoreTiming", "simulate_agents"}:
        from . import core

        return getattr(core, name)
    if name in {
        "load_or_build_tessellation",
        "maybe_build_diaries",
        "maybe_build_profiles",
        "run_simulation",
        "simulation_dates",
    }:
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
