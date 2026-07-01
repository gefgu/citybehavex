"""Discover experiments from ``configs/*.yaml`` and resolve their runs.

Each YAML config is one experiment. Simulation outputs are timestamp-stamped at
write time (``_YYYYMMDDTHHMMSS`` before the extension), so the concrete runs are
found by globbing the stem of ``simulation.output`` rather than trusting the
literal path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from citybehavex.config import load_config

from .config import CONFIGS_DIR, REPO_ROOT
from .datasource import run_summary

_STAMP_RE = re.compile(r"_(\d{8}T\d{6})$")


def _resolve(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class Run:
    run_id: str            # timestamp stamp, or "base" for the unstamped file
    path: Path
    mtime: float

    def to_dict(self, with_summary: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "path": str(self.path.relative_to(REPO_ROOT)),
            "mtime": self.mtime,
        }
        if with_summary:
            try:
                d["summary"] = run_summary(self.path)
            except Exception as exc:  # noqa: BLE001 - surface as metadata, don't 500
                d["summary_error"] = str(exc)
        return d


def _discover_runs(output_path: Optional[Path]) -> list[Run]:
    """All parquet runs for a ``simulation.output`` stem, newest first.

    Excludes the ``*_encounters.parquet`` sibling written alongside trajectories.
    """
    if output_path is None:
        return []
    stem = output_path.stem
    parent = output_path.parent
    if not parent.is_dir():
        return []

    runs: list[Run] = []
    for candidate in parent.glob(f"{stem}*{output_path.suffix}"):
        name = candidate.stem
        if name.endswith("_encounters"):
            continue
        if name == stem:
            run_id = "base"
        else:
            suffix = name[len(stem):]
            match = _STAMP_RE.match(suffix)
            if not match:
                continue
            run_id = match.group(1)
        runs.append(Run(run_id=run_id, path=candidate, mtime=candidate.stat().st_mtime))

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


@dataclass
class Experiment:
    id: str
    config_path: Path
    label: str
    synthetic_output: Optional[Path]
    observed_path: Optional[Path]
    params: dict[str, Any]
    runs: list[Run]

    def to_dict(self, with_summary: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": str(self.config_path.relative_to(REPO_ROOT)),
            "label": self.label,
            "observed_path": (
                str(self.observed_path.relative_to(REPO_ROOT))
                if self.observed_path else None
            ),
            "observed_exists": bool(self.observed_path and self.observed_path.exists()),
            "params": self.params,
            "runs": [r.to_dict(with_summary=with_summary) for r in self.runs],
        }

    def run(self, run_id: Optional[str]) -> Optional[Run]:
        if not self.runs:
            return None
        if run_id is None:
            return self.runs[0]
        for r in self.runs:
            if r.run_id == run_id:
                return r
        return None


def _load_experiment(config_path: Path) -> Experiment:
    cfg = load_config(str(config_path))
    synthetic_output = _resolve(cfg.simulation.output)
    observed_path = _resolve(cfg.comparison.path)
    params = {
        "agents": cfg.simulation.agents,
        "days": cfg.simulation.days,
        "start_date": cfg.simulation.start_date,
        "granularity_minutes": cfg.simulation.granularity_minutes,
        "car_speed_kmh": cfg.simulation.car_speed_kmh,
    }
    return Experiment(
        id=config_path.stem,
        config_path=config_path,
        label=cfg.comparison.label,
        synthetic_output=synthetic_output,
        observed_path=observed_path,
        params=params,
        runs=_discover_runs(synthetic_output),
    )


def list_experiments() -> list[Experiment]:
    if not CONFIGS_DIR.is_dir():
        return []
    return [_load_experiment(p) for p in sorted(CONFIGS_DIR.glob("*.yaml"))]


def get_experiment(exp_id: str) -> Optional[Experiment]:
    config_path = CONFIGS_DIR / f"{exp_id}.yaml"
    if not config_path.is_file():
        return None
    return _load_experiment(config_path)
