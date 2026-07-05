"""Discover experiments from ``configs/*.yaml`` and resolve their runs.

Each YAML config is one experiment. Simulation outputs are timestamp-stamped at
write time (``_YYYYMMDDTHHMMSS`` before the extension), so the concrete runs are
found by globbing the stem of ``simulation.output`` rather than trusting the
literal path.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from citybehavex.config.root import CityBehavExConfig
from citybehavex.config import load_config

from .config import CONFIGS_DIR, REPO_ROOT
from .datasource import run_summary

_STAMP_RE = re.compile(r"_(\d{8}T\d{6})$")


class ExperimentMutationError(ValueError):
    """Raised when an experiment edit/archive/delete cannot be applied."""


def _resolve(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _display_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class Run:
    run_id: str            # timestamp stamp, or "base" for the unstamped file
    path: Path
    mtime: float

    @property
    def encounters_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_encounters{self.path.suffix}")

    @property
    def moving_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_moving{self.path.suffix}")

    @property
    def activities_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_activities{self.path.suffix}")

    @property
    def crp_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_crp{self.path.suffix}")

    @property
    def social_network_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_social_network.json")

    def to_dict(self, with_summary: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "path": _display_path(self.path),
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

    Excludes the ``*_encounters.parquet``, ``*_moving.parquet`` and ``*_crp.parquet``
    siblings written alongside trajectories.
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
        if name.endswith("_encounters") or name.endswith("_moving") or name.endswith("_crp"):
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
    time_use_path: Optional[Path]
    time_use_label: str
    time_use_country: Optional[str]
    time_use_survey: Optional[int]
    time_use_weight_col: str
    profiles_enabled: bool
    profiles_output: Optional[Path]
    profiles_path: Optional[Path]
    road_nodes_path: Optional[Path]
    road_edges_path: Optional[Path]
    network_validation_config: Any
    params: dict[str, Any]
    special_days: list[dict[str, str]]
    runs: list[Run]

    def to_dict(self, with_summary: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": _display_path(self.config_path),
            "label": self.label,
            "simulation_output": _display_path(self.synthetic_output),
            "observed_path": _display_path(self.observed_path),
            "observed_exists": bool(self.observed_path and self.observed_path.exists()),
            "time_use_path": _display_path(self.time_use_path),
            "time_use_exists": bool(self.time_use_path and self.time_use_path.exists()),
            "time_use_label": self.time_use_label,
            "time_use_country": self.time_use_country,
            "time_use_survey": self.time_use_survey,
            "time_use_weight_col": self.time_use_weight_col,
            "network_validation": (
                self.network_validation_config.model_dump()
                if hasattr(self.network_validation_config, "model_dump")
                else self.network_validation_config
            ),
            "profiles_enabled": self.profiles_enabled,
            "profiles_output": _display_path(self.profiles_output),
            "profiles_path": _display_path(self.profiles_path),
            "profiles_exists": bool(self.profiles_path and self.profiles_path.exists()),
            "road_network_available": bool(
                self.road_nodes_path
                and self.road_nodes_path.exists()
                and self.road_edges_path
                and self.road_edges_path.exists()
            ),
            "params": self.params,
            "special_days": self.special_days,
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
    time_use_path = _resolve(cfg.comparison.time_use_path)
    profiles_output = _resolve(cfg.profiles.output)
    profiles_path = profiles_output if cfg.profiles.enabled else None
    road_distance_enabled = cfg.road_network.enabled and cfg.comparison.road_network_distance
    road_nodes_path = _resolve(cfg.road_network.nodes_output) if road_distance_enabled else None
    road_edges_path = _resolve(cfg.road_network.edges_output) if road_distance_enabled else None
    params = {
        "agents": cfg.simulation.agents,
        "days": cfg.simulation.days,
        "start_date": cfg.simulation.start_date,
        "granularity_minutes": cfg.simulation.granularity_minutes,
        "car_speed_kmh": cfg.simulation.car_speed_kmh,
        "social_graph_k": cfg.simulation.social_graph_k,
        "rho": cfg.simulation.rho,
        "gamma": cfg.simulation.gamma,
        "alpha": cfg.simulation.alpha,
        "dt_update_mob_sim_hours": cfg.simulation.dt_update_mob_sim_hours,
        "indipendency_window_hours": cfg.simulation.indipendency_window_hours,
    }
    special_days = [
        {"name": sd.name, "start_date": sd.start_date, "end_date": sd.end_date}
        for sd in cfg.diaries.special_days
    ]
    return Experiment(
        id=config_path.stem,
        config_path=config_path,
        label=cfg.comparison.label,
        synthetic_output=synthetic_output,
        observed_path=observed_path,
        time_use_path=time_use_path,
        time_use_label=cfg.comparison.time_use_label,
        time_use_country=cfg.comparison.time_use_country,
        time_use_survey=cfg.comparison.time_use_survey,
        time_use_weight_col=cfg.comparison.time_use_weight_col,
        profiles_enabled=cfg.profiles.enabled,
        profiles_output=profiles_output,
        profiles_path=profiles_path,
        road_nodes_path=road_nodes_path,
        road_edges_path=road_edges_path,
        network_validation_config=cfg.comparison.network_validation,
        params=params,
        special_days=special_days,
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


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ExperimentMutationError("experiment config must contain a YAML mapping")
    return raw


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.setdefault(name, {})
    if not isinstance(value, dict):
        raise ExperimentMutationError(f"{name!r} config section must be a mapping")
    return value


def update_experiment(exp_id: str, updates: dict[str, Any]) -> Experiment:
    config_path = CONFIGS_DIR / f"{exp_id}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(exp_id)

    raw = _read_yaml_mapping(config_path)
    simulation = _section(raw, "simulation")
    comparison = _section(raw, "comparison")
    profiles = _section(raw, "profiles")

    field_map = {
        "agents": (simulation, "agents"),
        "days": (simulation, "days"),
        "start_date": (simulation, "start_date"),
        "granularity_minutes": (simulation, "granularity_minutes"),
        "car_speed_kmh": (simulation, "car_speed_kmh"),
        "simulation_output": (simulation, "output"),
        "label": (comparison, "label"),
        "observed_path": (comparison, "path"),
        "time_use_path": (comparison, "time_use_path"),
        "time_use_label": (comparison, "time_use_label"),
        "time_use_country": (comparison, "time_use_country"),
        "time_use_survey": (comparison, "time_use_survey"),
        "time_use_weight_col": (comparison, "time_use_weight_col"),
        "profiles_enabled": (profiles, "enabled"),
        "profiles_output": (profiles, "output"),
    }
    for api_field, value in updates.items():
        if api_field not in field_map:
            continue
        section, config_field = field_map[api_field]
        section[config_field] = value

    try:
        CityBehavExConfig.model_validate(raw)
    except ValidationError as exc:
        raise ExperimentMutationError(str(exc)) from exc

    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return _load_experiment(config_path)


def archive_experiment(exp_id: str) -> Path:
    config_path = CONFIGS_DIR / f"{exp_id}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(exp_id)

    archived_dir = CONFIGS_DIR / ".archived"
    archived_dir.mkdir(parents=True, exist_ok=True)
    archived_path = archived_dir / config_path.name
    if archived_path.exists():
        raise ExperimentMutationError(f"archived config already exists: {archived_path.name}")
    shutil.move(str(config_path), str(archived_path))
    return archived_path


def delete_run(exp_id: str, run_id: str) -> list[Path]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise FileNotFoundError(exp_id)

    run = experiment.run(run_id)
    if run is None:
        raise FileNotFoundError(run_id)

    deleted: list[Path] = []
    for path in (run.path, run.encounters_path, run.moving_path, run.activities_path, run.crp_path, run.social_network_path):
        if path.exists():
            path.unlink()
            deleted.append(path)
    return deleted
