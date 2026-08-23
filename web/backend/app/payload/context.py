"""Shared comparison context for section payload builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ARTIFACT_SCHEMA_VERSION = "v3"


def path_mtime(path: str | Path | None) -> int | str:
    if path is None:
        return "none"
    p = Path(path)
    return int(p.stat().st_mtime) if p.exists() else "missing"


def _hashable_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _hashable_config(value.model_dump())
    if isinstance(value, SimpleNamespace):
        return _hashable_config(vars(value))
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable_config(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_config(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable_config(v) for v in value))
    return repr(value)


@dataclass(frozen=True)
class ComparisonContext:
    synthetic_path: str
    observed_path: Optional[str]
    observed_label: str
    synthetic_activities_path: Optional[str] = None
    time_use_path: Optional[str] = None
    time_use_label: str = "time-use"
    time_use_country: Optional[str] = None
    time_use_survey: Optional[int] = None
    time_use_weight_col: str = "propwt"
    transport_spatial_config: object | None = None
    evaluation_adaptation_config: object | None = None
    special_days: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def from_kwargs(
        cls,
        *,
        synthetic_path: str,
        observed_path: Optional[str],
        observed_label: str,
        synthetic_activities_path: Optional[str] = None,
        time_use_path: Optional[str] = None,
        time_use_label: str = "time-use",
        time_use_country: Optional[str] = None,
        time_use_survey: Optional[int] = None,
        time_use_weight_col: str = "propwt",
        transport_spatial_config: object | None = None,
        evaluation_adaptation_config: object | None = None,
        special_days: Optional[list[dict[str, str]]] = None,
    ) -> "ComparisonContext":
        return cls(
            synthetic_path=synthetic_path,
            observed_path=observed_path,
            observed_label=observed_label,
            synthetic_activities_path=synthetic_activities_path,
            time_use_path=time_use_path,
            time_use_label=time_use_label,
            time_use_country=time_use_country,
            time_use_survey=time_use_survey,
            time_use_weight_col=time_use_weight_col,
            transport_spatial_config=transport_spatial_config,
            evaluation_adaptation_config=evaluation_adaptation_config,
            special_days=tuple(
                (sd["name"], sd["start_date"], sd["end_date"]) for sd in (special_days or [])
            ),
        )

    def special_day_dicts(self) -> list[dict[str, str]]:
        return [
            {"name": name, "start_date": start, "end_date": end}
            for name, start, end in self.special_days
        ]

    def artifact_key(self, filter_key: str) -> tuple[object, ...]:
        return (
            "comparison-filter",
            ARTIFACT_SCHEMA_VERSION,
            self.synthetic_path,
            path_mtime(self.synthetic_path),
            self.observed_path,
            path_mtime(self.observed_path),
            self.synthetic_activities_path,
            path_mtime(self.synthetic_activities_path),
            self.time_use_path,
            path_mtime(self.time_use_path),
            self.observed_label,
            self.time_use_label,
            self.time_use_country,
            self.time_use_survey,
            self.time_use_weight_col,
            _hashable_config(self.transport_spatial_config),
            _hashable_config(self.evaluation_adaptation_config),
            self.special_days,
            filter_key,
        )
