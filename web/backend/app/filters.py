"""Day-type/time-of-day/special-day filter metadata and application.

Shared between ``payload.py`` (tier-2 comparison combination) and
``features.py`` (tier-1 per-file feature extraction) -- split out on its own
so neither module has to import the other just to filter a dataframe by the
same filter metadata shape.
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

FILTERS = [
    {"key": "all", "label": "All", "kind": "base"},
    {"key": "weekday", "label": "Weekday", "kind": "day"},
    {"key": "weekend", "label": "Weekend", "kind": "day"},
]

_TIME_FILTERS = [
    {"key": "morning", "label": "Morning", "kind": "time", "start": 6, "end": 12},
    {"key": "afternoon", "label": "Afternoon", "kind": "time", "start": 12, "end": 18},
    {"key": "evening", "label": "Evening", "kind": "time", "start": 18, "end": 24},
    {"key": "night", "label": "Night", "kind": "time", "start": 0, "end": 6},
]


def _special_day_filters(special_days: Optional[list[dict[str, str]]]) -> list[dict[str, Any]]:
    """Turn config-declared special days (e.g. an "emergency" date range) into
    the same filter-metadata shape as the built-in weekday/weekend filters."""
    return [
        {
            "key": sd["name"],
            "label": sd["name"].replace("_", " ").title(),
            "kind": "date_range",
            "start": sd["start_date"],
            "end": sd["end_date"],
        }
        for sd in (special_days or [])
    ]


def _empty_group(meta: dict[str, Any], blocks_key: str = "blocks") -> dict[str, Any]:
    return {"filter_key": meta["key"], "filter_label": meta["label"], blocks_key: {}}


def _to_datetime(col: pl.Series) -> pl.Series:
    """Coerce a datetime-ish column (string or already-parsed) to polars
    ``Datetime``, coercing unparsable values to null."""
    if col.dtype == pl.Utf8:
        return col.str.to_datetime(strict=False)
    if isinstance(col.dtype, pl.Datetime):
        return col
    return col.cast(pl.Datetime, strict=False)


def _filter_df(df: pl.DataFrame, datetime_col: str | None, meta: dict[str, Any]) -> pl.DataFrame:
    if meta["key"] == "all" or not datetime_col or datetime_col not in df.columns:
        return df
    dt = _to_datetime(df[datetime_col])
    if meta["kind"] == "day":
        mask = dt.dt.weekday() < 6
        if meta["key"] == "weekend":
            mask = ~mask
    elif meta["kind"] == "date_range":
        day = dt.dt.truncate("1d")
        start = pl.Series([meta["start"]]).str.to_datetime(strict=False)[0]
        end = pl.Series([meta["end"]]).str.to_datetime(strict=False)[0]
        mask = (day >= start) & (day <= end)
    else:
        hour = dt.dt.hour()
        mask = (hour >= int(meta["start"])) & (hour < int(meta["end"]))
    return df.filter(mask.fill_null(False))


def _filter_visits(visits: pl.DataFrame | None, meta: dict[str, Any]) -> pl.DataFrame | None:
    if visits is None:
        return None
    return _filter_df(visits, "start_timestamp", meta)
