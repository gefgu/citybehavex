from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .llm_diaries import DiaryBatch


def diary_batch_to_markov_training(
    batch: DiaryBatch,
    *,
    representative_day: str,
    granularity_minutes: int = 60,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Expand diary episodes into a per-slot training frame for the Markov learner.

    Each episode is expanded into one row per ``granularity_minutes`` slot it
    spans. Episodes shorter than one slot still emit (at least) their starting
    slot, so sub-slot activities are not silently dropped.
    """
    rows: list[dict[str, object]] = []
    base_day = pd.Timestamp(representative_day)
    freq = f"{granularity_minutes}min"

    for uid, diary in enumerate(batch.diaries, start=1):
        for episode in diary.episodes:
            start = base_day + pd.Timedelta(minutes=episode.start_minutes)
            end = base_day + pd.Timedelta(minutes=episode.end_minutes)
            first = start.floor(freq)
            last = (end - pd.Timedelta(microseconds=1)).floor(freq)
            if last < first:
                last = first
            for timestamp in pd.date_range(first, last, freq=freq):
                location = (
                    "home"
                    if episode.purpose == "HOME"
                    else f"{episode.purpose.lower()}_{uid}"
                )
                rows.append(
                    {
                        "uid": uid,
                        "datetime": timestamp,
                        "location": location,
                        "purpose": episode.purpose,
                    }
                )

    training = pd.DataFrame(rows)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        training.to_parquet(path, index=False)
    return training


def annotate_trajectory_purposes(
    traj_df: pd.DataFrame,
    batch: DiaryBatch,
    *,
    uid_col: str = "uid",
    datetime_col: str = "datetime",
    weekend_batch: Optional[DiaryBatch] = None,
) -> pd.DataFrame:
    """Assign a purpose to each trajectory record from the diary episode active at
    that clock time. When ``weekend_batch`` is given, weekend rows (Sat/Sun) are
    labelled from it and weekday rows from ``batch``.
    """
    out = traj_df.copy()
    if uid_col not in out.columns or datetime_col not in out.columns:
        return out

    def diary_lookup(b: DiaryBatch) -> dict[int, object]:
        return {index + 1: diary for index, diary in enumerate(b.diaries)}

    weekday_by_uid = diary_lookup(batch)
    weekend_by_uid = diary_lookup(weekend_batch) if weekend_batch is not None else weekday_by_uid
    n_weekday = len(batch.diaries)
    n_weekend = len(weekend_batch.diaries) if weekend_batch is not None else n_weekday

    purposes: list[str] = []
    for _, row in out.iterrows():
        uid = int(row[uid_col])
        ts = pd.Timestamp(row[datetime_col])
        if ts.dayofweek >= 5:
            diary = weekend_by_uid.get(((uid - 1) % n_weekend) + 1)
        else:
            diary = weekday_by_uid.get(((uid - 1) % n_weekday) + 1)
        minute = ts.hour * 60 + ts.minute
        purpose = "OTHER"
        if diary is not None:
            for episode in diary.episodes:
                if episode.start_minutes <= minute < episode.end_minutes:
                    purpose = episode.purpose
                    break
        purposes.append(purpose)
    out["purpose"] = purposes
    return out
