from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from .llm_diaries import DiaryBatch

if TYPE_CHECKING:
    from .schedule_ddcrp import DiaryBank


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


def annotate_trajectory_purposes_ddcrp(
    traj_df: pd.DataFrame,
    bank: "DiaryBank",
    chosen: np.ndarray,
    start_date: pd.Timestamp,
    *,
    uid_col: str = "uid",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Assign a purpose to each trajectory record from the diary that the agent
    actually used that day under ddCRP selection.

    ``chosen[agent, day_index]`` is the global bank index of the diary used by the
    agent (1-based ``uid``) on ``day_index = (date - start_date).days``.
    """
    out = traj_df.copy()
    if uid_col not in out.columns or datetime_col not in out.columns:
        return out

    start_day = pd.Timestamp(start_date).normalize()
    n_agents, n_days = chosen.shape

    purposes: list[str] = []
    for _, row in out.iterrows():
        ts = pd.Timestamp(row[datetime_col])
        agent = int(row[uid_col]) - 1
        day_index = (ts.normalize() - start_day).days
        purpose = "OTHER"
        if 0 <= agent < n_agents and 0 <= day_index < n_days:
            diary = bank.diaries[int(chosen[agent, day_index])]
            minute = ts.hour * 60 + ts.minute
            for episode in diary.episodes:
                if episode.start_minutes <= minute < episode.end_minutes:
                    purpose = episode.purpose
                    break
        purposes.append(purpose)
    out["purpose"] = purposes
    return out
