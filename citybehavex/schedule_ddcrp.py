"""Distance-dependent Chinese Restaurant Process (ddCRP) schedule selection.

Customers = days, tables = whole LLM diaries. Each simulated day an agent selects
one entire diary from a bank, weighted by:

* **calendar recency** -- ``exp(-lam * (t - t'))`` over past same-type days (habit), and
* **semantic similarity** -- cosine similarity between candidate diaries and the
  diaries the agent used on those past days (cross-schedule generalization + cold
  start), from ``nomic-embed-text-v2-moe`` embeddings.

Weekday and weekend are **hard-separated**: a weekday only draws from the weekday
bank and a weekend only from the weekend bank. An exploration mass ``rho * S^-gamma``
(``S`` = distinct schedules adopted of that type) is spread over not-yet-used
candidates, so new-but-similar diaries are reachable.

The output is the same ``DiaryArrays`` tuple the Rust core consumes (a per-slot
home(0)/away(non-zero) mask), plus a ``[n_agents, days]`` map of the diary each
agent used each day for purpose annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EmbeddingConfig, ScheduleConfig
from .embeddings import cosine_sim_matrix, embed_diaries
from .llm_diaries import Diary, DiaryBatch
from .trip_ditras import DiaryArrays


def diary_to_abs_locs(diary: Diary, slots_per_day: int, granularity_minutes: int) -> np.ndarray:
    """Per-slot abstract-location mask for one diary.

    HOME slots are ``0`` (home); each distinct non-HOME purpose gets a distinct
    positive code. Both Rust cores treat the array purely as a home/away mask (only
    ``== 0`` is tested), so the exact positive value is informational; physical
    location persistence across slots/days is the EPR's job.
    """
    locs = np.zeros(slots_per_day, dtype=np.int32)
    purpose_code: dict[str, int] = {}
    episodes = diary.episodes
    ep_i = 0
    for slot in range(slots_per_day):
        minute = slot * granularity_minutes
        # Episodes are ordered and gap-free; advance to the one covering `minute`.
        while ep_i < len(episodes) and episodes[ep_i].end_minutes <= minute:
            ep_i += 1
        if ep_i >= len(episodes):
            break
        purpose = episodes[ep_i].purpose
        if purpose == "HOME":
            continue
        code = purpose_code.get(purpose)
        if code is None:
            code = len(purpose_code) + 1
            purpose_code[purpose] = code
        locs[slot] = code
    return locs


@dataclass
class DiaryBank:
    """Combined weekday+weekend diary bank with precomputed similarity and slots."""

    diaries: list[Diary]
    is_weekend: np.ndarray  # bool[K]
    sim: np.ndarray  # float[K, K] cosine similarity (identity if no embeddings)
    slot_locs: np.ndarray  # int32[K, slots_per_day]
    slots_per_day: int
    granularity_minutes: int
    embedded: bool


def build_diary_bank(
    diary_batches: dict[str, DiaryBatch],
    embedding_config: EmbeddingConfig,
    granularity_minutes: int,
) -> DiaryBank:
    """Concatenate the weekday/weekend banks, embed them, precompute slot masks."""
    if 1440 % granularity_minutes != 0:
        raise ValueError("granularity_minutes must divide 1440")
    slots_per_day = 1440 // granularity_minutes

    diaries: list[Diary] = []
    is_weekend: list[bool] = []
    for day_type in ("weekday", "weekend"):
        batch = diary_batches.get(day_type)
        if batch is None:
            continue
        for diary in batch.diaries:
            diaries.append(diary)
            is_weekend.append(day_type == "weekend")
    if not diaries:
        raise ValueError("diary bank is empty")

    embeddings = embed_diaries(diaries, embedding_config)
    if embeddings is not None:
        sim = cosine_sim_matrix(embeddings)
        embedded = True
    else:
        sim = np.eye(len(diaries), dtype=np.float32)
        embedded = False

    slot_locs = np.stack(
        [diary_to_abs_locs(d, slots_per_day, granularity_minutes) for d in diaries]
    ).astype(np.int32)

    return DiaryBank(
        diaries=diaries,
        is_weekend=np.asarray(is_weekend, dtype=bool),
        sim=sim,
        slot_locs=slot_locs,
        slots_per_day=slots_per_day,
        granularity_minutes=granularity_minutes,
        embedded=embedded,
    )


def build_ddcrp_diary(
    bank: DiaryBank,
    start_date: pd.Timestamp,
    days: int,
    n_agents: int,
    random_state: int,
    params: ScheduleConfig,
) -> tuple[DiaryArrays, np.ndarray]:
    """Run ddCRP schedule selection for every agent and day.

    Returns ``(diary_arrays, chosen)`` where ``chosen[agent, day]`` is the global
    bank index of the diary that agent used that day.
    """
    slots_per_day = bank.slots_per_day
    slot_seconds = bank.granularity_minutes * 60
    slot_offsets = np.arange(slots_per_day, dtype=np.int64) * slot_seconds

    # Day metadata, computed once.
    day_ts = np.empty(days, dtype=np.int64)
    day_is_weekend = np.empty(days, dtype=bool)
    for d in range(days):
        day = pd.Timestamp(start_date) + pd.Timedelta(days=d)
        day_ts[d] = int(day.timestamp())
        day_is_weekend[d] = day.dayofweek >= 5

    weekday_idx = np.flatnonzero(~bank.is_weekend)
    weekend_idx = np.flatnonzero(bank.is_weekend)
    sim_t = bank.sim.astype(np.float64)

    chosen = np.empty((n_agents, days), dtype=np.int64)

    per_ts: list[np.ndarray] = []
    per_loc: list[np.ndarray] = []
    d_starts = np.empty(n_agents, dtype=np.int64)
    d_ends = np.empty(n_agents, dtype=np.int64)
    offset = 0

    sem_exp = 1.0 / params.semantic_temperature

    for agent in range(n_agents):
        rng = np.random.default_rng(np.random.SeedSequence([int(random_state), agent]))

        # memory: parallel arrays of (day_index, diary_idx); used_* track distinct
        # schedules for the exploration decay.
        mem_days: list[int] = []
        mem_diaries: list[int] = []
        used_weekday: set[int] = set()
        used_weekend: set[int] = set()

        if params.implant_memory:
            wd = int(weekday_idx[rng.integers(len(weekday_idx))]) if len(weekday_idx) else None
            we = int(weekend_idx[rng.integers(len(weekend_idx))]) if len(weekend_idx) else None
            for imp in (wd, we):
                if imp is not None:
                    mem_days.append(-1)  # "yesterday": mild discount on day 0
                    mem_diaries.append(imp)
                    (used_weekend if bank.is_weekend[imp] else used_weekday).add(imp)

        agent_locs = np.empty(days * slots_per_day, dtype=np.int32)
        agent_ts = np.empty(days * slots_per_day, dtype=np.int64)

        for d in range(days):
            is_we = bool(day_is_weekend[d])
            candidates = weekend_idx if is_we else weekday_idx
            used = used_weekend if is_we else used_weekday

            # Same-type, in-window memory entries.
            rel_k: list[int] = []
            rel_w: list[float] = []
            for md, mk in zip(mem_days, mem_diaries):
                if bank.is_weekend[mk] != is_we:
                    continue
                age = d - md
                if age > params.memory_window_days:
                    continue
                rel_k.append(mk)
                rel_w.append(np.exp(-params.lam * age))

            # Preferential-return weight per candidate (semantic x recency).
            if rel_k:
                sim_block = sim_t[np.ix_(candidates, rel_k)]
                sim_block = np.clip(sim_block, 0.0, 1.0) ** sem_exp
                w_ret = sim_block @ np.asarray(rel_w, dtype=np.float64)
            else:
                w_ret = np.zeros(len(candidates), dtype=np.float64)

            # Exploration mass spread over not-yet-used candidates.
            s_tau = max(len(used), 1)
            w_new = params.rho * (s_tau ** (-params.gamma))
            unused_mask = np.array([int(c) not in used for c in candidates])
            n_unused = int(unused_mask.sum())
            weight = w_ret.copy()
            if n_unused > 0:
                weight[unused_mask] += w_new / n_unused

            total = weight.sum()
            if not np.isfinite(total) or total <= 0:
                pick = int(candidates[rng.integers(len(candidates))])
            else:
                pick = int(candidates[rng.choice(len(candidates), p=weight / total)])

            chosen[agent, d] = pick
            mem_days.append(d)
            mem_diaries.append(pick)
            used.add(pick)

            base = d * slots_per_day
            agent_locs[base : base + slots_per_day] = bank.slot_locs[pick]
            agent_ts[base : base + slots_per_day] = day_ts[d] + slot_offsets

        per_ts.append(agent_ts)
        per_loc.append(agent_locs)
        d_starts[agent] = offset
        offset += agent_ts.size
        d_ends[agent] = offset

    diary_timestamps = (
        np.concatenate(per_ts) if per_ts else np.empty(0, dtype=np.int64)
    ).astype(np.int64)
    diary_abs_locs = (
        np.concatenate(per_loc) if per_loc else np.empty(0, dtype=np.int32)
    ).astype(np.int32)
    diary_arrays: DiaryArrays = (diary_timestamps, diary_abs_locs, d_starts, d_ends)
    return diary_arrays, chosen
