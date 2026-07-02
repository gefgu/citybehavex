"""Profile-driven CRP schedule selection.

Customers = days, tables = whole LLM diaries. Each simulated day an agent selects
one diary from a bank, weighted by:

* **popularity** — how many times this agent has already used that diary (n_k)
* **profile similarity** — cosine(profile_embedding, diary_embedding)

The weight for candidate k is: ``w_k = count_k * exp(s_k / T)``

where:
- ``count_k = n_k`` if the agent has used diary k before, else ``alpha``
- ``s_k = cosine(agent_profile_embedding, diary_embedding[k])``
- ``T`` and ``alpha`` are drawn per-agent from Beta distributions

This is a Chinese Restaurant Process with semantic smoothing. Weekday and weekend
are **hard-separated**: a weekday only draws from the weekday bank and vice versa.

Fixed purpose→code map in ``diary_to_abs_locs``:
  HOME=0, WORK=1, OTHER=2
This allows Rust to short-circuit WORK episodes to a per-agent persistent work
tile, mirroring the HOME short-circuit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed as _delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from citybehavex.embedding import embed_diaries
from citybehavex.embedding.config import EmbeddingConfig
from citybehavex.llm_diaries import Diary, DiaryBatch
from citybehavex.schedules.config import ScheduleConfig

DiaryArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Fixed purpose→code map. HOME always stays 0 (Rust invariant).
# WORK = 1 is reserved so Rust can pin WORK episodes to a persistent work tile.
_PURPOSE_CODE: dict[str, int] = {
    "HOME": 0,
    "WORK": 1,
    "OTHER": 2,
}


def diary_to_abs_locs(diary: Diary, slots_per_day: int, granularity_minutes: int) -> np.ndarray:
    """Per-slot abstract-location mask for one diary using a **fixed** purpose→code map.

    HOME slots are ``0``; WORK slots are ``1`` (so Rust can pin them to a
    per-agent work tile); other purposes get stable codes 2–6.
    """
    locs = np.zeros(slots_per_day, dtype=np.int32)
    episodes = diary.episodes
    ep_i = 0
    for slot in range(slots_per_day):
        minute = slot * granularity_minutes
        while ep_i < len(episodes) and episodes[ep_i].end_minutes <= minute:
            ep_i += 1
        if ep_i >= len(episodes):
            break
        purpose = episodes[ep_i].purpose
        code = _PURPOSE_CODE.get(purpose, 2)
        locs[slot] = code
    return locs


@dataclass
class DiaryBank:
    """Combined weekday+weekend diary bank with embeddings and slot masks."""

    diaries: list[Diary]
    is_weekend: np.ndarray          # bool[K]
    embeddings: np.ndarray | None   # float32[K, dim] or None (fallback: identity)
    slot_locs: np.ndarray           # int32[K, slots_per_day]
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
    embedded = embeddings is not None

    slot_locs = np.stack(
        [diary_to_abs_locs(d, slots_per_day, granularity_minutes) for d in diaries]
    ).astype(np.int32)

    return DiaryBank(
        diaries=diaries,
        is_weekend=np.asarray(is_weekend, dtype=bool),
        embeddings=embeddings,
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
    profile_embeddings: np.ndarray | None = None,
) -> tuple[DiaryArrays, np.ndarray]:
    """Run profile-driven CRP schedule selection for every agent and day.

    Weight formula per candidate k on day t for agent a:
      ``w_k = count_k * exp(s_k / T_a)``
    where ``count_k = n_k`` (times used so far) for previously-used diaries,
    ``count_k = alpha_a`` for new ones, and ``s_k`` is the cosine similarity
    between the agent's profile embedding and diary k's embedding.

    When ``profile_embeddings`` is ``None`` (embeddings off), ``s_k`` is treated
    as a constant → weights reduce to pure popularity-weighted CRP.

    Returns ``(diary_arrays, chosen)`` where ``chosen[agent, day]`` is the global
    bank index of the diary that agent used that day.
    """
    slots_per_day = bank.slots_per_day
    slot_seconds = bank.granularity_minutes * 60
    slot_offsets = np.arange(slots_per_day, dtype=np.int64) * slot_seconds

    # Day metadata.
    day_ts = np.empty(days, dtype=np.int64)
    day_is_weekend = np.empty(days, dtype=bool)
    for d in range(days):
        day = pd.Timestamp(start_date) + pd.Timedelta(days=d)
        day_ts[d] = int(day.timestamp())
        day_is_weekend[d] = day.dayofweek >= 5

    weekday_idx = np.flatnonzero(~bank.is_weekend)
    weekend_idx = np.flatnonzero(bank.is_weekend)

    # Pre-compute per-agent profile↔diary similarities [n_agents, K].
    # Falls back to zeros (constant) when embeddings are unavailable.
    K = len(bank.diaries)
    if profile_embeddings is not None and bank.embeddings is not None:
        # Both are L2-normalized: dot product = cosine similarity.
        agent_diary_sim = np.clip(
            profile_embeddings.astype(np.float64) @ bank.embeddings.astype(np.float64).T,
            0.0, 1.0,
        )  # [n_agents, K]
    else:
        agent_diary_sim = np.zeros((n_agents, K), dtype=np.float64)

    chosen = np.empty((n_agents, days), dtype=np.int64)

    slots = days * slots_per_day

    def _process_agent(agent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(np.random.SeedSequence([int(random_state), agent]))
        T_a = float(rng.beta(params.temperature_beta_a, params.temperature_beta_b))
        T_a = max(T_a, 1e-6)
        alpha_a = float(rng.beta(params.alpha_beta_a, params.alpha_beta_b))
        alpha_a = max(alpha_a, 1e-6)

        usage_counts = np.zeros(K, dtype=np.float64)
        agent_sims = agent_diary_sim[agent]

        agent_locs = np.empty(slots, dtype=np.int32)
        agent_ts = np.empty(slots, dtype=np.int64)
        agent_chosen = np.empty(days, dtype=np.int64)

        for d in range(days):
            is_we = bool(day_is_weekend[d])
            candidates = weekend_idx if is_we else weekday_idx
            sims_k = agent_sims[candidates]
            counts_k = usage_counts[candidates]
            effective_counts = np.where(counts_k > 0, counts_k, alpha_a)
            w = effective_counts * np.exp(sims_k / T_a)
            total = w.sum()
            if not np.isfinite(total) or total <= 0:
                pick = int(candidates[rng.integers(len(candidates))])
            else:
                pick = int(candidates[rng.choice(len(candidates), p=w / total)])
            agent_chosen[d] = pick
            usage_counts[pick] += 1.0
            base = d * slots_per_day
            agent_locs[base : base + slots_per_day] = bank.slot_locs[pick]
            agent_ts[base : base + slots_per_day] = day_ts[d] + slot_offsets

        return agent_locs, agent_ts, agent_chosen

    if _JOBLIB_AVAILABLE and n_agents > 4:
        results = Parallel(n_jobs=-1, backend="loky")(
            _delayed(_process_agent)(agent) for agent in range(n_agents)
        )
    else:
        results = [_process_agent(agent) for agent in range(n_agents)]

    diary_timestamps = np.empty(n_agents * slots, dtype=np.int64)
    diary_abs_locs = np.empty(n_agents * slots, dtype=np.int32)
    d_starts = np.arange(n_agents, dtype=np.int64) * slots
    d_ends = d_starts + slots
    for agent, (agent_locs, agent_ts, agent_chosen) in enumerate(results):
        off = agent * slots
        diary_abs_locs[off : off + slots] = agent_locs
        diary_timestamps[off : off + slots] = agent_ts
        chosen[agent] = agent_chosen

    diary_arrays: DiaryArrays = (diary_timestamps, diary_abs_locs, d_starts, d_ends)
    return diary_arrays, chosen
