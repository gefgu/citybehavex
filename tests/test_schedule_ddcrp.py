from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from citybehavex import embeddings as emb
from citybehavex.config import EmbeddingConfig, ScheduleConfig
from citybehavex.llm_diaries import Diary, DiaryBatch, LocationCountDistribution
from citybehavex.schedule_ddcrp import (
    DiaryBank,
    build_ddcrp_diary,
    build_diary_bank,
    diary_to_abs_locs,
)

GRAN = 60
SLOTS = 1440 // GRAN
MONDAY = pd.Timestamp("2026-06-15")  # a Monday -> Mon..Sun over 7 days


def _diary(diary_id: str, away_purpose: str, away_start: str, away_end: str) -> Diary:
    return Diary.model_validate(
        {
            "diary_id": diary_id,
            "episodes": [
                {"start": "00:00", "end": away_start, "purpose": "HOME"},
                {"start": away_start, "end": away_end, "purpose": away_purpose},
                {"start": away_end, "end": "24:00", "purpose": "HOME"},
            ],
        }
    )


def _diary_list(n: int, away_purpose: str = "WORK") -> list[Diary]:
    return [_diary(f"d-{i:03d}", away_purpose, "09:00", "17:00") for i in range(n)]


def _batch(diaries: list[Diary]) -> DiaryBatch:
    return DiaryBatch.model_validate(
        {
            "representative_day": "2026-01-01",
            "location_count_distribution": LocationCountDistribution(
                mu=1.0, sigma=0.5, max_locations=6
            ).model_dump(),
            "target_location_counts": [2] * len(diaries),
            "diaries": diaries,
        }
    )


def _batches(n_weekday: int = 10, n_weekend: int = 10) -> dict[str, DiaryBatch]:
    weekday = [
        _diary(f"wd-{i:03d}", "WORK", "09:00", "17:00") for i in range(n_weekday)
    ]
    weekend = [
        _diary(f"we-{i:03d}", "LEISURE", "12:00", "18:00") for i in range(n_weekend)
    ]
    return {"weekday": _batch(weekday), "weekend": _batch(weekend)}


# --- diary_to_abs_locs -----------------------------------------------------


def test_diary_to_abs_locs_home_zero_away_nonzero():
    diary = _diary("x", "WORK", "09:00", "17:00")
    locs = diary_to_abs_locs(diary, SLOTS, GRAN)
    assert locs.shape == (SLOTS,)
    assert locs.dtype == np.int32
    assert np.all(locs[:9] == 0)  # 00:00-09:00 HOME
    assert np.all(locs[9:17] != 0)  # 09:00-17:00 WORK
    assert np.all(locs[17:] == 0)  # 17:00-24:00 HOME


# --- embeddings cache + fallback -------------------------------------------


def test_embed_diaries_disabled_returns_none():
    diaries = _batches()["weekday"].diaries
    assert emb.embed_diaries(diaries, EmbeddingConfig(enabled=False)) is None


def test_embed_diaries_cache_round_trip(tmp_path, monkeypatch):
    diaries = _diary_list(3)
    calls = {"n": 0}

    def fake_post(base_url, model, texts, *, api_key, timeout):
        calls["n"] += 1
        # Deterministic non-normalized vectors.
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0

    monkeypatch.setattr(emb, "_server_reachable", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_post_embeddings", fake_post)

    cfg = EmbeddingConfig(
        base_url="http://localhost:9", dimensions=4, cache_dir=str(tmp_path)
    )
    first = emb.embed_diaries(diaries, cfg)
    assert first is not None and first.shape == (3, 4)
    # Rows are L2-normalized.
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)

    second = emb.embed_diaries(diaries, cfg)
    np.testing.assert_array_equal(first, second)
    assert calls["n"] == 1  # second call served entirely from cache


def test_embed_diaries_returns_none_on_server_failure():
    cfg = EmbeddingConfig(base_url=None, auto_launch=False)
    assert emb.embed_diaries(_diary_list(2), cfg) is None


# --- build_diary_bank ------------------------------------------------------


def test_build_diary_bank_identity_when_embeddings_off():
    bank = build_diary_bank(_batches(10, 10), EmbeddingConfig(enabled=False), GRAN)
    assert len(bank.diaries) == 20
    assert int((~bank.is_weekend).sum()) == 10
    assert int(bank.is_weekend.sum()) == 10
    assert bank.embedded is False
    np.testing.assert_array_equal(bank.sim, np.eye(20, dtype=np.float32))
    assert bank.slot_locs.shape == (20, SLOTS)


# --- build_ddcrp_diary -----------------------------------------------------


def _bank(n_weekday=10, n_weekend=10) -> DiaryBank:
    return build_diary_bank(
        _batches(n_weekday, n_weekend), EmbeddingConfig(enabled=False), GRAN
    )


def test_hard_weekday_weekend_filter():
    bank = _bank()
    weekday_set = set(np.flatnonzero(~bank.is_weekend).tolist())
    weekend_set = set(np.flatnonzero(bank.is_weekend).tolist())
    (_, _, _, _), chosen = build_ddcrp_diary(
        bank, MONDAY, days=7, n_agents=50, random_state=42, params=ScheduleConfig()
    )
    for d in range(7):
        is_we = (MONDAY + pd.Timedelta(days=d)).dayofweek >= 5
        valid = weekend_set if is_we else weekday_set
        assert set(chosen[:, d].tolist()) <= valid


def test_determinism():
    bank = _bank()
    params = ScheduleConfig()
    (_, _, _, _), c1 = build_ddcrp_diary(bank, MONDAY, 7, 30, 7, params)
    (_, _, _, _), c2 = build_ddcrp_diary(bank, MONDAY, 7, 30, 7, params)
    np.testing.assert_array_equal(c1, c2)


def test_diary_arrays_shapes_and_mask():
    bank = _bank()
    days, agents = 7, 12
    (ts, locs, starts, ends), chosen = build_ddcrp_diary(
        bank, MONDAY, days, agents, 1, ScheduleConfig()
    )
    expected = agents * days * SLOTS
    assert ts.shape == (expected,) and locs.shape == (expected,)
    assert starts.shape == (agents,) and ends.shape == (agents,)
    assert starts[0] == 0 and ends[-1] == expected
    assert np.all(locs >= 0)  # pure home/away mask
    assert chosen.shape == (agents, days)
    # First slot of agent 0 is midnight of the start date.
    assert ts[0] == int(MONDAY.timestamp())


def test_habit_pure_return_sticks_to_one_schedule_per_type():
    # rho=0 -> no exploration mass; implanted memory + identity sim means each
    # agent should reuse exactly one weekday and one weekend diary all week.
    bank = _bank()
    params = ScheduleConfig(rho=0.0, lam=1.0, implant_memory=True)
    (_, _, _, _), chosen = build_ddcrp_diary(bank, MONDAY, 7, 40, 3, params)
    for agent in range(chosen.shape[0]):
        row = chosen[agent]
        weekday_used = {
            int(row[d]) for d in range(7)
            if (MONDAY + pd.Timedelta(days=d)).dayofweek < 5
        }
        weekend_used = {
            int(row[d]) for d in range(7)
            if (MONDAY + pd.Timedelta(days=d)).dayofweek >= 5
        }
        assert len(weekday_used) == 1
        assert len(weekend_used) == 1


def test_exploration_spreads_across_bank():
    # High rho, gamma=0, no implant, no recency -> near-uniform exploration.
    bank = _bank(n_weekday=10, n_weekend=10)
    params = ScheduleConfig(rho=5.0, gamma=0.0, lam=0.0, implant_memory=False)
    (_, _, _, _), chosen = build_ddcrp_diary(bank, MONDAY, 7, 200, 9, params)
    weekday_cols = [d for d in range(7) if (MONDAY + pd.Timedelta(days=d)).dayofweek < 5]
    used = set(chosen[:, weekday_cols].ravel().tolist())
    assert len(used) >= 6  # most of the 10-diary weekday bank gets explored
