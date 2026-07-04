from __future__ import annotations

import numpy as np
import pandas as pd

from citybehavex.embedding import EmbeddingConfig
from citybehavex.embedding import service as emb
from citybehavex.schedules import ScheduleConfig
from citybehavex.llm_diaries import Diary, DiaryBatch, LocationCountDistribution
from citybehavex.schedules import (
    DiaryBank,
    build_ddcrp_diary,
    build_diary_bank,
    diary_to_abs_locs,
)

GRAN = 60
SLOTS = 1440 // GRAN
MONDAY = pd.Timestamp("2026-06-15")  # a Monday -> Mon..Sun over 7 days


def _calendar_day_types(start: pd.Timestamp, days: int) -> list[str]:
    return [
        "weekend" if (start + pd.Timedelta(days=d)).dayofweek >= 5 else "weekday"
        for d in range(days)
    ]


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
        _diary(f"we-{i:03d}", "OTHER", "12:00", "18:00") for i in range(n_weekend)
    ]
    return {"weekday": _batch(weekday), "weekend": _batch(weekend)}


# --- diary_to_abs_locs with fixed purpose→code map -------------------------


def test_diary_to_abs_locs_home_zero_work_one():
    diary = _diary("x", "WORK", "09:00", "17:00")
    locs = diary_to_abs_locs(diary, SLOTS, GRAN)
    assert locs.shape == (SLOTS,)
    assert locs.dtype == np.int32
    assert np.all(locs[:9] == 0)   # HOME
    assert np.all(locs[9:17] == 1)  # WORK must be 1 (fixed map)
    assert np.all(locs[17:] == 0)   # HOME


def test_diary_to_abs_locs_other_code():
    diary = _diary("x", "OTHER", "10:00", "18:00")
    locs = diary_to_abs_locs(diary, SLOTS, GRAN)
    assert np.all(locs[10:18] == 2)  # OTHER = 2 in fixed map


# --- embeddings cache + fallback -------------------------------------------


def test_embed_texts_disabled_returns_none():
    from citybehavex.embedding import embed_texts
    assert embed_texts(["hello"], EmbeddingConfig(enabled=False)) is None


def test_embed_diaries_disabled_returns_none():
    diaries = _batches()["weekday"].diaries
    assert emb.embed_diaries(diaries, EmbeddingConfig(enabled=False)) is None


def test_embed_texts_cache_round_trip(tmp_path, monkeypatch):
    from citybehavex.embedding import embed_texts
    calls = {"n": 0}

    def fake_post(base_url, model, texts, *, api_key, timeout):
        calls["n"] += 1
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0

    monkeypatch.setattr(emb, "_server_reachable", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_post_embeddings", fake_post)

    cfg = EmbeddingConfig(base_url="http://localhost:9", dimensions=4, cache_dir=str(tmp_path))
    first = embed_texts(["alpha", "beta", "gamma"], cfg)
    assert first is not None and first.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)

    second = embed_texts(["alpha", "beta", "gamma"], cfg)
    np.testing.assert_array_equal(first, second)
    assert calls["n"] == 1  # served entirely from cache


def test_embed_diaries_cache_round_trip(tmp_path, monkeypatch):
    diaries = _diary_list(3)
    calls = {"n": 0}

    def fake_post(base_url, model, texts, *, api_key, timeout):
        calls["n"] += 1
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0

    monkeypatch.setattr(emb, "_server_reachable", lambda *a, **k: True)
    monkeypatch.setattr(emb, "_post_embeddings", fake_post)

    cfg = EmbeddingConfig(
        base_url="http://localhost:9", dimensions=4, cache_dir=str(tmp_path)
    )
    first = emb.embed_diaries(diaries, cfg)
    assert first is not None and first.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)

    second = emb.embed_diaries(diaries, cfg)
    np.testing.assert_array_equal(first, second)
    assert calls["n"] == 1


def test_embed_texts_returns_none_on_server_failure():
    cfg = EmbeddingConfig(base_url=None, auto_launch=False)
    from citybehavex.embedding import embed_texts
    assert embed_texts(["hello"], cfg) is None


# --- build_diary_bank ------------------------------------------------------


def test_build_diary_bank_no_embeddings():
    bank = build_diary_bank(_batches(10, 10), EmbeddingConfig(enabled=False), GRAN)
    assert len(bank.diaries) == 20
    assert int((bank.day_type == "weekday").sum()) == 10
    assert int((bank.day_type == "weekend").sum()) == 10
    assert bank.embedded is False
    assert bank.embeddings is None
    assert bank.slot_locs.shape == (20, SLOTS)


# --- build_ddcrp_diary -----------------------------------------------------


def _bank(n_weekday=10, n_weekend=10) -> DiaryBank:
    return build_diary_bank(
        _batches(n_weekday, n_weekend), EmbeddingConfig(enabled=False), GRAN
    )


def test_hard_weekday_weekend_filter():
    bank = _bank()
    weekday_set = set(np.flatnonzero(bank.day_type == "weekday").tolist())
    weekend_set = set(np.flatnonzero(bank.day_type == "weekend").tolist())
    day_types = _calendar_day_types(MONDAY, 7)
    (_, _, _, _), chosen, _info = build_ddcrp_diary(
        bank, MONDAY, 7, day_types, n_agents=50, random_state=42, params=ScheduleConfig()
    )
    for d in range(7):
        valid = weekend_set if day_types[d] == "weekend" else weekday_set
        assert set(chosen[:, d].tolist()) <= valid


def test_hard_partition_with_special_day():
    """A special day type (e.g. an 'emergency') hard-partitions its own bank,
    even on a date that would otherwise be a weekday."""
    bank = build_diary_bank(
        {
            **_batches(10, 10),
            "emergency": _batch(
                [_diary(f"em-{i:03d}", "OTHER", "10:00", "11:00") for i in range(10)]
            ),
        },
        EmbeddingConfig(enabled=False),
        GRAN,
    )
    emergency_set = set(np.flatnonzero(bank.day_type == "emergency").tolist())
    weekday_set = set(np.flatnonzero(bank.day_type == "weekday").tolist())
    # Days 0-1 are a Monday/Tuesday (calendar weekdays) but forced to "emergency".
    day_types = ["emergency", "emergency", *_calendar_day_types(MONDAY, 7)[2:]]
    (_, _, _, _), chosen, _info = build_ddcrp_diary(
        bank, MONDAY, 7, day_types, n_agents=20, random_state=1, params=ScheduleConfig()
    )
    assert set(chosen[:, 0].tolist()) <= emergency_set
    assert set(chosen[:, 1].tolist()) <= emergency_set
    assert set(chosen[:, 0].tolist()) & weekday_set == set()


def test_determinism():
    bank = _bank()
    params = ScheduleConfig()
    day_types = _calendar_day_types(MONDAY, 7)
    (_, _, _, _), c1, _info1 = build_ddcrp_diary(bank, MONDAY, 7, day_types, 30, 7, params)
    (_, _, _, _), c2, _info2 = build_ddcrp_diary(bank, MONDAY, 7, day_types, 30, 7, params)
    np.testing.assert_array_equal(c1, c2)


def test_diary_arrays_shapes_and_mask():
    bank = _bank()
    days, agents = 7, 12
    day_types = _calendar_day_types(MONDAY, days)
    (ts, locs, starts, ends), chosen, _info = build_ddcrp_diary(
        bank, MONDAY, days, day_types, agents, 1, ScheduleConfig()
    )
    expected = agents * days * SLOTS
    assert ts.shape == (expected,) and locs.shape == (expected,)
    assert starts.shape == (agents,) and ends.shape == (agents,)
    assert starts[0] == 0 and ends[-1] == expected
    assert np.all(locs >= 0)
    assert chosen.shape == (agents, days)
    assert ts[0] == int(MONDAY.timestamp())


def test_profile_similarity_biases_selection():
    """Agents with injected profile embeddings should prefer similar diaries."""
    # Two diary groups: "work-heavy" (codes: WORK episodes) vs "other-heavy".
    work_diaries = [_diary(f"w-{i}", "WORK", "08:00", "18:00") for i in range(5)]
    other_diaries = [_diary(f"o-{i}", "OTHER", "10:00", "22:00") for i in range(5)]
    batches = {
        "weekday": _batch(work_diaries + other_diaries),
        "weekend": _batch(other_diaries + work_diaries),
    }
    bank = build_diary_bank(batches, EmbeddingConfig(enabled=False), GRAN)

    # Inject synthetic 2D embeddings: first 5 diaries (work) at (1,0), last 5 at (0,1).
    dim = 2
    diary_embs = np.zeros((10, dim), dtype=np.float32)
    diary_embs[:5] = [1.0, 0.0]
    diary_embs[5:] = [0.0, 1.0]
    bank = DiaryBank(
        diaries=bank.diaries,
        day_type=bank.day_type,
        embeddings=diary_embs,
        slot_locs=bank.slot_locs,
        slots_per_day=bank.slots_per_day,
        granularity_minutes=bank.granularity_minutes,
        embedded=True,
    )

    # "Worker" agent: embedding close to work diaries.
    # "Other" agent: embedding close to other diaries.
    profile_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    params = ScheduleConfig(temperature_beta_a=0.5, temperature_beta_b=10.0)  # low T → sharp
    day_types = _calendar_day_types(MONDAY, 5)
    _, chosen, _info = build_ddcrp_diary(
        bank, MONDAY, 5, day_types, n_agents=2, random_state=0, params=params,
        profile_embeddings=profile_embs,
    )
    weekday_cols = [d for d in range(5) if day_types[d] == "weekday"]

    worker_picks = chosen[0, weekday_cols]
    other_picks = chosen[1, weekday_cols]

    # Worker agent should mostly pick work diaries (idx 0-4).
    worker_work_frac = np.mean([p < 5 for p in worker_picks])
    other_other_frac = np.mean([p >= 5 for p in other_picks])
    # With sharp temperature, similarity should dominate — expect at least 60% correct.
    assert worker_work_frac >= 0.6, f"Worker only picked work {worker_work_frac:.0%}"
    assert other_other_frac >= 0.6, f"Other agent only picked other {other_other_frac:.0%}"


def test_precomputed_alignment_scores_override_embedding_similarity():
    bank = _bank(n_weekday=10, n_weekend=10)
    bank = DiaryBank(
        diaries=bank.diaries,
        day_type=bank.day_type,
        embeddings=np.ones((len(bank.diaries), 3), dtype=np.float32),
        slot_locs=bank.slot_locs,
        slots_per_day=bank.slots_per_day,
        granularity_minutes=bank.granularity_minutes,
        embedded=True,
    )
    profile_embs = np.ones((2, 2), dtype=np.float32)
    alignment_scores = np.zeros((2, len(bank.diaries)), dtype=np.float64)
    weekday = np.flatnonzero(bank.day_type == "weekday")
    alignment_scores[0, weekday[0]] = 1.0
    alignment_scores[1, weekday[1]] = 1.0

    params = ScheduleConfig(temperature_beta_a=0.5, temperature_beta_b=10.0)
    (_, _, _, _), chosen, info = build_ddcrp_diary(
        bank,
        MONDAY,
        1,
        ["weekday"],
        n_agents=2,
        random_state=3,
        params=params,
        profile_embeddings=profile_embs,
        agent_diary_sim=alignment_scores,
    )

    np.testing.assert_array_equal(info.agent_diary_sim, alignment_scores)
    assert chosen.shape == (2, 1)


def test_precomputed_alignment_scores_validate_shape():
    bank = _bank(n_weekday=10, n_weekend=10)
    with np.testing.assert_raises(ValueError):
        build_ddcrp_diary(
            bank,
            MONDAY,
            1,
            ["weekday"],
            n_agents=2,
            random_state=3,
            params=ScheduleConfig(),
            agent_diary_sim=np.zeros((1, len(bank.diaries))),
        )


def test_no_profile_embeddings_uses_popularity():
    """Without profile embeddings, selection should spread across the bank (exploration)."""
    bank = _bank(n_weekday=10, n_weekend=10)
    params = ScheduleConfig(alpha_beta_a=10.0, alpha_beta_b=1.0)  # high alpha → lots of exploration
    day_types = _calendar_day_types(MONDAY, 7)
    (_, _, _, _), chosen, _info = build_ddcrp_diary(bank, MONDAY, 7, day_types, 200, 9, params)
    weekday_cols = [d for d in range(7) if day_types[d] == "weekday"]
    used = set(chosen[:, weekday_cols].ravel().tolist())
    assert len(used) >= 6  # most of the 10-diary weekday bank gets explored
