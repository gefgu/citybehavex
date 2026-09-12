from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from citybehavex.profiles import AgentProfile  # noqa: E402
from cross_encoder_eval_lib import (  # noqa: E402
    ALIGNER_SPECS,
    already_done,
    compute_common_metrics,
    load_or_label_dataset,
    load_training_module,
    stratified_split,
    teacher_label,
    upsert_result,
)


def _profile(uid: int) -> AgentProfile:
    return AgentProfile(
        uid=uid,
        gender="female",
        name="Alice",
        age=35,
        education="bachelor",
        health=4,
        household="living alone",
        job="professional",
        has_car=True,
        has_bike=False,
        home_tile=1,
        work_tile=2,
    )


def test_teacher_label_maps_known_model_ids():
    assert teacher_label("Qwen/Qwen2.5-32B-Instruct-AWQ") == "qwen"
    assert teacher_label("mistral-small-3.2-24b") == "mistral"
    assert teacher_label("mistralai/Mistral-Small-3.2-24B-Instruct-2506") == "mistral"


def test_stratified_split_preserves_class_proportions():
    df = pd.DataFrame(
        {
            "group": ["a"] * 40 + ["b"] * 40 + ["c"] * 20,
            "value": np.arange(100),
        }
    )
    train_df, test_df = stratified_split(df, "group", test_size=0.2, seed=1)

    assert len(train_df) + len(test_df) == len(df)
    train_counts = train_df["group"].value_counts(normalize=True)
    test_counts = test_df["group"].value_counts(normalize=True)
    for group in ("a", "b", "c"):
        assert abs(train_counts[group] - test_counts[group]) < 0.1


def test_stratified_split_falls_back_when_a_class_has_one_row():
    df = pd.DataFrame({"group": ["a"] * 10 + ["b"], "value": np.arange(11)})
    with pytest.warns(UserWarning):
        train_df, test_df = stratified_split(df, "group", test_size=0.2, seed=1)
    assert len(train_df) + len(test_df) == len(df)


def test_compute_common_metrics_perfect_predictions_are_zero_error():
    y = np.linspace(0.0, 1.0, 20)
    metrics = compute_common_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["kendall_tau"] == pytest.approx(1.0)


def test_already_done_and_upsert_result_roundtrip(tmp_path):
    results_path = tmp_path / "results.parquet"
    assert already_done(results_path, "schedule", "qwen", 100) is False

    upsert_result(
        results_path,
        {"aligner": "schedule", "teacher": "qwen", "sample_size": 100, "mae": 0.1},
    )
    assert already_done(results_path, "schedule", "qwen", 100) is True
    assert already_done(results_path, "schedule", "qwen", 1000) is False

    upsert_result(
        results_path,
        {"aligner": "schedule", "teacher": "qwen", "sample_size": 100, "mae": 0.05},
    )
    df = pd.read_parquet(results_path)
    assert len(df) == 1
    assert df.loc[0, "mae"] == pytest.approx(0.05)


@pytest.mark.parametrize("sample_size", [3, 7])
def test_profile_coherence_prefix_equivalence(sample_size):
    """A size-N build is a byte-identical prefix of a larger build with the same seed."""
    spec = ALIGNER_SPECS["profile_coherence"]
    from cross_encoder_eval_lib import load_training_module

    module = load_training_module(spec.script_path)
    profiles = [_profile(uid) for uid in range(1, 6)]

    small = module.build_training_pairs(profiles, sample_size=sample_size, mutation_ratio=0.5, seed=42)
    large = module.build_training_pairs(profiles, sample_size=10, mutation_ratio=0.5, seed=42)

    assert small == large[:sample_size]


def test_load_or_label_dataset_extends_smaller_cache_without_relabeling(tmp_path, monkeypatch):
    spec = ALIGNER_SPECS["profile_coherence"]
    module = load_training_module(spec.script_path)

    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame([_profile(uid).model_dump() for uid in range(1, 6)]).to_parquet(
        profiles_path, index=False
    )

    call_sizes: list[int] = []

    def fake_label_pairs(pairs, **kwargs):
        call_sizes.append(len(pairs))
        return pd.DataFrame(
            [
                {
                    "profile_uid": pair.profile_uid,
                    "variant": pair.variant,
                    "profile_text": pair.profile_text,
                    "context_text": pair.context_text,
                    "candidate_text": pair.candidate_text,
                    "score": 0.5,
                }
                for pair in pairs
            ]
        )

    monkeypatch.setattr(module, "label_pairs", fake_label_pairs)

    common_kwargs = dict(
        spec=spec,
        profiles_path=profiles_path,
        diary_paths=None,
        teacher="mistral",
        model_id="mistral-small-3.2-24b",
        base_url="http://unused",
        seed=42,
        concurrency=2,
        cache_dir=tmp_path / "cache",
    )

    small = load_or_label_dataset(sample_size=3, **common_kwargs)
    assert len(small) == 3
    assert call_sizes == [3]

    large = load_or_label_dataset(sample_size=6, **common_kwargs)
    assert len(large) == 6
    assert call_sizes == [3, 3]  # only the 3 new pairs were labeled, not all 6
    assert large.iloc[:3].reset_index(drop=True).equals(small)


def test_vehicle_ownership_prefix_equivalence():
    spec = ALIGNER_SPECS["vehicle_ownership"]
    from cross_encoder_eval_lib import load_training_module

    module = load_training_module(spec.script_path)
    profiles = [_profile(uid) for uid in range(1, 6)]

    small = module.build_training_pairs(profiles, sample_size=4, seed=42)
    large = module.build_training_pairs(profiles, sample_size=9, seed=42)

    assert small == large[:4]
