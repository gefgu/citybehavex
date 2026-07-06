from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from citybehavex.activities import build_catalog
from citybehavex.llm_diaries import Diary, DiaryBatch, LocationCountDistribution
from citybehavex.profiles import AgentProfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_modernbert_activity_aligner.py"
SPEC = importlib.util.spec_from_file_location("train_modernbert_activity_aligner", SCRIPT_PATH)
aligner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = aligner
SPEC.loader.exec_module(aligner)


def _profile(uid: int = 1) -> AgentProfile:
    return AgentProfile(
        uid=uid,
        gender="female",
        name="Alice",
        age=35,
        education="bachelor",
        health=4,
        household="living alone",
        job="professional",
        has_car=False,
        has_bike=True,
        home_tile=1,
        work_tile=2,
    )


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


def test_parse_alignment_payload_clips_score_and_requires_reason():
    assert aligner.parse_alignment_payload({"reason": "ok", "score": 1.5}) == 1.0
    assert aligner.parse_alignment_payload('{"reason": "bad", "score": -0.2}') == 0.0


def test_build_training_pairs_uses_only_valid_activities():
    pairs = aligner.build_training_pairs(
        [_profile(1), _profile(2)],
        [
            _diary("wd", "WORK", "09:00", "17:00"),
            _diary("we", "OTHER", "12:00", "18:00"),
        ],
        sample_size=30,
        seed=7,
    )
    catalog = build_catalog()
    by_idx = {activity.idx: activity for activity in catalog}

    assert len(pairs) == 30
    assert all(pair.context_text for pair in pairs)
    assert all(pair.activity_text for pair in pairs)
    for pair in pairs:
        purpose = {"HOME": 0, "WORK": 1}.get(pair.purpose, 2)
        assert purpose in by_idx[pair.activity_idx].eligible_purposes


def test_label_pairs_does_not_persist_reason(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"reason": "this should not be stored", "score": 0.42}
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        aligner.requests,
        "post",
        lambda *args, **kwargs: Response(),
    )

    pair = aligner.TrainingPair(
        profile_uid=1,
        cluster_id=None,
        diary_id="d1",
        block_id=0,
        block_index=0,
        purpose="HOME",
        start="00:00",
        end="08:00",
        previous_activity_idx=-1,
        previous_activity="start",
        activity_idx=6,
        activity="cleanetc",
        profile_text="profile",
        context_text="context",
        activity_text="cleanetc: cleaning",
    )
    df = aligner.label_pairs(
        [pair],
        base_url="http://localhost:8081",
        model="test-model",
        api_key=None,
        timeout=1.0,
        retries=1,
    )

    assert "reason" not in df.columns
    assert df.loc[0, "score"] == 0.42
    assert df.loc[0, "activity"] == "cleanetc"
    assert df.loc[0, "context_text"] == "context"


def test_main_reuses_dataset_and_trains_with_mock(tmp_path, monkeypatch):
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame([_profile().model_dump()]).to_parquet(profiles_path, index=False)
    diary_path = tmp_path / "validated_diaries_weekday.json"
    diary_path.write_text(
        json.dumps(_batch([_diary("wd", "WORK", "09:00", "17:00")] * 10).model_dump()),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {
                "profile_uid": 1,
                "cluster_id": None,
                "diary_id": "wd",
                "block_id": 0,
                "block_index": 0,
                "purpose": "HOME",
                "start": "00:00",
                "end": "09:00",
                "previous_activity_idx": -1,
                "previous_activity": "start",
                "activity_idx": 6,
                "activity": "cleanetc",
                "profile_text": "profile",
                "context_text": "context",
                "activity_text": "cleanetc: cleaning",
                "score": 0.5,
            }
        ]
    ).to_parquet(dataset_path, index=False)

    trained = {}

    def fake_train(dataset, **kwargs):
        trained["rows"] = len(dataset)
        trained.update(kwargs)

    monkeypatch.setattr(aligner, "train_cross_encoder", fake_train)
    aligner.main(
        [
            "--profiles-path",
            str(profiles_path),
            "--diary-path",
            str(diary_path),
            "--llm-base-url",
            "http://unused",
            "--llm-model",
            "unused",
            "--dataset-output",
            str(dataset_path),
            "--output-model-path",
            str(tmp_path / "model"),
            "--reuse-dataset",
        ]
    )

    assert trained["rows"] == 1
    assert trained["output_model_path"] == str(tmp_path / "model")
    assert trained["device"] == "cpu"


def test_cli_defaults_point_to_activity_aligner():
    args = aligner.parse_args(
        [
            "--profiles-path",
            "profiles.parquet",
            "--diary-path",
            "validated_diaries_weekday.json",
            "--llm-base-url",
            "http://localhost:8081",
            "--llm-model",
            "model",
        ]
    )

    assert args.output_model_path == "models/modernbert-activity-aligner"
    assert args.dataset_output == "data/activity_alignment_scores.parquet"
    assert args.sample_size == 5000
