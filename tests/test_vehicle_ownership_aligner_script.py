from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from citybehavex.profiles import AgentProfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_modernbert_vehicle_ownership_aligner.py"
SPEC = importlib.util.spec_from_file_location("train_modernbert_vehicle_ownership_aligner", SCRIPT_PATH)
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
        has_car=True,
        has_bike=False,
        home_tile=1,
        work_tile=2,
    )


def test_parse_alignment_payload_clips_score_and_requires_reason():
    assert aligner.parse_alignment_payload({"reason": "ok", "score": 1.5}) == 1.0
    assert aligner.parse_alignment_payload('{"reason": "bad", "score": -0.2}') == 0.0


def test_build_training_pairs_alternates_car_and_bike_without_transport_leak():
    pairs = aligner.build_training_pairs(
        [_profile(1), _profile(2)],
        sample_size=4,
        seed=7,
        city_profile="dense metro with good transit",
    )

    assert [pair.vehicle for pair in pairs] == ["car", "bike", "car", "bike"]
    assert all("City context: dense metro" in pair.context_text for pair in pairs)
    assert all("They own" not in pair.profile_text for pair in pairs)
    assert all("public transport or walking" not in pair.profile_text for pair in pairs)


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
        vehicle="car",
        profile_text="profile",
        context_text="context",
        candidate_text="owns a car",
    )
    df = aligner.label_pairs(
        [pair],
        base_url="http://localhost:8081",
        model="test-model",
        api_key=None,
        timeout=1.0,
        retries=1,
        concurrency=1,
        progress_interval=1,
    )

    assert "reason" not in df.columns
    assert df.loc[0, "score"] == 0.42
    assert df.loc[0, "vehicle"] == "car"
    assert df.loc[0, "context_text"] == "context"


def test_main_reuses_dataset_and_trains_with_mock(tmp_path, monkeypatch):
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame([_profile().model_dump()]).to_parquet(profiles_path, index=False)
    dataset_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {
                "profile_uid": 1,
                "vehicle": "car",
                "profile_text": "profile",
                "context_text": "context",
                "candidate_text": "owns a car",
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


def test_cli_defaults_point_to_vehicle_ownership_aligner():
    args = aligner.parse_args(
        [
            "--profiles-path",
            "profiles.parquet",
            "--llm-base-url",
            "http://localhost:8081",
            "--llm-model",
            "Qwen/Qwen2.5-32B-Instruct-AWQ",
        ]
    )

    assert args.dataset_output == "data/vehicle_ownership_alignment_scores.parquet"
    assert args.output_model_path == "models/modernbert-vehicle-ownership-aligner"
    assert args.sample_size == 2000
    assert args.llm_concurrency == 8
