from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from citybehavex.activities.poi_semantic import build_poi_semantic_activity_data
from citybehavex.llm_diaries import Diary, DiaryBatch, LocationCountDistribution
from citybehavex.profiles import AgentProfile

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_modernbert_poi_type_aligner.py"
SPEC = importlib.util.spec_from_file_location("train_modernbert_poi_type_aligner", SCRIPT_PATH)
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


def _diary(diary_id: str) -> Diary:
    return Diary.model_validate(
        {
            "diary_id": diary_id,
            "episodes": [
                {"start": "00:00", "end": "09:00", "purpose": "HOME"},
                {"start": "09:00", "end": "17:00", "purpose": "WORK"},
                {"start": "17:00", "end": "19:00", "purpose": "OTHER"},
                {"start": "19:00", "end": "24:00", "purpose": "HOME"},
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


def test_build_training_pairs_only_uses_other_blocks_and_cycles_clusters():
    diaries = [_diary(f"d{i}") for i in range(3)]
    poi_data = build_poi_semantic_activity_data()

    pairs = aligner.build_training_pairs(
        [_profile(1), _profile(2)],
        diaries,
        sample_size=len(poi_data.semantic_clusters) * 2,
        seed=7,
    )

    assert len(pairs) == len(poi_data.semantic_clusters) * 2
    assert {pair.semantic_cluster for pair in pairs} == set(poi_data.semantic_clusters)
    assert all(pair.context_text for pair in pairs)
    assert all(pair.candidate_text for pair in pairs)
    assert all(pair.semantic_cluster in pair.candidate_text for pair in pairs)


def test_build_training_pairs_requires_other_blocks():
    home_only = Diary.model_validate(
        {
            "diary_id": "d0",
            "episodes": [{"start": "00:00", "end": "24:00", "purpose": "HOME"}],
        }
    )
    try:
        aligner.build_training_pairs([_profile(1)], [home_only], sample_size=5, seed=1)
    except ValueError as exc:
        assert "OTHER" in str(exc)
    else:
        raise AssertionError("expected ValueError when no OTHER blocks exist")


def test_alignment_prompt_mentions_place_type_and_context():
    poi_data = build_poi_semantic_activity_data()
    pair = aligner.TrainingPair(
        profile_uid=1,
        diary_id="d0",
        block_id=0,
        block_index=2,
        semantic_cluster=poi_data.semantic_clusters[0],
        profile_text="profile",
        context_text="context about the person and block",
        candidate_text=f"{poi_data.semantic_clusters[0]}: public place type with examples",
    )

    prompt = aligner.alignment_prompt(pair)

    assert "context about the person and block" in prompt
    assert poi_data.semantic_clusters[0] in prompt


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
                                {"reason": "this should not be stored", "score": 0.7}
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(aligner.requests, "post", lambda *args, **kwargs: Response())

    pair = aligner.TrainingPair(
        profile_uid=1,
        diary_id="d0",
        block_id=0,
        block_index=2,
        semantic_cluster="retail_shopping",
        profile_text="profile",
        context_text="context",
        candidate_text="retail_shopping: public place type",
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
    assert df.loc[0, "score"] == 0.7
    assert df.loc[0, "semantic_cluster"] == "retail_shopping"
    assert df.loc[0, "context_text"] == "context"


def test_main_reuses_dataset_and_trains_with_mock(tmp_path, monkeypatch):
    profiles_path = tmp_path / "profiles.parquet"
    pd.DataFrame([_profile().model_dump()]).to_parquet(profiles_path, index=False)
    diary_path = tmp_path / "validated_diaries_weekday.json"
    diary_path.write_text(
        json.dumps(_batch([_diary("d0")] * 10).model_dump()), encoding="utf-8"
    )
    dataset_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {
                "profile_uid": 1,
                "diary_id": "d0",
                "block_id": 0,
                "block_index": 2,
                "semantic_cluster": "retail_shopping",
                "profile_text": "profile",
                "context_text": "context",
                "candidate_text": "retail_shopping: public place type",
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


def test_cli_defaults_point_to_poi_type_aligner():
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

    assert args.output_model_path == "models/modernbert-poi-type-aligner"
    assert args.dataset_output == "data/poi_type_alignment_scores.parquet"
    assert args.sample_size == 2000
