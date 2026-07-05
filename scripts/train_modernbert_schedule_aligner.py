#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
import numpy as np


@dataclass(frozen=True)
class TrainingPair:
    profile_uid: int
    diary_id: str
    day_type: str
    profile_text: str
    diary_text: str


def parse_alignment_payload(payload: Any) -> float:
    """Extract and validate a [0, 1] score from LLM JSON output."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("alignment payload must be a JSON object")
    if "reason" not in payload or "score" not in payload:
        raise ValueError("alignment payload must contain reason and score")
    try:
        score = float(payload["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("alignment score must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError("alignment score must be finite")
    return float(min(1.0, max(0.0, score)))


def _parse_chat_json(response_payload: Any) -> Any:
    choices = response_payload.get("choices") if isinstance(response_payload, dict) else None
    if not choices:
        raise ValueError("LLM response has no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("LLM response choice has no message content")
    return json.loads(content)


def _infer_day_type(path: Path) -> str:
    stem = path.stem
    match = re.search(r"validated_diaries_(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def load_profiles(path: str | Path):
    from citybehavex.profiles import AgentProfile

    df = pd.read_parquet(path)
    return [AgentProfile.model_validate(row) for row in df.to_dict(orient="records")]


def load_diaries(paths: Sequence[str | Path]) -> list[tuple[str, object]]:
    from citybehavex.llm_diaries import DiaryBatch

    diaries: list[tuple[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        batch = DiaryBatch.model_validate(json.loads(path.read_text(encoding="utf-8")))
        day_type = _infer_day_type(path)
        diaries.extend((day_type, diary) for diary in batch.diaries)
    if not diaries:
        raise ValueError("no diaries found")
    return diaries


def build_training_pairs(
    profiles: Sequence[object],
    diaries: Sequence[tuple[str, object]],
    *,
    sample_size: int,
    seed: int,
) -> list[TrainingPair]:
    from citybehavex.embedding import diary_to_prose
    from citybehavex.profiles import profile_to_narrative

    if not profiles:
        raise ValueError("profiles are empty")
    if not diaries:
        raise ValueError("diaries are empty")
    rng = np.random.default_rng(seed)
    by_day_type: dict[str, list[object]] = {}
    for day_type, diary in diaries:
        by_day_type.setdefault(day_type, []).append(diary)
    day_types = sorted(by_day_type)
    pairs: list[TrainingPair] = []
    for i in range(sample_size):
        profile = profiles[int(rng.integers(len(profiles)))]
        day_type = day_types[i % len(day_types)]
        diary_bucket = by_day_type[day_type]
        diary = diary_bucket[int(rng.integers(len(diary_bucket)))]
        pairs.append(
            TrainingPair(
                profile_uid=profile.uid,
                diary_id=diary.diary_id,
                day_type=day_type,
                profile_text=profile_to_narrative(profile),
                diary_text=diary_to_prose(diary),
            )
        )
    return pairs


def alignment_prompt(pair: TrainingPair) -> str:
    return (
        "Assess whether this macro daily schedule aligns with the person's "
        "demographic and mobility profile. Return strictly valid JSON with the "
        "keys in this order: reason, score. The score must be a number from 0 "
        "to 1, where 0 means incompatible and 1 means highly aligned.\n\n"
        f"Profile:\n{pair.profile_text}\n\n"
        f"Schedule:\n{pair.diary_text}"
    )


def label_pairs(
    pairs: Sequence[TrainingPair],
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    retries: int,
) -> pd.DataFrame:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        last_error: Exception | None = None
        for _attempt in range(max(1, retries)):
            try:
                resp = requests.post(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a strict JSON scorer for mobility schedules.",
                            },
                            {"role": "user", "content": alignment_prompt(pair)},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                score = parse_alignment_payload(_parse_chat_json(resp.json()))
                rows.append(
                    {
                        "profile_uid": pair.profile_uid,
                        "diary_id": pair.diary_id,
                        "day_type": pair.day_type,
                        "profile_text": pair.profile_text,
                        "diary_text": pair.diary_text,
                        "score": score,
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry with final failure.
                last_error = exc
        else:
            raise RuntimeError(
                f"failed to label profile={pair.profile_uid} diary={pair.diary_id}: {last_error}"
            ) from last_error
    return pd.DataFrame(rows)


def train_cross_encoder(
    dataset: pd.DataFrame,
    *,
    base_model: str,
    output_model_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str | None,
) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from sentence_transformers import InputExample
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Install finetuning dependencies with `uv sync --extra finetuning`."
        ) from exc

    examples = [
        InputExample(texts=[row.profile_text, row.diary_text], label=float(row.score))
        for row in dataset.itertuples(index=False)
    ]
    if not examples:
        raise ValueError("training dataset is empty")
    model = CrossEncoder(base_model, num_labels=1, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        warmup_steps=max(1, len(loader) // 10),
        output_path=output_model_path,
    )
    model.save(output_model_path)


def _read_existing_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, orient="records", lines=True)


def _write_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-path", required=True)
    parser.add_argument("--diary-path", action="append", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--dataset-output", default="data/schedule_alignment_scores.parquet")
    parser.add_argument("--output-model-path", default="models/modernbert-schedule-aligner")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device passed to CrossEncoder. Defaults to CPU for RTX 5090/PyTorch compatibility.",
    )
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="Write the supervised label dataset and skip CrossEncoder training.",
    )
    parser.add_argument("--reuse-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_output)
    dataset = _read_existing_dataset(dataset_path) if args.reuse_dataset else None
    if dataset is None:
        profiles = load_profiles(args.profiles_path)
        diaries = load_diaries(args.diary_path)
        pairs = build_training_pairs(
            profiles,
            diaries,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        dataset = label_pairs(
            pairs,
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            timeout=args.llm_timeout_seconds,
            retries=args.llm_retries,
        )
        _write_dataset(dataset, dataset_path)
    if args.label_only:
        return
    train_cross_encoder(
        dataset,
        base_model=args.base_model,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
