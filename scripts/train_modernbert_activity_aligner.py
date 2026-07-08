#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

from citybehavex.activities.alignment import (
    START_PREVIOUS_ACTIVITY,
    ActivityBlock,
    diary_activity_blocks,
)


@dataclass(frozen=True)
class TrainingPair:
    profile_uid: int
    cluster_id: int | None
    diary_id: str
    block_id: int
    block_index: int
    purpose: str
    start: str
    end: str
    previous_activity_idx: int
    previous_activity: str
    activity_idx: int
    activity: str
    profile_text: str
    context_text: str
    activity_text: str


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


def load_diaries(paths: Sequence[str | Path]) -> list[object]:
    from citybehavex.llm_diaries import DiaryBatch

    diaries: list[object] = []
    for raw_path in paths:
        path = Path(raw_path)
        batch = DiaryBatch.model_validate(json.loads(path.read_text(encoding="utf-8")))
        _infer_day_type(path)
        diaries.extend(batch.diaries)
    if not diaries:
        raise ValueError("no diaries found")
    return diaries


def _purpose_code(purpose: str) -> int:
    if purpose == "HOME":
        return 0
    if purpose == "WORK":
        return 1
    return 2


def _eligible_activities(block: ActivityBlock, catalog: Sequence[object]) -> list[object]:
    purpose = _purpose_code(block.purpose)
    return [activity for activity in catalog if purpose in activity.eligible_purposes]


def _activity_text(activity: object) -> str:
    return f"{activity.name}: {activity.description}"


def _previous_activity_label(previous_idx: int, catalog: Sequence[object]) -> str:
    if previous_idx == START_PREVIOUS_ACTIVITY:
        return "start"
    if 0 <= previous_idx < len(catalog):
        return str(catalog[previous_idx].name)
    return "unknown"


def _previous_activity_text(previous_idx: int, catalog: Sequence[object]) -> str:
    if previous_idx == START_PREVIOUS_ACTIVITY:
        return "no previous micro-activity in this block"
    if 0 <= previous_idx < len(catalog):
        activity = catalog[previous_idx]
        return f"previous micro-activity was {activity.name}: {activity.description}"
    return "previous micro-activity is unknown"


_PERIODS: tuple[tuple[int, int, str], ...] = (
    (0, 6 * 60, "00-06"),
    (6 * 60, 12 * 60, "06-12"),
    (12 * 60, 18 * 60, "12-18"),
    (18 * 60, 24 * 60, "18-24"),
)


def _minutes(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    return 24 * 60 if hour == 24 else hour * 60 + minute


def _period_label(block: ActivityBlock) -> str:
    start = _minutes(block.start)
    end = _minutes(block.end)
    overlaps = [
        max(0, min(end, period_end) - max(start, period_start))
        for period_start, period_end, _label in _PERIODS
    ]
    period_index = max(range(len(overlaps)), key=lambda idx: (overlaps[idx], -idx))
    return _PERIODS[period_index][2]


def context_text(
    profile_text: str,
    block: ActivityBlock,
    previous_activity_idx: int,
    catalog: Sequence[object],
) -> str:
    return (
        f"{profile_text}\n"
        f"Schedule block: diary {block.diary_id}, block {block.episode_index}, "
        f"{block.purpose} from {block.start} to {block.end}.\n"
        f"Period group: {block.purpose} blocks mostly in the {_period_label(block)} period.\n"
        f"Transition/history context: {_previous_activity_text(previous_activity_idx, catalog)}.\n"
        "Score which valid time-use activity best fits this person, block, time, and history."
    )


def _hard_negative_activity(block: ActivityBlock, eligible: Sequence[object]) -> object | None:
    period = _period_label(block)
    names_by_priority: list[str]
    if block.purpose == "HOME" and period == "00-06":
        names_by_priority = ["eatdrink", "selfcare", "cleanetc"]
    elif block.purpose == "HOME" and period == "18-24":
        names_by_priority = ["selfcare", "eatdrink", "pkidcare", "ikidcare"]
    elif block.purpose == "WORK":
        names_by_priority = ["paidwork", "selfcare"]
    else:
        names_by_priority = ["eatdrink", "selfcare", "paidwork"]
    by_name = {activity.name: activity for activity in eligible}
    for name in names_by_priority:
        if name in by_name:
            return by_name[name]
    return None


def build_training_pairs(
    profiles: Sequence[object],
    diaries: Sequence[object],
    *,
    sample_size: int,
    seed: int,
) -> list[TrainingPair]:
    from citybehavex.activities import build_catalog
    from citybehavex.profiles import profile_to_narrative

    if not profiles:
        raise ValueError("profiles are empty")
    if not diaries:
        raise ValueError("diaries are empty")
    blocks = diary_activity_blocks(diaries)
    if not blocks:
        raise ValueError("diaries contain no activity blocks")

    catalog = build_catalog()
    previous_values = [START_PREVIOUS_ACTIVITY, *range(len(catalog))]
    rng = np.random.default_rng(seed)
    pairs: list[TrainingPair] = []
    attempts = 0
    max_attempts = max(sample_size * 20, 100)
    while len(pairs) < sample_size and attempts < max_attempts:
        attempts += 1
        profile = profiles[int(rng.integers(len(profiles)))]
        block = blocks[int(rng.integers(len(blocks)))]
        eligible = _eligible_activities(block, catalog)
        if not eligible:
            continue
        previous_idx = int(previous_values[int(rng.integers(len(previous_values)))])
        hard_negative = _hard_negative_activity(block, eligible)
        if hard_negative is not None and len(pairs) % 3 == 0:
            activity = hard_negative
        else:
            activity = eligible[int(rng.integers(len(eligible)))]
        profile_text = profile_to_narrative(profile)
        pairs.append(
            TrainingPair(
                profile_uid=int(profile.uid),
                cluster_id=None,
                diary_id=block.diary_id,
                block_id=int(block.block_id),
                block_index=int(block.episode_index),
                purpose=block.purpose,
                start=block.start,
                end=block.end,
                previous_activity_idx=previous_idx,
                previous_activity=_previous_activity_label(previous_idx, catalog),
                activity_idx=int(activity.idx),
                activity=str(activity.name),
                profile_text=profile_text,
                context_text=context_text(profile_text, block, previous_idx, catalog),
                activity_text=_activity_text(activity),
            )
        )
    if len(pairs) < sample_size:
        raise ValueError(f"only built {len(pairs)} training pairs after {attempts} attempts")
    return pairs


def alignment_prompt(pair: TrainingPair) -> str:
    return (
        "Assess whether this candidate time-use micro-activity aligns with the "
        "person, schedule block, time window, and previous activity context. "
        "Return strictly valid JSON with the keys in this order: reason, score. "
        "The score must be a number from 0 to 1, where 0 means incompatible and "
        "1 means highly aligned.\n\n"
        "Scoring guidance: use the period group strongly. HOME 00-06 usually "
        "fits sleep best and should not over-score eating, self-care, or chores. "
        "HOME evenings can fit TV/radio, food preparation, leisure, reading, or "
        "internet use. WORK daytime can fit paid work, but realistic meal breaks "
        "should score well when the time window supports them. Avoid inflating "
        "eatdrink, selfcare, and paidwork simply because they are eligible.\n\n"
        f"Context:\n{pair.context_text}\n\n"
        f"Candidate activity:\n{pair.activity_text}"
    )


def _pair_row(pair: TrainingPair, score: float) -> dict[str, Any]:
    return {
        "profile_uid": pair.profile_uid,
        "cluster_id": pair.cluster_id,
        "diary_id": pair.diary_id,
        "block_id": pair.block_id,
        "block_index": pair.block_index,
        "purpose": pair.purpose,
        "start": pair.start,
        "end": pair.end,
        "previous_activity_idx": pair.previous_activity_idx,
        "previous_activity": pair.previous_activity,
        "activity_idx": pair.activity_idx,
        "activity": pair.activity,
        "profile_text": pair.profile_text,
        "context_text": pair.context_text,
        "activity_text": pair.activity_text,
        "score": score,
    }


def label_pairs(
    pairs: Sequence[TrainingPair],
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    concurrency: int,
    progress_interval: int,
) -> pd.DataFrame:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def score_pair(pair: TrainingPair) -> dict[str, Any]:
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
                                "content": "You are a strict JSON scorer for human time-use activity alignment.",
                            },
                            {"role": "user", "content": alignment_prompt(pair)},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                score = parse_alignment_payload(_parse_chat_json(resp.json()))
                return _pair_row(pair, score)
            except Exception as exc:  # noqa: BLE001 - retry with final failure.
                last_error = exc
        raise RuntimeError(
            "failed to label "
            f"profile={pair.profile_uid} diary={pair.diary_id} "
            f"block={pair.block_index} activity={pair.activity}: {last_error}"
        ) from last_error

    if concurrency <= 1:
        rows = []
        for idx, pair in enumerate(pairs, start=1):
            rows.append(score_pair(pair))
            if progress_interval > 0 and idx % progress_interval == 0:
                print(f"Labeled {idx}/{len(pairs)} pairs", flush=True)
        return pd.DataFrame(rows)

    rows: list[dict[str, Any] | None] = [None] * len(pairs)
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(score_pair, pair): idx
            for idx, pair in enumerate(pairs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            rows[idx] = future.result()
            completed += 1
            if progress_interval > 0 and completed % progress_interval == 0:
                print(f"Labeled {completed}/{len(pairs)} pairs", flush=True)
    if any(row is None for row in rows):
        raise RuntimeError("labeling finished with missing rows")
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
        InputExample(texts=[row.context_text, row.activity_text], label=float(row.score))
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
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=8,
        help="Concurrent LLM labeling requests. Increase when the serving backend has batching headroom.",
    )
    parser.add_argument(
        "--llm-progress-interval",
        type=int,
        default=100,
        help="Print labeling progress every N completed pairs. Set to 0 to disable.",
    )
    parser.add_argument("--dataset-output", default="data/activity_alignment_scores.parquet")
    parser.add_argument("--output-model-path", default="models/modernbert-activity-aligner")
    parser.add_argument("--sample-size", type=int, default=5000)
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
            concurrency=args.llm_concurrency,
            progress_interval=args.llm_progress_interval,
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
