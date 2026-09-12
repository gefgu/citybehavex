#!/usr/bin/env python
"""Shared harness for evaluating the four ModernBERT CrossEncoder aligners.

Reuses each `scripts/train_modernbert_*_aligner.py` module's own
`load_profiles`/`load_diaries`/`build_training_pairs`/`label_pairs`/
`train_cross_encoder` functions (loaded by path via importlib, same pattern as
`notebooks/06_cross_reranker_evaluation/01_profile_coherence_evaluation.ipynb`)
instead of duplicating prompts, sampling, or training logic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_CACHE: dict[str, types.ModuleType] = {}


def load_training_module(script_path: str | Path) -> types.ModuleType:
    """Load a `train_modernbert_*_aligner.py` script as an importable module."""
    key = str(script_path)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(Path(script_path).stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


@dataclass(frozen=True)
class AlignerSpec:
    name: str
    script_path: Path
    needs_diaries: bool
    stratify_col: str
    text_cols: tuple[str, str]
    build_pairs: Callable[[types.ModuleType, list, list | None, int, int], list]
    extra_metrics: Callable[[pd.DataFrame], dict[str, Any]] = field(default=lambda df: {})


def _build_schedule_pairs(module, profiles, diaries, sample_size, seed):
    return module.build_training_pairs(profiles, diaries, sample_size=sample_size, seed=seed)


def _build_activity_pairs(module, profiles, diaries, sample_size, seed):
    return module.build_training_pairs(profiles, diaries, sample_size=sample_size, seed=seed)


def _build_poi_type_pairs(module, profiles, diaries, sample_size, seed):
    return module.build_training_pairs(profiles, diaries, sample_size=sample_size, seed=seed)


def _build_vehicle_pairs(module, profiles, _diaries, sample_size, seed):
    return module.build_training_pairs(profiles, sample_size=sample_size, seed=seed)


def _build_coherence_pairs(module, profiles, _diaries, sample_size, seed, *, mutation_ratio=0.5):
    return module.build_training_pairs(
        profiles, sample_size=sample_size, mutation_ratio=mutation_ratio, seed=seed
    )


def _coherence_extra_metrics(df: pd.DataFrame) -> dict[str, Any]:
    binary = df["variant"].eq("original").astype(int).to_numpy()
    metrics: dict[str, Any] = {}
    if len(np.unique(binary)) == 2:
        metrics["original_vs_mutated_roc_auc"] = float(roc_auc_score(binary, df["prediction"]))
        metrics["original_vs_mutated_pr_auc"] = float(
            average_precision_score(binary, df["prediction"])
        )
        original = df[df["variant"].eq("original")].groupby("profile_uid")["prediction"].mean()
        mutated = df[df["variant"].eq("mutated")].groupby("profile_uid")["prediction"].mean()
        common = original.index.intersection(mutated.index)
        if len(common):
            metrics["paired_original_beats_mutated"] = float(
                (original.loc[common] > mutated.loc[common]).mean()
            )
    return metrics


def _category_breakdown_metrics(category_col: str) -> Callable[[pd.DataFrame], dict[str, Any]]:
    def _extra(df: pd.DataFrame) -> dict[str, Any]:
        breakdown = (
            df.groupby(category_col)
            .apply(
                lambda g: float(mean_absolute_error(g["score"], g["prediction"])),
                include_groups=False,
            )
            .to_dict()
        )
        return {f"{category_col}_mae_breakdown": json.dumps(breakdown)}

    return _extra


ALIGNER_SPECS: dict[str, AlignerSpec] = {
    "schedule": AlignerSpec(
        name="schedule",
        script_path=REPO_ROOT / "scripts/train_modernbert_schedule_aligner.py",
        needs_diaries=True,
        stratify_col="day_type",
        text_cols=("profile_text", "diary_text"),
        build_pairs=_build_schedule_pairs,
        extra_metrics=_category_breakdown_metrics("day_type"),
    ),
    "activity": AlignerSpec(
        name="activity",
        script_path=REPO_ROOT / "scripts/train_modernbert_activity_aligner.py",
        needs_diaries=True,
        stratify_col="purpose",
        text_cols=("context_text", "activity_text"),
        build_pairs=_build_activity_pairs,
        extra_metrics=_category_breakdown_metrics("purpose"),
    ),
    "vehicle_ownership": AlignerSpec(
        name="vehicle_ownership",
        script_path=REPO_ROOT / "scripts/train_modernbert_vehicle_ownership_aligner.py",
        needs_diaries=False,
        stratify_col="vehicle",
        text_cols=("context_text", "candidate_text"),
        build_pairs=_build_vehicle_pairs,
        extra_metrics=_category_breakdown_metrics("vehicle"),
    ),
    "profile_coherence": AlignerSpec(
        name="profile_coherence",
        script_path=REPO_ROOT / "scripts/train_modernbert_profile_coherence_aligner.py",
        needs_diaries=False,
        stratify_col="variant",
        text_cols=("context_text", "candidate_text"),
        build_pairs=_build_coherence_pairs,
        extra_metrics=_coherence_extra_metrics,
    ),
    "poi_type": AlignerSpec(
        name="poi_type",
        script_path=REPO_ROOT / "scripts/train_modernbert_poi_type_aligner.py",
        needs_diaries=True,
        stratify_col="semantic_cluster",
        text_cols=("context_text", "candidate_text"),
        build_pairs=_build_poi_type_pairs,
        extra_metrics=_category_breakdown_metrics("semantic_cluster"),
    ),
}


def detect_served_model(base_url: str) -> str:
    """Return the model id currently loaded behind an OpenAI-compatible /v1/models."""
    response = requests.get(base_url.rstrip("/") + "/v1/models", timeout=10)
    response.raise_for_status()
    data = response.json()["data"]
    if not data:
        raise RuntimeError(f"{base_url} reports no served models")
    return str(data[0]["id"])


def teacher_label(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "mistral" in lowered:
        return "mistral"
    return "".join(ch if ch.isalnum() else "_" for ch in lowered)


def load_or_label_dataset(
    spec: AlignerSpec,
    *,
    profiles_path: str | Path,
    diary_paths: list[str] | None,
    teacher: str,
    model_id: str,
    base_url: str,
    sample_size: int,
    seed: int,
    concurrency: int,
    cache_dir: Path,
    mutation_ratio: float = 0.5,
    force: bool = False,
) -> pd.DataFrame:
    """Build+label `sample_size` pairs once per (aligner, teacher), cached to parquet.

    Sample sizes are nested prefixes of the same seeded draw sequence (verified for
    all four scripts): pairs[:n] of a size-`sample_size` build equals a standalone
    size-n build. So if a smaller cached size already exists for this (aligner,
    teacher) it is reused as a prefix and only the incremental pairs are labeled
    (and the result is checkpointed under its own `n{sample_size}` filename) —
    growing from 1000 to 10000 later doesn't relabel the first 1000.
    """
    aligner_dir = cache_dir / spec.name
    cache_path = aligner_dir / f"{teacher}_n{sample_size}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    module = load_training_module(spec.script_path)
    profiles = module.load_profiles(profiles_path)
    diaries = module.load_diaries(diary_paths) if spec.needs_diaries else None
    if spec.name == "profile_coherence":
        pairs = _build_coherence_pairs(
            module, profiles, diaries, sample_size, seed, mutation_ratio=mutation_ratio
        )
    else:
        pairs = spec.build_pairs(module, profiles, diaries, sample_size, seed)

    start_from = 0
    prefix_df: pd.DataFrame | None = None
    if aligner_dir.exists() and not force:
        smaller_sizes = []
        for candidate in aligner_dir.glob(f"{teacher}_n*.parquet"):
            try:
                n = int(candidate.stem.rsplit("_n", 1)[-1])
            except ValueError:
                continue
            if n < sample_size:
                smaller_sizes.append((n, candidate))
        if smaller_sizes:
            start_from, best_path = max(smaller_sizes)
            prefix_df = pd.read_parquet(best_path)
            print(f"[{spec.name}/{teacher}] extending cached {start_from} pairs to {sample_size}")

    remaining_pairs = pairs[start_from:]
    if remaining_pairs:
        new_df = module.label_pairs(
            remaining_pairs,
            base_url=base_url,
            model=model_id,
            api_key=None,
            timeout=120.0,
            retries=3,
            concurrency=concurrency,
            progress_interval=max(1, len(remaining_pairs) // 20),
        )
        dataset = pd.concat([prefix_df, new_df], ignore_index=True) if prefix_df is not None else new_df
    else:
        assert prefix_df is not None
        dataset = prefix_df

    aligner_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(cache_path, index=False)
    return dataset


def stratified_split(
    df: pd.DataFrame, stratify_col: str, *, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[stratify_col]
        )
    except ValueError as exc:
        warnings.warn(
            f"stratified split on {stratify_col!r} failed ({exc}); falling back to a "
            "plain random split"
        )
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_common_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall_tau": float(kendalltau(y_true, y_pred).statistic),
    }
    calibration = pd.DataFrame({"pred": y_pred, "true": y_true})
    calibration["bin"] = pd.cut(calibration["pred"], bins=np.linspace(0, 1, 11), include_lowest=True)
    agg = (
        calibration.groupby("bin", observed=False)
        .agg(pred=("pred", "mean"), true=("true", "mean"), n=("true", "size"))
        .dropna()
    )
    metrics["calibration_mae"] = (
        float(np.average(np.abs(agg["pred"] - agg["true"]), weights=agg["n"])) if len(agg) else None
    )
    return metrics


def train_and_evaluate(
    spec: AlignerSpec,
    *,
    teacher: str,
    sample_size: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    models_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    module = load_training_module(spec.script_path)
    output_model_path = models_root / spec.name / teacher / f"n{sample_size}"
    module.train_cross_encoder(
        train_df,
        base_model=base_model,
        output_model_path=str(output_model_path),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(str(output_model_path), device=device)
    col_a, col_b = spec.text_cols
    pairs = test_df[[col_a, col_b]].astype(str).values.tolist()
    predictions = np.asarray(model.predict(pairs, batch_size=128), dtype=float).reshape(-1)
    predictions = np.clip(predictions, 0.0, 1.0)
    del model

    test_df = test_df.copy()
    test_df["prediction"] = predictions

    metrics = compute_common_metrics(test_df["score"].to_numpy(), predictions)
    metrics.update(spec.extra_metrics(test_df))
    metrics.update(
        {
            "aligner": spec.name,
            "teacher": teacher,
            "sample_size": sample_size,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
    )
    return metrics, test_df


def upsert_result(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if results_path.exists():
        existing = pd.read_parquet(results_path)
        key = (row["aligner"], row["teacher"], row["sample_size"])
        mask = ~(
            (existing["aligner"] == key[0])
            & (existing["teacher"] == key[1])
            & (existing["sample_size"] == key[2])
        )
        existing = existing[mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_parquet(results_path, index=False)


def already_done(results_path: Path, aligner: str, teacher: str, sample_size: int) -> bool:
    if not results_path.exists():
        return False
    existing = pd.read_parquet(results_path)
    mask = (
        (existing["aligner"] == aligner)
        & (existing["teacher"] == teacher)
        & (existing["sample_size"] == sample_size)
    )
    return bool(mask.any())
