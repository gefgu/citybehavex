#!/usr/bin/env python
"""Evaluate cached CrossEncoder labels at multiple training epoch counts.

Each (aligner, sample size, epoch) is checkpointed immediately, so an
interrupted sweep resumes without repeating completed training runs.  The
script deliberately uses only cached labeled data; it makes no LLM requests.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_encoder_eval_lib import ALIGNER_SPECS, stratified_split, train_and_evaluate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="mistral")
    parser.add_argument("--epochs", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--aligners", nargs="+", default=[
        "schedule", "activity", "vehicle_ownership", "profile_coherence",
    ], choices=list(ALIGNER_SPECS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data/eval/cross_encoders"))
    parser.add_argument("--models-root", default=str(REPO_ROOT / "models/eval_epoch_sweep"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def largest_cached_dataset(data_root: Path, aligner: str, teacher: str) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in (data_root / aligner).glob(f"{teacher}_n*.parquet"):
        try:
            candidates.append((int(path.stem.rsplit("_n", 1)[1]), path))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No cached {teacher} labels for {aligner} under {data_root}")
    return max(candidates)[1]


def is_done(results: pd.DataFrame, aligner: str, teacher: str, size: int, epochs: int) -> bool:
    if results.empty:
        return False
    return bool(((results["aligner"] == aligner) & (results["teacher"] == teacher)
                 & (results["sample_size"] == size) & (results["epochs"] == epochs)).any())


def checkpoint(results_path: Path, results: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    row = pd.DataFrame([metrics])
    if not results.empty:
        key = ((results["aligner"] == metrics["aligner"]) & (results["teacher"] == metrics["teacher"])
               & (results["sample_size"] == metrics["sample_size"]) & (results["epochs"] == metrics["epochs"]))
        results = results.loc[~key]
    results = pd.concat([results, row], ignore_index=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path, index=False)
    return results


def main(argv=None) -> None:
    args = parse_args(argv)
    data_root, models_root = Path(args.data_root), Path(args.models_root)
    results_path = data_root / f"epoch_sweep_{args.teacher}.parquet"
    results = pd.read_parquet(results_path) if results_path.exists() else pd.DataFrame()

    for aligner_name in args.aligners:
        spec = ALIGNER_SPECS[aligner_name]
        dataset_path = largest_cached_dataset(data_root, aligner_name, args.teacher)
        dataset = pd.read_parquet(dataset_path)
        print(f"[{aligner_name}] using {len(dataset)} cached labels from {dataset_path.name}", flush=True)
        for size in args.sample_sizes:
            if len(dataset) < size:
                raise ValueError(f"{dataset_path} has {len(dataset)} labels, needs {size}")
            subset = dataset.iloc[:size].reset_index(drop=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                train_df, test_df = stratified_split(subset, spec.stratify_col, seed=args.seed)
            for epochs in args.epochs:
                if not args.force and is_done(results, aligner_name, args.teacher, size, epochs):
                    print(f"[{aligner_name}/n{size}/epochs={epochs}] already done", flush=True)
                    continue
                print(f"[{aligner_name}/n{size}/epochs={epochs}] training on {args.device}", flush=True)
                metrics, predictions = train_and_evaluate(
                    spec, teacher=args.teacher, sample_size=size, train_df=train_df, test_df=test_df,
                    base_model=args.base_model, epochs=epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, device=args.device,
                    models_root=models_root / f"epochs{epochs}",
                )
                metrics["epochs"] = epochs
                prediction_path = data_root / aligner_name / (
                    f"{args.teacher}_n{size}_epochs{epochs}_predictions.parquet"
                )
                predictions.to_parquet(prediction_path, index=False)
                results = checkpoint(results_path, results, metrics)
                print(f"[{aligner_name}/n{size}/epochs={epochs}] mae={metrics['mae']:.4f} "
                      f"spearman={metrics['spearman']:.4f}", flush=True)


if __name__ == "__main__":
    main()
