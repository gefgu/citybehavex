#!/usr/bin/env python
"""Drive the cross-encoder aligner evaluation sweep: 4 aligners x 3 sample sizes,
for whichever teacher LLM is currently served at --llm-base-url.

Resumable: skips (aligner, teacher, sample_size) combos already present in
`results.parquet` unless --force. Labels 10000 pairs once per (aligner, teacher)
and slices prefixes for the smaller sample sizes (see cross_encoder_eval_lib for
why that's equivalent to labeling each size independently).

Run once now (labels whichever model /v1/models currently reports), then again
after switching the sibling server to the other teacher model.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_encoder_eval_lib import (  # noqa: E402
    ALIGNER_SPECS,
    already_done,
    detect_served_model,
    load_or_label_dataset,
    stratified_split,
    teacher_label,
    train_and_evaluate,
    upsert_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-path",
        default=str(REPO_ROOT / "data/gparis/results/gparis_agent_profiles.parquet"),
    )
    parser.add_argument(
        "--diary-path",
        action="append",
        default=None,
        help="Repeatable. Defaults to gparis weekday+weekend diaries.",
    )
    parser.add_argument("--llm-base-url", default="http://200.134.10.65:8081")
    parser.add_argument("--llm-model", default=None, help="Override auto-detected model id.")
    parser.add_argument("--llm-concurrency", type=int, default=16)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument(
        "--aligners",
        nargs="+",
        default=list(ALIGNER_SPECS),
        choices=list(ALIGNER_SPECS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-ratio", type=float, default=0.5)
    parser.add_argument("--base-model", default="nomic-ai/modernbert-embed-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "data/eval/cross_encoders")
    )
    parser.add_argument("--models-root", default=str(REPO_ROOT / "models/eval"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    diary_paths = args.diary_path or [
        str(REPO_ROOT / "data/llm_diaries_gparis/validated_diaries_weekday.json"),
        str(REPO_ROOT / "data/llm_diaries_gparis/validated_diaries_weekend.json"),
    ]
    data_root = Path(args.data_root)
    models_root = Path(args.models_root)
    results_path = data_root / "results.parquet"

    model_id = args.llm_model or detect_served_model(args.llm_base_url)
    teacher = teacher_label(model_id)
    max_size = max(args.sample_sizes)
    print(f"Served model: {model_id!r} -> teacher={teacher!r}. Max sample size: {max_size}.")

    for aligner_name in args.aligners:
        spec = ALIGNER_SPECS[aligner_name]
        pending_sizes = [
            size
            for size in args.sample_sizes
            if args.force or not already_done(results_path, spec.name, teacher, size)
        ]
        if not pending_sizes:
            print(f"[{spec.name}/{teacher}] all sample sizes already done, skipping.")
            continue

        print(f"[{spec.name}/{teacher}] loading/labeling {max_size} pairs (cached if present)...")
        dataset = load_or_label_dataset(
            spec,
            profiles_path=args.profiles_path,
            diary_paths=diary_paths if spec.needs_diaries else None,
            teacher=teacher,
            model_id=model_id,
            base_url=args.llm_base_url,
            sample_size=max_size,
            seed=args.seed,
            concurrency=args.llm_concurrency,
            cache_dir=data_root,
            mutation_ratio=args.mutation_ratio,
            force=args.force,
        )
        print(f"[{spec.name}/{teacher}] dataset ready: {len(dataset)} labeled pairs.")

        for size in pending_sizes:
            print(f"[{spec.name}/{teacher}/n{size}] splitting + training + evaluating...")
            subset = dataset.iloc[:size].reset_index(drop=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                train_df, test_df = stratified_split(subset, spec.stratify_col, seed=args.seed)
                for warning in caught:
                    print(f"  warning: {warning.message}")

            metrics, predictions_df = train_and_evaluate(
                spec,
                teacher=teacher,
                sample_size=size,
                train_df=train_df,
                test_df=test_df,
                base_model=args.base_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                models_root=models_root,
            )

            predictions_path = data_root / spec.name / f"{teacher}_n{size}_predictions.parquet"
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_parquet(predictions_path, index=False)

            upsert_result(results_path, metrics)
            print(
                f"[{spec.name}/{teacher}/n{size}] done: "
                f"mae={metrics['mae']:.4f} spearman={metrics['spearman']:.4f} "
                f"n_train={metrics['n_train']} n_test={metrics['n_test']}"
            )

    print("Sweep complete for this teacher. Re-run after switching the sibling server "
          "to the other model to fill in the remaining combos.")


if __name__ == "__main__":
    main()
