#!/usr/bin/env python
"""Split a real trajectory/visitation dataset into two random user-id halves.

Used to build the ablation table's Ref. baseline (real half A vs real half B)
and to size each ablation config's `simulation.agents` at N/2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


def split_population(
    df: pl.DataFrame, uid_col: str, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    uids = df[uid_col].unique().sort().to_list()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(uids)
    half = len(shuffled) // 2
    uids_a = set(shuffled[:half].tolist())
    uids_b = set(shuffled[half:].tolist())
    df_a = df.filter(pl.col(uid_col).is_in(uids_a))
    df_b = df.filter(pl.col(uid_col).is_in(uids_b))
    return df_a, df_b


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--uid-col", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-a", required=True)
    parser.add_argument("--out-b", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = pl.read_parquet(args.input)
    if args.uid_col not in df.columns:
        raise ValueError(f"{args.uid_col!r} not in columns: {df.columns}")
    df_a, df_b = split_population(df, args.uid_col, args.seed)
    n_a = df_a[args.uid_col].n_unique()
    n_b = df_b[args.uid_col].n_unique()
    Path(args.out_a).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_b).parent.mkdir(parents=True, exist_ok=True)
    df_a.write_parquet(args.out_a)
    df_b.write_parquet(args.out_b)
    print(f"half A: {n_a} users, {df_a.height} rows -> {args.out_a}")
    print(f"half B: {n_b} users, {df_b.height} rows -> {args.out_b}")


if __name__ == "__main__":
    main()
