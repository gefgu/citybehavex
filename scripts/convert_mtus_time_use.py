#!/usr/bin/env python3
"""Convert an MTUS Stata time-use table to a Rust-backend-friendly Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TIME_USE_CATEGORIES = (
    "sleep",
    "personal care",
    "household",
    "work",
    "study",
    "shopping",
    "leisure",
    "travel",
    "other",
)

SOURCE_CATEGORY_MAP = {
    "sleep": ("sleep",),
    "personal care": ("eatdrink", "selfcare"),
    "household": (
        "foodprep",
        "cleanetc",
        "maintain",
        "garden",
        "petcare",
        "eldcare",
        "pkidcare",
        "ikidcare",
    ),
    "work": ("paidwork",),
    "study": ("educatn",),
    "shopping": ("shopserv",),
    "leisure": (
        "religion",
        "volorgwk",
        "sportex",
        "tvradio",
        "read",
        "compint",
        "goout",
        "leisure",
    ),
    "travel": ("commute", "travel"),
    "other": ("missing",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source MTUS .dta file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination Parquet path; defaults to INPUT with .parquet extension",
    )
    parser.add_argument("--weight-col", default="propwt")
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--survey-col", default="survey")
    parser.add_argument("--day-col", default="day")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".parquet")
    source_columns = sorted({col for cols in SOURCE_CATEGORY_MAP.values() for col in cols})
    columns = [
        args.country_col,
        args.survey_col,
        args.day_col,
        args.weight_col,
        *source_columns,
    ]
    df = pd.read_stata(args.input, columns=columns)
    out = df[[args.country_col, args.survey_col, args.day_col, args.weight_col]].copy()
    out = out.rename(
        columns={
            args.country_col: "country",
            args.survey_col: "survey",
            args.day_col: "day",
            args.weight_col: args.weight_col,
        }
    )
    for category, source_cols in SOURCE_CATEGORY_MAP.items():
        out[category] = df[list(source_cols)].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"Wrote {len(out):,} rows -> {output}")


if __name__ == "__main__":
    main()
