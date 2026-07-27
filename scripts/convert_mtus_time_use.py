#!/usr/bin/env python3
"""Convert an MTUS Stata time-use table to a Rust-backend-friendly Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Mirrors the Rust backend's time-use categories exactly
# -- the 25 raw MTUS harmonized activity codes. A prior version of this
# script aggregated these into a 9-bucket rollup so the Rust backend (which
# can't parse Stata) had a small, easy schema to read -- but that meant the
# Rust and Python comparisons were built from two different granularities of
# the same source data and could never numerically agree. Passing the raw
# codes straight through (still no Stata parsing needed in Rust -- this
# script remains the one-time Python-side pre-step) makes both backends read
# the exact same categories.
TIME_USE_CATEGORIES = (
    "sleep",
    "eatdrink",
    "selfcare",
    "paidwork",
    "educatn",
    "foodprep",
    "cleanetc",
    "maintain",
    "shopserv",
    "garden",
    "petcare",
    "eldcare",
    "pkidcare",
    "ikidcare",
    "religion",
    "volorgwk",
    "commute",
    "travel",
    "sportex",
    "tvradio",
    "read",
    "compint",
    "goout",
    "leisure",
    "missing",
)


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
    columns = [
        args.country_col,
        args.survey_col,
        args.day_col,
        args.weight_col,
        *TIME_USE_CATEGORIES,
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
    for category in TIME_USE_CATEGORIES:
        out[category] = pd.to_numeric(df[category], errors="coerce").fillna(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"Wrote {len(out):,} rows -> {output}")


if __name__ == "__main__":
    main()
