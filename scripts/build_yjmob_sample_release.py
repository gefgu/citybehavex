#!/usr/bin/env python
"""Build the public YJMOB-1k demo sample GitHub Release asset.

The shipped sample is entirely synthetic: it is a subsample (first
``--agents`` agents, first ``--days`` days) of an already-completed
CityBehavEx simulation run, not derived from the real YJMob100K challenge
dataset. That sidesteps the challenge dataset's redistribution restrictions
entirely -- nothing in the released asset originates from YJMob100K. Users
who want to validate against the real dataset still need to download and
preprocess it themselves; see the "Setting up data/ for the YJMOB scenario"
section of the README.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
from datetime import timedelta
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SAMPLE_TRAJECTORIES_NAME = "yjmob_1k_simulated_sample.parquet"
TESSELLATION_NAME = "yjmob_h3_tessellation.parquet"


def _build_sample(source_trajectories: Path, agents: int, days: int):
    dataset = ds.dataset(source_trajectories)
    uids = dataset.to_table(columns=["uid"])["uid"].to_pylist()
    first_uids = sorted(set(uids))[:agents]
    if len(first_uids) < agents:
        raise SystemExit(f"source has only {len(first_uids)} distinct agents, need {agents}")
    start = pc.min(dataset.to_table(columns=["datetime"])["datetime"]).as_py()
    cutoff = start + timedelta(days=days)
    return dataset.to_table(
        filter=(ds.field("uid").isin(first_uids)) & (ds.field("datetime") < cutoff)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-trajectories",
        type=Path,
        required=True,
        help="An already-completed CityBehavEx simulation trajectories parquet.",
    )
    parser.add_argument(
        "--source-tessellation",
        type=Path,
        required=True,
        help="The matching H3 tessellation parquet (built from public Overture Maps data).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=1000, help="Number of agents to keep.")
    parser.add_argument("--days", type=int, default=7, help="Number of days to keep.")
    args = parser.parse_args()

    if not args.source_tessellation.is_file():
        raise SystemExit(f"missing source tessellation: {args.source_tessellation}")

    sample = _build_sample(args.source_trajectories, args.agents, args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample_path = tmp_path / SAMPLE_TRAJECTORIES_NAME
        pq.write_table(sample, sample_path)
        provenance_path = tmp_path / "PROVENANCE.md"
        provenance_path.write_text(
            "This sample is synthetic CityBehavEx simulation output "
            f"({args.source_trajectories.name}), subsampled to the first "
            f"{args.agents} agents and first {args.days} days. It is not "
            "derived from the real YJMob100K challenge dataset.\n"
        )
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(
                sample_path, arcname=f"data/yjmob-1k/{SAMPLE_TRAJECTORIES_NAME}", recursive=False
            )
            archive.add(
                args.source_tessellation,
                arcname=f"data/yjmob-1k/{TESSELLATION_NAME}",
                recursive=False,
            )
            archive.add(provenance_path, arcname="PROVENANCE.md", recursive=False)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"asset: {args.output}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
