#!/usr/bin/env python
"""Build the license-cleared prepared YJMOB-1k GitHub Release asset.

The input directory must contain only redistributable derived files.  This
script deliberately does not download or include the original challenge CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

REQUIRED = (
    "data/yjmob-1k/yjmob_h3_tessellation.parquet",
    "data/yjmob-1k/yjmob_wgs84_simple.parquet",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.input_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing required prepared inputs: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name in REQUIRED:
            archive.add(args.input_dir / name, arcname=name, recursive=False)
        for name in ("LICENSE", "PROVENANCE.md"):
            candidate = args.input_dir / name
            if candidate.is_file():
                archive.add(candidate, arcname=name, recursive=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"asset: {args.output}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
