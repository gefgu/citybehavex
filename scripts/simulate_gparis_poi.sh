#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

"$REPO_ROOT/.venv/bin/python" -u -m citybehavex simulate \
  --min-lon 1.6 \
  --min-lat 48.5 \
  --max-lon 2.975 \
  --max-lat 49.16 \
  --poi-tessellation \
  --output "$REPO_ROOT/data/gparis_poi_trajectories.parquet" \
  --comparison "$REPO_ROOT/data/gparis_visitation_df.parquet" \
  --comparison-label "gparis" \
  --comparison-html "$REPO_ROOT/data/gparis_poi_comparison.html"
