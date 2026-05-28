#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

"$REPO_ROOT/.venv/bin/python" -m citybehavex simulate \
  --min-lon 120.88 \
  --min-lat 30.63 \
  --max-lon 122.06 \
  --max-lat 31.84 \
  --output "$REPO_ROOT/data/shanghai_trajectories.parquet" \
  --comparison "$REPO_ROOT/data/shanghai_quadratic_visitation_h3_sample.parquet" \
  --comparison-label "quadratic visitation" \
  --comparison-html "$REPO_ROOT/data/shanghai_comparison.html" \
  --comparison-datetime-col start_timestamp \
  --comparison-uid-col user_id \
  --comparison-lng-col lon
