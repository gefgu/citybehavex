#!/usr/bin/env bash
# Run one ablation config, time it, generate its report, and append a row to
# data/ablation_logs/manifest.jsonl.
#
# Usage: scripts/run_ablation.sh <dataset> <variant> <run_index> [config_path]
#   variant "full" uses the dataset's base config (configs/<dataset>_simulation.yaml)
#   any other variant uses configs/ablations/<dataset>/<dataset>_<variant>.yaml
#   config_path overrides both of the above (e.g. for the shanghai_500sample / *_ref cases)
#
# Set SKIP_REPORT=1 to only run simulate and record the trajectories path,
# skipping the (currently slow) report step -- backfill reports later with
# scripts/backfill_reports.sh once the report path is faster.
set -euo pipefail
SKIP_REPORT="${SKIP_REPORT:-0}"

DATASET="$1"
VARIANT="$2"
RUN_INDEX="$3"
CONFIG_PATH="${4:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$CONFIG_PATH" ]]; then
  if [[ "$VARIANT" == "full" ]]; then
    if [[ -f "configs/ablations/${DATASET}_half.yaml" ]]; then
      # Ablation-matrix "full" column: half-population config (Shanghai/YJMOB/YJMOB2).
      CONFIG_PATH="configs/ablations/${DATASET}_half.yaml"
    else
      # No half config yet (e.g. gparis, deferred): fall back to the dataset's
      # full-population base config (used as-is for the comparison tables).
      CONFIG_PATH="configs/${DATASET}_simulation.yaml"
    fi
  else
    CONFIG_PATH="configs/ablations/${DATASET}/${DATASET}_${VARIANT}.yaml"
  fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "config not found: $CONFIG_PATH" >&2
  exit 1
fi

LOG_DIR="data/ablation_logs"
mkdir -p "$LOG_DIR"
TAG="${DATASET}_${VARIANT}_run${RUN_INDEX}"
TIME_LOG="${LOG_DIR}/${TAG}_time.log"
STDOUT_LOG="${LOG_DIR}/${TAG}_stdout.log"
REPORT_JSON="${LOG_DIR}/${TAG}_report.json"
MANIFEST="${LOG_DIR}/manifest.jsonl"

echo "=== $TAG: simulate ($CONFIG_PATH) ==="
/usr/bin/time -v -o "$TIME_LOG" \
  uv run citybehavex simulate --config "$CONFIG_PATH" \
  > "$STDOUT_LOG" 2>&1

# simulation.output stem for this config, used to find the freshest stamped parquet.
OUTPUT_STEM=$(uv run python -c "
import yaml
with open('$CONFIG_PATH') as f:
    cfg = yaml.safe_load(f)
print(cfg['simulation']['output'])
")
STEM_NO_EXT="${OUTPUT_STEM%.parquet}"
TRAJ_PATH=$(ls -t "${STEM_NO_EXT}"_*.parquet 2>/dev/null | grep -vE '_(activities|activity_alignment|poi_activity_alignment|poi_type_alignment|crp|encounters|moving|social_network)\.parquet$' | head -1)

if [[ -z "$TRAJ_PATH" ]]; then
  echo "could not locate stamped trajectories parquet for stem ${STEM_NO_EXT}" >&2
  exit 1
fi
echo "trajectories: $TRAJ_PATH"

RT_MINUTES=$(grep "Elapsed (wall clock) time" "$TIME_LOG" | sed -E 's/.*: ([0-9:.]+)$/\1/' | \
  LC_ALL=C awk -F: '{ if (NF==3) print ($1*60)+$2+($3/60); else if (NF==2) print $1+($2/60); else print $1/60 }')
MEM_GB=$(grep "Maximum resident set size" "$TIME_LOG" | sed -E 's/.*: ([0-9]+)$/\1/' | \
  LC_ALL=C awk '{ print $1/1024/1024 }')

if [[ "$SKIP_REPORT" == "1" ]]; then
  echo "=== $TAG: report skipped (SKIP_REPORT=1) ==="
  REPORT_JSON=""
else
  echo "=== $TAG: report ==="
  # HTML report generation is deprecated (web UI builds charts on demand now);
  # --json is the only supported machine-readable output.
  uv run citybehavex report --config "$CONFIG_PATH" \
    --synthetic "$TRAJ_PATH" \
    --json "$REPORT_JSON"
fi

PYTHONPATH="" uv run python - "$DATASET" "$VARIANT" "$RUN_INDEX" "$CONFIG_PATH" "$TRAJ_PATH" "$REPORT_JSON" "$RT_MINUTES" "$MEM_GB" "$MANIFEST" <<'PYEOF'
import json
import sys
from datetime import datetime, timezone

dataset, variant, run_index, config_path, traj_path, report_json, rt_minutes, mem_gb, manifest_path = sys.argv[1:10]
row = {
    "dataset": dataset,
    "variant": variant,
    "run_index": int(run_index),
    "config_path": config_path,
    "trajectories_path": traj_path,
    "report_json_path": report_json or None,
    "rt_minutes": float(rt_minutes) if rt_minutes else None,
    "mem_gb": float(mem_gb) if mem_gb else None,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open(manifest_path, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"manifest row appended: {row}")
PYEOF

echo "=== $TAG: done (RT=${RT_MINUTES} min, Mem=${MEM_GB} GB) ==="
