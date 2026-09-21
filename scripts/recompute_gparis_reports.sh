#!/usr/bin/env bash
# Recompute (in place, overwriting the existing report JSON) every gparis
# report after wiring in real per-trip travel-time ground truth
# (comparison.trip_duration_path -- see configs/ablations/gparis_half.yaml
# and configs/idf_home_work_from_gparis_simulation.yaml). Only
# trip_duration_min changes; every other metric is recomputed identically.
# Does NOT touch the manifest: report_json_path values are unchanged, only
# the file content at those paths gets overwritten with the corrected
# numbers. Modeled directly on scripts/recompute_yjmob_reports.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST=data/ablation_logs/manifest.jsonl
LOG=data/ablation_logs/recompute_gparis_driver.log
mkdir -p data/ablation_logs
echo "=== recompute started $(date) ===" >> "$LOG"

# The "ref" row's own trajectories_path is gparis_visitation_full_data_half_a.parquet
# (used as --synthetic below, i.e. one real half standing in for "synthetic"),
# so it needs an explicit --comparison override to the other real half --
# report_json_path is unique per-row regardless of variant, so a single
# loop with a conditional --comparison argument covers every row.
HALF_B=data/gparis/gparis_visitation_full_data_half_b.parquet

python3 -c "
import json
rows = [json.loads(l) for l in open('$MANIFEST')]
for r in rows:
    if r['dataset'] == 'gparis' and r.get('report_json_path'):
        print(f\"{r['variant']}\t{r['run_index']}\t{r['config_path']}\t{r['trajectories_path']}\t{r['report_json_path']}\")
" | while IFS=$'\t' read -r variant idx config_path traj_path report_json; do
  tag="gparis_${variant}_run${idx}"
  echo "--- $(date) recompute $tag ---" >> "$LOG"
  if [[ "$variant" == "ref" ]]; then
    cmd=(uv run citybehavex report --config "$config_path" --synthetic "$traj_path" --comparison "$HALF_B" --json "$report_json")
  else
    cmd=(uv run citybehavex report --config "$config_path" --synthetic "$traj_path" --json "$report_json")
  fi
  if "${cmd[@]}" >> "$LOG" 2>&1; then
    echo "OK recompute $tag" >> "$LOG"
  else
    echo "FAILED recompute $tag" >> "$LOG"
  fi
done

echo "=== recompute finished $(date) ===" >> "$LOG"
