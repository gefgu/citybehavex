#!/usr/bin/env bash
# Recompute (in place, overwriting the existing report JSON) all yjmob/yjmob2
# ablation reports after fixing evaluation_adaptation.h3_resolution (was too
# fine, causing a spurious ~400+ visits_per_user Wasserstein gap -- see
# EVALUATION_NOTES.md). Does NOT touch the manifest: report_json_path values
# are unchanged, only the file content at those paths gets overwritten with
# the corrected numbers.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST=data/ablation_logs/manifest.jsonl
LOG=data/ablation_logs/recompute_yjmob_driver.log
mkdir -p data/ablation_logs
echo "=== recompute started $(date) ===" >> "$LOG"

python3 -c "
import json
rows = [json.loads(l) for l in open('$MANIFEST')]
for r in rows:
    if r['dataset'] in ('yjmob', 'yjmob2') and r['variant'] != 'ref' and r.get('report_json_path'):
        print(f\"{r['dataset']}\t{r['variant']}\t{r['run_index']}\t{r['config_path']}\t{r['trajectories_path']}\t{r['report_json_path']}\")
" | while IFS=$'\t' read -r dataset variant idx config_path traj_path report_json; do
  tag="${dataset}_${variant}_run${idx}"
  echo "--- $(date) recompute $tag ---" >> "$LOG"
  if uv run citybehavex report --config "$config_path" --synthetic "$traj_path" --json "$report_json" >> "$LOG" 2>&1; then
    echo "OK recompute $tag" >> "$LOG"
  else
    echo "FAILED recompute $tag" >> "$LOG"
  fi
done

echo "=== recompute finished $(date) ===" >> "$LOG"
