#!/usr/bin/env bash
# Recompute every manifest row with a report (in place, overwriting the
# existing report JSON) after fixing two bugs found this session:
# (1) citybehavex/cli.py's `report` command never auto-detected the
#     synthetic trajectory's own "purpose" column, so every VPD/ATM/DARD
#     value used a crude HOME/WORK/OTHER heuristic instead of the real
#     simulated activity labels;
# (2) yjmob/yjmob2's evaluation_adaptation.h3_resolution was too fine,
#     inflating visits_per_user by ~400.
# Does NOT touch the manifest: report_json_path values are unchanged, only
# file content at those paths gets overwritten. Ordered fast-to-slow.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST=data/ablation_logs/manifest.jsonl
LOG=data/ablation_logs/recompute_all_driver.log
mkdir -p data/ablation_logs
echo "=== recompute-all started $(date) ===" >> "$LOG"

python3 -c "
import json
rows = [json.loads(l) for l in open('$MANIFEST')]
order = {'gparis': 0, 'shanghai': 1, 'yjmob2': 2, 'yjmob': 3}
rows = [r for r in rows if r.get('report_json_path') and r['variant'] != 'ref']
rows.sort(key=lambda r: order.get(r['dataset'], 9))
for r in rows:
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

echo "=== recompute-all finished $(date) ===" >> "$LOG"
