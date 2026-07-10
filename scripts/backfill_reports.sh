#!/usr/bin/env bash
# Run the (now-fast) report step for every manifest row that was recorded
# with SKIP_REPORT=1 (report_json_path is null), without re-running simulate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST=data/ablation_logs/manifest.jsonl
LOG=data/ablation_logs/backfill_driver.log
mkdir -p data/ablation_logs
echo "=== backfill started $(date) ===" >> "$LOG"

python3 -c "
import json
rows = [json.loads(l) for l in open('$MANIFEST')]
for r in rows:
    if r.get('report_json_path') is None:
        print(f\"{r['dataset']}\t{r['variant']}\t{r['run_index']}\t{r['config_path']}\t{r['trajectories_path']}\")
" | while IFS=$'\t' read -r dataset variant idx config_path traj_path; do
  tag="${dataset}_${variant}_run${idx}"
  report_json="data/ablation_logs/${tag}_report.json"
  echo "--- $(date) backfill $tag ---" >> "$LOG"
  if uv run citybehavex report --config "$config_path" --synthetic "$traj_path" --json "$report_json" >> "$LOG" 2>&1; then
    python3 -c "
import json
from datetime import datetime, timezone
rows = [json.loads(l) for l in open('$MANIFEST')]
for r in rows:
    if r['dataset']=='$dataset' and r['variant']=='$variant' and r['run_index']==$idx and r.get('report_json_path') is None:
        r['report_json_path'] = '$report_json'
        r['timestamp'] = datetime.now(timezone.utc).isoformat()
with open('$MANIFEST', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
"
    echo "OK backfill $tag" >> "$LOG"
  else
    echo "FAILED backfill $tag" >> "$LOG"
  fi
done

echo "=== backfill finished $(date) ===" >> "$LOG"
uv run python scripts/aggregate_ablation_results.py --apply >> "$LOG" 2>&1
echo "=== aggregator applied $(date) ===" >> "$LOG"
