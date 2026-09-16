#!/usr/bin/env bash
# Run the (now-fast) report step for every manifest row that was recorded
# with SKIP_REPORT=1 (report_json_path is null), without re-running simulate.
#
# Set SKIP_NETWORK_VALIDATION=1 to pass --skip-network-validation through to
# every `citybehavex report` call -- the social/contact-network validation
# section (co-presence graph + degree/clustering/persistence/topological
# overlap) is currently a known slow path (minutes, vs seconds for every
# other section combined). Once that's fixed, rerun with
# SKIP_NETWORK_VALIDATION unset (or =0) to backfill it: rows already
# reported here still have report_json_path set, so re-run
# aggregate_ablation_results.py's underlying report step manually per-row,
# or just delete the affected report_json_path entries from the manifest and
# rerun this script to regenerate them with network validation included.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SKIP_NETWORK_VALIDATION="${SKIP_NETWORK_VALIDATION:-0}"
NV_FLAG=""
if [[ "$SKIP_NETWORK_VALIDATION" == "1" ]]; then
  NV_FLAG="--skip-network-validation"
fi

MANIFEST=data/ablation_logs/manifest.jsonl
LOG=data/ablation_logs/backfill_driver.log
mkdir -p data/ablation_logs
echo "=== backfill started $(date) (SKIP_NETWORK_VALIDATION=$SKIP_NETWORK_VALIDATION) ===" >> "$LOG"

python3 -c "
import json
rows = [json.loads(l) for l in open('$MANIFEST')]
# Report/CPC compute time tracks the real comparison dataset's row count, not
# agent count: shanghai (5.3M rows) < yjmob2 (14.6M) < yjmob (55.8M) -- run
# fastest-to-slowest so more of the table fills in early rather than stalling
# on yjmob's much slower reports first.
order = {'gparis': 0, 'shanghai': 1, 'yjmob2': 2, 'yjmob': 3}
rows = [r for r in rows if r.get('report_json_path') is None]
rows.sort(key=lambda r: order.get(r['dataset'], 9))
for r in rows:
    print(f\"{r['dataset']}\t{r['variant']}\t{r['run_index']}\t{r['config_path']}\t{r['trajectories_path']}\")
" | while IFS=$'\t' read -r dataset variant idx config_path traj_path; do
  tag="${dataset}_${variant}_run${idx}"
  report_json="data/ablation_logs/${tag}_report.json"
  echo "--- $(date) backfill $tag ---" >> "$LOG"
  if uv run citybehavex report --config "$config_path" --synthetic "$traj_path" --json "$report_json" $NV_FLAG >> "$LOG" 2>&1; then
    # flock around the read-modify-write: this rewrites the whole manifest,
    # so it must not interleave with run_ablation.sh's plain append (used by
    # the simulate sweep running in parallel to reuse idle CPU) -- same
    # lockfile/fd convention as there.
    (
      flock -x 200
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
    ) 200>"${MANIFEST}.lock"
    echo "OK backfill $tag" >> "$LOG"
  else
    echo "FAILED backfill $tag" >> "$LOG"
  fi
done

echo "=== backfill finished $(date) ===" >> "$LOG"
uv run python scripts/aggregate_ablation_results.py --apply >> "$LOG" 2>&1
echo "=== aggregator applied $(date) ===" >> "$LOG"
