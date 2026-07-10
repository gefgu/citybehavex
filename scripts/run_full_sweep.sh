#!/usr/bin/env bash
# Autonomous driver for the full ablation + comparison sweep. Continues past
# individual failures (logs them) so one bad run doesn't stall everything.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=data/ablation_logs/sweep_driver.log
mkdir -p data/ablation_logs
echo "=== sweep started $(date) ===" >> "$LOG"

MANIFEST=data/ablation_logs/manifest.jsonl

already_done() {
  local dataset="$1" variant="$2" idx="$3"
  [[ -f "$MANIFEST" ]] || return 1
  python3 -c "
import json, sys
target = ('$dataset', '$variant', $idx)
for line in open('$MANIFEST'):
    r = json.loads(line)
    if (r['dataset'], r['variant'], r['run_index']) == target:
        sys.exit(0)
sys.exit(1)
"
}

run() {
  local dataset="$1" variant="$2" idx="$3"
  if already_done "$dataset" "$variant" "$idx"; then
    echo "--- $(date) $dataset/$variant/run$idx: already in manifest, skipping ---" >> "$LOG"
    return
  fi
  # YJMOB/YJMOB2's real comparison dataset is tens of millions of rows;
  # report computation over that needs more headroom than Shanghai's
  # (irrelevant right now while SKIP_REPORT=1, but kept for when reports
  # come back on).
  local run_timeout=1800
  [[ "$dataset" == "yjmob" || "$dataset" == "yjmob2" ]] && run_timeout=7200
  echo "--- $(date) $dataset/$variant/run$idx ---" >> "$LOG"
  if timeout "$run_timeout" bash scripts/run_ablation.sh "$dataset" "$variant" "$idx" >> "$LOG" 2>&1; then
    echo "OK $dataset/$variant/run$idx" >> "$LOG"
  else
    echo "FAILED $dataset/$variant/run$idx (see log above)" >> "$LOG"
  fi
}

VARIANTS="no_profile no_micro_sched no_social no_transport no_feedback"

# Shanghai comparison-table runs already completed and applied to
# paper/comparision_table.tex -- not repeated here.

# Ablation matrix: 3 datasets x (full + 5 variants) x 3 rounds.
for round in 1 2 3; do
  for dataset in shanghai yjmob yjmob2; do
    run "$dataset" full "$round"
    for v in $VARIANTS; do
      run "$dataset" "$v" "$round"
    done
  done
done

echo "=== sweep finished $(date) ===" >> "$LOG"
uv run python scripts/aggregate_ablation_results.py --apply >> "$LOG" 2>&1
echo "=== aggregator applied $(date) ===" >> "$LOG"
