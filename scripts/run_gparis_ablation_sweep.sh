#!/usr/bin/env bash
# gparis-only ablation sweep: full (1500 agents, N/2 baseline) + 5 variants x 3
# rounds. Mirrors run_full_sweep.sh's skip-already-done logic but scoped to
# gparis only, since shanghai/yjmob/yjmob2 are already complete. Run with
# SKIP_REPORT=1 until the full gparis comparison dataset is ready -- reports
# get backfilled later via scripts/backfill_reports.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=data/ablation_logs/gparis_sweep_driver.log
mkdir -p data/ablation_logs
echo "=== gparis sweep started $(date) ===" >> "$LOG"

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
  echo "--- $(date) $dataset/$variant/run$idx ---" >> "$LOG"
  # 1500 agents is 3x the 500-agent comparison-table scale (which took
  # ~28.5min); give real headroom rather than reusing shanghai's 30min cap.
  if timeout 5400 bash scripts/run_ablation.sh "$dataset" "$variant" "$idx" >> "$LOG" 2>&1; then
    echo "OK $dataset/$variant/run$idx" >> "$LOG"
  else
    echo "FAILED $dataset/$variant/run$idx (see log above)" >> "$LOG"
  fi
}

VARIANTS="no_profile no_micro_sched no_social no_transport no_feedback"

for round in 1 2 3; do
  run gparis full "$round"
  for v in $VARIANTS; do
    run gparis "$v" "$round"
  done
done

echo "=== gparis sweep finished $(date) ===" >> "$LOG"
