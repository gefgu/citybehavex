#!/usr/bin/env bash
# Autonomous driver for the full ablation + comparison sweep. Continues past
# individual failures (logs them) so one bad run doesn't stall everything.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=data/ablation_logs/sweep_driver.log
mkdir -p data/ablation_logs
echo "=== sweep started $(date) ===" >> "$LOG"

run() {
  local dataset="$1" variant="$2" idx="$3"
  echo "--- $(date) $dataset/$variant/run$idx ---" >> "$LOG"
  if timeout 10800 bash scripts/run_ablation.sh "$dataset" "$variant" "$idx" >> "$LOG" 2>&1; then
    echo "OK $dataset/$variant/run$idx" >> "$LOG"
  else
    echo "FAILED $dataset/$variant/run$idx (see log above)" >> "$LOG"
  fi
}

VARIANTS="no_profile no_micro_sched no_social no_transport no_feedback"

# Shanghai comparison-table config: redo official 3 runs (no profile caching)
for i in 1 2 3; do
  echo "--- $(date) shanghai_500sample run$i ---" >> "$LOG"
  if timeout 10800 bash scripts/run_ablation.sh shanghai 500sample "$i" configs/shanghai_simulation_500sample.yaml >> "$LOG" 2>&1; then
    echo "OK shanghai_500sample run$i" >> "$LOG"
  else
    echo "FAILED shanghai_500sample run$i" >> "$LOG"
  fi
done

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
