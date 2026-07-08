# Ablation + comparison-table sweep — runbook

Status snapshot and exact commands to resume this work if it gets interrupted
(usage limits, session restart, etc.). See also
`/home/gustavo/.claude/plans/i-ve-made-some-updates-tingly-kazoo.md` for the
original plan with full context/rationale.

## What this is for

`paper/ablation.tex` and `paper/comparision/{spatial,temporal,semantic}_table.tex`
(gitignored — not committed, edited in place) are LaTeX tables comparing
CityBehavEx (CBX) against AgentSociety/CitySim baselines and an ablation study
(full model vs. 5 module-ablations) across 4 datasets: gparis, Shanghai, YJMOB,
YJMOB-disaster. Most cells are still `X.X` placeholders.

## Key methodology decisions (don't re-derive these)

- **Ablation variants** (`scripts/make_ablation_config.py`, `VARIANTS` dict):
  `no_profile` = `profiles.enabled: false`; `no_micro_sched` =
  `activities.enabled: false`; `no_social` = `simulation.alpha: 0.0`;
  `no_transport` = `road_network.enabled: false` + `rail_network.enabled: false`;
  `no_feedback` = reset manually-tuned hyperparameters (motif_exploration_rate,
  location_count_sigma, schedule beta params, activities.kappa/temperature,
  activities.durations block, work_from_home_probability,
  gravity_deterrence_exponent) back to Pydantic schema defaults, leaving
  module-enable flags alone.
- **Population halving** (ablation table only): Shanghai/YJMOB/YJMOB-disaster
  simulate at N/2 agents vs. a held-out random half of the real population
  (`scripts/split_real_population.py`, seed 42, split once, reused everywhere).
  The other real half gives the `Ref.` column (real half A vs. real half B,
  no simulation — just `citybehavex report --synthetic <half_a> --comparison
  <half_b>`, since `load_trajectory`'s column auto-detection already matches
  all 3 raw schemas). **gparis is deferred** — the user is adding a larger
  Paris reference dataset later; don't build gparis's N/2 config until then.
- **Comparison tables (spatial/temporal/semantic)** use a *different*,
  already-established methodology: fixed 500-agent samples compared against
  the *full* real dataset, matching how AgentSociety/CitySim were evaluated
  (see `spatial_table.tex`'s prose). This is NOT the N/2 split above. gparis's
  500-agent config is `configs/gparis_simulation.yaml` as-is. Shanghai's is
  `configs/shanghai_simulation_500sample.yaml` (new).
- **New ablation-table rows**: 4 synthetic-only network metrics already
  computed by `citybehavex/reports/network_validation.py`'s `synthetic_vs_random`
  block (degree, clustering coefficient, edge persistence, topological
  overlap — Wasserstein vs. a degree-preserving random null graph, no real
  friendship data needed). Filled for Shanghai/YJMOB/YJMOB-disaster; **left
  blank ("-") for gparis** per explicit instruction.
- **Caching vs. variance tradeoff**: setting `profiles.profiles_path` to the
  same path as `profiles.output` makes profile generation (and therefore
  most of the activity/schedule alignment cache) fully reused across repeated
  runs of an *unchanged* config — this is the right move while tuning one
  parameter at a time (10x+ speedup), but it makes repeated runs of the same
  config deterministic (std=0), which defeats the point of averaging 3 runs.
  **Rule: use `profiles_path` caching only during tuning/improvement
  iterations. Remove it (let profiles regenerate fresh) for the official
  3-run set that goes into a table**, matching how gparis's original 3 runs
  already worked (no caching, genuine run-to-run variance from LLM-driven
  profile calibration).
- **Bug fixed this session**: `visits_per_user` Wasserstein distance was
  comparing raw row-count (real data, uncollapsed) against collapsed
  stay-episodes (synthetic), massively inflating the distance for
  check-in-dense datasets like Shanghai (142 -> ~21 after the fix). Fixed in
  two places — `citybehavex/reports/comparison.py` (`generate_comparison_report`,
  CLI path) and `web/backend/app/payload/legacy.py` (`distribution_group`,
  live web UI path) — both now collapse the real side with `_collapse_to_stays`
  before computing/plotting visits-per-user. If you see an implausibly large
  Vf number again, check whether a *third* call site to
  `visits_per_user_wasserstein_distance` has been added without the same fix
  (`grep -rn visits_per_user_wasserstein_distance --include=*.py .`).
  Also clear `data/.web_cache/v8__<config>__*.json` after any comparison.py
  metric fix — the web backend disk-caches computed payloads by
  config+run-timestamp and won't recompute until those are cleared.

## Progress so far

- gparis: 3 official 500-agent runs done (fine-tuned aligners), values in
  `data/ablation_logs/manifest.jsonl` (dataset=gparis, variant=full) and
  already written into all 4 `.tex` files via the aggregator's `--apply`.
- Shanghai 500-agent comparison-table config: 3 rounds of improvement tried
  against `configs/shanghai_simulation_500sample.yaml` (see manifest entries
  `improve_r1`/`improve_r2`/`improve_r3` if not yet cleaned up, or check
  `data/ablation_logs/shanghai_improve_r*_report.json` directly): round 1
  (poi_type_choice_enabled) was a wash; round 2 (gravity_deterrence_exponent
  -2.0->-2.5) and round 3 (->-3.0, work_distance_max_km 60->30) meaningfully
  improved Δr/r_g (~30%/~21%) but still don't beat CitySim/AgentSociety.
  Round 3's settings are locked into `configs/shanghai_simulation_500sample.yaml`.
  **The official 3-run set for this config still needs to be (re)run without
  `profiles_path` caching** — an earlier 3-run attempt used cached profiles
  and produced std=0 (invalid for an error bar), was discarded from the
  manifest, and needs redoing.
- Ablation-matrix configs generated (not yet run): `configs/ablations/
  {shanghai,yjmob,yjmob2}_half.yaml` (N/2-agent full-model base) and
  `configs/ablations/{shanghai,yjmob,yjmob2}/{dataset}_{no_profile,
  no_micro_sched,no_social,no_transport,no_feedback}.yaml`. Real-population
  halves already split: `data/{shanghai,yjmob,yjmob2}/*_half_{a,b}.parquet`.
  **None of these 18 configs have been run yet** — this is the bulk of
  remaining work.

## How to resume

1. Check `data/ablation_logs/manifest.jsonl` for what's already run (one JSON
   line per run: dataset, variant, run_index, config_path, report_json_path,
   rt_minutes, mem_gb).
2. Confirm the 3 servers + vLLM are up before running anything:
   ```bash
   for p in 8081 8082 8083 8084; do curl -s -m 2 -o /dev/null -w "$p: %{http_code}\n" http://localhost:$p/health; done
   ```
   If down, restart the aligner rerank servers (schedule/activity/ownership):
   ```bash
   screen -dmS schedule-aligner bash -c 'PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages .venv/bin/python scripts/serve_schedule_aligner.py --model-path models/modernbert-schedule-aligner --port 8082 --device cuda --predict-batch-size 128; exec bash'
   screen -dmS activity-aligner bash -c 'PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages .venv/bin/python scripts/serve_schedule_aligner.py --model-path models/modernbert-activity-aligner --port 8083 --device cuda --predict-batch-size 128; exec bash'
   screen -dmS ownership-aligner bash -c 'PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages .venv/bin/python scripts/serve_schedule_aligner.py --model-path models/modernbert-vehicle-ownership-aligner --port 8084 --device cuda --predict-batch-size 128; exec bash'
   ```
3. **Redo Shanghai's official 3-run set** (comparison tables), no profile caching:
   ```bash
   for i in 1 2 3; do
     bash scripts/run_ablation.sh shanghai 500sample $i configs/shanghai_simulation_500sample.yaml
   done
   ```
4. **Run the ablation matrix**, one dataset at a time, smoke-test the
   full-model variant first before committing to the other 5:
   ```bash
   # Shanghai
   bash scripts/run_ablation.sh shanghai full 1          # smoke test, check RT/output sane
   for v in no_profile no_micro_sched no_social no_transport no_feedback; do
     bash scripts/run_ablation.sh shanghai $v 1
   done
   # YJMOB (highest risk: 50k agents, 75-day stream_output, never run at this scale with this pipeline)
   bash scripts/run_ablation.sh yjmob full 1
   for v in no_profile no_micro_sched no_social no_transport no_feedback; do
     bash scripts/run_ablation.sh yjmob $v 1
   done
   # YJMOB-disaster
   bash scripts/run_ablation.sh yjmob2 full 1
   for v in no_profile no_micro_sched no_social no_transport no_feedback; do
     bash scripts/run_ablation.sh yjmob2 $v 1
   done
   ```
   Each `run_ablation.sh <dataset> <variant> <run_index>` call: runs
   `simulate` under `/usr/bin/time -v`, locates the stamped trajectories
   parquet, runs `citybehavex report --json` (no `--output`/HTML — that flag
   was removed, HTML reports are deprecated in favor of the web UI), and
   appends one manifest row.
5. Compute each dataset's `Ref.` cell (real half A vs. real half B, run once,
   not part of the 3-run sweep):
   ```bash
   uv run citybehavex report \
     --synthetic data/shanghai/shanghai_data_raw_half_a.parquet \
     --comparison data/shanghai/shanghai_data_raw_half_b.parquet \
     --json data/ablation_logs/shanghai_ref_report.json
   # repeat for yjmob / yjmob2 with their half_a/half_b paths
   ```
   then add a manifest row by hand with `variant: "ref"` (see
   `scripts/aggregate_ablation_results.py`'s `aggregate()` — it doesn't
   special-case "ref" yet, so patching the `Ref.` column currently needs a
   small manual/aggregator addition; not yet wired up).
6. After round 1 (1 run per variant per dataset) — review with
   `uv run python scripts/aggregate_ablation_results.py` (dry-run by default),
   then run 2 more rounds per variant (`run_index` 2 and 3) for the same
   configs, then `--apply` to patch `paper/ablation.tex`.
7. Re-run `uv run python scripts/aggregate_ablation_results.py --apply` any
   time after new manifest rows land — it's idempotent and only touches the
   specific cells it has data for.

## Known gaps / things to double check before trusting numbers

- The `Ref.` column patching isn't wired into `aggregate_ablation_results.py`
  yet — only `wasserstein`/`jsd`/network-validation metrics for actual model
  variants are. Add a `variant == "ref"` branch before relying on it.
- Shanghai/YJMOB/YJMOB-disaster ablation configs were generated from a base
  config that does NOT include the Δr/r_g improvement tuning found for
  Shanghai's comparison-table config (gravity_deterrence_exponent,
  work_distance_max_km) — decide whether to carry those into the ablation
  matrix's `shanghai_half.yaml` before running it, or keep the ablation study
  on the untuned baseline intentionally (arguably more representative of
  ablating each module against a common, unoptimized reference point).
- `citybehavex report`'s HTML output was removed this session (see commit
  "Remove standalone HTML comparison report") — the web UI is now the only
  way to see charts; JSON is the only machine-readable output.
