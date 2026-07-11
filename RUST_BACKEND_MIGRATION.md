# Rust/axum backend migration

## Goal

Replace `web/backend/`'s FastAPI/Python server with a byte-compatible axum
server written entirely in Rust (`citybehavex-web/`), **without deleting or
modifying the Python one**. Motivation: axum + tokio give real multi-core
request parallelism; the Python server works around the GIL with a
`ProcessPoolExecutor` for CPU-bound builds, which is heavier and caps
throughput compared to native async/rayon parallelism. Both backends will be
runnable side by side (different ports); `web/frontend/` talks to whichever
one is running, unmodified.

This file is the living status/handoff doc. The original phased plan (the
source these phase numbers refer to) was written to
`~/.claude/plans/make-a-plan-to-polished-floyd.md` during planning — that
path is outside this repo and tied to the Claude Code session that wrote it,
so treat *this* file as the durable reference going forward, not that one.

## Architecture

Cargo workspace, root `Cargo.toml` `members = ["citybehavex-py", "citybehavex-core", "citybehavex-web"]`:

- **`citybehavex-core/`** — plain Rust lib (no PyO3), extracted out of
  `citybehavex-py`. Holds H3 batch conversion (`h3_batch.rs`),
  contraction-hierarchy road routing (`roads.rs`, `fast_paths` crate), and
  co-presence/graph-metrics for network validation (`network_graph.rs`).
  `citybehavex-py`'s PyO3 bindings are now thin wrappers around this crate —
  confirmed zero behavior change via the full Python test suite after the
  extraction.
- **`citybehavex-web/`** — the new axum binary. Depends on
  `citybehavex-core`, `fkmob-core` (path dep to `/home/gustavo/fkmob/fkmob-core`,
  a plain Rust lib — same pattern `citybehavex-py` already uses), `polars`
  (Rust) for dataframe pipelines, `duckdb` (bundled) for parquet metadata,
  `axum`/`tokio`/`tower-http`. Binds `CBX_WEB_RS_PORT` (default 8001) so it
  can run alongside the Python server (default 8000) during development.
- **Key architectural finding**: most of the "Rust-backed" numeric functions
  `citybehavex/reports/comparison.py` imports from the `fkmob` Python
  package are thin wrappers around real kernels living in `fkmob-core`
  itself (Wasserstein, activity transition counts, motif
  discovery/canonicalization, visitation-law distances, waiting times,
  trajectory CPC, STVD-EMD's exact seeded sliced-Wasserstein) — so
  `citybehavex-web` calls those kernels **directly**, with no PyO3/Python in
  the loop at all. A few (`jensen_shannon_divergence`,
  `time_bin_matrix_jensen_shannon_divergence`, `bin_visitation_law_data`,
  `fit_visitation_law`) are pure Python/numpy with no Rust kernel and were
  reimplemented directly from the documented formulas.

## What's done

### Phases 0–4 — infrastructure (commit `9d1ad13`)

- Axum server: CORS (matches `main.py`'s `localhost`/`127.0.0.1` regex +
  credentials), gzip, SPA static serving with the *exact* status-code
  semantics of `main.py`'s custom 404 handler (verified live).
- `citybehavex-core` extraction (see above).
- Full config layer: every `citybehavex/config/**` Pydantic model ported to
  serde structs with `deny_unknown_fields` + hand-written validators.
  Verified: all 8 real config files in `configs/*.yaml` parse and validate
  correctly (`citybehavex-web/src/settings/`).
- `GET/PATCH /api/experiments`, `POST .../archive`, `DELETE .../runs/{id}` —
  **byte-for-byte identical** to Python on all 5 real experiments in the
  repo, including DuckDB-derived per-run summaries
  (`citybehavex-web/src/experiments.rs`, `datasource.rs`).
- On-disk JSON cache with async in-flight request coalescing
  (`citybehavex-web/src/cache.rs`), mirroring `web/backend/app/cache.py`'s
  `Future`-sharing design via `tokio::sync::OnceCell`.

**Two real fidelity bugs found and fixed** during this phase (both by
testing against the live Python backend rather than trusting the source
reading alone):
1. Python's SPA-fallback exception handler collapses **every** `/api` 404 to
   a generic `{"detail":"Not Found"}` body app-wide (not just unmatched
   routes) whenever `web/frontend/dist` exists — replicated via response
   middleware in `main.rs`.
2. Writing an unquoted `start_date: 2026-01-01` to YAML gets misparsed back
   as a `datetime.date` by PyYAML instead of a string, breaking Pydantic
   validation on the next read. Fixed with a targeted quoting pass in
   `experiments.rs` (`yaml_scalar_would_lose_string_type`).

### Phase 5 — comparison compute engine (commit `b2e6045`)

Everything `web/backend/app/payload/legacy.py` reuses from
`citybehavex/reports/comparison.py` (per `reports_bridge.py`'s import
surface — not the CLI HTML-report entry points, which the web backend never
calls) is ported to `citybehavex-web/src/comparison/`:

| Module | Mirrors | Notes |
|---|---|---|
| `h3.rs` | `_h3_cells`, `_location_resolution` | via `citybehavex-core::h3_batch` |
| `panel.rs` | `_looks_like_panel_observations`, `_adapt_evaluation_dataframe`, `_collapse_to_stays` | verified vs. real `gparis_visitation_df.parquet` |
| `trajectory.rs` | `load_trajectory` | |
| `metrics.rs` | `wasserstein_distance`, `jensen_shannon_divergence`, `_common_part_of_commuters`, `waiting_times_minutes` | calls `fkmob_core::measures::evaluation::wasserstein` directly |
| `transport.rs` | `_synthetic_transport_leg_records` (lazy/streaming), `_observed_transport_leg_records`, `_transport_spatial_summary` | see bug #3 below |
| `mobility_laws.rs` | `_mobility_law_visits`, `_daily_location_lognormal_dataset`, `_distance_frequency_dataset` (+ fkmob's home-inference/binning/OLS-fit pipeline) | verified vs. real data, exact match to 6+ decimals |
| `stvd.rs` | `_stvd_hourly_histogram`, `_diff_stvd_layers`, `_compute_stvd_layers` | see bug #4 below |
| `micro_activity.rs` | `_micro_activity_daily_usage_data` | verified vs. 239K real rows |
| `visits.rs` | `_visits_for_comparison`, `_prepare_activity_visits`, `_motif_visits`, purpose heuristics | |
| `activity.rs` | `activity_transition_matrix`, `daily_activity_distribution`, `discover_daily_motifs_from_agents` | calls `fkmob-core`'s activity/motif kernels directly; verified vs. 10,248 real user-days, exact motif IDs |

**57 unit tests + 8 real-data cross-checks against the live Python
backend, all passing.** Two more bugs found:

3. **Python bug, not replicated (by user decision)**: `_haversine_km_expr`
   clamps via `pl.min_horizontal(a.sqrt(), lit(1.0))`, but `min_horizontal`
   silently skips nulls instead of propagating them. Since `a` is null for
   every transport leg's first waypoint (no predecessor), this adds a
   spurious `~20015 km` "jump" to *every* leg's `mean_jump_km` in the
   Transport Spatial chart. Confirmed root cause directly against the
   installed Polars (`min_horizontal(None, 1.0) == 1.0`). This port computes
   the physically-correct value instead — see the extensive comment on
   `haversine_km_expr` in `util.rs` and the cross-check test in
   `transport.rs`. **The parity harness (Phase 11, not yet built) must
   treat this field as a known exception, not a regression.**
4. **This port's own bug, fixed to match Python**: STVD peak-hour selection
   initially used Rust's `Iterator::max_by_key` (keeps the *last* maximal
   element on a tie), but Python's `max()` keeps the *first*. Fixed by
   reversing iteration order before `max_by_key` in `stvd.rs`.

**Deferred (needs a decision)**: `_truncated_powerlaw_dataset` (fkmob's
`fit_values_to_truncated_powerlaw`) is a *bounded* nonlinear least-squares
fit via scipy's Trust-Region-Reflective solver — no drop-in Rust
equivalent. Recommended approach: the `levenberg-marquardt` crate (built on
`nalgebra`) with a sigmoid reparameterization to handle the box constraints.
Stubbed in `mobility_laws.rs::truncated_powerlaw_dataset` with a clear
not-yet-implemented error. This blocks 2 of the 3 mobility-law curve
families (jump-length and radius-of-gyration truncated-power-law curves);
the third family (distance-frequency) and the log-normal daily-locations
curve are both done.

### Phase 6 — payload assembly (mostly complete, commits `9528f70` + current worktree)

Read `web/backend/app/payload/{context,store,sections}.py` in full — the
`ComparisonContext`/`ArtifactStore`/section-dispatch structure — and about a
third of `payload/legacy.py` (1857 lines, the actual payload-building
engine that turns `comparison.py`'s numeric output into the exact JSON
shapes `web/frontend/src/api.ts` expects).

Ported so far (`citybehavex-web/src/comparison/`):
- `filters.rs` — day/weekday-weekend/time-of-day/special-day filter
  metadata and application (`web/backend/app/filters.py`). Verified
  directly against Python.
- `ecdf.rs` — the empirical-CDF point computation Python gets from
  `skmob_vis._core.compute_ecdf`. Reimplemented directly (~30 lines) rather
  than taking `skmob-vis` as a Cargo dependency — its `[lib] crate-type =
  ["cdylib", "rlib"]` forces a PyO3 cdylib build (needing `-lpython`) on
  any consumer of the rlib output, which fails outside a Python-embedding
  context (see the long comment in `citybehavex-web/Cargo.toml`). Verified
  byte-for-byte against the actual Python-exposed function.
- `metric_row.rs` — the common `{filter_key, filter_label, metric_name,
  name, value, unit?}` row shape used by every metric list.
- Native progressive section wiring now exists for:
  - `metrics` — Wasserstein metric rows over jump-lengths, visits-per-user,
    RoG, dwell, and trip duration.
  - `distributions` — ECDF payload blocks for jump-lengths, visits-per-user,
    RoG, dwell, and trip duration, including observed-side adaptation.
  - `transport-spatial` — summary, mode-share bars, and jump ECDF using the
    already-ported transport leg extraction. This keeps the documented Rust
    correction for Python's `mean_jump_km` null-handling bug.
  - `micro-activity` — synthetic activity sidecar loading, day/special-day
    filtering, and `micro_activity_daily_usage_data` blocks.
  - `activity` — visit preparation, purpose-share bars, transition matrix
    difference/raw blocks, and daily activity profile difference/raw blocks.
  - `motifs` — daily motif distribution/literature-basis mapping, including
    the metrics-section "Daily motifs" JSD side effect.
  - `mobility-laws` — native law-block rendering for travel distance, radius
    of gyration, daily locations, and distance-frequency. The truncated
    power-law fit is now Rust-native via a deterministic bounded coarse-to-fine
    fit, not scipy/TRF.
  - `stvd` — native STVD GeoJSON layers using the already-ported
    `stvd.rs` histogram/diff primitives.
  - `social-network` — loads and validates the simulation social sidecar.
  - `time-use` — synthetic activity-segment aggregation plus observed
    CSV/Parquet MTUS weighted means are native. Configs that still point at
    `.dta` resolve a same-stem `.parquet` or `.csv` conversion before warning.
  - `profiles` — native mobility-profile metrics and deterministic
    Routiner/Regular/Scouter labeling now back `/charts/profiles`.

### Phases 7–9 — standalone routes (native coverage in current worktree)

- **Network validation**: `/network-validation` now returns a real
  `synthetic_vs_random` block from the simulation social-network sidecar,
  using `citybehavex-core::network_graph::compute_graph_metrics` for degree,
  clustering, and topological-overlap distributions. For graphs above 5000
  nodes the random baseline is skipped rather than doing an O(n²) generation
  in-request. Observed co-presence validation is still the remaining parity
  gap.
- **Home/work density maps**: `/home-work` now builds synthetic and observed
  HOME/WORK panels with DuckDB table reduction plus Rust `h3o` bucketing and
  GeoJSON polygon output. This intentionally avoids DuckDB's community H3
  extension.
- **Timeline**: `/timeline/meta`, `/timeline/legs`, `/timeline/agents/{uid}`,
  `/crp`, and `/social` now return native data instead of placeholders.
  `/timeline/legs` uses cached derived `timeline_legs` and optional
  `timeline_moving` parquet indexes, matching the Python large-run browsing
  strategy while adding road waypoints and profile character fields when the
  sidecars are available.

### Experiment loading performance (in progress)

The Experiments page (`GET /api/experiments?with_summary=true`) was slow
because it loaded configs sequentially and then opened DuckDB once per run
summary, also sequentially. The Rust backend now does the safe independent
parts in parallel:

- `list_experiments()` loads `configs/*.yaml` with `rayon` and sorts the final
  experiment list back by id for stable output.
- `Experiment::to_json(true)` and the list route build per-run summaries in
  parallel.
- `datasource::cached_run_summary` wraps `run_summary()` in an in-process
  bounded LRU cache keyed by `(path, mtime, file length)`, caching both
  successes and errors so repeated page loads do not repeatedly pay DuckDB
  schema/query cost.
- `routes/experiments.rs` logs `elapsed_ms`, experiment count, run count, and
  `with_summary` so cold vs. warm page-load behavior is visible in server logs.

The frontend still calls `fetchExperiments(true)` unchanged. If this is still
too slow on a larger run catalog, the next UI-side fallback is to render
`with_summary=false` immediately and fetch summaries lazily for the opened
experiment.

**Resolved along the way**: the STVD-EMD metric's coordinate reprojection
(previously flagged as needing investigation) is just standard EPSG:4326 →
EPSG:3857 (Web Mercator) — confirmed from `legacy.py`'s own
`Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)`. A
closed-form formula suffices; no `proj` crate dependency needed.

## What's left

- **Observed network validation parity**: the Rust endpoint now covers
  synthetic social-network validation, but not the observed daily co-presence
  graph path or `synthetic_vs_observed` metric comparison.
- **MTUS source conversion**: the Rust backend does not read Stata `.dta`
  directly at request time. Run
  `python scripts/convert_mtus_time_use.py data/mtus/MTUS_haf.dta` once, or
  provide a same-stem CSV/Parquet asset; Rust will use that converted table.
- **HTTP parity harness expansion**: `scripts/compare_web_backends.py` exists,
  but should be extended to cover all chart sections, home/work filters,
  timeline endpoints, and metrics export. Continue to whitelist only
  `transport_spatial.summary.*.mean_jump_km`.
- **Performance validation**: benchmark `/charts/*`, `/home-work`,
  `/timeline/legs`, and `/network-validation` cold/warm on both servers.

## How to build/test

```
# Build the new crates (no PYO3_PYTHON needed, unlike citybehavex-py):
cargo build -p citybehavex-core -p citybehavex-web

# Run all fast unit tests:
cargo test -p citybehavex-web --bin citybehavex-web

# Run the real-data cross-check tests (need this repo's data/ tree):
cargo test -p citybehavex-web --bin citybehavex-web -- --ignored

# Run the server (binds CBX_WEB_RS_PORT, default 8001):
cargo run -p citybehavex-web
# Python backend, for side-by-side comparison, still on 8000 as always:
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --port 8000
```

After the Phase 1 extraction, rebuilding `citybehavex-py` still needs the
`PYO3_PYTHON` + `pyo3/extension-module` incantation documented in this
repo's dev-environment notes — that's unchanged, `citybehavex-core` doesn't
need it since it has no PyO3 dependency.

## Git history

- `9d1ad13` — Phases 0–4 (infrastructure).
- `b2e6045` — Phase 5 (comparison compute engine).
- `9528f70` — Phase 6 start (filters/ecdf/metric-row).

Each commit stages only the Rust-rewrite files for that slice — the
repo has had other unrelated work in flight concurrently (a Python-side
`comparison.py`/`cache.py`/`legacy.py` diff already present before this
migration started, and separately, timeline/frontend work) which these
commits deliberately don't touch.
