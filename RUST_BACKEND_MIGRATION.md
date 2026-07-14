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

- **Network validation**: `/network-validation` returns
  `synthetic_vs_random`, `observed_vs_random`, and `synthetic_vs_observed`
  blocks, using `citybehavex-core::network_graph::compute_graph_metrics` for
  degree, clustering, and topological-overlap distributions, and
  `build_co_presence_edges` for the observed daily co-presence graph
  (`citybehavex-web/src/comparison/network_validation.rs`). For graphs above
  5000 nodes the random baseline is skipped rather than doing an O(n²)
  generation in-request. See "What's left" for the one known remaining gap
  (synthetic graph edge completeness).
- **Home/work density maps**: `/home-work` now builds synthetic and observed
  HOME/WORK panels with DuckDB table reduction plus Rust `h3o` bucketing and
  GeoJSON polygon output. This intentionally avoids DuckDB's community H3
  extension.
- **Timeline**: `/timeline/meta`, `/timeline/legs`, `/timeline/agents/{uid}`,
  `/crp`, and `/social` now return native data instead of placeholders.
  `/timeline/legs` uses cached derived `timeline_legs` and optional
  `timeline_moving` parquet indexes, matching the Python large-run browsing
  strategy while adding road waypoints and profile character fields when the
  sidecars are available. `/timeline/agents/{uid}`'s `narrative` and
  `encounters` (previously hardcoded `null`/`[]`) are now native too:
  `profile_to_narrative` (`citybehavex-web/src/routes/timeline.rs`) mirrors
  `citybehavex/profiles/agents.py`'s prose template exactly, and
  `query_agent_encounters` mirrors `web/backend/app/timeline_data.py`'s
  contact-profile/contact-narrative/activity-at-stop enrichment. One real fix
  needed getting there: the encounters parquet's `ts` is raw epoch seconds,
  and Python's `to_timestamp(ts)::TIMESTAMP` implicitly localizes it via the
  duckdb session's OS-derived `TimeZone` before comparing against
  `arrival`/`departure` -- an un-localized (UTC) conversion compared against
  the wrong wall-clock time by however many hours the local offset is
  (varying by historical DST for the date in question), silently matching
  the wrong stop. Verified against real Shanghai-500-sample data: 12/12
  encounters now match Python's contact_uid/ts/stop/activity/profile fields
  exactly (using `chrono::Local`'s epoch->local-datetime conversion, since
  DuckDB's own `AT TIME ZONE` needs the `icu` extension, which isn't in the
  bundled build of the duckdb-rs version pinned here and would otherwise
  need a live `INSTALL`/network fetch).

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

- **Synthetic network validation edge completeness**: `synthetic_vs_random`'s
  synthetic graph currently only reads the social-network sidecar's edges
  (`citybehavex-web/src/payload.rs::synthetic_social_graph`); Python's
  `_synthetic_validation_block` additionally unions in edges from the
  encounters sidecar (`_encounter_edges_and_persistence`) when present. This
  was discovered doing a real-data comparison while building observed
  network validation (below) -- Python's synthetic edge count is
  consistently higher than Rust's on real runs with an encounters sidecar,
  and it's also why `edge_persistence` has always been empty in Rust's
  `synthetic_vs_random` metrics (no encounter data was ever threaded in to
  compute it from). Not yet fixed.
- **Performance validation**: benchmark `/charts/*`, `/home-work`,
  `/timeline/legs`, and `/network-validation` cold/warm on both servers.
- **Harness flakiness under `--include-slow`'s concurrent load**: a handful
  of endpoints (`charts/metrics`'s `time_use_comparison`/`metrics.stvd`,
  `charts/activity`'s `daily_activity_difference.limit`) intermittently show
  up as failing in a full harness run, but consistently pass when the exact
  same request is re-issued in isolation immediately after (cache cleared,
  single request, no concurrent load) -- confirmed repeatedly while fixing
  the items below. Root cause not yet identified (candidate: cache-write
  contention or a DuckDB connection-pool limit under the harness's dense,
  rapid request sequence across every section/filter/experiment
  combination); flagging so a future full-harness "failure" isn't
  automatically read as a regression without a quick isolated re-check
  first.
- **Fixed root cause of a systemic per-user grouping bug (affected `jump_lengths`
  ECDF, `activity` transition matrix, and `mobility-laws`)**: `citybehavex-web/src/comparison/features.rs::jumps_rog`
  and `comparison/activity.rs::activity_transition_matrix` both cast the
  `uid` column to `Int64` for per-user boundary detection. Real observed
  survey data can have composite string uids (confirmed on
  `gparis_simulation`: values like `"10_2980"`); Polars silently nulls
  unparseable values on cast, and the follow-up `.unwrap_or(i64::MIN)`
  collapsed *every* such user into one fake shared ID -- confirmed this
  merged all 504 distinct `gparis` observed users into a single contiguous
  group, creating 503 spurious inter-user "jumps" at the false boundaries
  between what should have been separate users (504 - 1 = 503, exactly
  matching the observed discrepancy: Rust reported 10832 non-zero jumps vs
  Python's 10329). `mobility_laws.rs` already had the correct fix
  (`canonical_user_ids`: cast through directly for already-integer dtypes,
  otherwise factorize by first appearance) -- promoted that helper to
  `comparison::util` (`canonical_user_ids`/`canonical_user_ids_vec`) and
  reused it in the two buggy call sites. Verified against real
  `gparis_simulation` data: the `jump_lengths` ECDF now matches Python on
  all 400 downsampled points (previously diverged starting around point 30);
  `activity`'s `transition_difference.limit` now matches Python almost
  exactly (previously off by more than 4x).
- **Implemented `metrics.cpc` and `metrics.stvd` (previously always empty,
  not merely buggy -- nothing called these kernels at all)**: both
  underlying kernels already existed
  (`comparison::metrics::common_part_of_commuters`,
  `comparison::stvd::stvd_hourly_histogram` +
  `fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl`) but nothing
  wired them into `metrics_section_payload`. Added the wiring, including a
  from-scratch port of `legacy.py::_stvd_emd_distribution` (flattening a
  per-H3-cell hourly histogram into `(x, y, minutes-of-day, volume)` arrays,
  reprojecting each cell's centroid from EPSG:4326 to EPSG:3857 via the
  closed-form Web Mercator formula this doc already confirmed suffices).
  Verified against real `gparis_simulation` data: both `metrics.cpc` (3
  resolutions) and `metrics.stvd` (3 resolutions) now match Python
  **exactly**, full float precision.
- **Known accepted divergence: `profiles` cluster *assignment* on observed
  data**: after the fix below, per-user `regularity`/`diversity`/
  `stationarity`/`entropy` values are verified exact against Python. The
  Routiner/Regular/Scouter *cluster labeling* can still disagree on the
  `gparis` observed dataset specifically (synthetic-side clustering matched
  exactly in the same test). Root cause: Rust's k-means
  (`label_profile_clusters`, seeded from sorted n/6·n/2·5n/6 percentile
  points) isn't the same algorithm as Python's `sklearn.cluster.KMeans(
  random_state=0, n_init=10)` (k-means++ init, 10 restarts, numpy's own
  PRNG) -- bit-exact replication of sklearn's clustering in a from-scratch
  Rust implementation isn't practical, so this is treated the same as this
  codebase's other accepted algorithm-level (not data-level) divergences
  (e.g. the degree-preserving random-graph baseline's differing RNG). Every
  underlying per-user metric is provably correct; only which of the 3 named
  buckets a borderline user in a noisy real dataset lands in can differ.

### Resolved: `profiles` chart section returned 500 on real data

Root-caused and fixed -- two independent bugs, not the algorithmic gap
originally suspected:
1. **The real blocker**: `profile_visits_from_df` cast the `uid` column to
   `Int64`, silently nulling *every* row when `uid` is a composite string
   identifier (confirmed on `gparis_simulation`'s observed data: values like
   `"10_2980"`), because Polars' cast produces null rather than erroring on
   unparseable values. `ProfileVisit`/`ProfileRow`'s `uid` field is now
   `String` (matching Python's `compute_profiles`, which never assumes `uid`
   is numeric -- it's narwhals-generic throughout), fixing the crash for any
   dataset with non-integer user IDs.
2. `profile_visits_from_df` also excluded every row with
   `end_timestamp <= start_timestamp` up front, which Python's
   `compute_profiles` never does for its shared `df` (feeds every raw visit
   into `regularity`/`diversity`/`entropy`/intermittency's location-token
   derivation, unfiltered) -- only `_stationarity` filters
   `duration_minutes > 0`, scoped to its own dwell/span computation. Fixed by
   removing the blanket exclusion and re-scoping the positive-duration check
   to just the stationarity dwell/span accumulation in
   `compute_profiles_rows`.

Verified against real `gparis_simulation` data: `regularity`, `diversity`,
`stationarity`, and `entropy` box-plot quantiles for the synthetic side and
for the `"Routiner"` profile bucket on the observed side match Python
**exactly** (full float precision); see the item above for the one
remaining, accepted divergence (cluster labeling on noisier observed data).

### Resolved: the "cannot start a runtime from within a runtime" panic

Root-caused and fixed. `cache.rs::get_or_build` was `.await`ing its synchronous,
CPU/IO-bound `build` closure directly on the calling tokio worker thread. When
that closure's Polars work hit the new-streaming engine's lazy `scan_parquet`
path, `polars_stream::nodes::io_sources::multi_scan::MultiScan::update_state`
internally calls `polars_io::pl_async::RuntimeManager::block_on`, which itself
tries to enter a runtime -- doing that from a thread that's already driving
the axum handler's own async task trips tokio's "cannot start a runtime from
within a runtime" panic (confirmed via full backtrace:
`polars_stream::async_executor::task_scope` -> `MultiScan::update_state` ->
`RuntimeManager::block_on` -> `tokio::runtime::scheduler::multi_thread::MultiThread::block_on`).
Fixed by changing `get_or_build`'s `build` parameter from an async closure to
a plain synchronous one, run via `tokio::task::spawn_blocking` instead of
awaited inline -- `spawn_blocking` threads aren't considered "inside" the
async scheduler, so Polars' internal `block_on` no longer conflicts. All 5
call sites (`routes/charts.rs`) updated accordingly; verified with 9 repeated
cold-cache (`refresh=true`) requests against the previously-panicking
`transport-spatial`/`activity`/`metrics` sections on real `gparis_simulation`
data, zero panics.

### Resolved since the table above was written

- **Observed network validation** (`observed_vs_random` and
  `synthetic_vs_observed`) is now native: `citybehavex-web/src/comparison/network_validation.rs`
  builds the observed daily co-presence graph via
  `citybehavex-core::network_graph::build_co_presence_edges` (the same kernel
  the synthetic path already used), with `NetworkValidationConfig`'s
  `observed_enabled`/`location_mode`/`location_col`/`h3_resolution`/
  `max_group_size` all wired through `routes/charts.rs::network_validation_route`.
  Verified against real `shanghai_simulation_500sample` data: node/edge counts
  match Python exactly (500 nodes, 3534 edges); Wasserstein/clustering/
  topological-overlap numbers are close but not identical, which is expected
  since the degree-preserving random baseline uses a different RNG than
  Python's (same accepted-divergence category as `synthetic_vs_random`'s
  existing random baseline). `edge_persistence` is `null` for the observed
  side too, for the same reason noted in the item above.
- **MTUS source conversion**: not a Rust gap -- it's an accepted one-time
  pre-step. Run `python scripts/convert_mtus_time_use.py data/mtus/MTUS_haf.dta`
  once (or provide a same-stem CSV/Parquet asset); Rust resolves that
  converted table at request time. See `web/README.md`'s Notes section.
  **Updated**: the script used to aggregate MTUS's 25 raw harmonized activity
  codes down into a simplified 9-bucket rollup (its own invention, not
  Python's schema) so Rust had a small, easy-to-hardcode category list --
  but that meant Rust's time-use comparison was built from a coarser
  granularity than Python's (which reads the `.dta` directly via
  `pandas.read_stata`, seeing all 25 raw codes), so the two could never
  numerically agree even with matching category *names*. The script now
  passes all 25 raw codes through unaggregated (see the "Resolved" entry
  below); regenerate any existing converted sidecar after pulling this
  change.
- **HTTP parity harness expansion**: `scripts/compare_web_backends.py` now
  covers all 11 chart sections, home/work filter combinations, and the full
  timeline route surface (`meta`, `legs`, `agents/{uid}`, `/crp`, `/social`),
  and survives a connection-dropped request (e.g. a panicking handler) as a
  diffable failure instead of crashing the whole run. Its known-exception
  whitelist also now exempts the whole `transport_spatial.jump_ecdf` subtree,
  not just `mean_jump_km` -- running the expanded harness against real
  `gparis_simulation` data surfaced that the same null-unsafe-clamp bug
  documented below corrupts a *subset* of legs in a mode to ~20015 km (not a
  uniform per-leg offset), which reshuffles that mode's sort order/rank and
  shifts both coordinates of every downstream ECDF point, not just the
  directly-corrupted ones -- so the whole subtree is incomparable
  point-for-point wherever the bug bites, not just the corrupted values
  themselves. Same accepted root cause, just a second, structurally
  different place it leaks into the payload.
- **Implemented the 3 missing `metrics.jsd` rows** (`"Activity distribution"`,
  `"Activity transitions"`, `"Daily activity profile"` -- `"Daily motifs"`
  was already there via `build_motifs_block`): wired
  `sections::metrics::metrics_section_payload` to recompute these the same
  way `legacy.py::_activity_group` does, reusing the exact
  `prepared_visits_for_filter` visits already shared with the motifs JSD side
  effect and the existing `activity_transition_matrix`/
  `jensen_shannon_divergence`/`time_bin_matrix_jsd` kernels plus
  `sections::activity`'s alignment helpers (`align_square`, `align_daily`,
  `ordered_union`, `string_column`), promoted from private to `pub(crate)`
  for this reuse. Only computed when both synthetic and observed visits are
  present, matching Python's `if obs_v is not None and not obs_v.is_empty()`
  guard on this specific side effect (looser than the outer section's
  synthetic-only gate). One subtlety caught by real-data verification:
  Python's `_activity_group` calls `daily_activity_distribution(syn_v)` with
  no explicit bin size, which defaults to **10-minute** bins in
  `fastmob`'s Python wrapper. `sections::activity`'s own `daily_tuple`
  (shared by the `activity` chart's display matrix) was hardcoded to
  **60-minute** bins instead -- initially assumed to be a deliberate, coarser
  granularity for chart display, but a follow-up real-data harness run
  (below) showed Python's chart matrix reuses this exact same 10-minute
  tuple too (`_build_activity_block` takes `synth_daily`/`real_daily` as
  already-computed arguments from the same call `_activity_group` makes),
  so 60 minutes was simply wrong everywhere, not an intentional UI
  simplification. Fixed by changing `daily_tuple` itself to bin_size 10 --
  confirmed the frontend doesn't hardcode an assumed bin count
  (`web/frontend/src/charts/builders.ts` computes `n_bins / 24` dynamically).
  Verified against real `gparis_simulation` data across
  `all`/`weekday`/`weekend` filters: all 4 `metrics.jsd` rows match Python
  **exactly**, and `activity`'s `daily_activity_difference` chart matrix
  (previously 24 bins vs Python's 144) now matches too.
- **Fixed 3 more gaps found running the full `--include-slow` harness for
  the first time end-to-end** (surfaced immediately after the two entries
  above landed):
  - **`activity`/`daily_activity_difference` and `transition_difference`
    `"limit"` fields off in the 4th decimal place**: `legacy.py` computes
    `"limit"` from the *already-`.round(3)`-ed* matrix, not the raw one;
    `sections::activity::matrix_limit` took the max of the raw values. Fixed
    by rounding to 3 decimals before taking the max, matching Python's
    apparent behavior exactly (`50.4454680344345` → `50.445`, `Python`'s own
    value).
  - **`mobility-laws`' `travel_distance`/`radius_of_gyration`/
    `daily_locations` fit/reference curves used 100 points, Python uses
    200**: `legacy.py::_curve_x`'s `n: int = 200` default vs
    `sections::mobility_laws::curve_x`'s hardcoded `let n = 100usize`. Fixed
    by matching the default; verified `daily_locations`' curve now matches
    exactly. `radius_of_gyration`/`travel_distance`'s *fit parameters* still
    differ substantially, but that's the pre-existing, explicitly
    out-of-scope, already-documented scipy-TRF-vs-Rust-grid-search
    divergence noted above -- added a scoped exception to
    `compare_web_backends.py` for exactly those two blocks' `fits`/`series`
    fields (not `daily_locations`/`distance_frequency`, which use different,
    non-fitted dataset builders and should still be compared normally).
  - **`time_use` categories were a 9-item invented rollup, not MTUS's 25 raw
    codes**: see the "MTUS source conversion" entry above and the dedicated
    write-up below.
- **`time_use`/`metrics-export` -- three compounding gaps, found and fixed
  together**:
  1. **Wrong category scheme**: `sections::time_use::TIME_USE_CATEGORIES`
     hardcoded a 9-bucket rollup (`sleep`, `personal care`, `household`,
     ...) that doesn't exist anywhere in Python -- Python's real
     `TIME_USE_CATEGORIES` is the 25 raw MTUS harmonized activity codes
     (`sleep`, `eatdrink`, `selfcare`, `paidwork`, `educatn`, ...),
     conveniently identical to citybehavex's own native activity-catalog
     names (`settings::catalog`), since the simulation's taxonomy was
     modeled directly on MTUS. Fixing just the Rust constant wasn't enough,
     though: `scripts/convert_mtus_time_use.py` (the accepted one-time
     `.dta`-to-Parquet pre-step Rust reads instead of parsing Stata itself)
     *itself* aggregated those 25 raw columns down into the same invented
     9-bucket scheme before this fix, so Rust and Python were reading two
     different granularities of the same source data and could never
     numerically agree regardless of category names. Updated the script to
     pass all 25 raw columns through unaggregated instead, and regenerated
     `data/mtus/MTUS_haf.parquet` (git-ignored, regenerable, used by exactly
     one config) -- confirmed via `pandas.read_stata` on the original
     `.dta` that it genuinely has all 25 raw columns; Python's `_read_time_use_table`
     reads that `.dta` directly for the real MTUS granularity, which the
     regenerated Parquet now matches.
  2. **`percent_difference` wasn't rounded to 6 decimals** like every other
     field in the same row (`legacy.py` does `round(diff / obs * 100.0, 6)`);
     fixed in `sections::time_use`'s row builder. After (1) and (2),
     `/charts/time-use`'s `time_use_comparison` matches Python **exactly**,
     field-for-field, category-for-category.
  3. **`metrics-export` only ever computed a single filter**
     (`chart_section_payload(&ctx, "metrics", "all")`), but Python's
     `/metrics-export` route calls `_build_comparison_payload(filter_keys=None,
     ...)`, which internally loops over filters and concatenates -- and,
     confirmed against real data, over two *different* filter sets
     depending on metric type: `wasserstein`/`stvd` span every
     *distribution* filter (`all`/`weekday`/`weekend`/time-of-day buckets/
     special days -- 7 for `gparis_simulation`), while `jsd`/`cpc`/
     `time_use`/`time_use_comparison` only span the 3 *regular* filters
     (`all`/`weekday`/`weekend` -- confirmed via `/charts/metrics?filter=morning`
     on real data: Python already returns populated `wasserstein`/`stvd` for
     a time-of-day filter, but empty `jsd`/`cpc` and a `null`
     `time_use_comparison`). Added `payload::metrics_export_artifact`,
     looping `chart_section_payload` over `available_filters` (regular) for
     the jsd/cpc/time_use/time_use_comparison merge and separately over
     `distribution_filters` (all 7) for wasserstein/stvd. Also matched
     Python's *order* for the merged `jsd` array: `legacy.py` runs the
     "activity" JSD side effect (Activity distribution/transitions/Daily
     activity profile) to completion across every regular filter, *then*
     runs the separate "Daily motifs" side effect across every regular
     filter again -- so the merged order is metric-group-major, not
     filter-major; split by `metric_name` rather than assuming a fixed
     position, since either group can be legitimately empty for a given
     filter. Also fixed `time_use_metric_rows`'s row shape to match every
     other metric list's `{metric_name, name, ..., unit}` shape (it was
     using a bespoke `"metric"` key) and its unit string (`legacy.py` uses
     the literal `"pct points"`, not `"percentage points"`). Verified
     against real `gparis_simulation` data: `/metrics-export`'s full
     response (`metrics.wasserstein` 35 rows, `metrics.stvd` 21 rows,
     `metrics.jsd` 12 rows in the correct grouped order, `metrics.time_use`
     3 rows, `time_use_table` 75 rows) now matches Python **exactly**.
- **`stvd` chart's map layer**: `sections::stvd::stvd_section_payload`
  hardcoded its own `COLORS` palette and `threshold = 25.0`, both invented
  rather than copied from `legacy.py`'s `STVD_COLORS`/
  `STVD_VOLUME_THRESHOLD = 3.0` -- fixed to match exactly. The GeoJSON
  `features` array order also differs from Python's, but verified (aligning
  both sides by `properties.area`, the H3 cell hex ID) that it's the exact
  same 840 cells with identical geometry and properties on real
  `gparis_simulation` data, just in a different array position -- neither
  side sorts, each reflects its own internal grouping/iteration order,
  which GeoJSON rendering doesn't depend on. Added a targeted sort-before-
  diff step to `compare_web_backends.py` for this one list (keyed by
  `properties.area`) rather than a blanket exception, so a real future
  content mismatch in this field would still be caught.
- **`/home-work` -- three previously-untested gaps** (this route had never
  been exercised by the harness with real data before a full
  `--include-slow` run finally reached it):
  1. **`filter_options.jobs` was an entirely invented list**:
     `home_work.rs::JOBS` hardcoded 10 full ISCO-08 major-group labels
     ("Managers", "Professionals", ...) that don't correspond to anything
     the simulation actually writes. The real values (and Python's real
     `JOBS` list, `citybehavex.profiles.ILOSTAT_JOBS`) are 9 short lowercase
     codes ("manager", "professional", ...) -- the literal `job` field on
     every synthetic agent profile. This meant `/home-work?job=...` almost
     certainly never matched a single synthetic agent. Fixed by copying
     `ILOSTAT_JOBS` exactly.
  2. **`filter_options.age_brackets[].label` used a hyphen, Python uses an
     en dash** (`"16-24"` vs `"16–24"`, U+2013) -- cosmetic only, the actual
     filter keys/bounds already matched; fixed the display string.
  3. **The real/observed density map's modal-point algorithm didn't match**:
     `home_work.rs::modal_points` picked each user's single most-visited
     point by rounding raw lat/lng to 6 decimals (~0.1m) and grouping on
     that -- far tighter than GPS noise on repeat visits to "the same
     place," so it was splitting one real location into several
     near-duplicate candidate groups and could pick a different (or just
     differently-tied) winner than Python.
     `web/backend/app/home_work_data.py::_agent_density` instead groups by
     H3 resolution-12 cell (`_FINE_RESOLUTION`, ~10m) before picking the
     modal group. Rewrote `modal_points` to do the same fine-cell grouping
     in Rust via `h3o` (DuckDB now only does the raw filter query; the
     group-by-fine-cell-then-pick-the-mode step happens in Rust code,
     mirroring the SQL's `cnt DESC, fine_cell` tiebreak with
     `max_by_key((cnt, Reverse(cell)))`). Verified against real
     `gparis_simulation` data: the synthetic-side density map now matches
     Python **exactly** (386/386 H3 cells, identical `agent_count` per
     cell). The observed/real side is very close but not bit-exact (260 of
     261 cells match exactly, agent counts identical on every shared cell);
     the one remaining mismatch is one user landing in a different display
     cell, most likely because Python's `any_value(lat)`/`any_value(lng)`
     (picking an arbitrary representative point within a tied fine cell) is
     itself not deterministic in DuckDB -- there's no "more correct" answer
     to replicate bit-for-bit here, just two different arbitrary choices.
     Not chased further given the scale of improvement (from
     systematically wrong on nearly every cell to one non-deterministic
     edge case).

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
