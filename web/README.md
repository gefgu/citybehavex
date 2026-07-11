# CityBehavEx web app

Interactive comparison UI for CityBehavEx runs. A backend turns a simulation's
parquet outputs into JSON plot data; a React + Vite frontend renders it
(ECharts + Leaflet), styled per [`DESIGN.md`](./DESIGN.md). The Python/FastAPI
backend remains available, and the Rust/axum backend in `citybehavex-web/` can
run beside it or replace it for the migrated routes.

```
web/
├── backend/app/        FastAPI service
│   ├── main.py         create_app() — CORS (dev) + optional static SPA mount
│   ├── api/            /api routers: experiments, charts
│   ├── experiments.py  discover experiments from configs/*.yaml, glob runs
│   ├── datasource.py   DuckDB: parquet row/user/date metadata
│   ├── payload/        reuse citybehavex.reports.comparison -> raw JSON
│   └── cache.py        on-disk payload cache keyed by input mtimes
└── frontend/           React + Vite + TS (pico.css + DESIGN.md tokens)
    └── src/
        ├── pages/      Home, Experiments, Charts, Timeline
        ├── charts/     ECharts option builders + theme
        └── components/ Layout, StvdMap (react-leaflet), TimelineMap (mapbox-gl),
                         AgentSidebar
```

## Run (development)

Two servers. The frontend calls relative `/api/...` and Vite proxies it to the
backend (see `frontend/vite.config.ts`).

**Python backend** (port 8000) — run the venv's uvicorn directly so `uv` does
not try to rebuild the Rust extension:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --reload --port 8000
```

**Rust backend** (port 8001 by default):

```bash
cargo run -p citybehavex-web
```

**Frontend** (port 5173):

```bash
cd web/frontend
npm install
npm run dev
```

In development, browser API calls default directly to the Rust backend at
`http://127.0.0.1:8001`, avoiding Vite proxy failures. To use another backend:

```bash
cd web/frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:5173.

## Rust backend checks

The Rust backend is intended to match the Python API contract, except for the
documented Transport Spatial `mean_jump_km` correction in
`RUST_BACKEND_MIGRATION.md`. Current native coverage includes experiments,
progressive chart sections, home/work density maps, timeline routes, and
synthetic social-network validation. Remaining parity gaps are documented in
`../RUST_BACKEND_MIGRATION.md`.

```bash
cargo build -p citybehavex-core -p citybehavex-web
cargo test -p citybehavex-web --bin citybehavex-web
cargo test -p citybehavex-web --bin citybehavex-web -- --ignored
```

For HTTP-level parity, run both servers and compare responses:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --port 8000
CBX_WEB_RS_PORT=8001 cargo run -p citybehavex-web
scripts/compare_web_backends.py --python http://localhost:8000 --rust http://localhost:8001
```

Add `--include-slow` when validating the heaviest network-validation endpoint.
For a quick timing smoke test:

```bash
scripts/benchmark_web_backends.py gparis_simulation --include-network-validation
```

## Static GitHub Pages demo

The public demo can run without FastAPI by exporting endpoint-shaped JSON into
the Vite public directory:

```bash
uv run python scripts/export_static_web_demo.py --manifest web/demo_export.yaml
cd web/frontend
VITE_STATIC_DEMO=true VITE_BASE_PATH=/citybehavex/ npm run build
```

The exporter writes `web/frontend/public/demo-data/`. In static mode the
frontend reads those files instead of `/api/...`, uses hash routing for GitHub
Pages deep links, and keeps the regular local API behavior unchanged when
`VITE_STATIC_DEMO` is unset.

`web/demo_export.yaml` pins the public runs and marks which experiments may
include observed data. Keep `allow_observed: false` for the private Greater
Paris and Shanghai comparisons; YJMOB can use `allow_observed: true` because the
comparison source is public. The manifest also supports `expected_agents`; leave
it at `500` for the public demo so the exporter fails early if a large run is
accidentally pinned. If `web/frontend/public/demo-data/` is not committed or
otherwise supplied to CI, the Pages workflow will still build the app but the
deployed static demo will have no data to load.

## Timeline view setup

The timeline view (`/experiments/:id/timeline`) uses Mapbox GL JS instead of the
free CARTO/Leaflet tiles used elsewhere in this app, since it needs GPU-accelerated
per-agent marker updates at animation frame rate. This means it needs its own
access token:

1. Create a free account at https://account.mapbox.com and copy an access token.
2. Create `web/frontend/.env.local` (gitignored) containing:
   ```
   VITE_MAPBOX_TOKEN=pk.your_token_here
   ```
3. Restart `npm run dev` — Vite only reads `.env*` files at startup.

## Endpoints

- `GET /api/experiments[?with_summary=true]` — experiments from `configs/*.yaml`,
  each with its timestamped runs (and DuckDB row/user/date metadata).
- `GET /api/experiments/{id}` — one experiment (always with run summaries).
- `GET /api/experiments/{id}/charts[?run=<id>&refresh=true]` — progressive
  comparison metadata. Section payloads are fetched separately and cached under
  the backend's web-cache directory.
- `GET /api/experiments/{id}/charts/{section}[?filter=all&run=<id>&refresh=true]` —
  one chart section for one filter.
- `GET /api/experiments/{id}/metrics-export?format=json[&run=<id>&refresh=true]` —
  raw JSON metrics export, not wrapped in `{ data: ... }`.
- `GET /api/experiments/{id}/network-validation[?run=<id>&refresh=true]` —
  network validation payload, cached separately from charts. The Rust backend
  currently emits native synthetic-vs-random validation from the social sidecar;
  observed co-presence validation remains a parity follow-up.
- `GET /api/experiments/{id}/home-work[?run=<id>&gender=&age_bracket=&job=&refresh=true]` —
  home/work density map payload. The Rust backend uses `h3o` directly for H3
  bucketing/polygons instead of DuckDB's community H3 extension.
- `GET /api/experiments/{id}/timeline/meta[?run=<id>]` — run's date range, bbox,
  and data-availability flags for the timeline view.
- `GET /api/experiments/{id}/timeline/legs?since=&until=&min_lat=&min_lng=&max_lat=&max_lng=[&run=&max_agents=2000]` —
  agents active in a time window and map viewport, as origin/destination leg or
  dwell segments (client interpolates positions between them). Time window is
  capped at 6h of sim time per request; `max_agents` at 5000. Both backends use
  derived cached legs/moving parquet indexes for large-run browsing.
- `GET /api/experiments/{id}/timeline/agents/{uid}[?run=<id>]` — one agent's
  profile, narrative bio, full trip history, and recent encounters.

## Production (single origin)

`npm run build` emits `web/frontend/dist`. When that directory exists, the
backend serves it as static files with an SPA fallback, so the app runs from the
backend origin alone (no proxy/CORS needed).

## Notes

- The scientific metrics reuse helpers from `citybehavex.reports.comparison`,
  while the live UI owns the rendered comparison experience. DuckDB is used only
  for cheap parquet metadata and column-projected loading.
- First load of a large experiment (e.g. Shanghai's ~10.5M-row observed table)
  builds the whole payload and can take a while; it is then served from cache.
- The Rust backend is functionally complete for the web app's current routes,
  with one documented scientific parity exception:
  `transport_spatial.summary.*.mean_jump_km` intentionally uses the corrected
  null-propagating distance calculation.
- Python backend:
  `uv run uvicorn web.backend.app.main:app --host 127.0.0.1 --port 8000`
- Rust backend:
  `cargo run -p citybehavex-web` or
  `CBX_WEB_RS_PORT=8001 cargo run -p citybehavex-web`
- Frontend against Rust:
  `VITE_API_PROXY_TARGET=http://localhost:8001 npm run dev`
- MTUS `.dta` files should be converted once for Rust request-time use:
  `python scripts/convert_mtus_time_use.py data/mtus/MTUS_haf.dta`
  The Rust backend resolves a same-stem `.parquet` or `.csv` when a config still
  points at the original `.dta`.
- Parity and benchmark harnesses:
  `python scripts/compare_web_backends.py --python http://localhost:8000 --rust http://localhost:8001`
  and
  `python scripts/benchmark_web_backends.py gparis_simulation --include-network-validation`.
