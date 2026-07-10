# CityBehavEx web app

Interactive comparison UI for CityBehavEx runs. A FastAPI
backend turns a simulation's parquet outputs into JSON plot data; a React + Vite
frontend renders it (ECharts + Leaflet), styled per [`DESIGN.md`](./DESIGN.md).

```
web/
├── backend/app/        FastAPI service
│   ├── main.py         create_app() — CORS (dev) + optional static SPA mount
│   ├── api/            /api routers: experiments, charts
│   ├── experiments.py  discover experiments from configs/*.yaml, glob runs
│   ├── datasource.py   DuckDB: parquet row/user/date metadata
│   ├── payload.py      reuse citybehavex.reports.comparison -> raw JSON
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

**Backend** (port 8000) — run the venv's uvicorn directly so `uv` does not try to
rebuild the Rust extension:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --reload --port 8000
```

**Frontend** (port 5173):

```bash
cd web/frontend
npm install
npm run dev
```

Open http://localhost:5173.

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
- `GET /api/experiments/{id}/charts[?run=<id>&refresh=true]` — the full
  comparison payload (metrics, ECDFs, mobility laws, activity, profiles, motifs,
  STVD GeoJSON). `run` defaults to the latest; results are cached under
  `data/.web_cache/`.
- `GET /api/experiments/{id}/timeline/meta[?run=<id>]` — run's date range, bbox,
  and data-availability flags for the timeline view.
- `GET /api/experiments/{id}/timeline/legs?since=&until=&min_lat=&min_lng=&max_lat=&max_lng=[&run=&max_agents=2000]` —
  agents active in a time window and map viewport, as origin/destination leg or
  dwell segments (client interpolates positions between them). Time window is
  capped at 6h of sim time per request; `max_agents` at 5000. Backed by a derived,
  time-sorted "legs index" parquet cached under `data/.web_cache/timeline_legs/`
  (built once per run, from a `LAG()` window function over the raw trajectory
  table — expensive for the largest cities, hence cached).
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
