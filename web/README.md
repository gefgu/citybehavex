# CityBehavEx web app

Interactive replacement for the standalone comparison `.html` report. A FastAPI
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
        ├── pages/      Home, Experiments, Charts
        ├── charts/     ECharts option builders + theme
        └── components/ Layout, StvdMap (react-leaflet)
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

## Endpoints

- `GET /api/experiments[?with_summary=true]` — experiments from `configs/*.yaml`,
  each with its timestamped runs (and DuckDB row/user/date metadata).
- `GET /api/experiments/{id}` — one experiment (always with run summaries).
- `GET /api/experiments/{id}/charts[?run=<id>&refresh=true]` — the full
  comparison payload (metrics, ECDFs, mobility laws, activity, profiles, motifs,
  STVD GeoJSON). `run` defaults to the latest; results are cached under
  `data/.web_cache/`.

## Production (single origin)

`npm run build` emits `web/frontend/dist`. When that directory exists, the
backend serves it as static files with an SPA fallback, so the app runs from the
backend origin alone (no proxy/CORS needed).

## Notes

- The scientific metrics reuse `citybehavex.reports.comparison` unchanged, so the
  numbers match the standalone report exactly. DuckDB is used only for cheap
  parquet metadata and column-projected loading.
- First load of a large experiment (e.g. Shanghai's ~10.5M-row observed table)
  builds the whole payload and can take a while; it is then served from cache.
