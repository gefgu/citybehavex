# CityBehavEx web app

Interactive comparison UI for CityBehavEx runs. The FastAPI backend turns a
simulation's parquet outputs into JSON plot data (reusing
`citybehavex.reports.comparison`/`network_validation`); the React + Vite
frontend renders it with ECharts, Leaflet, and Mapbox GL.

```text
web/
├── backend/            FastAPI app (`app/`)
│   └── app/             API routers, payload builders, and cache/executor infra
└── frontend/           React + Vite + TS
    └── src/
        ├── pages/      Home, Experiments, Charts, Timeline
        ├── charts/     ECharts option builders + theme
        └── components/ Layout, StvdMap, timeline and summary components
```

## Run

Start the backend from the repository root:

```bash
uv run uvicorn app.main:app --app-dir web/backend --port 8000
```

Start the frontend:

```bash
cd web/frontend
npm install
npm run dev
```

In development, browser API calls default directly to the backend at
`http://127.0.0.1:8000`. To use another backend URL:

```bash
cd web/frontend
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:5173.

## Backend Checks

```bash
uv run pytest tests/test_reports.py
```

## Timeline View

The timeline view (`/experiments/:id/timeline`) uses Mapbox GL JS for
GPU-accelerated per-agent marker updates. To enable it, create
`web/frontend/.env.local`:

```bash
VITE_MAPBOX_TOKEN=pk.your_token_here
```

Restart `npm run dev` after creating or editing this file.

## Endpoints

- `GET /api/experiments[?with_summary=true]`
- `GET /api/experiments/{id}`
- `PATCH /api/experiments/{id}`
- `GET /api/experiments/{id}/charts[?run=<id>&refresh=true]`
- `GET /api/experiments/{id}/charts/{section}[?filter=all&run=<id>&refresh=true]`
- `GET /api/experiments/{id}/metrics-export?format=json[&run=<id>&refresh=true]`
- `GET /api/experiments/{id}/network-validation[?run=<id>&refresh=true]`
- `GET /api/experiments/{id}/home-work[?run=<id>&gender=&age_bracket=&job=&refresh=true]`
- `GET /api/experiments/{id}/timeline/meta[?run=<id>]`
- `GET /api/experiments/{id}/timeline/legs?since=&until=&min_lat=&min_lng=&max_lat=&max_lng=[&run=&max_agents=2000]`
- `GET /api/experiments/{id}/timeline/agents/{uid}[?run=<id>]`
- `GET /api/experiments/{id}/timeline/agents/{uid}/crp[?run=<id>]`
- `GET /api/experiments/{id}/timeline/agents/{uid}/social[?run=<id>]`

## Static Demo

Export endpoint-shaped JSON into the Vite public directory:

```bash
uv run python scripts/export_static_web_demo.py --manifest web/demo_export.yaml
cd web/frontend
VITE_STATIC_DEMO=true VITE_BASE_PATH=/citybehavex/ npm run build
```

The exporter writes `web/frontend/public/demo-data/`. In static mode the
frontend reads those files instead of `/api/...`, uses hash routing for GitHub
Pages deep links, and keeps the regular local API behavior unchanged when
`VITE_STATIC_DEMO` is unset.

### GitHub Pages release data

The Pages workflow downloads a versioned GitHub Release asset before building.
`web/demo_release.json` pins its release tag, asset name, and SHA-256 checksum.
Generated data stays out of Git. The initial null pin deliberately stops deployment
until a real archive is published; a frontend-only artifact would return 404 for
the experiments list.

On the machine with the runs named in `web/demo_export.yaml`, use the existing
export environment, then package the result from the repository root:

```bash
uv run python scripts/export_static_web_demo.py --manifest web/demo_export.yaml
python3 scripts/static_demo_archive.py pack --release-tag demo-data-2026-10-07 \
  --archive /tmp/demo-data.tar.gz
gh release create demo-data-2026-10-07 /tmp/demo-data.tar.gz \
  --repo gefgu/citybehavex --title 'Static demo data 2026-10-07' \
  --notes 'Exported using web/demo_export.yaml.'
```

Use a new dedicated release tag for each dataset update. The pack command validates
the export, creates an archive containing `demo-data/`, and writes the real checksum
to `web/demo_release.json`. Preserve the manifest's sampling and observed-data
restrictions. Publish the archive before committing the generated pin. Commit the
pin to `main` to trigger deployment, or manually run **Static Web Demo Pages** after
merging it. Release publication requires GitHub CLI authentication with write access;
Actions downloads using its existing read-only contents permission.

The archive utility needs Python 3.11.8+ (or 3.12+) and `gh` for downloads; it does
not import the exporter or require simulation dependencies. To reproduce the Pages
data preparation on a clean checkout:

```bash
GH_TOKEN=your_read_token python3 scripts/static_demo_archive.py fetch \
  --repository gefgu/citybehavex
cd web/frontend
npm ci
VITE_STATIC_DEMO=true VITE_BASE_PATH=/citybehavex/ npm run build
cd ../..
python3 scripts/static_demo_archive.py validate web/frontend/dist/demo-data
```

Fetch refuses to overwrite an existing `public/demo-data/`; move or remove an old
export first. Missing releases, checksum mismatches, malformed JSON, and missing
required payloads stop deployment before artifact upload. Both the extracted export
and the built artifact are validated. Chart sections omitted by the exporter because
their inputs are unavailable remain optional, but every run must have chart sections.

After deployment, check `/citybehavex/demo-data/experiments.json` returns JSON with
a nonempty `data` array, open `#/experiments`, and check charts and timeline views.
To roll back the dataset, restore the previous `web/demo_release.json` and redeploy;
retain older release assets so their pins stay reproducible.

## Production

`npm run build` emits `web/frontend/dist`. When that directory exists, the
backend serves it as static files with an SPA fallback, so the app runs from the
backend origin alone.

## Note on the previous Rust/axum backend

An earlier iteration of this backend was rewritten in Rust (axum,
`citybehavex-web`). That rewrite has been reverted in favor of this FastAPI
backend -- fastmob's Python API is the better fit here than driving
`fastmob-rs`/`fastmob-core` directly from Rust.
