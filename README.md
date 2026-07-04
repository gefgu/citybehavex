# CityBehavEx

CityBehavEx ships a small Rust extension (`citybehavex._core`, built with maturin)
that implements the project simulation core: agents follow a sub-hourly Markov
schedule (5/15-min slots, weekday/weekend chains), social/EPR-style logic chooses
locations, and a car trip-duration heuristic (`haversine / car_speed_kmh`) shifts
arrivals and departures off the slot grid. The extension path-depends on the
sibling `../skmob2` crate.

## Building

The project uses the **maturin** build backend. After cloning (or after any change
to the Rust sources), build the extension into the project `.venv`:

```bash
./scripts/update_local_skmob.sh        # if ../skmob2 Rust changed
./scripts/update_local_citybehavex.sh  # builds citybehavex._core
```

`scripts/update_local_citybehavex.sh` compiles `citybehavex-py` and installs the
package editable. Rust edits are **not** hot-reloaded — rerun the script after
changing any `.rs` file.

## Running

Run the configured Greater Paris simulation from the repository root:

```bash
uv run citybehavex simulate --config configs/gparis_simulation_core.yaml
```

LLM generation creates 30 weekday and 30 weekend diaries by default. Override
the number, within the supported range of 10 to 30, from the CLI:

```bash
uv run citybehavex simulate --config configs/gparis_simulation_core.yaml --diary-count 20
```

Equivalently, run the package as a Python module:

```bash
uv run python -m citybehavex simulate --config configs/gparis_simulation_core.yaml
```

Open the live web UI to analyze simulation runs and comparison data. The old
standalone HTML report path is deprecated; chart payloads are built on demand by
the web backend.

### Embedding model (ddCRP schedule selection)

Schedule selection embeds each diary with `nomic-embed-text-v2-moe`. With
`embedding.auto_launch: true` (the default) citybehavex starts the server below on
demand and shuts it down afterwards, caching vectors so it rarely reruns. To run it
yourself and reuse it (set `embedding.auto_launch: false` and point
`embedding.base_url` at it), serve the model with vLLM:

```bash
uv run --extra embeddings vllm serve nomic-ai/nomic-embed-text-v1.5 \
  --runner pooling --trust-remote-code --port 8001
```

The `embeddings` extra (vLLM) is required only for serving; without a GPU set
`embedding.enabled: false` to fall back to identity similarity.

## Web app

`web/` is the supported comparison UI: a FastAPI backend serves comparison and
synthetic-only chart payloads as JSON, and a React/Vite frontend renders them
(see `web/README.md` for details).

Run the two dev servers from the repository root. Use the venv's interpreter
directly for the backend — `uv run` would try to rebuild the Rust extension:

```bash
# backend (http://localhost:8000)
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --reload --port 8000
```

```bash
# frontend (http://localhost:5173, proxies /api to the backend)
cd web/frontend
npm install
npm run dev
```

Then open http://localhost:5173. The Experiments page is populated from
`configs/*.yaml`; opening a run's charts builds the payload on first request
(large cities take a while) and caches it under `data/.web_cache/`.

Node.js is provided via nvm; source it first if `node` is not on your PATH:

```bash
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"
```

## Updating local skmob2

Rebuild the sibling `../skmob2` Rust extension into this project's `.venv`:

```bash
./scripts/update_local_skmob.sh
```
