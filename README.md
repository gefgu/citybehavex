# CityBehavEx

**CityBehavEx** is a scalable, empirically validated, LLM-assisted urban behavior
simulation platform. It generates synthetic city-scale mobility trajectories and
lets users inspect, replay, debug, and evaluate them against observed mobility,
time-use, semantic, transport, and social-network patterns.

This repository accompanies an EACL Demo Track submission:

> **CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban
> Simulation Platform**

<img width="1915" height="1273" alt="Timeline View" src="https://github.com/user-attachments/assets/a68ae9c1-b426-407a-adc6-fc44fee13d60" />

The demo focuses on the complete workflow: configure a city scenario, run a
simulation, replay agent trajectories, inspect profiles and activity traces, and
compare synthetic behavior with empirical validation metrics through the web
dashboard.

## Why CityBehavEx?

Recent LLM-based urban simulators can produce rich behavior descriptions, but
they are often expensive to scale and weakly validated against real mobility
patterns. CityBehavEx separates semantic reasoning from trajectory execution:

- **Scalable simulation core.** A Rust-backed engine simulates large populations
  with sub-hourly schedules, exploration and preferential return, social ties,
  transport choices, and micro-activities.
- **LLM-assisted, not LLM-per-action.** LLMs help generate and calibrate diaries
  and semantic supervision, while fine-tuned cross-encoders score profile,
  schedule, POI, vehicle-ownership, and activity compatibility efficiently.
- **Empirical validation.** The dashboard reports spatial, temporal, semantic,
  time-use, transport, behavioral-profile, motif, OD, and social-network metrics.
- **Inspectable agents.** The interface supports trajectory replay, profile
  inspection, macro-schedules, micro-activities, transport legs, and social
  encounters.
- **Reproducible experiments.** Scenarios are configured with YAML files, outputs
  are stored as parquet/JSON sidecars, and chart payloads are cached.

In the paper experiments, CityBehavEx runs orders of magnitude faster than recent
LLM-based urban simulation baselines while matching empirical spatial and
temporal mobility distributions more closely.

## UI Screenshots:

### Experiments 
<img width="1465" height="1113" alt="image" src="https://github.com/user-attachments/assets/817e7206-cad5-4ff9-b18b-131333070c54" />

### Charts 
<img width="1648" height="2013" alt="image" src="https://github.com/user-attachments/assets/29632960-64a3-4978-a659-34da807cb139" />

## License

CityBehavEx is released under the **GNU Affero General Public License v3.0
(AGPLv3)**. See [`LICENSE`](LICENSE).

The AGPLv3 license is intentional: it guarantees that improvements to hosted or
modified versions of the simulator remain available to the research community.

## Repository Overview

```text
citybehavex/                 Python package and report/evaluation logic
citybehavex/aligners/        Packaged local aligner/embedding server (`--start-aligners`)
citybehavex/templates/       `citybehavex init` project scaffold and the YJMOB-1k asset manifest
citybehavex/project.py       `init`/`data download`/`doctor` CLI implementations
citybehavex/services.py      Local aligner process lifecycle helpers used by `simulate`
citybehavex-py/              Rust simulation core exposed as citybehavex._core
configs/                     Reproducible scenario and ablation configurations
scripts/                     Simulation, training, serving, and sweep utilities
web/backend/                 FastAPI backend for experiment discovery and charts
web/frontend/                React/Vite validation and trajectory-replay UI
data/                        Input/output location for scenarios and runs
models/                      Optional fine-tuned alignment models
```

The simulator reads scenario settings from `configs/*.yaml`. The web UI discovers
those configs, finds their generated runs, and builds validation views on demand.
The web backend is the FastAPI app in `web/backend/app`; see
[`web/README.md`](web/README.md) for development and deployment commands.

## Requirements

Core requirements:

- Python 3.11.4+ (earlier 3.11.x patches lack `tarfile`'s `filter=` support,
  which `citybehavex data download` relies on)

Additional requirements for the web dashboard (not needed for the pip-installed
CLI on its own):

- Node.js 18+ and npm, for the web frontend

Additional requirements for building from a source checkout (not needed when
installing the published wheel):

- Rust toolchain, for the simulation core (`citybehavex._core`)
- `uv`

Optional requirements:

- A CUDA GPU for embedding, cross-encoder, or LLM serving
- A Mapbox token for the high-performance animated timeline map
- An OpenAI-compatible LLM endpoint when regenerating diaries or training
  semantic aligners

The published wheel ships a prebuilt Rust extension, so `pip install
citybehavex` needs no Rust toolchain. Building from source uses `maturin` to
build that extension (the core simulation engine only — the web backend is
pure Python/FastAPI and needs no Rust toolchain either way). CityBehavEx
builds on Fastkit-Mobility:

- [Fastkit-Mobility](https://github.com/gefgu/fastmob) (`fastmob` on PyPI) —
  Rust-accelerated mobility analysis and visualization, available as
  [`fastmob==0.2.2`](https://pypi.org/project/fastmob/0.2.2/). Its
  `visualization` extra provides the Rust-backed ECharts visualizations.

## Quick Start

Install the published wheel:

```bash
python -m pip install citybehavex
citybehavex init my-citybehavex-project
citybehavex data download yjmob --project my-citybehavex-project
cd my-citybehavex-project
```

Validate the project, then run the local temporary aligner service with the
simulation:

```bash
citybehavex doctor --config configs/yjmob-1k.yaml
citybehavex simulate --config configs/yjmob-1k.yaml --start-aligners
```

No LLM setup needed for this first run — the project ships with bundled,
pre-generated diaries (`data/yjmob-1k/llm_diaries/validated_diaries_*.json`),
reused automatically instead of calling an LLM. To regenerate diaries
instead (e.g. after changing `diaries.city_profile`), set
`CITYBEHAVEX_LLM_BASE_URL`, `CITYBEHAVEX_LLM_API_KEY`, and
`CITYBEHAVEX_LLM_MODEL` to a real OpenAI-compatible endpoint and delete the
corresponding `validated_diaries_*.json` first.

`--start-aligners` requires CUDA by default. Use `--aligner-device cpu` only
when a GPU is unavailable; it is much slower.

The command writes simulation outputs under the paths configured in the YAML
file, typically inside `data/.../results/`. Existing caches are reused when
available.

> **No model training required.** Every shipped config already points at
> CityBehavEx's five pretrained ModernBERT aligners, hosted on the
> [Hugging Face Hub](https://huggingface.co/gefgu) and resolved automatically
> on first use — no local checkpoints or extra setup. See
> [Alignment Services](#alignment-services) below for the model list and how
> to serve them locally instead.

## Web Demo

> **Requires a source checkout, not just `pip install citybehavex`.**
> `web/` (the FastAPI backend and React frontend) isn't part of the
> published wheel/sdist at all -- clone the repo to run it.

Install the backend's extra dependencies (FastAPI + uvicorn aren't needed
for the CLI/simulation core, so they're not in the base install) and start
it:

```bash
uv sync --extra web
uv run uvicorn app.main:app --app-dir web/backend --port 8000
```

Start the frontend:

```bash
cd web/frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

By default the frontend talks directly to the FastAPI backend on port `8000`.
The Experiments page is populated from `configs/*.yaml`. Opening charts for a
run builds the validation payload on first request and caches it under
`data/.web_cache/`.

The FastAPI app can also single-origin-serve the built frontend (`npm run build`
in `web/frontend/`, then restart uvicorn — it serves both the API and the
static SPA whenever `web/frontend/dist` exists). See
[`web/README.md`](web/README.md) for the full endpoint list and deployment
notes.

### Timeline Map

The animated timeline uses Mapbox GL. To enable it, create
`web/frontend/.env.local`:

```bash
VITE_MAPBOX_TOKEN=pk.your_token_here
```

Restart `npm run dev` after creating or editing this file.

## Running the Main Components

### Simulation CLI

```bash
uv run citybehavex simulate --config configs/yjmob_simulation.yaml
```

Equivalent module entry point:

```bash
uv run python -m citybehavex simulate --config configs/yjmob_simulation.yaml
```

Override the number of generated candidate diaries:

```bash
uv run citybehavex simulate \
  --config configs/yjmob_simulation.yaml \
  --diary-count 20
```

### Validation Dashboard

The dashboard includes:

- ECDFs and fitted mobility laws for travel distance, radius of gyration, trip
  duration, dwell time, and visitation frequency
- visit-purpose distributions and activity-transition matrices
- daily routines, motifs, mobility profiles, and time-use summaries
- H3 spatio-temporal visit difference maps
- home/work maps, transport summaries, and social-network validation
- animated trajectory replay with agent-level inspection

For metric-by-metric calibration guidance, including config knobs and
fine-tuning levers, see [`CALIBRATION.md`](CALIBRATION.md).

### Alignment Services

CityBehavEx can run with cached alignment scores, simple fallbacks, or live
alignment services. The convention used by the project is:

```text
8081  diary-generation LLM, OpenAI-compatible chat endpoint (a sibling host)
8090  consolidated aligner + embedding server (scripts/serve_aligners.py)
```

All five ModernBERT CrossEncoder aligners (schedule, activity,
vehicle-ownership, profile-coherence, POI-type) and the diary/profile
embedding model are served by **one process** on port 8090. It lazily loads
whichever model a request names — the `model` field every request already
carries, taken directly from each config's `*_alignment_model` /
`embedding.model` field — evicts a model after `--idle-ttl-s` (default 600s)
of no requests, and exposes `POST /unload {"model": "..."}` for an immediate
evict.

The five pretrained aligners are published on the Hugging Face Hub and are
what every shipped config points at by default — no local checkpoint copy or
extra setup needed, `sentence-transformers` resolves them straight from the
Hub on first use, same as a local path:

| aligner | Hugging Face repo |
| --- | --- |
| schedule | [`gefgu/modernbert-schedule-aligner`](https://huggingface.co/gefgu/modernbert-schedule-aligner) |
| activity | [`gefgu/modernbert-activity-aligner`](https://huggingface.co/gefgu/modernbert-activity-aligner) |
| vehicle-ownership | [`gefgu/modernbert-vehicle-ownership-aligner`](https://huggingface.co/gefgu/modernbert-vehicle-ownership-aligner) |
| profile-coherence | [`gefgu/modernbert-profile-coherence-aligner`](https://huggingface.co/gefgu/modernbert-profile-coherence-aligner) |
| POI-type | [`gefgu/modernbert-poi-type-aligner`](https://huggingface.co/gefgu/modernbert-poi-type-aligner) |

Start the server with citybehavex's own venv — no separate CUDA/torch
environment needed, `pyproject.toml` already pins the CUDA 13 wheel index:

```bash
uv run python scripts/serve_aligners.py \
  --port 8090 \
  --device cuda \
  --predict-batch-size 128
```

For a one-off simulation, the equivalent public CLI flow is simply:

```bash
citybehavex simulate --config configs/yjmob-1k.yaml --start-aligners
```

Point every config's `*_alignment_base_url` and `embedding.base_url` at this
one port; only the `*_alignment_model` / `embedding.model` fields select which
checkpoint gets loaded for a given request — no separate service or port per
aligner. A `*_alignment_model` field accepts a Hugging Face repo id (the
default) or any local/compatible checkpoint path, selected directly in YAML
without restarting the server:

```yaml
activities:
  alignment_base_url: http://localhost:8090
  alignment_model: /path/to/my-modernbert-cross-encoder
```

The fine-tuning scripts expose the corresponding model flags. Use
`--base-model` to select the ModernBERT (or other compatible) starting model
and `--output-model-path` to choose where the fine-tuned CrossEncoder is saved.
For example:

```bash
.venv/bin/python scripts/train_modernbert_activity_aligner.py \
  --base-model nomic-ai/modernbert-embed-base \
  --output-model-path models/my-activity-aligner
```

Disable these services in YAML when running without the corresponding models or
GPU. The simulator will use configured fallbacks and existing caches where
possible.

## Configuration

Important scenario files:

```text
configs/gparis_simulation.yaml       Greater Paris scenario
configs/shanghai_simulation.yaml     Shanghai scenario
configs/yjmob_simulation.yaml        YJMOB regular scenario
configs/yjmob2_simulation.yaml       YJMOB disaster/special-event scenario
configs/ablations/                   Module-level ablation configurations
```

The major configurable modules are:

- `profiles`: synthetic population, home/work assignment, coherence and vehicle
  ownership alignment
- `schedule`: diary selection, semantic alignment, and exploration parameters
- `activities`: MTUS-grounded micro-schedules and activity alignment
- `transport`: walking, cycling, road, rail, and fallback travel behavior
- `social`: initial friendship formation and co-location-based tie updates
- `embedding`: optional embedding backend and cache behavior

To fix locations while retaining generated demographics, set
`profiles.profiles_path` to a JSON or Parquet table keyed by one-based `uid`.
Only supplied, non-null fields override generation. Integer locations are
legacy 0-based tessellation row indices; string locations are matched against
the tessellation's `tile_id` column, so POI IDs, UUIDs, H3 IDs, and HOME-anchor
IDs can be supplied directly:

```json
[
  {"uid": 1, "home_tile": "home_anchor_17", "work_tile": "88309959d1fffff"},
  {"uid": 2, "home_tile": "home_anchor_42", "work_tile": "aadfd6a4-ba82-46ae-92da-973e3c91aaee"}
]
```

Tile indices address the runtime tessellation, after any configured residential
home anchors are appended.

## Data Notes

The paper evaluates CityBehavEx with Greater Paris, Shanghai, and YJMOB mobility
datasets. Greater Paris and Shanghai are non-public datasets and cannot be
redistributed in this repository. YJMOB is public and is the recommended dataset
for artifact reviewers who need to reproduce an end-to-end run without private
data access.

Large simulation outputs are intentionally not committed. The repository expects
input and generated files under `data/`, with exact paths controlled by each YAML
configuration. `data/` itself is not part of the repository (it is gitignored) —
create it locally and populate it before running a scenario.

> **The packaged `citybehavex data download yjmob` sample is not this
> dataset.** It ships a small, entirely synthetic comparison baseline (the
> first 1,000 agents / first 7 days of an already-completed CityBehavEx run)
> so the pip-installed CLI has zero data-license entanglement. To validate
> against the real YJMob100K dataset, follow the steps below instead.

### Setting up `data/` for the YJMOB scenario

1. Create the directory: `mkdir -p data/yjmob`.
2. Obtain the YJMob100K dataset (released for the HuMob Challenge) and place its
   `dataset1` CSV as `data/yjmob/yjmob100k-dataset1.csv.gz`.
3. Run the preparation notebooks in order to produce the processed parquet files
   that `configs/yjmob_simulation.yaml` reads:
   `notebooks/01_yjmob_preparation/01_preprocessing.ipynb`, then `02_eda.ipynb`
   and `04_motif_distribution.ipynb` as needed for validation baselines.
4. For the disaster/special-event scenario (`configs/yjmob2_simulation.yaml`),
   run `notebooks/03_yjmob2_preparation/01_preprocessing.ipynb` against the
   corresponding YJMob2 release.

Greater Paris and Shanghai configs expect the same `data/<city>/...` layout, but
since their source datasets are private, they cannot be reproduced from this
repository alone.

## Reproducing Paper-Style Experiments

Run a configured experiment:

```bash
uv run citybehavex simulate --config configs/yjmob_simulation.yaml
```

Run module ablations:

```bash
./scripts/run_ablation.sh configs/ablations/yjmob/yjmob_no_profile.yaml
./scripts/run_ablation.sh configs/ablations/yjmob/yjmob_no_micro_sched.yaml
./scripts/run_ablation.sh configs/ablations/yjmob/yjmob_no_social.yaml
./scripts/run_ablation.sh configs/ablations/yjmob/yjmob_no_transport.yaml
```

Aggregate ablation logs:

```bash
uv run python scripts/aggregate_ablation_results.py
```

Then start the web demo and open the corresponding experiment to inspect charts,
timeline replay, metrics, and cached comparison payloads.

## Troubleshooting

- **Rust extension not found (source checkout only):** rerun
  `./scripts/update_local_citybehavex.sh`. The published wheel bundles a
  prebuilt extension and never hits this.
- **`fastmob` not found:** install the project dependencies with `uv sync`.
  The required `fastmob` features, including visualization, are installed from
  PyPI as extras.
- **Frontend cannot reach the API:** confirm the FastAPI backend is running on
  `http://localhost:8000` and the frontend on `http://localhost:5173`, and
  that `VITE_API_PROXY_TARGET` matches the backend you started.
- **Timeline map is blank:** set `VITE_MAPBOX_TOKEN` in
  `web/frontend/.env.local` and restart Vite.
- **Alignment endpoint errors:** either start the configured reranker/LLM service
  or disable that backend in the YAML config.
- **Large chart load is slow:** the first request builds and caches payloads; the
  next request should reuse `data/.web_cache/`.

## Citation

If you use CityBehavEx, please cite:

```bibtex
@misc{santos2026citybehavex,
  title     = {CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform},
  author    = {Santos, Gustavo H. and Viana, Aline and Silva, Thiago H.},
  year      = {2026},
  eprint    = {2607.12086},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2607.12086}
}
```

CityBehavEx depends on [Fastkit-Mobility](https://github.com/gefgu/fastmob)
([`fastmob` 0.2.2 on PyPI](https://pypi.org/project/fastmob/0.2.2/)) with its
`visualization` extra.
