# CityBehavEx

**CityBehavEx** is a scalable, empirically validated, LLM-assisted urban behavior
simulation platform. It generates synthetic city-scale mobility trajectories and
lets users inspect, replay, debug, and evaluate them against observed mobility,
time-use, semantic, transport, and social-network patterns.

This repository accompanies an EMNLP Demo Track submission:

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

## Requirements

Core requirements:

- Python 3.11+
- Rust toolchain
- `uv`
- Node.js 18+ and npm, for the web frontend

Optional requirements:

- A CUDA GPU for embedding, cross-encoder, or LLM serving
- A Mapbox token for the high-performance animated timeline map
- An OpenAI-compatible LLM endpoint when regenerating diaries or training
  semantic aligners

The Python package uses `maturin` to build the Rust extension. When installing
from source, make sure the mobility dependencies `fkmob` and `skmob-vis` are
available in the locations declared in `pyproject.toml`, or update those entries
to point to installed/vendored copies included with the artifact.

## Quick Start

From the repository root:

```bash
uv sync --extra web
./scripts/update_local_citybehavex.sh
```

Run a public-data-oriented YJMOB scenario:

```bash
uv run citybehavex simulate --config configs/yjmob_simulation.yaml
```

The command writes simulation outputs under the paths configured in the YAML
file, typically inside `data/.../results/`. Existing caches are reused when
available.

## Web Demo

Start the backend:

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir web/backend --reload --port 8000
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

The frontend proxies `/api` requests to the FastAPI backend on port `8000`.
The Experiments page is populated from `configs/*.yaml`. Opening charts for a
run builds the validation payload on first request and caches it under
`data/.web_cache/`.

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
8081  diary-generation LLM, OpenAI-compatible chat endpoint
8082  macro-schedule alignment reranker
8083  activity alignment reranker
8001  optional embedding server for schedule-selection embeddings
```

If `embedding.auto_launch: true` is enabled and embeddings are missing,
CityBehavEx can launch the embedding server on demand. To serve it manually:

```bash
uv run --extra embeddings vllm serve nomic-ai/nomic-embed-text-v1.5 \
  --runner pooling \
  --trust-remote-code \
  --port 8001
```

To serve a fine-tuned cross-encoder reranker:

```bash
.venv/bin/python scripts/serve_schedule_aligner.py \
  --model-path models/modernbert-activity-aligner \
  --port 8083 \
  --device cuda \
  --predict-batch-size 128
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

## Data Notes

The paper evaluates CityBehavEx with Greater Paris, Shanghai, and YJMOB mobility
datasets. Greater Paris and Shanghai are non-public datasets and cannot be
redistributed in this repository. YJMOB is public and is the recommended dataset
for artifact reviewers who need to reproduce an end-to-end run without private
data access.

Large simulation outputs are intentionally not committed. The repository expects
input and generated files under `data/`, with exact paths controlled by each YAML
configuration.

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

- **Rust extension not found:** rerun `./scripts/update_local_citybehavex.sh`.
- **Editable `fkmob` or `skmob-vis` not found:** make sure the artifact includes
  those dependencies or update `pyproject.toml` to point to installed versions.
- **Frontend cannot reach the API:** confirm the backend is running on
  `http://localhost:8000` and the frontend on `http://localhost:5173`.
- **Timeline map is blank:** set `VITE_MAPBOX_TOKEN` in
  `web/frontend/.env.local` and restart Vite.
- **Alignment endpoint errors:** either start the configured reranker/LLM service
  or disable that backend in the YAML config.
- **Large chart load is slow:** the first request builds and caches payloads; the
  next request should reuse `data/.web_cache/`.

## Citation

```bibtex
@misc{santos2026citybehavex,
      title={CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform}, 
      author={Gustavo H. Santos and Aline Viana and Thiago H Silva},
      year={2026},
      eprint={2607.12086},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.12086}, 
}
```
