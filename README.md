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

### Fine-tuning the ModernBERT schedule aligner

To label sampled profile-diary pairs with the LLM server and fine-tune the
macro-schedule alignment scorer, run:

```bash
uv run --extra finetuning python scripts/train_modernbert_schedule_aligner.py \
  --profiles-path data/gparis/results/gparis_agent_profiles.parquet \
  --diary-path data/llm_diaries_gparis/validated_diaries_weekday.json \
  --diary-path data/llm_diaries_gparis/validated_diaries_weekend.json \
  --llm-base-url http://localhost:8081 \
  --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dataset-output data/llm_diaries_gparis/schedule_alignment_scores.parquet \
  --output-model-path models/modernbert-schedule-aligner \
  --sample-size 1000 \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 2e-5
```

The script asks the LLM for a reason and a score, but only persists the numeric
alignment score and pair metadata. To use the trained scorer for macro-schedule
ddCRP selection, serve the saved model with TEI and set:

```yaml
schedule:
  similarity_backend: alignment_model
  alignment_base_url: http://localhost:8082
  alignment_model: models/modernbert-schedule-aligner
```

### Fine-tuning the ModernBERT vehicle ownership aligner

To replace fixed car/bike ownership probabilities with profile-conditioned
probabilities, first generate or load agent profiles, then label
transport-neutral profile/vehicle pairs with the LLM and fine-tune one shared
car+bike scorer:

```bash
uv run --extra finetuning python scripts/train_modernbert_vehicle_ownership_aligner.py \
  --profiles-path data/gparis/results/gparis_agent_profiles.parquet \
  --city-profile "Greater Paris metropolitan region, urban mobility with commuting, errands, leisure, healthcare, studies, and home routines." \
  --llm-base-url http://localhost:8081 \
  --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dataset-output data/llm_diaries_gparis/vehicle_ownership_alignment_scores.parquet \
  --output-model-path models/modernbert-vehicle-ownership-aligner \
  --sample-size 2000 \
  --llm-concurrency 8 \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 2e-5
```

The training query excludes existing transport ownership text so the model
learns from demographics and city context rather than echoing old random
labels. To use the scorer during profile generation, serve it with the same
rerank-compatible server on a free port and set:

```bash
PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages \
  .venv/bin/python scripts/serve_schedule_aligner.py \
  --model-path models/modernbert-vehicle-ownership-aligner \
  --port 8084 \
  --device cuda \
  --predict-batch-size 128
```

```yaml
profiles:
  ownership_alignment_backend: rerank
  ownership_alignment_base_url: http://localhost:8084
  ownership_alignment_model: models/modernbert-vehicle-ownership-aligner
  ownership_alignment_batch_size: 256
  ownership_alignment_cache_path: data/llm_diaries_gparis/vehicle_ownership_alignment_cache.npz
  ownership_alignment_concurrency: 4
```

At runtime, the car and bike scores are treated as Bernoulli probabilities.
The sampled booleans still populate `has_car` and `has_bike`, and the numeric
scores are saved as `car_ownership_score` and `bike_ownership_score`.

### Fine-tuning the ModernBERT profile coherence aligner

To repair incoherent demographic combinations before vehicle ownership
alignment, first run a simulation once so `profiles.output` exists, or point the
script at another generated profile parquet. Then label real profiles plus
synthetic inconsistent variants and fine-tune the scorer:

```bash
uv run --extra finetuning python scripts/train_modernbert_profile_coherence_aligner.py \
  --profiles-path data/gparis/results/gparis_agent_profiles.parquet \
  --city-profile "Greater Paris metropolitan region, urban mobility with commuting, errands, leisure, healthcare, studies, and home routines." \
  --llm-base-url http://localhost:8081 \
  --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dataset-output data/llm_diaries_gparis/profile_coherence_scores.parquet \
  --output-model-path models/modernbert-profile-coherence-aligner \
  --sample-size 2000 \
  --mutation-ratio 0.5 \
  --llm-concurrency 8 \
  --device cuda
```

Serve the saved model with the rerank-compatible server on a free port:

```bash
PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages \
  .venv/bin/python scripts/serve_schedule_aligner.py \
  --model-path models/modernbert-profile-coherence-aligner \
  --port 8085 \
  --device cuda \
  --predict-batch-size 128
```

Then enable it under `profiles`:

```yaml
profiles:
  coherence_alignment_backend: rerank
  coherence_alignment_base_url: http://localhost:8085
  coherence_alignment_model: models/modernbert-profile-coherence-aligner
  coherence_alignment_batch_size: 256
  coherence_alignment_cache_path: data/llm_diaries_gparis/profile_coherence_alignment_cache.npz
  coherence_alignment_concurrency: 4
  coherence_rerun_rounds: 3
  coherence_rerun_threshold: 0.6
```

Each round scores profile clusters, reruns demographics for invalid agents, and
preserves `uid`, `home_tile`, `work_tile`, `has_car`, and `has_bike`. If the
scorer is disabled or unavailable, profile generation continues unchanged.

### Fine-tuning the ModernBERT activity aligner

To label sampled profile/block/activity pairs with the same running LLM server
and fine-tune the micro-activity alignment scorer, run:

```bash
uv run --extra finetuning python scripts/train_modernbert_activity_aligner.py \
  --profiles-path data/gparis/results/gparis_agent_profiles.parquet \
  --diary-path data/llm_diaries_gparis/validated_diaries_weekday.json \
  --diary-path data/llm_diaries_gparis/validated_diaries_weekend.json \
  --llm-base-url http://localhost:8081 \
  --llm-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dataset-output data/llm_diaries_gparis/activity_alignment_scores.parquet \
  --output-model-path models/modernbert-activity-aligner \
  --sample-size 5000 \
  --llm-concurrency 8 \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 2e-5
```

The script scores only activity categories valid for each HOME/WORK/OTHER
schedule block and includes previous-activity context in the training query.
`--llm-concurrency` controls how many labeling requests are in flight at once;
raise it when the vLLM server has batching headroom, and lower it if requests
start timing out.
To use the trained scorer for micro-activity CRP alignment, serve the saved
model with the same rerank-compatible server and set:

```bash
PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages \
  .venv/bin/python scripts/serve_schedule_aligner.py \
  --model-path models/modernbert-activity-aligner \
  --port 8083 \
  --device cuda \
  --predict-batch-size 128
```

On this RTX 5090 workstation, keep the vLLM environment's CUDA 13 / PyTorch
build first on `PYTHONPATH` when serving rerankers on GPU; the project
environment's older CUDA 12.4 PyTorch build cannot run sm_120 kernels.

```yaml
activities:
  enabled: true
  alignment_backend: rerank
  alignment_base_url: http://localhost:8083
  alignment_model: models/modernbert-activity-aligner
  alignment_batch_size: 512
  alignment_cache_path: data/llm_diaries_gparis/activity_alignment_cache.npz
  alignment_concurrency: 4
  alignment_retries: 2
  alignment_checkpoint_every: 20
  prune_to_reachable: false  # set true to run the cheap reachability probe first and skip unvisited (cluster, block) pairs
```

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
