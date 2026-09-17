# AGENTS.md

## Distance Calculations

- Do not add custom Haversine implementations in citybehavex. Use fastmob's distance utilities directly, normally `fastmob.network.haversine_m_batch`, and convert metres to kilometres at the call site when needed.

## Serving the diary-generation LLM

The diary-generation model (`Qwen/Qwen2.5-32B-Instruct-AWQ`, port 8081) is served from a
sibling host (not this workstation — see `configs/*.yaml`'s `llm.base_url`), not from this
repo. Start it with `./serve.sh` in that host's `vllm` checkout (a `screen -S llm` session
is normally kept around for this). Use `--gpu-memory-utilization 0.75`, not the `0.9` in
`vllm/command.txt` — that GPU may also be serving other models concurrently, so 0.9 leaves
too thin a margin against a shared card. That same sibling endpoint may serve a different
model at any given time (e.g. `mistral-small-3.2-24b` instead of Qwen) — check
`GET {base_url}/v1/models` before assuming which one is live; only one model is loaded
there at once, so switching requires whoever runs that host's vLLM to restart it.

## Serving the aligner/embedding server

The ModernBERT CrossEncoder aligners actually deployed (schedule, activity,
vehicle-ownership — see `models/modernbert-*-aligner/`; profile-coherence is
currently disabled everywhere via `coherence_alignment_backend: none`, and
POI-type reuses the activity aligner's checkpoint rather than having its own,
see `citybehavex/activities/alignment.py`'s `model = config.poi_type_alignment_model
or config.alignment_model`) and the diary/profile embedding model are served
by one consolidated process, `scripts/serve_aligners.py`, on **port 8090**.
It lazily loads whichever model a request names (clients already send a
`model` field — the exact `*_alignment_model`/`embedding.model` checkpoint
path/id from config — on every `/rerank`, `/score_pairs`, and
`/v1/embeddings` call), evicts a model after `--idle-ttl-s` (default 600s) of
no requests, and exposes `POST /unload {"model": "..."}` for an immediate
evict. Start it with citybehavex's own venv -- no separate torch/CUDA
environment needed, `pyproject.toml` already pins the CUDA 13 wheel index:
```
uv run python scripts/serve_aligners.py --port 8090 --device cuda \
  --predict-batch-size 128
```
Every scenario config's `*_alignment_base_url` and `embedding.base_url` point at this one
port; only the `*_alignment_model`/`embedding.model` fields select which checkpoint gets
loaded. Simulation pipeline code (`citybehavex/simulation/{profile,schedule,activity}_pipeline.py`)
proactively calls `/unload` for a model right after the pipeline phase that needed it
finishes, so idle-TTL eviction is a backstop, not the only mechanism.

## Activity Aligner Fine-Tuning

- `scripts/train_modernbert_activity_aligner.py` labels profile/block/activity pairs through the configured OpenAI-compatible chat endpoint.
- Use `--llm-concurrency` to keep multiple labeling requests in flight so vLLM can batch work. Start with `--llm-concurrency 8`; increase when GPU utilization is low, and decrease if requests time out.
- Never serve local AI models on CPU on this workstation. Launch `scripts/serve_aligners.py` from citybehavex's own venv (see the invocation above) with `--device cuda`.
- Use `--predict-batch-size` on `scripts/serve_aligners.py` and `activities.alignment_batch_size` in configs to keep rerank inference batched. `--predict-batch-size 128` measured best on this workstation's RTX 5090 (shared with other GPU residents) — 256/512 measured ~15% *slower* (~1400 vs ~1650 pairs/sec) there, so it's contention-bound, not headroom-bound; re-measure with a quick `/score_pairs` timing loop if the GPU's other residents change. Pair with `activities.alignment_batch_size: 512` in configs.
- `scripts/serve_aligners.py` coalesces concurrent `/rerank` and `/score_pairs` requests **per loaded model** into fewer, larger `CrossEncoder.predict()` calls (a background thread per model drains whatever's queued within `--coalesce-window-ms`, default 20ms, up to `--coalesce-max-pairs`, default 2048) — this only helps when the client actually sends concurrent requests, so pair it with `activities.alignment_concurrency` (default 4) on the client side. Measured gain in this shared-GPU environment was modest (~1.1x) — the model itself is the bottleneck here, not request overhead.
- `citybehavex.activities.alignment.score_activity_alignment` also checkpoints its on-disk cache (`activities.alignment_cache_path`) atomically every `activities.alignment_checkpoint_every` batches (default 20), not just at the end, and retries a failed batch up to `activities.alignment_retries` times (default 2) before giving up — a crash mid-run now loses at most one checkpoint interval's worth of scores instead of the whole run.
- `citybehavex.schedules.alignment.score_alignment_matrix` (the macro-schedule/SW-CRP reranker, `schedule.alignment_cache_path`) has the same guarantee: cache keys are hashed on `(model, profile_text, diary_text)`, so a new simulation reusing the same profiles/diaries only re-sends whatever's actually missing (new profile clusters, new diaries) — verified this only issues one rerank call per genuinely-new row, zero calls when everything's already cached. Checkpoints atomically every `schedule.alignment_checkpoint_every` profile rows (default 5), including on early-return/failure, not just at the very end.
