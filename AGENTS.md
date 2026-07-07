# AGENTS.md

## Serving the diary-generation LLM

The diary-generation model (`Qwen/Qwen2.5-32B-Instruct-AWQ`, port 8081) is served from the
sibling `/home/gustavo/vllm` folder, not from this repo. Start it with `./serve.sh` there
(a `screen -S llm` session is normally kept around for this). Use `--gpu-memory-utilization
0.75` (the value already in `vllm/serve.sh`), not the `0.9` in `vllm/command.txt` — this GPU
also hosts the schedule aligner (TEI, port 8082, ~0.9 GB) and the activity aligner
(`scripts/serve_schedule_aligner.py`, port 8083, ~2.1 GB) persistently, so 0.9 leaves too
thin a margin against a shared card. Port convention: 8081 = diary LLM, 8082 = schedule
aligner, 8083 = activity aligner, 8001 = on-demand embedding server (auto-launched by
`embedding.auto_launch: true`, only spawned on a cache miss).

## Activity Aligner Fine-Tuning

- `scripts/train_modernbert_activity_aligner.py` labels profile/block/activity pairs through the configured OpenAI-compatible chat endpoint.
- Use `--llm-concurrency` to keep multiple labeling requests in flight so vLLM can batch work. Start with `--llm-concurrency 8`; increase when GPU utilization is low, and decrease if requests time out.
- Never serve local AI models on CPU on this workstation. For CrossEncoder rerankers, launch with the vLLM environment's CUDA 13 / PyTorch build first on `PYTHONPATH`, for example: `PYTHONPATH=/home/gustavo/vllm/.venv/lib/python3.12/site-packages .venv/bin/python scripts/serve_schedule_aligner.py --model-path models/modernbert-activity-aligner --port 8083 --device cuda --predict-batch-size 128`.
- Use `--predict-batch-size` on `scripts/serve_schedule_aligner.py` and `activities.alignment_batch_size` in configs to keep rerank inference batched. `--predict-batch-size 128` measured best on this workstation's RTX 5090 (shared with a persistent ~25GB vLLM engine) — 256/512 measured ~15% *slower* (~1400 vs ~1650 pairs/sec) there, so it's contention-bound, not headroom-bound; re-measure with a quick `/score_pairs` timing loop if the GPU's other residents change. Pair with `activities.alignment_batch_size: 512` in configs.
- `scripts/serve_schedule_aligner.py` coalesces concurrent `/rerank` and `/score_pairs` requests into fewer, larger `CrossEncoder.predict()` calls (a background thread drains whatever's queued within `--coalesce-window-ms`, default 20ms, up to `--coalesce-max-pairs`, default 2048) — this only helps when the client actually sends concurrent requests, so pair it with `activities.alignment_concurrency` (default 4) on the client side. Measured gain in this shared-GPU environment was modest (~1.1x) — the model itself is the bottleneck here, not request overhead.
- `citybehavex.activities.alignment.score_activity_alignment` also checkpoints its on-disk cache (`activities.alignment_cache_path`) atomically every `activities.alignment_checkpoint_every` batches (default 20), not just at the end, and retries a failed batch up to `activities.alignment_retries` times (default 2) before giving up — a crash mid-run now loses at most one checkpoint interval's worth of scores instead of the whole run.
- `citybehavex.schedules.alignment.score_alignment_matrix` (the macro-schedule/ddCRP reranker, `schedule.alignment_cache_path`) has the same guarantee: cache keys are hashed on `(model, profile_text, diary_text)`, so a new simulation reusing the same profiles/diaries only re-sends whatever's actually missing (new profile clusters, new diaries) — verified this only issues one rerank call per genuinely-new row, zero calls when everything's already cached. Checkpoints atomically every `schedule.alignment_checkpoint_every` profile rows (default 5), including on early-return/failure, not just at the very end.
- Keep the schedule reranker and activity reranker on separate ports when both are needed. The current convention is schedule alignment on `http://localhost:8082` and activity alignment on `http://localhost:8083`.
