#!/usr/bin/env python3
"""One-off: refresh existing ablation reports to add network validation.

These 35 combos were reported earlier tonight with --skip-network-validation
(the co-presence graph construction was a slow, buggy bottleneck at the
time). Now that it's fixed (fastmob.social.co_presence_graph_from_staypoints
+ the graph_from_edges contiguity fix), re-run `citybehavex report` WITHOUT
the skip flag for each already-reported combo, overwriting its existing
report JSON in place. Does not touch the manifest (report_json_path is
unchanged) or re-simulate anything.
"""
import json
import subprocess
import sys
import time

MANIFEST = "data/ablation_logs/manifest.jsonl"
VARIANTS = {"full", "no_profile", "no_micro_sched", "no_social", "no_transport"}
DATASETS = {"shanghai", "yjmob", "yjmob2"}

rows = [json.loads(line) for line in open(MANIFEST)]
targets = [
    r
    for r in rows
    if r["dataset"] in DATASETS and r["variant"] in VARIANTS and r.get("report_json_path")
]
targets.sort(key=lambda r: {"shanghai": 0, "yjmob2": 1, "yjmob": 2}.get(r["dataset"], 9))

print(f"=== network-validation backfill: {len(targets)} combos ===", flush=True)
for i, r in enumerate(targets, 1):
    tag = f"{r['dataset']}_{r['variant']}_run{r['run_index']}"
    t0 = time.time()
    print(f"--- [{i}/{len(targets)}] {tag} ---", flush=True)
    result = subprocess.run(
        [
            "uv", "run", "citybehavex", "report",
            "--config", r["config_path"],
            "--synthetic", r["trajectories_path"],
            "--json", r["report_json_path"],
        ],
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    if result.returncode == 0:
        print(f"OK {tag} ({dt:.1f}s)", flush=True)
    else:
        print(f"FAILED {tag} ({dt:.1f}s)", flush=True)
        print(result.stdout[-2000:], flush=True)
        print(result.stderr[-2000:], flush=True)

print("=== network-validation backfill done ===", flush=True)
