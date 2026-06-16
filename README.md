# CityBehavEx

CityBehavEx ships a small Rust extension (`citybehavex._core`, built with maturin)
that implements trip-duration-aware STS-EPR by default: agents follow a sub-hourly
Markov schedule (5/15-min slots, weekday/weekend chains), social/EPR logic chooses
locations, and a car trip-duration heuristic (`haversine / car_speed_kmh`) shifts
arrivals and departures off the slot grid. Trip-DITRAS is still available with
`--ditras`. The extension path-depends on the sibling `../skmob2` crate.

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
uv run citybehavex simulate --config configs/gparis_sts_epr.yaml
```

Run the legacy trip-DITRAS path explicitly:

```bash
uv run citybehavex simulate --config configs/gparis_sts_epr.yaml --ditras
```

LLM generation creates 30 weekday and 30 weekend diaries by default. Override
the number, within the supported range of 10 to 30, from the CLI:

```bash
uv run citybehavex simulate --config configs/gparis_sts_epr.yaml --diary-count 20
```

Equivalently, run the package as a Python module:

```bash
uv run python -m citybehavex simulate --config configs/gparis_sts_epr.yaml
```

Generate the comparison report again from the existing data without rerunning
the simulation:

```bash
uv run citybehavex report --config configs/gparis_sts_epr.yaml
```

Paths can also be provided explicitly:

```bash
uv run citybehavex report \
  --synthetic data/gparis_sts_epr_trajectories.parquet \
  --comparison data/gparis_visitation_df.parquet \
  --comparison-label gparis \
  --output data/gparis_sts_epr_comparison.html
```

### Embedding model (ddCRP schedule selection)

Schedule selection embeds each diary with `nomic-embed-text-v2-moe`. With
`embedding.auto_launch: true` (the default) citybehavex starts the server below on
demand and shuts it down afterwards, caching vectors so it rarely reruns. To run it
yourself and reuse it (set `embedding.auto_launch: false` and point
`embedding.base_url` at it), serve the model with vLLM:

```bash
uv run --extra embeddings vllm serve nomic-ai/nomic-embed-text-v2-moe \
  --task embed --trust-remote-code --port 8001
```

The `embeddings` extra (vLLM) is required only for serving; without a GPU set
`embedding.enabled: false` to fall back to identity similarity.

## Updating local skmob2

Rebuild the sibling `../skmob2` Rust extension into this project's `.venv`:

```bash
./scripts/update_local_skmob.sh
```
