# CityBehavEx

CityBehavEx ships a small Rust extension (`citybehavex._core`, built with maturin)
that implements a trip-duration-aware DITRAS: agents follow a sub-hourly Markov
schedule (5/15-min slots, weekday/weekend chains) while a car trip-duration
heuristic (`haversine / car_speed_kmh`) shifts arrivals and departures off the slot
grid. The extension path-depends on the sibling `../skmob2` crate.

## Building

The project uses the **maturin** build backend. After cloning (or after any change
to the Rust sources), build the extension into the project `.venv`:

```bash
./scripts/update_local_skmob.sh        # if ../skmob2 Rust changed
./scripts/update_local_citybehavex.sh  # builds citybehavex._core
```

`scripts/update_local_citybehavex.sh` compiles `citybehavex-py` (depending on
`../skmob2/skmob2-core` with `default-features = false`, so it avoids the C/SIMD
crates) and installs the package editable. Rust edits are **not** hot-reloaded —
rerun the script after changing any `.rs` file.

## Running

Run the configured Greater Paris simulation from the repository root:

```bash
uv run citybehavex simulate --config configs/gparis_ditras.yaml
```

Equivalently, run the package as a Python module:

```bash
uv run python -m citybehavex simulate --config configs/gparis_ditras.yaml
```

Generate the comparison report again from the existing data without rerunning
the simulation:

```bash
uv run citybehavex report --config configs/gparis_ditras.yaml
```

Paths can also be provided explicitly:

```bash
uv run citybehavex report \
  --synthetic data/gparis_ditras_trajectories.parquet \
  --comparison data/gparis_visitation_df.parquet \
  --comparison-label gparis \
  --output data/gparis_ditras_comparison.html
```

## Updating local skmob2

Rebuild the sibling `../skmob2` Rust extension into this project's `.venv`:

```bash
./scripts/update_local_skmob.sh
```
