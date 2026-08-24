#!/usr/bin/env python
"""Cache LA road-node snaps in the pandas parquet format CityBehavEx consumes.

This is scenario setup, not a simulator change.  The installed fastmob version
returns the snap vector as a PyArrow array; materialising it here lets the
existing simulator load its normal ``road_network.snap_output`` cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastmob.network import snap_locations_to_graph


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "la"
RESULTS = DATA / "results"
AGENTS = 132_026
RANDOM_STATE = 20210719


def main() -> None:
    tessellation = pd.read_parquet(DATA / "la_poi_tessellation.parquet")
    anchors = pd.read_parquet(DATA / "la_home_anchors_1pct.parquet")
    if len(anchors) != AGENTS:
        raise ValueError(f"expected {AGENTS:,} LA home anchors, found {len(anchors):,}")

    rng = np.random.default_rng(RANDOM_STATE)
    chosen = anchors.iloc[rng.choice(len(anchors), size=AGENTS, replace=False)].reset_index(drop=True)
    homes = pd.DataFrame(index=np.arange(AGENTS), columns=tessellation.columns)
    homes["lat"] = chosen["lat"].to_numpy(dtype=float)
    homes["lng"] = chosen["lng"].to_numpy(dtype=float)
    homes["tile_id"] = [f"home_anchor_{index + 1}" for index in range(AGENTS)]
    homes["category"] = "residential"
    homes["purpose"] = "HOME"
    homes["relevance"] = 1.0
    locations = pd.concat([tessellation, homes], ignore_index=True)

    nodes = pd.read_parquet(RESULTS / "la_road_graph_nodes.parquet")
    snapped = snap_locations_to_graph(locations, nodes, 750.0, lat_col="lat", lng_col="lng")
    values = np.asarray(snapped.to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(values) != len(locations):
        raise AssertionError("road snap count does not match the scenario location table")
    output = RESULTS / "la_road_graph_snap.parquet"
    pd.DataFrame({"road_node": values}).to_parquet(output, index=False)
    print(f"Saved {len(values):,} LA road-node snaps -> {output}")


if __name__ == "__main__":
    main()
