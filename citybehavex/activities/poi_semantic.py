from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from citybehavex.activities.catalog import build_catalog

_CATEGORY_DIR = Path(__file__).parents[1] / "category"
_POI_CLUSTER_CSV = _CATEGORY_DIR / "overture_poi_semantic_clusters.csv"
_POI_MASK_CSV = _CATEGORY_DIR / "poi_semantic_cluster_activity_mask.csv"
UNKNOWN_SEMANTIC_CLUSTER = "other_mixed"
UNKNOWN_SEMANTIC_CLUSTER_ID = 0


@dataclass(frozen=True)
class PoiSemanticActivityData:
    semantic_clusters: list[str]
    cluster_to_id: dict[str, int]
    category_to_cluster: dict[str, str]
    category_to_cluster_id: dict[str, int]
    mask_starts: np.ndarray
    mask_activities: np.ndarray


def _clean_category(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def load_poi_semantic_cluster_mapping(path: str | Path | None = None) -> dict[str, str]:
    df = pd.read_csv(path or _POI_CLUSTER_CSV)
    required = {"primary_category", "semantic_cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"POI semantic cluster mapping is missing columns: {sorted(missing)}")
    mapping: dict[str, str] = {}
    for row in df.itertuples(index=False):
        category = _clean_category(getattr(row, "primary_category"))
        cluster = getattr(row, "semantic_cluster")
        if category is not None and isinstance(cluster, str) and cluster.strip():
            mapping[category] = cluster.strip()
    return mapping


def semantic_cluster_for_category(
    category: object,
    mapping: Mapping[str, str],
) -> str:
    cleaned = _clean_category(category)
    if cleaned is None:
        return UNKNOWN_SEMANTIC_CLUSTER
    return mapping.get(cleaned, UNKNOWN_SEMANTIC_CLUSTER)


def load_poi_activity_mask(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or _POI_MASK_CSV)
    if "semantic_cluster" not in df.columns:
        raise ValueError("POI activity mask is missing semantic_cluster column")
    return df


def build_poi_semantic_activity_data(
    mapping_path: str | Path | None = None,
    mask_path: str | Path | None = None,
) -> PoiSemanticActivityData:
    category_to_cluster = load_poi_semantic_cluster_mapping(mapping_path)
    mask_df = load_poi_activity_mask(mask_path)
    if UNKNOWN_SEMANTIC_CLUSTER not in set(mask_df["semantic_cluster"]):
        raise ValueError(f"POI activity mask must include {UNKNOWN_SEMANTIC_CLUSTER!r}")

    semantic_clusters = [UNKNOWN_SEMANTIC_CLUSTER]
    semantic_clusters.extend(
        sorted(
            cluster
            for cluster in mask_df["semantic_cluster"].astype(str).unique()
            if cluster != UNKNOWN_SEMANTIC_CLUSTER
        )
    )
    cluster_to_id = {cluster: idx for idx, cluster in enumerate(semantic_clusters)}
    activity_idx = {activity.name: activity.idx for activity in build_catalog()}
    mask_by_cluster = mask_df.set_index("semantic_cluster")

    starts = [0]
    activities: list[int] = []
    for cluster in semantic_clusters:
        row = mask_by_cluster.loc[cluster]
        allowed = [
            activity_idx[name]
            for name in activity_idx
            if name in row.index and bool(row[name])
        ]
        activities.extend(allowed)
        starts.append(len(activities))

    category_to_cluster_id = {
        category: cluster_to_id.get(cluster, UNKNOWN_SEMANTIC_CLUSTER_ID)
        for category, cluster in category_to_cluster.items()
    }
    return PoiSemanticActivityData(
        semantic_clusters=semantic_clusters,
        cluster_to_id=cluster_to_id,
        category_to_cluster=category_to_cluster,
        category_to_cluster_id=category_to_cluster_id,
        mask_starts=np.asarray(starts, dtype=np.int64),
        mask_activities=np.asarray(activities, dtype=np.int64),
    )


def semantic_cluster_ids_for_categories(
    categories: pd.Series,
    data: PoiSemanticActivityData,
) -> np.ndarray:
    return (
        categories.map(lambda value: semantic_cluster_for_category(value, data.category_to_cluster))
        .map(lambda cluster: data.cluster_to_id.get(cluster, UNKNOWN_SEMANTIC_CLUSTER_ID))
        .fillna(UNKNOWN_SEMANTIC_CLUSTER_ID)
        .to_numpy(dtype=np.int64)
    )
