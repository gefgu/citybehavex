from __future__ import annotations

import pandas as pd

from citybehavex.profiles.agents import load_profiles
from citybehavex.profiles.metrics import _cluster_and_label


def test_cluster_and_label_ranks_kmeans_clusters_by_degree_of_return() -> None:
    """Unit test for the GaussianMixture -> KMeans swap: clusters must still
    group similar [intermittency, degree_of_return] points together and be
    labeled Routiner/Regular/Scouter in descending degree_of_return order."""
    profiles = pd.DataFrame(
        {
            "uid": [
                "routiner_0", "routiner_1", "routiner_2",
                "regular_0", "regular_1", "regular_2",
                "scouter_0", "scouter_1", "scouter_2",
            ],
            "intermittency": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, 9.0, 9.1, 8.9],
            "degree_of_return": [0.9, 0.95, 0.85, 0.5, 0.55, 0.45, 0.1, 0.15, 0.05],
        }
    )

    labeled = _cluster_and_label(profiles, n_clusters=3, random_state=0)
    labels = labeled.set_index("uid")["agent_type"]

    assert len({labels[f"routiner_{i}"] for i in range(3)}) == 1
    assert len({labels[f"regular_{i}"] for i in range(3)}) == 1
    assert len({labels[f"scouter_{i}"] for i in range(3)}) == 1

    names_order = ["Routiner", "Regular", "Scouter"]
    routiner_label = labels["routiner_0"]
    regular_label = labels["regular_0"]
    scouter_label = labels["scouter_0"]
    assert names_order.index(routiner_label) < names_order.index(regular_label) < names_order.index(scouter_label)


def test_load_profiles_rejects_partial_parquet_profile_artifact(tmp_path) -> None:
    path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "uid": [1],
            "gender": ["female"],
            "name": ["Ana"],
            "age": [30],
            "education": ["bachelor"],
            "health": [4],
            "household": ["single"],
            "job": ["professional"],
            "has_car": [True],
            "has_bike": [False],
            "home_tile": [10],
            "work_tile": [12],
        }
    ).to_parquet(path, index=False)

    assert load_profiles(str(path), 2) is None
