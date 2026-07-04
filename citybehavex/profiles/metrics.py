"""Mobility-profile metrics and clustering for comparison reports.

Computes per-user mobility metrics (intermittency, degree of return, regularity,
diversity, stationarity, entropy) from a stay-level visit table and clusters users
into Routiner / Regular / Scouter profiles (Amichi et al., 2020), so the comparison
report can show the intermittency-vs-degree-of-return scatter and the per-profile
metric box plots.

Ported from agents_transport_netmob/mobility_analysis/measures/individual.py and
adapted to the stay-level schema produced by reports._visits_for_comparison
(columns: uid, start_timestamp, end_timestamp, purpose, location_id).
"""

from __future__ import annotations

import pandas as pd
from skmob2.measures.individual.diversity import diversity as _diversity
from skmob2.measures.individual.entropy import trajectory_entropy as _entropy
from skmob2.measures.individual.mobility_profiling import (
    intermittance_and_degree_of_return as _intermittance_and_degree_of_return,
)
from skmob2.measures.individual.regularity import regularity as _regularity

#: Metrics shown in the per-profile box plots.
PROFILE_METRICS = ("regularity", "diversity", "stationarity", "entropy")


def _stationarity(visits: pd.DataFrame) -> pd.DataFrame:
    """Fraction of observed time each user spends stationary (port of
    compute_netmob_stationarity; the simulation-step factor cancels out)."""
    df = visits.copy()
    df["duration_minutes"] = (
        pd.to_datetime(df["end_timestamp"]) - pd.to_datetime(df["start_timestamp"])
    ).dt.total_seconds() / 60.0
    df = df[df["duration_minutes"] > 0]
    span = (
        df.groupby("uid")["end_timestamp"].max() - df.groupby("uid")["start_timestamp"].min()
    ).dt.total_seconds() / 60.0
    dwell = df.groupby("uid")["duration_minutes"].sum()
    stationarity = (dwell / span.replace(0.0, pd.NA)).rename("stationarity")
    return stationarity.reset_index()


def _cluster_and_label(
    profiles: pd.DataFrame, *, n_clusters: int = 3, random_state: int = 0
) -> pd.DataFrame:
    """Cluster users on [intermittency, degree_of_return] with KMeans and
    label clusters Routiner/Regular/Scouter by descending degree-of-return centroid."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = profiles[["intermittency", "degree_of_return"]].to_numpy()
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = kmeans.fit_predict(scaler.fit_transform(features))

    cluster_order = (
        profiles.groupby("cluster")["degree_of_return"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    names = ["Routiner", "Regular", "Scouter"]
    mapping = {cluster: names[rank] for rank, cluster in enumerate(cluster_order)}
    profiles["agent_type"] = profiles["cluster"].map(mapping)
    return profiles.drop(columns=["cluster"])


def compute_profiles(
    visits: pd.DataFrame, *, n_clusters: int = 3, random_state: int = 0
) -> pd.DataFrame:
    """Compute per-user mobility-profile metrics and assign profile labels.

    Parameters
    ----------
    visits:
        Stay-level table with columns ``uid``, ``start_timestamp``,
        ``end_timestamp``, ``purpose`` and ``location_id`` (as produced by
        ``reports._visits_for_comparison``).

    Returns
    -------
    pandas.DataFrame
        One row per clustered user with columns ``uid``, ``intermittency``,
        ``degree_of_return``, ``regularity``, ``diversity``, ``stationarity``,
        ``entropy`` and ``agent_type``.
    """
    df = visits.copy()
    df["location_token"] = df["location_id"].astype(str) + "_" + df["purpose"].astype(str)
    df = df.sort_values(["uid", "start_timestamp"]).reset_index(drop=True)

    profiles = _intermittance_and_degree_of_return(
        df, user_id_col="uid", location_id_col="location_token", impute_gaps=True
    )[["uid", "intermittency", "degree_of_return"]]

    for metric_df in (
        _regularity(df, user_id_col="uid", location_id_col="location_id", location_type_col="purpose"),
        _diversity(df, user_id_col="uid", location_id_col="location_id", location_type_col="purpose"),
        _entropy(
            df,
            user_id_col="uid",
            location_id_col="location_id",
            location_type_col="purpose",
            normalized=True,
        ),
        _stationarity(df),
    ):
        profiles = profiles.merge(metric_df, on="uid", how="left")

    # KMeans needs finite clustering features; sparse single-visit users may lack them.
    profiles = profiles.dropna(subset=["intermittency", "degree_of_return"]).reset_index(drop=True)
    if len(profiles) < n_clusters:
        raise ValueError(
            f"need at least {n_clusters} users with finite profiling metrics, got {len(profiles)}"
        )
    return _cluster_and_label(profiles, n_clusters=n_clusters, random_state=random_state)
