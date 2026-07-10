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

import polars as pl
from fkmob.measures.individual.diversity import diversity as _diversity
from fkmob.measures.individual.entropy import trajectory_entropy as _entropy
from fkmob.measures.individual.mobility_profiling import (
    intermittance_and_degree_of_return as _intermittance_and_degree_of_return,
)
from fkmob.measures.individual.regularity import regularity as _regularity

#: Metrics shown in the per-profile box plots.
PROFILE_METRICS = ("regularity", "diversity", "stationarity", "entropy")


def _stationarity(visits: pl.DataFrame) -> pl.DataFrame:
    """Fraction of observed time each user spends stationary (port of
    compute_netmob_stationarity; the simulation-step factor cancels out)."""
    df = visits.with_columns(
        (
            (pl.col("end_timestamp") - pl.col("start_timestamp")).dt.total_seconds() / 60.0
        ).alias("duration_minutes")
    )
    df = df.filter(pl.col("duration_minutes") > 0)
    per_user = df.group_by("uid", maintain_order=True).agg(
        [
            (pl.col("end_timestamp").max() - pl.col("start_timestamp").min())
            .dt.total_seconds()
            .__truediv__(60.0)
            .alias("span"),
            pl.col("duration_minutes").sum().alias("dwell"),
        ]
    )
    return per_user.select(
        "uid",
        pl.when(pl.col("span") == 0.0)
        .then(None)
        .otherwise(pl.col("dwell") / pl.col("span"))
        .alias("stationarity"),
    )


def _cluster_and_label(
    profiles: pl.DataFrame, *, n_clusters: int = 3, random_state: int = 0
) -> pl.DataFrame:
    """Cluster users on [intermittency, degree_of_return] with KMeans and
    label clusters Routiner/Regular/Scouter by descending degree-of-return centroid."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = profiles.select(["intermittency", "degree_of_return"]).to_numpy()
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster = kmeans.fit_predict(scaler.fit_transform(features))
    profiles = profiles.with_columns(pl.Series("cluster", cluster))

    cluster_order = (
        profiles.group_by("cluster")
        .agg(pl.col("degree_of_return").mean())
        .sort("degree_of_return", descending=True)["cluster"]
        .to_list()
    )
    names = ["Routiner", "Regular", "Scouter"]
    mapping = {cluster_id: names[rank] for rank, cluster_id in enumerate(cluster_order)}
    profiles = profiles.with_columns(
        pl.col("cluster").replace_strict(mapping).alias("agent_type")
    )
    return profiles.drop("cluster")


def compute_profiles(
    visits: pl.DataFrame, *, n_clusters: int = 3, random_state: int = 0
) -> pl.DataFrame:
    """Compute per-user mobility-profile metrics and assign profile labels.

    Parameters
    ----------
    visits:
        Stay-level table with columns ``uid``, ``start_timestamp``,
        ``end_timestamp``, ``purpose`` and ``location_id`` (as produced by
        ``reports._visits_for_comparison``).

    Returns
    -------
    polars.DataFrame
        One row per clustered user with columns ``uid``, ``intermittency``,
        ``degree_of_return``, ``regularity``, ``diversity``, ``stationarity``,
        ``entropy`` and ``agent_type``.
    """
    df = visits.with_columns(
        (pl.col("location_id").cast(pl.Utf8) + "_" + pl.col("purpose").cast(pl.Utf8)).alias(
            "location_token"
        )
    ).sort(["uid", "start_timestamp"])

    profiles = _intermittance_and_degree_of_return(
        df, user_id_col="uid", location_id_col="location_token", impute_gaps=True
    ).select(["uid", "intermittency", "degree_of_return"])

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
        profiles = profiles.join(metric_df, on="uid", how="left")

    # KMeans needs finite clustering features; sparse single-visit users may lack them.
    profiles = profiles.drop_nulls(subset=["intermittency", "degree_of_return"])
    if len(profiles) < n_clusters:
        raise ValueError(
            f"need at least {n_clusters} users with finite profiling metrics, got {len(profiles)}"
        )
    return _cluster_and_label(profiles, n_clusters=n_clusters, random_state=random_state)
