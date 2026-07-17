"""Mobility-profile metrics and clustering for comparison reports.

The underlying per-user metrics (intermittency, degree of return,
regularity, diversity, entropy, stationarity) and their 2-feature KMeans
clustering now live in ``fastmob.measures.individual.compute_profiles``
(migrated from this module, which was itself ported from
agents_transport_netmob/mobility_analysis/measures/individual.py). This
module stays only to adapt fastmob's output to citybehavex's own
Title-case ``"agent_type"`` convention ("Routiner"/"Regular"/"Scouter"),
which existing citybehavex/web report code depends on, rather than
fastmob's lowercase-plural ``"profile"`` convention
(``exploration_profiling``'s "routiners"/"regulars"/"scouters").

Column contract for ``compute_profiles``' input: ``uid``,
``start_timestamp``, ``end_timestamp``, ``purpose``, ``location_id`` (as
produced by ``reports._visits_for_comparison``).
"""

from __future__ import annotations

import narwhals as nw
import polars as pl
from fastmob.measures.individual import compute_profiles as _fastmob_compute_profiles
from fastmob.measures.individual.profile_classification import (
    _cluster_and_label as _fastmob_cluster_and_label,
)

#: Metrics shown in the per-profile box plots.
PROFILE_METRICS = ("regularity", "diversity", "stationarity", "entropy")

_PROFILE_TO_AGENT_TYPE = {"routiners": "Routiner", "regulars": "Regular", "scouters": "Scouter"}


def _relabel_agent_type(result: nw.DataFrame) -> nw.DataFrame:
    return result.rename({"profile": "agent_type"}).with_columns(
        nw.col("agent_type").replace_strict(_PROFILE_TO_AGENT_TYPE).alias("agent_type")
    )


def _cluster_and_label(
    profiles: pl.DataFrame, *, n_clusters: int = 3, random_state: int = 0
) -> pl.DataFrame:
    """Cluster users on [intermittency, degree_of_return] with KMeans and
    label clusters Routiner/Regular/Scouter by descending degree-of-return
    centroid -- thin wrapper over fastmob's clustering, remapped from its
    lowercase-plural convention to citybehavex's own Title-case one."""
    nw_profiles = nw.from_native(profiles, eager_only=True)
    result = _fastmob_cluster_and_label(
        nw_profiles, user_id_col="uid", n_clusters=n_clusters, random_state=random_state
    )
    return _relabel_agent_type(result).to_native()


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
    result = _fastmob_compute_profiles(
        visits,
        user_id_col="uid",
        location_id_col="location_id",
        start_col="start_timestamp",
        end_col="end_timestamp",
        purpose_col="purpose",
        n_clusters=n_clusters,
        random_state=random_state,
    )
    nw_result = nw.from_native(result, eager_only=True)
    return _relabel_agent_type(nw_result).to_native()
