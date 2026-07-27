from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import typer

from citybehavex.activities import (
    ActivitiesConfig,
    ProfileClusters,
    activity_descriptions,
    activity_duration_arrays,
    available_semantic_cluster_ids,
    build_catalog,
    build_eligibility_csr,
    build_poi_semantic_activity_data,
    score_activity_alignment,
    score_poi_semantic_alignment,
    score_poi_type_alignment,
    semantic_cluster_ids_for_categories,
)
from citybehavex.config import CityBehavExConfig
from citybehavex.embedding import embed_texts
from citybehavex.schedules import DiaryBank

def _configured_activity_duration_arrays(config: ActivitiesConfig) -> tuple[np.ndarray, np.ndarray]:
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    if config.act_dur_scale != 1.0:
        act_dur_mu = act_dur_mu + np.log(config.act_dur_scale)
    if config.act_dur_sigma_scale != 1.0:
        act_dur_sigma = act_dur_sigma * config.act_dur_sigma_scale

    if not config.durations:
        return act_dur_mu, act_dur_sigma

    activity_idx = {activity.name: activity.idx for activity in build_catalog()}
    for name, override in config.durations.items():
        idx = activity_idx[name]
        if override.mu_ln is not None:
            act_dur_mu[idx] = override.mu_ln
        if override.scale is not None:
            act_dur_mu[idx] = act_dur_mu[idx] + np.log(override.scale)
        if override.sigma_ln is not None:
            act_dur_sigma[idx] = override.sigma_ln
        if override.sigma_scale is not None:
            act_dur_sigma[idx] = act_dur_sigma[idx] * override.sigma_scale
    return act_dur_mu, act_dur_sigma


def _build_activity_data(
    config: CityBehavExConfig,
    tessellation_df: Optional[pd.DataFrame] = None,
    bank: Optional[DiaryBank] = None,
    profile_clusters: Optional[ProfileClusters] = None,
    output_path: Optional[str] = None,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Return activity arrays, plus optional contextual alignment tensor."""
    if not config.activities.enabled:
        return None, None, None, None, None, None, None, None, None, None, None, None
    act_dur_mu, act_dur_sigma = _configured_activity_duration_arrays(config.activities)
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    act_embs = None
    if config.activities.embed_activities:
        descriptions = activity_descriptions()
        act_embs = embed_texts(descriptions, config.embedding)
        if act_embs is not None:
            typer.echo(f"Activity embeddings: {act_embs.shape}")
        else:
            typer.echo("Activity embeddings unavailable — using count-only CRP")
    activity_alignment_scores = None
    activity_cluster_labels = None
    poi_semantic_scores = None
    location_semantic_cluster_ids = None
    poi_mask_starts = None
    poi_mask_activities = None
    poi_type_alignment_scores = None
    poi_data = None
    if tessellation_df is not None and "category" in tessellation_df.columns:
        poi_data = build_poi_semantic_activity_data()
        location_semantic_cluster_ids = semantic_cluster_ids_for_categories(
            tessellation_df["category"],
            poi_data,
        )
        poi_mask_starts = poi_data.mask_starts
        poi_mask_activities = poi_data.mask_activities
    if (
        config.activities.alignment_backend == "rerank"
        and bank is not None
        and profile_clusters is not None
    ):
        activity_cluster_labels = profile_clusters.labels
        aligned = score_activity_alignment(
            profile_clusters.narratives,
            bank.diaries,
            config.activities,
        )
        if aligned is not None:
            activity_alignment_scores, _blocks, metadata = aligned
            typer.echo(f"Micro-activity alignment scores: {activity_alignment_scores.shape}")
            if output_path is not None:
                alignment_path = output_path.replace(".parquet", "_activity_alignment.parquet")
                metadata.to_parquet(alignment_path, index=False)
                typer.echo(f"Saved micro-activity alignment scores -> {alignment_path}")
        else:
            typer.echo("Micro-activity alignment scorer unavailable — falling back to activity embeddings")
        if poi_data is not None:
            poi_aligned = score_poi_semantic_alignment(
                profile_clusters.narratives,
                config.activities,
                poi_data,
            )
            if poi_aligned is not None:
                poi_semantic_scores, poi_metadata = poi_aligned
                typer.echo(f"POI semantic activity alignment scores: {poi_semantic_scores.shape}")
                if output_path is not None:
                    poi_alignment_path = output_path.replace(".parquet", "_poi_activity_alignment.parquet")
                    poi_metadata.to_parquet(poi_alignment_path, index=False)
                    typer.echo(f"Saved POI semantic activity alignment scores -> {poi_alignment_path}")
            else:
                typer.echo("POI semantic activity scorer unavailable — falling back to activity embeddings/counts for OTHER")
            if config.activities.poi_type_choice_enabled and location_semantic_cluster_ids is not None:
                poi_type_aligned = score_poi_type_alignment(
                    profile_clusters.narratives,
                    bank.diaries,
                    config.activities,
                    poi_data,
                    available_cluster_ids=available_semantic_cluster_ids(location_semantic_cluster_ids),
                )
                if poi_type_aligned is not None:
                    poi_type_alignment_scores, _blocks, poi_type_metadata = poi_type_aligned
                    typer.echo(f"POI type alignment scores: {poi_type_alignment_scores.shape}")
                    if output_path is not None:
                        poi_type_alignment_path = output_path.replace(".parquet", "_poi_type_alignment.parquet")
                        poi_type_metadata.to_parquet(poi_type_alignment_path, index=False)
                        typer.echo(f"Saved POI type alignment scores -> {poi_type_alignment_path}")
                else:
                    typer.echo("POI type scorer unavailable — using legacy unrestricted OTHER location choice")
    typer.echo(f"Activities enabled: {len(act_dur_mu)} activities, kappa={config.activities.kappa}, T={config.activities.temperature}")
    return (
        act_embs,
        act_dur_mu,
        act_dur_sigma,
        purpose_act_starts,
        purpose_acts,
        activity_alignment_scores,
        activity_cluster_labels,
        poi_semantic_scores,
        location_semantic_cluster_ids,
        poi_mask_starts,
        poi_mask_activities,
        poi_type_alignment_scores,
    )
