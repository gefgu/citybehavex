from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from citybehavex.activities import cluster_profile_embeddings
from citybehavex.config import CityBehavExConfig
from citybehavex.embedding import embed_profiles
from citybehavex.llm.config import LLMConfig
from citybehavex.llm_diaries import DiariesConfig, DiaryValidationError
from citybehavex.profiles import (
    AgentProfile,
    AgentProfilesConfig,
    calibrate_demographic_weights,
    expand_coherence_scores,
    expand_vehicle_scores,
    generate_profiles,
    load_profiles,
    profile_to_narrative,
    profiles_to_frame,
    reroll_profile_demographics,
    score_profile_coherence_alignment,
    score_vehicle_ownership_alignment,
)

def resolve_calibrated_profiles_config(
    pc: AgentProfilesConfig, llm_config: LLMConfig, diaries_config: DiariesConfig
) -> AgentProfilesConfig:
    """Apply LLM-calibrated demographic weights to ``pc`` when ``llm_override`` is set.

    Falls back to ``pc``'s existing (default or configured) weights, with a
    warning, if calibration fails after its internal retries — a transient LLM
    outage should not abort the whole simulation run.
    """
    if not pc.llm_override:
        return pc
    city_profile = (
        diaries_config.city_profile
        or diaries_config.city_profile_weekday
        or diaries_config.city_profile_weekend
    )
    try:
        weights = calibrate_demographic_weights(llm_config, city_profile=city_profile)
    except DiaryValidationError as exc:
        typer.echo(f"Warning: LLM weight calibration failed, using default weights: {exc}")
        return pc
    if weights is None:
        return pc
    typer.echo("Calibrated demographic weights via LLM from city profile")
    return pc.model_copy(update=weights)


def _profile_city_context(diaries_config: DiariesConfig) -> str | None:
    return (
        diaries_config.city_profile
        or diaries_config.city_profile_weekday
        or diaries_config.city_profile_weekend
    )


def _apply_vehicle_ownership_alignment(
    profiles: list[AgentProfile],
    config: CityBehavExConfig,
) -> list[AgentProfile]:
    pc = config.profiles
    if (
        not profiles
        or pc.ownership_alignment_backend != "rerank"
        or not pc.ownership_alignment_base_url
    ):
        return profiles

    neutral_narratives = [profile_to_narrative(p, include_transport=False) for p in profiles]
    profile_embeddings = embed_profiles(neutral_narratives, config.embedding)
    if profile_embeddings is not None:
        clusters = cluster_profile_embeddings(
            neutral_narratives,
            profile_embeddings,
            pc.ownership_profile_cluster_similarity_threshold,
        )
    else:
        typer.echo(
            "Profile embeddings unavailable for ownership alignment — "
            "scoring profiles individually"
        )
        clusters = cluster_profile_embeddings(neutral_narratives, None, 1.0)

    scored = score_vehicle_ownership_alignment(
        clusters.narratives,
        pc,
        city_profile=_profile_city_context(config.diaries),
    )
    if scored is None:
        typer.echo("Vehicle ownership alignment scorer unavailable — keeping configured random ownership")
        return profiles

    cluster_scores, metadata = scored
    agent_scores = expand_vehicle_scores(cluster_scores, clusters)
    rng = np.random.default_rng(config.simulation.random_state + 17)
    car_score = np.clip(agent_scores[:, 0], 0.0, 1.0)
    bike_score = np.clip(agent_scores[:, 1], 0.0, 1.0)
    has_car = rng.random(len(profiles)) < car_score
    has_bike = rng.random(len(profiles)) < bike_score
    updated = [
        profile.model_copy(
            update={
                "has_car": bool(has_car[idx]),
                "has_bike": bool(has_bike[idx]),
                "car_ownership_score": float(car_score[idx]),
                "bike_ownership_score": float(bike_score[idx]),
            }
        )
        for idx, profile in enumerate(profiles)
    ]
    typer.echo(
        "Vehicle ownership alignment scores: "
        f"{len(clusters.narratives)} profile clusters for {len(profiles)} profiles; "
        f"car mean={float(car_score.mean()):.3f}, bike mean={float(bike_score.mean()):.3f}"
    )
    if pc.output:
        alignment_path = Path(pc.output).with_name(
            f"{Path(pc.output).stem}_ownership_alignment.parquet"
        )
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(alignment_path, index=False)
        typer.echo(f"Saved vehicle ownership alignment scores -> {alignment_path}")
    return updated


def _coherence_rerun_indices(scores: np.ndarray, threshold: float, rng: np.random.Generator) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    forced = clipped < float(threshold)
    probabilistic = rng.random(len(clipped)) < (1.0 - clipped)
    return np.flatnonzero(forced | probabilistic)


def _apply_profile_coherence_alignment(
    profiles: list[AgentProfile],
    config: CityBehavExConfig,
) -> list[AgentProfile]:
    pc = config.profiles
    if (
        not profiles
        or pc.coherence_alignment_backend != "rerank"
        or not pc.coherence_alignment_base_url
        or pc.coherence_rerun_rounds <= 0
    ):
        return profiles

    city_profile = _profile_city_context(config.diaries)
    rng = np.random.default_rng(config.simulation.random_state + 29)
    repaired = list(profiles)
    metadata_frames: list[pd.DataFrame] = []

    for round_idx in range(1, pc.coherence_rerun_rounds + 1):
        narratives = [profile_to_narrative(p, include_transport=False) for p in repaired]
        profile_embeddings = embed_profiles(narratives, config.embedding)
        if profile_embeddings is not None:
            clusters = cluster_profile_embeddings(
                narratives,
                profile_embeddings,
                pc.coherence_profile_cluster_similarity_threshold,
            )
        else:
            typer.echo(
                "Profile embeddings unavailable for coherence alignment — "
                "scoring profiles individually"
            )
            clusters = cluster_profile_embeddings(narratives, None, 1.0)

        scored = score_profile_coherence_alignment(
            clusters.narratives,
            pc,
            city_profile=city_profile,
        )
        if scored is None:
            typer.echo("Profile coherence scorer unavailable — keeping generated profiles")
            return repaired

        cluster_scores, metadata = scored
        agent_scores = expand_coherence_scores(cluster_scores, clusters)
        rerun_indices = _coherence_rerun_indices(
            agent_scores,
            pc.coherence_rerun_threshold,
            rng,
        )
        metadata = metadata.copy()
        metadata["round"] = round_idx
        metadata_frames.append(metadata)
        typer.echo(
            "Profile coherence alignment round "
            f"{round_idx}/{pc.coherence_rerun_rounds}: "
            f"{len(clusters.narratives)} clusters for {len(repaired)} profiles; "
            f"mean={float(agent_scores.mean()):.3f}; rerun={len(rerun_indices)}"
        )
        if len(rerun_indices) == 0:
            break
        repaired = reroll_profile_demographics(repaired, rerun_indices, pc, rng)

    if metadata_frames and pc.output:
        alignment_path = Path(pc.output).with_name(
            f"{Path(pc.output).stem}_coherence_alignment.parquet"
        )
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(metadata_frames, ignore_index=True).to_parquet(alignment_path, index=False)
        typer.echo(f"Saved profile coherence alignment scores -> {alignment_path}")
    return repaired


def maybe_build_profiles(
    config: CityBehavExConfig,
    tessellation_df: pd.DataFrame,
    relevance_column: str,
    home_tile_pool: np.ndarray | None = None,
) -> Optional[list[AgentProfile]]:
    """Generate or load agent profiles when ``profiles.enabled`` is true."""
    if not config.profiles.enabled:
        return None
    n = config.simulation.agents
    pc = config.profiles
    if pc.profiles_path:
        loaded = load_profiles(pc.profiles_path, n)
        if loaded is not None:
            typer.echo(f"Loaded {len(loaded)} agent profiles from {pc.profiles_path}")
            loaded = _apply_profile_coherence_alignment(loaded, config)
            loaded = _apply_vehicle_ownership_alignment(loaded, config)
            if pc.output:
                out = Path(pc.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                profiles_to_frame(loaded).to_parquet(str(out), index=False)
                typer.echo(f"Saved agent profiles -> {pc.output}")
            return loaded
        typer.echo(f"Warning: profiles_path {pc.profiles_path!r} not usable — generating")
    pc = resolve_calibrated_profiles_config(pc, config.llm, config.diaries)
    config = config.model_copy(update={"profiles": pc})
    rng = np.random.default_rng(config.simulation.random_state)
    profiles = generate_profiles(n, pc, rng, tessellation_df, relevance_column, home_tile_pool=home_tile_pool)
    profiles = _apply_profile_coherence_alignment(profiles, config)
    profiles = _apply_vehicle_ownership_alignment(profiles, config)
    typer.echo(f"Generated {len(profiles)} agent profiles")
    if pc.output:
        out = Path(pc.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        profiles_to_frame(profiles).to_parquet(str(out), index=False)
        typer.echo(f"Saved agent profiles -> {pc.output}")
    return profiles
