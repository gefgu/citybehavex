from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import typer

from citybehavex.activities import ProfileClusters, cluster_profile_embeddings, expand_cluster_scores
from citybehavex.config import CityBehavExConfig
from citybehavex.embedding import embed_profiles
from citybehavex.llm_diaries import DiaryBatch
from citybehavex.profiles import AgentProfile, profile_to_narrative
from citybehavex.schedules import DiaryBank, SwCrpAgentInfo, build_diary_bank, build_sw_crp_diary, score_alignment_matrix
from citybehavex.utils import ProgressReporter

_PROGRESS_INTERVAL_SECONDS = 5.0

def _build_schedule(
    config: CityBehavExConfig,
    diary_batches: dict[str, DiaryBatch],
    start_date: pd.Timestamp,
    profiles: Optional[list[AgentProfile]] = None,
) -> tuple[DiaryBank, tuple, np.ndarray, Optional[np.ndarray], SwCrpAgentInfo, Optional[ProfileClusters]]:
    """Build the diary bank and run profile-driven CRP schedule selection.

    Returns (bank, diary_arrays, chosen, profile_embeddings, crp_info).
    profile_embeddings is None when embeddings are disabled or unavailable.
    """
    bank = build_diary_bank(
        diary_batches,
        config.embedding,
        config.simulation.granularity_minutes,
    )
    counts = Counter(bank.day_type.tolist())
    typer.echo(
        f"SW-CRP schedule bank: {len(bank.diaries)} diaries "
        f"({', '.join(f'{n} {t}' for t, n in counts.items())}), "
        f"embeddings={'on' if bank.embedded else 'off (popularity CRP, no profile similarity)'}"
    )

    profile_embeddings = None
    narratives = None
    profile_clusters = None
    if profiles is not None:
        narratives = [profile_to_narrative(p) for p in profiles]
        if config.embedding.enabled:
            typer.echo(
                f"Profile embeddings: embedding {len(narratives)} profile narratives "
                f"(requires an embedding server reachable at "
                f"{config.embedding.base_url or '<auto-launch: ' + config.embedding.model + '>'}) ..."
            )
            profile_embeddings = embed_profiles(narratives, config.embedding)
            if profile_embeddings is None:
                raise RuntimeError(
                    "Profile embeddings are required for profile-driven schedule "
                    "selection (profile clustering + macro-schedule alignment) but "
                    "could not be computed. Check that the embedding server is "
                    f"reachable (embedding.base_url={config.embedding.base_url!r}, "
                    f"auto_launch={config.embedding.auto_launch}) and, if auto-launched, "
                    "that the GPU has enough free memory for it -- see the error above "
                    "and the vllm_embed.log next to embedding.cache_dir "
                    f"({config.embedding.cache_dir}) for details. To intentionally run "
                    "without profile similarity, set embedding.enabled: false."
                )
            typer.echo(f"Profile embeddings: {profile_embeddings.shape}")
            typer.echo("Profile alignment clusters: clustering profile embeddings ...")
            profile_clusters = cluster_profile_embeddings(
                narratives,
                profile_embeddings,
                config.activities.profile_cluster_similarity_threshold,
            )
            typer.echo(
                f"Profile alignment clusters: {len(profile_clusters.narratives)} "
                f"for {len(narratives)} profiles"
            )
        else:
            typer.echo(
                "Embeddings disabled (embedding.enabled: false) — using popularity "
                "CRP without profile similarity."
            )
            profile_clusters = cluster_profile_embeddings(narratives, None, 1.0)

    agent_diary_sim = None
    if (
        profiles is not None
        and narratives is not None
        and config.schedule.similarity_backend == "alignment_model"
    ):
        scoring_narratives = profile_clusters.narratives if profile_clusters is not None else narratives
        typer.echo(
            f"Macro-schedule alignment: scoring {len(scoring_narratives)} profile rows "
            f"x {len(bank.diaries)} diaries ..."
        )
        alignment_progress = ProgressReporter(
            "Macro-schedule alignment",
            len(scoring_narratives),
            "rows",
            rate_precision=2,
            min_interval_seconds=_PROGRESS_INTERVAL_SECONDS,
            emit=typer.echo,
        )

        def _report_alignment_progress(done: int, total: int, elapsed: float) -> None:
            alignment_progress.total = total
            alignment_progress.report(done, elapsed=elapsed)

        scored = score_alignment_matrix(
            scoring_narratives,
            bank.diaries,
            config.schedule,
            progress_callback=_report_alignment_progress,
        )
        if scored is not None and profile_clusters is not None:
            agent_diary_sim = expand_cluster_scores(scored, profile_clusters.labels)
        else:
            agent_diary_sim = scored
        if agent_diary_sim is not None:
            typer.echo(f"Macro-schedule alignment scores: {agent_diary_sim.shape}")
        else:
            typer.echo("Alignment scorer unavailable — falling back to embedding cosine")

    day_types = [
        config.diaries.resolve_day_type((start_date + pd.Timedelta(days=d)).date())
        for d in range(config.simulation.days)
    ]
    typer.echo(
        f"SW-CRP schedule selection: assigning diaries for "
        f"{config.simulation.agents} agents x {config.simulation.days} days ..."
    )
    sw_crp_started = time.perf_counter()
    sw_crp_progress = ProgressReporter(
        "SW-CRP schedule selection",
        config.simulation.agents,
        "agents",
        rate_precision=1,
        min_interval_seconds=_PROGRESS_INTERVAL_SECONDS,
        emit=typer.echo,
        started=sw_crp_started,
    )

    def _report_sw_crp_progress(done: int, total: int) -> None:
        sw_crp_progress.total = total
        sw_crp_progress.report(done)

    diary_arrays, chosen, crp_info = build_sw_crp_diary(
        bank,
        start_date,
        config.simulation.days,
        day_types,
        config.simulation.agents,
        config.simulation.random_state,
        config.schedule,
        profile_embeddings=profile_embeddings,
        agent_diary_sim=agent_diary_sim,
        progress_callback=_report_sw_crp_progress,
    )
    return bank, diary_arrays, chosen, profile_embeddings, crp_info, profile_clusters


def _save_crp_artifact(
    path: str,
    bank: DiaryBank,
    chosen: np.ndarray,
    crp_info: SwCrpAgentInfo,
) -> None:
    """Persist per-(agent, diary) SW-CRP state next to the trajectory output.

    ``build_sw_crp_diary`` computes T_a/alpha_a/similarity/usage-counts and then
    throws them away once the diary picks are baked into ``diary_arrays`` — the
    web UI's diary-selection debug panel needs them to reconstruct "what would
    this agent pick next", so they're written out in long form (one row per
    agent x bank diary) alongside the run.
    """
    n_agents, _days = chosen.shape
    K = len(bank.diaries)
    usage_counts = np.stack([np.bincount(chosen[a], minlength=K) for a in range(n_agents)])

    diary_ids = np.array([d.diary_id for d in bank.diaries])
    df = pd.DataFrame(
        {
            "agent": np.repeat(np.arange(n_agents, dtype=np.int64), K),
            "diary_id": np.tile(diary_ids, n_agents),
            "day_type": np.tile(bank.day_type, n_agents),
            "sim": crp_info.agent_diary_sim.reshape(-1),
            "usage_count": usage_counts.reshape(-1),
            "T_a": np.repeat(crp_info.T_per_agent, K),
            "alpha_a": np.repeat(crp_info.alpha_per_agent, K),
        }
    )
    df.to_parquet(path, index=False)
