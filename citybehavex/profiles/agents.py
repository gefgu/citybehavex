"""Agent demographic profile generation.

Each agent gets a rich persona (gender, age, education, health, household
composition, job, transport modes, home tile, work tile) that drives:
- which daily schedule it adopts (profile↔schedule ddCRP similarity)
- who its friends are (profile-embedding social graph)
- which micro-activities it chooses (profile↔activity similarity in Rust)

Attributes are sampled independently; a coherence feedback loop is deferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from citybehavex.math import sample_beta_scaled_ints, sample_multinomial_index, sample_weighted_indices
from citybehavex.profiles.config import AgentProfilesConfig

EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------------------
# Category labels (ordered to match config weight lists)
# ---------------------------------------------------------------------------

EDUCATION_LEVELS: list[str] = [
    "no diploma",
    "secondary or less",
    "vocational or technical",
    "bachelor",
    "master or above",
]

HEALTH_LEVELS: list[int] = [1, 2, 3, 4, 5]  # 1=very poor … 5=very good

HOUSEHOLD_TYPES: list[str] = [
    "shared housing",
    "couple with children",
    "couple without children",
    "living with another family member",
    "single parent",
    "living with parents",
    "living alone",
]

ILOSTAT_JOBS: list[str] = [
    "manager",
    "professional",
    "technician or associate professional",
    "clerical support worker",
    "service or sales worker",
    "agricultural or fishery worker",
    "craft or trades worker",
    "machine operator or assembler",
    "elementary worker",
]


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


class AgentProfile(BaseModel):
    """Demographic profile for one simulated agent."""

    model_config = ConfigDict(extra="forbid")

    uid: int
    gender: str  # "male" or "female"
    name: str
    age: int
    education: str
    health: int  # 1–5 Likert scale
    household: str
    job: str
    has_car: bool
    has_bike: bool
    home_tile: int  # index into tessellation DataFrame
    work_tile: int  # index into tessellation DataFrame


# ---------------------------------------------------------------------------
# Narrative templating (the single integration point for downstream embeddings)
# ---------------------------------------------------------------------------

_HEALTH_LABELS = {1: "very poor", 2: "poor", 3: "fair", 4: "good", 5: "very good"}


def profile_to_narrative(profile: AgentProfile) -> str:
    """Return a concise prose description of a profile for embedding.

    This is the single source of truth that all downstream modules embed:
    the ddCRP (schedule similarity), the social graph, and the activity CRP
    all operate on embeddings of this text.
    """
    transport: list[str] = []
    if profile.has_car:
        transport.append("a car")
    if profile.has_bike:
        transport.append("a bike")
    transport_str = (
        f"They own {' and '.join(transport)}."
        if transport
        else "They rely on public transport or walking."
    )
    health_label = _HEALTH_LABELS.get(profile.health, str(profile.health))
    return (
        f"{profile.name} is a {profile.age}-year-old {profile.gender} "
        f"working as a {profile.job}. "
        f"They have {profile.education} level education "
        f"and {health_label} health. "
        f"They live as: {profile.household}. "
        f"{transport_str}"
    )


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------


def _haversine_km(
    lat1: np.ndarray | float,
    lng1: np.ndarray | float,
    lat2: np.ndarray | float,
    lng2: np.ndarray | float,
) -> np.ndarray:
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lng1_rad = np.radians(np.asarray(lng1, dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lng2_rad = np.radians(np.asarray(lng2, dtype=float))
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _work_attractiveness_weights(work_weights: np.ndarray, config: AgentProfilesConfig) -> np.ndarray:
    weights = np.asarray(work_weights, dtype=float)
    return np.log1p(np.maximum(weights, 0.0))


def _distance_friction_weights(dist_km: np.ndarray, config: AgentProfilesConfig) -> np.ndarray:
    dist = np.asarray(dist_km, dtype=float)
    friction = np.exp(-float(config.work_distance_exponential_lambda) * np.maximum(dist, 0.0))
    correction_power = float(config.work_distance_density_correction_power)
    if correction_power > 0:
        correction_dist = np.maximum(dist, config.work_distance_min_km)
        friction = friction / (correction_dist**correction_power)
    return friction


def _sample_conditional_work_tiles(
    home_tiles: np.ndarray,
    work_pool: np.ndarray,
    rel_vals: np.ndarray,
    tessellation_df: pd.DataFrame,
    config: AgentProfilesConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.work_distance_model == "none":
        return work_pool[sample_weighted_indices(rel_vals[work_pool], len(home_tiles), rng)]

    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    if "lat" not in tessellation_df.columns or lng_col not in tessellation_df.columns:
        return work_pool[sample_weighted_indices(rel_vals[work_pool], len(home_tiles), rng)]

    lat = pd.to_numeric(tessellation_df["lat"], errors="coerce").to_numpy(dtype=float)
    lng = pd.to_numeric(tessellation_df[lng_col], errors="coerce").to_numpy(dtype=float)
    work_lat = lat[work_pool]
    work_lng = lng[work_pool]
    base_work_weights = rel_vals[work_pool].astype(float, copy=False)
    adjusted_work_weights = _work_attractiveness_weights(base_work_weights, config)
    sampled = np.empty(len(home_tiles), dtype=np.int64)
    global_fallback = work_pool[sample_weighted_indices(adjusted_work_weights, len(home_tiles), rng)]

    for i, home_tile in enumerate(home_tiles):
        if rng.random() < config.work_from_home_probability:
            sampled[i] = home_tile
            continue

        home_lat = lat[home_tile]
        home_lng = lng[home_tile]
        if not np.isfinite(home_lat) or not np.isfinite(home_lng):
            sampled[i] = global_fallback[i]
            continue

        dist_km = _haversine_km(home_lat, home_lng, work_lat, work_lng)
        finite = np.isfinite(dist_km)
        within = finite & (dist_km <= config.work_distance_max_km)
        candidate_mask = within if within.any() else finite
        if config.work_distance_fallback == "global" and not within.any():
            sampled[i] = global_fallback[i]
            continue

        candidate_weights = adjusted_work_weights[candidate_mask] * _distance_friction_weights(
            dist_km[candidate_mask],
            config,
        )
        if candidate_weights.sum() <= 0:
            sampled[i] = global_fallback[i]
            continue
        choice = sample_weighted_indices(candidate_weights, 1, rng)[0]
        sampled[i] = work_pool[candidate_mask][choice]
    return sampled


def generate_profiles(
    n: int,
    config: AgentProfilesConfig,
    rng: np.random.Generator,
    tessellation_df: pd.DataFrame,
    relevance_column: str = "total_poi_count",
    home_tile_pool: np.ndarray | None = None,
    work_tile_pool: np.ndarray | None = None,
) -> list[AgentProfile]:
    """Generate ``n`` agent profiles using the distribution config.

    Home tiles are sampled from `home_tile_pool` when provided (typically
    synthetic residential anchors appended to the simulator location table),
    otherwise uniformly for legacy/non-augmented runs.
    Work tiles are sampled weighted by POI/relevance count (commercial bias).
    """
    n_tiles = len(tessellation_df)
    if n_tiles == 0:
        raise ValueError("tessellation_df is empty — cannot assign home/work tiles")

    if work_tile_pool is not None:
        work_pool = np.asarray(work_tile_pool, dtype=np.int64)
    elif "purpose" in tessellation_df.columns:
        purpose = tessellation_df["purpose"].fillna("").astype(str).str.upper()
        work_pool = np.flatnonzero(purpose.ne("HOME").to_numpy())
    else:
        work_pool = np.arange(n_tiles, dtype=np.int64)
    if len(work_pool) == 0:
        raise ValueError("work_tile_pool is empty — cannot assign work tiles")

    # Work tile relevance weights (high POI → commercial → more workplaces)
    if relevance_column in tessellation_df.columns:
        rel_vals = tessellation_df[relevance_column].fillna(0).to_numpy(dtype=float)
        rel_vals = np.where(rel_vals <= 0, 0.1, rel_vals)
    else:
        rel_vals = np.ones(n_tiles, dtype=float)

    if home_tile_pool is not None:
        pool = np.asarray(home_tile_pool, dtype=np.int64)
        if len(pool) == 0:
            raise ValueError("home_tile_pool is empty — cannot assign home tiles")
        home_tiles = rng.choice(pool, size=n, replace=len(pool) < n)
    else:
        home_tiles = rng.integers(0, n_tiles, size=n)
    work_tiles = _sample_conditional_work_tiles(
        home_tiles,
        work_pool,
        rel_vals,
        tessellation_df,
        config,
        rng,
    )

    # Gender
    genders = rng.integers(0, 2, size=n)  # 0=female, 1=male

    # Age: Beta(a, b) scaled to [age_min, age_max]
    ages = sample_beta_scaled_ints(
        config.age_beta_a,
        config.age_beta_b,
        config.age_min,
        config.age_max,
        n,
        rng,
    )

    # Education, health, household, job — each independently multinomial
    educations = [sample_multinomial_index(config.education_weights, rng) for _ in range(n)]
    healths = [sample_multinomial_index(config.health_weights, rng) for _ in range(n)]
    households = [sample_multinomial_index(config.household_weights, rng) for _ in range(n)]
    jobs = [sample_multinomial_index(config.job_weights, rng) for _ in range(n)]

    # Transport modes (independent Bernoulli from config probabilities)
    has_car = rng.random(n) < config.car_probability
    has_bike = rng.random(n) < config.bike_probability

    # Names
    male_pool = config.male_names or ["Alex"]
    female_pool = config.female_names or ["Alex"]

    profiles: list[AgentProfile] = []
    for i in range(n):
        is_male = bool(genders[i])
        pool = male_pool if is_male else female_pool
        name = pool[int(rng.integers(0, len(pool)))]
        profiles.append(
            AgentProfile(
                uid=i + 1,
                gender="male" if is_male else "female",
                name=name,
                age=int(ages[i]),
                education=EDUCATION_LEVELS[educations[i]],
                health=HEALTH_LEVELS[healths[i]],
                household=HOUSEHOLD_TYPES[households[i]],
                job=ILOSTAT_JOBS[jobs[i]],
                has_car=bool(has_car[i]),
                has_bike=bool(has_bike[i]),
                home_tile=int(home_tiles[i]),
                work_tile=int(work_tiles[i]),
            )
        )
    return profiles


def load_profiles(path: str, n: int) -> Optional[list[AgentProfile]]:
    """Load hand-authored profiles from a JSON file.

    Returns ``None`` if the file doesn't exist or has fewer than ``n`` entries
    (caller should then fall back to generation).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) < n:
            return None
        return [AgentProfile.model_validate(entry) for entry in raw[:n]]
    except Exception:  # noqa: BLE001
        return None


def profiles_to_frame(profiles: list[AgentProfile]) -> pd.DataFrame:
    """Convert a list of profiles to a tidy DataFrame."""
    return pd.DataFrame([p.model_dump() for p in profiles])
