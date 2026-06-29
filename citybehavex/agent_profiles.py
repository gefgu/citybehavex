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

from .config import AgentProfilesConfig

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


def _sample_multinomial(weights: list[float], rng: np.random.Generator) -> int:
    """Sample one category index from unnormalized weights."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return int(rng.choice(len(w), p=w))


def _sample_tiles_weighted(
    relevances: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` tile indices weighted by ``relevances``."""
    total = relevances.sum()
    if total <= 0:
        return rng.integers(0, len(relevances), size=n)
    probs = relevances / total
    return rng.choice(len(relevances), size=n, p=probs)


def generate_profiles(
    n: int,
    config: AgentProfilesConfig,
    rng: np.random.Generator,
    tessellation_df: pd.DataFrame,
    relevance_column: str = "total_poi_count",
) -> list[AgentProfile]:
    """Generate ``n`` agent profiles using the distribution config.

    Home tiles are sampled uniformly (any tile can host residents).
    Work tiles are sampled weighted by POI/relevance count (commercial bias).
    """
    n_tiles = len(tessellation_df)
    if n_tiles == 0:
        raise ValueError("tessellation_df is empty — cannot assign home/work tiles")

    # Work tile relevance weights (high POI → commercial → more workplaces)
    if relevance_column in tessellation_df.columns:
        rel_vals = tessellation_df[relevance_column].fillna(0).to_numpy(dtype=float)
        rel_vals = np.where(rel_vals <= 0, 0.1, rel_vals)
    else:
        rel_vals = np.ones(n_tiles, dtype=float)

    # Home tiles: uniform
    home_tiles = rng.integers(0, n_tiles, size=n)
    # Work tiles: relevance-weighted
    work_tiles = _sample_tiles_weighted(rel_vals, n, rng)

    # Gender
    genders = rng.integers(0, 2, size=n)  # 0=female, 1=male

    # Age: Beta(a, b) scaled to [age_min, age_max]
    raw_beta = rng.beta(config.age_beta_a, config.age_beta_b, size=n)
    ages = (raw_beta * (config.age_max - config.age_min) + config.age_min).astype(int)

    # Education, health, household, job — each independently multinomial
    educations = [_sample_multinomial(config.education_weights, rng) for _ in range(n)]
    healths = [_sample_multinomial(config.health_weights, rng) for _ in range(n)]
    households = [_sample_multinomial(config.household_weights, rng) for _ in range(n)]
    jobs = [_sample_multinomial(config.job_weights, rng) for _ in range(n)]

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
