"""Agent demographic profile generation.

Each agent gets a rich persona (gender, age, education, health, household
composition, job, transport modes, home tile, work tile) that drives:
- which daily schedule it adopts (profile↔schedule SW-CRP similarity)
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
from fastmob.network import haversine_m_batch
from fastmob.utils._common import _factorize_arrow_values
from pydantic import BaseModel, ConfigDict, Field

from citybehavex.math import (
    sample_beta_scaled_ints,
    sample_multinomial_index,
    sample_weighted_indices,
)
from citybehavex.profiles.config import AgentProfilesConfig

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
    car_ownership_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bike_ownership_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    home_tile: int  # index into tessellation DataFrame
    work_tile: int  # index into tessellation DataFrame


class PartialAgentProfile(BaseModel):
    """A user-supplied subset of fields for one generated agent profile.

    ``uid`` identifies the simulated agent.  Every other field is optional:
    non-null values override the generated profile while omitted/null values
    retain their generated value.
    """

    model_config = ConfigDict(extra="forbid")

    uid: int
    gender: str | None = None
    name: str | None = None
    age: int | None = None
    education: str | None = None
    health: int | None = None
    household: str | None = None
    job: str | None = None
    has_car: bool | None = None
    has_bike: bool | None = None
    car_ownership_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bike_ownership_score: float | None = Field(default=None, ge=0.0, le=1.0)
    home_tile: int | str | None = None
    work_tile: int | str | None = None


# ---------------------------------------------------------------------------
# Narrative templating (the single integration point for downstream embeddings)
# ---------------------------------------------------------------------------

_HEALTH_LABELS = {1: "very poor", 2: "poor", 3: "fair", 4: "good", 5: "very good"}


def profile_to_narrative(profile: AgentProfile, *, include_transport: bool = True) -> str:
    """Return a concise prose description of a profile for embedding.

    This is the single source of truth that all downstream modules embed:
    the SW-CRP (schedule similarity), the social graph, and the activity CRP
    all operate on embeddings of this text.
    """
    transport_str = ""
    if include_transport:
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
    parts = [
        f"{profile.name} is a {profile.age}-year-old {profile.gender} "
        f"working as a {profile.job}. ",
        f"They have {profile.education} level education "
        f"and {health_label} health. ",
        f"They live as: {profile.household}. ",
    ]
    if transport_str:
        parts.append(transport_str)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------


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


def _sample_work_tiles(
    n: int,
    work_pool: np.ndarray,
    rel_vals: np.ndarray,
    config: AgentProfilesConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample workplaces from the POI/building-derived work score."""
    return work_pool[
        sample_weighted_indices(
            _work_attractiveness_weights(rel_vals[work_pool], config), n, rng
        )
    ]


def _sample_conditional_home_tiles(
    work_tiles: np.ndarray,
    home_pool: np.ndarray,
    tessellation_df: pd.DataFrame,
    config: AgentProfilesConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose residential HOME anchors conditional on each sampled workplace."""
    if config.work_distance_model == "none":
        sampled = home_pool[rng.integers(len(home_pool), size=len(work_tiles))]
        works_from_home = rng.random(len(work_tiles)) < config.work_from_home_probability
        sampled[works_from_home] = work_tiles[works_from_home]
        return sampled

    lng_col = "lng" if "lng" in tessellation_df.columns else "lon"
    if "lat" not in tessellation_df.columns or lng_col not in tessellation_df.columns:
        return home_pool[rng.integers(len(home_pool), size=len(work_tiles))]

    lat = pd.to_numeric(tessellation_df["lat"], errors="coerce").to_numpy(dtype=float)
    lng = pd.to_numeric(tessellation_df[lng_col], errors="coerce").to_numpy(dtype=float)
    home_lat = lat[home_pool]
    home_lng = lng[home_pool]
    sampled = np.empty(len(work_tiles), dtype=np.int64)
    global_fallback = home_pool[rng.integers(len(home_pool), size=len(work_tiles))]

    for i, work_tile in enumerate(work_tiles):
        if rng.random() < config.work_from_home_probability:
            sampled[i] = work_tile
            continue

        work_lat = lat[work_tile]
        work_lng = lng[work_tile]
        if not np.isfinite(work_lat) or not np.isfinite(work_lng):
            sampled[i] = global_fallback[i]
            continue

        dist_km = haversine_m_batch(
            np.full_like(home_lat, work_lat, dtype=np.float64),
            np.full_like(home_lng, work_lng, dtype=np.float64),
            home_lat,
            home_lng,
        ) / 1000.0
        finite = np.isfinite(dist_km)
        within = finite & (dist_km <= config.work_distance_max_km)
        candidate_mask = within if within.any() else finite
        if config.work_distance_fallback == "global" and not within.any():
            sampled[i] = global_fallback[i]
            continue

        candidate_weights = _distance_friction_weights(dist_km[candidate_mask], config)
        if candidate_weights.sum() <= 0:
            sampled[i] = global_fallback[i]
            continue
        choice = sample_weighted_indices(candidate_weights, 1, rng)[0]
        sampled[i] = home_pool[candidate_mask][choice]
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

    WORK tiles are sampled first from the POI/building-derived relevance score.
    HOME tiles are then sampled from `home_tile_pool` (typically synthetic,
    building-backed residential anchors) using the configured commute prior.
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
        home_pool = np.asarray(home_tile_pool, dtype=np.int64)
        if len(home_pool) == 0:
            raise ValueError("home_tile_pool is empty — cannot assign home tiles")
    else:
        home_pool = np.arange(n_tiles, dtype=np.int64)
    work_tiles = _sample_work_tiles(
        n,
        work_pool,
        rel_vals,
        config,
        rng,
    )
    home_tiles = _sample_conditional_home_tiles(
        work_tiles,
        home_pool,
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


def reroll_profile_demographics(
    profiles: list[AgentProfile],
    indices: np.ndarray | list[int],
    config: AgentProfilesConfig,
    rng: np.random.Generator,
    protected_fields: list[set[str]] | None = None,
) -> list[AgentProfile]:
    """Resample demographic attributes for selected profiles.

    Spatial anchors and transport ownership are intentionally preserved so this
    can be used as a coherence repair step before vehicle ownership alignment.
    Fields present in ``protected_fields`` are also retained for their agent.
    """
    selected = np.asarray(indices, dtype=np.int64)
    if len(selected) == 0:
        return profiles

    n = len(selected)
    genders = rng.integers(0, 2, size=n)
    ages = sample_beta_scaled_ints(
        config.age_beta_a,
        config.age_beta_b,
        config.age_min,
        config.age_max,
        n,
        rng,
    )
    educations = [sample_multinomial_index(config.education_weights, rng) for _ in range(n)]
    healths = [sample_multinomial_index(config.health_weights, rng) for _ in range(n)]
    households = [sample_multinomial_index(config.household_weights, rng) for _ in range(n)]
    jobs = [sample_multinomial_index(config.job_weights, rng) for _ in range(n)]
    male_pool = config.male_names or ["Alex"]
    female_pool = config.female_names or ["Alex"]

    updated = list(profiles)
    for local_idx, profile_idx in enumerate(selected):
        idx = int(profile_idx)
        profile = updated[idx]
        protected = protected_fields[idx] if protected_fields is not None else set()
        is_male = bool(genders[local_idx])
        effective_is_male = profile.gender == "male" if "gender" in protected else is_male
        pool = male_pool if effective_is_male else female_pool
        candidates = {
            "gender": "male" if is_male else "female",
            "name": pool[int(rng.integers(0, len(pool)))],
            "age": int(ages[local_idx]),
            "education": EDUCATION_LEVELS[educations[local_idx]],
            "health": HEALTH_LEVELS[healths[local_idx]],
            "household": HOUSEHOLD_TYPES[households[local_idx]],
            "job": ILOSTAT_JOBS[jobs[local_idx]],
        }
        updated[idx] = profile.model_copy(
            update={field: value for field, value in candidates.items() if field not in protected}
        )
    return updated


def load_profiles(path: str, n: int) -> Optional[list[AgentProfile]]:
    """Load hand-authored profiles from a JSON or parquet file.

    Returns ``None`` if the file doesn't exist or has fewer than ``n`` entries
    (caller should then fall back to generation).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() == ".parquet":
            frame = pd.read_parquet(p)
            if len(frame) < n:
                return None
            raw = frame.head(n).to_dict(orient="records")
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) < n:
            return None
        return [AgentProfile.model_validate(entry) for entry in raw[:n]]
    except Exception:  # noqa: BLE001
        return None


def _profile_override_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("profile overrides JSON must contain a list of records")
    if not all(isinstance(entry, dict) for entry in raw):
        raise ValueError("profile overrides must contain object records")
    return raw


def _null_to_none(value: object) -> object:
    """Normalize pandas' missing scalars without treating containers as null."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    return None if isinstance(missing, (bool, np.bool_)) and missing else value


def load_profile_overrides(path: str, n: int) -> dict[int, dict[str, object]] | None:
    """Load strict, sparse profile overrides keyed by one-based agent ``uid``.

    Missing files retain the historical fallback-to-generation behavior.  A
    present but malformed file is an error, so a requested fixed location can
    never be silently ignored.
    """
    p = Path(path)
    if not p.exists():
        return None

    try:
        records = _profile_override_records(p)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not read profile overrides from {p}: {exc}") from exc

    overrides: dict[int, dict[str, object]] = {}
    for row_number, record in enumerate(records, start=1):
        cleaned = {key: _null_to_none(value) for key, value in record.items()}
        try:
            parsed = PartialAgentProfile.model_validate(cleaned)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid profile override at row {row_number}: {exc}") from exc

        uid = parsed.uid
        if not 1 <= uid <= n:
            raise ValueError(f"profile override uid {uid} must be in 1..{n}")
        if uid in overrides:
            raise ValueError(f"duplicate profile override uid {uid}")
        overrides[uid] = parsed.model_dump(exclude_unset=True, exclude_none=True)
    return overrides


def apply_profile_overrides(
    profiles: list[AgentProfile],
    overrides: dict[int, dict[str, object]],
) -> tuple[list[AgentProfile], list[set[str]]]:
    """Overlay supplied fields and return their per-agent protection masks."""
    updated: list[AgentProfile] = []
    protected_fields: list[set[str]] = []
    for profile in profiles:
        supplied = overrides.get(profile.uid, {})
        protected = set(supplied).difference({"uid"})
        updated.append(profile.model_copy(update={key: supplied[key] for key in protected}))
        protected_fields.append(protected)
    return updated, protected_fields


def resolve_profile_override_tiles(
    overrides: dict[int, dict[str, object]], tessellation_df: pd.DataFrame
) -> dict[int, dict[str, object]]:
    """Resolve integer indices and string ``tile_id`` values to row indices.

    The runtime core addresses locations by row index.  String IDs are
    factorized with fastmob's Rust-backed helper to keep the ID normalization
    path efficient while retaining the tessellation's canonical row order.
    """
    n_tiles = len(tessellation_df)
    resolved = {uid: dict(supplied) for uid, supplied in overrides.items()}
    string_references = [
        value
        for supplied in resolved.values()
        for field in ("home_tile", "work_tile")
        if isinstance(value := supplied.get(field), str)
    ]

    id_to_index: dict[str, int] = {}
    if string_references:
        if "tile_id" not in tessellation_df.columns:
            raise ValueError("string profile location IDs require a tessellation tile_id column")
        raw_ids = tessellation_df["tile_id"]
        if raw_ids.isna().any():
            raise ValueError("tessellation tile_id contains missing values")
        normalized_ids = raw_ids.map(str).to_numpy(dtype=object)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("tessellation tile_id values must be unique for profile ID lookup")
        _, representatives = _factorize_arrow_values(normalized_ids.tolist(), sort=False)
        for representative in np.asarray(representatives, dtype=np.int64):
            id_to_index[normalized_ids[representative]] = int(representative)

    for uid, supplied in resolved.items():
        for field in ("home_tile", "work_tile"):
            if field not in supplied:
                continue
            tile = supplied[field]
            if isinstance(tile, str):
                try:
                    supplied[field] = id_to_index[tile]
                except KeyError as exc:
                    raise ValueError(
                        f"profile override uid {uid} has unknown {field} ID {tile!r}"
                    ) from exc
                continue
            if not isinstance(tile, (int, np.integer)) or isinstance(tile, bool):
                raise ValueError(f"profile override uid {uid} has non-integer {field}: {tile!r}")
            if not 0 <= int(tile) < n_tiles:
                raise ValueError(
                    f"profile override uid {uid} has {field}={tile}; "
                    f"expected an index in 0..{n_tiles - 1}"
                )
            supplied[field] = int(tile)
    return resolved


def validate_profile_override_tiles(overrides: dict[int, dict[str, object]], n_tiles: int) -> None:
    """Ensure explicitly supplied legacy integer indices address the table."""
    for uid, supplied in overrides.items():
        for field in ("home_tile", "work_tile"):
            if field not in supplied:
                continue
            tile = supplied[field]
            if not isinstance(tile, (int, np.integer)) or isinstance(tile, bool):
                raise ValueError(f"profile override uid {uid} has non-integer {field}: {tile!r}")
            if not 0 <= int(tile) < n_tiles:
                raise ValueError(
                    f"profile override uid {uid} has {field}={tile}; "
                    f"expected an index in 0..{n_tiles - 1}"
                )


def profiles_to_frame(profiles: list[AgentProfile]) -> pd.DataFrame:
    """Convert a list of profiles to a tidy DataFrame."""
    return pd.DataFrame([p.model_dump() for p in profiles])
