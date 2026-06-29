"""MTUS-based micro-activity catalog for CityBehavEx.

Each agent samples an activity on arrival at a location using a CRP weighted by
profile↔activity cosine similarity (when embeddings are available) or pure popularity.
The sampled duration modifies the departure time, realising early/late exit relative
to the diary-scheduled slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Fixed purpose codes (matches _PURPOSE_CODE in schedule_ddcrp.py and WORK_CODE in Rust)
N_PURPOSES = 7

# MTUS-derived activities: (name, description, mu_ln, sigma_ln, eligible_purpose_codes)
# mu_ln / sigma_ln are parameters of LogNormal in ln(hours):
#   duration_hours = exp(mu_ln + sigma_ln * z),  z ~ N(0,1)
#   duration_seconds passed to Rust = duration_hours * 3600
_CATALOG_RAW: list[tuple[str, str, float, float, list[int]]] = [
    ("sleep",         "Sleeping or resting at home",                   2.08, 0.30, [0]),
    ("personal_care", "Personal hygiene and grooming",                 -0.69, 0.50, [0, 5]),
    ("home_chores",   "Cooking, cleaning, household tasks",             0.00, 0.60, [0]),
    ("childcare",     "Looking after children at home",                 0.41, 0.70, [0]),
    ("paid_work",     "Working at the office or job site",              2.08, 0.30, [1]),
    ("work_break",    "Short coffee or lunch break during work",       -0.69, 0.40, [1, 6]),
    ("study",         "Studying, attending class, or doing homework",   1.10, 0.50, [2]),
    ("shopping",      "Grocery or retail shopping",                    -0.29, 0.50, [3]),
    ("errands",       "Administrative tasks, pick-ups, bank errands",  -0.69, 0.50, [3, 6]),
    ("recreation",    "Outdoor activities, parks, sightseeing",         0.69, 0.60, [4]),
    ("social_visit",  "Meeting friends or family socially",             0.92, 0.60, [4, 0]),
    ("sport",         "Sports, exercise, gym session",                  0.41, 0.40, [4]),
    ("entertainment", "Cinema, theatre, concert, restaurant",           0.92, 0.50, [4]),
    ("healthcare",    "Doctor visit, treatment, pharmacy",              0.00, 0.50, [5]),
    ("voluntary",     "Volunteering or community activities",           0.69, 0.60, [6]),
]

N_ACTIVITIES = len(_CATALOG_RAW)


@dataclass(frozen=True)
class Activity:
    idx: int
    name: str
    description: str
    mu_ln: float
    sigma_ln: float
    eligible_purposes: list[int]


def build_catalog() -> list[Activity]:
    return [
        Activity(
            idx=i, name=name, description=desc,
            mu_ln=mu, sigma_ln=sigma, eligible_purposes=purposes,
        )
        for i, (name, desc, mu, sigma, purposes) in enumerate(_CATALOG_RAW)
    ]


def build_eligibility_csr() -> tuple[np.ndarray, np.ndarray]:
    """Return CSR arrays (purpose_starts, purpose_acts) mapping purpose code → eligible activity indices."""
    from_purpose: list[list[int]] = [[] for _ in range(N_PURPOSES)]
    for i, (_, _, _, _, purposes) in enumerate(_CATALOG_RAW):
        for p in purposes:
            if 0 <= p < N_PURPOSES:
                from_purpose[p].append(i)

    starts = np.zeros(N_PURPOSES + 1, dtype=np.int64)
    acts: list[int] = []
    for p, eligible in enumerate(from_purpose):
        starts[p + 1] = starts[p] + len(eligible)
        acts.extend(eligible)
    return starts, np.asarray(acts, dtype=np.int64)


def activity_descriptions() -> list[str]:
    return [desc for _, desc, _, _, _ in _CATALOG_RAW]


def activity_duration_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_ln, sigma_ln) arrays of shape [N_ACTIVITIES] in ln(hours)."""
    mu = np.array([row[2] for row in _CATALOG_RAW], dtype=np.float64)
    sigma = np.array([row[3] for row in _CATALOG_RAW], dtype=np.float64)
    return mu, sigma
