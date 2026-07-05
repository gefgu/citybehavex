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
N_PURPOSES = 3

# MTUS HAF top-level activities: (name, description, mu_ln, sigma_ln, eligible_purpose_codes)
# mu_ln / sigma_ln are parameters of LogNormal in ln(hours):
#   duration_hours = exp(mu_ln + sigma_ln * z),  z ~ N(0,1)
#   duration_seconds passed to Rust = duration_hours * 3600
_CATALOG_RAW: list[tuple[str, str, float, float, list[int]]] = [
    ("sleep",    "Sleeping or resting at home",                                  2.08, 0.30, [0]),
    ("eatdrink", "Eating, drinking, coffee, lunch, and meal breaks",             -0.35, 0.45, [0, 1, 2]),
    ("selfcare", "Personal hygiene, grooming, and private care",                 -0.69, 0.50, [0, 2]),
    ("paidwork", "Working at the office or job site",                            2.08, 0.30, [1]),
    ("educatn",  "Studying, attending class, or doing homework",                  1.10, 0.50, [2]),
    ("foodprep", "Cooking and food preparation",                                  0.00, 0.60, [0]),
    ("cleanetc", "Cleaning, laundry, and other domestic work",                    0.00, 0.60, [0]),
    ("maintain", "Household maintenance, repairs, and administrative upkeep",     0.00, 0.60, [0]),
    ("shopserv", "Shopping and personal services",                               -0.29, 0.50, [2]),
    ("garden",   "Gardening and outdoor household work",                          0.41, 0.60, [0]),
    ("petcare",  "Caring for pets and domestic animals",                         -0.69, 0.50, [0]),
    ("eldcare",  "Caring for adults or older household members",                  0.41, 0.70, [0, 2]),
    ("pkidcare", "Physical childcare and supervision",                            0.41, 0.70, [0, 2]),
    ("ikidcare", "Interactive childcare, play, and homework help",                0.41, 0.70, [0, 2]),
    ("religion", "Religious practice, ceremonies, and worship",                   0.69, 0.60, [2]),
    ("volorgwk", "Volunteering, civic, and organizational work",                  0.69, 0.60, [2]),
    ("commute",  "Commuting to and from work or education",                      -0.29, 0.50, [1, 2]),
    ("travel",   "Travel for personal, household, and leisure activities",        -0.29, 0.50, [2]),
    ("sportex",  "Sports, exercise, and gym sessions",                            0.41, 0.40, [2]),
    ("tvradio",  "Watching TV, listening to radio, and passive media",            0.92, 0.60, [0, 2]),
    ("read",     "Reading books, news, and magazines",                            0.41, 0.50, [0, 2]),
    ("compint",  "Computer, internet, gaming, and online leisure",                0.41, 0.50, [0, 2]),
    ("goout",    "Going out to restaurants, cinema, theatre, or events",          0.92, 0.50, [2]),
    ("leisure",  "Social, recreational, and other leisure activities",            0.92, 0.60, [0, 2]),
    ("missing",  "Unclassified or missing diary time; shown in comparisons only", 0.00, 0.50, []),
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
