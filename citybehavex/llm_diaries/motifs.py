from __future__ import annotations

import numpy as np
from skmob_vis.motifs import LITERATURE_MOTIF_PERCENTAGES

from citybehavex.math import sample_multinomial_index

# Literature ordinal (1-17, Schneider et al. 2013) -> distinct location count,
# including HOME. Derived from the packed fkmob motif IDs in
# ``skmob_vis.motifs.LITERATURE_TO_FKMOB_MOTIF_ID`` (node count is
# permutation-invariant, unlike node identity).
MOTIF_LOCATION_COUNTS: dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 4,
    8: 4,
    9: 4,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
    14: 6,
    15: 6,
    16: 6,
    17: 6,
}

# Literature ordinal -> excursion pattern: one entry per home-to-home outing,
# each entry the number of stops in that outing before returning home. Derived
# by parsing the motif SVG assets in ``skmob-vis`` (home is the red-colored
# node) and decomposing the resulting directed graph into home-anchored
# excursions (Hierholzer circuit-finding, with minimal edge-multiplicity
# augmentation for motifs 5 and 9, whose presence-only edge set is not
# already Eulerian-balanced).
MOTIF_EXCURSION_PATTERNS: dict[int, tuple[int, ...]] = {
    1: (),
    2: (1,),
    3: (1, 1),
    4: (2,),
    5: (2, 1),
    6: (2, 1),
    7: (3,),
    8: (1, 1, 1),
    9: (1, 3),
    10: (3, 1),
    11: (4,),
    12: (2, 1, 1),
    13: (2, 2),
    14: (4, 1),
    15: (5,),
    16: (3, 1, 1),
    17: (3, 2),
}


def motif_weights_for_location_count(location_count: int) -> dict[int, float]:
    """Literature percentages of motifs with this many distinct locations.

    Renormalized to sum to 1 over the matching ordinals.
    """
    ordinals = [
        ordinal
        for ordinal, count in MOTIF_LOCATION_COUNTS.items()
        if count == location_count
    ]
    weights = {ordinal: LITERATURE_MOTIF_PERCENTAGES[ordinal] for ordinal in ordinals}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {ordinal: weight / total for ordinal, weight in weights.items()}


def sample_motif(location_count: int, rng: np.random.Generator) -> int | None:
    """Weighted draw of one literature motif ordinal for this location count.

    The Schneider et al. literature motif set only covers up to six distinct
    locations. Higher-count diaries are still valid; they just have no
    literature motif rule to condition on.
    """
    weights = motif_weights_for_location_count(location_count)
    if not weights:
        return None
    ordinals = list(weights.keys())
    index = sample_multinomial_index(list(weights.values()), rng)
    return ordinals[index]


def _describe_stop_count(stops: int) -> str:
    if stops == 1:
        return "a single place (direct round trip)"
    return f"{stops} places in a row before returning home"


def build_motif_rule(pattern: tuple[int, ...]) -> str:
    """Natural-language instruction describing a target excursion pattern.

    Returns ``""`` for an empty pattern (stay-home day; already covered by
    the location-count rule).
    """
    if not pattern:
        return ""

    if len(pattern) == 1:
        stops = pattern[0]
        return (
            "Structure the day as a single outing away from home: leave once, "
            f"visit {stops} place(s) in a row without returning home in between, "
            "then return home for the rest of the day.\n"
        )

    outings = "; ".join(
        f"outing {i} visits {_describe_stop_count(stops)}"
        for i, stops in enumerate(pattern, start=1)
    )
    return (
        f"Structure the day as {len(pattern)} separate outings away from home, "
        f"returning home between each one: {outings}.\n"
    )
