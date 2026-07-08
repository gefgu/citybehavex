from __future__ import annotations

import math

import numpy as np


def lognormal_location_probabilities(
    mu: float,
    sigma: float,
    max_locations: int,
) -> dict[int, float]:
    """Return rounded log-normal probabilities truncated to ``1..max_locations``."""
    from citybehavex.llm_diaries.models import LocationCountDistribution

    distribution = LocationCountDistribution(
        mu=mu,
        sigma=sigma,
        max_locations=max_locations,
    )

    def cdf(value: float) -> float:
        if value <= 0:
            return 0.0
        z = (math.log(value) - distribution.mu) / (
            distribution.sigma * math.sqrt(2.0)
        )
        return 0.5 * (1.0 + math.erf(z))

    probabilities = {
        count: cdf(count + 0.5) - cdf(count - 0.5)
        for count in range(1, distribution.max_locations + 1)
    }
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("location-count distribution has no probability in range")
    return {count: probability / total for count, probability in probabilities.items()}


def allocate_location_counts(
    mu: float,
    sigma: float,
    max_locations: int,
    n: int,
    max_one_location: int | None = None,
) -> list[int]:
    """Allocate ``n`` diaries to a truncated rounded log-normal distribution."""
    if n <= 0:
        return []
    probabilities = lognormal_location_probabilities(mu, sigma, max_locations)
    raw = {count: probability * n for count, probability in probabilities.items()}
    counts = {k: int(value) for k, value in raw.items()}
    assigned = sum(counts.values())
    remainder = sorted(
        probabilities, key=lambda k: (raw[k] - counts[k], probabilities[k]), reverse=True
    )
    for count in remainder[: n - assigned]:
        counts[count] += 1
        assigned += 1
    if max_one_location is not None and max_one_location >= 0 and counts.get(1, 0) > max_one_location:
        excess = counts[1] - max_one_location
        counts[1] = max_one_location
        destinations = [count for count in sorted(counts) if count != 1]
        if not destinations:
            counts[1] += excess
        else:
            total = sum(probabilities[count] for count in destinations)
            raw_extra = {
                count: probabilities[count] / total * excess
                for count in destinations
            }
            extras = {count: int(value) for count, value in raw_extra.items()}
            assigned_extra = sum(extras.values())
            extra_remainder = sorted(
                destinations,
                key=lambda count: (raw_extra[count] - extras[count], probabilities[count]),
                reverse=True,
            )
            for count in extra_remainder[: excess - assigned_extra]:
                extras[count] += 1
            for count, extra in extras.items():
                counts[count] += extra
    out: list[int] = []
    for k in sorted(counts):
        out.extend([k] * counts[k])
    return out


def sample_multinomial_index(weights: list[float], rng: np.random.Generator) -> int:
    """Sample one category index from unnormalized weights."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return int(rng.choice(len(w), p=w))


def sample_weighted_indices(
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` indices weighted by ``weights``; fall back to uniform if empty."""
    total = weights.sum()
    if total <= 0:
        return rng.integers(0, len(weights), size=n)
    probs = weights / total
    return rng.choice(len(weights), size=n, p=probs)


def sample_beta_scaled_ints(
    a: float,
    b: float,
    low: int,
    high: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample beta values scaled to integer range ``[low, high)``."""
    raw = rng.beta(a, b, size=n)
    return (raw * (high - low) + low).astype(int)
