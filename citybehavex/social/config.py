from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SocialNetworkConfig(BaseModel):
    """Tunables for the agent friendship graph.

    Friends are drawn from agents who share an agent's home or work H3 cell,
    weighted by profile-embedding similarity; each agent's target friend
    count (degree) is an independent draw from a log-normal distribution.

    This is deliberately not fit to the raw co-presence degree measured from
    the shanghai/yjmob observed datasets (mean ~1,070 and ~5,300
    respectively) -- that graph counts anyone ever sharing a venue/grid-cell
    on the same day, a much denser notion than a friendship tie. We also
    tried thresholding co-presence edges into "friends" via the RECAST
    method (comparing observed edge persistence/topological overlap against
    a degree-preserving random graph's distribution at a random-chance
    probability p_rnd, per Fournet & Barrat-style random-graph baselines):
    it gave a plausible result for shanghai (~200m venues; friendship
    degree mean ~103 at p_rnd=1e-3) but was degenerate for yjmob2 (fixed
    500m grid cells; co-presence is so dense there that even the random
    baseline saturates at max persistence, so no threshold separates
    "social" from "random"). Since neither dataset's spatial resolution is
    adjustable, we fall back to a simple, directly-configured log-normal
    instead of a data-fit one.
    """

    model_config = ConfigDict(extra="forbid")

    home_h3_resolution: int = Field(default=7, ge=0, le=15)
    work_h3_resolution: int = Field(default=7, ge=0, le=15)

    # Per-agent target degree ~ lognormal(degree_mu_ln, degree_sigma_ln),
    # rounded and clipped to [0, max_degree]. Defaults give a mean degree of
    # 10 (mu_ln = ln(10) - sigma_ln**2 / 2).
    degree_mu_ln: float = Field(default=2.1776, ge=0)
    degree_sigma_ln: float = Field(default=0.5, gt=0)
    max_degree: int = Field(default=200, gt=0)

    # Softmax temperature over cosine similarity when sampling friends from
    # an agent's home/work candidate pool: w_k = exp(cosine_k / T).
    similarity_temperature: float = Field(default=0.3, gt=0)

    # If an agent's home+work colocation pool exceeds this, uniformly
    # subsample down to it before computing similarities (bounds cost the
    # same way build_profile_social_graph's cluster-size cap does).
    max_candidate_pool: int = Field(default=2000, gt=0)

    # How many H3 rings to search outward from home/work when the local
    # cell has no other agents, before accepting an empty pool.
    max_ring_expansion: int = Field(default=2, ge=0)

    # Fallback k-NN profile-similarity graph, used when an agent lacks a
    # profile embedding or a real home tile (see build_profile_social_graph).
    social_graph_k: int = Field(default=20, gt=0)
    profile_graph_exact_threshold: int = Field(default=10_000, gt=0)
