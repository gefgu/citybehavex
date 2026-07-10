//! Mirrors `citybehavex/social/config.py::SocialNetworkConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SocialNetworkConfig {
    pub home_h3_resolution: i64,
    pub work_h3_resolution: i64,
    pub degree_mu_ln: f64,
    pub degree_sigma_ln: f64,
    pub max_degree: i64,
    pub similarity_temperature: f64,
    pub max_candidate_pool: i64,
    pub max_ring_expansion: i64,
    pub social_graph_k: i64,
    pub profile_graph_exact_threshold: i64,
    pub dynamic_friendships_enabled: bool,
    pub friendship_update_interval_hours: f64,
    pub encounter_window_hours: f64,
    pub regularity_threshold: f64,
    pub topological_overlap_threshold: f64,
    pub recast_random_baseline_samples: i64,
    pub recast_random_chance_probability: f64,
    pub strength_initial: f64,
    pub strength_growth_mu_ln: f64,
    pub strength_growth_sigma_ln: f64,
    pub strength_decay_rate: f64,
    pub max_dynamic_degree: i64,
    pub max_colocation_group_size: i64,
}

impl Default for SocialNetworkConfig {
    fn default() -> Self {
        Self {
            home_h3_resolution: 7,
            work_h3_resolution: 7,
            degree_mu_ln: 2.1776,
            degree_sigma_ln: 0.5,
            max_degree: 200,
            similarity_temperature: 0.3,
            max_candidate_pool: 2000,
            max_ring_expansion: 2,
            social_graph_k: 20,
            profile_graph_exact_threshold: 10_000,
            dynamic_friendships_enabled: true,
            friendship_update_interval_hours: 24.0,
            encounter_window_hours: 24.0 * 7.0,
            regularity_threshold: 0.3,
            topological_overlap_threshold: 0.05,
            recast_random_baseline_samples: 256,
            recast_random_chance_probability: 1.0e-3,
            strength_initial: 0.1,
            strength_growth_mu_ln: -2.3,
            strength_growth_sigma_ln: 0.5,
            strength_decay_rate: 0.05,
            max_dynamic_degree: 200,
            max_colocation_group_size: 50,
        }
    }
}

impl SocialNetworkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0..=15).contains(&self.home_h3_resolution) || !(0..=15).contains(&self.work_h3_resolution)
        {
            anyhow::bail!("home/work_h3_resolution must be in [0, 15]");
        }
        if self.degree_mu_ln < 0.0 {
            anyhow::bail!("degree_mu_ln must be >= 0");
        }
        if !(self.degree_sigma_ln > 0.0) {
            anyhow::bail!("degree_sigma_ln must be > 0");
        }
        if self.max_degree <= 0 {
            anyhow::bail!("max_degree must be > 0");
        }
        if !(self.similarity_temperature > 0.0) {
            anyhow::bail!("similarity_temperature must be > 0");
        }
        if self.max_candidate_pool <= 0 {
            anyhow::bail!("max_candidate_pool must be > 0");
        }
        if self.max_ring_expansion < 0 {
            anyhow::bail!("max_ring_expansion must be >= 0");
        }
        if self.social_graph_k <= 0 {
            anyhow::bail!("social_graph_k must be > 0");
        }
        if self.profile_graph_exact_threshold <= 0 {
            anyhow::bail!("profile_graph_exact_threshold must be > 0");
        }
        if !(self.friendship_update_interval_hours > 0.0) {
            anyhow::bail!("friendship_update_interval_hours must be > 0");
        }
        if !(self.encounter_window_hours > 0.0) {
            anyhow::bail!("encounter_window_hours must be > 0");
        }
        if !(0.0..=1.0).contains(&self.regularity_threshold) {
            anyhow::bail!("regularity_threshold must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.topological_overlap_threshold) {
            anyhow::bail!("topological_overlap_threshold must be in [0, 1]");
        }
        if self.recast_random_baseline_samples < 0 {
            anyhow::bail!("recast_random_baseline_samples must be >= 0");
        }
        if !(self.recast_random_chance_probability > 0.0
            && self.recast_random_chance_probability <= 1.0)
        {
            anyhow::bail!("recast_random_chance_probability must be in (0, 1]");
        }
        if !(self.strength_initial > 0.0) {
            anyhow::bail!("strength_initial must be > 0");
        }
        if !(self.strength_growth_sigma_ln > 0.0) {
            anyhow::bail!("strength_growth_sigma_ln must be > 0");
        }
        if !(0.0..=1.0).contains(&self.strength_decay_rate) {
            anyhow::bail!("strength_decay_rate must be in [0, 1]");
        }
        if self.max_dynamic_degree <= 0 {
            anyhow::bail!("max_dynamic_degree must be > 0");
        }
        if self.max_colocation_group_size < 2 {
            anyhow::bail!("max_colocation_group_size must be >= 2");
        }
        Ok(())
    }
}
