//! Mirrors `citybehavex/activities/config.py::{ActivityDurationOverride,ActivitiesConfig}`.

use super::catalog;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentBackend {
    None,
    Rerank,
}

impl Default for AlignmentBackend {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ActivityDurationOverride {
    pub mu_ln: Option<f64>,
    pub sigma_ln: Option<f64>,
    pub scale: Option<f64>,
    pub sigma_scale: Option<f64>,
}

impl ActivityDurationOverride {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, v) in [
            ("sigma_ln", self.sigma_ln),
            ("scale", self.scale),
            ("sigma_scale", self.sigma_scale),
        ] {
            if let Some(v) = v {
                if !(v > 0.0) {
                    anyhow::bail!("{name} must be > 0");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ActivitiesConfig {
    pub enabled: bool,
    pub kappa: f64,
    pub temperature: f64,
    pub embed_activities: bool,
    pub alignment_backend: AlignmentBackend,
    pub alignment_base_url: Option<String>,
    pub alignment_model: Option<String>,
    pub alignment_timeout_seconds: f64,
    pub alignment_batch_size: i64,
    pub alignment_cache_path: Option<String>,
    pub alignment_concurrency: i64,
    pub alignment_retries: i64,
    pub alignment_checkpoint_every: i64,
    pub profile_cluster_similarity_threshold: f64,
    pub history_weight: f64,
    pub materialize_travel: bool,
    pub poi_type_choice_enabled: bool,
    pub poi_type_choice_temperature: f64,
    pub poi_type_choice_alpha: f64,
    pub act_dur_scale: f64,
    pub act_dur_sigma_scale: f64,
    pub durations: BTreeMap<String, ActivityDurationOverride>,
}

impl Default for ActivitiesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kappa: 1.0,
            temperature: 0.5,
            embed_activities: false,
            alignment_backend: AlignmentBackend::default(),
            alignment_base_url: None,
            alignment_model: None,
            alignment_timeout_seconds: 120.0,
            alignment_batch_size: 32,
            alignment_cache_path: None,
            alignment_concurrency: 4,
            alignment_retries: 2,
            alignment_checkpoint_every: 20,
            profile_cluster_similarity_threshold: 0.94,
            history_weight: 1.0,
            materialize_travel: true,
            poi_type_choice_enabled: false,
            poi_type_choice_temperature: 0.5,
            poi_type_choice_alpha: 1.0,
            act_dur_scale: 1.0,
            act_dur_sigma_scale: 1.0,
            durations: BTreeMap::new(),
        }
    }
}

impl ActivitiesConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, v) in [
            ("kappa", self.kappa),
            ("temperature", self.temperature),
            ("alignment_timeout_seconds", self.alignment_timeout_seconds),
            ("poi_type_choice_temperature", self.poi_type_choice_temperature),
            ("act_dur_scale", self.act_dur_scale),
            ("act_dur_sigma_scale", self.act_dur_sigma_scale),
        ] {
            if !(v > 0.0) {
                anyhow::bail!("{name} must be > 0");
            }
        }
        if self.alignment_batch_size <= 0 {
            anyhow::bail!("alignment_batch_size must be > 0");
        }
        if self.alignment_concurrency < 1 || self.alignment_retries < 1
            || self.alignment_checkpoint_every < 1
        {
            anyhow::bail!("alignment_concurrency/retries/checkpoint_every must be >= 1");
        }
        if !(-1.0..=1.0).contains(&self.profile_cluster_similarity_threshold) {
            anyhow::bail!("profile_cluster_similarity_threshold must be in [-1, 1]");
        }
        if self.history_weight < 0.0 {
            anyhow::bail!("history_weight must be >= 0");
        }
        if self.poi_type_choice_alpha < 0.0 {
            anyhow::bail!("poi_type_choice_alpha must be >= 0");
        }

        for over in self.durations.values() {
            over.validate()?;
        }
        if !self.durations.is_empty() {
            let known: std::collections::HashSet<&str> = catalog::known_names().collect();
            let unknown: Vec<&str> = self
                .durations
                .keys()
                .map(|s| s.as_str())
                .filter(|name| !known.contains(name))
                .collect();
            if !unknown.is_empty() {
                anyhow::bail!(
                    "activities.durations contains unknown activity name(s): {}",
                    unknown.join(", ")
                );
            }
        }
        Ok(())
    }
}
