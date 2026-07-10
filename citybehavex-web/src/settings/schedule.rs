//! Mirrors `citybehavex/schedules/config.py::ScheduleConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityBackend {
    Embedding,
    AlignmentModel,
}

impl Default for SimilarityBackend {
    fn default() -> Self {
        Self::Embedding
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScheduleConfig {
    pub similarity_backend: SimilarityBackend,
    pub alignment_base_url: Option<String>,
    pub alignment_model: Option<String>,
    pub alignment_timeout_seconds: f64,
    pub alignment_batch_size: i64,
    pub alignment_cache_path: Option<String>,
    pub alignment_concurrency: i64,
    pub alignment_retries: i64,
    pub alignment_checkpoint_every: i64,
    pub temperature_beta_a: f64,
    pub temperature_beta_b: f64,
    pub alpha_beta_a: f64,
    pub alpha_beta_b: f64,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            similarity_backend: SimilarityBackend::default(),
            alignment_base_url: None,
            alignment_model: None,
            alignment_timeout_seconds: 120.0,
            alignment_batch_size: 32,
            alignment_cache_path: None,
            alignment_concurrency: 4,
            alignment_retries: 2,
            alignment_checkpoint_every: 5,
            temperature_beta_a: 2.0,
            temperature_beta_b: 5.0,
            alpha_beta_a: 2.0,
            alpha_beta_b: 5.0,
        }
    }
}

impl ScheduleConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        let positives: [(&str, f64); 7] = [
            ("alignment_timeout_seconds", self.alignment_timeout_seconds),
            ("alignment_batch_size", self.alignment_batch_size as f64),
            ("temperature_beta_a", self.temperature_beta_a),
            ("temperature_beta_b", self.temperature_beta_b),
            ("alpha_beta_a", self.alpha_beta_a),
            ("alpha_beta_b", self.alpha_beta_b),
            ("alignment_checkpoint_every", self.alignment_checkpoint_every as f64),
        ];
        for (name, v) in positives {
            if !(v > 0.0) {
                anyhow::bail!("{name} must be > 0");
            }
        }
        if self.alignment_concurrency < 1 {
            anyhow::bail!("alignment_concurrency must be >= 1");
        }
        if self.alignment_retries < 1 {
            anyhow::bail!("alignment_retries must be >= 1");
        }
        Ok(())
    }
}
