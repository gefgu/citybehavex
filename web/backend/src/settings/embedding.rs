//! Mirrors `citybehavex/embedding/config.py::EmbeddingConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EmbeddingConfig {
    pub enabled: bool,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub task_prefix: String,
    pub dimensions: i64,
    pub timeout_seconds: f64,
    pub auto_launch: bool,
    pub vllm_executable: String,
    pub vllm_port: i64,
    pub vllm_startup_timeout_seconds: f64,
    pub vllm_extra_args: Vec<String>,
    pub cache_dir: String,
    pub cache_path: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "nomic-ai/nomic-embed-text-v2-moe".to_string(),
            base_url: None,
            api_key: None,
            task_prefix: "clustering: ".to_string(),
            dimensions: 768,
            timeout_seconds: 120.0,
            auto_launch: true,
            vllm_executable: "vllm".to_string(),
            vllm_port: 8001,
            vllm_startup_timeout_seconds: 600.0,
            vllm_extra_args: Vec::new(),
            cache_dir: ".citybehavex/embeddings".to_string(),
            cache_path: None,
        }
    }
}

impl EmbeddingConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.dimensions <= 0 {
            anyhow::bail!("dimensions must be > 0");
        }
        Ok(())
    }

    pub fn resolved_cache_path(&self) -> String {
        self.cache_path.clone().unwrap_or_else(|| {
            format!(
                "{}/diary_embeddings.npz",
                self.cache_dir.trim_end_matches('/')
            )
        })
    }
}
