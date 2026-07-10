//! Mirrors `citybehavex/llm/config.py::LLMConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LLMConfig {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub temperature: f64,
    pub max_tokens: Option<i64>,
    pub timeout_seconds: f64,
    pub retries: i64,
    pub concurrency: i64,
    pub diary_count: i64,
    pub reuse_cache: bool,
    pub cache_dir: String,
    pub prompt_path: Option<String>,
    pub raw_response_path: Option<String>,
    pub validated_diaries_path: Option<String>,
    pub auto_launch: bool,
    pub vllm_port: i64,
    pub vllm_startup_timeout_seconds: f64,
    pub vllm_extra_args: Vec<String>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key: None,
            model: None,
            temperature: 0.4,
            max_tokens: None,
            timeout_seconds: 60.0,
            retries: 3,
            concurrency: 1,
            diary_count: 30,
            reuse_cache: true,
            cache_dir: ".citybehavex/llm_diaries".to_string(),
            prompt_path: None,
            raw_response_path: None,
            validated_diaries_path: None,
            auto_launch: false,
            vllm_port: 8080,
            vllm_startup_timeout_seconds: 600.0,
            vllm_extra_args: Vec::new(),
        }
    }
}

impl LLMConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.concurrency < 1 {
            anyhow::bail!("concurrency must be >= 1");
        }
        if !(10..=50).contains(&self.diary_count) {
            anyhow::bail!("diary_count must be in [10, 50]");
        }
        if self.auto_launch {
            if self.model.is_none() {
                anyhow::bail!("llm model must be provided when auto_launch is enabled");
            }
            return Ok(());
        }
        let present = [&self.base_url, &self.api_key]
            .into_iter()
            .filter(|v| v.is_some())
            .count()
            + usize::from(self.model.is_some());
        if present != 0 && present != 3 {
            anyhow::bail!("llm base_url, api_key, and model must be provided together");
        }
        Ok(())
    }
}
