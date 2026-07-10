//! Mirrors `citybehavex/tessellation/config.py::TessellationConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TessellationConfig {
    pub path: Option<String>,
    pub min_lon: Option<f64>,
    pub min_lat: Option<f64>,
    pub max_lon: Option<f64>,
    pub max_lat: Option<f64>,
    pub resolution: i64,
    pub enrich_overture: bool,
    pub overture_release: String,
    pub min_poi_count: i64,
    pub poi_tessellation: bool,
    pub output: String,
    pub relevance_column: String,
}

impl Default for TessellationConfig {
    fn default() -> Self {
        Self {
            path: None,
            min_lon: None,
            min_lat: None,
            max_lon: None,
            max_lat: None,
            resolution: 10,
            enrich_overture: false,
            overture_release: "2026-05-20.0".to_string(),
            min_poi_count: 1,
            poi_tessellation: false,
            output: "tessellation.parquet".to_string(),
            relevance_column: "total_poi_count".to_string(),
        }
    }
}

impl TessellationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        let bbox = [self.min_lon, self.min_lat, self.max_lon, self.max_lat];
        let has_any = bbox.iter().any(|v| v.is_some());
        let has_full = bbox.iter().all(|v| v.is_some());
        if has_any && !has_full {
            anyhow::bail!("bbox requires min_lon, min_lat, max_lon, and max_lat");
        }
        Ok(())
    }
}
