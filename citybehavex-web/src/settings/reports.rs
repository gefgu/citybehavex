//! Mirrors `citybehavex/reports/config.py` (`NetworkValidationConfig`,
//! `TransportSpatialConfig`, `EvaluationAdaptationConfig`, `ComparisonConfig`)
//! and the `ALL_REPORT_SECTIONS` constant from `citybehavex/reports/comparison.py`
//! that `ComparisonConfig.sections`'s validator checks against.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// `ACTIVITY_JSD_SECTIONS | {"cpc", "stvd", "micro_activity", "mobility_laws"}`
/// from `citybehavex/reports/comparison.py`.
pub const ALL_REPORT_SECTIONS: &[&str] = &[
    "activity_jsd",
    "activity_comparison",
    "motifs",
    "mobility_profiles",
    "cpc",
    "stvd",
    "micro_activity",
    "mobility_laws",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeWindow {
    Day,
}

impl Default for TimeWindow {
    fn default() -> Self {
        Self::Day
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocationMode {
    Auto,
    LocationCol,
    H3,
}

impl Default for LocationMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NetworkValidationConfig {
    pub enabled: bool,
    pub observed_enabled: bool,
    pub synthetic_enabled: bool,
    pub time_window: TimeWindow,
    pub location_mode: LocationMode,
    pub location_col: Option<String>,
    pub h3_resolution: i64,
    pub max_group_size: i64,
    pub random_seed: i64,
}

impl Default for NetworkValidationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            observed_enabled: false,
            synthetic_enabled: true,
            time_window: TimeWindow::default(),
            location_mode: LocationMode::default(),
            location_col: None,
            h3_resolution: 9,
            max_group_size: 200,
            random_seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TransportSpatialConfig {
    pub enabled: bool,
    pub observed_enabled: bool,
    pub synthetic_moving_path: Option<String>,
    pub uid_col: Option<String>,
    pub datetime_col: Option<String>,
    pub lat_col: Option<String>,
    pub lng_col: Option<String>,
    pub transport_col: Option<String>,
    pub mode_map: HashMap<String, String>,
}

impl Default for TransportSpatialConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            observed_enabled: false,
            synthetic_moving_path: None,
            uid_col: None,
            datetime_col: None,
            lat_col: None,
            lng_col: None,
            transport_col: None,
            mode_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationAdaptationMode {
    Auto,
    Force,
    Off,
}

impl Default for EvaluationAdaptationMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EvaluationAdaptationConfig {
    pub mode: EvaluationAdaptationMode,
    pub location_col: Option<String>,
    pub h3_resolution: i64,
}

impl Default for EvaluationAdaptationConfig {
    fn default() -> Self {
        Self {
            mode: EvaluationAdaptationMode::default(),
            location_col: None,
            h3_resolution: 10,
        }
    }
}

impl EvaluationAdaptationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0..=15).contains(&self.h3_resolution) {
            anyhow::bail!("h3_resolution must be in [0, 15]");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ComparisonConfig {
    pub path: Option<String>,
    pub label: String,
    pub time_use_path: Option<String>,
    pub time_use_label: String,
    pub time_use_country: Option<String>,
    pub time_use_survey: Option<i64>,
    pub time_use_weight_col: String,
    pub json_output: Option<String>,
    pub sections: Option<Vec<String>>,
    pub road_network_distance: bool,
    pub evaluation_adaptation: EvaluationAdaptationConfig,
    pub network_validation: NetworkValidationConfig,
    pub transport_spatial: TransportSpatialConfig,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            path: None,
            label: "observed".to_string(),
            time_use_path: None,
            time_use_label: "time-use".to_string(),
            time_use_country: None,
            time_use_survey: None,
            time_use_weight_col: "propwt".to_string(),
            json_output: None,
            sections: None,
            road_network_distance: true,
            evaluation_adaptation: EvaluationAdaptationConfig::default(),
            network_validation: NetworkValidationConfig::default(),
            transport_spatial: TransportSpatialConfig::default(),
        }
    }
}

impl ComparisonConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.evaluation_adaptation.validate()?;
        if !(0..=15).contains(&self.network_validation.h3_resolution) {
            anyhow::bail!("network_validation.h3_resolution must be in [0, 15]");
        }
        if let Some(sections) = &self.sections {
            let unknown: Vec<&str> = sections
                .iter()
                .map(|s| s.as_str())
                .filter(|s| !ALL_REPORT_SECTIONS.contains(s))
                .collect();
            if !unknown.is_empty() {
                let mut valid = ALL_REPORT_SECTIONS.to_vec();
                valid.sort_unstable();
                anyhow::bail!(
                    "Unknown comparison report section(s): {:?}. Valid sections: {:?}",
                    unknown,
                    valid
                );
            }
        }
        Ok(())
    }
}
