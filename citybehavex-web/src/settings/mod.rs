//! Port of `citybehavex/config/**` (Pydantic, `extra="forbid"` everywhere) to
//! serde structs with `#[serde(deny_unknown_fields, default)]` -- the direct
//! analogue, plus a `validate()` method per struct standing in for Pydantic's
//! `@field_validator`/`@model_validator(mode="after")`, called recursively
//! from `CityBehavExConfig::validate` after deserialization (mirrors
//! `config/io.py::load_config`'s `CityBehavExConfig.model_validate(...)`
//! call, which runs Pydantic's validators as part of construction).
//!
//! Struct field declaration order gives deterministic, order-preserving YAML
//! output from `#[derive(Serialize)]` on rewrite (the PATCH-experiment
//! endpoint, Phase 3), matching Python's `yaml.safe_dump(sort_keys=False)`
//! without needing an `IndexMap`.

pub mod activities;
pub mod catalog;
pub mod diaries;
pub mod embedding;
pub mod llm;
pub mod profiles;
pub mod reports;
pub mod roads;
pub mod schedule;
pub mod simulation;
pub mod social;
pub mod tessellation;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CityBehavExConfig {
    pub tessellation: tessellation::TessellationConfig,
    pub simulation: simulation::SimulationConfig,
    pub road_network: roads::RoadNetworkConfig,
    pub rail_network: roads::RailNetworkConfig,
    pub llm: llm::LLMConfig,
    pub diaries: diaries::DiariesConfig,
    pub embedding: embedding::EmbeddingConfig,
    pub schedule: schedule::ScheduleConfig,
    pub profiles: profiles::AgentProfilesConfig,
    pub activities: activities::ActivitiesConfig,
    pub social: social::SocialNetworkConfig,
    pub comparison: reports::ComparisonConfig,
}

impl Default for CityBehavExConfig {
    fn default() -> Self {
        Self {
            tessellation: Default::default(),
            simulation: Default::default(),
            road_network: Default::default(),
            rail_network: Default::default(),
            llm: Default::default(),
            diaries: Default::default(),
            embedding: Default::default(),
            schedule: Default::default(),
            profiles: Default::default(),
            activities: Default::default(),
            social: Default::default(),
            comparison: Default::default(),
        }
    }
}

impl CityBehavExConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.tessellation.validate()?;
        self.simulation.validate()?;
        self.road_network.validate()?;
        self.rail_network.validate()?;
        self.llm.validate()?;
        self.diaries.validate()?;
        self.embedding.validate()?;
        self.schedule.validate()?;
        self.profiles.validate()?;
        self.activities.validate()?;
        self.social.validate()?;
        self.comparison.validate()?;
        Ok(())
    }
}

/// `$VAR`/`${VAR}` env-var expansion over every string in a parsed YAML
/// value, applied before struct deserialization -- mirrors
/// `config/io.py::_expand_env`. Missing variables are left as literal text
/// (matching `os.path.expandvars`'s behavior of leaving unresolved
/// references alone rather than erroring or substituting empty string).
fn expand_env(value: Value) -> Value {
    match value {
        Value::String(s) => Value::String(expand_env_str(&s)),
        Value::Sequence(items) => Value::Sequence(items.into_iter().map(expand_env).collect()),
        Value::Mapping(map) => Value::Mapping(
            map.into_iter()
                .map(|(k, v)| (expand_env(k), expand_env(v)))
                .collect(),
        ),
        other => other,
    }
}

fn expand_env_str(input: &str) -> String {
    let re = regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
        .expect("static regex");
    re.replace_all(input, |caps: &regex::Captures| {
        let name = caps.get(1).or_else(|| caps.get(2)).unwrap().as_str();
        std::env::var(name).unwrap_or_else(|_| caps.get(0).unwrap().as_str().to_string())
    })
    .into_owned()
}

/// Loads and validates a `configs/*.yaml` file, mirroring
/// `config/io.py::load_config`. `None` path returns the all-defaults config,
/// same as the Python version.
pub fn load_config(path: Option<&Path>) -> anyhow::Result<CityBehavExConfig> {
    let Some(path) = path else {
        return Ok(CityBehavExConfig::default());
    };
    let raw_text = std::fs::read_to_string(path)?;
    let raw: Value = serde_yaml::from_str(&raw_text).unwrap_or(Value::Mapping(Default::default()));
    if !matches!(raw, Value::Mapping(_)) {
        anyhow::bail!("config file must contain a YAML mapping");
    }
    let expanded = expand_env(raw);
    let config: CityBehavExConfig = serde_yaml::from_value(expanded)?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        CityBehavExConfig::default().validate().unwrap();
    }

    #[test]
    fn expand_env_leaves_unresolved_vars_literal() {
        assert_eq!(expand_env_str("$DOES_NOT_EXIST_XYZ/foo"), "$DOES_NOT_EXIST_XYZ/foo");
    }

    #[test]
    fn expand_env_substitutes_present_vars() {
        // SAFETY: single-threaded test process section, no concurrent env access.
        unsafe {
            std::env::set_var("CBX_TEST_VAR_1", "value");
        }
        assert_eq!(expand_env_str("prefix-${CBX_TEST_VAR_1}-suffix"), "prefix-value-suffix");
        assert_eq!(expand_env_str("$CBX_TEST_VAR_1"), "value");
        unsafe {
            std::env::remove_var("CBX_TEST_VAR_1");
        }
    }

    #[test]
    fn unknown_top_level_key_is_rejected() {
        let yaml = "not_a_real_section:\n  foo: 1\n";
        let raw: Value = serde_yaml::from_str(yaml).unwrap();
        let result: Result<CityBehavExConfig, _> = serde_yaml::from_value(raw);
        assert!(result.is_err());
    }

    #[test]
    fn activities_durations_rejects_unknown_activity_name() {
        let yaml = "activities:\n  durations:\n    not_a_real_activity:\n      mu_ln: 1.0\n";
        let raw: Value = serde_yaml::from_str(yaml).unwrap();
        let config: CityBehavExConfig = serde_yaml::from_value(raw).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn gparis_config_loads_and_validates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("configs/gparis_simulation.yaml");
        if !path.exists() {
            return;
        }
        let config = load_config(Some(&path)).expect("gparis config should load and validate");
        assert_eq!(config.simulation.agents, 500);
        assert_eq!(config.simulation.days, 7);
        assert_eq!(config.simulation.relevance_column, "relevance");
        assert_eq!(config.simulation.granularity_minutes, 15);
        assert_eq!(config.simulation.car_speed_kmh, 50.0);
        assert_eq!(config.simulation.rho, 0.6);
        assert_eq!(config.simulation.gamma, 0.21);
        assert_eq!(config.simulation.alpha, 0.2);
        assert_eq!(config.simulation.dt_update_mob_sim_hours, 168.0);
        assert_eq!(config.simulation.gravity_deterrence_exponent, -2.5);
        assert!(config.road_network.enabled);
        assert_eq!(
            config.simulation.output,
            "data/gparis/results/gparis_simulation_core_trajectories.parquet"
        );
    }

    /// Every real config in the repo (not just gparis) must parse and
    /// validate cleanly -- this is the broadest available fidelity check on
    /// `deny_unknown_fields` short of the Phase 11 cross-server parity
    /// harness.
    #[test]
    fn every_repo_config_loads_and_validates() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let configs_dir = repo_root.join("configs");
        if !configs_dir.is_dir() {
            return;
        }
        let mut checked = 0;
        for entry in walk_yaml_files(&configs_dir) {
            checked += 1;
            load_config(Some(&entry))
                .unwrap_or_else(|e| panic!("{} failed to load/validate: {e}", entry.display()));
        }
        assert!(checked > 0, "expected to find at least one config yaml");
    }

    fn walk_yaml_files(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return out;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_yaml_files(&path));
            } else if path.extension().is_some_and(|e| e == "yaml") {
                out.push(path);
            }
        }
        out
    }
}
