//! Discover experiments from `configs/*.yaml` and resolve their runs.
//! Mirrors `web/backend/app/experiments.py`.
//!
//! Each YAML config is one experiment. Simulation outputs are timestamp-
//! stamped at write time (`_YYYYMMDDTHHMMSS` before the extension), so the
//! concrete runs are found by globbing the stem of `simulation.output`
//! rather than trusting the literal path.

use crate::config::{configs_dir, repo_root};
use crate::datasource::{RunSummary, run_summary};
use crate::settings::{self, CityBehavExConfig};
use regex::Regex;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::UNIX_EPOCH;

static STAMP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^_(\d{8}T\d{6})$").expect("static regex"));

#[derive(Debug, thiserror::Error)]
pub enum ExperimentError {
    #[error("unknown experiment {0:?}")]
    NotFound(String),
    #[error("unknown run {0:?}")]
    RunNotFound(String),
    #[error("{0}")]
    Mutation(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn resolve(path_str: Option<&str>) -> Option<PathBuf> {
    let path_str = path_str?;
    if path_str.is_empty() {
        return None;
    }
    let p = PathBuf::from(path_str);
    Some(if p.is_absolute() { p } else { repo_root().join(p) })
}

fn display_path(path: Option<&Path>) -> Option<String> {
    let path = path?;
    match path.strip_prefix(repo_root()) {
        Ok(rel) => Some(rel.display().to_string()),
        Err(_) => Some(path.display().to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct Run {
    pub run_id: String,
    pub path: PathBuf,
    pub mtime: f64,
}

impl Run {
    fn sibling(&self, suffix: &str) -> PathBuf {
        let stem = self.path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = self.path.extension().map(|e| e.to_string_lossy().to_string());
        let filename = match ext {
            Some(ext) => format!("{stem}{suffix}.{ext}"),
            None => format!("{stem}{suffix}"),
        };
        self.path.with_file_name(filename)
    }

    pub fn encounters_path(&self) -> PathBuf {
        self.sibling("_encounters")
    }
    pub fn moving_path(&self) -> PathBuf {
        self.sibling("_moving")
    }
    pub fn activities_path(&self) -> PathBuf {
        self.sibling("_activities")
    }
    pub fn crp_path(&self) -> PathBuf {
        self.sibling("_crp")
    }
    pub fn social_network_path(&self) -> PathBuf {
        let stem = self.path.file_stem().unwrap_or_default().to_string_lossy();
        self.path.with_file_name(format!("{stem}_social_network.json"))
    }

    pub fn to_json(&self, with_summary: bool) -> RunJson {
        let (summary, summary_error) = if with_summary {
            match run_summary(&self.path) {
                Ok(s) => (Some(s), None),
                Err(e) => (None, Some(e.to_string())),
            }
        } else {
            (None, None)
        };
        RunJson {
            run_id: self.run_id.clone(),
            path: display_path(Some(&self.path)),
            mtime: self.mtime,
            summary,
            summary_error,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunJson {
    pub run_id: String,
    pub path: Option<String>,
    pub mtime: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<RunSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_error: Option<String>,
}

/// All parquet runs for a `simulation.output` stem, newest first. Excludes
/// the `*_encounters`/`*_moving`/`*_crp` sidecar siblings.
fn discover_runs(output_path: Option<&Path>) -> Vec<Run> {
    let Some(output_path) = output_path else {
        return Vec::new();
    };
    let Some(parent) = output_path.parent() else {
        return Vec::new();
    };
    if !parent.is_dir() {
        return Vec::new();
    }
    let stem = output_path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let suffix = output_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();

    let mut runs = Vec::new();
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let candidate = entry.path();
        if !candidate.is_file() {
            continue;
        }
        let filename = candidate.file_name().unwrap_or_default().to_string_lossy().to_string();
        if !filename.starts_with(&stem) || !filename.ends_with(&suffix) {
            continue;
        }
        let name = candidate.file_stem().unwrap_or_default().to_string_lossy().to_string();
        if name.ends_with("_encounters") || name.ends_with("_moving") || name.ends_with("_crp") {
            continue;
        }
        let run_id = if name == stem {
            "base".to_string()
        } else {
            let name_suffix = &name[stem.len()..];
            match STAMP_RE.captures(name_suffix) {
                Some(caps) => caps.get(1).unwrap().as_str().to_string(),
                None => continue,
            }
        };
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        runs.push(Run { run_id, path: candidate, mtime });
    }
    runs.sort_by(|a, b| b.mtime.partial_cmp(&a.mtime).unwrap_or(std::cmp::Ordering::Equal));
    runs
}

#[derive(Debug, Clone, Serialize)]
pub struct SpecialDayJson {
    pub name: String,
    pub start_date: String,
    pub end_date: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamsJson {
    pub agents: i64,
    pub days: i64,
    pub start_date: Option<String>,
    pub granularity_minutes: i64,
    pub car_speed_kmh: f64,
    pub social_graph_k: i64,
    pub rho: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub dt_update_mob_sim_hours: f64,
    pub indipendency_window_hours: f64,
}

pub struct Experiment {
    pub id: String,
    pub config_path: PathBuf,
    pub label: String,
    pub synthetic_output: Option<PathBuf>,
    pub observed_path: Option<PathBuf>,
    pub time_use_path: Option<PathBuf>,
    pub time_use_label: String,
    pub time_use_country: Option<String>,
    pub time_use_survey: Option<i64>,
    pub time_use_weight_col: String,
    pub profiles_enabled: bool,
    pub profiles_output: Option<PathBuf>,
    pub profiles_path: Option<PathBuf>,
    pub road_nodes_path: Option<PathBuf>,
    pub road_edges_path: Option<PathBuf>,
    pub network_validation_config: settings::reports::NetworkValidationConfig,
    pub transport_spatial_config: settings::reports::TransportSpatialConfig,
    pub evaluation_adaptation_config: settings::reports::EvaluationAdaptationConfig,
    pub params: ParamsJson,
    pub special_days: Vec<SpecialDayJson>,
    pub runs: Vec<Run>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExperimentJson {
    pub id: String,
    pub config: Option<String>,
    pub label: String,
    pub simulation_output: Option<String>,
    pub observed_path: Option<String>,
    pub observed_exists: bool,
    pub time_use_path: Option<String>,
    pub time_use_exists: bool,
    pub time_use_label: String,
    pub time_use_country: Option<String>,
    pub time_use_survey: Option<i64>,
    pub time_use_weight_col: String,
    pub network_validation: settings::reports::NetworkValidationConfig,
    pub transport_spatial: settings::reports::TransportSpatialConfig,
    pub evaluation_adaptation: settings::reports::EvaluationAdaptationConfig,
    pub profiles_enabled: bool,
    pub profiles_output: Option<String>,
    pub profiles_path: Option<String>,
    pub profiles_exists: bool,
    pub road_network_available: bool,
    pub params: ParamsJson,
    pub special_days: Vec<SpecialDayJson>,
    pub runs: Vec<RunJson>,
}

impl Experiment {
    pub fn to_json(&self, with_summary: bool) -> ExperimentJson {
        ExperimentJson {
            id: self.id.clone(),
            config: display_path(Some(&self.config_path)),
            label: self.label.clone(),
            simulation_output: display_path(self.synthetic_output.as_deref()),
            observed_path: display_path(self.observed_path.as_deref()),
            observed_exists: self.observed_path.as_ref().is_some_and(|p| p.exists()),
            time_use_path: display_path(self.time_use_path.as_deref()),
            time_use_exists: self.time_use_path.as_ref().is_some_and(|p| p.exists()),
            time_use_label: self.time_use_label.clone(),
            time_use_country: self.time_use_country.clone(),
            time_use_survey: self.time_use_survey,
            time_use_weight_col: self.time_use_weight_col.clone(),
            network_validation: self.network_validation_config.clone(),
            transport_spatial: self.transport_spatial_config.clone(),
            evaluation_adaptation: self.evaluation_adaptation_config.clone(),
            profiles_enabled: self.profiles_enabled,
            profiles_output: display_path(self.profiles_output.as_deref()),
            profiles_path: display_path(self.profiles_path.as_deref()),
            profiles_exists: self.profiles_path.as_ref().is_some_and(|p| p.exists()),
            road_network_available: self.road_nodes_path.as_ref().is_some_and(|p| p.exists())
                && self.road_edges_path.as_ref().is_some_and(|p| p.exists()),
            params: ParamsJson {
                agents: self.params.agents,
                days: self.params.days,
                start_date: self.params.start_date.clone(),
                granularity_minutes: self.params.granularity_minutes,
                car_speed_kmh: self.params.car_speed_kmh,
                social_graph_k: self.params.social_graph_k,
                rho: self.params.rho,
                gamma: self.params.gamma,
                alpha: self.params.alpha,
                dt_update_mob_sim_hours: self.params.dt_update_mob_sim_hours,
                indipendency_window_hours: self.params.indipendency_window_hours,
            },
            special_days: self.special_days.clone(),
            runs: self.runs.iter().map(|r| r.to_json(with_summary)).collect(),
        }
    }

    pub fn run(&self, run_id: Option<&str>) -> Option<&Run> {
        match run_id {
            None => self.runs.first(),
            Some(run_id) => self.runs.iter().find(|r| r.run_id == run_id),
        }
    }
}

fn load_experiment(config_path: &Path) -> anyhow::Result<Experiment> {
    let cfg = settings::load_config(Some(config_path))?;
    let synthetic_output = resolve(Some(&cfg.simulation.output));
    let observed_path = resolve(cfg.comparison.path.as_deref());
    let time_use_path = resolve(cfg.comparison.time_use_path.as_deref());
    let profiles_output = resolve(Some(&cfg.profiles.output));
    let profiles_path = if cfg.profiles.enabled {
        profiles_output.clone()
    } else {
        None
    };
    let road_distance_enabled = cfg.road_network.enabled && cfg.comparison.road_network_distance;
    let road_nodes_path = if road_distance_enabled {
        resolve(Some(&cfg.road_network.nodes_output))
    } else {
        None
    };
    let road_edges_path = if road_distance_enabled {
        resolve(Some(&cfg.road_network.edges_output))
    } else {
        None
    };
    let params = ParamsJson {
        agents: cfg.simulation.agents,
        days: cfg.simulation.days,
        start_date: cfg.simulation.start_date.clone(),
        granularity_minutes: cfg.simulation.granularity_minutes,
        car_speed_kmh: cfg.simulation.car_speed_kmh,
        social_graph_k: cfg.social.social_graph_k,
        rho: cfg.simulation.rho,
        gamma: cfg.simulation.gamma,
        alpha: cfg.simulation.alpha,
        dt_update_mob_sim_hours: cfg.simulation.dt_update_mob_sim_hours,
        indipendency_window_hours: cfg.simulation.indipendency_window_hours,
    };
    let special_days = cfg
        .diaries
        .special_days
        .iter()
        .map(|sd| SpecialDayJson {
            name: sd.name.clone(),
            start_date: sd.start_date.clone(),
            end_date: sd.end_date.clone(),
        })
        .collect();

    let id = config_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    Ok(Experiment {
        id,
        config_path: config_path.to_path_buf(),
        label: cfg.comparison.label.clone(),
        runs: discover_runs(synthetic_output.as_deref()),
        synthetic_output,
        observed_path,
        time_use_path,
        time_use_label: cfg.comparison.time_use_label.clone(),
        time_use_country: cfg.comparison.time_use_country.clone(),
        time_use_survey: cfg.comparison.time_use_survey,
        time_use_weight_col: cfg.comparison.time_use_weight_col.clone(),
        profiles_enabled: cfg.profiles.enabled,
        profiles_output,
        profiles_path,
        road_nodes_path,
        road_edges_path,
        network_validation_config: cfg.comparison.network_validation.clone(),
        transport_spatial_config: cfg.comparison.transport_spatial.clone(),
        evaluation_adaptation_config: cfg.comparison.evaluation_adaptation.clone(),
        params,
        special_days,
    })
}

pub fn list_experiments() -> Vec<Experiment> {
    let dir = configs_dir();
    if !dir.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| match load_experiment(&p) {
            Ok(exp) => Some(exp),
            Err(e) => {
                tracing::warn!(path = %p.display(), error = %e, "failed to load experiment config");
                None
            }
        })
        .collect()
}

pub fn get_experiment(exp_id: &str) -> Option<Experiment> {
    let config_path = configs_dir().join(format!("{exp_id}.yaml"));
    if !config_path.is_file() {
        return None;
    }
    load_experiment(&config_path).ok()
}

fn read_yaml_mapping(path: &Path) -> Result<serde_yaml::Mapping, ExperimentError> {
    let text = std::fs::read_to_string(path)?;
    let raw: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| ExperimentError::Mutation(format!("invalid YAML: {e}")))?;
    match raw {
        serde_yaml::Value::Mapping(m) => Ok(m),
        serde_yaml::Value::Null => Ok(serde_yaml::Mapping::new()),
        _ => Err(ExperimentError::Mutation(
            "experiment config must contain a YAML mapping".to_string(),
        )),
    }
}

fn section<'a>(
    raw: &'a mut serde_yaml::Mapping,
    name: &str,
) -> Result<&'a mut serde_yaml::Mapping, ExperimentError> {
    let key = serde_yaml::Value::String(name.to_string());
    let entry = raw
        .entry(key)
        .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    entry.as_mapping_mut().ok_or_else(|| {
        ExperimentError::Mutation(format!("{name:?} config section must be a mapping"))
    })
}

fn set_field(map: &mut serde_yaml::Mapping, key: &str, value: serde_yaml::Value) {
    map.insert(serde_yaml::Value::String(key.to_string()), value);
}

/// Would this string, written as a bare (unquoted) YAML plain scalar, be
/// re-read back as something other than a string? PyYAML's default resolver
/// implicitly types plain scalars as bool/null/int/float/**date** (a
/// YAML-1.1/PyYAML-specific extension, not core YAML) -- `start_date:
/// 2026-01-01` written unquoted round-trips through `yaml.safe_load` as a
/// `datetime.date`, not `str`, which then fails Pydantic's `str` field
/// validation when the Python backend re-reads a config this Rust backend
/// wrote. `serde_yaml`'s emitter doesn't replicate PyYAML's implicit-typing
/// avoidance (it only quotes for lexical reasons, e.g. special characters),
/// so this has to be done explicitly before handing values to it.
fn yaml_scalar_would_lose_string_type(s: &str) -> bool {
    if s.is_empty() {
        return true; // bare empty scalar reads back as null
    }
    static NULL_WORDS: &[&str] = &["~", "null", "Null", "NULL"];
    static BOOL_WORDS: &[&str] = &[
        "y", "Y", "yes", "Yes", "YES", "n", "N", "no", "No", "NO", "true", "True", "TRUE",
        "false", "False", "FALSE", "on", "On", "ON", "off", "Off", "OFF",
    ];
    if NULL_WORDS.contains(&s) || BOOL_WORDS.contains(&s) {
        return true;
    }
    static INT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[-+]?[0-9]+$").unwrap());
    static FLOAT_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$").unwrap()
    });
    // PyYAML's timestamp resolver: a bare `YYYY-MM-DD` date, optionally
    // followed by a time component -- only the date prefix matters here
    // since `start_date` values are always plain dates in practice.
    static DATE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]").unwrap());
    INT_RE.is_match(s) || FLOAT_RE.is_match(s) || DATE_RE.is_match(s)
}

/// `Option<&String>` -> `serde_yaml::Value::String`/`Value::Null`. Values
/// that would lose their string-ness if left unquoted (see
/// `yaml_scalar_would_lose_string_type`) are flagged via `ambiguous_out` so
/// the caller can force-quote them in a post-processing pass over the
/// rendered text -- `serde_yaml::Value` has no public API to request quoted
/// scalar style, and its `Tagged` variant (tried first) turned out to emit
/// non-standard tag syntax (`!tag:yaml.org,2002:str value` / `!str value`)
/// that PyYAML's `SafeLoader` can't parse at all, which is strictly worse
/// than the unquoted-misparse problem it was meant to fix.
fn yaml_string_value(
    key: &str,
    value: Option<&String>,
    ambiguous_out: &mut Vec<(String, String)>,
) -> serde_yaml::Value {
    match value {
        None => serde_yaml::Value::Null,
        Some(s) => {
            if yaml_scalar_would_lose_string_type(s) {
                ambiguous_out.push((key.to_string(), s.clone()));
            }
            serde_yaml::Value::String(s.clone())
        }
    }
}

/// Rewrites `key: <bare-ambiguous-value>` lines to `key: '<escaped-value>'`
/// (single-quoted, YAML's own convention for a plain-string scalar) for the
/// specific keys/values the caller flagged as needing it. A substring
/// replace is safe here (not a general YAML text editor) because the
/// candidate values are exactly the literal strings this function's caller
/// just asked `serde_yaml` to emit moments ago -- there's no ambiguity about
/// which occurrence to quote.
fn force_quote_ambiguous_scalars(text: &str, pairs: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (key, value) in pairs {
        let bare = format!("{key}: {value}\n");
        if out.contains(&bare) {
            let quoted = format!("{key}: '{}'\n", value.replace('\'', "''"));
            out = out.replacen(&bare, &quoted, 1);
        }
    }
    out
}

/// `Option<Option<T>>` + `with = "::serde_with::rust::double_option"` on every
/// field distinguishes "field absent from the JSON body" (outer `None`) from
/// "field present with value `null`" (`Some(None)`) from "field present with
/// a value" (`Some(Some(v))`) -- the direct analogue of Pydantic's
/// `model_dump(exclude_unset=True)`, which only PATCHes keys the client
/// actually sent (including explicit `null`s, which clear a field).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ExperimentUpdate {
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub label: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub agents: Option<Option<i64>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub days: Option<Option<i64>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub start_date: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub granularity_minutes: Option<Option<i64>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub car_speed_kmh: Option<Option<f64>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub simulation_output: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub observed_path: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub time_use_path: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub time_use_label: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub time_use_country: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub time_use_survey: Option<Option<i64>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub time_use_weight_col: Option<Option<String>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub profiles_enabled: Option<Option<bool>>,
    #[serde(default, with = "::serde_with::rust::double_option")]
    pub profiles_output: Option<Option<String>>,
}

/// Applies `updates` to `configs/{exp_id}.yaml`, re-validating the *whole*
/// merged document against `CityBehavExConfig` before writing -- a partial/
/// invalid PATCH never corrupts the file. Mirrors `experiments.py::update_experiment`.
pub fn update_experiment(exp_id: &str, updates: &ExperimentUpdate) -> Result<Experiment, ExperimentError> {
    let config_path = configs_dir().join(format!("{exp_id}.yaml"));
    if !config_path.is_file() {
        return Err(ExperimentError::NotFound(exp_id.to_string()));
    }

    let mut raw = read_yaml_mapping(&config_path)?;

    // Python's `update_experiment` calls `_section(raw, name)` for
    // "simulation"/"comparison"/"profiles" unconditionally at the top of the
    // function (not per-field), which has the side effect of inserting an
    // empty mapping for any of the three that's missing from the file --
    // *even when no field from that section is actually being patched*.
    // Reproduced here for byte-for-byte output parity (confirmed against
    // the live Python backend: a PATCH that only touches `simulation`/
    // `comparison` fields still adds a `profiles: {}` block to a config
    // that didn't have one).
    section(&mut raw, "simulation")?;
    section(&mut raw, "comparison")?;
    section(&mut raw, "profiles")?;

    macro_rules! apply {
        ($field:expr, $section_name:expr, $key:expr) => {
            if let Some(inner) = &$field {
                let value = serde_yaml::to_value(inner).unwrap_or(serde_yaml::Value::Null);
                set_field(section(&mut raw, $section_name)?, $key, value);
            }
        };
    }
    let mut ambiguous = Vec::new();
    macro_rules! apply_str {
        ($field:expr, $section_name:expr, $key:expr) => {
            if let Some(inner) = &$field {
                let value = yaml_string_value($key, inner.as_ref(), &mut ambiguous);
                set_field(section(&mut raw, $section_name)?, $key, value);
            }
        };
    }
    apply!(updates.agents, "simulation", "agents");
    apply!(updates.days, "simulation", "days");
    apply_str!(updates.start_date, "simulation", "start_date");
    apply!(updates.granularity_minutes, "simulation", "granularity_minutes");
    apply!(updates.car_speed_kmh, "simulation", "car_speed_kmh");
    apply_str!(updates.simulation_output, "simulation", "output");
    apply_str!(updates.label, "comparison", "label");
    apply_str!(updates.observed_path, "comparison", "path");
    apply_str!(updates.time_use_path, "comparison", "time_use_path");
    apply_str!(updates.time_use_label, "comparison", "time_use_label");
    apply_str!(updates.time_use_country, "comparison", "time_use_country");
    apply!(updates.time_use_survey, "comparison", "time_use_survey");
    apply_str!(updates.time_use_weight_col, "comparison", "time_use_weight_col");
    apply!(updates.profiles_enabled, "profiles", "enabled");
    apply_str!(updates.profiles_output, "profiles", "output");

    let value = serde_yaml::Value::Mapping(raw.clone());
    let config: CityBehavExConfig = serde_yaml::from_value(value)
        .map_err(|e| ExperimentError::Mutation(e.to_string()))?;
    config
        .validate()
        .map_err(|e| ExperimentError::Mutation(e.to_string()))?;

    let text = serde_yaml::to_string(&serde_yaml::Value::Mapping(raw))
        .map_err(|e| ExperimentError::Mutation(e.to_string()))?;
    let text = force_quote_ambiguous_scalars(&text, &ambiguous);
    std::fs::write(&config_path, text)?;

    load_experiment(&config_path).map_err(|e| ExperimentError::Mutation(e.to_string()))
}

/// Moves `configs/{exp_id}.yaml` to `configs/.archived/`. Mirrors
/// `experiments.py::archive_experiment`.
pub fn archive_experiment(exp_id: &str) -> Result<PathBuf, ExperimentError> {
    let config_path = configs_dir().join(format!("{exp_id}.yaml"));
    if !config_path.is_file() {
        return Err(ExperimentError::NotFound(exp_id.to_string()));
    }
    let archived_dir = configs_dir().join(".archived");
    std::fs::create_dir_all(&archived_dir)?;
    let archived_path = archived_dir.join(config_path.file_name().unwrap());
    if archived_path.exists() {
        return Err(ExperimentError::Mutation(format!(
            "archived config already exists: {}",
            archived_path.file_name().unwrap().to_string_lossy()
        )));
    }
    std::fs::rename(&config_path, &archived_path)?;
    Ok(archived_path)
}

/// Deletes a run's trajectory parquet plus its `_encounters`/`_moving`/
/// `_activities`/`_crp` sidecars and `_social_network.json`. Mirrors
/// `experiments.py::delete_run`.
pub fn delete_run(exp_id: &str, run_id: &str) -> Result<Vec<PathBuf>, ExperimentError> {
    let experiment = get_experiment(exp_id).ok_or_else(|| ExperimentError::NotFound(exp_id.to_string()))?;
    let run = experiment
        .run(Some(run_id))
        .ok_or_else(|| ExperimentError::RunNotFound(run_id.to_string()))?;

    let candidates = [
        run.path.clone(),
        run.encounters_path(),
        run.moving_path(),
        run.activities_path(),
        run.crp_path(),
        run.social_network_path(),
    ];
    let mut deleted = Vec::new();
    for path in candidates {
        if path.exists() {
            std::fs::remove_file(&path)?;
            deleted.push(path);
        }
    }
    Ok(deleted)
}

#[cfg(test)]
mod yaml_quoting_tests {
    use super::*;

    #[test]
    fn dates_and_reserved_words_need_quoting() {
        for s in ["2026-01-01", "true", "false", "yes", "no", "null", "~", "", "42", "3.14"] {
            assert!(
                yaml_scalar_would_lose_string_type(s),
                "{s:?} should be flagged as ambiguous"
            );
        }
    }

    #[test]
    fn ordinary_strings_do_not_need_quoting() {
        for s in [
            "gparis",
            "data/gparis/results/trajectories.parquet",
            "MTUS France 2009",
            "propwt",
        ] {
            assert!(
                !yaml_scalar_would_lose_string_type(s),
                "{s:?} should NOT be flagged as ambiguous"
            );
        }
    }

    #[test]
    fn round_trips_through_pyyaml_compatible_quoting() {
        // A bare ISO date is exactly the case that broke: unquoted, PyYAML's
        // `safe_load` resolves it to `datetime.date`, not `str`, so a
        // Python backend re-reading a Rust-written config would fail
        // Pydantic's `str` validation on `start_date`.
        let mut ambiguous = Vec::new();
        let value = yaml_string_value(
            "start_date",
            Some(&"2026-01-01".to_string()),
            &mut ambiguous,
        );
        assert_eq!(value, serde_yaml::Value::String("2026-01-01".to_string()));
        assert_eq!(ambiguous, vec![("start_date".to_string(), "2026-01-01".to_string())]);

        let mut map = serde_yaml::Mapping::new();
        map.insert(serde_yaml::Value::String("start_date".into()), value);
        let rendered = serde_yaml::to_string(&map).unwrap();
        let quoted = force_quote_ambiguous_scalars(&rendered, &ambiguous);
        assert_eq!(quoted, "start_date: '2026-01-01'\n");

        // And it must parse back as a plain string, not get re-flagged.
        let reparsed: serde_yaml::Value = serde_yaml::from_str(&quoted).unwrap();
        let start_date = reparsed.get("start_date").unwrap().as_str().unwrap();
        assert_eq!(start_date, "2026-01-01");
    }

    #[test]
    fn safe_strings_are_left_unquoted() {
        let mut ambiguous = Vec::new();
        let value = yaml_string_value("label", Some(&"gparis".to_string()), &mut ambiguous);
        assert!(ambiguous.is_empty());
        let mut map = serde_yaml::Mapping::new();
        map.insert(serde_yaml::Value::String("label".into()), value);
        let rendered = serde_yaml::to_string(&map).unwrap();
        let quoted = force_quote_ambiguous_scalars(&rendered, &ambiguous);
        assert_eq!(quoted, "label: gparis\n");
    }
}
