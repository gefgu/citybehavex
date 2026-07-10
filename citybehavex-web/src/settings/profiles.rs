//! Mirrors `citybehavex/profiles/config.py::AgentProfilesConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocationInferenceMethod {
    PoiBuilding,
}

impl Default for LocationInferenceMethod {
    fn default() -> Self {
        Self::PoiBuilding
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkDistanceModel {
    Exponential,
    None,
}

impl Default for WorkDistanceModel {
    fn default() -> Self {
        Self::Exponential
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkDistanceFallback {
    Expand,
    Global,
}

impl Default for WorkDistanceFallback {
    fn default() -> Self {
        Self::Expand
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AgentProfilesConfig {
    pub enabled: bool,
    pub profiles_path: Option<String>,
    pub output: String,
    pub llm_override: bool,
    pub home_anchors_path: Option<String>,
    pub home_anchors_output: Option<String>,
    pub home_anchor_relevance: f64,
    pub home_anchor_h3_resolution: i64,
    pub location_inference_method: LocationInferenceMethod,
    pub overture_building_features_path: Option<String>,
    pub overture_building_features_output: Option<String>,
    pub overture_feature_h3_resolution: Option<i64>,
    pub home_poi_inverse_weight: f64,
    pub home_building_weight: f64,
    pub work_poi_weight: f64,
    pub work_building_weight: f64,
    pub work_distance_model: WorkDistanceModel,
    pub work_distance_exponential_lambda: f64,
    pub work_distance_max_km: f64,
    pub work_distance_min_km: f64,
    pub work_distance_fallback: WorkDistanceFallback,
    pub work_distance_density_correction_power: f64,
    pub work_from_home_probability: f64,

    pub age_beta_a: f64,
    pub age_beta_b: f64,
    pub age_min: i64,
    pub age_max: i64,

    pub education_weights: Vec<f64>,
    pub health_weights: Vec<f64>,
    pub household_weights: Vec<f64>,
    pub job_weights: Vec<f64>,

    pub car_probability: f64,
    pub bike_probability: f64,
    pub coherence_alignment_backend: AlignmentBackend,
    pub coherence_alignment_base_url: Option<String>,
    pub coherence_alignment_model: Option<String>,
    pub coherence_alignment_timeout_seconds: f64,
    pub coherence_alignment_batch_size: i64,
    pub coherence_alignment_cache_path: Option<String>,
    pub coherence_alignment_concurrency: i64,
    pub coherence_alignment_retries: i64,
    pub coherence_alignment_checkpoint_every: i64,
    pub coherence_profile_cluster_similarity_threshold: f64,
    pub coherence_rerun_rounds: i64,
    pub coherence_rerun_threshold: f64,

    pub ownership_alignment_backend: AlignmentBackend,
    pub ownership_alignment_base_url: Option<String>,
    pub ownership_alignment_model: Option<String>,
    pub ownership_alignment_timeout_seconds: f64,
    pub ownership_alignment_batch_size: i64,
    pub ownership_alignment_cache_path: Option<String>,
    pub ownership_alignment_concurrency: i64,
    pub ownership_alignment_retries: i64,
    pub ownership_alignment_checkpoint_every: i64,
    pub ownership_profile_cluster_similarity_threshold: f64,

    pub male_names: Vec<String>,
    pub female_names: Vec<String>,
}

fn strings(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

impl Default for AgentProfilesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            profiles_path: None,
            output: "agent_profiles.parquet".to_string(),
            llm_override: false,
            home_anchors_path: None,
            home_anchors_output: None,
            home_anchor_relevance: 1.0,
            home_anchor_h3_resolution: 9,
            location_inference_method: LocationInferenceMethod::default(),
            overture_building_features_path: None,
            overture_building_features_output: None,
            overture_feature_h3_resolution: None,
            home_poi_inverse_weight: 0.5,
            home_building_weight: 1.0,
            work_poi_weight: 0.75,
            work_building_weight: 1.0,
            work_distance_model: WorkDistanceModel::default(),
            work_distance_exponential_lambda: 0.3,
            work_distance_max_km: 60.0,
            work_distance_min_km: 0.25,
            work_distance_fallback: WorkDistanceFallback::default(),
            work_distance_density_correction_power: 1.0,
            work_from_home_probability: 0.05,

            age_beta_a: 2.0,
            age_beta_b: 5.0,
            age_min: 16,
            age_max: 80,

            education_weights: vec![0.08, 0.32, 0.23, 0.27, 0.10],
            health_weights: vec![0.02, 0.07, 0.21, 0.45, 0.25],
            household_weights: vec![0.08, 0.21, 0.14, 0.05, 0.06, 0.12, 0.34],
            job_weights: vec![0.10, 0.22, 0.16, 0.12, 0.18, 0.03, 0.08, 0.06, 0.05],

            car_probability: 0.55,
            bike_probability: 0.35,
            coherence_alignment_backend: AlignmentBackend::default(),
            coherence_alignment_base_url: None,
            coherence_alignment_model: None,
            coherence_alignment_timeout_seconds: 120.0,
            coherence_alignment_batch_size: 32,
            coherence_alignment_cache_path: None,
            coherence_alignment_concurrency: 4,
            coherence_alignment_retries: 2,
            coherence_alignment_checkpoint_every: 20,
            coherence_profile_cluster_similarity_threshold: 0.94,
            coherence_rerun_rounds: 3,
            coherence_rerun_threshold: 0.6,

            ownership_alignment_backend: AlignmentBackend::default(),
            ownership_alignment_base_url: None,
            ownership_alignment_model: None,
            ownership_alignment_timeout_seconds: 120.0,
            ownership_alignment_batch_size: 32,
            ownership_alignment_cache_path: None,
            ownership_alignment_concurrency: 4,
            ownership_alignment_retries: 2,
            ownership_alignment_checkpoint_every: 20,
            ownership_profile_cluster_similarity_threshold: 0.94,

            male_names: strings(&[
                "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
                "Thomas", "Charles", "Daniel", "Matthew", "Lucas", "Hugo", "Théo", "Nathan",
                "Maxime", "Pierre", "Antoine", "Louis", "Julien", "Nicolas", "Clément",
                "Alexandre", "Thomas",
            ]),
            female_names: strings(&[
                "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Susan", "Jessica", "Sarah",
                "Karen", "Emma", "Léa", "Clara", "Chloé", "Camille", "Manon", "Inès", "Lucie",
                "Anaïs", "Juliette", "Marie", "Zoé", "Alice", "Océane", "Pauline", "Charlotte",
            ]),
        }
    }
}

impl AgentProfilesConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, weights) in [
            ("education_weights", &self.education_weights),
            ("health_weights", &self.health_weights),
            ("household_weights", &self.household_weights),
            ("job_weights", &self.job_weights),
        ] {
            if weights.is_empty() {
                anyhow::bail!("{name} must not be empty");
            }
            if weights.iter().any(|&w| w < 0.0) {
                anyhow::bail!("{name} must be non-negative");
            }
            if weights.iter().sum::<f64>() <= 0.0 {
                anyhow::bail!("{name} must sum to a positive value");
            }
        }
        if self.age_min >= self.age_max {
            anyhow::bail!("age_min must be less than age_max");
        }
        Ok(())
    }
}
