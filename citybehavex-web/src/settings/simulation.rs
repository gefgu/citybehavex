//! Mirrors `citybehavex/simulation/config.py::SimulationConfig`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SimulationConfig {
    pub tessellation: Option<String>,
    pub min_lon: Option<f64>,
    pub min_lat: Option<f64>,
    pub max_lon: Option<f64>,
    pub max_lat: Option<f64>,
    pub agents: i64,
    pub days: i64,
    pub start_date: Option<String>,
    pub output: String,
    pub random_state: i64,
    pub relevance_column: String,
    pub granularity_minutes: i64,
    pub car_speed_kmh: f64,
    pub walking_speed_kmh: f64,
    pub bike_speed_kmh: f64,
    pub walking_threshold_mu_ln_km: f64,
    pub walking_threshold_sigma_ln: f64,
    pub bike_threshold_mu_ln_km: f64,
    pub bike_threshold_sigma_ln: f64,
    pub stream_output: bool,
    pub rho: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub dt_update_mob_sim_hours: f64,
    pub indipendency_window_hours: f64,
    pub gravity_deterrence_exponent: f64,
    pub gravity_origin_exponent: f64,
    pub gravity_destination_exponent: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            tessellation: None,
            min_lon: None,
            min_lat: None,
            max_lon: None,
            max_lat: None,
            agents: 500,
            days: 7,
            start_date: None,
            output: "trajectories.parquet".to_string(),
            random_state: 42,
            relevance_column: "total_poi_count".to_string(),
            granularity_minutes: 15,
            car_speed_kmh: 50.0,
            walking_speed_kmh: 4.8,
            bike_speed_kmh: 15.0,
            walking_threshold_mu_ln_km: -0.35,
            walking_threshold_sigma_ln: 0.45,
            bike_threshold_mu_ln_km: 1.4,
            bike_threshold_sigma_ln: 0.55,
            stream_output: false,
            rho: 0.6,
            gamma: 0.21,
            alpha: 0.2,
            dt_update_mob_sim_hours: 24.0 * 7.0,
            indipendency_window_hours: 0.5,
            gravity_deterrence_exponent: -2.0,
            gravity_origin_exponent: 1.0,
            gravity_destination_exponent: 1.0,
        }
    }
}

impl SimulationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.agents <= 0 {
            anyhow::bail!("agents must be positive");
        }
        if self.days <= 0 {
            anyhow::bail!("days must be positive");
        }
        if self.granularity_minutes <= 0 || 1440 % self.granularity_minutes != 0 {
            anyhow::bail!("granularity_minutes must be a positive divisor of 1440");
        }
        for (name, value) in [
            ("car_speed_kmh", self.car_speed_kmh),
            ("walking_speed_kmh", self.walking_speed_kmh),
            ("bike_speed_kmh", self.bike_speed_kmh),
        ] {
            if value <= 0.0 {
                anyhow::bail!("{name} must be positive");
            }
        }
        if self.walking_threshold_sigma_ln <= 0.0 || self.bike_threshold_sigma_ln <= 0.0 {
            anyhow::bail!("threshold sigma must be positive");
        }
        if !(self.rho > 0.0) {
            anyhow::bail!("rho must be > 0");
        }
        if !(self.gamma > 0.0) {
            anyhow::bail!("gamma must be > 0");
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            anyhow::bail!("alpha must be in [0, 1]");
        }
        if !(self.dt_update_mob_sim_hours > 0.0) {
            anyhow::bail!("dt_update_mob_sim_hours must be > 0");
        }
        if !(self.indipendency_window_hours > 0.0) {
            anyhow::bail!("indipendency_window_hours must be > 0");
        }

        let bbox = [self.min_lon, self.min_lat, self.max_lon, self.max_lat];
        let has_any = bbox.iter().any(|v| v.is_some());
        let has_full = bbox.iter().all(|v| v.is_some());
        if self.tessellation.is_some() && has_any {
            anyhow::bail!("provide either tessellation or bbox, not both");
        }
        if has_any && !has_full {
            anyhow::bail!("bbox requires min_lon, min_lat, max_lon, and max_lat");
        }
        Ok(())
    }
}
