//! Mirrors `citybehavex/roads/config.py::{RoadNetworkConfig,RailNetworkConfig}`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RoadNetworkConfig {
    pub enabled: bool,
    pub overture_release: Option<String>,
    pub nodes_output: String,
    pub edges_output: String,
    pub snap_output: String,
    pub snap_max_distance_m: f64,
    pub max_leg_waypoints: i64,
}

impl Default for RoadNetworkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            overture_release: None,
            nodes_output: "data/road_graph_nodes.parquet".to_string(),
            edges_output: "data/road_graph_edges.parquet".to_string(),
            snap_output: "data/road_graph_snap.parquet".to_string(),
            snap_max_distance_m: 750.0,
            max_leg_waypoints: 128,
        }
    }
}

impl RoadNetworkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RailNetworkConfig {
    pub enabled: bool,
    pub overture_release: Option<String>,
    pub nodes_output: String,
    pub edges_output: String,
    pub snap_output: String,
    pub snap_max_distance_m: f64,
    pub max_leg_waypoints: i64,
    pub classes: Vec<String>,
    pub speed_kmh_by_class: HashMap<String, f64>,
    pub default_speed_kmh: f64,
}

impl Default for RailNetworkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            overture_release: None,
            nodes_output: "data/rail_graph_nodes.parquet".to_string(),
            edges_output: "data/rail_graph_edges.parquet".to_string(),
            snap_output: "data/rail_graph_snap.parquet".to_string(),
            snap_max_distance_m: 1500.0,
            max_leg_waypoints: 128,
            classes: vec![
                "subway".to_string(),
                "tram".to_string(),
                "light_rail".to_string(),
                "monorail".to_string(),
                "standard_gauge".to_string(),
            ],
            speed_kmh_by_class: HashMap::from([
                ("subway".to_string(), 35.0),
                ("tram".to_string(), 22.0),
                ("light_rail".to_string(), 30.0),
                ("monorail".to_string(), 28.0),
                ("standard_gauge".to_string(), 45.0),
            ]),
            default_speed_kmh: 35.0,
        }
    }
}

impl RailNetworkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.classes.is_empty() {
            anyhow::bail!("classes must not be empty");
        }
        Ok(())
    }
}
