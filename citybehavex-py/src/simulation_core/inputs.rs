use fastmob_core::models::od::validate_equal_lengths;

pub(crate) struct CoreInputs<'a> {
    pub(crate) locations: LocationInputs<'a>,
    pub(crate) social_graph: SocialGraphInputs<'a>,
    pub(crate) diary: DiaryInputs<'a>,
    pub(crate) params: SimulationParams,
    pub(crate) initial_locations: InitialLocationInputs<'a>,
    pub(crate) activities: ActivityInputs<'a>,
    pub(crate) road_network: RoadNetworkInputs<'a>,
    pub(crate) rail_network: RoadNetworkInputs<'a>,
    pub(crate) transport: TransportInputs<'a>,
}

pub(crate) struct ValidatedInputShape {
    pub(crate) n_locations: usize,
}

impl CoreInputs<'_> {
    pub(crate) fn validate(&self) -> Result<ValidatedInputShape, String> {
        if self.params.n_agents > u32::MAX as usize {
            return Err(format!(
                "n_agents={} exceeds u32::MAX; agent ids are stored as u32",
                self.params.n_agents
            ));
        }
        let n_locations = self.locations.validate()?;
        self.social_graph.validate(self.params.n_agents)?;
        self.diary.validate(self.params.n_agents)?;
        self.params.validate()?;
        self.initial_locations.validate(self.params.n_agents)?;
        self.transport.validate(self.params.n_agents)?;
        self.validate_per_agent_ranges()?;
        Ok(ValidatedInputShape { n_locations })
    }

    fn validate_per_agent_ranges(&self) -> Result<(), String> {
        for agent in 0..self.params.n_agents {
            if self.diary.starts[agent] > self.diary.ends[agent]
                || self.diary.ends[agent] > self.diary.timestamps.len()
            {
                return Err("diary ranges must be ordered and within diary_timestamps".to_string());
            }
            if self.diary.ends[agent] > self.diary.abstract_locations.len() {
                return Err("diary ranges must be within diary_abs_locs".to_string());
            }
            if self.diary.ends[agent] > self.diary.block_ids.len() {
                return Err("diary ranges must be within diary_block_ids".to_string());
            }
            if self.social_graph.neighbor_starts[agent]
                > self.social_graph.neighbor_starts[agent + 1]
                || self.social_graph.neighbor_starts[agent + 1] > self.social_graph.neighbors.len()
            {
                return Err("neighbor_starts must be ordered and within neighbors".to_string());
            }
        }
        Ok(())
    }
}

pub(crate) struct RoadNetworkInputs<'a> {
    pub(crate) edge_from: &'a [usize],
    pub(crate) edge_to: &'a [usize],
    pub(crate) edge_weight_ds: &'a [usize],
    pub(crate) node_lats: &'a [f64],
    pub(crate) node_lngs: &'a [f64],
    pub(crate) location_node: &'a [i64],
    pub(crate) max_leg_waypoints: usize,
}

impl RoadNetworkInputs<'_> {
    pub(crate) fn enabled(&self) -> bool {
        !self.edge_from.is_empty()
    }
}

pub(crate) struct LocationInputs<'a> {
    pub(crate) lats: &'a [f64],
    pub(crate) lngs: &'a [f64],
    pub(crate) relevances: &'a [f64],
    pub(crate) distances: &'a [f64],
    pub(crate) semantic_cluster_ids: &'a [usize],
    pub(crate) poi_type_scores: &'a [f64],
    pub(crate) poi_type_choice_enabled: bool,
    pub(crate) poi_type_n_clusters: usize,
    pub(crate) poi_type_n_blocks: usize,
    pub(crate) poi_type_temperature: f64,
    pub(crate) poi_type_alpha: f64,
}

impl LocationInputs<'_> {
    pub(crate) fn validate(&self) -> Result<usize, String> {
        let n_locations = validate_equal_lengths(&[
            ("latitudes", self.lats.len()),
            ("longitudes", self.lngs.len()),
            ("relevances", self.relevances.len()),
        ])?;
        if n_locations < 2 {
            return Err("need at least 2 locations".to_string());
        }
        if n_locations > u32::MAX as usize {
            return Err(format!(
                "n_locations={} exceeds u32::MAX; location indices are stored as u32",
                n_locations
            ));
        }
        if !self.distances.is_empty() && self.distances.len() != n_locations * n_locations {
            return Err(format!(
                "distances must be empty or have length n_locations*n_locations={}, got {}",
                n_locations * n_locations,
                self.distances.len()
            ));
        }
        if self.poi_type_choice_enabled {
            if self.semantic_cluster_ids.len() != n_locations {
                return Err(format!(
                    "location_semantic_cluster_ids must have length n_locations={}, got {}",
                    n_locations,
                    self.semantic_cluster_ids.len()
                ));
            }
            if self.poi_type_n_blocks == 0 || self.poi_type_n_clusters == 0 {
                return Err(
                    "POI type alignment dimensions must be positive when enabled".to_string(),
                );
            }
            let expected = self.poi_type_n_blocks * self.poi_type_n_clusters;
            if self.poi_type_scores.len() % expected != 0 {
                return Err(format!(
                    "poi_type_alignment_scores length must be a multiple of blocks*clusters={}, got {}",
                    expected,
                    self.poi_type_scores.len()
                ));
            }
            if !(self.poi_type_temperature.is_finite() && self.poi_type_temperature > 0.0) {
                return Err("poi_type_choice_temperature must be positive".to_string());
            }
            if !(self.poi_type_alpha.is_finite() && self.poi_type_alpha >= 0.0) {
                return Err("poi_type_choice_alpha must be non-negative".to_string());
            }
        }
        Ok(n_locations)
    }
}

pub(crate) struct SocialGraphInputs<'a> {
    pub(crate) neighbor_starts: &'a [usize],
    pub(crate) neighbors: &'a [usize],
    pub(crate) edge_profile_sim: &'a [f64],
}

impl SocialGraphInputs<'_> {
    pub(crate) fn validate(&self, n_agents: usize) -> Result<(), String> {
        if self.neighbor_starts.len() != n_agents + 1 {
            return Err(format!(
                "neighbor_starts must have length n_agents+1={}, got {}",
                n_agents + 1,
                self.neighbor_starts.len()
            ));
        }
        Ok(())
    }
}

pub(crate) struct DiaryInputs<'a> {
    pub(crate) timestamps: &'a [i64],
    pub(crate) abstract_locations: &'a [i32],
    pub(crate) block_ids: &'a [i32],
    pub(crate) starts: &'a [usize],
    pub(crate) ends: &'a [usize],
}

impl DiaryInputs<'_> {
    pub(crate) fn validate(&self, n_agents: usize) -> Result<(), String> {
        if self.starts.len() < n_agents || self.ends.len() < n_agents {
            return Err(format!(
                "diary_starts/diary_ends must have at least {} entries",
                n_agents
            ));
        }
        Ok(())
    }
}

pub(crate) struct SimulationParams {
    pub(crate) rho: f64,
    pub(crate) gamma: f64,
    pub(crate) alpha: f64,
    pub(crate) gravity_deterrence_exponent: f64,
    pub(crate) gravity_origin_exponent: f64,
    pub(crate) gravity_destination_exponent: f64,
    pub(crate) start_ts: i64,
    pub(crate) end_ts: i64,
    pub(crate) indipendency_window_s: i64,
    pub(crate) dt_update_mob_sim_s: i64,
    pub(crate) slot_seconds: i64,
    pub(crate) car_speed_kmh: f64,
    pub(crate) walking_speed_kmh: f64,
    pub(crate) bike_speed_kmh: f64,
    pub(crate) n_agents: usize,
    pub(crate) master_seed: Option<u64>,
    pub(crate) dynamic_friendships_enabled: bool,
    pub(crate) friendship_update_interval_s: i64,
    pub(crate) encounter_window_s: i64,
    pub(crate) regularity_threshold: f64,
    pub(crate) topological_overlap_threshold: f64,
    pub(crate) recast_random_baseline_samples: usize,
    pub(crate) recast_random_chance_probability: f64,
    pub(crate) strength_initial: f64,
    pub(crate) strength_growth_mu_ln: f64,
    pub(crate) strength_growth_sigma_ln: f64,
    pub(crate) strength_decay_rate: f64,
    pub(crate) max_dynamic_degree: usize,
    pub(crate) max_colocation_group_size: usize,
}

impl SimulationParams {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.slot_seconds <= 0 {
            return Err("slot_seconds must be positive".to_string());
        }
        if self.indipendency_window_s <= 0 {
            return Err("indipendency_window_s must be positive".to_string());
        }
        if self.dt_update_mob_sim_s <= 0 {
            return Err("dt_update_mob_sim_s must be positive".to_string());
        }
        if self.friendship_update_interval_s <= 0 {
            return Err("friendship_update_interval_s must be positive".to_string());
        }
        if self.encounter_window_s <= 0 {
            return Err("encounter_window_s must be positive".to_string());
        }
        if !(self.regularity_threshold.is_finite()
            && (0.0..=1.0).contains(&self.regularity_threshold))
        {
            return Err("regularity_threshold must be in [0, 1]".to_string());
        }
        if !(self.topological_overlap_threshold.is_finite()
            && (0.0..=1.0).contains(&self.topological_overlap_threshold))
        {
            return Err("topological_overlap_threshold must be in [0, 1]".to_string());
        }
        if !(self.recast_random_chance_probability.is_finite()
            && self.recast_random_chance_probability > 0.0
            && self.recast_random_chance_probability <= 1.0)
        {
            return Err("recast_random_chance_probability must be in (0, 1]".to_string());
        }
        if !(self.strength_initial.is_finite() && self.strength_initial > 0.0) {
            return Err("strength_initial must be positive".to_string());
        }
        if !(self.strength_growth_sigma_ln.is_finite() && self.strength_growth_sigma_ln > 0.0) {
            return Err("strength_growth_sigma_ln must be positive".to_string());
        }
        if !(self.strength_decay_rate.is_finite()
            && (0.0..=1.0).contains(&self.strength_decay_rate))
        {
            return Err("strength_decay_rate must be in [0, 1]".to_string());
        }
        if self.max_dynamic_degree == 0 {
            return Err("max_dynamic_degree must be positive".to_string());
        }
        if self.max_colocation_group_size < 2 {
            return Err("max_colocation_group_size must be at least 2".to_string());
        }
        if !(self.car_speed_kmh.is_finite() && self.car_speed_kmh > 0.0) {
            return Err("car_speed_kmh must be positive".to_string());
        }
        if !(self.walking_speed_kmh.is_finite() && self.walking_speed_kmh > 0.0) {
            return Err("walking_speed_kmh must be positive".to_string());
        }
        if !(self.bike_speed_kmh.is_finite() && self.bike_speed_kmh > 0.0) {
            return Err("bike_speed_kmh must be positive".to_string());
        }
        Ok(())
    }
}

pub(crate) struct TransportInputs<'a> {
    pub(crate) has_car: &'a [bool],
    pub(crate) has_bike: &'a [bool],
    pub(crate) walking_threshold_km: &'a [f64],
    pub(crate) bike_threshold_km: &'a [f64],
}

impl TransportInputs<'_> {
    pub(crate) fn validate(&self, n_agents: usize) -> Result<(), String> {
        for (name, len) in [
            ("has_car", self.has_car.len()),
            ("has_bike", self.has_bike.len()),
            ("walking_threshold_km", self.walking_threshold_km.len()),
            ("bike_threshold_km", self.bike_threshold_km.len()),
        ] {
            if len < n_agents {
                return Err(format!("{name} must have at least {n_agents} entries"));
            }
        }
        Ok(())
    }
}

pub(crate) struct InitialLocationInputs<'a> {
    pub(crate) starting_locs: Option<&'a [usize]>,
    pub(crate) starting_locs_mode_relevance: bool,
    pub(crate) work_tiles: &'a [usize],
}

impl InitialLocationInputs<'_> {
    pub(crate) fn validate(&self, n_agents: usize) -> Result<(), String> {
        if let Some(starts) = self.starting_locs
            && starts.len() < n_agents
        {
            return Err(format!(
                "starting_locs must have at least {} entries",
                n_agents
            ));
        }
        if self.work_tiles.len() < n_agents {
            return Err(format!(
                "work_tiles must have at least {} entries",
                n_agents
            ));
        }
        Ok(())
    }
}

pub(crate) struct ActivityInputs<'a> {
    pub(crate) act_embs: &'a [f64],
    pub(crate) act_dur_mu: &'a [f64],
    pub(crate) act_dur_sigma: &'a [f64],
    pub(crate) purpose_act_starts: &'a [usize],
    pub(crate) purpose_acts: &'a [usize],
    pub(crate) profile_embs: &'a [f64],
    pub(crate) profile_act_sims: &'a [f64],
    pub(crate) contextual_scores: &'a [f64],
    pub(crate) cluster_labels: &'a [usize],
    pub(crate) n_clusters: usize,
    pub(crate) n_blocks: usize,
    pub(crate) n_previous: usize,
    pub(crate) poi_semantic_scores: &'a [f64],
    pub(crate) location_semantic_cluster_ids: &'a [usize],
    pub(crate) poi_mask_starts: &'a [usize],
    pub(crate) poi_mask_activities: &'a [usize],
    pub(crate) n_poi_semantic_clusters: usize,
    pub(crate) history_weight: f64,
    pub(crate) emb_dim: usize,
    pub(crate) kappa: f64,
    pub(crate) temperature: f64,
    pub(crate) materialize_travel: bool,
}

impl ActivityInputs<'_> {
    pub(crate) fn enabled(&self) -> bool {
        !self.act_dur_mu.is_empty()
    }
}
