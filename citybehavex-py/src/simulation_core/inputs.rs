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
}

pub(crate) struct SocialGraphInputs<'a> {
    pub(crate) neighbor_starts: &'a [usize],
    pub(crate) neighbors: &'a [usize],
    pub(crate) edge_profile_sim: &'a [f64],
}

pub(crate) struct DiaryInputs<'a> {
    pub(crate) timestamps: &'a [i64],
    pub(crate) abstract_locations: &'a [i32],
    pub(crate) block_ids: &'a [i32],
    pub(crate) starts: &'a [usize],
    pub(crate) ends: &'a [usize],
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
}

pub(crate) struct TransportInputs<'a> {
    pub(crate) has_car: &'a [bool],
    pub(crate) has_bike: &'a [bool],
    pub(crate) walking_threshold_km: &'a [f64],
    pub(crate) bike_threshold_km: &'a [f64],
}

pub(crate) struct InitialLocationInputs<'a> {
    pub(crate) starting_locs: Option<&'a [usize]>,
    pub(crate) starting_locs_mode_relevance: bool,
    pub(crate) work_tiles: &'a [usize],
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
