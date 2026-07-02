use crate::simulation_core::types::AgentState;

pub(crate) struct SimulationOutput {
    pub(crate) agents: Vec<i64>,
    pub(crate) lats: Vec<f64>,
    pub(crate) lngs: Vec<f64>,
    pub(crate) arrival: Vec<i64>,
    pub(crate) departure: Vec<i64>,
    pub(crate) duration: Vec<f64>,
    pub(crate) encounter_agent: Vec<i64>,
    pub(crate) encounter_contact: Vec<i64>,
    pub(crate) encounter_tile: Vec<i64>,
    pub(crate) encounter_ts: Vec<i64>,
    pub(crate) activity: Vec<i64>,
    pub(crate) stop_id: Vec<i64>,
    pub(crate) path_agent: Vec<i64>,
    pub(crate) path_stop_id: Vec<i64>,
    pub(crate) path_seq: Vec<i32>,
    pub(crate) path_lat: Vec<f64>,
    pub(crate) path_lng: Vec<f64>,
    pub(crate) path_t: Vec<i64>,
}

impl SimulationOutput {
    pub(crate) fn empty() -> Self {
        Self {
            agents: Vec::new(),
            lats: Vec::new(),
            lngs: Vec::new(),
            arrival: Vec::new(),
            departure: Vec::new(),
            duration: Vec::new(),
            encounter_agent: Vec::new(),
            encounter_contact: Vec::new(),
            encounter_tile: Vec::new(),
            encounter_ts: Vec::new(),
            activity: Vec::new(),
            stop_id: Vec::new(),
            path_agent: Vec::new(),
            path_stop_id: Vec::new(),
            path_seq: Vec::new(),
            path_lat: Vec::new(),
            path_lng: Vec::new(),
            path_t: Vec::new(),
        }
    }
}

/// Waypoints for road-following legs (or 2-point origin/destination pairs
/// when a trip falls back to straight-line routing). One entry per
/// destination-stop `stop_id`.
#[derive(Default)]
pub(crate) struct RoadPathOutputBuffers {
    pub(crate) agent: Vec<i64>,
    pub(crate) dest_stop_id: Vec<i64>,
    pub(crate) seq: Vec<i32>,
    pub(crate) lat: Vec<f64>,
    pub(crate) lng: Vec<f64>,
    pub(crate) t: Vec<i64>,
}

impl RoadPathOutputBuffers {
    pub(crate) fn push_leg(&mut self, agent: i64, dest_stop_id: i64, lats: &[f64], lngs: &[f64], times: &[i64]) {
        for (seq, ((&lat, &lng), &t)) in lats.iter().zip(lngs).zip(times).enumerate() {
            self.agent.push(agent);
            self.dest_stop_id.push(dest_stop_id);
            self.seq.push(seq as i32);
            self.lat.push(lat);
            self.lng.push(lng);
            self.t.push(t);
        }
    }
}

pub(crate) struct TripOutputBuffers {
    pub(crate) agents: Vec<i64>,
    pub(crate) lats: Vec<f64>,
    pub(crate) lngs: Vec<f64>,
    pub(crate) arrival: Vec<i64>,
    pub(crate) departure: Vec<i64>,
    pub(crate) duration: Vec<f64>,
    pub(crate) activity: Vec<i64>,
    pub(crate) stop_id: Vec<i64>,
    pub(crate) last_output_idx: Vec<usize>,
}

impl TripOutputBuffers {
    pub(crate) fn with_initial_agents(
        agents: &[AgentState],
        lats: &[f64],
        lngs: &[f64],
        start_ts: i64,
    ) -> Self {
        let n_agents = agents.len();
        let mut out = Self {
            agents: Vec::with_capacity(n_agents),
            lats: Vec::with_capacity(n_agents),
            lngs: Vec::with_capacity(n_agents),
            arrival: Vec::with_capacity(n_agents),
            departure: Vec::with_capacity(n_agents),
            duration: Vec::with_capacity(n_agents),
            activity: Vec::with_capacity(n_agents),
            stop_id: Vec::with_capacity(n_agents),
            last_output_idx: Vec::with_capacity(n_agents),
        };

        for (i, agent) in agents.iter().enumerate() {
            let stop_id = out.agents.len() as i64;
            out.last_output_idx.push(out.agents.len());
            out.agents.push(i as i64 + 1);
            out.lats.push(lats[agent.current_location]);
            out.lngs.push(lngs[agent.current_location]);
            out.arrival.push(start_ts);
            out.departure.push(start_ts);
            out.duration.push(0.0);
            out.activity.push(0);
            out.stop_id.push(stop_id);
        }

        out
    }

    pub(crate) fn into_output(
        self,
        encounter_agent: Vec<i64>,
        encounter_contact: Vec<i64>,
        encounter_tile: Vec<i64>,
        encounter_ts: Vec<i64>,
        paths: RoadPathOutputBuffers,
    ) -> SimulationOutput {
        SimulationOutput {
            agents: self.agents,
            lats: self.lats,
            lngs: self.lngs,
            arrival: self.arrival,
            departure: self.departure,
            duration: self.duration,
            encounter_agent,
            encounter_contact,
            encounter_tile,
            encounter_ts,
            activity: self.activity,
            stop_id: self.stop_id,
            path_agent: paths.agent,
            path_stop_id: paths.dest_stop_id,
            path_seq: paths.seq,
            path_lat: paths.lat,
            path_lng: paths.lng,
            path_t: paths.t,
        }
    }
}
