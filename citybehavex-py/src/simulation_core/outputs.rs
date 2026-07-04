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
    pub(crate) stop_id: Vec<i64>,
    /// The diary abstract-location code (0=HOME, 1=WORK, 2+=OTHER) that
    /// actually drove each stop, so callers can label purpose without
    /// re-deriving it from a stop's (possibly slot-shifted) arrival time.
    pub(crate) stop_abstract_loc: Vec<i32>,
    pub(crate) path_agent: Vec<i64>,
    pub(crate) path_stop_id: Vec<i64>,
    pub(crate) path_seq: Vec<i32>,
    pub(crate) path_lat: Vec<f64>,
    pub(crate) path_lng: Vec<f64>,
    pub(crate) path_t: Vec<i64>,
    pub(crate) act_agent: Vec<i64>,
    pub(crate) act_stop_id: Vec<i64>,
    pub(crate) act_seq: Vec<i32>,
    pub(crate) act_activity: Vec<i64>,
    pub(crate) act_arrival: Vec<i64>,
    pub(crate) act_departure: Vec<i64>,
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
            stop_id: Vec::new(),
            stop_abstract_loc: Vec::new(),
            path_agent: Vec::new(),
            path_stop_id: Vec::new(),
            path_seq: Vec::new(),
            path_lat: Vec::new(),
            path_lng: Vec::new(),
            path_t: Vec::new(),
            act_agent: Vec::new(),
            act_stop_id: Vec::new(),
            act_seq: Vec::new(),
            act_activity: Vec::new(),
            act_arrival: Vec::new(),
            act_departure: Vec::new(),
        }
    }
}

/// One row per micro-activity sampled during a stop's dwell window. Kept
/// separate from `TripOutputBuffers` so the stop table stays a clean "one row
/// per real physical visit" table even when a single stop spans several
/// sampled micro-activities (e.g. sleep -> breakfast -> get ready, all at
/// HOME). `last_idx[agent_idx]` mirrors `TripOutputBuffers.last_output_idx`:
/// it points at the most recently pushed activity row for that agent, so its
/// `departure` can be patched once the next activity (or the stop itself)
/// closes it out.
#[derive(Default)]
pub(crate) struct ActivityOutputBuffers {
    pub(crate) agent: Vec<i64>,
    pub(crate) stop_id: Vec<i64>,
    pub(crate) seq: Vec<i32>,
    pub(crate) activity: Vec<i64>,
    pub(crate) arrival: Vec<i64>,
    pub(crate) departure: Vec<i64>,
    pub(crate) last_idx: Vec<usize>,
}

impl ActivityOutputBuffers {
    pub(crate) fn with_capacity(n_agents: usize) -> Self {
        Self {
            agent: Vec::with_capacity(n_agents),
            stop_id: Vec::with_capacity(n_agents),
            seq: Vec::with_capacity(n_agents),
            activity: Vec::with_capacity(n_agents),
            arrival: Vec::with_capacity(n_agents),
            departure: Vec::with_capacity(n_agents),
            last_idx: Vec::with_capacity(n_agents),
        }
    }

    /// Push a new micro-activity row for `agent_idx`, recording its index in
    /// `last_idx` so a later call can patch its `departure`. Returns the new
    /// row's index (unused by callers today, kept for symmetry/testability).
    pub(crate) fn push(&mut self, agent_idx: usize, stop_id: i64, seq: i32, arrival: i64) -> usize {
        let idx = self.agent.len();
        self.agent.push(agent_idx as i64 + 1);
        self.stop_id.push(stop_id);
        self.seq.push(seq);
        self.activity.push(0); // placeholder, patched immediately after sampling
        self.arrival.push(arrival);
        self.departure.push(arrival); // placeholder, patched when this activity closes
        if agent_idx < self.last_idx.len() {
            self.last_idx[agent_idx] = idx;
        } else {
            self.last_idx.push(idx);
        }
        idx
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
    /// Take everything accumulated so far, leaving an empty buffer behind.
    /// Waypoint rows are never patched after being pushed (unlike stop/
    /// activity rows, which track an "open" row per agent for later
    /// departure-patching), so a day-boundary flush can safely drain the
    /// whole buffer with no risk of losing an in-progress row.
    pub(crate) fn take_day_chunk(&mut self) -> RoadPathOutputBuffers {
        std::mem::take(self)
    }

    pub(crate) fn push_leg(
        &mut self,
        agent: i64,
        dest_stop_id: i64,
        lats: &[f64],
        lngs: &[f64],
        times: &[i64],
    ) {
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
    pub(crate) stop_id: Vec<i64>,
    /// Abstract-location code that opened each stop (0=HOME, 1=WORK,
    /// 2+=OTHER); mirrors `SimulationOutput.stop_abstract_loc`.
    pub(crate) abstract_loc: Vec<i32>,
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
            stop_id: Vec::with_capacity(n_agents),
            abstract_loc: Vec::with_capacity(n_agents),
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
            out.stop_id.push(stop_id);
            // Bootstrap stop: every agent starts at their fixed home tile.
            out.abstract_loc.push(0);
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
        activities: ActivityOutputBuffers,
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
            stop_id: self.stop_id,
            stop_abstract_loc: self.abstract_loc,
            path_agent: paths.agent,
            path_stop_id: paths.dest_stop_id,
            path_seq: paths.seq,
            path_lat: paths.lat,
            path_lng: paths.lng,
            path_t: paths.t,
            act_agent: activities.agent,
            act_stop_id: activities.stop_id,
            act_seq: activities.seq,
            act_activity: activities.activity,
            act_arrival: activities.arrival,
            act_departure: activities.departure,
        }
    }
}
