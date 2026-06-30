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
            last_output_idx: Vec::with_capacity(n_agents),
        };

        for (i, agent) in agents.iter().enumerate() {
            out.last_output_idx.push(out.agents.len());
            out.agents.push(i as i64 + 1);
            out.lats.push(lats[agent.current_location]);
            out.lngs.push(lngs[agent.current_location]);
            out.arrival.push(start_ts);
            out.departure.push(start_ts);
            out.duration.push(0.0);
            out.activity.push(0);
        }

        out
    }

    pub(crate) fn into_output(
        self,
        encounter_agent: Vec<i64>,
        encounter_contact: Vec<i64>,
        encounter_tile: Vec<i64>,
        encounter_ts: Vec<i64>,
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
        }
    }
}
