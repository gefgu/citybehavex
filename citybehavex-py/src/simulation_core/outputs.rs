use crate::simulation_core::types::AgentState;

pub(crate) struct SimulationOutput {
    pub(crate) agents: Vec<u32>,
    pub(crate) loc_id: Vec<u32>,
    pub(crate) arrival: Vec<i32>,
    pub(crate) departure: Vec<i32>,
    pub(crate) duration: Vec<u32>,
    pub(crate) encounter_agent: Vec<u32>,
    pub(crate) encounter_contact: Vec<u32>,
    pub(crate) encounter_tile: Vec<u32>,
    pub(crate) encounter_ts: Vec<i32>,
    pub(crate) stop_id: Vec<u32>,
    /// The diary abstract-location code (0=HOME, 1=WORK, 2+=OTHER) that
    /// actually drove each stop, so callers can label purpose without
    /// re-deriving it from a stop's (possibly slot-shifted) arrival time.
    pub(crate) stop_abstract_loc: Vec<u8>,
    pub(crate) path_agent: Vec<u32>,
    pub(crate) path_stop_id: Vec<u32>,
    pub(crate) path_seq: Vec<u16>,
    pub(crate) path_lat: Vec<f32>,
    pub(crate) path_lng: Vec<f32>,
    pub(crate) path_t: Vec<i32>,
    pub(crate) path_mode: Vec<u8>,
    pub(crate) act_agent: Vec<u32>,
    pub(crate) act_stop_id: Vec<u32>,
    pub(crate) act_seq: Vec<u16>,
    pub(crate) act_activity: Vec<u16>,
    pub(crate) act_arrival: Vec<i32>,
    pub(crate) act_departure: Vec<i32>,
    pub(crate) act_block_id: Vec<i32>,
}

impl SimulationOutput {
    pub(crate) fn empty() -> Self {
        Self {
            agents: Vec::new(),
            loc_id: Vec::new(),
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
            path_mode: Vec::new(),
            act_agent: Vec::new(),
            act_stop_id: Vec::new(),
            act_seq: Vec::new(),
            act_activity: Vec::new(),
            act_arrival: Vec::new(),
            act_departure: Vec::new(),
            act_block_id: Vec::new(),
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
    pub(crate) agent: Vec<u32>,
    pub(crate) stop_id: Vec<u32>,
    pub(crate) seq: Vec<u16>,
    pub(crate) activity: Vec<u16>,
    pub(crate) arrival: Vec<i32>,
    pub(crate) departure: Vec<i32>,
    /// The diary block that drove this activity's contextual-alignment
    /// lookup (see `ActivityInputs`/`activity_weight`), exposed here purely
    /// so Python-side reachability analysis can tell which (cluster, block)
    /// pairs a run actually visited -- never read back by the Rust core
    /// itself once the activity has been sampled.
    pub(crate) block_id: Vec<i32>,
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
            block_id: Vec::with_capacity(n_agents),
            last_idx: Vec::with_capacity(n_agents),
        }
    }

    /// Push a new micro-activity row for `agent_idx`, recording its index in
    /// `last_idx` so a later call can patch its `departure`. Returns the new
    /// row's index (unused by callers today, kept for symmetry/testability).
    pub(crate) fn push(
        &mut self,
        agent_idx: usize,
        stop_id: u32,
        seq: i32,
        arrival: i64,
        block_id: i32,
    ) -> usize {
        let idx = self.agent.len();
        self.agent.push(agent_idx as u32 + 1);
        self.stop_id.push(stop_id);
        self.seq.push(seq as u16);
        self.activity.push(0); // placeholder, patched immediately after sampling
        self.arrival.push(arrival as i32);
        self.departure.push(arrival as i32); // placeholder, patched when this activity closes
        self.block_id.push(block_id);
        if agent_idx < self.last_idx.len() {
            self.last_idx[agent_idx] = idx;
        } else {
            self.last_idx.push(idx);
        }
        idx
    }

    /// Same compaction scheme as `TripOutputBuffers::take_day_chunk` --
    /// `stop_id` here is already the counter-derived FK value (not a
    /// position), so it needs no adjustment across compaction.
    pub(crate) fn take_day_chunk(&mut self) -> ActivityOutputBuffers {
        let n_rows = self.agent.len();
        let n_agents = self.last_idx.len();
        let mut flushed = ActivityOutputBuffers::default();
        let mut residual = ActivityOutputBuffers {
            agent: Vec::with_capacity(n_agents),
            stop_id: Vec::with_capacity(n_agents),
            seq: Vec::with_capacity(n_agents),
            activity: Vec::with_capacity(n_agents),
            arrival: Vec::with_capacity(n_agents),
            departure: Vec::with_capacity(n_agents),
            block_id: Vec::with_capacity(n_agents),
            last_idx: std::mem::take(&mut self.last_idx),
        };
        for i in 0..n_rows {
            let owner = (self.agent[i] - 1) as usize;
            if residual.last_idx[owner] == i {
                let new_idx = residual.agent.len();
                residual.agent.push(self.agent[i]);
                residual.stop_id.push(self.stop_id[i]);
                residual.seq.push(self.seq[i]);
                residual.activity.push(self.activity[i]);
                residual.arrival.push(self.arrival[i]);
                residual.departure.push(self.departure[i]);
                residual.block_id.push(self.block_id[i]);
                residual.last_idx[owner] = new_idx;
            } else {
                flushed.agent.push(self.agent[i]);
                flushed.stop_id.push(self.stop_id[i]);
                flushed.seq.push(self.seq[i]);
                flushed.activity.push(self.activity[i]);
                flushed.arrival.push(self.arrival[i]);
                flushed.departure.push(self.departure[i]);
                flushed.block_id.push(self.block_id[i]);
            }
        }
        *self = residual;
        flushed
    }
}

/// Waypoints for road-following legs (or 2-point origin/destination pairs
/// when a trip falls back to straight-line routing). One entry per
/// destination-stop `stop_id`.
#[derive(Default)]
pub(crate) struct RoadPathOutputBuffers {
    pub(crate) agent: Vec<u32>,
    pub(crate) dest_stop_id: Vec<u32>,
    pub(crate) seq: Vec<u16>,
    pub(crate) lat: Vec<f32>,
    pub(crate) lng: Vec<f32>,
    pub(crate) t: Vec<i32>,
    pub(crate) mode: Vec<u8>,
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
        agent: u32,
        dest_stop_id: u32,
        lats: &[f64],
        lngs: &[f64],
        times: &[i64],
        mode: u8,
    ) {
        for (seq, ((&lat, &lng), &t)) in lats.iter().zip(lngs).zip(times).enumerate() {
            self.agent.push(agent);
            self.dest_stop_id.push(dest_stop_id);
            self.seq.push(seq as u16);
            self.lat.push(lat as f32);
            self.lng.push(lng as f32);
            self.t.push(t as i32);
            self.mode.push(mode);
        }
    }
}

pub(crate) struct TripOutputBuffers {
    pub(crate) agents: Vec<u32>,
    /// Location-table index of the tile this stop occupies (replaces storing
    /// a copy of the tile's lat/lng on every row); callers join this back
    /// against the small O(n_locations) tessellation table downstream.
    pub(crate) loc_id: Vec<u32>,
    pub(crate) arrival: Vec<i32>,
    pub(crate) departure: Vec<i32>,
    pub(crate) duration: Vec<u32>,
    pub(crate) stop_id: Vec<u32>,
    /// Abstract-location code that opened each stop (0=HOME, 1=WORK,
    /// 2+=OTHER); mirrors `SimulationOutput.stop_abstract_loc`.
    pub(crate) abstract_loc: Vec<u8>,
    pub(crate) last_output_idx: Vec<usize>,
}

impl TripOutputBuffers {
    pub(crate) fn with_initial_agents(
        agents: &[AgentState],
        start_ts: i64,
        next_stop_id: &mut u32,
    ) -> Self {
        let n_agents = agents.len();
        let mut out = Self {
            agents: Vec::with_capacity(n_agents),
            loc_id: Vec::with_capacity(n_agents),
            arrival: Vec::with_capacity(n_agents),
            departure: Vec::with_capacity(n_agents),
            duration: Vec::with_capacity(n_agents),
            stop_id: Vec::with_capacity(n_agents),
            abstract_loc: Vec::with_capacity(n_agents),
            last_output_idx: Vec::with_capacity(n_agents),
        };

        for (i, agent) in agents.iter().enumerate() {
            let stop_id = *next_stop_id;
            *next_stop_id += 1;
            out.last_output_idx.push(out.agents.len());
            out.agents.push(i as u32 + 1);
            out.loc_id.push(agent.current_location as u32);
            out.arrival.push(start_ts as i32);
            out.departure.push(start_ts as i32);
            out.duration.push(0);
            out.stop_id.push(stop_id);
            // Bootstrap stop: every agent starts at their fixed home tile.
            out.abstract_loc.push(0);
        }

        out
    }

    /// Partition the buffer at a day boundary into "closed" rows (safe to
    /// flush -- their `departure` will never be patched again) and the
    /// still-open rows (exactly one per agent, tracked by
    /// `last_output_idx`), which are retained in a fresh residual buffer with
    /// `last_output_idx` remapped to the new (compacted) positions. Unlike
    /// `RoadPathOutputBuffers::take_day_chunk`, this can't be a blind
    /// `mem::take`: each agent has one row that may still be patched (its
    /// `departure`) whenever that agent's *next* relocation happens --
    /// possibly many days later -- so that row must never be flushed while
    /// still open. Bounds resident rows to O(n_agents) + O(one day's pushes)
    /// instead of O(rows for the whole run).
    pub(crate) fn take_day_chunk(&mut self) -> TripOutputBuffers {
        let n_rows = self.agents.len();
        let n_agents = self.last_output_idx.len();
        let mut flushed = TripOutputBuffers {
            agents: Vec::new(),
            loc_id: Vec::new(),
            arrival: Vec::new(),
            departure: Vec::new(),
            duration: Vec::new(),
            stop_id: Vec::new(),
            abstract_loc: Vec::new(),
            last_output_idx: Vec::new(),
        };
        let mut residual = TripOutputBuffers {
            agents: Vec::with_capacity(n_agents),
            loc_id: Vec::with_capacity(n_agents),
            arrival: Vec::with_capacity(n_agents),
            departure: Vec::with_capacity(n_agents),
            duration: Vec::with_capacity(n_agents),
            stop_id: Vec::with_capacity(n_agents),
            abstract_loc: Vec::with_capacity(n_agents),
            last_output_idx: std::mem::take(&mut self.last_output_idx),
        };
        for i in 0..n_rows {
            let owner = (self.agents[i] - 1) as usize;
            if residual.last_output_idx[owner] == i {
                let new_idx = residual.agents.len();
                residual.agents.push(self.agents[i]);
                residual.loc_id.push(self.loc_id[i]);
                residual.arrival.push(self.arrival[i]);
                residual.departure.push(self.departure[i]);
                residual.duration.push(self.duration[i]);
                residual.stop_id.push(self.stop_id[i]);
                residual.abstract_loc.push(self.abstract_loc[i]);
                residual.last_output_idx[owner] = new_idx;
            } else {
                flushed.agents.push(self.agents[i]);
                flushed.loc_id.push(self.loc_id[i]);
                flushed.arrival.push(self.arrival[i]);
                flushed.departure.push(self.departure[i]);
                flushed.duration.push(self.duration[i]);
                flushed.stop_id.push(self.stop_id[i]);
                flushed.abstract_loc.push(self.abstract_loc[i]);
            }
        }
        *self = residual;
        flushed
    }

    pub(crate) fn into_output(
        self,
        encounter_agent: Vec<u32>,
        encounter_contact: Vec<u32>,
        encounter_tile: Vec<u32>,
        encounter_ts: Vec<i32>,
        paths: RoadPathOutputBuffers,
        activities: ActivityOutputBuffers,
    ) -> SimulationOutput {
        SimulationOutput {
            agents: self.agents,
            loc_id: self.loc_id,
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
            path_mode: paths.mode,
            act_agent: activities.agent,
            act_stop_id: activities.stop_id,
            act_seq: activities.seq,
            act_activity: activities.activity,
            act_arrival: activities.arrival,
            act_departure: activities.departure,
            act_block_id: activities.block_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a synthetic stop row for `agent` (0-based), updating
    /// `last_output_idx` the same way `push_stop_record` does.
    fn push_trip_row(out: &mut TripOutputBuffers, agent: u32, stop_id: u32, arrival: i32) {
        let idx = out.agents.len();
        out.agents.push(agent + 1);
        out.loc_id.push(0);
        out.arrival.push(arrival);
        out.departure.push(arrival);
        out.duration.push(0);
        out.stop_id.push(stop_id);
        out.abstract_loc.push(0);
        if (agent as usize) < out.last_output_idx.len() {
            out.last_output_idx[agent as usize] = idx;
        } else {
            out.last_output_idx.push(idx);
        }
    }

    #[test]
    fn take_day_chunk_conserves_rows_and_remaps_open_indices() {
        let mut out = TripOutputBuffers {
            agents: Vec::new(),
            loc_id: Vec::new(),
            arrival: Vec::new(),
            departure: Vec::new(),
            duration: Vec::new(),
            stop_id: Vec::new(),
            abstract_loc: Vec::new(),
            last_output_idx: Vec::new(),
        };
        // 3 agents, interleaved pushes mimicking collect_sorted_moves'
        // (ts, agent) ordering. Agent 1 never relocates again after its
        // first push -- its row must survive compaction untouched.
        push_trip_row(&mut out, 0, 100, 0); // idx 0, agent 0's open row (for now)
        push_trip_row(&mut out, 1, 101, 1); // idx 1, agent 1's open row (stays open)
        push_trip_row(&mut out, 2, 102, 2); // idx 2, agent 2's open row (for now)
        push_trip_row(&mut out, 0, 103, 3); // idx 3, closes idx 0, new open row for agent 0
        push_trip_row(&mut out, 2, 104, 4); // idx 4, closes idx 2, new open row for agent 2

        let total_before = out.agents.len();
        let open_before: std::collections::HashSet<usize> =
            out.last_output_idx.iter().copied().collect();

        let flushed = out.take_day_chunk();

        // Row conservation.
        assert_eq!(flushed.agents.len() + out.agents.len(), total_before);
        assert_eq!(out.agents.len(), 3); // exactly one open row per agent survives

        // No stop_id appears in both flushed and residual.
        let flushed_ids: std::collections::HashSet<u32> = flushed.stop_id.iter().copied().collect();
        let residual_ids: std::collections::HashSet<u32> = out.stop_id.iter().copied().collect();
        assert!(flushed_ids.is_disjoint(&residual_ids));
        assert_eq!(flushed_ids.len() + residual_ids.len(), total_before);

        // The rows that were open before compaction (idx 1, 3, 4) must be
        // exactly the ones retained (by stop_id: 101, 103, 104).
        assert_eq!(open_before, [1usize, 3, 4].into_iter().collect());
        assert_eq!(residual_ids, [101u32, 103, 104].into_iter().collect());

        // `last_output_idx` correctly indexes into the compacted `out`.
        for agent in 0..3usize {
            let idx = out.last_output_idx[agent];
            assert_eq!(out.agents[idx], agent as u32 + 1);
        }
    }

    #[test]
    fn take_day_chunk_on_activity_buffers_conserves_rows_and_remaps() {
        let mut acts = ActivityOutputBuffers::with_capacity(2);
        // Agent 0: two activities (first closes, second stays open).
        acts.push(0, 100, 0, 0, 7);
        acts.push(0, 100, 1, 10, 8);
        // Agent 1: one activity (stays open).
        acts.push(1, 200, 0, 0, 9);

        let total_before = acts.agent.len();
        let flushed = acts.take_day_chunk();

        assert_eq!(flushed.agent.len() + acts.agent.len(), total_before);
        assert_eq!(acts.agent.len(), 2); // one open row per agent

        for agent in 0..2usize {
            let idx = acts.last_idx[agent];
            assert_eq!(acts.agent[idx], agent as u32 + 1);
        }
        // The closed first activity for agent 0 (seq=0) must have been flushed.
        assert!(flushed.seq.contains(&0) && flushed.agent.iter().all(|&a| a == 1));
        // block_id must survive the flush/residual split in lockstep with the
        // other parallel fields (same row, same position).
        assert_eq!(flushed.block_id, vec![7]);
        assert_eq!(acts.block_id.len(), 2);
        for (i, &agent) in acts.agent.iter().enumerate() {
            let expected = if agent == 1 { 8 } else { 9 };
            assert_eq!(acts.block_id[i], expected);
        }
    }
}
