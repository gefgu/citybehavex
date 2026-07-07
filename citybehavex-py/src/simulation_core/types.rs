use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::FxHashMap;

/// Abstract location code reserved for WORK episodes (matches diary_to_abs_locs fixed map).
pub(crate) const WORK_CODE: i32 = 1;

pub(crate) const GRAVITY_REJECTION_ATTEMPTS: usize = 16;

pub(crate) struct DiaryState {
    pub(crate) diary_start: usize,
    pub(crate) diary_end: usize,
    pub(crate) diary_idx: usize,
}

impl DiaryState {
    pub(crate) fn current_ts(&self, diary_timestamps: &[i64]) -> Option<i64> {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            Some(diary_timestamps[idx])
        } else {
            None
        }
    }

    pub(crate) fn current_abstract_location(&self, diary_abs_locs: &[i32]) -> i32 {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            diary_abs_locs[idx]
        } else {
            0
        }
    }

    pub(crate) fn current_block_id(&self, diary_block_ids: &[i32]) -> i32 {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end && idx < diary_block_ids.len() {
            diary_block_ids[idx]
        } else {
            -1
        }
    }

    pub(crate) fn advance(&mut self, diary_timestamps: &[i64], end_ts: i64) -> i64 {
        self.diary_idx += 1;
        self.current_ts(diary_timestamps).unwrap_or(end_ts + 3600)
    }
}

pub(crate) struct AgentState {
    pub(crate) current_location: usize,
    pub(crate) home_location: usize,
    pub(crate) work_location: usize,
    pub(crate) visited_locs: Vec<usize>,
    /// Sparse visit counts, keyed by location id; absence means a count of 0.
    /// Was a `Vec<u32>` sized by the FULL location count (bounded only by
    /// `n_locations`, ~100k+ for large cities) per agent -- with 100k agents
    /// that's tens of GB of mostly-zero memory. Agents realistically visit a
    /// tiny fraction of all locations over a run, so a sparse map (sized by
    /// actual distinct visits, matching `visited_locs`) is the correct
    /// asymptotic representation.
    pub(crate) visit_counts: FxHashMap<usize, u32>,
    pub(crate) poi_type_counts: FxHashMap<usize, u32>,
    pub(crate) visited_poi_types: Vec<usize>,
    pub(crate) total_visits: f64,
    pub(crate) s: f64,
    pub(crate) norm_sq: f64,
}

impl AgentState {
    pub(crate) fn new() -> Self {
        Self {
            current_location: 0,
            home_location: 0,
            work_location: 0,
            visited_locs: Vec::with_capacity(200),
            visit_counts: FxHashMap::with_capacity_and_hasher(200, Default::default()),
            poi_type_counts: FxHashMap::with_capacity_and_hasher(32, Default::default()),
            visited_poi_types: Vec::with_capacity(32),
            total_visits: 0.0,
            s: 0.0,
            norm_sq: 0.0,
        }
    }

    pub(crate) fn visit(&mut self, loc: usize) {
        let entry = self.visit_counts.entry(loc).or_insert(0);
        let old = *entry;
        self.norm_sq += (2 * old + 1) as f64;
        if old == 0 {
            self.s += 1.0;
            self.visited_locs.push(loc);
        }
        *entry += 1;
        self.total_visits += 1.0;
    }

    pub(crate) fn visit_poi_type(&mut self, semantic_cluster: usize) {
        let entry = self.poi_type_counts.entry(semantic_cluster).or_insert(0);
        if *entry == 0 {
            self.visited_poi_types.push(semantic_cluster);
        }
        *entry += 1;
    }
}

pub(crate) struct Scratch {
    pub(crate) candidates: Vec<usize>,
    pub(crate) cdf: Vec<f64>,
    pub(crate) act_cdf: Vec<f64>,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::with_capacity(200),
            cdf: Vec::with_capacity(200),
            act_cdf: Vec::with_capacity(16),
        }
    }
}

/// An encounter recorded when a social action selects a contact's location.
#[derive(Clone)]
pub(crate) struct Encounter {
    pub(crate) agent: u32,
    pub(crate) contact: u32,
    pub(crate) tile: u32,
    pub(crate) ts: i32,
}

pub(crate) struct AgentParData {
    pub(crate) rng: Xoshiro256PlusPlus,
    pub(crate) diary: DiaryState,
    pub(crate) scratch: Scratch,
    pub(crate) moves: Vec<(usize, i64, i32, i32)>,
    pub(crate) active_day: i64,
    pub(crate) active_abs_loc: i32,
    pub(crate) active_block_id: i32,
    pub(crate) neighbor_indices: Vec<usize>,
    pub(crate) edge_sim: Vec<f64>,
    pub(crate) edge_upd: Vec<i64>,
    pub(crate) encounters: Vec<Encounter>,
    pub(crate) activity_counts: Vec<u32>,
    pub(crate) last_activity: i32,
    pub(crate) pending_departure: i64,
    /// Ordinal of the next micro-activity sampled within the currently-open
    /// stop; reset to 0 whenever a real relocation opens a new stop.
    pub(crate) activity_seq: i32,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SocialMode {
    Exploration,
    Return,
}
