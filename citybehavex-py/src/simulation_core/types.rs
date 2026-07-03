use rand_xoshiro::Xoshiro256PlusPlus;

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
    pub(crate) visit_counts: Vec<u32>,
    pub(crate) total_visits: f64,
    pub(crate) s: f64,
    pub(crate) norm_sq: f64,
}

impl AgentState {
    pub(crate) fn new(n_locations: usize) -> Self {
        Self {
            current_location: 0,
            home_location: 0,
            work_location: 0,
            visited_locs: Vec::with_capacity(200),
            visit_counts: vec![0u32; n_locations],
            total_visits: 0.0,
            s: 0.0,
            norm_sq: 0.0,
        }
    }

    pub(crate) fn visit(&mut self, loc: usize) {
        let old = self.visit_counts[loc];
        self.norm_sq += (2 * old + 1) as f64;
        if old == 0 {
            self.s += 1.0;
            self.visited_locs.push(loc);
        }
        self.visit_counts[loc] += 1;
        self.total_visits += 1.0;
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
    pub(crate) agent: usize,
    pub(crate) contact: usize,
    pub(crate) tile: usize,
    pub(crate) ts: i64,
}

pub(crate) struct AgentParData {
    pub(crate) rng: Xoshiro256PlusPlus,
    pub(crate) diary: DiaryState,
    pub(crate) scratch: Scratch,
    pub(crate) moves: Vec<(usize, i64, i32)>,
    pub(crate) active_day: i64,
    pub(crate) active_abs_loc: i32,
    pub(crate) neighbor_indices: Vec<usize>,
    pub(crate) edge_sim: Vec<f64>,
    pub(crate) edge_upd: Vec<i64>,
    pub(crate) encounters: Vec<Encounter>,
    pub(crate) activity_counts: Vec<u32>,
    pub(crate) pending_departure: i64,
    /// Ordinal of the next micro-activity sampled within the currently-open
    /// stop; reset to 0 whenever a real relocation opens a new stop.
    pub(crate) activity_seq: i32,
    /// Cached CDF for gravity-exploration of unvisited tiles.
    /// Keyed by (source_tile, s_at_build); invalidated when either changes.
    pub(crate) explore_cache: Option<(usize, f64, Vec<usize>, Vec<f64>)>,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SocialMode {
    Exploration,
    Return,
}
