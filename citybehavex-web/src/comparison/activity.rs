//! Mirrors fkmob's `activity_transition_matrix`, `daily_activity_distribution`,
//! and `discover_daily_motifs_from_agents` -- the data-prep (factorization,
//! indexed/sorted per-user row ranges) lives here in Rust; the actual
//! counting/graph-canonicalization algorithms are `fkmob-core` kernels,
//! called directly with no PyO3 in the loop.

use polars::prelude::*;
use std::collections::BTreeSet;

/// Mirrors `fkmob`'s `_factorize_activities`: dense integer codes for each
/// distinct string value, categories sorted lexicographically (matching
/// Python's `sorted(set(values), key=str)` for string-typed columns).
pub struct Factorized {
    pub categories: Vec<String>,
    pub codes: Vec<u64>,
}

pub fn factorize(values: &[String]) -> Factorized {
    let categories: Vec<String> = values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index: std::collections::HashMap<&str, u64> = categories
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as u64))
        .collect();
    let codes = values.iter().map(|v| index[v.as_str()]).collect();
    Factorized { categories, codes }
}

/// Contiguous per-user `(start, end)` row ranges, assuming `uid` is already
/// grouped (physically sorted by user, and by whatever secondary order the
/// caller needs -- e.g. timestamp) -- the direct-index equivalent of
/// fkmob's `indices`/`ends` permutation-based ranges, since this port
/// physically sorts rather than computing a permutation over an unsorted
/// frame.
pub fn contiguous_user_ranges<T: PartialEq>(sorted_uid: &[T]) -> (Vec<usize>, Vec<usize>) {
    let indices: Vec<usize> = (0..sorted_uid.len()).collect();
    let mut ends = Vec::new();
    let mut i = 0;
    while i < sorted_uid.len() {
        let mut j = i + 1;
        while j < sorted_uid.len() && sorted_uid[j] == sorted_uid[i] {
            j += 1;
        }
        ends.push(j);
        i = j;
    }
    (indices, ends)
}

/// Mirrors fkmob's `activity_transition_matrix`: `df` must already be
/// sorted by `[uid_col, timestamp_col]` (the physical-sort equivalent of
/// fkmob's `_build_indexed_user_ranges_fast` over an already-time-sorted
/// frame). Returns `(categories, matrix)` where `matrix[from][to]` is the
/// percentage of all consecutive-visit transitions (summed across every
/// user) that went from activity `categories[from]` to `categories[to]`.
pub fn activity_transition_matrix(
    sorted_df: &DataFrame,
    uid_col: &str,
    activity_col: &str,
) -> anyhow::Result<(Vec<String>, Vec<Vec<f64>>)> {
    let activities: Vec<String> = sorted_df
        .column(activity_col)?
        .as_materialized_series()
        .cast(&DataType::String)?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or("UNKNOWN").to_string())
        .collect();
    let factorized = factorize(&activities);
    let n = factorized.categories.len();
    if n == 0 {
        return Ok((factorized.categories, Vec::new()));
    }

    let uid: Vec<i64> = sorted_df
        .column(uid_col)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_iter()
        .map(|v| v.unwrap_or(i64::MIN))
        .collect();
    let (indices, ends) = contiguous_user_ranges(&uid);

    let counts = fkmob_core::measures::individual::activity::activity_transition_counts(
        &factorized.codes,
        &indices,
        &ends,
        n,
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let total: u64 = counts.iter().sum();
    let mut matrix = vec![vec![0.0f64; n]; n];
    if total > 0 {
        for from in 0..n {
            for to in 0..n {
                matrix[from][to] = counts[from * n + to] as f64 / total as f64 * 100.0;
            }
        }
    }
    Ok((factorized.categories, matrix))
}

/// Mirrors fkmob's `daily_activity_distribution`. `start_minutes`/`end_minutes`
/// are minute-of-day (`0..1440`); a visit with no resolvable end time should
/// pass the same value for both (matching that visit occupying only its
/// start-minute's bin). Returns `(categories, matrix[n_categories][n_bins])`,
/// percentages per time-bin column (`NaN` for a bin with zero activity
/// across every category, matching the Rust kernel's own convention).
pub fn daily_activity_distribution(
    activity: &[String],
    start_minutes: &[i64],
    end_minutes: &[i64],
    valid_rows: &[bool],
    bin_size_minutes: usize,
) -> anyhow::Result<(Vec<String>, Vec<Vec<f64>>)> {
    let factorized = factorize(activity);
    let n = factorized.categories.len();
    let n_bins = 1440 / bin_size_minutes;
    if n == 0 {
        return Ok((factorized.categories, Vec::new()));
    }

    let flat = fkmob_core::measures::individual::activity::daily_activity_percentages(
        &factorized.codes,
        start_minutes,
        end_minutes,
        valid_rows,
        n,
        bin_size_minutes,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    let matrix = (0..n)
        .map(|a| flat[a * n_bins..(a + 1) * n_bins].to_vec())
        .collect();
    Ok((factorized.categories, matrix))
}

#[derive(Debug, Clone)]
pub struct DailyMotif {
    pub user_id: String,
    pub date_id: i32,
    pub motif_id: i64,
    pub num_nodes: i32,
    pub num_edges: i32,
}

#[derive(Debug, Clone)]
pub struct MotifDistributionRow {
    pub motif_id: i64,
    pub count: i64,
    pub percentage: f64,
}

/// Mirrors fkmob's `discover_daily_motifs_from_agents`: `visits` must have
/// `[uid, location_id, purpose, start_timestamp, end_timestamp]` (the shape
/// `visits_for_comparison`/`motif_visits` produce), sorted by
/// `[uid, start_timestamp]`. Builds the `location_id + "_" + purpose` node
/// names, extracts hour-of-day and days-since-epoch, and calls
/// `fkmob-core`'s `compute_daily_motifs` directly.
pub fn discover_daily_motifs_from_agents(
    sorted_visits: &DataFrame,
) -> anyhow::Result<(Vec<DailyMotif>, Vec<MotifDistributionRow>)> {
    let uid = sorted_visits.column("uid")?.cast(&DataType::String)?;
    let uid = uid.str()?;
    let location_id = sorted_visits
        .column("location_id")?
        .cast(&DataType::String)?;
    let location_id = location_id.str()?;
    let purpose = sorted_visits.column("purpose")?.cast(&DataType::String)?;
    let purpose = purpose.str()?;
    let start_ts = sorted_visits
        .column("start_timestamp")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let start_ts = start_ts.datetime()?.clone();
    let end_ts = sorted_visits
        .column("end_timestamp")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let end_ts = end_ts.datetime()?.clone();

    let n = sorted_visits.height();
    let mut unique_ids = Vec::with_capacity(n);
    let mut purposes = Vec::with_capacity(n);
    let mut start_hours = Vec::with_capacity(n);
    let mut end_hours = Vec::with_capacity(n);
    let mut date_ids = Vec::with_capacity(n);
    let durations: Vec<Option<f64>> = vec![None; n];
    let mut uid_labels_per_row = Vec::with_capacity(n);

    const MICROS_PER_DAY: i64 = 86_400_000_000;
    const MICROS_PER_HOUR: i64 = 3_600_000_000;

    for i in 0..n {
        let loc = location_id.get(i).unwrap_or("");
        let purp = purpose.get(i).unwrap_or("");
        unique_ids.push(format!("{loc}_{purp}"));
        purposes.push(purp.to_string());
        let start_us = start_ts.phys.get(i).unwrap_or(0);
        let end_us = end_ts.phys.get(i).unwrap_or(start_us);
        start_hours.push(((start_us.rem_euclid(MICROS_PER_DAY)) / MICROS_PER_HOUR) as u32);
        end_hours.push(((end_us.rem_euclid(MICROS_PER_DAY)) / MICROS_PER_HOUR) as u32);
        date_ids.push(start_us.div_euclid(MICROS_PER_DAY) as i32);
        uid_labels_per_row.push(uid.get(i).unwrap_or("").to_string());
    }

    let (_indices, ends) = contiguous_user_ranges(&uid_labels_per_row);
    let mut user_ranges = Vec::with_capacity(ends.len());
    let mut user_id_labels = Vec::with_capacity(ends.len());
    let mut start = 0usize;
    for &end in &ends {
        user_ranges.push((start, end));
        user_id_labels.push(uid_labels_per_row[start].clone());
        start = end;
    }

    let (out_user_ids, out_date_ids, out_motif_ids) =
        fkmob_core::measures::individual::motifs::compute_daily_motifs(
            unique_ids,
            purposes,
            start_hours,
            end_hours,
            date_ids,
            durations,
            user_ranges,
            user_id_labels,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut daily = Vec::with_capacity(out_user_ids.len());
    let mut counts: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for ((user_id, date_id), motif_id) in out_user_ids
        .into_iter()
        .zip(out_date_ids)
        .zip(out_motif_ids)
    {
        let (num_nodes, num_edges) = decode_motif_id(motif_id);
        daily.push(DailyMotif {
            user_id,
            date_id,
            motif_id,
            num_nodes,
            num_edges,
        });
        *counts.entry(motif_id).or_insert(0) += 1;
    }

    let total = daily.len() as f64;
    let mut distribution: Vec<MotifDistributionRow> = counts
        .into_iter()
        .map(|(motif_id, count)| MotifDistributionRow {
            motif_id,
            count,
            percentage: if total > 0.0 {
                count as f64 / total * 100.0
            } else {
                0.0
            },
        })
        .collect();
    distribution.sort_by_key(|r| r.motif_id);

    Ok((daily, distribution))
}

/// Mirrors the Python-side motif_id decode: top bits (`>> 36`) are the node
/// count, low 36 bits are the canonical adjacency bitmask (popcount = edge
/// count). `motif_id == -1` (the &gt;6-node overflow sentinel) decodes to
/// `(-1, -1)`.
fn decode_motif_id(motif_id: i64) -> (i32, i32) {
    if motif_id == -1 {
        return (-1, -1);
    }
    let n_nodes = (motif_id >> 36) as i32;
    let adjacency_matrix = motif_id & ((1i64 << 36) - 1);
    let num_edges = adjacency_matrix.count_ones() as i32;
    (n_nodes, num_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_sorts_categories_lexicographically() {
        let values = vec![
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "a".to_string(),
        ];
        let f = factorize(&values);
        assert_eq!(f.categories, vec!["a", "b", "c"]);
        assert_eq!(f.codes, vec![1, 0, 2, 0]);
    }

    #[test]
    fn contiguous_user_ranges_groups_correctly() {
        let uid = vec![1, 1, 1, 2, 3, 3];
        let (indices, ends) = contiguous_user_ranges(&uid);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(ends, vec![3, 4, 6]);
    }

    #[test]
    fn transition_matrix_counts_consecutive_pairs_per_user() {
        // user 1: a -> b -> a (2 transitions); user 2: b -> b (1 transition, self-loop counted).
        let df = df![
            "uid" => [1i64, 1, 1, 2, 2],
            "activity" => ["a", "b", "a", "b", "b"],
        ]
        .unwrap();
        let (categories, matrix) = activity_transition_matrix(&df, "uid", "activity").unwrap();
        assert_eq!(categories, vec!["a", "b"]);
        // 3 total transitions: a->b, b->a, b->b.
        let a_idx = 0;
        let b_idx = 1;
        assert!((matrix[a_idx][b_idx] - (1.0 / 3.0 * 100.0)).abs() < 1e-9);
        assert!((matrix[b_idx][a_idx] - (1.0 / 3.0 * 100.0)).abs() < 1e-9);
        assert!((matrix[b_idx][b_idx] - (1.0 / 3.0 * 100.0)).abs() < 1e-9);
    }

    #[test]
    fn decode_motif_id_overflow_sentinel() {
        assert_eq!(decode_motif_id(-1), (-1, -1));
    }

    #[test]
    fn decode_motif_id_extracts_nodes_and_popcount_edges() {
        // 3 nodes, adjacency bits 0b101 (2 edges) packed into the low 36 bits.
        let motif_id = (3i64 << 36) | 0b101;
        assert_eq!(decode_motif_id(motif_id), (3, 2));
    }

    #[test]
    fn discover_daily_motifs_finds_home_visit_home_triangle() {
        let visits = df![
            "uid" => ["u1", "u1", "u1"],
            "location_id" => ["home", "work", "home"],
            "purpose" => ["HOME", "VISIT", "HOME"],
            "start_timestamp" => [
                "2026-01-01T00:00:00", "2026-01-01T09:00:00", "2026-01-01T18:00:00",
            ],
            "end_timestamp" => [
                "2026-01-01T08:00:00", "2026-01-01T17:00:00", "2026-01-01T23:00:00",
            ],
        ]
        .unwrap()
        .lazy()
        .with_columns([
            col("start_timestamp").str().to_datetime(
                Some(TimeUnit::Microseconds),
                None,
                StrptimeOptions::default(),
                lit("raise"),
            ),
            col("end_timestamp").str().to_datetime(
                Some(TimeUnit::Microseconds),
                None,
                StrptimeOptions::default(),
                lit("raise"),
            ),
        ])
        .collect()
        .unwrap();

        let (daily, distribution) = discover_daily_motifs_from_agents(&visits).unwrap();
        assert_eq!(daily.len(), 1);
        assert_eq!(daily[0].user_id, "u1");
        assert_eq!(daily[0].num_nodes, 2); // home, work
        assert_eq!(distribution.len(), 1);
        assert_eq!(distribution[0].percentage, 100.0);
    }

    /// Cross-checked against the live Python backend: built the exact same
    /// visits (via `_prepare_activity_visits` on the real gparis synthetic
    /// trajectory, `used_heuristic=False` since a real `purpose` column
    /// exists) and ran `fkmob.activity_transition_matrix`/
    /// `daily_activity_distribution` on it directly. 39578 visit rows;
    /// transition matrix categories `[HOME, OTHER, WORK]` with
    /// `HOME->OTHER=16.513472`, `HOME->WORK=21.237985`,
    /// `OTHER->HOME=26.976207`, `OTHER->WORK=6.649509`,
    /// `WORK->HOME=10.688587`, `WORK->OTHER=17.18315` (all diagonal 0, all
    /// percentages of the grand total across every user). Daily
    /// distribution (bin_size_minutes=60, 24 bins): `HOME` row[0..6] and
    /// `WORK` row[8..14] as below.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet"]
    fn gparis_activity_metrics_match_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let path = repo_root.join(
            "data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet",
        );
        let traj = super::super::trajectory::load_trajectory(&path).unwrap();

        let cols: Vec<&str> = traj
            .df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        let activity_col = crate::columns::detect_in(&cols, crate::columns::ACTIVITY_CANDIDATES);
        let result = super::super::visits::prepare_activity_visits(
            &traj.df,
            "synthetic",
            Some(&traj.uid_col),
            Some(&traj.datetime_col),
            activity_col.as_deref(),
            None,
            Some(&traj.lat_col),
            Some(&traj.lng_col),
            10,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.visits.height(), 39578);
        assert!(!result.used_heuristic);

        let sorted_visits = result
            .visits
            .lazy()
            .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
            .collect()
            .unwrap();

        let (categories, matrix) =
            activity_transition_matrix(&sorted_visits, "uid", "purpose").unwrap();
        assert_eq!(categories, vec!["HOME", "OTHER", "WORK"]);
        let expected = [
            [0.0f64, 16.513472, 21.237985],
            [26.976207, 0.75109, 6.649509],
            [10.688587, 17.18315, 0.0],
        ];
        for (row, expected_row) in matrix.iter().zip(expected.iter()) {
            for (v, e) in row.iter().zip(expected_row.iter()) {
                assert!((v - e).abs() < 1e-5, "got {v} want {e}");
            }
        }

        let purpose: Vec<String> = sorted_visits
            .column("purpose")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .flatten()
            .map(str::to_string)
            .collect();
        let start_ts = sorted_visits
            .column("start_timestamp")
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let start_ts = start_ts.datetime().unwrap().clone();
        let end_ts = sorted_visits
            .column("end_timestamp")
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let end_ts = end_ts.datetime().unwrap().clone();
        const MICROS_PER_DAY: i64 = 86_400_000_000;
        const MICROS_PER_MINUTE: i64 = 60_000_000;
        let n = sorted_visits.height();
        let start_minutes: Vec<i64> = (0..n)
            .map(|i| {
                start_ts.phys.get(i).unwrap_or(0).rem_euclid(MICROS_PER_DAY) / MICROS_PER_MINUTE
            })
            .collect();
        let end_minutes: Vec<i64> = (0..n)
            .map(|i| end_ts.phys.get(i).unwrap_or(0).rem_euclid(MICROS_PER_DAY) / MICROS_PER_MINUTE)
            .collect();
        let valid_rows = vec![true; n];

        let (dist_categories, dist_matrix) =
            daily_activity_distribution(&purpose, &start_minutes, &end_minutes, &valid_rows, 60)
                .unwrap();
        assert_eq!(dist_categories, vec!["HOME", "OTHER", "WORK"]);
        let home_row = &dist_matrix[dist_categories.iter().position(|c| c == "HOME").unwrap()];
        let expected_home_head = [
            97.29753215,
            97.22222222,
            97.22222222,
            97.22222222,
            97.22222222,
            97.22222222,
        ];
        for (got, want) in home_row[..6].iter().zip(expected_home_head.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
        let work_row = &dist_matrix[dist_categories.iter().position(|c| c == "WORK").unwrap()];
        let expected_work_mid = [
            50.93947271,
            42.30568597,
            38.73090482,
            38.54118374,
            37.77661483,
            35.64643016,
        ];
        for (got, want) in work_row[8..14].iter().zip(expected_work_mid.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    /// Cross-checked against the live Python backend's
    /// `discover_daily_motifs_from_agents` on the same `_motif_visits`-
    /// transformed visits (from the real gparis trajectory): 10248
    /// user-day rows; uid "1"'s first 5 days' `(motif_id, num_nodes,
    /// num_edges)`; a 17-row motif distribution with 10 known
    /// `(motif_id, count, percentage)` triples.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet"]
    fn gparis_motif_discovery_matches_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let path = repo_root.join(
            "data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet",
        );
        let traj = super::super::trajectory::load_trajectory(&path).unwrap();
        let cols: Vec<&str> = traj
            .df
            .get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect();
        let activity_col = crate::columns::detect_in(&cols, crate::columns::ACTIVITY_CANDIDATES);
        let result = super::super::visits::prepare_activity_visits(
            &traj.df,
            "synthetic",
            Some(&traj.uid_col),
            Some(&traj.datetime_col),
            activity_col.as_deref(),
            None,
            Some(&traj.lat_col),
            Some(&traj.lng_col),
            10,
            None,
        )
        .unwrap()
        .unwrap();
        let sorted_visits = result
            .visits
            .lazy()
            .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
            .collect()
            .unwrap();
        let motif_input = super::super::visits::motif_visits(&sorted_visits).unwrap();

        let (daily, distribution) = discover_daily_motifs_from_agents(&motif_input).unwrap();
        assert_eq!(daily.len(), 10248);

        let uid1: Vec<&DailyMotif> = daily.iter().filter(|d| d.user_id == "1").take(5).collect();
        let expected_uid1 = [
            (274877933592i64, 4i32, 5i32),
            (137438953478, 2, 2),
            (137438953478, 2, 2),
            (274877933592, 4, 5),
            (206158430348, 3, 3),
        ];
        for (row, (motif_id, num_nodes, num_edges)) in uid1.iter().zip(expected_uid1.iter()) {
            assert_eq!(row.motif_id, *motif_id);
            assert_eq!(row.num_nodes, *num_nodes);
            assert_eq!(row.num_edges, *num_edges);
        }

        assert_eq!(distribution.len(), 17);
        let expected_dist = [
            (68719476736i64, 38i64, 0.370804f64),
            (137438953478, 3403, 33.206479),
            (206158430348, 2131, 20.794301),
            (206158430434, 438, 4.274005),
            (206158430436, 2288, 22.326308),
            (206158430444, 37, 0.361046),
            (206158430446, 33, 0.322014),
            (274877923864, 162, 1.580796),
            (274877933588, 44, 0.429352),
            (274877933592, 858, 8.372365),
        ];
        for (motif_id, count, percentage) in expected_dist {
            let row = distribution
                .iter()
                .find(|r| r.motif_id == motif_id)
                .unwrap();
            assert_eq!(row.count, count);
            assert!(
                (row.percentage - percentage).abs() < 1e-5,
                "motif {motif_id}: got {}",
                row.percentage
            );
        }
    }
}
