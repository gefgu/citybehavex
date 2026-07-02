//! Car routing over an Overture-derived road graph using contraction
//! hierarchies (`fast_paths`): the graph is prepared once per simulation run,
//! then every trip's shortest path is a fast bidirectional search instead of
//! a fresh Dijkstra.

use std::collections::HashMap;

pub(crate) struct RoadGraph {
    fast_graph: fast_paths::FastGraph,
    edge_weight_ds: HashMap<(usize, usize), i64>,
}

impl RoadGraph {
    /// Builds and prepares the contraction hierarchy from directed edges.
    /// Parallel edges between the same (from, to) pair are deduped, keeping
    /// the smallest weight; self-loops are dropped (`fast_paths` disallows them).
    pub(crate) fn build(edge_from: &[usize], edge_to: &[usize], edge_weight_ds: &[usize]) -> Self {
        let mut deduped: HashMap<(usize, usize), usize> = HashMap::with_capacity(edge_from.len());
        for i in 0..edge_from.len() {
            let (from, to) = (edge_from[i], edge_to[i]);
            if from == to {
                continue;
            }
            let w = edge_weight_ds[i].max(1);
            deduped
                .entry((from, to))
                .and_modify(|existing| {
                    if w < *existing {
                        *existing = w;
                    }
                })
                .or_insert(w);
        }

        let mut input_graph = fast_paths::InputGraph::new();
        for (&(from, to), &w) in &deduped {
            input_graph.add_edge(from, to, w);
        }
        input_graph.freeze();
        let fast_graph = fast_paths::prepare(&input_graph);

        let edge_weight_ds = deduped
            .into_iter()
            .map(|((from, to), w)| ((from, to), w as i64))
            .collect();

        Self {
            fast_graph,
            edge_weight_ds,
        }
    }

    pub(crate) fn new_calculator(&self) -> fast_paths::PathCalculator {
        fast_paths::create_calculator(&self.fast_graph)
    }

    /// Returns (total_weight_ds, node_path including endpoints) or `None` if
    /// `from`/`to` are in disconnected components of the graph.
    pub(crate) fn shortest_path(
        &self,
        calc: &mut fast_paths::PathCalculator,
        from: usize,
        to: usize,
    ) -> Option<(usize, Vec<usize>)> {
        let path = calc.calc_path(&self.fast_graph, from, to)?;
        Some((path.get_weight(), path.get_nodes().clone()))
    }

    pub(crate) fn edge_weight_ds(&self, from: usize, to: usize) -> i64 {
        *self.edge_weight_ds.get(&(from, to)).unwrap_or(&0)
    }
}

/// Given the full node path and per-node cumulative time, subsample down to
/// at most `max_points`, always keeping the first and last point.
pub(crate) fn subsample_waypoints(
    lats: &[f64],
    lngs: &[f64],
    times: &[i64],
    max_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let n = lats.len();
    if n <= max_points || max_points < 2 {
        return (lats.to_vec(), lngs.to_vec(), times.to_vec());
    }
    let mut idxs = Vec::with_capacity(max_points);
    let step = (n - 1) as f64 / (max_points - 1) as f64;
    for i in 0..max_points {
        let idx = ((i as f64) * step).round() as usize;
        idxs.push(idx.min(n - 1));
    }
    idxs.dedup();
    (
        idxs.iter().map(|&i| lats[i]).collect(),
        idxs.iter().map(|&i| lngs[i]).collect(),
        idxs.iter().map(|&i| times[i]).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_path_over_a_chain() {
        // 0 -> 1 -> 2 -> 3, plus a direct but slower 0 -> 3 edge.
        let edge_from = vec![0, 1, 2, 0];
        let edge_to = vec![1, 2, 3, 3];
        let edge_weight = vec![10, 10, 10, 100];
        let graph = RoadGraph::build(&edge_from, &edge_to, &edge_weight);
        let mut calc = graph.new_calculator();
        let (weight, nodes) = graph.shortest_path(&mut calc, 0, 3).expect("path exists");
        assert_eq!(weight, 30);
        assert_eq!(nodes, vec![0, 1, 2, 3]);
    }

    #[test]
    fn disconnected_pair_returns_none() {
        let edge_from = vec![0, 2];
        let edge_to = vec![1, 3];
        let edge_weight = vec![5, 5];
        let graph = RoadGraph::build(&edge_from, &edge_to, &edge_weight);
        let mut calc = graph.new_calculator();
        assert!(graph.shortest_path(&mut calc, 0, 3).is_none());
    }

    #[test]
    fn subsample_keeps_endpoints_and_caps_length() {
        let lats: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let lngs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let times: Vec<i64> = (0..20).collect();
        let (lat, lng, t) = subsample_waypoints(&lats, &lngs, &times, 5);
        assert!(lat.len() <= 5);
        assert_eq!(lat.first(), Some(&0.0));
        assert_eq!(lat.last(), Some(&19.0));
        assert_eq!(lng.first(), Some(&0.0));
        assert_eq!(t.last(), Some(&19));
    }
}
