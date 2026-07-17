//! Formerly the shared Rust core for CityBehavEx: H3 batch conversion,
//! contraction-hierarchy road routing, and co-presence/graph-metrics
//! computation all lived here (reachable directly, with no Python in the
//! loop, by both `citybehavex-py` and `citybehavex-web`). All three have
//! since been retired in favor of consuming the same kernels directly from
//! `fastmob-core` (which both crates already depend on), leaving this crate
//! empty. Kept as a placeholder in the workspace rather than removed
//! outright -- a call the workspace's maintainers should make deliberately,
//! not as a side effect of this migration.
