//! Mirrors `legacy Python backend/payload/legacy.py`'s section builders --
//! orchestration that combines the primitives in `comparison::*` (Phase 5)
//! and `comparison::{filters,ecdf,metric_row,features}` (Phase 6 so far)
//! into the exact JSON shapes `web/frontend/src/api.ts` expects.
//!
//! One submodule per payload section, mirroring `legacy.py`'s own rough
//! section groupings (the Python file isn't split into modules, but the
//! comment banners inside it mark the same boundaries).

pub mod activity;
pub mod distributions;
pub mod metrics;
pub mod micro_activity;
pub mod mobility_laws;
pub mod motifs;
pub mod profiles;
pub mod social_network;
pub mod stvd;
pub mod time_use;
pub mod transport_spatial;
