//! Fixed filesystem layout, mirroring `web/backend/app/config.py`.
//!
//! The Python version resolves paths relative to its own source file's
//! location in the repo tree (three parents up from
//! `web/backend/app/config.py`). The Rust equivalent uses the compile-time
//! `CARGO_MANIFEST_DIR` (this crate lives one level under the repo root, at
//! `citybehavex-web/`) so it's equally fixed to the repo layout, with no
//! runtime env-var override.

use std::path::PathBuf;

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("citybehavex-web must live one level under the repo root")
        .to_path_buf()
}

pub fn configs_dir() -> PathBuf {
    repo_root().join("configs")
}

pub fn data_dir() -> PathBuf {
    repo_root().join("data")
}

/// Deliberately a different directory than the Python backend's
/// `data/.web_cache/` -- see the plan's Phase 4 note on why the two servers
/// don't share a cache directory.
pub fn cache_dir() -> PathBuf {
    data_dir().join(".web_cache_rs")
}

pub fn frontend_dist_dir() -> PathBuf {
    repo_root().join("web").join("frontend").join("dist")
}
