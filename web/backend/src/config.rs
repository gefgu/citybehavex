//! Fixed filesystem layout for the Rust web backend.
//!
//! The backend crate lives at `web/backend`, so `CARGO_MANIFEST_DIR` is two
//! levels below the repository root.

use std::path::PathBuf;

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("citybehavex-web must live at web/backend")
        .to_path_buf()
}

pub fn configs_dir() -> PathBuf {
    repo_root().join("configs")
}

pub fn data_dir() -> PathBuf {
    repo_root().join("data")
}

pub fn cache_dir() -> PathBuf {
    data_dir().join(".web_cache")
}

pub fn frontend_dist_dir() -> PathBuf {
    repo_root().join("web").join("frontend").join("dist")
}
