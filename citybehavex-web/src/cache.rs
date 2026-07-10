//! On-disk cache for comparison payloads, mirroring `web/backend/app/cache.py`.
//!
//! Building a payload loads and processes the full observed table (millions
//! of rows for some cities), so results are cached as JSON keyed by the
//! mtimes of the input parquets. A changed input invalidates the entry
//! automatically (no TTL, no explicit invalidation route).
//!
//! `get_or_build` also de-duplicates concurrent requests for the same
//! still-uncached key: without this, two browser tabs (or a page reload
//! racing its own previous request) hitting the same cold cache would each
//! redundantly run the full expensive build. The in-flight registry is
//! per-process, coalescing concurrent requests landing in this same server.
//!
//! Uses its own cache directory (`data/.web_cache_rs/`, see
//! `config::cache_dir`), separate from the Python backend's
//! `data/.web_cache/` -- the two servers' cache files are not intended to be
//! byte-compatible or shared, only each internally self-consistent.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

pub const PAYLOAD_CACHE_VERSION: &str = "v9";
const MAX_CACHE_KEY_PREFIX: usize = 120;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Build(#[from] anyhow::Error),
}

/// A previously-cached payload is discarded (forcing a rebuild) when
/// `extra_key["section"] == "mobility-laws"` and the cached payload's
/// `mobility_laws` field is `null` -- a migration-era guard so old cache
/// entries written by a previously-buggy build self-heal on next request.
fn is_stale_payload(payload: &Value, extra_key: Option<&Value>) -> bool {
    let Some(extra_key) = extra_key.and_then(Value::as_object) else {
        return false;
    };
    if extra_key.get("section").and_then(Value::as_str) == Some("mobility-laws")
        && payload.get("mobility_laws").is_some_and(Value::is_null)
    {
        return true;
    }
    false
}

fn safe_part(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn mtime_or(path: &Path, missing: &str) -> Value {
    match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(t) => {
            let secs = t
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            json!(secs)
        }
        Err(_) => json!(missing),
    }
}

pub struct Cache {
    dir: PathBuf,
    inflight: Mutex<HashMap<String, Arc<OnceCell<Value>>>>,
}

impl Cache {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            inflight: Mutex::new(HashMap::new()),
        }
    }

    fn key(
        &self,
        exp_id: &str,
        run_id: &str,
        synthetic: &Path,
        observed: Option<&Path>,
        extra_paths: &[&Path],
        extra_key: Option<&Value>,
    ) -> String {
        let syn_mtime = mtime_or(synthetic, "missing");
        let obs_mtime = match observed {
            Some(p) if p.exists() => mtime_or(p, "missing"),
            _ => json!("synthetic-only"),
        };
        let extra: Vec<Value> = extra_paths
            .iter()
            .map(|p| json!([p.display().to_string(), p.file_stem().map(|s| s.to_string_lossy().to_string()), mtime_or(p, "missing")]))
            .collect();
        let key_parts = json!({
            "version": PAYLOAD_CACHE_VERSION,
            "exp_id": exp_id,
            "run_id": run_id,
            "synthetic": [synthetic.display().to_string(), syn_mtime],
            "observed": [observed.map(|p| p.display().to_string()), obs_mtime],
            "extra": extra,
            "extra_key": extra_key,
        });
        // `serde_json::Map` is `BTreeMap`-backed by default (no
        // `preserve_order` feature), so `to_string` emits object keys in
        // sorted order at every nesting level -- the same guarantee
        // `json.dumps(..., sort_keys=True)` gives the Python version, which
        // is what makes this hash stable for a given logical key.
        let digest_input = serde_json::to_string(&key_parts).expect("json serialize");
        let digest = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(digest_input.as_bytes());
            hex::encode(hasher.finalize())
        };
        let digest = &digest[..16];
        let prefix = format!(
            "{PAYLOAD_CACHE_VERSION}__{}__{}",
            safe_part(exp_id),
            safe_part(run_id)
        );
        let prefix: String = prefix.chars().take(MAX_CACHE_KEY_PREFIX).collect();
        format!("{prefix}__{digest}.json")
    }

    /// Cache-or-build, coalescing concurrent callers for the same key.
    #[allow(clippy::too_many_arguments)]
    pub async fn get_or_build<F, Fut>(
        &self,
        exp_id: &str,
        run_id: &str,
        synthetic: &Path,
        observed: Option<&Path>,
        extra_paths: &[&Path],
        extra_key: Option<&Value>,
        refresh: bool,
        build: F,
    ) -> Result<Value, CacheError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = anyhow::Result<Value>>,
    {
        std::fs::create_dir_all(&self.dir)?;
        let cache_key = self.key(exp_id, run_id, synthetic, observed, extra_paths, extra_key);
        let cache_file = self.dir.join(&cache_key);

        if !refresh && cache_file.exists() {
            if let Ok(text) = std::fs::read_to_string(&cache_file) {
                if let Ok(cached) = serde_json::from_str::<Value>(&text) {
                    if !is_stale_payload(&cached, extra_key) {
                        return Ok(cached);
                    }
                }
            }
        }

        let cell = {
            let mut inflight = self.inflight.lock().unwrap();
            inflight
                .entry(cache_key.clone())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        let did_build = Arc::new(AtomicBool::new(false));
        let did_build_flag = did_build.clone();
        let result = cell
            .get_or_try_init(|| async move {
                did_build_flag.store(true, Ordering::SeqCst);
                build().await
            })
            .await
            .cloned();

        if did_build.load(Ordering::SeqCst) {
            self.inflight.lock().unwrap().remove(&cache_key);
            if let Ok(payload) = &result {
                let _ = std::fs::write(&cache_file, serde_json::to_string(payload).unwrap_or_default());
            }
        }
        Ok(result?)
    }

    /// Like `get_or_build` but for a parquet-file cache artifact keyed by a
    /// single input's mtime (e.g. a derived per-run precomputation), rather
    /// than the two-mtime JSON comparison-payload cache above. No in-flight
    /// coalescing (matches the Python version, which is synchronous).
    pub fn get_or_build_parquet(
        &self,
        cache_name: &str,
        key_parts: &[&str],
        input_path: &Path,
        build: impl FnOnce(&Path) -> anyhow::Result<()>,
    ) -> anyhow::Result<PathBuf> {
        let subdir = self.dir.join(cache_name);
        std::fs::create_dir_all(&subdir)?;
        let mtime = std::fs::metadata(input_path)?
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let out = subdir.join(format!("{}__{mtime}.parquet", key_parts.join("__")));
        if !out.exists() {
            build(&out)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cbx-cache-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"x").unwrap();
    }

    #[test]
    fn safe_part_replaces_unsafe_chars() {
        assert_eq!(safe_part("gparis_simulation"), "gparis_simulation");
        assert_eq!(safe_part("weird/id with spaces!"), "weird-id-with-spaces-");
    }

    #[tokio::test]
    async fn cache_hit_avoids_rebuild() {
        let dir = tmp_dir("hit");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Cache::new(dir.join("cache"));
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let calls = calls.clone();
            let value = cache
                .get_or_build("exp", "run", &synthetic, None, &[], None, false, || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(json!({"n": 1}))
                })
                .await
                .unwrap();
            assert_eq!(value, json!({"n": 1}));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "second/third call should hit disk cache");
    }

    #[tokio::test]
    async fn different_extra_key_produces_different_cache_entry() {
        let dir = tmp_dir("extra-key");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Cache::new(dir.join("cache"));

        let a = cache
            .get_or_build("exp", "run", &synthetic, None, &[], Some(&json!({"section": "a"})), false, || async {
                Ok(json!({"which": "a"}))
            })
            .await
            .unwrap();
        let b = cache
            .get_or_build("exp", "run", &synthetic, None, &[], Some(&json!({"section": "b"})), false, || async {
                Ok(json!({"which": "b"}))
            })
            .await
            .unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn concurrent_calls_for_same_key_coalesce_into_one_build() {
        let dir = tmp_dir("coalesce");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Arc::new(Cache::new(dir.join("cache")));
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let synthetic = synthetic.clone();
            let calls = calls.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_build("exp", "run", &synthetic, None, &[], None, false, || async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(json!({"n": 1}))
                    })
                    .await
                    .unwrap()
            }));
        }
        for h in handles {
            assert_eq!(h.await.unwrap(), json!({"n": 1}));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "concurrent identical requests must coalesce");
    }

    #[tokio::test]
    async fn refresh_forces_rebuild() {
        let dir = tmp_dir("refresh");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Cache::new(dir.join("cache"));
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = calls.clone();
            cache
                .get_or_build("exp", "run", &synthetic, None, &[], None, true, || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(json!({"n": 1}))
                })
                .await
                .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2, "refresh=true must always rebuild");
    }

    #[tokio::test]
    async fn stale_mobility_laws_payload_is_rebuilt() {
        let dir = tmp_dir("stale");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Cache::new(dir.join("cache"));
        let extra_key = json!({"section": "mobility-laws"});

        // First build "poisons" the cache with a null mobility_laws, as a
        // previously-buggy build might have.
        cache
            .get_or_build("exp", "run", &synthetic, None, &[], Some(&extra_key), false, || async {
                Ok(json!({"mobility_laws": null}))
            })
            .await
            .unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let value = cache
            .get_or_build("exp", "run", &synthetic, None, &[], Some(&extra_key), false, || async move {
                calls2.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"mobility_laws": {"groups": []}}))
            })
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "stale null-mobility_laws entry must trigger a rebuild");
        assert_eq!(value, json!({"mobility_laws": {"groups": []}}));
    }

    #[tokio::test]
    async fn build_error_does_not_poison_the_cache_for_the_next_call() {
        let dir = tmp_dir("error-retry");
        let synthetic = dir.join("synthetic.parquet");
        touch(&synthetic);
        let cache = Cache::new(dir.join("cache"));

        let err = cache
            .get_or_build("exp", "run", &synthetic, None, &[], None, false, || async {
                anyhow::bail!("boom")
            })
            .await;
        assert!(err.is_err());

        let ok = cache
            .get_or_build("exp", "run", &synthetic, None, &[], None, false, || async {
                Ok(json!({"n": 1}))
            })
            .await
            .unwrap();
        assert_eq!(ok, json!({"n": 1}));
    }

    #[test]
    fn get_or_build_parquet_only_builds_once() {
        let dir = tmp_dir("parquet");
        let input = dir.join("input.parquet");
        touch(&input);
        let cache = Cache::new(dir.join("cache"));
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let calls = calls.clone();
            let out = cache
                .get_or_build_parquet("legs_index", &["exp", "run"], &input, |out| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::fs::write(out, b"data").map_err(Into::into)
                })
                .unwrap();
            assert!(out.exists());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
