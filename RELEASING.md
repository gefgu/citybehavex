# Releasing CityBehavEx

`v0.1.0` shipped 2026-09-19: Linux x86_64 wheels + sdist, live on PyPI.

1. `fastmob-core` (crates.io) and `fastmob-vis` (PyPI) are both published --
   done. CityBehavEx's release build must succeed from a clean checkout
   without the ignored `../fastmob` Cargo/Python development overrides.
2. Build the synthetic YJMOB-1k demo sample asset with
   `python scripts/build_yjmob_sample_release.py --source-trajectories <completed-run.parquet> --source-tessellation <yjmob_h3_tessellation.parquet> --output citybehavex-yjmob-1k-vX.Y.Z.tar.gz`.
   The asset is a subsample (first `--agents`/first `--days`, default 1000/7)
   of an already-completed CityBehavEx run, not the real YJMob100K dataset,
   so it carries no data-license restrictions. Publish it under its own
   `yjmob-1k-vX.Y.Z` GitHub Release tag -- deliberately **not** a `vX.Y.Z`
   tag, since that pattern triggers the PyPI/TestPyPI publish workflows below
   -- then replace the template manifest URL/version/SHA-256.
3. Optional: tag `vX.Y.Z-rcN` to build and publish wheels to TestPyPI first,
   and install/smoke test the release candidate from a clean platform.
4. Tag `vX.Y.Z` to publish the verified wheels and sdist to PyPI and attach
   the artifacts to the GitHub Release.

`release-testpypi.yml` builds CPython 3.11-3.13 wheels for Linux x86_64,
macOS x86_64/arm64, and Windows x86_64. `release-pypi.yml` currently builds
Linux x86_64 wheels + the sdist only, to skip the macos-13 runner queue --
macOS/Windows users fall back to the sdist (needs a Rust toolchain locally)
until a follow-up release restores the full platform matrix there too. The
external diary LLM is never part of a package release.
