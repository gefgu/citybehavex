# Releasing CityBehavEx

`v0.1.0`-`v0.1.2` shipped 2026-09-19: Linux x86_64 wheels + sdist, live on
PyPI. `v0.1.3` restores the full platform matrix (see below).

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

Both `release-testpypi.yml` and `release-pypi.yml` build CPython 3.11-3.15
(incl. free-threaded) wheels for Linux x86_64, macOS x86_64/arm64, and
Windows x86_64, plus the sdist. The external diary LLM is never part of a
package release.
