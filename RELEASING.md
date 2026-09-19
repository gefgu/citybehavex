# Releasing CityBehavEx

1. Publish compatible `fastmob-core` to crates.io and `fastmob-vis` to PyPI.
   CityBehavEx's release build must succeed from a clean checkout without the
   ignored `../fastmob` Cargo/Python development overrides.
2. Build the license-cleared prepared YJMOB-1k asset with
   `python scripts/build_yjmob_sample_release.py --input-dir <prepared-dir> --output citybehavex-yjmob-1k-vX.Y.Z.tar.gz`.
   Attach it to the matching GitHub Release, then replace the template
   manifest URL/version/SHA-256 before tagging the package release.
3. Tag `vX.Y.Z-rcN` to build and publish wheels to TestPyPI. Install and smoke
   test the release candidate from a clean supported platform.
4. Tag `vX.Y.Z` to publish the verified wheels and sdist to PyPI and attach
   the artifacts to the GitHub Release.

The release workflows build CPython 3.11--3.13 wheels for Linux x86_64,
macOS x86_64/arm64, and Windows x86_64. The external diary LLM is never part
of a package release.
