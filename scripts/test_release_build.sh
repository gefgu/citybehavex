#!/usr/bin/env bash
# Mirror CI's Linux wheel-build step locally via Docker, before pushing a
# release/rc tag. Uses the same ghcr.io/pyo3/maturin image PyO3/maturin-action
# runs under the hood for `manylinux: auto`, so a failure here is a failure
# the tag-triggered GitHub Actions run would also hit.
#
# This only exercises the build (matches release-pypi.yml / release-testpypi.yml's
# `build` job). It does not `pip install` the wheel: citybehavex's real
# dependency `fastmob-vis` isn't published on PyPI yet, so an install attempt
# would fail on that external blocker regardless of whether the build itself
# is correct -- see RELEASING.md.
#
# Builds from a clean `git archive` snapshot of HEAD, not the live working
# tree: this machine's .cargo/config.toml patches fastmob-core to a relative
# ../fastmob sibling path for local dev, which is gitignored and doesn't
# exist inside the container (or in a real CI checkout) -- mounting the
# working tree as-is makes the build fail on that local-only file, not on
# anything CI would actually hit.
#
# macOS and Windows wheels cannot be built locally (no genuine Apple/Microsoft
# toolchain on this machine) and stay CI-only.
#
# Usage:
#   bash scripts/test_release_build.sh
#
# Requires: docker. If your user isn't in the `docker` group, run this with
# sudo instead: `sudo bash scripts/test_release_build.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MATURIN_IMAGE="ghcr.io/pyo3/maturin"
OUT_DIR="dist_local_test"
SNAPSHOT_DIR="$(mktemp -d)"
trap 'rm -rf "$SNAPSHOT_DIR"' EXIT

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: can't talk to the Docker daemon (permission denied or not running)."
    echo "       Retry as: sudo bash scripts/test_release_build.sh"
    exit 1
fi

echo "==> Exporting a clean git snapshot of HEAD (no local-only dotfiles) ..."
git archive HEAD | tar -x -C "$SNAPSHOT_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "==> Building manylinux x86_64 wheel (docker: ${MATURIN_IMAGE}) ..."
docker run --rm -v "$SNAPSHOT_DIR":/io -v "$REPO_ROOT/$OUT_DIR":/io/"$OUT_DIR" "$MATURIN_IMAGE" \
    build --release --out "$OUT_DIR" --find-interpreter

echo "==> Built:"
ls -lh "$OUT_DIR"

WHEEL=$(ls "$OUT_DIR"/*.whl | head -1)
echo "==> Checking metadata on ${WHEEL} ..."
python3 -m zipfile -l "$WHEEL" | grep -E "\.dist-info/METADATA$"
python3 -m zipfile -e "$WHEEL" "$OUT_DIR/extracted"
grep -E "^(Name|Version|Summary|License-File)" "$OUT_DIR"/extracted/*.dist-info/METADATA

echo "==> Checking citybehavex._core is bundled in the wheel ..."
python3 -m zipfile -l "$WHEEL" | grep -q "_core" && echo "OK: _core extension present"

if [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER":"$SUDO_USER" "$OUT_DIR"
fi

echo "==> Done. This only validates the build; it does not pip-install the"
echo "    wheel (fastmob-vis isn't on PyPI yet -- see RELEASING.md)."
