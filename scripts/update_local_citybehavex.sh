#!/usr/bin/env bash
set -euo pipefail

# Build the citybehavex Rust extension (citybehavex._core, the trip-duration-aware
# DITRAS) into the project virtualenv with maturin. Mirrors update_local_skmob.sh.
#
# The extension path-depends on ../skmob2/skmob2-core with default-features = false,
# so it does NOT pull in the C/SIMD crates (numkong/wass). If you also changed the
# skmob2 Rust core, run scripts/update_local_skmob.sh first.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SKMOB2_CORE_DIR="${ROOT_DIR}/../skmob2/skmob2-core"

require_path() {
    local path="$1"
    local description="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing ${description}: ${path}" >&2
        exit 1
    fi
}

require_path "${ROOT_DIR}/pyproject.toml" "project pyproject.toml"
require_path "${ROOT_DIR}/Cargo.toml" "citybehavex Cargo workspace"
require_path "${ROOT_DIR}/citybehavex-py/Cargo.toml" "citybehavex-py crate"
require_path "${VENV_DIR}/bin/python" "project virtual environment Python"
require_path "${SKMOB2_CORE_DIR}/Cargo.toml" "skmob2-core path dependency"

MATURIN_CMD=()
if [[ -x "${VENV_DIR}/bin/maturin" ]]; then
    MATURIN_CMD=("${VENV_DIR}/bin/maturin")
elif command -v maturin >/dev/null 2>&1; then
    MATURIN_CMD=(maturin)
elif command -v uv >/dev/null 2>&1; then
    MATURIN_CMD=(uv tool run --from "maturin>=1.13,<2.0" maturin)
else
    echo "Unable to find maturin." >&2
    echo "Install maturin in ${VENV_DIR} or on PATH." >&2
    exit 1
fi

echo "Building citybehavex._core into ${VENV_DIR} with maturin develop --release"
echo "  source: ${ROOT_DIR}"
echo "  maturin: ${MATURIN_CMD[*]}"

(
    cd "${ROOT_DIR}"
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="${VENV_DIR}" \
        PATH="${VENV_DIR}/bin:${PATH}" \
        "${MATURIN_CMD[@]}" develop --release
)

echo "Verifying the local citybehavex build"
"${VENV_DIR}/bin/python" - <<'PY'
import citybehavex._core as core

if not hasattr(core, "trip_ditras_simulate_agents"):
    raise SystemExit("citybehavex._core.trip_ditras_simulate_agents is unavailable")
print(f"citybehavex._core: {core.__file__}")
print("trip_ditras_simulate_agents: OK")
PY

echo "Done"
