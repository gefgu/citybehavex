#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FKMOB_DIR="${ROOT_DIR}/../fkmob"
VENV_DIR="${ROOT_DIR}/.venv"

require_path() {
    local path="$1"
    local description="$2"

    if [[ ! -e "$path" ]]; then
        echo "Missing ${description}: ${path}" >&2
        exit 1
    fi
}

require_path "${ROOT_DIR}/pyproject.toml" "project pyproject.toml"
require_path "${VENV_DIR}/bin/python" "project virtual environment Python"
require_path "${FKMOB_DIR}/pyproject.toml" "fkmob package"
require_path "${FKMOB_DIR}/Cargo.toml" "fkmob Rust manifest"
require_path "${FKMOB_DIR}/fkmob-py/Cargo.toml" "fkmob Python Rust manifest"

MATURIN_CMD=()
if [[ -x "${VENV_DIR}/bin/maturin" ]]; then
    MATURIN_CMD=("${VENV_DIR}/bin/maturin")
elif [[ -x "${FKMOB_DIR}/.venv/bin/maturin" ]]; then
    MATURIN_CMD=("${FKMOB_DIR}/.venv/bin/maturin")
elif command -v maturin >/dev/null 2>&1; then
    MATURIN_CMD=(maturin)
elif command -v uv >/dev/null 2>&1; then
    MATURIN_CMD=(uv tool run --from "maturin>=1.13,<2.0" maturin)
else
    echo "Unable to find maturin." >&2
    echo "Install maturin in ${VENV_DIR}, ${FKMOB_DIR}/.venv, or on PATH." >&2
    exit 1
fi

echo "Building fkmob into ${VENV_DIR} with maturin develop --release"
echo "  source: ${FKMOB_DIR}"
echo "  maturin: ${MATURIN_CMD[*]}"

(
    cd "${FKMOB_DIR}"
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="${VENV_DIR}" \
        PATH="${VENV_DIR}/bin:${PATH}" \
        "${MATURIN_CMD[@]}" develop --release
)

echo "Verifying the local fkmob build"
FKMOB_DIR="${FKMOB_DIR}" "${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path

import fkmob
import fkmob._core as core
from fkmob import discover_daily_motifs_from_agents, waiting_times

expected = Path(os.environ["FKMOB_DIR"]).resolve()
package_path = Path(fkmob.__file__).resolve()
core_path = Path(core.__file__).resolve()

if not package_path.is_relative_to(expected):
    raise SystemExit(
        f"fkmob imported from {package_path}, expected a path under {expected}"
    )
if not core_path.is_relative_to(expected):
    raise SystemExit(
        f"fkmob._core imported from {core_path}, expected a path under {expected}"
    )
if not callable(waiting_times) or not callable(discover_daily_motifs_from_agents):
    raise SystemExit("Current fkmob public measure APIs are unavailable")

print(f"fkmob: {package_path}")
print(f"fkmob._core: {core_path}")
print("Public measure APIs: OK")
PY

echo "Done"
