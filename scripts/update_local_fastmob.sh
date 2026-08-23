#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTMOB_DIR="${ROOT_DIR}/../fastmob"
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
require_path "${FASTMOB_DIR}/pyproject.toml" "fastmob package"
require_path "${FASTMOB_DIR}/Cargo.toml" "fastmob Rust manifest"
require_path "${FASTMOB_DIR}/fastmob-py/Cargo.toml" "fastmob Python Rust manifest"

MATURIN_CMD=()
if [[ -x "${VENV_DIR}/bin/maturin" ]]; then
    MATURIN_CMD=("${VENV_DIR}/bin/maturin")
elif [[ -x "${FASTMOB_DIR}/.venv/bin/maturin" ]]; then
    MATURIN_CMD=("${FASTMOB_DIR}/.venv/bin/maturin")
elif command -v maturin >/dev/null 2>&1; then
    MATURIN_CMD=(maturin)
elif command -v uv >/dev/null 2>&1; then
    MATURIN_CMD=(uv tool run --from "maturin>=1.13,<2.0" maturin)
else
    echo "Unable to find maturin." >&2
    echo "Install maturin in ${VENV_DIR}, ${FASTMOB_DIR}/.venv, or on PATH." >&2
    exit 1
fi

echo "Building fastmob into ${VENV_DIR} with maturin develop --release"
echo "  source: ${FASTMOB_DIR}"
echo "  maturin: ${MATURIN_CMD[*]}"

(
    cd "${FASTMOB_DIR}"
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="${VENV_DIR}" \
        PATH="${VENV_DIR}/bin:${PATH}" \
        "${MATURIN_CMD[@]}" develop --release
)

echo "Verifying the local fastmob build"
FASTMOB_DIR="${FASTMOB_DIR}" "${VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path

import fastmob
import fastmob._core as core
from fastmob import waiting_times
from fastmob.measures.individual import daily_motifs

expected = Path(os.environ["FASTMOB_DIR"]).resolve()
package_path = Path(fastmob.__file__).resolve()
core_path = Path(core.__file__).resolve()

if not package_path.is_relative_to(expected):
    raise SystemExit(
        f"fastmob imported from {package_path}, expected a path under {expected}"
    )
if not core_path.is_relative_to(expected):
    raise SystemExit(
        f"fastmob._core imported from {core_path}, expected a path under {expected}"
    )
if not callable(waiting_times) or not callable(daily_motifs):
    raise SystemExit("Current fastmob public measure APIs are unavailable")

print(f"fastmob: {package_path}")
print(f"fastmob._core: {core_path}")
print("Public measure APIs: OK")
PY

echo "Done"
