#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKMOB2_DIR="${ROOT_DIR}/../skmob2"
SKMOB_VIZ_DIR="${ROOT_DIR}/../skmob-viz"
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
require_path "${ROOT_DIR}/uv.lock" "project uv.lock"
require_path "${VENV_DIR}" "project uv venv"
require_path "${SKMOB2_DIR}/pyproject.toml" "skmob2 package"
require_path "${SKMOB2_DIR}/Cargo.toml" "skmob2 Rust manifest"
require_path "${SKMOB_VIZ_DIR}/pyproject.toml" "skmob-viz package"
require_path "${SKMOB_VIZ_DIR}/Cargo.toml" "skmob-viz Rust manifest"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available on PATH" >&2
    exit 1
fi

unset CONDA_PREFIX
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

echo "Rebuilding editable local packages into ${VENV_DIR}"
echo "  skmob2:    ${SKMOB2_DIR}"
echo "  skmob-vis: ${SKMOB_VIZ_DIR}"

uv sync \
    --project "${ROOT_DIR}" \
    --reinstall-package skmob2 \
    --reinstall-package skmob-vis

echo "Verifying imports"
"${VENV_DIR}/bin/python" - <<'PY'
import skmob2
import skmob_vis

print(f"skmob2: {skmob2.__file__}")
print(f"skmob_vis: {skmob_vis.__file__}")
PY

echo "Done"
