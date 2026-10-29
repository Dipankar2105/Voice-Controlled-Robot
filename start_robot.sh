#!/usr/bin/env bash
# ==============================================================================
# SS_PROJECT_VCR_FRESH — Authoritative Single-Command Robot Launcher
# ==============================================================================
# Guarantees:
# 1. Locates project root independently of invocation directory.
# 2. Automatically selects and activates the local Python virtual environment.
# 3. Performs a strictly non-invasive configuration and asset preflight check.
# 4. Fails closed with descriptive messages if assets or dependencies are missing.
# 5. Starts the authoritative runtime and local telemetry dashboard exactly once.
# 6. Never automatically moves motors during startup (motors default to stopped/disabled).
# 7. Handles SIGINT (Ctrl+C) and SIGTERM gracefully, releasing all physical resources.
# ==============================================================================

set -eo pipefail

# 1. Locate Project Root Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "  SS_PROJECT_VCR_FRESH — Autonomous Voice-Controlled Robot Launcher"
echo "  Project Root: ${SCRIPT_DIR}"
echo "======================================================================"

# 2. Select and Activate Python Virtual Environment
VENV_PATH=""
if [ -d "${SCRIPT_DIR}/.venv" ]; then
    VENV_PATH="${SCRIPT_DIR}/.venv"
elif [ -d "${SCRIPT_DIR}/venv" ]; then
    VENV_PATH="${SCRIPT_DIR}/venv"
fi

if [ -n "${VENV_PATH}" ]; then
    echo "[INFO] Activating virtual environment: ${VENV_PATH}"
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="${VENV_PATH}/bin/python3"
else
    echo "[WARN] No local virtual environment found (.venv or venv)."
    echo "       Falling back to system python3. (Using a virtualenv is strongly recommended)"
    PYTHON_BIN="python3"
fi

# Verify Python version
if ! command -v "${PYTHON_BIN}" &> /dev/null; then
    echo "[ERROR] Python 3 executable '${PYTHON_BIN}' not found in PATH." >&2
    exit 1
fi

PY_VER="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[INFO] Using Python version: ${PY_VER}"

# 3. Perform Non-Invasive Configuration & Asset Preflight Check
echo "[INFO] Running static configuration and model asset preflight check..."
if "${PYTHON_BIN}" -m app.main --check-config; then
    :
else
    PREFLIGHT_CODE=$?
    echo "[FATAL] Preflight check FAILED with exit code ${PREFLIGHT_CODE}." >&2
    echo "        Missing model files, invalid pin mapping, or malformed YAML configuration." >&2
    echo "        Robot will NOT start. Exiting cleanly." >&2
    exit "${PREFLIGHT_CODE}"
fi

echo "[INFO] Configuration and asset preflight check PASSED."
echo "[INFO] Launching authoritative runtime server..."
echo "       (Motors default stopped; zero physical actuation on startup)"
echo "----------------------------------------------------------------------"

# 4. Load operator token if present and not already set in environment
if [ -z "${ROBOT_API_TOKEN:-}" ] && [ -f "${SCRIPT_DIR}/.robot_token" ]; then
    export ROBOT_API_TOKEN="$(cat "${SCRIPT_DIR}/.robot_token" | tr -d '\r\n')"
fi

# 5. Launch Unified Application Server (Exec replaces shell for clean signal forwarding)
LAUNCH_ARGS=("$@")
if [ "$#" -eq 0 ]; then
    LAUNCH_ARGS=("--mode" "REAL_HARDWARE")
fi

exec "${PYTHON_BIN}" -m app.main --serve "${LAUNCH_ARGS[@]}"
