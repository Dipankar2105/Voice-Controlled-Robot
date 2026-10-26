#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="/home/admin/SS_Project_VCR_FRESH"
TOKEN_FILE="$BASE_DIR/.robot_token"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: Token file $TOKEN_FILE not found." >&2
    exit 1
fi
export ROBOT_API_TOKEN="$(cat "$TOKEN_FILE" | tr -d '\r\n')"
cd "$BASE_DIR"
exec "$BASE_DIR/.venv/bin/python" -m app.main --serve --mode REAL_HARDWARE --port 8000
