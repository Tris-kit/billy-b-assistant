#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"
SERVICE_NAME="billy.service"
SERVICE_WAS_ACTIVE=0

cleanup() {
    if [[ "$SERVICE_WAS_ACTIVE" -eq 1 ]]; then
        echo
        echo "Starting $SERVICE_NAME again..."
        sudo systemctl start "$SERVICE_NAME"
    fi
}

trap cleanup EXIT

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing virtual environment at $PYTHON_BIN"
    echo "Run the Python setup first:"
    echo "  python3 -m venv venv"
    echo "  source ./venv/bin/activate"
    echo "  pip3 install -r ./requirements.txt"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import dotenv, numpy, lgpio" >/dev/null 2>&1; then
    echo "The virtual environment is missing required packages."
    echo "Activate it and install requirements:"
    echo "  source ./venv/bin/activate"
    echo "  pip3 install -r ./requirements.txt"
    exit 1
fi

if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$SERVICE_NAME"; then
    SERVICE_WAS_ACTIVE=1
    echo "Stopping $SERVICE_NAME so GPIO is free..."
    sudo systemctl stop "$SERVICE_NAME"
fi

echo "Running Billy motor diagnostic..."
"$PYTHON_BIN" "$ROOT_DIR/test/motor_diagnostic.py" "$@"
