#!/usr/bin/env bash
set -euo pipefail

echo "[setup] creating venv if missing..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

VENV_PY="./.venv/bin/python"

echo "[deps] installing requirements..."
"${VENV_PY}" -m pip install -r requirements.txt

echo "[run] launching app..."
"${VENV_PY}" src/app.py