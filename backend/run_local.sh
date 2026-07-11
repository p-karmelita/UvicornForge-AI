#!/usr/bin/env bash
# Helper script to run the backend locally for development.
# If ROCm is available and torch is ROCm-enabled, set USE_ROCM=1 to attempt GPU usage.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$PWD"

if [ ! -d ".venv" ]; then
  echo "Creating backend virtual environment..."
  python -m venv .venv
fi

. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

PORT=8000
if command -v ss >/dev/null 2>&1; then
  OLD_PID="$(ss -tlnp | awk -v port=":${PORT}" '$4 ~ port {print $6}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)"
  if [ -n "${OLD_PID:-}" ]; then
    echo "Stopping previous server on port ${PORT} (pid ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 1
  fi
fi

python -m uvicorn app:app --host 0.0.0.0 --port "${PORT}" --reload
