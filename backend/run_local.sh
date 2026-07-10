#!/usr/bin/env bash
# Helper script to run the backend locally for development.
# If ROCm is available and torch is ROCm-enabled, set USE_ROCM=1 to attempt GPU usage.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="$PWD"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found; installing runtime dependencies into a venv..."
  python -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install fastapi uvicorn pydantic
fi

# Run uvicorn
python -m uvicorn app:app --host 0.0.0.0 --port 8000
