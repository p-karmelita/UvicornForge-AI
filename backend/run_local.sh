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

# Run uvicorn
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
