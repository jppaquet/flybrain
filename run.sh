#!/usr/bin/env bash
# Starts the local server (connectome dashboard + 3D fly). Arguments go to dashboard.py,
# e.g. ./run.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -f malecns_graph.npz ]; then
  echo "Not set up yet: run ./setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python -u dashboard.py "$@"
