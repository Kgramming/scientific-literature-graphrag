#!/usr/bin/env bash
# Phase 0 environment setup: creates .venv and installs the project (editable) + dev extras.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
  echo "created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
echo "done. Activate with: source .venv/bin/activate"
