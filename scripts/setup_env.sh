#!/usr/bin/env bash
# Phase 0 environment setup: creates a FRESH .venv on Python 3.12 and installs
# the project (editable) + dev extras.
#
# Why Python 3.12: the spaCy 3.7.5 -> thinc 8.2.5 -> blis 0.7.11 chain and
# scispaCy 0.6.2 have reliable cp312 ARM64 macOS wheels; Python 3.13 has no
# wheels for blis (source build fails) and scispaCy declares requires_python<3.13.
# pyproject.toml also gates this via requires-python = ">=3.10,<3.13".
set -euo pipefail
cd "$(dirname "$0")/.."

# Allow override: SGRAG_PYTHON=/path/to/python ./scripts/setup_env.sh
PYBIN="${SGRAG_PYTHON:-python3.12}"

if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYBIN' not found." >&2
  echo "Install Python 3.12 first, then re-run this script:" >&2
  echo "  macOS:  brew install python@3.12   (then ensure python3.12 is on PATH)" >&2
  echo "  Ubuntu: sudo apt install python3.12 python3.12-venv" >&2
  echo "  or:     SGRAG_PYTHON=/path/to/python3.12 ./scripts/setup_env.sh" >&2
  exit 1
fi

PYVER="$("$PYBIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYVER" != "3.12" ]; then
  echo "ERROR: $PYBIN is Python $PYVER, but this project requires Python 3.12." >&2
  exit 1
fi
echo "Using $PYBIN (Python $PYVER)"

# From-scratch build: remove any existing venv (e.g. one created with 3.13).
if [ -d .venv ]; then
  echo "Removing existing .venv for a clean rebuild..."
  rm -rf .venv
fi

"$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python --version
pip --version
python -m pip install --upgrade pip
pip install -e ".[dev]"
echo "done. Activate with: source .venv/bin/activate"
