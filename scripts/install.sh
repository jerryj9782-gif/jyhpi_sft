#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3.11}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[test]'
echo 'Installed in the project .venv; activate with: source .venv/bin/activate'
