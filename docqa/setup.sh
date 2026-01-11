#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  uv venv
fi

uv sync

if [ -d "frontend" ]; then
  if [ ! -d "frontend/node_modules" ]; then
    (cd frontend && npm install)
  fi
fi

echo "Setup complete. Run ./start-backend.sh and ./start-frontend.sh in separate terminals."
