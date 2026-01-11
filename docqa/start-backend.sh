#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run ./setup.sh first." >&2
  exit 1
fi

uv run uvicorn backend.app.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT"
