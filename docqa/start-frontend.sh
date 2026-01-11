#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [ ! -d "frontend" ]; then
  echo "frontend directory not found." >&2
  exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Missing frontend/node_modules. Run ./setup.sh first." >&2
  exit 1
fi

cd frontend
npm run dev
