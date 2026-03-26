#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-$SRC_DIR/../indent_discovery_v2_shareable}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

rsync -a \
  --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'workspace/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.coverage' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude '.DS_Store' \
  --exclude '.streamlit/secrets.toml' \
  "$SRC_DIR/" "$DEST_DIR/"

cat <<EOF
Shareable copy prepared at:
  $DEST_DIR

Next steps:
  cd "$DEST_DIR"
  git init
  git add .
  git commit -m "Initial commit"
EOF
