#!/usr/bin/env bash
# Mirror this repo to ~/roboticsexp on the cluster. runs/ is excluded, so a sync
# never deletes results that exist only on the cluster.
set -euo pipefail

HOST="${RX_HOST:-futurama}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude '.venv/' --exclude 'runs/' --exclude '__pycache__/' --exclude '.DS_Store' \
  "$REPO_ROOT/" "$HOST:roboticsexp/"
ssh -q "$HOST" 'mkdir -p roboticsexp/runs/slurm'
echo "synced $REPO_ROOT -> $HOST:roboticsexp/"
