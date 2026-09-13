#!/usr/bin/env bash
# Copy run outputs (logs, checkpoints, videos) from the cluster into ./runs.
# Extra arguments go to rsync, e.g.  scripts/pull_runs.sh --include '*/' --include '*.mp4' --exclude '*'
set -euo pipefail

HOST="${RX_HOST:-futurama}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az "$@" "$HOST:roboticsexp/runs/" "$REPO_ROOT/runs/"
echo "pulled $HOST:roboticsexp/runs/ -> $REPO_ROOT/runs/"
