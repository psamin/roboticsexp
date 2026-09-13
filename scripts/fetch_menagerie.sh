#!/usr/bin/env bash
# Fetch only the Menagerie files PandaPickCube needs (franka_emika_panda/), at the
# commit playground pins, into the path playground checks before cloning. Without
# this, the first env load clones all of mujoco_menagerie (~590 MB).
#
# Run inside the project env:  uv run bash scripts/fetch_menagerie.sh
set -euo pipefail

read -r DEST SHA < <(python -c 'from mujoco_playground._src import mjx_env as m; print(m.MENAGERIE_PATH, m.MENAGERIE_COMMIT_SHA)')
if [ -e "$DEST" ]; then
  echo "menagerie already present: $DEST"
  exit 0
fi

TMP="$DEST.partial"
rm -rf "$TMP"
mkdir -p "$TMP"
git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$TMP" sparse-checkout set franka_emika_panda
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$SHA"
git -C "$TMP" checkout -q FETCH_HEAD
mv "$TMP" "$DEST"
echo "menagerie $SHA (franka_emika_panda only) -> $DEST"
