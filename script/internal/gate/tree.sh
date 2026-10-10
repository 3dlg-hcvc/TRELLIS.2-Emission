#!/bin/bash
# bash script/internal/gate/tree.sh <commit> <dest>
# Unpacks `git archive <commit>` into <dest> (must not exist) and links the main checkout's ignored
# outputs/ and data/ into it, so the gate runs the committed code with the local checkpoints and data
# (the configs and eval_vae_recon.py read both relative to the working directory). Run from a checkout
# of this repo. Overrides: GATE_REPO (the repo to archive), GATE_MAIN (where outputs/ and data/ live).
set -euo pipefail
C="${1:?commit}"; D="${2:?dest}"
R=${GATE_REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
MAIN=$(realpath -e "${GATE_MAIN:-/local-scratch/localhome/dya78/code/lightgen/TRELLIS2}")
[ -d "$MAIN/outputs" ] && [ -d "$MAIN/data" ] || { echo "no outputs/ or data/ in $MAIN" >&2; exit 2; }
[ -e "$D" ] && { echo "$D exists" >&2; exit 2; }
SHA=$(git -C "$R" rev-parse --verify "$C^{commit}")
mkdir -p "$D"; git -C "$R" archive "$SHA" | tar -x -C "$D"
ln -s "$MAIN/outputs" "$D/outputs"; ln -s "$MAIN/data" "$D/data"
echo "$SHA" > "$D/.gate_commit"
echo "TREE_DONE $D $SHA"
