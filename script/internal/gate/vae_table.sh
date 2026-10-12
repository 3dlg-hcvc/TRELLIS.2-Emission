#!/bin/bash
# bash script/internal/gate/vae_table.sh <tree> <out_dir> [<shapes_file>]
# The VAE table (data_toolkit/eval_vae_recon.py, arms pretrained_pad and ours_pbr2emission) on <tree>
# (a tree from tree.sh), written to <out_dir>/per_shape.csv; the log goes to <out_dir>.log. Shapes
# default to the 204 agentic_clean test shapes. Overrides: GATE_NE, GATE_LOCK, GATE_LOCK_WAIT, GATE_ACTIVATE (as infer.sh).
set -euo pipefail
T=$(realpath -e "${1:?tree}"); OUT=$(realpath -m "${2:?out dir}")
NE=$(realpath -e "${GATE_NE:-/local-scratch/localhome/dya78/code/lightgen/evaluation/newdata_eval}")
SHAS=$(realpath -e "${3:-$NE/eval_sets/test_agentic_clean.txt}")
[ -f "$T/data_toolkit/eval_vae_recon.py" ] || { echo "missing $T/data_toolkit/eval_vae_recon.py" >&2; exit 2; }
grep -q '[^[:space:]]' "$SHAS" || { echo "empty shapes file $SHAS" >&2; exit 2; }
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"; echo "[gpu-lock] waiting $(date +%T)"
  flock -w "${GATE_LOCK_WAIT:-10800}" 9 || { echo "[gpu-lock] timeout"; exit 3; }; echo "[gpu-lock] acquired $(date +%T)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"; mkdir -p "$OUT"; [ -f "$T/.gate_commit" ] && cp "$T/.gate_commit" "$OUT/"
cd "$T"
printf -v CMD '%q ' python data_toolkit/eval_vae_recon.py --shapes "$SHAS" \
  --sets pretrained,pbr2emission --arms pretrained_pad,ours_pbr2emission --out_dir "$OUT"
if [ -n "${GATE_ACTIVATE:-}" ]; then RUN=(bash -c "source $(printf %q "$GATE_ACTIVATE") && $CMD")
else RUN=(conda run --no-capture-output -n trellis2 bash -c "$CMD"); fi
"${RUN[@]}" 2>&1 | tee "$OUT.log"
[ -s "$OUT/per_shape.csv" ] || { echo "no per_shape.csv" >&2; exit 4; }
echo "VAE_TABLE_DONE $(($(wc -l < "$OUT/per_shape.csv") - 1)) rows"
