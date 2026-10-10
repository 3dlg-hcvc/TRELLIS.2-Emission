#!/bin/bash
# bash script/internal/gate/infer.sh <tree> <out_dir> [<shas_file>]
# The paper's seed-0 inference, the command `evaluation/newdata_eval/sweeps/job_dit.sh test
# last-step0050000.ckpt 0` builds, on <tree> (a tree from tree.sh, or a checkout with outputs/ and the
# paper checkpoints), written to <out_dir>/pred_voxels/<sha>.npz; the log goes to <out_dir>.log.
# Shapes default to the 204 agentic_clean test shapes. Prints GATE_DONE <n> only when every requested
# shape has an npz (the predictor skips a shape with missing inputs and still exits 0), else exits 5.
# Overrides: GATE_NE (the parent's evaluation/newdata_eval), GATE_LOCK (a lock file, or "none"),
# GATE_LOCK_WAIT (seconds, default 10800).
set -euo pipefail
T=$(realpath -e "${1:?tree}"); OUT=$(realpath -m "${2:?out dir}")
NE=$(realpath -e "${GATE_NE:-/local-scratch/localhome/dya78/code/lightgen/evaluation/newdata_eval}")
SHAS=$(realpath -e "${3:-$NE/eval_sets/test_agentic_clean.txt}")
DIT=$T/outputs/emission_dit_pbr2emission_74k_v3_agentic; VAE=$T/outputs/emission_vae_pbr2emission_74k_v3_agentic
for f in "$DIT/ckpts/last-step0050000.ckpt" "$DIT/config.workstation.yaml" "$VAE/ckpts/step0035200-0.0058.ckpt" \
         "$VAE/config.yaml" "$T/data_toolkit/predict_emission_voxels_twostream.py"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
mapfile -t IDS < <(grep -v '^[[:space:]]*$' "$SHAS")
[ "${#IDS[@]}" -gt 0 ] || { echo "empty shapes file $SHAS" >&2; exit 2; }
JOINED=$(IFS=,; echo "${IDS[*]}")
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"; echo "[gpu-lock] waiting $(date +%T)"
  flock -w "${GATE_LOCK_WAIT:-10800}" 9 || { echo "[gpu-lock] timeout"; exit 3; }; echo "[gpu-lock] acquired $(date +%T)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$OUT"; mkdir -p "$OUT"; [ -f "$T/.gate_commit" ] && cp "$T/.gate_commit" "$OUT/"
cd "$T"
printf -v CMD '%q ' python data_toolkit/predict_emission_voxels_twostream.py \
  --dit_ckpt "$DIT/ckpts/last-step0050000.ckpt" --dit_cfg "$DIT/config.workstation.yaml" \
  --vae_ckpt "$VAE/ckpts/step0035200-0.0058.ckpt" --vae_cfg "$VAE/config.yaml" \
  --data_root "$NE/out/data_root" --thumbnail_dir "$NE/out/data_root/thumbnails" \
  --shas "$JOINED" --n all --guidance 3.0 --steps 50 --seed 0 --out_dir "$OUT"
conda run --no-capture-output -n trellis2 bash -c "$CMD" 2>&1 | tee "$OUT.log"
miss=0; for id in "${IDS[@]}"; do [ -f "$OUT/pred_voxels/$id.npz" ] || { echo "NO OUTPUT $id"; miss=$((miss+1)); }; done
[ "$miss" = 0 ] || { echo "GATE_INCOMPLETE $miss of ${#IDS[@]} shapes have no npz" >&2; exit 5; }
echo "GATE_DONE ${#IDS[@]}"
