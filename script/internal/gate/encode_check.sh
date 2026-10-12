#!/bin/bash
# bash script/internal/gate/encode_check.sh <tree> <out_dir> [<shas_file>]
# The encode check: data_toolkit/encode_emission_twostream_dataset.py with the paper VAE
# (outputs/emission_vae_pbr2emission_74k_v3_agentic, step0035200-0.0058.ckpt, the encoder that made the
# paper DiT's latents) on the given shapes (default subset20.txt), --aggregate_stats. The root it reads
# and writes is <out_dir>: emission_voxels_256/ links to the paper gate inputs, index.parquet lists the
# shapes; the latents land in <out_dir>/emission_latents/gate/. Compare two runs with compare_encode.py.
# Overrides: GATE_NE, GATE_LOCK, GATE_LOCK_WAIT, GATE_ACTIVATE (as infer.sh). Prints ENCODE_DONE <n>.
set -euo pipefail
T=$(realpath -e "${1:?tree}"); OUT=$(realpath -m "${2:?out dir}")
NE=$(realpath -e "${GATE_NE:-/local-scratch/localhome/dya78/code/lightgen/evaluation/newdata_eval}")
SHAS=$(realpath -e "${3:-$(dirname "$(readlink -f "$0")")/subset20.txt}")
VAE=$T/outputs/emission_vae_pbr2emission_74k_v3_agentic
for f in "$VAE/ckpts/step0035200-0.0058.ckpt" "$VAE/config.yaml" "$T/data_toolkit/encode_emission_twostream_dataset.py"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
mapfile -t IDS < <(grep -v '^[[:space:]]*$' "$SHAS"); [ "${#IDS[@]}" -gt 0 ] || { echo "empty shapes file" >&2; exit 2; }
[ -e "$OUT" ] && { echo "$OUT exists; use a new out dir" >&2; exit 2; }
mkdir -p "$OUT/emission_voxels_256"
for u in "${IDS[@]}"; do ln -s "$NE/out/data_root/emission_voxels_256/$u.vxz" "$OUT/emission_voxels_256/$u.vxz"; done
[ "$(find "$OUT/emission_voxels_256" -xtype l | wc -l)" = 0 ] || { echo "dangling voxel links" >&2; exit 2; }
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"; echo "[gpu-lock] waiting $(date +%T)"
  flock -w "${GATE_LOCK_WAIT:-10800}" 9 || { echo "[gpu-lock] timeout"; exit 3; }; echo "[gpu-lock] acquired $(date +%T)"
fi
[ -f "$T/.gate_commit" ] && cp "$T/.gate_commit" "$OUT/"
cd "$T"
printf -v CMD '%q ' python - "$OUT/index.parquet" "${IDS[@]}"
PARQ='import sys, pandas as pd; pd.DataFrame({"success": [True] * (len(sys.argv) - 2)}, index=pd.Index(sys.argv[2:], name="sha256")).to_parquet(sys.argv[1])'
printf -v ENC '%q ' python data_toolkit/encode_emission_twostream_dataset.py --data_root "$OUT" --parquet "$OUT/index.parquet" \
  --vae_ckpt "$VAE/ckpts/step0035200-0.0058.ckpt" --vae_cfg "$VAE/config.yaml" --latent_name gate --resolution 256 --aggregate_stats
ALL="python -c $(printf %q "$PARQ") $(printf '%q ' "$OUT/index.parquet" "${IDS[@]}") && $ENC"
if [ -n "${GATE_ACTIVATE:-}" ]; then RUN=(bash -c "source $(printf %q "$GATE_ACTIVATE") && $ALL")
else RUN=(conda run --no-capture-output -n trellis2 bash -c "$ALL"); fi
"${RUN[@]}" 2>&1 | tee "$OUT.log"
n=$(find "$OUT/emission_latents/gate" -name '*.npz' | wc -l)
[ "$n" = "${#IDS[@]}" ] && [ -s "$OUT/emission_latents/gate/normalization_stats.json" ] || { echo "ENCODE_INCOMPLETE $n of ${#IDS[@]}" >&2; exit 5; }
echo "ENCODE_DONE $n"
