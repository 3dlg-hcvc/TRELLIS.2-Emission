#!/bin/bash
# bash script/internal/gate/make_check_root.sh <dest> [<python>]
# The training check's data root: the 12 release shapes of check_root_splits.json (8 train, 2 val, 2 test)
# copied from jupiter in the layout script/internal/alliance/lightgenbench_unpack.sh builds, so the VAE
# and the DiT train on it with the release configs: emission_voxels_256/, pbr_voxels_256/, thumbnails/,
# shape_latents/shape_enc_next_dc_f16c32_fp16_256/, pbr_latents/tex_enc_next_dc_f16c32_fp16_256/,
# emission_latents/emission_pbr2emission_lightgenbench_v1/ (the release VAE's latents and their
# normalization_stats.json), and the index files data_toolkit/build_lightgenbench_index.py writes
# (run with <python>, default the trellis2 env's). Copies, so the root can travel to another host.
# GATE_JUP overrides jupiter's root.
set -euo pipefail
D=$(realpath -m "${1:?dest}"); PY=${2:-/localhome/dya78/software/miniconda3/envs/trellis2/bin/python}
HERE=$(dirname "$(readlink -f "$0")"); REPO=$(cd "$HERE/../../.." && pwd)
JUP=${GATE_JUP:-/cs/3dlg-jupiter-project/lightgen}; REL=$JUP/datasets/dataset_release; LAT=$JUP/datasets/dataset_release_latents
NAME=emission_pbr2emission_lightgenbench_v1
ETAR=$JUP/outputs/emission_vae_pbr2emission_lightgenbench_v1/emission_latents/emission_latents_$NAME.tar
[ -e "$D" ] && { echo "$D exists" >&2; exit 2; }
P=$D/pbr_latents/tex_enc_next_dc_f16c32_fp16_256; Q=$D/shape_latents/shape_enc_next_dc_f16c32_fp16_256
mkdir -p "$D/emission_voxels_256" "$D/pbr_voxels_256" "$D/thumbnails" "$P" "$Q"
cp "$HERE/check_root_splits.json" "$D/splits.json"
mapfile -t IDS < <("$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); [print(u) for k in ("train","val","test") for u in s[k]]' "$D/splits.json")
for u in "${IDS[@]}"; do
  cp "$REL/$u/emission_voxels.vxz" "$D/emission_voxels_256/$u.vxz"; cp "$REL/$u/pbr_voxels.vxz" "$D/pbr_voxels_256/$u.vxz"
  cp "$REL/$u/thumbnail.png" "$D/thumbnails/$u.png"
  cp "$LAT/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/$u.npz" "$P/$u.npz"
  cp "$LAT/shape_latents/shape_enc_next_dc_f16c32_fp16_256/$u.npz" "$Q/$u.npz"
done
tar xf "$ETAR" -C "$D" "emission_latents/$NAME/normalization_stats.json" \
  $(printf "emission_latents/$NAME/%s.npz " "${IDS[@]}")
(cd "$REPO" && "$PY" data_toolkit/build_lightgenbench_index.py --splits "$D/splits.json" --out-dir "$D" \
  && "$PY" data_toolkit/build_lightgenbench_index.py --splits "$D/splits.json" --out-dir "$D" --check)
n=${#IDS[@]}
for d in emission_voxels_256 pbr_voxels_256 thumbnails "$P" "$Q" "emission_latents/$NAME"; do
  c=$(find "$D/${d#$D/}" -maxdepth 1 -type f ! -name normalization_stats.json | wc -l)
  [ "$c" = "$n" ] || { echo "short $d: $c of $n" >&2; exit 3; }; done
echo "CHECK_ROOT_DONE $D $n shapes"
