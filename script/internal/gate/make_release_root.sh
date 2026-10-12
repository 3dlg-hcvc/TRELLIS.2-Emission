#!/bin/bash
# bash script/internal/gate/make_release_root.sh <dest> [<shas_file>]
# Copies the release inputs of the given shapes (default release_test20.txt: the first 20 of the release
# test split) into <dest> in the layout predict_emission_voxels_twostream.py reads, as
# trellis2_release/infer_local.sh lays them out (it links them; this copies, so the root can travel to
# another host): emission_voxels_256/<id>.vxz, thumbnails/<id>.png,
# pbr_latents/tex_enc_next_dc_f16c32_fp16_256/<id>.npz, shape_latents/shape_enc_next_dc_f16c32_fp16_256/<id>.npz,
# and shapes.txt. Sources: jupiter datasets/dataset_release{,_latents} (GATE_JUP overrides the root).
set -euo pipefail
D="${1:?dest}"; S="${2:-$(dirname "$0")/release_test20.txt}"
JUP=${GATE_JUP:-/cs/3dlg-jupiter-project/lightgen}; REL=$JUP/datasets/dataset_release; LAT=$JUP/datasets/dataset_release_latents
[ -e "$D" ] && { echo "$D exists" >&2; exit 2; }
P=$D/pbr_latents/tex_enc_next_dc_f16c32_fp16_256; Q=$D/shape_latents/shape_enc_next_dc_f16c32_fp16_256
mkdir -p "$D/emission_voxels_256" "$D/thumbnails" "$P" "$Q"
grep -v '^[[:space:]]*$' "$S" > "$D/shapes.txt"
while read -r u; do
  cp "$REL/$u/emission_voxels.vxz" "$D/emission_voxels_256/$u.vxz"; cp "$REL/$u/thumbnail.png" "$D/thumbnails/$u.png"
  cp "$LAT/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/$u.npz" "$P/$u.npz"
  cp "$LAT/shape_latents/shape_enc_next_dc_f16c32_fp16_256/$u.npz" "$Q/$u.npz"
done < "$D/shapes.txt"
n=$(grep -c . "$D/shapes.txt"); for d in emission_voxels_256 thumbnails; do [ "$(ls "$D/$d" | wc -l)" = "$n" ] || { echo "short $d" >&2; exit 3; }; done
echo "RELEASE_ROOT_DONE $D $n shapes"
