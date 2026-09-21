#!/bin/bash
# Unpack the staged LightgenBench tars into the flat tree trellis2/datasets/lightgen_slat.py reads.
# Called by script/alliance/lightgenbench_v1.sh inside the job, with <root> under $SLURM_TMPDIR
# (fir's /scratch file quota cannot hold 36,826 loose files per tree; a node-local disk can).
#
#   bash script/alliance/lightgenbench_unpack.sh vae <root>    # the VAE and the encode read these
#       <root>/emission_voxels_256/<uuid>.vxz, <root>/pbr_voxels_256/<uuid>.vxz
#   bash script/alliance/lightgenbench_unpack.sh dit <root>    # the DiT reads these
#       <root>/thumbnails/<uuid>.png
#       <root>/shape_latents/shape_enc_next_dc_f16c32_fp16_256/<uuid>.npz
#       <root>/pbr_latents/tex_enc_next_dc_f16c32_fp16_256/<uuid>.npz
#       <root>/emission_latents/<LATENT_NAME>/{<uuid>.npz,normalization_stats.json}
#   both: <root>/df_lightgenbench.parquet, <root>/data_splits_lightgenbench.json
#
# Sources: $HFD = what lightgenbench_stage.sh downloaded (release members are <uuid>/<file>, so a
# tar --transform renames them on the way out); $LAT holds latents.tar (shape + PBR latents of the
# stock TRELLIS.2-4B encoders, jupiter datasets/dataset_release_latents/tars/) and, for the DiT,
# emission_latents_<LATENT_NAME>.tar written by the encode stage. Every tree is counted: a short
# tree stops the job here and not at some sample hours in.
set -euo pipefail
STAGE=${1:?vae|dit}; ROOT=${2:?destination root}
HFD=${HFD:-$HOME/scratch/lightgen/data/lightgenbench}
LAT=${LAT:-$HOME/scratch/lightgen/data/lightgenbench_trellis2}
LATENT_NAME=${LATENT_NAME:-emission_pbr2emission_lightgenbench_v1}
N=${N:-36826}                 # every shape of the release; override only to test on a subset
T0=$(date +%s)

FREE_GB=$(df -BG --output=avail "$(dirname "$ROOT")" | tail -1 | tr -dc 0-9)
[ "${FREE_GB}" -ge 40 ] || { echo "FATAL: only ${FREE_GB} GB free under $(dirname "$ROOT"), need 40"; exit 1; }
mkdir -p "$ROOT"
cp "$HFD/trellis2_index/df_lightgenbench.parquet" "$HFD/trellis2_index/data_splits_lightgenbench.json" "$ROOT/"

count() {  # <dir> <extension>: no glob, 36,826 paths overflow argv
    ls "$1" | grep -c "\.$2\$" || true
}
need() {   # <dir> <extension>
    local n; n=$(count "$1" "$2"); echo "[unpack] $1: $n (expected $N)"
    [ "$n" -eq "$N" ] || { echo "FATAL: $1 holds $n .$2 files, expected $N"; exit 1; }
}

case "$STAGE" in
  vae)
    mkdir -p "$ROOT/emission_voxels_256" "$ROOT/pbr_voxels_256"
    find "$HFD"/data/{train,val,test}/voxels -name 'voxels-*.tar' | sort \
      | xargs -P 8 -I{} tar xf {} -C "$ROOT" \
          --transform 's|^\([^/]*\)/emission_voxels\.vxz$|emission_voxels_256/\1.vxz|' \
          --transform 's|^\([^/]*\)/pbr_voxels\.vxz$|pbr_voxels_256/\1.vxz|'
    need "$ROOT/emission_voxels_256" vxz
    need "$ROOT/pbr_voxels_256" vxz
    ;;
  dit)
    mkdir -p "$ROOT/thumbnails"
    find "$HFD"/data/{train,val,test}/thumbnail -name 'thumbnail-*.tar' | sort \
      | xargs -P 8 -I{} tar xf {} -C "$ROOT/thumbnails" --transform 's|^\([^/]*\)/thumbnail\.png$|\1.png|'
    tar xf "$LAT/latents.tar" -C "$ROOT" shape_latents pbr_latents
    tar xf "$LAT/emission_latents_${LATENT_NAME}.tar" -C "$ROOT"
    need "$ROOT/thumbnails" png
    need "$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_256" npz
    need "$ROOT/pbr_latents/tex_enc_next_dc_f16c32_fp16_256" npz
    need "$ROOT/emission_latents/${LATENT_NAME}" npz
    [ -f "$ROOT/emission_latents/${LATENT_NAME}/normalization_stats.json" ] || { echo "FATAL: normalization_stats.json missing"; exit 1; }
    ;;
  *) echo "unknown stage '$STAGE'"; exit 2 ;;
esac
echo "[unpack] $STAGE -> $ROOT in $(( $(date +%s) - T0 )) s; $(du -sh "$ROOT" | cut -f1)"
