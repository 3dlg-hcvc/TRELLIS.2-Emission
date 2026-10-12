#!/bin/bash
#SBATCH -J lightgenbench_stage_trellis2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Download the part of the LightgenBench release that TRELLIS.2-Emission trains on, verify it, and
# build the two index files the loader reads. Run ONCE per cluster, where the internet is reachable:
# on the login node (nibi, rorqual and trillium compute nodes have none), or as a CPU job on fir.
#
#   bash script/internal/alliance/lightgenbench_stage.sh                                  # a login node, from the repo root
#   sbatch --account=def-msavva_cpu script/internal/alliance/lightgenbench_stage.sh        # fir
#
# What lands in $DEST (about 80 files, 17 GB, so the /scratch file quota does not notice):
#   splits.json, checksums.sha256
#   data/{train,val,test}/voxels/voxels-*.tar          39 tars, members <uuid>/{emission,pbr}_voxels.vxz
#   data/{train,val,test}/thumbnail/thumbnail-*.tar    39 tars, members <uuid>/thumbnail.png
#   trellis2_index/{df_lightgenbench.parquet,data_splits_lightgenbench.json}
# The atlas and multiview tars are not downloaded (TEXGen's and Hunyuan3D-Paint's input). $DEST is
# the directory the TEXGen lane stages into, so tars already there are reused, not fetched again.
# The tars stay packed: script/internal/alliance/lightgenbench_unpack.sh unpacks them into $SLURM_TMPDIR
# at the start of every job. The repo is public, so no token is read.
set -euo pipefail

REPO_ID=3dlg-hcvc/LightgenBench
# sha256 of splits.json as released 2026-09-18 (the repo's own checksums.sha256 row). Every
# cluster must train on this split, so it is pinned here and not read from the download.
SPLITS_SHA256=6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e
N_FILES=79                   # splits.json + 39 voxel tars + 39 thumbnail tars

REPO=${REPO:-$HOME/scratch/TRELLIS.2-lightning}
DEST=${DEST:-$HOME/scratch/lightgen/data/lightgenbench}
WORKERS=${WORKERS:-8}

module load StdEnv/2023 gcc python/3.11 arrow/21.0.0 2>/dev/null || true
source "${REPO}/env/bin/activate"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "${DEST}"
echo "=== stage LightgenBench (voxels + thumbnails) -> ${DEST}  $(hostname)  $(date -Iseconds) ==="

DEST="${DEST}" REPO_ID="${REPO_ID}" WORKERS="${WORKERS}" python - <<'PY'
import os
import huggingface_hub
from huggingface_hub import snapshot_download
print("[hf] huggingface_hub", huggingface_hub.__version__)
snapshot_download(
    repo_id=os.environ["REPO_ID"], repo_type="dataset", local_dir=os.environ["DEST"],
    allow_patterns=["splits.json", "checksums.sha256",
                    "data/*/voxels/*.tar", "data/*/thumbnail/*.tar"],
    max_workers=int(os.environ["WORKERS"]),
)
PY

cd "${DEST}"
echo "${SPLITS_SHA256}  splits.json" | sha256sum -c -
grep -E '  (splits\.json|data/(train|val|test)/(voxels|thumbnail)/[^/]+\.tar)$' checksums.sha256 > .want_trellis2.sha256
GOT=$(wc -l < .want_trellis2.sha256)
if [ "${GOT}" -ne "${N_FILES}" ]; then
    echo "FATAL: checksums.sha256 lists ${GOT} files for TRELLIS.2, expected ${N_FILES}"
    exit 2
fi
echo "[sha256] checking ${GOT} files with ${WORKERS} workers  $(date -Iseconds)"
# one sha256sum per line ("<hash>  <path>" -> $0 $1); xargs exits 123 if any check fails
xargs -P "${WORKERS}" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < .want_trellis2.sha256
echo "[sha256] ok - ${GOT}/${N_FILES} files match checksums.sha256  $(date -Iseconds)"

python "${REPO}/data_toolkit/build_lightgenbench_index.py" --splits "${DEST}/splits.json" --out-dir "${DEST}/trellis2_index"
du -sh "${DEST}/data" | sed 's/^/[size] /'
echo "=== staged  $(date -Iseconds) ==="
