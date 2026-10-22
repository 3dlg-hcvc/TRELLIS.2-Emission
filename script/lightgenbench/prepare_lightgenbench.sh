#!/bin/bash
# Download the LightgenBench release (Hugging Face 3dlg-hcvc/LightgenBench) and lay it out the way the emission
# VAE and DiT datasets read it.
#
#   bash script/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>
#
# <download_dir>  the Hugging Face files TRELLIS.2-Emission uses: splits.json, checksums.sha256 and
#                 data/{train,val,test}/{voxels,dual_grid,thumbnail}/*.tar (about 30 GB). The dataset is gated:
#                 accept its terms on the Hugging Face page and run `hf auth login` first. Files already
#                 there are kept; SKIP_DOWNLOAD=1 uses <download_dir> as it is.
# <data_root>     written: emission_voxels_256/<uuid>.vxz, pbr_voxels_256/<uuid>.vxz, dual_grid_256/<uuid>.vxz,
#                 thumbnails/<uuid>.png, df_lightgenbench.parquet and data_splits_lightgenbench.json
#                 (data_toolkit/build_lightgenbench_index.py).
# The VAE trains on these; the DiT also needs the shape latents (from the dual grids), the PBR latents and the
# emission latents (README.md).
# Run from the repo root in the python env of this repo (README.md, Environment Setup).
set -euo pipefail
DL=$(realpath -m "${1:?download_dir}"); ROOT=$(realpath -m "${2:?data_root}")
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
REPO_ID=${REPO_ID:-3dlg-hcvc/LightgenBench}
# The released splits.json: every published number uses this split.
SPLITS_SHA256=${SPLITS_SHA256:-6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e}
WORKERS=${WORKERS:-8}
mkdir -p "$DL" "$ROOT"

if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  DL="$DL" REPO_ID="$REPO_ID" WORKERS="$WORKERS" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["REPO_ID"], repo_type="dataset", local_dir=os.environ["DL"],
                  allow_patterns=["splits.json", "checksums.sha256", "data/*/voxels/*.tar", "data/*/dual_grid/*.tar",
                                  "data/*/thumbnail/*.tar"],
                  max_workers=int(os.environ["WORKERS"]))
PY
fi

cd "$DL"
echo "$SPLITS_SHA256  splits.json" | sha256sum -c --quiet - || { echo "splits.json is not the released one" >&2; exit 2; }
grep -E '  data/(train|val|test)/(voxels|dual_grid|thumbnail)/[^/]+\.tar$' checksums.sha256 > "$ROOT/.tars.sha256"
[ -s "$ROOT/.tars.sha256" ] || { echo "checksums.sha256 lists no voxel, dual-grid or thumbnail tar" >&2; exit 2; }
echo "[prepare] checking $(wc -l < "$ROOT/.tars.sha256") tars against checksums.sha256"
xargs -P "$WORKERS" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < "$ROOT/.tars.sha256"

echo "[prepare] unpacking into $ROOT"
mkdir -p "$ROOT/emission_voxels_256" "$ROOT/pbr_voxels_256" "$ROOT/dual_grid_256" "$ROOT/thumbnails"
# Only the tars just checked; members are <uuid>/<file>, and --transform renames them into the flat folders.
awk '{print $2}' "$ROOT/.tars.sha256" | grep '/voxels/' | xargs -P "$WORKERS" -I{} tar xf {} -C "$ROOT" \
    --transform 's|^\([^/]*\)/emission_voxels\.vxz$|emission_voxels_256/\1.vxz|' \
    --transform 's|^\([^/]*\)/pbr_voxels\.vxz$|pbr_voxels_256/\1.vxz|'
awk '{print $2}' "$ROOT/.tars.sha256" | grep '/dual_grid/' | xargs -P "$WORKERS" -I{} tar xf {} -C "$ROOT/dual_grid_256" \
    --transform 's|^\([^/]*\)/dual_grid\.vxz$|\1.vxz|'
awk '{print $2}' "$ROOT/.tars.sha256" | grep '/thumbnail/' | xargs -P "$WORKERS" -I{} tar xf {} -C "$ROOT/thumbnails" \
    --transform 's|^\([^/]*\)/thumbnail\.png$|\1.png|'
N=$(python -c 'import json, sys; s = json.load(open(sys.argv[1])); print(sum(len(v) for v in s.values()))' splits.json)
for d in emission_voxels_256 pbr_voxels_256 dual_grid_256 thumbnails; do
  n=$(find "$ROOT/$d" -maxdepth 1 -type f \( -name '*.vxz' -o -name '*.png' \) | wc -l)
  [ "$n" = "$N" ] || { echo "$ROOT/$d holds $n files, splits.json lists $N shapes" >&2; exit 3; }
done
cd "$REPO"
python data_toolkit/build_lightgenbench_index.py --splits "$DL/splits.json" --out-dir "$ROOT"
python data_toolkit/build_lightgenbench_index.py --splits "$DL/splits.json" --out-dir "$ROOT" --check
echo "[prepare] done: $N shapes in $ROOT"
