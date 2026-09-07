#!/bin/bash
# build_env.sh — rebuild the fir training environment on another Alliance cluster, from fir's exact package list.
#
#   cd /scratch/$USER/TRELLIS.2-lightning && bash script/alliance/build_env.sh [--modules "<module list>"]
#
# Three passes, in this order (the first must succeed before the second runs):
#   1. wheelhouse:  pip install --no-index  for every pin in requirements-fir-<date>.txt that the cluster's wheelhouse has
#   2. PyPI:        the pins the wheelhouse lacks (login nodes reach PyPI on every cluster; vulcan via its proxy)
#   3. source:      the six CUDA extensions the repo's setup.sh compiles (o_voxel, cumesh, flex_gemm, nvdiffrast, nvdiffrec, utils3d)
# then a smoke test. Run it inside a CPU allocation (salloc -c 8 --mem=32G --time=3:00:00) — compiles are heavy and the
# login node is not for that. Idempotent: re-run after a failure and it skips what is already installed.
#
# Modules: the fir set by default. trillium-gpu has no cuda/* module — pass --modules "StdEnv/2023 python/3.12 nvhpc/25.1 gcc opencv arrow/22.0.0 sparsehash".
set -u
MODULES="StdEnv/2023 python/3.12 cuda/12.6 gcc opencv arrow/22.0.0 sparsehash"
[ "${1:-}" = "--modules" ] && { MODULES=$2; shift 2; }
REQ=$(ls "$(dirname "$0")"/requirements-fir-*.txt | sort | tail -1)
[ -f setup.sh ] || { echo "run from the repo root (setup.sh not found)"; exit 2; }
mkdir -p log; TS=$(date +%Y%m%d-%H%M%S); LOG="log/${TS}-build_env.log"; exec > >(tee -a "$LOG") 2>&1
echo "build_env: $(hostname) $(date -Is) modules=[$MODULES] req=$REQ"
module purge >/dev/null 2>&1; module load $MODULES || { echo "module load failed"; exit 1; }
[ -x env/bin/python ] || virtualenv --no-download env || exit 1
source env/bin/activate; pip install --no-index --upgrade pip >/dev/null 2>&1 || true

# pass 1 + 2: try every pin from the wheelhouse first; whatever it can't find goes to the PyPI list
grep -vE '^(pip|setuptools|wheel)=' "$REQ" | grep -vE '^(o_voxel|cumesh|flex_gemm|nvdiffrast|nvdiffrec_render|utils3d)=' > "$LOG.pins"
: > "$LOG.pypi"; ok=0; miss=0
while read -r pin; do
  name=${pin%%==*}
  if pip show -q "$name" 2>/dev/null; then ok=$((ok+1)); continue; fi
  if pip install --no-index --no-deps -q "$pin" 2>/dev/null; then ok=$((ok+1)); else echo "$pin" >> "$LOG.pypi"; miss=$((miss+1)); fi
done < "$LOG.pins"
echo "pass 1 (wheelhouse): $ok installed/present, $miss deferred to PyPI"
if [ -s "$LOG.pypi" ]; then
  echo "pass 2 (PyPI): $(tr '\n' ' ' < "$LOG.pypi")"
  while read -r pin; do name=${pin%%==*}; ver=${pin#*==}; ver=${ver%%+*}    # drop +computecanada tags for PyPI
    pip install --no-deps -q "$name==$ver" 2>/dev/null || pip install --no-deps -q "$name" || echo "  FAILED: $pin"; done < "$LOG.pypi"
fi
# dependency closure check: pip's resolver sees what --no-deps skipped
pip check 2>&1 | head -20

# pass 3: the compiled extensions, via the repo's own installer (sourced, as it expects)
echo "pass 3 (source builds via setup.sh)"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"    # A100 / A40 / L40S-RTX40xx / H100+H200
. ./setup.sh --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec 2>&1 | tail -40

echo "smoke test"
python - <<'PY'
import importlib
mods = ["torch","torchvision","lightning","flash_attn","nvdiffrast","o_voxel","flex_gemm","cumesh","kornia","timm","transformers","diffusers","lpips","trimesh","utils3d","PIL"]
ok=fail=0
for m in mods:
    try: importlib.import_module(m); print(f"OK   {m}"); ok+=1
    except Exception as e: print(f"FAIL {m}: {type(e).__name__}: {e}"); fail+=1
import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} (False is expected in a CPU allocation)")
print(f"SUMMARY: {ok} ok, {fail} fail")
PY
echo "build_env: done $(date -Is) — log $LOG"
