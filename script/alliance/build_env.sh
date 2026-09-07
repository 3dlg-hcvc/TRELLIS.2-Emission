#!/bin/bash
# Build the lightgen Python environment on an Alliance cluster, in two steps that match where the network is:
#   bash script/alliance/build_env.sh prepare [--modules "…"]   # LOGIN node (has internet): venv + wheelhouse pins + PyPI pins + clone the extension sources
#   bash script/alliance/build_env.sh compile [--modules "…"]   # inside an allocation (compute nodes have no internet on nibi/rorqual/trillium): compile the CUDA extensions, smoke test
# Run both from the repo root. Idempotent: re-run after a failure and it skips what is already there.
# Modules: fir's exact set by default (its env is Python 3.11.5; the launchers load the same modules). trillium-gpu has no
# cuda/* module — pass --modules "StdEnv/2023 gcc python/3.11 nvhpc/25.1 arrow/21.0.0 opencv/4.13.0 sparsehash".
set -u
STEP=${1:-}; shift || true
MODULES="StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash"
[ "${1:-}" = "--modules" ] && { MODULES=$2; shift 2; }
[ -f setup.sh ] || { echo "run from the repo root (setup.sh not found)"; exit 2; }
case "$STEP" in prepare|compile) ;; *) echo "usage: build_env.sh prepare|compile [--modules \"…\"]"; exit 2 ;; esac
REQ=$(ls "$(dirname "$0")"/requirements-fir-*.txt | sort | tail -1)
SRC=$HOME/scratch/.lightgen-build/src            # extension sources: cloned by 'prepare', compiled by 'compile'
REPO=$PWD
mkdir -p log; TS=$(date +%Y%m%d-%H%M%S); LOG="log/${TS}-build_env-${STEP}.log"; exec > >(tee -a "$LOG") 2>&1
echo "build_env $STEP: $(hostname) $(date -Is) modules=[$MODULES] req=$REQ"
module purge >/dev/null 2>&1; module load $MODULES || { echo "module load failed"; exit 1; }
export TMPDIR=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}; export PIP_CACHE_DIR=$TMPDIR/pip-cache-$USER   # EXDEV trap: keep pip's temp off the parallel FS
[ -x env/bin/python ] || virtualenv --no-download env || exit 1
source env/bin/activate
if [ "$STEP" = prepare ]; then
  # pass 1 + 2: every pin from the wheelhouse first; what it can't find comes from PyPI (the login node has the network).
  # pip stays what virtualenv seeded; setuptools/wheel ARE installed — the --no-build-isolation compiles need them.
  grep -vE '^pip=' "$REQ" | grep -vE '^(o_voxel|cumesh|flex_gemm|nvdiffrast|nvdiffrec_render|utils3d)=' > "$LOG.pins"
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
  pip check 2>&1 | head -20     # what --no-deps skipped; the six extensions are still missing here, that is expected
  # sources for 'compile' — exactly what setup.sh clones (it clones into /tmp on the build node, which has no network here)
  mkdir -p "$SRC"; cd "$SRC"
  [ -d nvdiffrast ] || git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git nvdiffrast
  [ -d nvdiffrec ]  || git clone -q -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec
  [ -d CuMesh ]     || git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git CuMesh
  [ -d FlexGEMM ]   || git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM
  [ -d utils3d ]    || { git clone -q https://github.com/EasternJournalist/utils3d.git utils3d && git -C utils3d checkout -q 9a4eb15e4021b67b12c460c7057d642626897ec8; }
  ls -d "$SRC"/*
else
  # pass 3: the extensions setup.sh compiles, from the sources 'prepare' cloned, with setup.sh's flags (+ --no-deps: no network)
  export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"    # A100 / A40 / L40S-RTX40xx / H100+H200
  export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || echo /nonexistent/bin/nvcc)")")}
  echo "CUDA_HOME=$CUDA_HOME nvcc: $(nvcc --version 2>&1 | tail -1)"
  for pkg in nvdiffrast:nvdiffrast nvdiffrec_render:nvdiffrec cumesh:CuMesh flex_gemm:FlexGEMM o_voxel:"$REPO/o-voxel" utils3d:utils3d; do
    mod=${pkg%%:*}; d=${pkg#*:}; [ "${d#/}" = "$d" ] && d="$SRC/$d"
    if python -c "import $mod" 2>/dev/null; then echo "present: $mod"; continue; fi
    echo "--- building $mod from $d $(date +%H:%M:%S)"
    pip install --no-build-isolation --no-deps -q "$d" || echo "FAILED to build $mod"
  done
  echo "smoke test"
  python - <<'PY'
import importlib
mods = ["torch","torchvision","lightning","flash_attn","nvdiffrast","nvdiffrec_render","o_voxel","flex_gemm","cumesh","kornia","timm","transformers","diffusers","lpips","trimesh","utils3d","PIL"]
ok = fail = 0
for m in mods:
    try: importlib.import_module(m); print("OK  ", m); ok += 1
    except Exception as e: print("FAIL", m + ":", type(e).__name__ + ":", str(e)[:160]); fail += 1
import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} (False is expected in a CPU allocation)")
print(f"SUMMARY: {ok} ok, {fail} fail")
PY
fi
echo "build_env $STEP: done $(date -Is) — log $LOG"
