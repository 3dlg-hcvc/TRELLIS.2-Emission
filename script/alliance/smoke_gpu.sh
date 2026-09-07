#!/bin/bash
# GPU smoke test for a freshly built env: every compiled extension imports on a GPU node and CUDA does work.
# Submit from the repo root with the cluster's 1-GPU form, e.g.
#   sbatch --account=def-msavva_gpu --gpus-per-node=h100:1 -c 4 --mem=16G --time=00:15:00 -J smoke_gpu --output=log/smoke_gpu_%j.out script/alliance/smoke_gpu.sh
# Prints "SMOKE_GPU: ok" or "SMOKE_GPU: FAIL <module>: <error>" lines; exit code = number of failures.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2
module load StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash || { echo "SMOKE_GPU: FAIL module load"; exit 1; }
source env/bin/activate || { echo "SMOKE_GPU: FAIL no env"; exit 1; }
echo "SMOKE_GPU: $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -Is)"
python - <<'PY'
import importlib, sys, torch
fails = 0
for m in ["torch","torchvision","lightning","flash_attn","nvdiffrast","nvdiffrec_render","o_voxel","flex_gemm","cumesh","kornia","timm","transformers","diffusers","lpips","trimesh","utils3d","PIL"]:
    try: importlib.import_module(m)
    except Exception as e: print(f"SMOKE_GPU: FAIL {m}: {type(e).__name__}: {str(e)[:160]}"); fails += 1
try:
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    x = torch.randn(1024, 1024, device="cuda"); y = (x @ x).sum().item()
    print(f"SMOKE_GPU: torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} matmul ok")
except Exception as e: print(f"SMOKE_GPU: FAIL cuda: {type(e).__name__}: {str(e)[:160]}"); fails += 1
print("SMOKE_GPU: ok" if fails == 0 else f"SMOKE_GPU: {fails} failures"); sys.exit(fails)
PY
