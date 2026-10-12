#!/bin/bash
# bash script/internal/gate/coverage.sh <tree> <out_dir>
# The coverage run (REFACTOR_TODO.md step 9; no code change): line coverage of <tree> while it runs
#   the VAE training check fresh and resume (train-check.sh), the predictor on 3 paper and 3 release
#   shapes (infer.sh), the encode check and the VAE table on 3 shapes, and both emission datasets
#   built on the check root with every item loaded and collated (the DiT's data path; the DiT trainer
#   itself is upstream's and trains only on Solar L40S).
# coverage 7 comes from the texgen env (C/coverage_pkg links it; the trellis2 env has none; both are
# Python 3.10); traincheck/sitecustomize.py starts it in every process (COVERAGE_PROCESS_START).
# Output: <out_dir>/report.txt (every file of the tree that ran, with its missing lines) and
# <out_dir>/ours.txt (our files only). "Never ran" is not "dead": multi-GPU, gradient accumulation,
# --pred_only and the elastic memory controller's branches do not run here.
set -euo pipefail
T=$(realpath -e "${1:?tree}"); OUT=$(realpath -m "${2:?out dir}")
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
C=/local-scratch/localhome/dya78/lightgen_eval_local/trellis2_refactor_checks
PY=/localhome/dya78/software/miniconda3/envs/trellis2/bin/python
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
[ -d "$C/coverage_pkg/coverage" ] || { echo "no $C/coverage_pkg/coverage" >&2; exit 2; }
mkdir -p "$OUT"
cat > "$OUT/.coveragerc" <<RC
[run]
parallel = True
relative_files = True
data_file = $OUT/.coverage
source = $T
omit = */o-voxel/*
       */script/internal/*
RC
export COVERAGE_PROCESS_START=$OUT/.coveragerc PYTHONPATH="$HERE/traincheck:$C/coverage_pkg${PYTHONPATH:+:$PYTHONPATH}"
head -3 "$HERE/subset20.txt" > "$OUT/paper3.txt"; head -3 "$C/release_root/shapes.txt" > "$OUT/release3.txt"
for m in fresh resume; do bash "$HERE/train-check.sh" "$T" "$OUT/vae_$m" vae "$m" > "$OUT/vae_$m.log" 2>&1; echo "vae_$m rc=$?"; done
bash "$HERE/infer.sh" "$T" "$OUT/paper3" "$OUT/paper3.txt" > /dev/null 2>&1; echo "paper3 rc=$?"
GATE_RUN=release bash "$HERE/infer.sh" "$T" "$OUT/release3" "$OUT/release3.txt" > /dev/null 2>&1; echo "release3 rc=$?"
bash "$HERE/encode_check.sh" "$T" "$OUT/encode3" "$OUT/paper3.txt" > /dev/null 2>&1; echo "encode3 rc=$?"
bash "$HERE/vae_table.sh" "$T" "$OUT/vae_table3" "$OUT/paper3.txt" > /dev/null 2>&1; echo "vae_table3 rc=$?"
(cd "$T" && "$PY" - "$C/check_root" <<'PYEOF'
import sys, yaml
from trellis2 import datasets
root = sys.argv[1]
for cfg_path, name in (("configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml", "vae"),
                       ("configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml", "dit")):
    cfg = yaml.safe_load(open(cfg_path))["dataset"]
    args = dict(cfg["args"], data_root=root, parquet_file=f"{root}/df_lightgenbench.parquet",
                split_json=f"{root}/data_splits_lightgenbench.json")
    if name == "dit":
        args["thumbnail_dir"] = f"{root}/thumbnails"
    for split in ("train", "val"):
        ds = getattr(datasets, cfg["name"])(**dict(args, split=split))
        items = [ds[i] for i in range(len(ds))]
        batch = type(ds).collate_fn(items)
        print(name, split, len(ds), sorted(batch) if isinstance(batch, dict) else type(batch))
PYEOF
) > "$OUT/datasets.log" 2>&1; echo "datasets rc=$?"
cd "$T" && "$PY" -m coverage combine --rcfile="$OUT/.coveragerc" -q && "$PY" -m coverage report --rcfile="$OUT/.coveragerc" -m > "$OUT/report.txt"
cd "$OUT"
grep -E "^(train\.py|trellis2/(datasets/(emission_helpers|sparse_voxel_emission|structured_latent_emission)|trainers/(basic|callbacks|lightgen_hooks|vae/emission_vae))\.py|data_toolkit/(init_emission_vae_from_pbr_vae|bake_pbr2emission_v3_init|dual_grid_from_glb|encode_shape_from_dual_grid|encode_pbr_from_voxels|encode_emission_latents|encode_emission_twostream_dataset|eval_vae_recon|build_lightgenbench_index|predict_emission|predict_emission_voxels_twostream)\.py|vis/)" report.txt > ours.txt || true
echo "COVERAGE_DONE $OUT"
