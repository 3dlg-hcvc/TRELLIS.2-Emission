#!/bin/bash
# bash script/internal/gate/train-check.sh <tree> <out_dir> vae|dit fresh|resume [<resume_ckpt>]
# The training check (REFACTOR_TODO.md, D5): the real train.py with a release config, on the check root
# (make_check_root.sh), batch 1, one GPU, through traincheck/sitecustomize.py, which seeds and records
# every step to <out_dir>/record.jsonl; summarize_ckpt.py appends the saved checkpoint. Compare two
# records with compare_traincheck.py.
#   fresh   from the configs' initialization (the VAE's baked init, the DiT's pretrained denoiser):
#           16 steps = 2 epochs of the 8 train shapes, validation after each epoch
#   resume  --ckpt <resume_ckpt> (default: the release VAE's ckpts/last.ckpt, step 35,800; the release
#           DiT's snaps/last-step0050000.ckpt, step 50,000), 16 more steps: the code under test loads a
#           published checkpoint (weights, EMA, optimizer, sampler and elastic state)
# Both save last.ckpt every 4 steps. Lightning 2.6 puts off a save whose monitor (val/loss) is not there
# yet or that falls on an epoch's last batch, so the saves land at steps 12 and 16 (fresh; 35812 and
# 35816 for the VAE resume): the step-12 save is followed by a training step, so CheckpointTimingCallback
# prints RESUME_AFTER_SAVE, which the snapshot watchers parse.
# The run's own checkpoints (13-21 GB) are deleted after they are summarized unless KEEP_CKPT=1.
# val_num_samples=8: the final snapshot (SnapshotCallback.on_train_end, always on) draws that many shapes
# from the 8 training shapes; upstream's default 16 makes the DiT's run_snapshot slice an empty batch.
# Overrides: GATE_CHECK_ROOT (default C/check_root), GATE_LOCK, GATE_LOCK_WAIT, GATE_ACTIVATE (as
# infer.sh), TRAINCHECK_TMPBASE (a short directory with room for one checkpoint), TRAINCHECK_STEPS
# (steps to run, default 16), TRAINCHECK_START (the resume checkpoint's global step, default 35800 vae /
# 50000 dit), TRAINCHECK_SAVE_EVERY (last.ckpt every N steps, default 4). Prints TRAINCHECK_DONE.
set -euo pipefail
T=$(realpath -e "${1:?tree}"); OUT=$(realpath -m "${2:?out dir}"); NET=${3:?vae or dit}; MODE=${4:?fresh or resume}
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
C=/local-scratch/localhome/dya78/lightgen_eval_local/trellis2_refactor_checks
ROOT=$(realpath -e "${GATE_CHECK_ROOT:-$C/check_root}")
case "$NET" in
  vae) CFG=configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml; START=${TRAINCHECK_START:-35800}
       CKPT=${5:-$T/outputs/emission_vae_pbr2emission_lightgenbench_v1/ckpts/last.ckpt}; EXTRA=() ;;
  dit) CFG=configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml; START=${TRAINCHECK_START:-50000}
       CKPT=${5:-$T/outputs/emission_dit_pbr2emission_lightgenbench_v1/snaps/last-step0050000.ckpt}
       EXTRA=(--override "dataset.args.thumbnail_dir=$ROOT/thumbnails") ;;
  *) echo "net must be vae or dit" >&2; exit 2 ;;
esac
case "$MODE" in
  fresh)  MAX=${TRAINCHECK_STEPS:-16}; EVERY=${TRAINCHECK_SAVE_EVERY:-4}; CK=(--ckpt none) ;;
  resume) [ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 2; }
          MAX=$((START + ${TRAINCHECK_STEPS:-16})); EVERY=${TRAINCHECK_SAVE_EVERY:-4}; CK=(--ckpt "$CKPT") ;;
  *) echo "mode must be fresh or resume" >&2; exit 2 ;;
esac
[ -e "$OUT" ] && { echo "$OUT exists; use a new out dir" >&2; exit 2; }
mkdir -p "$OUT"
LOCK=${GATE_LOCK:-/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
if [ "$LOCK" != none ]; then
  exec 9>"$LOCK"; echo "[gpu-lock] waiting $(date +%T)"
  flock -w "${GATE_LOCK_WAIT:-10800}" 9 || { echo "[gpu-lock] timeout"; exit 3; }; echo "[gpu-lock] acquired $(date +%T)"
fi
set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate trellis2; fi
set -u
export TRELLIS2_TRAINCHECK_OUT=$OUT/record.jsonl PYTHONPATH="$HERE/traincheck${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Lightning saves through a temp file in $TMPDIR; the workstation's /tmp is small and the path must stay short.
mkdir -p "${TRAINCHECK_TMPBASE:=/local-scratch/localhome/dya78/tmp}"
TMPDIR=$(mktemp -d "$TRAINCHECK_TMPBASE/tc.XXXXXX") && [ -d "$TMPDIR" ] || { echo "mktemp failed" >&2; exit 2; }
export TMPDIR
trap 'rm -rf "$TMPDIR"; [ "${KEEP_CKPT:-0}" = 1 ] || find "$OUT/run/ckpts" -name "*.ckpt" -delete 2>/dev/null || true' EXIT
echo "host=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES") tree=$(cat "$T/.gate_commit" 2>/dev/null || echo "$T") net=$NET mode=$MODE" | tee "$OUT/gpu.txt"
cd "$T"
python -u train.py --config "$CFG" --output_dir "$OUT/run" "${CK[@]}" --num_gpus 1 \
  --override "dataset.args.data_root=$ROOT" \
  --override "dataset.args.parquet_file=$ROOT/df_lightgenbench.parquet" \
  --override "dataset.args.split_json=$ROOT/data_splits_lightgenbench.json" "${EXTRA[@]}" \
  --override trainer.args.batch_size_per_gpu=1 --override trainer.args.val_batch_size_per_gpu=1 \
  --override trainer.args.max_steps=$MAX --override trainer.args.i_log=1 \
  --override trainer.args.i_sample=0 --override trainer.args.skip_initial_samples=true \
  --override trainer.args.val_num_samples=8 \
  --override trainer.args.save_top_k=1 --override trainer.args.ckpt_last_every_n_steps=$EVERY \
  --override trainer.args.val_every_epochs=1 --override trainer.args.num_sanity_val_steps=0 \
  --override trainer.args.dataloader_num_workers=0 --override trainer.args.wandb_enable=false \
  2>&1 | tee "$OUT/train.log"
grep -q '"kind": "fit_end"' "$OUT/record.jsonl" || { echo "ABORT: no fit_end record" >&2; exit 3; }
python - "$OUT/record.jsonl" <<'PY'
import json, math, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
bad = [r["gs"] for r in recs if r["kind"] == "train" and not (r["loss"] is not None and math.isfinite(r["loss"]))]
steps = sum(r["kind"] == "train" for r in recs); vals = sum(r["kind"] == "val" for r in recs)
sys.exit(f"ABORT: non-finite loss at {bad}" if bad else ("ABORT: no training step" if not steps else ("ABORT: no validation" if not vals else 0)))
PY
python - "$OUT/train.log" >> "$OUT/record.jsonl" <<'PY'
import json, re, sys
t = open(sys.argv[1], errors="replace").read()
print(json.dumps({"kind": "log", "save_start": re.findall(r"\[ckpt-timing\] SAVE_START step=(\d+)", t),
                  "resume_after_save": re.findall(r"\[ckpt-timing\] RESUME_AFTER_SAVE step=(\d+)", t)}))
PY
LAST=$OUT/run/ckpts/last.ckpt
[ -f "$LAST" ] || { echo "ABORT: no $LAST" >&2; ls "$OUT/run/ckpts" >&2; exit 3; }
env -u TRELLIS2_TRAINCHECK_OUT python "$HERE/traincheck/summarize_ckpt.py" "$LAST" >> "$OUT/record.jsonl"
echo TRAINCHECK_DONE
