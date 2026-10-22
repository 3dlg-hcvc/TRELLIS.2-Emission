#!/bin/bash
# The pbr->emission chain on LightgenBench, one stage per call, in the order below:
#
#   TEXVERSE_1K=<dir> bash script/lightgenbench/run_chain.sh latents <data_root>   # shape + PBR latents
#   bash script/lightgenbench/run_chain.sh vae <data_root>                         # bake the init, train the VAE
#   VAE_CKPT=<ckpt> bash script/lightgenbench/run_chain.sh encode <data_root>      # emission latents + their stats
#   bash script/lightgenbench/run_chain.sh dit <data_root>                         # train the DiT
#   VAE_CKPT=<ckpt> DIT_CKPT=<ckpt> bash script/lightgenbench/run_chain.sh predict <data_root>   # the test split
#
# <data_root> is prepare_lightgenbench.sh's output; arguments after it go to train.py (vae, dit), such as
# --override trainer.args.dataloader_num_workers=4. The configs are
# configs/scvae/emission_vae_pbr2emission_lightgenbench_v1.yaml (VAE, 35,800 steps) and
# configs/gen/emission_dit_pbr2emission_lightgenbench_v1.yaml (DiT, 50,000 steps); by default the VAE
# trains with 16 shapes per GPU on 4 GPUs (global 64; the config says 32) and the DiT with 8 per GPU
# (global 32). A checkpoint is always named: nothing here picks one.
#
# env: NUM_GPUS (4); BS (16 for the VAE, 8 for the DiT; another global batch than the release's is
#      refused unless ANY_BATCH=1); MAX_STEPS; OUTPUTS (outputs: runs land in <OUTPUTS>/<config name>);
#      RESUME=1 to continue a run directory that holds checkpoints (a resume skips the finetune load,
#      so a first launch must start empty); WORKERS (latents: dual-grid processes, 8); LIGHTGEN_REPO
#      (latents: the lightgen repo, default the directory holding this repo); SEED (predict, 0);
#      PRED_OUT (predict: <OUTPUTS>/predict_test_seed<SEED>); CHAIN_TMPDIR (below). wandb runs offline
#      unless logged in.
# Each call tees its output to log/<TS>-lightgenbench_<stage>.log. Run from anywhere; it works in the repo root.
set -euo pipefail
STAGE=${1:?latents|vae|encode|dit|predict}; ROOT=$(realpath "${2:?data_root}"); shift 2; EXTRA=("$@")
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
abs() { [ -z "${1:-}" ] || realpath -m "$1"; }   # caller paths are relative to where the caller stands
OUTPUTS=$(abs "${OUTPUTS:-$REPO/outputs}"); TEXVERSE_1K=$(abs "${TEXVERSE_1K:-}"); LIGHTGEN_REPO=$(abs "${LIGHTGEN_REPO:-}")
VAE_CKPT=$(abs "${VAE_CKPT:-}"); DIT_CKPT=$(abs "${DIT_CKPT:-}"); PRED_OUT=$(abs "${PRED_OUT:-}"); CHAIN_TMPDIR=$(abs "${CHAIN_TMPDIR:-}")
cd "$REPO"; NUM_GPUS=${NUM_GPUS:-4}
VAE_NAME=emission_vae_pbr2emission_lightgenbench_v1; DIT_NAME=emission_dit_pbr2emission_lightgenbench_v1
LATENT_NAME=emission_pbr2emission_lightgenbench_v1
SHAPE_DIR=$ROOT/shape_latents/shape_enc_next_dc_f16c32_fp16_256; PBR_DIR=$ROOT/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
EMI_DIR=$ROOT/emission_latents/$LATENT_NAME
INDEX=$ROOT/df_lightgenbench.parquet; SPLITS=$ROOT/data_splits_lightgenbench.json
mkdir -p log; LOG=log/$(date +%Y%m%d-%H%M%S)-lightgenbench_${STAGE}.log
exec > >(tee -a "$LOG") 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Lightning writes each checkpoint to a temp file in $TMPDIR before moving it into place (the VAE's is
# 13 GB, the DiT's 21 GB), and Python makes Unix sockets there, whose paths must stay under 108
# characters. A small /tmp: CHAIN_TMPDIR=<a short path on a disk with room>.
[ -z "$CHAIN_TMPDIR" ] || { mkdir -p "$CHAIN_TMPDIR" && export TMPDIR=$CHAIN_TMPDIR; }
[ -n "${WANDB_API_KEY:-}" ] || grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=${WANDB_MODE:-offline}

fail() { echo "[chain] FATAL: $*" >&2; exit 2; }
case "$STAGE" in vae|dit) ;; *) [ ${#EXTRA[@]} -eq 0 ] || fail "only the vae and dit stages take arguments after <data_root>" ;; esac
[ -f "$INDEX" ] && [ -f "$SPLITS" ] || fail "$ROOT holds no index: run prepare_lightgenbench.sh first"
need() {  # <dir> <extension>: every shape of the index has its file there, and every .npz opens whole
  python - "$INDEX" "$1" "$2" <<'PY' || fail "$1 is incomplete"
import os, sys, zipfile
import pandas as pd
index, d, ext = sys.argv[1:4]
uuids = [str(u) for u in pd.read_parquet(index).index]
missing, bad = [], []
for u in uuids:
    f = os.path.join(d, f"{u}.{ext}")
    if not os.path.exists(f):
        missing.append(u)
    elif ext == "npz":
        try:
            with zipfile.ZipFile(f) as z:
                if z.testzip() is not None:
                    bad.append(u)
        except Exception:
            bad.append(u)
msg = f"[chain] {d}: {len(uuids) - len(missing) - len(bad)} of {len(uuids)}"
if missing:
    msg += f"; {len(missing)} missing (first {missing[0]})"
if bad:
    msg += f"; {len(bad)} unreadable, delete them and rerun the stage: {' '.join(bad[:5])}"
print(msg, flush=True)
os._exit(1 if missing or bad else 0)   # not sys.exit: pyarrow's teardown can abort at exit
PY
}
# GPUs: the caller's CUDA_VISIBLE_DEVICES (a scheduler's allocation), else 0..NUM_GPUS-1
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
per_gpu() {  # <command...>: once per GPU with --rank/--world_size; need() after it decides
  local pids=() r p bad=0
  [ "${#GPUS[@]}" -ge "$NUM_GPUS" ] || fail "NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES lists ${#GPUS[@]}"
  for r in $(seq 0 $((NUM_GPUS - 1))); do CUDA_VISIBLE_DEVICES=${GPUS[$r]} "$@" --rank "$r" --world_size "$NUM_GPUS" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [ $bad -eq 0 ] || echo "[chain] a rank of ${*:1:3} exited with an error (its log above)"
}
ckpt_cfg() {  # <ckpt>: the run's config.yaml, one level above ckpts/ or snaps/
  [ -f "$1" ] || fail "no checkpoint $1"; echo "$(dirname "$(dirname "$(realpath "$1")")")/config.yaml"
}
tmp_room() {  # <GB>: the temp folder holds one checkpoint, and its path is short
  local t=${TMPDIR:-/tmp} gb
  [ ${#t} -le 60 ] || fail "TMPDIR $t is ${#t} characters, too long for the Unix sockets made under it: set CHAIN_TMPDIR to a shorter path"
  gb=$(df -BG --output=avail "$t" | tail -1 | tr -dc 0-9)
  [ "$gb" -ge "$1" ] || fail "$t has $gb GB free and a checkpoint passes through it ($1 GB): set CHAIN_TMPDIR to a short path on a disk with room"
}
train() {  # <config> <run name> <per-GPU batch> <release global batch> <checkpoint GB> [<override>...]
  local cfg=$1 out=$OUTPUTS/$2 bs=$3 global=$4; tmp_room "$5"; shift 5
  if [ $((NUM_GPUS * bs)) -ne "$global" ] && [ "${ANY_BATCH:-0}" != 1 ]; then
    fail "$NUM_GPUS GPUs x $bs = $((NUM_GPUS * bs)); the release's global batch is $global (ANY_BATCH=1 to run anyway)"
  fi
  local ckpt=last
  if [ "${RESUME:-0}" != 1 ]; then
    ckpt=none   # a fresh run: no resume, so the finetune load happens
    if compgen -G "$out/ckpts/*.ckpt" >/dev/null || compgen -G "$out/ckpts/misc_step*.pt" >/dev/null; then
      fail "$out holds checkpoints: RESUME=1 continues that run (without the finetune load); a fresh run needs an empty directory"
    fi
  fi
  python -u train.py --config "$cfg" --output_dir "$out" --ckpt "$ckpt" --num_gpus "$NUM_GPUS" \
    --override trainer.args.batch_size_per_gpu="$bs" ${MAX_STEPS:+--override trainer.args.max_steps=$MAX_STEPS} \
    --override dataset.args.data_root="$ROOT" --override dataset.args.parquet_file="$INDEX" \
    --override dataset.args.split_json="$SPLITS" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
}

case "$STAGE" in
  latents)
    [ -n "${TEXVERSE_1K:-}" ] || fail "set TEXVERSE_1K to a TexVerse-1K download (glbs/glbs_1k/<group>/<uuid>_1024.glb)"
    need "$ROOT/pbr_voxels_256" vxz
    W=${WORKERS:-8}; pids=(); bad=0
    for r in $(seq 0 $((W - 1))); do
      python -u data_toolkit/dual_grid_from_glb.py --parquet "$INDEX" --glb_root "$TEXVERSE_1K" --out_root "$ROOT" \
        --loader lightgenbench --check_cells ${LIGHTGEN_REPO:+--lightgen_repo "$LIGHTGEN_REPO"} \
        --rank "$r" --world_size "$W" & pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    [ $bad -eq 0 ] || echo "[chain] a dual-grid process failed; failed shapes are in $ROOT/dual_grid_256/failed_rank*.txt"
    need "$ROOT/dual_grid_256" vxz
    per_gpu python -u data_toolkit/encode_shape_from_dual_grid.py --data_root "$ROOT" --parquet "$INDEX" --resolution 256
    per_gpu python -u data_toolkit/encode_pbr_from_voxels.py --data_root "$ROOT" --parquet "$INDEX" --resolution 256
    need "$SHAPE_DIR" npz; need "$PBR_DIR" npz
    ;;
  vae)
    need "$ROOT/emission_voxels_256" vxz; need "$ROOT/pbr_voxels_256" vxz
    CFG=configs/scvae/$VAE_NAME.yaml
    # The bake reads TRELLIS.2's PBR VAE from the Hugging Face cache only: fetch it (offline: the cache).
    python - <<'PY' || fail "TRELLIS.2's PBR VAE is neither downloadable nor in the Hugging Face cache"
import os
from huggingface_hub import hf_hub_download
for name in ("tex_enc_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16"):
    for ext in (".json", ".safetensors"):
        hf_hub_download("microsoft/TRELLIS.2-4B", f"ckpts/{name}{ext}")
os._exit(0)
PY
    # The VAE's init (gitignored, seconds); --verify-config stops a config that would drop the derived tensors.
    python -u data_toolkit/init_emission_vae_from_pbr_vae.py --out data/ckpt_init/pbr2emission_v3 --verify-config "$CFG"
    train "$CFG" "$VAE_NAME" "${BS:-16}" 64 15
    ;;
  encode)
    [ -n "${VAE_CKPT:-}" ] || { ls "$OUTPUTS/$VAE_NAME/ckpts" 2>/dev/null; fail "set VAE_CKPT (the checkpoint of lowest val/loss)"; }
    VCFG=$(ckpt_cfg "$VAE_CKPT")
    need "$ROOT/emission_voxels_256" vxz; need "$ROOT/pbr_voxels_256" vxz
    # The encoder keeps latents already on disk, so a folder must hold one VAE's latents only.
    mkdir -p "$EMI_DIR"; PROV="$EMI_DIR/vae_ckpt.txt"; WANT="$VAE_CKPT $(stat -c %s "$VAE_CKPT")"
    if [ -f "$PROV" ]; then
      [ "$(cat "$PROV")" = "$WANT" ] || fail "$EMI_DIR holds the latents of $(cut -d' ' -f1 "$PROV"); move it away to encode with $VAE_CKPT"
    elif compgen -G "$EMI_DIR/*.npz" >/dev/null; then
      fail "$EMI_DIR holds latents of an unknown VAE; move it away first"
    fi
    echo "$WANT" > "$PROV"
    ENC=(python -u data_toolkit/encode_emission_latents.py --data_root "$ROOT" --parquet "$INDEX"
         --vae_ckpt "$VAE_CKPT" --vae_cfg "$VCFG" --latent_name "$LATENT_NAME" --resolution 256)
    per_gpu "${ENC[@]}"
    "${ENC[@]}" --rank 0 --world_size "$NUM_GPUS" --aggregate_stats
    need "$EMI_DIR" npz
    echo "[chain] $EMI_DIR/normalization_stats.json:"; cat "$EMI_DIR/normalization_stats.json"; echo
    ;;
  dit)
    CFG=configs/gen/$DIT_NAME.yaml
    need "$ROOT/thumbnails" png; need "$SHAPE_DIR" npz; need "$PBR_DIR" npz; need "$EMI_DIR" npz
    # The config's emission_normalization must be these latents' stats: another VAE's latents live in
    # another space, and wrong stats mis-scale every input without an error.
    python - "$CFG" "$EMI_DIR/normalization_stats.json" <<'PY' || fail "copy the mean and std of $EMI_DIR/normalization_stats.json into dataset.args.emission_normalization of $CFG"
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))["dataset"]["args"]["emission_normalization"]
got = json.load(open(sys.argv[2]))
import math
if not all(math.isfinite(x) for k in ("mean", "std") for x in got[k]) or min(got["std"]) <= 0:
    sys.exit("[chain] normalization_stats.json holds a NaN, an infinity or a std <= 0: the encode went wrong")
bad = [k for k in ("mean", "std") if len(cfg[k]) != len(got[k]) or max(abs(a - b) for a, b in zip(cfg[k], got[k])) > 1e-5]
print("[chain] emission_normalization vs normalization_stats.json:", "MISMATCH in " + ", ".join(bad) if bad else "ok")
sys.exit(1 if bad else 0)
PY
    train "$CFG" "$DIT_NAME" "${BS:-8}" 32 22 --override dataset.args.thumbnail_dir="$ROOT/thumbnails"
    ;;
  predict)
    [ -n "${VAE_CKPT:-}" ] && [ -n "${DIT_CKPT:-}" ] || fail "set VAE_CKPT and DIT_CKPT (the VAE the latents came from, and a DiT checkpoint)"
    need "$ROOT/thumbnails" png; need "$SHAPE_DIR" npz; need "$PBR_DIR" npz; need "$ROOT/emission_voxels_256" vxz
    SEED=${SEED:-0}; OUT=${PRED_OUT:-$OUTPUTS/predict_test_seed$SEED}
    SHAS=$(python -c 'import json, os, sys, pandas as pd
idx = [str(u) for u in pd.read_parquet(sys.argv[1]).index]
s = json.load(open(sys.argv[2]))["test"]
print(",".join(idx[i] for i in (s["indices"] if isinstance(s, dict) else s)), flush=True); os._exit(0)' "$INDEX" "$SPLITS")
    echo "[chain] predicting $(echo "$SHAS" | tr , '\n' | grep -c .) test shapes into $OUT"
    python -u data_toolkit/predict_emission.py \
      --dit_ckpt "$DIT_CKPT" --dit_cfg "$(ckpt_cfg "$DIT_CKPT")" --vae_ckpt "$VAE_CKPT" --vae_cfg "$(ckpt_cfg "$VAE_CKPT")" \
      --data_root "$ROOT" --thumbnail_dir "$ROOT/thumbnails" --shas "$SHAS" --n all \
      --guidance 3.0 --steps 50 --seed "$SEED" --out_dir "$OUT"
    ;;
  *) fail "unknown stage '$STAGE'" ;;
esac
echo "[chain] $STAGE done; log $LOG"
