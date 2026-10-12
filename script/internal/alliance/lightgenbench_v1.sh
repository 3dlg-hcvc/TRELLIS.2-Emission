#!/bin/bash
# The paper's pbr->emission chain (TRELLIS.2-Emission) on the released dataset, Hugging Face
# 3dlg-hcvc/LightgenBench (train 36,426 / val 200 / test 200), on ANY Alliance H100 cluster:
#   STAGE=vae     emission_vae_pbr2emission_lightgenbench_v1   4 x 16 = global 64, 35,800 steps
#   STAGE=encode  emission latents of all 36,826 shapes with that VAE, + normalization_stats.json
#   STAGE=dit     emission_dit_pbr2emission_lightgenbench_v1   4 x 8 = global 32, 50,000 steps
# The three wrappers beside this file (emission_vae_…, encode_emission_…, emission_dit_…) set STAGE,
# so hedge.sh gets one launcher path per stage. Run on the cluster's login node from the repo root:
#
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=01:00:00 bash script/internal/alliance/emission_vae_pbr2emission_lightgenbench_v1.sh
#   bash script/internal/alliance/emission_vae_pbr2emission_lightgenbench_v1.sh [--dry-run]
#   VAE_CKPT=step00NNNNN-0.00NN.ckpt bash script/internal/alliance/encode_emission_pbr2emission_lightgenbench_v1.sh
#   bash script/internal/alliance/emission_dit_pbr2emission_lightgenbench_v1.sh
#   env: MAX_STEPS, OUT_SUFFIX (a probe goes to its own directory: a run dir that already holds a
#        checkpoint RESUMES, and a resume skips the finetune load), WALLTIME, CLUSTER, ACCOUNT,
#        RESUME=1 to submit into a run dir that already holds checkpoints on purpose
#
# What is the paper's and what is this launcher's. Configs: the paper's, data paths and names only
# (configs/{scvae,gen}/*_lightgenbench_v1.yaml). Gates: those of script/fir/*_74k_v3_agentic.sh (global
# batch guard, sick-node gate, init bake with --verify-config, data counts, requeue cap 3, SIGTERM
# lead 900 s for the VAE and 1500 s for the DiT's 20 GB last.ckpt). Cluster table, offline weight
# cache, wandb online-if-reachable, read-only HOME, sleep 60 for hedged submission:
# script/alliance/emission_dit_74k.sh (deleted; tag pre-refactor). New here: the data is unpacked from the staged release tars
# into $SLURM_TMPDIR (script/internal/alliance/lightgenbench_{stage,unpack}.sh), and the DiT job copies
# last.ckpt to snaps/last-step<N>.ckpt every 10,000 steps, which the paper run did with a separate
# watcher job (script/star2/dit_agentic_snapshot_last_venus19.sbatch, deleted; tag pre-refactor) — the paper's checkpoint was
# chosen among those snapshots by IoU on the validation split, not by val/loss.
# Spec: lightgen docs/superpowers/specs/2026-09-20-trellis2-emission-lightgenbench-retrain-design.md
set -u
STAGE="${STAGE:?set STAGE=vae|encode|dit, or use the wrapper scripts}"
BRANCH="lightgen"; N_SHAPES=36826
MAX_STEPS="${MAX_STEPS:-}"; OUT_SUFFIX="${OUT_SUFFIX:-}"; DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
CLUSTER="${CLUSTER:-$(scontrol show config 2>/dev/null | awk -F'= ' '/^ClusterName/{print $2}')}"
REPO="${REPO:-$HOME/scratch/TRELLIS.2-lightning}"
HFD="${HFD:-$HOME/scratch/lightgen/data/lightgenbench}"
LAT="${LAT:-$HOME/scratch/lightgen/data/lightgenbench_trellis2}"
VAE_NAME="emission_vae_pbr2emission_lightgenbench_v1"
DIT_NAME="emission_dit_pbr2emission_lightgenbench_v1"
LATENT_NAME="emission_pbr2emission_lightgenbench_v1"
SNAP_EVERY=10000; DIT_STEPS=50000
NTASKS=4; CPUS=12
case "$STAGE" in
  vae)    NAME="$VAE_NAME"; CONFIG="configs/scvae/${VAE_NAME}.yaml"; BS="${BS:-16}"; GLOBAL_BATCH=64; LEAD=900
          WALLTIME="${WALLTIME:-24:00:00}"; WANDB_BASE="vae_pbr->emission_lightgenbench_v1" ;;
  dit)    NAME="$DIT_NAME"; CONFIG="configs/gen/${DIT_NAME}.yaml"; BS="${BS:-8}"; GLOBAL_BATCH=32; LEAD=1500
          WALLTIME="${WALLTIME:-24:00:00}"; WANDB_BASE="dit_pbr->emission_lightgenbench_v1" ;;
  encode) NAME="encode_${LATENT_NAME}"; CONFIG=""; BS=0; GLOBAL_BATCH=0; LEAD=300; NTASKS=1; CPUS=48
          WALLTIME="${WALLTIME:-04:00:00}"; WANDB_BASE=""
          VAE_CKPT="${REPO}/outputs/${VAE_NAME}/ckpts/${VAE_CKPT:?set VAE_CKPT=<best filename>.ckpt}"
          VAE_CFG="${REPO}/outputs/${VAE_NAME}/config.yaml" ;;
  *) echo "unknown STAGE '$STAGE'"; exit 2 ;;
esac
OUTPUT_DIR="$REPO/outputs/${NAME}${OUT_SUFFIX}"
WANDB_NAME="${WANDB_NAME:-${WANDB_BASE}${OUT_SUFFIX:+${OUT_SUFFIX}_${CLUSTER}}}"   # probes on several clusters keep separate rows
if [ "$STAGE" != encode ] && [ $((NTASKS * BS)) -ne "$GLOBAL_BATCH" ]; then
    echo "REFUSING TO SUBMIT: ${NTASKS} GPUs x BS ${BS} = $((NTASKS * BS)), the paper's global batch is ${GLOBAL_BATCH}." >&2; exit 1
fi

MODULES="StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash"
SIGNAL="#SBATCH --signal=B:SIGTERM@${LEAD}"; MEM="#SBATCH --mem=1024G"; GPU="#SBATCH --gpus-per-node=h100:4"; TIME="#SBATCH --time=${WALLTIME}"
case "$CLUSTER" in
  fir)       ACCT=rrg-msavva_gpu ;;
  nibi)      ACCT=def-msavva_gpu ;;
  rorqual)   ACCT=def-msavva_gpu; MEM="#SBATCH --mem=450G" ;;                       # 4xH100 nodes carry 510G
  killarney) ACCT=aip-msavva ;;                                                    # 8xH100 nodes, 2060G: half a node
  tamia)     ACCT=aip-msavva; MEM="#SBATCH --mem=450G"; GPU="#SBATCH --gres=gpu:h100:4" ;;   # whole 4xH100 node: 48 cpu, 500000M
  grillium|trillium*) ACCT=def-msavva; MEM=""; SIGNAL=""; GPU="#SBATCH --gpus-per-node=4"
             TIME="#SBATCH -t $(( $(echo "$WALLTIME" | awk -F: '{print $1*60+$2}') ))" ;;     # SciNet's wrapper: minutes, no --mem, no --signal
  *) echo "unknown cluster '$CLUSTER' — set CLUSTER=…"; exit 2 ;;
esac
ACCOUNT="${ACCOUNT:-$ACCT}"

# Pre-flight on the login node: a target that cannot run is refused here, not after hours in a queue.
fail() { echo "PREFLIGHT ($CLUSTER, $STAGE): $*"; [ $DRY = 1 ] || exit 2; }
[ -x "$REPO/env/bin/python" ] || fail "environment missing: $REPO/env"
[ -f "$HFD/trellis2_index/df_lightgenbench.parquet" ] || fail "release not staged: run script/internal/alliance/lightgenbench_stage.sh ($HFD)"
W="hub/models--microsoft--TRELLIS.2-4B/snapshots/*/ckpts/tex_enc_next_dc_f16c32_fp16.safetensors"
ls "$HOME"/scratch/.cache/huggingface/$W >/dev/null 2>&1 || ls "$HOME"/.cache/huggingface/$W >/dev/null 2>&1 \
   || fail "microsoft/TRELLIS.2-4B is in neither ~/scratch/.cache/huggingface nor ~/.cache/huggingface"
if [ "$STAGE" = dit ]; then
    [ -f "$LAT/latents.tar" ] || fail "missing $LAT/latents.tar (shape + PBR latents)"
    [ -f "$LAT/emission_latents_${LATENT_NAME}.tar" ] || fail "missing $LAT/emission_latents_${LATENT_NAME}.tar (run the encode stage, or copy it from the cluster that ran it)"
fi
if [ "$STAGE" = encode ]; then [ -f "$VAE_CKPT" ] || fail "missing $VAE_CKPT"; [ -f "$VAE_CFG" ] || fail "missing $VAE_CFG"; fi
if [ "$STAGE" != encode ] && [ -z "${RESUME:-}" ] && ls "$OUTPUT_DIR"/ckpts/*.ckpt >/dev/null 2>&1; then
    fail "$OUTPUT_DIR already holds checkpoints: this job would RESUME them and skip the finetune load. RESUME=1 if that is meant."
fi
echo "Submitting ${NAME}${OUT_SUFFIX} on ${CLUSTER} (${ACCOUNT}): walltime ${WALLTIME}${BS:+; ${NTASKS} x BS ${BS} = global $((NTASKS * BS))}; max_steps ${MAX_STEPS:-<config>}"

mkdir -p "$REPO/log"; JOB="$REPO/log/job_${NAME}${OUT_SUFFIX}_${CLUSTER}.sh"   # log/ is gitignored
cat > "$JOB" <<JOB
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}
${GPU}
${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NTASKS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
${MEM}
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
${SIGNAL}
handle_timeout() { echo "SIGTERM: stopping gracefully..."; [ -n "\${SNAP_PID:-}" ] && kill \$SNAP_PID 2>/dev/null; [ -n "\${TRAIN_PID:-}" ] && kill -0 \$TRAIN_PID 2>/dev/null && { kill -TERM \$TRAIN_PID; wait \$TRAIN_PID; }; scontrol requeue \${SLURM_JOB_ID}; exit 0; }
trap 'handle_timeout' SIGTERM
sleep 60   # hedged submission: a job cancelled in its first minute has cost nothing (compute-access hedged-submission.md)
echo "====================================== Job \${SLURM_JOB_NAME} (\${SLURM_JOB_ID}) cluster ${CLUSTER} node \$(hostname) restart=\${SLURM_RESTART_COUNT:-0} started \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# A shared node can hand over a card that a foreign process still holds (fir fc10218, 2026-08-20 and 08-24: 69-71 GB on GPU 0).
BUSY=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1 > 2048' | wc -l)
if [ "\${BUSY}" -gt 0 ]; then echo "FATAL: \${BUSY} GPU(s) on \$(hostname) already hold >2 GB; node is sick. Requeueing."; scontrol requeue \${SLURM_JOB_ID}; sleep 60; exit 0; fi
module load ${MODULES}
cd ${REPO}
# Code is updated on the login node at submit time (below); compute nodes on nibi/rorqual/trillium have no route to GitHub, so this is a bounded retry only.
timeout 30 git fetch origin ${BRANCH} 2>/dev/null && git checkout -q ${BRANCH} && git pull -q --ff-only origin ${BRANCH} || echo "git update skipped in-job (no network or local changes)"
echo "repo at: \$(git log --oneline -1)"
source ${REPO}/env/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0   # NFS-home autotune cache races between ranks
# Pretrained weights come from a Hugging Face cache staged ahead of time: in scratch where one was built (2026-09-07), else the home default (fir).
[ -d \$HOME/scratch/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B ] && export HF_HOME=\$HOME/scratch/.cache/huggingface
[ -d \$HOME/scratch/.cache/torch ] && export TORCH_HOME=\$HOME/scratch/.cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; echo "HF_HOME=\${HF_HOME:-<default ~/.cache/huggingface>}"
if [ -f \$HOME/.netrc ] && curl -sSI -m 8 https://api.wandb.ai >/dev/null 2>&1; then echo "wandb: online"; else export WANDB_MODE=offline; echo "wandb: OFFLINE (no login or no internet on this node)"; fi
# trillium compute nodes mount \$HOME read-only: every per-user cache that would land there goes to the node-local disk instead.
if ! ( : > \$HOME/.wtest.\$\$ ) 2>/dev/null; then export TRITON_CACHE_DIR=\$SLURM_TMPDIR/.triton CUDA_CACHE_PATH=\$SLURM_TMPDIR/.nv MPLCONFIGDIR=\$SLURM_TMPDIR/.mpl WANDB_CACHE_DIR=\$SLURM_TMPDIR/.wandb WANDB_CONFIG_DIR=\$SLURM_TMPDIR/.wandb_cfg XDG_CACHE_HOME=\$SLURM_TMPDIR/.cache; echo "HOME read-only: caches redirected to \$SLURM_TMPDIR"; else rm -f \$HOME/.wtest.\$\$; fi
set -o pipefail
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}${OUT_SUFFIX}-${CLUSTER}.log"
ROOT=\$SLURM_TMPDIR/lightgenbench
export HFD=${HFD} LAT=${LAT} LATENT_NAME=${LATENT_NAME}
JOB

if [ "$STAGE" = encode ]; then cat >> "$JOB" <<JOB
bash script/internal/alliance/lightgenbench_unpack.sh vae \$ROOT 2>&1 | tee -a "\${LOG}"; [ \${PIPESTATUS[0]} -eq 0 ] || exit 1
{
  pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r python -u data_toolkit/encode_emission_latents.py \\
      --data_root \$ROOT --parquet \$ROOT/df_lightgenbench.parquet \\
      --vae_ckpt ${VAE_CKPT} --vae_cfg ${VAE_CFG} \\
      --latent_name ${LATENT_NAME} --resolution 256 --rank \$r --world_size 4 &
    pids+=(\$!)
  done
  fail=0; for p in \${pids[@]}; do wait \$p || fail=1; done
  [ \$fail -eq 0 ] || { echo "ERROR: an encode rank failed"; exit 1; }
  python -u data_toolkit/encode_emission_latents.py \\
      --data_root \$ROOT --parquet \$ROOT/df_lightgenbench.parquet \\
      --vae_ckpt ${VAE_CKPT} --vae_cfg ${VAE_CFG} \\
      --latent_name ${LATENT_NAME} --resolution 256 --rank 0 --world_size 4 --aggregate_stats || exit 1
  NLAT=\$(ls \$ROOT/emission_latents/${LATENT_NAME} | grep -c '\.npz\$')
  echo "latents: \${NLAT} (expected ${N_SHAPES})"
  [ "\${NLAT}" -eq ${N_SHAPES} ] || { echo "FATAL: \${NLAT} of ${N_SHAPES} latents present; encode incomplete"; exit 1; }
  cat \$ROOT/emission_latents/${LATENT_NAME}/normalization_stats.json; echo
  # One tar in scratch (file quota), written under a temporary name so a cut-short tar is never picked up.
  mkdir -p ${LAT}
  tar cf ${LAT}/emission_latents_${LATENT_NAME}.tar.part -C \$ROOT emission_latents/${LATENT_NAME} || exit 1
  mv ${LAT}/emission_latents_${LATENT_NAME}.tar.part ${LAT}/emission_latents_${LATENT_NAME}.tar
  cp \$ROOT/emission_latents/${LATENT_NAME}/normalization_stats.json ${LAT}/normalization_stats_${LATENT_NAME}.json
  ( cd ${LAT} && sha256sum emission_latents_${LATENT_NAME}.tar | tee emission_latents_${LATENT_NAME}.tar.sha256 )
  echo "vae_ckpt: ${VAE_CKPT}" | tee ${LAT}/emission_latents_${LATENT_NAME}.provenance.txt; sha256sum ${VAE_CKPT} | tee -a ${LAT}/emission_latents_${LATENT_NAME}.provenance.txt
} 2>&1 | tee -a "\${LOG}"
exit \${PIPESTATUS[0]}
JOB
else
UNPACK=vae; [ "$STAGE" = dit ] && UNPACK=dit
cat >> "$JOB" <<JOB
bash script/internal/alliance/lightgenbench_unpack.sh ${UNPACK} \$ROOT 2>&1 | tee -a "\${LOG}"; [ \${PIPESTATUS[0]} -eq 0 ] || exit 1
python data_toolkit/build_lightgenbench_index.py --splits ${HFD}/splits.json --out-dir \$ROOT --check 2>&1 | tee -a "\${LOG}"; [ \${PIPESTATUS[0]} -eq 0 ] || exit 1
JOB
if [ "$STAGE" = vae ]; then cat >> "$JOB" <<JOB
# Bake the derived init here (gitignored, deterministic, seconds). The verifier is a hard gate:
# without it the trainer would silently drop the 5 derived tensors and cold-start.
python -u data_toolkit/init_emission_vae_from_pbr_vae.py --out data/ckpt_init/pbr2emission_v3 --verify-config ${CONFIG} 2>&1 | tee -a "\${LOG}"
[ \${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: init bake failed"; exit 1; }
JOB
else cat >> "$JOB" <<JOB
# The config's emission_normalization must be the stats of THESE latents: arrays from another VAE
# belong to another latent space and would mis-scale every input, silently.
python - <<PY 2>&1 | tee -a "\${LOG}"
import json, sys, yaml
cfg = yaml.safe_load(open("${CONFIG}"))["dataset"]["args"]["emission_normalization"]
got = json.load(open("\$ROOT/emission_latents/${LATENT_NAME}/normalization_stats.json"))
bad = [k for k in ("mean", "std") if len(cfg[k]) != 32 or max(abs(a - b) for a, b in zip(cfg[k], got[k])) > 1e-5]
print("[norm] config emission_normalization vs normalization_stats.json:", "MISMATCH " + str(bad) if bad else "ok")
sys.exit(1 if bad else 0)
PY
[ \${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: ${CONFIG} does not carry the normalization of the staged emission latents"; exit 1; }
# Keep the trajectory: last.ckpt is overwritten every 2,000 steps and save_top_k is 1, so copy it out
# every ${SNAP_EVERY} steps. The trainer prints RESUME_AFTER_SAVE step=N once every save of step N is done.
snap_one() {  # \$1 = step, \$2 = log
    local S=\$1 F T M SZ; F=\$(printf 'last-step%07d.ckpt' "\$S"); [ -f "${OUTPUT_DIR}/snaps/\$F" ] && return 0
    newest() { grep -oE 'SAVE_START step=[0-9]+' "\$1" | tail -1 | grep -oE '[0-9]+\$'; }   # last.ckpt is step S only while no later save has begun
    [ "\$(newest "\$2")" = "\$S" ] || { echo "[snap] step=\$S: a later save has begun, last.ckpt is no longer step \$S; skipped"; return 1; }
    T=\$(grep -E "SAVE_START step=\$S " "\$2" | tail -1 | grep -oE ' t=[0-9]+' | cut -d= -f2)
    M=\$(stat -c %Y ${OUTPUT_DIR}/ckpts/last.ckpt 2>/dev/null || echo 0); SZ=\$(stat -c %s ${OUTPUT_DIR}/ckpts/last.ckpt 2>/dev/null || echo 0)
    if [ -z "\${T:-}" ] || [ "\$M" -lt "\$T" ]; then echo "[snap] step=\$S: last.ckpt mtime \$M is older than the save (t=\${T:-?}); skipped"; return 1; fi
    mkdir -p ${OUTPUT_DIR}/snaps; cp ${OUTPUT_DIR}/ckpts/last.ckpt ${OUTPUT_DIR}/snaps/\$F.tmp || { rm -f ${OUTPUT_DIR}/snaps/\$F.tmp; return 1; }
    if [ "\$(newest "\$2")" != "\$S" ] || [ "\$(stat -c %Y ${OUTPUT_DIR}/ckpts/last.ckpt)" != "\$M" ] || [ "\$(stat -c %s ${OUTPUT_DIR}/snaps/\$F.tmp)" != "\$SZ" ]; then
        echo "[snap] step=\$S: last.ckpt changed under the copy; discarded"; rm -f ${OUTPUT_DIR}/snaps/\$F.tmp; return 1; fi
    mv ${OUTPUT_DIR}/snaps/\$F.tmp ${OUTPUT_DIR}/snaps/\$F || { echo "[snap] step=\$S: mv FAILED"; return 1; }
    echo "[snap] step=\$S -> snaps/\$F (\$SZ B) \$(date '+%F %T')"
}
# After a requeue the new log holds no SAVE_START for the checkpoint that survived, so the loop above would never
# copy it. With no trainer running yet there is no race: read its step and copy it if it is one to keep.
snap_retained() {
    local S F; [ -f ${OUTPUT_DIR}/ckpts/last.ckpt ] || return 0
    S=\$(python -c "import torch; print(int(torch.load('${OUTPUT_DIR}/ckpts/last.ckpt', map_location='cpu', mmap=True, weights_only=False)['global_step']))" 2>/dev/null) \
        || { echo "[snap] retained last.ckpt: global_step unreadable; not snapshotted"; return 0; }
    echo "[snap] retained last.ckpt is step \$S"
    [ "\$S" -gt 0 ] && [ \$((S % ${SNAP_EVERY})) -eq 0 ] || return 0
    F=\$(printf 'last-step%07d.ckpt' "\$S"); [ -f "${OUTPUT_DIR}/snaps/\$F" ] && return 0
    mkdir -p ${OUTPUT_DIR}/snaps; cp ${OUTPUT_DIR}/ckpts/last.ckpt ${OUTPUT_DIR}/snaps/\$F.tmp && mv ${OUTPUT_DIR}/snaps/\$F.tmp ${OUTPUT_DIR}/snaps/\$F \
        && echo "[snap] step=\$S (retained across a requeue) -> snaps/\$F" || { rm -f ${OUTPUT_DIR}/snaps/\$F.tmp; echo "[snap] step=\$S: copy of the retained checkpoint FAILED"; return 1; }
}
snap_loop() { while true; do for S in \$(grep -oE 'RESUME_AFTER_SAVE step=[0-9]+' "\$1" 2>/dev/null | grep -oE '[0-9]+\$' | sort -un); do [ \$((S % ${SNAP_EVERY})) -eq 0 ] && snap_one "\$S" "\$1"; done; sleep 30; done; }
JOB
fi
[ "$STAGE" = dit ] && cat >> "$JOB" <<JOB
snap_retained 2>&1 | tee -a "\${LOG}"
[ \${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: the retained last.ckpt is a step to keep and could not be copied; not starting a trainer that would overwrite it."; exit 3; }
JOB
EXTRA=""; [ "$STAGE" = dit ] && EXTRA='--override dataset.args.thumbnail_dir=${ROOT}/thumbnails'   # single quotes: ${ROOT} is the job's
cat >> "$JOB" <<JOB
srun python -u train.py \\
    --config ${CONFIG} --output_dir ${OUTPUT_DIR} --ckpt last --num_gpus ${NTASKS} \\
    --override trainer.args.batch_size_per_gpu=${BS} \\
    --override "trainer.args.wandb_name=${WANDB_NAME}" \\
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \\
    --override dataset.args.data_root=\${ROOT} \\
    --override dataset.args.parquet_file=\${ROOT}/df_lightgenbench.parquet \\
    --override dataset.args.split_json=\${ROOT}/data_splits_lightgenbench.json ${EXTRA} \\
    > >(tee -a "\${LOG}") 2>&1 &
TRAIN_PID=\$!
JOB
[ "$STAGE" = dit ] && cat >> "$JOB" <<JOB
snap_loop "\${LOG}" >> "\${LOG}" 2>&1 & SNAP_PID=\$!
JOB
cat >> "$JOB" <<JOB
wait \$TRAIN_PID; EXIT_CODE=\$?
JOB
[ "$STAGE" = dit ] && cat >> "$JOB" <<JOB
kill \$SNAP_PID 2>/dev/null
# The final save prints SAVE_START but no RESUME_AFTER_SAVE (the run ends instead of resuming): take it here.
SNAP_FAIL=0
if [ \$EXIT_CODE -eq 0 ] && grep -qE "SAVE_START step=${DIT_STEPS} " "\${LOG}"; then snap_one ${DIT_STEPS} "\${LOG}" 2>&1 | tee -a "\${LOG}"; [ \${PIPESTATUS[0]} -eq 0 ] || SNAP_FAIL=1; fi
JOB
cat >> "$JOB" <<JOB
if [ \$EXIT_CODE -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then echo "Training crashed (exit \$EXIT_CODE) at \$(date). Requeueing..."; scontrol requeue \${SLURM_JOB_ID}; exit 0; fi
    echo "Training crashed (exit \$EXIT_CODE); restart cap reached. NOT requeueing."; exit \$EXIT_CODE
fi
echo "Training completed at \$(date). Outputs: ${OUTPUT_DIR} — back them up to the project store from the workstation (compute-access post-run-backup.md)."
if [ "\${SNAP_FAIL:-0}" -ne 0 ]; then echo "BUT the step-${DIT_STEPS} snapshot was NOT written: copy ckpts/last.ckpt to snaps/ by hand before anything resumes this run."; exit 3; fi
JOB
fi
sed -i '/^[[:space:]]*$/d' "$JOB"   # drop the blank lines left by empty MEM/SIGNAL on trillium and an unset MAX_STEPS (a blank line would end the srun command)
if [ $DRY = 1 ]; then cat "$JOB"; exit 0; fi
cd -P "$REPO" || exit 2                                   # -P: killarney's submit plugin wants a physical /scratch cwd
# Update the code here, on the login node (it has the network); never edit on the cluster, pull what was pushed.
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH" || echo "WARNING: git pull --ff-only failed — local changes or diverged branch; submitting the tree as it is"
sbatch "$JOB"
