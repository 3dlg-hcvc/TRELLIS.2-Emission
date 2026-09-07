#!/bin/bash
# pbr->emission DiT v3 on the _agentic split — cs-venus-05 (4x RTX Pro 6000 Blackwell Max-Q).
# Third horse in the 2026-08-24 race: fir 56669773 (dya78/rrg-msavva) and fir 56675384
# (xya120/rrg-angelx) are the other two. First to start wins; the others are cancelled.
#
# SELF-STAGING. Unlike the _v2 venus-05 launcher this is ported from, the _agentic emission
# latents did not exist when venus-05 was last staged -- they were encoded on the workstation
# 2026-08-24 and pushed to jupiter. venus-05 mounts jupiter at /3dlg-jupiter-project (no /cs
# prefix), so the job stages them itself on first run and skips the work on a requeue.
# pbr_latents / shape_latents / thumbnails / parquet were staged for the _v2 work and persist
# on /localscratch across reboots; the job gates on them rather than re-copying.
#
# Usage:  bash script/star2/emission_dit_pbr2emission_74k_v3_agentic_venus05.sh
#         OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=2:00:00 bash ...   # probe into a separate dir

CONFIG="configs/gen/emission_dit_pbr2emission_74k_v3_agentic.yaml"
NAME="emission_dit_pbr2emission_74k_v3_agentic"
BACKUP_STORE="/3dlg-jupiter-project/lightgen"   # project store as a Solar node sees it (AGENTS.md "Filesystem")
TAG="venus05"
REPO="/localscratch/dya78/TRELLIS.2-lightning"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="${REPO}/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
WALLTIME="${WALLTIME:-4-00:00:00}"
WANDB_NAME="${WANDB_NAME:-dit_pbr->emission_74k_v3_agentic_venus05}"
SPLIT_MD5="159492b2b8d104ab63f3d13eeea394d0"
LATENT_NAME="emission_pbr2emission_74k_v3_agentic"

echo "Submitting ${NAME}${OUT_SUFFIX}_${TAG}: 4 x BS 8 = global 32; walltime ${WALLTIME}; max_steps ${MAX_STEPS:-<config 50000>}"

ssh star2 sbatch << EOF
#!/bin/zsh
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:4
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=/localscratch/dya78/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@900

handle_timeout() {
    echo "SIGTERM received. Stopping training gracefully..."
    if [ -n "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null; wait \$TRAIN_PID
    fi
    scontrol requeue \${SLURM_JOB_ID}; exit 0
}
trap 'handle_timeout' TERM

echo "======================================"
echo "Job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})  restart_count=\${SLURM_RESTART_COUNT:-0}"
echo "Started: \$(date)  Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================"

source ~/.zshrc
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate trellis2-bw

cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
echo "repo at: \$(git log --oneline -1)"

D=${REPO}/data/lightgen_74k_newbake
J=/3dlg-jupiter-project/lightgen/trellis2_bw/lightgen_74k_newbake

# --- self-stage the two _agentic artefacts (idempotent; a requeue skips both) ---
mkdir -p \${D}/emission_latents
NEMIT=\$(ls \${D}/emission_latents/${LATENT_NAME} 2>/dev/null | grep -c "\.npz\$")
if [ "\${NEMIT}" -lt 72421 ]; then
    echo "staging ${LATENT_NAME} from jupiter (have \${NEMIT}/72421)..."
    [ -f \${J}/emission_latents/agentic_latents.tar ] || { echo "FATAL: jupiter tar missing"; exit 1; }
    cp \${J}/emission_latents/agentic_latents.tar /localscratch/dya78/ || exit 1
    tar xf /localscratch/dya78/agentic_latents.tar -C \${D}/emission_latents/ || exit 1
    rm -f /localscratch/dya78/agentic_latents.tar
    NEMIT=\$(ls \${D}/emission_latents/${LATENT_NAME} | grep -c "\.npz\$")
fi
echo "emission_latents: \${NEMIT} (expected 72421)"
[ "\${NEMIT}" -ge 72421 ] || { echo "FATAL: emission latents incomplete"; exit 1; }
[ -f \${D}/emission_latents/${LATENT_NAME}/normalization_stats.json ] || { echo "FATAL: stats json missing"; exit 1; }

SPLIT=\${D}/data_splits_emissive_74k_stratified_newbake_vae_agentic.json
[ -f "\${SPLIT}" ] || cp \${J}/data_splits_emissive_74k_stratified_newbake_vae_agentic.json \${SPLIT} || exit 1
GOT=\$(md5sum \${SPLIT} | cut -d' ' -f1)
[ "\${GOT}" = "${SPLIT_MD5}" ] || { echo "FATAL: split md5 \${GOT} != ${SPLIT_MD5}"; exit 1; }
echo "split md5 OK"

# --- gate the artefacts staged for the _v2 work (must already be here) ---
NPBR=\$(ls \${D}/pbr_latents/tex_enc_next_dc_f16c32_fp16_256 2>/dev/null | grep -c "\.npz\$")
NSHP=\$(ls \${D}/shape_latents/shape_enc_next_dc_f16c32_fp16_256 2>/dev/null | grep -c "\.npz\$")
NTHM=\$(ls \${D}/thumbnails 2>/dev/null | wc -l)
echo "pbr=\${NPBR} shape=\${NSHP} thumbs=\${NTHM} (each expected 72421)"
for v in \${NPBR} \${NSHP} \${NTHM}; do
    [ "\${v}" -ge 72421 ] || { echo "FATAL: a _v2-era artefact is missing on /localscratch; node needs re-staging"; exit 1; }
done

export TMPDIR=/localscratch/dya78/tmp
export HF_HUB_OFFLINE=1
export HF_HOME=/localscratch/dya78/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
export WANDB_MODE=offline
# Lightning asserts ntasks == devices under a real Slurm job name; masking it makes
# --num_gpus 4 native-spawn instead -- the star2 single-task convention.
export SLURM_JOB_NAME=bash
set -o pipefail

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG="log/\${TS}-${NAME}${OUT_SUFFIX}_${TAG}.log"

python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    --override trainer.args.batch_size_per_gpu=8 \
    --override "trainer.args.wandb_name=${WANDB_NAME}" \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then
        echo "Training crashed (exit \$TRAIN_EXIT) at \$(date). Requeueing..."
        scontrol requeue \${SLURM_JOB_ID}; exit 0
    fi
    echo "Training crashed (exit \$TRAIN_EXIT); restart cap reached. NOT requeueing."; exit \$TRAIN_EXIT
fi
echo "Training completed at \$(date)."
# Post-run backup (AGENTS.md "After a training run"): a CPU-only job on this node mirrors the run
# into the project store and verifies sha256 on both ends — after the GPUs are released, never
# touching this job's exit code. BACKUP_STORE is the project's store as seen from a Solar node.
if [ -f ${REPO}/script/star2/backup_run.sh ]; then
  sbatch -J backup_${NAME}${OUT_SUFFIX:-} -w "\$(hostname -s)" -c 2 --mem=4G --time=8:00:00 \\
    --output=/localscratch/dya78/backup_${NAME}${OUT_SUFFIX:-}_%j.out \\
    --wrap="bash ${REPO}/script/star2/backup_run.sh '${OUTPUT_DIR}' '${BACKUP_STORE}' '${NAME}${OUT_SUFFIX:-}'" \\
    || echo "WARNING: backup job not submitted — run backup_run.sh by hand."
else
  echo "WARNING: ${REPO}/script/star2/backup_run.sh missing on this node — pull the repo, then back up by hand."
fi
exit \$TRAIN_EXIT
EOF
echo "Submitted: ${NAME}${OUT_SUFFIX}_${TAG}"
