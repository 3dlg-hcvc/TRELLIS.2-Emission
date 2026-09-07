#!/bin/bash
# Launch the albedo->emission VAE v3 on cs-venus-19 (RTX Pro 6000 Blackwell SE
# 96 GB). Copied from emission_vae_pbr2emission_74k_v3_venus19.sh -- the other
# chain's launcher for the same node -- so the two runs share a submission path
# as well as a recipe.
#
# WHY VENUS-19 SPECIFICALLY. pbr->emission v3 trained here. Running albedo->
# emission v3 on the same node removes the last non-architectural difference
# between the two chains (site and silicon), on top of the config-level parity
# the v3 recipe already gives them.
#
# THIS IS A SECOND HORSE. The same run is queued on fir as job 55297448.
# Whichever starts first wins; cancel the other immediately -- two live runs
# would burn a node someone else wants and put two histories in one wandb
# project. The wandb names differ by the _venus19 suffix so a brief overlap is
# still legible.
#
# WHAT THIS RUN IS. decoder.latent_channels 32 -> 64, so the decoder consumes
# cat([z_emission 32, z_pbr 32]) -- the conditioning the pbr->emission decoder
# has had since its own v3. See the header of
# configs/scvae/emission_pbr_vae_ft_74k_v3.yaml.
#
# GLOBAL BATCH 64 -- the number that has to hold for the two chains to stay
# comparable. NUM_GPUS x BATCH_SIZE must always multiply to 64:
#   4 x 16  (default -- all of venus-19)
#   2 x 32  (use when sharing the node)
# DDP makes these identical: both average gradients over the same 64 samples and
# there is no BatchNorm anywhere in the model.
#
# VRAM is not the constraint on 96 GB cards. Note the two numbers in circulation
# for BS 32 on this node -- 19.4 GB (probe job 244547) and 74 GB (the newbake
# long tail, checkpoints.md 2026-07-23). They are not in conflict: the first is a
# typical batch, the second is what the heaviest samples cost. Both fit here;
# neither would fit reliably on a 48 GB card, which is why this run is not on the
# idle L40S nodes.
#
# --ckpt last is SAFE here and required for requeue: resolve_ckpt_path returns
# None on an empty output dir, so resume_mode stays None and on_fit_start still
# calls _apply_finetune_checkpoint (train.py:640-655, basic.py:104-106). A
# NON-empty output dir sets resume_mode=lightning, which returns BEFORE the
# finetune load -- so a probe must never share an output dir with the real run.
#
# Usage:
#   bash script/star2/emission_pbr_vae_ft_74k_v3_venus19.sh                       # full 35,800-step run
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=2:00:00 bash script/star2/emission_pbr_vae_ft_74k_v3_venus19.sh

CONFIG="configs/scvae/emission_pbr_vae_ft_74k_v3.yaml"
NAME="emission_pbr_vae_ft_74k_v3"
BACKUP_STORE="/3dlg-jupiter-project/lightgen"   # project store as a Solar node sees it (AGENTS.md "Filesystem")
TAG="venus19"
REPO="/localscratch/dya78/TRELLIS.2-lightning"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="${REPO}/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WALLTIME="${WALLTIME:-4-00:00:00}"
WANDB_NAME="${WANDB_NAME:-vae_albedo->emission_74k_v3_venus19}"

# The albedo->emission VAE's global batch. Comparing the two chains requires this
# to hold, and it is the easiest thing to break silently by changing GPU count
# without changing the micro-batch. Refuse to submit rather than train 3 days at
# the wrong effective batch size. GLOBAL_BATCH=<n> to deliberately run something
# else.
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
ACTUAL=$((NUM_GPUS * BATCH_SIZE))
if [ "${ACTUAL}" -ne "${GLOBAL_BATCH}" ]; then
    echo "REFUSING TO SUBMIT: NUM_GPUS(${NUM_GPUS}) x BATCH_SIZE(${BATCH_SIZE}) = ${ACTUAL}," >&2
    echo "  but the albedo->emission baseline's global batch is ${GLOBAL_BATCH}." >&2
    echo "  Fix the micro-batch, or set GLOBAL_BATCH=${ACTUAL} if you mean it." >&2
    exit 1
fi

echo "Submitting job: ${NAME}_${TAG}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS} x batch ${BATCH_SIZE} = global ${ACTUAL}"
echo "Max steps: ${MAX_STEPS:-<config default 35800>}"

ssh star2 sbatch << EOF
#!/bin/zsh
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH -w cs-venus-19
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_se:${NUM_GPUS}
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/localscratch/dya78/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@900

handle_timeout() {
    echo "SIGTERM received. Stopping training gracefully..."
    if [ -n "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null; then
        kill -TERM \$TRAIN_PID 2>/dev/null
        wait \$TRAIN_PID
    fi
    scontrol requeue \${SLURM_JOB_ID}
    exit 0
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

export TMPDIR=/localscratch/dya78/tmp
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_HOME=/localscratch/dya78/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
export SLURM_JOB_NAME=bash

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG="log/\${TS}-${NAME}${OUT_SUFFIX}_${TAG}.log"

# Fail fast and loudly if the data this chain needs is not on THIS node.
# /localscratch is per-node on star2, and venus-19 ran pbr->emission, which reads
# pbr_voxels and encodes them live. albedo->emission v3 instead reads the
# PRE-ENCODED pbr_latents, which that run never needed -- so their presence here
# is not implied by anything and has to be checked.
# Count without a glob: 72,421 paths in the argv is "Argument list too long", and
# the count silently comes back 0.
VOXDIR=${REPO}/data/lightgen_74k_newbake/emission_voxels_256
LATDIR=${REPO}/data/lightgen_74k_newbake/pbr_latents/tex_enc_next_dc_f16c32_fp16_256
for d in "\${VOXDIR}" "\${LATDIR}"; do
    if [ ! -d "\${d}" ]; then
        echo "FATAL: \${d} does not exist on \$(hostname)."
        echo "--- what IS under \${REPO}/data/lightgen_74k_newbake (so one failed queue wait is enough) ---"
        ls -la "${REPO}/data/lightgen_74k_newbake" 2>/dev/null || echo "  (the data root itself is missing)"
        echo "--- pbr_latents subdirs ---"
        ls -la "${REPO}/data/lightgen_74k_newbake/pbr_latents" 2>/dev/null || echo "  (no pbr_latents dir)"
        echo "--- free space on /localscratch ---"
        df -h /localscratch 2>/dev/null | tail -1
        exit 1
    fi
done
NVOX=\$(ls "\${VOXDIR}" | grep -c "\\.vxz\$")
NLAT=\$(ls "\${LATDIR}" | grep -c "\\.npz\$")
echo "emission voxels: \${NVOX} (expected 72421)"
echo "pbr latents:     \${NLAT} (expected 72421)"
if [ "\${NVOX}" -lt 72421 ] || [ "\${NLAT}" -lt 72421 ]; then
    echo "FATAL: incomplete data on \$(hostname) — refusing to train on a different split than fir."
    exit 1
fi

# Bake the initial weights here rather than shipping them: data/ is gitignored,
# so a cluster clone cannot receive them via git pull, and the bake takes
# seconds and is deterministic. Exactly one tensor changes shape on this chain
# (decoder.from_latent.weight 32 -> 64, new half zeroed); the loader would
# otherwise warn once and SKIP it, cold-starting the decoder's only latent
# gateway. The snapshot hash differs per host, so resolve it rather than
# hardcoding.
SNAP=\$(ls -d \${HF_HOME}/hub/models--microsoft--TRELLIS.2-4B/snapshots/*/ckpts 2>/dev/null | head -1)
if [ -z "\${SNAP}" ]; then
    echo "FATAL: no TRELLIS.2-4B snapshot under \${HF_HOME}/hub — cannot bake the init."
    exit 1
fi
export TRELLIS2_SNAP="\${SNAP}"
echo "snapshot: \${TRELLIS2_SNAP}"

python -u data_toolkit/bake_albedo2emission_v3_init.py \
    --out data/ckpt_init/albedo2emission_v3 2>&1 | tee -a "\${LOG}"
BAKE_EXIT=\${pipestatus[1]}
if [ \$BAKE_EXIT -ne 0 ]; then
    echo "Init bake failed (exit \$BAKE_EXIT) — refusing to train from an unverified init."
    exit \$BAKE_EXIT
fi
python -u data_toolkit/verify_albedo2emission_v3/verify_identity.py 2>&1 | tee -a "\${LOG}"
if [ \${pipestatus[1]} -ne 0 ]; then
    echo "Baked init failed verification — refusing to train."
    exit 1
fi
python -u data_toolkit/verify_albedo2emission_v3/verify_config_diff.py 2>&1 | tee -a "\${LOG}"
if [ \${pipestatus[1]} -ne 0 ]; then
    echo "Config drifted from the pbr->emission v3 recipe — refusing to train."
    exit 1
fi

python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus ${NUM_GPUS} \
    --override "trainer.args.wandb_name=${WANDB_NAME}" \
    --override trainer.args.batch_size_per_gpu=${BATCH_SIZE} \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then
        echo "Training crashed (exit code \$TRAIN_EXIT) at \$(date). Requeueing job..."
        scontrol requeue \${SLURM_JOB_ID}
        exit 0
    fi
    echo "Training crashed (exit code \$TRAIN_EXIT) at \$(date); restart cap reached (\${SLURM_RESTART_COUNT}). NOT requeueing."
    exit \$TRAIN_EXIT
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
