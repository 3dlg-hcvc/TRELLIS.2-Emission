#!/bin/bash
# Launch the pbr->emission VAE v3 on cs-venus-19 (RTX Pro 6000 Blackwell SE
# 96 GB). Copied from emission_dit_albedo2emission_pbrcond_74k_v2_venus19.sh,
# which is the venus-19-proven pattern; the deltas are the config, the GPU count
# and the absence of a DinoV3 image-cond override (a VAE has no image cond).
#
# WHAT THIS RUN IS. v3 gives the pbr->emission VAE the same architecture as the
# albedo->emission VAE apart from input/output channels, and finetunes it from
# the same pretrained PBR VAE. The shipped _v2 trained its emission encoder and
# decoder from random init at half width and half depth (128.63 M vs 828.60 M).
# See the header of configs/scvae/emission_vae_pbr2emission_74k_v3.yaml.
#
# GLOBAL BATCH 64 — the albedo->emission VAE's global batch, and the number that
# has to hold for the two chains to stay comparable. NUM_GPUS x BATCH_SIZE must
# always multiply to 64:
#   4 x 16  (default — all of venus-19, ~2x the throughput of the 2-GPU split)
#   2 x 32  (what albedo->emission itself ran; use when sharing the node)
# DDP makes these mathematically identical: both average gradients over the same
# 64 samples, and there is no BatchNorm anywhere in the model. Raising BATCH_SIZE
# to 32 on 4 GPUs would silently double the global batch to 128 and break the
# comparison this run exists to make.
#
# VRAM is not the constraint: measured 19.4 GB / 15.9 GB of 96 GB per card at
# BS 32 (probe job 244547), so BS 16 has ample headroom.
#
# --ckpt last is SAFE here and required for requeue: resolve_ckpt_path returns
# None on an empty output dir, so resume_mode stays None and on_fit_start still
# calls _apply_finetune_checkpoint (train.py:640-655, basic.py:104-106). But a
# NON-empty output dir sets resume_mode=lightning, which returns BEFORE the
# finetune load — so a probe must never share an output dir with the real run.
# OUT_SUFFIX exists for exactly that.
#
# Usage:
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=2:00:00 bash script/star2/emission_vae_pbr2emission_74k_v3_venus19.sh
#   bash script/star2/emission_vae_pbr2emission_74k_v3_venus19.sh   # full 35,800-step run

CONFIG="configs/scvae/emission_vae_pbr2emission_74k_v3_nonzero_nocopy.yaml"
NAME="emission_vae_pbr2emission_74k_v3_nonzero_nocopy"
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
WANDB_NAME="${WANDB_NAME:-vae_pbr->emission_74k_v3_nonzero_nocopy_venus19}"

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

# Bake the initial weights here rather than shipping them: data/ is gitignored,
# so a cluster clone cannot receive them via git pull, and the bake is
# deterministic and takes seconds. It reads the pretrained pair out of this
# node's HF cache (HF_HOME above), resolves the 4 shape-mismatched tensors plus
# the input_layer.bias fold, and asserts an exact 284/284 match against the
# config. Without this the trainer would silently skip those tensors and
# cold-start — the defect v3 exists to fix.
python -u data_toolkit/bake_pbr2emission_v3_init.py \
    --out data/ckpt_init/pbr2emission_v3 \
    --verify-config ${CONFIG} 2>&1 | tee -a "\${LOG}"
BAKE_EXIT=\${pipestatus[1]}
if [ \$BAKE_EXIT -ne 0 ]; then
    echo "Init bake failed (exit \$BAKE_EXIT) — refusing to train from an unverified init."
    exit \$BAKE_EXIT
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
