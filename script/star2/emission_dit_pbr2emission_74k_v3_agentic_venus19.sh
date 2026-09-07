#!/bin/bash
# pbr->emission DiT v3 on the _agentic split — cs-venus-19 (4x RTX Pro 6000 Blackwell SE 96 GB).
# Standby site: venus-19 is the ONE place besides fir where this chain can run today, because it
# already has the trellis2-bw env, the repo, pbr_latents + shape_latents + thumbnails (72,421
# each) and both HF snapshots. Verified 2026-08-25; the only things missing were the _agentic
# emission latents and split, staged from jupiter the same day.
#
# (venus-05 was checked and rejected: it has NO trellis2-bw env and NO repo — only the Hunyuan
# and TEXGen envs — and its pbr/shape latent trees are gone. It would need a ~3-4 h bootstrap.)
#
# KEEP the DinoV3 --override below. Unlike the fir launchers, which resolve the hub id from
# ~/.cache under HF_HUB_OFFLINE=1, venus-19's transformers 5.14 + hf_hub 1.24 cannot resolve
# hub ids offline against its older cache layout, so the local snapshot path is required.
#
# Usage:
#   bash script/star2/emission_dit_pbr2emission_74k_v3_agentic_venus19.sh
#   OUT_SUFFIX=_probe MAX_STEPS=100 WALLTIME=2:00:00 bash script/star2/emission_dit_pbr2emission_74k_v3_agentic_venus19.sh

CONFIG="configs/gen/emission_dit_pbr2emission_74k_v3_agentic.yaml"
NAME="emission_dit_pbr2emission_74k_v3_agentic"
BACKUP_STORE="/3dlg-jupiter-project/lightgen"   # project store as a Solar node sees it (AGENTS.md "Filesystem")
TAG="venus19"
REPO="/localscratch/dya78/TRELLIS.2-lightning"
# OUT_SUFFIX (e.g. _probe) keeps a probe out of the real run dir: a NON-empty
# output dir flips resume_mode=lightning and silently SKIPS the finetune load
# (same trap as the VAE launcher documents; mechanism copied from it).
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUTPUT_DIR="${REPO}/outputs/${NAME}${OUT_SUFFIX}"
BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"
WALLTIME="${WALLTIME:-2-00:00:00}"
WANDB_NAME="${WANDB_NAME:-dit_pbr->emission_74k_v3_agentic_venus19}"

echo "Submitting job: ${NAME}_${TAG}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

ssh star2 sbatch << EOF
#!/bin/zsh
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}${OUT_SUFFIX}_${TAG}
#SBATCH -w cs-venus-19
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_se:4
#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=/localscratch/dya78/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
# last.ckpt for the 1.3B DiT is ~15-20 GB; /localscratch NVMe writes it in
# ~1-2 min. 900 s of SIGTERM lead is ample (fir needed 1500 s on Lustre).
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
# venus-19 root FS was 100% full -> HF cache relocated to /localscratch
# (~/.cache/huggingface is now a symlink there)
export HF_HOME=/localscratch/dya78/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
# flex_gemm autotune-cache races across ranks -> in-memory autotune only
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0
export FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0
# Lightning: disable the SLURM-coordinated launcher (ntasks=1 vs devices=4
# ValueError) so --num_gpus 4 native-spawns — star2 single-task convention,
# same as script/star2/emission_dit_1k_probe.sh.
export SLURM_JOB_NAME=bash

mkdir -p log
TS=\$(date +%Y%m%d-%H%M%S)
LOG="log/\${TS}-${NAME}_${TAG}.log"

# NOTE: > >(tee) process substitution, NOT a tee pipe — \$! must be train.py's
# PID (the TERM trap signals it and \$? must be its exit), and a backgrounded
# pipeline would hand both to tee instead.
python -u train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt last \
    --num_gpus 4 \
    --override "trainer.args.wandb_name=${WANDB_NAME}" \
    --override trainer.args.image_cond_model.args.model_name=/localscratch/dya78/hf_home/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51 \
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID
TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -ne 0 ]; then
    # Cap requeues: transient faults deserve a retry, deterministic errors must
    # not crash-loop (242795 self-requeued 70x on a config error before the cap).
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

echo "Submitted: ${NAME}_${TAG}"
