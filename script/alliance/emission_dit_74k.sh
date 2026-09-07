#!/bin/bash
# Launch emission DiT 74k training (4 x H100, one node) on ANY Alliance cluster — the fir launcher made cluster-aware.
# Run it on the cluster's login node from the repo root (hedge.sh does: cd ~/scratch/TRELLIS.2-lightning && bash script/alliance/emission_dit_74k.sh).
#
#   bash script/alliance/emission_dit_74k.sh [--dry-run]      # --dry-run prints the sbatch script instead of submitting
#   env: MAX_STEPS=100 for a bounded probe; WALLTIME=HH:MM:SS (default 03:00:00); CLUSTER=<name> to override detection
#
# Per-cluster facts (compute-access skill, alliance/overview.md, 2026-09-07): account, GPU request form, memory ceiling,
# sbatch quirks. trillium's sbatch is SciNet's wrapper: no --mem, no --gres, --gpus-per-node=N only, -t in minutes,
# no --signal; it needs a script file, so every cluster submits a file (never stdin). The module set is fir's everywhere.
set -u
CONFIG="configs/gen/emission_dit_74k.yaml"; NAME="emission_dit_74k"; BRANCH="lightgen"
MAX_STEPS="${MAX_STEPS:-}"; WALLTIME="${WALLTIME:-03:00:00}"; DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
CLUSTER="${CLUSTER:-$(scontrol show config 2>/dev/null | awk -F'= ' '/^ClusterName/{print $2}')}"
REPO="${REPO:-$HOME/scratch/TRELLIS.2-lightning}"; OUTPUT_DIR="$REPO/outputs/$NAME"; TAR="$HOME/scratch/lightgen/lightgen_74k_dit.tar"
MODULES="StdEnv/2023 gcc python/3.11 cuda/12.6 arrow/21.0.0 opencv/4.13.0 sparsehash"
SIGNAL="#SBATCH --signal=B:SIGTERM@1500"; MEM="#SBATCH --mem=1024G"; GPU="#SBATCH --gpus-per-node=h100:4"; TIME="#SBATCH --time=${WALLTIME}"
case "$CLUSTER" in
  fir)       ACCOUNT=rrg-msavva ;;
  nibi)      ACCOUNT=def-msavva_gpu ;;
  rorqual)   ACCOUNT=def-msavva_gpu; MEM="#SBATCH --mem=450G" ;;                       # 4xH100 nodes carry 510G
  killarney) ACCOUNT=aip-msavva ;;                                                    # 8xH100 nodes, 2060G: half a node
  tamia)     ACCOUNT=aip-msavva; MEM="#SBATCH --mem=900G"; GPU="#SBATCH --gres=gpu:h100:4" ;;   # whole 4xH100 node (1000G)
  grillium|trillium*) ACCOUNT=def-msavva; MEM=""; SIGNAL=""; GPU="#SBATCH --gpus-per-node=4"
             TIME="#SBATCH -t $(( $(echo "$WALLTIME" | awk -F: '{print $1*60+$2}') ))" ;;     # minutes; same modules as fir
  *) echo "unknown cluster '$CLUSTER' — set CLUSTER=…"; exit 2 ;;
esac
[ -f "$TAR" ] || { echo "dataset tar missing: $TAR"; [ $DRY = 1 ] || exit 2; }
[ -x "$REPO/env/bin/python" ] || { echo "environment missing: $REPO/env"; [ $DRY = 1 ] || exit 2; }
mkdir -p "$REPO/log"; JOB="$REPO/log/job_${NAME}_${CLUSTER}.sh"   # log/ is gitignored
cat > "$JOB" <<JOB
#!/bin/bash
#SBATCH --mail-user=yangdongchen1@gmail.com
#SBATCH --mail-type=END
#SBATCH -J ${NAME}
${GPU}
${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=%N-%j.out
#SBATCH --open-mode=append
${MEM}
#SBATCH --account=${ACCOUNT}
#SBATCH --requeue
${SIGNAL}
# 25-min lead: the 20 GB last.ckpt save takes ~10-13 min on Lustre; a short window SIGKILLs mid-write -> 0-byte ckpt.
handle_timeout() { echo "SIGTERM: stopping training gracefully..."; [ -n "\$TRAIN_PID" ] && kill -0 \$TRAIN_PID 2>/dev/null && { kill -TERM \$TRAIN_PID; wait \$TRAIN_PID; }; scontrol requeue \${SLURM_JOB_ID}; exit 0; }
trap 'handle_timeout' SIGTERM
sleep 60   # hedged submission: a job cancelled in its first minute has cost nothing (compute-access hedged-submission.md)
echo "====================================== Job \${SLURM_JOB_NAME} (\${SLURM_JOB_ID}) cluster ${CLUSTER} node \$(hostname) started \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module load ${MODULES}
cd ${REPO}
git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}
source ${REPO}/env/bin/activate
cd \$SLURM_TMPDIR
bash ${REPO}/script/fir/unpack_74k_dit_data.sh ${TAR}
DATA_ROOT=\${SLURM_TMPDIR}/lightgen_74k
cd ${REPO}
mkdir -p log; TS=\$(date +%Y%m%d-%H%M%S); LOG="log/\${TS}-${NAME}-${CLUSTER}.log"
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0   # NFS-home autotune cache races between ranks
# Pretrained weights (DINOv3) come from a Hugging Face cache staged in scratch — compute nodes on nibi/rorqual/trillium have no internet.
export HF_HOME=\$HOME/scratch/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# wandb goes online only where it can: a login (~/.netrc) AND a reachable API; otherwise offline — 'wandb sync outputs/<run>/wandb/offline-run-*' later from a login node.
if [ -f \$HOME/.netrc ] && curl -sSI -m 8 https://api.wandb.ai >/dev/null 2>&1; then echo "wandb: online"; else export WANDB_MODE=offline; echo "wandb: OFFLINE (no login or no internet on this node)"; fi
# trillium compute nodes mount \$HOME read-only: every per-user cache that would land there goes to the node-local disk instead.
if [ ! -w \$HOME ]; then export TRITON_CACHE_DIR=\$SLURM_TMPDIR/.triton CUDA_CACHE_PATH=\$SLURM_TMPDIR/.nv MPLCONFIGDIR=\$SLURM_TMPDIR/.mpl WANDB_CACHE_DIR=\$SLURM_TMPDIR/.wandb WANDB_CONFIG_DIR=\$SLURM_TMPDIR/.wandb_cfg XDG_CACHE_HOME=\$SLURM_TMPDIR/.cache; echo "HOME read-only: caches redirected to \$SLURM_TMPDIR"; fi
set -o pipefail
srun python -u train.py \\
    --config ${CONFIG} --output_dir ${OUTPUT_DIR} --ckpt last --num_gpus 4 \\
    ${MAX_STEPS:+--override trainer.args.max_steps=${MAX_STEPS}} \\
    --override dataset.args.data_root=\${DATA_ROOT} \\
    --override dataset.args.parquet_file=\${DATA_ROOT}/df_SomgProc_final.parquet \\
    --override dataset.args.split_json=\${DATA_ROOT}/data_splits_emissive_74k_stratified_voxonly.json \\
    --override dataset.args.thumbnail_dir=\${DATA_ROOT}/thumbnails \\
    > >(tee "\${LOG}") 2>&1 &
TRAIN_PID=\$!
wait \$TRAIN_PID; EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    if [ \${SLURM_RESTART_COUNT:-0} -lt 3 ]; then echo "Training crashed (exit \$EXIT_CODE) at \$(date). Requeueing..."; scontrol requeue \${SLURM_JOB_ID}; exit 0; fi
    echo "Training crashed (exit \$EXIT_CODE); restart cap reached. NOT requeueing."; exit \$EXIT_CODE
fi
echo "Training completed at \$(date). Outputs: ${OUTPUT_DIR} — back them up to the project store from the workstation (compute-access post-run-backup.md)."
JOB
sed -i '/^$/d' "$JOB"   # drop the blank lines left by empty MEM/SIGNAL on trillium
if [ $DRY = 1 ]; then cat "$JOB"; else cd -P "$REPO" && sbatch "$JOB"; fi   # -P: killarney's submit plugin wants a physical /scratch cwd
