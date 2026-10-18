#!/bin/bash
# The training checks (train-check.sh) of one TRELLIS.2-Emission tree on one Solar node's L40S GPUs, one task per
# GPU at a time. GPU type = L40S; compare a record only with records of the same type (REFACTOR_TODO.md,
# How we work). The shared Solar env ~/envs/trellis2 runs on sm_89 only; this job refuses other cards.
#
# On star2's shared $HOME (rsync from the workstation; C = lightgen_eval_local/trellis2_refactor_checks):
#   rsync -a <tree>/ star2:t2refactor/trees/<name>/ --exclude outputs --exclude data   (a tree.sh tree)
#   rsync -a --delete script/internal/gate/ star2:t2refactor/tooling/gate/
#   rsync -a --delete C/check_root/ star2:t2refactor/check_root/
#   rsync -a TRELLIS.2-Emission/data/ckpt_init/pbr2emission_v3/ star2:t2refactor/ckpt_init/pbr2emission_v3/   (once)
# The resume checkpoints are read from jupiter (/3dlg-jupiter-project, read only): the release VAE's
# ckpts/last.ckpt and the release DiT's snaps/last-step0050000.ckpt with their config.yaml.
#
# Submit (never `sbatch --export` on star2; never -w on our nodes, a pinned job evicts another lab's):
#   sed -e 's|__TREE__|pre-refactor|' -e 's|__LABEL__|pre-refactor_r1|' -e 's|__NGPU__|2|' \
#       -e 's|__TASKS__|dit:fresh dit:resume|' -e 's|__NCPU__|16|' -e 's|__MEM__|160G|' -e 's|__TIME__|01:30:00|' \
#       script/internal/gate/star2/checks.sbatch > /tmp/t2checks.sbatch
#   ssh star2 'sbatch --deadline=now+<time + 5 min>' < /tmp/t2checks.sbatch
# __TASKS__: net:mode pairs (vae|dit : fresh|resume); task j goes to GPU j mod NGPU, each GPU runs its
# tasks one after another. Output: ~/t2refactor/out/__LABEL__/<net>_<mode>/{record.jsonl,gpu.txt,train.log}
# and status.txt (one line per task, then CHECKS_DONE or CHECKS_INCOMPLETE).
# The DiT's training does not repeat exactly (its sparse attention runs flash-attention's backward
# without the deterministic option): compare its records within the envelope of several pre-refactor
# runs. Each GPU keeps its own flex_gemm autotune cache file under the job directory.
# Exit 0 only when every task wrote TRAINCHECK_DONE and its outputs reached ~/t2refactor/out; on a
# timeout or a kill the outputs so far are copied to <label>/partial/ before the scratch goes.
#SBATCH --job-name=t2refactor___LABEL__
#SBATCH --gres=gpu:l40s:__NGPU__
#SBATCH --cpus-per-task=__NCPU__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=/home/dya78/t2refactor/logs/checks-%j.out
set -uo pipefail
TREE_NAME="__TREE__"; LABEL="__LABEL__"; NGPU=__NGPU__; TASKS="__TASKS__"
H=$HOME/t2refactor; JOB=${SLURM_JOB_ID:-$$}
LS=/localscratch/dya78/t2refactor; W=$LS/jobs/$JOB; OUT=$W/out; HOUT=$H/out/$LABEL
JUP=/3dlg-jupiter-project/lightgen
V=emission_vae_pbr2emission_lightgenbench_v1; D=emission_dit_pbr2emission_lightgenbench_v1
say() { echo "[t2checks $(date +%H:%M:%S)] $*"; }
die() { echo "CHECKS_FAIL label=$LABEL node=$(hostname -s) job=$JOB reason=$*"; mkdir -p "$HOUT"; echo "CHECKS_FAIL $*" >> "$HOUT/status.txt"; exit 1; }
cleanup() { [ -d "$OUT" ] && rsync -rlt --exclude run "$OUT/" "$HOUT/partial/" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT; trap 'exit 143' TERM
say "node=$(hostname -s) job=$JOB tree=$TREE_NAME label=$LABEL tasks='$TASKS' gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | xargs)
[ "$CC" = "8.9" ] || die "compute_cap $CC: the shared trellis2 env runs on sm_89 only"
[ -d "$H/trees/$TREE_NAME" ] || die "no tree $H/trees/$TREE_NAME"
mkdir -p "$W/main/outputs/$V/ckpts" "$W/main/outputs/$D/snaps" "$W/main/data/ckpt_init" "$OUT" "$HOUT" "$W/tmp" || die "mkdir"

# --- the tree, the check root and the init on node-local disk; the resume checkpoints linked from jupiter
rsync -rlt "$H/trees/$TREE_NAME/" "$W/tree/" || die "tree copy"
ln -s "$W/main/outputs" "$W/tree/outputs"; ln -s "$W/main/data" "$W/tree/data"
rsync -rlt "$H/check_root/" "$W/check_root/" || die "check_root copy"
rsync -rlt "$H/ckpt_init/" "$W/main/data/ckpt_init/" || die "ckpt_init copy"
ln -s "$JUP/outputs/$V/ckpts/last.ckpt" "$W/main/outputs/$V/ckpts/last.ckpt"
ln -s "$JUP/outputs/$V/config.yaml" "$W/main/outputs/$V/config.yaml"
ln -s "$JUP/outputs/$D/snaps/last-step0050000.ckpt" "$W/main/outputs/$D/snaps/last-step0050000.ckpt"
ln -s "$JUP/outputs/$D/config.yaml" "$W/main/outputs/$D/config.yaml"
[ "$(find "$W/main" -xtype l | wc -l)" = 0 ] || die "dangling checkpoint links"

# --- environment: the shared Solar env and the jupiter HF cache (as trellis2_release/infer_solar.sbatch)
cat > "$W/activate.sh" <<EOF
export PATH=$HOME/envs/trellis2/bin:\$PATH
export HF_HOME=$JUP/trellis2_bw/eval_inputs/hf_home HF_HUB_CACHE=$JUP/trellis2_bw/eval_inputs/hf_home/hub
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export XDG_CACHE_HOME=$W/xdg TRITON_CACHE_DIR=$W/triton
EOF

# --- tasks, one queue per GPU; GPU g of the queues is the g-th device Slurm allocated
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU - 1)))}"
[ "${#DEVS[@]}" -ge "$NGPU" ] || die "Slurm allocated ${#DEVS[@]} GPUs (${CUDA_VISIBLE_DEVICES:-unset}), NGPU=$NGPU"
i=0; declare -a Q
for t in $TASKS; do g=$((i % NGPU)); Q[$g]="${Q[$g]:-} $t"; i=$((i + 1)); done
run_queue() {  # <gpu> <tasks...>
  local g=$1; shift
  for t in "$@"; do
    local net=${t%%:*} mode=${t##*:} o=$OUT/${t/:/_}
    say "gpu $g: $net $mode"
    CUDA_VISIBLE_DEVICES=${DEVS[$g]} GATE_LOCK=none GATE_ACTIVATE=$W/activate.sh GATE_CHECK_ROOT=$W/check_root \
      FLEX_GEMM_AUTOTUNE_CACHE_PATH=$W/flex_gemm_autotune_$g.json TRAINCHECK_TMPBASE=$W/tmp \
      bash "$H/tooling/gate/train-check.sh" "$W/tree" "$o" "$net" "$mode" > "$OUT/${t/:/_}.log" 2>&1
    local r=$?
    mkdir -p "$HOUT/${t/:/_}"
    local c=0; rsync -rlt --exclude run "$o/" "$HOUT/${t/:/_}/" && cp "$OUT/${t/:/_}.log" "$HOUT/${t/:/_}/job.log" || c=1
    if [ $r = 0 ] && [ $c = 0 ] && grep -q TRAINCHECK_DONE "$OUT/${t/:/_}.log"; then echo "OK $t gpu=$g" >> "$HOUT/status.txt"
    else echo "FAIL $t gpu=$g rc=$r copy=$c" >> "$HOUT/status.txt"; tail -5 "$OUT/${t/:/_}.log"; fi
  done
}
for g in $(seq 0 $((NGPU - 1))); do [ -n "${Q[$g]:-}" ] && run_queue "$g" ${Q[$g]} & done
wait
NOK=$(grep -c '^OK ' "$HOUT/status.txt" 2>/dev/null || true); NT=$(echo $TASKS | wc -w)
if [ "$NOK" = "$NT" ] && ! grep -q '^FAIL' "$HOUT/status.txt"; then echo CHECKS_DONE >> "$HOUT/status.txt"; say CHECKS_DONE
else echo "CHECKS_INCOMPLETE ($NOK of $NT ok)" >> "$HOUT/status.txt"; say CHECKS_INCOMPLETE; exit 1; fi
