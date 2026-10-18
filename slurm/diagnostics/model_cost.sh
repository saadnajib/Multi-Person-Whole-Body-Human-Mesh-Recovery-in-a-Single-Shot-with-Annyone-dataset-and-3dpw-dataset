#!/bin/bash
#SBATCH --job-name=model-cost
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --exclude=gifu,glendale
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/model_cost_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/model_cost_%j.err
#
# model_cost.sh: params, FLOPs and (BENCH=1) ms/image of one or more checkpoints, on a GPU node.
#
# scripts/analysis/model_cost.py imports model.py, which needs EGL (the renderer); the login node has no EGL,
# so the script must run inside the container on a compute node, like every evaluation job.
#
# Use (from the repo root on login1; the script submits itself):
#   bash slurm/diagnostics/model_cost.sh                       # NAVER's Anny model vs the final model, with --bench
#   CKPTS="a.pt b.pt" TAG=mytag bash slurm/diagnostics/model_cost.sh
#   BENCH=0 bash slurm/diagnostics/model_cost.sh               # params and analytic FLOPs only
#
# Output: results/diagnostics/model_cost_<TAG>.json and the printed table in logs/model_cost_<jobid>.out
# (read it with: grep -vE "^Using cache|Warning|warn" logs/model_cost_<jobid>.out | tail -n 30).
# Latency is measured for model.py's Model class only; the original-repo class (NAVER's checkpoints) gets params and
# FLOPs, and its ms cell stays empty (model_cost.py, bench()).
# Never on V100 (gifu, glendale) or H100/H200: same rule as every other job.
set -euo pipefail
P="/netscratch/najib/multi-hmr"
cd "$P"
mkdir -p logs results/diagnostics

CKPTS="${CKPTS:-$P/models/multiHMR/multiHMR_672_L_anny.pt $P/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt}"
TAG="${TAG:-naver_vs_final}"
BENCH="${BENCH:-1}"
TIME="${TIME:-00:30:00}"
OUT_JSON="$P/results/diagnostics/model_cost_${TAG}.json"

CKPT_ARGS=""
for c in $CKPTS; do
  [ -f "$c" ] || { echo "ERROR: checkpoint not found: $c"; exit 1; }
  CKPT_ARGS="$CKPT_ARGS --ckpt $c"
done
[ -f scripts/analysis/model_cost.py ] || { echo "ERROR: scripts/analysis/model_cost.py missing (git pull)"; exit 1; }
BENCH_ARG=""; [ "$BENCH" = 1 ] && BENCH_ARG="--bench"

if [ "${MC_JOB:-0}" != 1 ]; then
  # ---- on the login node: submit this file as the job ----
  echo "=== model_cost: $CKPTS"
  echo "    bench $BENCH, time $TIME; json $OUT_JSON"
  JOB=$(sbatch --parsable --export=ALL,MC_JOB=1,CKPTS="$CKPTS",TAG="$TAG",BENCH="$BENCH" \
        --job-name="model-cost-${TAG}" --time="$TIME" "$P/slurm/diagnostics/model_cost.sh")
  echo "Submitted batch job $JOB"
  echo "log: $P/logs/model_cost_${JOB}.out   (send it when squeue -u najib no longer lists the job)"
  exit 0
fi

# ---- on the compute node ----
srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader || true
python scripts/analysis/model_cost.py ${CKPT_ARGS} ${BENCH_ARG} --json '${OUT_JSON}'
echo DONE: '${OUT_JSON}'"
