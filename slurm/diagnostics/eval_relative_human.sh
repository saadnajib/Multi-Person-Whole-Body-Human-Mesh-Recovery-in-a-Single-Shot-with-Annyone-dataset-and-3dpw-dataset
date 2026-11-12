#!/bin/bash
# Relative Human (RH) evaluation: predicted Anny age / gender per RH age group on real photos
# (docs/handover.md 3.4 and section 5 item 12.1; docs/experiments_plan.md "Relative Human evaluation").
# Array: index 0 = v5 00099 (plain), index 1 = exp6 fine-tuned v5 (its bone fix is in the checkpoint args, BONE_FIX='').
#   sbatch --array=0-1 slurm/diagnostics/eval_relative_human.sh          # both models, val split
#   SPLIT=test sbatch --array=0-1 slurm/diagnostics/eval_relative_human.sh
#   MODEL=/path/ckpt.pt BONE_FIX=none sbatch slurm/diagnostics/eval_relative_human.sh   # one explicit model
#   MODEL=<exp8 run>/train/checkpoints/00029.pt BONE_FIX=<exp8 run>/bone_fix.json TAG=student_vitb_e30_00029 \
#     sbatch --array=0 slurm/diagnostics/eval_relative_human.sh    # the ViT-B student with its own bone fix (JSON path)
# Model class: model.py's Model only (diag_common.load_model, incl. projector-fed ViT-B / ViT-S students). The
#   original-repo class (models/multiHMR/multiHMR_672_L_anny.pt, train.py --model_class multi_hmr_original) is refused
#   by load_model ("not a model.py Model"): its output has no path through this script.
# PRINT_ONLY=1 bash slurm/diagnostics/eval_relative_human.sh   # print the resolved MODEL / TAG / OUT line and exit
# Re-score from the cache (no GPU): python scripts/diagnostics/eval_relative_human.py --score_only --out <OUT> --iou 0.5
# Env: MODEL, BONE_FIX (''), SPLIT (val), N (0 = all), OUT (results/diagnostics/relative_human_<split>_<tag>), RH_ROOT, TAG.
#SBATCH --job-name=eval-rh
#SBATCH --partition=A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV
#SBATCH --exclude=gifu,glendale
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval_rh_%A_%a.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval_rh_%A_%a.err
set -euo pipefail
[ "${PRINT_ONLY:-0}" = 1 ] || mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

ROOT_DIR=/netscratch/najib/multi-hmr
MODELS=(
  "${ROOT_DIR}/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"
  "${ROOT_DIR}/results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/train/checkpoints/00001.pt"
)
TAGS=(anny_s2_shape_v5_00099 exp6_v5_bonefix_00001)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ -z "${MODEL:-}" ]; then
  MODEL="${MODELS[$IDX]}"; TAG="${TAG:-${TAGS[$IDX]}}"
elif [ "$(basename "$(dirname "$MODEL")")" = checkpoints ]; then
  # <run>/checkpoints/<epoch>.pt and the exp7 / exp8 layout <run>/train/checkpoints/<epoch>.pt -> <run>_<epoch>
  # (before 12 Nov the second layout gave train_<epoch> for every run: one shared OUT, i.e. one shared cache)
  RUN_DIR_="$(dirname "$(dirname "$MODEL")")"
  [ "$(basename "$RUN_DIR_")" = train ] && RUN_DIR_="$(dirname "$RUN_DIR_")"
  TAG="${TAG:-$(basename "$RUN_DIR_")_$(basename "$MODEL" .pt)}"
else
  TAG="${TAG:-$(basename "$MODEL" .pt)}"
fi
BONE_FIX="${BONE_FIX:-}"
SPLIT="${SPLIT:-val}"
N="${N:-0}"
RH_ROOT="${RH_ROOT:-/netscratch/najib/data/Relative_Human/RelativeHuman}"
OUT="${OUT:-results/diagnostics/relative_human_${SPLIT}_${TAG}}"
echo "[eval_rh] idx=${IDX} MODEL=${MODEL} TAG=${TAG} BONE_FIX='${BONE_FIX}' SPLIT=${SPLIT} N=${N} OUT=${OUT}"
[ "${PRINT_ONLY:-0}" = 1 ] && exit 0

COMMAND=$(cat <<EOF2
set -euo pipefail
source ${ROOT_DIR}/slurm/env.sh
[ -f "${MODEL}" ] || { echo "ERROR: checkpoint not found: ${MODEL}"; exit 1; }
[ -f "${RH_ROOT}/${SPLIT}_annots.npz" ] || { echo "ERROR: RH annotations not found: ${RH_ROOT}/${SPLIT}_annots.npz"; exit 1; }
python scripts/diagnostics/eval_relative_human.py --model "${MODEL}" --bone_fix "${BONE_FIX}" \\
    --rh_root "${RH_ROOT}" --split "${SPLIT}" --n ${N} --out "${OUT}"
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="${ROOT_DIR}" \
  bash -c "$COMMAND"
