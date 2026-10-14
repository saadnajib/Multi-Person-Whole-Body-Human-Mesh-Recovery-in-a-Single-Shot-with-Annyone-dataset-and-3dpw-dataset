#!/bin/bash
# =============================================================================
# FINAL TEST-SET EVALUATION: every final model on Anny-One holdout + 3DPW-TEST
# =============================================================================
# One Slurm ARRAY task per model (list below), identical settings, one log
# each:  logs/eval-test3dpw-<jobid>_<task>.out
#
#   sbatch run_eval_3dpw_test.sh              # all models
#   sbatch --array=3 run_eval_3dpw_test.sh    # just one of them (0-based index)
#
# 3DPW-test (all 9,722 converted images, every third frame of the 24 test
# sequences) is used ONLY here, after every training decision was taken on
# the validation split. The 13-20 mm conversion residual is a floor on it.
# =============================================================================
#SBATCH --job-name=mhmr-eval-test3dpw
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval-test3dpw-%A_%a.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval-test3dpw-%A_%a.err
#SBATCH --array=0-4%2

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

P="/netscratch/najib/multi-hmr"
# >>> THE MODELS: "checkpoint|head_dim|backbone|label"  (array index = line number, from 0)
#     head_dim 1024 for projector-fed students (v6+), 0 for the teacher-class models.
#     A checkpoint given with a * is resolved to the newest matching file (saved copies
#     in models/multiHMR/ first, then the run's own checkpoints/ folder).
MODELS=(
  "$P/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt|0|dinov2_vitl14|teacher_v5_synthetic"
  "$P/models/multiHMR/anny_teacher_v9_mixed_00062.pt|0|dinov2_vitl14|teacher_v9_mixed_ep62"
  "$P/models/multiHMR/anny_distill_vits_v6c_00149.pt|1024|dinov2_vits14|student_v6c_synthetic"
  "$P/models/multiHMR/anny_distill_vits_v7_mixed_00130.pt|1024|dinov2_vits14|student_v7_mixed"
  "$P/models/multiHMR/anny_distill_vits_v8_final_*.pt|1024|dinov2_vits14|student_v8_final"
  "$P/models/multiHMR/anny_v10_scratch_mixed_00299.pt|0|dinov2_vitl14|model_v10_scratch_mixed"
)
# add more lines above and widen --array accordingly, e.g. teacher v10:
#   "$P/models/multiHMR/anny_v10_scratch_mixed_*.pt|0|dinov2_vitl14|model_v10_scratch_mixed"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
ENTRY="${MODELS[$TASK]:-}"
[ -n "$ENTRY" ] || { echo "ERROR: no model at array index $TASK (${#MODELS[@]} entries)"; exit 1; }
IFS='|' read -r CKPT_PAT HEAD_DIM BACKBONE LABEL <<< "$ENTRY"

# resolve wildcards: newest saved copy, else newest checkpoint in the run folder
CKPT_ARG=$(ls -1 $CKPT_PAT 2>/dev/null | sort | tail -1 || true)
if [ -z "$CKPT_ARG" ]; then
  RUN=$(basename "$CKPT_PAT" | sed 's/_[0-9*]*\.pt$//; s/_\*\.pt$//')
  CKPT_ARG=$(ls -1 "$P/logs/anny_model/$RUN/checkpoints/"*.pt 2>/dev/null | sort | tail -1 || true)
fi
[ -n "$CKPT_ARG" ] || { echo "ERROR: no checkpoint matches $CKPT_PAT (copy it to models/multiHMR/ first)"; exit 1; }
echo "task $TASK: $LABEL  ->  $CKPT_ARG  (head_dim $HEAD_DIM, $BACKBONE)"
RUN_NAME="$LABEL"
# >>> HOW MANY TEST SAMPLES (100 = exactly the untrained holdout) <<<
TEST_N=100
LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
VAL_3DPW_SPLIT="test"
VAL_3DPW_N=0        # 0 = all 9,722 test images

COMMAND=$(cat <<EOF
set -euo pipefail

source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch

CKPT="${CKPT_ARG}"
[ -f "\$CKPT" ] || { echo "ERROR: checkpoint not found: \$CKPT"; exit 1; }
echo "Evaluating \$CKPT on the last ${TEST_N} AnnyOne samples and ${VAL_3DPW_N} 3DPW-${VAL_3DPW_SPLIT} images"
[ -f "${LABELS}/index_${VAL_3DPW_SPLIT}.txt" ] || { echo "ERROR: no 3DPW labels at ${LABELS}"; exit 1; }
grep -q "mix_3dpw_labels" train.py || { echo "ERROR: train.py lacks apply_mixed_data_fix.py"; exit 1; }

# --eval_only 1 : no training, just run evaluate() and print the metrics.
# --test_anny_n : builds the AnnyOne test split (see header).
# Architecture flags must MATCH the checkpoint or the weights will not load.
python train.py \\
    --eval_only 1 \\
    --train_data AnnyOne \\
    --test_anny_n ${TEST_N} \\
    --val_anny_n 100 \\
    --person_center head \\
    --pretrained "\$CKPT" \\
    --pretrained_remap 0 \\
    --load_only_backbone 0 \\
    --backbone ${BACKBONE} \\
    --head_dim ${HEAD_DIM} \\
    --mix_3dpw_labels "${LABELS}" \\
    --val_3dpw_split ${VAL_3DPW_SPLIT} \\
    --val_3dpw_n ${VAL_3DPW_N} \\
    --img_size 672 \\
    --batch_size 1 \\
    --num_workers 4 \\
    --amp 0 \\
    --use_anny_shape 1 \\
    --mask_helper_joints_train 1 \\
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \\
    --name evaltest_${LABEL}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"