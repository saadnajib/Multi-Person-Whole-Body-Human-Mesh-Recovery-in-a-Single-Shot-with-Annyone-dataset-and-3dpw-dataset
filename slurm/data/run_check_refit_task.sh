#!/bin/bash
# 3DPW -> Anny v2.3 refit: post-check of the new fits of ONE sequence (or all) - CPU only, seconds per sequence.
# scripts/data/check_refit_task.py: gender value vs 3DPW gender + convention, frame grid, frames where both actors are
# valid but only one is fitted, joint error old vs new. Exit 1 (job FAILED) = a new .npz breaks the expected
# conventions; read the VIOLATION lines in the log.
#
# USAGE (repo root on the cluster):
#   SPLIT=train SEQ=courtyard_captureSelfies_00 sbatch slurm/data/run_check_refit_task.sh   # the one-task check
#   SEQ=all sbatch slurm/data/run_check_refit_task.sh                                      # every sequence in NEW_ROOT
# Overrides: NEW_ROOT (default threedpw_anny_v23), OLD_ROOT (default threedpw_anny_full; '' = no comparison),
#            SEQ_ROOT (3DPW sequenceFiles), EXPECT_GENDER (anny), EXPECT_GRID (shared).
#SBATCH --job-name=check-refit
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/check-refit-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/check-refit-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

NEW_ROOT="${NEW_ROOT:-/netscratch/najib/multi-hmr/threedpw_anny_v23}"
OLD_ROOT="${OLD_ROOT-/netscratch/najib/multi-hmr/threedpw_anny_full}"
SEQ_ROOT="${SEQ_ROOT:-/netscratch/najib/multi-hmr/data/3DPW/sequenceFiles}"
SPLIT="${SPLIT:-}"
SEQ="${SEQ:-courtyard_captureSelfies_00}"
EXPECT_GENDER="${EXPECT_GENDER:-anny}"
EXPECT_GRID="${EXPECT_GRID:-shared}"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch
python scripts/data/check_refit_task.py --seq "${SEQ}" --split "${SPLIT}" \\
  --new_root "${NEW_ROOT}" --old_root "${OLD_ROOT}" --seq_root "${SEQ_ROOT}" \\
  --expect_gender_convention "${EXPECT_GENDER}" --expect_frame_grid "${EXPECT_GRID}"
EOF
)

srun -K \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
