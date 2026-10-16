#!/bin/bash
# =============================================================================
# 3DPW -> Anny, STEP 2 of 3: convert ONE (sequence, person) - Slurm ARRAY task
# =============================================================================
# Submit as an array over the lines of threedpw_tasks.txt (written by step 1):
#     N=$(wc -l < threedpw_tasks.txt)
#     sbatch --array=0-$((N-1))%8 run_smpl_to_anny_task.sh
# %8 = at most 8 tasks on GPUs at once. A task that already has its .npz is
# skipped, so a failed/killed array can simply be resubmitted.
#
# Output: threedpw_anny_full/<split>/<seq>_p<person>.npz  (+ a log per task)
#
# COST: v2.1 measured 11 s/frame with the 40-frame settings (280 + 200 iters).
# Adjacent frames warm-start from each other, so far fewer iterations are
# needed here: 20/80/60 + 100 refine ~= 6 s/frame. At STRIDE=3 (10 fps, every
# third image; neighbours are near-duplicates anyway) a 1000-frame sequence
# takes ~35 min. STRIDE=1 converts every image (~3x longer).
#
# VALIDATE FIRST on one sequence you already know (outdoors_fencing_01 at the
# 40-frame settings gave 15.8 mm): run the array with --array=<its line>, and
# check the PER-JOINT table in its log is still ~16-20 mm. If it is worse,
# raise WARM_ITERS / REFINE_ITERS.
#
# ENVIRONMENT OVERRIDES (defaults = the v2.2 run that made threedpw_anny_full):
#   OUT_ROOT     fits directory (default /netscratch/najib/multi-hmr/threedpw_anny_full)
#   REFIT_FLAGS  extra smpl_to_anny.py flags, appended to the call (default empty)
# v2.3 refit (gender in Anny's convention + one frame grid for both actors), into a NEW directory
# (one line; both variables are needed):
#   OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
# Array index = line number - 1 of threedpw_tasks.txt. Guards against mixing fits made with different flags in one
# directory (an existing .npz is never redone):
#   - a non-empty REFIT_FLAGS with the default OUT_ROOT is refused;
#   - an empty REFIT_FLAGS with another OUT_ROOT is refused unless ALLOW_DEFAULT_FLAGS=1 (v2.2 behaviour into a new
#     directory, e.g. to reproduce the old fits);
#   - outside the default OUT_ROOT, an existing .npz is skipped only if the gender_convention / frame_grid it records
#     are the ones REFIT_FLAGS requests (scripts/data/check_refit_task.py --match_flags, in the container, ~1 min);
#     otherwise the task FAILS: delete that .npz or use another OUT_ROOT. In the default OUT_ROOT an existing .npz is
#     skipped without a check, as before.
#   GPUS         GPUs for the srun step (default 1). GPUS=0 runs on CPU (smpl_to_anny.py falls back to CPU when no
#                GPU is visible), for a CPU partition when the GPU queue is long; submit it with matching sbatch
#                options, e.g.  GPUS=0 OUT_ROOT=... REFIT_FLAGS="..." sbatch --partition=<cpu partition> --gpus=0
#                --cpus-per-task=8 --time=12:00:00 --array=9-10 slurm/data/run_smpl_to_anny_task.sh
#                CPU speed is not measured yet: read the 'done: ... (N s)' line of the first task before the full array.
#   CONDA_ENV    conda env activated by slurm/env.sh (default multihmr = PyTorch 2.0.1+cu117: sm_37-sm_86 only, so it
#                fails on H100/H200 with "no kernel image"). CONDA_ENV=multihmr_cu121 (a clone of multihmr with
#                PyTorch 2.1.2+cu121) also runs on H100/H200. B200 and RTXB6000 (Blackwell) work with neither.
# =============================================================================
#SBATCH --job-name=smpl2anny-all
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/smpl2anny-all-%A_%a.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/smpl2anny-all-%A_%a.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

TASKS="/netscratch/najib/multi-hmr/configs/threedpw_tasks.txt"
EXTRACT_DIR="/netscratch/najib/multi-hmr/data/3DPW"
SMPL_MODEL="/netscratch/najib/multi-hmr/models/smpl/SMPL_NEUTRAL.pkl"
DEFAULT_OUT_ROOT="/netscratch/najib/multi-hmr/threedpw_anny_full"
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
OUT_ROOT="${OUT_ROOT%/}"
REFIT_FLAGS="${REFIT_FLAGS:-}"   # e.g. "--gender_convention anny --frame_grid shared" (v2.3)
GPUS="${GPUS:-1}"                 # 0 = CPU only (see the header)
CONDA_ENV="${CONDA_ENV:-multihmr}"   # conda env for the converter (slurm/env.sh); multihmr_cu121 = PyTorch with sm_90 for H100/H200
if [ "${GPUS}" = "0" ]; then SRUN_GPUS=(); else SRUN_GPUS=(--gpus="${GPUS}"); fi
if [ -n "${REFIT_FLAGS}" ] && [ "${OUT_ROOT}" = "${DEFAULT_OUT_ROOT}" ]; then
  echo "ERROR: REFIT_FLAGS='${REFIT_FLAGS}' with the default OUT_ROOT ${DEFAULT_OUT_ROOT}:"
  echo "       the old fits would be mixed with new ones. Set OUT_ROOT to a new directory,"
  echo "       e.g. OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23"
  exit 1
fi
if [ -z "${REFIT_FLAGS}" ] && [ "${OUT_ROOT}" != "${DEFAULT_OUT_ROOT}" ] && [ "${ALLOW_DEFAULT_FLAGS:-0}" != "1" ]; then
  echo "ERROR: OUT_ROOT=${OUT_ROOT} with an empty REFIT_FLAGS would write v2.2-behaviour fits"
  echo "       (--gender_convention makehuman, --frame_grid per_person) into a new directory, and later runs"
  echo "       with flags would skip them. For the v2.3 refit set both, e.g."
  echo "       REFIT_FLAGS=\"--gender_convention anny --frame_grid shared\"; to really use the defaults there,"
  echo "       set ALLOW_DEFAULT_FLAGS=1."
  exit 1
fi

# ---- conversion settings ----------------------------------------------------
STRIDE=3                      # 1 = every image; 3 = every third (10 fps)
ITERS="150 400 250"           # first frame of a sequence (cold start)
WARM_ITERS="20 80 60"         # every following frame (warm-started)
REFINE_ITERS=100              # pass 2 with the shared body
FREE_PHENOTYPES="height,proportions"
SET_PHENOTYPES="age=0.55"     # joints cannot measure adult age; AnnyOne mean
# -----------------------------------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((TASK_ID + 1))p" "${TASKS}")
[ -n "${LINE}" ] || { echo "ERROR: no line $((TASK_ID + 1)) in ${TASKS}"; exit 1; }
read -r SPLIT SEQ PERSON NFRAMES <<< "${LINE}"
OUT="${OUT_ROOT}/${SPLIT}/${SEQ}_p${PERSON}.npz"
echo "task ${TASK_ID}: ${SPLIT}/${SEQ} person ${PERSON} (${NFRAMES} frames) -> ${OUT}"
echo "extra flags (REFIT_FLAGS): '${REFIT_FLAGS}'"

if [ -f "${OUT}" ]; then
  if [ "${OUT_ROOT}" = "${DEFAULT_OUT_ROOT}" ]; then
    echo "already converted - skipping"; exit 0
  fi
  # Another OUT_ROOT: skip only a fit made with the same flags (needs numpy, so in the container).
  echo "already converted - checking that it was made with the requested flags"
  COMMAND=$(cat <<EOF
set -uo pipefail
export CONDA_ENV="${CONDA_ENV}"
source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch
python scripts/data/check_refit_task.py --match_flags "${OUT}" --refit_flags="${REFIT_FLAGS}"
RC=\$?
if [ \$RC -eq 0 ]; then
  echo "already converted with the requested flags - skipping"
else
  echo "ERROR: ${OUT} exists but was not made with REFIT_FLAGS='${REFIT_FLAGS}' (or cannot be read)."
  echo "       Delete it to refit this task, or use another OUT_ROOT. Not refitted."
  exit 1
fi
EOF
)
else
COMMAND=$(cat <<EOF
set -uo pipefail
export CONDA_ENV="${CONDA_ENV}"
source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch
mkdir -p "${OUT_ROOT}/${SPLIT}"
[ -f "${SMPL_MODEL}" ] || { echo "ERROR: SMPL model missing: ${SMPL_MODEL}"; exit 1; }
PKL="${EXTRACT_DIR}/sequenceFiles/${SPLIT}/${SEQ}.pkl"
[ -f "\$PKL" ] || { echo "ERROR: \$PKL not found (run run_3dpw_list.sh first)"; exit 1; }

T0=\$(date +%s)
python scripts/data/smpl_to_anny.py \\
  --pkl "\$PKL" --person ${PERSON} \\
  --smpl_model_path "${SMPL_MODEL}" \\
  --out "${OUT}.tmp.npz" \\
  --max_frames 1000000 --frame_stride ${STRIDE} \\
  --iters ${ITERS} --warm_iters ${WARM_ITERS} --refine_iters ${REFINE_ITERS} \\
  --free_phenotypes "${FREE_PHENOTYPES}" --set_phenotypes "${SET_PHENOTYPES}" \\
  --shared_shape 1 --quiet 1 ${REFIT_FLAGS}
RC=\$?
if [ \$RC -eq 0 ] && [ -f "${OUT}.tmp.npz" ]; then
  mv "${OUT}.tmp.npz" "${OUT}"
  T1=\$(date +%s)
  echo "done: ${OUT}  (\$((T1-T0)) s)"
else
  rm -f "${OUT}.tmp.npz"
  echo "FAILED (rc=\$RC): ${SPLIT}/${SEQ} person ${PERSON}"
  exit 1
fi
EOF
)
fi

srun -K \
  ${SRUN_GPUS[@]+"${SRUN_GPUS[@]}"} \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib,/ds-av:/ds-av \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
