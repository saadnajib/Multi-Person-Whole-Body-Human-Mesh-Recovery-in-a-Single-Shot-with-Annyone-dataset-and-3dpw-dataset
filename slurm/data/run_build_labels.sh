#!/bin/bash
# 3DPW -> Anny, STEP 3 of 3: merge all fits into per-image label files. CPU only, ~1 min.
#SBATCH --job-name=3dpw-labels
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/3dpw-labels-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/3dpw-labels-%j.err

set -euo pipefail
# FITS / IMAGES / OUT can be overridden from the environment, e.g. OUT=.../labels_rebuilt sbatch ... to keep the
# labels that earlier runs trained and evaluated on.
# Before building, every task of configs/threedpw_tasks.txt must have its fit FITS/<split>/<seq>_p<person>.npz (exact
# name, non-empty). If any is missing (task failed, timed out or never ran) it lists them, builds NOTHING and exits 1:
# the builder would otherwise use whatever fits exist and give labels with missing actors. ALLOW_MISSING=1 builds
# anyway and still exits 1 (never train or evaluate on such labels).
FITS="${FITS:-/netscratch/najib/multi-hmr/threedpw_anny_full}"
IMAGES="${IMAGES:-/netscratch/najib/multi-hmr/data/3DPW/imageFiles}"
OUT="${OUT:-/netscratch/najib/multi-hmr/threedpw_anny_full/labels}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch
[ -s configs/threedpw_tasks.txt ] || { echo "ERROR: configs/threedpw_tasks.txt missing or empty"; exit 1; }
echo "unconverted tasks (no .npz yet):"
N_TASK=0; N_MISS=0
while read -r SPLIT SEQ PERSON N || [ -n "\${SPLIT:-}" ]; do
  [ -n "\${SPLIT}" ] || continue
  N_TASK=\$(( N_TASK + 1 ))
  if [ ! -s "${FITS}/\${SPLIT}/\${SEQ}_p\${PERSON}.npz" ]; then
    echo "  \${SPLIT} \${SEQ} p\${PERSON}"; N_MISS=\$(( N_MISS + 1 ))
  fi
done < configs/threedpw_tasks.txt
[ "\${N_MISS}" -eq 0 ] && echo "  (none: all \${N_TASK} tasks have a fit under ${FITS})"
if [ "\${N_MISS}" -gt 0 ] && [ "${ALLOW_MISSING}" != 1 ]; then
  echo "ERROR: \${N_MISS} of \${N_TASK} tasks have no fit under ${FITS}: labels NOT built. Rerun those tasks (same"
  echo "       OUT_ROOT / REFIT_FLAGS; tasks with a fit are skipped), then this job. ALLOW_MISSING=1 builds anyway."
  exit 1
fi
python scripts/data/build_threedpw_anny_labels.py --fits "${FITS}" --images "${IMAGES}" --out "${OUT}"
if [ "\${N_MISS}" -gt 0 ]; then
  echo "ERROR: ALLOW_MISSING=1: ${OUT} was built with \${N_MISS} of \${N_TASK} tasks unfitted (actors missing);"
  echo "       do not train or evaluate on it."
  exit 1
fi
EOF
)

srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
