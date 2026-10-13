#!/bin/bash
# CHECK 4: do the prediction heads receive gradients, and do decoded meshes respond to their outputs?
# CKPT must be a file path (the sbatch wrapper checks it exists):  CKPT=/path/to/00149.pt sbatch slurm/diagnostics/diag_gradients.sh
# Optional: MASK=0 replays training with --mask_helper_joints_train 0 (output dir gets a _mask0 suffix);
#           INDICES=579860,579885 uses explicit dataset indices instead of 4 holdout images.
#SBATCH --job-name=diag-gradients
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_gradients_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_gradients_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
if [ "$(basename "$(dirname "$CKPT")")" = checkpoints ]; then   # logs/anny_model/<run>/checkpoints/<ep>.pt
  TAG="${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT" .pt)}"
else                                                             # models/multiHMR/<name>.pt
  TAG="${TAG:-$(basename "$CKPT" .pt)}"
fi
MASK="${MASK:-1}"
INDICES="${INDICES:-}"
SUFFIX=""; [ "${MASK}" = 0 ] && SUFFIX="_mask0"
EXTRA=""; [ -n "${INDICES}" ] && EXTRA="--indices ${INDICES}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
python scripts/diagnostics/diag_gradients.py --ckpt "${CKPT}" \\
    --mask_helper_joints_train ${MASK} ${EXTRA} \\
    --out results/diagnostics/gradients_${TAG}${SUFFIX}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
