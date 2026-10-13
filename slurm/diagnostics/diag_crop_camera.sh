#!/bin/bash
# E4: demo crop camera (full-image fov) vs corrected crop camera on the real photos in example_data.
# Override:  CKPT=... IMG_FOLDER=/path sbatch slurm/diagnostics/diag_crop_camera.sh
#SBATCH --job-name=diag-cropK
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_cropK_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_cropK_%j.err
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
IMG_FOLDER="${IMG_FOLDER:-/netscratch/najib/multi-hmr/example_data}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
python -c "import torchvision" 2>/dev/null || pip install --quiet torchvision
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
python scripts/diagnostics/diag_crop_camera.py --ckpt "${CKPT}" --img_folder "${IMG_FOLDER}" \\
    --fov 55 --person_frac 0.22 --crop_margin 0.35 --detector_thresh 0.8 --det_thresh 0.15 --nms_kernel_size 5 \\
    --out results/diagnostics/crop_camera_${TAG}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
