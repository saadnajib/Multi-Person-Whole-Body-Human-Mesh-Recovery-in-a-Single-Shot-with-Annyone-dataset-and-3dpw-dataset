#!/bin/bash
# E3: same trace on the 50 youngest people among the first 20k TRAINING samples (in-sample).
# Override:  CKPT=... SCAN=50000 N=50 sbatch slurm/diagnostics/diag_young.sh
#SBATCH --job-name=diag-young
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_young_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_young_%j.err
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
SCAN="${SCAN:-20000}"
N="${N:-50}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
python scripts/diagnostics/diag_holdout_trace.py --ckpt "${CKPT}" --select youngest --scan ${SCAN} --n ${N} \\
    --out results/diagnostics/young_${TAG}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
