#!/bin/bash
# Manifest-based suite, checks 1 and 5 in one job:
#   build_manifest.py        -> results/diagnostics/manifest.json + data/diag_manifest/{train,holdout} symlinks
#   diag_gt_decode.py        -> CHECK 1: GT decoded and overlaid with the dataset camera, no network
#   diag_holdout_trace.py    -> CHECK 5 (+E1 metrics) on the manifest train split and on the holdout split
# CKPT must be a file path:  CKPT=/path/to/00099.pt sbatch slurm/diagnostics/diag_manifest.sh
#SBATCH --job-name=diag-manifest
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_manifest_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_manifest_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
if [ "$(basename "$(dirname "$CKPT")")" = checkpoints ]; then
  TAG="${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT" .pt)}"
else
  TAG="${TAG:-$(basename "$CKPT" .pt)}"
fi
REBUILD="${REBUILD:-0}"   # 1 = rebuild the manifest even if it exists (changes the fixed sample set!)

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
if [ ! -f results/diagnostics/manifest.json ] || [ "${REBUILD}" = "1" ]; then
  echo "=== building the manifest"
  python scripts/diagnostics/build_manifest.py --out results/diagnostics/manifest.json
else
  echo "=== reusing the existing manifest (REBUILD=1 to rebuild)"
fi
echo "=== CHECK 1: GT decode, no network"
python scripts/diagnostics/diag_gt_decode.py --manifest results/diagnostics/manifest.json --out results/diagnostics/gt_decode
echo "=== CHECK 5 + E1 metrics on the manifest, TRAIN split (in-sample)"
python scripts/diagnostics/diag_holdout_trace.py --ckpt "${CKPT}" --manifest results/diagnostics/manifest.json \\
    --manifest_split train --n_vis 40 --out results/diagnostics/manifest_trace_train_${TAG}
echo "=== CHECK 5 + E1 metrics on the manifest, HOLDOUT split"
python scripts/diagnostics/diag_holdout_trace.py --ckpt "${CKPT}" --manifest results/diagnostics/manifest.json \\
    --manifest_split holdout --n_vis 20 --out results/diagnostics/manifest_trace_holdout_${TAG}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
