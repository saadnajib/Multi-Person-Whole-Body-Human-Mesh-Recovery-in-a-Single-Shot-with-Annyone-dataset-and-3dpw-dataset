#!/bin/bash
# Body-model probe (no network, no checkpoint): CHECK 2 Anny age sweep in rest pose (vis/age_sweep.jpg) +
# CHECK 3 single-joint perturbations (neck/finger/wrist: LBS attribution, moved joints, local-bone check) +
# the 88 masked helper bones by name. 1 GPU (EGL mesh renders; falls back to CPU / vertex scatter), ~5 min.
# Reads results/diagnostics/phenotypes/all_shapes.npy (from diag_phenotypes.sh) for the mean shape if present.
# Override:  OUT=results/diagnostics/bodymodel_v2 sbatch slurm/diagnostics/diag_bodymodel.sh
#SBATCH --job-name=diag-bodymodel
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_bodymodel_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_bodymodel_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"
OUT="${OUT:-results/diagnostics/bodymodel}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
python scripts/diagnostics/diag_bodymodel_probe.py --device cuda --out "${OUT}"
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
