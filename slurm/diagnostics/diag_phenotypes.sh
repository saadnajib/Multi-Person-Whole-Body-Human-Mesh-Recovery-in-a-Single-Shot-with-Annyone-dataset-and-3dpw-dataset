#!/bin/bash
# E2: GT phenotype / age distribution of AnnyOne + Anny age sweep. CPU only, ~5 min.
# Override:  N=50000 sbatch slurm/diagnostics/diag_phenotypes.sh
#SBATCH --job-name=diag-pheno
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/diag_pheno_%j.out
#SBATCH --error=logs/diag_pheno_%j.err
set -euo pipefail
mkdir -p logs
N="${N:-20000}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
python scripts/diagnostics/diag_phenotypes.py --n ${N} --out results/diagnostics/phenotypes
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
