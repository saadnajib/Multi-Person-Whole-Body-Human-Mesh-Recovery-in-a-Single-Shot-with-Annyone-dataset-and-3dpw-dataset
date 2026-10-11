#!/bin/bash
#SBATCH --job-name=test_anny_loader
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_loader_%j.out
#SBATCH --error=logs/test_loader_%j.err

# We wrap the entire process in a COMMAND variable to pass into the container.
COMMAND=$(cat <<'EOF'
source /netscratch/najib/multi-hmr/slurm/env.sh   # conda, caches, cd, DINOv2 patch

# 4. Run the Dataloader Test!
python scripts/analysis/test_loader.py

EOF
)

# 5. Execute the command inside the DFKI Enroot/Pyxis Container
srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"