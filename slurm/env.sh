#!/bin/bash
# Shared job environment. Sourced at the start of every active Slurm job,
# inside the container:  source /netscratch/najib/multi-hmr/slurm/env.sh
# Edit the paths here once instead of in every script.

export PROJECT_ROOT="${PROJECT_ROOT:-/netscratch/najib/multi-hmr}"
SCRATCH="/netscratch/najib"

# conda
source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-multihmr}"   # CONDA_ENV overrides, e.g. multihmr_cu121 for H100/H200 (sm_90)

# caches on scratch (home directory is over quota)
export HF_HOME="$SCRATCH/hf_cache"
export TORCH_HOME="$SCRATCH/torch_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"
export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# anny writes to a hard-coded ~/.cache/anny; route it to scratch
mkdir -p "$SCRATCH/anny_cache" "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn "$SCRATCH/anny_cache" "$HOME/.cache/anny"

# one thread per worker: avoids dataloader deadlocks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$PROJECT_ROOT"

# DINOv2 from torch.hub uses `X | None` hints that Python 3.9 rejects: download once, strip them
DINOV2_DIR="$TORCH_HOME/hub/facebookresearch_dinov2_main/dinov2"
if [ ! -d "$DINOV2_DIR" ]; then
  python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" >/dev/null 2>&1 || true
fi
grep -rlZ --include="*.py" ' | None' "$DINOV2_DIR" 2>/dev/null \
  | xargs -0 -r sed -i 's/ | None//g' || true   # only touches files not yet patched
