#!/bin/bash
# CHECK 6: can the model overfit the ~40 manifest TRAIN images in a short, capped run?
#   1. evaluate the starting checkpoint on the manifest train subset and on the manifest holdout subset
#   2. train for MAX_ITER iterations on the train subset only (val disabled; holdout never seen)
#   3. evaluate the last checkpoint on both subsets again
# Success on the train subset is NOT validation performance: it only shows the optimisation path works.
# Needs results/diagnostics/manifest.json + data/diag_manifest/{train,holdout} (build_manifest.py).
# Writes nothing under logs/anny_model; its checkpoints go to results/diagnostics/overfit_<TAG>/.
# CKPT must be a file path:  CKPT=/path/to/00099.pt MAX_ITER=400 sbatch slurm/diagnostics/diag_overfit.sh
#SBATCH --job-name=diag-overfit
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_overfit_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_overfit_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all

CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
if [ "$(basename "$(dirname "$CKPT")")" = checkpoints ]; then
  TAG="${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT" .pt)}"
else
  TAG="${TAG:-$(basename "$CKPT" .pt)}"
fi
BACKBONE="${BACKBONE:-dinov2_vitl14}"   # dinov2_vits14 + HEAD_DIM=1024 for the distilled students
HEAD_DIM="${HEAD_DIM:-0}"
MAX_ITER="${MAX_ITER:-400}"             # 40 epochs x 10 iters at batch 4 ~ 10 passes over 40 images
LR="${LR:-1e-4}"
OUT="results/diagnostics/overfit_${TAG}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f results/diagnostics/manifest.json ] || { echo "ERROR: run build_manifest.py first (slurm/diagnostics/diag_manifest.sh)"; exit 1; }
[ -d data/diag_manifest/train ] && [ -d data/diag_manifest/holdout ] || { echo "ERROR: data/diag_manifest/{train,holdout} missing"; exit 1; }
mkdir -p "${OUT}"
N_TRAIN=\$(ls data/diag_manifest/train/*.pkl | wc -l)
N_HOLD=\$(ls data/diag_manifest/holdout/*.pkl | wc -l)
echo "manifest subsets: train \${N_TRAIN} images, holdout \${N_HOLD} images"

# shared eval flags: the eval_only path; helper bones UNMASKED at inference exactly like train.py evaluate()
evaluate () {   # \$1 = checkpoint, \$2 = subset folder, \$3 = label
  python train.py --eval_only 1 --train_data AnnyOne --anny_data_folder "\$2" --test_anny_n 100000 --val_anny_n 0 \\
      --person_center head --pretrained "\$1" --pretrained_remap 0 --load_only_backbone 0 \\
      --backbone ${BACKBONE} --head_dim ${HEAD_DIM} --img_size 672 --batch_size 1 --num_workers 4 --amp 0 \\
      --use_anny_shape 1 --mask_helper_joints_train 1 --tf32 1 \\
      --save_dir "${OUT}/evals" --name "\$3" > "${OUT}/eval_\$3.log" 2>&1 \
      || { echo "ERROR: evaluation \$3 failed; tail of ${OUT}/eval_\$3.log:"; tail -n 30 "${OUT}/eval_\$3.log"; exit 1; }
  grep -E "EVAL METRICS|^    - " "${OUT}/eval_\$3.log" || echo "  (no metrics lines found in eval_\$3.log)"
}

echo "=== 1. BEFORE: ${CKPT}"
evaluate "${CKPT}" data/diag_manifest/train   before_train
evaluate "${CKPT}" data/diag_manifest/holdout before_holdout

echo "=== 2. CAPPED TRAINING on the train subset only (${MAX_ITER} iters, lr ${LR})"
# recipe = anny_s2_shape_v5 / partial-freeze runs (same flags the thesis models were trained with),
# except: tiny data, short schedule, no val set, checkpoints kept in \${OUT}
python train.py --train_data AnnyOne --anny_data_folder data/diag_manifest/train --train_n 0 --val_anny_n 0 \\
    --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone ${BACKBONE} --head_dim ${HEAD_DIM} --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --start_2d_epoch 0 --alpha_j2d 0.1 --alpha_v2d 0.1 \\
    --batch_size 4 --num_workers 4 --n_iters_per_epoch 10 --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 1000 --amp 0 --tf32 1 --eval_freq 100000 --nb_max_ckpt 1 \\
    --log_freq 10 --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
grep -E "loss|ERROR|Error|nan" "${OUT}/train.log" | tail -n 40 || true

LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

echo "=== 3. AFTER: \${LAST}"
evaluate "\${LAST}" data/diag_manifest/train   after_train
evaluate "\${LAST}" data/diag_manifest/holdout after_holdout

python - <<'PY'
import json, os, re
out = os.environ.get('OUT_DIR')
res = {}
for lab in ('before_train', 'before_holdout', 'after_train', 'after_holdout'):
    txt = open(f"{out}/eval_{lab}.log", errors='ignore').read()
    vals = {k: float(v) for k, v in re.findall(r"^\s+- (pve|pa_pve|mpjpe|recall|precision): ([0-9.]+)", txt, flags=re.M)}
    res[lab] = vals
s = dict(results=res)
try:
    s['train_pve_drop_pct'] = 100 * (1 - res['after_train']['pve'] / res['before_train']['pve'])
    s['holdout_pve_drop_pct'] = 100 * (1 - res['after_holdout']['pve'] / res['before_holdout']['pve'])
except Exception as e:
    s['error'] = str(e)
json.dump(s, open(f"{out}/summary.json", 'w'), indent=2)
print(json.dumps(s, indent=2))
print("READ: train PVE must fall clearly (>30%) for the optimisation path to be considered intact; the holdout")
print("      number is the GENERALISATION read and is expected to move little. A train PVE that does not fall")
print("      on 40 images points at decoding/optimisation, not at data.")
PY
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}'; $COMMAND"
