#!/bin/bash
# exp5_neck_fix: fine-tune with the GEODESIC per-bone rotation loss (--rotmat_loss geodesic, loss.py) instead of the
# L1 over the 9 rotation-matrix entries, then dump the neck/head bones on the holdout (--dump_bones 1).
#
# WHY (docs/experiments_plan.md, "Neck root cause"): v5, v9 (149 more epochs) and every v5 pilot carry the same local
# error, neck01-03 62.7 deg and head+jaw 89.5 deg, per person 62.7 +/- 1.6 and 89.5 +/- 1.5, while the GLOBAL head
# error is 12.0 deg like v10's 12.9. The code audit refuted the mask, weight, eval and 3DPW causes (useful_rotmat is 1 on
# bones 100-104, non-zero decpose gradients, V0 == V1, same GT tensor in training and eval, v5 never saw 3DPW). Two
# bookkeeping causes are still UNTESTED: H3, an ancestor of the inherited heads trained in another pose_param, and H4,
# raw dataset neck/head blocks that are not proper rotations (so the SVD-projected target disagrees with the GT mesh).
# slurm/diagnostics/diag_neck_bones.sh checks both (H3 needs the ancestor checkpoints traced); read it before this.
# The leading optimisation hypothesis [Likely, not yet measured per bone] is a compensated 180-deg twist flip: one neck
# bone turned half a turn about its own axis and the head turned back, so joints and head orientation stay right while
# the neck skin collapses.
# The L1 over matrix entries has a barrier along that twist (4.83 at 135 deg vs 4.00 at 180 deg), so it can hold a flip
# for any number of steps; the rotation angle has no such barrier (synthetic Adam test: L1 stays at ~165 deg from a
# 175-deg start, geodesic reaches 0.1 deg). Whether v5 really holds a flip is what slurm/diagnostics/diag_neck_bones.sh
# decides; this script is the training-side candidate and the guard against re-trapping after any repair.
#
# WHAT RUNS (one GPU; ~45-60 min with the defaults):
#   1. the diag manifest's image indices -> ${OUT}/exclude_manifest_indices.txt (as in exp3/exp4)
#   2. train.py from CKPT with the production v5 recipe (the exp3 command) plus
#        --rotmat_loss ${ROTMAT_LOSS} --alpha_rotmat ${ALPHA_ROTMAT}      ROTMAT_LOSS=l1 = control arm (alpha 0.1)
#      alpha 0.33 for geodesic keeps the small-angle weight of the L1 at 0.1: at a 2-deg error the 9-entry L1 is 3.27x
#      (GT near identity) to 3.50x (arbitrary GT) the angle in radians (scratchpad test, 4000 samples).
#   3. diag_holdout_trace.py --dump_bones 1 on the last checkpoint (100-image holdout, V0-V3), and on CKPT itself when
#      no baseline dump exists at BASELINE_DUMP (diag_neck_bones.sh writes it)
#   4. scripts/experiments/exp5_neck_summary.py: per-bone verdict and median/mean local error neck01/02/03/head/jaw,
#      the group means, neck-skin girth, global head, PVE, PVE_hn and recall, next to the baseline dump and the
#      control arm (CONTROL_RUN). SUMMARY_ONLY=1 runs only this step, on finished run dirs, without srun or a GPU.
#   ALPHA_GROT (exp3's global term, default 0) is not rescaled by --rotmat_loss geodesic: the angle is ~3.5x smaller than
#   the 9-entry L1, so pass ~3.5x the L1-era value (17.5 for exp3's 5) to give it the same weight.
#
# USAGE (repo root on the cluster). Two arms, the control first; never a third arm just for the comparison:
#   C=$(ROTMAT_LOSS=l1 sbatch --parsable slurm/experiments/exp5_neck_fix.sh)      # control arm: same schedule, L1, alpha 0.1
#   CONTROL_RUN=results/experiments/exp5_neck_fix/pilot_l1_a0.1_anny_s2_shape_v5_00099 \
#       sbatch --dependency=afterok:$C slurm/experiments/exp5_neck_fix.sh         # geodesic, alpha 0.33; summary incl. control
#   or both at once (no CONTROL_RUN), then, after both finished, the comparison alone (seconds, login node is fine):
#   SUMMARY_ONLY=1 CONTROL_RUN=results/experiments/exp5_neck_fix/pilot_l1_a0.1_anny_s2_shape_v5_00099 \
#       bash slurm/experiments/exp5_neck_fix.sh                                   # RUN defaults to the geodesic arm
#   A CONTROL_RUN without a finished holdout_trace/ aborts the job at start (before any training), and the summary
#   exits 2, instead of silently comparing against the baseline only.
#   MAX_ITER=10000 RUN=long_geo sbatch --time=05:00:00 slurm/experiments/exp5_neck_fix.sh   # 10 epochs (~1.7 h training)
#   CKPT=<repaired checkpoint> sbatch slurm/experiments/exp5_neck_fix.sh       # after a zero-training repair: keep it repaired
#
# SUCCESS CRITERION (V1 settings, holdout, matched people). Three reads printed side by side; a guard miss does not hide
# the neck read:
#   neck:      FIXED = neck and head group means < 15 deg (v10: 5.0 / 4.1) and neck-skin girth pred/GT >= 0.9;
#              MOVING = every bone the baseline dump labels FLIP_*, PARTIAL_FLIP or CONST_OFFSET drops its MEAN local
#              error by >= 3 deg, and by >= 3 deg more than in the control arm (the v5 arms' group means span 62.5-62.8 /
#              89.1-90.1 deg; the mean, so a PARTIAL_FLIP bone counts); else NO MOVEMENT / NO TARGET
#   guards:    PVE <= ref + 8 mm, recall >= ref - 2 points, global head <= ref + 3 deg; ref = the control arm, or the v5
#              numbers 70.5 / 90.2 / 12.0 without one. Tolerances = the measured spread of the six 2-epoch v5-recipe arms
#              (exp2-4 pilots and controls): PVE 68.7-76.3, recall 88.6-90.2, global head 11.3-13.7; the two identical
#              control arms (exp3, exp4) differ by 7.6 mm / 0.8 points / 1.9 deg. The earlier +1 mm / -1 point guard
#              failed all six, controls included.
#   control:   whether the control arm was compared (without it MOVING cannot be told from fine-tuning drift)
#   A null pilot does NOT refute the flip hypothesis: v5's decpose 6D columns have norms 120-341 (neck02) and 61-182 (head)
#   (diag_gradients), Adam moves each weight by about lr per step, and undoing a flip means reversing such a column, which
#   2,000 steps at lr 1e-5 may not do [Guessing]. The zero-training test is the decpose surgery on the bone the dump names.
#SBATCH --job-name=exp5-neck-fix
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=02:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp5_neck_fix_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp5_neck_fix_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
# TAG as in slurm/diagnostics/diag_neck_bones.sh, so BASELINE_DUMP finds its dump: logs/anny_model/<run>/checkpoints/<ep>.pt
# -> <run>_<ep>; results/experiments/<exp>/<name>/train/checkpoints/<ep>.pt -> <name>_<ep>; models/multiHMR/<x>.pt -> <x>
if [ "$(basename "$(dirname "$CKPT")")" = checkpoints ]; then
  CK_RUN_DIR="$(dirname "$(dirname "$CKPT")")"; CK_RUN="$(basename "$CK_RUN_DIR")"
  [ "$CK_RUN" = train ] && CK_RUN="$(basename "$(dirname "$CK_RUN_DIR")")"
  TAG="${TAG:-${CK_RUN}_$(basename "$CKPT" .pt)}"
else
  TAG="${TAG:-$(basename "$CKPT" .pt)}"
fi
ROTMAT_LOSS="${ROTMAT_LOSS:-geodesic}"               # l1 = control arm (production loss)
if [ "${ROTMAT_LOSS}" = geodesic ]; then ALPHA_ROTMAT="${ALPHA_ROTMAT:-0.33}"; else ALPHA_ROTMAT="${ALPHA_ROTMAT:-0.1}"; fi
ALPHA_GROT="${ALPHA_GROT:-0}"                        # exp3's global term; geodesic there too: ~3.5x the L1-era alpha
GROT_BONES="${GROT_BONES:-head,neck03,wrist.L,wrist.R}"
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"                         # 2 epochs
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-1}"
BASELINE_DUMP="${BASELINE_DUMP:-results/diagnostics/neck_bones/${TAG}/bones_summary.json}"
CONTROL_RUN="${CONTROL_RUN:-}"                       # the finished control arm's run dir (ROTMAT_LOSS=l1, same schedule)
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"                     # 1: only step 4 on finished run dirs (no srun, no GPU)
RUN="${RUN:-pilot_${ROTMAT_LOSS}_a${ALPHA_ROTMAT}_${TAG}}"
OUT="results/experiments/exp5_neck_fix/${RUN}"
EXCL="${OUT}/exclude_manifest_indices.txt"
N_EPOCHS=$(( MAX_ITER / N_ITERS ))
EXPECTED_TRAIN_MIN=$(( MAX_ITER * 40 / 67 / 60 ))    # MAX_ITER * batch 4 / 6.7 images/s, in minutes
CONTROL_ARG=""; [ -n "${CONTROL_RUN}" ] && CONTROL_ARG="--control_run ${CONTROL_RUN}"

if [ "${SUMMARY_ONLY}" = "1" ]; then
  BASE="${BASELINE_DUMP}"; [ -f "${BASE}" ] || BASE="${OUT}/holdout_trace_baseline/bones_summary.json"
  [ -d "${OUT}" ] || { echo "ERROR: SUMMARY_ONLY=1 but run dir ${OUT} does not exist (set RUN=<run name>)"; exit 1; }
  exec python3 scripts/experiments/exp5_neck_summary.py --run_dir "${OUT}" --base_dump "${BASE}" ${CONTROL_ARG}
fi
if [ -n "${CONTROL_RUN}" ] && [ ! -f "${CONTROL_RUN}/holdout_trace/bones_summary.json" ]; then
  echo "ERROR: CONTROL_RUN=${CONTROL_RUN} has no holdout_trace/bones_summary.json yet: the control arm has not finished"
  echo "       (or the path is wrong). Nothing was trained. Submit this arm with sbatch --dependency=afterok:<control job id>,"
  echo "       or drop CONTROL_RUN and compare later with SUMMARY_ONLY=1 CONTROL_RUN=... bash slurm/experiments/exp5_neck_fix.sh"
  exit 2
fi

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp5_neck_fix run ${RUN}"
echo "    start: ${CKPT}"
echo "    rotmat_loss=${ROTMAT_LOSS} alpha_rotmat=${ALPHA_ROTMAT} alpha_global_rot=${ALPHA_GROT} lr=${LR} exclude_manifest=${EXCLUDE_MANIFEST}"
echo "    ${N_ITERS} iters/epoch x ${N_EPOCHS} epochs, batch 4 = $(( MAX_ITER * 4 )) images, ~${EXPECTED_TRAIN_MIN} min at 6.7 images/s"

echo "=== 1. MANIFEST IMAGES -> ${EXCL}"
EXCL_ARG=""
if [ "${EXCLUDE_MANIFEST}" = "1" ] && [ -f "${MANIFEST}" ]; then
  python - <<'PY'
import json, os
man = json.load(open(os.environ['MANIFEST_PATH']))
idx = sorted({int(e['idx']) for e in man['entries']})
with open(os.environ['EXCL_PATH'], 'w') as f:
    f.write('# AnnyOne indices of the diag-manifest images (%d), from %s; never sampled by this fine-tune\n'
            % (len(idx), os.environ['MANIFEST_PATH']))
    f.write('\n'.join(str(i) for i in idx) + '\n')
print('wrote', os.environ['EXCL_PATH'], ':', len(idx), 'images excluded')
PY
  EXCL_ARG="--exclude_indices_file ${EXCL}"
else
  echo "no exclusion (EXCLUDE_MANIFEST=${EXCLUDE_MANIFEST}); the production shuffled loader is used"
fi

echo "=== 2. TRAINING -> ${OUT}/train.log"
T0=\$(date +%s)
python train.py \\
    --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
    --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone dinov2_vitl14 --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
    --rotmat_loss ${ROTMAT_LOSS} --alpha_rotmat ${ALPHA_ROTMAT} \\
    --alpha_global_rot ${ALPHA_GROT} --global_rot_bones "${GROT_BONES}" --start_2d_epoch 0 \\
    \${EXCL_ARG} \\
    --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
    --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
    --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s"
grep -E "\[loss\] (rotmat_loss|joint weights|global_rot)|\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall): |NaN|WARNING|Error" "${OUT}/train.log" || true
echo "--- rotmat term share on the first and last logged lines"
python - <<'PY'
import os, re
a = float(os.environ['ALPHA_ROTMAT_VAL'])
lines = [l[l.index('EPOCH='):] for l in open(os.environ['OUT_DIR'] + '/train.log') if 'EPOCH=' in l]
def val(l, k):
    m = re.search(k + r': ([0-9.eE+-]+)', l)
    return float(m.group(1)) if m else None
for tag, l in (('first', lines[0] if lines else None), ('last', lines[-1] if lines else None)):
    if l is None:
        print('  no EPOCH= lines in train.log'); break
    r, t = val(l, 'rotmat'), val(l, 'total')
    if r is None or t is None:
        print('  %s line: rotmat/total not found' % tag); continue
    print('  %s line: raw rotmat %.3f x alpha %g = %.2f = %.1f%% of total %.1f' % (tag, r, a, a * r, 100 * a * r / max(t, 1e-9), t))
PY

LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

trace () {   # \$1 = checkpoint, \$2 = output dir
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\$1" --out "\$2" --select holdout --n 100 --n_vis 6 \\
      --dump_bones 1 > "\$2.log" 2>&1 || { echo "WARNING: trace \$2 failed; tail:"; tail -n 30 "\$2.log"; return 0; }
  grep -A 3 "=== ROTATION ERROR BY BONE GROUP" "\$2.log" | grep -E "^variant|^V1_eval_masked" || true
  sed -n '/=== NECK\/HEAD BONE DUMP/,/^wrote .*bones.npz/p' "\$2.log"
}
echo "=== 3a. HOLDOUT TRACE + BONE DUMP on \${LAST}"
trace "\${LAST}" "${OUT}/holdout_trace"
BASE="${BASELINE_DUMP}"
if [ ! -f "\${BASE}" ]; then
  echo "=== 3b. no baseline dump at \${BASE}: tracing ${CKPT} with the same code (+~5 min)"
  trace "${CKPT}" "${OUT}/holdout_trace_baseline"
  BASE="${OUT}/holdout_trace_baseline/bones_summary.json"
fi

echo "=== 4. SUMMARY"
python scripts/experiments/exp5_neck_summary.py --run_dir "${OUT}" --base_dump "\${BASE}" ${CONTROL_ARG}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' MANIFEST_PATH='${MANIFEST}' EXCL_PATH='${EXCL}' ALPHA_ROTMAT_VAL='${ALPHA_ROTMAT}'; $COMMAND"
