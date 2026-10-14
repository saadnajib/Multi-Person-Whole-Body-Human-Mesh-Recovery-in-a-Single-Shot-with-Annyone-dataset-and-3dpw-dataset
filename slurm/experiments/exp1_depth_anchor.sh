#!/bin/bash
# exp1_depth_anchor: fine-tune the production checkpoint with the GT anchor moved from the ROOT to the joint
# the model anchors (--gt_center_joint person_center), then re-run the fixed holdout trace.
#
# WHY (docs/diagnostics_findings.md, "The depth defect"): the model attaches the mesh to the detected pixel
# by its HEAD and reports the head as transl (model.py person_center_idx), while train.py prepare_gt built
# transl / dist / heatmap cell / offset at joint 0 (root). The 2D terms can only reconcile the two by
# shrinking the body, i.e. predicting a larger depth: pred depth = 0.33*GT + 2.36 m on the holdout, body
# drawn ~35% too small. The flag makes target and prediction describe the same joint; nothing else changes.
#
# WHAT RUNS (one GPU, ~1 h with the defaults):
#   1. train.py from CKPT with the production v5 recipe (same freeze pattern, lr 1e-5, alphas, batch 4) plus
#        --gt_center_joint ${GT_CENTER}   person_center = the fix; GT_CENTER=root = control arm, same schedule
#        --start_2d_epoch 0               the epoch counter restarts at 0 (no resume): keeps j2d/v2d ON as at epoch 99
#        --n_iters_per_epoch 1000 --max_iter ${MAX_ITER}   2 epochs = 8,000 images ~ 20 min at the measured 6.7 images/s
#        --eval_freq 1 --nb_max_ckpt 2 --debug_anchor ${DEBUG_ANCHOR} --alpha_dist/--alpha_transl ${ALPHA_*}
#      Checkpoints go to ${OUT}/train/checkpoints/0000N.pt, never next to the production run: train.py keeps
#      the --nb_max_ckpt highest epoch numbers in a directory and would delete a pilot's 00000.pt there.
#      LR: 1e-5 is the v5 recipe's starting value, but 8x the LR the checkpoint ended at (StepLR every 30 epochs,
#      x0.5: 1.25e-6 at epochs 90-99), and the optimizer starts with fresh Adam moments. The restart alone can move
#      every number, so the control arm (GT_CENTER=root, same LR and schedule) must be run and read side by side
#      (CONTROL_RUN below).
#   2. scripts/diagnostics/diag_holdout_trace.py on the LAST checkpoint: the fixed 100-image holdout
#      (--select holdout, V1 = helper bones masked) and the manifest holdout split. Nothing else is evaluated.
#   3. a summary printing the new numbers next to the baseline anny_s2_shape_v5/00099 (same holdout, V1).
#
# USAGE (repo root on the cluster):
#   sbatch slurm/experiments/exp1_depth_anchor.sh                                # the fix, 2-epoch pilot
#   GT_CENTER=root sbatch slurm/experiments/exp1_depth_anchor.sh                 # control arm: same schedule, old targets
#   ALPHA_DIST=5 ALPHA_TRANSL=5 sbatch slurm/experiments/exp1_depth_anchor.sh    # variant: direct depth terms x5
#   BASELINE_TRACE=1 sbatch slurm/experiments/exp1_depth_anchor.sh               # also trace CKPT itself (+5-10 min)
#   CONTROL_RUN=results/experiments/exp1_depth_anchor/pilot_root_dist1_transl1_anny_s2_shape_v5_00099 \
#       sbatch slurm/experiments/exp1_depth_anchor.sh                            # also print the control arm side by side
#
# LONGER RUN (only after a pilot that meets the criterion below):
#   10 epochs = MAX_ITER=10000 = 40,000 images. At the measured 6.7 images/s: 40000 / 6.7 = 5970 s ~ 1.7 h of
#   training, plus 10 holdout evaluations and startup. Rescale from the pilot's own line
#       [throughput] epoch 1: ... images/s
#   with   hours = MAX_ITER * 4 / images_per_s / 3600   and raise the job limit to match:
#   MAX_ITER=10000 RUN=long_person_center sbatch --time=04:00:00 slurm/experiments/exp1_depth_anchor.sh
#
# SUCCESS CRITERION (printed at the end), 100-image holdout, V1 settings, matched people:
#   depth_fit_slope moves from 0.33 toward 1 and depth_fit_intercept_m from 2.36 m toward 0,
#   with PVE not worse than 70.5 + 1 mm and recall not worse than 90.2 - 1 point.
#   Meeting it justifies the longer run; it does not by itself show the fix works.
#SBATCH --job-name=exp1-depth-anchor
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp1_depth_anchor_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp1_depth_anchor_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
if [ "$(basename "$(dirname "$CKPT")")" = checkpoints ]; then   # logs/anny_model/<run>/checkpoints/<ep>.pt
  TAG="${TAG:-$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT" .pt)}"
else
  TAG="${TAG:-$(basename "$CKPT" .pt)}"
fi
GT_CENTER="${GT_CENTER:-person_center}"   # root = control arm (production targets, same schedule)
ALPHA_DIST="${ALPHA_DIST:-1}"             # production 1; documented variant 5
ALPHA_TRANSL="${ALPHA_TRANSL:-1}"         # production 1; documented variant 5
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"              # 2 epochs
DEBUG_ANCHOR="${DEBUG_ANCHOR:-2}"         # anchor print on the first N batches
BASELINE_TRACE="${BASELINE_TRACE:-0}"     # 1 = also trace CKPT with the same code (apples-to-apples read)
CONTROL_RUN="${CONTROL_RUN:-}"            # a finished control-arm run dir (its summary.json); empty = no side-by-side
RUN="${RUN:-pilot_${GT_CENTER}_dist${ALPHA_DIST}_transl${ALPHA_TRANSL}_${TAG}}"
OUT="results/experiments/exp1_depth_anchor/${RUN}"
N_EPOCHS=$(( MAX_ITER / N_ITERS ))
EXPECTED_TRAIN_MIN=$(( MAX_ITER * 40 / 67 / 60 ))   # MAX_ITER * batch 4 / 6.7 images/s, in minutes

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp1_depth_anchor run ${RUN}"
echo "    start: ${CKPT}"
echo "    gt_center_joint=${GT_CENTER} alpha_dist=${ALPHA_DIST} alpha_transl=${ALPHA_TRANSL} lr=${LR}"
echo "    ${N_ITERS} iters/epoch x ${N_EPOCHS} epochs, batch 4 = $(( MAX_ITER * 4 )) images"
echo "    expected training time at the measured 6.7 images/s: ~${EXPECTED_TRAIN_MIN} min (+ one 100-image V0 evaluation per epoch)"

echo "=== 1. TRAINING -> ${OUT}/train.log"
T0=\$(date +%s)
python train.py \\
    --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
    --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone dinov2_vitl14 --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
    --alpha_dist ${ALPHA_DIST} --alpha_transl ${ALPHA_TRANSL} --start_2d_epoch 0 \\
    --gt_center_joint ${GT_CENTER} --debug_anchor ${DEBUG_ANCHOR} \\
    --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
    --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
    --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s (startup + ${N_EPOCHS} epochs + ${N_EPOCHS} holdout evaluations)"
echo "--- anchor check on the first ${DEBUG_ANCHOR} batches (a consistent target keeps pred offset_y in [-0.5, 0.5])"
grep -E "\[debug_anchor\]" "${OUT}/train.log" || echo "  (no debug_anchor lines)"
echo "--- throughput (images/s) and in-training V0 evaluation (train.py evaluate(), helper bones UNMASKED; its GT cohort"
echo "    follows --gt_center_joint through prepare_gt, so compare V0 only within an arm; the trace below is the fixed read)"
grep -E "\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): |NaN|WARNING|Error" "${OUT}/train.log" || true

LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

echo "=== 2. FIXED HOLDOUT TRACE (100 images, --select holdout, V1 settings) on \${LAST}"
python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${LAST}" --n 100 --select holdout \\
    --out "${OUT}/holdout_trace" > "${OUT}/holdout_trace.log" 2>&1 \\
    || { echo "ERROR: holdout trace failed; tail of ${OUT}/holdout_trace.log:"; tail -n 30 "${OUT}/holdout_trace.log"; exit 1; }
grep -A 12 "=== PLACEMENT" "${OUT}/holdout_trace.log" || true

if [ -f results/diagnostics/manifest.json ]; then
  echo "=== 3. MANIFEST HOLDOUT TRACE on \${LAST}"
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${LAST}" --manifest results/diagnostics/manifest.json \\
      --manifest_split holdout --n_vis 20 --out "${OUT}/manifest_trace_holdout" > "${OUT}/manifest_trace_holdout.log" 2>&1 \\
      || { echo "WARNING: manifest trace failed; tail:"; tail -n 30 "${OUT}/manifest_trace_holdout.log"; }
else
  echo "=== 3. manifest trace SKIPPED: results/diagnostics/manifest.json is missing (slurm/diagnostics/diag_manifest.sh builds it; never rebuild an existing one)"
fi

if [ "${BASELINE_TRACE}" = "1" ]; then
  echo "=== 4. BASELINE TRACE on ${CKPT} with the same code and holdout"
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "${CKPT}" --n 100 --select holdout \\
      --out "${OUT}/holdout_trace_baseline" > "${OUT}/holdout_trace_baseline.log" 2>&1 \\
      || { echo "WARNING: baseline trace failed; tail:"; tail -n 30 "${OUT}/holdout_trace_baseline.log"; }
fi

python - <<'PY'
import json, os
out = os.environ['OUT_DIR']
# anny_s2_shape_v5/00099 on the same 100-image holdout, V1 (docs/diagnostics_findings.md). The size ratio there
# is a ratio of medians (82 / 127 px); the depth ratio was not reported (means 3.00 / 1.92 m); head_px_err is
# the summary.json placement MEAN 64.5 px (the findings table quotes the 46.0 px median; the root median is 14 px).
BASE = dict(depth_fit_slope=0.33, depth_fit_intercept_m=2.36, depth_ratio_median=None, size_ratio_median_px=0.67,
            pve=70.5, recall=90.2, head_px_err=64.5, root_px_err=None)
KEYS = ['depth_fit_slope', 'depth_fit_intercept_m', 'depth_ratio_median', 'size_ratio_median_px',
        'pve', 'recall', 'head_px_err', 'root_px_err']
LABEL = dict(head_px_err='head_px_err (mean)', root_px_err='root_px_err (mean)')   # placement rows are means

def pick(path):
    if not os.path.isfile(path):
        return {}
    try:
        v = json.load(open(path))['variants']['V1_eval_masked']
    except Exception as e:
        print('  could not read', path, e)
        return {}
    pl = v.get('placement', {})
    d = {k: pl.get(k) for k in KEYS if k not in ('pve', 'recall')}
    d.update(pve=v.get('pve'), recall=v.get('recall'))
    return d

def f(x, nd=2):
    return '   -' if x is None else format(x, '.' + str(nd) + 'f')

new = pick(out + '/holdout_trace/summary.json')
rerun = pick(out + '/holdout_trace_baseline/summary.json')
man = pick(out + '/manifest_trace_holdout/summary.json')
print()
print('=== exp1_depth_anchor SUMMARY: 100-image holdout, V1 (helper bones masked), matched people ===')
print(format('metric', '<24') + format('baseline(findings)', '>20') + format('baseline(rerun)', '>17')
      + format('this run', '>12') + format('manifest holdout', '>18'))
for k in KEYS:
    print(format(LABEL.get(k, k), '<24') + format(f(BASE.get(k)), '>20') + format(f(rerun.get(k)), '>17')
          + format(f(new.get(k)), '>12') + format(f(man.get(k)), '>18'))
print('head_px_err baseline: mean 64.50 px (the row above, same statistic as the trace); median 46.00 px is the figure')
print('in docs/diagnostics_findings.md. The trace reports means only.')
ctrl_dir = os.environ.get('CONTROL_RUN', '')
ctrl = {}
if ctrl_dir:
    try:
        ctrl = json.load(open(os.path.join(ctrl_dir, 'summary.json'))).get('this_run') or {}
    except Exception as e:
        print('  could not read the control run', ctrl_dir, e)
    print('--- arms side by side: this run vs control (CONTROL_RUN=' + ctrl_dir + ')')
    print(format('metric', '<24') + format('this run', '>12') + format('control', '>12') + format('this - control', '>16'))
    for k in ('depth_fit_slope', 'depth_fit_intercept_m', 'pve', 'recall'):
        a, c = new.get(k), ctrl.get(k)
        print(format(k, '<24') + format(f(a), '>12') + format(f(c), '>12')
              + format(f(None if None in (a, c) else a - c), '>16'))
s, b, pve, rec = (new.get(k) for k in ('depth_fit_slope', 'depth_fit_intercept_m', 'pve', 'recall'))
ok = None
if None not in (s, b, pve, rec):
    toward = abs(1.0 - s) < abs(1.0 - BASE['depth_fit_slope']) and abs(b) < abs(BASE['depth_fit_intercept_m'])
    kept = pve <= BASE['pve'] + 1.0 and rec >= BASE['recall'] - 1.0
    ok = bool(toward and kept)
    print('criterion: slope toward 1 and intercept toward 0: ' + ('yes' if toward else 'NO')
          + '; PVE <= ' + f(BASE['pve'] + 1.0, 1) + ' and recall >= ' + f(BASE['recall'] - 1.0, 1) + ': '
          + ('yes' if kept else 'NO'))
if ok is None:
    verdict = 'INCOMPLETE (missing numbers; see the trace logs)'
elif ok:
    verdict = 'PILOT MEETS THE CRITERION -> worth the 10-epoch run (header of this script)'
else:
    verdict = 'PILOT DOES NOT MEET THE CRITERION'
print('VERDICT: ' + verdict)
print('This is a 2-epoch pilot read (8,000 images, fresh Adam state, no seed control): it justifies or vetoes the')
print('longer run and nothing more. Compare with GT_CENTER=root (control arm) before attributing a change to the flag.')
json.dump(dict(baseline_findings=BASE, baseline_rerun=rerun, this_run=new, manifest_holdout=man, meets_criterion=ok,
               control_run=ctrl_dir, control=ctrl), open(out + '/summary.json', 'w'), indent=2)
print('wrote ' + out + '/summary.json')
PY
echo "SUCCESS CRITERION: depth_fit_slope toward 1 (baseline 0.33), depth_fit_intercept_m toward 0 (baseline 2.36 m),"
echo "                   PVE <= 71.5 (baseline 70.5), recall >= 89.2 (baseline 90.2), all on the 100-image holdout, V1."
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' CONTROL_RUN='${CONTROL_RUN}'; $COMMAND"
