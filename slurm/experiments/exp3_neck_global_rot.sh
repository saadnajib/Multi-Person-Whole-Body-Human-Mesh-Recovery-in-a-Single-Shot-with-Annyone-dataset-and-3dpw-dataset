#!/bin/bash
# exp3_neck_global_rot: fine-tune the production checkpoint with a GLOBAL head/neck/wrist orientation term
# (--alpha_global_rot, loss.py) on top of the production loss, then re-run the fixed holdout and manifest traces.
#
# WHY (docs/diagnostics_findings.md, "Neck"): the body-model probe passed every single-joint perturbation, so no
# joint-mapping or rotation-convention error exists and nothing is "corrected" here. What the diagnostics measured
# is a supervision mismatch: the per-bone (local) rotmat L1 reports 62.7 deg on the neck bones and 89.5 deg on
# the head while the GLOBAL head orientation is off by 12.0 deg (holdout) / 19.9 deg (manifest train). Four bones
# in series can realise one head pose many ways; the local term penalises every decomposition but the annotator's
# and the v4/v5 neck boost (really x2.41 after renormalisation) moved nothing. The new term compares the post-FK
# camera-frame rotation of head, neck03, wrist.L, wrist.R with the GT one (same per-bone L1 the rotmat term uses,
# averaged over the 4 bones and the people). Both sides come from forward-kinematics passes that already run;
# no architecture change, only a few extra tensor ops per step.
#
# ALPHA (default ALPHA_GROT=5; the pilot's first log line decides whether to keep it):
#   per-bone L1 over 9 entries ~ 0.0615 x error_deg for a rotation error about a generic axis (sampled: 12 deg ->
#   0.74, 22.7 deg -> 1.39). At the measured errors (head 12.0, wrists 22.7, neck03 assumed like the head) the raw
#   term is ~ (0.74 + 0.74 + 1.39 + 1.39) / 4 = 1.07 per person; training people are harder than matched holdout
#   people, so expect 1.0-1.5. The weighted rotmat term is 0.1 x 44.6 = 4.5 in the overfit log (first logged
#   epochs of the production model, docs/diagnostics_findings.md) and about twice that if the 12.9 % loss share at
#   epoch 99 is taken literally (~8.8 of a ~68 total). alpha 5 x 1.07 = 5.3 lands between the two, i.e. the new
#   term starts at roughly the weighted rotmat term (~7 % of the total). The summary prints
#   alpha*global_rot / (0.1*rotmat) from the first logged line; rescale with ALPHA_GROT if it leaves 0.5-2.
#
# WHAT RUNS (one GPU, ~2.5 h with the defaults; see --time):
#   1. the diag manifest's image indices -> ${OUT}/exclude_manifest_indices.txt, passed as --exclude_indices_file
#      so the manifest-train read (the hard neck poses, in-sample for the base model) is not re-seen by the pilot
#      (with 8,000 draws from 579,860 images this is a formality; it matters for the longer run)
#   2. train.py from CKPT with the production v5 recipe (same freeze pattern, lr 1e-5, alphas, boosts, batch 4) plus
#        --alpha_global_rot ${ALPHA_GROT} --global_rot_bones "${GROT_BONES}"      ALPHA_GROT=0 = control arm
#        --start_2d_epoch 0              the epoch counter restarts at 0 (no resume): keeps j2d/v2d ON as at epoch 99
#        --n_iters_per_epoch 1000 --max_iter ${MAX_ITER}   2 epochs = 8,000 images ~ 20 min at the measured 6.7 images/s
#        --eval_freq 1 --nb_max_ckpt 2 --log_freq 50
#      Checkpoints go to ${OUT}/train/checkpoints/0000N.pt, never next to the production run: train.py keeps the
#      --nb_max_ckpt highest epoch numbers in a directory and would delete a pilot's 00000.pt there.
#   3. scripts/diagnostics/diag_holdout_trace.py on the LAST checkpoint, V1 settings (helper bones masked), on
#      exactly: the fixed 100-image holdout (--select holdout), the manifest holdout split, the manifest train split
#      (MANIFEST_TRAIN=0 skips it). Nothing else is evaluated.
#   4. a summary printing the new numbers next to the anny_s2_shape_v5/00099 baseline (same sets, V1).
#
# USAGE (repo root on the cluster):
#   sbatch slurm/experiments/exp3_neck_global_rot.sh                               # alpha 5, 2-epoch pilot
#   ALPHA_GROT=0 sbatch slurm/experiments/exp3_neck_global_rot.sh                  # control arm: same schedule, term off
#   ALPHA_GROT=10 sbatch slurm/experiments/exp3_neck_global_rot.sh                 # if the first line shows the term < 0.5x rotmat
#   GROT_BONES=head,neck03 sbatch slurm/experiments/exp3_neck_global_rot.sh        # neck only (wrists are the secondary read)
#   BASELINE_TRACE=1 sbatch slurm/experiments/exp3_neck_global_rot.sh             # also trace CKPT itself (+40-60 min)
#
# LONGER RUN (only after a pilot that meets the criterion below):
#   10 epochs = MAX_ITER=10000 = 40,000 images; at the measured 6.7 images/s: 40000 / 6.7 = 5970 s ~ 1.7 h of
#   training plus 10 holdout evaluations and the traces; 20 epochs ~ 3.3 h. Rescale from the pilot's own line
#       [throughput] epoch 1: ... images/s
#   with   hours = MAX_ITER * 4 / images_per_s / 3600   and raise the job limit to match:
#   MAX_ITER=10000 RUN=long_alpha5 sbatch --time=06:00:00 slurm/experiments/exp3_neck_global_rot.sh
#
# SUCCESS CRITERION (printed at the end), V1 settings, matched people:
#   primary:   rot_global_head_deg below 12.0 on the 100-image holdout and below 19.9 on the manifest train split
#              (8.6 on the manifest holdout), with holdout PVE <= 70.5 + 1 mm and recall >= 90.2 - 1 point;
#   also read: neck-category PVE (76.3 manifest holdout / 93.7 manifest train), PVE_hn (52.3 holdout), shape bias
#              (the term back-propagates into the shape head through the shape-dependent rest frames);
#   secondary: the wrists' global error (22.7 deg) and the fingers' local error (17.3 deg): fingers are children of
#              the wrists, so a better wrist frame moves every finger's global pose, in either direction.
#   Meeting it justifies the longer run; it does not by itself show the fix works.
#SBATCH --job-name=exp3-neck-global-rot
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp3_neck_global_rot_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp3_neck_global_rot_%j.err
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
ALPHA_GROT="${ALPHA_GROT:-5}"                       # 0 = control arm (term off, same schedule)
GROT_BONES="${GROT_BONES:-head,neck03,wrist.L,wrist.R}"
GT_CENTER="${GT_CENTER:-root}"                     # train.py --gt_center_joint: lets a pilot stack on an exp1 checkpoint; root = today's behaviour
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"                        # 2 epochs
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"   # build_manifest.py default (slurm/diagnostics/diag_manifest.sh)
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-1}"           # 0 = the production shuffled loader (manifest images may be drawn)
MANIFEST_TRAIN="${MANIFEST_TRAIN:-1}"               # 0 = skip the manifest train-split trace (~1 h)
BASELINE_TRACE="${BASELINE_TRACE:-0}"               # 1 = also trace CKPT with the same code (apples-to-apples read)
BONES_TAG=$(echo "${GROT_BONES}" | tr -d ' ' | tr ',.' '-_')
RUN="${RUN:-pilot_grot${ALPHA_GROT}_${BONES_TAG}_${TAG}}"
OUT="results/experiments/exp3_neck_global_rot/${RUN}"
EXCL="${OUT}/exclude_manifest_indices.txt"
N_EPOCHS=$(( MAX_ITER / N_ITERS ))
EXPECTED_TRAIN_MIN=$(( MAX_ITER * 40 / 67 / 60 ))   # MAX_ITER * batch 4 / 6.7 images/s, in minutes

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp3_neck_global_rot run ${RUN}"
echo "    start: ${CKPT}"
echo "    alpha_global_rot=${ALPHA_GROT} global_rot_bones=${GROT_BONES} gt_center_joint=${GT_CENTER} lr=${LR} exclude_manifest=${EXCLUDE_MANIFEST}"
echo "    ${N_ITERS} iters/epoch x ${N_EPOCHS} epochs, batch 4 = $(( MAX_ITER * 4 )) images"
echo "    expected training time at the measured 6.7 images/s: ~${EXPECTED_TRAIN_MIN} min (+ one 100-image V0 evaluation per epoch)"

echo "=== 1. MANIFEST IMAGES -> ${EXCL} (zero sampling weight in the fine-tune)"
EXCL_ARG=""
if [ "${EXCLUDE_MANIFEST}" = "1" ] && [ -f "${MANIFEST}" ]; then
  python - <<'PY'
import json, os
man = json.load(open(os.environ['MANIFEST_PATH']))
ents = man['entries']
idx = sorted({int(e['idx']) for e in ents})
with open(os.environ['EXCL_PATH'], 'w') as f:
    f.write('# AnnyOne indices of the %d diag-manifest people (%d images), from %s; never sampled by this fine-tune\n'
            % (len(ents), len(idx), os.environ['MANIFEST_PATH']))
    f.write('\n'.join(str(i) for i in idx) + '\n')
print('wrote', os.environ['EXCL_PATH'], ':', len(idx), 'images excluded')
PY
  EXCL_ARG="--exclude_indices_file ${EXCL}"
else
  echo "no exclusion (EXCLUDE_MANIFEST=${EXCLUDE_MANIFEST}, manifest present: \$([ -f "${MANIFEST}" ] && echo yes || echo no)); the production shuffled loader is used"
fi

echo "=== 2. TRAINING -> ${OUT}/train.log"
T0=\$(date +%s)
python train.py \\
    --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
    --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone dinov2_vitl14 --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
    --alpha_global_rot ${ALPHA_GROT} --global_rot_bones "${GROT_BONES}" --start_2d_epoch 0 \\
    --gt_center_joint "${GT_CENTER}" \\
    \${EXCL_ARG} \\
    --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
    --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
    --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s (startup + ${N_EPOCHS} epochs + ${N_EPOCHS} holdout evaluations)"
echo "--- term configuration, throughput (images/s) and in-training V0 evaluation (train.py evaluate(), helper bones UNMASKED)"
grep -E "\[loss\] global_rot|\[loss\] joint weights|\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): |NaN|WARNING|Error" "${OUT}/train.log" || true
echo "--- magnitude check: alpha*global_rot vs 0.1*rotmat on the first and last logged lines (target 0.5-2x)"
python - <<'PY'
import os, re
alpha = float(os.environ['ALPHA_GROT_VAL'])
# EPOCH= prints share physical lines with tqdm's '\r'-separated bar (stderr merged), so match anywhere and trim
lines = [l[l.index('EPOCH='):] for l in open(os.environ['OUT_DIR'] + '/train.log') if 'EPOCH=' in l]
def val(l, k):
    m = re.search(k + r': ([0-9.eE+-]+)', l)
    return float(m.group(1)) if m else None
for tag, l in (('first', lines[0]) if lines else ('first', None), ('last', lines[-1]) if lines else ('last', None)):
    if l is None:
        print('  no EPOCH= lines in train.log'); break
    r, g, t = val(l, 'rotmat'), val(l, 'global_rot'), val(l, 'total')
    if g is None:
        print('  %s line: rotmat %.2f (x0.1 = %.2f), no global_rot term logged (alpha %g)' % (tag, r, 0.1 * r, alpha)); continue
    print('  %s line: raw global_rot %.3f x alpha %g = %.2f | raw rotmat %.2f x 0.1 = %.2f | ratio %.2f | %.1f%% of total %.1f'
          % (tag, g, alpha, alpha * g, r, 0.1 * r, alpha * g / max(0.1 * r, 1e-9), 100 * alpha * g / max(t, 1e-9), t))
PY

LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

trace () {   # \$1 = checkpoint, \$2 = output dir, \$3... = selection flags
  local ck="\$1" od="\$2"; shift 2
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${ck}" --out "\${od}" "\$@" > "\${od}.log" 2>&1 \\
      || { echo "WARNING: trace \${od} failed; tail:"; tail -n 30 "\${od}.log"; return 0; }
  grep -E "^\[select\]|^V1_eval_masked|^=== (PER-VARIANT|GLOBAL|PER CATEGORY)|^neck |^variant " "\${od}.log" | head -n 24 || true
}

echo "=== 3a. FIXED HOLDOUT TRACE (100 images, --select holdout, V1 settings) on \${LAST}"
trace "\${LAST}" "${OUT}/holdout_trace" --select holdout --n 100
if [ -f "${MANIFEST}" ]; then
  echo "=== 3b. MANIFEST HOLDOUT-SPLIT TRACE on \${LAST}"
  trace "\${LAST}" "${OUT}/manifest_trace_holdout" --manifest "${MANIFEST}" --manifest_split holdout --n_vis 20
  if [ "${MANIFEST_TRAIN}" = "1" ]; then
    echo "=== 3c. MANIFEST TRAIN-SPLIT TRACE (hard neck poses; in-sample for the base model, excluded from this fine-tune) on \${LAST}"
    trace "\${LAST}" "${OUT}/manifest_trace_train" --manifest "${MANIFEST}" --manifest_split train --n_vis 40
  fi
else
  echo "=== 3b/3c. manifest traces SKIPPED: ${MANIFEST} is missing (slurm/diagnostics/diag_manifest.sh builds it; never rebuild an existing one)"
fi
if [ "${BASELINE_TRACE}" = "1" ]; then
  echo "=== 3d. BASELINE TRACES on ${CKPT} with the same code and selections"
  trace "${CKPT}" "${OUT}/holdout_trace_baseline" --select holdout --n 100
  if [ -f "${MANIFEST}" ]; then
    trace "${CKPT}" "${OUT}/manifest_trace_holdout_baseline" --manifest "${MANIFEST}" --manifest_split holdout --n_vis 20
    [ "${MANIFEST_TRAIN}" = "1" ] && trace "${CKPT}" "${OUT}/manifest_trace_train_baseline" --manifest "${MANIFEST}" --manifest_split train --n_vis 40
  fi
fi

python - <<'PY'
import json, os
out = os.environ['OUT_DIR']
V = 'V1_eval_masked'
# anny_s2_shape_v5/00099, V1 settings, same sets (docs/diagnostics_findings.md: global head 12.0 / 8.6 / 19.9 deg, local
# neck 62.7 / head 89.5 / fingers 17.3 deg, neck-category PVE 76.3 holdout / 93.7 train with PVE_hn 79 on train, PVE 70.5,
# recall 90.2; PVE_hn 52.3 and global wrist 22.7 deg from the same holdout trace).
BASE = {
    'holdout': dict(rot_global_head_deg=12.0, pve=70.5, pve_headneck=52.3, recall=90.2, rot_global_wrist_deg=22.7,
                    rot_neck=62.7, rot_head=89.5, rot_fingers=17.3, shape_bias_age=-0.015),
    'manifest_holdout': dict(rot_global_head_deg=8.6, neck_pve=76.3),
    'manifest_train': dict(rot_global_head_deg=19.9, neck_pve=93.7, neck_pve_headneck=79.0),
}
KEYS = ['rot_global_head_deg', 'rot_global_wrist_deg', 'rot_neck', 'rot_head', 'rot_fingers', 'pve', 'pve_headneck',
        'recall', 'shape_bias_age', 'neck_n', 'neck_pve', 'neck_pve_headneck', 'neck_rot_global_head_deg']

def load(path):
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print('  could not read', path, e); return None

def read(summ):
    if not summ:
        return {}
    v = summ['variants'][V]
    g = v.get('rot_err_deg_by_group', {})
    d = dict(rot_global_head_deg=v['rot_global_deg']['head'], rot_global_wrist_deg=v['rot_global_deg']['wrist'],
             rot_neck=g.get('neck'), rot_head=g.get('head'), rot_fingers=g.get('fingers'), pve=v.get('pve'),
             pve_headneck=v.get('pve_headneck'), recall=v.get('recall'), shape_bias_age=v.get('shape_bias', {}).get('age'))
    pt = (summ.get('per_tag') or {}).get(V, {}).get('neck')
    if pt:
        d.update(neck_n=pt.get('n'), neck_pve=pt.get('pve'), neck_pve_headneck=pt.get('pve_headneck'),
                 neck_rot_global_head_deg=pt.get('rot_global_head_deg'))
    return d

def f(x, nd=1):
    return '   -' if x is None else format(x, '.' + str(nd) + 'f')

sets = [('holdout', 'holdout_trace', '100-image holdout (--select holdout)'),
        ('manifest_holdout', 'manifest_trace_holdout', 'manifest holdout split'),
        ('manifest_train', 'manifest_trace_train', 'manifest train split (hard neck poses, excluded from the fine-tune)')]
res = {}
print()
print('=== exp3_neck_global_rot SUMMARY: V1 (helper bones masked), matched people; deg for rot_*, mm for pve_* ===')
for name, sub, title in sets:
    new = read(load(out + '/' + sub + '/summary.json')); rerun = read(load(out + '/' + sub + '_baseline/summary.json'))
    res[name] = dict(this_run=new, baseline_rerun=rerun)
    if not new and not rerun:
        print('--- ' + title + ': no trace output'); continue
    print('--- ' + title)
    print(format('metric', '<26') + format('baseline(findings)', '>20') + format('baseline(rerun)', '>17') + format('this run', '>12'))
    for k in KEYS:
        if BASE[name].get(k) is None and new.get(k) is None and rerun.get(k) is None:
            continue
        nd = 3 if k == 'shape_bias_age' else (0 if k == 'neck_n' else 1)
        print(format(k, '<26') + format(f(BASE[name].get(k), nd), '>20') + format(f(rerun.get(k), nd), '>17') + format(f(new.get(k), nd), '>12'))
H, T = res['holdout']['this_run'], res['manifest_train']['this_run']
h_head, pve, rec = H.get('rot_global_head_deg'), H.get('pve'), H.get('recall')
ok = None
if None not in (h_head, pve, rec):
    head_better = h_head < BASE['holdout']['rot_global_head_deg']
    kept = pve <= BASE['holdout']['pve'] + 1.0 and rec >= BASE['holdout']['recall'] - 1.0
    t_head = T.get('rot_global_head_deg')
    train_better = None if t_head is None else t_head < BASE['manifest_train']['rot_global_head_deg']
    ok = bool(head_better and kept and (train_better is not False))
    print('criterion: holdout global head error < 12.0: ' + ('yes' if head_better else 'NO')
          + '; manifest-train global head error < 19.9: ' + ('n/a' if train_better is None else ('yes' if train_better else 'NO'))
          + '; PVE <= ' + f(BASE['holdout']['pve'] + 1.0) + ' and recall >= ' + f(BASE['holdout']['recall'] - 1.0) + ': ' + ('yes' if kept else 'NO'))
    w, fg = H.get('rot_global_wrist_deg'), H.get('rot_fingers')
    print('secondary: global wrist ' + f(w) + ' deg (baseline 22.7), local fingers ' + f(fg) + ' deg (baseline 17.3): '
          + ('not worse' if (w is not None and fg is not None and w <= 22.7 + 1.0 and fg <= 17.3 + 1.0) else 'CHECK'))
if ok is None:
    verdict = 'INCOMPLETE (missing numbers; see the trace logs)'
elif ok:
    verdict = 'PILOT MEETS THE CRITERION -> worth the 10-epoch run (header of this script)'
else:
    verdict = 'PILOT DOES NOT MEET THE CRITERION'
print('VERDICT: ' + verdict)
print('This is a 2-epoch pilot read (8,000 images, fresh Adam state, no seed control): it justifies or vetoes the')
print('longer run and nothing more. Compare with ALPHA_GROT=0 (control arm) before attributing a change to the term.')
json.dump(dict(baseline_findings=BASE, results=res, meets_criterion=ok), open(out + '/summary.json', 'w'), indent=2)
print('wrote ' + out + '/summary.json')
PY
echo "SUCCESS CRITERION: rot_global_head_deg < 12.0 on the 100-image holdout and < 19.9 on the manifest train split (8.6 manifest"
echo "                   holdout), with holdout PVE <= 71.5 (baseline 70.5) and recall >= 89.2 (baseline 90.2); also read the"
echo "                   neck-category PVE (76.3 holdout / 93.7 train) and PVE_hn (52.3 holdout). Secondary: global wrist error"
echo "                   (22.7 deg) and local finger error (17.3 deg) not worse than +1 deg. All V1 settings, matched people."
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' MANIFEST_PATH='${MANIFEST}' EXCL_PATH='${EXCL}' ALPHA_GROT_VAL='${ALPHA_GROT}'; $COMMAND"
