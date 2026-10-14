#!/bin/bash
# exp4_fingers: measure the hands before changing anything, test MAGNIFICATION at inference on the production
# checkpoint (no training), then an optional 2-epoch pilot with the region-normalised rotation loss.
#
# WHY (docs/diagnostics_findings.md, "Fingers"): finger rotation error 17.3 deg on the holdout vs 15.1 deg with the
# fingers zeroed (--relax_hands): the predicted fingers are worse than the rest pose. Fingers get the smallest
# gradients of any bone group (|g| 4.7 vs root 598) and the worst 6D conditioning (0.25-0.30); every single-joint
# perturbation of the body model PASSED, so the rig and the decoding are not the cause. Two hypotheses are left and
# this script separates them without a new architecture:
#   resolution  - a hand covers ~1-3 of the 14 px tokens at 672 px (step 1 measures it); if feeding the SAME network
#                 a person crop (more tokens per hand, nothing retrained) lowers the finger error below the relaxed
#                 floor, the input resolution is the bottleneck and a crop-based second pass is the cheapest fix;
#   supervision - the rotmat term is a SUM over bones, so the 30 phalanges already hold 40% of the active budget (70%
#                 under the production boosts) and the 4-bone neck chain 5%; --rotmat_region_norm (loss.py) averages
#                 within body / hands / head-neck and across the regions instead (x0.61 body, x0.83 hands, x6.25
#                 head-neck under the helper mask). It is mainly a head-neck re-weighting and slightly LOWERS the hand
#                 share, so it is a control for "more finger weight helps", not a finger boost: the v5 boosts already
#                 gave the hands x2.41 "with no measurable effect".
#
# WHAT WAS VERIFIED BY READING THE CODE (reader notes, re-read at the cited lines; nothing here was executed):
#   flip / left-right: --flip (default 1) and --brightness are parsed (train.py argparse, --flip/--brightness) but never
#     read: 'args.flip' occurs only in the commented-out BEDLAM DataLoader block (train.py ~1058). AnnyOne.__getitem__
#     applies ImageOps.contain, ImageOps.pad and normalize_rgb only (datasets/annyone.py:42-46); ThreeDPWAnny the same
#     (datasets/threedpw_anny.py:78-82); collate only stacks (datasets/bedlam.py:365-420); prepare_gt has no mirroring.
#     The only flip code in the repo (ImageOps.mirror + SMPL-X flip pairs) is datasets/bedlam.py:225-228,266,290-291
#     and is not on the AnnyOne path. So: no horizontal flip, no L/R swap, no 163-bone L/R permutation exists or is
#     needed, and there is no augmentation at all on the AnnyOne path. The hand bones are found by NAME: model.py
#     hand_bone_mask = the 30 phalanges finger{1..5}-{1,2,3}.{L,R} (MakeHuman's thumb is finger1-*; no bone is named
#     'thumb'); the 8 metacarpals and 2 wrists are outside it; --boost_hand_weight covers 32 (phalanges + wrists);
#     docs/diagnostics_findings.md says "28 finger/thumb bones" (a miscount, 30), README says 38 (= incl. metacarpals).
#     All 40 hand bones are in the 75 ACTIVE bones of useful_rotmat (model.py:132-146): none is helper-masked.
#   visibility masking: no per-joint visibility annotation exists or is used. The batch carries imagename, K, n_humans,
#     valid_humans, anny_pose, anny_shape (collate keeps ndarray keys only; tight_bbox is a list, segm_color an int).
#     People are dropped WHOLE when the anchor joint is behind the camera (z <= 0.1), off-image or in an occupied 14 px
#     cell (train.py prepare_gt, the block after 'n_patch = ...'); every kept person has all 163 bones supervised by
#     rotmat / j3d / v3d, fingers like the body. The 2D terms keep GT points with 0 < p < img_size only (loss.py, the
#     idx_v2d / idx_j2d lines), identically for every joint; self-occluded or person-occluded points inside the image
#     are supervised as visible. The only per-bone masking is the helper identity mask applied to BOTH sides under
#     --mask_helper_joints_train (train.py:193-204, model.py:360-363). No "label available" mask existed.
#   3DPW identity fingers (CONFIRMED, so --mask_3dpw_hands was implemented): scripts/data/smpl_to_anny.py:561-575
#     initialises every bone's 6D rotation to identity and lets only the root and the bones carrying a mapped SMPL
#     joint rotate; 667-673 zeroes every other bone's gradient, so all 30 phalanges and 8 metacarpals stay at EXACT
#     identity (finger2-1.L/R carry SMPL's hand joint at their own head and get no joint gradient either).
#     scripts/data/build_threedpw_anny_labels.py:105-118 writes those identities into anny_pose for all 163 bones,
#     datasets/threedpw_anny.py:93-98 passes them on, and prepare_gt / loss.py cannot tell them from AnnyOne labels:
#     in the mixed runs (train.py WeightedRandomSampler, --mix_3dpw_ratio 0.3) 30% of the samples teach flat fingers
#     through rotmat, j3d and v3d. The production v5 lineage is PURE AnnyOne and unaffected, so this pilot does not
#     use the flag; slurm/history/train_distill_vits_v7_mixed.sh, slurm/train/train_teacher_v9_mixed.sh,
#     slurm/train/train_distill_vits_v8_final.sh and slurm/train/train_v10_scratch_mixed.sh should add
#     --mask_3dpw_hands 1 (train.py tags every human has_hand_labels 1/0; loss.py drops the finger/metacarpal bones of
#     the 0-tagged people from the rotmat and global_rot terms; j3d/v3d are untouched).
#   6D init (reader-confirmed, NOT changed here): HPH.init_body_pose = [1,0,0,1,0,0] per bone (model.py:565) is decoded
#     by utils.rot6d_to_rotmat with columns x[0:3], x[3:6] (utils/humans.py:20-21), i.e. two parallel columns: the init
#     is singular, not the rest pose, and the 0.25-0.30 "conditioning" of the fingers is |sin| of exactly that angle.
#     A post-load re-parametrisation is its own experiment (listed under unresolved items in the commit).
#
# WHAT RUNS (one GPU; ~1.5 h with the defaults, see the per-step estimates):
#   1. scripts/experiments/measure_hand_px.py on the 100-image holdout and on the diag manifest (both splits): hand bbox
#      size at 1280 px, at the 672 px input and under the x${CROP_FACTOR} person crop, in px and 14 px tokens, with the
#      fractions of hands under 1 / 2 / 4 tokens. Body-model passes only (~3-5 min).
#   2. scripts/diagnostics/diag_holdout_trace.py --person_crop 1 on the PRODUCTION checkpoint: the 4 usual variants
#      plus V4_person_crop (one forward per kept GT person on a crop-factor x GT-joint-bbox crop of the original render,
#      crop-corrected K, V1 settings) on the fixed holdout, the manifest holdout split and the manifest train split.
#      Prints the V4-vs-V1 paired block (same people): rot_fingers vs V1 (17.3 on the findings' holdout) and vs the
#      V3 relaxed floor (15.1), PVE / PA-PVE / PVE_hn deltas, by magnification. Placement/depth are excluded for V4
#      (crop camera). ~30 min for the three traces.
#   3. PILOT=1 (default): train.py from CKPT with the production v5 recipe plus --rotmat_region_norm ${REGION_NORM}
#      (REGION_NORM=0 = control arm, same schedule) and --alpha_global_rot ${ALPHA_GROT} (0 unless combined on purpose),
#      --start_2d_epoch 0, 2 x 1000 iterations at batch 4 = 8,000 images (~20 min at the measured 1.65-1.71 it/s =
#      6.7 images/s, plus one 100-image V0 evaluation per epoch), manifest images excluded from sampling, then the
#      same three traces on the last checkpoint (~25 min). Checkpoints stay under ${OUT}/train/checkpoints.
#   4. a summary next to the anny_s2_shape_v5/00099 baseline and the decision rule below.
#
# USAGE (repo root on the cluster):
#   sbatch slurm/experiments/exp4_fingers.sh                               # steps 1-4 with the defaults
#   PILOT=0 sbatch slurm/experiments/exp4_fingers.sh                       # inference only (steps 1, 2, 4): ~40 min
#   CROP_FACTOR=2.0 sbatch slurm/experiments/exp4_fingers.sh               # looser crop (person ~50% of the crop)
#   REGION_NORM=0 sbatch slurm/experiments/exp4_fingers.sh                 # pilot control arm: same schedule, sum over bones
#   ALPHA_GROT=5 sbatch slurm/experiments/exp4_fingers.sh                  # combine with exp3's global term on purpose
#   MANIFEST_TRAIN=0 sbatch slurm/experiments/exp4_fingers.sh              # skip the 40-image manifest train traces
#
# LONGER RUN (only if the pilot's finger error moves): MAX_ITER=10000 RUN=long_regionnorm sbatch --time=06:00:00 ...;
#   hours = MAX_ITER * 4 / images_per_s / 3600 with images_per_s from the pilot's own "[throughput]" line.
#
# DECISION RULE (printed at the end; V1 settings, paired people):
#   if V4 (crop) rot_fingers < the relaxed floor (V3 rot_fingers on the same pairs; 15.1 on the findings' holdout) with
#      PVE / PA-PVE / PVE_hn not worse than +1 mm  -> RESOLUTION is the bottleneck: a crop-based second pass (full frame
#      for body and placement, person crop for the finger rotations) is the cheapest next step;
#   else -> SUPERVISION is: read the region-norm pilot (rot_fingers vs 17.3 and vs its own V3 floor, head/neck local and
#      global error, PVE <= 71.5, recall >= 89.2). A pilot that does not move the fingers either points at the 6D init
#      and the finger gradient magnitude, not at the loss weighting.
#   Nothing here shows a fix works; the numbers decide.
#SBATCH --job-name=exp4-fingers
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp4_fingers_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp4_fingers_%j.err
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
CROP_FACTOR="${CROP_FACTOR:-1.3}"
PILOT="${PILOT:-1}"                                 # 0 = inference-only experiment (steps 1, 2, 4)
REGION_NORM="${REGION_NORM:-1}"                     # 0 = pilot control arm (same schedule, the sum over bones)
ALPHA_GROT="${ALPHA_GROT:-0}"                       # exp3's global orientation term, off unless combined on purpose
GROT_BONES="${GROT_BONES:-head,neck03,wrist.L,wrist.R}"
GT_CENTER="${GT_CENTER:-root}"                     # train.py --gt_center_joint: lets a pilot stack on an exp1 checkpoint; root = today's behaviour
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"                        # 2 epochs
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"   # build_manifest.py default (slurm/diagnostics/diag_manifest.sh)
MANIFEST_TRAIN="${MANIFEST_TRAIN:-1}"               # 0 = skip the manifest train-split traces
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-1}"           # 0 = the production shuffled loader (manifest images may be drawn)
RUN="${RUN:-crop${CROP_FACTOR}_regionnorm${REGION_NORM}_grot${ALPHA_GROT}_${TAG}}"
OUT="results/experiments/exp4_fingers/${RUN}"
EXCL="${OUT}/exclude_manifest_indices.txt"
N_EPOCHS=$(( MAX_ITER / N_ITERS ))
EXPECTED_TRAIN_MIN=$(( MAX_ITER * 40 / 67 / 60 ))   # MAX_ITER * batch 4 / 6.7 images/s, in minutes

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp4_fingers run ${RUN}"
echo "    checkpoint: ${CKPT}"
echo "    crop_factor=${CROP_FACTOR} pilot=${PILOT} rotmat_region_norm=${REGION_NORM} alpha_global_rot=${ALPHA_GROT} gt_center_joint=${GT_CENTER} lr=${LR}"
echo "    pilot schedule: ${N_ITERS} iters/epoch x ${N_EPOCHS} epochs, batch 4 = $(( MAX_ITER * 4 )) images, ~${EXPECTED_TRAIN_MIN} min at 6.7 images/s"
HAVE_MANIFEST=0; [ -f "${MANIFEST}" ] && HAVE_MANIFEST=1
[ "\${HAVE_MANIFEST}" = "1" ] || echo "NOTE: ${MANIFEST} is missing; manifest steps are skipped (slurm/diagnostics/diag_manifest.sh builds it; never rebuild an existing one)"

echo "=== 1. HAND PIXEL SIZE (labels through the body model; no network)"
T0=\$(date +%s)
python scripts/experiments/measure_hand_px.py --select holdout --n 100 --crop_factor ${CROP_FACTOR} \\
    --out "${OUT}/hand_px_holdout.json" > "${OUT}/hand_px_holdout.log" 2>&1 \\
    || { echo "WARNING: measure_hand_px (holdout) failed; tail:"; tail -n 20 "${OUT}/hand_px_holdout.log"; }
echo "--- holdout (100 images), \$(( \$(date +%s) - T0 )) s"
sed -n '/=== HAND SIZE/,\$p' "${OUT}/hand_px_holdout.log" || true
if [ "\${HAVE_MANIFEST}" = "1" ]; then
  T0=\$(date +%s)
  python scripts/experiments/measure_hand_px.py --manifest "${MANIFEST}" --manifest_split both --crop_factor ${CROP_FACTOR} \\
      --out "${OUT}/hand_px_manifest.json" > "${OUT}/hand_px_manifest.log" 2>&1 \\
      || { echo "WARNING: measure_hand_px (manifest) failed; tail:"; tail -n 20 "${OUT}/hand_px_manifest.log"; }
  echo "--- manifest (train + holdout splits), \$(( \$(date +%s) - T0 )) s"
  sed -n '/=== HAND SIZE/,\$p' "${OUT}/hand_px_manifest.log" || true
fi

trace () {   # \$1 = checkpoint, \$2 = output dir, \$3... = selection / variant flags; prints what a reader needs
  local ck="\$1" od="\$2"; shift 2
  local t0=\$(date +%s)
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${ck}" --out "\${od}" "\$@" > "\${od}.log" 2>&1 \\
      || { echo "WARNING: trace \${od} failed; tail:"; tail -n 30 "\${od}.log"; return 0; }
  local dt=\$(( \$(date +%s) - t0 ))
  local nimg
  nimg=\$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_images'])" "\${od}/summary.json" 2>/dev/null || echo 0)
  echo "throughput: \${nimg} images in \${dt} s = \$(python -c "print('%.2f' % (\${nimg} / max(\${dt}, 1)))") images/s (every variant, GT body-model passes and panels included)"
  grep -E "^\[select\]|^GT people" "\${od}.log" || true
  sed -n '/=== PER-VARIANT METRICS/,/^\$/p;/=== ROTATION ERROR BY BONE GROUP/,/^\$/p;/=== GLOBAL (camera-frame)/,/^  read/p;/=== V4_person_crop vs/,/^wrote /p' "\${od}.log" || true
}

echo "=== 2. PERSON-CROP TRACE on the production checkpoint (inference only; does magnification alone fix the fingers?)"
echo "=== 2a. FIXED HOLDOUT (100 images, --select holdout), V0-V3 + V4_person_crop x${CROP_FACTOR}"
trace "${CKPT}" "${OUT}/crop_trace_holdout" --select holdout --n 100 --person_crop 1 --crop_factor ${CROP_FACTOR}
if [ "\${HAVE_MANIFEST}" = "1" ]; then
  echo "=== 2b. MANIFEST HOLDOUT SPLIT"
  trace "${CKPT}" "${OUT}/crop_trace_manifest_holdout" --manifest "${MANIFEST}" --manifest_split holdout --n_vis 20 --person_crop 1 --crop_factor ${CROP_FACTOR}
  if [ "${MANIFEST_TRAIN}" = "1" ]; then
    echo "=== 2c. MANIFEST TRAIN SPLIT (in-sample for the base model; the 'hands' people are frame-filling and truncated)"
    trace "${CKPT}" "${OUT}/crop_trace_manifest_train" --manifest "${MANIFEST}" --manifest_split train --n_vis 40 --person_crop 1 --crop_factor ${CROP_FACTOR}
  fi
fi

if [ "${PILOT}" = "1" ]; then
  echo "=== 3. PILOT FINE-TUNE: production recipe + --rotmat_region_norm ${REGION_NORM} --alpha_global_rot ${ALPHA_GROT}"
  EXCL_ARG=""
  if [ "${EXCLUDE_MANIFEST}" = "1" ] && [ "\${HAVE_MANIFEST}" = "1" ]; then
    python - <<'PY'
import json, os
man = json.load(open(os.environ['MANIFEST_PATH']))
idx = sorted({int(e['idx']) for e in man['entries']})
with open(os.environ['EXCL_PATH'], 'w') as f:
    f.write('# AnnyOne indices of the %d diag-manifest people (%d images); never sampled by this fine-tune\n' % (len(man['entries']), len(idx)))
    f.write('\n'.join(str(i) for i in idx) + '\n')
print('wrote', os.environ['EXCL_PATH'], ':', len(idx), 'images excluded from sampling')
PY
    EXCL_ARG="--exclude_indices_file ${EXCL}"
  fi
  GROT_ARGS=""
  [ "${ALPHA_GROT}" != "0" ] && GROT_ARGS="--alpha_global_rot ${ALPHA_GROT} --global_rot_bones ${GROT_BONES}"
  T0=\$(date +%s)
  python train.py \\
      --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
      --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
      --backbone dinov2_vitl14 --img_size 672 \\
      --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
      --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
      --rotmat_region_norm ${REGION_NORM} \${GROT_ARGS} --start_2d_epoch 0 \\
      --gt_center_joint "${GT_CENTER}" \\
      \${EXCL_ARG} \\
      --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
      --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
      --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
      --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
      || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
  echo "training wall time: \$(( \$(date +%s) - T0 )) s (startup + ${N_EPOCHS} epochs + ${N_EPOCHS} holdout evaluations)"
  echo "--- loss configuration, throughput (images/s) and in-training V0 evaluation (helper bones UNMASKED, not the V1 read)"
  grep -E "\[loss\] (rotmat_region_norm|joint weights|global_rot)|\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): |NaN|WARNING|Error" "${OUT}/train.log" || true
  echo "--- rotmat term on the first and last logged lines (region norm keeps the scale: n_active x the per-bone mean)"
  grep -o 'EPOCH=.*' "${OUT}/train.log" | sed -n '1p;\$p' || true   # EPOCH= shares lines with tqdm's bar: match anywhere
  LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
  [ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
  echo "last checkpoint: \${LAST}"
  echo "=== 3a. FIXED HOLDOUT TRACE on the pilot checkpoint (V0-V3; V3 = relaxed floor of this model)"
  trace "\${LAST}" "${OUT}/pilot_trace_holdout" --select holdout --n 100
  if [ "\${HAVE_MANIFEST}" = "1" ]; then
    echo "=== 3b. MANIFEST HOLDOUT SPLIT on the pilot checkpoint"
    trace "\${LAST}" "${OUT}/pilot_trace_manifest_holdout" --manifest "${MANIFEST}" --manifest_split holdout --n_vis 20
    if [ "${MANIFEST_TRAIN}" = "1" ]; then
      echo "=== 3c. MANIFEST TRAIN SPLIT on the pilot checkpoint (excluded from the fine-tune's sampling)"
      trace "\${LAST}" "${OUT}/pilot_trace_manifest_train" --manifest "${MANIFEST}" --manifest_split train --n_vis 40
    fi
  fi
else
  echo "=== 3. PILOT SKIPPED (PILOT=0)"
fi

echo "=== 4. SUMMARY"
python - <<'PY'
import json, os
out = os.environ['OUT_DIR']
V1, V3, V4 = 'V1_eval_masked', 'V3_demo_relaxhands', 'V4_person_crop'
crop_factor = os.environ['CROP_FACTOR_VAL']
# anny_s2_shape_v5/00099 on the fixed 100-image holdout, V1 settings (docs/diagnostics_findings.md): fingers 17.3 deg
# (V3 relaxed 15.1), PVE 70.5, PVE_hn 52.3, recall 90.2, local neck 62.7 / head 89.5, global head 12.0, global wrist 22.7
BASE = dict(rot_fingers=17.3, rot_fingers_relaxed=15.1, pve=70.5, pve_headneck=52.3, recall=90.2, rot_neck=62.7,
            rot_head=89.5, rot_global_head_deg=12.0, rot_global_wrist_deg=22.7)

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def f(x, nd=1):
    return '   -' if x is None else format(x, '.' + str(nd) + 'f')

def variant(summ, v):
    if not summ or v not in summ.get('variants', {}):
        return {}
    s = summ['variants'][v]
    g = s.get('rot_err_deg_by_group', {})
    return dict(rot_fingers=g.get('fingers'), rot_wrist=g.get('wrist'), rot_neck=g.get('neck'), rot_head=g.get('head'),
                pve=s.get('pve'), pa_pve=s.get('pa_pve'), pve_headneck=s.get('pve_headneck'), recall=s.get('recall'),
                rot_global_head_deg=(s.get('rot_global_deg') or {}).get('head'),
                rot_global_wrist_deg=(s.get('rot_global_deg') or {}).get('wrist'))

res = dict(baseline_findings=BASE, crop_factor=crop_factor, hand_px={}, crop={}, pilot={})
print()
print('=== exp4_fingers SUMMARY (V1 settings = helper bones masked; deg for rot_*, mm for pve_*) ===')
hp = load(out + '/hand_px_holdout.json')
if hp:
    sz = hp['sizes']
    keys = list(sz)
    inp = next((sz[k] for k in keys if k.startswith('network input') and 'fully inside' in k), {})
    crp = next((sz[k] for k in keys if k.startswith('person crop') and 'fully inside' in k), {})
    res['hand_px'] = dict(input=inp, crop=crp, crop_stats=hp.get('crop'), fingertips=hp.get('fingertips'))
    print('--- step 1, holdout hands fully inside the image (n %s): long side at the 672px input median %s px = %s tokens, '
          '%s%% under 2 tokens; under the x%s person crop median %s px = %s tokens, %s%% under 2 tokens (median magnification x%s)'
          % (inp.get('n'), f(inp.get('median_px')), f(inp.get('median_tokens'), 2), f(100 * inp.get('frac_lt_2tok', 0), 0), crop_factor,
             f(crp.get('median_px')), f(crp.get('median_tokens'), 2), f(100 * crp.get('frac_lt_2tok', 0), 0),
             f((hp.get('crop') or {}).get('scale_vs_input_median'), 2)))
else:
    print('--- step 1: no hand_px_holdout.json')

ct = load(out + '/crop_trace_holdout/summary.json')
pc = (ct or {}).get('paired_crop') or {}
v1c, v4c, v3c = variant(ct, V1), variant(ct, V4), variant(ct, V3)
print('--- step 2, production checkpoint, 100-image holdout: per-variant means (own cohorts) and the V4-vs-V1 PAIRED read')
print(format('metric', '<22') + format('findings V1', '>13') + format('V1 rerun', '>10') + format('V3 relaxed', '>12') + format('V4 crop', '>10'))
for k in ('rot_fingers', 'rot_wrist', 'rot_global_wrist_deg', 'rot_global_head_deg', 'pve', 'pa_pve', 'pve_headneck', 'recall'):
    print(format(k, '<22') + format(f(BASE.get(k)), '>13') + format(f(v1c.get(k)), '>10') + format(f(v3c.get(k)), '>12') + format(f(v4c.get(k)), '>10'))
ok_crop = None
if pc.get('n_pairs'):
    pv, dv = pc['per_variant'], pc['delta_vs_reference'].get(V4, {})
    floor = (pv.get(V3) or {}).get('rot_fingers')
    print('paired on %d people matched under both V1 and V4 (V4 recall %s%%, %s crops without a detection):'
          % (pc['n_pairs'], f(pc.get('recall')), pc.get('crops_without_detection')))
    for k in ('rot_fingers', 'rot_wrist', 'rot_global_wrist_deg', 'rot_global_head_deg', 'pve', 'pa_pve', 'pve_headneck'):
        print('  %-22s V1 %8s   V4 %8s   d %8s   V3 relaxed %8s' % (k, f(pv[V1].get(k)), f(pv[V4].get(k)), f(dv.get(k)), f((pv.get(V3) or {}).get(k))))
    for lab, b in (pc.get('by_magnification') or {}).items():
        print('  magnification %-9s n=%-4d rot_fingers %s -> %s   PVE %s -> %s' % (lab, b['n'], f(b['rot_fingers_ref']), f(b['rot_fingers_crop']), f(b['pve_ref']), f(b['pve_crop'])))
    print('  EXCLUDED for V4: ' + pc.get('excluded', ''))
    fv4, fv1 = pv[V4].get('rot_fingers'), pv[V1].get('rot_fingers')
    body_ok = all(dv.get(k) is not None and dv[k] <= 1.0 for k in ('pve', 'pa_pve', 'pve_headneck'))
    if fv4 is not None and fv1 is not None:
        floor_v = floor if floor is not None else BASE['rot_fingers_relaxed']
        if fv4 < floor_v and body_ok:
            ok_crop = 'resolution'
        elif fv4 < fv1 - 0.5:
            ok_crop = 'partial'
        else:
            ok_crop = 'supervision'
    res['crop'] = dict(paired=pc, verdict=ok_crop, body_not_worse=body_ok)
else:
    print('  no V4-vs-V1 paired block (trace failed or no crop was matched)')
for sub, title in (('crop_trace_manifest_holdout', 'manifest holdout'), ('crop_trace_manifest_train', 'manifest train')):
    s = load(out + '/' + sub + '/summary.json')
    if s:
        pt = s.get('per_tag') or {}
        h1, h4 = (pt.get(V1) or {}).get('hands') or {}, (pt.get(V4) or {}).get('hands') or {}
        p = (s.get('paired_crop') or {})
        print('  %s: paired n=%s rot_fingers V1 %s -> V4 %s (d %s); hands-category rot_fingers V1 %s (n %s) -> V4 %s (n %s)'
              % (title, p.get('n_pairs'), f((p.get('per_variant') or {}).get(V1, {}).get('rot_fingers')),
                 f((p.get('per_variant') or {}).get(V4, {}).get('rot_fingers')), f((p.get('delta_vs_reference') or {}).get(V4, {}).get('rot_fingers')),
                 f(h1.get('rot_fingers')), h1.get('n'), f(h4.get('rot_fingers')), h4.get('n')))
        res['crop'][sub] = dict(paired=p, hands_tag=dict(V1=h1, V4=h4))

pt_ = load(out + '/pilot_trace_holdout/summary.json')
ok_pilot = None
if pt_:
    p1, p3 = variant(pt_, V1), variant(pt_, V3)
    print('--- step 3, region-norm pilot (rotmat_region_norm %s, alpha_global_rot %s), 100-image holdout, V1 (own cohort):'
          % (os.environ['REGION_NORM_VAL'], os.environ['ALPHA_GROT_VAL']))
    print(format('metric', '<22') + format('baseline V1', '>13') + format('pilot V1', '>10') + format('pilot V3', '>10'))
    for k in ('rot_fingers', 'rot_wrist', 'rot_neck', 'rot_head', 'rot_global_head_deg', 'rot_global_wrist_deg', 'pve', 'pve_headneck', 'recall'):
        print(format(k, '<22') + format(f(BASE.get(k)), '>13') + format(f(p1.get(k)), '>10') + format(f(p3.get(k)), '>10'))
    res['pilot'] = dict(V1=p1, V3=p3)
    fg, pve, rec = p1.get('rot_fingers'), p1.get('pve'), p1.get('recall')
    if None not in (fg, pve, rec):
        kept = pve <= BASE['pve'] + 1.0 and rec >= BASE['recall'] - 1.0
        floor_p = p3.get('rot_fingers') if p3.get('rot_fingers') is not None else BASE['rot_fingers_relaxed']
        ok_pilot = dict(fingers_better_than_baseline=fg < BASE['rot_fingers'], fingers_below_relaxed_floor=fg < floor_p,
                        body_kept=kept, head_local_better=(p1.get('rot_head') or 1e9) < BASE['rot_head'],
                        head_global_better=(p1.get('rot_global_head_deg') or 1e9) < BASE['rot_global_head_deg'])
        print('pilot read: fingers %s vs 17.3 baseline / %s relaxed floor; local head %s vs 89.5, global head %s vs 12.0; PVE <= 71.5 and recall >= 89.2: %s'
              % (f(fg), f(floor_p), f(p1.get('rot_head')), f(p1.get('rot_global_head_deg')), 'yes' if kept else 'NO'))
        res['pilot']['criteria'] = ok_pilot
else:
    print('--- step 3: no pilot trace (PILOT=0 or the trace failed)')

print()
if ok_crop == 'resolution':
    print('VERDICT (step 2): the person crop brings the finger error BELOW the relaxed floor with the body unchanged ->')
    print('  RESOLUTION is the bottleneck; a crop-based second pass (full frame for body/placement, person crop for the')
    print('  finger rotations transplanted into the full-frame prediction) is the cheapest next step.')
elif ok_crop == 'partial':
    print('VERDICT (step 2): magnification helps the fingers but they stay above the relaxed floor (or the body regressed):')
    print('  resolution is PART of the problem; the predicted fingers are still worse than the rest pose, so supervision')
    print('  (and the 6D init) remains in the way. Read the pilot.')
elif ok_crop == 'supervision':
    print('VERDICT (step 2): magnification alone does not fix the fingers -> SUPERVISION is the bottleneck;')
    print('  the region-norm pilot result (step 3) decides whether the loss weighting is part of it.')
else:
    print('VERDICT (step 2): INCOMPLETE (missing numbers; see the trace logs)')
if ok_pilot is not None:
    if ok_pilot['fingers_better_than_baseline'] and ok_pilot['body_kept']:
        print('VERDICT (step 3): the region-norm pilot moved the fingers below 17.3 with the body kept%s -> worth the 10-epoch run.'
              % (' and below the relaxed floor' if ok_pilot['fingers_below_relaxed_floor'] else ' (still above the relaxed floor)'))
    else:
        print('VERDICT (step 3): the region-norm pilot did not move the fingers (or the body regressed): the loss WEIGHTING is not')
        print('  what limits them; next candidates are the degenerate 6D init (model.py init_body_pose vs utils.rot6d_to_rotmat)')
        print('  and the finger gradient magnitude, each as its own experiment.')
    print('  head/neck read of the same pilot: local head %s (89.5), global head %s (12.0): region norm is mainly a x6 head-neck re-weighting.'
          % ('better' if ok_pilot['head_local_better'] else 'not better', 'better' if ok_pilot['head_global_better'] else 'not better'))
print('All of this is a 2-epoch pilot / single-checkpoint inference read on 100 holdout images; it justifies or vetoes the next')
print('step and shows nothing else. Compare with REGION_NORM=0 (control arm) before attributing a pilot change to the term.')
json.dump(res, open(out + '/summary.json', 'w'), indent=2)
print('wrote ' + out + '/summary.json')
PY
echo "DECISION RULE: if V4 (crop) rot_fingers < the relaxed floor (V3 on the same pairs; 15.1 on the findings' holdout) with"
echo "               PVE / PA-PVE / PVE_hn not worse than +1 mm, RESOLUTION is the bottleneck and a crop-based second pass is the"
echo "               cheapest next step; otherwise SUPERVISION is, and the region-norm pilot (rot_fingers vs 17.3 / its V3 floor,"
echo "               head local/global error, PVE <= 71.5, recall >= 89.2) decides whether the loss weighting is part of it."
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' MANIFEST_PATH='${MANIFEST}' EXCL_PATH='${EXCL}' CROP_FACTOR_VAL='${CROP_FACTOR}' REGION_NORM_VAL='${REGION_NORM}' ALPHA_GROT_VAL='${ALPHA_GROT}'; $COMMAND"
