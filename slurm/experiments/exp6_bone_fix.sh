#!/bin/bash
# exp6_bone_fix: does a CONSTANT output-level correction of the flipped bones (R_b <- R_b @ C_b, model.py / bone_fix.py)
# remove them on unseen images without training, and does fine-tuning with it in place keep them removed?
#
# WHY (docs/experiments_plan.md "exp6 bone fix"; measured by slurm/diagnostics/diag_neck_bones.sh, jobs 3573518/3573519):
#   v5 lineage (frozen_bb_v2 00299 .. v5 00099, v9 00149, long_pc_dist5, both exp5 pilots): neck02 FLIP_TWIST (median
#   175.6 deg, relSpr 5.3) + head FLIP_TWIST/FLIP_SWING (177.9, relSpr ~15) as a compensated pair (global head 12 deg is
#   right), neck-skin girth 0.46 of GT, PVE_neck 27-28 mm (v10: 1.00, 10.3); metacarpal1.L FLIP_SWING (constant 180,
#   relSpr 0, child error 185-200 mm vs joint 90-106) in the 6 checkpoints traced by diag_neck_bones (the exp5 pilots did
#   not dump metacarpals). v10 00290: neck fine; constant half-turns on metacarpal2.L/2.R/3.R (GT identity, relSpr 0-0.3,
#   global 167-170; child error 182-185 mm for metacarpal2.L/3.R (FLIP_SWING) and 136 mm for metacarpal2.R (FLIP_TWIST),
#   vs joint 95-120); half-turns on shoulder01.L (relSpr 21.3) and upperarm02.L (relSpr 2.4) that form a
#   compensated CHAIN with upperarm01.L between them (SPREAD 58 deg = ~2x its GT angle, glob 169 -> 18 at upperarm02.L),
#   the same pattern as v5's neck02 -> neck03 -> head.
#   exp5 (2 epochs, lr 1e-5): the geodesic loss moved neck02/head < 1 deg, the L1 control 0: fine-tuning does not
#   un-flip [Likely: neck02 and head must rotate back together; a partial move of one raises the loss].
#   A flip with a small relSpr is a (nearly) constant output offset, so one constant per bone should undo it at zero
#   training cost. Predicted from the measured relSpr [Likely]: neck02 -> ~5 deg, metacarpals -> ~0, head -> ~15 deg
#   (its half-turn compensates neck02 per person, which no constant undoes), global head up. On v10, BONES=auto would
#   correct shoulder01.L + upperarm02.L but not upperarm01.L between them, which leaves upperarm01.L's conjugated
#   rotation uncompensated: left elbow/forearm/hand off by ~upperarm01.L's local error (synthetic: 0 -> 56 deg global,
#   elbow ~260 mm), and shoulder01.L itself stays ~20 deg (relSpr 21.3). So v10 runs with the three metacarpals only.
#   compute_bone_fix.py prints the predicted local and GLOBAL errors on the estimation images before any holdout run,
#   and exits 4 (GLOBAL GUARD) when the fix raises any dumped bone's predicted global median by > 5 deg (likely for
#   v5's head, the rise above; the v5 run has TRAIN=1 and continues).
#
# WHAT RUNS (one GPU; ~30 min with TRAIN=1, ~10 min with TRAIN=0):
#   a. diag_holdout_trace.py --select train_sample (N_EST seeded random TRAINING images; never the last-100 holdout or
#      the diag manifest images) --bone_fix none --dump_bones 1 --dump_bone_names all (every active bone), V1 only
#      -> ${OUT}/est_trace; scripts/diagnostics/compute_bone_fix.py (--bones ${BONES}, V1_eval_masked) -> ${OUT}/bone_fix.json
#      (exit 3 = no bone qualifies: NO TARGET, the job stops there; exit 4 = GLOBAL GUARD, a compensated chain: the fix
#      raises a dumped bone's predicted global median by > 5 deg; then ON_GLOBAL_RISE decides: continue = run anyway
#      (the zero-training column is expected to fail there; default with TRAIN=1, since the fine-tune can still act),
#      stop = end the job with the safe subset printed (default with TRAIN=0), safe = re-estimate with the SAFE SUBSET
#      compute_bone_fix.py printed (the selected bones minus every corrected bone on a risen bone's chain))
#   b. ZERO-TRAINING TEST: holdout trace of CKPT with --bone_fix bone_fix.json (100 images, V0-V3, n_vis ${N_VIS})
#      -> ${OUT}/holdout_fix; the same ${N_VIS} images WITHOUT the fix (V1/V2 panels only) -> ${OUT}/holdout_nofix_vis;
#      ${OUT}/compare/<idx>.jpg = the two panel rows stacked (top without, bottom with). No baseline dump at
#      BASELINE_DIR -> a full no-fix holdout trace -> ${OUT}/holdout_baseline is used instead.
#   c. TRAIN=1: train.py from CKPT with the exp5 control recipe (L1, alpha_rotmat 0.1, alpha_global_rot 0, 2 epochs x
#      1000 iters, batch 4, lr 1e-5, manifest images excluded) plus --bone_fix bone_fix.json (saved in the checkpoint
#      args, re-applied on load) -> ${OUT}/train; holdout trace of its last checkpoint (its own fix) -> ${OUT}/holdout_ft
#   d. scripts/experiments/exp6_bone_fix_summary.py: per-bone verdict/median (neck chain, corrected and baseline-flagged
#      bones), girth, PVE_neck (neck01-centred), PVE, PA-PVE, PVE_hn, recall, global head, finger rotation for
#      baseline | zero-training | fine-tuned | control (the exp5 L1 run of the same TAG), plus the two verdicts.
#      SUMMARY_ONLY=1 runs only this step on a finished run dir (no srun, no GPU).
#
# PASS CRITERIA (V1, 100-image holdout, matched people):
#   zero-training vs the baseline: every corrected bone median < 15 deg (v5: neck02, head, metacarpal1.L; v10: the
#     three metacarpals), neck-skin girth >= 0.9, global head <= baseline + 3 deg, PVE <= baseline + 1 mm.
#   fine-tuned vs the control (else the baseline): corrected bones median < 15 deg, girth >= 0.9, PVE <= ref + 8 mm,
#     recall >= ref - 2, global head <= ref + 3 deg (exp5 guards: the spread of six 2-epoch v5-recipe arms).
# HOW TO READ: zero-training PASS = the flips are a fixed output offset of the head and the constant removes them on
#   unseen images; then the fine-tuned column says whether training keeps them removed (and lets the head's residual
#   shrink). A corrected bone still > 15 deg with a baseline relSpr ~15 (v5 head) = a per-person compensation that no
#   constant undoes; global head up = that residual shows in where the head points; PVE up with a GLOBAL GUARD warning
#   in step a = the compensated chain the guard predicted (continue), not a new effect. CAVEAT: the correction is ONE
#   constant per bone estimated on N_EST training images; it does not say why the head converged to the flip.
#
# USAGE (repo root on the cluster):
#   sbatch slurm/experiments/exp6_bone_fix.sh                                           # v5 00099, TRAIN=1
#   CKPT=logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt TRAIN=0 BONES=metacarpal2.L,metacarpal2.R,metacarpal3.R \
#       sbatch --time=00:30:00 slurm/experiments/exp6_bone_fix.sh                       # v10's uncompensated flips, no training
#   BONES=neck02,head sbatch slurm/experiments/exp6_bone_fix.sh                         # explicit bones (RUN gets a suffix)
#   SUMMARY_ONLY=1 RUN=<run name> bash slurm/experiments/exp6_bone_fix.sh               # the comparison alone; TAG,
#       baseline and control come from <run>/bone_fix.json (source.ckpt), e.g. for the v10 run:
#   SUMMARY_ONLY=1 RUN=anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R bash slurm/experiments/exp6_bone_fix.sh
# Env: CKPT, TAG (default from the path, as diag_neck_bones.sh), N_EST=200, SEED=0, TRAIN=1, BONES=auto, N_VIS=12,
#   ON_GLOBAL_RISE (continue|stop|safe; default continue with TRAIN=1, stop with TRAIN=0),
#   CONTROL_RUN (default results/experiments/exp5_neck_fix/pilot_l1_a0.1_${TAG}: the exp5 L1 control of v5 00099),
#   BASELINE_DIR (default results/diagnostics/neck_bones/${TAG}), LR=1e-5, N_ITERS=1000, MAX_ITER=2000, RUN.
#SBATCH --job-name=exp6-bone-fix
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp6_bone_fix_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp6_bone_fix_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

CKPT_SET="${CKPT:-}${TAG:-}"                          # non-empty: the user named the checkpoint (SUMMARY_ONLY check)
CKPT="${CKPT:-/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
# TAG as in slurm/diagnostics/diag_neck_bones.sh / exp5, so BASELINE_DIR and CONTROL_RUN find their runs
tag_of () {
  local ck="$1" rd rn
  if [ "$(basename "$(dirname "$ck")")" = checkpoints ]; then
    rd="$(dirname "$(dirname "$ck")")"; rn="$(basename "$rd")"
    [ "$rn" = train ] && rn="$(basename "$(dirname "$rd")")"
    echo "${rn}_$(basename "$ck" .pt)"
  else
    basename "$ck" .pt
  fi
}
TAG="${TAG:-$(tag_of "$CKPT")}"
N_EST="${N_EST:-200}"
SEED="${SEED:-0}"
TRAIN="${TRAIN:-1}"
BONES="${BONES:-auto}"
if [ "${TRAIN}" = "1" ]; then ON_GLOBAL_RISE="${ON_GLOBAL_RISE:-continue}"; else ON_GLOBAL_RISE="${ON_GLOBAL_RISE:-stop}"; fi
case "${ON_GLOBAL_RISE}" in continue|stop|safe) ;; *) echo "ERROR: ON_GLOBAL_RISE=${ON_GLOBAL_RISE} (continue|stop|safe)"; exit 1;; esac
N_VIS="${N_VIS:-12}"
DUMP_NAMES="${DUMP_NAMES:-all}"                       # bones dumped by every trace ('all' = every active bone)
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"                         # 2 epochs, as the exp5 control
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SKIP_EST="${SKIP_EST:-0}"                            # 1: reuse ${OUT}/bone_fix.json from an earlier run of the same RUN
BSUF=""; [ "${BONES}" != auto ] && BSUF="_$(echo "${BONES}" | tr ',.' '-_' | cut -c1-60)"
RUN="${RUN:-${TAG}_n${N_EST}_s${SEED}${BSUF}}"
OUT="results/experiments/exp6_bone_fix/${RUN}"
if [ "${SUMMARY_ONLY}" = "1" ]; then
  # the run's own checkpoint (bone_fix.json source.ckpt) decides TAG, so baseline and control are that checkpoint's,
  # not the default CKPT's (v5); without it, RUN must at least be a run of TAG
  JCK="$(python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("source") or {}).get("ckpt") or "")' \
         "${OUT}/bone_fix.json" 2>/dev/null || true)"
  if [ -z "${CKPT_SET}" ]; then
    if [ -n "${JCK}" ]; then
      CKPT="${JCK}"; TAG="$(tag_of "${JCK}")"
      echo "SUMMARY_ONLY: checkpoint from ${OUT}/bone_fix.json: ${JCK} -> TAG ${TAG}"
    else
      case "${RUN}" in
        "${TAG}"_*) ;;
        *) echo "ERROR: RUN=${RUN} is not a run of TAG=${TAG} and ${OUT}/bone_fix.json names no checkpoint;" \
                "pass CKPT=<its checkpoint> or TAG=<tag>"; exit 1;;
      esac
    fi
  elif [ -n "${JCK}" ] && [ "$(tag_of "${JCK}")" != "${TAG}" ]; then
    echo "WARNING: ${OUT}/bone_fix.json was estimated on ${JCK} (TAG $(tag_of "${JCK}")), not TAG=${TAG}:" \
         "the baseline and control below belong to TAG=${TAG}"
  fi
fi
BASELINE_DIR="${BASELINE_DIR:-results/diagnostics/neck_bones/${TAG}}"
CONTROL_RUN="${CONTROL_RUN:-results/experiments/exp5_neck_fix/pilot_l1_a0.1_${TAG}}"

# step d as one shell snippet, run on the login node (SUMMARY_ONLY=1, python3) or at the end of the job (container python)
# BASE falls back to this run's own no-fix trace when no diag_neck_bones.sh dump exists for TAG
SUMMARY_CMD=$(cat <<EOF3
BASE="${BASELINE_DIR}"
{ [ -f "\${BASE}/bones_summary.json" ] && [ -f "\${BASE}/summary.json" ]; } || BASE="${OUT}/holdout_baseline"
FT_ARG=""; [ -f "${OUT}/holdout_ft/summary.json" ] && FT_ARG="--ft_dir ${OUT}/holdout_ft"
CT_ARG=""; [ -n "${CONTROL_RUN}" ] && CT_ARG="--control_run ${CONTROL_RUN}"
python3 scripts/experiments/exp6_bone_fix_summary.py --baseline_dir "\${BASE}" --fix_dir "${OUT}/holdout_fix" \
    \${FT_ARG} \${CT_ARG} --bone_fix "${OUT}/bone_fix.json" --out "${OUT}/summary.json"
EOF3
)
if [ "${SUMMARY_ONLY}" = "1" ]; then
  [ -d "${OUT}" ] || { echo "ERROR: SUMMARY_ONLY=1 but run dir ${OUT} does not exist (set RUN=<run name>)"; exit 1; }
  bash -c "set -euo pipefail; ${SUMMARY_CMD}"; exit $?
fi

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp6_bone_fix run ${RUN}"
echo "    ckpt: ${CKPT} (tag ${TAG}); estimation: ${N_EST} training images, seed ${SEED}, bones ${BONES}; TRAIN=${TRAIN};" \
     "ON_GLOBAL_RISE=${ON_GLOBAL_RISE}"
echo "    baseline: ${BASELINE_DIR}; control: ${CONTROL_RUN}"

trace () {   # \$1 = checkpoint, \$2 = output dir, rest = extra args; prints the V1 group row and the neck-skin lines
  local CK="\$1" OD="\$2"; shift 2
  local T0=\$(date +%s)
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${CK}" --out "\${OD}" "\$@" > "\${OD}.log" 2>&1 \\
      || { echo "ERROR: trace \${OD} failed; tail:"; tail -n 30 "\${OD}.log"; return 1; }
  echo "    \${OD}: done in \$(( \$(date +%s) - T0 )) s (log \${OD}.log)"
  grep -E "^\[load_model\] bone_fix|^\[bone_fix\]|^\[select\]" "\${OD}.log" || true
  grep -A 3 "=== ROTATION ERROR BY BONE GROUP" "\${OD}.log" | grep -E "^variant|^V1_eval_masked" || true
  grep -E "^neck skin V1_eval_masked" "\${OD}.log" || true
}

echo "=== a. ESTIMATION: ${N_EST} seeded random TRAINING images, no fix -> ${OUT}/est_trace"
if [ "${SKIP_EST}" = "1" ] && [ -f "${OUT}/bone_fix.json" ]; then
  echo "    SKIP_EST=1: reusing ${OUT}/bone_fix.json"
else
  rm -f "${OUT}/bone_fix_guarded.json"     # only an ON_GLOBAL_RISE=safe run of THIS estimation may leave one
  trace "${CKPT}" "${OUT}/est_trace" --select train_sample --n ${N_EST} --seed ${SEED} --exclude_manifest "${MANIFEST}" \\
      --variants V1_eval_masked --n_vis 0 --bone_fix none --dump_bones 1 --dump_bone_names "${DUMP_NAMES}"
  set +e
  python scripts/diagnostics/compute_bone_fix.py --trace_dir "${OUT}/est_trace" --variant V1_eval_masked \\
      --bones "${BONES}" --out "${OUT}/bone_fix.json"
  RC=\$?
  set -e
  if [ "\${RC}" = 3 ]; then echo "VERDICT: NO TARGET (no bone qualifies on the training sample); nothing tested"; exit 0; fi
  if [ "\${RC}" = 4 ]; then
    SAFE=\$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["global_guard"]["safe_subset"]))' \\
           "${OUT}/bone_fix.json")
    echo "WARNING: GLOBAL GUARD (compute_bone_fix.py exit 4): the fix (BONES=${BONES}) raises a predicted global median by > 5 deg"
    echo "    (a compensated chain, see the WARNING above); safe subset: \${SAFE:-NONE}; ON_GLOBAL_RISE=${ON_GLOBAL_RISE}"
    case "${ON_GLOBAL_RISE}" in
      continue)
        echo "    continuing with the full fix: the zero-training column is expected to FAIL on PVE / global head there" ;;
      safe)
        [ -n "\${SAFE}" ] || { echo "VERDICT: NO TARGET (every selected bone is on a compensated chain); nothing tested"; exit 0; }
        mv "${OUT}/bone_fix.json" "${OUT}/bone_fix_guarded.json"
        echo "    re-estimating with the safe subset (the full fix kept as ${OUT}/bone_fix_guarded.json)"
        python scripts/diagnostics/compute_bone_fix.py --trace_dir "${OUT}/est_trace" --variant V1_eval_masked \\
            --bones "\${SAFE}" --out "${OUT}/bone_fix.json" \\
            || { echo "ERROR: compute_bone_fix.py with the safe subset \${SAFE} failed"; exit 1; } ;;
      *)
        echo "VERDICT: STOPPED (ON_GLOBAL_RISE=stop): ${OUT}/bone_fix.json is kept for reading; to test it anyway, rerun"
        echo "    with SKIP_EST=1 ON_GLOBAL_RISE=continue (same RUN), or with BONES=\${SAFE:-<explicit list>}"
        exit 0 ;;
    esac
  elif [ "\${RC}" != 0 ]; then
    echo "ERROR: compute_bone_fix.py exit \${RC}"; exit 1
  fi
fi

echo "=== b. ZERO-TRAINING TEST: holdout with --bone_fix ${OUT}/bone_fix.json -> ${OUT}/holdout_fix"
trace "${CKPT}" "${OUT}/holdout_fix" --select holdout --n 100 --n_vis ${N_VIS} --bone_fix "${OUT}/bone_fix.json" \\
    --dump_bones 1 --dump_bone_names "${DUMP_NAMES}"
echo "--- the same ${N_VIS} holdout images WITHOUT the fix (panels only) -> ${OUT}/holdout_nofix_vis"
trace "${CKPT}" "${OUT}/holdout_nofix_vis" --select holdout --n ${N_VIS} --n_vis ${N_VIS} --bone_fix none \\
    --variants V1_eval_masked,V2_demo || echo "WARNING: no-fix panels failed; the comparison images are skipped"
python - <<'PY'
import glob, os
from PIL import Image, ImageDraw
out = os.environ['OUT_DIR']
os.makedirs(os.path.join(out, 'compare'), exist_ok=True)
n = 0
for p in sorted(glob.glob(os.path.join(out, 'holdout_fix', 'vis', '*.jpg'))):
    q = os.path.join(out, 'holdout_nofix_vis', 'vis', os.path.basename(p))
    if not os.path.isfile(q):
        continue
    rows = []
    for path, lab in ((q, 'WITHOUT bone fix'), (p, 'WITH bone fix (zero training)')):
        im = Image.open(path).convert('RGB')
        ImageDraw.Draw(im).text((8, im.height - 20), lab, fill=(255, 255, 0))
        rows.append(im)
    W = max(r.width for r in rows)
    can = Image.new('RGB', (W, sum(r.height for r in rows)), (0, 0, 0))
    y = 0
    for r in rows:
        can.paste(r, (0, y)); y += r.height
    can.save(os.path.join(out, 'compare', os.path.basename(p)), quality=90)
    n += 1
print('wrote %d side-by-side panels to %s/compare (top: without, bottom: with the fix)' % (n, out))
PY
if [ ! -f "${BASELINE_DIR}/bones_summary.json" ] || [ ! -f "${BASELINE_DIR}/summary.json" ]; then
  echo "--- no baseline dump at ${BASELINE_DIR}: full no-fix holdout trace -> ${OUT}/holdout_baseline"
  trace "${CKPT}" "${OUT}/holdout_baseline" --select holdout --n 100 --n_vis 0 --bone_fix none \\
      --dump_bones 1 --dump_bone_names "${DUMP_NAMES}"
fi

if [ "${TRAIN}" = "1" ]; then
  echo "=== c. FINE-TUNE WITH THE FIX: exp5 control recipe (L1, alpha_rotmat 0.1, ${MAX_ITER} iters, lr ${LR}) + --bone_fix"
  EXCL="${OUT}/exclude_manifest_indices.txt"
  EXCL_ARG=""
  if [ -f "${MANIFEST}" ]; then
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
    EXCL_ARG="--exclude_indices_file \${EXCL}"
  else
    echo "WARNING: ${MANIFEST} not found; the control run excluded its images, this one cannot"
  fi
  T0=\$(date +%s)
  python train.py \\
      --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
      --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
      --backbone dinov2_vitl14 --img_size 672 \\
      --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
      --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
      --rotmat_loss l1 --alpha_rotmat 0.1 \\
      --alpha_global_rot 0 --global_rot_bones "head,neck03,wrist.L,wrist.R" --start_2d_epoch 0 \\
      \${EXCL_ARG} --bone_fix "${OUT}/bone_fix.json" \\
      --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
      --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
      --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
      --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
      || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
  echo "training wall time: \$(( \$(date +%s) - T0 )) s"
  grep -E "^\[bone_fix\]|\[loss\] joint weights|\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall): |NaN|Error" "${OUT}/train.log" || true
  LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
  [ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
  echo "last checkpoint: \${LAST} (its args carry bone_fix_spec; the trace re-applies it with --bone_fix '')"
  trace "\${LAST}" "${OUT}/holdout_ft" --select holdout --n 100 --n_vis ${N_VIS} --dump_bones 1 \\
      --dump_bone_names "${DUMP_NAMES}"
  [ -f "${CONTROL_RUN}/holdout_trace/summary.json" ] \\
      || echo "note: no exp5 control at ${CONTROL_RUN}: the fine-tuned column is judged against the baseline (drift not separable)"
else
  echo "=== c. skipped (TRAIN=0)"
fi

echo "=== d. SUMMARY"
${SUMMARY_CMD}
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' MANIFEST_PATH='${MANIFEST}' EXCL_PATH='${OUT}/exclude_manifest_indices.txt'; $COMMAND"
