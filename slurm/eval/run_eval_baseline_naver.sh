#!/bin/bash
# run_eval_baseline_naver.sh: the BASELINE number for requirement 1b. Evaluates NAVER's own Multi-HMR-Anny checkpoint
# (models/multiHMR/multiHMR_672_L_anny.pt) on the AnnyOne test split (last 100 images) and every 3DPW-test image of
# the v23 labels, through the SAME train.py evaluate() as every model of ours (slurm/eval/run_eval_3dpw_test.sh,
# identical flags), so its PVE / PA-PVE / MPJPE / recall sit in the same table as the final model and the exp8 student.
#
# WHY A CODE CHANGE WAS NEEDED (verified in train.py / demo.py / slurm/history/train_step1_pretrained_bb.sh):
#   multiHMR_672_L_anny.pt was trained with the ORIGINAL repo class Multi_HMR (multi_hmr_anny/multi_hmr.py: keys
#   encoder.backbone.*, decoder.*, mlp_pose.*), not with model.py's Model. train.py --pretrained_remap 1 only renames
#   encoder.backbone.* -> backbone.encoder.* (the ViT); every head tensor (decoder / mlp_pose / mlp_shape / mlp_dist /
#   encoder.mlp_det) has no counterpart in Model and is DROPPED, so "--pretrained_remap 1" evaluates NAVER's ViT under
#   RANDOM heads: meaningless numbers (train_step1 used exactly that to start fresh heads, never to evaluate). So
#   train.py got --model_class multi_hmr_original (default model: nothing changes for any other run; --eval_only 1
#   only): it builds Multi_HMR from the checkpoint's own saved args with xat_dim 1024 (as demo.py's load_model does for
#   original-repo checkpoints), loads ALL its weights (refuses below 95%), and wraps it so evaluate() gets an empty list
#   when nothing is detected (the class returns ({}, []) then). run_eval_3dpw_test.sh passes it via EVAL_MODEL_CLASS.
#   Nothing else in evaluate() needs changing: the persons carry j2d / v3d / j3d / transl_pelvis like Model's.
#
# WHAT THE NUMBERS ARE (read before comparing):
#   - same evaluate(), same data, same GT intrinsics K (passed to the model; Multi_HMR's own FOV head is then unused),
#     det_thresh 0.2, NMS 3, fp32, PVE / PA-PVE pelvis-centred, MPJPE on the 163 Anny joints (evaluate()'s generic
#     branch for AnnyOne and 3DPW-Anny), recall from greedy 2D matching.
#   - Multi_HMR forces the helper bones to identity by design (its useful_rotmat), our Model does not in evaluate()
#     (V0). It has no bone fix and was never trained on 3DPW.
#   - [Unknown] whether NAVER's AnnyOne training split contains our last-100 test images: if it does, its AnnyOne
#     number is in-sample (optimistic). The 3DPW-test number has no such issue (no 3DPW in its training data, per
#     slurm/history/train_step1_pretrained_bb.sh: "annyone, bedlam, coco, mpii").
#   - 3DPW numbers are against our converted v23 labels only (fit floor ~14-16 mm); compare only with models evaluated
#     on the same labels (log line "labels: ...").
#
# USAGE (repo root on the cluster, login node):
#   CHECK_ONLY=1 bash slurm/eval/run_eval_baseline_naver.sh   # checks only, submits nothing
#   bash slurm/eval/run_eval_baseline_naver.sh                # checks, then one sbatch (array task 0, --time 00:40:00)
#   DRY_RUN=1 bash slurm/eval/run_eval_baseline_naver.sh      # checks, print the sbatch command
#   EVAL_3DPW_N=50 bash slurm/eval/run_eval_baseline_naver.sh # quick test on 50 3DPW-test images (run name _n50)
# ENV: CKPT (models/multiHMR/multiHMR_672_L_anny.pt), EVAL_LABELS (threedpw_anny_v23/labels), LABEL
#   (baseline_naver_multiHMR_672_L_anny), EVAL_3DPW_N (0 = all), TIME (00:40:00; the v23 test took 13-14 min for our
#   ViT-L Model [Certain, 3612073-5]; Multi_HMR's decoder is larger [Guessing] up to 2x), PARTITIONS (the eval script's
#   sm<=86 list), CHECK_ONLY, DRY_RUN.
# OUTPUT: logs/eval-test3dpw-<jobid>_0.out. Read:
#   grep -E "^task |^labels: |^bone fix: |\[model_class\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): " \
#       logs/eval-test3dpw-<jobid>_0.out
#   The "[model_class] multi_hmr_original: ... loaded N/N tensors (100.0%)" line must be there; "annyone-test-1" is the
#   AnnyOne block, "threedpw_anny-test-..." the 3DPW one.
set -euo pipefail

REPO="${REPO:-/netscratch/najib/multi-hmr}"
SELF="slurm/eval/run_eval_baseline_naver.sh"
EVAL_SCRIPT="slurm/eval/run_eval_3dpw_test.sh"
CKPT="${CKPT:-${REPO}/models/multiHMR/multiHMR_672_L_anny.pt}"
EVAL_LABELS="${EVAL_LABELS:-${REPO}/threedpw_anny_v23/labels}"
EVAL_LABELS="${EVAL_LABELS%/}"
LABEL="${LABEL:-baseline_naver_multiHMR_672_L_anny}"
EVAL_3DPW_N="${EVAL_3DPW_N:-0}"
TIME="${TIME:-00:40:00}"
PARTITIONS="${PARTITIONS:-A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "=== baseline evaluation: NAVER Multi-HMR-Anny ${CKPT}"
echo "    labels ${EVAL_LABELS}; AnnyOne last 100; 3DPW-test images ${EVAL_3DPW_N} (0 = all); run label ${LABEL}"
cd "${REPO}" 2>/dev/null || { echo "ERROR: REPO=${REPO} not found"; exit 1; }
fail=0
[ -f "${CKPT}" ] || { echo "REFUSED: checkpoint not found: ${CKPT}"; fail=1; }
[ -f "${EVAL_SCRIPT}" ] || { echo "REFUSED: ${EVAL_SCRIPT} not found (run from the repo root after git pull)"; fail=1; }
if [ -f "${EVAL_SCRIPT}" ] && ! grep -q EVAL_MODEL_CLASS "${EVAL_SCRIPT}"; then
  echo "REFUSED: ${EVAL_SCRIPT} has no EVAL_MODEL_CLASS override (old copy): git pull"; fail=1
fi
grep -q -- "--model_class" train.py || { echo "REFUSED: train.py has no --model_class (old copy): git pull"; fail=1; }
case "${EVAL_LABELS} $(readlink -f "${EVAL_LABELS}" 2>/dev/null || true)" in
  *threedpw_anny_full*) echo "REFUSED: EVAL_LABELS=${EVAL_LABELS} is under threedpw_anny_full (old labels); use v23"; fail=1 ;;
esac
[ -s "${EVAL_LABELS}/index_test.txt" ] || { echo "REFUSED: ${EVAL_LABELS}/index_test.txt is missing or empty"; fail=1; }
[[ "${EVAL_3DPW_N}" =~ ^[0-9]+$ ]] || { echo "REFUSED: EVAL_3DPW_N=${EVAL_3DPW_N} (a non-negative integer)"; fail=1; }
case ",${PARTITIONS}," in
  *V100*|*B200*|*RTXB6000*|*H100*|*H200*) echo "REFUSED: PARTITIONS=${PARTITIONS}: torch 2.0.1+cu117 needs sm 8.0-8.6 GPUs"; fail=1 ;;
esac
# optional key check (needs torch on this host: conda activate multihmr); the job itself refuses below 95% loaded
if [ -f "${CKPT}" ] && command -v python >/dev/null 2>&1 && python -c 'import torch' >/dev/null 2>&1; then
  python - "${CKPT}" <<'PY' || fail=1
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = ck.get('model_state_dict', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
n_orig = sum(1 for k in sd if k.startswith(('encoder.backbone.', 'decoder.', 'dec_to_token.', 'mlp_pose.')))
n_model = sum(1 for k in sd if k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')))
a = ck.get('args') if isinstance(ck, dict) else None
kw = dict(vars(a)) if hasattr(a, '__dict__') else dict(a or {})
print('[check] keys: %d original-class (encoder.backbone./decoder./mlp_pose.), %d model.py-class; saved args: %s; '
      'backbone=%s img_size=%s' % (n_orig, n_model, 'yes' if kw else 'NO', kw.get('backbone'), kw.get('img_size')))
if n_orig <= n_model or not kw:
    print('REFUSED: not an original-repo Multi_HMR checkpoint with saved args; evaluate it with run_eval_3dpw_test.sh')
    sys.exit(1)
PY
else
  echo "    (no python with torch here: key check skipped; the job prints the '[model_class] ... loaded' line and refuses"
  echo "     below 95%)"
fi
if command -v squeue >/dev/null 2>&1; then
  Q="$(squeue -h -u "${USER:-$(id -un)}" -n eval-baseline-naver -o '%i %T' 2>/dev/null | tr '\n' ' ' || true)"
  if [ -n "${Q// /}" ]; then
    echo "REFUSED: a job named eval-baseline-naver is already queued or running (${Q% }): scancel it or wait"; fail=1
  fi
fi
[ "${fail}" = 0 ] || { echo "NOT SUBMITTED"; exit 1; }
[ "${CHECK_ONLY}" = 1 ] && { echo "CHECK_ONLY=1: every check passed; nothing submitted"; exit 0; }
# the overrides travel as environment variables (sbatch exports the environment), as in exp7's header commands;
# EVAL_BONE_FIX is set empty so a value left in this shell cannot reach the job
CMD=(env EVAL_MODEL_CLASS=multi_hmr_original EVAL_LABELS="${EVAL_LABELS}" EVAL_3DPW_N="${EVAL_3DPW_N}" EVAL_BONE_FIX=
     MODEL_ENTRY="${CKPT}|0|dinov2_vitl14|${LABEL}"
     sbatch --array=0 --job-name=eval-baseline-naver --time="${TIME}" --partition="${PARTITIONS}" --exclude=gifu,glendale
     "${EVAL_SCRIPT}")
echo "--- submit: ${CMD[*]}"
[ "${DRY_RUN}" = 1 ] && { echo "DRY_RUN=1: nothing submitted"; exit 0; }
"${CMD[@]}"
echo "log: ${REPO}/logs/eval-test3dpw-<jobid>_0.out (read with the grep in the header of ${SELF})"
