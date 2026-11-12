#!/bin/bash
# exp10_kd_ablation: the supervisor's feedback of 12 Nov on the thesis draft, "Knowledge distillation: missing important
# ablation: ViT-B without distillation. Same training data and schedule. Otherwise, difficult to establish what
# distillation actually contributes." Two arms of the exp8 ViT-B student, each one exp8 job (KD_MODE, see exp8's header):
#
#   nokd        the two distillation LOSSES off (--lambda_kd 0 --lambda_kd_out 0); everything else the reported recipe,
#               INCLUDING the teacher's heads copied into the student and frozen for the first 5 epochs and the teacher's
#               bone fix during training. Isolates what the distillation losses contribute on top of the head transfer.
#               It does NOT isolate the teacher: the heads (14.0 M of the 101.4 M parameters) are still the teacher's.
#   supervised  no teacher in training at all: no distillation loss, no head copy (projector and heads at model.py's
#               fresh initialisation), no head freeze, --bone_fix none. Isolates everything the teacher contributes
#               (losses + head initialisation). It does NOT separate the two (that is what nokd is for), and it gives
#               fresh heads only the 30-epoch schedule tuned for transferred heads (lr 3e-5, 2D terms from epoch 0).
#
# WHAT IS FIXED in both arms, through exp8 itself: the architecture (dinov2_vitb14, --head_dim 1024 = projector
#   768 -> 1024 and teacher-width heads: parameters and FLOPs identical to the Student), the backbone init (NAVER's
#   multiHMR_672_B.pt, backbone only), the data (AnnyOne without the 55 manifest images + 30% 3DPW-train, v23 labels),
#   targets and loss weights, augmentation, SEED 0, EPOCHS 30 x N_ITERS 1000 x BATCH 8, LR 3e-5 (x0.5 every 12 epochs),
#   and every post-step (the model's own bone fix estimated on 200 training images, holdout traces, model cost, the
#   3DPW-test evaluation, final_table.json). As in the exp9 sweep, EVAL_FREQ 1 and VAL_LOSS 1: the in-training
#   validation metrics (AnnyOne holdout, 300 3DPW-validation images) and the validation loss exist for every epoch.
# TEACHER NUMBERS: the teacher is not re-measured (TEACHER_REMEASURE=0): its holdout trace and 3DPW-test log are read
#   from the reported run's folder (job 3614191: holdout_teacher/, eval_3dpw_test_teacher.log; exp8 refuses unless that
#   log is this TEACHER on these labels). Saves two model evaluations (~20 min) per job. When those files are missing,
#   or with TEACHER_REUSE=0, the arm re-measures the teacher as every earlier exp8 job did.
# RUN NAMES: student_vitb_from_<teacher tag>_e30_nokd and ..._e30_supervised (exp8 refuses an ablation mode whose RUN
#   lacks _<mode>, a distilled run named like an ablation, and any folder that already holds checkpoints): the reported
#   run ..._e30 and the sweep runs ..._e30_kdT<t>_sweep are never touched. "from_<teacher tag>" in the supervised name
#   names the teacher the arm is COMPARED with (post-steps), not one it learned from.
#
# EXPECTED COST per arm: [Likely] 6-7 h of training on an L40S incl. the 30 in-training evaluations (the distilled
#   sweep arms: 7.3 h in total on an L40S, 13.3 h on an RTX A6000; these arms skip the ViT-L teacher forward of every
#   training step, so they should be somewhat faster [Guessing: 10-25%]), plus ~40 min of post-steps. --time 16:00:00.
#
# OUTPUTS: results/experiments/exp8_distill_vitb/<RUN>/ per arm (exp8's layout: settings.txt, train.log, train/,
#   bone_fix.json, holdout_student/, eval_3dpw_test_student.log, model_cost.json, final_table.json), the job logs
#   logs/exp8_distill_vitb_<jobid>.out and this script's record logs/exp10_kd_ablation_<date>.txt (arm, RUN, job id).
#   Afterwards, on the login node (CPU; both commands are printed with the real names at the end of a submission):
#     python scripts/analysis/kd_ablation_table.py --reported <...>_e30 --sweep_t4 <...>_e30_kdT4_sweep \
#         --nokd <...>_e30_nokd --supervised <...>_e30_supervised --out results/experiments/exp10_kd_ablation
#     python scripts/analysis/plot_loss_curves.py <...>_e30_nokd <...>_e30_supervised --out results/experiments/exp10_kd_ablation/loss_curves
#
# MODES (passed through to exp8 as its environment): CHECK_ONLY=1 runs exp8's guards for both arms and submits nothing;
#   DRY_RUN=1 prints each sbatch line; LOCAL_CHECK=1 works anywhere and prints every composed command; POST_ONLY=1
#   re-runs only the post-steps of an arm that finished training (exp8's POST_ONLY; --time then exp8's own).
#
# USAGE (repo root on the cluster, /netscratch/najib/multi-hmr):
#   conda activate multihmr
#   CHECK_ONLY=1 bash slurm/experiments/exp10_kd_ablation.sh      # guards for both arms, submits nothing
#   bash slurm/experiments/exp10_kd_ablation.sh                   # submits 2 jobs (exp8-<RUN> each)
#   ARMS=supervised bash slurm/experiments/exp10_kd_ablation.sh   # one arm
#   POST_ONLY=1 ARMS=nokd bash slurm/experiments/exp10_kd_ablation.sh   # finish an arm whose post-steps failed
#   LOCAL_CHECK=1 bash slurm/experiments/exp10_kd_ablation.sh     # anywhere: print what would run
# ENV (defaults): ARMS ("nokd supervised"), VAL_LOSS (1), EVAL_FREQ (1), TIME (16:00:00; not passed with POST_ONLY=1),
#   ABL_TAG ('' ; a suffix _<tag> for a repeated arm), TEACHER_REUSE (auto | 1 | 0: read the teacher's numbers from the
#   reported run; auto = when its files are there and match), TEACHER / STUDENT_BACKBONE / EPOCHS / LABELS (exp8's
#   defaults; they enter the names and the reuse check here, exp8 reads them itself), CHECK_ONLY, DRY_RUN, LOCAL_CHECK,
#   POST_ONLY (0). Every other exp8 variable set in the environment reaches exp8 unchanged. Not allowed here, because the
#   arms define them: RUN, KD_MODE, KD_T, LAMBDA_KD, LAMBDA_KD_OUT, INIT_HEADS, STUDENT_TRAIN_FIX, FREEZE_HEADS_EPOCHS.
set -euo pipefail

REPO="${REPO:-/netscratch/najib/multi-hmr}"
SELF="slurm/experiments/exp10_kd_ablation.sh"
EXP8="${EXP8:-slurm/experiments/exp8_distill_vitb.sh}"
abs () { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${REPO}/$1" ;; esac; }
posint () { [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $1=$2 (a positive integer)"; exit 1; }; }
onezero () { case "$2" in 0|1) ;; *) echo "ERROR: $1=$2 (0|1)"; exit 1 ;; esac; }
tag_of () {   # = exp8_distill_vitb.sh tag_of: <run>_<epoch> for .../<run>[/train]/checkpoints/<epoch>.pt, else the file stem
  local ck="$1" rd rn
  if [ "$(basename "$(dirname "$ck")")" = checkpoints ]; then
    rd="$(dirname "$(dirname "$ck")")"; rn="$(basename "$rd")"
    [ "$rn" = train ] && rn="$(basename "$(dirname "$rd")")"
    echo "${rn}_$(basename "$ck" .pt)"
  else
    basename "$ck" .pt
  fi
}

# ---------------------------------------------------------------- settings (the RUN names mirror exp8's rule)
ARMS_IN="${ARMS:-nokd supervised}"
VAL_LOSS="${VAL_LOSS:-1}"; onezero VAL_LOSS "${VAL_LOSS}"
EVAL_FREQ="${EVAL_FREQ:-1}"; posint EVAL_FREQ "${EVAL_FREQ}"
POST_ONLY="${POST_ONLY:-0}"; onezero POST_ONLY "${POST_ONLY}"
TIME_SET="${TIME:-}"
TIME="${TIME:-16:00:00}"
ABL_TAG="${ABL_TAG:-}"
case "${ABL_TAG}" in *[!A-Za-z0-9_]*) echo "ERROR: ABL_TAG=${ABL_TAG} (letters, digits, _)"; exit 1 ;; esac
TEACHER_REUSE="${TEACHER_REUSE:-auto}"
case "${TEACHER_REUSE}" in auto|0|1) ;; *) echo "ERROR: TEACHER_REUSE=${TEACHER_REUSE} (auto | 0 | 1)"; exit 1 ;; esac
TEACHER="$(abs "${TEACHER:-${REPO}/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt}")"
STUDENT_BACKBONE="${STUDENT_BACKBONE:-dinov2_vitb14}"
case "${STUDENT_BACKBONE}" in
  dinov2_vitb14) BB_SHORT=vitb ;;
  dinov2_vits14) BB_SHORT=vits ;;
  *) echo "ERROR: STUDENT_BACKBONE=${STUDENT_BACKBONE} (dinov2_vitb14 | dinov2_vits14)"; exit 1 ;;
esac
EPOCHS="${EPOCHS:-30}"; posint EPOCHS "${EPOCHS}"
LABELS_CHK="$(abs "${LABELS:-${REPO}/threedpw_anny_v23/labels}")"; LABELS_CHK="${LABELS_CHK%/}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_CHECK="${LOCAL_CHECK:-0}"
for v in RUN KD_MODE KD_T LAMBDA_KD LAMBDA_KD_OUT INIT_HEADS STUDENT_TRAIN_FIX FREEZE_HEADS_EPOCHS; do
  [ -z "${!v:-}" ] || { echo "ERROR: ${v}=${!v} is set; the arms define it (ARMS=\"nokd supervised\" chooses the arms, ABL_TAG a name suffix)"; exit 1; }
done
[ -z "${TEACHER_REMEASURE:-}" ] || { echo "ERROR: TEACHER_REMEASURE=${TEACHER_REMEASURE} is set; use TEACHER_REUSE=0 (re-measure) or 1 (reuse) here"; exit 1; }
ARMS=()
for a in ${ARMS_IN}; do
  case "${a}" in nokd|supervised) ;; *) echo "ERROR: ARMS=\"${ARMS_IN}\": unknown arm ${a} (nokd | supervised)"; exit 1 ;; esac
  case " ${ARMS[*]:-} " in *" ${a} "*) echo "ERROR: ARMS=\"${ARMS_IN}\": ${a} twice"; exit 1 ;; esac
  ARMS+=("${a}")
done
[ "${#ARMS[@]}" -gt 0 ] || { echo "ERROR: ARMS is empty"; exit 1; }
TEACHER_TAG="$(tag_of "${TEACHER}")"
PREFIX="student_${BB_SHORT}_from_${TEACHER_TAG}_e${EPOCHS}"
RESULTS="results/experiments/exp8_distill_vitb"
REPORTED_ABS="$(abs "${RESULTS}/${PREFIX}")"
run_of () { printf '%s\n' "${PREFIX}_${1}${ABL_TAG:+_${ABL_TAG}}"; }
describe () {
  case "$1" in
    nokd) echo "distillation losses off (lambda_kd 0, lambda_kd_out 0); teacher heads copied + frozen 5 epochs, teacher bone fix in training" ;;
    supervised) echo "no teacher in training: no distillation loss, no head copy, no head freeze, --bone_fix none" ;;
  esac
}

MODE=submit
[ "${LOCAL_CHECK}" = 1 ] && MODE=local
[ "${CHECK_ONLY}" = 1 ] && MODE=check
[ "${DRY_RUN}" = 1 ] && [ "${MODE}" = submit ] && MODE=dry

# the teacher's numbers: from the reported run (TEACHER_REMEASURE=0) when its trace and its 3DPW-test log are there and
# are this teacher on these labels (the two greps are exp8's own guard); else re-measured in each arm
T_TRACE="${TEACHER_TRACE_DIR:-${REPORTED_ABS}/holdout_teacher}"
T_LOG="${TEACHER_EVAL_LOG:-${REPORTED_ABS}/eval_3dpw_test_teacher.log}"
REUSE="${TEACHER_REUSE}"; REUSE_WHY=""
if [ "${REUSE}" = auto ]; then
  if [ "${MODE}" = local ]; then
    REUSE=1; REUSE_WHY="planned; LOCAL_CHECK does not look for the files"
  elif [ -f "${T_TRACE}/summary.json" ] && [ -f "${T_LOG}" ] \
       && grep -qF "labels: ${LABELS_CHK} " "${T_LOG}" && grep -qF "  ->  ${TEACHER}  (" "${T_LOG}"; then
    REUSE=1; REUSE_WHY="both files found; the log is this teacher on these labels"
  else
    REUSE=0; REUSE_WHY="${T_TRACE}/summary.json or ${T_LOG} missing, or not this teacher / these labels"
  fi
fi

echo "=== exp10_kd_ablation: the exp8 ViT-B student WITHOUT distillation (supervisor feedback, 12 Nov)"
echo "    fixed: exp8 defaults, teacher ${TEACHER} (tag ${TEACHER_TAG}), ${STUDENT_BACKBONE}, ${EPOCHS} epochs;"
echo "    per epoch: evaluation (EVAL_FREQ ${EVAL_FREQ}) + validation loss (VAL_LOSS ${VAL_LOSS}); --time $([ "${POST_ONLY}" = 1 ] && [ -z "${TIME_SET}" ] && echo "exp8's (POST_ONLY)" || echo "${TIME}")"
echo "    arms:"
for a in "${ARMS[@]}"; do
  echo "      ${a}  ->  ${RESULTS}/$(run_of "${a}")"
  echo "          $(describe "${a}")"
done
if [ "${REUSE}" = 1 ]; then
  echo "    teacher numbers: reused (TEACHER_REMEASURE=0) from ${T_TRACE} and ${T_LOG}${REUSE_WHY:+ (${REUSE_WHY})}"
else
  echo "    teacher numbers: re-measured in each arm (TEACHER_REMEASURE=1)${REUSE_WHY:+: ${REUSE_WHY}}"
fi
echo "    not touched: the reported run ${RESULTS}/${PREFIX} (job 3614191) and the sweep runs ${PREFIX}_kdT*_sweep"
[ "${POST_ONLY}" = 1 ] && echo "    POST_ONLY=1: nothing is trained; exp8 re-runs the post-steps of each arm's finished training"
if [ "${MODE}" != local ]; then
  [ -f "${EXP8}" ] || { echo "ERROR: run this from the repo root (${EXP8} not found in ${PWD})"; exit 1; }
  [ -n "${SLURM_JOB_ID:-}" ] && [ "${MODE}" = submit ] \
    && { echo "REFUSED: SLURM_JOB_ID=${SLURM_JOB_ID} is set (an salloc / srun shell?): submit from a login shell"; exit 1; }
else
  EXP8="$(abs "${EXP8}")"
  [ -f "${EXP8}" ] || EXP8="$(dirname "$0")/exp8_distill_vitb.sh"
  [ -f "${EXP8}" ] || { echo "ERROR: exp8_distill_vitb.sh not found next to ${SELF}"; exit 1; }
fi
RECORD=""
if [ "${MODE}" = submit ]; then
  mkdir -p "${REPO}/logs"
  RECORD="${REPO}/logs/exp10_kd_ablation_$(date +%Y%m%d_%H%M%S).txt"
  { echo "# exp10_kd_ablation $(date -Is) git $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
    echo "# ARMS=\"${ARMS[*]}\" VAL_LOSS=${VAL_LOSS} EVAL_FREQ=${EVAL_FREQ} TIME=${TIME} TEACHER=${TEACHER} EPOCHS=${EPOCHS} TEACHER_REMEASURE=$(( 1 - REUSE )) POST_ONLY=${POST_ONLY}"
    echo "# arm  RUN  job_id"; } > "${RECORD}"
fi

# ---------------------------------------------------------------- one exp8 call per arm
JOBS=(); FAILED=()
for a in "${ARMS[@]}"; do
  RUN_A="$(run_of "${a}")"
  echo
  echo "--- ${a}: RUN=${RUN_A} (mode ${MODE})"
  ENV_A=(KD_MODE="${a}" RUN="${RUN_A}" VAL_LOSS="${VAL_LOSS}" EVAL_FREQ="${EVAL_FREQ}" POST_ONLY="${POST_ONLY}")
  if [ "${POST_ONLY}" != 1 ] || [ -n "${TIME_SET}" ]; then ENV_A+=(TIME="${TIME}"); fi
  if [ "${REUSE}" = 1 ]; then ENV_A+=(TEACHER_REMEASURE=0 TEACHER_TRACE_DIR="${T_TRACE}" TEACHER_EVAL_LOG="${T_LOG}")
  else ENV_A+=(TEACHER_REMEASURE=1); fi
  # exp8's pre-flight (label and fit checks on netscratch) takes a few minutes per arm: stream its output live
  TMP_A="$(mktemp)"
  set +e
  env -u TIME "${ENV_A[@]}" bash "${EXP8}" 2>&1 | tee "${TMP_A}" | sed -u 's/^/    /'
  RC=${PIPESTATUS[0]}
  set -e
  OUT_A="$(cat "${TMP_A}")"; rm -f -- "${TMP_A}"
  if [ "${RC}" != 0 ]; then
    echo "    exp8 exit ${RC} for arm ${a}: NOT SUBMITTED"
    FAILED+=("${a}")
    continue
  fi
  if [ "${MODE}" = submit ]; then
    JID="$(printf '%s\n' "${OUT_A}" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [ -z "${JID}" ]; then
      echo "    no 'Submitted batch job' line for arm ${a}: treat as NOT SUBMITTED (see exp8's output above)"
      FAILED+=("${a}")
      continue
    fi
    echo "    submitted: job ${JID} (exp8-${RUN_A}) -> logs/exp8_distill_vitb_${JID}.out, ${RESULTS}/${RUN_A}"
    echo "${a}  ${RUN_A}  ${JID}" >> "${RECORD}"
    JOBS+=("${a}:${JID}")
  fi
done

# ---------------------------------------------------------------- summary and the collection commands
echo
case "${MODE}" in
  local) echo "LOCAL_CHECK=1: nothing checked against the cluster, nothing submitted" ;;
  check) echo "CHECK_ONLY=1: exp8's guards ran for ${#ARMS[@]} arm(s); nothing submitted" ;;
  dry)   echo "DRY_RUN=1: nothing submitted" ;;
  submit)
    echo "submitted ${#JOBS[@]} of ${#ARMS[@]} arm(s): ${JOBS[*]:-none} (arm:job id); record ${RECORD}"
    echo "queue: squeue -u \${USER} -o '%i %j %T %M %L %R' | grep exp8-${PREFIX}_"
    echo "first minutes of a job (logs/exp8_distill_vitb_<jobid>.out): '[teacher] RESULT: PASS', then '=== 1. TRAINING';"
    echo "  in <RUN>/train.log: '[dinov2_hub] dinov2_vitb14 built from the local hub cache', '[pretrained] backbone: ...',"
    echo "  nokd: '[distill] NO distillation loss' and EPOCH= lines without 'kd:'; supervised: '[model] projector-fed heads, no teacher'" ;;
esac
[ "${#FAILED[@]}" = 0 ] || echo "NOT SUBMITTED for arm(s): ${FAILED[*]}"
T4="${RESULTS}/${PREFIX}_kdT4_sweep"
echo "collect (login node, CPU, after the jobs end):"
echo "  python scripts/analysis/kd_ablation_table.py --reported ${RESULTS}/${PREFIX} --sweep_t4 ${T4} --nokd ${RESULTS}/$(run_of nokd) --supervised ${RESULTS}/$(run_of supervised) --out results/experiments/exp10_kd_ablation"
echo "  python scripts/analysis/plot_loss_curves.py ${RESULTS}/$(run_of nokd) ${RESULTS}/$(run_of supervised) --out results/experiments/exp10_kd_ablation/loss_curves"
[ "${#FAILED[@]}" = 0 ] || exit 1
exit 0
