#!/bin/bash
# exp9_kd_temperature: the supervisor's two requests of 22 Oct on the ViT-B student (exp8), "without changing other parts
# of the setup": (1) a sweep of the knowledge-distillation TEMPERATURE (x = temperature, y = error; report the value that
# minimises the errors) and (2) training AND validation LOSS curves, to decide whether training longer helps (both still
# falling -> train more; train falling, validation flat or rising -> over-fitting, stop).
#
# WHAT IS VARIED: KD_T only = train.py --kd_temperature, the softmax temperature of the feature-KD term (distill.py:
#   KL(teacher || student) on the temperature-softened projected backbone tokens, softmax over channels, loss x T^2).
#   The output KD (--lambda_kd_out, an L1 on rotmat / shape / depth) has no temperature and is unchanged. TEMPS
#   (default "1 2 4 8 16"; 4 = the reported recipe, re-run here so the sweep has its own T 4 point: its difference to
#   the reported run is the run-to-run noise the collector prints).
# WHAT IS FIXED: everything else at exp8's defaults, through exp8 itself (one sbatch per temperature): the same teacher
#   (TEACHER, exp7 fix arm 00009), ViT-B student with NAVER's HMR-trained init, head_dim 1024, heads copied and frozen
#   5 epochs, the teacher's bone fix, the v23 3DPW mix, SEED 0, EPOCHS 30 x N_ITERS 1000 x BATCH 8, LR 3e-5 (x0.5 every
#   12 epochs), lambda_kd 1.0, lambda_kd_out 1.0, the post-steps (own bone fix, holdout traces, model cost, 3DPW-test
#   evaluation, final table). Two settings differ from the reported run for the LOSS CURVES, in every arm alike:
#   EVAL_FREQ 1 (the in-training evaluation every epoch instead of every 5th; it only measures, it never trains) and
#   VAL_LOSS 1 (train.py --val_loss 1: the training criterion on the validation images at each evaluation, in eval mode
#   under no_grad; a new flag whose default 0 leaves train.py's behaviour unchanged). Neither changes the optimisation.
# RUN NAMES: student_vitb_from_<teacher tag>_e30_kdT<t>_sweep (RUN passed explicitly, so even the T 4 arm never lands
#   in the reported run's folder student_vitb_from_<teacher tag>_e30 of job 3614191, which this script never touches;
#   exp8 refuses to write into a folder that already holds checkpoints anyway).
#
# EXPECTED COST: [Likely] ~6.5 h per run on an L40S for the training incl. the per-epoch evaluations (the reported run:
#   5.8 h for 30 epochs with 6 evaluations, 11.6 images/s; an evaluation of 100 holdout + 300 3DPW-validation images with
#   the extra student + teacher forward of --val_loss is ~1-2 min [Guessing], x 30), plus ~1 h of post-steps (two full
#   3DPW-test evaluations, three holdout traces, the bone fix, model_cost.py), so ~7.5 h wall per run; 5 temperatures
#   = ~33-38 GPU-h, in parallel if the queue allows (five jobs, --time TIME, default 16:00:00 = exp8's 14:00 + 2 h for
#   the 24 extra evaluations). An A40 / RTXA6000 is ~1.7x slower.
#
# OUTPUTS: one exp8 run folder per temperature, results/experiments/exp8_distill_vitb/<RUN>/ (train.log with the
#   per-epoch "***EVAL METRICS" blocks and "[val_loss]" lines, train/ tensorboard events incl. val_loss/*, holdout_student/
#   summary.json, eval_3dpw_test_student.log, final_table.json, ...; see exp8's header), the job logs
#   logs/exp8_distill_vitb_<jobid>.out, and this script's submission record logs/exp9_kd_temperature_<date>.txt (temperature,
#   RUN, job id). Afterwards, on the login node (CPU):
#     python scripts/analysis/kd_temperature_sweep.py --runs "results/experiments/exp8_distill_vitb/<prefix>_kdT*_sweep" \
#         --reported results/experiments/exp8_distill_vitb/<prefix> --out results/experiments/exp9_kd_temperature
#     python scripts/analysis/plot_loss_curves.py results/experiments/exp8_distill_vitb/<prefix>_kdT*_sweep --out results/experiments/exp9_kd_temperature/loss_curves
#   (both commands are printed, with the real names, at the end of a submission).
#
# MODES (all passed through to exp8 as its environment): CHECK_ONLY=1 runs exp8's guards for every temperature and submits
#   nothing (loads the teacher checkpoint once per temperature, seconds to a minute each); DRY_RUN=1 prints each sbatch
#   line; LOCAL_CHECK=1 works anywhere without the cluster and prints every composed command. Submit mode needs
#   'conda activate multihmr' and the repo root as the working directory (exp8 sbatch-es itself by its relative path).
#
# USAGE (repo root on the cluster, /netscratch/najib/multi-hmr):
#   conda activate multihmr
#   CHECK_ONLY=1 bash slurm/experiments/exp9_kd_temperature.sh       # guards for all 5 arms, submits nothing
#   bash slurm/experiments/exp9_kd_temperature.sh                    # submits 5 jobs (exp8-<RUN> each)
#   TEMPS="0.5 32" bash slurm/experiments/exp9_kd_temperature.sh     # extra arms later (own RUN names)
#   LOCAL_CHECK=1 bash slurm/experiments/exp9_kd_temperature.sh      # anywhere: print what would run
# ENV (defaults): TEMPS ("1 2 4 8 16"), VAL_LOSS (1), EVAL_FREQ (1), TIME (16:00:00), SWEEP_TAG (sweep; the RUN suffix),
#   TEACHER / STUDENT_BACKBONE / EPOCHS (exp8's defaults; they only enter the RUN names here, exp8 reads them itself),
#   CHECK_ONLY, DRY_RUN, LOCAL_CHECK (0). Every other exp8 variable set in the environment reaches exp8 unchanged (do not
#   set RUN or KD_T: this script sets them per arm).
set -euo pipefail

REPO="${REPO:-/netscratch/najib/multi-hmr}"
SELF="slurm/experiments/exp9_kd_temperature.sh"
EXP8="${EXP8:-slurm/experiments/exp8_distill_vitb.sh}"
abs () { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${REPO}/$1" ;; esac; }
posint () { [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $1=$2 (a positive integer)"; exit 1; }; }
posnum () { [[ "$2" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] || { echo "ERROR: $1=$2 (a non-negative number)"; exit 1; }; }
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
TEMPS="${TEMPS:-1 2 4 8 16}"
VAL_LOSS="${VAL_LOSS:-1}"; onezero VAL_LOSS "${VAL_LOSS}"
EVAL_FREQ="${EVAL_FREQ:-1}"; posint EVAL_FREQ "${EVAL_FREQ}"
TIME="${TIME:-16:00:00}"
SWEEP_TAG="${SWEEP_TAG:-sweep}"
case "${SWEEP_TAG}" in ''|*[!A-Za-z0-9_]*) echo "ERROR: SWEEP_TAG=${SWEEP_TAG} (letters, digits, _)"; exit 1 ;; esac
TEACHER="$(abs "${TEACHER:-${REPO}/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt}")"
STUDENT_BACKBONE="${STUDENT_BACKBONE:-dinov2_vitb14}"
case "${STUDENT_BACKBONE}" in
  dinov2_vitb14) BB_SHORT=vitb ;;
  dinov2_vits14) BB_SHORT=vits ;;
  *) echo "ERROR: STUDENT_BACKBONE=${STUDENT_BACKBONE} (dinov2_vitb14 | dinov2_vits14)"; exit 1 ;;
esac
EPOCHS="${EPOCHS:-30}"; posint EPOCHS "${EPOCHS}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_CHECK="${LOCAL_CHECK:-0}"
[ -z "${RUN:-}" ] || { echo "ERROR: RUN=${RUN} is set; this script names every arm itself (SWEEP_TAG changes the suffix)"; exit 1; }
[ -z "${KD_T:-}" ] || { echo "ERROR: KD_T=${KD_T} is set; use TEMPS=\"...\" for the values to sweep"; exit 1; }
TEACHER_TAG="$(tag_of "${TEACHER}")"
PREFIX="student_${BB_SHORT}_from_${TEACHER_TAG}_e${EPOCHS}"
RESULTS="results/experiments/exp8_distill_vitb"
ARMS=()
for t in ${TEMPS}; do
  posnum TEMPS "${t}"
  ARMS+=("${t}")
done
[ "${#ARMS[@]}" -gt 0 ] || { echo "ERROR: TEMPS is empty"; exit 1; }
run_of () { printf '%s\n' "${PREFIX}_kdT${1}_${SWEEP_TAG}"; }

echo "=== exp9_kd_temperature: KD temperature sweep of the exp8 ViT-B student"
echo "    temperatures ${ARMS[*]} (--kd_temperature); fixed: exp8 defaults, teacher ${TEACHER} (tag ${TEACHER_TAG}),"
echo "    ${STUDENT_BACKBONE}, ${EPOCHS} epochs; per epoch: evaluation (EVAL_FREQ ${EVAL_FREQ}) + validation loss (VAL_LOSS ${VAL_LOSS}); --time ${TIME}"
echo "    runs:"
for t in "${ARMS[@]}"; do echo "      KD_T=${t}  ->  ${RESULTS}/$(run_of "${t}")"; done
echo "    the reported run ${RESULTS}/${PREFIX} (job 3614191) is not touched"
MODE=submit
[ "${LOCAL_CHECK}" = 1 ] && MODE=local
[ "${CHECK_ONLY}" = 1 ] && MODE=check
[ "${DRY_RUN}" = 1 ] && [ "${MODE}" = submit ] && MODE=dry
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
  RECORD="${REPO}/logs/exp9_kd_temperature_$(date +%Y%m%d_%H%M%S).txt"
  { echo "# exp9_kd_temperature $(date -Is) git $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
    echo "# TEMPS=\"${TEMPS}\" VAL_LOSS=${VAL_LOSS} EVAL_FREQ=${EVAL_FREQ} TIME=${TIME} TEACHER=${TEACHER} EPOCHS=${EPOCHS}"
    echo "# temperature  RUN  job_id"; } > "${RECORD}"
fi

# ---------------------------------------------------------------- one exp8 call per temperature
JOBS=(); FAILED=()
for t in "${ARMS[@]}"; do
  RUN_T="$(run_of "${t}")"
  echo
  echo "--- KD_T=${t}: RUN=${RUN_T} (mode ${MODE})"
  # exp8's pre-flight (label and fit checks on netscratch) takes a few minutes per arm: stream its output live
  # (indented) instead of printing it only when exp8 returns, so a slow arm does not look like a hang
  TMP_T="$(mktemp)"
  set +e
  KD_T="${t}" RUN="${RUN_T}" VAL_LOSS="${VAL_LOSS}" EVAL_FREQ="${EVAL_FREQ}" TIME="${TIME}" bash "${EXP8}" 2>&1 \
    | tee "${TMP_T}" | sed -u 's/^/    /'
  RC=${PIPESTATUS[0]}
  set -e
  OUT_T="$(cat "${TMP_T}")"; rm -f -- "${TMP_T}"
  if [ "${RC}" != 0 ]; then
    echo "    exp8 exit ${RC} for KD_T=${t}: NOT SUBMITTED"
    FAILED+=("${t}")
    continue
  fi
  if [ "${MODE}" = submit ]; then
    JID="$(printf '%s\n' "${OUT_T}" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [ -z "${JID}" ]; then
      echo "    no 'Submitted batch job' line for KD_T=${t}: treat as NOT SUBMITTED (see exp8's output above)"
      FAILED+=("${t}")
      continue
    fi
    echo "    submitted: job ${JID} (exp8-${RUN_T}) -> logs/exp8_distill_vitb_${JID}.out, ${RESULTS}/${RUN_T}"
    echo "${t}  ${RUN_T}  ${JID}" >> "${RECORD}"
    JOBS+=("${t}:${JID}")
  fi
done

# ---------------------------------------------------------------- summary and the collection commands
echo
case "${MODE}" in
  local) echo "LOCAL_CHECK=1: nothing checked against the cluster, nothing submitted" ;;
  check) echo "CHECK_ONLY=1: exp8's guards ran for ${#ARMS[@]} arm(s); nothing submitted" ;;
  dry)   echo "DRY_RUN=1: nothing submitted" ;;
  submit)
    echo "submitted ${#JOBS[@]} of ${#ARMS[@]} arm(s): ${JOBS[*]:-none} (temperature:job id); record ${RECORD}"
    echo "queue: squeue -u \${USER} -o '%i %j %T %M %L %R' | grep exp8-${PREFIX}_kdT" ;;
esac
[ "${#FAILED[@]}" = 0 ] || echo "NOT SUBMITTED for KD_T: ${FAILED[*]}"
echo "collect (login node, CPU, after the jobs end):"
echo "  python scripts/analysis/kd_temperature_sweep.py --runs \"${RESULTS}/${PREFIX}_kdT*_${SWEEP_TAG}\" --reported ${RESULTS}/${PREFIX} --out results/experiments/exp9_kd_temperature"
echo "  python scripts/analysis/plot_loss_curves.py ${RESULTS}/${PREFIX}_kdT*_${SWEEP_TAG} ${RESULTS}/${PREFIX} --out results/experiments/exp9_kd_temperature/loss_curves"
[ "${#FAILED[@]}" = 0 ] || exit 1
exit 0
