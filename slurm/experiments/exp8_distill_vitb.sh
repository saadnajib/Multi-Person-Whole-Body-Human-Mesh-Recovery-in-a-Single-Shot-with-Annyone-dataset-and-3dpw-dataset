#!/bin/bash
# exp8_distill_vitb: supervisor requirement 1b, the SMALLER-MODEL route. A DINOv2 ViT-B/14 student distilled from the
# FINAL model (exp7, ViT-L) must keep PVE and MPJPE within 95% of the teacher on BOTH the AnnyOne holdout and the
# 3DPW test set (student error <= teacher error / 0.95) with fewer parameters AND FLOPs and faster inference.
# Structure, guards, modes and log checks follow slurm/experiments/exp7_final_model.sh. Nothing here has been run yet.
#
# RECIPE (each choice read from the code; history in slurm/history/train_distill_vits_v*.sh, README "Pipeline 3"):
#   teacher   TEACHER (default the final model, exp7 fix arm 00009; the MASK_3DPW_SHAPE=1 run is the alternative)
#             frozen, ViT-L backbone in bf16 (--teacher_bf16 1). Its saved args must say --gt_center_joint
#             person_center and carry a bone fix (guard), and its recipe must match the flags below (guard).
#   student   STUDENT_BACKBONE dinov2_vitb14 (dinov2_vits14 allowed), whole backbone trained (--freeze_backbone 0).
#             Backbone init STUDENT_INIT: NAVER's HMR-trained Multi-HMR ViT-B (models/multiHMR/multiHMR_672_B.pt; ViT-S:
#             multiHMR_672_S.pt as v4-v8) through --pretrained_remap 1 --load_only_backbone 1 (the ViT only; train.py
#             aborts below 90% of the backbone loaded). v4 showed the HMR-trained init beats the ImageNet DINOv2 one (PVE
#             193 -> 156 on ViT-S). STUDENT_INIT=dinov2 uses the ImageNet DINOv2 weights instead.
#   heads     HEAD_DIM 1024 = the teacher's head width (ViT-L backbone, head_dim 0). --init_heads_from_teacher copies
#             only tensors whose name AND shape match (train.py ~1160): at head_dim 0 a ViT-B student builds every head
#             at 768 (+99 camera channels), so the whole INPUT side of the head (HPH token embedding, cross-attention
#             K/V, learned queries, mlp_classif, mlp_offset) cannot transfer; that was the v3-v5 failure (README; v6
#             header). With --head_dim 1024, model.py's feat_proj (Linear 768 -> 1024, 0.79 M params) feeds heads built
#             at the teacher width, so every head tensor transfers and the feature KD compares the projected tokens with
#             the teacher's without a loss-side projector (v6: 152 / 105 vs v5 159 / 113). Cost: the heads stay
#             teacher-sized (14.0 M params, ~14 GFLOPs per image), next to 86.6 M / 590 GFLOPs of ViT-B.
#             --freeze_heads_epochs FREEZE_HEADS_EPOCHS (5, as v6; capped at EPOCHS-1) keeps the copied heads frozen
#             while backbone + projector learn to feed them.
#   bone fix  during training: STUDENT_TRAIN_FIX=teacher (default) = the teacher's own fix, written from its saved args
#             to <RUN>/teacher_bone_fix.json and passed as --bone_fix. The copied heads were trained WITH that fix (their
#             raw neck02 / head / ... output is the half-turn the fix undoes), and train.py itself warns when a teacher
#             with a fix teaches a student without one (train.py ~1180). Without it the student's FK, losses and output KD
#             would see the flipped bones, which exp5 showed fine-tuning does not undo.
#             after training (the student's OWN fix, as the task asks): diag_holdout_trace.py --select train_sample on
#             N_EST (200) seeded TRAINING images with --bone_fix none (raw head output) -> compute_bone_fix.py (auto: every
#             bone whose V1 median local error > 90 deg with a FLIP/CONST verdict) -> <RUN>/bone_fix.json (exp6 step a).
#             compute_bone_fix estimates the absolute correction from the raw output, so this JSON REPLACES the training
#             fix in the holdout trace and in the 3DPW evaluation. No qualifying bone (exit 3) = no fix: the JSON records
#             it with no bones and the student is evaluated with --bone_fix none. GLOBAL GUARD (exit 4, a compensated chain
#             such as neck02 -> head, expected for this lineage as for the teacher's own exp6 fix): ON_GLOBAL_RISE
#             continue (default; the teacher's 4-bone fix was used the same way) | safe (re-estimate with the safe subset).
#   targets   identical to the teacher recipe (exp7 fix arm): --gt_center_joint person_center --alpha_dist 5
#             --alpha_transl 5 (so the student learns the fixed depth), the 55 diag-manifest images excluded
#             (--exclude_indices_file), 30% real 3DPW-train from the v23 labels (--mix_3dpw_labels, --mix_3dpw_ratio 0.3),
#             --mask_3dpw_hands 1, MASK_3DPW_SHAPE (0), helper-joint mask, use_anny_shape, alpha_shape 20, j2d/v2d 0.1,
#             neck/hand boost x4, L1 rotmat 0.1, global-rot 0, --start_2d_epoch 0 (the heads are a trained teacher's, not
#             fresh: no reason to delay the 2D terms).
#   KD        v8 values: --lambda_kd 1.0 (feature KL on the tokens the heads see, T 4, softmax over channels)
#             --lambda_kd_out 1.0 (L1 to the teacher's rotmat / shape / depth on the same image; its depth term is
#             weighted by --alpha_dist 5, like the task term). Colour jitter as v6-v8 (brightness/contrast/saturation 0.2,
#             hue 0.05).
#   schedule  BATCH 8 (v6/v7; v6c/v8 used 16 on 80 GB for ViT-S: ViT-B at 16 would not fit 40-48 GB [Guessing]),
#             N_ITERS 1000, EPOCHS 30 = 240,000 images, LR 3e-5 (v8), StepLR x0.5 every LR_DECAY_EVERY (2/5 of EPOCHS,
#             as v8's 60 of 150), EVAL_FREQ 5 (holdout + 300 3DPW-validation images). v8 trained 150 epochs x 16 = 2.4 M
#             images on top of v6c: ~67 h at the throughput below, not a one-night run; 30 epochs is 10% of that and 6x
#             the teacher's own 40,000-image fine-tune. If the in-training holdout curve still falls at the end, rerun
#             longer (EPOCHS=60: new RUN name, ~2x the time).
#
# THROUGHPUT AND GPU-HOURS (estimate; nothing of this run measured yet):
#   [Certain] the teacher recipe ran at 10.9 images/s on an A100-PCIE-40GB (exp7 fix arm 3611336) and 6.7 on an A40
#     (control 3611339); ViT-L forward + backward through its last 4 blocks.
#   [Certain] v6b (ViT-S student, bf16 ViT-L teacher, batch 8): 0.47 s/it = 17 images/s (train_distill_vits_v6c.sh header).
#   [Guessing] this student: teacher forward (ViT-L, bf16) + ViT-B forward AND backward through all 12 blocks + two
#     head/mesh passes ~ the cost of one exp7 step: ~10 images/s on an A100 (6-12 by node), not the 2x of a plain ViT-B
#     fine-tune (~22/s), because the frozen ViT-L teacher still runs every step. 30 epochs = 240,000 images: ~6.7 GPU-h
#     at 10/s, ~10 h at 6.7/s, plus ~1 h of post-steps (two 3DPW-test evaluations of 8,190 images, 13-14 min each for
#     ViT-L [Certain, 3612073-5]; three holdout traces; the bone-fix estimation; model_cost.py). The default --time
#     (TIME) assumes 6 images/s: 24 min per epoch + 120 min = 14:00:00 for 30 epochs. Read the real rate from the
#     smoke test's "[throughput]" line and the first epochs of the real run.
#
# WHAT RUNS (one job, one GPU):
#   0. checks inside the job (as at submission): v23 labels (copy of exp7's label check), teacher args (recipe, person
#      centre, bone fix), DINOv2 weights in the torch cache, GPU compute capability 8.0-8.9
#   1. training -> <RUN>/train.log, <RUN>/train/checkpoints (the last NB_MAX_CKPT epochs); start-up lines checked
#   2. the student's own bone fix (est_trace on N_EST training images, compute_bone_fix.py) -> <RUN>/bone_fix.json
#   3. holdout traces (100 AnnyOne holdout images, V0-V3, diag_holdout_trace.py): student WITH its new fix ->
#      holdout_student/; student with its training fix (the teacher's) -> holdout_student_trainfix/ (reference);
#      teacher re-measured -> holdout_teacher/ (TEACHER_REMEASURE=1)
#   4. scripts/analysis/model_cost.py --ckpt TEACHER --ckpt <student> [--bench]: params, FLOPs, ms/image -> model_cost.json
#   5. slurm/eval/run_eval_3dpw_test.sh (AnnyOne test-100 + every 3DPW-test image of LABELS) for the student with
#      EVAL_BONE_FIX=<RUN>/bone_fix.json (or none) and for the teacher (its own fix) -> eval_3dpw_test_{student,teacher}.log
#   6. final table -> log + <RUN>/final_table.json: teacher | student | ratio | 95% target | met? for AnnyOne holdout
#      PVE / PA-PVE / MPJPE / recall (V1 trace, the numbers the teacher was reported with), AnnyOne test-100 and 3DPW test
#      PVE / PA-PVE / MPJPE / recall (evaluate()), params, FLOPs, inference ms/image, training images/s.
#
# THE 95% RULE (supervisor 1b; computed from the teacher's numbers RE-MEASURED in this job, never from constants):
#   for each error metric e: target = teacher_e / 0.95; met = student_e <= target. PASS = met for PVE and MPJPE on the
#   AnnyOne holdout AND on 3DPW test, and the student has fewer params AND fewer FLOPs (AND is faster when --bench ran).
#   PA-PVE and recall get the same "met?" column for information (recall: student >= 0.95 x teacher). Reference values
#   from exp7 (for orientation only): AnnyOne holdout PVE 73.7 / PA-PVE 50.3 / recall 87.0, 3DPW test v23 PVE 89.0 /
#   PA-PVE 57.6 / MPJPE 88.3 -> targets PVE <= 77.6 and <= 93.7.
#
# GUARDS: the job REFUSES (exit 1, before any training) when
#   - TEACHER is missing, its saved args lack --gt_center_joint person_center or a bone fix (bone_fix_spec), its head
#     width differs from HEAD_DIM (with INIT_HEADS=1), or its recipe differs from this run's targets (alpha_dist,
#     alpha_transl, mask_3dpw_hands, mask_3dpw_shape, the real path of mix_3dpw_labels vs LABELS); ALLOW_RECIPE_MISMATCH=1
#     turns the recipe ones (not the person centre / bone fix ones) into WARNINGs
#   - the labels fail the exp7 label check (old threedpw_anny_full labels, missing / empty index files, index lines
#     outside LABELS, a refit task without its fit or a fit newer than the labels, wrong gender convention / frame grid,
#     wrong gender values in a sample of label files)
#   - STUDENT_INIT (unless dinov2), the manifest, the DINOv2 hub code or <TORCH_CACHE>/hub/checkpoints/<backbone>_pretrain.pth
#     is missing (train.py always builds the backbone with the ImageNet DINOv2 weights first; compute nodes have no
#     internet [Likely, train_distill_vits_v6.sh]); the refusal prints the wget command for the login node
#   - slurm/eval/run_eval_3dpw_test.sh lacks MODEL_ENTRY / EVAL_LABELS / EVAL_BONE_FIX (an old copy), or train.py lacks a
#     flag used here
#   - PARTITIONS names V100, B200, RTXB6000, H100 or H200, or the GPU's compute capability is not 8.0-8.9 (torch 2.0.1
#     +cu117 has no sm_90+ kernels; V100 = 7.0 has no TF32 / bf16); --exclude=gifu,glendale always
#   - <RUN>/train/checkpoints already holds .pt files (OVERWRITE=1 moves the folder to <RUN>_replaced_<jobid>), a job
#     named exp8-<RUN> is queued or running, or a plain-sbatch job exp8-distill-vitb is (IGNORE_PLAIN_SBATCH=1)
#   After training it stops before the post-steps when train.log lacks "[distill] teacher loaded", "[distill] heads
#   initialised from teacher" (INIT_HEADS=1), "[distill] projector-fed heads" (HEAD_DIM != backbone width), "[mixed]
#   training on", "[loss] mask_3dpw_hands" (MASK_3DPW_HANDS=1), "[loss] mask_3dpw_shape:" (MASK_3DPW_SHAPE=1),
#   "[pretrained] backbone:" (STUDENT_INIT file), two "[bone_fix] ON, <n> bone(s)" lines (student + teacher, with
#   STUDENT_TRAIN_FIX=teacher), or shows train.py's teacher-has-a-fix-but-the-student-not WARNING.
#   After step 5 each eval log must show its "task 0: <label>  ->  <ckpt>", "labels: <LABELS>" and "bone fix: ..." line.
#
# MODES: submit mode when SLURM_JOB_ID is empty; job mode only with SLURM_JOB_ID AND EXP8_JOB=1 (set by the submit mode
#   and by the #SBATCH --export line). The submit mode runs the checks before sbatch; they need Python 3.8+ with NumPy
#   AND torch (the teacher's args are pickled in its checkpoint): conda activate multihmr first. Reading the teacher
#   loads its 1.2 GB checkpoint on the CPU (seconds to a minute on the login node).
#   LOCAL_CHECK=1: anywhere, no cluster paths: prints the settings, the composed train.py command, the post-step commands
#   and the sbatch line, syntax-checks the embedded Python, exits 0. CHECK_ONLY=1 / DRY_RUN=1 as exp7.
#   TABLE_ONLY=1: login node, rebuilds step 6 of an existing <RUN> from <RUN>/table_inputs.json (same settings / RUN).
#
# USAGE (repo root on the cluster, /netscratch/najib/multi-hmr):
#   conda activate multihmr
#   CHECK_ONLY=1 bash slurm/experiments/exp8_distill_vitb.sh          # guards only, submits nothing
#   SMOKE=1 bash slurm/experiments/exp8_distill_vitb.sh               # 1 epoch x 20 iterations + every post-step on small
#                                                                     # inputs (RUN ..._e1_it20_smoke); ~1 h incl. start-up
#   bash slurm/experiments/exp8_distill_vitb.sh                       # the real run (30 epochs, --time 14:00:00)
#   TEACHER=/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt \
#     MASK_3DPW_SHAPE=1 bash slurm/experiments/exp8_distill_vitb.sh   # student of the shape-masked teacher (own RUN name)
#   STUDENT_BACKBONE=dinov2_vits14 bash slurm/experiments/exp8_distill_vitb.sh   # a ViT-S student, same recipe
#   TABLE_ONLY=1 bash slurm/experiments/exp8_distill_vitb.sh          # rebuild the final table (same env as the run)
#   LOCAL_CHECK=1 bash slurm/experiments/exp8_distill_vitb.sh         # anywhere: print what would run
# ENV (defaults): TEACHER (exp7 fix arm 00009), STUDENT_BACKBONE (dinov2_vitb14 | dinov2_vits14), STUDENT_INIT
#   (models/multiHMR/multiHMR_672_B.pt; _S for ViT-S | dinov2), HEAD_DIM (1024), INIT_HEADS (1), STUDENT_TRAIN_FIX
#   (teacher | none), EPOCHS (30), N_ITERS (1000), BATCH (8), LR (3e-5), LR_DECAY_EVERY (2/5 of EPOCHS), LAMBDA_KD (1.0),
#   LAMBDA_KD_OUT (1.0), KD_T (4.0), FREEZE_HEADS_EPOCHS (5, at most EPOCHS-1), START_2D_EPOCH (0), ALPHA_DIST /
#   ALPHA_TRANSL (5), MIX_RATIO (0.3), LABELS (threedpw_anny_v23/labels), FITS (parent of LABELS), MASK_3DPW_HANDS (1),
#   MASK_3DPW_SHAPE (0), VAL_3DPW_N (300), EVAL_FREQ (5), NB_MAX_CKPT (3), NUM_WORKERS (12), N_EST (200), SEED (0),
#   BONES (auto), ON_GLOBAL_RISE (continue | safe), N_VIS (12), EVAL_TEST (1), EVAL_3DPW_N (0 = all), TEACHER_REMEASURE (1;
#   0 = reuse TEACHER_TRACE_DIR / TEACHER_EVAL_LOG, default the teacher run's holdout_ft/ and eval_3dpw_test.log), BENCH
#   (1), RUN (student_<vitb|vits>_from_<teacher tag>_e<EPOCHS>[_it<N_ITERS>][_b<BATCH>][_shpmask][_nofix][_dinov2][_smoke]),
#   TIME (from EPOCHS, see above), SMOKE (0; 1 = EPOCHS 1, N_ITERS 20, VAL_3DPW_N 20, N_VIS 2, N_EST 20, EVAL_3DPW_N 50,
#   EVAL_FREQ 1 unless set), PARTITIONS, TORCH_CACHE (/netscratch/najib/torch_cache), MANIFEST, TASKS, N_LABEL_CHECK (300),
#   ALLOW_RECIPE_MISMATCH (0), CHECK_ONLY, DRY_RUN, OVERWRITE, TABLE_ONLY, LOCAL_CHECK, IGNORE_PLAIN_SBATCH.
#
# OUTPUTS (results/experiments/exp8_distill_vitb/<RUN>/): settings.txt, teacher_check.json, teacher_bone_fix.json,
#   exclude_manifest_indices.txt, train.log / train_lines.log, train/checkpoints, est_trace/ + bone_fix.json (+
#   bone_fix_guarded.json with ON_GLOBAL_RISE=safe), holdout_student/, holdout_student_trainfix/, holdout_teacher/,
#   model_cost.json, eval_3dpw_test_student.log, eval_3dpw_test_teacher.log, table_inputs.json, final_table.json,
#   last_checkpoint.txt. Eval run folders: logs/anny_model/evaltest_exp8_<RUN>[_teacher]__threedpw_anny_v23.
#
# RISKS: (1) the student's bone fix is a fresh estimate on 200 training images; a bone that flips only in the student
#   or a compensated chain the guard flags can still cost PVE (compare holdout_student with holdout_student_trainfix).
#   (2) ViT-B may miss 95% on 3DPW even if it meets it on AnnyOne: the ViT-S students stayed far from their teachers
#   (v8: 3DPW-val 123.6 vs the teacher's ~95-103). (3) FLOPs are analytic for the backbone (+ hooked linears for the
#   heads); the Anny mesh is not counted; latency is batch 1 on whatever GPU the job got (both models on the same GPU).
#   (4) BATCH 8 with a fully trained ViT-B + the ViT-L teacher at 672 px may not fit a 40 GB card [Guessing]: the smoke
#   test shows it within minutes if it lands on one (PARTITIONS=A100-40GB,A100-PCI SMOKE=1 forces that); then BATCH=4.
#SBATCH --job-name=exp8-distill-vitb
#SBATCH --partition=A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV
#SBATCH --exclude=gifu,glendale
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp8_distill_vitb_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp8_distill_vitb_%j.err
#SBATCH --export=ALL,EXP8_JOB=1
set -euo pipefail

REPO="${REPO:-/netscratch/najib/multi-hmr}"   # repo root on the cluster; override only for a local check
SELF="slurm/experiments/exp8_distill_vitb.sh"
abs () { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${REPO}/$1" ;; esac; }
posint () { [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $1=$2 (a positive integer)"; exit 1; }; }
nonneg () { [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: $1=$2 (a non-negative integer)"; exit 1; }; }
posnum () { [[ "$2" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] || { echo "ERROR: $1=$2 (a non-negative number)"; exit 1; }; }
onezero () { case "$2" in 0|1) ;; *) echo "ERROR: $1=$2 (0|1)"; exit 1 ;; esac; }
tag_of () {   # = exp6_bone_fix.sh tag_of: <run>_<epoch> for .../<run>[/train]/checkpoints/<epoch>.pt, else the file stem
  local ck="$1" rd rn
  if [ "$(basename "$(dirname "$ck")")" = checkpoints ]; then
    rd="$(dirname "$(dirname "$ck")")"; rn="$(basename "$rd")"
    [ "$rn" = train ] && rn="$(basename "$(dirname "$rd")")"
    echo "${rn}_$(basename "$ck" .pt)"
  else
    basename "$ck" .pt
  fi
}
run_dir_of () {   # the run folder of .../<run>/train/checkpoints/<epoch>.pt (exp7 layout), else empty
  local ck="$1" d
  d="$(dirname "$ck")"
  [ "$(basename "$d")" = checkpoints ] || return 0
  d="$(dirname "$d")"
  [ "$(basename "$d")" = train ] && d="$(dirname "$d")"
  echo "$d"
}

# ---------------------------------------------------------------- settings
SMOKE="${SMOKE:-0}"; onezero SMOKE "${SMOKE}"
if [ "${SMOKE}" = 1 ]; then   # exp7's smoke pattern; every post-step runs on small inputs
  EPOCHS="${EPOCHS:-1}"; N_ITERS="${N_ITERS:-20}"; VAL_3DPW_N="${VAL_3DPW_N:-20}"; N_VIS="${N_VIS:-2}"
  N_EST="${N_EST:-20}"; EVAL_3DPW_N="${EVAL_3DPW_N:-50}"; EVAL_FREQ="${EVAL_FREQ:-1}"
fi
TEACHER="$(abs "${TEACHER:-${REPO}/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt}")"
STUDENT_BACKBONE="${STUDENT_BACKBONE:-dinov2_vitb14}"
case "${STUDENT_BACKBONE}" in
  dinov2_vitb14) BB_SHORT=vitb; BB_WIDTH=768; NAVER_INIT="multiHMR_672_B.pt" ;;
  dinov2_vits14) BB_SHORT=vits; BB_WIDTH=384; NAVER_INIT="multiHMR_672_S.pt" ;;
  *) echo "ERROR: STUDENT_BACKBONE=${STUDENT_BACKBONE} (dinov2_vitb14 | dinov2_vits14)"; exit 1 ;;
esac
STUDENT_INIT="${STUDENT_INIT:-${REPO}/models/multiHMR/${NAVER_INIT}}"
[ "${STUDENT_INIT}" = dinov2 ] || STUDENT_INIT="$(abs "${STUDENT_INIT}")"
HEAD_DIM="${HEAD_DIM:-1024}"; nonneg HEAD_DIM "${HEAD_DIM}"
INIT_HEADS="${INIT_HEADS:-1}"; onezero INIT_HEADS "${INIT_HEADS}"
STUDENT_TRAIN_FIX="${STUDENT_TRAIN_FIX:-teacher}"
case "${STUDENT_TRAIN_FIX}" in teacher|none) ;; *) echo "ERROR: STUDENT_TRAIN_FIX=${STUDENT_TRAIN_FIX} (teacher | none)"; exit 1 ;; esac
EPOCHS="${EPOCHS:-30}"; posint EPOCHS "${EPOCHS}"
N_ITERS="${N_ITERS:-1000}"; posint N_ITERS "${N_ITERS}"
BATCH="${BATCH:-8}"; posint BATCH "${BATCH}"
MAX_ITER=$(( EPOCHS * N_ITERS ))
LR="${LR:-3e-5}"; posnum LR "${LR}"
LR_DECAY_EVERY="${LR_DECAY_EVERY:-$(( EPOCHS * 2 / 5 > 0 ? EPOCHS * 2 / 5 : 1 ))}"; posint LR_DECAY_EVERY "${LR_DECAY_EVERY}"
LAMBDA_KD="${LAMBDA_KD:-1.0}"; posnum LAMBDA_KD "${LAMBDA_KD}"
LAMBDA_KD_OUT="${LAMBDA_KD_OUT:-1.0}"; posnum LAMBDA_KD_OUT "${LAMBDA_KD_OUT}"
KD_T="${KD_T:-4.0}"; posnum KD_T "${KD_T}"
FREEZE_HEADS_EPOCHS="${FREEZE_HEADS_EPOCHS:-$(( EPOCHS - 1 < 5 ? EPOCHS - 1 : 5 ))}"; nonneg FREEZE_HEADS_EPOCHS "${FREEZE_HEADS_EPOCHS}"
START_2D_EPOCH="${START_2D_EPOCH:-0}"; nonneg START_2D_EPOCH "${START_2D_EPOCH}"
ALPHA_DIST="${ALPHA_DIST:-5}"; posnum ALPHA_DIST "${ALPHA_DIST}"
ALPHA_TRANSL="${ALPHA_TRANSL:-5}"; posnum ALPHA_TRANSL "${ALPHA_TRANSL}"
MIX_RATIO="${MIX_RATIO:-0.3}"; posnum MIX_RATIO "${MIX_RATIO}"
LABELS="${LABELS:-${REPO}/threedpw_anny_v23/labels}"
LABELS="$(abs "${LABELS%/}")"
FITS="$(abs "${FITS:-$(dirname "${LABELS}")}")"
MASK_3DPW_HANDS="${MASK_3DPW_HANDS:-1}"; onezero MASK_3DPW_HANDS "${MASK_3DPW_HANDS}"
MASK_3DPW_SHAPE="${MASK_3DPW_SHAPE:-0}"; onezero MASK_3DPW_SHAPE "${MASK_3DPW_SHAPE}"
VAL_3DPW_N="${VAL_3DPW_N:-300}"; nonneg VAL_3DPW_N "${VAL_3DPW_N}"
EVAL_FREQ="${EVAL_FREQ:-5}"; posint EVAL_FREQ "${EVAL_FREQ}"
NB_MAX_CKPT="${NB_MAX_CKPT:-3}"; posint NB_MAX_CKPT "${NB_MAX_CKPT}"
NUM_WORKERS="${NUM_WORKERS:-12}"; nonneg NUM_WORKERS "${NUM_WORKERS}"
N_EST="${N_EST:-200}"; posint N_EST "${N_EST}"
SEED="${SEED:-0}"; nonneg SEED "${SEED}"
BONES="${BONES:-auto}"
ON_GLOBAL_RISE="${ON_GLOBAL_RISE:-continue}"
case "${ON_GLOBAL_RISE}" in continue|safe) ;; *) echo "ERROR: ON_GLOBAL_RISE=${ON_GLOBAL_RISE} (continue | safe)"; exit 1 ;; esac
N_VIS="${N_VIS:-12}"; nonneg N_VIS "${N_VIS}"
EVAL_TEST="${EVAL_TEST:-1}"; onezero EVAL_TEST "${EVAL_TEST}"
EVAL_3DPW_N="${EVAL_3DPW_N:-0}"; nonneg EVAL_3DPW_N "${EVAL_3DPW_N}"
TEACHER_REMEASURE="${TEACHER_REMEASURE:-1}"; onezero TEACHER_REMEASURE "${TEACHER_REMEASURE}"
BENCH="${BENCH:-1}"; onezero BENCH "${BENCH}"
ALLOW_RECIPE_MISMATCH="${ALLOW_RECIPE_MISMATCH:-0}"; onezero ALLOW_RECIPE_MISMATCH "${ALLOW_RECIPE_MISMATCH}"
N_LABEL_CHECK="${N_LABEL_CHECK:-300}"; nonneg N_LABEL_CHECK "${N_LABEL_CHECK}"
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"
TASKS="$(abs "${TASKS:-configs/threedpw_tasks.txt}")"
TORCH_CACHE="${TORCH_CACHE:-/netscratch/najib/torch_cache}"   # = TORCH_HOME of slurm/env.sh
PARTITIONS="${PARTITIONS:-A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
TABLE_ONLY="${TABLE_ONLY:-0}"
LOCAL_CHECK="${LOCAL_CHECK:-0}"
IGNORE_PLAIN_SBATCH="${IGNORE_PLAIN_SBATCH:-0}"
EVAL_SCRIPT="${REPO}/slurm/eval/run_eval_3dpw_test.sh"

TEACHER_TAG="$(tag_of "${TEACHER}")"
TEACHER_RUN_DIR="$(run_dir_of "${TEACHER}")"
TEACHER_TRACE_DIR="${TEACHER_TRACE_DIR:-${TEACHER_RUN_DIR:+${TEACHER_RUN_DIR}/holdout_ft}}"
TEACHER_EVAL_LOG="${TEACHER_EVAL_LOG:-${TEACHER_RUN_DIR:+${TEACHER_RUN_DIR}/eval_3dpw_test.log}}"
SUFFIX=""
[ "${N_ITERS}" = 1000 ] || SUFFIX="${SUFFIX}_it${N_ITERS}"
[ "${BATCH}" = 8 ] || SUFFIX="${SUFFIX}_b${BATCH}"
[ "${MASK_3DPW_SHAPE}" = 1 ] && SUFFIX="${SUFFIX}_shpmask"
[ "${STUDENT_TRAIN_FIX}" = none ] && SUFFIX="${SUFFIX}_nofix"
[ "${STUDENT_INIT}" = dinov2 ] && SUFFIX="${SUFFIX}_dinov2"
[ "${SMOKE}" = 1 ] && SUFFIX="${SUFFIX}_smoke"
RUN="${RUN:-student_${BB_SHORT}_from_${TEACHER_TAG}_e${EPOCHS}${SUFFIX}}"
case "${RUN}" in ''|*/*|.|..) echo "ERROR: RUN=${RUN} (one folder name, no /)"; exit 1 ;; esac
OUT="results/experiments/exp8_distill_vitb/${RUN}"
OUT_ABS="$(abs "${OUT}")"
STUDENT_LABEL="exp8_${RUN}"
TEACHER_LABEL="exp8_${RUN}_teacher"
SPLITS="train,validation"; [ "${EVAL_TEST}" = 1 ] && SPLITS="train,validation,test"
# --time: 6 images/s assumed (slow node), + 1 min per epoch for the in-training evaluations; post-steps 60 min + 30 per
# full 3DPW-test evaluation (13-14 min measured for ViT-L on v23), 10 per partial one
EPOCH_MIN=$(( (N_ITERS * BATCH + 359) / 360 + 1 ))
EVAL_MIN=30; [ "${EVAL_3DPW_N}" = 0 ] || EVAL_MIN=10
N_EVALS=0; [ "${EVAL_TEST}" = 1 ] && N_EVALS=$(( 1 + TEACHER_REMEASURE ))
NEED_MIN=$(( 60 + N_EVALS * EVAL_MIN + EPOCHS * EPOCH_MIN ))
TIME="${TIME:-$(printf '%02d:%02d:00' $(( NEED_MIN / 60 )) $(( NEED_MIN % 60 )))}"
[ "${HEAD_DIM}" = 0 ] || [ "${HEAD_DIM}" = "${BB_WIDTH}" ] && PROJ_EXPECTED=0 || PROJ_EXPECTED=1
if [ "${INIT_HEADS}" = 1 ] && [ "${HEAD_DIM}" = 0 ]; then
  echo "NOTE: INIT_HEADS=1 with HEAD_DIM=0: the head's input layers cannot transfer (v3-v5); the teacher check refuses"
  echo "      unless the teacher's head width equals ${BB_WIDTH}"
fi

# ---------------------------------------------------------------- the train.py command (job and LOCAL_CHECK)
if [ "${STUDENT_INIT}" = dinov2 ]; then
  INIT_ARGS=""                                   # Model(pretrained_backbone=1): ImageNet DINOv2 weights from torch.hub
else
  INIT_ARGS="--pretrained ${STUDENT_INIT} --pretrained_remap 1 --load_only_backbone 1 --min_backbone_match 90"
fi
HEADS_ARGS=""
[ "${INIT_HEADS}" = 1 ] && HEADS_ARGS="--init_heads_from_teacher 1 --freeze_heads_epochs ${FREEZE_HEADS_EPOCHS}"
if [ "${STUDENT_TRAIN_FIX}" = teacher ]; then TRAIN_FIX_ARG="--bone_fix ${OUT_ABS}/teacher_bone_fix.json"
else TRAIN_FIX_ARG="--bone_fix none"; fi
HANDS_ARG=""; [ "${MASK_3DPW_HANDS}" = 1 ] && HANDS_ARG="--mask_3dpw_hands 1"
SHAPE_ARG=""; [ "${MASK_3DPW_SHAPE}" = 1 ] && SHAPE_ARG="--mask_3dpw_shape 1"
TRAIN_CMD="python train.py --train_data AnnyOne --train_n 0 --val_anny_n 100 --person_center head \
--backbone ${STUDENT_BACKBONE} --img_size 672 --head_dim ${HEAD_DIM} ${INIT_ARGS} --freeze_backbone 0 \
--distill_teacher_ckpt ${TEACHER} ${HEADS_ARGS} --teacher_bf16 1 \
--lambda_kd ${LAMBDA_KD} --lambda_kd_out ${LAMBDA_KD_OUT} --kd_temperature ${KD_T} --kd_softmax_dim channel \
${TRAIN_FIX_ARG} --mask_helper_joints_train 1 --use_anny_shape 1 \
--alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \
--rotmat_loss l1 --alpha_rotmat 0.1 --alpha_global_rot 0 --start_2d_epoch ${START_2D_EPOCH} \
--gt_center_joint person_center --alpha_dist ${ALPHA_DIST} --alpha_transl ${ALPHA_TRANSL} --debug_anchor 2 \
--exclude_indices_file ${OUT_ABS}/exclude_manifest_indices.txt \
--mix_3dpw_labels ${LABELS} --mix_3dpw_ratio ${MIX_RATIO} --mix_3dpw_split train \
--val_3dpw_split validation --val_3dpw_n ${VAL_3DPW_N} ${HANDS_ARG} ${SHAPE_ARG} \
--brightness 0.2 --contrast 0.2 --saturation 0.2 --hue 0.05 \
--batch_size ${BATCH} --num_workers ${NUM_WORKERS} --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \
--learning_rate ${LR} --lr_decay_every ${LR_DECAY_EVERY} --lr_decay_gamma 0.5 --amp 0 --tf32 1 \
--eval_freq ${EVAL_FREQ} --nb_max_ckpt ${NB_MAX_CKPT} --log_freq 50 \
--save_dir ${OUT_ABS} --name train"
TRAIN_CMD="$(printf '%s' "${TRAIN_CMD}" | tr -s ' ')"
pretty () { printf '%s\n' "$1" | sed 's/ --/ \\\n    --/g'; }

# ---------------------------------------------------------------- python checks (shared by both modes)
# 1. label check: exp7_final_model.sh's CHECK_PY, copied without its bone-fix part (exp7 checks the fix it starts from;
#    here the teacher check below does). Standard library + NumPy. Exit 0 = pass, 1 = refused, 2 = bad arguments.
LABEL_CHECK_PY=$(cat <<'PY'
import argparse, glob, json, os, pickle, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--fits', required=True)
ap.add_argument('--splits', default='train,validation,test')
ap.add_argument('--n_sample', type=int, default=300)
ap.add_argument('--base', default='.')   # relative index lines are read from here (train.py runs in the repo root)
a = ap.parse_args()
REAL_LABELS = os.path.realpath(a.labels)
EXPECT = {'m': 0.0, 'f': 1.0}   # Anny: 0 = male, 1 = female (smpl_to_anny.py GENDER_CONVENTIONS['anny'])
TOL = 0.05
bad = []

def s(x):
    try:
        return str(np.asarray(x).item())
    except Exception:
        return str(x)

# fits behind the labels, then sampled label files (exp7 sections 2 and 3, unchanged)
for split in [x.strip() for x in a.splits.split(',') if x.strip()]:
    idx = os.path.join(a.labels, 'index_%s.txt' % split)
    try:
        files = [l.strip() for l in open(idx) if l.strip()]
    except Exception as e:
        bad.append('%s unreadable: %s' % (idx, e))
        continue
    if not files:
        bad.append('%s is empty' % idx)
        continue
    files = [f if os.path.isabs(f) else os.path.join(a.base, f) for f in files]
    # every line must point into LABELS: an index copied from the old labels would otherwise only be caught by the
    # gender sample below. Real folder per line (one realpath per folder), real file for the sampled lines.
    real_dir, n_old, n_out = {}, 0, 0
    for f in files:
        d = os.path.dirname(f)
        if d not in real_dir:
            real_dir[d] = os.path.realpath(d)
        rd = real_dir[d]
        if 'threedpw_anny_full' in f or 'threedpw_anny_full' in rd:
            n_old += 1
            if n_old <= 3:
                bad.append('%s: line %s points into threedpw_anny_full (old labels)' % (idx, f))
        elif rd != REAL_LABELS and not rd.startswith(REAL_LABELS + os.sep):
            n_out += 1
            if n_out <= 3:
                bad.append('%s: line %s (real folder %s) is not under LABELS (real path %s)' % (idx, f, rd, REAL_LABELS))
    if n_old > 3:
        bad.append('%s: %d more lines point into threedpw_anny_full' % (idx, n_old - 3))
    if n_out > 3:
        bad.append('%s: %d more lines are not under LABELS' % (idx, n_out - 3))
    seqs = sorted({os.path.basename(os.path.dirname(f)) for f in files})
    fit_gender, n_fit, n_frames = {}, 0, 0
    for seq in seqs:
        # exact fit names only (<seq>_p<k>.npz): a leftover <seq>_p<k>.npz.tmp.npz of a killed task is not a fit
        npzs = sorted(p for p in glob.glob(os.path.join(a.fits, split, seq + '_p*.npz'))
                      if re.fullmatch(re.escape(seq) + r'_p[0-9]+\.npz', os.path.basename(p)))
        if not npzs:
            bad.append('%s/%s: no fit .npz under %s (set FITS to the folder the labels were built from)'
                       % (split, seq, os.path.join(a.fits, split)))
            continue
        for p in npzs:
            n_fit += 1
            try:
                d = np.load(p, allow_pickle=False)
                conv = s(d['gender_convention']) if 'gender_convention' in d.files else 'missing (v2.2 or older fit)'
                grid = s(d['frame_grid']) if 'frame_grid' in d.files else 'missing (v2.2 or older fit)'
                ver = s(d['version']) if 'version' in d.files else '?'
                g = s(d['gender']) if 'gender' in d.files else ''
                gi = [str(x) for x in d['phenotype_labels']].index('gender')
                vals = np.asarray(d['anny_shape'], dtype=np.float64)[:, gi]
            except Exception as e:
                bad.append('%s unreadable: %s: %s' % (p, type(e).__name__, e))
                continue
            n_frames += len(vals)
            if conv != 'anny':
                bad.append('%s: gender_convention %s (need anny; version %s)' % (p, conv, ver))
            if grid != 'shared':
                bad.append('%s: frame_grid %s (need shared; version %s)' % (p, grid, ver))
            if g not in EXPECT:
                bad.append('%s: 3DPW gender %r is not m/f' % (p, g))
                continue
            off = np.abs(vals - EXPECT[g]) > TOL
            if off.any():
                bad.append('%s: 3DPW gender %s but Anny gender value %.2f..%.2f in %d of %d frames (need %.1f)'
                           % (p, g, vals.min(), vals.max(), int(off.sum()), len(vals), EXPECT[g]))
            fit_gender[os.path.basename(p)[:-4]] = (g, gi)
    n_h = n_off = n_unk = 0
    pick = np.unique(np.linspace(0, len(files) - 1, min(max(a.n_sample, 0), len(files))).astype(int)) if a.n_sample > 0 else []
    for i in pick:
        rf = os.path.realpath(files[i])
        if os.path.islink(files[i]) and ('threedpw_anny_full' in rf or not rf.startswith(REAL_LABELS + os.sep)):
            bad.append('%s: sampled label %s is a link to %s, outside LABELS (real path %s)' % (idx, files[i], rf, REAL_LABELS))
        try:
            with open(files[i], 'rb') as fh:
                lab = pickle.load(fh)
        except Exception as e:
            bad.append('%s unreadable: %s: %s' % (files[i], type(e).__name__, e))
            continue
        for h in lab.get('humans', []):
            key = h.get('name', '')
            if key not in fit_gender:
                n_unk += 1
                continue
            g, gi = fit_gender[key]
            v = float(np.asarray(h['anny_shape'], dtype=np.float64).reshape(-1)[gi])
            n_h += 1
            if abs(v - EXPECT[g]) > TOL:
                n_off += 1
                if n_off <= 3:
                    bad.append('%s: %s (3DPW gender %s) has Anny gender value %.2f (need %.1f): labels not built from these fits?'
                               % (files[i], key, g, v, EXPECT[g]))
    if n_off > 3:
        bad.append('%s: %d more sampled people with a wrong gender value' % (split, n_off - 3))
    if a.n_sample > 0 and n_h == 0:
        bad.append('%s: no sampled label person could be matched to a fit (name <seq>_p<k>)' % split)
    print('[check] %-10s %6d label files (%d old-label lines, %d outside LABELS), %3d sequences, %3d fits (%d fitted'
          ' frames) checked; label sample: %d people in %d files, %d wrong gender, %d without a fit'
          % (split, len(files), n_old, n_out, len(seqs), n_fit, n_frames, n_h, len(pick), n_off, n_unk))
for b in bad:
    print('REFUSED: ' + b)
print('[check] RESULT: ' + ('PASS' if not bad else 'FAIL (%d problem(s))' % len(bad)))
sys.exit(1 if bad else 0)
PY
)

# 2. teacher check: its saved args (recipe, person centre, bone fix, head width), optionally writes the teacher's fix as
#    a JSON for the student's --bone_fix and a small arch record. Needs torch (+ the repo's bone_fix.py for --write_fix).
TEACHER_CHECK_PY=$(cat <<'PY'
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True)
ap.add_argument('--labels', required=True)
ap.add_argument('--head_dim', type=int, required=True)
ap.add_argument('--init_heads', type=int, default=1)
ap.add_argument('--alpha_dist', type=float, required=True)
ap.add_argument('--alpha_transl', type=float, required=True)
ap.add_argument('--mask_hands', type=int, required=True)
ap.add_argument('--mask_shape', type=int, required=True)
ap.add_argument('--allow_mismatch', type=int, default=0)
ap.add_argument('--write_fix', default='')
ap.add_argument('--write_json', default='')
ap.add_argument('--repo', default='.')
a = ap.parse_args()
sys.path.insert(0, a.repo)
import torch
WIDTH = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024}
bad, mism = [], []
print('[teacher] reading %s (whole checkpoint on the CPU)' % a.teacher, flush=True)
ck = torch.load(a.teacher, map_location='cpu', weights_only=False)
ta = ck.get('args') if isinstance(ck, dict) else None
kw = dict(vars(ta)) if hasattr(ta, '__dict__') else dict(ta or {})
sd = ck.get('model_state_dict', {}) if isinstance(ck, dict) else {}
if not kw:
    bad.append('no saved args in the teacher checkpoint')
n_model = sum(1 for k in sd if k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')))
if n_model == 0:
    bad.append('the teacher state dict has no model.py Model keys (backbone.encoder.*, x_attention_head.*)')
bb = kw.get('backbone')
hd = int(kw.get('head_dim') or 0)
width = hd if hd > 0 else WIDTH.get(bb)
spec = kw.get('bone_fix_spec') or {}
bones = sorted((spec.get('bones') or {}).keys()) if isinstance(spec, dict) else []
print('[teacher] epoch %s; backbone %s, head_dim %s (head width %s), img_size %s; gt_center_joint %s; alpha_dist %s, '
      'alpha_transl %s; mask_3dpw_hands %s, mask_3dpw_shape %s; mix_3dpw_labels %s (ratio %s); exclude_indices_file %s'
      % (ck.get('epoch') if isinstance(ck, dict) else '?', bb, hd, width, kw.get('img_size'), kw.get('gt_center_joint'),
         kw.get('alpha_dist'), kw.get('alpha_transl'), kw.get('mask_3dpw_hands'), kw.get('mask_3dpw_shape'),
         kw.get('mix_3dpw_labels'), kw.get('mix_3dpw_ratio'), kw.get('exclude_indices_file')))
print('[teacher] bone fix in its args: %s' % (', '.join(bones) if bones else 'NONE'))
if kw.get('gt_center_joint') != 'person_center':
    bad.append('teacher gt_center_joint = %r, not person_center: not the final model\'s depth recipe' % kw.get('gt_center_joint'))
if not bones:
    bad.append('teacher args carry no bone fix (bone_fix_spec): not the final model')
if a.init_heads and width is not None and a.head_dim != width:
    bad.append('HEAD_DIM=%d but the teacher heads are %s wide: --init_heads_from_teacher would leave the head input layers '
               'random (set HEAD_DIM=%s)' % (a.head_dim, width, width))
if kw.get('img_size') not in (672, None):
    bad.append('teacher img_size %s, the student trains at 672 (token grids must match for the feature KD)' % kw.get('img_size'))

def num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d

if num(kw.get('alpha_dist'), 1.0) != a.alpha_dist:
    mism.append('alpha_dist: teacher %s, student %s' % (kw.get('alpha_dist'), a.alpha_dist))
if num(kw.get('alpha_transl'), 1.0) != a.alpha_transl:
    mism.append('alpha_transl: teacher %s, student %s' % (kw.get('alpha_transl'), a.alpha_transl))
if int(kw.get('mask_3dpw_hands') or 0) != a.mask_hands:
    mism.append('mask_3dpw_hands: teacher %s, student %s' % (kw.get('mask_3dpw_hands'), a.mask_hands))
if int(kw.get('mask_3dpw_shape') or 0) != a.mask_shape:
    mism.append('mask_3dpw_shape: teacher %s, student %s (MASK_3DPW_SHAPE)' % (kw.get('mask_3dpw_shape'), a.mask_shape))
tl = kw.get('mix_3dpw_labels') or ''
if not tl or os.path.realpath(tl) != os.path.realpath(a.labels):
    mism.append('3DPW labels: teacher %r, student %s' % (tl, a.labels))
if not kw.get('exclude_indices_file'):
    print('WARNING: the teacher args name no exclude_indices_file; the student excludes the manifest images anyway')
for m in mism:
    print(('WARNING (ALLOW_RECIPE_MISMATCH=1): ' if a.allow_mismatch else 'REFUSED: recipe differs from the teacher: ') + m)
if a.write_fix and bones and not bad:
    import bone_fix as bfl
    meta = {k: v for k, v in spec.items() if k not in ('bones', 'format')}
    meta.update(teacher_ckpt=a.teacher, note="the teacher's own fix (its checkpoint args, bone_fix_spec), written by "
                "exp8 for the student's training: the student's heads start as copies of the teacher's")
    bfl.save(a.write_fix, spec['bones'], **meta)
    print('[teacher] wrote %s (%d bones)' % (a.write_fix, len(bones)))
if a.write_json:
    json.dump(dict(teacher=a.teacher, epoch=ck.get('epoch') if isinstance(ck, dict) else None, backbone=bb, head_dim=hd,
                   head_width=width, bones=bones, gt_center_joint=kw.get('gt_center_joint'),
                   alpha_dist=kw.get('alpha_dist'), alpha_transl=kw.get('alpha_transl'),
                   mask_3dpw_hands=kw.get('mask_3dpw_hands'), mask_3dpw_shape=kw.get('mask_3dpw_shape'),
                   mix_3dpw_labels=tl, exclude_indices_file=kw.get('exclude_indices_file'),
                   recipe_mismatches=mism, refused=bad), open(a.write_json, 'w'), indent=2)
for b in bad:
    print('REFUSED: ' + b)
fail = bool(bad) or (bool(mism) and not a.allow_mismatch)
print('[teacher] RESULT: ' + ('FAIL' if fail else 'PASS'))
sys.exit(1 if fail else 0)
PY
)

# 3. final table (standard library only: runs in the container or on the login node with TABLE_ONLY=1).
#    Arg: <RUN folder>; reads <RUN>/table_inputs.json written by the job.
FINAL_PY=$(cat <<'PY'
import json, os, re, sys
out = sys.argv[1]
inp = json.load(open(os.path.join(out, 'table_inputs.json')))
V = 'V1_eval_masked'
RULE = 0.95


def trace(d):
    if not d:          # no trace planned (e.g. STUDENT_TRAIN_FIX=none): never read a summary.json from the CWD
        return {}
    try:
        s = json.load(open(os.path.join(d, 'summary.json')))
        v = s['variants'][V]
    except Exception:
        return {}
    pl = v.get('placement') or {}
    return dict(pve=v.get('pve'), pa_pve=v.get('pa_pve'), mpjpe=v.get('mpjpe'), recall=v.get('recall'),
                root_err_mm=pl.get('root_err_mm'), ckpt=s.get('ckpt'), bone_fix=s.get('bone_fix'))


def eval_log(p):
    """{dataset key: {metric: value}} from the '***EVAL METRICS - <name>***' blocks of a train.py evaluate() log."""
    res, cur = {}, None
    try:
        lines = open(p, errors='replace').read().replace('\r', '\n').split('\n')
    except Exception:
        return res
    for line in lines:
        m = re.search(r'\*\*\*EVAL METRICS - (.+?)\*\*\*', line)
        if m:
            cur = m.group(1).strip()
            res[cur] = {}
            continue
        if cur is None:
            continue
        m = re.match(r'^\s*- (\w+): (\S+)\s*$', line)
        if m:
            try:
                res[cur][m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.strip():
            cur = None
    return res


def pick(res, prefix):
    keys = [k for k in res if k.startswith(prefix)]
    return res[keys[-1]] if keys else {}


def throughput(p):
    try:
        txt = open(p, errors='replace').read().replace('\r', '\n')
    except Exception:
        return None
    v = [(int(e), float(r)) for e, r in re.findall(r'\[throughput\] epoch (\d+): .*?= [0-9.]+ it/s, ([0-9.]+) images/s', txt)]
    if not v:
        return None
    later = sorted(r for e, r in v if e >= 1) or sorted(r for _, r in v)
    return later[len(later) // 2]


def cost(p):
    try:
        r = json.load(open(p))['results']
    except Exception:
        return {}, {}
    def one(x):
        b = (x.get('bench') or {}).get('full_1person') or {}
        return dict(params=(x.get('params') or {}).get('network'), flops=x.get('total_flops'), ms=b.get('median_ms'),
                    gpu=(x.get('bench') or {}).get('gpu'))
    return (one(r[0]) if len(r) > 0 else {}), (one(r[1]) if len(r) > 1 else {})


T, S = trace(inp['teacher_trace']), trace(inp['student_trace'])
S0 = trace(inp.get('student_trace_trainfix') or '')
te, se = eval_log(inp['teacher_eval_log']), eval_log(inp['student_eval_log'])
TA, SA = pick(te, 'annyone-test'), pick(se, 'annyone-test')
TP, SP = pick(te, 'threedpw_anny-test'), pick(se, 'threedpw_anny-test')
TC, SC = cost(inp['model_cost_json'])
t_ips, s_ips = throughput(inp.get('teacher_train_log') or ''), throughput(inp['student_train_log'])
rows = []   # (section, metric, teacher, student, kind, in_rule); kind: 'err' lower better, 'rec' higher, 'size', 'info'
for k, lab, rule in (('pve', 'PVE', True), ('pa_pve', 'PA-PVE', False), ('mpjpe', 'MPJPE', True), ('recall', 'recall', False)):
    rows.append(('AnnyOne holdout (100 img, V1 trace)', lab, T.get(k), S.get(k), 'rec' if k == 'recall' else 'err', rule))
rows.append(('AnnyOne holdout (100 img, V1 trace)', 'root err mm', T.get('root_err_mm'), S.get('root_err_mm'), 'info', False))
for k, lab in (('pve', 'PVE'), ('pa_pve', 'PA-PVE'), ('mpjpe', 'MPJPE'), ('recall', 'recall')):
    rows.append(('AnnyOne test-100 (evaluate(), V0)', lab, TA.get(k), SA.get(k), 'rec' if k == 'recall' else 'err', False))
for k, lab, rule in (('pve', 'PVE', True), ('pa_pve', 'PA-PVE', False), ('mpjpe', 'MPJPE', True), ('recall', 'recall', False)):
    rows.append(('3DPW test (evaluate(), ' + os.path.basename(os.path.dirname(inp['labels'].rstrip('/'))) + ')', lab,
                 TP.get(k), SP.get(k), 'rec' if k == 'recall' else 'err', rule))
rows.append(('cost', 'params (M, network)', TC.get('params') and TC['params'] / 1e6, SC.get('params') and SC['params'] / 1e6, 'size', True))
rows.append(('cost', 'GFLOPs / image (1 person)', TC.get('flops') and TC['flops'] / 1e9, SC.get('flops') and SC['flops'] / 1e9, 'size', True))
rows.append(('cost', 'inference ms / image', TC.get('ms'), SC.get('ms'), 'size', TC.get('ms') is not None))
rows.append(('cost', 'training images/s (not comparable)', t_ips, s_ips, 'info', False))


def f(x, nd=1):
    return '-' if x is None else format(x, '.%df' % nd)


checks, table = [], []
for sec, lab, t, s, kind, rule in rows:
    ratio = (s / t) if (t not in (None, 0) and s is not None) else None
    if kind == 'err':
        target = None if t is None else t / RULE
        met = None if (target is None or s is None) else s <= target
        tg = '<= ' + f(target)
    elif kind == 'rec':
        target = None if t is None else t * RULE
        met = None if (target is None or s is None) else s >= target
        tg = '>= ' + f(target)
    elif kind == 'size':
        met = None if (t is None or s is None) else s < t
        tg = '< teacher'
    else:
        met, tg = None, ''
    table.append(dict(section=sec, metric=lab, teacher=t, student=s, ratio=ratio, target=tg, met=met, in_rule=rule))
    if rule:
        checks.append((sec + ' ' + lab, met))
smoke = bool(inp.get('smoke'))
print('=== exp8 FINAL TABLE: teacher %s vs student %s' % (inp['teacher'], inp['student']))
print('    student evaluated with bone fix: %s; teacher with its own (checkpoint args)' % inp.get('student_fix'))
if smoke:
    print('    SMOKE TEST: 20 training iterations, %s 3DPW-test images, %s estimation images: NOT a result'
          % (inp.get('eval_3dpw_n'), inp.get('n_est')))
print()
print('| section | metric | teacher | student | student/teacher | 95% target | met? | in the 1b rule |')
print('|---|---|---|---|---|---|---|---|')
for r in table:
    nd = 2 if r['metric'].startswith(('params', 'inference')) else 1
    print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
        r['section'], r['metric'], f(r['teacher'], nd), f(r['student'], nd), f(r['ratio'], 3), r['target'],
        {True: 'yes', False: 'NO', None: '-'}[r['met']], 'yes' if r['in_rule'] else ''))
if S0:
    print()
    print('reference: student with its TRAINING fix (the teacher\'s), holdout V1: PVE %s, PA-PVE %s, MPJPE %s, recall %s '
          '(vs its own fix above: PVE %s)' % (f(S0.get('pve')), f(S0.get('pa_pve')), f(S0.get('mpjpe')), f(S0.get('recall')),
                                               f(S.get('pve'))))
if TC.get('gpu'):
    print('latency: batch 1, fp32 + TF32, full forward with one person, both models on %s' % TC.get('gpu'))
print()
missing = [k for k, m in checks if m is None]
failed = [k for k, m in checks if m is False]
if missing:
    verdict = 'INCOMPLETE (no value for: %s)' % '; '.join(missing)
elif failed:
    verdict = 'FAIL (not met: %s)' % '; '.join(failed)
else:
    verdict = 'PASS: student error <= teacher error / 0.95 for PVE and MPJPE on the AnnyOne holdout and 3DPW test, and smaller'
print('VERDICT (requirement 1b, smaller-model route)%s: %s' % (' [SMOKE: not a result]' if smoke else '', verdict))
if TC.get('params') and SC.get('params') and TC.get('flops') and SC.get('flops'):
    print('size: %.2fx fewer params, %.2fx fewer FLOPs%s' % (TC['params'] / SC['params'], TC['flops'] / SC['flops'],
          (', %.2fx faster' % (TC['ms'] / SC['ms'])) if TC.get('ms') and SC.get('ms') else ''))
print('Teacher numbers are re-measured in this job (or reused from its own run when TEACHER_REMEASURE=0); compare 3DPW '
      'numbers only with models evaluated on the same labels.')
json.dump(dict(inputs=inp, rows=table, verdict=verdict, smoke=smoke), open(os.path.join(out, 'final_table.json'), 'w'), indent=2)
print('wrote ' + os.path.join(out, 'final_table.json'))
PY
)

# ---------------------------------------------------------------- bash checks (paths; no python)
bash_checks () {
  local fail=0 f s real
  [ -f "${TEACHER}" ] || { echo "REFUSED: teacher not found: ${TEACHER}"; fail=1; }
  if [ "${STUDENT_INIT}" != dinov2 ] && [ ! -f "${STUDENT_INIT}" ]; then
    echo "REFUSED: student backbone init not found: ${STUDENT_INIT}. On the login node (internet):"
    echo "           cd ${REPO}/models/multiHMR && wget https://download.europe.naverlabs.com/ComputerVision/MultiHMR/${NAVER_INIT}"
    echo "         or STUDENT_INIT=dinov2 (ImageNet DINOv2 init; v4 showed the HMR-trained init is better)"
    fail=1
  fi
  f="${TORCH_CACHE}/hub/checkpoints/${STUDENT_BACKBONE}_pretrain.pth"
  if [ ! -s "${f}" ]; then
    echo "REFUSED: ${f} is missing: train.py builds every student with the ImageNet DINOv2 weights first"
    echo "         (Model(pretrained_backbone=1)) and compute nodes cannot download. On the login node:"
    echo "           mkdir -p ${TORCH_CACHE}/hub/checkpoints && wget -nc -P ${TORCH_CACHE}/hub/checkpoints https://dl.fbaipublicfiles.com/dinov2/${STUDENT_BACKBONE}/${STUDENT_BACKBONE}_pretrain.pth"
    fail=1
  fi
  [ -d "${TORCH_CACHE}/hub/facebookresearch_dinov2_main" ] \
    || { echo "REFUSED: no DINOv2 hub code under ${TORCH_CACHE}/hub (slurm/env.sh fetches it on first use)"; fail=1; }
  [ -f "$(abs "${MANIFEST}")" ] || { echo "REFUSED: $(abs "${MANIFEST}") missing: the 55 manifest images cannot be excluded (teacher recipe)"; fail=1; }
  case ",${PARTITIONS}," in
    *V100*|*B200*|*RTXB6000*|*H100*|*H200*)
      echo "REFUSED: PARTITIONS=${PARTITIONS}: training runs only on sm 8.0-8.6 GPUs (torch 2.0.1+cu117; no V100/B200/RTXB6000/H100/H200)"
      fail=1 ;;
  esac
  for s in init_heads_from_teacher head_dim freeze_heads_epochs teacher_bf16 lambda_kd_out gt_center_joint \
           exclude_indices_file mask_3dpw_hands bone_fix; do
    grep -q -- "--${s}" "${REPO}/train.py" || { echo "REFUSED: train.py has no --${s} (old copy: git pull)"; fail=1; }
  done
  [ "${MASK_3DPW_SHAPE}" = 1 ] && ! grep -q -- "--mask_3dpw_shape" "${REPO}/loss.py" \
    && { echo "REFUSED: loss.py has no --mask_3dpw_shape (git pull)"; fail=1; }
  grep -q "train_sample" "${REPO}/scripts/diagnostics/diag_holdout_trace.py" \
    || { echo "REFUSED: diag_holdout_trace.py has no --select train_sample (git pull)"; fail=1; }
  for f in scripts/diagnostics/compute_bone_fix.py scripts/analysis/model_cost.py bone_fix.py; do
    [ -f "${REPO}/${f}" ] || { echo "REFUSED: ${f} not found (git pull)"; fail=1; }
  done
  if [ "${EVAL_TEST}" = 1 ]; then
    if [ ! -f "${EVAL_SCRIPT}" ]; then
      echo "REFUSED: ${EVAL_SCRIPT} not found (EVAL_TEST=1 runs it in step 5)"; fail=1
    else
      for s in MODEL_ENTRY EVAL_LABELS EVAL_BONE_FIX EVAL_3DPW_N; do
        grep -q "${s}" "${EVAL_SCRIPT}" || { echo "REFUSED: ${EVAL_SCRIPT} lacks ${s}: an old copy (git pull)"; fail=1; }
      done
    fi
    if [ "${TEACHER_REMEASURE}" = 0 ]; then
      if [ -z "${TEACHER_EVAL_LOG}" ] || [ ! -f "${TEACHER_EVAL_LOG}" ]; then
        echo "REFUSED: TEACHER_REMEASURE=0 but no teacher 3DPW-test log (TEACHER_EVAL_LOG=${TEACHER_EVAL_LOG:-unset})"; fail=1
      elif ! grep -qF "labels: ${LABELS} " "${TEACHER_EVAL_LOG}" || ! grep -qF "  ->  ${TEACHER}  (" "${TEACHER_EVAL_LOG}"; then
        echo "REFUSED: ${TEACHER_EVAL_LOG} is not ${TEACHER} on ${LABELS} (its 'task 0' / 'labels:' lines): re-measure"
        fail=1
      fi
    fi
  fi
  if [ "${TEACHER_REMEASURE}" = 0 ] && [ ! -f "${TEACHER_TRACE_DIR:-/nonexistent}/summary.json" ]; then
    echo "REFUSED: TEACHER_REMEASURE=0 but no teacher holdout trace (TEACHER_TRACE_DIR=${TEACHER_TRACE_DIR:-unset})"; fail=1
  fi
  # ---- v23 labels: exp7_final_model.sh bash_checks, copied unchanged
  real="$(readlink -f "${LABELS}" 2>/dev/null || printf '%s' "${LABELS}")"
  case "${LABELS} ${real}" in
    *threedpw_anny_full*)
      echo "REFUSED: LABELS=${LABELS} (real path ${real}) is under threedpw_anny_full: the OLD labels (v2.2 fits:"
      echo "         gender reversed for Anny, second actor often unfitted). Train only on threedpw_anny_v23/labels."
      fail=1 ;;
  esac
  for s in ${SPLITS//,/ }; do
    f="${LABELS}/index_${s}.txt"
    if [ -s "${f}" ]; then
      echo "    ${f}: $(grep -c . "${f}") label files"
    else
      echo "REFUSED: ${f} is missing or empty: build and check the v23 labels first (handover 4e, steps 3-5)"
      fail=1
    fi
  done
  # every refit task of these splits must have its fit, by exact name, and none may be newer than the split's index
  # (the label builder takes whatever fits exist; a partial refit gives labels with missing actors)
  local sp sq pe nf fit n_exp n_miss n_new miss
  if [ ! -s "${TASKS}" ]; then
    echo "REFUSED: task list ${TASKS} is missing or empty: cannot check that every refit task has its fit"
    fail=1
  else
    for s in ${SPLITS//,/ }; do
      n_exp=0; n_miss=0; n_new=0; miss=""
      while read -r sp sq pe nf || [ -n "${sp:-}" ]; do   # also a last line without a newline
        [ "${sp}" = "${s}" ] || continue
        n_exp=$(( n_exp + 1 ))
        fit="${FITS}/${sp}/${sq}_p${pe}.npz"
        if [ ! -s "${fit}" ]; then
          n_miss=$(( n_miss + 1 )); [ "${n_miss}" -le 8 ] && miss="${miss} ${sq}_p${pe}"
        elif [ -f "${LABELS}/index_${s}.txt" ] && [ "${fit}" -nt "${LABELS}/index_${s}.txt" ]; then
          n_new=$(( n_new + 1 ))
        fi
      done < "${TASKS}"
      if [ "${n_exp}" -eq 0 ]; then
        echo "REFUSED: ${TASKS} lists no task of split ${s}"; fail=1; continue
      fi
      echo "    ${FITS}/${s}: $(( n_exp - n_miss )) of ${n_exp} refit tasks have their fit"
      if [ "${n_miss}" -gt 0 ]; then
        echo "REFUSED: ${n_miss} refit task(s) of ${s} have no fit under ${FITS}/${s}:${miss}$([ "${n_miss}" -gt 8 ] && echo ' ...')"
        echo "         (failed, timed out or not run). Rerun them (same OUT_ROOT / REFIT_FLAGS; tasks with a fit are"
        echo "         skipped), then rebuild the labels. Labels from a partial refit miss actors."
        fail=1
      fi
      if [ "${n_new}" -gt 0 ]; then
        echo "REFUSED: ${n_new} fit(s) of ${s} are newer than ${LABELS}/index_${s}.txt: the labels were built before"
        echo "         these fits existed. Rebuild the labels (run_build_labels.sh with FITS / OUT as before)."
        fail=1
      fi
    done
  fi
  # train.py keeps only the NB_MAX_CKPT highest epoch numbers in its checkpoint folder (exp7's guard)
  local ck old
  ck="${OUT_ABS}/train/checkpoints"
  if compgen -G "${ck}/*.pt" >/dev/null; then
    old="$(cd "${ck}" && ls -1 -- *.pt | tr '\n' ' ')"; old="${old% }"
    if [ "${OVERWRITE}" = 1 ]; then
      echo "    OVERWRITE=1: ${ck} holds ${old}; the job moves ${OUT_ABS} to ${RUN}_replaced_<jobid> before training"
    else
      echo "REFUSED: ${ck} already holds checkpoints (${old}); use another RUN=... or OVERWRITE=1"
      fail=1
    fi
  fi
  return "${fail}"
}

find_check_python () {   # Python 3.8+ with NumPy AND torch (the teacher's args are pickled in its checkpoint)
  local p
  for p in python3 python; do
    if command -v "${p}" >/dev/null 2>&1 \
       && "${p}" -c 'import sys, numpy, torch; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
      echo "${p}"; return 0
    fi
  done
  return 0
}

teacher_check_args () {   # $1 = extra args
  printf '%s' "--teacher ${TEACHER} --labels ${LABELS} --head_dim ${HEAD_DIM} --init_heads ${INIT_HEADS} \
--alpha_dist ${ALPHA_DIST} --alpha_transl ${ALPHA_TRANSL} --mask_hands ${MASK_3DPW_HANDS} --mask_shape ${MASK_3DPW_SHAPE} \
--allow_mismatch ${ALLOW_RECIPE_MISMATCH} --repo ${REPO} $1"
}

print_settings () {
  echo "=== exp8_distill_vitb ${RUN}"
  echo "    teacher ${TEACHER} (tag ${TEACHER_TAG})"
  echo "    student ${STUDENT_BACKBONE}, init ${STUDENT_INIT}, head_dim ${HEAD_DIM} (feat_proj expected: ${PROJ_EXPECTED}),"\
       "heads from teacher ${INIT_HEADS} (frozen ${FREEZE_HEADS_EPOCHS} epochs), training bone fix: ${STUDENT_TRAIN_FIX}"
  echo "    KD: lambda_kd ${LAMBDA_KD}, lambda_kd_out ${LAMBDA_KD_OUT}, T ${KD_T}; targets: person_center, alpha_dist"\
       "${ALPHA_DIST}, alpha_transl ${ALPHA_TRANSL}, 3DPW ${LABELS} x ${MIX_RATIO}, mask hands ${MASK_3DPW_HANDS},"\
       "mask shape ${MASK_3DPW_SHAPE}"
  echo "    ${EPOCHS} epochs x ${N_ITERS} iterations x batch ${BATCH} = $(( MAX_ITER * BATCH )) images, lr ${LR} (x0.5 every"\
       "${LR_DECAY_EVERY} epochs); time limit ${TIME} (allowance ${NEED_MIN} min at 6 images/s)"
  echo "    post: bone fix on ${N_EST} training images (seed ${SEED}, BONES ${BONES}, ON_GLOBAL_RISE ${ON_GLOBAL_RISE});"\
       "3DPW test ${EVAL_TEST} (${EVAL_3DPW_N} images, 0 = all); teacher re-measured ${TEACHER_REMEASURE}; bench ${BENCH}"
  [ "${SMOKE}" = 1 ] && echo "    SMOKE=1: a start-up / pipeline test, not a result"
  echo "    output ${OUT}"
}

write_table_inputs () {   # deterministic paths of every table input (bash only)
  local tt te
  if [ "${TEACHER_REMEASURE}" = 1 ]; then tt="${OUT_ABS}/holdout_teacher"; te="${OUT_ABS}/eval_3dpw_test_teacher.log"
  else tt="${TEACHER_TRACE_DIR}"; te="${TEACHER_EVAL_LOG}"; fi
  local ttl=""; [ -n "${TEACHER_RUN_DIR}" ] && ttl="${TEACHER_RUN_DIR}/train.log"
  mkdir -p "${OUT_ABS}"
  cat > "${OUT_ABS}/table_inputs.json" <<EOF3
{
  "run": "${RUN}",
  "teacher": "${TEACHER}",
  "student": "$(cat "${OUT_ABS}/last_checkpoint.txt" 2>/dev/null || echo "${OUT_ABS}/train/checkpoints/<last>.pt")",
  "student_fix": "$(cat "${OUT_ABS}/student_fix_arg.txt" 2>/dev/null || echo '?')",
  "labels": "${LABELS}",
  "teacher_trace": "${tt}",
  "student_trace": "${OUT_ABS}/holdout_student",
  "student_trace_trainfix": "$([ "${STUDENT_TRAIN_FIX}" = teacher ] && echo "${OUT_ABS}/holdout_student_trainfix" || true)",
  "teacher_eval_log": "${te}",
  "student_eval_log": "${OUT_ABS}/eval_3dpw_test_student.log",
  "model_cost_json": "${OUT_ABS}/model_cost.json",
  "student_train_log": "${OUT_ABS}/train.log",
  "teacher_train_log": "${ttl}",
  "smoke": ${SMOKE},
  "eval_3dpw_n": ${EVAL_3DPW_N},
  "n_est": ${N_EST}
}
EOF3
}

# ================================================================ LOCAL CHECK (anywhere; no cluster paths)
if [ "${LOCAL_CHECK}" = 1 ]; then
  print_settings
  echo "--- LOCAL_CHECK=1: no path / label / teacher / queue checks; composed commands only"
  echo "--- train.py (job step 1):"
  pretty "${TRAIN_CMD}"
  echo "--- post-steps (job steps 2-5):"
  echo "python scripts/diagnostics/diag_holdout_trace.py --ckpt <last> --out ${OUT}/est_trace --select train_sample --n ${N_EST}"\
       "--seed ${SEED} --exclude_manifest ${MANIFEST} --variants V1_eval_masked --n_vis 0 --bone_fix none --dump_bones 1 --dump_bone_names all"
  echo "python scripts/diagnostics/compute_bone_fix.py --trace_dir ${OUT}/est_trace --variant V1_eval_masked --bones ${BONES} --out ${OUT}/bone_fix.json"
  echo "python scripts/diagnostics/diag_holdout_trace.py --ckpt <last> --out ${OUT}/holdout_student --select holdout --n 100 --n_vis ${N_VIS}"\
       "--bone_fix <${OUT}/bone_fix.json | none> --dump_bones 1 --dump_bone_names all"
  [ "${STUDENT_TRAIN_FIX}" = teacher ] && echo "python scripts/diagnostics/diag_holdout_trace.py --ckpt <last> --out ${OUT}/holdout_student_trainfix --select holdout --n 100 --n_vis 0 --dump_bones 1 --dump_bone_names all"
  [ "${TEACHER_REMEASURE}" = 1 ] && echo "python scripts/diagnostics/diag_holdout_trace.py --ckpt ${TEACHER} --out ${OUT}/holdout_teacher --select holdout --n 100 --n_vis 0 --dump_bones 1 --dump_bone_names all"
  echo "python scripts/analysis/model_cost.py --ckpt ${TEACHER} --ckpt <last>$([ "${BENCH}" = 1 ] && echo ' --bench' || true) --json ${OUT}/model_cost.json"
  if [ "${EVAL_TEST}" = 1 ]; then
    echo "MODEL_ENTRY=\"<last>|${HEAD_DIM}|${STUDENT_BACKBONE}|${STUDENT_LABEL}\" EVAL_LABELS=${LABELS} EVAL_BONE_FIX=<bone_fix.json | none> EVAL_3DPW_N=${EVAL_3DPW_N} bash slurm/eval/run_eval_3dpw_test.sh"
    [ "${TEACHER_REMEASURE}" = 1 ] && echo "MODEL_ENTRY=\"${TEACHER}|<teacher head_dim>|<teacher backbone>|${TEACHER_LABEL}\" EVAL_LABELS=${LABELS} EVAL_BONE_FIX= EVAL_3DPW_N=${EVAL_3DPW_N} bash slurm/eval/run_eval_3dpw_test.sh"
  fi
  echo "--- sbatch (submit mode): sbatch --export=ALL,EXP8_JOB=1 --job-name=exp8-${RUN} --time=${TIME} --partition=${PARTITIONS} --exclude=gifu,glendale ${SELF}"
  PYC=""
  for p in python3 python; do command -v "${p}" >/dev/null 2>&1 && { PYC="${p}"; break; }; done
  if [ -n "${PYC}" ]; then
    for blk in LABEL_CHECK_PY TEACHER_CHECK_PY FINAL_PY; do
      printf '%s\n' "${!blk}" | "${PYC}" -c 'import sys; compile(sys.stdin.read(), sys.argv[1], "exec")' "${blk}" \
        && echo "    embedded ${blk}: compiles" || { echo "ERROR: embedded ${blk} does not compile"; exit 1; }
    done
  fi
  echo "LOCAL_CHECK=1: nothing checked against the cluster, nothing submitted"
  exit 0
fi

# ================================================================ TABLE-ONLY MODE (login node, no Slurm)
if [ "${TABLE_ONLY}" = 1 ]; then
  echo "=== exp8_distill_vitb TABLE_ONLY: ${OUT}"
  [ -f "${OUT_ABS}/table_inputs.json" ] || { echo "ERROR: ${OUT_ABS}/table_inputs.json not found (same RUN / settings as the run?)"; exit 1; }
  PYT=""
  for p in python3 python; do
    command -v "${p}" >/dev/null 2>&1 && "${p}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' \
      >/dev/null 2>&1 && { PYT="${p}"; break; }
  done
  [ -n "${PYT}" ] || { echo "ERROR: no Python 3 found"; exit 1; }
  write_table_inputs   # refresh (student / fix lines from the run folder)
  printf '%s\n' "${FINAL_PY}" | "${PYT}" - "${OUT_ABS}"
  exit $?
fi

# ================================================================ which mode
MODE=submit
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ "${EXP8_JOB:-}" = 1 ]; then
    MODE=job
  elif [ "${CHECK_ONLY}" != 1 ] && [ "${DRY_RUN}" != 1 ]; then
    echo "REFUSED: SLURM_JOB_ID=${SLURM_JOB_ID} is set but EXP8_JOB is not (an salloc / srun shell?). Run it from a"
    echo "         login shell, or 'sbatch ${SELF}'. CHECK_ONLY=1 or DRY_RUN=1 work here. Nothing trained, nothing submitted."
    exit 1
  fi
fi

# ================================================================ SUBMIT MODE
if [ "${MODE}" = submit ]; then
  print_settings
  echo "--- pre-flight on $(hostname 2>/dev/null || echo this host)"
  bash_checks || { echo "NOT SUBMITTED"; exit 1; }
  if command -v squeue >/dev/null 2>&1; then
    QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n "exp8-${RUN}" -o '%i %T' 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${QUEUED// /}" ]; then
      echo "REFUSED: a job named exp8-${RUN} is already queued or running (${QUEUED% }): it writes ${OUT}."
      echo "NOT SUBMITTED"; exit 1
    fi
    QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n exp8-distill-vitb -o '%i %T' 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${QUEUED// /}" ] && [ "${IGNORE_PLAIN_SBATCH}" != 1 ]; then
      echo "REFUSED: job(s) named exp8-distill-vitb (a plain sbatch of this file, RUN unknown) are queued or running"
      echo "         (${QUEUED% }). Check logs/exp8_distill_vitb_<jobid>.out line 1, then IGNORE_PLAIN_SBATCH=1 if it writes another RUN."
      echo "NOT SUBMITTED"; exit 1
    fi
  fi
  PYCHK="$(find_check_python)"
  if [ -z "${PYCHK}" ]; then
    echo "REFUSED: no Python 3.8+ with NumPy and torch here (tried python3, python): the label and teacher checks cannot run."
    echo "         Run 'conda activate multihmr' first. (Plain sbatch ${SELF} skips this pre-flight; the job repeats it.)"
    echo "NOT SUBMITTED"; exit 1
  fi
  echo "    checks with $(command -v "${PYCHK}")"
  if ! printf '%s\n' "${LABEL_CHECK_PY}" | "${PYCHK}" - --labels "${LABELS}" --fits "${FITS}" --splits "${SPLITS}" \
      --n_sample "${N_LABEL_CHECK}" --base "${REPO}"; then
    echo "NOT SUBMITTED"; exit 1
  fi
  # shellcheck disable=SC2046
  if ! printf '%s\n' "${TEACHER_CHECK_PY}" | "${PYCHK}" - $(teacher_check_args ""); then
    echo "NOT SUBMITTED"; exit 1
  fi
  [ "${CHECK_ONLY}" = 1 ] && { echo "CHECK_ONLY=1: every check passed; nothing submitted"; exit 0; }
  CMD=(sbatch --export=ALL,EXP8_JOB=1 --job-name="exp8-${RUN}" --time="${TIME}" --partition="${PARTITIONS}"
       --exclude=gifu,glendale "${SELF}")
  echo "--- submit: ${CMD[*]}   (from ${PWD}; the settings above travel as environment variables)"
  [ "${DRY_RUN}" = 1 ] && { echo "DRY_RUN=1: nothing submitted"; exit 0; }
  [ -f "${SELF}" ] || { echo "ERROR: run this from the repo root (${SELF} not found in ${PWD})"; exit 1; }
  unset PY_VENV
  exec "${CMD[@]}"
fi

# ================================================================ JOB MODE
mkdir -p "${REPO}/logs"
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"
unset PY_VENV
export CONDA_ENV=multihmr
print_settings
echo "--- job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-$(hostname)}; checks before any GPU work"
bash_checks || { echo "REFUSED: nothing trained"; exit 1; }
if [ "${OVERWRITE}" = 1 ] && [ -e "${OUT_ABS}" ]; then
  mv "${OUT_ABS}" "${OUT_ABS}_replaced_${SLURM_JOB_ID}"
  echo "OVERWRITE=1: moved the old ${OUT} to ${OUT}_replaced_${SLURM_JOB_ID}"
fi
mkdir -p "${OUT_ABS}"
rm -f "${OUT_ABS}/last_checkpoint.txt" "${OUT_ABS}/student_fix_arg.txt" "${OUT_ABS}/final_table.json"
LIMIT="$(squeue -h -j "${SLURM_JOB_ID}" -o %l 2>/dev/null || true)"
echo "    time limit ${LIMIT:-unknown} (allowance ${NEED_MIN} min)"

COMMAND=$(cat <<EOF2
set -euo pipefail
unset PY_VENV; export CONDA_ENV=multihmr
source ${REPO}/slurm/env.sh
source ${REPO}/slurm/egl.sh
python -c 'import sys, torch, xformers, torchvision; c = torch.cuda.get_device_capability(0); print("[env]", torch.cuda.get_device_name(0), "compute capability %d.%d" % c, "| torch", torch.__version__, "| xformers", xformers.__version__); sys.exit(0 if (8, 0) <= c < (9, 0) else 3)' \\
    || { echo "REFUSED: this GPU is not compute capability 8.0-8.9 (no V100 / H100 / B200 for this env), or env multihmr broken"; exit 1; }
mkdir -p "${OUT}"

echo "=== 0. CHECKS (inside the job)"
python - --labels "${LABELS}" --fits "${FITS}" --splits "${SPLITS}" --n_sample ${N_LABEL_CHECK} --base "${REPO}" <<'PYCHECK' \\
    || { echo "REFUSED: label check failed; nothing trained"; exit 1; }
${LABEL_CHECK_PY}
PYCHECK
python - $(teacher_check_args "--write_fix ${OUT_ABS}/teacher_bone_fix.json --write_json ${OUT_ABS}/teacher_check.json") <<'PYCHECK' \\
    || { echo "REFUSED: teacher check failed; nothing trained"; exit 1; }
${TEACHER_CHECK_PY}
PYCHECK
TEACHER_BB=\$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["backbone"])' "${OUT_ABS}/teacher_check.json")
TEACHER_HD=\$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["head_dim"])' "${OUT_ABS}/teacher_check.json")
NB_T=\$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["bones"]))' "${OUT_ABS}/teacher_check.json")
echo "\${TEACHER_BB} \${TEACHER_HD}" > "${OUT}/teacher_arch.txt"
python - "${MANIFEST}" "${OUT_ABS}/exclude_manifest_indices.txt" <<'PY'
import json, sys
man, out = sys.argv[1], sys.argv[2]
idx = sorted({int(e['idx']) for e in json.load(open(man))['entries']})
with open(out, 'w') as f:
    f.write('# AnnyOne indices of the diag-manifest images (%d), from %s; never sampled (as the teacher, exp7)\n' % (len(idx), man))
    f.write('\n'.join(str(i) for i in idx) + '\n')
print('wrote', out, ':', len(idx), 'images excluded')
PY
TRAIN_CMD="${TRAIN_CMD}"
{
  echo "exp8_distill_vitb ${RUN}, job ${SLURM_JOB_ID}, \$(date -Is), git \$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "TEACHER=${TEACHER} STUDENT_BACKBONE=${STUDENT_BACKBONE} STUDENT_INIT=${STUDENT_INIT} HEAD_DIM=${HEAD_DIM} INIT_HEADS=${INIT_HEADS}"
  echo "STUDENT_TRAIN_FIX=${STUDENT_TRAIN_FIX} EPOCHS=${EPOCHS} N_ITERS=${N_ITERS} BATCH=${BATCH} LR=${LR} LR_DECAY_EVERY=${LR_DECAY_EVERY}"
  echo "LAMBDA_KD=${LAMBDA_KD} LAMBDA_KD_OUT=${LAMBDA_KD_OUT} KD_T=${KD_T} FREEZE_HEADS_EPOCHS=${FREEZE_HEADS_EPOCHS} START_2D_EPOCH=${START_2D_EPOCH}"
  echo "ALPHA_DIST=${ALPHA_DIST} ALPHA_TRANSL=${ALPHA_TRANSL} LABELS=${LABELS} MIX_RATIO=${MIX_RATIO} MASK_3DPW_HANDS=${MASK_3DPW_HANDS} MASK_3DPW_SHAPE=${MASK_3DPW_SHAPE}"
  echo "N_EST=${N_EST} SEED=${SEED} BONES=${BONES} ON_GLOBAL_RISE=${ON_GLOBAL_RISE} EVAL_TEST=${EVAL_TEST} EVAL_3DPW_N=${EVAL_3DPW_N} TEACHER_REMEASURE=${TEACHER_REMEASURE} SMOKE=${SMOKE} TIME=${TIME}"
  echo "\${TRAIN_CMD}"
} > "${OUT}/settings.txt"

echo "=== 1. TRAINING -> ${OUT}/train.log"
T0=\$(date +%s)
eval "\${TRAIN_CMD}" > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s"
tr '\r' '\n' < "${OUT}/train.log" > "${OUT}/train_lines.log"
grep -E "^\[distill\]|^\[pretrained\]|^\[bone_fix\]|^\[mixed\]|^\[age_balance\]|^\[loss\] mask_3dpw|^\[speed\]|^Loaded .*3DPW-Anny|^3DPW-Anny val" "${OUT}/train_lines.log" || true
echo "--- throughput and in-training evaluation (V0; student with its training fix)"
grep -E "\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall): |Error|Traceback" "${OUT}/train_lines.log" || true
echo "NaN/Inf gradient steps skipped: \$(grep -c 'NaN/Inf GRADIENTS' "${OUT}/train_lines.log" || true)"
MISSING=""
grep -qE "^\[distill\] teacher loaded from" "${OUT}/train_lines.log" || MISSING="\${MISSING} '[distill] teacher loaded from';"
if [ "${INIT_HEADS}" = 1 ] && ! grep -qE "^\[distill\] heads initialised from teacher" "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[distill] heads initialised from teacher';"
fi
if [ "${PROJ_EXPECTED}" = 1 ] && ! grep -qE "^\[distill\] projector-fed heads" "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[distill] projector-fed heads' (no feat_proj);"
fi
if [ "${STUDENT_INIT}" != dinov2 ] && ! grep -qE "^\[pretrained\] backbone: " "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[pretrained] backbone:' (student init not loaded);"
fi
grep -qE "^\[mixed\] training on" "${OUT}/train_lines.log" || MISSING="\${MISSING} '[mixed] training on';"
if [ "${MASK_3DPW_HANDS}" = 1 ] && ! grep -qE "^\[loss\] mask_3dpw_hands" "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[loss] mask_3dpw_hands';"
fi
if [ "${MASK_3DPW_SHAPE}" = 1 ] && ! grep -qE "^\[loss\] mask_3dpw_shape:" "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[loss] mask_3dpw_shape:';"
fi
NB_ON=\$(grep -cE "^\[bone_fix\] ON, \${NB_T} bone\(s\)" "${OUT}/train_lines.log" || true)
if [ "${STUDENT_TRAIN_FIX}" = teacher ]; then
  [ "\${NB_ON}" -ge 2 ] || MISSING="\${MISSING} two '[bone_fix] ON, \${NB_T} bone(s)' lines (student + teacher; found \${NB_ON});"
  grep -q "but the student does not" "${OUT}/train_lines.log" && MISSING="\${MISSING} (train.py warned: the student trains WITHOUT the teacher's fix);"
fi
if [ -n "\${MISSING}" ]; then
  echo "ERROR: train.log lacks:\${MISSING}"
  echo "       Not the planned student; stopping before the post-steps."
  exit 1
fi
grep -qE "^\[age_balance\] exclusion-only" "${OUT}/train_lines.log" \\
    || echo "WARNING: no '[age_balance] exclusion-only' line: the manifest images may have been sampled"
LAST=\$(ls "${OUT_ABS}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint"; exit 1; }
echo "last checkpoint: \${LAST}"

trace () {   # \$1 = checkpoint, \$2 = output dir, rest = extra args
  local CK="\$1" OD="\$2"; shift 2
  local T1=\$(date +%s)
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${CK}" --out "\${OD}" "\$@" > "\${OD}.log" 2>&1 \\
      || { echo "ERROR: trace \${OD} failed; tail:"; tail -n 30 "\${OD}.log"; return 1; }
  echo "    \${OD}: done in \$(( \$(date +%s) - T1 )) s (log \${OD}.log)"
  grep -E "^\[load_model\] bone_fix|^\[bone_fix\]|^\[select\]" "\${OD}.log" || true
}

echo "=== 2. THE STUDENT'S OWN BONE FIX: ${N_EST} seeded training images, raw head output -> ${OUT}/bone_fix.json"
rm -f "${OUT}/bone_fix.json" "${OUT}/bone_fix_guarded.json"
trace "\${LAST}" "${OUT}/est_trace" --select train_sample --n ${N_EST} --seed ${SEED} --exclude_manifest "${MANIFEST}" \\
    --variants V1_eval_masked --n_vis 0 --bone_fix none --dump_bones 1 --dump_bone_names all
set +e
python scripts/diagnostics/compute_bone_fix.py --trace_dir "${OUT}/est_trace" --variant V1_eval_masked \\
    --bones "${BONES}" --out "${OUT}/bone_fix.json"
RC=\$?
set -e
STUDENT_FIX="${OUT_ABS}/bone_fix.json"
case "\${RC}" in
  0) ;;
  3) echo "NO FLIPPED BONE in the student's raw output (no bone with a FLIP/CONST verdict and median > 90 deg): no fix;"
     echo "    the student is evaluated with --bone_fix none"
     python - "${OUT}/bone_fix.json" "\${LAST}" <<'PY'
import sys, datetime
import bone_fix as bfl
bfl.save(sys.argv[1], {}, status='NO_TARGET', source=dict(ckpt=sys.argv[2]),
         note='compute_bone_fix.py exit 3: no bone qualifies on the estimation images; evaluated with --bone_fix none',
         created=datetime.datetime.now().isoformat(timespec='seconds'))
print('wrote', sys.argv[1], '(no bones)')
PY
     STUDENT_FIX=none ;;
  4) SAFE=\$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["global_guard"]["safe_subset"]))' "${OUT}/bone_fix.json")
     echo "WARNING: GLOBAL GUARD (exit 4): a compensated chain; safe subset: \${SAFE:-NONE}; ON_GLOBAL_RISE=${ON_GLOBAL_RISE}"
     if [ "${ON_GLOBAL_RISE}" = safe ]; then
       mv "${OUT}/bone_fix.json" "${OUT}/bone_fix_guarded.json"
       if [ -n "\${SAFE}" ]; then
         python scripts/diagnostics/compute_bone_fix.py --trace_dir "${OUT}/est_trace" --variant V1_eval_masked \\
             --bones "\${SAFE}" --out "${OUT}/bone_fix.json" || { echo "ERROR: re-estimation with \${SAFE} failed"; exit 1; }
       else
         echo "    every selected bone is on a compensated chain: no fix (--bone_fix none)"; STUDENT_FIX=none
         cp "${OUT}/bone_fix_guarded.json" "${OUT}/bone_fix.json"
       fi
     else
       echo "    continuing with the full fix (as the teacher's own exp6 fix)"
     fi ;;
  *) echo "ERROR: compute_bone_fix.py exit \${RC}"; exit 1 ;;
esac
echo "\${STUDENT_FIX}" > "${OUT}/student_fix_arg.txt"
python - "${OUT}/bone_fix.json" "${OUT_ABS}/teacher_bone_fix.json" <<'PY' || true
import json, sys
import numpy as np
import bone_fix as bfl
s = json.load(open(sys.argv[1])).get('bones') or {}
try:
    t = json.load(open(sys.argv[2])).get('bones') or {}
except Exception:
    t = {}
print('student fix: ' + (', '.join('%s %.1f deg' % (n, bfl.angle_deg(np.array(C))) for n, C in s.items()) or 'none'))
print('teacher fix: ' + (', '.join('%s %.1f deg' % (n, bfl.angle_deg(np.array(C))) for n, C in t.items()) or 'none'))
for n in sorted(set(s) & set(t)):
    print('    %s: student vs teacher correction differ by %.1f deg' % (n, float(bfl.angle_between_deg(np.array(s[n]), np.array(t[n])))))
only_s, only_t = sorted(set(s) - set(t)), sorted(set(t) - set(s))
if only_s:
    print('    only in the student fix: ' + ', '.join(only_s))
if only_t:
    print('    only in the teacher fix: ' + ', '.join(only_t))
PY

echo "=== 3. HOLDOUT TRACES (100 AnnyOne holdout images, V0-V3)"
if [ "\${STUDENT_FIX}" = none ]; then SFIX_ARG="none"; else SFIX_ARG="\${STUDENT_FIX}"; fi
trace "\${LAST}" "${OUT}/holdout_student" --select holdout --n 100 --n_vis ${N_VIS} --bone_fix "\${SFIX_ARG}" \\
    --dump_bones 1 --dump_bone_names all
if [ "${STUDENT_TRAIN_FIX}" = teacher ]; then
  trace "\${LAST}" "${OUT}/holdout_student_trainfix" --select holdout --n 100 --n_vis 0 --dump_bones 1 --dump_bone_names all
fi
if [ "${TEACHER_REMEASURE}" = 1 ]; then
  trace "${TEACHER}" "${OUT}/holdout_teacher" --select holdout --n 100 --n_vis 0 --dump_bones 1 --dump_bone_names all
fi
for D in holdout_student holdout_student_trainfix holdout_teacher; do
  [ -f "${OUT}/\${D}.log" ] || continue
  echo "--- \${D}:"
  grep -A 12 "=== PER-VARIANT METRICS" "${OUT}/\${D}.log" | grep -E "^variant|^V1_eval_masked" || true
done

echo "=== 4. MODEL COST -> ${OUT}/model_cost.json"
if python scripts/analysis/model_cost.py --ckpt "${TEACHER}" --ckpt "\${LAST}" $([ "${BENCH}" = 1 ] && echo --bench || true) \\
    --json "${OUT}/model_cost.json" > "${OUT}/model_cost.log" 2>&1; then
  { grep -vE "^Using cache|^\[model\]|Warning|warn" "${OUT}/model_cost.log" || true; } | tail -n 14
else
  echo "WARNING: model_cost.py failed (the table's cost rows stay empty); tail:"; tail -n 20 "${OUT}/model_cost.log"
fi
echo "\${LAST}" > "${OUT}/last_checkpoint.txt"
EOF2
)

echo "=== job step: checks, training, bone fix, traces, cost (container)"
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="${REPO}" \
  bash -c "$COMMAND"

LAST_ABS="$(cat "${OUT_ABS}/last_checkpoint.txt")"
STUDENT_FIX="$(cat "${OUT_ABS}/student_fix_arg.txt")"
read -r TEACHER_BB TEACHER_HD < "${OUT_ABS}/teacher_arch.txt"
if [ "${STUDENT_FIX}" = none ]; then FIX_DESC="forced off (--bone_fix none)"
else FIX_DESC="${STUDENT_FIX} (replaces the checkpoint's own)"; fi

# 5. the test evaluations (each its own srun step inside this allocation, as exp7 step 4); the log must be THIS model on
#    THESE labels with THIS fix
eval_one () {   # $1 label, $2 ckpt, $3 head_dim, $4 backbone, $5 EVAL_BONE_FIX value, $6 expected fix text, $7 log
  local log="$7"
  echo "--- ${1}: ${2} (bone fix ${5:-inherit})"
  MODEL_ENTRY="${2}|${3}|${4}|${1}" EVAL_LABELS="${LABELS}" EVAL_BONE_FIX="${5}" EVAL_3DPW_N="${EVAL_3DPW_N}" \
    EVAL_MODEL_CLASS=model bash "${EVAL_SCRIPT}" > "${log}" 2>&1 \
    || { echo "ERROR: evaluation ${1} failed; tail of ${log}:"; tail -n 30 "${log}"; return 1; }
  if ! grep -qF "task 0: ${1}  ->  ${2}" "${log}" || ! grep -qF "labels: ${LABELS} " "${log}" \
     || ! grep -qF "bone fix: ${6}" "${log}"; then
    mv "${log}" "${log%.log}.NOT_THIS_RUN.log"
    echo "ERROR: ${log} lacks 'task 0: ${1}  ->  ${2}', 'labels: ${LABELS}' or 'bone fix: ${6}': renamed .NOT_THIS_RUN.log"
    return 1
  fi
  tr '\r' '\n' < "${log}" | grep -E "^task |^labels: |^bone fix: |^\[bone_fix\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall): " || true
}
if [ "${EVAL_TEST}" = 1 ]; then
  echo "=== 5. TEST EVALUATION (slurm/eval/run_eval_3dpw_test.sh: AnnyOne test-100 + 3DPW-test of ${LABELS}, ${EVAL_3DPW_N} images, 0 = all)"
  eval_one "${STUDENT_LABEL}" "${LAST_ABS}" "${HEAD_DIM}" "${STUDENT_BACKBONE}" "${STUDENT_FIX}" "${FIX_DESC}" \
    "${OUT_ABS}/eval_3dpw_test_student.log" || exit 1
  if [ "${TEACHER_REMEASURE}" = 1 ]; then
    eval_one "${TEACHER_LABEL}" "${TEACHER}" "${TEACHER_HD}" "${TEACHER_BB}" "" "the checkpoint's own" \
      "${OUT_ABS}/eval_3dpw_test_teacher.log" || exit 1
  else
    echo "    teacher: reusing ${TEACHER_EVAL_LOG} (TEACHER_REMEASURE=0)"
  fi
else
  echo "=== 5. test evaluation skipped (EVAL_TEST=0): the table's AnnyOne test-100 and 3DPW rows stay empty"
fi

echo "=== 6. FINAL TABLE -> ${OUT}/final_table.json"
write_table_inputs
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="${REPO}" \
  bash -c "source ${REPO}/slurm/env.sh >/dev/null 2>&1; python - '${OUT_ABS}' <<'PYTABLE'
${FINAL_PY}
PYTABLE" \
  || { echo "WARNING: the table step failed in the container; rebuild it with TABLE_ONLY=1 (same settings) on the login node"; }
echo "=== done: ${OUT}"
