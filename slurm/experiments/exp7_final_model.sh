#!/bin/bash
# exp7_final_model: the planned THESIS MODEL. One fine-tune of v5 00099 that puts the three fixes together:
#   (a) depth anchor fix, exactly as the best exp1 run long_pc_dist5: --gt_center_joint person_center and the direct
#       depth terms x5 (--alpha_dist 5 --alpha_transl 5)
#   (b) exp6 bone fix: --bone_fix with v5's bone_fix.json (neck02, head, metacarpal1.L, finger4-1.R; R_b <- R_b @ C_b,
#       saved in every checkpoint's args and re-applied on load)
#   (c) real 3DPW-train in every batch as in v9 (--mix_3dpw_labels, 30%, train split; 3DPW-validation evaluated each
#       epoch), but from the NEW converter-v2.3 labels (threedpw_anny_v23/labels: Anny gender convention m -> 0.0,
#       f -> 1.0, one frame grid for both actors), plus --mask_3dpw_hands 1 (3DPW finger labels are converter
#       placeholders; docs/handover.md section 5, item 7)
# Everything else is the v5 recipe as in exp6 step c / the exp5 L1 control: backbone frozen except the last 4 blocks,
# helper mask, use_anny_shape, alpha_shape 20, j2d/v2d 0.1, neck/hand boost x4, L1 rotmat 0.1, lr 1e-5, batch 4, and
# the diag-manifest images excluded from sampling. Nothing here has been run yet.
#
# STATUS OF THE PIECES (100-image AnnyOne holdout, V1, matched people):
#   [Certain] each fix was measured ALONE: depth fix long_pc_dist5: root error 0.54 m (v5 1.21 m), size 0.84 (v5 0.63),
#     depth slope 0.38 (v5 0.33, still far from 1), PVE 74.77 (v5 70.5) (3572482). Bone fix, exp6 fine-tuned 2 epochs:
#     neck02 4.7 deg, head 11.2 deg, girth 0.997 (v5 175.6, 177.9, 0.461), PVE 68.8 (3579014). 3DPW mix: v9 e62
#     3DPW-test PVE 102.5 vs v5 119.1, against the OLD labels (handover table, [Likely] run_eval_3dpw_test.sh).
#   [Unknown] whether the three keep their effects when combined: never run together. This job measures it.
#
# PREREQUISITES (do not submit before all four):
#   1. the refit check task passed: RESULT: PASS from
#        python3 scripts/data/check_refit_task.py --split train --seq courtyard_captureSelfies_00 \
#            --new_root threedpw_anny_v23 --old_root threedpw_anny_full --seq_root data/3DPW/sequenceFiles
#   2. all 87 refit tasks finished (87 .npz under threedpw_anny_v23/{train,validation,test}) and the same check with
#      --seq all prints RESULT: PASS. That check reads only the .npz files that exist: it does NOT notice a missing
#      task. The completeness guard below and run_build_labels.sh (refuses to build while a task has no fit) do.
#   3. v23 labels built (run_build_labels.sh with FITS/OUT = threedpw_anny_v23[/labels]) AFTER all 87 fits exist, and
#      the conversion check (check_3dpw_conversion.sh -> results/diagnostics/check_3dpw_conversion_v23) read
#   4. this file and slurm/eval/run_eval_3dpw_test.sh (its MODEL_ENTRY / EVAL_LABELS overrides) committed in ONE
#      commit and pushed to main, then git pull on the cluster. Step 4 runs the eval script from the repo: the OLD one
#      ignores both overrides and would evaluate v5 00099 on threedpw_anny_full (guarded below, but commit both).
#
# GUARDS: the job REFUSES (exit 1, before any training) when
#   - LABELS, or its real path, is under threedpw_anny_full: the old labels (v2.2 fits: gender reversed for Anny,
#     second actor often unfitted; labels_rebuilt is in there too and comes from the same old fits)
#   - index_train.txt or index_validation.txt (and index_test.txt with EVAL_TEST=1) is missing or empty
#   - a line of an index file contains threedpw_anny_full, or its real folder is not under the real LABELS path (an
#     index copied from the old labels)
#   - results/experiments/exp7_final_model/<RUN>/train/checkpoints already holds .pt files (train.py keeps only the
#     NB_MAX_CKPT highest epoch numbers in that folder, so old 00008/00009 would stay and the new checkpoints would be
#     deleted as they are written). OVERWRITE=1 moves the old <RUN> folder to <RUN>_replaced_<jobid> when the job
#     starts. Also refused in the submit mode: a job named exp7-<RUN> already queued or running, or any job named
#     exp7-final-model (a plain sbatch of this file; its RUN cannot be read from squeue; IGNORE_PLAIN_SBATCH=1 skips
#     this one check when you know it writes another RUN)
#   - a refit task of configs/threedpw_tasks.txt (SPLIT SEQ PERSON N, 87 lines) in the splits used (train, validation;
#     test with EVAL_TEST=1) has no fit FITS/<split>/<seq>_p<person>.npz (exact names: a leftover .npz.tmp.npz never
#     counts), or a fit is newer than that split's index file (labels built before the refit finished). A partial
#     refit would otherwise give labels with missing actors, the v2.2 defect
#   - EVAL_TEST=1 and slurm/eval/run_eval_3dpw_test.sh lacks MODEL_ENTRY or EVAL_LABELS (the old script: step 4 would
#     evaluate v5 00099 against threedpw_anny_full and file it as this run's result)
#   - a fit behind the labels (FITS/<split>/<seq>_p<k>.npz; FITS = parent folder of LABELS) was not made with
#     --gender_convention anny and --frame_grid shared, or its Anny gender value does not match the 3DPW gender
#   - in a sample of label files (N_LABEL_CHECK per split, default 300) a person's gender value does not match the
#     3DPW gender of its fit (catches labels built from other fits)
#   - CKPT is missing, or BONE_FIX is missing or was estimated on another checkpoint (its source.ckpt)
#   - the GPU has compute capability 9.0 or higher (torch 2.0.1+cu117 has no kernels for it)
#   After training it stops before the trace and the test evaluation when train.log lacks a line it should have:
#   "[mixed] training on", "[loss] mask_3dpw_hands" (MASK_3DPW_HANDS=1), "[bone_fix] ON, <n> bone(s)" with n = the
#   bones in BONE_FIX (ARM=fix); a missing "[age_balance] exclusion-only" line only prints a WARNING.
#   After step 4 it fails unless eval_3dpw_test.log shows "task 0: exp7_<RUN>  ->  <last checkpoint>" and
#   "labels: <LABELS>"; a log without them is renamed eval_3dpw_test.NOT_THIS_RUN.log.
#   MODES: the submit mode runs when SLURM_JOB_ID is empty; the job mode only when SLURM_JOB_ID is set AND EXP7_JOB=1,
#   which the submit mode passes (sbatch --export=ALL,EXP7_JOB=1) and the #SBATCH --export line sets for a plain
#   sbatch. Run inside an salloc / srun shell (SLURM_JOB_ID set, no EXP7_JOB) it refuses instead of training in that
#   allocation; CHECK_ONLY=1 and DRY_RUN=1 still work there.
#   The bash checks and the label checks run BEFORE sbatch in the submit mode below, so a bad label folder is caught
#   before days in the queue; the job repeats them at start. The label check needs Python 3.8+ with NumPy (the label
#   pickles use the newest pickle protocol): on the login node run `conda activate multihmr` first. Without such a
#   Python the submit mode (CHECK_ONLY=1 included) refuses and exits 1.
#   FIRST REAL USE of --mask_3dpw_hands with --mix_3dpw_labels, --exclude_indices_file and --bone_fix together (no log
#   of this combination exists). When the job starts, read the start of train.log within the first minutes:
#     grep -E "^\[loss\] mask_3dpw_hands|^\[mixed\] training on|^\[age_balance\] exclusion-only|^\[bone_fix\] ON" \
#         results/experiments/exp7_final_model/<RUN>/train.log
#   all four should print (bone_fix: "ON, 4 bone(s)"). Optional smoke run, submitted next to the real one (its RUN
#   differs automatically, it ends with _it20), so whichever starts first shows a startup crash:
#     EPOCHS=1 N_ITERS=20 EVAL_TEST=0 VAL_3DPW_N=20 N_VIS=2 bash slurm/experiments/exp7_final_model.sh
#
# GPUs: only partitions whose GPUs can run the training env. Training needs the normal conda env multihmr (torch 2.0.1
#   +cu117, xformers 0.0.20, torchvision 0.15.2). That torch has kernels for sm_37 to sm_86 only: on an H200 NVL (sm_90)
#   it stopped with "no kernel image is available for execution on the device" [Certain, jobs 3587852/3587853]. The venv
#   with torch 2.1.2+cu121 (PY_VENV, made for the converter on H100) does not fit xformers 0.0.20 / torchvision 0.15.2,
#   so it is not used here: the job unsets PY_VENV and forces CONDA_ENV=multihmr. H100/H200 would need a whole new
#   training env. Default partitions: A100 (8.0), RTXA6000 (8.6), L40S (8.9).
#   [Likely] L40S runs the sm_86 kernels (CUDA runs 8.6 code on any 8.x GPU of 8.6 or newer); no log of ours shows one.
#   NOT in the default list: V100-32GB and RTX3090. [Guessing] this recipe never ran there (batch 4, fp32, ViT-L at
#   672): V100 has no TF32 and may be ~3x slower (10 epochs + trace + the ~9.7k-image test evaluation could pass 5 h;
#   a timeout loses the trace and the test evaluation), and RTX3090 (24 GB) may run out of memory. Both had the same
#   22-26 Oct queue estimate as the others, so leaving them out costs little. To add them anyway:
#   PARTITIONS=A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV,V100-32GB,RTX3090 TIME=08:00:00
#
# USAGE (repo root on the cluster, /netscratch/najib/multi-hmr):
#   conda activate multihmr                                   # login node: Python 3.8+ with NumPy for the label check
#   CHECK_ONLY=1 bash slurm/experiments/exp7_final_model.sh   # guards only, login node, seconds; submits nothing;
#                                                             # exit 1 when a guard fails or the label check cannot run
#   bash slurm/experiments/exp7_final_model.sh                # guards, then sbatch with --time from EPOCHS (10 -> 05:00:00)
#   OVERWRITE=1 bash slurm/experiments/exp7_final_model.sh    # rerun of the same RUN (e.g. after a timeout): the job moves
#                                                             # the old <RUN> folder to <RUN>_replaced_<jobid> first
#   DRY_RUN=1 bash slurm/experiments/exp7_final_model.sh      # guards, then print the sbatch command only
#   EPOCHS=30 bash slurm/experiments/exp7_final_model.sh      # longer run (--time grows: 30 epochs -> 11:00:00)
#   ARM=control bash slurm/experiments/exp7_final_model.sh    # matched control: same labels, schedule and restart,
#                                                             # production targets (root, x1), no bone fix
#   sbatch slurm/experiments/exp7_final_model.sh              # also works: 10 epochs, the #SBATCH --time below; the
#                                                             # guards then run only when the job starts
#   TABLE_ONLY=1 bash slurm/experiments/exp7_final_model.sh   # login node, no Slurm: rebuild only the step-3 table and
#   TABLE_ONLY=1 ARM=control bash slurm/...                   # compare.json of an existing <RUN> (same ARM / EPOCHS /
#                                                             # RUN as its training) with CONTROL_RUN. Use it after
#                                                             # both arms finished: run side by side, the fix job's own
#                                                             # table usually says "no control trace yet"
# ENV (defaults): ARM (fix | control), EPOCHS (10), N_ITERS (1000 iterations per epoch, as the exp1 10-epoch runs;
#   batch 4 = 4,000 images per epoch), RUN (v5_pc_dist5_bonefix_v23_e<EPOCHS>; control v5_root_dist1_nofix_v23_e<EPOCHS>;
#   both get _it<N_ITERS> appended when N_ITERS is not 1000, so a short test run never shares the real run's folder),
#   OVERWRITE (0; 1 = replace an existing <RUN> folder that holds checkpoints),
#   TIME (60 + 60 if EVAL_TEST + 18 per epoch, in minutes), CKPT (v5 00099), BONE_FIX (the exp6 v5 JSON | inherit | none),
#   LABELS (/netscratch/najib/multi-hmr/threedpw_anny_v23/labels), FITS (parent of LABELS), ALPHA_DIST / ALPHA_TRANSL
#   (5 fix, 1 control), LR (1e-5), MIX_RATIO (0.3), VAL_3DPW_N (300), MASK_3DPW_HANDS (1), EVAL_FREQ (1),
#   NB_MAX_CKPT (2), EVAL_TEST (1), N_VIS (12), N_LABEL_CHECK (300), PARTITIONS, CONTROL_RUN (for the table),
#   TASKS (configs/threedpw_tasks.txt), TABLE_ONLY (0), IGNORE_PLAIN_SBATCH (0).
#   Alternative start (handover item 7): CKPT=/netscratch/najib/multi-hmr/models/multiHMR/exp6_v5_bonefix_00001.pt
#   BONE_FIX=inherit (the exp6 fine-tuned checkpoint, its own fix inherited). Default here: v5 00099 + the same JSON.
#
# RUNTIME (realistic, from the logs):
#   [Certain] this recipe on AnnyOne alone: 328-339 s per 1,000-iteration epoch, 11.8-12.2 images/s (long_pc_dist5,
#     3572482:35-114); 10 epochs + 10 holdout evaluations = 3,496 s (3572482:7); exp6 12.7-12.9 images/s (3579014:66,74).
#   [Unknown] the 3DPW batches (1920x1080 JPEGs) and the 300-image 3DPW-validation evaluation per epoch: never timed.
#     v9's own comment guessed 8-10 min per epoch (train_teacher_v9_mixed.sh:16, never measured).
#   Estimate, 10 epochs on an A100 / L40S / RTXA6000 node: ~6-10 min per epoch -> 1-1.7 h of training, + ~2 min trace,
#     + the 3DPW-test evaluation of ~9,800 images, [Guessing] 15-40 min (its script allows 20 min; no log of it here).
#     Total ~1.5-2.5 h. A slow node (6.7 images/s once, 3571598): up to ~2x. The default --time (5 h for 10 epochs)
#     covers ~4 images/s. Queue wait comes on top (A100/L40S/RTXA6000 estimated 22-26 Oct on 17 Oct).
#
# OUTPUTS (results/experiments/exp7_final_model/<RUN>/):
#   settings.txt                 every setting and the exact train.py command
#   train.log, train/checkpoints the last NB_MAX_CKPT epochs; 0000<EPOCHS-1>.pt is the candidate thesis model
#   holdout_ft/, holdout_ft.log  fixed 100-image AnnyOne holdout trace (V0-V3, placement), per-bone dump of every bone,
#                                N_VIS panels (as exp6 step c)
#   compare.json + log table     this run next to v5, v9 e149, long_pc_dist5, exp6 fine-tuned, exp5 control and the
#                                exp7 control arm (columns whose trace folder is missing print "-")
#   eval_3dpw_test.log           slurm/eval/run_eval_3dpw_test.sh on the last checkpoint with EVAL_LABELS=LABELS: AnnyOne
#                                test (last 100) and every 3DPW-test image of the v23 labels (output name
#                                evaltest_exp7_<RUN>__threedpw_anny_v23)
#   exclude_manifest_indices.txt the AnnyOne indices never sampled (diag manifest)
#
# COMPARE AGAINST (holdout V1; reference values with their jobs):
#   v5 00099          PVE 70.5, recall 90.2, root 1.21 m, slope 0.33, size 0.63, girth 0.461, neck02 FLIP 175.6 deg
#   v9 e62 / e149     PVE 75.0 / 77.0, root 1.39 / 1.44 m (3571963, 3571964): 3DPW mixing cost synthetic PVE
#   long_pc_dist5     root 0.54 m, size 0.84, slope 0.38, PVE 74.77, recall 87.80 (depth fix alone, 10 epochs)
#   exp6 fine-tuned   neck02 4.7, head 11.2 deg, girth 0.997, PVE 68.8, recall 89.4 (bone fix alone, 2 epochs)
#   exp5 L1 control   PVE 76.9, recall 91.1, girth 0.461 (same restart, no fix, 2 epochs)
#   ARM=control       same labels, schedule and restart without the two fixes: the fair reference for PVE / recall
#   Read: neck kept = neck02 and head medians < 15 deg and girth >= 0.9 (exp6's rule); depth kept = root error < 0.75 m
#   (the five exp1 fix runs 0.53-0.73 m, controls 1.11-1.44 m); cost = PVE / recall vs ARM=control (exp5 guards: PVE
#   <= control + 8 mm, recall >= control - 2). The table prints these; it does not grade 3DPW.
#   3DPW-test: the old numbers (v5 119.1, v9 e62 102.5) are against threedpw_anny_full and CANNOT be compared with v23.
#   Re-evaluate the baselines on v23 first (any time after the labels exist; one job each, --array=0 is required;
#   one line, v5 00099, v9 e62, exp6 fine-tuned):
#     cd /netscratch/najib/multi-hmr && for M in "logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt|teacher_v5" "models/multiHMR/anny_teacher_v9_mixed_00062.pt|teacher_v9_e62" "models/multiHMR/exp6_v5_bonefix_00001.pt|exp6_v5_bonefix"; do EVAL_LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels MODEL_ENTRY="/netscratch/najib/multi-hmr/${M%%|*}|0|dinov2_vitl14|${M##*|}" sbatch --array=0 slurm/eval/run_eval_3dpw_test.sh; done
#   (EVAL_LABELS, not LABELS: the eval script ignores a LABELS variable; output names get __threedpw_anny_v23 appended)
#   3DPW numbers are against converted labels only (the fits' own joint error, ~14-16 mm mean in the v2.3 test task, is a
#   floor). Say "best on our 3DPW-Anny v23 labels", never "best model".
#SBATCH --job-name=exp7-final-model
#SBATCH --partition=A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp7_final_model_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp7_final_model_%j.err
#SBATCH --export=ALL,EXP7_JOB=1
set -euo pipefail

REPO="${REPO:-/netscratch/najib/multi-hmr}"   # repo root on the cluster; override only for a local dry run
SELF="slurm/experiments/exp7_final_model.sh"
abs () { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "${REPO}/$1" ;; esac; }
posint () { [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $1=$2 (a positive integer)"; exit 1; }; }

# ---------------------------------------------------------------- settings
ARM="${ARM:-fix}"
case "${ARM}" in fix|control) ;; *) echo "ERROR: ARM=${ARM} (fix|control)"; exit 1 ;; esac
CKPT="${CKPT:-${REPO}/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt}"
EPOCHS="${EPOCHS:-10}"
N_ITERS="${N_ITERS:-1000}"
posint EPOCHS "${EPOCHS}"; posint N_ITERS "${N_ITERS}"
MAX_ITER=$(( EPOCHS * N_ITERS ))
ITER_TAG=""; [ "${N_ITERS}" = 1000 ] || ITER_TAG="_it${N_ITERS}"   # in the default RUN names: test runs get their own folder
LABELS="${LABELS:-${REPO}/threedpw_anny_v23/labels}"
LABELS="$(abs "${LABELS%/}")"
FITS="$(abs "${FITS:-$(dirname "${LABELS}")}")"
if [ "${ARM}" = fix ]; then
  GT_CENTER=person_center
  ALPHA_DIST="${ALPHA_DIST:-5}"
  ALPHA_TRANSL="${ALPHA_TRANSL:-5}"
  BONE_FIX="${BONE_FIX:-results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json}"
  RUN="${RUN:-v5_pc_dist${ALPHA_DIST}_bonefix_v23_e${EPOCHS}${ITER_TAG}}"
else
  GT_CENTER=root
  ALPHA_DIST="${ALPHA_DIST:-1}"
  ALPHA_TRANSL="${ALPHA_TRANSL:-1}"
  BONE_FIX="${BONE_FIX:-none}"
  [ "${BONE_FIX}" = none ] || { echo "ERROR: ARM=control runs without a bone fix; BONE_FIX=${BONE_FIX}"; exit 1; }
  RUN="${RUN:-v5_root_dist${ALPHA_DIST}_nofix_v23_e${EPOCHS}${ITER_TAG}}"
fi
LR="${LR:-1e-5}"
MIX_RATIO="${MIX_RATIO:-0.3}"
VAL_3DPW_N="${VAL_3DPW_N:-300}"
MASK_3DPW_HANDS="${MASK_3DPW_HANDS:-1}"
EVAL_FREQ="${EVAL_FREQ:-1}"
NB_MAX_CKPT="${NB_MAX_CKPT:-2}"
EVAL_TEST="${EVAL_TEST:-1}"
N_VIS="${N_VIS:-12}"
N_LABEL_CHECK="${N_LABEL_CHECK:-300}"
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"
PARTITIONS="${PARTITIONS:-A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV}"
CONTROL_RUN="${CONTROL_RUN:-results/experiments/exp7_final_model/v5_root_dist1_nofix_v23_e${EPOCHS}${ITER_TAG}}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
TABLE_ONLY="${TABLE_ONLY:-0}"
IGNORE_PLAIN_SBATCH="${IGNORE_PLAIN_SBATCH:-0}"
TASKS="$(abs "${TASKS:-configs/threedpw_tasks.txt}")"   # the 87 refit tasks: SPLIT SEQ PERSON N
EVAL_SCRIPT="${REPO}/slurm/eval/run_eval_3dpw_test.sh"   # step 4
case "${RUN}" in ''|*/*|.|..) echo "ERROR: RUN=${RUN} (one folder name, no /)"; exit 1 ;; esac
OUT="results/experiments/exp7_final_model/${RUN}"
SPLITS="train,validation"; [ "${EVAL_TEST}" = 1 ] && SPLITS="train,validation,test"
NEED_MIN=$(( 60 + 18 * EPOCHS )); [ "${EVAL_TEST}" = 1 ] && NEED_MIN=$(( NEED_MIN + 60 ))
TIME="${TIME:-$(printf '%02d:%02d:00' $(( NEED_MIN / 60 )) $(( NEED_MIN % 60 )))}"
case "${BONE_FIX}" in
  none)    BONE_FIX_ARG="--bone_fix none" ;;
  inherit) BONE_FIX_ARG="" ;;                         # train.py inherits CKPT's own fix (load_only_backbone 0)
  *)       BONE_FIX_ARG="--bone_fix $(abs "${BONE_FIX}")" ;;
esac
BF_ABS="${BONE_FIX}"; case "${BF_ABS}" in none|inherit) ;; *) BF_ABS="$(abs "${BF_ABS}")" ;; esac
HANDS_ARG=""; [ "${MASK_3DPW_HANDS}" = 1 ] && HANDS_ARG="--mask_3dpw_hands 1"

# ---------------------------------------------------------------- python checks (shared by both modes)
# Label + bone-fix check. Standard library + NumPy; run on the login node (submit mode) and inside the job.
# Exit 0 = pass, 1 = refused (each reason printed as REFUSED), 2 = bad arguments.
CHECK_PY=$(cat <<'PY'
import argparse, glob, json, os, pickle, re, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--fits', required=True)
ap.add_argument('--splits', default='train,validation,test')
ap.add_argument('--n_sample', type=int, default=300)
ap.add_argument('--bone_fix', default='none')
ap.add_argument('--ckpt', default='')
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

def tag_of(ck):   # = exp6_bone_fix.sh / exp6_bone_fix_summary.py tag_of
    if not ck:
        return ''
    d = os.path.dirname(ck)
    b = os.path.basename(ck)
    b = b[:-3] if b.endswith('.pt') else b
    if os.path.basename(d) == 'checkpoints':
        rn = os.path.basename(os.path.dirname(d))
        if rn == 'train':
            rn = os.path.basename(os.path.dirname(os.path.dirname(d)))
        return rn + '_' + b
    return b

# 1. bone fix: the JSON must exist and come from the same checkpoint as --ckpt
if a.bone_fix not in ('none', 'inherit', ''):
    try:
        spec = json.load(open(a.bone_fix))
        src = (spec.get('source') or {}).get('ckpt') or ''
        names = sorted((spec.get('bones') or {}).keys())
        if not names:
            bad.append('bone fix %s lists no bones' % a.bone_fix)
        if src and a.ckpt and tag_of(src) != tag_of(a.ckpt):
            bad.append('bone fix %s was estimated on %s (tag %s), not on CKPT %s (tag %s)'
                       % (a.bone_fix, src, tag_of(src), a.ckpt, tag_of(a.ckpt)))
        print('[check] bone fix %s: %s; estimated on %s' % (a.bone_fix, ', '.join(names), src or '(source.ckpt not recorded)'))
    except Exception as e:
        bad.append('bone fix %s unreadable: %s: %s' % (a.bone_fix, type(e).__name__, e))

# 2. fits behind the labels, 3. sampled label files
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

# Step-3 comparison table (standard library; scripts/experiments/exp6_bone_fix_summary.py readers). Run by the job
# after the trace, and alone with TABLE_ONLY=1. Args: <RUN folder> <CONTROL_RUN> <bone fix | none | inherit> <ARM>.
TABLE_PY=$(cat <<'PY'
import json, os, sys
sys.path.insert(0, 'scripts/experiments')
from exp6_bone_fix_summary import column, f      # same readers as the exp6 summary
out, ctrl_run, bf, arm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
V = 'V1_eval_masked'
cols = [('v5 00099', 'results/diagnostics/neck_bones/anny_s2_shape_v5_00099'),
        ('v9 e149', 'results/diagnostics/neck_bones/anny_teacher_v9_mixed_00149'),
        ('long_pc_dist5', 'results/diagnostics/neck_bones/long_pc_dist5_00009'),
        ('exp6 ft', 'results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/holdout_ft'),
        ('exp5 ctrl', 'results/experiments/exp5_neck_fix/pilot_l1_a0.1_anny_s2_shape_v5_00099/holdout_trace'),
        ('exp7 ctrl', os.path.join(ctrl_run, 'holdout_ft')),
        ('THIS RUN', os.path.join(out, 'holdout_ft'))]
if os.path.normpath(cols[-2][1]) == os.path.normpath(cols[-1][1]):
    cols.pop(-2)                                  # this run is the control arm
bones = ['neck02', 'head', 'metacarpal1.L', 'finger4-1.R']
if bf not in ('none', 'inherit', '') and os.path.isfile(bf):
    bones = list(dict.fromkeys(['neck02', 'head'] + sorted(json.load(open(bf)).get('bones', {}))))
data = {}
for lab, d in cols:
    c = column(d)
    try:
        pl = json.load(open(os.path.join(d, 'summary.json')))['variants'][V].get('placement', {})
    except Exception:
        pl = {}
    data[lab] = dict(dir=d, found=os.path.isfile(os.path.join(d, 'summary.json')), metrics=c['metrics'],
                     placement={k: pl.get(k) for k in ('root_err_mm', 'depth_fit_slope', 'depth_fit_intercept_m',
                                                       'size_ratio_median_px')},
                     bones={b: (c['bones'].get(b) or {}) for b in bones})
rows = [('PVE', lambda x: x['metrics'].get('pve'), 1), ('PA-PVE', lambda x: x['metrics'].get('pa_pve'), 1),
        ('recall', lambda x: x['metrics'].get('recall'), 1),
        ('root err mm', lambda x: x['placement'].get('root_err_mm'), 0),
        ('depth slope', lambda x: x['placement'].get('depth_fit_slope'), 2),
        ('depth icpt m', lambda x: x['placement'].get('depth_fit_intercept_m'), 2),
        ('size ratio', lambda x: x['placement'].get('size_ratio_median_px'), 2),
        ('girth pred/GT', lambda x: x['metrics'].get('girth'), 3),
        ('PVE_neck mm', lambda x: x['metrics'].get('pve_neck_local'), 1),
        ('global head deg', lambda x: x['metrics'].get('rot_global_head_deg'), 1),
        ('rot fingers deg', lambda x: x['metrics'].get('rot_fingers'), 1)]
rows += [(b + ' median', (lambda b: lambda x: x['bones'][b].get('err_median'))(b), 1) for b in bones]
W = 14
print('=== exp7 vs references: 100-image AnnyOne holdout, V1 (helper bones masked), matched people; - = no trace found')
print(format('', '<18') + ''.join(format(l, '>' + str(W)) for l, _ in cols))
for name, get, nd in rows:
    print(format(name, '<18') + ''.join(format(f(get(data[l]), nd), '>' + str(W)) for l, _ in cols))
print(format('verdict ' + bones[0], '<18') + ''.join(format(str(data[l]['bones'][bones[0]].get('verdict') or '-'), '>' + str(W)) for l, _ in cols))
print(format('verdict head', '<18') + ''.join(format(str(data[l]['bones']['head'].get('verdict') or '-'), '>' + str(W)) for l, _ in cols))
me = data['THIS RUN']
checks = {}
nk = [me['bones'][b].get('err_median') for b in ('neck02', 'head')]
gi = me['metrics'].get('girth')
checks['neck kept (neck02, head < 15 deg; girth >= 0.9)'] = (None if None in nk + [gi] else
                                                           bool(max(nk) < 15 and gi >= 0.9))
r = me['placement'].get('root_err_mm')
checks['depth kept (root error < 750 mm)'] = None if r is None else bool(r < 750)
ct = data.get('exp7 ctrl')
if ct and ct['found']:
    p, pc, rc, rcc = me['metrics'].get('pve'), ct['metrics'].get('pve'), me['metrics'].get('recall'), ct['metrics'].get('recall')
    checks['cost vs exp7 control (PVE <= ctrl + 8, recall >= ctrl - 2)'] = (None if None in (p, pc, rc, rcc) else
                                                                           bool(p <= pc + 8 and rc >= rcc - 2))
else:
    checks['cost vs exp7 control'] = ('this run IS the control arm' if arm == 'control'
                                      else 'no control trace yet (run ARM=control)')
if arm == 'control':
    print('  ARM=control: the depth and bone fixes are OFF by design; the two checks below are for reference only')
for k, v in checks.items():
    print('  ' + k + ': ' + ('yes' if v is True else 'NO' if v is False else ('unknown' if v is None else v)))
print('These are AnnyOne-holdout reads of one run (no seed, fresh Adam state). 3DPW-test: section 4 of the job log.')
json.dump(dict(columns=data, checks={k: v for k, v in checks.items()}), open(os.path.join(out, 'compare.json'), 'w'), indent=2)
print('wrote ' + os.path.join(out, 'compare.json'))
PY
)

# bash-only checks (no python): paths and index files. Returns 1 on any refusal.
bash_checks () {
  local fail=0 real f s
  [ -f "$(abs "${CKPT}")" ] || { echo "REFUSED: checkpoint not found: ${CKPT}"; fail=1; }
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
  # step 4 runs the eval script from the repo: the old one (before MODEL_ENTRY / EVAL_LABELS) ignores both overrides,
  # evaluates v5 00099 against threedpw_anny_full, and the log would be read as this run's 3DPW-test result
  if [ "${EVAL_TEST}" = 1 ]; then
    if [ ! -f "${EVAL_SCRIPT}" ]; then
      echo "REFUSED: ${EVAL_SCRIPT} not found (EVAL_TEST=1 runs it in step 4)"; fail=1
    elif ! grep -q MODEL_ENTRY "${EVAL_SCRIPT}" || ! grep -q EVAL_LABELS "${EVAL_SCRIPT}"; then
      echo "REFUSED: ${EVAL_SCRIPT} lacks MODEL_ENTRY or EVAL_LABELS: the OLD eval script. Step 4 would evaluate"
      echo "         v5 00099 against threedpw_anny_full. Commit and push it together with this file, then git pull"
      echo "         (or run with EVAL_TEST=0)."
      fail=1
    fi
  fi
  case "${BONE_FIX}" in
    none|inherit) ;;
    *) [ -f "$(abs "${BONE_FIX}")" ] || { echo "REFUSED: bone fix not found: $(abs "${BONE_FIX}")"; fail=1; } ;;
  esac
  [ -f "$(abs "${MANIFEST}")" ] || echo "WARNING: $(abs "${MANIFEST}") is missing: the diag-manifest images cannot be excluded"
  # train.py keeps only the NB_MAX_CKPT highest epoch numbers in its checkpoint folder (train.py, "Cleaning old ckpt"):
  # into a folder that still holds 00008/00009, every new lower-epoch checkpoint is deleted as soon as it is written.
  local ck old
  ck="$(abs "${OUT}")/train/checkpoints"
  if compgen -G "${ck}/*.pt" >/dev/null; then
    old="$(cd "${ck}" && ls -1 -- *.pt | tr '\n' ' ')"; old="${old% }"
    if [ "${OVERWRITE}" = 1 ]; then
      echo "    OVERWRITE=1: ${ck} holds ${old}; the job moves $(abs "${OUT}") to ${RUN}_replaced_<jobid> before training"
    else
      echo "REFUSED: ${ck} already holds checkpoints (${old}). train.py keeps only the ${NB_MAX_CKPT} highest epoch"
      echo "         numbers there: the old ones would stay, the new ones would be deleted as they are written, and the"
      echo "         trace / table / test evaluation could read the old model. Use another RUN=..., or OVERWRITE=1 (the job"
      echo "         then moves the old folder to ${RUN}_replaced_<jobid> before training)."
      fail=1
    fi
  fi
  return "${fail}"
}

# Python for the label check on the login node: 3.8+ (the label pickles are written with pickle.HIGHEST_PROTOCOL by
# build_threedpw_anny_labels.py, protocol 5 under the env's Python) with NumPy. Prints the command, or nothing.
find_check_python () {
  local p
  for p in python3 python; do
    if command -v "${p}" >/dev/null 2>&1 \
       && "${p}" -c 'import sys, numpy; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
      echo "${p}"; return 0
    fi
  done
  return 0
}

print_settings () {
  echo "=== exp7_final_model ${RUN} (ARM=${ARM})"
  echo "    start ${CKPT}"
  echo "    depth: --gt_center_joint ${GT_CENTER} --alpha_dist ${ALPHA_DIST} --alpha_transl ${ALPHA_TRANSL}"
  echo "    bone fix: ${BONE_FIX}   3DPW: ${LABELS} (fits ${FITS}), ${MIX_RATIO} of each batch, val n ${VAL_3DPW_N}," \
       "mask_3dpw_hands ${MASK_3DPW_HANDS}"
  echo "    ${EPOCHS} epochs x ${N_ITERS} iterations x batch 4 = $(( MAX_ITER * 4 )) images, lr ${LR}; time limit ${TIME}" \
       "(allowance ${NEED_MIN} min); test eval ${EVAL_TEST}"
  echo "    output ${OUT}"
}

# ================================================================ TABLE-ONLY MODE (login node, no Slurm)
# Rebuilds step 3 (table + compare.json) of an existing <RUN> folder, e.g. once the fix arm and ARM=control both
# finished. Needs only <RUN>/holdout_ft/summary.json; the control column prints "-" while CONTROL_RUN has no trace.
if [ "${TABLE_ONLY}" = 1 ]; then
  echo "=== exp7_final_model TABLE_ONLY: ${OUT} (ARM=${ARM}) vs control ${CONTROL_RUN}"
  cd "${REPO}" || { echo "ERROR: REPO=${REPO} not found"; exit 1; }
  [ -f "${OUT}/holdout_ft/summary.json" ] || {
    echo "ERROR: ${REPO}/${OUT}/holdout_ft/summary.json not found: no finished trace for RUN=${RUN}. Give the same"
    echo "       ARM / EPOCHS / N_ITERS / RUN as the training run."; exit 1; }
  [ -f "${CONTROL_RUN}/holdout_ft/summary.json" ] \
    || echo "NOTE: no ${CONTROL_RUN}/holdout_ft/summary.json: the control column and the cost line stay empty"
  PYT=""
  for p in python3 python; do
    command -v "${p}" >/dev/null 2>&1 && "${p}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' \
      >/dev/null 2>&1 && { PYT="${p}"; break; }
  done
  [ -n "${PYT}" ] || { echo "ERROR: no Python 3 found (tried python3, python)"; exit 1; }
  printf '%s\n' "${TABLE_PY}" | "${PYT}" - "${OUT}" "${CONTROL_RUN}" "${BF_ABS}" "${ARM}"
  exit $?
fi

# ================================================================ which mode
# Job mode only inside a job this wrapper submitted (sbatch --export=ALL,EXP7_JOB=1) or a plain sbatch of this file
# (the #SBATCH --export line). An salloc / srun shell also has SLURM_JOB_ID but no EXP7_JOB: refuse rather than train
# in that allocation; CHECK_ONLY=1 / DRY_RUN=1 run the pre-flight there and submit nothing.
MODE=submit
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ "${EXP7_JOB:-}" = 1 ]; then
    MODE=job
  elif [ "${CHECK_ONLY}" != 1 ] && [ "${DRY_RUN}" != 1 ]; then
    echo "REFUSED: SLURM_JOB_ID=${SLURM_JOB_ID} is set but EXP7_JOB is not: this looks like an salloc / srun shell, not"
    echo "         a job of this wrapper. Run it from a login shell (it then submits with sbatch), or use"
    echo "         'sbatch ${SELF}'. CHECK_ONLY=1 or DRY_RUN=1 work here. Nothing trained, nothing submitted."
    exit 1
  fi
fi

# ================================================================ SUBMIT MODE (not inside a Slurm job)
if [ "${MODE}" = submit ]; then
  print_settings
  echo "--- pre-flight on $(hostname 2>/dev/null || echo this host)"
  bash_checks || { echo "NOT SUBMITTED"; exit 1; }
  if command -v squeue >/dev/null 2>&1; then
    QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n "exp7-${RUN}" -o '%i %T' 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${QUEUED// /}" ]; then
      echo "REFUSED: a job named exp7-${RUN} is already queued or running (${QUEUED% }): two jobs would write the same"
      echo "         folder ${OUT}. Cancel it (scancel) or use another RUN=..."
      echo "NOT SUBMITTED"; exit 1
    fi
    # a plain 'sbatch slurm/experiments/exp7_final_model.sh' is named exp7-final-model; its RUN is not visible here
    QUEUED="$(squeue -h -u "${USER:-$(id -un)}" -n exp7-final-model -o '%i %T' 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${QUEUED// /}" ]; then
      if [ "${IGNORE_PLAIN_SBATCH}" = 1 ]; then
        echo "    IGNORE_PLAIN_SBATCH=1: job(s) named exp7-final-model (${QUEUED% }) assumed to write another RUN"
      else
        echo "REFUSED: job(s) named exp7-final-model (a plain sbatch of this file) are queued or running (${QUEUED% })."
        echo "         Their RUN cannot be read from squeue; if one writes ${OUT}, two jobs would share the folder."
        echo "         Check its log (logs/exp7_final_model_<jobid>.out, first line '=== exp7_final_model <RUN>'),"
        echo "         then cancel it, or rerun with IGNORE_PLAIN_SBATCH=1 if it writes another RUN."
        echo "NOT SUBMITTED"; exit 1
      fi
    fi
  fi
  PYCHK="$(find_check_python)"
  if [ -z "${PYCHK}" ]; then
    echo "REFUSED: no Python 3.8+ with NumPy here (tried python3, python), so the label check cannot run on this host."
    echo "         Run 'conda activate multihmr' first, then this command again. (Plain sbatch ${SELF} skips this"
    echo "         pre-flight; the checks then run only when the job starts, after the queue wait.)"
    echo "NOT SUBMITTED"; exit 1
  fi
  echo "    label check with $(command -v "${PYCHK}") ($("${PYCHK}" -c 'import sys, numpy; print("Python %d.%d.%d, NumPy %s" % (sys.version_info[:3] + (numpy.__version__,)))'))"
  if ! printf '%s\n' "${CHECK_PY}" | "${PYCHK}" - --labels "${LABELS}" --fits "${FITS}" --splits "${SPLITS}" \
      --n_sample "${N_LABEL_CHECK}" --bone_fix "${BF_ABS}" --ckpt "$(abs "${CKPT}")" --base "${REPO}"; then
    echo "    (if the problems above say 'unreadable', this Python may not match the env: run 'conda activate multihmr')"
    echo "NOT SUBMITTED"; exit 1
  fi
  [ "${CHECK_ONLY}" = 1 ] && { echo "CHECK_ONLY=1: every check passed; nothing submitted"; exit 0; }
  [ -n "${PY_VENV:-}" ] && echo "note: PY_VENV=${PY_VENV} is set in this shell; the job ignores it (training needs env multihmr)"
  CMD=(sbatch --export=ALL,EXP7_JOB=1 --job-name="exp7-${RUN}" --time="${TIME}" --partition="${PARTITIONS}" "${SELF}")
  echo "--- submit: ${CMD[*]}   (from ${PWD}; the settings above travel as environment variables)"
  if [ "${DRY_RUN}" = 1 ]; then echo "DRY_RUN=1: nothing submitted"; exit 0; fi
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
OUT_ABS="$(abs "${OUT}")"
if [ "${OVERWRITE}" = 1 ] && [ -e "${OUT_ABS}" ]; then
  mv "${OUT_ABS}" "${OUT_ABS}_replaced_${SLURM_JOB_ID}"
  echo "OVERWRITE=1: moved the old ${OUT} to ${OUT}_replaced_${SLURM_JOB_ID} (delete it when it is no longer needed)"
fi
rm -f "${OUT_ABS}/last_checkpoint.txt"   # step 4 must never read a pointer left by an earlier run
to_min () {   # Slurm time limit ([D-]HH:MM:SS, MM:SS) -> minutes; empty if unknown
  local t="$1" d=0 h=0 m=0
  case "$t" in ''|UNLIMITED|INVALID|NOT_SET) return 0 ;; esac
  case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
  IFS=: read -r -a p <<< "$t"
  case "${#p[@]}" in 3) h="${p[0]}"; m="${p[1]}" ;; 2) m="${p[0]}" ;; *) m="${p[0]}" ;; esac
  echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}
LIMIT="$(squeue -h -j "${SLURM_JOB_ID}" -o %l 2>/dev/null || true)"
LIMIT_MIN="$(to_min "${LIMIT}" || true)"
if [ -n "${LIMIT_MIN}" ] && [ "${LIMIT_MIN}" -lt "${NEED_MIN}" ]; then
  echo "WARNING: time limit ${LIMIT} (${LIMIT_MIN} min) < allowance ${NEED_MIN} min for ${EPOCHS} epochs: the job may be"
  echo "         killed before the trace / test evaluation. Next time submit with: bash ${SELF} (sets --time)."
fi
COMMAND=$(cat <<EOF2
set -euo pipefail
unset PY_VENV; export CONDA_ENV=multihmr
source ${REPO}/slurm/env.sh
source ${REPO}/slurm/egl.sh
python -c 'import sys, torch, xformers, torchvision; c = torch.cuda.get_device_capability(0); print("[env]", torch.cuda.get_device_name(0), "compute capability %d.%d" % c, "| torch", torch.__version__, "| xformers", xformers.__version__, "| torchvision", torchvision.__version__); sys.exit(3 if c[0] >= 9 else 0)' \\
    || { echo "REFUSED: this GPU (compute capability 9.0 or newer) or the env multihmr cannot run training; see the header"; exit 1; }
mkdir -p "${OUT}"

echo "=== 0. LABEL CHECK (inside the job)"
python - --labels "${LABELS}" --fits "${FITS}" --splits "${SPLITS}" --n_sample ${N_LABEL_CHECK} \\
    --bone_fix "${BF_ABS}" --ckpt "$(abs "${CKPT}")" --base "${REPO}" <<'PYCHECK' || { echo "REFUSED: label / bone-fix check failed; nothing trained"; exit 1; }
${CHECK_PY}
PYCHECK

EXCL="${OUT}/exclude_manifest_indices.txt"
EXCL_ARG=""
if [ -f "${MANIFEST}" ]; then
  python - "${MANIFEST}" "\${EXCL}" <<'PY'
import json, sys
man, out = sys.argv[1], sys.argv[2]
idx = sorted({int(e['idx']) for e in json.load(open(man))['entries']})
with open(out, 'w') as f:
    f.write('# AnnyOne indices of the diag-manifest images (%d), from %s; never sampled by this fine-tune\n' % (len(idx), man))
    f.write('\n'.join(str(i) for i in idx) + '\n')
print('wrote', out, ':', len(idx), 'images excluded')
PY
  EXCL_ARG="--exclude_indices_file \${EXCL}"
else
  echo "WARNING: ${MANIFEST} not found: the diag-manifest images stay in the sampling pool"
fi

TRAIN_CMD="python train.py \\
    --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
    --person_center head --pretrained $(abs "${CKPT}") --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone dinov2_vitl14 --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 \\
    --rotmat_loss l1 --alpha_rotmat 0.1 --alpha_global_rot 0 --global_rot_bones head,neck03,wrist.L,wrist.R \\
    --start_2d_epoch 0 \\
    --gt_center_joint ${GT_CENTER} --alpha_dist ${ALPHA_DIST} --alpha_transl ${ALPHA_TRANSL} --debug_anchor 2 \\
    \${EXCL_ARG} ${BONE_FIX_ARG} \\
    --mix_3dpw_labels ${LABELS} --mix_3dpw_ratio ${MIX_RATIO} --mix_3dpw_split train \\
    --val_3dpw_split validation --val_3dpw_n ${VAL_3DPW_N} ${HANDS_ARG} \\
    --batch_size 4 --num_workers 12 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
    --eval_freq ${EVAL_FREQ} --nb_max_ckpt ${NB_MAX_CKPT} --log_freq 50 \\
    --save_dir ${OUT} --name train"
{
  echo "exp7_final_model ${RUN}, job ${SLURM_JOB_ID}, \$(date -Is), git \$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "ARM=${ARM} CKPT=${CKPT} BONE_FIX=${BONE_FIX} LABELS=${LABELS} FITS=${FITS}"
  echo "EPOCHS=${EPOCHS} N_ITERS=${N_ITERS} MAX_ITER=${MAX_ITER} LR=${LR} MIX_RATIO=${MIX_RATIO} VAL_3DPW_N=${VAL_3DPW_N}"
  echo "MASK_3DPW_HANDS=${MASK_3DPW_HANDS} EVAL_FREQ=${EVAL_FREQ} NB_MAX_CKPT=${NB_MAX_CKPT} EVAL_TEST=${EVAL_TEST} TIME=${TIME}"
  echo "\${TRAIN_CMD}"
} > "${OUT}/settings.txt"

echo "=== 1. TRAINING -> ${OUT}/train.log (settings: ${OUT}/settings.txt)"
T0=\$(date +%s)
eval "\${TRAIN_CMD}" > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s (startup + ${EPOCHS} epochs + evaluations)"
tr '\r' '\n' < "${OUT}/train.log" > "${OUT}/train_lines.log"
grep -E "^\[bone_fix\]|^\[mixed\]|^Loaded .*3DPW-Anny|^3DPW-Anny val|^\[age_balance\]|^\[loss\] joint weights|^\[loss\] mask_3dpw_hands" "${OUT}/train_lines.log" || true
echo "--- anchor check, first 2 batches (consistent targets keep pred offset_y in [-0.5, 0.5])"
grep -E "^\[debug_anchor\]" "${OUT}/train_lines.log" | head -n 40 || echo "  (no debug_anchor lines)"
echo "--- throughput and in-training evaluation (V0; AnnyOne holdout and 3DPW-val against v23; cohort follows --gt_center_joint)"
grep -E "\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall): |Error|Traceback" "${OUT}/train_lines.log" || true
echo "NaN/Inf gradient steps skipped: \$(grep -c 'NaN/Inf GRADIENTS' "${OUT}/train_lines.log" || true)"
# first real use of --mask_3dpw_hands + --mix_3dpw_labels + --exclude_indices_file + --bone_fix together: every flag
# must have printed its start-up line, or the model is not the planned one
MISSING=""
grep -qE "^\[mixed\] training on" "${OUT}/train_lines.log" || MISSING="\${MISSING} '[mixed] training on' (no 3DPW in the batches);"
if [ "${MASK_3DPW_HANDS}" = 1 ] && ! grep -qE "^\[loss\] mask_3dpw_hands" "${OUT}/train_lines.log"; then
  MISSING="\${MISSING} '[loss] mask_3dpw_hands' (3DPW placeholder fingers supervised);"
fi
if [ "${ARM}" = fix ]; then
  NB_EXP=""
  case "${BONE_FIX}" in
    none|inherit) ;;
    *) NB_EXP=\$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1])).get("bones") or {}))' "${BF_ABS}") ;;
  esac
  if [ -n "\${NB_EXP}" ]; then
    grep -qE "^\[bone_fix\] ON, \${NB_EXP} bone\(s\)" "${OUT}/train_lines.log" \\
        || MISSING="\${MISSING} '[bone_fix] ON, \${NB_EXP} bone(s)' (trained without the planned bone fix);"
  else
    grep -qE "^\[bone_fix\] ON" "${OUT}/train_lines.log" || MISSING="\${MISSING} '[bone_fix] ON' (trained without the bone fix);"
  fi
fi
if [ -n "\${MISSING}" ]; then
  echo "ERROR: train.log lacks:\${MISSING}"
  echo "       Not the planned model; stopping before the trace and the test evaluation."
  exit 1
fi
if [ -n "\${EXCL_ARG}" ] && ! grep -qE "^\[age_balance\] exclusion-only" "${OUT}/train_lines.log"; then
  echo "WARNING: no '[age_balance] exclusion-only' line: the diag-manifest images may have been sampled"
fi
if [ "${ARM}" = control ] && grep -qE "^\[bone_fix\] ON" "${OUT}/train_lines.log"; then
  echo "ERROR: ARM=control but a bone fix was ON; stopping"; exit 1
fi
LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

echo "=== 2. HOLDOUT TRACE + BONE DUMP (100 images, --select holdout, V0-V3, every bone) -> ${OUT}/holdout_ft"
T0=\$(date +%s)
python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${LAST}" --out "${OUT}/holdout_ft" --select holdout --n 100 \\
    --n_vis ${N_VIS} --dump_bones 1 --dump_bone_names all > "${OUT}/holdout_ft.log" 2>&1 \\
    || { echo "ERROR: trace failed; tail:"; tail -n 30 "${OUT}/holdout_ft.log"; exit 1; }
echo "    done in \$(( \$(date +%s) - T0 )) s"
grep -E "^\[load_model\] bone_fix|^\[bone_fix\]" "${OUT}/holdout_ft.log" || true
grep -A 12 "=== PLACEMENT" "${OUT}/holdout_ft.log" || true
grep -A 3 "=== ROTATION ERROR BY BONE GROUP" "${OUT}/holdout_ft.log" | grep -E "^variant|^V1_eval_masked" || true
grep -E "^neck skin V1_eval_masked" "${OUT}/holdout_ft.log" || true

echo "=== 3. COMPARISON TABLE -> ${OUT}/compare.json"
python - "${OUT}" "${CONTROL_RUN}" "${BF_ABS}" "${ARM}" <<'PYTABLE'
${TABLE_PY}
PYTABLE
echo "\${LAST}" > "${OUT}/last_checkpoint.txt"
EOF2
)

echo "=== job step: training, trace and comparison (container)"
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="${REPO}" \
  bash -c "$COMMAND"

if [ "${EVAL_TEST}" != 1 ]; then
  echo "=== 4. 3DPW-test evaluation skipped (EVAL_TEST=${EVAL_TEST})"; exit 0
fi
# 4. the v9-era test evaluation, unchanged except for its two environment overrides (MODEL_ENTRY, EVAL_LABELS); it runs
#    its own srun step inside this allocation. No --bone_fix: train.py inherits the checkpoint's fix (load_only_backbone 0).
LAST_ABS="$(abs "$(cat "${REPO}/${OUT}/last_checkpoint.txt")")"
echo "=== 4. 3DPW-TEST EVALUATION (slurm/eval/run_eval_3dpw_test.sh) of ${LAST_ABS} against ${LABELS}"
MODEL_ENTRY="${LAST_ABS}|0|dinov2_vitl14|exp7_${RUN}" EVAL_LABELS="${LABELS}" \
  bash "${REPO}/slurm/eval/run_eval_3dpw_test.sh" > "${REPO}/${OUT}/eval_3dpw_test.log" 2>&1 \
  || { echo "ERROR: test evaluation failed; tail of ${OUT}/eval_3dpw_test.log:"; tail -n 30 "${REPO}/${OUT}/eval_3dpw_test.log"; exit 1; }
# the log must be THIS model on THESE labels: an eval script without the overrides evaluates its list entry 0 (v5 00099)
# on threedpw_anny_full and still exits 0
if ! grep -qF "task 0: exp7_${RUN}  ->  ${LAST_ABS}" "${REPO}/${OUT}/eval_3dpw_test.log" \
   || ! grep -qF "labels: ${LABELS} " "${REPO}/${OUT}/eval_3dpw_test.log"; then
  mv "${REPO}/${OUT}/eval_3dpw_test.log" "${REPO}/${OUT}/eval_3dpw_test.NOT_THIS_RUN.log"
  echo "ERROR: the test-evaluation log lacks 'task 0: exp7_${RUN}  ->  ${LAST_ABS}' or 'labels: ${LABELS}'."
  echo "       It is not this model on these labels (old slurm/eval/run_eval_3dpw_test.sh?). Its lines:"
  grep -E "^task |^labels: |Evaluating" "${REPO}/${OUT}/eval_3dpw_test.NOT_THIS_RUN.log" | head -n 5 || true
  echo "       Renamed to ${OUT}/eval_3dpw_test.NOT_THIS_RUN.log: do not use its numbers. Fix the eval script, then run"
  echo "       step 4 alone: EVAL_LABELS=${LABELS} MODEL_ENTRY=\"${LAST_ABS}|0|dinov2_vitl14|exp7_${RUN}\" sbatch --array=0 slurm/eval/run_eval_3dpw_test.sh"
  exit 1
fi
tr '\r' '\n' < "${REPO}/${OUT}/eval_3dpw_test.log" | grep -E "^task |Evaluating|Loaded .*3DPW-Anny|^\[bone_fix\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): " || true
echo "Compare these 3DPW-test numbers only with models evaluated on the same LABELS (header: re-evaluate v5, v9, exp6)."
echo "=== done: ${OUT}"
