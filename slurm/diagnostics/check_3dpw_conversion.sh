#!/bin/bash
# 3DPW -> Anny conversion check: fresh, independent evidence that the converted 3DPW-Anny labels match the ORIGINAL
# 3DPW ground truth. Read-only: no label, .npz or checkpoint is written or changed.
#   scripts/diagnostics/check_3dpw_conversion.py on a fixed, seeded sample (PER_SEQ label images from EVERY converted
#   sequence of every split, all people in those files) -> ${OUT}/
#     summary.json      every number, per split + overall, verdicts with thresholds, legend, worst 10 people, loader check
#     per_person.csv    one row per sampled person (sequence, frame, all metrics)
#     how_to_read.txt   the legend + the overall verdict lines
#     vis/*.jpg         ~12 panels [photo | 3DPW SMPL GT mesh + OpenPose 2D | Anny label mesh + joints]
#   per_person.csv is rewritten after every sequence; summary.json + how_to_read.txt are written as soon as the
#   measurements exist, then again after the loader check and after the renders. 'STAGE:' / "stage" says how far the
#   run got, so a pyrender/EGL segfault or the time limit near the end does not lose the measurements.
#
# USAGE (repo root on the cluster):
#   sbatch slurm/diagnostics/check_3dpw_conversion.sh                 # default sample, ~2,000 images
#   PER_SEQ=80 sbatch slurm/diagnostics/check_3dpw_conversion.sh      # larger sample (raise --time beyond ~120)
#   SPLITS=test PER_SEQ=0 sbatch ...                                  # every test label file (~9,700; ~1.5-3 h)
#
# NEEDS (all already on /netscratch from the conversion; the job checks them first):
#   threedpw_anny_full/labels/index_<split>.txt + the label pkls and 3DPW images they list   (what training loads)
#   data/3DPW/sequenceFiles/<split>/<seq>.pkl                                                (original SMPL GT)
#   models/smpl/SMPL_{MALE,FEMALE}.pkl (the files train.py loads via smplx.create('models', 'smpl', gender))
#   threedpw_anny_full/<split>/<seq>_p<k>.npz (optional: the bone mapping + virtual offsets the converter used, and
#     every fitted frame's joint error for I4; without them the smpl_to_anny.py defaults are recomputed with
#     models/smpl/SMPL_NEUTRAL.pkl and I4 / the I3 causes are not measured)
#   configs/threedpw_tasks.txt (optional: tasks without a .npz are listed in I4)
#
# RUNTIME: the 3DPW-Anny labels hold 52 sequences (18 train / 10 validation / 24 test, index files of commit 329eed8),
# so PER_SEQ=40 samples ~2,080 images, ~2,700 people (13 test sequences have 2 actors). Measured locally, CPU only,
# 4 cores, synthetic data of real size (Anny 19,158 vertices, SMPL-size 6,900): 0.25-0.4 s per person (Anny forward
# x2 + rest pose, SMPL forward, 4 KD-trees of ~60k points), 19 s for 21 people incl. ~8 s start-up, 5 renders and the
# loader check. -> ~12-20 min for the default sample, plus ~1 min per 10 sequence pkls read from netscratch.
# I1 gender refit (GENDER_REFIT_PER_SEQ=2 -> ~104 people, 2 refits of 160 fit_frame iterations each): 25 s per person
# measured locally on 4 CPU cores; on a GPU ~7-8 s per person if the converter's v2.1 speed (11 s per 480 iterations)
# holds [Likely] -> ~13 min. The 1 h 30 min limit leaves ~2x margin for slow nodes / netscratch.
#
# HOW TO READ (the job log prints a table per split, per-joint and per-OpenPose-point tables, the worst 10 people and
# the verdicts; how_to_read.txt has the full legend):
#   V0 SMPL forward vs 3DPW jointPositions <= 5 mm: the SMPL mesh is the body the converter fitted to.
#   V1 3D joints, FIT points (bone head + virtual offset) vs 3DPW joints, body: median <= 20 mm and p95 <= 40 mm = PASS
#      (converter tuned to 16-20 mm; 3DPW's own GT ~26 mm [Likely]). 'bone heads' adds the fixed torso rig gap.
#      V1-V4 see only the labels the builder KEPT (it drops > 80 mm, visible < 0.3, box < 40 px); I4 shows all fits.
#   V1b saved vs recomputed error <= 1 mm: the label pkl decodes to the fit the converter reported.
#   V2 surface SMPL <-> Anny skin, nearest point, both directions: median <= 20 mm PASS, <= 35 WARN (62.7 mm = best
#      3DPW PA-PVE so far, v9). A LOWER bound on per-vertex error. Region rows say where (head, hands, torso).
#   V3 2D vs OpenPose (independent of SMPL and of the fit), graded in mm at the joint depth (px x depth / focal, so a
#      close actor is not failed for its larger pixel errors): Anny median - raw 3DPW median <= 15 mm PASS (= 5 px at
#      6 m). Raw 3DPW median > 45 mm = the 3DPW camera / 2D data itself is off: V3 FAIL, not the Anny fit.
#   V4 left/right + mirror >= 99%, limb / stature ratios in [0.95, 1.05], people with a joint > 100 mm <= 1%.
#   V5 training loader (threedpw_anny.py + collate): anny_pose/anny_shape passed through unchanged, human order kept,
#      padded 672 K maps original pixels to the right 672 pixel (<= 0.5 px). train.py itself is NOT run: the Anny
#      decode here is an FP32 copy of prepare_gt; training uses TF32, and 'tf32_gap' (GPU only) says how much that moves
#      the GT joints/vertices.
#      A V2 PASS does NOT mean gender or body shape is right (a wrong-gender fit stays at 7-10 mm in a synthetic test).
#   I1 gender: (a) 'male end' of Anny's gender value from the rest bodies at 0 and 1, next to the label gender of 3DPW
#      'm'/'f' people; (b) GENDER_REFIT_PER_SEQ people per sequence refitted from the label with its own gender
#      (control) and with g -> 1-g: flip lower in joint error AND surface for most people = label gender wrong; a gain
#      near 0 = not shown, not 'not reversed'. Table '=== I1 GENDER ===' in the log, rows in gender_refit.csv.
#   I2 head orientation, I3 coverage (unlabelled valid actors in
#   two-person images split into 'not fitted' = stride phase vs 'dropped by a builder filter'), I4 all converter fits
#   before the builder filters (error of every fitted frame, drops per filter, expected vs indexed label images).
#   OVERALL is the worst V grade; a V check that was not measured makes it NOT MEASURED, never PASS. A FAIL names WHAT
#   is wrong, the per-sequence lines and worst-10 list say WHERE.
#SBATCH --job-name=check-3dpw-conv
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/check_3dpw_conversion_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/check_3dpw_conversion_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

LABELS="${LABELS:-threedpw_anny_full/labels}"
FITS="${FITS:-threedpw_anny_full}"
SEQ_ROOT="${SEQ_ROOT:-data/3DPW/sequenceFiles}"
SMPL_DIR="${SMPL_DIR:-models}"
SPLITS="${SPLITS:-train,validation,test}"
PER_SEQ="${PER_SEQ:-40}"
SEED="${SEED:-0}"
N_VIS="${N_VIS:-12}"
GENDER_REFIT_PER_SEQ="${GENDER_REFIT_PER_SEQ:-2}"   # I1 refit test; 0 = off
OUT="${OUT:-results/diagnostics/check_3dpw_conversion}"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh          # non-fatal: without EGL the panels use vertex dots
# KD-tree queries run on the CPU: give them the job's cores (env.sh sets 1 thread for the dataloader case)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "=== inputs"
ls "${LABELS}"/index_*.txt || { echo "ERROR: no label index under ${LABELS}"; exit 1; }
wc -l "${LABELS}"/index_*.txt
[ -d "${SEQ_ROOT}" ] || { echo "ERROR: 3DPW sequenceFiles not found: ${SEQ_ROOT}"; exit 1; }
for G in MALE FEMALE NEUTRAL; do
  ls "${SMPL_DIR}"/smpl/SMPL_\${G}.* 2>/dev/null || echo "WARNING: ${SMPL_DIR}/smpl/SMPL_\${G}.(pkl|npz) not found"
done
echo "converter fits (.npz) per split:"; for S in train validation test; do
  echo "  \${S}: \$(ls ${FITS}/\${S}/*_p*.npz 2>/dev/null | wc -l)"; done
md5sum scripts/data/smpl_to_anny.py scripts/data/build_threedpw_anny_labels.py datasets/threedpw_anny.py || true
python -c "import scipy; print('scipy', scipy.__version__)" || echo "WARNING: no scipy: brute-force nearest points (slow)"
echo "=== check"
python scripts/diagnostics/check_3dpw_conversion.py \\
  --labels "${LABELS}" --fits "${FITS}" --seq_root "${SEQ_ROOT}" --smpl_dir "${SMPL_DIR}" \\
  --splits "${SPLITS}" --per_seq ${PER_SEQ} --seed ${SEED} --n_vis ${N_VIS} \\
  --gender_refit_per_seq ${GENDER_REFIT_PER_SEQ} --out "${OUT}"
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
