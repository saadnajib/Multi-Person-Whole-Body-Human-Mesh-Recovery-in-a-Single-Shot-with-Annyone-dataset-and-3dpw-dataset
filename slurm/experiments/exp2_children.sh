#!/bin/bash
# exp2_children: measure how often the training batches contain a child, then fine-tune the production checkpoint
# with age-balanced sampling (--age_balance) and per-phenotype shape weights (--shape_weights), and re-run the fixed
# evaluation. docs/experiments/exp2_children.md has the read-only verification and the plan.
#
# WHY (docs/diagnostics_findings.md, "Children"): the shape path is correct (11 phenotypes -> sigmoid -> body model,
# L1 against the dataset's own phenotypes, same visibility filter as the pose terms) and in-sample babies ARE
# predicted as babies; but 2.53% of training people have age < 0.3, roughly 4 in 5 batch-4 batches contain none
# (arithmetic, measured by step 2 below), the holdout has no child and 3DPW labels pin age to 0.55. On real photos
# small people come out as short adults (age 0.56-0.66). The lever is sampling and loss weighting, not architecture.
#
# WHAT RUNS (one GPU):
#   1. scripts/experiments/build_age_index.py  (once; pkl reads only, ${INDEX_WORKERS} processes; prints s per 10k files)
#   2. the diag manifest's image indices -> ${OUT}/exclude_manifest_indices.txt (kept OUT of the fine-tune)
#      scripts/experiments/measure_batch_ages.py on ${N_MEASURE} real training batches, baseline AND balanced:
#      supervised people per age bucket (after prepare_gt's visibility filter), childless-batch fraction, loader images/s
#   3. train.py from CKPT with the production v5 recipe (same freeze pattern, lr 1e-5, alphas, boosts, batch 4) plus
#        --age_balance ${AGE_BALANCE} --age_balance_child_frac ${CHILD_FRAC} --age_balance_child_age ${CHILD_AGE}
#        --exclude_indices_file <manifest idx> --shape_weights "${SHAPE_WEIGHTS}"   (age x3, height x2 by default)
#        --start_2d_epoch 0      the epoch counter restarts at 0 (no resume): keeps j2d/v2d ON as at epoch 99
#        --n_iters_per_epoch 1000 --max_iter ${MAX_ITER}   2 epochs = 8,000 images ~ 20 min at the measured 6.7 images/s
#      Checkpoints go to ${OUT}/train/checkpoints/0000N.pt, never next to the production run (train.py keeps the
#      --nb_max_ckpt highest epoch numbers of a directory and would delete a pilot's 00000.pt there).
#   4. scripts/diagnostics/diag_holdout_trace.py on the LAST checkpoint, V1 settings, on exactly three fixed sets:
#        --select holdout (100 images; adults only: the regression check), the manifest TRAIN split (its 10 children
#        are in-sample for the base model and excluded from this fine-tune), --select youngest (50 in-sample babies).
#   5. a summary next to the anny_s2_shape_v5/00099 baseline numbers, and the caveats.
#
# USAGE (repo root on the cluster):
#   sbatch slurm/experiments/exp2_children.sh                            # index + measurement + pilot + traces
#   AGE_BALANCE=0 sbatch slurm/experiments/exp2_children.sh              # control arm: same exclusion + shape weights, uniform draws
#   SHAPE_WEIGHTS='' sbatch slurm/experiments/exp2_children.sh           # sampler only, unweighted shape loss
#   CHILD_FRAC=0.15 sbatch slurm/experiments/exp2_children.sh            # milder oversampling
#   SKIP_TRAIN=1 sbatch slurm/experiments/exp2_children.sh               # steps 1-2 only (the measurement the plan asks for)
#   BASELINE_TRACE=1 sbatch slurm/experiments/exp2_children.sh           # also trace CKPT itself with the same code (+20-30 min)
#
# LONGER RUN (only after a pilot that meets the criterion below): 10 epochs = MAX_ITER=10000 = 40,000 images,
#   40000 / 6.7 images/s = 5970 s ~ 1.7 h of training plus 10 holdout evaluations. Rescale from the pilot's own
#   "[throughput] epoch 1: ... images/s" line: hours = MAX_ITER * 4 / images_per_s / 3600.
#   MAX_ITER=10000 RUN=long_balanced sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
#
# SUCCESS CRITERION (printed at the end; baselines from docs/diagnostics_findings.md, V1 settings):
#   adults (100-image holdout): PVE <= 70.5 + 1 mm, recall >= 90.2 - 1, age bias within +/-0.03 (baseline -0.015);
#   children, IN-SAMPLE ONLY: manifest child PVE < 47.8, age bias toward 0 from +0.078, shape-swap gain < 6.2 mm;
#   youngest-50 predicted age closer to the GT 0.015 than the baseline 0.10.
#   No held-out child exists: meeting this justifies the longer run and the real-photo read (E4), nothing more.
#SBATCH --job-name=exp2-children
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/exp2_children_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/exp2_children_%j.err
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
AGE_BALANCE="${AGE_BALANCE:-1}"           # 0 = control arm (same exclusion and shape weights, uniform draws with replacement)
CHILD_FRAC="${CHILD_FRAC:-0.25}"          # target fraction of sampled IMAGES whose youngest person is a child
CHILD_AGE="${CHILD_AGE:-0.3}"             # child threshold on the Anny age phenotype (1/3 ~ 11 y)
SHAPE_WEIGHTS="${SHAPE_WEIGHTS-1,3,1,1,2,1,1,1,1,1,1}"   # gender,AGE x3,muscle,weight,HEIGHT x2,...; '' = unweighted
LR="${LR:-1e-5}"
N_ITERS="${N_ITERS:-1000}"
MAX_ITER="${MAX_ITER:-2000}"              # 2 epochs
N_MEASURE="${N_MEASURE:-200}"             # batches per measurement run (x batch 4 images)
INDEX="${INDEX:-results/experiments/exp2_children/age_index.npz}"   # train.py --age_balance_index default
INDEX_WORKERS="${INDEX_WORKERS:-16}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"
MANIFEST="${MANIFEST:-results/diagnostics/manifest.json}"           # build_manifest.py default (slurm/diagnostics/diag_manifest.sh)
SKIP_MEASURE="${SKIP_MEASURE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
BASELINE_TRACE="${BASELINE_TRACE:-0}"
YOUNG_SCAN="${YOUNG_SCAN:-20000}"         # slurm/diagnostics/diag_young.sh defaults
YOUNG_N="${YOUNG_N:-50}"
SW_TAG=$([ -n "${SHAPE_WEIGHTS}" ] && echo sw || echo nosw)
RUN="${RUN:-pilot_ab${AGE_BALANCE}_cf${CHILD_FRAC}_${SW_TAG}_${TAG}}"
OUT="results/experiments/exp2_children/${RUN}"
EXCL="${OUT}/exclude_manifest_indices.txt"
N_EPOCHS=$(( MAX_ITER / N_ITERS ))
EXPECTED_TRAIN_MIN=$(( MAX_ITER * 40 / 67 / 60 ))   # MAX_ITER * batch 4 / 6.7 images/s, in minutes

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
mkdir -p "${OUT}"
echo "=== exp2_children run ${RUN}"
echo "    start: ${CKPT}"
echo "    age_balance=${AGE_BALANCE} child_frac=${CHILD_FRAC} child_age=${CHILD_AGE} shape_weights='${SHAPE_WEIGHTS}' lr=${LR}"
echo "    ${N_ITERS} iters/epoch x ${N_EPOCHS} epochs, batch 4 = $(( MAX_ITER * 4 )) images"
echo "    expected training time at the measured 6.7 images/s: ~${EXPECTED_TRAIN_MIN} min (+ one 100-image V0 evaluation per epoch)"

echo "=== 1. PER-IMAGE AGE INDEX -> ${INDEX}"
if [ ! -f "${INDEX}" ] || [ "${REBUILD_INDEX}" = "1" ]; then
  T0=\$(date +%s)
  python scripts/experiments/build_age_index.py --out "${INDEX}" --workers ${INDEX_WORKERS} 2>&1 | tee "${OUT}/age_index.log"
  echo "index build wall time: \$(( \$(date +%s) - T0 )) s (pkl reads only; the scan is I/O bound)"
else
  echo "reusing ${INDEX} (REBUILD_INDEX=1 to rebuild)"
  [ -f "${INDEX%.npz}_summary.json" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('n_total','seconds_per_10k','train','holdout') if k in d}, indent=1))" "${INDEX%.npz}_summary.json" || true
fi

echo "=== 2a. MANIFEST IMAGES -> ${EXCL} (zero sampling weight in the fine-tune)"
if [ -f "${MANIFEST}" ]; then
  python - <<'PY'
import json, os
man = json.load(open(os.environ['MANIFEST_PATH']))
ents = man['entries']
idx = sorted({int(e['idx']) for e in ents})
n_child = sum(1 for e in ents if 'child' in e.get('tags', []))
with open(os.environ['EXCL_PATH'], 'w') as f:
    f.write(f"# AnnyOne indices of the {len(ents)} diag-manifest people ({len(idx)} images, {n_child} child entries), "
            f"from {os.environ['MANIFEST_PATH']}; never sampled by this fine-tune\n")
    f.write("\n".join(str(i) for i in idx) + "\n")
print(f"wrote {len(idx)} image indices ({n_child} child entries) to {os.environ['EXCL_PATH']}")
PY
  EXCL_ARG="${EXCL}"
else
  echo "WARNING: ${MANIFEST} is missing (slurm/diagnostics/diag_manifest.sh builds it; never rebuild an existing one):"
  echo "         no exclusion; the manifest-train read below is then in-sample for the fine-tune too"
  echo "# no manifest found at submit time; no exclusion" > "${EXCL}"
  EXCL_ARG=""
fi

if [ "${SKIP_MEASURE}" != "1" ]; then
  echo "=== 2b. AGE-GROUP REPRESENTATION IN ${N_MEASURE} REAL TRAINING BATCHES: baseline loader (train.py default shuffle)"
  python scripts/experiments/measure_batch_ages.py --n_batches ${N_MEASURE} --batch_size 4 --num_workers 8 --val_anny_n 100 \\
      --child_age ${CHILD_AGE} --age_balance_child_age ${CHILD_AGE} --out "${OUT}/batch_ages_baseline.json" \\
      > "${OUT}/batch_ages_baseline.log" 2>&1 \\
      || { echo "ERROR: baseline measurement failed; tail:"; tail -n 30 "${OUT}/batch_ages_baseline.log"; exit 1; }
  grep -E "^\[batch_ages\]|^\[loader\]|^\[filter\]|WARN" "${OUT}/batch_ages_baseline.log" || true
  echo "=== 2c. same, balanced sampler (--age_balance 1, child_frac ${CHILD_FRAC}, manifest excluded)"
  python scripts/experiments/measure_batch_ages.py --n_batches ${N_MEASURE} --batch_size 4 --num_workers 8 --val_anny_n 100 \\
      --age_balance 1 --age_balance_index "${INDEX}" --age_balance_child_frac ${CHILD_FRAC} --age_balance_child_age ${CHILD_AGE} \\
      --exclude_indices_file "\${EXCL_ARG}" --child_age ${CHILD_AGE} --out "${OUT}/batch_ages_balanced.json" \\
      > "${OUT}/batch_ages_balanced.log" 2>&1 \\
      || { echo "ERROR: balanced measurement failed; tail:"; tail -n 30 "${OUT}/batch_ages_balanced.log"; exit 1; }
  grep -E "^\[batch_ages\]|^\[age_balance\]|^\[loader\]|WARN" "${OUT}/batch_ages_balanced.log" || true
fi

if [ "${SKIP_TRAIN}" = "1" ]; then
  echo "SKIP_TRAIN=1: measurement only; no fine-tune, no trace. Outputs under ${OUT}."
  exit 0
fi

echo "=== 3. TRAINING -> ${OUT}/train.log"
T0=\$(date +%s)
python train.py \\
    --train_data AnnyOne --train_n 0 --val_anny_n 100 \\
    --person_center head --pretrained "${CKPT}" --pretrained_remap 0 --load_only_backbone 0 \\
    --backbone dinov2_vitl14 --img_size 672 \\
    --freeze_backbone 1 --unfreeze_last_n_blocks 4 --mask_helper_joints_train 1 --use_anny_shape 1 \\
    --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 --start_2d_epoch 0 \\
    --age_balance ${AGE_BALANCE} --age_balance_index "${INDEX}" \\
    --age_balance_child_frac ${CHILD_FRAC} --age_balance_child_age ${CHILD_AGE} \\
    --exclude_indices_file "\${EXCL_ARG}" --shape_weights "${SHAPE_WEIGHTS}" \\
    --batch_size 4 --num_workers 8 --n_iters_per_epoch ${N_ITERS} --max_iter ${MAX_ITER} \\
    --learning_rate ${LR} --lr_decay_every 30 --lr_decay_gamma 0.5 --amp 0 --tf32 1 \\
    --eval_freq 1 --nb_max_ckpt 2 --log_freq 50 \\
    --save_dir "${OUT}" --name train > "${OUT}/train.log" 2>&1 \\
    || { echo "ERROR: training failed; tail of ${OUT}/train.log:"; tail -n 40 "${OUT}/train.log"; exit 1; }
echo "training wall time: \$(( \$(date +%s) - T0 )) s (startup + ${N_EPOCHS} epochs + ${N_EPOCHS} holdout evaluations)"
echo "--- sampler / loss configuration as train.py printed it"
grep -E "^\[age_balance\]|^\[loss\]" "${OUT}/train.log" || echo "  (no [age_balance]/[loss] lines: both flags off?)"
echo "--- throughput (images/s) and in-training V0 evaluation (train.py evaluate(), helper bones UNMASKED, adults only)"
grep -E "\[throughput\]|EVAL METRICS| - (pve|pa_pve|mpjpe|recall|precision): |NaN|WARNING|Error" "${OUT}/train.log" || true

LAST=\$(ls "${OUT}/train/checkpoints/"*.pt 2>/dev/null | sort | tail -n 1 || true)
[ -n "\${LAST}" ] || { echo "ERROR: training produced no checkpoint; see ${OUT}/train.log"; exit 1; }
echo "last checkpoint: \${LAST}"

trace () {   # \$1 = checkpoint, \$2 = output dir, \$3... = selection flags
  local ck="\$1" od="\$2"; shift 2
  python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${ck}" --out "\${od}" "\$@" > "\${od}.log" 2>&1 \\
      || { echo "WARNING: trace \${od} failed; tail:"; tail -n 30 "\${od}.log"; return 0; }
  grep -E "^\[select\]|^V1_eval_masked|^=== (PER-VARIANT|AGE BUCKETS|SHAPE|PER CATEGORY)|^child |^variant " "\${od}.log" | head -n 20 || true
}

echo "=== 4a. FIXED HOLDOUT TRACE (100 images, --select holdout, V1; adults only = regression check) on \${LAST}"
trace "\${LAST}" "${OUT}/holdout_trace" --select holdout --n 100
if [ -f "${MANIFEST}" ]; then
  echo "=== 4b. MANIFEST TRAIN-SPLIT TRACE (10 children, in-sample for the base model, excluded from this fine-tune) on \${LAST}"
  trace "\${LAST}" "${OUT}/manifest_trace_train" --manifest "${MANIFEST}" --manifest_split train --n_vis 40
else
  echo "=== 4b. manifest trace SKIPPED: ${MANIFEST} is missing"
fi
echo "=== 4c. YOUNGEST-${YOUNG_N} TRACE (--select youngest over the first ${YOUNG_SCAN} training pkls; IN-SAMPLE) on \${LAST}"
trace "\${LAST}" "${OUT}/young_trace" --select youngest --scan ${YOUNG_SCAN} --n ${YOUNG_N}
if [ "${BASELINE_TRACE}" = "1" ]; then
  echo "=== 4d. BASELINE TRACES on ${CKPT} with the same code and selections"
  trace "${CKPT}" "${OUT}/holdout_trace_baseline" --select holdout --n 100
  [ -f "${MANIFEST}" ] && trace "${CKPT}" "${OUT}/manifest_trace_train_baseline" --manifest "${MANIFEST}" --manifest_split train --n_vis 40
  trace "${CKPT}" "${OUT}/young_trace_baseline" --select youngest --scan ${YOUNG_SCAN} --n ${YOUNG_N}
fi

python - <<'PY'
import json, os
out = os.environ['OUT_DIR']
V = 'V1_eval_masked'
# anny_s2_shape_v5/00099, V1 settings (docs/diagnostics_findings.md): holdout = 100 images / 222 people; manifest child
# n=10 (in-sample); youngest-50 = GT age 0.015 babies (in-sample).
BASE = dict(holdout=dict(pve=70.5, recall=90.2, age_bias=-0.015),
            child=dict(pve=47.8, age_bias=0.078, swap_gain=6.2, n=10),
            young=dict(pred_age=0.10, gt_age=0.015, pve=48.8))

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

def f(x, nd=2):
    return '   -' if x is None else format(x, '.' + str(nd) + 'f')

def holdout_read(d):
    if not d or V not in d.get('variants', {}):
        return {}
    v = d['variants'][V]
    return dict(pve=v.get('pve'), recall=v.get('recall'), age_bias=(v.get('shape_bias') or {}).get('age'),
                age_abs=(v.get('shape_abs_err') or {}).get('age'), height_bias=(v.get('shape_bias') or {}).get('height'),
                depth_slope=(v.get('placement') or {}).get('depth_fit_slope'))

def child_read(d):
    try:
        c = d['per_tag'][V]['child']
    except Exception:
        return {}
    return dict(pve=c.get('pve'), age_bias=c.get('pheno_err_age'), swap_gain=c.get('shape_swap_gain_mm'), n=c.get('n'))

def young_read(d):
    if not d or V not in d.get('variants', {}):
        return {}
    v = d['variants'][V]
    b = v.get('age_buckets') or {}
    first = next(iter(b.values()), {}) if b else {}
    return dict(pve=v.get('pve'), pred_age=first.get('pred_age'), gt_age=first.get('gt_age'), n=first.get('n'),
                age_bias=(v.get('shape_bias') or {}).get('age'),
                rest_h_gt=(v.get('rest_pose_m') or {}).get('skel_height_gt_shape'),
                rest_h_pred=(v.get('rest_pose_m') or {}).get('skel_height_pred_shape'))

def meas_read(d):
    if not d:
        return {}
    return dict(people_lt_0p3=(d.get('frac_below_after') or {}).get('0.3'), childless_batches=d.get('batches_without_child_after'),
                people_per_batch=d.get('people_per_batch_after'), img_with_child=d.get('images_with_child_frac'),
                loader_ips=d.get('loader_images_per_s'))

H = holdout_read(load(out + '/holdout_trace/summary.json')); Hb = holdout_read(load(out + '/holdout_trace_baseline/summary.json'))
C = child_read(load(out + '/manifest_trace_train/summary.json')); Cb = child_read(load(out + '/manifest_trace_train_baseline/summary.json'))
Y = young_read(load(out + '/young_trace/summary.json')); Yb = young_read(load(out + '/young_trace_baseline/summary.json'))
M0 = meas_read(load(out + '/batch_ages_baseline.json')); M1 = meas_read(load(out + '/batch_ages_balanced.json'))
print()
print('=== exp2_children SUMMARY (V1 = helper bones masked; baseline(findings) = anny_s2_shape_v5/00099) ===')
print('--- batch measurement (supervised people, after the visibility filter)')
print(format('', '<34') + format('baseline loader', '>16') + format('balanced loader', '>16'))
for k, lab in (('people_lt_0p3', 'people with age < 0.3'), ('childless_batches', 'batches with no child'),
               ('img_with_child', 'sampled images with a child'), ('people_per_batch', 'people per batch'),
               ('loader_ips', 'loader images/s')):
    a, b = M0.get(k), M1.get(k)
    pct = k != 'people_per_batch' and k != 'loader_ips'
    g = (lambda x: '   -' if x is None else (format(100 * x, '.1f') + '%' if pct else format(x, '.2f')))
    print(format(lab, '<34') + format(g(a), '>16') + format(g(b), '>16'))
print('--- ADULT REGRESSION CHECK: 100-image holdout (no children in it)')
print(format('metric', '<24') + format('baseline(findings)', '>20') + format('baseline(rerun)', '>17') + format('this run', '>12'))
for k, nd in (('pve', 1), ('recall', 1), ('age_bias', 3), ('age_abs', 3), ('height_bias', 3), ('depth_slope', 2)):
    print(format(k, '<24') + format(f(BASE['holdout'].get(k), nd), '>20') + format(f(Hb.get(k), nd), '>17') + format(f(H.get(k), nd), '>12'))
print('--- CHILDREN, IN-SAMPLE: manifest train split, tag child (excluded from this fine-tune, seen by the base model)')
for k, nd in (('n', 0), ('pve', 1), ('age_bias', 3), ('swap_gain', 1)):
    print(format(k, '<24') + format(f(BASE['child'].get(k), nd), '>20') + format(f(Cb.get(k), nd), '>17') + format(f(C.get(k), nd), '>12'))
print('--- CHILDREN, IN-SAMPLE: youngest-50 babies (seen in training, resampled by the balanced loader)')
for k, nd in (('n', 0), ('gt_age', 3), ('pred_age', 3), ('age_bias', 3), ('pve', 1), ('rest_h_gt', 2), ('rest_h_pred', 2)):
    print(format(k, '<24') + format(f(BASE['young'].get(k), nd), '>20') + format(f(Yb.get(k), nd), '>17') + format(f(Y.get(k), nd), '>12'))
ok_adult = ok_child = None
if None not in (H.get('pve'), H.get('recall'), H.get('age_bias')):
    ok_adult = (H['pve'] <= BASE['holdout']['pve'] + 1.0 and H['recall'] >= BASE['holdout']['recall'] - 1.0
                and abs(H['age_bias']) <= 0.03)
    print('adults kept (PVE <= 71.5, recall >= 89.2, |age bias| <= 0.03): ' + ('yes' if ok_adult else 'NO'))
if None not in (C.get('pve'), C.get('age_bias')) or Y.get('pred_age') is not None:
    parts = []
    if C.get('pve') is not None:
        parts.append(C['pve'] < BASE['child']['pve'] and abs(C['age_bias']) < abs(BASE['child']['age_bias']))
        print('manifest children improved (PVE < 47.8 and |age bias| < 0.078): ' + ('yes' if parts[-1] else 'NO'))
    if Y.get('pred_age') is not None:
        parts.append(abs(Y['pred_age'] - (Y.get('gt_age') or BASE['young']['gt_age'])) < abs(BASE['young']['pred_age'] - BASE['young']['gt_age']))
        print('youngest-50 predicted age closer to GT than the baseline 0.10: ' + ('yes' if parts[-1] else 'NO'))
    ok_child = all(parts) if parts else None
if ok_adult is None or ok_child is None:
    verdict = 'INCOMPLETE (missing numbers; see the trace logs)'
elif ok_adult and ok_child:
    verdict = 'PILOT MEETS THE CRITERION (in-sample children, adults kept) -> worth the 10-epoch run and the E4 real-photo read'
elif ok_adult:
    verdict = 'adults kept but no in-sample child improvement -> raise CHILD_FRAC / shape weights or run longer before judging'
else:
    verdict = 'ADULT REGRESSION -> lower CHILD_FRAC or the shape weights; do not run longer'
print('VERDICT: ' + verdict)
json.dump(dict(baseline_findings=BASE, measurement_baseline=M0, measurement_balanced=M1, holdout=H, holdout_baseline_rerun=Hb,
               manifest_child=C, manifest_child_baseline_rerun=Cb, youngest=Y, youngest_baseline_rerun=Yb,
               adults_kept=ok_adult, children_improved_in_sample=ok_child),
          open(out + '/summary.json', 'w'), indent=2)
print('wrote ' + out + '/summary.json')
PY
echo "CAVEATS: no held-out child exists in AnnyOne's last 100 images, so the holdout numbers only show whether the adults"
echo "         survived; every child number above is IN-SAMPLE (manifest children were seen by the base model for 100 epochs"
echo "         and only kept out of this fine-tune; the youngest-50 were resampled by the balanced loader). The informative"
echo "         reads are the manifest-child rows and REAL PHOTOS: CKPT=\${LAST} sbatch slurm/diagnostics/diag_crop_camera.sh"
echo "         (E4; baseline predicts age 0.56-0.66 for small people), both with that in-sample caveat. This is a 2-epoch"
echo "         pilot (8,000 images, fresh Adam state, no seed control): compare with AGE_BALANCE=0 before attributing a"
echo "         change to the sampler; nothing here is a measured result until the traces above have been run."
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "export OUT_DIR='${OUT}' MANIFEST_PATH='${MANIFEST}' EXCL_PATH='${EXCL}'; $COMMAND"
