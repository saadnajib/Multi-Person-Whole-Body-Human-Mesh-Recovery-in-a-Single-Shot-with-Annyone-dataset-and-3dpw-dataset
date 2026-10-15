#!/bin/bash
# Neck root cause: per-bone dump of the neck/head chain on the 100-image holdout, one trace per checkpoint.
#   scripts/diagnostics/diag_holdout_trace.py --dump_bones 1 (V0-V3 as usual) -> results/diagnostics/neck_bones/<tag>/
#     bones.npz, bones_summary.json (+ the usual summary.json, per_person.csv, vis/), full log in <tag>.log
# For each bone (spine01, neck01, neck02, neck03, head, jaw, clavicles, shoulder01s, wrists + EXTRA_BONES), per matched
# person and variant: local error, pred/GT angle from identity, global error, swing/twist about the bone's rest axis,
# across-person spread of pred / GT / error rotation, joint error at the bone's own head ('joint': moved only by the bones
# above it) and at its child's head ('child': the first joint its own rotation moves); neck-skin PVE and girth ratio.
# The group means 62.7 (neck01-03) / 89.5 (head+jaw) of v5 have never been split by bone; this does it, without training.
# The same job runs the two open bookkeeping checks: H4 (det / singular values of the RAW dataset blocks of the dumped
# bones: proper rotations?) and H3 (the checkpoint's saved pose_param and loss args; for the inherited heads, trace the
# ancestors, CKPTS line 2 below, since v5's args describe only v5's own run).
#
# USAGE (repo root on the cluster):
#   sbatch slurm/diagnostics/diag_neck_bones.sh                                   # the four default checkpoints below
#   CKPTS="models/multiHMR/anny_s2_frozen_bb_v2_00299.pt logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt" \
#       sbatch slurm/diagnostics/diag_neck_bones.sh                               # when did the offset appear (lineage)
#   CKPTS=<patched.pt> sbatch slurm/diagnostics/diag_neck_bones.sh               # any other checkpoint
# CKPTS: space- or comma-separated checkpoint paths (relative to the repo root or absolute). A missing
# models/multiHMR/<run>_<epoch>.pt falls back to logs/anny_model/<run>/checkpoints/<epoch>.pt; a checkpoint that is
# still missing is skipped with a warning. Tag = <run>_<epoch> (the experiment name for .../<name>/train/checkpoints/).
#
# RUNTIME: the existing holdout trace runs its 100-image loop in 65-105 s per checkpoint (progress lines "91/100 (65-104s)"
# of diag_holdout_3569947/3571596/3571794/3571963/3571964/3571797.out) plus model/anny start-up; the dump adds one
# body-model call per image and small per-person tensor work. ~3-5 min per checkpoint, ~20 min for the four defaults;
# the 1 h limit leaves margin for slow nodes and longer CKPTS lists (raise --time beyond ~10 checkpoints).
#
# HOW TO READ (bones_summary.json, V1_eval_masked; the job log prints each NECK/HEAD BONE DUMP block and a table).
# A label says WHAT the rotations are; each mechanism needs its own test (docs/experiments_plan.md "Neck root cause").
#   FIRST the bookkeeping lines of each block: "raw GT blocks ... NOT proper rotations" -> H4 open (the rotmat target
#     disagrees with the GT mesh); "checkpoint args ... pose_param=" anything but local-bone/None -> H3 (wrong space).
#   'joint' is the error at a bone's OWN head, which its own rotation never moves (anny FK), so it only reflects the bones
#     above it; read 'child' (its child's head) for what the bone itself does: child well above joint = its GLOBAL
#     orientation is off (= its local error only when the parent chain is right).
#   FLIP_TWIST (> 150 deg in >= half the people, bone axis moves < 30 deg) on exactly one of neck01/02/03, with 'child'
#     ~ 'joint' on that bone + head FLIP_* (FLIP_SWING expected, see below) with one axis, small 'glob' and 'child' ~
#     'joint' + jaw OK -> a compensated half-turn flip is PRESENT (consistent with H1).
#     Whether the L1 holds it: exp5 vs its l1 control. Where it came from: the lineage traces (CKPTS line 2).
#   WHERE the flip sits, from 'joint' (compare each bone with the one above; errors add as vectors). Exact only for neck03:
#     neck03 flipped -> neck03/head 'joint' no higher than neck02's. neck02 -> head 'joint' up ~9-21 mm at rest, ~12-30 mm
#     (mean) with 10-20 deg GT neck bends. neck01 -> neck03 up 0-11 mm (14-31 bent), head up 3-10 mm (19-59 bent; anny
#     default rig, docs table "flip | neck02 joint | ..."). Such a rise fits H1; read it with '|gt|' (GT angle).
#   FLIP_SWING -> the local error tilts the bone axis (>= 30 deg) relative to the GT-POSED parent frame, so it moves the
#     child joint only when the parent chain is right. Under H1 the compensating head is expected to read FLIP_SWING (its
#     error is a half-turn about the flipped neck bone's axis, 19-30 deg off the head axis on the default rig at rest, so
#     its swing is ~2x that angle in the GT pose, ~38-61 deg at rest) while its 'glob' stays small and 'child' ~ 'joint'.
#     A FLIP_SWING bone with large 'glob' and 'child' well above 'joint' is a tilt flip that does move its child: not
#     part of the H1 pattern.
#   PARTIAL_FLIP (> 150 deg in 10-50% of the people) -> per-image flips; read mean vs median.
#   all three neck bones CONST_OFFSET with one shared axis (relAng ~62.7) -> consistent with a convention/frame offset
#   FIXED_OUTPUT (pred spread << GT spread) -> the head emits one rotation regardless of the image
#   v10 neck OK with flips on arm/metacarpal bones -> consistent with a per-init random draw AND with a recipe-dependent
#     cause (v10 differs in mask timing, boost, 3DPW from step 0, unfrozen blocks, LR); it does not decide between them
#   girth pred/GT < ~0.8 on the neck skin -> the neck skin is thinner than GT (the 0.8 comes from an LBS simulation);
#     that it is what the renders show needs a render of the decpose-surgery checkpoint
#SBATCH --job-name=diag-neck-bones
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/diag_neck_bones_%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/diag_neck_bones_%j.err
set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs
export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

# v5 production, v9 after 149 more epochs (same heads), v10 (independent fresh heads; traced at 00290), exp1 10-epoch pilot
CKPTS="${CKPTS:-models/multiHMR/anny_s2_shape_v5_00099.pt models/multiHMR/anny_teacher_v9_mixed_00149.pt logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt results/experiments/exp1_depth_anchor/long_pc_dist5/train/checkpoints/00009.pt}"
# v10 has its large constant local errors on the arms and metacarpals (clavicle_s 50.3, arm 36.3 deg): dump them for every
# checkpoint, so the v5 rows are the control for the v10 rows
EXTRA_BONES="${EXTRA_BONES:-upperarm01.L,upperarm02.L,lowerarm01.L,lowerarm02.L,upperarm01.R,upperarm02.R,lowerarm01.R,lowerarm02.R,metacarpal1.L,metacarpal2.L,metacarpal3.L,metacarpal4.L,metacarpal1.R,metacarpal2.R,metacarpal3.R,metacarpal4.R}"
N="${N:-100}"
N_VIS="${N_VIS:-6}"
OUT_ROOT="${OUT_ROOT:-results/diagnostics/neck_bones}"
CKPTS="$(echo "${CKPTS}" | tr ',' ' ')"

COMMAND=$(cat <<EOF2
set -euo pipefail
source /netscratch/najib/multi-hmr/slurm/env.sh
source /netscratch/najib/multi-hmr/slurm/egl.sh
mkdir -p "${OUT_ROOT}"
TAGS=""
for CK in ${CKPTS}; do
  if [ ! -f "\${CK}" ] && [ "\$(basename "\$(dirname "\${CK}")")" = multiHMR ]; then   # models/multiHMR/<run>_<epoch>.pt
    NM=\$(basename "\${CK}" .pt); ALT="logs/anny_model/\${NM%_*}/checkpoints/\${NM##*_}.pt"   # -> its training run dir
    [ -f "\${ALT}" ] && { echo "[ckpt] \${CK} missing, using \${ALT}"; CK="\${ALT}"; }
  fi
  [ -f "\${CK}" ] || { echo "WARNING: checkpoint not found: \${CK}; skipped"; continue; }
  if [ "\$(basename "\$(dirname "\${CK}")")" = checkpoints ]; then
    D=\$(dirname "\$(dirname "\${CK}")"); B=\$(basename "\${D}")
    [ "\${B}" = train ] && B=\$(basename "\$(dirname "\${D}")")          # results/experiments/<exp>/<name>/train/checkpoints
    TAG="\${B}_\$(basename "\${CK}" .pt)"
  else
    TAG=\$(basename "\${CK}" .pt)
  fi
  OD="${OUT_ROOT}/\${TAG}"
  echo "=== \${TAG}: \${CK} -> \${OD} (log \${OD}.log)"
  T0=\$(date +%s)
  if python scripts/diagnostics/diag_holdout_trace.py --ckpt "\${CK}" --n ${N} --n_vis ${N_VIS} \\
        --dump_bones 1 --dump_bone_names "${EXTRA_BONES}" --out "\${OD}" > "\${OD}.log" 2>&1; then
    echo "    done in \$(( \$(date +%s) - T0 )) s"
    grep -E "^\[load_model\] .*(ckpt_args|WARNING)|^\[dump_bones\]" "\${OD}.log" || true
    grep -A 3 "=== ROTATION ERROR BY BONE GROUP" "\${OD}.log" | grep -E "^variant|^V1_eval_masked" || true
    sed -n '/=== NECK\/HEAD BONE DUMP/,/^wrote .*bones.npz/p' "\${OD}.log"
    TAGS="\${TAGS} \${TAG}"
  else
    echo "WARNING: trace failed for \${CK}; its saved args (H3; load_model prints them before building the model):"
    grep -E "^\[load_model\] .*(saved ckpt_args|WARNING)" "\${OD}.log" || echo "    none: torch.load itself failed"
    echo "tail of \${OD}.log:"; tail -n 30 "\${OD}.log"
    TAGS="\${TAGS} \${TAG}"                                           # the table still shows its pose_param
  fi
done

export OUT_ROOT_DIR="${OUT_ROOT}" TAG_LIST="\${TAGS}"
python - <<'PY'
import json, os
root, tags = os.environ['OUT_ROOT_DIR'], os.environ['TAG_LIST'].split()
BONES = ['neck01', 'neck02', 'neck03', 'head', 'jaw']
def f(x, nd=1):
    return '-' if x is None else format(x, '.' + str(nd) + 'f')
print()
print('=== NECK BONES ACROSS CHECKPOINTS (V1_eval_masked: verdict median/mean local error deg; girth = neck-skin radius pred/GT;')
print('    rawGT = H4 check of the raw dataset blocks; pose_param = H3, as saved in the checkpoint) ===')
print(format('checkpoint', '<36') + ''.join(format(b, '>24') for b in BONES) + format('girth', '>7') + format('PVE_nk', '>8')
      + format('rawGT', '>7') + '  pose_param')
for t in tags:
    p = os.path.join(root, t, 'bones_summary.json')
    if not os.path.isfile(p):
        pp = 'not printed (torch.load failed)'
        lg = os.path.join(root, t + '.log')
        for ln in (open(lg, errors='replace').read().splitlines() if os.path.isfile(lg) else []):
            if ln.startswith('[load_model]') and 'saved ckpt_args' in ln and 'pose_param=' in ln:
                toks = ln.split('pose_param=', 1)[1].split(' ')     # value up to the next key=value token
                n = next((i for i, x in enumerate(toks[1:], 1) if '=' in x), len(toks))
                pp = ' '.join(toks[:n])
        print(format(t[:35], '<36') + format('trace FAILED, no bones_summary.json (see ' + t + '.log)', '<142')
              + '  ' + pp); continue
    js = json.load(open(p))
    s = js['variants']
    v = s.get('V1_eval_masked') or next(iter(s.values()))
    cells = []
    for b in BONES:
        d = v['bones'].get(b)
        cells.append(format('-' if d is None else (d['verdict'] + ' ' + f(d['err_median']) + '/' + f(d['err_mean'])), '>24'))
    sk = v['neck_skin']
    g = (v.get('gt_raw') or {}).get('verdict', '-')
    g = 'ok' if g.startswith('proper') else ('BAD' if g.startswith('NOT') else '-')
    pp = (js.get('ckpt_args') or {}).get('pose_param', '-')
    print(format(t[:35], '<36') + ''.join(cells) + format(f(sk['neck_girth_ratio_mean'], 2), '>7')
          + format(f(sk['pve_neckskin_local_mm']), '>8') + format(g, '>7') + '  ' + str(pp))
print('PVE_nk = neck-skin PVE centred at each mesh\'s own neck01 joint (mm). Full per-bone tables (incl. clavicles, arms,')
print('metacarpals, raw-GT det/singular values) are in each <tag>/bones_summary.json and in the blocks above.')
PY
EOF2
)
srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
