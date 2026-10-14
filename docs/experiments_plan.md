# Experiment plan for the week

Production checkpoint: `anny_s2_shape_v5/checkpoints/00099.pt` (`slurm/history/train_s2_shape_v5.sh`). Evidence:
`docs/diagnostics_findings.md`. Experiment commits on top of `0d8ef72`: `6b0a305` (exp1), `6976688` (exp2),
`9ad20bc` (exp3), `3d51a7d` (exp4). One review-fix commit follows them; the review findings, fixed and open, are in
section 4.

> **Nothing in this plan has been run.** Every number is a measurement of the production checkpoint quoted
> from `docs/diagnostics_findings.md`, a value read from code, or arithmetic. No improvement is claimed.

**Fixed for every experiment.** Train = AnnyOne indices 0-579,859. Holdout = the last 100 images
(579,860-579,959, `--val_anny_n 100`). Evaluation = `scripts/diagnostics/diag_holdout_trace.py`, variant **V1**
(fp32/TF32, helper bones masked, `relax_hands` off, det_thresh 0.2, NMS 3), on the 100-image holdout and on the
diag manifest (`results/diagnostics/manifest.json`, never rebuilt). Every new flag defaults off. With the
defaults, the only observable change in train.py is one `[throughput]` line per epoch.

## 1. Overview

| exp | failure addressed | type | flag(s), default = previous behaviour | commit | pilot runtime (est.) | decision metric (baseline 00099, V1) |
|---|---|---|---|---|---|---|
| exp1 | depth: pred = 0.33·GT + 2.36 m, body drawn ~35% too small | confirmed bug fix | `--gt_center_joint person_center` (default `root`); `--debug_anchor N` (0) | `6b0a305` | 20 min train + ~10 min eval; job ~1 h | `depth_fit_slope` 0.33 → 1 and `depth_fit_intercept_m` 2.36 → 0; guards: PVE ≤ 71.5 (70.5), recall ≥ 89.2 (90.2) |
| exp2 | children come out as short adults | sampling (+ shape-loss weighting) | `--age_balance 1` (0), `--age_balance_child_frac 0.25`, `--age_balance_child_age 0.3`, `--age_balance_index`, `--exclude_indices_file` (''), `--shape_weights` ('') | `6976688` | index scan (unmeasured) + 2×200-batch measurement + 20 min train + ~10 min eval + 2 traces; limit 4 h | manifest child tag (n=10, excluded from the fine-tune): PVE < 47.8, \|age bias\| < 0.078; no holdout regression: PVE ≤ 71.5, recall ≥ 89.2, \|age bias\| ≤ 0.03 (−0.015); youngest-50 is a secondary in-sample read only |
| exp3 | neck: global head orientation off by 12.0° / 19.9° | supervision | `--alpha_global_rot 5` (0.0), `--global_rot_bones head,neck03,wrist.L,wrist.R` | `9ad20bc` | 20 min train + ~10 min eval + 2 manifest traces; script budgets ~2.5 h | `rot_global_deg.head` < 12.0 (holdout) and < 19.9 (manifest train); same PVE/recall guards; fingers 17.3° not worse by > 1° |
| exp4a | fingers worse than the rest pose (17.3° vs 15.1°): resolution? | inference-only measurement | trace `--person_crop 1` (0), `--crop_factor 1.3`; `measure_hand_px.py` | `3d51a7d` | no training; ~40 min | paired V4 `rot_fingers` below the paired V3 relaxed floor (15.1), with Δpve, Δpa_pve, Δpve_hn ≤ +1 mm |
| exp4b | fingers: supervision? (a control, not a finger boost) | supervision | `--rotmat_region_norm 1` (0) | `3d51a7d` | 20 min train + ~10 min eval + traces; job ~1.5 h | V1 `rot_fingers` vs 17.3 and vs its own V3 floor; head 89.5° local / 12.0° global; PVE/recall guards |
| exp4c | 3DPW identity fingers supervised as GT (mixed runs only) | confirmed bug fix, not in the production lineage | `--mask_3dpw_hands 1` (0) | `3d51a7d` | no script, not scheduled | n/a this week |
| exp5 | neck: local neck/head error 62.7° / 89.5°, constant over the v5 lineage (global head 12.0°) | diagnostic first, then supervision (loss form) | trace `--dump_bones 1` (0); `--rotmat_loss geodesic` (default `l1`) | not committed | dump ~20 min for 4 checkpoints; pilot ~20 min train, job ~1 h | dump: verdict per bone plus the H3/H4 bookkeeping lines; pilot: flagged bones' mean local error −3° beyond the l1 control arm; guards vs the control arm: PVE +8 mm, recall −2, global head +3° |

Every baseline the scripts print (recall 90.2, PA-PVE 53.3, holdout age bias −0.015, PVE_hn 52.3, global wrist
22.7°, head px error 64.5 mean / 46 median, ...) is now listed in `docs/diagnostics_findings.md`, "Baseline numbers
used by the experiment scripts", with its source (`results/diagnostics/REPORT.md` on the cluster). Confirm them
with the Day-1 baseline re-trace (section 4) before comparing anything against them.

## 2. Experiments

**Common pilot recipe.** Every training script copies the `train_s2_shape_v5.sh` flags (freeze_backbone 1 +
unfreeze_last_n_blocks 4, mask_helper_joints_train 1, use_anny_shape 1, alpha_shape 20, alpha_j2d/v2d 0.1,
boost_neck/hand 4, batch 4, lr 1e-5, amp 0, person_center head, img_size 672), starts from `CKPT=00099.pt` and
adds only `--start_2d_epoch 0`, `--n_iters_per_epoch 1000 --max_iter 2000`, `--eval_freq 1`, `--nb_max_ckpt 2`,
`--log_freq 50` and `--save_dir results/experiments/<exp>/<RUN> --name train` (a fresh directory).
`--start_2d_epoch 0` is mandatory: train.py has no resume, the epoch counter restarts at 0, and the production
value 20 would keep j2d/v2d off for the whole pilot. Each script has a **control arm** (same schedule, new flag
off); read every pilot against it, not only against 00099. Pilots restart at lr 1e-5 with fresh Adam state,
while v5 ran its last 10 epochs at 1.25e-6 (`LR=` overrides it). A pilot sees 8,000 images (1.4% of the training
set) without seed control: it is a go/no-go read for the longer run, never a validation result.

**Runtime arithmetic** (6.7 images/s measured at batch 4 with 4 unfrozen blocks, i.e. 1.65-1.71 it/s):

| run | iterations | images | training time | plus |
|---|---|---|---|---|
| pilot, 2 epochs | 2,000 | 8,000 | ~20 min | ~10 min evaluation, startup, script extras |
| longer, 10 epochs (`MAX_ITER=10000`) | 10,000 | 40,000 | ~1.7 h | 10 in-training V0 evaluations + traces |
| full, 100 epochs | 100,000 | 400,000 | ~16.5 h | not planned this week |

The 6.7 images/s comes from a 40-image overfit pilot. Rescale from the pilot's own `train.log` line
`[throughput] epoch 1: ... images/s` (epoch 0 includes worker spin-up): `hours = MAX_ITER × 4 / images_per_s / 3600`,
and raise `--time` to match.

### exp1_depth_anchor: confirmed bug fix (`6b0a305`)

**Hypothesis.** The model attaches the mesh to the detected pixel by its HEAD and reports the head as transl
(`model.py:121, 430-432, 442`), while `prepare_gt` builds transl, dist, heatmap cell and offset at joint 0, the
ROOT (`train.py:213-222, 239-245`). The 2D terms compare absolute projections made with the predicted transl
(`model.py:496-497`, `loss.py:117-120`), so they can reconcile the two only by shrinking the body, i.e. by
predicting more depth. j3d/v3d are pelvis-centred and blind to this (`loss.py:104-106`).

**Evidence [Certain].** Holdout, 222 people: pred root depth = 0.33·GT + 2.36 m (mean 3.00 vs 1.92 m), 100%
predicted farther, r(GT depth, error) = −0.76; skeletal height right (1.405 vs 1.409 m) but head-to-root 82 vs
127 px (0.67). Same slope on the manifest splits and the youngest-50 (not a generalisation effect). dist and
transl are 0.6% and 1.9% of the loss. Capped overfit: transl 1.67 → 1.89 and j2d 58 → 75 rose while 3D fell.

**What changed.** train.py `--gt_center_joint person_center` sets target transl and dist to
`self.model.person_center_idx` on AnnyOne batches; heatmap cell, offset, visibility filter and cell occupancy
follow; `transl_pelvis` stays joint 0; the SMPL-X/3DPW branch is untouched. `--debug_anchor N` prints pred
offset_y/loc_y/dist next to the GT anchor and the GT head/root rows for N iterations. The trace's placement block
gains `depth_fit_slope`, `depth_fit_intercept_m`, `depth_ratio_median`, `size_ratio_median_px`, `depth_fit_n`.

**Configuration and env overrides.** `GT_CENTER` (person_center; `root` = control arm), `ALPHA_DIST` and
`ALPHA_TRANSL` (1; variant 5), `DEBUG_ANCHOR` (2), `BASELINE_TRACE` (0), `CONTROL_RUN` (empty; a finished
`GT_CENTER=root` run directory whose slope/intercept/PVE/recall are printed side by side), `LR`, `N_ITERS`,
`MAX_ITER`, `RUN`.
Evaluation: holdout trace and manifest-holdout trace on the last checkpoint.
```
sbatch slurm/experiments/exp1_depth_anchor.sh                                # the fix
GT_CENTER=root sbatch slurm/experiments/exp1_depth_anchor.sh                 # control arm
ALPHA_DIST=5 ALPHA_TRANSL=5 sbatch slurm/experiments/exp1_depth_anchor.sh    # variant: direct depth terms x5
MAX_ITER=10000 RUN=long_person_center sbatch --time=04:00:00 slurm/experiments/exp1_depth_anchor.sh   # only after a passing pilot
```
**Runtime.** About 1 h (limit 3 h): 20 min training, 2 in-training V0 evaluations, 2 traces. 10 epochs: ~1.7 h
of training.

**Success** (`variants.V1_eval_masked.placement` in `holdout_trace/summary.json`): slope toward 1 from 0.33 AND
intercept toward 0 from 2.36 m, with PVE ≤ 71.5 and recall ≥ 89.2. Secondary: size ratio toward 1 from 0.67; the
`[debug_anchor]` lines (root targets: pred offset_y outside [−0.5, 0.5], pred dist above GT dist).

**Would / would not prove.** A slope change relative to the `GT_CENTER=root` arm would show that the anchor
target drives depth. It would not show that depth generalises to real photos, which have no metric GT. The
in-training V0 numbers use a cohort that moves with the anchor, so compare them only within an arm. The trace
keeps its own root-based cohort (`diag_common.visible_gt_mask`).

### exp2_children: sampling (`6976688`; details in `docs/experiments/exp2_children.md`)

**Hypothesis.** The adult bias is a data prior, not a bug: every link of the shape path checks out (section 3a).
With 2.53% child people, roughly 79% of batch-4 batches contain no child. That figure is arithmetic
(0.9747⁹), not a measurement [Guessing]. Raising the child share and the age-error weight should move the
child predictions without breaking the adults.

**Evidence [Certain].** 2.53% of training people have age < 0.3, 1.06% < 0.2; the holdout has none. Youngest-50
in-sample babies (GT 0.015) are predicted at 0.10 with head/height 101% of GT: the representation works. Manifest
children (n=10): PVE 47.8, 41.6 with GT shape (gain 6.2 mm), age bias +0.078. Real photos: age 0.56-0.66 for
small people. Shape is 25.7% of the loss; age has the largest shape gradient.

**What changed.** `--age_balance`: WeightedRandomSampler (with replacement) over the training Subset; child
images get `child_frac` of the draws, `--exclude_indices_file` indices get weight 0. `--shape_weights`: 11
multipliers on the per-phenotype L1, not renormalised (`1,3,1,1,2,1,1,1,1,1,1` raises the weight sum 11 → 14,
+27% shape share at alpha 20). Scripts: `build_age_index.py` (pkl-only scan), `measure_batch_ages.py` (the real
loader; ages of the supervised people after the visibility filter; childless-batch fraction).

**Configuration and env overrides.** `AGE_BALANCE` (1; 0 = control, which keeps the exclusion and the shape
weights), `CHILD_FRAC` (0.25), `CHILD_AGE` (0.3), `SHAPE_WEIGHTS` (age ×3, height ×2; `''` = unweighted),
`N_MEASURE` (200), `INDEX`, `INDEX_WORKERS` (16), `SKIP_MEASURE`, `SKIP_TRAIN`, `BASELINE_TRACE`,
`YOUNG_SCAN` (20000), `YOUNG_N` (50), `LR`, `N_ITERS`, `MAX_ITER`, `RUN`. Evaluation: holdout, manifest train
split, and youngest-50.
```
SKIP_TRAIN=1 sbatch slurm/experiments/exp2_children.sh      # index + batch-age measurement only
sbatch slurm/experiments/exp2_children.sh                   # + pilot + traces
AGE_BALANCE=0 sbatch slurm/experiments/exp2_children.sh     # control arm (sampler off)
SHAPE_WEIGHTS='' sbatch slurm/experiments/exp2_children.sh  # sampler only
MAX_ITER=10000 RUN=long_balanced sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
```
**Runtime.** Index scan I/O bound, unmeasured (30-90 min single process, minutes with 16; it prints seconds per
10k files); measurements a few minutes; training 20 min + 2 evaluations; three traces 20-30 min. Limit 4 h.

**Decision rule.** The manifest child tag (n=10, in the diag manifest and excluded from the fine-tune): PVE < 47.8
and |age bias| < 0.078 (also read the shape-swap gain vs 6.2 mm), plus no regression on the 100-image holdout:
PVE ≤ 71.5, recall ≥ 89.2, |age bias| ≤ 0.03. The youngest-50 trace (predicted age vs GT 0.015, baseline 0.10)
is kept as a secondary, in-sample read and does not enter the rule: those are training people, oversampled by
the balanced loader. Measurement: childless-batch fraction, baseline loader vs balanced loader.

**Would / would not prove.** No held-out child exists, so every child number is in-sample: the manifest
children were seen by the base model for 100 epochs (only excluded from this fine-tune) and the youngest-50 come
from the training pkls and are oversampled. A pass shows the shape head moves toward children when it sees more
of them and the adults did not break; it does not show generalisation to real children. The only out-of-domain
read is qualitative: `CKPT=<pilot ckpt> sbatch slurm/diagnostics/diag_crop_camera.sh` (E4, 20 people, no GT
age). The threshold 0.3 ≈ 11 y assumes the default Anny age anchors.

### exp3_neck_global_rot: supervision (`9ad20bc`)

**Hypothesis.** A 4-bone neck chain can realise one head pose in many ways. The per-bone local rotmat L1
penalises every decomposition except the annotator's, so the global head pose gets a noisy signal. An L1 on the
post-FK camera-frame orientation of head, neck03, wrist.L and wrist.R gives one consistent target.

**Evidence [Certain].** Local error neck 62.7°, head 89.5°, spine 5.7°; global head error 12.0° (holdout), 8.6°
(manifest holdout), 19.9° (manifest train). Neck length 99.3% of GT. Neck-category PVE 93.7 (train) / 76.3
(holdout). All 16 body-model perturbations pass: no mapping or convention error. `--boost_neck_weight 4` is
really ×2.41 after the mean-1 renormalisation and moved the neck chain from ~0.7% to ~1.1% of the total loss,
with no measurable effect.

**What changed.** model.py exposes `out['global_rotmat']` (a view of the already computed
`bone_poses[..., :3, :3]`); `prepare_gt` stores the GT counterpart from the same FK pass, only when alpha > 0;
loss.py adds the per-bone L1 over 9 entries, mean over the selected bones and people (alpha 0: not computed, not
logged); `configure_global_rot` refuses unknown, duplicate or masked bones and bones below a masked one.

**Configuration and env overrides.** `ALPHA_GROT` (5; 0 = control arm), `GROT_BONES`, `EXCLUDE_MANIFEST` (1),
`MANIFEST_TRAIN` (1), `BASELINE_TRACE`, `LR`, `N_ITERS`, `MAX_ITER`, `RUN`. Alpha 5 [Guessing]: per-bone L1 ≈
0.0615 × error_deg, so the raw term is ~1.07 per person; the weighted rotmat term is 4.5-8.8; 5 × 1.07 ≈ 5.3 is
~7% of the total. Rescale if the first log line puts alpha·global_rot outside 0.5-2× of 0.1·rotmat. Evaluation:
holdout, manifest holdout, manifest train.
```
sbatch slurm/experiments/exp3_neck_global_rot.sh                           # alpha 5
ALPHA_GROT=0 sbatch slurm/experiments/exp3_neck_global_rot.sh              # control arm
GROT_BONES=head,neck03 sbatch slurm/experiments/exp3_neck_global_rot.sh    # neck only
MAX_ITER=10000 RUN=long_alpha5 sbatch --time=06:00:00 slurm/experiments/exp3_neck_global_rot.sh
```
**Runtime.** Training 20 min + evaluations; the three 4-variant traces are unmeasured; the script budgets ~2.5 h
(limit 5 h); `BASELINE_TRACE=1` adds 40-60 min.

**Success.** `rot_global_deg.head` < 12.0 (holdout) and < 19.9 (manifest train); PVE ≤ 71.5; recall ≥ 89.2.
Also read neck PVE, PVE_hn, `shape_bias.age`; secondary: global wrist error, local fingers not worse by > 1°.

**Would / would not prove.** A drop relative to the `ALPHA_GROT=0` arm shows that the term reaches the head
orientation. It does not make the local neck bones match GT, and that is not expected, since the decomposition
is not unique. The term also back-propagates into the shape head through the shape-dependent rest frames. The
wrists are the parents of the fingers, so the fingers may move in either direction.

### exp4_fingers: measurement first, then a supervision control (`3d51a7d`)

**Hypotheses.** Resolution: at the median holdout scale an adult hand is ~36×16 px (~3 tokens), a finger ~1 token
long and 0.24 token wide [Likely]. Supervision: the rotmat sum already gives the 30 phalanges 40% of the active
budget (70% under the v5 boosts) with no effect.

**Evidence [Certain].** Finger error 17.3° vs 15.1° relaxed to the rest pose; finger |g| 4.7 vs root 598; 6D
conditioning fingers 0.25-0.30 vs neck 0.99; a finger perturbation moves only that finger; the "hands" manifest
people are truncated (63% of joints in the image), so their PVE is confounded.

**What changed.** `measure_hand_px.py`: hand bbox in px and tokens at 1280 px, at 672 px and under the crop.
Trace `--person_crop 1` adds `V4_person_crop`: V1 settings, one forward per kept GT person on a square crop of
the original render (side = factor × long side of the GT-joint bbox, clamped inside the image), crop camera
K = [f·r, (cx−left)·r; (cy−top)·r], r = 672/side; a paired V4-vs-V1 block with the V3 floor and magnification
bins; V0-V3 outputs unchanged. `--rotmat_region_norm 1`: rotmat L1 averaged within body (41 bones), hands (30),
head-neck (4) and across regions, × 75 (×0.61 / ×0.83 / ×6.25), replacing the boosts in this term only.
`--mask_3dpw_hands 1`: drops the 38 finger/thumb/metacarpal bones of 3DPW people from rotmat and global_rot.
Combined with `--rotmat_region_norm 1`, the masked person's hand residuals are zeroed before the per-bone region
weights, so that person's hands region contributes zero rather than being excluded from the mean (a dilution by
the 3DPW share; irrelevant to AnnyOne-only pilots).

**Configuration and env overrides.** `CROP_FACTOR` (1.3), `PILOT` (1; 0 = inference only), `REGION_NORM`
(1; 0 = control arm), `ALPHA_GROT` (0), `MANIFEST_TRAIN` (1), `EXCLUDE_MANIFEST` (1), `LR`, `N_ITERS`,
`MAX_ITER`, `RUN`.
```
PILOT=0 sbatch slurm/experiments/exp4_fingers.sh                    # exp4a: hand sizes + crop trace, no training
CROP_FACTOR=2.0 PILOT=0 sbatch slurm/experiments/exp4_fingers.sh    # looser crop (person ~50% of the frame)
sbatch slurm/experiments/exp4_fingers.sh                            # exp4a + exp4b region-norm pilot
REGION_NORM=0 sbatch slurm/experiments/exp4_fingers.sh              # exp4b control arm
```
**Runtime.** About 40 min with `PILOT=0`. About 1.5 h with the pilot (limit 4 h).

**Success.** exp4a: paired V4 `rot_fingers` below the paired V3 floor with Δpve, Δpa_pve, Δpve_hn ≤ +1 mm →
resolution is the bottleneck; V4 > 0.5° better than V1 but above the floor → partial; otherwise supervision.
exp4b: `rot_fingers` vs 17.3 and vs its own V3 floor; head 89.5° local / 12.0° global; PVE/recall guards.

**Would / would not prove.** V4 is not comparable on placement or depth (excluded): at factor 1.3 the crop camera
has a ~31° fov vs 51-110° in training, and the depth prior was learned full-frame. Recall on frame-filling crops
is unknown. For a 300-px person the 1.3 crop is ×3.4 vs the 672 input but only ×1.9 real detail [Likely]. Region
normalisation lowers the hand share (40% → 33%), so a null finger result says nothing about upweighting. Neither
arm tests the 6D-init defect (3h). exp4c is not exercised (the production lineage is pure AnnyOne).

### Neck root cause: per-bone dump, then exp5_neck_fix (`--rotmat_loss`, not committed)

**What is known.** [Certain] Local (parent-relative) error neck01-03 62.7°, head+jaw 89.5° on v5 00099, the same on
v9 00062 / 00149 and in all six v5 pilot and control arms (neck 62.5-62.8°, head 89.1-90.1°); per person 62.7 ± 1.6 and
89.5 ± 1.5 (222 holdout people); V0 (unmasked) = V1. Global head error 12.0° (v5) vs 12.9° (v10). v10, which had its own
fresh heads, gets 5.0° / 4.1° on the neck and head but ~50° on clavicle/shoulder and 36.3° on the arms. These are group
means: the per-bone split has never been measured.

**Bookkeeping: what is refuted and what is not.** The code audit refuted the mask, weight, eval and 3DPW causes:
`useful_rotmat` is 1 on bones 100-104, the boost raises neck01-03/head to ×2.41 (jaw ×0.60), all five bones get
non-zero decpose gradients, training and trace use the same GT tensor and SVD projection, and the v5 lineage never saw
3DPW. Two bookkeeping hypotheses are **untested** [Guessing]:
- H3: an ancestor of the inherited heads was trained in another `pose_param` (archive/train_v1.py had world /
  world-orient), which model.py ignores.
- H4: the raw AnnyOne neck/head blocks are not proper rotations, so the SVD-projected rotmat target disagrees with the
  GT mesh and joints, which train.py builds from the raw blocks.

The dump below now checks both. It records the determinant and singular values of the raw blocks (H4), and the
checkpoint's saved `pose_param` and loss args (H3; `diag_common.load_model` also prints them). For H3 the ancestors have
to be traced too, since v5's args describe only v5's own run. Read these lines first.

**Leading hypothesis** [Likely]: a compensated 180° flip, an optimisation problem. One neck bone is turned half a turn
about its own axis and the head turns back, so joints and head orientation stay correct. The flip would be seeded by the
singular 6D init (3h) in the lineage's only fresh-head phase (`anny_s2_frozen_bb`, inherited by v3, v4, v5, v9 and every
pilot). The L1 over matrix entries would hold it in place, since it has a barrier along a twist (4.83 at 135° vs 4.00 at
180°).

**Diagnostic: `diag_holdout_trace.py --dump_bones 1`.** The bones dumped are the parent spine bone, neck01-03, head
and jaw, plus the clavicles, shoulder01s and wrists. `slurm/diagnostics/diag_neck_bones.sh` also adds the arm and
metacarpal bones. For each matched person, variant and bone it records:
- the local error, and the angle of pred and of GT from identity;
- the global error;
- the swing/twist split of E = R_gtᵀ R_pred about the bone's rest axis;
- the bone-head joint error;
- the raw local and global rotations;
- det and singular values of the raw dataset block (H4).

The rest axis is the child-offset direction in the bone's own rest frame (neck01→neck02→neck03→head, spine01→neck01,
clavicle→shoulder01). Head, jaw and the wrists have no connected child and use local +Y (Blender head→tail). The script
prints the cosine of each child offset with +Y.

Per bone and variant it summarises:
- mean / median / p90 error and the fraction of people above 150°;
- the across-person spread of pred, of GT and of E, each measured as the mean geodesic distance to the chordal mean
  (this is the constancy test);
- the raw-GT check: det range, share of det < 0, singular-value range, and a verdict (proper rotations if det > 0 and
  |s − 1| ≤ 1e-3);
- a neck-skin PVE over the surface vertices skinned ≥ 50% to neck01-03 (head excluded). It is reported pelvis-centred,
  like PVE_hn, and also centred at each mesh's own neck01 joint, with a girth ratio pred/GT.

**Verdict per bone** (in this order):

| label | rule |
|---|---|
| `FLIP_TWIST` | error > 150° in ≥ half the people, and the bone's own axis moves < 30° (median swing). That is a half-turn about an axis within ~15° of the bone, so the child joint stays put |
| `FLIP_SWING` | the same flip, but the bone axis moves ≥ 30°: the child joint moves |
| `PARTIAL_FLIP` | > 150° in 10-50% of the people; the mean is far above the median |
| `OK` | median < 15° |
| `FIXED_OUTPUT` | pred spread < ¼ of GT spread |
| `CONST_OFFSET` | R_pred ≈ R_gt C with one C (spread of E < 10°) |
| `SPREAD` | none of the above |

The swing decides between the two flip labels. Near 180° the twist angle about the bone axis reads ~180 for any rotation
axis not perpendicular to the bone: a 180° turn about an axis 75° off the bone still gives |twist| ≈ 176°.

Outputs: `<out>/bones.npz`, `<out>/bones_summary.json` and a printed table with a legend. With the default 0, every output
is unchanged.
```
sbatch slurm/diagnostics/diag_neck_bones.sh     # v5 00099, v9 00149, v10 00290, exp1 long_pc_dist5 00009
CKPTS="models/multiHMR/anny_s2_frozen_bb_v2_00299.pt logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt logs/anny_model/anny_s2_boosted_v4/checkpoints/00299.pt" \
    sbatch slurm/diagnostics/diag_neck_bones.sh # lineage: where the offset first appears; also H3 for the ancestors
```
**Runtime.** One job, no training. The existing holdout trace runs its 100-image loop in 65-105 s per checkpoint
(progress lines of the diag_holdout_35* logs), plus start-up. That makes ~3-5 min per checkpoint and ~20 min for the
four defaults; the job limit is 1 h. Output goes to `results/diagnostics/neck_bones/<tag>/` and `<tag>.log`. The job log
ends with a cross-checkpoint table: verdict and median/mean per bone, girth, neck-skin PVE, raw-GT check, `pose_param`.

**How to read it (V1 rows of `bones_summary.json`).** A label says what the rotations are, not why. Each mechanism
has its own deciding test:

| what is claimed | decided by |
|---|---|
| a compensated flip is present | the dump (this job) |
| the L1 holds it | exp5 against its l1 control arm |
| where it came from (seeded at init, or created when the helper mask was switched on) | the lineage traces |
| it is the visible thinning | girth, plus a render of the decpose-surgery checkpoint |
| bookkeeping H3 / H4 | the `checkpoint args` and `raw GT blocks` lines (H3 for the ancestors: the lineage traces) |

| dump outcome | consistent with | next step |
|---|---|---|
| `raw GT blocks ... NOT proper rotations` | H4: the rotmat target and the GT mesh disagree for those bones | build the GT mesh/joints from the projected rotations in prepare_gt and the trace (candidate flag, not implemented) before anything else |
| `pose_param` other than `local-bone` / None | H3: rotations read in the wrong space | honour pose_param or refuse the checkpoint before anything else |
| exactly one of neck01/02/03 `FLIP_TWIST`, head `FLIP_*` with one axis, jaw `OK`, bone-head joint error small; v9 identical to v5 | a compensated flip (H1) | zero-training decpose surgery on that bone; exp5 vs control tests the L1 trap; lineage traces test the origin |
| v10's neck `OK` with its flips on arm/metacarpal bones | a per-init random draw, **or** a recipe-dependent cause (v10 differs in helper-mask timing, boost, 3DPW from step 0, unfrozen blocks, LR) | does not decide between them; the lineage traces and the other fresh-head runs (anny_partial_freeze = v8, anny_v7_scratch) do |
| the flipped bone is neck01 or neck02 and girth pred/GT < ~0.8 | a flip that thins the neck skin (the 0.8 comes from an LBS simulation; PVE_hn is equal for v5 and v10) | render the surgery checkpoint before calling it the long-neck look |
| the flipped bone is neck03 and girth ≥ ~0.9 | a flip that is nearly invisible in the mesh; the long-neck look has another cause (skeleton length is 99.3% of GT) | look at shape / rendering |
| `FLIP_SWING` | a flip that tilts the bone and moves its child joint, not the H1 twist | inspect joint error and global error of that bone |
| `PARTIAL_FLIP` | per-image flips in a minority of people | read mean vs median; per-person rows in bones.npz |
| neck01-03 all `CONST_OFFSET` with one shared axis (relAng ~62.7°) | a convention or frame offset (bookkeeping) | check H3 / H4 lines, then the mapping |
| `FIXED_OUTPUT` (pred spread < ¼ of GT spread) | the head emits one rotation whatever the image | inspect the decpose rows of those bones |
| `SPREAD` with large medians | ordinary input-dependent error, no constant offset | supervision (exp3) |

**Fix flag: `--rotmat_loss {l1,geodesic}`** (loss.py, default `l1`). With `l1`, the original code path runs
unchanged; the default loss equals HEAD bit for bit on random inputs in four configurations (plain, boost 4/4, region
norm, hand mask + global_rot). With `geodesic`, the per-bone angle replaces the 9-entry L1 in the rotmat term, in
global_rot and in train.py's output-level distillation rotation term (`--lambda_kd_out`). The angle is computed as
atan2(|vee(M−Mᵀ)|, tr M − 1), which keeps the gradient finite at 0° and 180°. The hand mask, joint/region weights, sum
over bones and mean over people are the same. Use `--alpha_rotmat 0.33`: at a 2° error the 9-entry L1 is 3.27× (GT near
identity) to 3.50× (arbitrary GT) the angle in radians. `--alpha_global_rot` is **not** rescaled, so pass ~3.5× the
L1-era value under geodesic (17.5 for exp3's 5). Distilling from a flipped teacher (v9) still pulls toward its flip, under
either distance. Synthetic check: 16 bones started 175° off by twist, 600 Adam steps on the 6D vector. L1 ends at a
median 164° (16/16 still above 90°); geodesic ends at 0.1° (0/16).
```
C=$(ROTMAT_LOSS=l1 sbatch --parsable slurm/experiments/exp5_neck_fix.sh)   # control arm (alpha 0.1)
CONTROL_RUN=results/experiments/exp5_neck_fix/pilot_l1_a0.1_anny_s2_shape_v5_00099 \
    sbatch --dependency=afterok:$C slurm/experiments/exp5_neck_fix.sh   # geodesic, alpha 0.33; summary includes the control
# or submit both without CONTROL_RUN, then, once both finished (seconds, no GPU):
SUMMARY_ONLY=1 CONTROL_RUN=results/experiments/exp5_neck_fix/pilot_l1_a0.1_anny_s2_shape_v5_00099 \
    bash slurm/experiments/exp5_neck_fix.sh
MAX_ITER=10000 RUN=long_geo sbatch --time=05:00:00 slurm/experiments/exp5_neck_fix.sh
```
Run `diag_neck_bones.sh` first: its v5 dump is the baseline. Without it, the job traces CKPT itself (+~5 min). A
`CONTROL_RUN` whose `holdout_trace/` is not finished aborts the job before training, and makes the summary exit 2. The
job never silently falls back to a baseline-only comparison. The comparison lives in
`scripts/experiments/exp5_neck_summary.py` (stdlib only).
**Runtime:** ~20 min of training, plus in-training evaluation and 1-2 dump traces; ~45-60 min in total, job limit 2.5 h.

**Success criterion (V1, holdout).** Three reads, printed side by side; a guard miss is reported next to the neck read,
not instead of it.
- Neck. FIXED: neck and head group means < 15° and girth ≥ 0.9. MOVING: every bone the baseline dump labels `FLIP_*`,
  `PARTIAL_FLIP` or `CONST_OFFSET` drops its mean local error by ≥ 3°, and by ≥ 3° more than in the control arm. The mean
  is used so a `PARTIAL_FLIP` bone counts, and because the 3° floor comes from group means.
- Guards: PVE ≤ ref + 8 mm, recall ≥ ref − 2 points, global head ≤ ref + 3°. ref is the control arm, or the v5 numbers
  70.5 / 90.2 / 12.0 without one.
- Control: whether the control arm was compared. Without it, MOVING cannot be told from fine-tuning drift.

**Measured noise floor** (the six 2-epoch v5-recipe arms, exp2-exp4 pilots and controls, holdout V1):
- PVE 68.7-76.3 mm (7.6 mm spread);
- recall 88.6-90.2 (1.6 points);
- global head 11.3-13.7°;
- local group means 62.5-62.8° (neck) and 89.1-90.1° (head).

The two identical-recipe control arms (exp3, exp4) differ by 7.6 mm PVE, 0.8 recall points and 1.9° global head. The
earlier guard (PVE ≤ 71.5, recall ≥ 89.2) failed all six arms, controls included. The new tolerances cover this spread.

**Would / would not prove.** A drop beyond the control arm shows that the loss form reaches the flagged bone. A null
pilot does not refute the flip. v5's decpose 6D columns have norms of 120-341 (neck02) and 61-182 (head); Adam moves
each weight by about lr per step, and undoing a flip means reversing such a column, which 2,000 steps at 1e-5 may not
do [Guessing]. The decisive zero-training test is the decpose surgery on the bone the dump names (not implemented
here: it needs that bone). The 6D-init defect (3h) only matters for new fresh-head runs.

## 3. Verified in code

Readers' facts. Line numbers refer to `89db760`, whose `.py` files are identical to `0d8ef72`. The experiment
commits shift lines in train.py, loss.py, model.py and the trace.

**3a. Shape outputs reach the body model and are supervised.**
- [Certain] `HPH.decshape` = `nn.Linear(1024, 11)` with default init, plus a zero `init_betas` residual; single
  pass. `model.py:544, 564-572, 647, 654-657`
- [Certain] sigmoid is applied once; all 11 phenotypes go by name into the body model; `out['shape']` is
  post-sigmoid; nothing is clamped or dropped on the training path. `model.py:353, 390, 413, 419, 455`
- [Certain] The loss is an L1 summed over phenotypes, mean over people, × alpha_shape (20). No per-phenotype
  weight, no 0.5 prior. `loss.py:96-97, 138, 173`
- [Certain] GT = clamp(anny_shape, 0, 1) in positional order. The same vector drives the GT mesh and the target.
  `train.py:146-169, 205`
- [Certain] The shape target uses the same `idx_vis` filter as every other target (root in front, inside the
  image, one person per cell). `train.py:205, 213-222, 239-277, 294-311`
- [Certain] Inference toggles are gated by `not is_training`; train.py sets only `mask_helper_joints_train`, so
  `evaluate()` is V0. `model.py:153-193, 360-404`; `train.py:859-862`
- [Likely] Index 1 = age is supported only empirically (the youngest-by-index-1 render as 0.59 m babies).
  `diag_common.py:29-30`

**3b. Normalisation and adult-biased defaults.**
- [Certain] The ImageNet normalisation mismatch (C1a) is false in code (findings). No phenotype normalisation or
  remap exists (3a).
- [Likely] Zero logits give 0.5 on every phenotype, about 18 y on the Anny anchors. The init is not
  adult-biased; the bias is learned from data. `model.py:566, 647, 390`; `diag_common.py:339-343`
- [Certain] `MEAN_PARAMS` is imported but unused; no SMPL mean shape enters the head (`model.py:30`). "0.5 = 25 y"
  exists only in a docstring (`smpl_to_anny.py:149-150`).
- [Certain] The constant age 0.55 enters training only with `--mix_3dpw_labels`, which v5 did not use.
  `build_threedpw_anny_labels.py:90,112`; `train.py:1113-1127`

**3c. Flip and left/right conventions.**
- [Certain] `--flip` and `--brightness` are parsed but never read; `args.flip` appears only in a commented-out
  BEDLAM block. `train.py:1058, 1343-1344`
- [Certain] AnnyOne and ThreeDPWAnny do contain + pad + normalize only, collate only stacks: no mirroring, no
  augmentation (`datasets/annyone.py:42-46`; `datasets/threedpw_anny.py:78-82`; `datasets/bedlam.py:365-426`).
  The only flip code is BEDLAM's, off this path (`datasets/bedlam.py:225-228, 266, 290-291`).
- [Certain] All 16 single-joint perturbations pass; pose parameters act in the bone rest frame (findings,
  checks 2+3).

**3d. Hand labels and visibility masking.**
- [Likely] The pkl `anny_pose` is (163,4,4) for every bone; the "164" comment in train.py is stale.
  `docs/anny_shape_report.txt:11-16`; `train.py:132-133`
- [Certain] The rotmat target is the SVD-orthonormalised 3×3 of all 163 bones. `train.py:132-135, 177-186`
- [Certain] `hand_bone_mask` is selected by name ('finger'/'thumb'). [Likely] On the rig that is 30 phalanges
  including the thumbs; metacarpals and wrists are outside it; the hand boost covers 32. `model.py:168-176`;
  `loss.py:73-75`
- [Likely] All 40 hand bones are active. The 88 masked bones are 28 toes, 2 breasts and 58 face bones.
  `model.py:136-146`
- [Certain] There is no per-joint visibility. A person is dropped whole if the root is behind the camera,
  off-image, or in an occupied cell. `train.py:250-277, 293-309`
- [Certain] The 2D terms keep GT points with 0 < p < img_size, the same rule for every joint; there is no
  occlusion handling. `loss.py:117-120`
- [Certain] The only per-bone mask is the helper identity mask, applied to both sides. `model.py:360-363`;
  `train.py:187-199`

**3e. 3DPW finger labels.**
- [Certain] The converter leaves every unmapped bone at exact identity: phalanges and metacarpals are
  gradient-zeroed, and finger2-1 gets no joint gradient. `smpl_to_anny.py:553-575, 609-621, 667-673`
- [Certain] These identities are written into `anny_pose` and supervised as GT in the mixed runs (ratio 0.3:
  v7/v8/v9/v10). v5 is unaffected. `build_threedpw_anny_labels.py:105-118`; `datasets/threedpw_anny.py:93-105`

**3f. Resolution limits of the model.**
- [Certain] The four HPH tables `cross_queries_x/y` and `cross_values_x/y` have length img_size//14 = 48 and are
  indexed by the detected token row/col. The checkpoint cannot run at 896 or 1288 px without retraining.
  `model.py:208, 515, 551-562, 588-601`
- [Certain] All other grid-dependent code is dynamic and the K/fov path is scale-consistent
  (`model.py:172-174, 188-220, 343-344`); 1280 is not a multiple of 14 (`model.py:75-77`).
- [Certain] `--crops` is dead, and `evaluate()` has no TTA. `train.py:1342, 544-583`;
  `datasets/bedlam.py:232-238`
- [Likely] Arithmetic: a hand spans ~3 tokens at 672 px; a 1.3 crop of a 300-px person fills 77% of the frame
  at a ~31° fov.

**3g. Pilot mechanics.**
- [Certain] There is no resume: epoch, iter and Adam state restart at 0, hence `--start_2d_epoch 0`.
  `train.py:48-49, 315, 344-348`; `loss.py:143-145`
- [Certain] Retention keeps the `--nb_max_ckpt` highest numbers of a directory, hence a fresh directory per run
  (`train.py:351-364`). There is no seed control, so two arms see different images.
- [Certain] v5's final LR is 1e-5 × 0.5³ = 1.25e-6 (StepLR every 30 epochs). `train_s2_shape_v5.sh:122-124`;
  `train.py:330-333`

**3h. Confirmed defects and their status.**

| defect | where | affects 00099? | status |
|---|---|---|---|
| Head anchor vs root targets | `train.py:213-222, 239-245` | yes (depth) | exp1 |
| 3DPW identity fingers as GT | see 3e | no (mixed runs) | exp4c flag, not scheduled |
| 6D "identity" init `[1,0,0,1,0,0]` is singular under `rot6d_to_rotmat` (columns x[0:3], x[3:6]) | `model.py:565`; `utils/humans.py:20-21` | yes (fingers' 0.25-0.30 conditioning is \|sin\| of this angle) [Likely link] | **not implemented** (it seeds the neck flip hypothesis, section 2 "Neck root cause"; distinct from exp5_neck_fix). Candidate: after load, set the init to `[1,0,0,0,1,0]` and shift `decpose.bias` by `(0,0,0,+1,-1,0)` per bone (output-neutral), then reset the 30 phalanx rows to exact identity; the buffer is restored from the checkpoint, so a constant change alone is overwritten |
| Boost renormalised over 163 bones: "×4" is ×2.41, rest ×0.60 | `loss.py:76`, help `loss.py:188-193` | interpretation of v4/v5 | not fixed (proposed `--boost_renorm 0`) |
| `prepare_gt` reads global `args.img_size`; `train_n_iters`/`evaluate` read global `args.amp` | `train.py:239, 448, 570` | no | not fixed |
| `--use_anny_shape` default 0 → GT phenotypes 0 (newborn) | `train.py:148-159, 1337` | no (v5 passes 1) | not fixed |
| Missing `--pretrained` silently skipped | `train.py:734` | no | scripts guard with `[ -f CKPT ]` |
| `clip_dist` stored as a 1-tuple (flag dead) | `model.py:70, 261` | no | not fixed |
| Demo crop path uses the full-image K | `demo.py:366` | placement only (E4) | not fixed |

## 4. Recommended order for the week

**exp1 goes first** because it changes the detection anchor: the heatmap cell, the offset, and the cohort that
`prepare_gt` keeps. exp2-4 pilots start from 00099 with root targets and read against their own control arms,
so they stay independent.

Stacking any of exp2-4 onto an exp1 checkpoint needs `--gt_center_joint person_center` on its train.py line. Each
of the three scripts passes `GT_CENTER` through as `--gt_center_joint` (default `root` = today's behaviour), so
stack with `CKPT=<exp1 checkpoint> GT_CENTER=person_center`. With `CKPT=<exp1 checkpoint>` alone they would
silently train on root targets again.

**Never combine flags until each pilot has been read.**

| day | submit | read |
|---|---|---|
| 1 | Baseline re-trace (below); exp1 pilot + `GT_CENTER=root`; exp4 `PILOT=0`; exp2 `SKIP_TRAIN=1` | Confirm the unverified baselines; get the baseline `depth_fit_*` from the same code |
| 2 | exp3 pilot + `ALPHA_GROT=0`; exp2 pilot + `AGE_BALANCE=0` (reuses the index); if exp1 passes, its 10-epoch run | exp1 both arms; exp4a decision rule; batch-age measurement |
| 3 | exp4b pilot + `REGION_NORM=0` if exp4a says supervision | exp2, exp3 pilots; exp1 long run |
| 4-5 | One longer run at a time, for pilots that passed (~1.7 h of training each) | Final traces; decide on stacking (exp1 base first) |

**Evaluate every checkpoint with the same two commands.** The scripts already run the first command in-job. The
second covers manifest splits a script skips: exp1 skips the train split, exp2 skips the holdout split.
```
LAST=$(ls /netscratch/najib/multi-hmr/results/experiments/<exp>/<RUN>/train/checkpoints/*.pt | sort | tail -n 1)
CKPT=$LAST TAG=<exp>_<RUN> sbatch slurm/diagnostics/diag_holdout.sh    # 100-image holdout; read V1_eval_masked
CKPT=$LAST TAG=<exp>_<RUN> sbatch slurm/diagnostics/diag_manifest.sh   # manifest train + holdout splits
TAG=anny_s2_shape_v5_00099_rerun sbatch slurm/diagnostics/diag_holdout.sh    # Day-1 baseline (default CKPT = 00099)
TAG=anny_s2_shape_v5_00099_rerun sbatch slurm/diagnostics/diag_manifest.sh
```
Always pass `TAG`. Without it, every pilot derives `train_00001` and overwrites the previous one, and the baseline
would overwrite `results/diagnostics/{holdout,manifest_trace_*}_anny_s2_shape_v5_00099`, the evidence behind the
findings.
`results/experiments/` is git-ignored except `*.json` and `*.md`: pilot logs, images, checkpoints and npz stay
untracked, the summaries can be committed.

**Known issues from review.**

The review-fix commit addressed three of these: the batch-age guard (exp2: a batch without any person is now counted
as skipped, as `prepare_gt` returns None for it), the EPOCH parsing (exp3's alpha check, and exp4's first/last-line
grep, now match `EPOCH=` anywhere on the line and trim) and the baseline provenance (exp1 labels `head_px_err`
64.5 as the mean and prints the 46.0 median, the trace's size line names the ratio of medians, and the baseline
table is in `docs/diagnostics_findings.md`). The lr row stays open; exp1 can now print its control arm side by
side (`CONTROL_RUN`). The `[throughput]` line per epoch is the one accepted deviation from a byte-identical
default train.log.

| exp | issue | what to do |
|---|---|---|
| exp1 | The printed "baseline(findings)" `head_px_err` 64.5 is in no committed document (the findings quote a 46 px median). The size-ratio baseline 0.67 is a ratio of medians, while the column is a median of ratios. | Fixed: `head_px_err` is labelled as the mean with the median beside it, the baselines are tabled, and 0.67 is confirmed as the median of per-person ratios (the column's statistic; 82/127 px = 0.65 is the ratio of medians). Still compare with the Day-1 re-trace or `BASELINE_TRACE=1` |
| all | lr 1e-5 is 8× v5's last lr, with fresh Adam state; each script's VERDICT compares one arm with the 00099 baseline only | Always read against the control arm; `LR=1.25e-6` is an option |
| exp2 | `measure_batch_ages.py` raises `KeyError('anny_shape')` on a batch where no image has a person | Fixed: counted as skipped |
| exp3 | The alpha check keeps only lines that start with `EPOCH=`. tqdm shares those lines, so it likely prints "no EPOCH= lines" | Fixed: substring match, trimmed (exp4's grep too) |

## 5. Statement

Nothing in this plan has been executed. It was written on a machine without torch, GPU or data. The code
changes were checked only with `py_compile`, `bash -n` and pure-python or stubbed tests. The runtimes are
extrapolated from one throughput measurement. The success criteria say what a measurement would show, not what
it will show. No experiment here is claimed to improve the model until its pilot and the fixed traces have run
and been read against the control arm.
