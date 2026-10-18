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
| exp5 | neck: local neck/head error 62.7° / 89.5°, constant over v5 00099, v9 00062/00149 and the six exp2-exp4 pilot and control runs (exp1 runs and the ancestors not yet traced; global head 12.0°) | diagnostic first, then supervision (loss form, candidate) | trace `--dump_bones 1` (0); `--rotmat_loss geodesic` (default `l1`) | `bd7cfa3` (WIP, branch `claude/gallant-dijkstra-bnlowv`; review fixes on top) | dump ~20 min for 4 checkpoints; pilot ~20 min train, job ~1 h | dump: verdict per bone plus the H3/H4 bookkeeping lines; pilot: flagged bones' mean local error −3° beyond the l1 control arm; guards vs the control arm: PVE +8 mm, recall −2, global head +3° |
| exp6 | measured half-turns: v5 lineage neck02 (175.6°) + head (177.9°) compensated pair, girth 0.46, metacarpal1.L 180° (constant); v10 metacarpal2.L/2.R/3.R (constant) and a compensated left-arm chain shoulder01.L → upperarm01.L → upperarm02.L; exp5 fine-tuning did not move them | output-level correction (zero-training test, then a fine-tune with it) | `--bone_fix <json>` in train.py / trace / demo ('' = none or the checkpoint's own, `none` = off); trace `--select train_sample`, `--dump_bone_names all` | uncommitted (this working tree) | ~30 min with TRAIN=1, ~10 min with TRAIN=0; limit 1 h | zero-training vs baseline: corrected bones median < 15°, girth ≥ 0.9, global head ≤ +3°, PVE ≤ +1 mm; fine-tuned vs the exp5 l1 control: same bones/girth, exp5 guards |

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

### Neck root cause: per-bone dump, then exp5_neck_fix (`--rotmat_loss`; WIP `bd7cfa3` on `claude/gallant-dijkstra-bnlowv`)

**What is known.** [Certain] Local (parent-relative) error neck01-03 62.7°, head+jaw 89.5° on v5 00099, the same on v9
00062 / 00149 and in all six exp2-exp4 pilot and control arms (neck 62.5-62.8°, head 89.1-90.1°; the exp1 depth runs
printed no rotation table); per person 62.7 ± 1.6 and 89.5 ± 1.5 (222 holdout people); V0 (unmasked) = V1. Global head
error 12.0° (v5) vs 12.9° (v10). v10, which had its own fresh heads, gets 5.0° / 4.1° on the neck and head but ~50° on
clavicle/shoulder and 36.3° on the arms. These are group means: the per-bone split has never been measured.

**Bookkeeping: what is refuted and what is not.** The code audit refuted the mask, weight, eval and 3DPW causes:
`useful_rotmat` is 1 on bones 100-104, the boost raises neck01-03/head to ×2.41 (jaw ×0.60), all five bones get
non-zero decpose gradients, training and trace use the same GT tensor and SVD projection, and v5 and all its ancestors
were trained on AnnyOne only (v9 added 30% 3DPW only after the offset existed; its 3DPW labels pin neck02, neck03 and
jaw to identity, which matters when reading the v9 rows of the dump). Two bookkeeping hypotheses are **untested**
[Guessing]:
- H3: an ancestor of the inherited heads was trained in another `pose_param` (archive/train_v1.py had world /
  world-orient), which model.py ignores.
- H4: the raw AnnyOne neck/head blocks are not proper rotations, so the SVD-projected rotmat target disagrees with the
  GT mesh and joints, which train.py builds from the raw blocks.

The dump below now checks both. It records the determinant and singular values of the raw blocks (H4), and the
checkpoint's saved `pose_param` and loss args (H3; `diag_common.load_model` also prints them). For H3 the ancestors have
to be traced too, since v5's args describe only v5's own run. Read these lines first.

**Leading hypothesis** [Likely]: a compensated 180° flip, an optimisation problem. One neck bone is turned half a turn
about its own axis and the head turns back, so the head's orientation stays correct and the joints stay
*approximately* correct. That is exact only for a neck03 flip, because the head joint sits on neck03's axis. A neck01
or neck02 flip swings everything below it about that bone's axis, so a joint off the axis moves by twice its distance
from it, and the head's compensating rotation cannot move the head's own joint back. On anny 0.6.1's default rig
(local simulation with anny's own model, all phenotypes set to 0.3 / 0.5 / 0.7, the head rotated so its global
orientation equals GT; scratchpad flip_joints.py, not in the repo):

| flip | neck02 joint | neck03 joint | head joint, GT neck at rest | head joint, GT swings 10° / 20° on neck01-03 (mean, max) |
|---|---|---|---|---|
| neck03 | 0 | 0 | 0 | 0 |
| neck02 | 0 | 0 | 9.4-21.1 mm | 12-23 (max 28) / 20-30 (max 40) mm |
| neck01 | 0 | 0.1-11.4 mm at rest; 14-18 / 28-31 mm with the swings | 3.3-9.5 mm | 19-28 (max 44) / 37-59 (max 82) mm |

So a neck01/neck02 flip pays a position cost (j3d/j2d/v3d) that grows with the GT neck bend, which weakens the L1-trap
argument below (the barrier holds against the rotation term only); a neck03 flip pays none but barely changes the
mesh. How bent the AnnyOne GT necks are has never been measured; the dump's `|gt|` column (GT angle from identity)
bounds it. The earlier cluster probe (diag_bodymodel, 8d9b27a9:134-149) called the neck02/neck03/head joints static
under a 30° neck01 twist. It used the rest pose, the dataset-mean shape and a 2 mm "moved" threshold. Here a 30° twist
moves the head 1.1 mm at phenotype 0.3 and 2.5 mm at 0.5, and a 180° flip moves it 2 / (2 sin 15°) ≈ 3.9× as far. So
the probe bounds the rest-pose shift but says nothing about posed necks. The "neck03 ~25 mm, head ~40 mm" reported in
the previous review round came from a posed check (random 20° GT on every bone) and agrees with the posed columns. The
flip would be seeded by the singular 6D init (3h) in the lineage's only fresh-head phase (`anny_s2_frozen_bb`,
inherited by v3, v4, v5, v9 and every pilot). [Measured, diag_neck_bones_3573518 (v5 00099; the ancestors in 3573519 agree): neck02 is a twist flip
(|tw| 175.7, swing 3.3; bone axis local +Y, cos 1.000), and the head is mostly twist too (|tw| ~178, swing 28-32,
never near 180). A half-turn about local X, which is where the singular init starts a bone, would tilt the axis by
180 deg (FLIP_SWING) and does not give that; in the v5 line the init fits metacarpal1.L's FLIP_SWING only.] The L1 over matrix entries would hold it in place, since it has a
barrier along a twist (4.83 at 135° vs 4.00 at 180°); for a neck01/02 flip the position terms push the other way, and
which wins is untested.

**Diagnostic: `diag_holdout_trace.py --dump_bones 1`.** The bones dumped are the parent spine bone, neck01-03, head
and jaw, plus the clavicles, shoulder01s and wrists. `slurm/diagnostics/diag_neck_bones.sh` also adds the arm and
metacarpal bones. For each matched person, variant and bone it records:
- the local error, and the angle of pred and of GT from identity;
- the global error;
- the swing/twist split of E = R_gtᵀ R_pred about the bone's rest axis;
- the joint error at the bone's own head (`joint`). The bone's own rotation never moves that point (anny FK:
  pose_b = pose_parent · rest_parent⁻¹ · rest_b · delta_b, and delta_b has no translation), so it only reflects
  the bones above it;
- the joint error at the head of its twist-axis child (`child`), the first joint the bone's own rotation moves.
  `child` well above `joint` means the bone's global orientation is off; that equals its local error only when
  the parent chain is right;
- the raw local and global rotations;
- det and singular values of the raw dataset block (H4).

The rest axis is the child-offset direction in the bone's own rest frame (neck01→neck02→neck03→head, spine01→neck01,
clavicle→shoulder01). Head, jaw and the wrists use the child whose head is at their tail (special06.L, special04,
metacarpal1.L/R on anny's default rig; helper bones count), so their axis is local +Y (Blender head→tail); the log
shows e.g. `head=child:special06.L (1.000)`. A bone without such a child falls back to local +Y. The script prints
the cosine of each child offset with +Y.

Per bone and variant it summarises:
- mean / median / p90 error and the fraction of people above 150°;
- median swing and median |twist|, the circular mean of the twist (`twist_circular_mean`, atan2 of mean sin and mean
  cos; a signed mean is meaningless near ±180°, where the sign is round-off) with its resultant length, and the share of
  people whose twist is undefined (swing above ~178°, NaN in bones.npz);
- the across-person spread of pred, of GT and of E, each measured as the mean geodesic distance to the chordal mean
  (this is the constancy test);
- the raw-GT check: det range, share of det < 0, singular-value range, and a verdict (proper rotations if det > 0 and
  |s − 1| ≤ 1e-3);
- a neck-skin PVE over the surface vertices skinned ≥ 50% to neck01-03 (head excluded). It is reported pelvis-centred,
  like PVE_hn, and also centred at each mesh's own neck01 joint, with a girth ratio pred/GT.

**Verdict per bone** (in this order):

| label | rule |
|---|---|
| `FLIP_TWIST` | error > 150° in ≥ half the people, and the bone's own axis moves < 30° (median swing). That is a half-turn about an axis within ~15° of the bone, so relative to the parent frame the child joint moves < 0.52 bone lengths |
| `FLIP_SWING` | the same flip, but the local error tilts the bone axis ≥ 30° relative to the GT-posed parent frame. The child joint moves only when the parent chain is right: a bone that compensates an upstream flip (the head under H1) reads `FLIP_SWING` with a small global error |
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
| exactly one of neck01/02/03 `FLIP_TWIST` with `child` ≈ `joint`; head `FLIP_*` (`FLIP_SWING` expected) with one axis, small `glob` and `child` ≈ `joint`; jaw `OK`; v9 identical to v5 | a compensated flip (H1) | zero-training decpose surgery on that bone; exp5 vs control tests the L1 trap; lineage traces test the origin |
| where the flip sits, from the `joint` column (pelvis-relative mean error; compare each bone with the bone above, since errors add as vectors) | neck03 flipped: neck03 and head `joint` no higher than neck02's. neck02 flipped: neck03 `joint` ≈ neck02's (neck03's head is neck02's tail), head `joint` higher by ~9-21 mm at rest, ~12-30 mm with 10-20° GT bends. neck01 flipped: neck02 `joint` ≈ neck01's, neck03 higher by 0-11 mm at rest, ~14-31 mm bent, head higher by ~3-10 mm at rest, ~19-59 mm bent (table above) | a 10-30 mm rise at neck03/head is what a neck01/02 flip predicts, not evidence against H1. Read it with `|gt|`: a straight GT neck with a large rise, or a bent one with none, does not fit the flip location the verdicts name |
| v10's neck `OK` with its flips on arm/metacarpal bones | a per-init random draw, **or** a recipe-dependent cause (v10 differs in helper-mask timing, boost, 3DPW from step 0, unfrozen blocks, LR) | does not decide between them; the lineage traces and the other fresh-head runs (anny_partial_freeze = v8, anny_v7_scratch) do |
| the flipped bone is neck01 or neck02 and girth pred/GT < ~0.8 | a flip that thins the neck skin (the 0.8 comes from an LBS simulation; PVE_hn is equal for v5 and v10). These flips also move the neck03/head joints (row above), so they are not invisible to the position losses | render the surgery checkpoint before calling it the long-neck look |
| the flipped bone is neck03 and girth ≥ ~0.9 | a flip that is nearly invisible in the mesh; the long-neck look has another cause (skeleton length is 99.3% of GT) | look at shape / rendering |
| `FLIP_SWING` | the local error tilts the bone axis relative to the GT-posed parent frame; it moves the child joint only when the parent chain is right. Under H1 the compensating head is expected here: its error is a half-turn about the flipped neck bone's axis, which lies 19-30° off the head axis on the default rig at rest, so its swing is ~2× that (~38-61° at rest) | tell the cases apart by `glob` and `child`: a compensating bone has small `glob` and `child` ≈ `joint`; a tilt flip that moves its child has large `glob` and `child` well above `joint`. Never read the bone's own `joint` column for this: its own rotation cannot move it |
| `PARTIAL_FLIP` | per-image flips in a minority of people | read mean vs median; per-person rows in bones.npz |
| neck01-03 all `CONST_OFFSET` with one shared axis (relAng ~62.7°) | a convention or frame offset (bookkeeping) | check H3 / H4 lines, then the mapping |
| `FIXED_OUTPUT` (pred spread < ¼ of GT spread) | the head emits one rotation whatever the image | inspect the decpose rows of those bones |
| `SPREAD` with large medians | ordinary input-dependent error, no constant offset | supervision (exp3) |

**Candidate flag: `--rotmat_loss {l1,geodesic}`** (loss.py, default `l1`). It is a candidate, not an established fix
(the name exp5_neck_fix is historical): neither the flip nor the L1 trap has been measured on a real checkpoint. With
`l1`, the original code path runs unchanged; the default loss equals `ac17023` (the last commit before any neck work)
bit for bit on random inputs in four configurations (plain, boost 4/4, region norm, hand mask + global_rot). With
`geodesic`, the per-bone angle replaces the 9-entry L1 in the rotmat term, in global_rot and in train.py's
output-level distillation rotation term (`--lambda_kd_out`). The angle is computed as atan2(|vee(M−Mᵀ)|, tr M − 1),
which keeps the gradient finite at 0° and 180°. It runs in float32 with autocast switched off. In fp16 the masked
helper bones (pred = GT = identity, vee = 0) got NaN gradients (0.5/eps = 5e5 overflows fp16), which made train.py
skip every optimizer step. The fix is tested under CPU fp16 autocast only; GPU `--amp 1` (torch.cuda.amp with
GradScaler) is not tested yet. exp5 passes `--amp 0`, so nothing planned depends on it; run a short `--amp 1` geodesic
job and grep its log for skipped steps before using it in production. The hand mask, joint/region weights, sum over
bones and mean over people are the same. Use `--alpha_rotmat 0.33`: at a 2° error the 9-entry L1 is 3.27× (GT near
identity) to 3.50× (arbitrary GT) the angle in radians. `--alpha_global_rot` is **not** rescaled, so pass ~3.5× the
L1-era value under geodesic (17.5 for exp3's 5). If v9 carries the flip, distilling from it pulls toward that flip,
under either distance. Synthetic checks, both on single bones with only the rotation term (no j3d/v3d/j2d position
terms), so they say nothing about whether a coupled neck-bone + head flip escapes in the real model:
- 16 bones started 175° off by twist, 600 Adam steps on the 6D vector, two seeded GT draws (test_neck_dump.log): L1
  ends at a median of 166.6° in both draws (range 123.8-180°; 16/16 still above 90° in both); geodesic at 0.1° (max
  0.6°; 0/16). Earlier unseeded runs of the same test gave L1 medians of 162.4-169.0°.
- 8 bones started 160-180° off by twist (trap_sim.log): L1 median 168.0° (8/8 above 90°); geodesic median 0.1°, but
  1/8 still above 90° (114.9°); squared Frobenius median 0.1° (0/8).
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

### exp6_bone_fix: a constant output-level correction of the flipped bones (uncommitted)

**Why (measured, `diag_neck_bones` jobs 3573518/3573519, `exp5_neck_fix` jobs 3573520/3573521; V1, 100-image holdout).**
- v5 lineage (frozen_bb_v2 00299, partialft_v3 00054, boosted_v4 00299, v5 00099, v9 00149, long_pc_dist5 00009, both
  exp5 pilots): neck02 `FLIP_TWIST` (median 175.6°, |twist| 175.7, swing 3.3, relSpr 5.3) and head `FLIP_TWIST` /
  `FLIP_SWING` (177.9°, twist 178, swing ~29, relSpr ~15), a compensated pair: neck03 local OK (~6°) but global ~173°,
  head global ~12° (right). neck01 ~5°, jaw ~2°. Neck-skin girth pred/GT 0.46 (v10 1.00), PVE_neck (neck01-centred)
  27-28 mm (v10 10.3). metacarpal1.L `FLIP_SWING`, a constant 180° (pred spread 0, GT identity), in all 6 checkpoints
  traced by diag_neck_bones (the exp5 pilots dumped only the 12 default bones, no metacarpals): child error 185-200 mm vs
  joint 90-106 mm. Already in frozen_bb_v2 00299, before the helper mask, the boost and alpha_shape 20. H3 and H4
  refuted for every checkpoint.
- v10 00290: neck and head fine. Constant half-turns on metacarpal2.L (`FLIP_SWING` 180°), metacarpal2.R
  (`FLIP_TWIST` 180°) and metacarpal3.R (`FLIP_SWING` 180°): GT identity, relSpr 0.0-0.3, global 167-170°, i.e.
  uncompensated. Child error 182-185 mm for metacarpal2.L/3.R (`FLIP_SWING`) and 136 mm for metacarpal2.R
  (`FLIP_TWIST`: a half-turn about the bone barely moves its child), vs joint 95-120 mm. In the left arm, a compensated
  chain like v5's neck: shoulder01.L `FLIP_TWIST` 166° (relSpr 21.3: prdSpr 10.7 vs gtSpr 24.6, so not a constant) with
  global 166.7°; upperarm01.L `SPREAD` 58° (swing 55.7, about twice its GT angle of 28°: its GT rotation conjugated by
  the half-turn above) with global 169.0°; upperarm02.L `FLIP_TWIST` 180° (GT identity, relSpr 2.4) with global 18.0°,
  i.e. the chain is right again from upperarm02.L down (lowerarm01.L global 19.7°).
- exp5 (2 epochs, lr 1e-5): geodesic neck02 175.6 → 175.0, head 177.9 → 177.1; the l1 control did not move. Read
  [Likely]: un-flipping needs neck02 and head to rotate back together; a partial move of one breaks the compensation
  and raises the loss, so gradient steps from inside the flip do not leave it.

A flip with a small relSpr is a near-constant offset of the head's output, so one constant rotation per bone should
undo it with no training. That is cheaper and more direct than the decpose surgery the exp5 section proposed.

**What.** `model.py` applies `R_b ← R_b · C_b` to the head's final local rotmats of the listed bones, right after
`rot6d_to_rotmat` and before the helper mask, `relax_hands`, `rest_pose`, FK and the losses. Training, `evaluate()`,
the diagnostics and the demo therefore all see the corrected rotations, and gradients flow through the constant. The
head is a single pass whose query token is built from the constant `init_body_pose` buffer, so nothing feeds the
uncorrected rotation back. `C` is a non-persistent buffer `[163,3,3]` (identity elsewhere) plus a bool mask. With no
fix, `forward()` skips the step (the same tensor is returned, bit-identical to HEAD). `bone_fix.py` holds the JSON
format `{"bones": {"neck02": 3×3, ...}, "source": ..., "estimated_on": ..., "stats": ...}`. Names are resolved against
the rig (an unknown name is a hard error), and every matrix must be a proper rotation. Every loader prints one
`[bone_fix]` line with the bones and angles.
- `train.py --bone_fix <json>`: the parsed file is saved in the checkpoint args (`bone_fix_spec`). A model fine-tuned
  with a fix re-applies it whenever it is rebuilt from its args (`diag_common.load_model`, `demo.py`,
  `distill.build_teacher`), even after the JSON file is gone. With `--bone_fix ''` it inherits the `--pretrained`
  checkpoint's fix when that checkpoint's heads are loaded (`load_only_backbone 0`), and `none` drops it.
- `diag_holdout_trace.py` / `demo.py --bone_fix`: `''` = the checkpoint's own fix (none for every existing
  checkpoint), `none` = off, a path = that fix instead.
- `scripts/diagnostics/compute_bone_fix.py`: `C_b` = SO(3) projection (SVD, det +1) of `mean_i R_pred_iᵀ R_gt_i`, using
  `R_pred` from the dump's new `R_pred_local_raw` (the prediction without any fix). `--bones auto` takes every dumped,
  active bone whose verdict is `FLIP_TWIST/FLIP_SWING/CONST_OFFSET/FIXED_OUTPUT` with a median above 90°. It prints, per
  bone, the angle of `C_b`, the median/p90 local error before and after, the same with a 2-fold split by image
  (out-of-sample inside the estimation set), and the singular values of the mean. It then prints the predicted
  **global** error of every dumped bone at or below a corrected one, `G_b'' = Π_a (G_a C_a G_aᵀ) G_b`, which is exact
  for anny's FK. Exit status 3 means no bone qualifies.
- **Global guard** (`--max_global_rise 5`): when the fix raises the predicted global median of any dumped bone by more
  than 5°, `compute_bone_fix.py` prints a WARNING naming those bones and the corrected bones on their chains, plus the
  **safe subset** (the selected bones minus every corrected bone on a risen bone's chain, repeated until nothing
  rises), stores both in the JSON (`global_guard`), and exits 4 with the JSON written. This is the compensated-chain
  case: `--bones auto` corrects the flipped ends of a chain but not the link in between (v10 upperarm01.L is `SPREAD`,
  median 58° < 90°), and one right-constant per flipped bone leaves that link's conjugated rotation uncompensated. In a
  synthetic v10 arm (local 180/56/180°, global upperarm01.L 180° / upperarm02.L 0°), the auto fix moves upperarm02.L,
  and with it the forearm, hand and fingers, from 0° to 56° global, and the elbow by ~264 mm (review script, no rest
  offsets; with the real rig's rest offsets in `test_bone_fix.py`: 2° → 54°). `exp6_bone_fix.sh` acts
  on exit 4 by `ON_GLOBAL_RISE`: `continue` (default with TRAIN=1: the fine-tune can still act on the residual), `stop`
  (default with TRAIN=0: the zero-training test would only confirm the predicted failure) or `safe` (re-estimate with
  the safe subset; the full fix is kept as `bone_fix_guarded.json`, which every new estimation deletes first, and the
  summary reports a safe subset only when `bone_fix.json` holds exactly that file's safe subset). Keeping such a chain
  intact needs a two-sided constant on the link, `R_l ← K_lᵀ R_l K_l` with `K_l` the parent's correction carried
  through the rest offset (not implemented).
- Estimation never touches the holdout. `diag_holdout_trace.py --select train_sample --n 200 --seed 0` draws seeded
  random training indices (all but the last 100), minus the diag manifest images and `--exclude_indices_file`, and
  writes `selected_indices.txt`. `--dump_bone_names all` dumps every active bone (75), so `auto` can find a flip
  anywhere. Under `--dump_bones 1`, `bones.npz` now also holds `R_pred_local_raw`, `bone_fix_C`, `bone_fix_on`,
  `bone_hand`, `bone_parent_index`, `rig_parents`, `rig_names` and `bone_active`; `--dump_bones 0` output is unchanged.
  `R_pred_local_raw` is the prediction the same variant would give without the fix: the fix is not undone where the
  variant forces identity anyway (helper bones under the helper mask; finger/thumb bones, `bone_hand`, under
  `relax_hands` in V3). `compute_bone_fix.py --variant V3_demo_relaxhands` never selects a finger/thumb bone.

**Commands** (repo root on the cluster; ~30 min with TRAIN=1, ~10 min with TRAIN=0):
```
sbatch slurm/experiments/exp6_bone_fix.sh                                   # v5 00099: estimate, zero-training test, 2-epoch fine-tune
CKPT=logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt TRAIN=0 BONES=metacarpal2.L,metacarpal2.R,metacarpal3.R \
    sbatch --time=00:30:00 slurm/experiments/exp6_bone_fix.sh               # v10's uncompensated flips, no training
SUMMARY_ONLY=1 RUN=anny_s2_shape_v5_00099_n200_s0 bash slurm/experiments/exp6_bone_fix.sh   # the table alone, login node
SUMMARY_ONLY=1 RUN=anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R \
    bash slurm/experiments/exp6_bone_fix.sh                                 # the v10 table
```
In SUMMARY_ONLY mode the TAG (and so the baseline and control) comes from the run's `bone_fix.json` (`source.ckpt`),
not from the default CKPT (v5); without it, a RUN that does not start with `<TAG>_` is an error. v10 runs with the
three metacarpals because `--bones auto` would also correct shoulder01.L and upperarm02.L, the two ends of the
compensated chain above (the global guard would stop that run). Outputs go to
`results/experiments/exp6_bone_fix/<TAG>_n200_s0[_<bones>]/`:
- `est_trace/` (training sample, no fix) and `bone_fix.json`;
- `holdout_fix/` (zero-training) and `holdout_nofix_vis/` (the same 12 images without the fix);
- `compare/<idx>.jpg`: the two panel rows stacked, top without the fix, bottom with it;
- `train/` and `holdout_ft/` (TRAIN=1 only);
- `summary.json`.

The baseline is `results/diagnostics/neck_bones/<TAG>/` (`diag_neck_bones.sh`). Without it, the job traces CKPT itself
into `holdout_baseline/`. The control is `results/experiments/exp5_neck_fix/pilot_l1_a0.1_<TAG>`; it only exists for
v5, and v10 runs without one.

**Pass criteria** (V1, holdout, matched people):
- Zero-training, against the baseline: every corrected bone has a median below 15° (v5: neck02, head, metacarpal1.L),
  girth ≥ 0.9, global head ≤ baseline + 3°, and PVE ≤ baseline + 1 mm.
- Fine-tuned, against the exp5 l1 control (or the baseline when there is none): the same bone and girth conditions,
  plus the exp5 guards PVE ≤ ref + 8 mm, recall ≥ ref − 2 and global head ≤ ref + 3°.

The summary table has one column each for baseline, zero-training, fine-tuned and control. Its rows: verdict and median
for the neck chain, the corrected bones and every baseline-flagged bone; girth, PVE_neck, PVE, PA-PVE, PVE_hn, recall,
global head and finger rotation.

**What to expect, from the measured spreads [Likely].** After the fix, a bone's local error is about its relSpr (the
mean distance of its per-person error rotations from their mean):
- neck02 → ~5°, metacarpal1.L (v5) and the three v10 metacarpals → ~0-1° (relSpr 0-0.3), with v10 PVE and global head
  about unchanged (nothing above the metacarpals changes);
- head → ~15°, borderline against the 15° criterion. Its half-turn compensates the neck02 flip person by person (the
  error is a 180° turn about an axis that moves with the GT neck03/head pose), which no constant can undo.
- v10 with `--bones auto` (not the planned run): upperarm02.L and the metacarpals → ~0-2° (relSpr 0-2.4), shoulder01.L
  → ~20° (relSpr 21.3; shoulder01 sits at 12-15° even unflipped: v5 14.3 L / 12.0 R, v10 14.7 R), and upperarm01.L,
  the link of the compensated chain, is no longer compensated: the left-arm global error and PVE rise. The global
  guard fires in step a (predicted global rise on upperarm02.L, lowerarm01/02.L, wrist.L and the left hand below it;
  upperarm01.L itself drops from 169° to about its local error), and with TRAIN=0 the job stops there.

The head's remaining ~15° is no longer compensated, so the **global** head error is likely to rise. In a synthetic
compensated pair (relSpr 23°), the predicted global head error went from 2° to 22°. `compute_bone_fix.py` prints this
prediction on the training sample in step a, before the holdout runs. If it rises by more than 3°, the zero-training
column fails on global head; above 5° the global guard fires (exit 4) for the v5 run too, and with TRAIN=1 the job
continues by default (`ON_GLOBAL_RISE=continue`), so the WARNING in the v5 log is this expected rise. The fine-tuned column then shows whether training starting from the un-flipped state
removes the residual. The neck skin should recover either way, because neck02 is the bone that twists it.

**How to read.**

| outcome | meaning | next |
|---|---|---|
| zero-training PASS | the flips are fixed output offsets of the head; one constant per bone removes them on unseen images | keep the fix in the checkpoint args (fine-tuned column), render `compare/`, re-run the exp1-exp4 reads on the fixed model |
| neck02 / metacarpals < 15°, girth ≥ 0.9, but head ≈ 15° and/or global head > +3° | consistent with the compensation residual above | the fine-tuned column decides; otherwise a chain-preserving head correction `d_h ← X⁻¹ C_n2⁻¹ X d_h`, with X = rest(neck02→neck03) · predicted neck03 · rest(neck03→head), which keeps the head's global rotation exactly (not implemented) |
| a corrected bone does not drop on the holdout although `after` / `2-fold` dropped on the training sample | the offset is not constant across images (train ≠ holdout) | read the per-person rows in `holdout_fix/bones.npz` |
| girth stays < 0.9 with neck02 corrected | the thinning has another cause | render; look at shape |
| v10 (three metacarpals): → ~0°, PVE / global head unchanged | the same constant-offset mechanism in a fresh-head run | apply to v10 itself; the left-arm chain needs the two-sided link correction |
| GLOBAL GUARD warning in step a, then PVE / global head up in the zero-training column | the compensated chain the guard predicted, not a new effect | run the safe subset (`ON_GLOBAL_RISE=safe` or `BONES=`) for the constant flips; the chain needs the two-sided correction |

**Caveat.** The correction is one constant per bone, estimated on 200 training images. It removes a symptom and
says nothing about why the heads converged to the flip (origin: present in frozen_bb_v2 00299). The holdout is never
used for estimation. A model fine-tuned with `--bone_fix` keeps the fix in its args and needs it at inference;
`--bone_fix none` turns it off for any diagnostic.

**Verified offline** (`scratchpad/bonefix/test_bone_fix.py`, 94 checks, CPU, stub anny on the real 163-bone default rig):
- With no fix, `Model.forward` is bit-identical to HEAD `model.py` in eval and training, with and without the helper mask.
- On random 6D inputs through `rot6d_to_rotmat`, the fixed bones equal `R @ C` exactly (`torch.equal`) and every other
  bone is untouched. The same holds under autocast. The gradient is `dL/dR' Cᵀ`.
- In the full forward, the fixed bones are HEAD's rotmat times C, the other bones, shape, loc and scores are unchanged,
  and the mesh moves.
- The fix survives `train.py --bone_fix` → checkpoint → `load_model`, even with the JSON file deleted. The `''`,
  `none`, `<path>` and `--pretrained` inheritance cases all behave as described above.
- For `R_pred = R_gt F_b` with 2° noise and 300 people, `compute_bone_fix.py` recovers `C_b = F_bᵀ` within 0.11°, the
  corrected median is ~1°, and the predicted global errors equal FK of the corrected locals (1.6e-11).
- Global guard, synthetic v10 arm with the real rig's rest offsets (local shoulder01.L / upperarm01.L / upperarm02.L
  179.6 / 54.3 / 179.6°, global upperarm01.L 179° and upperarm02.L 2°): `auto` picks shoulder01.L, upperarm02.L and
  two metacarpals, the predicted global upperarm02.L, forearm, wrist and fingers rise from 2-3° to ~54° (equal to FK
  of the corrected locals), exit 4, implicated = shoulder01.L + upperarm02.L, safe subset = the metacarpals; re-run
  with the safe subset: exit 0, metacarpals ~1°, finger3-1.L global 179° → 3°. Uncompensated flips do not fire it. The
  synthetic v5 pair fires it too (global head 2° → 22°, safe subset empty), and the NOTE line names the head (22°).
- `train_sample` never draws holdout, manifest or excluded indices.
- The summary verdicts were checked on synthetic JSONs, as were the step-a guard lines and the warning when the
  baseline trace belongs to another checkpoint than the fix.
- The generated slurm command passes `bash -n` and runs end to end with fake tools, for v5 with TRAIN=1, v10 with
  TRAIN=0 (auto and the three metacarpals), and exit 4 under `ON_GLOBAL_RISE` stop (default with TRAIN=0), safe and
  continue (default with TRAIN=1). `SUMMARY_ONLY=1 RUN=<v10 run>` without CKPT takes the v10 baseline from
  `bone_fix.json`; without `source.ckpt` it stops unless TAG or CKPT is given.

### 3DPW converter fix (v2.3 flags)

A label fix, not a training experiment. The converter work (from commit `40fd436` on): `scripts/data/smpl_to_anny.py`
v2.3, `slurm/data/run_smpl_to_anny_task.sh`, the new `scripts/data/check_refit_task.py` and
`slurm/data/run_check_refit_task.sh`, the `.gitignore` exceptions and this section. Use it from `main` on GitHub;
nothing has to be merged, committed or force-added by hand.
(`.gitignore`'s `data/` line also matched `scripts/data/` and `slurm/data/`;
`!scripts/data/` and `!slurm/data/` now follow it, and the `*.npz`, `*.pkl` and `__pycache__/` rules still apply inside
both folders.) Step 0 below checks that the cluster copy has it.

**What was wrong (measured by `check_3dpw_conversion` job 3575369 on `threedpw_anny_full/labels`, converter md5
`f502c8b2`).**
1. *The gender was reversed.* `smpl_to_anny.py` v2.2 wrote 1.0 for a 3DPW 'm' and 0.0 for an 'f' (MakeHuman
   convention). Anny uses 0 = male, 1 = female: `model_data.py` has `gender=['male','female']`, and
   `shape_distribution.py` has `gender <= 0.5 -> boys`. The I1 probe agrees: rest stature is 1.753 m at 0 vs
   1.632 m at 1, shoulders 0.369 vs 0.308 m. Label values were 'm' 1.000 (1,801 sampled people) and 'f' 0.000 (734).
   I1 refit test (104 people, label refitted with g and with 1-g):
   - Men (73): joint error label/same/flip 19.0/19.2/15.3 mm, so the flip was lower for 72 of 73 (median gain
     3.4 mm). The surface was also better with the flip (18.6 → 17.4 mm, 93% of people).
   - Women (31): joint error 13.7/13.7/16.3 mm, so the flip was lower for only 26%. The surface was better with the
     flip (17.0 → 16.2 mm, 87%).
   So the joints confirm the reversal only for men. For women only the surface and Anny's code support it.
2. *Two actors were fitted on different frames.* `find_valid_frames` applied the stride (3) inside each person's own
   valid-frame list. After an invalid gap, the two actors of a sequence therefore sat on different phases of the
   stride. I3 found 407 of 960 sampled two-person images with a valid actor and no label (train 115/280,
   validation 42/160, test 250/520). All of them were 'not fitted'; 0 were dropped by a builder filter. I4 shows the
   same pattern per sequence: `courtyard_captureSelfies_00` has 457 person-frames on 401 images (only 56 images with
   both actors), and `courtyard_rangeOfMotions_00` has 399 on 316. `courtyard_giveDirections_00` has no gaps and shows
   566 on 283, i.e. both actors on every image.

**The flags.** The defaults reproduce v2.2 bit for bit (tested, see below), so the labels v7-v10 were trained on can
still be rebuilt.

| flag (`smpl_to_anny.py`) | default = v2.2 | fix |
|---|---|---|
| `--gender_convention` | `makehuman`: m → 1.0, f → 0.0 | `anny`: m → 0.0, f → 1.0 (`gender_phenotype_value`, `:286`) |
| `--frame_grid` | `per_person`: `valid_idx[::stride]` | `shared`: the valid frames with `frame % stride == 0`, one absolute grid for every person (`select_frames`, `:801`) |

- The log line now names the convention and the value, e.g. `gender phenotype fixed at 0.0 from 3DPW gender 'm'
  (--gender_convention anny: m -> 0.0, f -> 1.0: Anny's own convention ...)`. A second line prints the frame grid.
- Each `.npz` also stores `gender_convention`, `frame_grid`, `frame_stride`, `max_frames`, `gender_value` and
  `valid_mask` (the per-frame validity used), and says `version='2.3'`. The label builder (`load_fit`) and
  `check_3dpw_conversion.py` look keys up by name and only print the version, so they accept 2.3. Both were tested on
  v2.3 files.
- A person valid from frame 0 with no gap gets the same frames under both grids.
- `run_smpl_to_anny_task.sh` reads two environment variables:
  - `OUT_ROOT` (default `threedpw_anny_full`, unchanged) sets the fits directory.
  - `REFIT_FLAGS` (default empty) is appended to the `smpl_to_anny.py` call. It should contain only the two new
    flags: it comes last, so it would also override the script's `--frame_stride`, `--max_frames` or `--fix_gender`,
    and the skip check below compares only `gender_convention` and `frame_grid`.

  An existing `.npz` is never refitted, so three guards keep fits made with different flags out of one directory:
  - A non-empty `REFIT_FLAGS` with the default `OUT_ROOT` is refused.
  - An empty `REFIT_FLAGS` with any other `OUT_ROOT` is refused unless `ALLOW_DEFAULT_FLAGS=1`. Without this guard,
    dropping `REFIT_FLAGS` from the v23 command would fill `threedpw_anny_v23` with v2.2-behaviour fits, and every
    later correct submission would skip them. `ALLOW_DEFAULT_FLAGS=1` is for reproducing v2.2 into a new folder.
  - Outside the default `OUT_ROOT`, an existing `.npz` is skipped only if the `gender_convention` and `frame_grid`
    it records match what `REFIT_FLAGS` asks for (`check_refit_task.py --match_flags`, run in the container, about a
    minute per skipped task). A file without these keys counts as v2.2 (`makehuman`, `per_person`). On a mismatch,
    or an unreadable file, the task fails with `ERROR: ... exists but was not made with REFIT_FLAGS=...`. Delete
    that `.npz` or choose another `OUT_ROOT`.

  In the default `OUT_ROOT` an existing `.npz` is skipped without a check, as before. Array index = line − 1 of
  `configs/threedpw_tasks.txt`, so the 87 tasks are 0-86.
- `run_build_labels.sh` (`FITS`/`IMAGES`/`OUT`) and `check_3dpw_conversion.sh` (`LABELS`/`FITS`/`OUT`) could already
  point at another root, so they are unchanged.
- `check_refit_task.py` checks one sequence or all of them. It compares the gender value with the 3DPW gender and the
  convention, checks the frame grid, and counts frames where both actors are valid in 3DPW but only one is fitted. It
  also shows the joint error mean/median of the old vs new fits. It exits 1 on a convention violation; error changes
  are reported, not graded.

**Commands, in order** (on the cluster). Run them one at a time: each step starts only after the previous one has
finished (and, for steps 2 and 3, passed). `sbatch` prints the job id; `squeue -u najib` lists only queued and running
jobs, so a job it no longer shows has ended.

0. After that merge, update the cluster copy and check it (no branch to merge, nothing to add):
   ```
   cd /netscratch/najib/multi-hmr && git pull && grep -c REFIT_FLAGS slurm/data/run_smpl_to_anny_task.sh; grep -c gender_convention scripts/data/smpl_to_anny.py; grep -c ALLOW_DEFAULT_FLAGS slurm/data/run_smpl_to_anny_task.sh; ls scripts/data/check_refit_task.py slurm/data/run_check_refit_task.sh
   ```
   Go on only if all three numbers are above `0` and both files are listed; otherwise the merge has not reached
   `main` yet, so wait and run the line again. The old task script ignores `OUT_ROOT` and `REFIT_FLAGS`, and all 87
   `.npz` files exist in `threedpw_anny_full`. So with the old script, commands 1 and 3 print
   `already converted - skipping` and exit 0 for every task without fitting anything, and command 4 then builds from
   an empty `threedpw_anny_v23`.
1. One check task: `courtyard_captureSelfies_00`, train, array indices 9 (p0) and 10 (p1), i.e. lines 10-11.
   - Two actors of mixed gender: its 40 sampled images in 3575369 had one labelled person each with label gender
     mean 0.60, so 24 images of the man and 16 of the woman.
   - It is the strongest stride-phase case (56 of 401 images with both actors).
   ```
   cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=9-10 slurm/data/run_smpl_to_anny_task.sh
   ```
   Wait until `squeue -u najib` no longer lists this job and both logs, `logs/smpl2anny-all-<jobid>_9.out` and
   `_10.out`, contain a `done: ...` line (a `FAILED` or `ERROR` line instead = stop here). If the job is gone from `squeue` but a
   log has no `done:` line, Slurm killed it (time limit, memory, node): read `logs/smpl2anny-all-<jobid>_9.err` /
   `_10.err` or run `sacct -j <jobid>`, and stop. Then check in each log:
   - `extra flags (REFIT_FLAGS): '--gender_convention anny --frame_grid shared'` and `frame grid: shared`.
   - The gender line shows 0.0 for the man and 1.0 for the woman.
   - The converter's summary line `fitted N frames; joint error mean X mm, max Y mm` gives X at about 16-20 mm. In
     3575369's I4 the v2.2 train sequences had per-sequence medians of 13.7-20.3 mm (`courtyard_captureSelfies_00`
     18.8 mm). The man may come out lower, the woman ~2 mm higher (see I1).
   - Do not grade the PER-JOINT table against 16-20 mm. It is never flat: pelvis, hips, neck and shoulders sit at
     ~27-38 mm by rig definition (I4 pooled table: pelvis 28.4, hips 33.9/37.9, neck 27.6, shoulders 35.5/28.7 mm
     mean). Only a row far above its I4 value is a warning.
   - `done: ... (N s)` gives the measured s/frame: N divided by the fitted frames.
2. Post-check of that sequence (CPU, seconds). It compares against the old fits in `threedpw_anny_full` by default.
   Submit it only after both step-1 tasks have finished (see above). Earlier, it stops with `ERROR: no new .npz ...`
   (exit 2) or checks only one actor.
   ```
   cd /netscratch/najib/multi-hmr && SPLIT=train SEQ=courtyard_captureSelfies_00 sbatch slurm/data/run_check_refit_task.sh
   ```
   It passes when (`logs/check-refit-<jobid>.out`):
   - the log ends with `RESULT: PASS` and the job exits 0;
   - `[p0+p1] NEW: ... both valid in 3DPW but only one fitted: 0/N`, while the `OLD` line shows a large share;
   - the man's label value is ~0.0001 and the woman's ~0.9999 (`[pK] gender: 3DPW 'm', label value median ...`);
   - also expected, NOT a gate (the script does not grade it): the man's joint error is lower than before (the
     woman's may rise ~2 mm, see I1). If it is not lower, note it and continue; report it. Read the paired line first,
     `[pK] joint error mean/median, M frames fitted by both: old ... -> new ...`. The old per-person grid put the two
     actors on different stride phases, so an actor may share only a few frames with the new `frame % 3 == 0` grid,
     or none (`[pK] no frame fitted by both runs`). If M is small (under ~30) or the line is missing, use the
     unpaired per-person means instead: `[pK] joint error mean/median, all fitted frames: old ... -> new ...`,
     which is always printed.
3. All 87 tasks. Submit only after step 2 passed. Never submit it while the step-1 job is still queued or running:
   tasks 9 and 10 would find no `.npz` yet and refit the same people again, both jobs writing the same
   `<seq>_p<k>.npz.tmp.npz`. Tasks 9 and 10 are skipped because their `.npz` already exist and record the same flags
   (the log says `already converted with the requested flags - skipping`). A task to redo must have its `.npz`
   deleted first. A task whose existing `.npz` records other flags fails instead of skipping (see the guards above).
   ```
   cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
   ```
   When the array is finished:
   - `ls /netscratch/najib/multi-hmr/threedpw_anny_v23/*/*_p*.npz | wc -l` should give 87.
   - Optionally, `cd /netscratch/najib/multi-hmr && SEQ=all sbatch slurm/data/run_check_refit_task.sh` should print
     PASS for every sequence.
4. Labels (CPU, ~1 min), submitted in one block with step 5, only after the step-3 array has finished and the count
   above is 87.
5. The conversion check on v23, ~19-20 min. 3575369 took 997 s on the 52 sequences of the old index; the v23 labels
   cover 59 (the six train sequences and `courtyard_basketball_01` added), i.e. ~2,360 sampled images and ~118
   people in the I1 refits, still well inside the 1 h 30 min limit. It waits for the label job
   (`--dependency=afterok`).
   It must not start earlier: it only tests that `index_*.txt` exists, and the builder writes `index_train.txt`
   before the other splits are done. Keep `OUT=` on the check, otherwise it overwrites the 3575369 results.
   ```
   cd /netscratch/najib/multi-hmr && L=$(FITS=/netscratch/najib/multi-hmr/threedpw_anny_v23 OUT=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --parsable slurm/data/run_build_labels.sh) && echo "labels job ${L%%;*}" && LABELS=threedpw_anny_v23/labels FITS=threedpw_anny_v23 OUT=results/diagnostics/check_3dpw_conversion_v23 sbatch --dependency=afterok:${L%%;*} slurm/diagnostics/check_3dpw_conversion.sh
   ```
   If the label job fails, the check stays pending with reason `DependencyNeverSatisfied`; cancel it with
   `scancel <jobid>`.

**Runtime (estimated, not measured).** Each frame costs about 260 Anny forward+backward passes: pass 1 warm-started
20+80+60, pass 2 100 refine, plus a few forwards and the bounding box.
- 3575369's I1 refit ran 104×2 refits of 160 iterations in 678 s, i.e. ≤ 20 ms per iteration including its surface
  metrics.
- v2.1 measured 11 s for 480 iterations, i.e. 23 ms per iteration. The task script estimates ~6 s/frame.
- So 5.3-6 s/frame. The v2.2 run fitted 22,915 frames (I4). The shared grid keeps about n_valid/3 frames per person,
  as before, so the count stays about the same.
- Total: ~34-38 GPU-h, plus 87 start-ups of ~1-2 min, i.e. ~35-41 GPU-h. With `%8` that is ~5-6 h of wall time if 8
  GPUs are free.
- The longest one-person tasks are `downtown_windowShopping_00` (608 frames, ~55-60 min), then
  `outdoors_parcours_00` (544, ~50 min). I4 gives only the sum for two-person sequences (`downtown_sitOnStairs_00`
  877, `downtown_bus_00` 850, ~425-440 per person if split evenly). A person cannot get more than a third of the
  sequence's frames: at most 726 for `downtown_bus_00` (2,178 frames), ~65-75 min. All are well under the 4 h limit.
- The check run (tasks 9-10, ~230 frames each) takes ~25 min.

**Pass criteria (v23 check vs 3575369).**
- OVERALL: expect `OVERALL (V0-V5): FAIL` again, as in 3575369 (train FAIL from V4e, test FAIL from V4c). That is
  not a failed refit.
  - Expected to change: I1 (label gender 'm' ~0, 'f' ~1, men's flip gain gone) and I3 (unlabelled valid actors ~0
    apart from builder filters), plus the I4 versions and the label counts (below).
  - Not expected to change: V4c limb ratios (forearm 0.879, thigh 1.119 overall; FAIL in test, WARN in validation)
    and V4e people with a joint > 100 mm (train 0.052 FAIL, overall 0.032 WARN). The flags do not target them, so
    they should stay about the same. Test V4c may improve if it is linked to the gender bug (test has only men, all
    labelled with the female body; possible, not shown), and train V4e sits right on the 0.05 line. A move in either
    direction is not a refit failure. OVERALL FAIL is the likely result; WARN would be good news.
  - Judge the refit by I1, I3, I4, the label counts and V1-V3 as listed below.
- I1:
  - Label gender 'm' 0.000 and 'f' 1.000; the probe still says male end = 0.
  - Men's refit flip gain is gone: median gain ≤ 0 and a flip-better share well below 0.5 (was +3.4 mm and 0.99).
  - Women: expect the mirror of 3575369 (on joints, the flip about 2-3 mm better; on the surface, worse). The joints
    never confirmed the female end for women, so this is reported, not graded.
- I3: 'valid actor unlabelled' ~0 of the two-person images (was 407/960), and 'not fitted' ~0. The remainder should
  be builder drops (error > 80 mm, visible < 0.3, box < 40 px) or frames the converter's NaN filter removes even
  though `campose_valid` is true.
- V1/V2/V3 not worse than 3575369, allowing ~1 mm because the seeded sample is drawn from a different index:
  - V1 body median 18.7 / p95 24.5 mm;
  - V2 medians 18.9 / 18.3 mm;
  - V3 Anny − raw 1.8 mm2d.

  Men should improve. Women may lose ~2-3 mm in V1 (refit test 13.7 → 16.3 mm), and train holds the all-female
  climbing/slalom/freestyle sequences, so read V1 train (20.2, WARN) per gender in `per_person.csv`.
- I4: 87 npz, `versions` {'2.3': 87}, and label images expected = indexed for every sequence (the images were
  re-extracted).
- Label images: compare with the 18,936 that the v2.2 fits imply (I4 `expected_label_images`), not with the old
  index's 16,631. The old index lacked the images of 8 sequences (6 train sequences, part of
  `courtyard_captureSelfies_00`, validation `courtyard_basketball_01`), which have now been re-extracted.
  - From I4: one-person sequences give 9,707 images; two-person sequences give 13,046 kept person-frames on 9,229
    images.
  - With both actors on one image wherever both are valid, expect ~16,200-18,900 label images: at least 9,707 +
    13,046 / 2 (every two-person image holds both actors), at most 18,936.
  - With I3's one-sided shares (train 41%, validation 26%, test 48%), about 16,900: train ~5,900 (old index 4,451),
    validation ~2,650, test ~8,300. So more than 16,631 is expected, not a failure.
  - Labelled people (person-frames) stay about v2.2's 22,753, which is ~2,700 more than the old index held.
- V4d stature ratio (0.972, PASS) is a graded V check (PASS band 0.95-1.05) and feeds OVERALL. It may move, since
  men now get the taller male body, but should stay PASS.

**Models.** v7-v10 were trained and 3DPW-evaluated on `threedpw_anny_full/labels`, and they are unaffected until a
model is retrained or re-evaluated on v23.
- `slurm/train/train_teacher_v9_mixed.sh:55`, `train_distill_vits_v8_final.sh:42`, `train_v10_scratch_mixed.sh:45`,
  `slurm/eval/run_eval_3dpw.sh:39`, `run_eval_3dpw_teacher.sh:39` and `run_eval_3dpw_test.sh:61` set `LABELS` with a
  plain assignment, `LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"`.
- So an environment override is overwritten without a warning:
  `LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch slurm/eval/run_eval_3dpw.sh` still evaluates on
  the old labels.
- To use v23, edit that line in each script you run, then check the `--mix_3dpw_labels` path in the job log. (A
  separate change could make it `LABELS="${LABELS:-/netscratch/najib/multi-hmr/threedpw_anny_full/labels}"`.)
- 3DPW numbers measured against the old labels cannot be compared with numbers against v23; re-evaluate the baseline
  on v23 first.
- The old labels stay reproducible: with the default flags and no `OUT_ROOT`, the task script writes the v2.2 fits.

**Verified offline** (`scratchpad/convfix`, 122 + 36 + 1 checks, CPU, real Anny 0.6.1, synthetic 3DPW data; the
numpy-only scripts also under Python 3.9 + numpy 1.24):
- Frame selection over 9,600 random (validity pattern, stride, max_frames) cases:
  - `per_person` equals HEAD's `find_valid_frames`.
  - `shared` frames lie on `frame % stride == 0`, are valid, number ≤ max_frames, and are identical for both actors
    wherever both are valid.
  - `check_refit_task.py`'s validity mask equals the converter's.
- Gender values for both conventions and genders. HEAD's 26 argparse defaults are unchanged, and there are exactly
  two new options.
- End to end (`main()` twice on a 2-actor sequence):
  - HEAD and the new defaults give bit-identical `.npz` files in every key but `version` (31 of 32; 2.2 → 2.3,
    plus the six new keys) and identical label pkls.
  - `anny`+`shared` gives the man 0.0001 and the woman 0.9999, and both actors share all 10 both-valid grid images.
    The old grid shared 3 of 17.
  - The builder accepts v2.3 files under Python 3.9.
- `check_refit_task.py` passes v23 fits and fails default-flag fits (13 violations), v2.2 fits, a doctored gender and
  a dropped frame.
- `check_refit_task.py --match_flags`: for 10 flag strings it reads the same `gender_convention`/`frame_grid` as
  the converter's own argparse. It returns 0/1/2 as expected on v2.3, v2.2 and default-flag fits, an unreadable file
  and a bad value, under both Pythons.
- `run_smpl_to_anny_task.sh` with fake sbatch/srun/python:
  - Defaults give the same python call as HEAD, with output in `threedpw_anny_full`. There an existing `.npz` is
    skipped without any check, as before.
  - The env overrides add the flags and write into the new root.
  - Resubmitting with the same flags (also written as `--flag=value`) skips without refitting, using the real check
    under Python 3.9. Other flags, a v2.2 fit, a default-flag fit or an unreadable file in the new root make the task
    fail, and no file changes.
  - The guards refuse `REFIT_FLAGS` with the old root, and an empty `REFIT_FLAGS` with a new root (all 87 tasks,
    no python call, no folder) unless `ALLOW_DEFAULT_FLAGS=1`.
  - Indices 9/10 map to captureSelfies p0/p1 and 86 to the last task.
- `.gitignore`: the two new files are no longer ignored; `*.npz`, `*.pkl`, `__pycache__/` and the top-level `data/`
  still are.
- `run_check_refit_task.sh` exits 0 / 1 as expected. `bash -n` passes on all touched scripts.

**Unfitted phenotypes in mixed runs (`--mask_3dpw_shape`, default 0; exp7 `MASK_3DPW_SHAPE`, default 0).** v2.3 still
fits only height and proportions and takes gender from the 3DPW metadata; age stays at 0.55 and muscle, weight, cupsize,
firmness, african, asian and caucasian at 0.5, and the shape L1 trains every 3DPW person toward these constants. With
`--mask_3dpw_shape 1` (refused without `--mix_3dpw_labels`) the shape L1 of people tagged `has_hand_labels` 0 is zero
on those 8 phenotypes, found by name; gender, height, proportions and every AnnyOne person stay supervised. Limits: it
adds no child signal (3DPW has no children); the v3d/v2d mesh terms still carry the neutral body, because the 3DPW GT
mesh is built from the constants; and the 3DPW-test PVE is computed against the same neutral-body GT meshes, so a
masked model can score worse there.
`--mask_3dpw_shape_names` (exp7/exp8 `MASK_3DPW_SHAPE_NAMES`, default '' = the 8) masks a comma-separated subset
instead, checked against `phenotype_labels`; next run `muscle,weight,cupsize,firmness,african,asian,caucasian` (RUN
`..._shpmask_noage`) keeps age 0.55 supervised, the label that seemed to carry the gender/children gains of the 8-mask.

### exp8: ViT-B student distilled from the final model (requirement 1b) (WIP commits aca1896 + 3fe870d, not on main, never run)

**Purpose.** Supervisor requirement 1b, smaller-model route: a model that is not exactly Multi-HMR because it is much
smaller (fewer parameters AND FLOPs, faster inference) while its PVE and MPJPE stay within 95% of the large model on
BOTH the AnnyOne holdout and 3DPW test. Script: `slurm/experiments/exp8_distill_vitb.sh` (exp7's structure, guards and
modes). Cost table: `scripts/analysis/model_cost.py`. Baseline: `slurm/eval/run_eval_baseline_naver.sh`.

**Recipe.** Teacher = the final model (exp7 fix arm, `v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt`),
frozen, its ViT-L in bf16. Student = DINOv2 ViT-B/14 with NAVER's HMR-trained ViT-B weights
(`models/multiHMR/multiHMR_672_B.pt`, backbone only; [Guessing] that NAVER serves that file and that its ViT loads at
>= 90% as the ViT-S one did - the job's guard refuses without the file, train.py refuses below `--min_backbone_match`),
fully trained. `--head_dim 1024` puts a 768 -> 1024 projector
(`feat_proj`) in front of heads built at the teacher width, so `--init_heads_from_teacher` copies every head tensor (at
head_dim 0 the head's input layers cannot transfer: the v3-v5 failure). The copied heads stay frozen for 5 epochs.
The student trains with the teacher's own bone fix, because its heads start as the teacher's (train.py warns
otherwise). Targets are exactly the teacher's: `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5`, the 55
manifest images excluded, 30% 3DPW-train from the v23 labels, `--mask_3dpw_hands 1`, the v5 loss weights. KD as v8:
feature KL 1.0 (T 4), output L1 1.0. Batch 8, lr 3e-5, 30 epochs x 1000 iterations = 240,000 images.

**After training (the student's own bone fix).** `diag_holdout_trace.py --select train_sample` on 200 seeded training
images with `--bone_fix none`, then `compute_bone_fix.py` (auto) -> `<RUN>/bone_fix.json`. It replaces the training fix
in the holdout trace and in the 3DPW-test evaluation (`run_eval_3dpw_test.sh` gained `EVAL_BONE_FIX`). No qualifying
bone = no fix (`--bone_fix none`), recorded in the JSON. The GLOBAL GUARD (exit 4) continues by default, as the teacher's
own exp6 fix did (`ON_GLOBAL_RISE=safe` re-estimates with the safe subset). The teacher is re-measured in the same job
(holdout trace + 3DPW test), so the 95% targets never come from constants.

**The 95% rule.** For PVE and MPJPE on the AnnyOne holdout (V1 trace) and on 3DPW test (v23, `evaluate()`):
student error <= teacher error / 0.95. PASS = all four met, fewer params AND FLOPs, and faster when the benchmark ran.
PA-PVE and recall get the same column for information. With exp7's numbers the targets would be PVE <= 77.6 (AnnyOne)
and <= 93.7 (3DPW). The job's last block prints the table and the verdict (`<RUN>/final_table.json`).

**Commands** (repo root on the cluster, after `git pull` and `conda activate multihmr`):

```
CHECK_ONLY=1 bash slurm/experiments/exp8_distill_vitb.sh      # guards only (labels, teacher args, weights, queue)
SMOKE=1 bash slurm/experiments/exp8_distill_vitb.sh           # 20 iterations + every post-step on small inputs
bash slurm/experiments/exp8_distill_vitb.sh                   # the real run (--time 14:00:00)
TEACHER=/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt \
  MASK_3DPW_SHAPE=1 bash slurm/experiments/exp8_distill_vitb.sh   # student of the shape-masked teacher
TABLE_ONLY=1 bash slurm/experiments/exp8_distill_vitb.sh      # rebuild the final table (same env as the run)
bash slurm/eval/run_eval_baseline_naver.sh                    # NAVER multiHMR_672_L_anny on AnnyOne test-100 + 3DPW test v23
python scripts/analysis/model_cost.py --no-build --all        # analytic backbone table (anywhere)
```

The shape-masked teacher needs `MASK_3DPW_SHAPE=1` too: the teacher check refuses a student whose targets differ from
its teacher's. If the job times out during step 5, rerun that evaluation alone (the `MODEL_ENTRY` / `EVAL_BONE_FIX`
lines it printed), copy its log to `<RUN>/eval_3dpw_test_{student,teacher}.log`, then run `TABLE_ONLY=1`.

**What PASS means.** The student meets 1b's smaller-model clause on our two test sets: within 95% of the final model on
PVE and MPJPE on the AnnyOne holdout and on 3DPW test (v23 labels), with about 3.1x fewer parameters and 3.2x fewer
FLOPs (analytic backbone + hooked head linears, counted locally on a stand-in backbone [Likely]: ViT-B student 101.4 M /
603 GFLOPs vs teacher 318.4 M / 1927 GFLOPs at 672 px, one person; the job re-counts both from the real checkpoints). It
says nothing about other datasets, and the 3DPW numbers hold only against our converted labels.

**Risks.**
- Student bone fix: the fix is a fresh estimate on 200 training images. A compensated chain (neck02 -> head) or a bone
  that flips only in the student can still cost PVE. Compare `holdout_student/` with `holdout_student_trainfix/`.
- ViT-B may miss 95% on 3DPW even if AnnyOne passes. The ViT-S students stayed far from their teachers (v8 final 123.6
  vs its teacher v9 102.5 PVE on 3DPW test, old labels; handover model table), and 30 epochs is 10% of v8's images. A FAIL there calls for a longer run (`EPOCHS=60`) before any
  claim.
- FLOPs are analytic for the backbone (+ hooked linear layers for the heads). The Anny mesh is not counted. Latency is
  batch 1 on whatever GPU the job got (teacher and student on the same one).
- Memory: batch 8 with a fully trained ViT-B next to the ViT-L teacher may not fit 40 GB [Guessing]. The smoke test shows
  it in minutes if it lands on such a card; else `BATCH=4`.
- Throughput is a guess (~10 images/s on an A100: the ViT-L teacher still runs every step), so the 14:00 limit assumes
  6 images/s. Read the smoke test's `[throughput]` line.

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
| 6D "identity" init `[1,0,0,1,0,0]` is singular under `rot6d_to_rotmat` (columns x[0:3], x[3:6]) | `model.py:569`; `utils/humans.py:20-21` | yes (fingers' 0.25-0.30 conditioning is \|sin\| of this angle) [Likely link] | **not implemented** (it seeds the neck flip hypothesis, section 2 "Neck root cause"; measured: neck02 is a twist flip (\|tw\| 175.7, swing 3.3), which a half-turn about local X does not give; the init fits metacarpal1.L's FLIP_SWING only; distinct from exp5_neck_fix). Candidate: after load, set the init to `[1,0,0,0,1,0]` (delta = `(0,0,0,-1,+1,0)` per bone, 978 values) and compensate BOTH places the init enters the head: the residual (`decpose.bias -= delta`, `model.py:650`) and the token input (`token = cat([x_central, init_body_pose, init_betas, init_cam])`, `model.py:627-632`, fed to `transformer.to_token_embedding`, an `nn.Linear`: `to_token_embedding.bias -= W[:, D:D+978] @ delta`, D = width of x_central). The decpose shift alone is NOT output-neutral: on the real `HPH` with random weights it changed rotmat entries by up to 1.86; with both shifts the change is 1e-14 in float64 and 2.6e-6 in float32 (round-off, so the check cannot be bit-exact; synthetic scratchpad check, not in the repo). Require a before/after prediction match (max \|d rotmat\| <= 1e-5 in fp32) before crediting any later change to it. Then reset the 30 phalanx rows to exact identity (a real change, intended); the buffer is restored from the checkpoint, so a constant change alone is overwritten |
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

## Relative Human evaluation (children, gender on real photos)

**What it measures** (docs/handover.md 3.4, section 5 item 12.1). On Relative Human (RH) real photos, labelled per
person with an age group (adult / teen / kid / baby) and gender: the model runs once per whole image, as the plain
demo.py path does (672 px, fov 55, det_thresh 0.3, nms 3). Each detection's box comes from its projected vertices.
Detections are matched one-to-one to RH people by IoU (greedy, >= 0.3) against `bbox_wb` (whole body) when the
person has one, else `bbox`. Per group it reports: recall, mean and median predicted Anny age, the fraction below
`--child_thr` 0.45 (child-shaped), mean height and proportions, and gender accuracy (prediction < 0.5 = male). It also
reports the child-vs-adult AUC of predicted age (kid + baby against adult). The script is
`scripts/diagnostics/eval_relative_human.py`; the predictions are cached per image under `<out>/pred/`.

**Commands**
```
sbatch --array=0-1 slurm/diagnostics/eval_relative_human.sh            # 0 = v5 00099, 1 = exp6 fine-tuned v5 (own bone fix)
SPLIT=test sbatch --array=0-1 slurm/diagnostics/eval_relative_human.sh  # test split (1,836 images)
N=20 sbatch --array=0 slurm/diagnostics/eval_relative_human.sh          # smoke test
python scripts/diagnostics/eval_relative_human.py --score_only --out results/diagnostics/relative_human_val_anny_s2_shape_v5_00099 --iou 0.5 --child_thr 0.35
```
Outputs: `summary.txt` (also in the log, `[rh]` lines) and `per_person.csv` (every RH person, matched or not, with
the 11 phenotypes and the predicted box).

**How to read it**
- A working model puts the kid and baby mean age clearly below the adult mean, with AUC well above 0.5. AUC ~0.5
  means age carries no child signal. Given the median pull (3.4), expect every group near 0.6.
- Always quote recall next to it. Multi-HMR at 672 px misses small people, and the AUC covers detected people only,
  so a low kid or baby recall biases it.
- Gender: read the overall accuracy and the per-group accuracy against the male / female mean predictions. 3.4
  predicts men pulled towards 0.59.
- `--child_thr` and `--iou` are reading choices. Re-score with `--score_only` (no GPU) to show that the conclusion
  does not depend on them.

**License**: the RH repository (https://github.com/Arthur151/Relative_Human) has no license file. Use it for
evaluation only: no training, and do not redistribute the images.
