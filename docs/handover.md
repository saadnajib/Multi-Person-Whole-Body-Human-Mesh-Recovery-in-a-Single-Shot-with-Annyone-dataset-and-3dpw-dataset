# Handover: project state on Thursday 15 October 2026 (repo `main` at `64266dc`)

This file replaces the chat history. It is written for an assistant that cannot see that chat, the cluster or any
temporary files. Everything you need is either here or in this repository. **Where this file disagrees with
`README.md`, `docs/diagnostics_findings.md`, `docs/experiments_plan.md` or `docs/experiments/exp2_children.md`, trust
this file** (section 7 lists the outdated lines). The speaker notes `docs/supervisor_talk_notes_2026-10-16.md` are
current.

Supervisor meeting: **Friday 16 October 2026**, with the deck `docs/supervisor_update_2026-10-16.pptx`.

---

## 0. Paste this first

The student pastes this block as the first message, together with this file.

```
You are helping me finish my master thesis. Act as a critical advisor, not a cheerleader.

Project in 5 lines:
- Multi-HMR (multi-person whole-body mesh recovery; ViT-L DINOv2 backbone at 672 px + cross-attention head),
  moved from SMPL-X to the Anny body model (163 bones, 11 shape values in [0,1]).
- Trained on AnnyOne (~580k synthetic images) and on 3DPW converted from SMPL to Anny by my own fitting script.
- Main model v5 (epoch 99): 70.5 mm PVE on a fixed 100-image synthetic holdout (setting "V1").
- This week: found a depth-target bug (fix works on placement, likely costs 2.6-4.9 mm PVE), a half-turn flip of two
  neck bones in every v5-line model (a correction, exp6, is running), and three defects in my 3DPW labels.
- Fingers and children: tested, nothing helped yet.
I run jobs on a Slurm cluster you cannot see. I paste or upload logs, in batches, and say "done" when all are in.

How to answer:
- First sentence: challenge my assumption or name what is missing. No warm-up.
- Tag every claim [Certain], [Likely] or [Guessing].
- No praise phrases ("Great question", "You're absolutely right", "Absolutely", "Definitely").
- Uncomfortable answer first. Simple, short words.
- To disagree: "I disagree because ... Here's what I'd do instead ... The risk is ...".
  Do not give in to pushback without new evidence.
- Commands: complete and copy-pasteable, run from /netscratch/najib/multi-hmr. Never use <placeholders>.
  Never put a destructive git command (reset, checkout --, clean, push -f) in the same block as other commands.

Before answering anything, read the whole handover.md I attached. It overrides the other docs in the repo.
```

---

## 1. Project in brief

**Model.** Multi-HMR: DINOv2 ViT-L/14 backbone at 672 px plus the HPH cross-attention head. It detects people on a
patch grid and, per person, predicts Anny pose (163 bones as 6D rotations), 11 phenotypes (gender, age, muscle, weight,
height, proportions, cupsize, firmness, african, asian, caucasian) and a translation. The model anchors each person at
the **head** joint (`--person_center head`). 88 helper bones (toes, breasts, face) are masked to identity in training
(`--mask_helper_joints_train 1`), leaving 75 active bones.

**Data.**
- AnnyOne: 579,960 synthetic 1280x1280 renders. Indices 0-579,859 are training. The last 100 (579,860-579,959,
  `--val_anny_n 100`) are the **holdout**. It has no children.
- Diagnostic manifest `results/diagnostics/manifest.json`: 40 training images (adults, children, visible hands, hard
  necks) plus 15 held-out images with no children (notes :13).
- 3DPW converted to Anny (`scripts/data/smpl_to_anny.py`, `scripts/data/build_threedpw_anny_labels.py`): labels for
  4,451 train / 2,458 validation / 9,722 test images, every third frame (README :227). Mixed into v9/v10 batches at 30%.

**Models** (checkpoints are not in git; cluster paths below).

| name | what | AnnyOne holdout V1 PVE (job:line) | 3DPW-test PVE / PA-PVE* |
|---|---|---|---|
| v5 = `anny_s2_shape_v5` epoch 99 | ViT-L, synthetic only. Reference for every diagnostic | **70.5** (3570222:40) | 119.1 / 68.8 |
| v9 = `anny_teacher_v9_mixed` epochs 62, 149 | v5 + 30% 3DPW-train. Teacher of the final student | 75.0 / 77.0 (3571963:40, 3571964:40) | 102.5 / 62.7 |
| v10 = `anny_v10_scratch_mixed` epoch 290 | ViT-L from scratch, mixed data from the start | 74.1 (3571794:40) | 111.5 / 76.2 |
| students v6c, v8 final | ViT-S distilled | not traced | 139.5 / 88.0, 123.6 / 81.2 |

\* From `slurm/demo/run_demo_compare.sh:5-9`, measured with `train.py evaluate()` (V0, unmasked) against **our converted
labels** on a stride-3 subset. They are not comparable with published 3DPW numbers, and the labels have known defects
(section 3.5). Say "best on our 3DPW-Anny labels", never "best model".

**v5 line** = the chain of checkpoints that led to v5 and v9: `anny_s2_frozen_bb_v2` 00299 -> `anny_s2_partialft_v3`
00054 -> `anny_s2_boosted_v4` 00299 -> v5 00099 -> v9. The exp1/exp5 fine-tunes also start from v5.

**Metrics.** PVE = mean vertex error after aligning the pelvis (mm). PA-PVE = after Procrustes. Recall = share of GT
people detected. All three are pelvis-centred and **cannot see placement or depth errors**. The trace
(`scripts/diagnostics/diag_holdout_trace.py`) adds:
- `root mm` = absolute 3D distance between predicted and true root (pelvis) joint. Use this for depth.
- `root dz m` = signed mean depth offset. Near and far errors cancel in it. **Never quote it.**
- depth slope / intercept = linear fit of predicted root depth against true depth (perfect: 1 and 0 m).
- size ratio = predicted head-to-root distance in pixels / true (perfect: 1).
- rotation error per bone group (local = relative to parent bone), global head orientation error (deg).
- neck girth = neck-skin radius pred/GT; PVE_neck = neck-skin vertex error centred at neck01 (mm).

**Evaluation settings** (`diag_holdout_trace.py:14-24`):
- V0: as `train.py evaluate()`, fp32, dataset camera, helper bones **unmasked**.
- V1: V0 + helper bones masked. **The headline setting.**
- V2: as `demo.py` on synthetic images (fp16, camera from FOV, masked).
- V3: V2 + `--relax_hands 1` (fingers set to rest pose).
- V4: V1 on a square crop around each GT person (`--person_crop 1`, exp4 only).

**Headline.** v5, V1: PVE 70.5 mm, PA-PVE 53.3, recall 90.2% (3570222:40). The thesis draft still says 74 mm; that is
V0 (unmasked helper bones add +3.8 mm on the same people, README :321). Change the draft to 70.5.

**What is not the cause** [Certain] (notes slide 4, `docs/diagnostics_findings.md`): GT parameters decode onto the
people; all 16 single-joint perturbations move the right body part; every active bone and shape output gets gradient
(two of three camera outputs get none); no joint-mapping or rotation-convention error. The overfit test on 40 images
lowers the error by 36%, but the translation and 2D losses rose while memorising (1.67 -> 1.89, 58 -> 75), probably an
early sign of the depth bug [Likely]. Training skips 3-5 optimiser steps per 2,000 for NaN/Inf gradients in every run
(e.g. 3572482:32-34); clipping absorbs it; mention it in the thesis as a known issue.

---

## 2. Where things are

**Repository.** github.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset,
branch `main`, HEAD `64266dc` (15 Oct). The cluster copy is updated with `git pull`.

| path | what |
|---|---|
| `train.py`, `model.py`, `loss.py`, `demo.py`, `bone_fix.py` | training, model, losses, demo, the exp6 bone correction |
| `scripts/diagnostics/diag_holdout_trace.py` | the evaluation used for every number here (V0-V4, placement, per-bone dump) |
| `scripts/diagnostics/compute_bone_fix.py` | exp6: estimates one correction rotation per flipped bone |
| `scripts/diagnostics/check_3dpw_conversion.py` | end-to-end check of the 3DPW->Anny labels |
| `scripts/experiments/` | exp summaries (`exp5_neck_summary.py`, `exp6_bone_fix_summary.py`), `measure_batch_ages.py`, `measure_hand_px.py`, `build_age_index.py` |
| `scripts/data/smpl_to_anny.py` | the SMPL->Anny fitter (gender bug at :1008) |
| `scripts/data/build_threedpw_anny_labels.py` | fits -> per-image label files |
| `slurm/experiments/exp1..exp6_*.sh` | one job script per experiment; the header of each explains why, what, how to read |
| `slurm/diagnostics/*.sh` | `diag_holdout.sh`, `diag_neck_bones.sh`, `check_3dpw_conversion.sh`, others |
| `slurm/data/run_build_labels.sh`, `run_smpl_to_anny_task.sh` | 3DPW label build and the per-person fit array job |
| `slurm/train/` | v9, v10, final student training scripts |
| `docs/` | this file, the deck, speaker notes, findings, plan, `deck_src/` (deck source) |
| `results/experiments/<exp>/<RUN>/` | experiment outputs on the cluster; only `*.json` and `*.md` are tracked in git |

**Key flags** (all default off / old behaviour).

| flag | where | effect |
|---|---|---|
| `--gt_center_joint person_center` | train.py | depth fix: build transl/dist/heatmap/offset targets at the anchored head, not the root |
| `--alpha_dist`, `--alpha_transl` | train.py | depth loss weights (exp1 used x5, x20 via `ALPHA_DIST`/`ALPHA_TRANSL`) |
| `--age_balance 1`, `--shape_weights` | train.py | exp2 child-oversampling sampler, per-phenotype loss weights |
| `--alpha_global_rot`, `--global_rot_bones` | train.py | exp3 loss on global head/neck03/wrist orientation |
| `--rotmat_region_norm 1` | train.py | exp4 rotation loss averaged per body region |
| `--mask_3dpw_hands 1` | train.py | ignore 3DPW finger labels (they are rest pose). Use in any mixed fine-tune |
| `--rotmat_loss geodesic` | train.py | exp5 geodesic rotation loss (default `l1`) |
| `--bone_fix <json>` | train.py, trace, demo.py | exp6 constant correction `R <- R·C` for listed bones; saved into checkpoint args |
| `--dump_bones 1`, `--dump_bone_names all` | trace | per-bone error table (verdicts FLIP_TWIST, FLIP_SWING, ...) |
| `--relax_hands 1` | demo.py | sets bones whose name has "finger"/"thumb" to rest. **Does not touch metacarpals** (model.py:165-177) |
| `--anchor_to_box 1` (default) | demo.py:785 | real-photo demos place each body from its detection box, so they hide the depth bug |

**Cluster.** DFKI Slurm. Repo at `/netscratch/najib/multi-hmr`, conda env `multihmr`, enroot container, shared setup
`slurm/env.sh`. Logs: `/netscratch/najib/multi-hmr/logs/<jobname>_<jobid>.out` and `.err` (label build:
`3dpw-labels-<jobid>.out`). Checkpoints: `logs/anny_model/<run>/checkpoints/<epoch>.pt` (only the last 10 kept) and
copies in `models/multiHMR/<run>_<epoch>.pt`. v9 lives only at `models/multiHMR/anny_teacher_v9_mixed_00062.pt` and
`_00149.pt`. Status: `squeue -u najib`.

**How work flows.** Code is changed in the repo, pushed to GitHub `main`; the student runs
`cd /netscratch/najib/multi-hmr && git pull`, then `sbatch slurm/...` from the repo root, then uploads the `.out`
logs. Diagnostics need `TAG=` or their outputs overwrite each other. Throughput: 6.7 images/s at batch 4 with 4 blocks
unfrozen; a 2-epoch pilot is ~12 min of training, a 10-epoch run about one hour (notes :61).

**Citation style below.** `3572482:126` = line 126 of the job's `.out` log (job index in section 8). The student has
these logs; ask for one when you need to check a number.

---

## 3. Status per problem

### 3.1 Depth (the main result)

**Mechanism** [Certain]. The model attaches each mesh to its detection cell by the HEAD joint and reports the head as
its translation, but `train.py prepare_gt` built the transl, dist, heatmap-cell and offset targets at joint 0, the
ROOT. The 2D losses resolve the conflict by pushing the body away, so it is drawn too small
(`docs/diagnostics_findings.md:86-104`). A one-batch debug print showed predicted offsets of -2 to -19 patches against
a legal range of +/-0.5 (`--debug_anchor`, 3571599:9-14). Pelvis-centred PVE cannot see it; the real-photo demos hide it because they
place bodies from the detection box. The synthetic demo and the trace panels show it (`docs/figures/`).

**Measured** (V1, 100-image holdout, matched people; root = `root mm` column of the PLACEMENT table).

| run | job:lines | ep | depth loss | root err | size | slope | icpt m | PVE | recall | pelvis px |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 baseline | 3571596:55,64; 3570222:40 | - | x1 | **1.21 m** | 0.63 | 0.33 | 2.36 | 70.5 | 90.2 | 23.4 |
| control `pilot_root_dist1_transl1_...` | 3571599:49,58,67-68 | 2 | x1 | 1.11 | 0.66 | 0.33 | 2.28 | 72.32 | 89.84 | 23.3 |
| fix `pilot_person_center_dist1_...` | 3571598:66,75,84-85 | 2 | x1 | 0.69 | 0.78 | 0.30 | 1.93 | 77.25 | 87.80 | 37.3 |
| fix `pilot_person_center_dist5_...` | 3572418:52,61,70-71 | 2 | x5 | 0.66 | 0.80 | 0.44 | 1.62 | 76.77 | 87.80 | 38.1 |
| fix `pilot_person_center_dist20_...` | 3572483:57,66,75-76 | 2 | x20 | **0.53** | 0.87 | 0.43 | 1.44 | 75.66 | 87.40 | 31.4 |
| control `long_root` | 3571934:134,143,152-153 | 10 | x1 | 1.44 | 0.59 | 0.34 | 2.59 | 72.19 | 89.43 | 24.3 |
| fix `long_person_center` | 3571933:125,134,143-144 | 10 | x1 | 0.73 | 0.78 | 0.30 | 1.98 | 75.51 | 86.99 | 38.5 |
| fix `long_pc_dist5` | 3572482:126,135,144-145 | 10 | x5 | **0.54** | 0.84 | 0.38 | 1.59 | 74.77 | 87.80 | 33.1 |
| v9 epoch 62 | 3571963:40,55,64 | - | - | 1.39 | 0.59 | 0.31 | 2.60 | 75.0 | 88.6 | 25.1 |
| v9 epoch 149 | 3571964:40,55,64 | - | - | 1.44 | 0.57 | 0.29 | 2.68 | 77.0 | 89.4 | 25.5 |
| v10 epoch 290 | 3571794:40,55,64 | - | - | 1.49 | 0.57 | 0.28 | 2.71 | 74.1 | 89.0 | 24.0 |

Run directories: `results/experiments/exp1_depth_anchor/<run>/`. All runs fine-tune v5 00099.

What this says:
- [Certain] The fix moves the root: 0.53-0.73 m in all five fix runs vs 1.11-1.44 m in both controls. Size goes from
  0.59-0.66 to 0.78-0.87. The head now lands within 7-10 px (v5: 64.5 px).
- [Certain] Near vs far is still not separated: slope 0.30 with the switch alone, 0.38-0.44 with a stronger depth loss.
  Predictions are pulled toward ~2.5-2.9 m; cause unknown (notes :53, :55). Holdout people are at 1.92 m on average
  (README :114).
- [Certain] By its own success rule exp1 **failed**: every run, controls included, prints
  `PVE <= 71.5 and recall >= 89.2: NO` (e.g. 3572482:150), and the slope did not move toward 1.
- [Likely] The fix has a cost. Every fix run is worse than its control: PVE +4.93 / +4.45 / +3.34 (2-epoch runs vs
  3571599) and +3.32 / +2.58 (10-epoch runs vs 3571934), so +2.6 to +4.9 mm; recall -1.6 to -2.4 points. Each gap is
  inside pilot noise (section 3.6), but all five point the same way. The five fix runs share only two controls, and
  x5/x20 have no control at the same weight.
- [Certain] The pelvis 2D error got worse: ~23 px -> 31-39 px. [Likely] because the head is now pinned while the body is
  still drawn too small; not checked.
- [Certain] x20 for 2 epochs ties x5 for 10 epochs (0.53 vs 0.54 m). Longer training lowered x5's root error 0.66 ->
  0.54 m but not the slope (0.44 -> 0.38). Root error has no measured noise floor.
- [Certain] v9 and v10 carry the same bug (table above). [Likely] the ViT-S students too (same training code; not
  measured).
- In 4 of 12 rendered holdout images both v5 and the fix miss people very close to the camera [Likely detection; not
  counted] (notes :49).

**Refuted / do not use:** "depth error 1.08 -> 0.39 m" (signed `root dz`, 3572482:126); size 0.67 for v5 (an older
statistic in the findings doc and the exp1 script's "baseline(findings)" column; the trace's own value is 0.63).

**Running:** maybe `long_pc_dist20` (x20, 10 epochs). Never confirmed as submitted; no result.

**Next:** see section 5, items 3-6.

### 3.2 Neck

**Measured** [Certain] (per-bone dump, `slurm/diagnostics/diag_neck_bones.sh`, jobs 3573518 and 3573519; V1 holdout,
222 people). In v5 00099:

| bone | verdict | median error | note (3573518 line) |
|---|---|---|---|
| neck01 | OK | 5.3 deg | :12 |
| neck02 | **FLIP_TWIST** | **175.6 deg** | twist 175.7, swing 3.3: a half-turn about its own axis (:13) |
| neck03 | OK local | 5.7 deg | but global 173.3 deg, carried from neck02 (:14) |
| head | **FLIP_TWIST** | **177.9 deg** | swing 28.8, **global only 12.0 deg** (:15) |
| jaw | OK | 2.2 deg | :16 |

- The two half-turns cancel, so the head still points the right way (global 12 deg) and PVE does not show it. Last
  week's "62.7 deg neck / 89.5 deg head" were group means of these bones: (5.3 + 175.6 + 5.7)/3 ~ 62; (177.9 + 2.2)/2 ~ 90.
- But the neck skin is twisted thin: girth pred/GT **0.46**, PVE_neck 27.3 mm (3573518:69). v10: girth 1.00, PVE_neck
  10.3 mm, neck02 3.9 deg, head 6.8 deg (3573518:189, :191, :245, :359).
- The flip is in **all 8 v5-line checkpoints**: frozen_bb_v2 00299, partialft_v3 00054, boosted_v4 00299
  (3573519:269-271), v5 00099, v9 00149, long_pc_dist5 00009 (3573518:357-360), and both exp5 runs (3573520, 3573521).
  neck02 175.0-176.0, head 177.1-178.0 (FLIP_SWING instead of FLIP_TWIST in partialft_v3 and the exp5 geodesic run),
  girth 0.46, PVE_neck 27.3-28.7 mm. It is there from the first stage (frozen backbone), before the helper mask, the
  x4 boost and `alpha_shape 20`.
- The depth-fixed `long_pc_dist5` keeps the flip and girth 0.46, so **depth is not the neck cause**.
- Same mechanism elsewhere (3573518):
  - v5 line: `metacarpal1.L` FLIP_SWING, constant 180 deg (left index finger base bent backwards; child error
    185-200 mm) (:31).
  - v10: `shoulder01.L` FLIP_TWIST 166.3 (not constant, relSpr 21.3) (:195); `upperarm01.L` 58 deg local but 169 deg
    global (:199); `upperarm02.L` FLIP_TWIST 180 (:200), which turns the chain back, so the arm is right again below it;
    `metacarpal2.L` 180, `metacarpal2.R` 180, `metacarpal3.R` 180, all uncompensated (:208, :212, :213). This is the
    "deformed left arm" the student saw in v10 renders, and most of why v10's shoulder/arm group errors are 50 / 36 deg
    (3571794:47).

**Refuted** [Certain]:
- H4 "the GT blocks are not proper rotations": all are proper rotations (3573518:72).
- H3 "the checkpoint uses another pose parameterisation": `pose_param=None` (local-bone) in every checkpoint (:73).
- Mask, loss-weight, evaluation and 3DPW explanations: refuted by a code audit (`docs/experiments_plan.md:237-245`).
- "Weak supervision on a redundant chain" and "the long neck comes from the depth bug": refuted by the flip, by v10
  learning the neck with the same loss, and by `long_pc_dist5`.
- exp3 (global head loss, 2 epochs, job 3571957): global head 11.3 deg vs 11.7 / 13.6 for two identical controls
  (3571957:98; 3571960:318; 3571958:99): inside noise. Neck bones stayed at 62.5 (3571957:100).
- exp5 (2 epochs, lr 1e-5): the geodesic loss did not move the flip: neck02 175.6 -> 175.0, head 177.9 -> 177.1; the L1
  control stayed at 175.7 / 177.9 (3573521:89-91; verdict "NO MOVEMENT", :103). [Likely] neck02 and head would have to
  turn back together; a partial move raises the loss.

**Origin: unknown.** The 6D init `[1,0,0,1,0,0]` (`model.py:641`) is singular under `rot6d_to_rotmat`
(`utils/humans.py:20-21`). It starts a bone turned about its local X axis, i.e. a tilt (FLIP_SWING). That fits
metacarpal1.L, **not** neck02's twist [Certain about the geometry; the origin of the neck flip is open].

**Not yet done:** render the flip to confirm it is the thin/long-neck look seen in the overlays.

**Running: exp6** (`--bone_fix`, code in `d902e00`, final in `81dbdcb`). One constant rotation `C` per flipped bone,
estimated on 200 random training images (never the holdout) by `compute_bone_fix.py`, applied as `R <- R·C` right after
the 6D decoding. Then a zero-training test on the holdout and a 2-epoch fine-tune with the fix in place. Expected
[Likely]: neck02 -> ~5 deg, metacarpals -> ~0-1 deg, head -> ~15 deg (its half-turn compensates neck02 differently per
person, which one constant cannot undo), global head error rises. Commands and pass criteria in section 4.

**Not implemented:** the two-sided correction `R_l <- K_l^T R_l K_l` (needed to keep a compensated chain such as
v10's left arm intact) and a chain-preserving head correction (`docs/experiments_plan.md:534-558`).

### 3.3 Fingers

**Measured** [Certain] (v5, V1):
- Predicted fingers 17.3 deg; fingers set to rest pose 15.1 deg (3571960:290). The prediction is worse than no
  prediction, so all demos use `--relax_hands 1` and say so.
- Hands are small: median 2.25 tokens long at 672 px, 41% under 2 tokens (3571960:287).
- exp4a, test-time person crop (x1.3, no retraining): paired 17.3 -> 17.2 deg (3571960:299); body PVE nearly doubles
  (124.8, 3571960:294) because the model never saw such a camera.
- exp4b, region-normalised rotation loss: 17.5 deg vs 17.4 for its control (3571959:316; 3571960:314). It cut the
  hands' share of the rotation loss from ~70% (with the x4 hand boost) to 33%, so it is a rebalance, not an
  upweighting (notes :35). The earlier x4 hand boost (v4) did not help either.
- v5's `metacarpal1.L` is flipped 180 deg (3.2). [Likely] it adds ~4.7 deg to both finger numbers, so it does not explain
  the 17.3 vs 15.1 gap. `--relax_hands` does not reset metacarpals, so [Likely] the bent left index base is visible in
  the demos.

**Refuted:** "fingers are a resolution problem" for test-time zoom only. **Not shown:** that a model trained on larger
hands would not help. "Under-trained" / "supervision or capacity" is a guess [Guessing].

**Next:** train on larger hand crops; otherwise a thesis limitation. Low priority.

### 3.4 Children

**Measured:**
- [Certain] The shape head is complete and can make a child in-sample: the 50 youngest training people (GT age 0.015)
  are predicted at 0.10 (3571955:174-175).
- [Certain] Natural batches: 2.1-2.3% of supervised people have age < 0.3 (3571955:154; 3571956:149; 3571601:35).
  The balanced sampler (`--age_balance 1`, child fraction 0.25) raises this to 9.7-11.1%; batches without a child
  82.5-84.5% -> 36.0-43.0%; images with a child ~5-6% -> 23.0-25.4% (3571955:154-156; 3571956:149-151;
  3571601:49-51).
- [Certain] 2-epoch pilot (job 3571955) vs control (3571956, sampler off):
  - manifest children (n = 10, in-sample): PVE 48.5 vs 53.1, age bias 0.078 vs 0.085; v5 baseline 47.8 / 0.078
    (3571955:169-170; 3571956:164-165). Inside noise; no effect.
  - real photos, five small people (skeletal height 0.6-1.14 m): predicted age 0.57-0.67 with the sampler and in the
    control, 0.60-0.67 for v5 (diag_cropK jobs 3572419, 3572420, 3569949, lines 7, 10, 13, 58, 70). The holdout adults
    sit mostly at 0.4-0.8 (notes :39).
  - both arms print "ADULT REGRESSION" (adult PVE 75.1 / 74.9 vs 70.5): that is the restart cost of any fine-tune,
    not the sampler.
- The 3DPW labels fix age at 0.55 for everyone (3575369:232), so mixing 3DPW adds no child signal.

**Age scale** [Likely]: by Anny's own calibration, age 0.10 ~ 2 y, 0.3 ~ 7 y, 0.5 ~ 12.7 y, 0.55 ~ 13.6 y, 0.67 ~ 16 y
(linear between Anny's anchors). The labels in our code assume 0.33 = 11 y and 0.67 = 25 y. Which one AnnyOne follows is
not settled (notes :41). Avoid years; compare with the holdout adults instead.

**Verdict so far:** no effect from a 2-epoch sampler. [Guessing] a synthetic-to-real gap plus an old-age bias.

**Next:** one 10-epoch balanced run (section 5); if nothing, a thesis limitation.

### 3.5 3DPW -> Anny labels

**How they are made** (notes slide 5). Valid frames, every third one **per person**; Anny is fitted per frame by
gradient descent so its 24 mapped joints land on the 24 SMPL joints (positions only; SMPL rotations and shape are not
used). Gender comes from 3DPW, age is fixed at 0.55, only height and proportions are free. One median body per person,
then a pose refit. Only the root and 18 bones move; 144 of 163 bones stay identity, including neck02, neck03, head and
the fingers. The label builder drops fits > 80 mm, people < 30% visible and boxes < 40 px.

**Measured** [Certain] (job 3575369, `scripts/diagnostics/check_3dpw_conversion.py`, ~2,000 sampled images):

| check | result | line |
|---|---|---|
| body joints vs 3DPW | median 18.7 mm, p95 24.5 | :329 |
| surface, nearest point | 18.9 / 18.3 mm (lower bound; worst region torso 28.2) | :331 |
| 2D | 11.28 -> 12.39 px, +1.1 px over raw 3DPW | :332 |
| left/right, mirror | 100% | :333-334 |
| loader passes labels unchanged | PASS, max diff 0.000 | :338 |
| converter fits | 87 fits, 22,915 frames, median 18.2 mm, max 70.9 | :342 |

**Defects:**
1. **Gender reversed.** `smpl_to_anny.py:1008` sets `1.0 if gender == 'm' else 0.0`, but Anny's male end is 0 (rest
   stature 1.753 m at 0 vs 1.632 m at 1; 3575369:24, :313). Every man is labelled 1.000, every woman 0.000 (:225-226).
   Refit with flipped gender: men better in 72/73 samples, joints 19.2 -> 15.3 mm. Women: joints worse for most
   (13.7 -> 16.3 mm), surface closer for 27/31 (:316-317). So the fix is confirmed only for men.
2. **Missing actor.** 407 of 960 two-person images have one valid actor without a label; all 407 because that actor was
   not fitted on that frame (per-person stride phase) (:228-229).
3. **Missing labels.** Train has 4,451 label images where the fits give 6,437 (:31). Six train sequences have none
   (courtyard_arguing_00, backpack_00, basketball_00, bodyScannerMotions_00, box_00, capoeira_00) and captureSelfies_00
   is partial (:63-89). Validation: courtyard_basketball_01 not built (2,458 of 2,777; :26, :33), and all 162 frames
   of outdoors_crosscountry_00 are dropped by the < 30% visibility filter (:96); the rebuild will not recover that one.

**By design, not bugs:** neck02/neck03/head at rest in 100% of labels; age 0.55; head turn kept only through neck01:
overall 9.5 vs 20.2 deg in SMPL (:223-224). Train and validation keep most of it (18.5 vs 21.2; 19.1 vs 19.3); test
loses most (6.3 vs 20.0; cause unknown).

**Script grade "OVERALL FAIL"** (:343-346) comes from: limb ratios (V4c) forearm 0.879, thigh 1.119 overall, failing
only in the test split (train passes with 0.965 / 1.035) (:214-215, :335); and V4e, 5.2% of train people with a joint
over 100 mm (FAIL in train; 3.2% overall = WARN) (:218, :337). Cause of the test-split limb ratios: unknown. All
checked test people are men labelled 1, so a link to the gender defect is possible, not shown (notes :27).

**Running:** label rebuild into `threedpw_anny_full/labels_rebuilt` (b395a10). Not confirmed.

**Next:** fix gender and use one shared frame grid for both actors, re-fit all 87 tasks into a **new** folder, rebuild,
re-check (section 5). Only then test anything on 3DPW.

### 3.6 Pilot noise (read every 2-epoch result against this)

The exp3 control (3571958) and exp4 control (3571960) are identical configs. They differ by **7.6 mm PVE** (76.3 vs
68.7; 3571958:104, 3571960:320), 1.9 deg global head (13.6 vs 11.7) and 0.8 recall points. Over six 2-epoch v5-recipe
arms: PVE 68.7-76.3, recall 88.6-90.2, global head 11.3-13.7 (`docs/experiments_plan.md:413-419`). So one pilot
resolves only ~10 mm PVE, ~3 deg, ~2 recall points. No random seed is set anywhere in the code. Restarting a fine-tune
alone costs ~2.4 mm (notes :61).

---

## 4. Pending jobs

All commands run on the cluster after `cd /netscratch/najib/multi-hmr && git pull`. On 15 Oct 15:39 the student said
"since it's running" without job IDs. **Treat every job below as "maybe submitted"**: ask for `squeue -u najib` or the
`.out` file before resubmitting.

### 4a. exp6 on v5 (bone correction + 2-epoch fine-tune)

```
sbatch slurm/experiments/exp6_bone_fix.sh
```
- ~30 min (limit 1 h). Log `logs/exp6_bone_fix_<jobid>.out`. Run dir
  `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/` (`bone_fix.json`, `summary.json`, `compare/*.jpg`
  = top row without, bottom row with the fix).
- The final table has 4 columns: baseline | zero-training | fine-tuned | control (the exp5 L1 run).
- **Zero-training PASS** needs all of: neck02, head and metacarpal1.L each below 15 deg; girth >= 0.9; global head <=
  baseline + 3 deg; PVE <= baseline + 1 mm.
- **Fine-tuned PASS** needs: the same bone and girth conditions; PVE <= control + 8 mm; recall >= control - 2;
  global head <= control + 3 deg.
- **Expected:** a "GLOBAL GUARD" warning (exit 4) in step a; the job continues. neck02 near 5 deg; head near 15 deg, so
  the zero-training column may fail on the head or global head. The fine-tuned column decides.
- If girth >= 0.9: render/look at `compare/` to confirm it is the look from the overlays.
- If the head stays ~15 deg and global head rises: implement the chain-preserving head correction (plan :558).

### 4b. exp6 on v10 (three metacarpals, no training)

```
CKPT=logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt TRAIN=0 BONES=metacarpal2.L,metacarpal2.R,metacarpal3.R sbatch --time=00:30:00 slurm/experiments/exp6_bone_fix.sh
```
- ~10 min. Expected: the three metacarpals -> ~0-1 deg, PVE and global head about unchanged, zero-training PASS.
- Do **not** use `BONES=auto` on v10: it would also correct shoulder01.L and upperarm02.L, break the compensated
  left-arm chain, and the guard would stop the job.

Re-print a finished table without a GPU (login node):
```
SUMMARY_ONLY=1 RUN=anny_s2_shape_v5_00099_n200_s0 bash slurm/experiments/exp6_bone_fix.sh
SUMMARY_ONLY=1 RUN=anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R bash slurm/experiments/exp6_bone_fix.sh
```

### 4c. 3DPW label rebuild (no new fit)

```
OUT=/netscratch/najib/multi-hmr/threedpw_anny_full/labels_rebuilt sbatch slurm/data/run_build_labels.sh
```
- ~1 min of CPU (limit 30 min). Log `logs/3dpw-labels-<jobid>.out`. Writes to a new folder, so the old labels that
  v9/v10 trained on stay intact.
- Read the per-sequence lines `[train] <seq> ... N images`, the per-split totals `[split] N images, M people` and
  `TOTAL`. Expected [Likely]: train ~6,437 (was 4,451), validation ~2,777 minus crosscountry (was 2,458), test 9,722
  unchanged. The "unconverted tasks" list at the end should be empty.
- The builder silently skips a frame whose image file is missing (`build_threedpw_anny_labels.py:123-125`). If a train
  sequence still shows 0 images, its images are probably missing under `data/3DPW/imageFiles/<seq>/`.
- Then re-check the new labels (my suggestion; not run yet):
```
LABELS=threedpw_anny_full/labels_rebuilt OUT=results/diagnostics/check_3dpw_conversion_rebuilt sbatch slurm/diagnostics/check_3dpw_conversion.sh
```
- Training scripts hard-code the old folder (`slurm/train/train_teacher_v9_mixed.sh:55`,
  `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`). Do not overwrite `labels/`.
- The rebuild does **not** fix gender or the missing actor; both are decided inside the fit.

### 4d. Depth x20 for 10 epochs (never confirmed submitted)

```
ALPHA_DIST=20 ALPHA_TRANSL=20 MAX_ITER=10000 RUN=long_pc_dist20 sbatch --time=04:00:00 slurm/experiments/exp1_depth_anchor.sh
```
- ~1-2 h. Log `logs/exp1_depth_anchor_<jobid>.out`.
- Read, in this order: PLACEMENT row `V1_eval_masked`, column `root mm` (never `root dz m`); then depth slope and size
  ratio; then PVE and recall in the SUMMARY.
- Compare with `long_pc_dist5`: 544.5 mm, size 0.84, slope 0.38, PVE 74.77, recall 87.80 (3572482:126-145), and with
  `long_root`: 1438.8 mm, 0.59, 0.34, 72.19, 89.43 (3571934). There is no control at x20.
- Use x20 in a v9 fine-tune only if it clearly beats x5 on root error and size with PVE and recall no worse.
  Otherwise keep x5.

---

## 5. Next steps, in priority order

1. **Read the pending results** (section 4). Analyse only after the student says "done".
2. **Neck.** If exp6 passes on neck02 and girth: render `compare/`, keep the fix (it is saved in checkpoint args).
   If the head stays ~15 deg or global head rises: implement the chain-preserving head correction; for v10's left arm
   the two-sided correction `R_l <- K_l^T R_l K_l`. Find the origin of the flip only if time allows.
3. **3DPW labels (after Friday, before any 3DPW claim).**
   - In `scripts/data/smpl_to_anny.py:1008` set men to 0 and women to 1 (Anny convention). The refit confirmed this
     only for men; check women after the re-fit.
   - Pick both actors' frames on one shared grid (stride applied to the sequence, not per person;
     `find_valid_frames`, `smpl_to_anny.py:737-776`).
   - Re-fit all 87 tasks into a **new** folder. `run_smpl_to_anny_task.sh` hard-codes
     `OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_full` and **skips any person that already has a file**, so
     make `OUT_ROOT` overridable first. ~40 GPU hours by the script's own 6 s/frame estimate, 5-6 h with `%8`
     (not measured).
   - Rebuild labels from the new fits (set `FITS` to the new fit folder and `OUT` to a new label folder), then re-run
     the check on them.
4. **Supervisor's decision (slide 16).** Fine-tune v9 now with `--gt_center_joint person_center` and x5, or first fix
   the labels (item 3) and then fine-tune? The student proposes labels first, because 3DPW is 30% of each batch and the
   men would keep learning the wrong body. Either fine-tune should add `--mask_3dpw_hands 1`. A v9 fine-tune keeps the
   neck flip unless it also uses `--bone_fix`. Fine-tune ~1 h; expected from v5: size ~0.84, root ~0.5 m, cost ~2.6 mm
   PVE and ~1.6 recall points (notes :77).
5. **Depth slope stuck at 0.3-0.44.** Measure the training-set depth distribution properly (so far only batch means of
   ~2.5-3.2 m) and the fix run's slope on training images. [Guessing] it needs more unfrozen blocks or a higher LR on
   the depth head.
6. **Cost of the fix.** PVE paired on the same people; recall on people whose head is visible ([Likely, untested]: the
   recall drop is a cohort effect from moving the detection cell to the head). Report root error next to PVE in the
   thesis.
7. **Pilot noise.** Set a seed, or repeat the key run, before reading any 2-epoch difference under ~10 mm.
8. **People very close to the camera.** Count misses (4 of 12 rendered images, both models).
9. **Children:** `MAX_ITER=10000 RUN=long_balanced sbatch --time=05:00:00 slurm/experiments/exp2_children.sh`.
   **Fingers:** train on larger hand crops. If neither moves, both become thesis limitations.
10. **Docs and thesis.** Fix the stale lines in section 7; thesis draft 74 -> 70.5 mm (V1).
11. **Lower priority:** v10 shape drift (only epochs 290-293 traced; trace e.g. 50/100/200 if they exist); students'
    depth bug (not measured); output-neutral fix of the 6D init (only matters for new heads; plan :680); geodesic loss
    with `--amp 1` untested; why crosscountry_00 counts as not visible; test-split limb ratios and lost head turn;
    repeat the overfit test with the depth fix; Anny age calibration.

---

## 6. Corrections log: claims that turned out wrong (do not repeat)

**Depth**
- "Depth error 1.08 -> 0.56 / 0.39 / 0.34 m." Those are signed means (`root dz`). Use absolute root error: 1.21 ->
  0.54 m (x5, 10 ep).
- "Bodies ~13% too small after the x5 10-epoch run." 0.87 was the x20 2-epoch size; x5 10-epoch is 0.84 (~16% small).
- "x20 is the best candidate." It ties x5 (0.53 vs 0.54 m).
- "The fix costs no measurable PVE." Every fix run is worse than its control (+2.6 to +4.9 mm, -1.6 to -2.4 recall).
- "v5 size ratio 0.67." The trace's value is 0.63.
- "Thesis numbers looked fine while the overlays looked wrong." The real-photo overlays are placed from the detection
  box and never showed the depth bug; only the synthetic demo and trace panels do.
- "Only the V1 column of the trace panels shows the model's depth." All three prediction columns show the shrink.

**Neck**
- "Weak per-bone supervision on a redundant 4-bone chain" and "the long-neck look is mostly the depth defect."
  Refuted (3.2).
- "Neck: no bug found." Superseded by the measured half-turn flip.
- "The neck offset is bookkeeping (mask or convention)." Refuted by the code audit and H3/H4.
- "The 6D init explains the neck and v10" / "every bone starts with a random twist." The init gives a tilt about X; it
  fits metacarpal1.L only. Origin unknown.
- "The 6D outputs grew too large to move" (Adam scale argument). Refuted by review.
- "62.7 / 89.5 deg are per-bone errors." They are group means.
- "v10's neck is right because of 3DPW head-down poses" and "bulky torsos from 3DPW shape labels." Unsupported: 3DPW
  labels keep the head at rest; the deformed v10 arm is the flip mechanism.
- "v9 checkpoint at `logs/anny_model/anny_teacher_v9_mixed/checkpoints/00062.pt`." Real path:
  `models/multiHMR/anny_teacher_v9_mixed_00062.pt` (and `_00149.pt`).

**Fingers**
- "Fingers need a hand-crop branch." Not shown; only test-time zoom was tested.
- "The index-base flip is part of the finger gap." It adds ~4.7 deg to both the predicted and the rest-pose numbers.
- "Fingers are under-trained / a supervision or capacity problem." A guess.

**Children**
- "Age 0.3 ~ 11 y", "0.55 ~ 20 y", "0.5 ~ 18 y". These use the code's assumed anchors; Anny's calibration gives ~7,
  ~13.6 and ~12.7 y.
- "11% children in batches." Measured 9.7-11.1% depending on the run; 9.9% in the measurement job 3571601.

**3DPW labels**
- "48% of two-person images miss an actor." Measured 42% (407/960).
- "2 of 12 validation sequences unlabelled / 28% smaller." One not built, one dropped by visibility; 31% of train
  labels missing.
- "The fit error was never measured." Now 18.7 mm.
- "The 3DPW labels are accurate." Say "pass": the nearest-point surface test is a lower bound and blind to gender.
- "OVERALL FAIL is from limb ratios and 3.2% outliers." FAIL is from limb ratios (test split) and from 5.2% outliers in
  the train split; the 3.2% overall is only WARN.

**Still untested [Likely] claims; keep the tag**
- "The label builder ran before all fits finished" (or before images were extracted).
- "The recall drop is a cohort artefact."

---

## 7. Stale docs: outdated lines elsewhere in the repo

Trust this file over these. Fix them when there is time (one commit per doc).

**`README.md`**
- :114 "numbers look normal while the overlays look wrong" (real-photo demos use box placement).
- :115 whole neck row (weak supervision, redundant chain, long neck = depth defect).
- :116 "Under-trained" (a guess; missing the crop/region-norm nulls and the metacarpal1.L flip).
- :117 "about 11 years", "0.55 (about 20 years)" (code anchors); exp2 null missing.
- :119 "No improvement is claimed until its holdout trace has been re-run" (it has).
- :126 "Future work (not measured)" (both measured, both null). :128 "Nothing from them has been measured yet."
- :211-214 fit description omits the reversed gender and the per-person stride.
- :227 "4,451 train / 2,458 validation" without the expected 6,437 / 2,777 and the pending rebuild.
- :312-325 3DPW-val column needs the converted-label caveat. :331 v4_restpose reading ignores the neck flip / girth 0.46.
- :36-80 layout omits `bone_fix.py`, `scripts/diagnostics`, `scripts/experiments`, `slurm/diagnostics`,
  `slurm/experiments`, `docs/deck_src`. :243 "x4" is effectively x2.41 (boost renormalisation).

**`docs/diagnostics_findings.md`**
- :14-16 neck/fingers origins. :77 size 0.65/0.67 (trace: 0.63). :83-84 "overlays look wrong".
- :118-146 "minimal fix" and one-batch check: done; the fix is the flag; x5 and x20 measured.
- :184 "Origin: supervision". :186-188 "no long neck in the parameters" (neck02 half-turn). :189-193 redundant chain.
- :196-197 "[Likely] long-neck look from the depth defect" (refuted). :201 "under-trained".
- :216-217 "1/3 ~ 11 y", "0.55 ~ 20 y". :251 "root depth error 1.08 m" (signed; absolute 1.21 m).
- :270-280 "What to do before Friday" (all stale). No neck-flip section, no 3DPW check.

**`docs/experiments_plan.md`**
- :8-9 and :740-746 "Nothing in this plan has been run."
- :27 exp5 as WIP `bd7cfa3` (merged in `637dc5c`); ancestors "not yet traced" (traced).
- :28 and :428 exp6 "uncommitted" (it is in `d902e00` / `81dbdcb`).
- :94, :735 size baseline 0.67 (trace: 0.63). :150-152 exp3 redundant-chain hypothesis (refuted).
- :229 heading "WIP bd7cfa3". :235 "per-bone split never measured". :241-250 H3/H4 "untested" (refuted).
- :252 "Leading hypothesis [Likely]" (flip now measured). :273-278 6D-init sentence only half corrected.
- :368-370 "neither the flip nor the L1 trap measured". :422-426 decpose surgery (superseded by exp6).
- :258, :569 cite scratch files (`flip_joints.py`, `test_bone_fix.py`) that are not in the repo.
- :680 line refs from an older commit (6D init now at `model.py:641`). :688-707 week schedule is history.
- No results section for exp1-exp5 (use section 3 here).

**`docs/experiments/exp2_children.md`**
- :3-5 "no number here comes from running the experiment" (pilot ran, null). :18 "0.5 = about 18 y". :104-105 age-anchor
  question answered (Anny calibration).

---

## 8. History since 14 October

**Commits on `main` after `586faad`** (the old handover):

| commit | what |
|---|---|
| cbe86b4 | first supervisor deck (12 slides) and speaker notes |
| f6edb05 | deck: corrected neck slide, post-pilot numbers, depth before/after figures |
| a93af19 | deck: absolute root error, run-correct size ratios, fix-vs-control cost |
| ac17023 | deck: final checked 15 slides + notes |
| bd7cfa3 | WIP neck: per-bone dump, exp5 scripts, `--rotmat_loss` candidate |
| 637dc5c | neck: `--dump_bones`, geodesic `--rotmat_loss` (default l1), exp5 pilot |
| fed4ca5 | 3DPW conversion audit slides, `check_3dpw_conversion.py/.sh`, `check_anny_fit.py` update |
| b395a10 | `run_build_labels.sh`: FITS, IMAGES, OUT overridable |
| d902e00 | bone fix code: `bone_fix.py`, model/train/demo wiring, `compute_bone_fix.py`, exp6 script and summary |
| 81dbdcb | bone fix final (2-line summary docstring; carries the feature commit message) |
| 64266dc | deck slide 6 = measured conversion check; slides 1, 5, 14-16 and notes updated (17 slides) |

Not committed yet at the time of writing: this file and `docs/deck_src/`.

**Job index** (log = `logs/<jobname>_<jobid>.out` on the cluster):

| job | log name | what |
|---|---|---|
| 3570222 | diag_holdout | v5 00099 trace: headline V1 70.5 |
| 3571596 | diag_holdout | v5 re-trace: root 1.21 m, size 0.63 |
| 3571794 | diag_holdout | v10 00290 trace |
| 3571963, 3571964 | diag_holdout | v9 epoch 62, epoch 149 |
| 3571598, 3571599 | exp1_depth_anchor | 2-ep fix x1, 2-ep control |
| 3572418, 3572483 | exp1_depth_anchor | 2-ep fix x5, 2-ep fix x20 |
| 3571933, 3571934, 3572482 | exp1_depth_anchor | 10-ep fix x1, 10-ep control, 10-ep fix x5 |
| 3571601 | exp2_children | earlier sampler run (`pilot_ab1_...`); its batch-age measurement is cited |
| 3571955, 3571956 | exp2_children | sampler pilot, control |
| 3572419, 3572420, 3569949 | diag_cropK | real-photo ages: exp2 pilot, control, v5 |
| 3571957, 3571958 | exp3_neck_global_rot | pilot, control |
| 3571959, 3571960 | exp4_fingers | region-norm pilot; control + crop trace + hand size |
| 3573518, 3573519 | diag_neck_bones | per-bone dumps: v5, v9 e149, v10, long_pc_dist5; lineage v2/v3/v4 |
| 3573520, 3573521 | exp5_neck_fix | L1 control, geodesic |
| 3575369 | check_3dpw_conversion | 3DPW->Anny label check |

**Experiments, one line each:** exp1 depth anchor (placement much better, slope not, likely cost; rule failed);
exp2 children sampler (null); exp3 global head loss (null); exp4 fingers crop + region-norm (both null); exp5
geodesic loss (flip did not move); exp6 bone correction (running).

---

## 9. Deck

**Files:** `docs/supervisor_update_2026-10-16.pptx` (17 slides), `docs/supervisor_talk_notes_2026-10-16.md` (spoken
part + "If asked:" part per slide), `docs/figures/depth_before_after_579860.jpg` and `_579862.jpg`.

**Source:** `docs/deck_src/build.js` (pptxgenjs). Read `docs/deck_src/README.md` first: two absolute paths in
`build.js` (a theme helper and the figure folder) point to the old build machine and must be changed before it runs.
Notes are generated from `build.js`; never edit the `.md` alone. For one-word changes, edit the pptx in PowerPoint.

**Before the meeting:** open the pptx once in PowerPoint and check slides 5, 6 and 16 for text that wraps one line
longer (they were only rendered with LibreOffice).

**Slides:**
1. Multi-HMR with Anny: what this week found (agenda).
2. Where we started: three visible failures (neck, fingers, children).
3. What I did: test each stage on its own (holdout + manifest).
4. What is not the cause: the structural checks pass (+ overfit warnings).
5. 3DPW to Anny: how the labels are made.
6. 3DPW labels: joints and surface pass, three defects.
7. Neck: two bones turned half a turn.
8. Fingers: test-time zoom and a loss rebalance do not help.
9. Children: child-like in training, adult on real photos.
10. The hidden bug: head anchored, pelvis targeted.
11. Depth fix: bodies drawn much closer to true size (figure 579862).
12. Depth fix: much better size, near vs far still weak (table).
13. Four experiments: results (+ noise floor).
14. Corrections made, and still to make.
15. Still open, and what I test next (7 items).
16. What I claim today, and what I do not (+ the question to the supervisor).
17. Backup: before and after, holdout image 579860.

**The question on slide 16:** fine-tune v9 now (fixed targets, depth loss x5), or first fix the 3DPW labels, re-fit,
then fine-tune? The student proposes the second. Record the supervisor's answer here after the meeting.

---

## 10. Working with the student

**Answer style** (standing preferences):
- Advisor voice. The first sentence challenges an assumption or names what is missing. Never open with agreement.
- Tag claims [Certain] / [Likely] / [Guessing].
- No flattery: never "Great question", "You're absolutely right", "That makes a lot of sense", "Absolutely",
  "Definitely".
- Disagree with structure: "I disagree because... Here's what I'd do instead... The risk in your approach is...".
- Uncomfortable answer first, no warm-up. Do not fold on pushback without new information.
- Simple, short wording. The student pastes commands directly, so they must run as written.

**Practical lessons:**
- Repo on the cluster: `/netscratch/najib/multi-hmr`. Always `git pull` first, then `sbatch` from the repo root.
  Logs: `/netscratch/najib/multi-hmr/logs`.
- **Never give commands with `<placeholders>`.** On 14 Oct the student pasted `CKPT=<full path ...>` literally. Write
  out real paths.
- **Never put a destructive git command in the same copy-paste block as other commands.** Once a `git status` and a
  `git reset --hard` were pasted together and the reset wiped uncommitted cluster edits.
- The student sends logs in batches (up to 5 files per message) and then says "done". Acknowledge each batch in one
  line; analyse only after "done".
- Always pass `TAG=` to diagnostic jobs. Give each experiment its own `RUN=` name.
- Read every pilot against its own control arm, never only against the v5 baseline (restart cost ~2.4 mm).
- Before calling a 2-epoch difference real, compare it with the noise floor (3.6).
- Put key numbers into the docs: the next assistant will not see the logs unless the student uploads them.
- After the meeting: write the supervisor's answers into section 9 of this file.
