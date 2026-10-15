# Handover: project state on Thursday 15 October 2026

This file replaces the chat history. It is written for an assistant that cannot see that chat, the cluster or any
temporary files. Everything you need is either here or in this repository. **Where this file disagrees with
`README.md`, `docs/diagnostics_findings.md`, `docs/experiments_plan.md` or `docs/experiments/exp2_children.md`, trust
this file** (section 7 lists the outdated lines). The speaker notes `docs/supervisor_talk_notes_2026-10-16.md` are
current except for known stale lines: slide 13 ("The slope did not move toward 1"), slides 6, 15 and 16 ("the rebuild
is running", never confirmed) and slide 16 ("about one hour of training", "77.0 against 70.5"). Section 9 gives the
replacements, and which slides change when a pending result arrives.

**Which version you have.** This file, `docs/deck_src/` and all code are on GitHub `main` (merged on 15 Oct, after
`64266dc`). To check a copy, run `grep -c '^## 0a\. Tomorrow' docs/handover.md`: it prints `1` for this version and `0`
for every older one (the 14-Oct `586faad`, or the work-in-progress `c6419ce`). On `0` you are reading an old copy: ask
the student for the right file. The student should attach this file to the chat directly, not rely on you browsing
GitHub. Download link: section 8.

**Meeting date: open.** Planned for **Friday 16 October 2026**, with the deck `docs/supervisor_update_2026-10-16.pptx`.
On 15 Oct (16:07) the student asked for a draft message asking the supervisor to move it to **Monday 19 October**.
Whether it was sent is unknown. An earlier message to the supervisor (content unknown) had no reply. Ask which date
holds before planning: it decides whether the pending results (section 4) go into the deck first.

**Citation key** (used from section 1 on):
- `3572482:126` = line 126 of that Slurm job's `.out` log. Job index in section 8. The student has every log; ask for
  one when you need to check a number.
- `README :L` = `README.md`; `notes :L` = `docs/supervisor_talk_notes_2026-10-16.md`; `plan :L` =
  `docs/experiments_plan.md`; `findings` = `docs/diagnostics_findings.md`.
- `file.py:L` = that file in the repo. All line numbers are as of `64266dc`.

---

## 0. Paste this first

The student pastes this block, section 0a and section 0b into the first message as text, then attaches this file.
[Guessing] ChatGPT may only search a large attachment (this one is ~90 KB) instead of reading all of it; text in the
message itself is sure to be read.

```
You are helping me finish my master thesis. Act as a critical advisor, not a cheerleader.

Project in brief:
- Multi-HMR (multi-person whole-body mesh recovery; ViT-L DINOv2 backbone at 672 px + cross-attention head),
  moved from SMPL-X to the Anny body model (163 bones, 11 shape values in [0,1]).
- Trained on AnnyOne (~580k synthetic images) and on 3DPW converted from SMPL to Anny by my own fitting script.
- Main model v5 (epoch 99): 70.5 mm PVE on a fixed 100-image synthetic holdout (setting "V1").
- Depth-target bug: the fix halves the absolute root error (1.21 -> 0.54 m), but the depth slope stays 0.30-0.44
  (1 = right) and bodies are still drawn ~16% too small. It likely costs 2.6-4.9 mm PVE.
- Neck: two bones are turned half a turn in every v5-line model. A correction, exp6, is merged; its result is not in.
- 3DPW labels: three defects (reversed gender, a missing actor in two-person frames, missing labels).
- Fingers and children: tested, nothing helped yet.
Supervisor meeting: planned Fri 16 Oct; I may have asked to move it to Mon 19 Oct. Ask me which date holds.
Thesis: my 13 Oct brief said a one-week deadline. Ask me the submission date and the chapter list of my draft
before planning anything that takes more than a day.
Possibly submitted 14-15 Oct, none confirmed: x20 depth run (14 Oct), 3DPW label rebuild and exp6 (15 Oct).
Ask me to run the status block from handover section 4 (squeue alone misses jobs that already finished).

You cannot see my repo or the cluster. The repo is public:
github.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset
Ask me to paste files, lines and logs. I run jobs on a Slurm cluster, upload logs in batches, and say "done".

Priorities: fix the model before editing slides or docs. Every change goes behind a default-off flag. Never
overwrite production checkpoints or threedpw_anny_full/labels.

How to answer:
- First sentence: challenge my assumption or name what is missing. No warm-up.
- Tag every claim [Certain], [Likely] or [Guessing].
- No praise phrases ("Great question", "You're absolutely right", "Absolutely", "Definitely").
- Uncomfortable answer first. Simple, short words.
- To disagree: "I disagree because ... Here's what I'd do instead ... The risk is ...".
  Do not give in to pushback without new evidence.
- Commands: complete and copy-pasteable, run from /netscratch/najib/multi-hmr. Never use <placeholders>.
  Never put a destructive git command (reset, checkout --, clean, push -f) in the same block as other commands.
- Code changes: first ask me to paste the current lines (example: sed -n '1000,1012p' scripts/data/smpl_to_anny.py).
  Then give ONE block that makes the change and checks it. Commit and push to main in a SEPARATE block, after I
  confirm the check. No pull requests. Handover section 2 shows the exact pattern.

Before answering anything, read the whole handover.md I attached; if you can only search it, search it before each
answer. It overrides the other docs in the repo.
```

## 0a. Tomorrow, in order

Paste this section with section 0. Each step names the section with the details.

1. Ask the student: does the meeting hold on Friday 16 Oct or Monday 19 Oct (top of file)? What is the thesis
   submission date, and which chapters does the draft have (section 10)?
2. Get the job status with the status block (section 4, top). Resubmit a job only if both `sacct` and its output
   folder show nothing.
3. Read what arrived: exp6 (4a, then its three-case table), label rebuild (4c), x20 depth run (4d). Ask for the `.out`
   files; analyse only after the student says "done".

If the meeting is on **Friday**:

4. Hand-edit the pptx in PowerPoint, using the table in section 9. First the lines that are stale whatever arrives
   (slides 6, 15, 16: "rebuild running"; slide 13 notes: the slope; slide 16 notes: "one hour", "77.0"). Then one row
   per result that arrived.
5. Open the pptx in PowerPoint and check slides 5, 6 and 16 for text that wraps one line longer.
6. After the meeting: write the supervisor's answer to the slide-16 question into section 9.

If the meeting is on **Monday**:

4. Start the 3DPW re-fit first, because it needs ~5-6 h of wall time (not measured): three code changes, then the
   commands in 4e ("3DPW re-fit chain"; NEEDS CODE).
5. If exp6 is case (a) or (b) of its table: real photos with the fix (section 5, item 2; NEEDS CODE).
6. Before the meeting: Friday steps 4-5 (deck edits, PowerPoint check).

## 0b. Numbers at a glance

V1 setting, 100-image AnnyOne holdout, matched people, unless noted. Sources in section 3.

- v5 epoch 99: PVE 70.5 mm, PA-PVE 53.3, recall 90.2%. Root error 1.21 m, size ratio 0.63, depth slope 0.33.
- Depth fix: root error 0.53-0.73 m (controls 1.11-1.44 m), size 0.78-0.87. Slope 0.30 with the switch alone and
  0.38-0.44 with the stronger depth loss, far from 1. Best runs: x5 for 10 epochs 0.54 m / 0.84 / 0.38; x20 for 2
  epochs 0.53 m / 0.87 / 0.43.
- Cost of the fix vs its control: PVE +2.6 to +4.9 mm, recall -1.6 to -2.4 points (each inside pilot noise, all the
  same direction).
- Pilot noise: two identical configs differ by 7.6 mm PVE and 1.9 deg global head. One pilot resolves ~10 mm, ~3 deg,
  ~2 recall points.
- Neck, v5 (all eight v5-line checkpoints alike): neck02 175.6 deg (half-turn), head 177.9 deg, global head only
  12 deg, neck girth pred/GT 0.46, PVE_neck 27.3 mm. v10: girth 1.00, PVE_neck 10.3 mm. exp5 geodesic loss: neck02
  175.6 -> 175.0.
- Other half-turns: v5 metacarpal1.L 180 deg; v10 shoulder01.L 166, upperarm02.L 180, metacarpal2.L/2.R/3.R 180.
- Fingers: predicted 17.3 deg vs 15.1 at rest pose; test-time crop 17.2; region-norm loss 17.5 vs 17.4 (control).
- Children: sampler raises children per batch 2.1% -> 9.7-11%; real photos stay at age 0.57-0.67 either way.
- 3DPW labels: body joints 18.7 mm median (p95 24.5); surface 18.9 / 18.3 mm (lower bound). Gender reversed (men:
  19.2 -> 15.3 mm when flipped); 407 of 960 two-person images miss an actor; train 4,451 of 6,437 images labelled,
  validation 2,458 of 2,777.
- 3DPW-test PVE, against our converted labels only (not comparable with published numbers): v5 119.1, v9 102.5,
  v10 111.5.

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
  4,451 train / 2,458 validation / 9,722 test images (README :227), every third frame (README :212). Mixed into
  v9/v10 batches at 30%.

**Models** (checkpoints are not in git; cluster paths below).

| name | what | AnnyOne holdout V1 PVE (job:line) | 3DPW-test PVE / PA-PVE* |
|---|---|---|---|
| v5 = `anny_s2_shape_v5` epoch 99 | ViT-L, synthetic only. Reference for every diagnostic | **70.5** (3570222:40) | 119.1 / 68.8 |
| v9 = `anny_teacher_v9_mixed` epochs 62, 149 | v5 + 30% 3DPW-train. Teacher of the final student | 75.0 / 77.0 (3571963:40, 3571964:40) | 102.5 / 62.7 |
| v10 = `anny_v10_scratch_mixed` epoch 290 | ViT-L. Despite "scratch": DINOv2 backbone from NAVER's `multiHMR_672_L_anny` + fresh heads (`train_v10_scratch_mixed.sh:3-8`), last 4 blocks trainable, 30% 3DPW from the start | 74.1 (3571794:40) | 111.5 / 76.2 |
| students v6c, v8 final | ViT-S distilled | not traced | 139.5 / 88.0, 123.6 / 81.2 |

\* Quoted in the header of `slurm/demo/run_demo_compare.sh:5-9`, which names neither the method nor the epochs.
[Likely] they come from `slurm/eval/run_eval_3dpw_test.sh`: `train.py --eval_only 1` (`evaluate()`, V0, unmasked) on
"all 9,722 converted images, every third frame of the 24 test sequences" (`:11-12`); its model list has v9 00062 and
v10 00299 (`:36`, `:40`), as does `run_demo_compare.sh:93-98`. No log of that run is in the chat. The scores are
against **our converted labels**, so they are not comparable with published 3DPW numbers, and the labels have known
defects (section 3.5). Say "best on our 3DPW-Anny labels", never "best model".

**Exact checkpoint paths** (relative to `/netscratch/najib/multi-hmr`; prefix that for `CKPT=`):
- v5 00099: `logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt` (copy: `models/multiHMR/anny_s2_shape_v5_00099.pt`).
- v9: `models/multiHMR/anny_teacher_v9_mixed_00062.pt` and `models/multiHMR/anny_teacher_v9_mixed_00149.pt`.
- v10 00290: `logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt`.
- v5 lineage: `models/multiHMR/anny_s2_frozen_bb_v2_00299.pt`, `logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt`,
  `logs/anny_model/anny_s2_boosted_v4/checkpoints/00299.pt` (paths from 3573519:1, :89, :177).
- NAVER's original (backbone source of frozen_bb_v2 and v10): `models/multiHMR/multiHMR_672_L_anny.pt`.
- Experiment runs: `results/experiments/<exp>/<RUN>/train/checkpoints/`, last epoch `00001.pt` (2 epochs) or
  `00009.pt` (10 epochs); only the last 2 are kept. Example: `results/experiments/exp1_depth_anchor/long_pc_dist5/train/checkpoints/00009.pt`.
  Run names are in section 3.1 and the job index (section 8).

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

**Evaluation settings** (`diag_holdout_trace.py:14-24`; the name each one has in the logs in brackets):
- V0 (`V0_eval_unmasked`): as `train.py evaluate()`, fp32, dataset camera, helper bones **unmasked**.
- V1 (`V1_eval_masked`): V0 + helper bones masked. **The headline setting.**
- V2 (`V2_demo`): as `demo.py` on synthetic images (fp16, camera from FOV, masked).
- V3 (`V3_demo_relaxhands`): V2 + `--relax_hands 1` (fingers set to rest pose).
- V4 (`V4_person_crop`): V1 on a square crop around each GT person (`--person_crop 1`, exp4 only).

**Experiments** (one job script `slurm/experiments/expN_*.sh` and one output folder `results/experiments/expN_*/`
each): exp1 depth anchor, exp2 children sampler, exp3 global head loss, exp4 fingers (test-time crop and region-norm
loss), exp5 geodesic rotation loss, exp6 bone correction. Results in section 3, one-line verdicts in section 8.

**Headline.** v5, V1: PVE 70.5 mm, PA-PVE 53.3, recall 90.2% (3570222:40). The thesis draft still says 74 mm; that is
V0 (unmasked helper bones add +3.8 mm on the same people, README :321). Change the draft to 70.5. The thesis draft is
**not in the repo** (the student's own file): ask for the section to be pasted.

**What is not the cause** [Certain] (notes slide 4, `docs/diagnostics_findings.md`): GT parameters decode onto the
people; all 16 single-joint perturbations move the right body part; every active bone and shape output gets gradient
(two of three camera outputs get none); no joint-mapping or rotation-convention error. The overfit test on 40 images
lowers the error by 36%. [Guessing] its translation and 2D losses may have risen while memorising, an early sign of the
depth bug: first vs last 5 epochs give 1.67 -> 1.89 and 58 -> 75, but these are single-batch prints (one per epoch),
and first-half vs second-half averages fall instead (2.10 -> 1.96, 69 -> 63). Too noisy to use.

Every fine-tune skips some optimiser steps for NaN/Inf gradients: 1-7 per 2,000 in the 2-epoch runs (e.g. 3572418: 1,
3573520: 7) and 13-23 per 10,000 in the 10-epoch runs (unique `NaN/Inf GRADIENTS at iter` lines). Clipping absorbs it;
mention it in the thesis as a known issue.

**Terms you will meet in the logs.** Full legends: `scripts/diagnostics/README.md` (trace metrics),
`diag_holdout_trace.py:64-70` and `:436-445` (bone verdicts), `slurm/experiments/exp6_bone_fix.sh:24-25, :31-35`
(guard).
- **Three naming schemes.** V0-V4 = evaluation settings of the trace (above). v5/v9/v10 = models.
  `check_3dpw_conversion.py` has its own grade IDs V0-V5 and I1-I4 (e.g. its "V4c" = limb-length ratios); they are
  unrelated to the V0-V4 settings. This file writes them as "check V1", "check I1", ...
- **`--person_center head` vs `--gt_center_joint person_center`.** The first is the model's anchor: the joint each
  detection cell and predicted translation refer to (head, in every model). The second is the exp1 fix: it builds the
  training targets at that same anchor joint instead of at the root.
- **PLACEMENT** (log section of the trace): absolute errors, not pelvis-centred, one row per setting: `root mm`,
  `root dz m`, `head px`, `root px`, ... **SUMMARY** (last section of an experiment log, e.g.
  `=== exp1_depth_anchor SUMMARY ...`): PVE, recall and the other headline numbers, plus the script's own success rule.
- **pelvis px** (column in 3.1) = PLACEMENT's `root px`: 2D distance in pixels between predicted and true root (pelvis).
- **skeletal height** = head joint (skull base) to the mean ankle joint, about 25 cm less than stature
  (`scripts/diagnostics/README.md:32`).
- **trace panels** = the images the trace saves under `<output folder>/vis/`: GT next to the model's predictions, each
  drawn where the model places it in 3D. They show the depth bug; the real-photo demos do not.
- **restart cost** = what 2 more epochs of fine-tuning v5 cost with nothing changed: ~2.4 mm PVE (notes :61). This is
  why each pilot is read against its own control arm, not against v5.
- **matched people**: GT people paired with a detection by the greedy 2D matcher of `train.py`; every mean is over
  them. Recall = matched / GT people.
- **tokens**: 14-px patches of the 672-px input (48 x 48 grid). **HPH**: Multi-HMR's cross-attention head.
- Per-bone dump columns (one row per bone): `med`/`mean`/`p90` = local (parent-relative) error, deg; `|tw|`/`swing` =
  twist about the bone's own rest axis / tilt of that axis; `glob` = error of the bone's camera-frame orientation;
  `joint` = mm error at the bone's own head (moved only by bones above it); `child` = mm error at its child's head
  (the first joint its own rotation moves; "child error"); `prdSpr`/`gtSpr` = spread of predicted / GT rotations
  across people; `relSpr` = spread of the per-person error rotations around their mean (small = the same offset in
  everyone, so one constant correction can undo it); `relAng` = angle of that mean error.
- Verdicts: **FLIP_TWIST** = error > 150 deg in at least half the people while the bone axis moves < 30 deg (a
  half-turn about the bone itself). **FLIP_SWING** = such a flip that tilts the axis >= 30 deg. **PARTIAL_FLIP** =
  more than 150 deg in 10-50% of people. **OK** = median < 15 deg. **FIXED_OUTPUT** = predicted spread < 1/4 of GT
  spread. **CONST_OFFSET** = R_pred ~ R_gt C with one C (relSpr < 10). **SPREAD** = large, image-dependent error.
- **compensated chain**: a flipped bone whose half-turn a second flipped bone further down undoes, so the bones
  below point the right way (v5: neck02 -> head; v10: shoulder01.L -> upperarm02.L). Correcting only one of the pair
  breaks the bones in between.
- **girth** = neck-skin radius pred/GT. **PVE_neck** = vertex error over the 171 neck-skin vertices (skinned >= 50%
  to neck01-03), reported neck01-centred. **PVE_hn** = vertex error over head + neck vertices (neck01 joint to the
  crown). **neck%** = predicted / GT neck length, %.
- **GLOBAL GUARD** (`compute_bone_fix.py`): exit 4 = the correction is predicted to raise some bone's global error by
  more than 5 deg (a compensated chain); `ON_GLOBAL_RISE` decides (continue / stop / safe). Exit 3 = no bone
  qualifies for a correction; the job stops.
- **x4 boost**: `--boost_neck_weight 4` / `--boost_hand_weight 4` are renormalised over 163 bones, so "x4" is
  really x2.41 for the boosted bones and x0.60 for the rest (plan :681).
- **stride phase**: each 3DPW actor's every-third-frame grid starts at that actor's own first valid frame, so two
  actors can sit on different frames (0, 3, 6 vs 1, 4, 7).
- **`%8`** in `sbatch --array=0-86%8`: at most 8 array tasks run at once.
- **decpose surgery**: an earlier idea (plan :422-426) to edit the pose-decoder weights of a checkpoint to undo a
  flip. Superseded by exp6.
- **diag_cropK**: job and log name of `slurm/diagnostics/diag_crop_camera.sh` (real-photo crop-camera read).

---

## 2. Where things are

**Repository.** github.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset
(public), branch `main`, HEAD `64266dc` (15 Oct) plus, once merged, this file (see the top). The cluster copy is
updated with `git pull`. Raw files can be read at
`raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/`
followed by the path (e.g. `.../main/train.py`), if your tools can fetch URLs; otherwise ask the student to paste.

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
| `slurm/eval/` | `run_eval_3dpw_test.sh` (3DPW-test via `train.py --eval_only 1`, V0; [Likely] the source of the 3DPW-test column in section 1), `run_eval_3dpw.sh`, `run_eval_3dpw_teacher.sh`, `run_test.sh` |
| `slurm/history/` | finished training runs of the v5 line (`train_step2_frozen_bb.sh` = frozen_bb_v2, ...) |
| `slurm/demo/` | `run_demo.sh` (real photos; copies v5 00099 under the misleading name `anny_distill_vits_v2_00099`, :58-59), `run_demo_syn.sh` (AnnyOne images), `run_demo_compare.sh` (five models), `run_demo_3dpw.sh`, `run_demo_student.sh`. All hard-code their checkpoints; none passes `--bone_fix` |
| `configs/threedpw_tasks.txt` | the 87 fit tasks (split, sequence, person, frames), one per array index 0-86 |
| `scripts/diagnostics/README.md` | legend of the trace metrics (PVE_hn, neck%, skeletal height); its "not yet executed" (:9) is stale |
| `docs/` | this file, the deck, speaker notes, findings, plan, `deck_src/` (deck source) |
| `docs/cluster_commands.txt` | old scratch commands. **Stale**: uses `--pose_param`, which `train.py` does not have |
| `docs/3dpw_adapter_check.md` | first one-sequence 3DPW adapter check (downtown_bar_00). **Superseded** by job 3575369 |
| `results/experiments/<exp>/<RUN>/` | experiment outputs on the cluster. `.gitignore` lets `*.json` and `*.md` through, but none is committed (`git ls-files results/experiments results/diagnostics` is empty): GitHub has no summaries. Ask the student to paste `summary.json` |

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
copies in `models/multiHMR/<run>_<epoch>.pt`. v9: use the `models/multiHMR/` copies for both epochs (section 1).
Epoch 62 is not under `logs/anny_model/anny_teacher_v9_mixed/checkpoints/` (job 3571597 failed on that path); that
folder exists, but its contents were never listed. Job status: the status block at the top of section 4
(`squeue -u najib` lists only queued and running jobs).

**How work flows.** The student runs `cd /netscratch/najib/multi-hmr && git pull`, then `sbatch slurm/...` from the
repo root, then uploads the `.out` logs. Until 15 Oct the code changes came from an assistant that could push to
GitHub; you cannot (next paragraph).
- Throughput: ~12 images/s measured in most runs (11.7-12.9 at batch 4 with 4 blocks unfrozen; one slow node 6.7,
  job 3571598). A 2-epoch pilot trains in ~11-20 min, a 10-epoch run in ~1-1.7 h, plus evaluations; set `--time`
  with margin. The job-script headers still quote the old 6.7 images/s estimate.
- `TAG` (output folder name of a diagnostic) defaults to `<run>_<epoch>` for `logs/anny_model/<run>/checkpoints/`
  and to `<name>` for `models/multiHMR/<name>.pt`. For an experiment checkpoint
  (`results/experiments/<exp>/<RUN>/train/checkpoints/`), `diag_holdout.sh` and `diag_crop_camera.sh` derive
  `train_<epoch>`, so two runs overwrite each other: pass `TAG=` there, and when re-tracing the same checkpoint with
  other settings. `diag_neck_bones.sh`, exp5 and exp6 already use the run name.

**Working without repo access (you, from now on).** You cannot read or push the repo or see the cluster.
1. Before changing a file, ask the student to `git pull` and paste the current lines, e.g.
   `cd /netscratch/najib/multi-hmr && git pull && sed -n '36,42p' slurm/data/run_smpl_to_anny_task.sh`.
   Never edit from memory or from this file's line numbers alone.
2. Give each change as ONE copy-paste block, run in `/netscratch/najib/multi-hmr`, that makes the change and checks
   it. For an edit, a Python heredoc that replaces an exact old text and refuses if it does not match exactly once
   (safer than a hand-written diff). For a new file, a full-file heredoc (`cat > path <<'EOF'`). Worked example,
   the first change of the 3DPW re-fit (section 5, item 3; section 4e):
   ```
   cd /netscratch/najib/multi-hmr && python - <<'EOF'
   p = "slurm/data/run_smpl_to_anny_task.sh"
   s = open(p).read()
   old = 'OUT_ROOT="/netscratch/najib/multi-hmr/threedpw_anny_full"'
   new = 'OUT_ROOT="${OUT_ROOT:-/netscratch/najib/multi-hmr/threedpw_anny_full}"'
   assert s.count(old) == 1, "old text not found exactly once; nothing changed"
   open(p, "w").write(s.replace(old, new))
   print("changed", p)
   EOF
   bash -n slurm/data/run_smpl_to_anny_task.sh && git diff
   ```
   Checks: `bash -n` for shell scripts, `python -m py_compile file.py` for Python, then `git diff`.
3. After the student confirms the diff, commit and push in a SEPARATE block:
   ```
   cd /netscratch/najib/multi-hmr && git add slurm/data/run_smpl_to_anny_task.sh && git commit -m "run_smpl_to_anny_task.sh: OUT_ROOT overridable from the environment" && git push origin main
   ```
   Everything goes on `main`. No branches, no pull requests (the student does not use them). If `git pull` or
   `git push` reports a conflict, stop and ask for the output; never answer with `reset` or `checkout --`.
4. Keep the flag rule: new behaviour behind a flag whose default is the old behaviour; new outputs in new folders.

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
| control `pilot_root_dist1_transl1_anny_s2_shape_v5_00099` | 3571599:49,58,67-68 | 2 | x1 | 1.11 | 0.66 | 0.33 | 2.28 | 72.32 | 89.84 | 23.3 |
| fix `pilot_person_center_dist1_transl1_anny_s2_shape_v5_00099` | 3571598:66,75,84-85 | 2 | x1 | 0.69 | 0.78 | 0.30 | 1.93 | 77.25 | 87.80 | 37.3 |
| fix `pilot_person_center_dist5_transl5_anny_s2_shape_v5_00099` | 3572418:52,61,70-71 | 2 | x5 | 0.66 | 0.80 | 0.44 | 1.62 | 76.77 | 87.80 | 38.1 |
| fix `pilot_person_center_dist20_transl20_anny_s2_shape_v5_00099` | 3572483:57,66,75-76 | 2 | x20 | **0.53** | 0.87 | 0.43 | 1.44 | 75.66 | 87.40 | 31.4 |
| control `long_root` | 3571934:134,143,152-153 | 10 | x1 | 1.44 | 0.59 | 0.34 | 2.59 | 72.19 | 89.43 | 24.3 |
| fix `long_person_center` | 3571933:125,134,143-144 | 10 | x1 | 0.73 | 0.78 | 0.30 | 1.98 | 75.51 | 86.99 | 38.5 |
| fix `long_pc_dist5` | 3572482:126,135,144-145 | 10 | x5 | **0.54** | 0.84 | 0.38 | 1.59 | 74.77 | 87.80 | 33.1 |
| v9 epoch 62 | 3571963:40,55,64 | - | - | 1.39 | 0.59 | 0.31 | 2.60 | 75.0 | 88.6 | 25.1 |
| v9 epoch 149 | 3571964:40,55,64 | - | - | 1.44 | 0.57 | 0.29 | 2.68 | 77.0 | 89.4 | 25.5 |
| v10 epoch 290 | 3571794:40,55,64 | - | - | 1.49 | 0.57 | 0.28 | 2.71 | 74.1 | 89.0 | 24.0 |

Run directories: `results/experiments/exp1_depth_anchor/` + the run name (the 10-epoch ones are `long_root`,
`long_person_center`, `long_pc_dist5`). All runs fine-tune v5 00099. To print a control next to a new run, pass the
control's run directory, e.g. `CONTROL_RUN=results/experiments/exp1_depth_anchor/long_root`
(`exp1_depth_anchor.sh:74`, `:180-187`).

What this says:
- [Certain] The fix moves the root: 0.53-0.73 m in all five fix runs vs 1.11-1.44 m in both controls. Size goes from
  0.59-0.66 to 0.78-0.87. The head now lands within 7-10 px (v5: 64.5 px).
- [Certain] Near vs far is still not separated: slope 0.30 with the switch alone, 0.38-0.44 with a stronger depth loss.
  Predictions are pulled toward ~2.5-2.9 m; cause unknown (notes :53, :55). Holdout people are at 1.92 m on average
  (README :114).
- [Certain] By its own success rule exp1 **failed**: every run, controls included, prints
  `PVE <= 71.5 and recall >= 89.2: NO` (e.g. 3572482:150). The rule's slope clause prints `yes` for x5, x20,
  x5-10-epoch and even the 2-epoch control (3572418:76, 3572483:81, 3572482:150, 3571599:73), because it only asks
  for any move toward 1 with a smaller intercept. The slope itself moved only from 0.33 to 0.30-0.44, far from 1.
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

**Submitted?** `long_pc_dist20` (x20, 10 epochs): the command was given on 14 Oct (~19:07, "Run this tonight"); never
confirmed, no job ID, no log. Check with the status block (section 4). Command and reading in 4d.

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
  girth 0.46-0.47 (0.468 in the exp5 geodesic run, 3573521:61), PVE_neck 27.3-28.7 mm. It is there from the first
  stage (frozen backbone), before the helper mask, the x4 boost and `alpha_shape 20`.
- The depth-fix run `long_pc_dist5` keeps the flip and girth 0.46, so **depth is not the neck cause**.
- Same mechanism elsewhere (3573518):
  - v5 line: `metacarpal1.L` FLIP_SWING, constant 180 deg (left index finger base bent backwards; child error
    185-200 mm) (:31).
  - v10: `shoulder01.L` FLIP_TWIST 166.3 (not constant, relSpr 21.3) (:195); `upperarm01.L` 58 deg local but 169 deg
    global (:199); `upperarm02.L` FLIP_TWIST 180 (:200), which turns the chain back, so the arm is right again below it;
    `metacarpal2.L` 180, `metacarpal2.R` 180, `metacarpal3.R` 180, all uncompensated (:208, :212, :213). [Certain]
    for these measured flips. [Likely] they are the "deformed left arm" the student saw in v10 renders (no render
    links the two) and most of why v10's shoulder/arm group errors are 50 / 36 deg (3571794:47).

**Refuted** [Certain]:
- H4 "the GT blocks are not proper rotations": all are proper rotations (3573518:72).
- H3 "the checkpoint uses another pose parameterisation": `pose_param=None` (local-bone) in every checkpoint (:73).
- Mask, loss-weight, evaluation and 3DPW explanations: refuted by a code audit (`docs/experiments_plan.md:237-245`).
- "Weak supervision on a redundant chain": refuted by the flip and by v10 learning the neck with the same loss.
- "The thin neck comes from the depth bug", for the measured numbers: the depth-fix run `long_pc_dist5` keeps girth
  0.46 and PVE_neck ~28 mm (3573518:357-360). For the look: the real-photo overlays place each body from its detection
  box (`--anchor_to_box 1`), so depth cannot shape them, yet they show the neck too (judged by eye, not measured).
- exp3 (global head loss, 2 epochs, job 3571957): global head 11.3 deg vs 11.7 / 13.6 for two identical controls
  (3571957:98; 3571960:318; 3571958:99): inside noise. Neck bones stayed at 62.5 (3571957:100).
- exp5 (2 epochs, lr 1e-5): the geodesic loss did not move the flip: neck02 175.6 -> 175.0, head 177.9 -> 177.1; the L1
  control stayed at 175.7 / 177.9 (3573521:89-91; verdict "NO MOVEMENT", :103). [Likely] neck02 and head would have to
  turn back together; a partial move raises the loss.

**Not refuted, not shown:** that the visible long/thin-neck look in the overlays IS this flip [Likely; render
pending, exp6 `compare/`].

**Origin: unknown.** The 6D init `[1,0,0,1,0,0]` (`model.py:641`) is singular under `rot6d_to_rotmat`
(`utils/humans.py:20-21`). It starts a bone turned about its local X axis, i.e. a tilt (FLIP_SWING). That fits
metacarpal1.L, **not** neck02's twist [Certain about the geometry; the origin of the neck flip is open].
[Certain] frozen_bb_v2 (the first v5-line stage) and v10 both started from the same pretrained backbone
(`multiHMR_672_L_anny`) with fresh heads. frozen_bb_v2 flipped neck02; v10 did not (it flipped arm and metacarpal bones
instead). The traced frozen_bb_v2 00299 was itself trained on from the run's own epoch 283 (saved args:
`pretrained=.../anny_s2_frozen_bb_v2/checkpoints/00283.pt`, 3573519:3). [Likely] frozen_bb_v2's recipe: fresh heads at
LR 1e-4 (`slurm/history/train_step2_frozen_bb.sh:104-127`), then a resume at LR 3e-5 with decay every 30 epochs
(`slurm/history/train_step2_resume.sh:98-123`). Only [Likely], because the history scripts were edited after the runs
(the resume script's comment still calls the LR "1e-4 UNCHANGED", `:89`). v10: LR 1e-5, last 4 backbone blocks
trainable (`slurm/train/train_v10_scratch_mixed.sh:52-86`). They also differ in backbone fully frozen vs partly
trainable, helper mask, `alpha_shape 20` and 3DPW from step 0. So "fresh heads" alone does not explain it, and the
recipe is not ruled out.

**Not yet done:** render the flip to confirm it is the thin/long-neck look seen in the overlays.

**exp6 (maybe submitted 15 Oct; no job ID or log; check with the status block in section 4)** (`--bone_fix`, code
in `d902e00`, final in `81dbdcb`). One constant rotation `C` per flipped bone,
estimated on 200 random training images (never the holdout) by `compute_bone_fix.py`, applied as `R <- R·C` right after
the 6D decoding. Then a zero-training test on the holdout and a 2-epoch fine-tune with the fix in place. Expected
[Likely]: neck02 -> ~5 deg, metacarpals -> ~0-1 deg, head -> ~15 deg (its half-turn compensates neck02 differently per
person, which one constant cannot undo), global head error rises. Commands, pass criteria and the three-case reading
in section 4a.

**Not implemented:** the two-sided correction `R_l <- K_l^T R_l K_l` (needed to keep a compensated chain such as
v10's left arm intact; plan :489, :561) and a chain-preserving head correction (plan :558; the expected head residual
it would address is explained at plan :534-551).

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

**Next:** lowest priority: a 10-epoch run with its own control and the real-photo read (section 5, item 9); if
nothing, a thesis limitation.

### 3.5 3DPW -> Anny labels

**How they are made** (notes slide 5). Valid frames, every third one **per person**; Anny is fitted per frame by
gradient descent so its 24 mapped joints land on the 24 SMPL joints (positions only; SMPL rotations and shape are not
used). Gender comes from 3DPW, age is fixed at 0.55, only height and proportions are free. One median body per person,
then a pose refit. Only the root and 18 bones move; 144 of 163 bones stay identity, including neck02, neck03, head and
the fingers. The label builder drops fits > 80 mm, people < 30% visible and boxes < 40 px.

**Measured** [Certain] (job 3575369, `scripts/diagnostics/check_3dpw_conversion.py`, ~2,000 sampled images):

| check | result | line |
|---|---|---|
| body joints vs 3DPW, at the fit points (bone head + virtual offset) | median 18.7 mm, p95 24.5 | :329 |
| same, train split alone | median 20.2 mm: grade WARN (PASS needs <= 20) | :187, :344 |
| bare bone heads (the GT joints `train.py` uses) | median 32.5 mm; mostly a fixed torso rig-definition gap (pelvis 110, spine2 77, collars 54-55 mm; limb joints equal the fit points) | :188, :246-269, :329 |
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
3. **Missing labels.** Train has 4,451 label images where the fits give 6,437 (:31). Six train sequences have none:
   `courtyard_arguing_00`, `courtyard_backpack_00`, `courtyard_basketball_00`, `courtyard_bodyScannerMotions_00`,
   `courtyard_box_00`, `courtyard_capoeira_00`; `courtyard_captureSelfies_00` is partial, 319 of 401 (:25, :63-70).
   Validation: `courtyard_basketball_01` not built (319 images; 2,458 of 2,777; :26, :33, :89). All 162 frames of
   `outdoors_crosscountry_00` are dropped by the < 30% visibility filter (:96); they are already outside the 2,777, and
   the rebuild will not recover them.

**By design, not bugs:** neck02/neck03/head at rest in 100% of labels; age 0.55; head turn kept only through neck01:
overall 9.5 vs 20.2 deg in SMPL (:223-224). Train and validation keep most of it (18.5 vs 21.2; 19.1 vs 19.3); test
loses most (6.3 vs 20.0; cause unknown).

**Script grade "OVERALL FAIL"** (:343-346). The IDs below are `check_3dpw_conversion.py`'s own grade IDs (V0-V5,
I1-I4), unrelated to the V0-V4 evaluation settings of section 1. It comes from: check V4c (limb-length ratios),
forearm 0.879, thigh 1.119 overall, failing only in the test split (train passes with 0.965 / 1.035) (:214-215, :335);
and check V4e (people with a joint over 100 mm), 5.2% in train = FAIL there, 3.2% overall = WARN (:218, :337). Cause
of the test-split limb ratios: unknown. All checked test people are men labelled 1, so a link to the gender defect is
possible, not shown (notes :27).

**Submitted?** label rebuild into `threedpw_anny_full/labels_rebuilt` (b395a10): maybe on 15 Oct; not confirmed, no job
ID, no log. Check with the status block (section 4).

**Next:** fix gender and use one shared frame grid for both actors, re-fit all 87 tasks into a **new** folder, rebuild,
re-check (section 5). Only then test anything on 3DPW.

### 3.6 Pilot noise (read every 2-epoch result against this)

The exp3 control (3571958) and exp4 control (3571960) are identical configs. They differ by **7.6 mm PVE** (76.3 vs
68.7; 3571958:104, 3571960:320), 1.9 deg global head (13.6 vs 11.7) and 0.8 recall points. Over six 2-epoch v5-recipe
arms: PVE 68.7-76.3, recall 88.6-90.2, global head 11.3-13.7 (`docs/experiments_plan.md:413-419`). So one pilot
resolves only ~10 mm PVE, ~3 deg, ~2 recall points. `train.py` sets no random seed (the trace's `--seed` only picks
which training images exp6 samples). Restarting a fine-tune alone costs ~2.4 mm (notes :61).

---

## 4. Pending jobs

All commands run on the cluster in `/netscratch/najib/multi-hmr`, after `git pull`; every block below starts with the
`cd`. On 15 Oct 15:39 the student said "since it's running" without job IDs. **Treat every job below as "maybe
submitted"** and check first.

**Status block.** `squeue -u najib` lists only queued and running jobs. exp6 (~30 min) and the rebuild (~1 min) were
maybe sent on 15 Oct, so by now they have probably finished and `squeue` shows nothing. This block also shows
finished jobs and their outputs (`;` so that a missing folder does not stop the rest):
```
cd /netscratch/najib/multi-hmr; sacct -u najib -S 2026-10-14 --format=JobID,JobName%28,State,Elapsed; ls -lt logs | head -20; ls results/experiments/exp6_bone_fix/ results/experiments/exp1_depth_anchor/; ls threedpw_anny_full/labels_rebuilt | head
```
What to look for: job names `exp6-bone-fix`, `3dpw-labels` and `exp1-depth-anchor` in `sacct` (State `COMPLETED`,
`FAILED`, `TIMEOUT`, ...); run folder `anny_s2_shape_v5_00099_n200_s0` under `exp6_bone_fix/`; `long_pc_dist20` under
`exp1_depth_anchor/`; index files in `labels_rebuilt`. If `sacct` prints an error (accounting may be off on this
cluster [Guessing]), use the `ls` lines alone. **Resubmit a job only if both `sacct` and its output folder show nothing.** If the folder exists
but the job failed, ask for its `.out` and `.err` before resubmitting.

### 4a. exp6 on v5 (bone correction + 2-epoch fine-tune)

```
cd /netscratch/najib/multi-hmr && sbatch slurm/experiments/exp6_bone_fix.sh
```
- ~30 min (limit 1 h). Log `logs/exp6_bone_fix_<jobid>.out`. Run dir
  `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/` (`bone_fix.json`, `summary.json`, `compare/*.jpg`
  = top row without, bottom row with the fix).
- The final table has 4 columns: baseline | zero-training | fine-tuned | control (the exp5 L1 run).
- **Zero-training PASS** needs all of: neck02, head and metacarpal1.L each below 15 deg; girth >= 0.9; global head <=
  baseline + 3 deg; PVE <= baseline + 1 mm.
- **Fine-tuned PASS** needs: the same bone and girth conditions; PVE <= control + 8 mm; recall >= control - 2;
  global head <= control + 3 deg.
- **Expected** [Likely]: a "GLOBAL GUARD" warning (exit 4) in step a; the job continues. neck02 near 5 deg; head near
  15 deg, so both verdict lines may say FAIL on the head or global head even though the neck skin recovers. That is
  case (b) below, not a failure of the idea.

**Reading exp6: three cases.** Use the final table (columns `baseline | zero-training | fine-tuned | control`; rows
`neck02 *`, `head *`, `girth pred/GT`, `global head`, `PVE`, `recall`; `*` = corrected bone) and the two verdict
lines `ZERO-TRAINING FIX:` and `FINE-TUNED FIX:`. Judge on the fine-tuned column; on the zero-training column if
training did not run. A "NO TARGET" stop (exit 3: no bone qualified) counts as case (c).

| case | what the log shows | what to claim | next |
|---|---|---|---|
| (a) | `FINE-TUNED FIX: PASS`, or `ZERO-TRAINING FIX: PASS` with the fine-tuned line FAIL or not run | "A fixed correction at the network output removes the neck flip (neck02 175.6 -> X deg; head 177.9 -> Y deg) and the neck skin recovers (girth 0.46 -> Z), within the PVE / recall / global-head guards." Fine-tuned PASS: add "fine-tuning keeps it removed". Zero-training PASS only: say "without training" and name what the fine-tune failed on | keep the fix (it is saved in the fine-tuned checkpoint args); look at `compare/`; real photos (section 5, item 2) |
| (b) | `neck02 *` < 15 and `girth pred/GT` >= 0.9, but the verdict line says FAIL on `head` (>= 15), on `global head` (> ref + 3) or on PVE / recall | "The neck skin recovers with a fixed output correction (girth 0.46 -> Z, neck02 175.6 -> X deg); the head keeps a ~Y deg residual (global head A -> B deg)." Name any PVE / recall cost. Do **not** claim a neck fix | the expected outcome [Likely]. Next: the chain-preserving head correction (plan :558; NEEDS CODE in `bone_fix.py`); look at `compare/` |
| (c) | `neck02 *` still >= 15, or `girth pred/GT` < 0.9 | neck02 >= 15: "a constant correction does not undo the flip on unseen images". neck02 < 15 but girth < 0.9: "neck02 is turned back, but the neck stays thin: the thinning has another cause" (plan :560) | the origin stays open; look at `compare/` and the shape outputs |

Fill X, Y, Z, A, B from the table; never write the letters into a slide.

### 4b. exp6 on v10 (three metacarpals, no training)

```
cd /netscratch/najib/multi-hmr && CKPT=logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt TRAIN=0 BONES=metacarpal2.L,metacarpal2.R,metacarpal3.R sbatch --time=00:30:00 slurm/experiments/exp6_bone_fix.sh
```
- ~10 min. Expected: the three metacarpals -> ~0-1 deg, PVE and global head about unchanged, zero-training PASS.
- Do **not** use `BONES=auto` on v10: it would also correct shoulder01.L and upperarm02.L, break the compensated
  left-arm chain, and the guard would stop the job.

Re-print a finished table without a GPU (login node):
```
cd /netscratch/najib/multi-hmr && SUMMARY_ONLY=1 RUN=anny_s2_shape_v5_00099_n200_s0 bash slurm/experiments/exp6_bone_fix.sh
SUMMARY_ONLY=1 RUN=anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R bash slurm/experiments/exp6_bone_fix.sh
```

### 4c. 3DPW label rebuild (no new fit)

```
cd /netscratch/najib/multi-hmr && OUT=/netscratch/najib/multi-hmr/threedpw_anny_full/labels_rebuilt sbatch slurm/data/run_build_labels.sh
```
- ~1 min of CPU (limit 30 min). Log `logs/3dpw-labels-<jobid>.out`. Writes to a new folder, so the old labels that
  v9/v10 trained on stay intact.
- Read the per-sequence lines (e.g. `[train] courtyard_arguing_00 ... N images`), the per-split totals
  `[split] N images, M people` and `TOTAL`. Expected [Likely]: train ~6,437 (was 4,451), validation ~2,777 (was
  2,458; +319 from `courtyard_basketball_01`; crosscountry is already excluded from the 2,777 and stays at 0), test
  9,722 unchanged (3575369:31, :33, :35). The "unconverted tasks" list at the end should be empty.
- The builder silently skips a frame whose image file is missing (`build_threedpw_anny_labels.py:123-125`). If a train
  sequence still shows 0 images, its images are probably missing under `data/3DPW/imageFiles/<seq>/`.
- Then re-check the new labels (my suggestion; not run yet):
```
cd /netscratch/najib/multi-hmr && LABELS=threedpw_anny_full/labels_rebuilt OUT=results/diagnostics/check_3dpw_conversion_rebuilt sbatch slurm/diagnostics/check_3dpw_conversion.sh
```
- Training scripts hard-code the old folder (`slurm/train/train_teacher_v9_mixed.sh:55`,
  `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`). Do not overwrite `labels/`.
- The rebuild does **not** fix gender or the missing actor; both are decided inside the fit.

### 4d. Depth x20 for 10 epochs (command given 14 Oct; never confirmed)

```
cd /netscratch/najib/multi-hmr && ALPHA_DIST=20 ALPHA_TRANSL=20 MAX_ITER=10000 RUN=long_pc_dist20 CONTROL_RUN=results/experiments/exp1_depth_anchor/long_root sbatch --time=04:00:00 slurm/experiments/exp1_depth_anchor.sh
```
- ~1-2 h. Log `logs/exp1_depth_anchor_<jobid>.out`. `CONTROL_RUN` prints `long_root` (x1, 10 epochs) side by side; it
  is not a same-weight control. Once `long_root_dist20` (4e) exists, that is the matching control:
  `CONTROL_RUN=results/experiments/exp1_depth_anchor/long_root_dist20`.
- Read, in this order: PLACEMENT row `V1_eval_masked`, column `root mm` (never `root dz m`); then depth slope and size
  ratio; then PVE and recall in the SUMMARY.
- Compare with `long_pc_dist5`: 544.5 mm, size 0.84, slope 0.38, PVE 74.77, recall 87.80 (3572482:126-145), and with
  `long_root`: 1438.8 mm, 0.59, 0.34, 72.19, 89.43 (3571934). There is no control at x20.
- Use x20 in a v9 fine-tune only if it clearly beats x5 on root error and size with PVE and recall no worse.
  Otherwise keep x5.

### 4e. Command cookbook (not pending; for the next steps)

All from `/netscratch/najib/multi-hmr` after `git pull`. **NEEDS CODE** = a change must be made first (section 2,
"Working without repo access"); the file and function are named.

Trace any checkpoint (V0-V3, placement, per-group errors; ~10 min; log `logs/diag_holdout_<jobid>.out`, output
`results/diagnostics/holdout_<TAG>/`):
```
cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt sbatch slurm/diagnostics/diag_holdout.sh
```
An experiment checkpoint needs its own `TAG` (section 2):
```
cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/results/experiments/exp1_depth_anchor/long_pc_dist5/train/checkpoints/00009.pt TAG=long_pc_dist5_00009 sbatch slurm/diagnostics/diag_holdout.sh
```
Per-bone dump (verdicts, girth, PVE_neck; ~3-5 min per checkpoint; log `logs/diag_neck_bones_<jobid>.out`):
```
cd /netscratch/najib/multi-hmr && CKPTS="models/multiHMR/anny_teacher_v9_mixed_00062.pt logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt" sbatch slurm/diagnostics/diag_neck_bones.sh
```
Real-photo shape and age read (`example_data`; log `logs/diag_cropK_<jobid>.out`; ages per person on the
`age` lines):
```
cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt sbatch slurm/diagnostics/diag_crop_camera.sh
```
Depth: the missing control at x20 (same schedule as `long_pc_dist20`, old root targets):
```
cd /netscratch/najib/multi-hmr && GT_CENTER=root ALPHA_DIST=20 ALPHA_TRANSL=20 MAX_ITER=10000 RUN=long_root_dist20 sbatch --time=04:00:00 slurm/experiments/exp1_depth_anchor.sh
```
exp6 on v9 (zero training; only if the v5 result is a PASS and a v9 fine-tune is planned; the head guard is expected
to fire, so let it continue; ~20-30 min, not measured: v9 00062 has no stored no-fix dump, so the job traces one;
never run, the env vars are the ones documented in the script header):
```
cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt TRAIN=0 ON_GLOBAL_RISE=continue sbatch slurm/experiments/exp6_bone_fix.sh
```
Its `bone_fix.json` lands in `results/experiments/exp6_bone_fix/anny_teacher_v9_mixed_00062_n200_s0/`.

3DPW re-fit chain (section 5, item 3). **NEEDS CODE** first, three changes. Both fitter fixes go behind default-off
flags (the brief's rule, section 10): the old labels that v9/v10 trained on must stay reproducible, and the new
behaviour is switched on only by the v2 re-fit command below.
1. `slurm/data/run_smpl_to_anny_task.sh`:
   - `:40`: make `OUT_ROOT` overridable (the worked example in section 2). Without it the array writes into the old
     folder and skips every person that already has a file.
   - add a line `REFIT_FLAGS="${REFIT_FLAGS:-}"` right after it (empty = old behaviour), and append `${REFIT_FLAGS}`
     to the `python scripts/data/smpl_to_anny.py` call, i.e. `--shared_shape 1 --quiet 1` (`:78`) becomes
     `--shared_shape 1 --quiet 1 ${REFIT_FLAGS}`. The command text is an unquoted heredoc, so the value is filled in
     when the job starts.
2. `scripts/data/smpl_to_anny.py`, gender. New flag in the argparse block (`:798-853`):
   `--gender_convention`, choices `makehuman` / `anny`, default `makehuman` (= old: men 1.0). At `:1008` (inside
   `main`), `anny` gives `0.0 if gender == 'm' else 1.0`; `makehuman` keeps `1.0 if gender == 'm' else 0.0`. Change
   the print at `:1011-1014` to name the convention actually used. Also correct the module docstring at `:150`
   ("gender 0 = female, 1 = male"): Anny's male end is 0 (3575369:24, :313). A comment needs no flag.
3. `scripts/data/smpl_to_anny.py`, frame grid. New flag `--frame_grid`, choices `per_person` / `shared`, default
   `per_person` (= old). `find_valid_frames` (`:737`) gets an argument `shared_grid=False`; at `:775`, `shared_grid`
   gives `picked = valid_idx[valid_idx % frame_stride == 0][:max_frames]` (every actor on frames 0, 3, 6, ... of the
   sequence), otherwise the old `picked = valid_idx[::frame_stride][:max_frames]` (a grid starting at each actor's
   own first valid frame). The call at `:932` passes `shared_grid=(args.frame_grid == 'shared')`.
Checks: `python -m py_compile scripts/data/smpl_to_anny.py`, `bash -n slurm/data/run_smpl_to_anny_task.sh`, `git diff`.

Then validate on one task first (the script header asks for this, `run_smpl_to_anny_task.sh:19-22`): its log must show
the gender value 0.0 for a man and the per-joint error still ~16-20 mm.
```
cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_full_v2 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0 slurm/data/run_smpl_to_anny_task.sh
```
If it looks right, the rest (fits ~40 GPU hours by the script's own 6 s/frame estimate, ~5-6 h wall time with 8 at
once; not measured; task 0 is skipped because its file exists):
```
cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_full_v2 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
```
Logs `logs/smpl2anny-all-<arrayjobid>_<task>.out`. When all 87 are done:
```
cd /netscratch/najib/multi-hmr && FITS=/netscratch/najib/multi-hmr/threedpw_anny_full_v2 OUT=/netscratch/najib/multi-hmr/threedpw_anny_full_v2/labels sbatch slurm/data/run_build_labels.sh
```
```
cd /netscratch/najib/multi-hmr && LABELS=threedpw_anny_full_v2/labels FITS=threedpw_anny_full_v2 OUT=results/diagnostics/check_3dpw_conversion_v2 sbatch slurm/diagnostics/check_3dpw_conversion.sh
```
Pass: check I1 (gender labels) shows men 0.000 / women 1.000; check I3 (unlabelled actors) "valid actor unlabelled"
near 0 of 960; check V1 (body joints) median still ~19 mm. Training scripts read `threedpw_anny_full/labels`
(hard-coded); a fine-tune on the new labels must pass the new folder.

---

## 5. Next steps, in priority order

1. **Read the pending results** (section 4). Analyse only after the student says "done".
2. **Neck.** Act on the exp6 case from the three-case table in 4a.
   - Case (a) or (b): look at `compare/` (top row without, bottom row with the fix). In case (a) keep the fix (it is
     saved in the fine-tuned checkpoint's args and re-applied on load).
   - Real photos with the fix (cases (a) and (b)): **NEEDS CODE** (a copy of `slurm/demo/run_demo.sh`; it hard-codes v5
     at :58-59). Point `CKPT_SRC` at
     `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/train/checkpoints/00001.pt` (fix in its args)
     and give `CKPT_NAME` a new name; or keep v5 and add
     `--bone_fix results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json` to the
     `python3.9 demo.py` call in the copied script (`run_demo.sh:95`; the flag is defined at `demo.py:819`).
   - Case (b): **NEEDS CODE** in `bone_fix.py`, the chain-preserving head correction (plan :558). For v10's left arm:
     the two-sided correction `R_l <- K_l^T R_l K_l` (plan :489). Find the origin of the flip only if time allows.
   - Case (c): the origin question comes back first; no further fix is planned.
3. **3DPW labels (after the meeting, before any 3DPW claim).** Three code changes (two new default-off flags in the
   fitter, one env var in the array job), then re-fit all 87 tasks into a new folder, rebuild, re-check: exact changes
   and commands in 4e ("3DPW re-fit chain"). The gender refit confirmed the flip only for men; check the women in the
   new check I1 (gender labels). If the meeting moves to Monday, the re-fit (~5-6 h wall time, not measured) could
   start before it.
4. **Supervisor's decision (slide 16).** Fine-tune v9 now with `--gt_center_joint person_center` and x5, or first fix
   the labels (item 3) and then fine-tune? The student proposes labels first, because 3DPW is 30% of each batch and the
   men would keep learning the wrong body. Expected from v5: size ~0.84, root ~0.5 m, cost ~2.6 mm PVE and ~1.6 recall
   points (notes :77). **The v9 fine-tune script does not exist yet (NEEDS CODE).** Recipe:
   - Copy the training call of `slurm/train/train_teacher_v9_mixed.sh` (it has the 3DPW mix flags), into a new
     script, e.g. `slurm/experiments/exp7_v9_depth.sh`. Do **not** reuse `exp1_depth_anchor.sh`: it trains on AnnyOne
     only, so `CKPT=v9` would silently drop the 30% 3DPW. Do not run `train_teacher_v9_mixed.sh` itself: it starts
     from v5, runs 150 epochs and writes to `--name anny_teacher_v9_mixed`.
   - Start: `--pretrained /netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt`
     `--pretrained_remap 0 --load_only_backbone 0`. Epoch 62 is the teacher of the final student
     (`train_distill_vits_v8_final.sh:57`), the better of the two on the AnnyOne holdout (75.0 vs 77.0) and the epoch
     `run_demo_compare.sh` uses. 00149 only if the supervisor wants the last state.
   - Keep from the v9 script: freeze pattern, alphas, boosts, `--mix_3dpw_ratio 0.3 --mix_3dpw_split train
     --val_3dpw_split validation --val_3dpw_n 300`, `--tf32 1`. Set `--start_2d_epoch 0 --learning_rate 1e-5
     --lr_decay_every 30 --n_iters_per_epoch 1000 --max_iter 10000 --eval_freq 1 --nb_max_ckpt 2`.
   - Add: `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5 --mask_3dpw_hands 1`. Optional:
     `--bone_fix results/experiments/exp6_bone_fix/anny_teacher_v9_mixed_00062_n200_s0/bone_fix.json` (estimate it on v9
     first, 4e; the v5 JSON belongs to v5). Without it the neck flip stays.
   - Output: `--save_dir /netscratch/najib/multi-hmr/results/experiments/exp7_v9_depth/v9e62_pc_dist5 --name train`.
     Never `logs/anny_model/anny_teacher_v9_mixed`: `train.py` keeps only the `--nb_max_ckpt` highest epoch numbers of
     a folder (`train.py:400-410`), so a run written there deletes checkpoints (its own low epochs, or old v9 ones).
   - Control arm, same script and weight: `--gt_center_joint root` (still x5), output `.../exp7_v9_depth/v9e62_root_dist5`.
   - Labels (`--mix_3dpw_labels`): slide-16 option 1 ("now") = `/netscratch/najib/multi-hmr/threedpw_anny_full/labels`
     (what v9 trained on; gender reversed, 31% of train missing) or `labels_rebuilt` if the rebuild passed (missing
     sequences back, gender still reversed). Option 2 ("labels first") = `/netscratch/najib/multi-hmr/threedpw_anny_full_v2/labels`
     after item 3.
   - Time: the v9 script quotes ~8-10 min per 1,000-iteration epoch with mixed batches (`train_teacher_v9_mixed.sh:16`),
     so 10 epochs ~1.5-2 h plus evaluations; `--time=04:00:00`.
   - Read: trace the last checkpoint of each arm with `diag_holdout.sh` and a `TAG` (4e); root error, slope, size,
     PVE, recall vs the control. The training log's 3DPW-val numbers are against converted labels only.
5. **Depth slope stuck at 0.3-0.44.** Measure the training-set depth distribution properly (so far only batch means of
   ~2.5-3.2 m) and the fix run's slope on training images. Both **NEED CODE**: extend
   `scripts/experiments/measure_batch_ages.py` (it already draws real training batches and applies a copy of
   `prepare_gt`'s visibility filter) to also print GT root-depth percentiles; and pass `--select train_sample`
   through `slurm/diagnostics/diag_holdout.sh` (the trace supports it, `diag_holdout_trace.py:527`; the wrapper passes
   only `--ckpt --n --out`). [Guessing] it needs more unfrozen blocks or a higher LR on the depth head.
6. **Cost of the fix.** PVE paired on the same people; recall on people whose head is visible ([Likely, untested]: the
   recall drop is a cohort effect from moving the detection cell to the head). **NEEDS CODE**, no tool exists: paired
   PVE = join two runs' `holdout_trace/per_person.csv` on `idx` + `gid` for `variant == V1_eval_masked`; head-visible
   recall needs a head-visibility column in `diag_holdout_trace.py`. Report root error next to PVE in the thesis.
7. **Pilot noise.** Set a seed, or repeat the key run, before reading any 2-epoch difference under ~10 mm. **NEEDS
   CODE**: `train.py` has no seed at all. Add `--seed` (default -1 = off, old behaviour) and, at the top of `main()`
   (`train.py:899`), seed `random`, `numpy` and `torch`, and give the training DataLoader a seeded generator.
8. **People very close to the camera.** Count misses (4 of 12 rendered images, both models).
9. **Children (lowest priority).** The exp2 script's own gate says not to run longer (`exp2_children.sh:38`; both
   pilots print "ADULT REGRESSION ... do not run longer", 3571955:183, 3571956:178). Running it anyway is a deliberate
   override: the control fails the same gate, so the gate measures the restart cost, not the sampler [Likely]. Run
   both arms (default `GT_CENTER=root`, i.e. without the depth fix):
   ```
   cd /netscratch/najib/multi-hmr && MAX_ITER=10000 RUN=long_balanced sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && AGE_BALANCE=0 MAX_ITER=10000 RUN=long_control sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
   ```
   Then the real-photo read on both final checkpoints (the real target; `TAG` needed, section 2):
   ```
   cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/results/experiments/exp2_children/long_balanced/train/checkpoints/00009.pt TAG=long_balanced_00009 sbatch slurm/diagnostics/diag_crop_camera.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/results/experiments/exp2_children/long_control/train/checkpoints/00009.pt TAG=long_control_00009 sbatch slurm/diagnostics/diag_crop_camera.sh
   ```
   Compare the five small people's ages with 0.57-0.67. **Fingers:** train on larger hand crops (no script). If
   neither moves, both become thesis limitations.
10. **Docs and thesis.** Fix the stale lines in section 7; thesis draft 74 -> 70.5 mm (V1; the draft is outside the
    repo).
11. **Lower priority:** v10 shape drift (only epochs 290-293 traced). First list what exists (only the last 10 epochs
    are kept per run): `ls /netscratch/najib/multi-hmr/logs/anny_model/anny_v10_scratch_mixed/checkpoints/` and
    `ls /netscratch/najib/multi-hmr/models/multiHMR/ | grep v10`; then trace each with `diag_holdout.sh` (4e).
    Students' depth bug (not measured); output-neutral fix of the 6D init (only matters for new heads; plan :680);
    geodesic loss with `--amp 1` untested; why crosscountry_00 counts as not visible; test-split limb ratios and lost
    head turn; repeat the overfit test with the depth fix; Anny age calibration.

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
- Quoting ages in years ("0.3 ~ 11 y", "0.55 ~ 20 y", "0.5 ~ 18 y"). Do not quote years at all: the code's anchors
  (0.33 = 11 y, 0.67 = 25 y) and Anny's calibration (0.3 ~ 7 y, 0.55 ~ 13.6 y) disagree, and which one AnnyOne uses
  is not settled (3.4). Compare predicted ages with the holdout adults instead.
- "11% children in batches." Measured 9.7-11.1% depending on the run; 9.9% in the measurement job 3571601.

**3DPW labels**
- "48% of two-person images miss an actor." Measured 42% (407/960).
- "2 of 12 validation sequences unlabelled / 28% smaller." One not built, one dropped by visibility; 31% of train
  labels missing.
- "The fit error was never measured." Now 18.7 mm.
- "The 3DPW labels are accurate." Say "pass": the nearest-point surface test is a lower bound and blind to gender.
- "OVERALL FAIL is from limb ratios and 3.2% outliers." FAIL is from limb ratios (test split) and from 5.2% outliers in
  the train split; the 3.2% overall is only WARN.

**Numbers and workflow (corrected while checking this file against the logs)**
- "Job 3571601 is an earlier sampler run." It is measurement only (`SKIP_TRAIN=1`, 3571601:54): age index and batch
  ages, natural 2.14% -> balanced 9.91% (:35, :49-51); no training.
- "exp1: the slope did not move toward 1." The rule's slope clause printed `yes` for x5, x20 and even the 2-epoch
  control. The slope is 0.30 with the switch alone and 0.38-0.44 with the stronger depth loss (baseline 0.33), far
  from 1. The deck's slide-13 notes still use the old wording.
- "v10 is a ViT-L from scratch." Pretrained DINOv2 backbone, fresh heads.
- "v9 exists only in `models/multiHMR/`." Only epoch 62 is known to be missing under `logs/`.
- "6.7 images/s." Measured ~12 in most runs; 6.7 on one node.
- "3-5 NaN skips per 2,000 in every run." 1-7 per 2,000 (13-23 per 10,000).
- "Validation after the rebuild: ~2,777 minus crosscountry." The 2,777 already excludes crosscountry.
- "The visible long-neck look is the flip [Certain]." Only the measured girth / PVE_neck are shown; the look is
  [Likely] until rendered.

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

**Others**
- `scripts/diagnostics/README.md:9` "written and syntax-checked, not yet executed" (all E1-E4 ran).
- `docs/cluster_commands.txt`: old scratch commands; `--pose_param` does not exist in `train.py`.
- `docs/3dpw_adapter_check.md`: one-sequence pilot; superseded by the full check (job 3575369).
- Job-script headers (`exp1_depth_anchor.sh`, `exp2_children.sh`, others): time estimates at 6.7 images/s (measured
  ~12 in most runs).
- Deck and notes (fix both in the pptx by hand, then in `docs/deck_src/build.js`; table in section 9):
  - notes slide 13 (notes :59): "The slope did not move toward 1" (see section 6).
  - "rebuild running now", never confirmed: slide 6 text (`build.js:187`) and notes :25; slide 15 item 7 text
    (`build.js:384`) and notes :69; slide 16 notes :77.
  - notes slide 16 (notes :77): "The fine-tune is about one hour of training" (section 5, item 4: ~1.5-2 h with 30%
    3DPW batches); "On the AnnyOne holdout v9 is not the best: 77.0 against 70.5" (77.0 is epoch 149; the planned
    start, epoch 62, is 75.0).

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

This file and `docs/deck_src/`: the branch `claude/gallant-dijkstra-bnlowv`. Pushed there: only the WIP commit
`c6419ce`. The final version (this file, `docs/deck_src/README.md`, `docs/deck_src/apply_theme.js` and the two-line
`build.js` change) was not committed when this was written. None of it is on `main` until the branch is merged.

**Getting this file.** It is on `main`. On the cluster: `cd /netscratch/najib/multi-hmr && git pull`, then
`grep -c '^## 0a\. Tomorrow' docs/handover.md` must print `1`. To give it to ChatGPT, download
`https://raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/docs/handover.md`
(save the page as `handover.md`) and attach it.

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
| 3571597 | diag_holdout | failed: v9 00062 not under `logs/anny_model/anny_teacher_v9_mixed/checkpoints/` |
| 3571601 | exp2_children | measurement only (`SKIP_TRAIN=1`): age index + batch-age measurement, natural 2.14% -> balanced 9.91% (:35, :49-51); no training |
| 3571955, 3571956 | exp2_children | sampler pilot, control |
| 3572419, 3572420, 3569949 | diag_cropK | real-photo ages: exp2 pilot, control, v5 |
| 3571957, 3571958 | exp3_neck_global_rot | pilot, control |
| 3571959, 3571960 | exp4_fingers | region-norm pilot; control + crop trace + hand size |
| 3573518, 3573519 | diag_neck_bones | per-bone dumps: v5, v9 e149, v10, long_pc_dist5; lineage v2/v3/v4 |
| 3573520, 3573521 | exp5_neck_fix | L1 control, geodesic |
| 3575369 | check_3dpw_conversion | 3DPW->Anny label check |

**Experiments, one line each:** exp1 depth anchor (placement much better, slope not, likely cost; rule failed);
exp2 children sampler (null); exp3 global head loss (null); exp4 fingers crop + region-norm (both null); exp5
geodesic loss (flip did not move); exp6 bone correction (maybe submitted 15 Oct; no job ID or log yet).

---

## 9. Deck

**Files:** `docs/supervisor_update_2026-10-16.pptx` (17 slides), `docs/supervisor_talk_notes_2026-10-16.md` (spoken
part + "If asked:" part per slide), `docs/figures/depth_before_after_579860.jpg` and `_579862.jpg`.

**Source:** `docs/deck_src/build.js` (pptxgenjs) and its theme helper `docs/deck_src/apply_theme.js`; instructions in
`docs/deck_src/README.md`. No setup beyond `cd docs/deck_src && npm ci && node build.js`, on a machine with node >= 18
and internet access for `npm ci` (probably the student's laptop; the cluster may lack node). Checked on 15 Oct: the
rebuild gives the same slides, theme, media and layouts as the published pptx, and byte-identical notes. Only on the
branch until it is merged (section 8). Notes are generated from `build.js`; never edit the `.md` alone.

**Before the meeting** (whichever date holds): open the pptx once in PowerPoint and check slides 5, 6 and 16 for text
that wraps one line longer (they were only rendered with LibreOffice).

**Deck edits before the meeting** (stale lines, and pending results that arrive). Edit the pptx by hand in PowerPoint
(slide text and its speaker notes); port the same changes into `build.js` after the meeting and rebuild, so the `.md`
notes match again.

The first three rows apply whatever arrives (stale today). The exp6 rows follow the three cases of 4a.

| result | slides to change | what changes |
|---|---|---|
| always | 13 (notes :59) | "The slope did not move toward 1" -> "the slope is 0.30 with the switch alone and 0.38-0.44 with the stronger depth loss (baseline 0.33), far from 1" |
| always: rebuild not confirmed | 6 (text "rebuild running now"; notes :25), 15 (item 7 "Running now: rebuild"; notes :69), 16 (notes :77, "the rebuild is running") | "running now" -> "to run" (or the counts, row "label rebuild ran" below) |
| always | 16 (notes :77) | "The fine-tune is about one hour of training" -> "about 1.5-2 hours with 30% 3DPW batches"; "77.0 against 70.5" -> "75.0 at epoch 62, the planned start (77.0 at epoch 149), against 70.5" |
| exp6 v5 case (a): script PASS | 7, 15 (item 1), 16, 1 | slide 7 "Next (exp6): ..." becomes the measured result; slide 15 item 1 "not fixed" becomes the numbers; slide 16: move "a neck fix" from "Not claimed" to "Claimed" only for `FINE-TUNED FIX: PASS`; for a zero-training PASS only, claim "a fixed output correction removes the flip without training"; slide 1 line "found, not yet fixed" -> "found; a fixed output correction removes it" |
| exp6 v5 case (b): neck02 < 15 and girth >= 0.9, verdict FAIL on head / global head / PVE / recall | 7, 15 (item 1), 16 | slide 7 "Next (exp6)" -> "exp6: the neck skin recovers with a fixed output correction (girth 0.46 -> Z); the head keeps a ~Y deg residual"; slide 15 item 1: the same numbers + "next: a head correction that keeps the chain"; slide 16: keep "a neck fix" under "Not claimed", add "the neck skin recovers with a fixed output correction" under "Claimed" |
| exp6 v5 case (c): neck02 >= 15 or girth < 0.9 | 7, 15 (item 1) | "Next (exp6)" -> what was tried and that it did not work (neck02 or girth numbers); origin still open |
| exp6 v10 metacarpals, PASS only | 7 (v10 column) | one line: the three metacarpals corrected without training |
| label rebuild ran | 6, 15 (item 7), notes :25, :69, :77 | the rebuilt train / validation / test counts (expected ~6,437 / ~2,777 / 9,722) |
| `long_pc_dist20` | 12 (table note), 15 (item 3, "No result yet for the ×20 run") | root error, size, slope, PVE, recall of x20 10 epochs next to x5 10 epochs |

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

**What the student wants first.** On 14 Oct the student asked: "are you fixing my issues of the project or are you just
editing slides?" Project fixes come before doc and slide polishing; say which one you are doing.

**Deadline: unknown, ask.** The student's 13 Oct prompt began "You are helping me finish a human mesh recovery thesis
under a one-week deadline. My supervisor meeting is Friday, 16 October 2026." Neither the submission date, the time
left, the GPU budget nor the draft's chapter list is in this chat. Ask for the submission date and the chapter list
before planning anything that takes more than a day (e.g. the ~40 GPU-hour 3DPW re-fit vs the v9 fine-tune vs
writing).

**The original brief (13-14 Oct), still binding:**
- Implement the smallest changes likely to help; "preserve each change as an independent experiment" (a flag,
  default off = old behaviour; its own `RUN=` folder).
- Prioritise confirmed implementation bugs. No new architecture unless the diagnostics justify it.
- Keep data splits and evaluation settings fixed (the 100-image holdout, V1). Keep the baseline and production
  checkpoints intact; new outputs go to new folders.
- Reuse pretrained weights. Start with a short pilot, measure throughput, estimate the longer run before launching
  it. "Do not launch full training or upgrade dependencies blindly."
- Give a configuration, a patch, a runtime estimate and an exact command for each experiment. "Never claim a fix
  works without measured results."

**Practical lessons:**
- Repo on the cluster: `/netscratch/najib/multi-hmr`. Always `git pull` first, then `sbatch` from the repo root.
  Logs: `/netscratch/najib/multi-hmr/logs`.
- **Never give commands with `<placeholders>`.** On 14 Oct the student pasted `CKPT=<full path ...>` literally. Write
  out real paths.
- **Never put a destructive git command in the same copy-paste block as other commands.** Once a `git status` and a
  `git reset --hard` were pasted together and the reset wiped uncommitted cluster edits.
- The student sends logs in batches (up to 5 files per message) and then says "done". Acknowledge each batch in one
  line; analyse only after "done".
- Pass `TAG=` to a diagnostic job for any experiment checkpoint or repeated setting (section 2). Give each
  experiment its own `RUN=` name.
- You cannot push. Code changes follow section 2, "Working without repo access": paste first, one change block,
  a separate commit-and-push block.
- Read every pilot against its own control arm, never only against the v5 baseline (restart cost ~2.4 mm).
- Before calling a 2-epoch difference real, compare it with the noise floor (3.6).
- Put key numbers into the docs: the next assistant will not see the logs unless the student uploads them.
- After the meeting: write the supervisor's answers into section 9 of this file.
