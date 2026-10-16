# Handover: project state on Friday 16 October 2026 (early morning)

This file replaces the chat history. It is written for an assistant that cannot see that chat, the cluster or any
temporary files. Everything you need is either here or in this repository. **Where this file disagrees with
`README.md`, `docs/diagnostics_findings.md`, `docs/experiments_plan.md` or `docs/experiments/exp2_children.md`, trust
this file** (section 7 lists the outdated lines). The deck `docs/supervisor_update_2026-10-16.pptx` (18 slides) and
the speaker notes `docs/supervisor_talk_notes_2026-10-16.md` match this file. Section 9 lists the slides and the
lines that will go stale when the pending results arrive.

**Which version you have.** Everything this file describes is on GitHub `main`: all code, the 3DPW converter fix
(v2.3 flags and the re-fit post-check, 4e), this file, the 18-slide deck with its notes and its source
`docs/deck_src/`, and the figures in `docs/figures/`. A `git pull` on the cluster gets all of it; nothing waits on a
side branch. To check a copy of this file:
- `grep -c '^## 0a\. Tomorrow' docs/handover.md` prints `1` for the 15-Oct and later versions and `0` for every older
  one (the 14-Oct `586faad`, or the work-in-progress `c6419ce`).
- `grep -c ALLOW_DEFAULT_FLAGS docs/handover.md` prints a number above `0` only in this version (16 Oct, after the
  merge; it quotes the re-fit steps in 4e). `0` means an older copy, which may still call the converter fix
  unmerged, the real-image demo unseen or the deck 17 slides long. Ask the student for the current file.

The student should attach this file to the chat directly, not rely on you browsing GitHub. Download links: section 8.

**What happened on 16 Oct** (all in this version): the student sent the 12 exp6 compare panels (neck checked by eye
on the holdout; 3.2, 4a). The real-image demo ran (job 3581312: v5 next to the exp6 fine-tuned checkpoint with the
fix, on 4 photos and 8 3DPW-test frames; the student sent 8 of these 12 images). By eye, on those 8, the thin necks
get normal width with the fix; the fixed side also had the 2-epoch fine-tune (3.2, "Real images"). Slide 8 shows a
holdout pair and a real pair (section 9). The image folders of the 8 affected 3DPW sequences were re-extracted and a
second label rebuild was submitted (job 3581320, log not read yet; 3.5, 4c). The converter fix and this file were
merged into `main` (section 8).

**Meeting date: open.** Planned for **Friday 16 October 2026**, with the deck `docs/supervisor_update_2026-10-16.pptx`.
On 15 Oct (16:07) the student asked for a draft message asking the supervisor to move it to **Monday 19 October**.
Whether it was sent is unknown; still open on the evening of 15 Oct. An earlier message to the supervisor (content
unknown) had no reply. Ask which date holds before planning: it decides whether the result of the second label
rebuild (job 3581320, 4c) can go into the deck first. The real-image demo is already on slide 8 (section 9).

**Citation key** (used from section 1 on):
- `3572482:126` = line 126 of that Slurm job's `.out` log. Job index in section 8. The student has every log; ask for
  one when you need to check a number.
- `README :L` = `README.md`; `plan :L` = `docs/experiments_plan.md`; `findings` = `docs/diagnostics_findings.md`.
- `notes, slide N` = the speaker notes of slide N in `docs/supervisor_talk_notes_2026-10-16.md` (18 slides);
  "If asked" = that slide's second part. The notes are cited by slide, not by line: their line numbers moved with
  every rebuild. Copies from before 16 Oct had 17 slides: the neck before/after slide (now 8) was added on 16 Oct,
  so their slides 8-17 are today's 9-18.
- `file.py:L` = that file in the repo. Line numbers are as of `64266dc`, unless a commit is named next to them. Of
  the files cited by line, only these have changed on `main` since: `slurm/demo/run_demo_compare.sh` (`fbda1b4`, 4b;
  cited "at `fbda1b4`"); the two 3DPW converter files `scripts/data/smpl_to_anny.py` and
  `slurm/data/run_smpl_to_anny_task.sh` (converter v2.3, 4e; current lines are named where they matter); and
  `docs/experiments_plan.md`, where the converter section was inserted at :593 (`plan` lines from :593 on are given
  as in the current file).

---

## 0. Paste this first

The student pastes this block, section 0a and section 0b into the first message as text, then attaches this file.
[Guessing] ChatGPT may only search a large attachment (this one is ~150 KB) instead of reading all of it; text in the
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
- Neck: two bones are turned half a turn in every v5-line model. exp6 (a fixed correction at the network output,
  then a 2-epoch fine-tune) turns them back, and the neck gets its thickness back (girth 0.46 -> 1.00). Without the
  fine-tune the head points worse. One run, no seed: no sign of a PVE cost, but no proven PVE gain either.
  Synthetic holdout renders of the correction alone, no fine-tune (16 Oct): thin necks without it, normal width
  with it. Real images (16 Oct, job 3581312, the fine-tuned checkpoint vs v5, 4 photos + 8 3DPW-test frames, 8 of
  the 12 seen): the same, judged by eye, no metric; the fixed side also had the 2-epoch fine-tune.
- 3DPW labels: three defects (reversed gender, a missing actor in two-person frames, missing labels). A label rebuild
  gave the same counts; I then re-extracted the image folders of 8 sequences, and a second rebuild (job 3581320)
  is waiting to be read. The converter fix for gender and the missing actor is in the repo (default-off flags); the
  re-fit with it has not been run.
- Fingers: flipped hand bones were a large part of the finger error (exp6 on v5: 17.3 -> 8.2 deg).
- Children: tested, nothing helped yet. New (16 Oct): adult men on real photos get female-looking bodies, in v5 and
  in the fixed model alike; cause unknown.
Supervisor meeting: planned Fri 16 Oct; I may have asked to move it to Mon 19 Oct. Ask me which date holds.
Thesis: my 13 Oct brief said a one-week deadline. Ask me the submission date and the chapter list of my draft
before planning anything that takes more than a day.
Finished 15 Oct: exp6 on v5 and on v10, 3DPW label rebuild. 16 Oct: the exp6 neck panels, the real-image demo with
the fix (job 3581312), the 3DPW image re-extraction (results in handover section 3). Pending, in this order:
(1) the log of the second label rebuild (job 3581320); (2) the 3DPW re-fit with the converter fix (steps 0-5,
handover 4e); (3) the final model: a v9-style fine-tune from the fixed v5 with the depth fix, the bone fix and the
fixed 3DPW labels, no script yet (handover sections 4 and 5).
Never confirmed: the x20 depth run (14 Oct). Ask me to run the status block from handover section 4 (squeue alone
misses finished jobs).

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
2. Status block (section 4, top), for the rebuild job 3581320 and the x20 depth run (4d, never confirmed). Resubmit
   a job only if both `sacct` and its output folder show nothing. Analyse logs only after the student says "done".
3. Pending (1): the log of the second label rebuild, `logs/3dpw-labels-3581320.out` (4c). Check its output folder
   first (it must not be the production `labels/`), then compare the counts with the expected train ~6,437 /
   validation ~2,777 / test 9,722. These add up to 18,936, the number of label images the current fits imply; the
   converter section of `docs/experiments_plan.md` uses the same 18,936. If the counts match, run the re-check (4c).
   This takes minutes; do it first, whichever day the meeting is.
4. Pending (2): the 3DPW re-fit with the converter fix. The fix is on `main` (v2.3 flags, default off); there is
   nothing to write. Run steps 0-5 one at a time: start each step only after the previous one has finished, and
   steps 2 and 3 only after the step before them passed. What to check after each step: 4e ("3DPW re-fit"). On
   Friday, start it after the meeting, unless the student wants it running during the meeting; it writes to a new
   folder, so starting early is harmless. The commands, as in the converter section of `docs/experiments_plan.md`:
   - Step 0, check the cluster copy (all three numbers above `0`, both files listed; otherwise stop):
     ```
     cd /netscratch/najib/multi-hmr && git pull && grep -c REFIT_FLAGS slurm/data/run_smpl_to_anny_task.sh; grep -c gender_convention scripts/data/smpl_to_anny.py; grep -c ALLOW_DEFAULT_FLAGS slurm/data/run_smpl_to_anny_task.sh; ls scripts/data/check_refit_task.py slurm/data/run_check_refit_task.sh
     ```
   - Step 1, one check task (~25 min); wait until the job has left `squeue -u najib` and both logs have a `done:`
     line:
     ```
     cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=9-10 slurm/data/run_smpl_to_anny_task.sh
     ```
   - Step 2, its post-check (CPU, seconds); go on only on `RESULT: PASS`:
     ```
     cd /netscratch/najib/multi-hmr && SPLIT=train SEQ=courtyard_captureSelfies_00 sbatch slurm/data/run_check_refit_task.sh
     ```
   - Step 3, all 87 tasks (~5-6 h of wall time, estimated, not measured); wait until the array has finished and
     87 `.npz` files exist:
     ```
     cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
     ```
   - Steps 4 and 5, labels and the conversion check (one block; the check waits for the label job):
     ```
     cd /netscratch/najib/multi-hmr && L=$(FITS=/netscratch/najib/multi-hmr/threedpw_anny_v23 OUT=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --parsable slurm/data/run_build_labels.sh) && echo "labels job ${L%%;*}" && LABELS=threedpw_anny_v23/labels FITS=threedpw_anny_v23 OUT=results/diagnostics/check_3dpw_conversion_v23 sbatch --dependency=afterok:${L%%;*} slurm/diagnostics/check_3dpw_conversion.sh
     ```
5. Pending (3), after the re-fit and its check pass: the final model, a v9-style fine-tune from the fixed v5 with the
   depth fix, the bone fix and the fixed 3DPW labels (section 5, item 7). **No script or command exists for it yet**
   (NEEDS CODE). Optional before it: the longer fine-tune with the fix and its matched control (section 5, item 3;
   4e). If that run is no worse than its control, its checkpoint is the better start for the final model. Slide 17
   describes the fine-tune as one of v9 epoch 62; if option 2 is chosen, the student should say at the meeting that
   this one starts from the fixed v5 (section 5, item 7).

Done on 16 Oct, nothing to ask for: the exp6 v5 compare panels (3.2, 4a); the real-image demo with the fix (job
3581312; 3.2, 4b; figures `docs/figures/neck_before_after_3dpw_bar.jpg` and `..._bar_office.jpg`); the
re-extraction of the 8 image folders (3.5, 4c). The demo also showed a new open issue in both models: adult men get
female-looking bodies (3.4). Optional: a correction-only demo without the fine-tune (section 5, item 2).

If the meeting is on **Friday**:

6. Check that the student's deck is the 18-slide version (section 9, "Check a copy"). A `git pull` gives it.
7. Open the pptx in PowerPoint and check slides 1, 2, 5, 6, 7, 8, 9, 15, 16 and 17 for text that wraps one line
   longer (slide 15's Wording card already ends at the card's bottom edge in the LibreOffice render).
8. After the meeting: write the supervisor's answer to the slide-17 question into section 9. If it is "labels first"
   (the student's proposal), go on with 0a steps 4 and 5. If it is "fine-tune v9 now", section 5, item 6 comes before
   them.

If the meeting is on **Monday**:

6. Start 0a step 4 (the re-fit) before the meeting; its result can then go into the deck (section 9, "Results still to
   come").
7. Start the longer fine-tune with the fix and its matched control (section 5, item 3; 4e), if GPUs are free next to
   the re-fit.
8. Before the meeting: Friday steps 6-7 (deck check, PowerPoint check).

## 0b. Numbers at a glance

V1 setting, 100-image AnnyOne holdout, matched people, unless noted. Sources in section 3.

- v5 epoch 99: PVE 70.5 mm, PA-PVE 53.3, recall 90.2%. Root error 1.21 m, size ratio 0.63, depth slope 0.33.
- Depth fix: root error 0.53-0.73 m (controls 1.11-1.44 m), size 0.78-0.87. Slope 0.30 with the switch alone and
  0.38-0.44 with the stronger depth loss, far from 1. Best runs: x5 for 10 epochs 0.54 m / 0.84 / 0.38; x20 for 2
  epochs 0.53 m / 0.87 / 0.43.
- Cost of the depth fix vs its control: PVE +2.6 to +4.9 mm, recall -1.6 to -2.4 points (each inside pilot noise, all the
  same direction).
- Pilot noise: two identical configs differ by 7.6 mm PVE and 1.9 deg global head. One pilot resolves ~10 mm, ~3 deg,
  ~2 recall points.
- Neck, v5 (all eight v5-line checkpoints alike): neck02 175.6 deg (half-turn), head 177.9 deg, global head only
  12 deg, neck girth pred/GT 0.46, PVE_neck 27.3 mm. v10: girth 1.00, PVE_neck 10.3 mm. exp5 geodesic loss: neck02
  175.6 -> 175.0.
- exp6 on v5 (job 3579014): one fixed rotation each for neck02, head, metacarpal1.L and finger4-1.R (178.8-179.8 deg,
  estimated on 200 training images). Zero-training verdict FAIL (global head 19.1 vs limit 15.0); fine-tuned PASS.

  | V1 holdout | v5 | fix, no training | fix + 2-epoch fine-tune | control (2 epochs, no fix) |
  |---|---|---|---|---|
  | neck02 / head, deg | 175.6 / 177.9 | 4.5 / 12.4 | 4.7 / 11.2 | 175.7 / 177.9 |
  | girth pred/GT; PVE_neck mm | 0.461; 27.3 | 1.004; 10.8 | 0.997; 10.7 | 0.461; 27.9 |
  | PVE / PA-PVE / PVE_hn, mm | 70.5 / 53.3 / 52.3 | 71.3 / 52.2 / 63.1 | 68.8 / 50.9 / 51.8 | 76.9 / 55.2 / 55.8 |
  | recall %; global head deg | 90.2; 12.0 | 90.2; 19.1 | 89.4; 14.4 | 91.1; 13.8 |
  | fingers deg | 17.3 | 8.2 | 8.3 | 17.4 |

  One run each, no seed; identical configs differ by up to 7.6 mm PVE. Say "no sign of a PVE cost", not "PVE
  improved".
- exp6 v5 compare panels (12 holdout images 579860-579871; the correction with no training): thin, stalk-like necks
  without it, normal width with it, in every panel where a neck is visible [Certain, by eye]. Figure
  `docs/figures/neck_before_after_v5.jpg`. Synthetic only.
- Real-image demo (job 3581312; v5 vs the exp6 fine-tuned checkpoint, which applies the fix from its saved args;
  4 photos + 8 3DPW-test frames, never trained on): thin, stalk-like necks with v5, normal width with the fix
  [Certain for the 8 images the student sent; judged by eye, no metric]. Clearest on `downtown_bar_00` frame 701
  and `office_phoneCall_00` frame 440. Figures `docs/figures/neck_before_after_3dpw_bar.jpg` and
  `..._bar_office.jpg`. The two columns also differ by the 2-epoch fine-tune.
- Both demo models, real photos: adult men get female-looking bodies. The log's predicted gender is 0.46-0.76 on
  the photos (median 0.60-0.62), and Anny's male end is 0 (3.4). Cause unknown [Guessing: a synthetic-to-real gap].
- exp6 on v10 (job 3579015, no training, metacarpal2.L/2.R/3.R): PASS. PVE 74.1 -> 69.9, PA-PVE 55.6 -> 51.1,
  fingers 32.0 -> 17.8 deg; global head 12.9 and recall 89.0 unchanged. The PVE drop is certain only for these 100
  images (no training, so no run-to-run noise); [Likely] on other images. shoulder01.L (166) and upperarm02.L (180)
  stay flipped: they need a two-sided correction, not implemented.
- Other half-turns: v5 metacarpal1.L 180 deg, finger4-1.R 178.8 (found by exp6 on training images); v10
  shoulder01.L 166, upperarm02.L 180, metacarpal2.L/2.R/3.R 180; v10 finger2-1.L 178.8, finger5-2.L 180 (training
  images; not corrected).
- Fingers: predicted 17.3 deg vs 15.1 at rest pose; with the exp6 fix 8.2; test-time crop 17.2; region-norm loss 17.5
  vs 17.4 (control).
- Children: sampler raises children per batch 2.1% -> 9.7-11%; real photos stay at age 0.57-0.67 either way.
- 3DPW labels: body joints 18.7 mm median (p95 24.5); surface 18.9 / 18.3 mm (lower bound). Gender reversed (men:
  19.2 -> 15.3 mm when flipped); 407 of 960 two-person images miss an actor; train 4,451 of 6,437 images labelled,
  validation 2,458 of 2,777. Rebuild (job 3579266): the same 4,451 / 2,458 / 9,722; [Likely] image files missing.
  The image folders of the 8 sequences were re-extracted on 16 Oct (captureSelfies_00: 480 -> 750 files). Second
  rebuild, job 3581320: log pending; expected train ~6,437, validation ~2,777, test 9,722 (18,936 in all, the count
  the current fits imply).
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
  necks) plus 15 held-out images with no children (notes, slide 3).
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
- exp6, v5 fine-tuned 2 epochs with the bone fix: `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/train/checkpoints/00001.pt`
  (the fix is saved in its args and re-applied on load; 3579014:80). Demo copy
  `models/multiHMR/exp6_v5_bonefix_00001.pt` (made 16 Oct; the demo job 3581312 loaded it under that name and applied
  the fix from its args, 3581312:419-427).
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
- **restart cost** = what 2 more epochs of fine-tuning v5 cost with nothing changed: ~2.4 mm PVE (notes, slide 14, If
  asked). This is why each pilot is read against its own control arm, not against v5.
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
  below point the right way (v5: neck02 -> head; v10: shoulder01.L -> upperarm02.L). A constant on each flipped bone
  still leaves the link in between wrong (exp6 v5: global head 19.1 without training, 3579014:32-35, :113); keeping
  the chain needs a two-sided correction.
- **girth** = neck-skin radius pred/GT. **PVE_neck** = vertex error over the 171 neck-skin vertices (skinned >= 50%
  to neck01-03), reported neck01-centred. **PVE_hn** = vertex error over head + neck vertices (neck01 joint to the
  crown). **neck%** = predicted / GT neck length, %.
- **GLOBAL GUARD** (`compute_bone_fix.py`): exit 4 = the correction is predicted to raise some bone's global error by
  more than 5 deg (a compensated chain); `ON_GLOBAL_RISE` decides (continue / stop / safe). Exit 3 = no bone
  qualifies for a correction; the job stops.
- **x4 boost**: `--boost_neck_weight 4` / `--boost_hand_weight 4` are renormalised over 163 bones, so "x4" is
  really x2.41 for the boosted bones and x0.60 for the rest (plan :932).
- **stride phase**: each 3DPW actor's every-third-frame grid starts at that actor's own first valid frame, so two
  actors can sit on different frames (0, 3, 6 vs 1, 4, 7).
- **`%8`** in `sbatch --array=0-86%8`: at most 8 array tasks run at once.
- **decpose surgery**: an earlier idea (plan :422-426) to edit the pose-decoder weights of a checkpoint to undo a
  flip. Superseded by exp6.
- **diag_cropK**: job and log name of `slurm/diagnostics/diag_crop_camera.sh` (real-photo crop-camera read).

---

## 2. Where things are

**Repository.** github.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset
(public), branch `main`. Since the 16-Oct merge (section 8) it holds everything this file refers to: the code as of
`64266dc`, the `DEMO_MODELS` option of the demo script (`fbda1b4`, 4b), the 3DPW converter v2.3 with its re-fit
post-check (4e), this file, the deck with its notes and source, and the figures. The cluster copy is updated with
`git pull`. Raw files can be read at
`raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/`
followed by the path (e.g. `.../main/train.py`), if your tools can fetch URLs; otherwise ask the student to paste.

| path | what |
|---|---|
| `train.py`, `model.py`, `loss.py`, `demo.py`, `bone_fix.py` | training, model, losses, demo, the exp6 bone correction |
| `scripts/diagnostics/diag_holdout_trace.py` | the evaluation used for every number here (V0-V4, placement, per-bone dump) |
| `scripts/diagnostics/compute_bone_fix.py` | exp6: estimates one correction rotation per flipped bone |
| `scripts/diagnostics/check_3dpw_conversion.py` | end-to-end check of the 3DPW->Anny labels |
| `scripts/experiments/` | exp summaries (`exp5_neck_summary.py`, `exp6_bone_fix_summary.py`), `measure_batch_ages.py`, `measure_hand_px.py`, `build_age_index.py` |
| `scripts/data/smpl_to_anny.py` | the SMPL->Anny fitter, v2.3: `--gender_convention` and `--frame_grid` (4e). Their defaults keep the old behaviour, including the reversed gender (`gender_phenotype_value`, :286) |
| `scripts/data/build_threedpw_anny_labels.py` | fits -> per-image label files |
| `scripts/data/check_refit_task.py`, `slurm/data/run_check_refit_task.sh` | post-check of a v2.3 re-fit against the old fits (gender values, shared frame grid, actors fitted on the same frames; 4e, step 2) |
| `slurm/experiments/exp1..exp6_*.sh` | one job script per experiment; the header of each explains why, what, how to read |
| `slurm/diagnostics/*.sh` | `diag_holdout.sh`, `diag_neck_bones.sh`, `check_3dpw_conversion.sh`, others |
| `slurm/data/run_build_labels.sh`, `run_smpl_to_anny_task.sh` | 3DPW label build and the per-person fit array job (`OUT_ROOT`, `REFIT_FLAGS` and three guards since v2.3, 4e) |
| `slurm/train/` | v9, v10, final student training scripts |
| `slurm/eval/` | `run_eval_3dpw_test.sh` (3DPW-test via `train.py --eval_only 1`, V0; [Likely] the source of the 3DPW-test column in section 1), `run_eval_3dpw.sh`, `run_eval_3dpw_teacher.sh`, `run_test.sh` |
| `slurm/history/` | finished training runs of the v5 line (`train_step2_frozen_bb.sh` = frozen_bb_v2, ...) |
| `slurm/demo/` | `run_demo.sh` (real photos; copies v5 00099 under the misleading name `anny_distill_vits_v2_00099`, :58-59), `run_demo_syn.sh` (AnnyOne images), `run_demo_compare.sh` (five models), `run_demo_3dpw.sh`, `run_demo_student.sh`. All hard-code their checkpoints; none passes `--bone_fix`, but `demo.py` re-applies a fix saved in the checkpoint args (`demo.py:151-155`) |
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
| `--gender_convention anny`, `--frame_grid shared` | smpl_to_anny.py | 3DPW converter v2.3: Anny's gender convention (m -> 0.0, f -> 1.0) and one frame grid for both actors. Passed through `REFIT_FLAGS` of `run_smpl_to_anny_task.sh` (4e) |

**Cluster.** DFKI Slurm. Repo at `/netscratch/najib/multi-hmr`, conda env `multihmr`, enroot container, shared setup
`slurm/env.sh`. Logs: `/netscratch/najib/multi-hmr/logs/<jobname>_<jobid>.out` and `.err` (label build:
`3dpw-labels-<jobid>.out`). Checkpoints: `logs/anny_model/<run>/checkpoints/<epoch>.pt` (only the last 10 kept) and
copies in `models/multiHMR/<run>_<epoch>.pt`. v9: use the `models/multiHMR/` copies for both epochs (section 1).
Epoch 62 is not under `logs/anny_model/anny_teacher_v9_mixed/checkpoints/` (job 3571597 failed on that path); that
folder exists, but its contents were never listed. Job status: the status block at the top of section 4
(`squeue -u najib` lists only queued and running jobs).

**How work flows.** The student runs `cd /netscratch/najib/multi-hmr && git pull`, then `sbatch slurm/...` from the
repo root, then uploads the `.out` logs. Until 16 Oct (the `DEMO_MODELS` option, converter v2.3 and the 16-Oct merge
included) the code changes came from an assistant that could push to GitHub; you cannot (next paragraph).
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
   (safer than a hand-written diff). For a new file, a full-file heredoc (`cat > path <<'EOF'`). Worked example
   (the pattern only: this change, `OUT_ROOT` made overridable, is already on `main` as part of converter v2.3, so
   today the block refuses, which is what the assert is for). Keep the `\n` at both ends of `old`: without them it
   also matches inside the current line `DEFAULT_OUT_ROOT="..."`, the assert passes, and the edited script then
   refuses every re-fit, even into a new folder. Anchor every short old text this way.
   ```
   cd /netscratch/najib/multi-hmr && python - <<'EOF'
   p = "slurm/data/run_smpl_to_anny_task.sh"
   s = open(p).read()
   old = '\nOUT_ROOT="/netscratch/najib/multi-hmr/threedpw_anny_full"\n'
   new = '\nOUT_ROOT="${OUT_ROOT:-/netscratch/najib/multi-hmr/threedpw_anny_full}"\n'
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
  Predictions are pulled toward ~2.5-2.9 m; cause unknown (notes, slide 13, and its If asked). Holdout people are at
  1.92 m on average (README :114).
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
  counted] (notes, slide 12).

**Refuted / do not use:** "depth error 1.08 -> 0.39 m" (signed `root dz`, 3572482:126); size 0.67 for v5 (an older
statistic in the findings doc and the exp1 script's "baseline(findings)" column; the trace's own value is 0.63).

**Submitted?** `long_pc_dist20` (x20, 10 epochs): the command was given on 14 Oct (~19:07, "Run this tonight"); never
confirmed, no job ID, no log. Check with the status block (section 4). Command and reading in 4d.

**Next:** see section 5, items 6-9.

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
  And in the exp6 compare panels the thin neck goes away with the bone correction alone while the depth bug stays
  (bodies still drawn too small; see "Visual check" below).
- exp3 (global head loss, 2 epochs, job 3571957): global head 11.3 deg vs 11.7 / 13.6 for two identical controls
  (3571957:98; 3571960:318; 3571958:99): inside noise. Neck bones stayed at 62.5 (3571957:100).
- exp5 (2 epochs, lr 1e-5): the geodesic loss did not move the flip: neck02 175.6 -> 175.0, head 177.9 -> 177.1; the L1
  control stayed at 175.7 / 177.9 (3573521:89-91; verdict "NO MOVEMENT", :103). [Likely] neck02 and head would have to
  turn back together; a partial move raises the loss.

**The thin-neck look is this flip: shown on synthetic holdout renders and, by eye, on real images ([Likely]
there: the fixed side also had the 2-epoch fine-tune).** In the 12 exp6 compare panels the thin neck turns normal
when only the output correction changes (16 Oct; "Visual check" below) [Certain for those renders]. On real
images (job 3581312; "Real images" below) the fine-tuned checkpoint with the fix has normal-width necks where v5 has
thin stalks [Certain for the 8 images the student sent; by eye, no metric].
There the two models also differ by the 2-epoch fine-tune, so "the correction, not the fine-tune, widens the neck on
real images" is [Likely]: on the holdout the same fine-tune without the fix keeps girth 0.461 (control column below).

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

**Not yet done:** a number for the neck on real images (no ground truth there; the demo is judged by eye only).
Optional: the fine-tuned checkpoint's own holdout panels (4a).

**exp6 on v5: the fix passes after a 2-epoch fine-tune (one run, no seed; holdout panels checked by eye for the
correction without training; real images checked by eye for the fine-tuned checkpoint, job 3581312)** (job 3579014;
`--bone_fix`, code in `d902e00`, final in `81dbdcb`). One constant rotation `C` per flipped bone, estimated on 200
random training images (never the holdout) by `compute_bone_fix.py`, applied as `R <- R·C` right after the 6D decoding
(3579014:2, :7, :48). Then a zero-training test on the holdout, and a 2-epoch fine-tune with the fix in place, with
the exp5 L1 recipe (3579014:59). The control is the exp5 L1 run: the same recipe without the fix (3579014:94).
- Bones picked by the script: neck02, head, metacarpal1.L, finger4-1.R; C = 178.8, 179.2, 179.8, 179.7 deg (:12, :37).
  finger4-1.R is new: FLIP_TWIST 178.8 on the training images (:13); the earlier dump (3573518) did not cover finger
  bones.
- The GLOBAL GUARD fired, as expected: on the training images the fix raises the predicted global error of head
  8.4 -> 17.9 deg and jaw 9.1 -> 17.5 (:31). The job went on (`ON_GLOBAL_RISE=continue`, :43-44).

| V1, 100-image holdout | v5 baseline | fix, no training | fix + 2-epoch fine-tune | control: 2 epochs, no fix | 3579014 line |
|---|---|---|---|---|---|
| neck02 median, deg | 175.6 FLIP_TWIST | 4.5 | 4.7 | 175.7 FLIP_TWIST | :101 |
| head median, deg | 177.9 FLIP_TWIST | 12.4 | 11.2 | 177.9 FLIP_TWIST | :103 |
| metacarpal1.L / finger4-1.R, deg | 180.0 / not dumped | 0.0 / 10.2 | 0.0 / 10.3 | - | :105-106 |
| girth pred/GT | 0.461 | 1.004 | 0.997 | 0.461 | :107 |
| PVE_neck (neck01-centred), mm | 27.3 | 10.8 | 10.7 | 27.9 | :108 |
| PVE, mm | 70.5 | 71.3 | 68.8 | 76.9 | :109 |
| PA-PVE, mm | 53.3 | 52.2 | 50.9 | 55.2 | :110 |
| PVE_hn, mm | 52.3 | 63.1 | 51.8 | 55.8 | :111 |
| recall, % | 90.2 | 90.2 | 89.4 | 91.1 | :112 |
| global head, deg | 12.0 | 19.1 | 14.4 | 13.8 | :113 |
| fingers, deg | 17.3 | 8.2 | 8.3 | 17.4 | :114 |

Verdict lines: `ZERO-TRAINING FIX: FAIL (rot_global_head_deg 19.1 vs limit 15.0)` (:120); `FINE-TUNED FIX: PASS`
(:121). By the three-case rule written before the run (15-Oct afternoon handover `64ac9e3`, its 4a), this is case
(a) (fine-tuned PASS), not the expected case (b) (head >= 15 deg).

What this says:
- [Certain] The fix turns both neck bones back on unseen holdout images (neck02 175.6 -> 4.5 deg without training,
  4.7 after the fine-tune), and the neck skin gets its thickness back: girth 0.46 -> 1.00, PVE_neck 27.3 -> 10.7 mm
  (v10 level: 10.3). The control without the fix keeps the flip (175.7, girth 0.461). These changes are far larger than
  the pilot noise (3.6).
- [Certain] Without training, the head is not right: the head bone keeps 12.4 deg, global head rises 12.0 -> 19.1 deg
  and PVE_hn 52.3 -> 63.1 mm. The guard predicted this rise before the holdout test (:31, :44). Do not claim "no
  training needed".
- [Likely] Why: the compensated chain the guard warned about (:32-35). The head's offset differs per person (relSpr
  15.5, :16; the log's own reading, :123), so one constant cannot undo it. The deck gives this one reason (slide 7;
  section 9).
- [Certain] After 2 epochs with the fix in place, the head residual shrinks (head 11.2 deg, global head 14.4, PVE_hn
  51.8) and the fine-tune checks against the control pass (:119, :121): corrected bones under 15 deg, girth >= 0.9,
  PVE <= control + 8 mm, recall >= control - 2 points (89.4 vs a limit of 89.1: a 0.3-point margin), global head <=
  control + 3 deg (14.4 vs 16.8). The PVE check only rules out a cost above 8 mm; it does not show a gain.
- [Likely, not certain] PVE 68.8 against 76.9 for the control (-8.1 mm) and 70.5 for v5. One run each, no seed; two
  identical configs differed by 7.6 mm (3.6), and the control's 76.9 is above the earlier range of six arms
  (68.7-76.3). Say "no sign of a PVE cost", not "PVE improved".
- [Certain] Global head 14.4 vs 13.8 (control) and recall 89.4 vs 91.1 (control) are inside the noise (1.9 deg, ~2
  points).
- The training log's own eval (pve 71.0, then 72.7; :68, :76) is `train.py evaluate()`, [Likely] with helper bones
  unmasked (V0-like, +3.8 mm on v5, README :321). Quote the trace's 68.8, never the 72.7.
- Not shown: why the flip arose (origin still open; the fix is a constant and says nothing about why, :124); a
  measured neck number on real images (the demo, "Real images" below, is judged by eye); the fine-tuned checkpoint's
  own holdout renders (on the holdout only the correction without training was looked at).

**Visual check: the exp6 compare panels (seen early 16 Oct).** The student sent all 12 v5 panels,
`results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/compare/`, holdout images 579860-579871 (the first
12 of the holdout; `N_VIS=12`). Each panel: top row without the fix, bottom row with the correction and **zero
training** (step b of the script: same checkpoint, same images, only the output correction differs; 3579014:58).
- [Certain, from the panels] Without the fix the predicted necks are thin stalks; with the correction they have
  normal width, in every panel where a neck is visible. This matches the measured girth 0.46 -> 1.00 (:107).
- [Certain for these renders] So the thin-neck look is the flip: the rows differ only by the correction, and of the
  four corrected bones only neck02 and head sit on the neck (metacarpal1.L and finger4-1.R are hand bones).
- Figure for the deck (slide 8, section 9): `docs/figures/neck_before_after_v5.jpg` (1254 x 1118; in git as
  `8577bb2`). Left column "Before: v5 as trained", right column "After: correction on neck02 +
  head" (the hand bones are also corrected; hands not judged: the same correction turns back metacarpal1.L and
  finger4-1.R, so the blue person's hands differ between the columns too; the slide caption says "On the neck, only
  ..."); top row holdout image 579862, bottom row 579863. Cut from two compare panels. Slide 8 shows only the 579862
  pair from this file, cropped out, next to the real pair of job 3581312 ("Real images" below), with the deck's own
  header bars (section 9). The 579863 pair is on no slide.
- Caveats, say them with the figure:
  - The panels show the correction **without the fine-tune**: head direction 19.1 deg against 12.0 for v5 in that
    setting; the 2-epoch fine-tune brings it to 14.4 (table above). The fine-tuned checkpoint's own panels were not
    looked at (4a).
  - v5 base model: the bodies are still drawn too small (the separate depth bug, 3.1). The figure shows neck shape,
    not placement.
  - Synthetic holdout only. Real images: next paragraph (job 3581312).
  - Judged by eye; the number behind it is the girth (and PVE_neck 27.3 -> 10.8 mm, :108).

**Real images: v5 next to the fine-tuned checkpoint with the fix (job 3581312, seen 16 Oct).** Log
`logs/demo-compare-3581312.out`. Run with the `DEMO_MODELS` route of 4b: the log lists only the two models, v5
00099 and `exp6_v5_bonefix_00001` (3581312:14, :418), so the variable reached the job. The second model is the exp6
fine-tuned checkpoint: correction plus the 2-epoch fine-tune.
- Demo settings of `slurm/demo/run_demo_compare.sh` (`:158-171` at `fbda1b4`), the same for both: person detector
  (threshold 0.8), `--anchor_to_box 1`, `--relax_hands 1`, `--mask_helper_joints 1` (3581312:22, :429).
- Images: the 4 real photos in `example_data`, and 8 3DPW **test** frames in `demo_images/3dpw_test/`, the middle
  frame of each of 8 test sequences (`run_demo_compare.sh:115-124` at `fbda1b4`).
- Neither model trained on 3DPW: v5 is synthetic only, and the exp6 fine-tune is AnnyOne only (`exp6_bone_fix.sh`
  passes no `--mix_3dpw_*`).
- [Certain, from the log] The fixed model ran with the fix: `[load_model] bone_fix: checkpoint args (bone_fix_spec)`,
  then `[bone_fix] ON, 4 bone(s)`, neck02 178.8, head 179.2, metacarpal1.L 179.8, finger4-1.R 179.7 deg (3581312:424,
  :427; again :604, :607). v5 prints `bone_fix: none in the checkpoint args` (:18, :191). All four output folders
  are `ok` (:832-837).
- [Certain for the 8 images the student sent; judged by eye, no metric] With v5 the necks are thin stalks, with
  visible tendon-like ridges and pinched under the head. With the fix they have normal width. Clearest on 3DPW
  `downtown_bar_00` frame 701 and `office_phoneCall_00` frame 440. Also visible on `downtown_walking_00`,
  `downtown_weeklyMarket_00` and the real photos (tennis, singers, glasses).
- [Likely] The correction, not the fine-tune, makes the difference: the columns differ by both, but on the holdout
  the fine-tune without the fix (exp5 L1 control) keeps girth 0.461 and the correction alone gives 1.004 (table
  above). A correction-only demo on the same images would settle it (optional, section 5, item 2).
- Figures (in git as `a01eb5b`):
  - `docs/figures/neck_before_after_3dpw_bar.jpg` (1134 x 570): one pair, "Before: v5" | "After: v5 + neck fix",
    3DPW test `downtown_bar_00` frame 701.
  - `docs/figures/neck_before_after_3dpw_bar_office.jpg` (1134 x 1118): the bar pair on top,
    `office_phoneCall_00` frame 440 below.
  - The label "v5 + neck fix" is short for the fine-tuned checkpoint, whose fix also turns back metacarpal1.L and
    finger4-1.R (hand bones).
- Caveats, say them with the figures:
  - The "after" column is correction plus the 2-epoch fine-tune, so not every difference between the columns comes
    from the correction.
  - Fingers are set to rest in both (`--relax_hands 1`). metacarpal1.L is still corrected, because `relax_hands` does
    not reset metacarpals (3.3). Hands were not judged.
  - The demo places each body from its detection box (`--anchor_to_box 1`). Bodies still look small or misplaced in
    depth in places, in both models: the depth bug (3.1) is in both.
  - A figure in a painting (`flat_guitar_01`) is detected as a person by both models: a detector false positive,
    seen by eye (the detector keeps 2 boxes on that frame, 3581312:375, :788).
  - Seen in the same images and in both models, unrelated to the neck: adult men on the photos get female-looking
    bodies (3.4).

**exp6 on v10: three metacarpals corrected without training (bone medians; the v10 panels were not sent)** (job
3579015; `TRAIN=0`, bones metacarpal2.L, metacarpal2.R, metacarpal3.R; :2, :12).
- [Certain] `ZERO-TRAINING FIX: PASS` (:86); the guard did not fire (:60). metacarpal2.L / 2.R / 3.R 180.0 -> 0.0 / 1.8
  / 0.0 deg (:70-72), fingers 32.0 -> 17.8 deg (:80). Global head 12.9 (:79), recall 89.0 (:78) and girth 0.998 (:73)
  are unchanged.
- PVE 74.1 -> 69.9 (:75), PA-PVE 55.6 -> 51.1 (:76): [Certain] for these 100 holdout images only. No training is
  involved (same checkpoint, same images, only the output correction differs), so the run-to-run noise of 3.6 does
  not apply. [Likely] on other images. Never call it a certain PVE gain without "on these 100 images".
- [Certain] Not corrected: shoulder01.L 166.2 and upperarm02.L 180.0, still FLIP_TWIST (:68-69). They form a
  compensated chain, so they need a two-sided correction (not implemented, below).
- The training images show two more flipped v10 finger bones, finger2-1.L FLIP_TWIST 178.8 and finger5-2.L FLIP_SWING
  180.0 (:13); the holdout dump did not cover finger bones, and they were not corrected.

**Not implemented:** the two-sided correction `R_l <- K_l^T R_l K_l` (needed to keep a compensated chain such as
v10's left arm intact; plan :489, :561). The chain-preserving head correction (plan :558) was meant for case (b); after
the fine-tune the v5 head is already under 15 deg, so [Likely] it is no longer needed for v5.

### 3.3 Fingers

**Measured** [Certain] (v5, V1):
- Predicted fingers 17.3 deg; fingers set to rest pose 15.1 deg (3571960:290). Without the bone fix the prediction is
  worse than no prediction, so all demos use `--relax_hands 1` and say so (with the fix: below).
- Hands are small: median 2.25 tokens long at 672 px, 41% under 2 tokens (3571960:287).
- exp4a, test-time person crop (x1.3, no retraining): paired 17.3 -> 17.2 deg (3571960:299); body PVE nearly doubles
  (124.8, 3571960:294) because the model never saw such a camera.
- exp4b, region-normalised rotation loss: 17.5 deg vs 17.4 for its control (3571959:316; 3571960:314). It cut the
  hands' share of the rotation loss from ~70% (with the x4 hand boost) to 33%, so it is a rebalance, not an
  upweighting (notes, slide 9, If asked). The earlier x4 hand boost (v4) did not help either.
- v5's `metacarpal1.L` is flipped 180 deg (3.2). `--relax_hands` does not reset metacarpals, so [Likely] the bent left
  index base is visible in the demos.

**Flipped bones (exp6, 3.2)** [Certain]:
- Flipped hand bones are a large part of the finger error. v5: correcting metacarpal1.L and finger4-1.R lowers the
  finger error 17.3 -> 8.2 deg without training (3579014:114) and 8.3 after the fine-tune; the control stays at 17.4.
  v10: correcting three metacarpals lowers it 32.0 -> 17.8 (3579015:80).
- So the old reading "the prediction is worse than no prediction" holds only without the fix: v5 with the fix (8.2) is
  well below the 15.1 rest-pose number. That rest-pose number still contains the metacarpal1.L flip (relax_hands keeps
  metacarpals), so a fair rest-pose floor with the fix was not measured.
- The earlier "[Likely] metacarpal1.L adds ~4.7 deg" was for that bone alone. Correcting it together with finger4-1.R
  removes 9.1 deg; the split between the two is not measured.
- v10 has two more flipped finger bones on training images (finger2-1.L, finger5-2.L; 3579015:13), not corrected.

**Refuted:** "fingers are a resolution problem" for test-time zoom only. **Not shown:** that a model trained on larger
hands would not help. "Under-trained" / "supervision or capacity" is a guess [Guessing].

**Next:** the 16-Oct looks at the exp6 panels and at the real-image demo (job 3581312; both in 3.2) judged the neck
only, not the hands, and the demo set the fingers to rest (`--relax_hands 1`). A run with `--relax_hands 0` would
show the fixed model's own fingers (NEEDS a copy of the demo call, not written). Then larger hand crops; otherwise a
thesis limitation. Low priority.

### 3.4 Children, and body shape on real photos

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
    sit mostly at 0.4-0.8 (notes, slide 10).
  - both arms print "ADULT REGRESSION" (adult PVE 75.1 / 74.9 vs 70.5): that is the restart cost of any fine-tune,
    not the sampler.
- The 3DPW labels fix age at 0.55 for everyone (3575369:232), so mixing 3DPW adds no child signal.

**Age scale** [Likely]: by Anny's own calibration, age 0.10 ~ 2 y, 0.3 ~ 7 y, 0.5 ~ 12.7 y, 0.55 ~ 13.6 y, 0.67 ~ 16 y
(linear between Anny's anchors). The labels in our code assume 0.33 = 11 y and 0.67 = 25 y. Which one AnnyOne follows is
not settled (notes, slide 10, If asked). Avoid years; compare with the holdout adults instead.

**Verdict so far:** no effect from a 2-epoch sampler. [Guessing] a synthetic-to-real gap plus an old-age bias.

**Next:** lowest priority: a 10-epoch run with its own control and the real-photo read (section 5, item 12); if
nothing, a thesis limitation.

**Open, new on 16 Oct: adult men get female-looking bodies on real photos.** Seen in the real-image demo (job
3581312, 3.2), in v5 **and** in the fixed model, so the neck fix neither causes nor removes it.
- [Certain for the 8 images the student sent; by eye] Adult men on the real photos get female-looking bodies:
  breasts, narrow shoulders. Example: the male tennis players.
- [Certain] It is not the 3DPW gender-label bug (3.5): v5 never saw 3DPW, and the exp6 fine-tune trained on AnnyOne
  only.
- [Certain, from the log] The demo prints the predicted phenotypes of every detection (`[shape]` lines):
  - gender on the four photos: 0.46-0.76 (median 0.60-0.62) in both models;
  - gender on the 8 3DPW frames: 0.32-0.73 (median 0.54-0.56).
  - Anny's male end is 0 (3.5), so no one on the photos is predicted near the male end.
  - The lines repeat per detection crop and are not matched to people, so these are ranges, not a count of people.
- [Likely] These gender values are what the eye sees.
- Cause: unknown. [Guessing] the shape head's phenotypes do not carry over from synthetic to real images. Age shows
  the same pattern: predicted age is 0.57-0.67 on real photos (above), mid-range like gender.
- Not checked: predicted gender on the synthetic holdout. The trace reports a per-phenotype shape error and bias
  (`diag_holdout_trace.py:27`), so the v5 trace log 3570222 [Likely] has a gender line. Ask for it; it costs no job.
  A small gender bias there would point to the synthetic-to-real gap. Priority: after the 3DPW labels.

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
1. **Gender reversed.** `smpl_to_anny.py:1008` sets `1.0 if gender == 'm' else 0.0` (at `64266dc`; since v2.3 this is
   the default `--gender_convention makehuman`, and `anny` gives the fix, 4e), but Anny's male end is 0 (rest
   stature 1.753 m at 0 vs 1.632 m at 1; 3575369:24, :313). Every man is labelled 1.000, every woman 0.000 (:225-226).
   Refit with flipped gender: men better in 72/73 samples, joints 19.2 -> 15.3 mm. Women: joints worse for most
   (13.7 -> 16.3 mm), surface closer for 27/31 (:316-317). So the fix is confirmed only for men.
2. **Missing actor.** 407 of 960 two-person images have one valid actor without a label; all 407 because that actor was
   not fitted on that frame (per-person stride phase) (:228-229).
3. **Missing labels.** Train has 4,451 label images where the fits give 6,437 (:31); [Likely] cause: missing image
   files (rebuild below). Six train sequences have none:
   `courtyard_arguing_00`, `courtyard_backpack_00`, `courtyard_basketball_00`, `courtyard_bodyScannerMotions_00`,
   `courtyard_box_00`, `courtyard_capoeira_00`; `courtyard_captureSelfies_00` is partial, 319 of 401 (:25, :63-70).
   Validation: `courtyard_basketball_01` not built (319 images; 2,458 of 2,777; :26, :33, :89). All 162 frames of
   `outdoors_crosscountry_00` are dropped by the < 30% visibility filter (:96); they are already outside the 2,777, and
   a rebuild cannot recover them.

**By design, not bugs:** neck02/neck03/head at rest in 100% of labels; age 0.55; head turn kept only through neck01:
overall 9.5 vs 20.2 deg in SMPL (:223-224). Train and validation keep most of it (18.5 vs 21.2; 19.1 vs 19.3); test
loses most (6.3 vs 20.0; cause unknown).

**Script grade "OVERALL FAIL"** (:343-346). The IDs below are `check_3dpw_conversion.py`'s own grade IDs (V0-V5,
I1-I4), unrelated to the V0-V4 evaluation settings of section 1. It comes from: check V4c (limb-length ratios),
forearm 0.879, thigh 1.119 overall, failing only in the test split (train passes with 0.965 / 1.035) (:214-215, :335);
and check V4e (people with a joint over 100 mm), 5.2% in train = FAIL there, 3.2% overall = WARN (:218, :337). Cause
of the test-split limb ratios: unknown. All checked test people are men labelled 1, so a link to the gender defect is
possible, not shown (notes, slide 6, If asked: Limb lengths).

**Label rebuild (job 3579266): the same counts, so it did not recover the missing labels.** Output
`threedpw_anny_full/labels_rebuilt` (b395a10):
- train 4,451 images, 5,183 people (3579266:25); the six train sequences still have 0 images (:1-6);
  captureSelfies_00 still 319 (:7).
- validation 2,458 images (:39); `courtyard_basketball_01` still 0 (:27); crosscountry 0, its 162 frames dropped by the
  visibility filter (:34, :39).
- test 9,722 (:65); total 16,631 label files, 20,028 people (:67). These match the old labels (3575369:31, :33, :35).
- The "unconverted tasks (no .npz yet)" list is empty (:68): all 87 fits exist.

What this says:
- [Certain] The old explanation "the label builder ran before all fits finished" is wrong (section 6).
- [Certain] The builder skips any frame whose image file is missing, and does not count it
  (`build_threedpw_anny_labels.py:123-125`: `if not os.path.isfile(img): continue`). The counters for the other drop
  reasons (error, visibility, box size) are 0 in train (3579266:25).
- [Likely] The image folders of those 8 sequences (six train, captureSelfies_00, basketball_01) are missing or
  incomplete on the cluster, from an incomplete unzip of `/ds-av/public_datasets/3DPW/original/imageFiles.zip`. The
  extract step unzips only if `data/3DPW/imageFiles` does not exist yet (`slurm/data/run_3dpw_list.sh:36`), so a broken
  first unzip is never repeated [Certain about the code; the broken unzip itself is not checked]. Check command: 4c.

**Image re-extraction (16 Oct) and second rebuild (job 3581320, log pending).** The image-folder check is done:
the image folders of the 8 sequences were re-extracted. File counts after extraction:

| sequence | images | sequence | images |
|---|---|---|---|
| courtyard_arguing_00 | 765 | courtyard_box_00 | 1,041 |
| courtyard_backpack_00 | 1,262 | courtyard_capoeira_00 | 435 |
| courtyard_basketball_00 | 468 | courtyard_captureSelfies_00 | 480 -> 750 |
| courtyard_bodyScannerMotions_00 | 1,257 | courtyard_basketball_01 (validation) | 958 |

- [Certain] captureSelfies_00 was incomplete (480 of 750 files). For the other seven, the counts before extraction
  are not in this file.
- [Certain, arithmetic] basketball_01's 958 images give about 319-320 every-third frames (958 / 3), close to its
  319 missing validation labels (above).
- Then the labels were rebuilt again, job 3581320 (log `logs/3dpw-labels-3581320.out`, not read yet).
  - [Likely] Written into `threedpw_anny_full/labels_rebuilt` with the 4c command. Without `OUT=`, the script
    writes into the production `threedpw_anny_full/labels` (`run_build_labels.sh:18`). The log names the folder:
    each split's summary line ends in `-> <folder>/index_<split>.txt` (`build_threedpw_anny_labels.py:143-146`).
    Check it first (4c).
  - Expected [Likely]: train ~6,437 (was 4,451), validation ~2,777 (was 2,458), test 9,722 unchanged. Together
    18,936: the label images the current fits imply (check I4's `expected_label_images`, as quoted in the
    converter section of `docs/experiments_plan.md`).
  - If train stays at 4,451, missing images were not the cause (4c says what to ask next).
  - Do not confuse these counts with the ones expected after the re-fit (4e): the v23 labels should have fewer
    images, ~16,900, with about the same number of labelled people, because both actors then share one image.
- The image cause stays [Likely] until that log is read.

**Next, in this order:** (1) read the rebuild log 3581320, then re-check the rebuilt labels (4c). (2) Re-fit with the
converter fix, which is on `main` (v2.3: gender by Anny's convention, one shared frame grid for both actors): all 87
tasks into a **new** folder, rebuild, re-check, steps 0-5 in 4e (section 5, item 7). (3) Only then train or test
anything on 3DPW (the final model, section 5, item 7).

### 3.6 Pilot noise (read every 2-epoch result against this)

The exp3 control (3571958) and exp4 control (3571960) are identical configs. They differ by **7.6 mm PVE** (76.3 vs
68.7; 3571958:104, 3571960:320), 1.9 deg global head (13.6 vs 11.7) and 0.8 recall points. Over six 2-epoch v5-recipe
arms: PVE 68.7-76.3, recall 88.6-90.2, global head 11.3-13.7 (`docs/experiments_plan.md:413-419`). So one pilot
resolves only ~10 mm PVE, ~3 deg, ~2 recall points. `train.py` sets no random seed (the trace's `--seed` only picks
which training images exp6 samples). Restarting a fine-tune alone costs ~2.4 mm (notes, slide 14, If asked). The exp5
L1 control, which is also exp6's control, sits just above that range (PVE 76.9, 3573521:98).

---

## 4. Pending jobs

All commands run on the cluster in `/netscratch/najib/multi-hmr`, after `git pull`; every block below starts with the
`cd`.

**Done on 15 Oct (no longer pending):** exp6 on v5 (job 3579014), exp6 on v10 (job 3579015), label rebuild (job
3579266). Results in 3.2, 3.3 and 3.5. **Done on 16 Oct:**
- the exp6 v5 compare panels were looked at (4a; result in 3.2);
- the real-image demo with the fix ran (job 3581312, 4b; result in 3.2, "Real images");
- the image-folder check is done and the 8 folders were re-extracted (4c; counts in 3.5).

**Still pending, in this order:**
1. The log of the second label rebuild, job 3581320 (4c). Expected train ~6,437, validation ~2,777, test 9,722:
   18,936 label images in all, the count the current fits imply, as in the converter section of
   `docs/experiments_plan.md`.
2. The 3DPW re-fit with the converter fix (v2.3 flags, on `main`): steps 0-5 in 4e, one at a time. Start each step
   only after the previous one has finished, and steps 2 and 3 only after the step before them passed its checks.
   Not started.
3. Then the final model: a v9-style fine-tune from the fixed v5 with the depth fix, the bone fix and the fixed 3DPW
   labels (section 5, item 7). No script or command exists for it yet (NEEDS CODE).

Also open: the x20 depth run (4d; the command was given on 14 Oct, the submission was never confirmed).

**Status block.** `squeue -u najib` lists only queued and running jobs. This block also shows finished jobs and their
outputs (`;` so that a missing folder does not stop the rest):
```
cd /netscratch/najib/multi-hmr; sacct -u najib -S 2026-10-14 --format=JobID,JobName%28,State,Elapsed; ls -lt logs | head -20; ls results/experiments/exp1_depth_anchor/; ls -l logs/3dpw-labels-3581320.out
```
What to look for: job names `exp1-depth-anchor` and `3dpw-labels` (job 3581320) in `sacct` (State `COMPLETED`,
`FAILED`, `TIMEOUT`, ...); `long_pc_dist20` under `exp1_depth_anchor/`; the rebuild log
`logs/3dpw-labels-3581320.out` (read it as in 4c). If `sacct` prints an error (accounting may be off on this
cluster [Guessing]), use the `ls` lines alone. **Resubmit a job only if both `sacct` and its output folder show
nothing.** If the folder exists but the job failed, ask for its `.out` and `.err` before resubmitting.

### 4a. exp6 compare panels (done early 16 Oct; v5 panels seen)

The student sent all 12 v5 panels (holdout images 579860-579871). Result [Certain, from the panels]: thin-stalk necks
in the top row (without the fix), normal width in the bottom row (correction, zero training), in every panel where a
neck is visible. Reading, caveats and the figure `docs/figures/neck_before_after_v5.jpg`: 3.2, "Visual check".
Nothing more is needed for the deck.

Optional, low priority (the real-image demo, 4b, is done). List the fine-tuned checkpoint's panels and the v10
panels (a "No such file" line only means that folder is not there):
```
cd /netscratch/najib/multi-hmr && ls results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/holdout_ft/vis/; ls results/experiments/exp6_bone_fix/anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R/compare/
```
- The v5 panels show the **zero-training** correction (step b), where the head still points worse (global head 19.1
  deg). The fine-tuned checkpoint's own trace panels (head 14.4 deg) are [Likely] in
  `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/holdout_ft/vis/` (the trace of the fine-tuned
  checkpoint runs with `--n_vis ${N_VIS}`, `exp6_bone_fix.sh:292`; this folder was never listed). They would show the
  neck and head of the model that would actually be used.
- The 12 v10 panels (3579015:49; metacarpal correction) were not sent. Only the hands differ between their rows.

### 4b. Real-image demo of the fine-tuned checkpoint with the fix (done 16 Oct, job 3581312)

Ran on 16 Oct as job 3581312 (log `logs/demo-compare-3581312.out`) by the `DEMO_MODELS` route: v5 00099 next to
`exp6_v5_bonefix_00001`, on the 4 photos and the 8 3DPW-test frames. Result, caveats and figures: 3.2, "Real
images". Nothing more is required for the neck; optional: a correction-only demo of v5 with the exp6
`bone_fix.json`, to separate the correction from the fine-tune on real images (section 5, item 2).

What the log confirms [Certain]:
- The copy `models/multiHMR/exp6_v5_bonefix_00001.pt` exists: `resolve_ckpt` resolved the name to
  `exp6_v5_bonefix_00001`, and that model loaded with `bone_fix: checkpoint args` (3581312:418-427). Its "not found"
  fallback warning would go to stderr (`run_demo_compare.sh:136`, `>&2`), so to `logs/demo-compare-3581312.err`, not
  the `.out`; the `.err` the student uploaded has no "not found" line (only pip and library warnings).
- `DEMO_MODELS` reached the job: the log has only two `MODEL:` lines (:14, :418), not the five default models.
- The fix was re-applied from the checkpoint args (`[bone_fix] ON, 4 bone(s)`, :427, :607). v5 has none (:18,
  :191).
- All four output folders are `ok` (:832-837): `demo_out_anny_s2_shape_v5_00099_photos/` and `_3dpwtest/`,
  `demo_out_exp6_v5_bonefix_00001_photos/` and `_3dpwtest/`. The job finished within the default time limit (its
  SUMMARY is printed).
- The montage commands printed at the end (:839-847) label each folder with its model name and run as printed
  (login node).

To re-run, e.g. for the 10-epoch fine-tune of section 5, item 3: first `cp -n` its checkpoint to
`models/multiHMR/` under a new name, then put that name in `DEMO_MODELS`. The command that ran:
```
cd /netscratch/najib/multi-hmr && DEMO_MODELS="anny_s2_shape_v5|00099 exp6_v5_bonefix|00001" sbatch slurm/demo/run_demo_compare.sh
```
- `DEMO_MODELS` replaces the model list (`run_demo_compare.sh:101-104` from `fbda1b4` on).
- Output folders are named `demo_out_<run>_<epoch>_<set>/`. A re-run of the same model writes into the same folder,
  so move the old one first if it must be kept.
- The four-block route of the earlier version (edit `MODELS` in the script, commit, sbatch) is not needed.
- Fingers: a run with `--relax_hands 0` NEEDS a copy of the demo call (3.3).

### 4c. Image folders re-extracted (16 Oct); label rebuild 3581320 pending (pending item 1)

**Done on 16 Oct:**
- The image-folder check. The first label rebuild (job 3579266) had given the same counts as the old labels (3.5).
- The re-extraction of the image folders of the 8 affected sequences. Counts in 3.5; captureSelfies_00 went from 480
  to 750 files.
- A second rebuild, submitted as job 3581320. Its log `logs/3dpw-labels-3581320.out` has not been read yet.

To re-count the folders (login node, seconds; `courtyard_jacket_00` is a control with 412 labelled images,
3579266:12; the last number is the count of sequence folders, 60 if all are there, 3575369:25-27):
```
cd /netscratch/najib/multi-hmr && for s in courtyard_arguing_00 courtyard_backpack_00 courtyard_basketball_00 courtyard_bodyScannerMotions_00 courtyard_box_00 courtyard_capoeira_00 courtyard_captureSelfies_00 courtyard_basketball_01 courtyard_jacket_00; do echo "$s $(ls data/3DPW/imageFiles/$s 2>/dev/null | wc -l)"; done; ls data/3DPW/imageFiles | wc -l
```

**Reading the log of 3581320**, in this order:
1. **Output folder.** Each split's summary line ends in `-> <folder>/index_<split>.txt`
   (`build_threedpw_anny_labels.py:143-146`). It should be `threedpw_anny_full/labels_rebuilt`.
   - [Likely] the job was submitted with the command below.
   - Without `OUT=`, the script writes into the production `threedpw_anny_full/labels` (`run_build_labels.sh:18`):
     the labels v9/v10 trained on. If the summary lines show `threedpw_anny_full/labels/index_...`, stop and tell
     the student before anything else: those labels were overwritten. The 3579266 copy in `labels_rebuilt` has the
     same counts as the old labels (3.5) [Likely the same files: same fits, same builder].
2. **Counts.** Expected [Likely]: train ~6,437 (was 4,451), validation ~2,777 (was 2,458; +319 from
   `courtyard_basketball_01`), test 9,722 unchanged (3575369:31, :33, :35).
   - Crosscountry is already outside the 2,777 and stays at 0.
   - The six train sequences and `courtyard_basketball_01` should no longer show 0 images, and captureSelfies_00
     should show more than 319.
   - The "unconverted tasks" list should stay empty.
3. **If train stays at 4,451**, missing images were not the cause. Ask for
   `ls data/3DPW/imageFiles/courtyard_arguing_00 | head -3` (file names, for example) before doing anything else.

The rebuild command given in the 15-Oct version of this section (~1 min of CPU, limit 30 min):
```
cd /netscratch/najib/multi-hmr && OUT=/netscratch/najib/multi-hmr/threedpw_anny_full/labels_rebuilt sbatch slurm/data/run_build_labels.sh
```
If the counts match, re-check the new labels (not run yet):
```
cd /netscratch/najib/multi-hmr && LABELS=threedpw_anny_full/labels_rebuilt OUT=results/diagnostics/check_3dpw_conversion_rebuilt sbatch slurm/diagnostics/check_3dpw_conversion.sh
```
- Training scripts hard-code the old folder (`slurm/train/train_teacher_v9_mixed.sh:55`,
  `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`). Never overwrite `labels/`.
- The rebuild does **not** fix gender or the missing actor; both are decided inside the fit (the re-fit with the
  converter fix, 4e).
- If a re-extraction is ever needed again, the full `unzip -n -q ... -d data/3DPW` command is in the 15-Oct copy of
  this section (`git show 2ff8594:docs/handover.md | grep "unzip -n -q"`). `-n` never overwrites existing files.

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

### 4e. Command cookbook, and the 3DPW re-fit (pending item 2)

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
exp6 on v9 (zero training; run it when a v9 fine-tune is planned: the v5 fine-tuned fix passed, 3579014:121; the
head guard is expected to fire as on v5, so let it continue; expect the zero-training verdict to FAIL on global head
as on v5, which does not matter here, since only its `bone_fix.json` is needed; ~20-30 min, not measured: v9 00062 has
no stored no-fix dump, so the job traces one; never run, the env vars are the ones documented in the script header):
```
cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt TRAIN=0 ON_GLOBAL_RISE=continue sbatch slurm/experiments/exp6_bone_fix.sh
```
Its `bone_fix.json` lands in `results/experiments/exp6_bone_fix/anny_teacher_v9_mixed_00062_n200_s0/`.

Longer fine-tune with the fix (section 5, item 3; never run). 10 epochs, the same recipe as the 2-epoch run, plus a
matched 10-epoch control without the fix (the exp5 L1 recipe; the 2-epoch control is not a fair reference for 10
epochs). Training ~1 h each at the measured ~12.7 images/s (3579014:66), plus traces; not measured at 10 epochs.
Control first:
```
cd /netscratch/najib/multi-hmr && ROTMAT_LOSS=l1 MAX_ITER=10000 RUN=long_l1 sbatch --time=05:00:00 slurm/experiments/exp5_neck_fix.sh
```
The fix run (it re-estimates `C` on the same 200 seeded training images, ~2 min):
```
cd /netscratch/najib/multi-hmr && MAX_ITER=10000 RUN=anny_s2_shape_v5_00099_n200_s0_long CONTROL_RUN=results/experiments/exp5_neck_fix/long_l1 sbatch --time=04:00:00 slurm/experiments/exp6_bone_fix.sh
```
When both have finished, re-print the table with the 10-epoch control (login node, no GPU):
```
cd /netscratch/najib/multi-hmr && SUMMARY_ONLY=1 RUN=anny_s2_shape_v5_00099_n200_s0_long CONTROL_RUN=results/experiments/exp5_neck_fix/long_l1 bash slurm/experiments/exp6_bone_fix.sh
```
Read: neck02, head, girth, global head, PVE, recall in the fine-tuned column against the control. Last checkpoint
`results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0_long/train/checkpoints/00009.pt`.

Re-print a finished exp6 table without a GPU (login node):
```
cd /netscratch/najib/multi-hmr && SUMMARY_ONLY=1 RUN=anny_s2_shape_v5_00099_n200_s0 bash slurm/experiments/exp6_bone_fix.sh
```
```
cd /netscratch/najib/multi-hmr && SUMMARY_ONLY=1 RUN=anny_v10_scratch_mixed_00290_n200_s0_metacarpal2_L-metacarpal2_R-metacarpal3_R bash slurm/experiments/exp6_bone_fix.sh
```

**3DPW re-fit with the converter fix: steps 0-5 (pending item 2; section 5, item 7).** The commands are copied
unchanged from `docs/experiments_plan.md`, section "3DPW converter fix (v2.3 flags)" (`plan :593-843`), which also
gives the background, the guards, the runtime estimate and the full pass criteria. What is on `main`:
- `scripts/data/smpl_to_anny.py` v2.3 has two new flags, `--gender_convention` (default `makehuman` = old: men 1.0;
  `anny`: men 0.0, women 1.0) and `--frame_grid` (default `per_person` = old; `shared`: every actor on the frames
  with `frame % 3 == 0`). The defaults reproduce the old fits, so the labels v9/v10 trained on stay reproducible
  (the brief's flag rule, section 10). Each new `.npz` records `gender_convention`, `frame_grid` and version `2.3`.
- `slurm/data/run_smpl_to_anny_task.sh` reads `OUT_ROOT` (default `threedpw_anny_full`) and `REFIT_FLAGS` (default
  empty, appended to the fitter call). Three guards: a non-empty `REFIT_FLAGS` with the old folder is refused; an
  empty `REFIT_FLAGS` with a new folder is refused unless `ALLOW_DEFAULT_FLAGS=1`; in a new folder an existing
  `.npz` is skipped only if it records the same flags, otherwise the task fails (delete that `.npz` or choose another
  `OUT_ROOT`).
- `scripts/data/check_refit_task.py` with `slurm/data/run_check_refit_task.sh`: the post-check of step 2.

**Run the steps one at a time.** Start each step only after the previous one has finished, and steps 2 and 3 only
after the step before them passed its checks. `sbatch` prints the job id; `squeue -u najib` lists only queued and
running jobs, so a job it no longer shows has ended. Analyse each log only after the student says "done".

0. Update the cluster copy and check it:
   ```
   cd /netscratch/najib/multi-hmr && git pull && grep -c REFIT_FLAGS slurm/data/run_smpl_to_anny_task.sh; grep -c gender_convention scripts/data/smpl_to_anny.py; grep -c ALLOW_DEFAULT_FLAGS slurm/data/run_smpl_to_anny_task.sh; ls scripts/data/check_refit_task.py slurm/data/run_check_refit_task.sh
   ```
   Go on only if all three numbers are above `0` and both files are listed. Otherwise the cluster copy is old: ask
   for the `git pull` output and run the line again. With the old task script, steps 1 and 3 would print
   `already converted - skipping` for every task without fitting anything (all 87 `.npz` exist in
   `threedpw_anny_full`), and step 4 would build from an empty `threedpw_anny_v23`.
1. One check task: `courtyard_captureSelfies_00`, train, array indices 9 and 10 (p0 and p1). Two actors of mixed
   gender, and the strongest stride-phase case (56 of 401 images with both actors). ~25 min (estimated).
   ```
   cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=9-10 slurm/data/run_smpl_to_anny_task.sh
   ```
   Wait until `squeue -u najib` no longer lists the job and both logs, `logs/smpl2anny-all-<jobid>_9.out` and
   `_10.out` (the job id `sbatch` printed), contain a `done: ...` line. A `FAILED` or `ERROR` line instead: stop. If
   the job is gone but a log has no `done:` line, Slurm killed it (time limit, memory, node): ask for the matching
   `.err` file or `sacct -j` with that job id, and stop. Then check in each log:
   - `extra flags (REFIT_FLAGS): '--gender_convention anny --frame_grid shared'` and `frame grid: shared`;
   - the gender line shows 0.0 for the man and 1.0 for the woman;
   - the summary line `fitted N frames; joint error mean X mm, max Y mm` gives X at about 16-20 mm (the man may come
     out lower, the woman ~2 mm higher). Do not grade the PER-JOINT table against 16-20 mm: pelvis, hips, neck and
     shoulders sit at ~27-38 mm by rig definition; only a row far above its value in 3575369's check I4 is a
     warning;
   - `done: ... (N s)`: N divided by the fitted frames is the first measured s/frame.
2. Post-check of that sequence (CPU, seconds). Submit it only after both step-1 tasks have finished; earlier it stops
   with `ERROR: no new .npz ...` (exit 2) or checks only one actor. It compares with the old fits in
   `threedpw_anny_full`.
   ```
   cd /netscratch/najib/multi-hmr && SPLIT=train SEQ=courtyard_captureSelfies_00 sbatch slurm/data/run_check_refit_task.sh
   ```
   It passes when (`logs/check-refit-<jobid>.out`): the log ends with `RESULT: PASS` and the job exits 0;
   `[p0+p1] NEW: ... both valid in 3DPW but only one fitted: 0/N` while the `OLD` line shows a large share; the
   man's label value is ~0.0001 and the woman's ~0.9999. Not a gate: the man's joint error should be lower than
   before (the woman's may rise ~2 mm); if it is not, note it, report it and go on. Read the paired line
   `[pK] joint error mean/median, M frames fitted by both: old ... -> new ...` first; if M is under ~30 or the line
   is missing, use the unpaired `[pK] joint error mean/median, all fitted frames: old ... -> new ...`.
3. All 87 tasks. Submit only after step 2 passed, and never while the step-1 job is still queued or running (tasks 9
   and 10 would fit the same people again into the same temporary file). Tasks 9 and 10 are skipped, because their
   `.npz` exist with the same flags (`already converted with the requested flags - skipping`). A task to redo needs
   its `.npz` deleted first. ~35-41 GPU hours, ~5-6 h of wall time if 8 GPUs are free (estimated, not measured; the
   longest task ~65-75 min, all well under the 4 h limit).
   ```
   cd /netscratch/najib/multi-hmr && OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
   ```
   Logs `logs/smpl2anny-all-<arrayjobid>_<task>.out`. When the array is finished,
   `ls /netscratch/najib/multi-hmr/threedpw_anny_v23/*/*_p*.npz | wc -l` must give 87. Optional:
   `cd /netscratch/najib/multi-hmr && SEQ=all sbatch slurm/data/run_check_refit_task.sh` should print PASS for every
   sequence.
4. and 5. The labels (CPU, ~1 min) and the conversion check on v23 (~19-20 min), in one block, only after the step-3
   array has finished and the count above is 87. The check waits for the label job (`--dependency=afterok`). It must
   not start earlier: it only tests that `index_*.txt` exists, and the builder writes `index_train.txt` before the
   other splits are done. Keep `OUT=` on the check, otherwise it overwrites the 3575369 results.
   ```
   cd /netscratch/najib/multi-hmr && L=$(FITS=/netscratch/najib/multi-hmr/threedpw_anny_v23 OUT=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --parsable slurm/data/run_build_labels.sh) && echo "labels job ${L%%;*}" && LABELS=threedpw_anny_v23/labels FITS=threedpw_anny_v23 OUT=results/diagnostics/check_3dpw_conversion_v23 sbatch --dependency=afterok:${L%%;*} slurm/diagnostics/check_3dpw_conversion.sh
   ```
   If the label job fails, the check stays pending with reason `DependencyNeverSatisfied`; cancel it with `scancel`
   and the check's job id.

**Reading the v23 check** (full criteria: the plan's "Pass criteria"):
- `OVERALL (V0-V5): FAIL` again is expected, as in 3575369 (limb ratios in test, outliers in train). It is not a
  failed re-fit. Judge the re-fit by the checks below.
- Check I1 (gender labels): men 0.000, women 1.000; the men's flip gain is gone (median gain <= 0, was +3.4 mm).
  Women: reported, not graded (the joints never confirmed the female end for them).
- Check I3 (unlabelled actors): "valid actor unlabelled" ~0 of the two-person images (was 407 of 960).
- Check I4: 87 `.npz`, versions {'2.3': 87}.
- Checks V1-V3 not worse than in 3575369 by more than ~1 mm (V1 body joints median 18.7 / p95 24.5 mm; V2 surface
  18.9 / 18.3 mm). Men should improve; women may lose ~2-3 mm in V1.
- Label counts: ~16,200-18,936 label images, about 16,900 (train ~5,900, validation ~2,650, test ~8,300), with about
  22,753 labelled people. Read this against the 18,936 the old fits imply (the upper bound, and what the rebuild
  3581320 should give, 4c), not against the old index's 16,631. **Fewer images** than 18,936 with about the same
  people is expected, not a loss: on one shared grid both actors of a two-person sequence sit on the same image
  instead of on two.

**Using the v23 labels.** The training and 3DPW evaluation scripts set
`LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"` with a plain assignment
(`slurm/train/train_teacher_v9_mixed.sh:55`, `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`,
`slurm/eval/run_eval_3dpw.sh:39`, `run_eval_3dpw_teacher.sh:39`, `run_eval_3dpw_test.sh:61`), so `LABELS=...` in
front of `sbatch` is silently ignored. Edit that line in a copy of the script, then check the `--mix_3dpw_labels`
path in the job log. 3DPW numbers against the old labels cannot be compared with numbers against v23: re-evaluate
the baseline on v23 first.

---

## 5. Next steps, in priority order

Analyse any new log only after the student says "done". The pending chain, in order (section 4, 0a steps 3-5):
(1) the rebuild log 3581320 (item 5; minutes, can go first on any day); (2) the 3DPW re-fit, steps 0-5 in 4e
(item 7); (3) the final model (end of item 7; no script yet). Item 3 can run before (3), or next to (2) if GPUs are
free. Item 6 depends on the supervisor's answer to slide 17.

1. **exp6 panels: done (early 16 Oct).** The 12 v5 compare panels show thin-stalk necks without the correction and
   normal width with it (3.2, "Visual check"; figure `docs/figures/neck_before_after_v5.jpg`, deck slide 8). So "the
   thin neck seen in renders is the flip" can be said for the synthetic holdout renders, with the caveats there (no
   fine-tune, depth bug still in, synthetic only). Nothing left to do here.
2. **Real images with the fix: done (16 Oct, job 3581312).** The demo of the fine-tuned checkpoint
   `models/multiHMR/exp6_v5_bonefix_00001.pt` next to v5, on 4 photos and 8 3DPW-test frames.
   - Result: v5's thin, stalk-like necks have normal width with the fix [Certain for the 8 images the student sent;
     by eye, no metric] (3.2, "Real images"; figures `docs/figures/neck_before_after_3dpw_bar.jpg` and
     `..._bar_office.jpg`, deck slide 8).
   - So "the thin neck on real images goes away in the fine-tuned fixed model" can be said, by eye; that the
     correction rather than the 2-epoch fine-tune does it stays [Likely].
   - Optional: a correction-only real-image demo would remove the fine-tune confound: v5 00099 with
     `--bone_fix results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json` and no fine-tune.
     `demo.py` takes a JSON path there (`demo.py:117-120`, `:819-822`; v5 builds `Model`, 3581312:19, so the fix
     applies). `run_demo_compare.sh` passes no `--bone_fix`, so this NEEDS a copy of its demo call
     (`run_demo_compare.sh:152-171`) with that flag and a new `--out_folder` (the default name
     `demo_out_anny_s2_shape_v5_00099_<set>/` would overwrite v5's folders). Not run, not asked for yet.
   - Optional: `--relax_hands 0` for the hands (3.3).
   - The predicted gender on the holdout, for the new female-body observation (3.4), costs no job: ask for it.
3. **Longer fine-tune with the fix.** 10 epochs, with a matched 10-epoch control without the fix (commands in 4e,
   "Longer fine-tune with the fix"). Why: at 2 epochs the evidence is only "no sign of a PVE cost" (68.8 vs 76.9
   for the control; one run, no seed, inside the noise of 3.6), and recall passed by 0.3 points (89.4 vs a limit of
   89.1); a longer run against its own control is the next evidence. [Guessing] the head residual (11.2 deg) and global head
   (14.4) shrink further. Keep the fix: it is saved in the checkpoint args and re-applied on load.
4. **Two-sided correction for compensated chains (v10's left arm).** **NEEDS CODE** in `bone_fix.py`:
   `R_l <- K_l^T R_l K_l` (plan :489, :561), so that shoulder01.L and upperarm02.L (166, 180 deg; 3579015:68-69) can be
   corrected without breaking upperarm01.L between them. New behaviour behind a default-off option. Only worth it if
   v10 stays in the thesis. The chain-preserving head correction (plan :558) is [Likely] no longer needed for v5
   (head 11.2 deg after the fine-tune).
5. **3DPW images, then rebuild.** The image check and the re-extraction of the 8 folders are done (16 Oct; counts
   in 3.5). The second rebuild was submitted as job 3581320; its log is not read yet (pending item 1).
   - Read it as in 4c: output folder first, then counts against train ~6,437 / validation ~2,777 / test 9,722.
   - If they match, re-check (4c). This would bring back the 31% missing train labels without a re-fit [Likely].
   - The label build after the re-fit (item 7) uses the same images, so they are ready for it.
6. **Supervisor's decision (slide 17).** Fine-tune v9 now with `--gt_center_joint person_center` and x5, or first fix
   the labels (item 7) and then fine-tune? The student proposes labels first, because 3DPW is 30% of each batch and the
   men would keep learning the wrong body. Expected from v5: size ~0.84, root ~0.5 m, cost ~2.6 mm PVE and ~1.6 recall
   points (notes, slide 17, If asked: Option 1). **The v9 fine-tune script does not exist yet (NEEDS CODE).** Recipe:
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
   - Add: `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5 --mask_3dpw_hands 1`. Recommended now that
     the v5 fix passed after fine-tuning (3579014:121), in both arms:
     `--bone_fix results/experiments/exp6_bone_fix/anny_teacher_v9_mixed_00062_n200_s0/bone_fix.json` (estimate it on v9
     first, 4e; the v5 JSON belongs to v5). Without it the neck flip stays.
   - Output: `--save_dir /netscratch/najib/multi-hmr/results/experiments/exp7_v9_depth/v9e62_pc_dist5 --name train`.
     Never `logs/anny_model/anny_teacher_v9_mixed`: `train.py` keeps only the `--nb_max_ckpt` highest epoch numbers of
     a folder (`train.py:400-410`), so a run written there deletes checkpoints (its own low epochs, or old v9 ones).
   - Control arm, same script and weight: `--gt_center_joint root` (still x5), output `.../exp7_v9_depth/v9e62_root_dist5`.
   - Labels (`--mix_3dpw_labels`): slide-17 option 1 ("now") = `/netscratch/najib/multi-hmr/threedpw_anny_full/labels`
     (what v9 trained on; gender reversed, 31% of train missing) or `labels_rebuilt` once item 5 brings the missing
     sequences back (gender still reversed; the second rebuild 3581320 should bring them back, log not read yet).
     Option 2 ("labels first") = `/netscratch/najib/multi-hmr/threedpw_anny_v23/labels` after item 7.
   - Time: the v9 script's comment expects ~8-10 min per 1,000-iteration epoch at batch 4
     (`train_teacher_v9_mixed.sh:16`; an estimate, never measured), so 10 epochs ~1.3-1.7 h plus evaluations. Not
     measured with 3DPW batches; the exp6 fine-tune on AnnyOne alone took 315 s per epoch (12.7 images/s,
     3579014:66). Say "about 1 to 2 hours". `--time=04:00:00`.
   - Read: trace the last checkpoint of each arm with `diag_holdout.sh` and a `TAG` (4e); root error, slope, size,
     PVE, recall vs the control. The training log's 3DPW-val numbers are against converted labels only.
7. **3DPW labels: re-fit with the converter fix (pending item 2; after the meeting, before any 3DPW claim).**
   - The converter fix is on `main` (v2.3): two default-off flags in the fitter (`--gender_convention anny`,
     `--frame_grid shared`), `OUT_ROOT` / `REFIT_FLAGS` and three guards in the array job, and the post-check
     `scripts/data/check_refit_task.py` (`docs/experiments_plan.md`, section "3DPW converter fix (v2.3 flags)").
     Nothing to write.
   - Run steps 0-5 in 4e ("3DPW re-fit"), one at a time, waiting for each to finish: check the cluster copy, one
     check task, its post-check, all 87 tasks into `threedpw_anny_v23`, then the labels and the conversion check.
   - The gender refit confirmed the flip only for men; check the women in the new check I1 (gender labels).
   - If the meeting moves to Monday, the re-fit (~5-6 h wall time, not measured) could start before it.
   - The images are re-extracted (item 5), so the label build after the re-fit finds them. The builder skips frames
     without image files.

   **Then the final model (pending item 3): a v9-style fine-tune from the fixed v5, with the depth fix, the bone fix
   and the fixed 3DPW labels** (after the re-fit and its check pass). **No script or command exists for it yet
   (NEEDS CODE).** It is v9's recipe (v5 + 30% 3DPW-train, `slurm/train/train_teacher_v9_mixed.sh`), but starting
   from the fixed v5 and trained on the fixed labels, with the depth fix on.
   - Write a copy of that script, e.g. `slurm/train/train_v11_fixed_mixed.sh` (section 2 pattern: full-file heredoc,
     check, then a separate commit block). Never edit or rerun the v9 script itself: it starts from v5 and writes to
     `anny_teacher_v9_mixed`.
   - In the copy, set `CKPT=/netscratch/najib/multi-hmr/models/multiHMR/exp6_v5_bonefix_00001.pt` (the v9 script
     builds `CKPT` from v5 at `:47`), the 2-epoch fixed v5. Use item 3's 10-epoch checkpoint instead if it has run
     and is no worse than its control.
   - Keep the script's `--load_only_backbone 0` and pass no `--bone_fix`: `train.py` then inherits the checkpoint's
     bone fix (`train.py:1008-1023`). The training log should print `[bone_fix] ON, 4 bone(s)` with "inherited from
     --pretrained" (`bone_fix.py:161-162`).
   - Labels: in the copy, set `LABELS="/netscratch/najib/multi-hmr/threedpw_anny_v23/labels"` (the v9 script assigns
     it plainly at `:55`, so an environment override is ignored; 4e, "Using the v23 labels"), and check the
     `--mix_3dpw_labels` path in the job log. Plus `--mask_3dpw_hands 1` (3DPW fingers are rest pose).
   - Depth fix: add `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5` (x5, as in item 6 and on slide
     17; x20 only if 4d clearly beats it). The slide-17 question to the supervisor is about the order (v9 now or
     labels first); [Likely] both options include the depth fix (the slide says so only for option 1). If the
     supervisor says otherwise, follow that.
   - The deck describes a different start. Slide 17 and its notes frame the fine-tune as one of v9: epoch 62 is "the
     planned start", "about 1 to 2 hours of training", with a bone fix "estimated on v9 first" (notes, If asked:
     "Option 1", "Bone flips in v9 and v10"). The final model here starts from the fixed v5 instead, inherits its bone
     fix (no v9 estimate needed) and may run much longer ("Length" below). If option 2 is chosen, the student should say
     this at the meeting. If the supervisor wants the deck's version, item 6's recipe with the option-2 labels is it.
   - A new `--save_dir` and `--name`, never `logs/anny_model/anny_teacher_v9_mixed` (item 6 says why).
   - Length: v9 ran up to 150 epochs (`--max_iter 150000`), and the student was distilled from epoch 62. Set the length
     from the deadline (section 10). The v9 script estimates ~8-10 min per epoch (never measured), so 60 epochs
     would be ~8-10 h.
   - Read the result:
     - V1 trace (4e) and per-bone dump: neck02, head, girth;
     - 3DPW-test against the v23 labels;
     - the demo next to v5 + fix (4b);
     - compare with v9 00062.
8. **Depth slope stuck at 0.3-0.44.** Measure the training-set depth distribution properly (so far only batch means of
   ~2.5-3.2 m) and the fix run's slope on training images. Both **NEED CODE**: extend
   `scripts/experiments/measure_batch_ages.py` (it already draws real training batches and applies a copy of
   `prepare_gt`'s visibility filter) to also print GT root-depth percentiles; and pass `--select train_sample`
   through `slurm/diagnostics/diag_holdout.sh` (the trace supports it, `diag_holdout_trace.py:527`; the wrapper passes
   only `--ckpt --n --out`). [Guessing] it needs more unfrozen blocks or a higher LR on the depth head.
9. **Cost of the fix.** PVE paired on the same people; recall on people whose head is visible ([Likely, untested]: the
   recall drop is a cohort effect from moving the detection cell to the head). **NEEDS CODE**, no tool exists: paired
   PVE = join two runs' `holdout_trace/per_person.csv` on `idx` + `gid` for `variant == V1_eval_masked`; head-visible
   recall needs a head-visibility column in `diag_holdout_trace.py`. Report root error next to PVE in the thesis.
10. **Pilot noise.** Set a seed, or repeat the key run, before reading any 2-epoch difference under ~10 mm. **NEEDS
    CODE**: `train.py` has no seed at all. Add `--seed` (default -1 = off, old behaviour) and, at the top of `main()`
    (`train.py:899`), seed `random`, `numpy` and `torch`, and give the training DataLoader a seeded generator.
11. **People very close to the camera.** Count misses (4 of 12 rendered holdout images, v5 and the depth fix).
12. **Children (lowest priority).** The exp2 script's own gate says not to run longer (`exp2_children.sh:38`; both
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
13. **Docs and thesis.** Fix the stale lines in section 7; thesis draft 74 -> 70.5 mm (V1; the draft is outside the
    repo).
14. **Lower priority:** v10 shape drift (only epochs 290-293 traced). First list what exists (only the last 10 epochs
    are kept per run): `ls /netscratch/najib/multi-hmr/logs/anny_model/anny_v10_scratch_mixed/checkpoints/` and
    `ls /netscratch/najib/multi-hmr/models/multiHMR/ | grep v10`; then trace each with `diag_holdout.sh` (4e).
    Students' depth bug (not measured); output-neutral fix of the 6D init (only matters for new heads; plan :931);
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
- "exp6 will leave the head near 15 deg: case (b)" (the 15-Oct expectation, [Likely] then). Measured: head 12.4 deg
  without training, 11.2 after the fine-tune; `FINE-TUNED FIX: PASS` (3579014:103, :121). The zero-training verdict did
  fail, on global head (19.1 vs 15.0, :120).
- "Neck: found, not yet fixed." Superseded: a fixed output correction plus a 2-epoch fine-tune removes the flip and
  restores the girth (3.2). Still not claimed: a PVE gain (single run), the flip's origin.

**Fingers**
- "Fingers need a hand-crop branch." Not shown; only test-time zoom was tested.
- "The index-base flip is part of the finger gap." It adds ~4.7 deg to both the predicted and the rest-pose numbers.
- "Fingers are under-trained / a supervision or capacity problem." A guess.
- "Fingers: tested, nothing helped" and "the prediction is worse than no prediction." True only without the bone fix:
  correcting the flipped hand bones lowers the finger error 17.3 -> 8.2 deg on v5 (3579014:114) and 32.0 -> 17.8 on v10
  (3579015:80).
- "metacarpal1.L adds ~4.7 deg to the finger numbers" covers that one bone only; with finger4-1.R also corrected the
  drop is 9.1 deg.

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
- "The label builder ran before all fits finished, so a rebuild recovers the missing labels." Wrong: the rebuild
  (job 3579266) gives the same counts (train 4,451, validation 2,458, test 9,722; :25, :39, :65), and its "unconverted
  tasks" list is empty (:68), so all fits exist. [Likely] cause: the builder skips frames without an image file
  (`build_threedpw_anny_labels.py:123-125`), and the image folders of 8 sequences were incomplete on the cluster. They
  were re-extracted 16 Oct (captureSelfies_00: 480 -> 750 files); whether that brings the labels back is the second
  rebuild, job 3581320 (3.5, 4c).

**Numbers and workflow (corrected while checking this file against the logs)**
- "Job 3571601 is an earlier sampler run." It is measurement only (`SKIP_TRAIN=1`, 3571601:54): age index and batch
  ages, natural 2.14% -> balanced 9.91% (:35, :49-51); no training.
- "exp1: the slope did not move toward 1." The rule's slope clause printed `yes` for x5, x20 and even the 2-epoch
  control. The slope is 0.30 with the switch alone and 0.38-0.44 with the stronger depth loss (baseline 0.33), far
  from 1.
- "v10 is a ViT-L from scratch." Pretrained DINOv2 backbone, fresh heads.
- "v9 exists only in `models/multiHMR/`." Only epoch 62 is known to be missing under `logs/`.
- "6.7 images/s." Measured ~12 in most runs; 6.7 on one node.
- "3-5 NaN skips per 2,000 in every run." 1-7 per 2,000 (13-23 per 10,000).
- "Validation after the rebuild: ~2,777 minus crosscountry." The 2,777 already excludes crosscountry.
- "The visible long-neck look is the flip [Certain]." On 15 Oct only the measured girth / PVE_neck were shown. Since
  16 Oct the look is shown on 12 synthetic holdout renders (exp6 panels, 3.2), and by eye for the fine-tuned
  checkpoint on the 8 of the 12 demo images the student sent (job 3581312 ran 4 photos and 8 3DPW-test frames; 3.2,
  "Real images") [Certain for those 8 images]. On real images, that the correction rather than the fine-tune does it
  stays [Likely].

**Still untested claims; keep the tag**
- "The image folders of 8 3DPW sequences are missing on the cluster (incomplete unzip)." Re-extracted on 16 Oct;
  captureSelfies_00 went from 480 to 750 files, so at least that one was incomplete [Certain]. Whether this brings
  the labels back: the rebuild log 3581320 (4c).
- "The recall drop is a cohort artefact."
- "On real images the correction, not the 2-epoch fine-tune, removes the thin neck." The demo (job 3581312) compares
  v5 with the fine-tuned checkpoint, so both changed. The holdout control without the fix keeps girth 0.461 (3.2).
  Test: the optional correction-only demo (section 5, item 2).
- "Men get female-looking bodies on real photos because of a synthetic-to-real gap in the shape head." [Guessing]
  (3.4).

---

## 7. Stale docs: outdated lines elsewhere in the repo

Trust this file over these. Fix them when there is time (one commit per doc).

**`README.md`**
- :114 "numbers look normal while the overlays look wrong" (real-photo demos use box placement).
- :115 whole neck row (weak supervision, redundant chain, long neck = depth defect).
- :116 "Under-trained" (a guess; missing the crop/region-norm nulls, the flipped hand bones and the exp6 result).
- :117 "about 11 years", "0.55 (about 20 years)" (code anchors); exp2 null missing.
- :119 "No improvement is claimed until its holdout trace has been re-run" (it has).
- :126 "Future work (not measured)" (both measured, both null). :128 "Nothing from them has been measured yet."
- :211-214 fit description omits the reversed gender and the per-person stride.
- :227 "4,451 train / 2,458 validation" without the expected 6,437 / 2,777, and without the cause: a rebuild gave the
  same counts (3579266), [Likely] image files missing.
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
- :8-9 and :991-997 "Nothing in this plan has been run." (The section "3DPW converter fix (v2.3 flags)", :593-843,
  is current; its step 0 says "After that merge": that is the 16-Oct merge, done.)
- :27 exp5 as WIP `bd7cfa3` (merged in `637dc5c`); ancestors "not yet traced" (traced).
- :28 and :428 exp6 "uncommitted" (it is in `d902e00` / `81dbdcb`). No exp6 results (use 3.2 here).
- :94, :986 size baseline 0.67 (trace: 0.63). :150-152 exp3 redundant-chain hypothesis (refuted).
- :229 heading "WIP bd7cfa3". :235 "per-bone split never measured". :241-250 H3/H4 "untested" (refuted).
- :252 "Leading hypothesis [Likely]" (flip now measured). :273-278 6D-init sentence only half corrected.
- :368-370 "neither the flip nor the L1 trap measured". :422-426 decpose surgery (superseded by exp6).
- :258, :569 cite scratch files (`flip_joints.py`, `test_bone_fix.py`) that are not in the repo.
- :931 line refs from an older commit (6D init now at `model.py:641`). :939-958 week schedule is history.
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
- Deck and notes: up to date (18 slides, rebuilt from `build.js` on 16 Oct). Section 9 lists the slides, how to check
  a copy, and the lines that go stale when the pending results arrive.

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
| c6419ce | WIP handover rewrite and deck source (superseded by `64ac9e3`) |
| 64ac9e3 | handover rewritten (15-Oct afternoon version); deck source in `docs/deck_src/` |
| 2ff8594 | 15-Oct evening update: exp6 and rebuild results in this file, deck and notes (17 slides) |
| 8577bb2 | neck figure `docs/figures/neck_before_after_v5.jpg` from the exp6 compare panels (v5, correction only) |
| fbda1b4 | `run_demo_compare.sh`: model list overridable with `DEMO_MODELS` (4b) |

**The 16-Oct merge.** On 16 Oct the work branch `claude/gallant-dijkstra-bnlowv` was fast-forwarded into `main`, so
`main` now has a straight history up to this file. The branch's commits from `5e9f416` on are work-in-progress steps
of one piece of work. Most of their messages say "not for main yet"; that was true only before the merge, and they
need not be read one by one. (`a01eb5b`, the real-image figures, is a normal commit.) Together they brought:
- this file, 16-Oct version (the neck panels, the real-image demo, the image re-extraction, the pending chain);
- the 18-slide deck: `docs/deck_src/build.js`, and the notes and pptx rebuilt from it (section 9);
- the real-image neck figures `docs/figures/neck_before_after_3dpw_bar.jpg` and `..._bar_office.jpg` (`a01eb5b`; job
  3581312, 3.2);
- the 3DPW converter v2.3 (4e): `--gender_convention` and `--frame_grid` in `scripts/data/smpl_to_anny.py` (default =
  old behaviour); `OUT_ROOT`, `REFIT_FLAGS` and three guards in `slurm/data/run_smpl_to_anny_task.sh`; the re-fit
  post-check `scripts/data/check_refit_task.py` and `slurm/data/run_check_refit_task.sh`; `.gitignore` exceptions,
  so that files under `scripts/data/` and `slurm/data/` are tracked like any other;
- the section "3DPW converter fix (v2.3 flags)" of `docs/experiments_plan.md`, the source of the re-fit steps in 4e.

**Getting this file.** On the cluster: `cd /netscratch/najib/multi-hmr && git pull`. Then
`grep -c '^## 0a\. Tomorrow' docs/handover.md` must print `1`, and `grep -c ALLOW_DEFAULT_FLAGS docs/handover.md` a
number above `0` (top of file). To give it to ChatGPT, download
`https://raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/docs/handover.md`
(save the page as `handover.md`) and attach it. The deck (18 slides) is at
`https://raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/docs/supervisor_update_2026-10-16.pptx`.

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
| 3579014 | exp6_bone_fix | exp6 on v5: fix on neck02, head, metacarpal1.L, finger4-1.R; zero-training FAIL (global head 19.1 vs 15.0, :120), fine-tuned PASS (:121) |
| 3579015 | exp6_bone_fix | exp6 on v10: metacarpal2.L/2.R/3.R, no training; PASS (:86) |
| 3579266 | 3dpw-labels | label rebuild into `threedpw_anny_full/labels_rebuilt`: the same counts as the old labels (:25, :39, :65, :67). Log name `logs/3dpw-labels-3579266.out` (dash, not underscore) |
| 3581312 | demo-compare | real-image demo: v5 00099 vs `exp6_v5_bonefix_00001` (the exp6 fine-tuned checkpoint; `[bone_fix] ON, 4 bone(s)` from its args, :427, :607), `DEMO_MODELS` route, 4 photos + 8 3DPW-test frames. By eye, on the 8 of these 12 images the student sent: thin necks with v5, normal width with the fix (the fixed side also had the 2-epoch fine-tune; 3.2). Log name `logs/demo-compare-3581312.out` |
| 3581320 | 3dpw-labels | second label rebuild, after the image folders of 8 sequences were re-extracted on 16 Oct. Files per folder after extraction: courtyard_arguing_00 765, backpack_00 1,262, basketball_00 468, bodyScannerMotions_00 1,257, box_00 1,041, capoeira_00 435, captureSelfies_00 480 -> 750, basketball_01 958. Log `logs/3dpw-labels-3581320.out` not read yet; expected train ~6,437, validation ~2,777, test 9,722 (4c) |

**Experiments, one line each:** exp1 depth anchor (placement much better, slope not, likely cost; rule failed);
exp2 children sampler (null); exp3 global head loss (null); exp4 fingers crop + region-norm (both null); exp5
geodesic loss (flip did not move); exp6 bone correction (v5: flip removed and neck girth back after a 2-epoch
fine-tune with the fix, fine-tuned PASS, zero-training FAIL on global head, no sign of a PVE cost; real images by
eye, 8 of 12 images seen: neck normal width with the fix, job 3581312; v10: three metacarpals fixed without
training, PASS).

---

## 9. Deck

**Files:** `docs/supervisor_update_2026-10-16.pptx` (18 slides), `docs/supervisor_talk_notes_2026-10-16.md` (spoken
part + "If asked:" part per slide), `docs/figures/depth_before_after_579860.jpg` and `_579862.jpg`,
`docs/figures/neck_before_after_v5.jpg` and `docs/figures/neck_before_after_3dpw_bar.jpg` (the two pairs on slide 8).
The two-pair figure `docs/figures/neck_before_after_3dpw_bar_office.jpg` is not on a slide: keep it for the thesis or
for questions. Its baked-in label "After: v5 + neck fix" leaves out the 2-epoch fine-tune; caption it "fine-tuned 2
epochs with the fix" when shown (same for `neck_before_after_3dpw_bar.jpg`; slide 8 crops the label off).

**Source:** `docs/deck_src/build.js` (pptxgenjs) and its theme helper `docs/deck_src/apply_theme.js`; instructions in
`docs/deck_src/README.md`. No setup beyond `cd docs/deck_src && npm ci && node build.js`, on a machine with node >= 18
and internet access for `npm ci` (probably the student's laptop; the cluster may lack node). Checked on 15 Oct: the
rebuild gives the same slides, theme, media and layouts as the published pptx, and byte-identical notes. The notes
and the pptx on `main` were rebuilt from the current `build.js` on 16 Oct (~05:05) and match it. Notes are generated
from `build.js`; never edit the `.md` alone.

**Before the meeting** (whichever date holds): open the pptx once in PowerPoint and check slides 1, 2, 5, 6, 7, 8, 9,
15, 16 and 17 for text that wraps one line longer. They were only rendered with LibreOffice. Slide 8's header bars
have an 11-pt second line sized to fit the bar; slide 15's Wording card already ends at the card's bottom edge in the
LibreOffice render.

**Check a copy** (after `git pull`; the pptx itself must also be opened once):
```
cd /netscratch/najib/multi-hmr && unzip -l docs/supervisor_update_2026-10-16.pptx | grep -c 'ppt/slides/slide[0-9]*\.xml$'; grep -c "before and after the fix" docs/deck_src/build.js; grep -c runForBus docs/supervisor_talk_notes_2026-10-16.md; grep -o "check pending\|image folders will show it\|I will check, re-extract and rebuild\|check is still to do\|I have not checked that yet\|empty or short" docs/deck_src/build.js | wc -l
```
The current deck prints `18`, `2`, `1`, `0` (18 slides; slide 8's title and its notes; the slide 8 "If asked" line on
the four unseen demo images; no line that still calls the image check undone). Anything else is an older copy: ask
the student to `git pull` again. If no current copy can be had before the meeting, hand-edit the pptx in PowerPoint
from the slide list below (slide text and speaker notes), then port the changes into `build.js` after the meeting and
rebuild, so the `.md` notes match again.

**Slides** (current titles; quotes from `build.js`, which is the reference if a wording here differs):
1. Multi-HMR with Anny: what this week found (agenda). "Neck: two bones turned half a turn; a pilot fix passes (one
   run)." and "Fingers: half the error was two flipped bones. Children: open."
2. Where we started: three visible failures (neck, fingers, children). Neck card: "A pilot fix restores it (one run;
   on real images by eye only)." Fingers card: "half the error was two flipped bones".
3. What I did: test each stage on its own (holdout + manifest).
4. What is not the cause: the structural checks pass (+ overfit warnings). The slide card and its notes call the
   overfit rise in translation and 2D losses (1.67 -> 1.89; 58 -> 75) "probably the first sign of the depth bug".
   Section 1 rates this a guess: those are single-batch prints, and the first-half vs second-half averages fall
   instead. If asked, the student should say exactly that. The next time the deck is rebuilt, change "probably" to
   "possibly" in `build.js` (in the slide 4 card and in the slide 4 `N()` notes).
5. 3DPW to Anny: how the labels are made.
6. 3DPW labels: joints and surface pass, three defects. "Labels missing" bullet: "6 train sequences, part of
   captureSelfies_00, val basketball_01. Images re-extracted; second rebuild job 3581320 pending." The notes give the
   3579266 counts, say the old guess "the builder ran before the fits finished" is wrong, and give the re-extraction
   (captureSelfies_00 480 -> 750 files) and the pending rebuild.
7. Neck: two half-turned bones, and a pilot fix. Table v5 before | fix, no training | fix + 2-epoch fine-tune |
   control (no fix): neck02 175.6 / 4.5 / 4.7 / 175.7 deg, head 177.9 / 12.4 / 11.2 / 177.9, neck thickness 0.46 /
   1.00 / 1.00 / 0.46, neck-skin error 27.3 / 10.8 / 10.7 / 27.9 mm, body PVE 70.5 / 71.3 / 68.8 / 76.9, head
   direction 12.0 / 19.1 / 14.4 / 13.8. The fix with no training "fails its 15° limit" on head direction, "likely a
   per-person head offset"; with the fine-tune it "passes the fine-tune checks, each within a tolerance of its
   control". "Caveat: one run, no fixed seed (identical runs differ by up to 7.6 mm PVE), so no PVE gain is
   claimed." "Open: why the bones flipped. Next slide: the fix alone on a holdout image; fix + fine-tune on a real
   3DPW frame." "Next: a longer fine-tune with the fix (slide 16)."
8. Neck: before and after the fix. Two before | after pairs in one row, yellow rings on two necks per pair.
   - Left: synthetic holdout 579862, from `neck_before_after_v5.jpg`; header bars "Before (v5) / as trained" (red)
     and "After (correction) / no training" (teal). Caption "Synthetic holdout 579862 (correction only, no
     training)".
   - Right: 3DPW test `downtown_bar_00` frame 701, from `neck_before_after_3dpw_bar.jpg` (job 3581312); header bars
     "Before (v5) / real image" and "After (fixed model) / fix + 2-epoch fine-tune". Caption "3DPW test frame (real,
     never trained on), downtown_bar_00".
   - Below: "Left: same model, no training; on the neck, only neck02 and the head are corrected. Right: fixed model,
     fine-tuned 2 epochs.", then the thickness 0.46 -> 1.00 on the 100-image holdout (job 3579014) and "Real-image
     demo: job 3581312". "Limits:" the head points 19° off without the fine-tune (12° before), 14° after it; holdout
     bodies are still too small (the separate depth bug); "Real images judged by eye only: no neck ground truth, no
     metric."
   - Notes, main part: thin stalks with v5, normal width with the fix; measured on the holdout (0.46 to 1.00), by eye
     only on real images: "clearest here and on an office frame, and also visible on two more 3DPW sequences and
     three photos"; head direction 14 degrees after the fine-tune, against 12.
   - Notes, "If asked":
     - the right column differs from v5 by the correction and by two epochs of training; "On the holdout, the same
       2-epoch fine-tune without the fix keeps thickness 0.46, so the correction is the likely cause; not tested on
       real images." The demo log shows the fix on for four bones;
     - "I have looked at 8 of these 12 images so far; the other four, the runForBus, sitOnStairs and fencing frames
       and one photo, not yet.";
     - the female-looking bodies on men in both models: not the 3DPW gender defect (v5 never saw 3DPW), cause
       unknown, a guess (3.4);
     - bodies still small or misplaced in depth in both models, and a figure in a painting detected as a person;
     - the hands differ too (two hand bones turned back) and were not judged (slide 9);
     - all twelve holdout panels show the thin neck and its fix wherever a neck is visible.
9. Fingers: about half the error was two flipped bones. 17.3 deg before, 8.3 with the fix after the fine-tune (8.2
   with no training), 15.1 at rest pose before the fix. The notes give the v10 metacarpal result (PVE 74.1 -> 69.9,
   fingers 32.0 -> 17.8).
10. Children: child-like in training, adult on real photos.
11. The hidden bug: head anchored, pelvis targeted.
12. Depth fix: bodies drawn much closer to true size (figure 579862).
13. Depth fix: much better size, near vs far still weak (table). Note: "No control was run at ×5 or ×20."
14. Four experiments: results (+ noise floor).
15. Corrections made, and still to make. The Wording card names the README and the findings as still carrying last
    week's neck story. That is narrower than section 7, which also lists the plan's neck lines (`plan :150-152`,
    `:252`); the slide is left as it is.
16. Still open, and what I test next (7 items). Item 1, "Bone flips: confirm the pilot fix": "Neck: a longer run;
    real images by eye only (slide 8). v10's left arm: two-sided fix, not built yet." Item 3: "No result yet for the
    ×20 run at 10 epochs." Item 5: "Both v5 and the depth fix miss them (4 of 12 holdout renders): not the fix.
    Likely detection; not counted." Item 7, "3DPW labels: three defects (slide 6)": "Images of the missing sequences
    re-extracted 16 Oct; second rebuild job 3581320 pending. Then gender (m → 0, f → 1), one shared frame grid,
    re-fit all 87 tasks, rebuild, re-check. Proposed: only then use 3DPW." Notes: "the 8 real images I have seen so
    far".
17. What I claim today, and what I do not (+ the question to the supervisor). Claimed: "in one pilot run, a fixed
    correction plus a 2-epoch fine-tune restores the neck and passes the fine-tune checks"; two flipped hand bones
    were half the finger error. Not claimed: "a proven neck fix (one run, no seed; head direction 14.4° vs 12.0°) or
    why the bones flipped". The question box adds: "v9 also has the neck flip (measured at epoch 149) and needs its
    own bone fix; v10 has a similar left-arm flip." Notes: "in the 8 of its 12 images I have seen, judged by eye
    only"; "about 1 to 2 hours of training", not measured with 3DPW batches; "75.0 at epoch 62, the planned start".
    So the slide and notes describe the fine-tune as starting from v9 00062 (1-2 h, with a v9 bone fix estimated
    first), while the final model of section 5, item 7 starts from the fixed v5 and needs no v9 estimate; if option 2
    is chosen, the student should say this at the meeting.
18. Backup: before and after, holdout image 579860.

**Results still to come: the lines they make stale.** When a result arrives, edit these lines in `build.js` and
rebuild (or hand-edit the pptx as above).

| result | slide text | speaker notes | what changes |
|---|---|---|---|
| label rebuild 3581320 (4c; pending item 1) | 6 ("Labels missing": "... second rebuild job 3581320 pending."), 16 item 7 (first sentence) | 6 (main and If asked, "Missing labels"), 16 (main and If asked, "3DPW labels"), 17 (If asked, "3DPW labels") | the rebuilt train / validation / test counts (expected ~6,437 / ~2,777 / 9,722); if train stays at 4,451, that the image cause is ruled out (4c) |
| 3DPW re-fit and the v23 check (4e; pending item 2) | 6 (the gender and two-person defects), 16 item 7 ("Then gender ..., re-fit all 87 tasks, rebuild, re-check."), 17 (question box) | 6 (If asked, "Gender"), 16 (main: "Then gender, frame grid and a re-fit of all 87 tasks, before any 3DPW test."; If asked: "about 40 GPU hours ... 5 to 6 hours with 8 GPUs", and "otherwise old and new fits would mix without a warning", which is already out of date: since v2.3 the task script refuses instead of mixing, 4e), 17 (If asked, "Option 2") | check I1 / I3 results, the v23 label counts (expected ~16,900 images), the measured runtime |
| the other four demo images (runForBus, sitOnStairs, fencing frames, one photo) | none | 8 (If asked: "I have looked at 8 of these 12 images so far ..."), 16 (main: "the 8 real images I have seen so far"; If asked: "I have looked at 8 of these 12 images so far"), 17 (If asked: "in the 8 of its 12 images I have seen") | how many of the 12 show normal necks with the fix |
| longer fine-tune with the fix (4e; section 5, item 3) | 7 ("Next:" line), 16 item 1 ("a longer run") | 7 (If asked), 16 (main: "I still need a longer run") | neck02, head, girth, global head, PVE, recall at 10 epochs against the 10-epoch control |
| `long_pc_dist20` (4d) | 13 (table note), 16 item 3 ("No result yet for the ×20 run at 10 epochs.") | 13 (main), 16 (If asked), 17 (If asked: "Five, not twenty, because ...") | root error, size, slope, PVE, recall of x20 for 10 epochs next to x5 for 10 epochs |
| the supervisor's answer (slide 17) | 16 item 7 ("Proposed: only then use 3DPW."), 17 (question box) | 16, 17 | record it in the paragraph below |
| the final model's start, if option 2 is chosen (section 5, item 7; pending item 3) | 17 (question box: "Fine-tune v9 ... with the fixed targets and depth ×5? Or first fix gender and frame grid, re-fit all 3DPW tasks, then fine-tune?") | 17 (If asked: "Option 1", with "75.0 at epoch 62, the planned start" and "about 1 to 2 hours"; "Option 2"; "Bone flips in v9 and v10", with a bone fix "estimated on v9 first") | that the final model starts from the fixed v5 (`exp6_v5_bonefix_00001`) and inherits its bone fix; its length, set from the deadline |

**The question on slide 17:** fine-tune v9 now (fixed targets, depth loss x5), or first fix the 3DPW labels, re-fit,
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
