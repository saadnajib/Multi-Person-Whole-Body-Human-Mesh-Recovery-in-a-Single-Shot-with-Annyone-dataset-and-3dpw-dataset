# Handover: project state on Saturday 17 October 2026

This file replaces the chat history. It is written for an assistant that cannot see that chat, the cluster or any
temporary files. Everything you need is either here or in this repository. **Where this file disagrees with
`README.md`, `docs/diagnostics_findings.md`, `docs/experiments_plan.md` or `docs/experiments/exp2_children.md`, trust
this file** (section 7 lists the outdated lines). The deck `docs/supervisor_update_2026-10-16.pptx` (18 slides) and
the speaker notes `docs/supervisor_talk_notes_2026-10-16.md` match this file. Section 9 lists the slides and the
lines that will go stale when the pending results arrive.

**Which version you have.** All code this file describes is on GitHub `main` as of the evening of 17 Oct
(`a0926fd` and the docs commit after it): the 3DPW converter fix (v2.3 flags and the re-fit post-check, 4e); the
options that run the converter on H100/H200 (`GPUS`, `CONDA_ENV`, `PY_VENV`, 4f); the final-model script
`slurm/experiments/exp7_final_model.sh` together with the `MODEL_ENTRY` / `EVAL_LABELS` overrides of
`slurm/eval/run_eval_3dpw_test.sh` it needs (`b188b63`; section 5, item 7; never run); the default-off flag
`--mask_3dpw_shape` (`loss.py`, `train.py`, exp7's `MASK_3DPW_SHAPE`; `a0926fd`; 3.4); the 18-slide deck with its
source `docs/deck_src/`, the figures in `docs/figures/` and the corrected children slide (section 9). A `git pull` on
the cluster gets all of it. After the pull, `ls slurm/experiments/exp7_final_model.sh`,
`grep -c EVAL_LABELS slurm/eval/run_eval_3dpw_test.sh` (above `0`) and `grep -c mask_3dpw_shape loss.py` (above `0`)
confirm the three. To check a copy of this file:
- `grep -c '^## 0a\. Tomorrow' docs/handover.md` prints `1` for the 15-Oct and later versions and `0` for every older
  one (the 14-Oct `586faad`, or the work-in-progress `c6419ce`).
- `grep -c ALLOW_DEFAULT_FLAGS docs/handover.md` prints a number above `0` only in the 16-Oct and later versions (after
  the merge; they quote the re-fit steps in 4e). `0` means an older copy, which may still call the converter fix
  unmerged, the real-image demo unseen or the deck 17 slides long. Ask the student for the current file.
- `grep -c 7,589 docs/handover.md` prints a number above `0` in the second 16-Oct version and later (with the result
  of the second label rebuild, job 3581320). `0` means the earlier 16-Oct copy, which still calls that rebuild pending.
- `grep -c 3589270 docs/handover.md` prints a number above `0` in the 17-Oct versions (the re-fit test, job
  3589270, and the H100 route). `0` means a 16-Oct copy: it still calls the re-fit test unrun, plans every job on the
  old GPUs and leaves the meeting date open.
- `grep -c 3590385 docs/handover.md` prints a number above `0` in the later 17-Oct versions (17 Oct, later: the
  first full re-fit, 51 of 87 done, 36 failed on the V100 nodes gifu/glendale, and the rerun command in 0a, step 4).
- `grep -c 'Relative Human' docs/handover.md` prints a number above `0` only in this version (17 Oct, evening: the
  children diagnosis corrected, the ranked plan, the 3DPW shape mask; 3.4, section 5 item 12). `0` means an older
  copy: it still counts the "five small people" as children predicted as adults, which is wrong.

The student should attach this file to the chat directly, not rely on you browsing GitHub. Download links: section 8.

**What happened on 16 Oct** (all in this version): the student sent the 12 exp6 compare panels (neck checked by eye
on the holdout; 3.2, 4a). The real-image demo ran (job 3581312: v5 next to the exp6 fine-tuned checkpoint with the
fix, on 4 photos and 8 3DPW-test frames; the student sent 8 of these 12 images). By eye, on those 8, the thin necks
get normal width with the fix; the fixed side also had the 2-epoch fine-tune (3.2, "Real images"). Slide 8 shows a
holdout pair and a real pair (section 9). The image folders of the 8 affected 3DPW sequences were re-extracted and a
second label rebuild ran (job 3581320; 3.5, 4c): train 6,437 images (was 4,451), validation 2,777 (was 2,458), test
9,722 unchanged, 18,936 in all, exactly what the current fits imply. So the missing-labels defect is fixed; the
reversed gender and the missing actor remain until the re-fit (4e). The rebuilt labels in
`threedpw_anny_full/labels_rebuilt` still come from the old fits: do not train on them. The converter fix and this
file were merged into `main` (section 8).

**What happened in the night of 16-17 Oct** (all in this version; details in 4e and 4f):
- The re-fit test could not start on the old GPUs: the queue estimated 22 Oct, then 26 Oct after a cancel and
  resubmit (reason `Priority`; no reservation on the cluster) [Certain, the student's `squeue --start` output].
- The H100/H200 nodes were free, but the normal env's PyTorch 2.0.1+cu117 has no code for them: jobs 3587852 and
  3587853 crashed on an H200 NVL with `no kernel image is available for execution on the device` [Certain, `.err`].
- Cloning the conda env with a newer PyTorch failed (`Disk quota exceeded`). What worked: a small venv holding only
  PyTorch 2.1.2+cu121 on top of the conda env, `PY_VENV=/netscratch/najib/venv_cu121` (`93c4eda`; 4f). It is for the
  3DPW converter only; training, demo and evaluation stay in `multihmr` on the old GPUs.
- With it the re-fit test (4e, step 1) ran on an H100 node, job 3589270 [Certain, logs]: the man now gets gender 0.0
  (old fits: 1.0 [Likely], 0b) and a mean joint error of 14.3 mm; the woman 1.0 and 15.5 mm; both on one shared
  frame grid; ~4.4 s per frame. Full re-fit job 3590385: 51 of 87 done, 36 failed on the V100 nodes gifu/glendale
  and are rerun without them (0a, step 4). Not run yet: the post-checks, labels and check, the final model (0a).

**What happened on the evening of 17 Oct** (this version; details in 3.4 and section 5, item 12):
- The children evidence was re-checked. The "five small people (skeletal height 0.6-1.14 m)" are mostly adults,
  crouching or seated [Likely, by eye]. The one real child in our photos (a boy) is predicted as an adult body
  [Certain, log]. The shape head regresses to the training median, in the synthetic tails and on real photos.
- A ranked plan: measure on Relative Human (real photos with age groups) first; then a limitation (Path A) or
  BEV-style weak labels (Path B, 1-2 weeks of code after exp7). Both are for the supervisor on Monday (0a, step 8).
- Another session added `--mask_3dpw_shape` (default off; exp7 env `MASK_3DPW_SHAPE`, default 0; commit `a0926fd`,
  not on `main` yet). On or off in exp7 is a Monday decision (3.4).
- Deck: slide 10 (children) corrected, slides 14 and 16 and notes 8, 10, 14, 16, 17 made consistent (section 9).

**Meeting date: Monday 19 October 2026** (the student, 17 Oct), with the deck `docs/supervisor_update_2026-10-16.pptx`
(18 slides; slides unchanged since 16 Oct except slides 10, 14 and 16, changed on the evening of 17 Oct with the
children correction; speaker notes updated 17 Oct, section 9). Already in the deck: the
real-image demo (slide 8), the label rebuild result (job 3581320, slides 6 and 16) and, in the notes only, the re-fit
test (job 3589270). If the full re-fit and its check finish before the meeting, their result can go in (section 9,
"Results still to come"); otherwise the student says that they are running or queued.

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
  `slurm/data/run_smpl_to_anny_task.sh` (converter v2.3, 4e; current lines are named where they matter); `slurm/env.sh`
  and again the task script (`4c2113b`, `1a5d0c3`, `93c4eda`: `GPUS`, `CONDA_ENV`, `PY_VENV`, 4f); and
  `docs/experiments_plan.md`, where the converter section was inserted at :593 (`plan` lines from :593 on are given
  as in the current file).

---

## 0. Paste this first

The student pastes this block, section 0a and section 0b into the first message as text, then attaches this file.
[Guessing] ChatGPT may only search a large attachment (this one is ~190 KB) instead of reading all of it; text in the
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
- 3DPW labels: three defects (reversed gender, a missing actor in two-person frames, missing labels). Missing labels:
  fixed on 16 Oct. Image files of 8 sequences were missing; after re-extracting them, a second rebuild (job 3581320)
  gave train 6,437 / validation 2,777 / test 9,722 label images, exactly what the fits imply. Those labels still
  come from the old fits (gender reversed), so I do not train on them. The converter fix for gender and the missing
  actor is in the repo (default-off flags). Re-fit test on one two-person sequence (17 Oct, job 3589270, H100): the
  man now gets gender 0.0 and 14.3 mm mean joint error, the woman 1.0 and 15.5 mm, both on one shared frame grid.
  Full re-fit (87 tasks, job 3590385, 17 Oct): 51 done; the 36 that landed on two V100 nodes failed in under a
  minute (a node problem, not the data) and are being rerun without those nodes.
- Fingers: flipped hand bones were a large part of the finger error (exp6 on v5: 17.3 -> 8.2 deg).
- Children: open. The shape head regresses to the training median, in the synthetic tails and on real photos (age,
  gender, height). The one real child in my photos (a boy) is predicted as an adult; the "five small people" I used
  before are mostly crouching adults. A 2-epoch age-balanced sampler pilot was inconclusive. Adult men on real photos
  get female-looking bodies (gender near the median 0.6; male = 0): the same failure. Plan: measure on Relative Human
  first, then limitation or BEV-style weak labels. A default-off 3DPW shape mask exists (decision for the final model).
Supervisor meeting: Monday 19 Oct.
Cluster: the queue for the old GPUs (V100, A100, RTX3090, RTXA6000, L40S) was estimated at 22-26 Oct. The H100/H200
run only the 3DPW converter, through a small venv with PyTorch 2.1.2 (PY_VENV). Training, demo and evaluation need my
normal env (PyTorch 2.0.1), so they wait for the old GPUs.
Thesis: my 13 Oct brief said a one-week deadline. Ask me the submission date and the chapter list of my draft
before planning anything that takes more than a day.
Finished 15 Oct: exp6 on v5 and on v10, 3DPW label rebuild. 16 Oct: the exp6 neck panels, the real-image demo with
the fix (job 3581312), the 3DPW image re-extraction and the second label rebuild (job 3581320) (results in handover
section 3). 17 Oct: the 3DPW re-fit test on H100 (job 3589270). Pending, in this order (handover 0a, 4e, 4f, 5):
(1) the post-check of that test, on the login node; (2) the full re-fit, 87 tasks, H100 route; (3) labels and the
conversion check on the new fits (v23); (4) the final model: a v9-style fine-tune of v5 with its bone fix, the depth
fix and the v23 labels (script drafted, not yet reviewed: slurm/experiments/exp7_final_model.sh). I submit it only
after the supervisor answers the slide-17 question on Monday.
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

## 0a. Tomorrow (Sunday 18 Oct) and the meeting on Monday 19 Oct, in order

Steps 3-4 can start today (Saturday 17 Oct) if the student is online: step 3 takes seconds and step 4 ~3.5-4 h on
free H100s, which leaves Sunday for step 5 and a margin before Monday.

Paste this section with section 0. Each step names the section with the details. State on Saturday 17 Oct: the
re-fit test (4e, step 1) ran and its logs were read (job 3589270, 0b); nothing later in the 3DPW chain has been
reported. The four pending items are steps 3-6, in this order.

1. Ask the student: the thesis submission date and the draft's chapters (section 10); and what has run since job
   3589270 (the post-check? the full array?). Ask for that output before giving the next step.
2. Status block (section 4, top), for the x20 depth run (4d, never confirmed) and for anything of steps 3-6 already
   submitted. **Never cancel or resubmit a queued job** (section 10); resubmit only if both `sacct` and the output
   folder show nothing. Analyse logs only after the student says "done".
3. Pending (1): the post-check of the test, **on the login node** (NumPy only, seconds). Not with
   `run_check_refit_task.sh`: that script has no partition line, so it lands in the CPU partition `batch` (216 of 216
   CPUs busy on 16 Oct) and waits. The login shell must have the conda env `multihmr` active (the prompt shows
   `(multihmr)`; the command activates it; if `conda activate` says the shell is not set up, run
   `source /netscratch/najib/miniconda3/etc/profile.d/conda.sh` once first, as `slurm/env.sh` does). Go on only on
   `RESULT: PASS`; what else to read: 4e, step 2.
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && python scripts/data/check_refit_task.py --seq courtyard_captureSelfies_00 --split train --new_root /netscratch/najib/multi-hmr/threedpw_anny_v23 --old_root /netscratch/najib/multi-hmr/threedpw_anny_full --seq_root /netscratch/najib/multi-hmr/data/3DPW/sequenceFiles --expect_gender_convention anny --expect_frame_grid shared
   ```
4. Pending (2): the full re-fit. **First run 17 Oct, job 3590385** [Certain, the student's `sacct` and check loop]:
   51 of 87 fits done (tasks 9 and 10 skipped as planned, 18 s each). The other 36 tasks all FAILED after 33-40 s,
   every one on the two V100-32GB nodes `gifu` (16) and `glendale` (20); all 51 that worked ran on other nodes
   (H200-AV serv-3312: 36, A100-PCI serv-3315: 7, RTX3090 serv-9216/9218: 4, RTXA6000-AV serv-9224: 3, H200
   serv-3346: 1). The error, in `logs/smpl2anny-all-3590385_21.err`: `smpl_to_anny.py: error: unrecognized
   arguments:` followed only by blanks; the `.out` ends `FAILED (rc=2)`. Same script and flags worked on every other
   node, so the cause is those two nodes [Certain]; how is unknown ([Guessing] the command reaches Python differently
   there, e.g. the `\` line breaks become literal blanks). Failed tasks leave no `.npz`, so a rerun refits them.
   Rerun of only the 36 failed tasks, V100-32GB dropped and both nodes excluded, 3-hour limit (the longest task
   that worked took 1 h 10 min, task 70):
   ```
   cd /netscratch/najib/multi-hmr && PY_VENV=/netscratch/najib/venv_cu121 OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --partition=H100,H100-PCI,H100-RP,H200,H200-PCI,H200-AV,A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTX3090,RTXA6000,RTXA6000-AV --exclude=gifu,glendale --gpus=1 --time=03:00:00 --array=21-30,41,42,49,53-64,66,72,73,77,79-81,83-86%12 slurm/data/run_smpl_to_anny_task.sh
   ```
   Missing before the rerun: 21-30, 41, 42, 49, 53-64, 66, 72, 73, 77, 79-81, 83-86 (mostly the long `downtown_*`
   test sequences). Done when the missing-task check of 4e, step 3 prints nothing and counts 87, and the post-check
   of every sequence ends with `RESULT: PASS` (4e, step 3; required before step 6). Never use V100-32GB for jobs
   that pass `PY_VENV` (4f).
5. Pending (3): labels and the conversion check on v23 (4e, steps 4-5), after step 4 is complete. Do not
   `sbatch slurm/data/run_build_labels.sh`: it asks for a GPU on the old partitions only (`:4-5`), so the label
   build (NumPy only, ~1 min) and everything after it would wait in the 22-26 Oct queue [Certain for the script;
   the queue date is the 16-Oct estimate]. Build the labels on the login node instead (4e, step 4). The command
   first checks that every task has its fit and builds nothing if one is missing (the builder alone would use
   whatever fits exist and give labels with missing actors):
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && M=$(awk '{print NR-1, $0}' configs/threedpw_tasks.txt | while read -r I SPLIT SEQ PERSON N; do [ -s "threedpw_anny_v23/$SPLIT/${SEQ}_p$PERSON.npz" ] || echo "task $I"; done) && if [ -n "$M" ]; then echo "labels NOT built, fits missing:" $M; else python scripts/data/build_threedpw_anny_labels.py --fits /netscratch/najib/multi-hmr/threedpw_anny_v23 --images /netscratch/najib/multi-hmr/data/3DPW/imageFiles --out /netscratch/najib/multi-hmr/threedpw_anny_v23/labels; fi
   ```
   Then the conversion check through the venv on H100 (4e, step 5; [Likely]: never run in the venv). The check
   script also asks for the old partitions by default; the 4e command passes `--partition` with the H100/H200.
6. Pending (4): the final model (section 5, item 7): a fine-tune of v5 00099 with v5's bone fix, the depth fix and
   the v23 labels, 10 epochs, plus its matched control (`ARM=control`). It trains, so it needs the normal env and
   **only the old GPUs** (4f): expect days in the queue.
   - Sunday: its script `slurm/experiments/exp7_final_model.sh` is on `main` (`b188b63`, never run). Check it against section 5, item 7 (start checkpoint, bone fix, epochs, control arm; item 7 now
     follows the script's defaults, so pick one start if the review changes it). The change to
     `slurm/eval/run_eval_3dpw_test.sh` is in the same commit: exp7's test evaluation calls that script with `MODEL_ENTRY` and
     `EVAL_LABELS` (exp7, end of the script), and the older version (`93c4eda`) ignored both, so it would
     evaluate the first model of its list (v5 00099) against the old labels in `threedpw_anny_full` and print those
     numbers in exp7's log [Certain, from the two scripts]. exp7's header (prerequisite 4) also asks for both on
     `main`. So commit the two files together, in one commit, push it to `main`, and `git pull` on the cluster.
     Check that both arrived: `ls slurm/experiments/exp7_final_model.sh` and
     `grep -c EVAL_LABELS slurm/eval/run_eval_3dpw_test.sh` (must print more than `0`).
   - Once the v23 labels exist (step 5), run its guards alone on the login node (seconds; it submits nothing; the
     label check needs the `multihmr` Python, otherwise it refuses with exit 1):
     ```
     cd /netscratch/najib/multi-hmr && git pull && conda activate multihmr && CHECK_ONLY=1 bash slurm/experiments/exp7_final_model.sh
     ```
   - **Submit it after the meeting**, if the supervisor picks "labels first" (step 9) and step 5's check passed; then
     once, and leave it queued (commands in section 5, item 7). Submitting it on Sunday would answer the slide-17
     question for the supervisor.
   - If the student wants it queued before the meeting anyway (to gain queue time; [Guessing] about a day), say so
     plainly at the meeting. If the supervisor then picks option 1, cancelling it is allowed: it is a change of plan,
     not a cancel-and-resubmit to start sooner, so it duplicates nothing (section 10).
   - Optional before it: the longer fine-tune with the fix and its control (section 5, item 3); it trains too, so the
     same queue.
   - New on the evening of 17 Oct: `MASK_3DPW_SHAPE` (default `0` = old behaviour; `1` passes `--mask_3dpw_shape 1`
     to both arms and adds `_shpmask` to the run names). Leave it at `0` until the supervisor decides (step 8; 3.4;
     section 5, item 7).
7. Before the meeting: check that the deck is the 18-slide version with the 17-Oct notes and the corrected slide 10
   (section 9, "Check a copy"). Open the pptx in PowerPoint and check slides 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16 and
   17 for text that wraps one line longer (slide 15's Wording card already ends at the card's bottom edge in the
   LibreOffice render; slides 10, 14 and 16 were re-rendered on 17 Oct evening and fit). Optional, no GPU: start the
   Relative Human download: **done 17 Oct** (`/netscratch/najib/data/Relative_Human/RelativeHuman/`, 8,741 images,
   zip verified; labels for 5,218 train / 635 val / 1,836 test images). The evaluation script and its job exist
   (section 5, item 12, step 1); submit the job after the meeting, or let it queue: `sbatch --array=0-1
   slurm/diagnostics/eval_relative_human.sh` from the repo root (old GPUs only).
8. In the meeting: the deck as it is. On 3DPW: the re-fit test (notes 6, 16, 17) and whether the full re-fit is
   done, running or queued (say which; never call the labels fixed before the v23 check passed). On training: it
   waits for the v23 labels and for the old GPUs (4f). Slide 17 describes the fine-tune as one of v9 epoch 62; the
   final model of step 6 starts from v5 00099 with v5's bone fix, so the student should say so if option 2 is chosen
   (note 17, "Option 2", now says it; section 5, item 7). Also say whether the final model is queued (by default it
   is not: step 6). On children (slide 10 and its notes; 3.4), three points:
   - Children are a measured open problem. The shape head regresses to the training median, in the synthetic tails
     and on real photos. Oversampling synthetic children is unproven (2-epoch pilot inconclusive) and has no
     mechanism for real photos.
   - The one real child in our photos is predicted as an adult (age 0.60 = the training median; height phenotype
     0.48). The earlier "five small people" were mostly crouching or seated adults.
   - Proposal: measure on Relative Human first, then decide between a limitation (Path A) and BEV-style weak labels
     (Path B, 1-2 weeks of code, after exp7). The 3DPW shape mask exists, default off; on or off in exp7 is a
     decision (with it on, exp7's 3DPW-test PVE can move for a reason unrelated to the depth and bone fixes).
9. After the meeting: write the supervisor's answers into section 9: the slide-17 question, the children path (A or
   B) and `MASK_3DPW_SHAPE` on or off (if on, update section 5, item 7 and note 17 too). "Labels first" (the
   student's proposal): finish steps 3-5, then submit the final model (step 6). "Fine-tune v9 now": section 5, item 6
   comes first and the final model is not submitted (or is cancelled, if it was queued early; step 6); item 6 trains
   too, so it also waits for the old GPUs. Steps 3-5 go on either way: they do not train, and option 2 needs them.

Done on 16 Oct, nothing to ask for: the exp6 v5 compare panels (3.2, 4a); the real-image demo with the fix (job
3581312; 3.2, 4b; figures `docs/figures/neck_before_after_3dpw_bar.jpg` and `..._bar_office.jpg`); the
re-extraction of the 8 image folders and the second label rebuild (job 3581320: train 6,437 / validation 2,777 /
test 9,722 label images, as expected; 3.5, 4c). Its labels still come from the old fits: do not train on them.
The demo also showed a new open issue in both models: adult men get female-looking bodies (3.4; since 17 Oct
evening: the same pull to the median as age). Optional: a correction-only demo without the fine-tune (section 5,
item 2).

Done on 17 Oct: the re-fit test (job 3589270; 0b, 4e step 1) and the H100 route for the converter (4f). The full
re-fit (job 3590385) made 51 of 87 fits; the 36 that ran on the V100 nodes gifu/glendale failed and are rerun (0a, 4).
Evening: the children re-check, the ranked plan and the corrected slide 10 (3.4; section 5, item 12; section 9);
the `--mask_3dpw_shape` flag by another session (commit `a0926fd`, not on `main` yet).

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
  the photos (median 0.60-0.62), and Anny's male end is 0 (3.4). It is the shape head's pull to the training median
  [Certain, log]; the cause of the pull (synthetic-to-real gap or the loss) is not ranked (3.4).
- Shape head on real photos (all 299 `[shape]` lines of demo 3581312; ~30 distinct people, nearly all adults):
  median gender 0.59, age 0.61, height 0.50 = the training medians 0.588 / 0.601 / 0.500 (3569948:8-18) [Certain].
  Proportions ~0.82 (training 0.725) and weight 0.59 (0.506): fixed offsets. On synthetic data the same pull shows in
  both tails: holdout GT age 0.6-0.8 -> 0.57 (3570222:89); in-sample babies 0.015 -> 0.10 (3569950:61).
- exp6 on v10 (job 3579015, no training, metacarpal2.L/2.R/3.R): PASS. PVE 74.1 -> 69.9, PA-PVE 55.6 -> 51.1,
  fingers 32.0 -> 17.8 deg; global head 12.9 and recall 89.0 unchanged. The PVE drop is certain only for these 100
  images (no training, so no run-to-run noise); [Likely] on other images. shoulder01.L (166) and upperarm02.L (180)
  stay flipped: they need a two-sided correction, not implemented.
- Other half-turns: v5 metacarpal1.L 180 deg, finger4-1.R 178.8 (found by exp6 on training images); v10
  shoulder01.L 166, upperarm02.L 180, metacarpal2.L/2.R/3.R 180; v10 finger2-1.L 178.8, finger5-2.L 180 (training
  images; not corrected).
- Fingers: predicted 17.3 deg vs 15.1 at rest pose; with the exp6 fix 8.2; test-time crop 17.2; region-norm loss 17.5
  vs 17.4 (control).
- Children: sampler raises children per batch 2.1% -> 9.7-11%; 2-epoch pilot inconclusive (10 in-sample children
  PVE 48.5 vs 53.1 control; baby age floor 0.100 -> 0.102). Real photos: the one real child (the boy in
  `360_F_392252267...jpg`) gets age 0.60, height phenotype 0.48, an adult rest body [Certain, log]. The old "five
  small people, age 0.57-0.67" were mostly crouching or seated adults [Likely, by eye]: not child evidence (3.4).
- 3DPW labels: body joints 18.7 mm median (p95 24.5); surface 18.9 / 18.3 mm (lower bound). Gender reversed (men:
  19.2 -> 15.3 mm when flipped); 407 of 960 two-person images miss an actor; train 4,451 of 6,437 images labelled,
  validation 2,458 of 2,777. Rebuild (job 3579266): the same 4,451 / 2,458 / 9,722; cause: image files missing.
  The image folders of the 8 sequences were re-extracted on 16 Oct (captureSelfies_00: 480 -> 750 files). Second
  rebuild, job 3581320, into `labels_rebuilt`: train 6,437 images / 7,589 people, validation 2,777 / 3,314, test
  9,722 / 11,850; 18,936 label files, 22,753 people, exactly what the current fits imply. Missing labels: fixed.
  Still the old fits (gender reversed, missing actor): not for training. The re-fit test ran (next bullet); the full
  re-fit (job 3590385) made 51 of 87 fits, the other 36 failed on two V100 nodes and are rerun (0a, step 4).
- 3DPW re-fit test (converter v2.3, `--gender_convention anny --frame_grid shared`; job 3589270; train
  `courtyard_captureSelfies_00`; node serv-3341, an H100, through `PY_VENV`) [Certain, logs]:
  - man (p1, task 10): gender 0.0 (old fits 1.0 [Likely]); joint error mean 14.3 / max 15.8 mm; 231 frames; 1,013 s;
  - woman (p0, task 9): gender 1.0 (old 0.0 [Likely]); 15.5 / 34.5 mm; 226 frames; 978 s; one hard stretch, frames
    396-498 at 20-28 mm per frame in the first pass (the other frames 11-20 mm); her 34.5-mm worst frame comes after
    the shared-body pass and is higher than any first-pass frame (28.3), frame unknown (4e, step 1);
  - the old values are [Likely], not read per person: v2.2's default convention gives m -> 1.0, f -> 0.0, and every
    sampled man was 1.000 and every woman 0.000 in 3575369:225-226; the post-check prints the per-person values;
  - shared grid: every fitted frame is a multiple of 3; 225 frames fitted for both; 457 person-frames in all, as in
    the old fits; ~4.4 s per frame;
  - the old fits of this sequence: median 18.8 / p95 21.9 / max 35.7 mm, both actors pooled (3575369:70). Not a
    paired comparison, and it sets the new means (14.3, 15.5) against an old median; the post-check prints mean and
    median, old and new, per person and paired (4e, step 2; not run yet).
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
  v9/v10 batches at 30%. The rebuild of 16 Oct (job 3581320, `labels_rebuilt`, same old fits) has 6,437 / 2,777 /
  9,722 (3.5).

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
  (`scripts/diagnostics/README.md:32`). `diag_crop_camera.py:136-141` measures it on the **posed** body, so crouching
  or sitting shortens it: it is not body size (3.4). The rest-pose value is computed there (:141) but not printed.
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
| `scripts/data/check_refit_task.py`, `slurm/data/run_check_refit_task.sh` | post-check of a v2.3 re-fit against the old fits (gender values, shared frame grid, actors fitted on the same frames; 4e, step 2). NumPy only: run the `.py` on the login node; the `.sh` has no partition line and lands in the CPU partition `batch` |
| `slurm/experiments/exp1..exp6_*.sh` | one job script per experiment; the header of each explains why, what, how to read |
| `slurm/diagnostics/*.sh` | `diag_holdout.sh`, `diag_neck_bones.sh`, `check_3dpw_conversion.sh`, others |
| `slurm/data/run_build_labels.sh`, `run_smpl_to_anny_task.sh` | 3DPW label build and the per-person fit array job (`OUT_ROOT`, `REFIT_FLAGS` and three guards since v2.3, 4e; `GPUS`, `CONDA_ENV`, `PY_VENV` since 16-17 Oct, 4f). The label build is NumPy only but asks for a GPU on the old partitions (4e, steps 4-5) |
| `slurm/env.sh` | shared job setup, sourced inside every container command: `conda activate ${CONDA_ENV:-multihmr}`, then the venv `$PY_VENV` if set (4f); caches on netscratch; `cd` to the repo |
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
| `--mask_3dpw_shape 1` | loss.py, train.py | new 17 Oct evening (another session; commit `a0926fd`, not on `main` yet): no shape L1 on the 8 converter constants (age, muscle, weight, cupsize, firmness, african, asian, caucasian) for 3DPW-Anny people; gender, height, proportions stay. Default 0. exp7: `MASK_3DPW_SHAPE` (3.4) |
| `--rotmat_loss geodesic` | train.py | exp5 geodesic rotation loss (default `l1`) |
| `--bone_fix <json>` | train.py, trace, demo.py | exp6 constant correction `R <- R·C` for listed bones; saved into checkpoint args |
| `--dump_bones 1`, `--dump_bone_names all` | trace | per-bone error table (verdicts FLIP_TWIST, FLIP_SWING, ...) |
| `--relax_hands 1` | demo.py | sets bones whose name has "finger"/"thumb" to rest. **Does not touch metacarpals** (model.py:165-177) |
| `--anchor_to_box 1` (default) | demo.py:785 | real-photo demos place each body from its detection box, so they hide the depth bug |
| `--gender_convention anny`, `--frame_grid shared` | smpl_to_anny.py | 3DPW converter v2.3: Anny's gender convention (m -> 0.0, f -> 1.0) and one frame grid for both actors. Passed through `REFIT_FLAGS` of `run_smpl_to_anny_task.sh` (4e) |

**Cluster.** DFKI Slurm. Repo at `/netscratch/najib/multi-hmr`, conda env `multihmr` (PyTorch 2.0.1+cu117: old GPUs
only; the H100 route for the converter is in 4f), enroot container, shared setup `slurm/env.sh`. Logs:
`/netscratch/najib/multi-hmr/logs/<jobname>_<jobid>.out` and `.err` (label build: `3dpw-labels-<jobid>.out`).
Checkpoints: `logs/anny_model/<run>/checkpoints/<epoch>.pt` (only the last 10 kept) and
copies in `models/multiHMR/<run>_<epoch>.pt`. v9: use the `models/multiHMR/` copies for both epochs (section 1).
Epoch 62 is not under `logs/anny_model/anny_teacher_v9_mixed/checkpoints/` (job 3571597 failed on that path); that
folder exists, but its contents were never listed. Job status: the status block at the top of section 4
(`squeue -u najib` lists only queued and running jobs). Home folder `/home/najib`: a personal quota of about 10 GB,
full on 16 Oct [Likely]; `Downloads` (4.4 GB) was moved to `/netscratch/najib/home_moved/`. Keep caches on netscratch
(`slurm/env.sh` does that inside jobs; section 10).

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

**Corrected on the evening of 17 Oct.** Earlier copies (and `findings :222-223`) used "five small people (skeletal
height 0.6-1.14 m), predicted age 0.57-0.67" as real children predicted as adults. That is wrong: most of them are
adults. This section replaces that reading. The plan is in section 5, item 12.

**In short:**
- [Certain] The shape head regresses to the training median. It shows in both tails of the synthetic data and,
  strongest, on real photos. Children and the female-looking men (below) are the same failure.
- [Certain, log] The one real child in our photos (a boy) is predicted as an adult body.
- [Certain] Oversampling synthetic children is unproven at 2 epochs. [Likely] It has no mechanism for real photos.
- The evidence cannot rank the cause: synthetic-to-real gap, or "L1 through a sigmoid + rare children". Say so.

**The real-photo measurement, corrected:**
- [Likely, by eye; two independent reads of the photos] `example_data/360_F_392252267...jpg` holds one standing boy,
  a crouching woman and a crouching man. The WhatsApp photo has no child; its "small" people (diag_cropK lines 58 and
  70) are a seated man and cut-off adults. The boy is the only child among the ~20 people in `example_data`.
- [Certain, code] The "skeletal height" in `diag_crop_camera.py:136-141` is measured on the posed body
  (`best['j3d']`). Crouching or sitting shortens it. The script computes rest joints (:141) but prints no rest stature.
- [Certain, log] Their height phenotype is 0.47-0.56 (3569949, 3572419, 3572420, lines 7-71): adult rest bodies
  ([Likely] about 1.75 m stature).
- [Certain, log] The boy is predicted at age 0.60, height phenotype 0.48: an adult rest body.
- So "five small people, age 0.57-0.67" says nothing about children. The old item-12 read ("compare the five small
  people's ages") cannot show a child effect.

**The real-photo collapse** (all 299 `[shape]` lines of demo 3581312):
- [Certain] Median predicted gender 0.59, age 0.61, height 0.50. The training medians are 0.588 / 0.601 / 0.500
  (diag_pheno 3569948:8-18).
- [Certain] Proportions sit at ~0.82 (training 0.725) and weight at 0.59 (training 0.506): fixed offsets, not the
  median. `model.py:205-207` already notes the proportions offset.
- [Certain] The 299 lines are 2 checkpoints x 12 images x one line per crop: about 30 distinct people, nearly all
  adults, each counted several times. So a narrower spread (std) than AnnyOne is **not** evidence: an adult sample
  must have a narrower age spread than AnnyOne, which holds babies and old people. Only the gender spread is: 14% of
  the lines are below 0.5 although men dominate the frames.

**The same pull on synthetic data:**
- [Certain] Holdout people with GT age 0.6-0.8 (mean 0.67, n = 53) are predicted at 0.57 (3570222:89).
- [Certain] In-sample babies (the 50 youngest training people, GT age 0.015) are predicted at 0.10 (3571955:174-175).
  The head reaches the right size by trading: height -0.25, gender abs error 0.23 (3569950:61,76).
- [Certain] Near the mode the head is accurate: gender bias -0.008 / abs error 0.068, age bias -0.015 / 0.037
  (3570222:75). But the holdout has nobody under age 0.513 (3569948:33), so "accurate on synthetic" holds only there.
- So the cause cannot be ranked between the synthetic-to-real gap and the loss (L1 through a sigmoid, plus rarity).
  Do not call it "mainly a domain gap".

**The shape head and its loss** [Certain, code]:
- `decshape = nn.Linear(1024, 11)` (`model.py:620`), then `torch.sigmoid` (`model.py:462`). Zero logits give 0.5.
- L1 on the post-sigmoid values, summed over the 11 dims, mean over people (`loss.py:272-276`, as before the 17-Oct
  mask edit), times `alpha_shape` 20. `--shape_weights` is one weight per phenotype, never per person.
- [Certain, maths] L1 regresses to the median when the image cues are weak. The gradient through the sigmoid is
  s(1-s): 0.25 at s = 0.5, 0.015 at s = 0.015. [Likely] That is part of why babies floor at 0.10.
- No per-person shape mask existed before `--mask_3dpw_shape` (below).

**How Anny makes a child** [Certain: anny 0.6.1 code, and a scratch probe in rest pose]:
- Age blends five shapes, newborn / baby / child / young / old, at anchors `linspace(-1/3, 1, 5)`: age 0 = baby,
  0.333 = child, 0.667 = young, 1 = old. Values are clamped to the anchor range, so newborn needs age < 0, which a
  sigmoid can never give.
- The child shape change happens below about age 0.4-0.5. From 0.67 to 1.0, stature and body ratios are almost flat.
- The child cue is narrow shoulders relative to the head, not stature: shoulder width / head width is 2.24 at age 0.1
  and 3.00 at age 0.6 (height 0.5).
- "Small height + age 0.6" is a short, broad-shouldered adult. The smallest body at age 0.6 is 1.22 m.
- The proportions phenotype cannot make a child: 0 -> 1 at age 0.6 moves leg/stature by only -0.016.

**Age scales.** Three disagree; do not quote years:
- [Certain] `diag_common.py:526-527` and `train.py:1604`: 1/3 = 11 y, 2/3 = 25 y.
- [Certain] `build_manifest.py:6` (MakeHuman slider): 0.19 = 11 y, 0.5 = 25 y.
- [Certain] Anny's calibration (`data/shape_calibration/boys.pth`): age 0 / 0.05 / 0.215 / 0.415 / 0.67 / 0.77 /
  0.83 / 1.0 = 0 / 1 / 4 / 11 / 16 / 18 / 64 / 110 years.
- [Likely] AnnyOne does not follow Anny's calibration: under it, 60.8% of AnnyOne people would be under 16.
- Use body terms instead: below ~0.4-0.5 is child-shaped on any scale, so `--age_balance_child_age 0.3` selects
  clearly child-shaped bodies [Certain, probe].
- [Certain] AnnyOne ages: 1.06% < 0.2, 2.53% < 0.3, 4.45% < 0.4; median 0.601; 41% in 0.5-0.6 (3569948:9,27,32,35).

**The sampler (exp2)** [Certain]:
- Natural batches: 2.1-2.3% of supervised people have age < 0.3 (3571955:154; 3571956:149; 3571601:35). The
  balanced sampler (`--age_balance 1`, child fraction 0.25) raises this to 9.7-11.1%; batches without a child
  82.5-84.5% -> 36.0-43.0%; images with a child ~5-6% -> 23.0-25.4% (3571955:154-156; 3571956:149-151;
  3571601:49-51).
- 2-epoch pilot (job 3571955) vs control (3571956, sampler off):
  - manifest children (n = 10, in-sample): PVE 48.5 vs 53.1, age bias 0.078 vs 0.085; v5 baseline 47.8 / 0.078
    (3571955:169-170; 3571956:164-165). 4.6 mm in the sampler's favour at n = 10: inconclusive, not null.
  - the in-sample baby floor did not move (0.100 -> 0.102; 3571955:174-175), after only 2 epochs = 8,000 images at
    lr 1e-5 with all but 4 backbone blocks frozen (3571955:3).
  - both arms print "ADULT REGRESSION" (adult PVE 75.1 / 74.9 vs 70.5): the restart cost of any fine-tune.
- Verdict: **unproven at 2 epochs**, not "cannot work". [Likely] no mechanism for real photos. Low priority.

**3DPW labels and body shape:**
- [Certain] The converter fits only height and proportions. Gender comes from 3DPW metadata (correct only in v2.3).
  Age 0.55 and 0.5 on muscle, weight, cupsize, firmness, african, asian, caucasian are constants
  (`run_smpl_to_anny_task.sh:94-95`; `smpl_to_anny.py:1101-1124`; 3575369:232).
- [Certain] The shape L1 trains every 3DPW person toward these constants at full per-person weight.
- [Likely] Share: 3DPW is ~30% of the images in a mixed batch, but 3DPW images hold ~1.2 people vs 2.73 for AnnyOne
  (3571601:21), so ~16% of supervised people.
- [Certain] The v3d / v2d mesh terms carry the same neutral body: prepare_gt builds the 3DPW GT mesh from these values
  (`train.py:163-170`).
- [Likely] 3DPW PVE rewards the neutral body: it is computed against those neutral-body GT meshes
  (`train.py:~647-662`). [Guessing] Part of v9's 3DPW gain (102.5 vs 119.1) may come from that.
- [Certain] Not the root cause: v5 never saw 3DPW and shows the same real-photo pull. 3DPW has no children.

**`--mask_3dpw_shape` (new on the evening of 17 Oct; another session; commit `a0926fd`, not on `main` yet):**
- What: with `1`, people tagged `has_hand_labels 0` (3DPW-Anny) get no shape L1 on the 8 constants. Gender, height
  and proportions stay supervised. AnnyOne people are unchanged. It reuses the `HandLabelFlag` tag.
- Default `0`. In exp7: env `MASK_3DPW_SHAPE` (default `0`; `1` adds `_shpmask` to the run names and requires the
  `[loss] mask_3dpw_shape:` line in `train.log`). Check that it arrived: `grep -c mask_3dpw_shape loss.py`.
- [Certain] It adds no child signal. The v3d / v2d mesh terms still carry the neutral body.
- On or off in exp7 is a **Monday decision** (0a, step 8). Caveat: with it on, exp7's 3DPW-test PVE can move for a
  reason unrelated to the depth and bone fixes, and it compares worse against the neutral-body GT. If on: re-evaluate
  the baselines with the 8 phenotypes neutralised at inference (`model.neutral_phenotype_names`, `model.py:470-477`)
  or caveat the number; update section 5, item 7 and note 17.

**Next:** section 5, item 12. In one line: measure on Relative Human first, then the supervisor picks a limitation
(Path A) or BEV-style weak labels (Path B, 1-2 weeks of code, after exp7).

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
- [Certain] The pull to the median above: on all 299 lines the median gender is 0.59, the training median 0.588.
  The cause of the pull is not ranked (above).
- Read on 17 Oct (no job): on the synthetic holdout gender is accurate near the mode, bias -0.008 / abs error 0.068
  (3570222:75). That alone does not prove a synthetic-to-real gap: the holdout has no young tail (nobody under
  0.513), and its 0.6-0.8 bucket already regresses (above).

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
3. **Missing labels (fixed 16 Oct).** Train has 4,451 label images where the fits give 6,437 (:31); cause: missing
   image files, fixed by the re-extraction and the second rebuild (below). Six train sequences have none in the old
   labels:
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
- [Certain since the second rebuild, below] The image folders of those 8 sequences (six train, captureSelfies_00,
  basketball_01) were missing or incomplete on the cluster; [Likely] from an incomplete unzip of
  `/ds-av/public_datasets/3DPW/original/imageFiles.zip`. The
  extract step unzips only if `data/3DPW/imageFiles` does not exist yet (`slurm/data/run_3dpw_list.sh:36`), so a broken
  first unzip is never repeated [Certain about the code; the broken unzip itself is not checked]. Check command: 4c.

**Image re-extraction (16 Oct) and second rebuild (job 3581320): the missing labels are back.** The image-folder
check is done: the image folders of the 8 sequences were re-extracted. File counts after extraction:

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
- Then the labels were rebuilt again, job 3581320 (log `logs/3dpw-labels-3581320.out`, read on 16 Oct), from the
  same old v2.2 fits (gender reversed, per-person frame grid).
  - [Certain] Output folder `threedpw_anny_full/labels_rebuilt`: every split's summary line ends in
    `threedpw_anny_full/labels_rebuilt/index_<split>.txt` (3581320:25, :39, :65). The production `labels/` were not
    touched. It wrote into the same folder as 3579266 and replaced its output [Likely: same fits and builder, so
    the 3579266 files are a subset].
  - Counts [Certain]:

    | split | old labels and rebuild 3579266 | rebuild 3581320 |
    |---|---|---|
    | train | 4,451 images, 5,183 people | 6,437 images, 7,589 people (:25) |
    | validation | 2,458 images, 2,995 people | 2,777 images, 3,314 people (:39) |
    | test | 9,722 images, 11,850 people | 9,722 images, 11,850 people (:65) |
    | total | 16,631 label files, 20,028 people | 18,936 label files, 22,753 people (:67) |

  - All six empty train sequences now have labels (:1-6): arguing_00 255, backpack_00 421, basketball_00 251,
    bodyScannerMotions_00 419, box_00 347, capoeira_00 211. captureSelfies_00 319 -> 401 (:7). Validation
    `courtyard_basketball_01` 0 -> 319 (:27). Together +1,986 train and +319 validation images: the whole gap.
  - `outdoors_crosscountry_00` is still 0: its 162 frames are dropped by the visibility filter (:34, :39), as
    before; they were never in the 2,777. No frame is dropped for error or box size in any split. The
    "unconverted tasks" list is empty (:68).
  - 18,936 is exactly check I4's `expected_label_images` for the current fits (3575369; as quoted in the converter
    section of `docs/experiments_plan.md`).
- [Certain] So the missing labels came from missing image files, and that defect is fixed. Gender and the missing
  actor are not: both are decided inside the fit (the re-fit, 4e).
- **Do not train on `labels_rebuilt`**: same old fits, gender reversed. The labels to train on come from the re-fit
  (`threedpw_anny_v23/labels`, 4e). The re-check of `labels_rebuilt` (4c) is optional and was not run.
- Do not confuse these counts with the ones expected after the re-fit (4e): the v23 labels should have fewer
  images, ~16,900, with about the same number of labelled people (22,753 here), because both actors then share one
  image.

**Next, in this order:** (1) Re-fit with the converter fix, which is on `main` (v2.3: gender by Anny's convention,
one shared frame grid for both actors): all 87 tasks into a **new** folder, rebuild, re-check, steps 0-5 in 4e
(section 5, item 7). The test on `courtyard_captureSelfies_00` ran on 17 Oct (job 3589270; 0b): man gender 0.0 and
14.3 mm, woman 1.0 and 15.5 mm, one shared frame grid [Certain, logs]. That fits defect 1 (men better with the
flipped gender) [Likely, unpaired: one sequence against the pooled old fits; the paired numbers come from its
post-check], but it is one sequence; its post-check and the other 85 tasks are next. (2) Only then train or test
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
- the image-folder check is done and the 8 folders were re-extracted (4c; counts in 3.5);
- the second label rebuild ran (job 3581320, 4c): train 6,437 / validation 2,777 / test 9,722 label images, 18,936
  in all, exactly what the current fits imply (3.5). Missing labels fixed; old fits, so not for training.

**Done on 17 Oct:** the re-fit test, 4e step 1 (job 3589270, on an H100 node through `PY_VENV`; 0b, 4f). Before it,
jobs 3587852/3587853 crashed on an H200 (4f).

**Still pending, in this order** (0a steps 3-6). Start each only after the one before has finished and passed its
checks.
1. The post-check of the re-fit test, on the login node (4e, step 2).
2. The full re-fit, 87 tasks, through `PY_VENV` with `--time=02:00:00` (4e, step 3; 4f).
3. Labels and the conversion check on v23 (4e, steps 4-5).
4. The final model: a v9-style fine-tune of v5 with its bone fix, the depth fix and the v23 labels (section 5,
   item 7). Script: `slurm/experiments/exp7_final_model.sh` (on `main` since `b188b63`; never run). It trains, so
   old GPUs only (4f). Submit it after the meeting, if the supervisor picks labels first (0a, step 6).

Also open: the x20 depth run (4d; the command was given on 14 Oct, the submission was never confirmed).

**Status block.** `squeue -u najib` lists only queued and running jobs. This block also shows finished jobs and their
outputs (`;` so that a missing folder does not stop the rest):
```
cd /netscratch/najib/multi-hmr; sacct -u najib -S 2026-10-14 --format=JobID,JobName%28,State,Elapsed; ls -lt logs | head -20; ls results/experiments/exp1_depth_anchor/
```
What to look for: job name `exp1-depth-anchor` in `sacct` (State `COMPLETED`, `FAILED`, `TIMEOUT`, ...);
`long_pc_dist20` under `exp1_depth_anchor/`; for the re-fit, job name `smpl2anny-all` (one line per array task).
(The label rebuild 3581320 is done and read, 4c.) If `sacct` prints an error (accounting may be off on this cluster
[Guessing]), use the `ls` lines alone. **Resubmit a job only if both
`sacct` and its output folder show nothing.** If the folder exists but the job failed, ask for its `.out` and `.err` before resubmitting.

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

### 4c. Image folders re-extracted and labels rebuilt (16 Oct, job 3581320; done)

**Done on 16 Oct:**
- The image-folder check. The first label rebuild (job 3579266) had given the same counts as the old labels (3.5).
- The re-extraction of the image folders of the 8 affected sequences. Counts in 3.5; captureSelfies_00 went from 480
  to 750 files.
- A second rebuild, job 3581320. Its log `logs/3dpw-labels-3581320.out` was read on 16 Oct: train 6,437 /
  validation 2,777 / test 9,722 label images, as expected (full counts in 3.5).

To re-count the folders (login node, seconds; `courtyard_jacket_00` is a control with 412 labelled images,
3579266:12; the last number is the count of sequence folders, 60 if all are there, 3575369:25-27):
```
cd /netscratch/najib/multi-hmr && for s in courtyard_arguing_00 courtyard_backpack_00 courtyard_basketball_00 courtyard_bodyScannerMotions_00 courtyard_box_00 courtyard_capoeira_00 courtyard_captureSelfies_00 courtyard_basketball_01 courtyard_jacket_00; do echo "$s $(ls data/3DPW/imageFiles/$s 2>/dev/null | wc -l)"; done; ls data/3DPW/imageFiles | wc -l
```

**What the log of 3581320 showed** (every check this section asked for passed):
1. **Output folder** [Certain]: `threedpw_anny_full/labels_rebuilt` in every summary line (3581320:25, :39, :65;
   the summary format is `build_threedpw_anny_labels.py:143-146`). The production `threedpw_anny_full/labels` (what
   v9/v10 trained on; `run_build_labels.sh:22` is the default without `OUT=`) was not overwritten.
2. **Counts** [Certain]: train 6,437 (was 4,451), validation 2,777 (was 2,458; +319 from `courtyard_basketball_01`),
   test 9,722 unchanged (old: 3575369:31, :33, :35); 18,936 label files, 22,753 people (:67).
   - The six train sequences and `courtyard_basketball_01` no longer show 0 images; captureSelfies_00 has 401 (was
     319).
   - Crosscountry stays at 0 (162 frames dropped by visibility, :34, :39); it was never in the 2,777.
   - The "unconverted tasks" list is empty (:68).
3. So missing image files were the cause. Nothing more to ask for.

The rebuild command given in the 15-Oct version of this section (~1 min of CPU, limit 30 min):
```
cd /netscratch/najib/multi-hmr && OUT=/netscratch/najib/multi-hmr/threedpw_anny_full/labels_rebuilt sbatch slurm/data/run_build_labels.sh
```
Optional re-check of the rebuilt labels (~19-20 min; not run). Not needed for the plan: the labels to train on
come from the re-fit, whose own check is step 5 in 4e:
```
cd /netscratch/najib/multi-hmr && LABELS=threedpw_anny_full/labels_rebuilt OUT=results/diagnostics/check_3dpw_conversion_rebuilt sbatch slurm/diagnostics/check_3dpw_conversion.sh
```
- Training scripts hard-code the old folder (`slurm/train/train_teacher_v9_mixed.sh:55`,
  `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`). Never overwrite `labels/`.
- The rebuild did **not** fix gender or the missing actor; both are decided inside the fit (the re-fit with the
  converter fix, 4e). Do not train on `labels_rebuilt`.
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

### 4e. Command cookbook, and the 3DPW re-fit (pending items 1-3)

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

**3DPW re-fit with the converter fix: steps 0-5 (pending items 1-3; section 5, item 7).** The commands come from
`docs/experiments_plan.md`, section "3DPW converter fix (v2.3 flags)" (`plan :593-843`), which also gives the
background, the guards and the full pass criteria. **Changed on 17 Oct** (4f): steps 1 and 3 run through the venv
(`PY_VENV`) with an explicit partition list, so they can use the free H100/H200; step 3 has a 2-hour limit; step 2
and the label build of step 4 run on the login node. The plan's commands for these steps are older (section 7).
What is on `main`:
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
- Since 16-17 Oct `run_smpl_to_anny_task.sh` also reads `GPUS` (`0` = CPU), `CONDA_ENV` and `PY_VENV`, and
  `slurm/env.sh` activates them (4f). Defaults unchanged.

**Run the steps one at a time.** Start each step only after the previous one has finished, and steps 2 and 3 only
after the step before them passed its checks. `sbatch` prints the job id; `squeue -u najib` lists only queued and
running jobs, so a job it no longer shows has ended. Analyse each log only after the student says "done".

0. Update the cluster copy and check it (done 16-17 Oct: the student pulled up to `93c4eda`; repeat after any new
   commit):
   ```
   cd /netscratch/najib/multi-hmr && git pull && grep -c REFIT_FLAGS slurm/data/run_smpl_to_anny_task.sh; grep -c gender_convention scripts/data/smpl_to_anny.py; grep -c ALLOW_DEFAULT_FLAGS slurm/data/run_smpl_to_anny_task.sh; grep -c PY_VENV slurm/env.sh; ls scripts/data/check_refit_task.py slurm/data/run_check_refit_task.sh
   ```
   Go on only if all four numbers are above `0` and both files are listed. Otherwise the cluster copy is old: ask
   for the `git pull` output and run the line again. With the old task script, steps 1 and 3 would print
   `already converted - skipping` for every task without fitting anything (all 87 `.npz` exist in
   `threedpw_anny_full`), and step 4 would build from an empty `threedpw_anny_v23`.
1. One check task: `courtyard_captureSelfies_00`, train, array indices 9 and 10 (p0 and p1). Two actors of mixed
   gender, and the strongest stride-phase case (56 of 401 images with both actors). **Done 17 Oct, job 3589270**
   (logs `logs/smpl2anny-all-3589270_9.out` and `_10.out`; their `.err` files name the task jobs 3589271 and
   3589270), on serv-3341, an H100 node (`sinfo -s` of 16 Oct lists it under H100 and H100-RP), through the venv.
   The command [Likely: the student did not paste the `sbatch` line; the logs confirm the venv, the flags and the
   folder]:
   ```
   cd /netscratch/najib/multi-hmr && PY_VENV=/netscratch/najib/venv_cu121 OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --partition=H100,H100-PCI,H100-RP,H200,H200-PCI,H200-AV,A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTX3090,RTXA6000,RTXA6000-AV,V100-32GB --gpus=1 --time=00:45:00 --array=9-10 slurm/data/run_smpl_to_anny_task.sh
   ```
   What the logs show, against the checks this step asked for [Certain, logs]:
   - both: `extra flags (REFIT_FLAGS): '--gender_convention anny --frame_grid shared'`, `frame grid: shared (stride
     3 ...)` and a `done:` line. Their `.err` files load PyTorch from `/netscratch/najib/venv_cu121/` and hold only
     two warnings (TypedStorage deprecated; no NVIDIA Warp), no error;
   - gender: man (p1) `gender phenotype fixed at 0.0 from 3DPW gender 'm'`; woman (p0) `fixed at 1.0 from 3DPW gender
     'f'`;
   - summary: man `fitted 231 frames; joint error mean 14.3 mm, max 15.8 mm`; woman `fitted 226 frames; joint error
     mean 15.5 mm, max 34.5 mm`. This line is printed after pass 2, the refit with the shared median body
     (`smpl_to_anny.py:1203-1205`). Both means are below the expected 16-20 mm; the man's max (15.8 mm) is far below
     the 30-mm "good fit" line. The woman has one hard stretch, frames 396-498 at 20-28 mm per frame in the first
     pass (the other frames 11-20 mm; her highest first-pass frame is 28.3 mm, frame 405). Her worst frame after the
     shared-body pass is 34.5 mm: above the 30-mm line, below the builder's 80-mm drop, and higher than any
     first-pass frame, so the shared body made at least one frame worse by 6 mm or more. Which frame it is is
     unknown: the log prints pass-2 errors for 5 frames only, and neither the post-check nor the v23 check's I4
     prints frame numbers. The `.npz` has them; this lists her 5 worst frames (login node, NumPy only, seconds):
     ```
     cd /netscratch/najib/multi-hmr && conda activate multihmr && python -c "import numpy as np; d=np.load('threedpw_anny_v23/train/courtyard_captureSelfies_00_p0.npz'); e=d['joint_err_mm']; f=d['frames']; i=np.argsort(e)[-5:]; print([(int(f[k]), round(float(e[k]), 1)) for k in i])"
     ```
   - PER-JOINT tables: highest rows man right_hip 34.4, left_hip 30.7, spine3 29.9 mm; woman left_hip 40.6,
     right_hip 38.2, neck 32.7 mm. The woman's hips and neck are 2-3 times her mean joint error (15.5 mm), so they do
     stand above the rest. But the highest rows (hips, neck / spine3) are the same joints that top the old pooled
     table (3575369:39-62: right_hip 37.9, left_shoulder 35.5, left_hip 33.9, pelvis 28.4, neck 27.6, spine3 23.2),
     so this is no sign of a new mapping error [Likely]; the old table shows the torso gap between the two rigs.
     Against the old pooled per-joint means (3575369:39-62, all 87 tasks), 12 of the woman's 24 joints are higher
     (e.g. neck 32.7 vs 27.6, left_hip 40.6 vs 33.9, ankles 16-17 vs ~10), and 8 of the man's (e.g. spine2 21.0 vs
     12.7, head 14.1 vs 8.5, spine3 29.9 vs 23.2). That is one sequence against the pool of 87, so it is unpaired,
     like the 14.3-vs-18.8 comparison below (which also sets a new mean against an old median). Per-joint and
     overall changes cannot be read until the paired post-check (step 2); it compares the overall error only, so per
     joint there is no paired number yet;
   - frames: every fitted frame is a multiple of 3; 225 frames are fitted for both, 6 only for the man and 1 only
     for the woman (by the shared-grid rule these are frames where the other actor is not valid in 3DPW; the
     post-check confirms it);
   - time: 1,013 s / 231 frames and 978 s / 226 frames, so **~4.4 s per frame on an H100** (the converter call only,
     without the container start); the task script's header estimate was 6 s.
   For comparison, 3575369:70 gives the old fits of this sequence median 18.8 / p95 21.9 / max 35.7 mm over 457
   person-frames, both actors pooled (the new fits also have 226 + 231 = 457). That is not a paired comparison, and
   it is a new mean (14.3, 15.5) against an old median; step 2 prints mean and median for old and new, and the
   paired line.
2. Post-check of that sequence (seconds; **the next step**). Run it **on the login node**, not with
   `run_check_refit_task.sh`: that script has no partition line, so it lands in the CPU partition `batch` (216 of 216
   CPUs busy on 16 Oct) and waits. `check_refit_task.py` is NumPy only (its docstring). The login shell must have the
   conda env `multihmr` active (the prompt shows `(multihmr)`). It compares with the old fits in `threedpw_anny_full`.
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && python scripts/data/check_refit_task.py --seq courtyard_captureSelfies_00 --split train --new_root /netscratch/najib/multi-hmr/threedpw_anny_v23 --old_root /netscratch/najib/multi-hmr/threedpw_anny_full --seq_root /netscratch/najib/multi-hmr/data/3DPW/sequenceFiles --expect_gender_convention anny --expect_frame_grid shared
   ```
   It passes when: the output ends with `RESULT: PASS` (exit code 0; `echo $?` right after it prints `0`);
   `[p0+p1] NEW: ... both valid in 3DPW but only one fitted: 0/N` while the `OLD` line shows a large share; the
   man's label value is ~0.0001 and the woman's ~0.9999. Not a gate: the man's joint error should be lower than
   before (the woman's may rise ~2 mm); if it is not, note it, report it and go on. Read the paired line
   `[pK] joint error mean/median, M frames fitted by both: old ... -> new ...` first; if M is under ~30 or the line
   is missing, use the unpaired `[pK] joint error mean/median, all fitted frames: old ... -> new ...`. If `python`
   is not found or `import numpy` fails, the env is not active: ask for the prompt line and the error.
3. All 87 tasks, through the venv, 2-hour limit. Submit only after step 2 passed, and never while another array on
   `threedpw_anny_v23` is queued or running (two tasks would fit the same person into the same temporary file).
   Tasks 9 and 10 are skipped, because their `.npz` exist with the same flags (`already converted with the requested
   flags - skipping`). A task to redo needs its `.npz` deleted first.
   ```
   cd /netscratch/najib/multi-hmr && PY_VENV=/netscratch/najib/venv_cu121 OUT_ROOT=/netscratch/najib/multi-hmr/threedpw_anny_v23 REFIT_FLAGS="--gender_convention anny --frame_grid shared" sbatch --partition=H100,H100-PCI,H100-RP,H200,H200-PCI,H200-AV,A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTX3090,RTXA6000,RTXA6000-AV --exclude=gifu,glendale --gpus=1 --time=02:00:00 --array=0-86%8 slurm/data/run_smpl_to_anny_task.sh
   ```
   - Time [an estimate from the measured 4.4 s per frame]: the longest task is [Likely] `downtown_windowShopping_00`
     p0, 608 fitted frames in the old fits (3575369:121), so ~45 min on an H100. The two-person sequences have fewer
     per person on average (`downtown_bus_00`: 850 for both, about 425 each, 3575369:104); the log does not split
     them per person, so one actor could have more, at most 726 (2,178 frames / 3), ~55 min. The 2-hour limit holds
     either way. All tasks: the old fits had 22,915 fitted frames (3575369:342), so ~28 GPU hours; [Likely] ~3.5-4 h
     of wall time with 8 at once if H100/H200 GPUs stay free.
   - The list also has the old GPUs, whose speed with these settings is not measured. [Guessing] a long task on a
     V100 or RTX3090 could pass 2 h; it then ends `TIMEOUT` without an `.npz` (the file is written only at the end).
     Resubmit only that index (`--array=` with its number, `--time=04:00:00`) after the array has finished.
   - B200 and RTXB6000 are left out on purpose: no PyTorch build here has code for them (task script header).
   - V100-32GB is left out, and its nodes `gifu` and `glendale` excluded: in the first full run (job 3590385) all
     36 tasks that landed there failed within 40 s (argparse: `unrecognized arguments:` with only blanks), while
     all 51 elsewhere worked [Certain]. The rerun of those 36 is in 0a, step 4.
   Logs `logs/smpl2anny-all-<arrayjobid>_<task>.out`. When the array has left `squeue`, this lists every task without
   an `.npz` (its array index first) and then counts the `.npz`, which must be 87:
   ```
   cd /netscratch/najib/multi-hmr && awk '{print NR-1, $0}' configs/threedpw_tasks.txt | while read -r I SPLIT SEQ PERSON N; do [ -f "threedpw_anny_v23/$SPLIT/${SEQ}_p$PERSON.npz" ] || echo "missing: task $I $SPLIT $SEQ p$PERSON"; done; ls threedpw_anny_v23/*/*_p*.npz | wc -l
   ```
   For a missing task, ask for its `.out` and `.err` before resubmitting it. Then, on the login node, the post-check
   of every sequence (NumPy only; never timed on all 87 tasks). **Required before the final model**: exp7's header
   lists `RESULT: PASS` from this command as prerequisite 2, but its guards do not run or enforce it [Certain, the
   script only names it in its header], so the student must run it and read the result:
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && python scripts/data/check_refit_task.py --seq all --new_root /netscratch/najib/multi-hmr/threedpw_anny_v23 --old_root /netscratch/najib/multi-hmr/threedpw_anny_full --seq_root /netscratch/najib/multi-hmr/data/3DPW/sequenceFiles --expect_gender_convention anny --expect_frame_grid shared
   ```
4. and 5. The labels and the conversion check on v23, only after the step-3 array has finished and the count above
   is 87. The check must not start before the labels are complete: it only tests that `index_*.txt` exists, and the
   builder writes `index_train.txt` before the other splits are done. Keep `OUT=` on the check, otherwise it
   overwrites the 3575369 results.
   - **Queue problem** [Certain, from the scripts]: `run_build_labels.sh:4-5` and `check_3dpw_conversion.sh:65-66`
     ask for a GPU on the old partitions only (estimated 22-26 Oct on 16 Oct), and both run in the normal env. The
     label build itself needs no GPU: `build_threedpw_anny_labels.py` imports only argparse, glob, os, pickle, sys
     and NumPy (its docstring: "Pure NumPy - no torch or Anny needed"; "~1 min", `run_build_labels.sh:2`). So a plain
     `sbatch slurm/data/run_build_labels.sh` would put the labels, the check and exp7's `CHECK_ONLY` step (0a, step 6)
     behind that queue for nothing.
   - **Step 4: build the labels on the login node** (17 Oct; not run yet), like the post-check. [Likely] fine there
     (NumPy only, ~1 min); the Slurm script asks for 48 GB (`:7`), so if the login node stops it (memory), use the
     same script with the free H100/H200 nodes added (NumPy only, so the missing sm_90 kernels do not matter
     [Likely]). The command first checks that every task has its fit (the same test as the loop in step 3) and builds
     nothing if one is missing:
     ```
     cd /netscratch/najib/multi-hmr && conda activate multihmr && M=$(awk '{print NR-1, $0}' configs/threedpw_tasks.txt | while read -r I SPLIT SEQ PERSON N; do [ -s "threedpw_anny_v23/$SPLIT/${SEQ}_p$PERSON.npz" ] || echo "task $I"; done) && if [ -n "$M" ]; then echo "labels NOT built, fits missing:" $M; else python scripts/data/build_threedpw_anny_labels.py --fits /netscratch/najib/multi-hmr/threedpw_anny_v23 --images /netscratch/najib/multi-hmr/data/3DPW/imageFiles --out /netscratch/najib/multi-hmr/threedpw_anny_v23/labels; fi
     ```
     Fallback only if the login node refuses it:
     ```
     cd /netscratch/najib/multi-hmr && FITS=/netscratch/najib/multi-hmr/threedpw_anny_v23 OUT=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --partition=H100,H100-PCI,H100-RP,H200,H200-PCI,H200-AV,A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV slurm/data/run_build_labels.sh
     ```
     The plan's one-block route (the check waits for the label job with `--dependency=afterok`) also works, but waits
     in the old-GPU queue; do not use it before Monday:
     ```
     cd /netscratch/najib/multi-hmr && L=$(FITS=/netscratch/najib/multi-hmr/threedpw_anny_v23 OUT=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --parsable slurm/data/run_build_labels.sh) && echo "labels job ${L%%;*}" && LABELS=threedpw_anny_v23/labels FITS=threedpw_anny_v23 OUT=results/diagnostics/check_3dpw_conversion_v23 sbatch --dependency=afterok:${L%%;*} slurm/diagnostics/check_3dpw_conversion.sh
     ```
     If its label job fails, the check stays pending with reason `DependencyNeverSatisfied`; cancel it with
     `scancel` and the check's job id.
   - The builder prints one summary line per split, ending in `threedpw_anny_v23/labels/index_<split>.txt`.
   - **Step 5: the conversion check**, through the venv, so it can use the H100/H200. The check imports torch, Anny,
     smplx and scipy, not xformers or torchvision [Likely, from its imports; never run in the venv]:
     ```
     cd /netscratch/najib/multi-hmr && PY_VENV=/netscratch/najib/venv_cu121 LABELS=threedpw_anny_v23/labels FITS=threedpw_anny_v23 OUT=results/diagnostics/check_3dpw_conversion_v23 sbatch --partition=H100,H100-PCI,H100-RP,H200,H200-PCI,H200-AV,A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTX3090,RTXA6000,RTXA6000-AV --exclude=gifu,glendale --gpus=1 --time=01:30:00 slurm/diagnostics/check_3dpw_conversion.sh
     ```
     Unlike the task script, `check_3dpw_conversion.sh` does not write `PY_VENV` into its container command; it
     relies on Slurm passing the submit environment into the job and the container [Likely]. So check in its `.err`
     (`logs/check_3dpw_conversion_<jobid>.err`) that PyTorch loads from `/netscratch/najib/venv_cu121/`. If it stops
     on an H100/H200 with `no kernel image`, the venv was not active and nothing is lost; the fix is the same small
     change `93c4eda` made in the task script (NEEDS CODE, section 2 pattern).

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
  3581320 gave, with 22,753 people; 3.5), not against the old index's 16,631. **Fewer images** than 18,936 with
  about the same people is expected, not a loss: on one shared grid both actors of a two-person sequence sit on the same image
  instead of on two.
- One sequence to check by hand: `courtyard_captureSelfies_00` had 401 label images; [Likely] ~232 now (the frames
  of either actor in job 3589270: 225 + 6 + 1) with the same 457 people, if the builder drops none, as before.

**Using the v23 labels.** The training and 3DPW evaluation scripts set
`LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"` with a plain assignment
(`slurm/train/train_teacher_v9_mixed.sh:55`, `train_v10_scratch_mixed.sh:45`, `train_distill_vits_v8_final.sh:42`,
`slurm/eval/run_eval_3dpw.sh:39`, `run_eval_3dpw_teacher.sh:39`, `run_eval_3dpw_test.sh:61`), so `LABELS=...` in
front of `sbatch` is silently ignored. Edit that line in a copy of the script, then check the `--mix_3dpw_labels`
path in the job log. 3DPW numbers against the old labels cannot be compared with numbers against v23: re-evaluate
the baseline on v23 first. Exceptions:
- the final-model script `exp7_final_model.sh` reads `LABELS` from the environment (default the v23 labels) and
  checks it;
- `run_eval_3dpw_test.sh`, with the change of 17 Oct (uncommitted, not on `main` up to `93c4eda`; reviewed and
  committed together with exp7, 0a step 6). It reads `EVAL_LABELS`, not `LABELS` (a `LABELS` exported in the shell
  never switches it), and `MODEL_ENTRY="ckpt|head_dim|backbone|label"` for one model, which only array task 0 runs
  (submit with `--array=0`). With labels other than `threedpw_anny_full/labels`, the run name gets
  `__<labels folder>` appended, e.g. `evaltest_teacher_v5__threedpw_anny_v23`, so the old `evaltest_<name>` folders
  stay; the log prints the labels used [Certain, from the uncommitted script as read at ~04:15 UTC on 17 Oct].
  Without the change (the committed version) both variables are ignored: it evaluates its model list against the
  old labels. The re-evaluation of the baselines on v23 is in exp7's header ("3DPW-test").

### 4f. Cluster, GPUs and the H100 route for the converter (16-17 Oct)

**The queue** [Certain, from the student's terminal output on 16 Oct, unless tagged]:
- The old GPU partitions (V100-32GB, A100-*, RTX3090, RTXA6000*, L40S*) were full. The re-fit test, as job 3585580
  (1.5-hour limit, 11 GPU partitions), got a start estimate of 22 Oct 07:01, reason `Priority`; cancelled and resubmitted
  as job 3585810, it got 26 Oct. `scontrol show reservation`: "No reservations in the system". So [Likely] low
  fair-share priority after this week's many jobs, not maintenance. Slurm's estimate assumes that every running job
  uses its full limit (3 days on those partitions), so real starts are often earlier [Likely].
- `sinfo -s` at 16:42 (times CEST): every partition fully allocated, including the CPU partition `batch` (216 of 216
  CPUs busy), so the CPU option `GPUS=0` (`4c2113b`) did not help. That night the H100/H200 jobs started within minutes:
  3587852 at 00:33, about 13 minutes after its command was given; 3589270 at 01:49, within about two minutes.
- **V100 nodes `gifu` and `glendale`** [Certain, job 3590385]: every refit task there died in 33-40 s with
  `smpl_to_anny.py: error: unrecognized arguments:` and only blanks after it; the same tasks run elsewhere (H200,
  A100, RTX3090, RTXA6000) worked. Cause unknown ([Guessing] the multi-line `bash -c` command arrives changed on those
  nodes). Leave V100-32GB out of every `--partition` list and add `--exclude=gifu,glendale`, at least for jobs with
  `PY_VENV`; [Guessing] training jobs may hit the same, which is one more reason exp7 leaves V100 out.
- Time limits (`sinfo -s`): H100, H100-RP, H100-PCI, H200, H200-PCI, H200-AV, B200 and RTXB6000 1 day; V100, A100-40GB,
  A100-80GB, A100-PCI, RTX3090, RTXA6000, L40S and `batch` 3 days; A100-RP, RTXA6000-AV and L40S-AV unlimited.

**Why the normal env fails on H100/H200** [Certain, `smpl2anny-all-3587853_10.err`]: `multihmr` has PyTorch
2.0.1+cu117, built for sm_37, sm_50, sm_60, sm_70, sm_75, sm_80 and sm_86 only. Jobs 3587852 (task 9) and 3587853
(task 10) ran on serv-3312, an H200 NVL (sm_90), and stopped in `anny_rest_positions` with `CUDA error: no kernel
image is available for execution on the device`; nothing was written. B200 and RTXB6000 (Blackwell) are newer still;
the task script's header says neither PyTorch build runs there.

**What failed: a conda clone** (`conda create --name multihmr_cu121 --clone multihmr`; 5.0 GB, 46,928 files).
- First `Disk quota exceeded` while writing `/home/najib/.conda/aau_token`. [Likely] the home folder has a personal
  quota of about 10 GB and was full: 9.7 GB used, mostly `Downloads` (4.4 GB) and `isaacsim` (5.2 GB); `quota` is not
  installed, so the limit itself is not shown. `Downloads` was moved to `/netscratch/najib/home_moved/` (nothing
  deleted).
- The clone with `--offline` and the caches on netscratch then failed at the very end (`Rolling back transaction`,
  `[Errno 122] Disk quota exceeded`), although netscratch passed a 1-GB write and a 2,000-file test. Cause unknown
  [Guessing: conda writes to a home path at the end]. Nothing was created: `multihmr_cu121` does not exist, so never
  use `CONDA_ENV=multihmr_cu121`.

**What works: the venv `PY_VENV=/netscratch/najib/venv_cu121`.** Built on the login node on 17 Oct (~01:45 CEST) with:
```
/netscratch/najib/miniconda3/envs/multihmr/bin/python -m venv --system-site-packages /netscratch/najib/venv_cu121
```
```
/netscratch/najib/venv_cu121/bin/pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
- [Certain, pip output] It installed `torch-2.1.2+cu121` and `triton-2.1.0` into the venv only ("Not uninstalling
  torch at /netscratch/najib/miniconda3/envs/multihmr/... outside environment"): `multihmr` is unchanged. Everything
  else comes from `multihmr` (`--system-site-packages`).
- [Certain, pip output] pip warned that `xformers 0.0.20`, `torchvision 0.15.2` and `torchaudio 2.0.2` require
  `torch==2.0.1`. Inside the venv those three do not match PyTorch.
- A check was given on 17 Oct; its output was not pasted. It should print `2.1.2+cu121`, a list with `sm_90`, and a
  path inside `/netscratch/najib/venv_cu121/`:
  ```
  /netscratch/najib/venv_cu121/bin/python -c "import torch, anny, roma, smplx; print(torch.__version__, torch.version.cuda, torch._C._cuda_getArchFlags(), torch.__file__)"
  ```
- [Certain, logs] It works for the converter: the `.err` files of job 3589270 load PyTorch from
  `/netscratch/najib/venv_cu121/lib/python3.9/site-packages/torch/`, and both tasks fitted on CUDA on serv-3341 (an
  H100 node) without error.
- How the scripts use it: `slurm/env.sh` runs `conda activate ${CONDA_ENV:-multihmr}`, then activates `$PY_VENV` if
  it is set (`93c4eda`); `run_smpl_to_anny_task.sh` writes `CONDA_ENV` and `PY_VENV` into its container command
  (`1a5d0c3`, `93c4eda`). Without `PY_VENV` every script behaves as before.

**What the venv is NOT for** [Certain about the versions; Likely about the consequences]: training, the demo, the
evaluations and the model diagnostics (`train.py`, `demo.py`, `diag_holdout_trace.py`, ...). They need the xformers
and torchvision that match PyTorch 2.0.1, so they run in plain `multihmr` (no `PY_VENV`). That PyTorch runs only on
the old GPUs: V100 (sm_70), A100 (sm_80), RTX3090 and RTXA6000 (sm_86), and L40S (sm_89, which runs sm_86 code).
Not on H100/H200 (sm_90), B200 or RTXB6000. So:
- **The final model, and every other fine-tune, waits for the old-GPU queue.** Submit it once (the final model after
  the meeting, 0a step 6) and leave it queued.
- `slurm/train/train_teacher_v9_mixed.sh:27` and `train_v10_scratch_mixed.sh:24` list `RTXB6000`; [Likely] a job that
  lands there crashes like 3587853. A copy for the final model should drop it (the `exp*.sh` and the diagnostic
  scripts do not list it).
- Training on H100/H200 would need a whole new env (PyTorch >= 2.1 with matching xformers and torchvision), never
  built or tested. The brief says "do not upgrade dependencies blindly" (section 10): not this week.
- NumPy-only steps need neither a GPU nor the queue: `check_refit_task.py` and `build_threedpw_anny_labels.py` run on
  the login node (4e, steps 2 and 4).

**Do not reuse:** any command with `CONDA_ENV=multihmr_cu121` (the env does not exist); the 16-Oct command with
`--partition=PARTITION` (a placeholder; it failed); the 16-Oct partition list with `B200` and `RTXB6000`.

---

## 5. Next steps, in priority order

Analyse any new log only after the student says "done". The pending chain, in order (section 4, 0a steps 3-6):
(1) the post-check of the re-fit test, (2) the full re-fit, (3) labels and check on v23, all in 4e (item 7); (4) the
final model (end of item 7; script drafted, submitted only after the meeting). Everything below that trains or runs
the model (training, demo, trace) needs the old GPUs; only the 3DPW converter can use the H100/H200 (4f). Item 3 can
run before (4). Item 6 depends on the supervisor's answer to slide 17. The label rebuild (item 5) is done.

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
5. **3DPW images, then rebuild: done (16 Oct).** The image check and the re-extraction of the 8 folders are done
   (counts in 3.5). The second rebuild, job 3581320, gave train 6,437 / validation 2,777 / test 9,722 label images,
   18,936 in all, exactly what the current fits imply (3.5, 4c): the 31% missing train labels are back.
   - Written into `threedpw_anny_full/labels_rebuilt` from the old fits: gender still reversed, the second actor
     still missing. Do not train on it; the re-fit (item 7) is next.
   - The re-check of `labels_rebuilt` (4c) is optional and not run.
   - The label build after the re-fit (item 7) uses the same images, so they are ready for it.
6. **Supervisor's decision (slide 17).** Fine-tune v9 now with `--gt_center_joint person_center` and x5, or first fix
   the labels (item 7) and then fine-tune? The student proposes labels first, because 3DPW is 30% of each batch and the
   men would keep learning the wrong body. Expected from v5: size ~0.84, root ~0.5 m, cost ~2.6 mm PVE and ~1.6 recall
   points (notes, slide 17, If asked: Option 1). **The v9 fine-tune script does not exist yet (NEEDS CODE).** Recipe:
   - Copy the training call of `slurm/train/train_teacher_v9_mixed.sh` (it has the 3DPW mix flags), into a new
     script, e.g. `slurm/experiments/exp8_v9_depth.sh` (exp7 is the final model, item 7). Do **not** reuse
     `exp1_depth_anchor.sh`: it trains on AnnyOne only, so `CKPT=v9` would silently drop the 30% 3DPW. Do not run
     `train_teacher_v9_mixed.sh` itself: it starts from v5, runs 150 epochs and writes to
     `--name anny_teacher_v9_mixed`. It trains, so old GPUs only, without RTXB6000 (4f).
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
   - Output: `--save_dir /netscratch/najib/multi-hmr/results/experiments/exp8_v9_depth/v9e62_pc_dist5 --name train`.
     Never `logs/anny_model/anny_teacher_v9_mixed`: `train.py` keeps only the `--nb_max_ckpt` highest epoch numbers of
     a folder (`train.py:400-410`), so a run written there deletes checkpoints (its own low epochs, or old v9 ones).
   - Control arm, same script and weight: `--gt_center_joint root` (still x5), output `.../exp8_v9_depth/v9e62_root_dist5`.
   - Labels (`--mix_3dpw_labels`): slide-17 option 1 ("now") = `/netscratch/najib/multi-hmr/threedpw_anny_full/labels`
     (what v9 trained on; gender reversed, 31% of train missing). `labels_rebuilt` has that 31% back (job 3581320,
     item 5), but from the same old fits (gender reversed, second actor missing): do not train on it; the re-fit is
     next. Option 2 ("labels first") = `/netscratch/najib/multi-hmr/threedpw_anny_v23/labels` after item 7.
   - Time: the v9 script's comment expects ~8-10 min per 1,000-iteration epoch at batch 4
     (`train_teacher_v9_mixed.sh:16`; an estimate, never measured), so 10 epochs ~1.3-1.7 h plus evaluations. Not
     measured with 3DPW batches; the exp6 fine-tune on AnnyOne alone took 315 s per epoch (12.7 images/s,
     3579014:66). Say "about 1 to 2 hours". `--time=04:00:00`.
   - Read: trace the last checkpoint of each arm with `diag_holdout.sh` and a `TAG` (4e); root error, slope, size,
     PVE, recall vs the control. The training log's 3DPW-val numbers are against converted labels only.
7. **3DPW labels: re-fit with the converter fix (pending items 1-3; before any 3DPW claim).**
   - The converter fix is on `main` (v2.3): two default-off flags in the fitter (`--gender_convention anny`,
     `--frame_grid shared`), `OUT_ROOT` / `REFIT_FLAGS` and three guards in the array job, and the post-check
     `scripts/data/check_refit_task.py` (`docs/experiments_plan.md`, section "3DPW converter fix (v2.3 flags)").
     Nothing to write.
   - Run steps 0-5 in 4e ("3DPW re-fit"), one at a time, waiting for each to finish: check the cluster copy, one
     check task, its post-check, all 87 tasks into `threedpw_anny_v23`, then the labels and the conversion check.
     Steps 0 and 1 are done (job 3589270: man gender 0.0, 14.3 mm; woman 1.0, 15.5 mm; one shared grid; 0b). Step 2
     on the login node; step 3 through `PY_VENV` with `--time=02:00:00`; steps 4-5: mind the queue problem (4e).
   - The gender refit in the 15-October check (job 3575369) confirmed the flip only for men; check the women in the new check I1 (gender labels). In the
     test the woman fits at 15.5 mm with gender 1.0; her old per-person value comes with the post-check.
   - The meeting is on Monday 19 Oct. The full re-fit (~3.5-4 h on free H100s, estimated from the measured 4.4 s per
     frame) can finish before it if it starts by Sunday; starting on Saturday leaves a margin.
   - The images are re-extracted (item 5; the rebuild 3581320 found all of them), so the label build after the
     re-fit finds them. The builder skips frames without image files.

   **Then the final model (pending item 4): a v9-style fine-tune of v5 with the depth fix, the bone fix and the
   fixed 3DPW labels** (after the re-fit and its check pass). **Script: `slurm/experiments/exp7_final_model.sh`**,
   written on 17 Oct, on `main` since `b188b63` together with the `MODEL_ENTRY` / `EVAL_LABELS` change to
   `slurm/eval/run_eval_3dpw_test.sh` it needs (4e, "Using the v23 labels"); never run. Read its header before the
   student submits it. Before submitting, check the four prerequisites in its header: its
   guards check the labels and the fits, but not the `--seq all` post-check (4e, step 3), which the student runs by
   hand. This item follows the script's defaults (chosen on 17 Oct so that this file, note 17 and the script say the
   same thing). If the review changes a default, change it here and in note 17 too.
   - What it is: v9's 3DPW mixing (30% 3DPW-train in every batch, 3DPW-validation each epoch) on top of the exp6
     fine-tune recipe (v5, last 4 backbone blocks trained, lr 1e-5, batch 4). It is its own script: never edit or
     rerun `slurm/train/train_teacher_v9_mixed.sh` (it starts from v5, runs 150 epochs and writes to
     `anny_teacher_v9_mixed`). Output `results/experiments/exp7_final_model/<RUN>/`, never
     `logs/anny_model/anny_teacher_v9_mixed` (item 6 says why).
   - Start: **v5 00099** (`logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt`) with **v5's bone fix JSON**,
     `--bone_fix results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json`: the same start and
     fix as the exp6 fine-tune (3579014). Why not the exp6 fine-tuned checkpoint: the control arm restarts from the
     same v5 00099, so the two arms differ only in the two fixes; from `exp6_v5_bonefix_00001.pt` the fix arm would
     have 2 extra epochs that the control lacks. That checkpoint stays the script's documented alternative
     (`CKPT=/netscratch/najib/multi-hmr/models/multiHMR/exp6_v5_bonefix_00001.pt BONE_FIX=inherit`); its log line
     would then say "inherited from --pretrained" instead.
   - Expected log line (in `train.log`): `[bone_fix] ON, 4 bone(s), R_b <- R_b @ C_b on the head's local rotmats (from
     bone_fix_spec (checkpoint/run args); estimated on train_sample: ...)`, the same line as in the exp6 fine-tune
     (3579014:62), for neck02 178.8, head 179.2, metacarpal1.L 179.8 and finger4-1.R 179.7 deg (3579014:37). No
     "--pretrained carries a bone fix" line, since v5 00099 carries none [Certain from the code: `train.py`'s bone-fix
     block, `model.py:197-198`; the exact text is [Likely]: exp7 never ran]. The script stops before the trace and the
     test evaluation if `train.log` has no `[bone_fix] ON` line.
   - Depth fix: `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5` (x5, as long_pc_dist5, item 6 and
     slide 17; x20 only if 4d clearly beats it). [Likely] both slide-17 options include the depth fix (the slide says
     so only for option 1). If the supervisor says otherwise, follow that.
   - Labels: `LABELS` defaults to `/netscratch/najib/multi-hmr/threedpw_anny_v23/labels`. The script refuses labels
     under `threedpw_anny_full` and fits not made with `--gender_convention anny --frame_grid shared`. Plus
     `--mask_3dpw_hands 1` (3DPW fingers are rest pose).
   - 3DPW shape mask (new on the evening of 17 Oct, another session; commit `a0926fd`, not on `main` yet):
     `MASK_3DPW_SHAPE`, default `0` = unmasked, as all earlier runs. `1` passes `--mask_3dpw_shape 1` to both arms
     (fix and control stay matched) and adds `_shpmask` to the run names, so a masked run never shares a folder with
     an unmasked one. On or off is a Monday decision (0a, step 8; 3.4). The default keeps this item, note 17 and the
     script in step. If the supervisor picks `1`: pass `MASK_3DPW_SHAPE=1` to both submit commands below, read the
     3DPW-test PVE with the caveat in 3.4 (it can move for a reason unrelated to the two fixes), and update this item
     and note 17.
   - Length: **10 epochs by default** (`EPOCHS=10`, 1,000 iterations each; `--time` 5 h by the script's formula).
     Longer only with `EPOCHS=` set from the deadline (section 10), e.g. `EPOCHS=30` (`--time` 11 h). v9 ran up to 150
     epochs, and the student was distilled from epoch 62. [Guessing] ~6-10 min per epoch with 3DPW batches (never
     timed; the script header's estimate), so 10 epochs ~1.5-2.5 h with the test evaluation, plus the queue.
   - Control: `ARM=control`: same labels, schedule and restart, old targets (root, x1), no bone fix. It is the fair
     reference for PVE and recall (the 2-epoch exp5 control is not one for 10 epochs). Submit both, or say why not.
     Item 3's longer fine-tune measures the bone fix alone; it is optional next to this pair.
   - GPUs: it trains, so plain `multihmr` (the script unsets `PY_VENV`) and only the old GPUs (4f). Its default
     partition list (draft read at ~04:05 UTC on 17 Oct) is A100, L40S and RTXA6000; V100 and RTX3090 only on request
     (its header says why); never H100/H200, B200 or RTXB6000, and it refuses a GPU of compute capability 9.0 or
     higher. Expect days in the queue (22-26 Oct estimated on 16 Oct for those partitions).
   - When: guards first (`CHECK_ONLY=1`, 0a step 6), once the v23 labels exist. Submit after the meeting, if the
     supervisor picks labels first, once each, and leave both queued:
     ```
     cd /netscratch/najib/multi-hmr && bash slurm/experiments/exp7_final_model.sh
     ```
     ```
     cd /netscratch/najib/multi-hmr && ARM=control bash slurm/experiments/exp7_final_model.sh
     ```
   - The deck describes a different fine-tune for option 1. Slide 17 and its notes frame it as one of v9 epoch 62
     ("the planned start", "about 1 to 2 hours of training", a bone fix "estimated on v9 first"; notes, If asked:
     "Option 1", "Bone flips in v9 and v10"). Since 17 Oct note 17's "Option 2" line says the fine-tune after the
     re-fit starts from v5 with v5's bone fix and the depth fix, 10 epochs, next to a control without the two fixes.
     If the supervisor wants the deck's v9 version, item 6's recipe with the option-2 labels is it.
   - Read the result:
     - the script's compare table (holdout V1: neck02, head, girth, root error, PVE, recall against the control; the
       read rules are in its header);
     - 3DPW-test against the v23 labels, only next to baselines re-evaluated on v23 (the script header gives the
       command; it needs the uncommitted `run_eval_3dpw_test.sh` change, 4e, "Using the v23 labels");
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
12. **Children: measure first (ranked plan, evening of 17 Oct; diagnosis in 3.4).** Nothing fixes children before
    Monday [Certain: the repo has no labelled real children, and nothing trains before the meeting]. What the
    supervisor gets: the corrected diagnosis, the plan, and two decisions (the children path; the 3DPW mask in exp7).
    In order:
    1. **Relative Human (RH) val evaluation: the thesis number.**
       - Data [Certain, RH repo]: 24,814 people: 1,534 babies, 2,720 kids, 1,067 teens, 19,493 adults. Labels per
         person: age group (0 adult, 1 teen, 2 kid, 3 baby, -1 unknown), gender (0 male, the same direction as Anny),
         depth layer, boxes, 2D pose. Download: the Google Drive link in the README of
         https://github.com/Arthur151/Relative_Human. No license file [Certain]; evaluation-only use in a thesis is
         low risk [Guessing]. **Downloaded 17 Oct** to `/netscratch/najib/data/Relative_Human/RelativeHuman/` (1.1 GB
         zip, `unzip -t` clean, 8,741 images; `{train,val,test}_annots.npz` with 5,218 / 635 / 1,836 labelled images;
         boxes are `[x1, y1, x2, y2]` pixels, confirmed from RH's `demo.py` crop code).
       - Metric: mean predicted Anny age per group; child-vs-adult AUC of predicted age; gender accuracy (prediction
         < 0.5 = male); per-group recall next to them. Multi-HMR at 672 px misses small people, so an AUC over
         detected people alone is biased [Likely].
       - CODE WRITTEN 17 Oct (not yet run on a GPU or on the real data; CPU test of matching/AUC/stats passes):
         `scripts/diagnostics/eval_relative_human.py` and `slurm/diagnostics/eval_relative_human.sh`
         (`docs/experiments_plan.md`, section "Relative Human evaluation"). Array index 0 = v5 00099, index 1 = the
         exp6 fine-tuned checkpoint with its own bone fix. Command, repo root: `sbatch --array=0-1
         slurm/diagnostics/eval_relative_human.sh`; smoke test `N=20 sbatch --array=0 ...`; `SPLIT=test` for the test
         split (may need more than the 2-hour limit; the job is resumable, cached predictions are skipped). Output
         `results/diagnostics/relative_human_<split>_<tag>/summary.txt` (lines start `[rh]`) and `per_person.csv`;
         re-score without a GPU: `python scripts/diagnostics/eval_relative_human.py --score_only --out <OUT> --iou 0.5`.
         Boxes come from the projected predicted mesh; fov 55 as in the demo [Guessing: RH has no intrinsics]; no EXIF
         rotation handling [Guessing: could misalign a few photos]. Inference only
         [Guessing: under 2 GPU-hours, old GPUs, so the queue]. Start the download now (0a, step 7); numbers after
         Monday.
    2. **Fix our own real-photo read** (prerequisite for anything "by eye"). NEEDS CODE [Guessing: half a day]: in
       `diag_crop_camera.py`, print rest-pose stature (rest joints exist at :141) and shoulder width / head width on
       the rest mesh; match the `[shape]` lines (`demo.py:484-493`) to people. Rides the next diag run.
    3. **The 3DPW shape mask:** default off; on or off for exp7 is a Monday decision (3.4; item 7).
    4. **Limitation paragraph** (writing only): the median pull in both synthetic tails and on real photos (gender,
       age, height; proportions and weight at offsets), cause not ranked; the one real child predicted as an adult;
       ~2% children in AnnyOne batches; the sampler unproven at 2 epochs; 3DPW-Anny labels carry a neutral body on 8
       of 11 phenotypes, so 3DPW PVE does not measure body shape [Likely]. It goes in the thesis in any case, with
       step 1's number if that ran.
    5. **Anny-Fit, for figures only** (https://github.com/naver/anny-fit; NAVER, the group behind Anny; non-commercial
       license [Certain]). A test-time refit of Multi-HMR's Anny predictions. Qwen2.5-VL reads age group and gender
       from head crops; the answers map to baby 0, toddler 0.1, kid 0.33, teen 0.5, adult 0.66, senior 0.999 (male 0,
       female 1) and serve as init and as a squared loss, weight 10 (`configs/demo/multihmr.yaml`) [Certain, code].
       The paper claims a 40.7 mm per-joint gain on toddlers over Multi-HMR [Likely]. The VLM needs a recent
       torch/transformers; our env is torch 2.0.1 [Certain]; several days, environment-dependent [Guessing]. A
       post-process, not the thesis contribution.
    6. **Path B: weak age-group supervision from RH, BEV-style.** Only if the supervisor makes children a goal, and
       after exp7.
       - BEV [Certain, ROMP code]: age bands with zero loss inside; the adult term is a point target alpha^2, because
         alpha 0 is the adult body in SMPL+A; age loss weight 4000; RH 16% of samples.
       - For Anny the adult band must be an interval, about [0.6, 1.0], not a point: a point would pull 78% of RH
         people (adults) to one value, the median attractor again [Likely]. The baby band ends at 0 (no newborn).
       - Cost [Likely]: 1-2 weeks of code and tests before the first GPU-hour. Every GT target comes from
         prepare_gt's Anny FK of the GT pose (`train.py:132-255`); an RH person has none. So: a second target path
         (2D keypoint -> heatmap / offset), validity masks in 8 loss terms, and a collate that tolerates missing keys
         (`bedlam.py:413`: np.stack raises KeyError and the batch is dropped).
    7. **10-epoch sampler run** (low priority; unproven, not "cannot work"): commands below.
    8. **AGORA kids** (257 child scans, non-commercial, synthetic, so not the real-photo gap): future work.
    9. **Pseudo-labels for 3DPW age** (MiVOLO, Apache-2.0 code): future work. 3DPW has no children, so no child
       signal.

    Papers (numbers came through search summaries, arXiv was blocked: [Likely]): BEV https://arxiv.org/abs/2112.08274
    (code https://github.com/Arthur151/ROMP, data https://github.com/Arthur151/Relative_Human; AGORA child mesh error
    125.9 with the weak labels, 146.0 without); Anny-Fit https://arxiv.org/abs/2605.04728; Anny
    https://arxiv.org/abs/2511.03589; AionHMR https://arxiv.org/abs/2512.05259.

    Step 7, the sampler run. The exp2 script's own gate says not to run longer (`exp2_children.sh:38`; both
    pilots print "ADULT REGRESSION ... do not run longer", 3571955:183, 3571956:178). Running it anyway is a deliberate
    override: the control fails the same gate, so the gate measures the restart cost, not the sampler [Likely]. Run
    both arms (default `GT_CENTER=root`, i.e. without the depth fix):
    ```
    cd /netscratch/najib/multi-hmr && MAX_ITER=10000 RUN=long_balanced sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
    ```
    ```
    cd /netscratch/najib/multi-hmr && AGE_BALANCE=0 MAX_ITER=10000 RUN=long_control sbatch --time=05:00:00 slurm/experiments/exp2_children.sh
    ```
    Then the real-photo read on both final checkpoints (`TAG` needed, section 2):
    ```
    cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/results/experiments/exp2_children/long_balanced/train/checkpoints/00009.pt TAG=long_balanced_00009 sbatch slurm/diagnostics/diag_crop_camera.sh
    ```
    ```
    cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/results/experiments/exp2_children/long_control/train/checkpoints/00009.pt TAG=long_control_00009 sbatch slurm/diagnostics/diag_crop_camera.sh
    ```
    Read: step 1's RH metric when it exists. On `example_data` only the boy in `360_F_392252267...jpg` is a child;
    the other "small" people are crouching or seated adults (3.4), so do not compare their ages.

    **Fingers:** train on larger hand crops (no script). If neither children (after step 1) nor fingers move, both
    become thesis limitations.
13. **Docs and thesis.** Fix the stale lines in section 7; thesis draft 74 -> 70.5 mm (V1; the draft is outside the
    repo).
14. **Lower priority:** v10 shape drift (only epochs 290-293 traced). First list what exists (only the last 10 epochs
    are kept per run): `ls /netscratch/najib/multi-hmr/logs/anny_model/anny_v10_scratch_mixed/checkpoints/` and
    `ls /netscratch/najib/multi-hmr/models/multiHMR/ | grep v10`; then trace each with `diag_holdout.sh` (4e).
    Students' depth bug (not measured); output-neutral fix of the 6D init (only matters for new heads; plan :931);
    geodesic loss with `--amp 1` untested; why crosscountry_00 counts as not visible; test-split limb ratios and lost
    head turn; repeat the overfit test with the depth fix; Anny age calibration (partly answered on 17 Oct: three
    scales in the code, and [Likely] AnnyOne does not follow Anny's own calibration; 3.4).

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
- Quoting ages in years ("0.3 ~ 11 y", "0.55 ~ 20 y", "0.5 ~ 18 y"). Do not quote years at all: the code uses three
  scales (0.33 = 11 y and 0.67 = 25 y; the MakeHuman slider 0.19 = 11 y and 0.5 = 25 y; Anny's calibration 0.3 ~ 7 y,
  0.55 ~ 13.6 y), and [Likely] AnnyOne does not follow Anny's calibration (3.4). Use body terms (child-shaped below
  ~0.4-0.5).
- "11% children in batches." Measured 9.7-11.1% depending on the run; 9.9% in the measurement job 3571601.
- "Five small people (skeletal height 0.6-1.14 m) are real children predicted at adult ages 0.57-0.67." Mostly
  crouching or seated adults [Likely, by eye]: the skeletal height is measured on the posed body
  (`diag_crop_camera.py:136-141`), and their height phenotype is 0.47-0.56. The one real child, the boy in
  `360_F_392252267...jpg`, gets age 0.60 [Certain, log] (3.4).
- "On real photos the head outputs the training median for every one of the 11 phenotypes." Only gender, age and
  height; proportions (~0.82 vs 0.725) and weight (0.59 vs 0.506) sit at fixed offsets (3.4).
- "The real-photo spread is 3-4x narrower than training (age std 0.050 vs 0.161), so the head collapsed." The 299
  lines are ~30 people, nearly all adults, each counted several times: not evidence. Only the gender spread is.
- "The head is accurate on synthetic images, so the cause is the synthetic-to-real gap." It is accurate only near the
  mode; it regresses in both synthetic tails (3570222:89; 3569950:61,76). The cause is not ranked (3.4).
- "Age-balanced sampling cannot fix it; the pilot changed nothing." Unproven at 2 epochs: 48.5 vs 53.1 mm at n = 10 is
  inconclusive; only the in-sample baby floor did not move (0.100 -> 0.102).
- "Use BEV's bands for Anny, adult as a point target." BEV's adult target is a point because alpha 0 is the adult body
  in SMPL+A. In Anny adults span ~0.6-1.0, so the adult band must be an interval (section 5, item 12).
- "RH weak labels cost 3-5 code days." 1-2 weeks: an RH person has none of the 3D targets prepare_gt builds (item 12).
- "3DPW people are 30% of the shape supervision." 30% of images, ~16% of supervised people (3.4).

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
  tasks" list is empty (:68), so all fits exist. Cause: the builder skips frames without an image file
  (`build_threedpw_anny_labels.py:123-125`), and the image folders of 8 sequences were incomplete on the cluster. They
  were re-extracted 16 Oct (captureSelfies_00: 480 -> 750 files), and the second rebuild (job 3581320) then gave
  train 6,437, validation 2,777, test 9,722, exactly what the fits imply (3.5).

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

**Cluster and the 3DPW re-fit (17 Oct)**
- "The 3DPW re-fit has not been run." Its step 1 (the test on `courtyard_captureSelfies_00`) ran on 17 Oct (job
  3589270); the post-check and the other 85 tasks have not.
- "The re-fit takes ~5-6 h of wall time, ~35-41 GPU hours." That was the 6 s/frame header estimate. Measured on 17
  Oct: ~4.4 s per frame on an H100, so ~28 GPU hours; the wall time depends on free GPUs.
- "Splitting the 2-task array into two single jobs makes it start sooner." No: Slurm schedules each array task as
  its own job, and the queue reason was `Priority`.
- "PyTorch 2.0.1+cu117 may run on H100/H200." It does not: jobs 3587852/3587853 crashed on an H200 with `no kernel
  image` (4f). Only the converter runs there, through `PY_VENV`.

**Still untested claims; keep the tag**
- "The image files were missing because of an incomplete unzip." That they were missing is now shown: after the
  re-extraction the rebuild 3581320 gives all 18,936 expected label images (3.5) [Certain]. The unzip as the cause
  stays [Likely]: captureSelfies_00 had 480 of 750 files; for the other seven there is no count from before.
- "The recall drop is a cohort artefact."
- "On real images the correction, not the 2-epoch fine-tune, removes the thin neck." The demo (job 3581312) compares
  v5 with the fine-tuned checkpoint, so both changed. The holdout control without the fix keeps girth 0.461 (3.2).
  Test: the optional correction-only demo (section 5, item 2).
- "Men get female-looking bodies on real photos because of a synthetic-to-real gap in the shape head." [Guessing]
  The pull to the training median is measured [Certain, 3.4]; that the gap rather than the loss (L1 through a
  sigmoid, plus rarity) causes it is not separated.

---

## 7. Stale docs: outdated lines elsewhere in the repo

Trust this file over these. Fix them when there is time (one commit per doc).

**`README.md`**
- :114 "numbers look normal while the overlays look wrong" (real-photo demos use box placement).
- :115 whole neck row (weak supervision, redundant chain, long neck = depth defect).
- :116 "Under-trained" (a guess; missing the crop/region-norm nulls, the flipped hand bones and the exp6 result).
- :117 "about 11 years", "0.55 (about 20 years)" (code anchors); exp2 result (inconclusive at 2 epochs) missing.
- :119 "No improvement is claimed until its holdout trace has been re-run" (it has).
- :126 "Future work (not measured)" (both measured, both null). :128 "Nothing from them has been measured yet."
- :211-214 fit description omits the reversed gender and the per-person stride.
- :227 "4,451 train / 2,458 validation" without the cause (image files missing on the cluster) and without the
  rebuild of 16 Oct (job 3581320, `labels_rebuilt`, old fits): 6,437 / 2,777 / 9,722 (3.5).
- :312-325 3DPW-val column needs the converted-label caveat. :331 v4_restpose reading ignores the neck flip / girth 0.46.
- :36-80 layout omits `bone_fix.py`, `scripts/diagnostics`, `scripts/experiments`, `slurm/diagnostics`,
  `slurm/experiments`, `docs/deck_src`. :243 "x4" is effectively x2.41 (boost renormalisation).

**`docs/diagnostics_findings.md`**
- :14-16 neck/fingers origins. :77 size 0.65/0.67 (trace: 0.63). :83-84 "overlays look wrong".
- :118-146 "minimal fix" and one-batch check: done; the fix is the flag; x5 and x20 measured.
- :184 "Origin: supervision". :186-188 "no long neck in the parameters" (neck02 half-turn). :189-193 redundant chain.
- :196-197 "[Likely] long-neck look from the depth defect" (refuted). :201 "under-trained".
- :216-217 "1/3 ~ 11 y", "0.55 ~ 20 y". :251 "root depth error 1.08 m" (signed; absolute 1.21 m).
- :222-223 "skeletal heights 0.62-0.99 m ... small adult ... the failure users see": those short skeletal heights are
  posed bodies of mostly crouching or seated adults, not small people (3.4).
- :270-280 "What to do before Friday" (all stale; :279 "oversample age < 0.4" is now step 7 of section 5, item 12).
  No neck-flip section, no 3DPW check.

**`docs/experiments_plan.md`**
- :8-9 and :991-997 "Nothing in this plan has been run." (The section "3DPW converter fix (v2.3 flags)", :593-843,
  is current on the flags, guards and pass criteria; its step 0 says "After that merge": that is the 16-Oct merge,
  done. Its commands predate 17 Oct: steps 1 and 3 without `PY_VENV` and the H100 partitions, step 2 as a Slurm job,
  the runtime as an estimate. Use 4e here.)
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
- :3-5 "no number here comes from running the experiment" (pilot ran; inconclusive at 2 epochs). :18 "0.5 = about
  18 y". :104-105 age-anchor question: the anchors are `linspace(-1/3, 1, 5)` [Certain, anny 0.6.1]; the years stay
  open (three scales; [Likely] AnnyOne does not follow Anny's calibration; 3.4).

**Others**
- `scripts/diagnostics/README.md:9` "written and syntax-checked, not yet executed" (all E1-E4 ran).
- `docs/cluster_commands.txt`: old scratch commands; `--pose_param` does not exist in `train.py`.
- `docs/3dpw_adapter_check.md`: one-sequence pilot; superseded by the full check (job 3575369).
- Job-script headers (`exp1_depth_anchor.sh`, `exp2_children.sh`, others): time estimates at 6.7 images/s (measured
  ~12 in most runs).
- `slurm/data/run_smpl_to_anny_task.sh` header: "~6 s/frame", "a 1000-frame sequence takes ~35 min" (measured 17 Oct:
  ~4.4 s per frame on an H100, 4e).
- Deck and notes: up to date (18 slides, rebuilt from `build.js` on 16 Oct; notes 6, 16 and 17 rebuilt on 17 Oct with
  the re-fit test; slides 10, 14, 16 and notes 8, 10, 14, 16, 17 rebuilt on the evening of 17 Oct with the children
  correction). Section 9 lists the slides, how to check a copy, and the lines that go stale when the pending
  results arrive.

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
| 4c2113b | after the 16-Oct merge: `run_smpl_to_anny_task.sh` reads `GPUS` (`GPUS=0` = CPU; unused, `batch` was full; 4f) |
| 1a5d0c3 | `CONDA_ENV` override in `slurm/env.sh` and the task script, for a cloned env with sm_90 support (the clone failed; 4f) |
| 93c4eda | `PY_VENV`: a venv with PyTorch 2.1.2+cu121 on top of `multihmr`, activated by `slurm/env.sh` (the H100 route, 4f) |

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
`grep -c '^## 0a\. Tomorrow' docs/handover.md` must print `1`, and `grep -c ALLOW_DEFAULT_FLAGS docs/handover.md`,
`grep -c 7,589 docs/handover.md` and `grep -c 3589270 docs/handover.md` each a number above `0` (top of file). To
give it to ChatGPT, download
`https://raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/docs/handover.md`
(save the page as `handover.md`) and attach it. The deck (18 slides) is at
`https://raw.githubusercontent.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset/main/docs/supervisor_update_2026-10-16.pptx`.

**Job index** (log = `logs/<jobname>_<jobid>.out` on the cluster):

| job | log name | what |
|---|---|---|
| 3570222 | diag_holdout | v5 00099 trace: headline V1 70.5; shape bias gender -0.008 / abs 0.068, age -0.015 / 0.037 (:75); GT age 0.6-0.8 predicted 0.57 (:89) |
| 3569948 | diag_pheno | AnnyOne phenotype distribution: training medians gender 0.588, age 0.601, height 0.500 (:8-18); no holdout person under age 0.513 (:33) |
| 3569950 | diag_young | the 50 youngest training people: age 0.015 -> 0.10, height -0.25, gender abs error 0.23 (:61, :76) |
| 3571596 | diag_holdout | v5 re-trace: root 1.21 m, size 0.63 |
| 3571794 | diag_holdout | v10 00290 trace |
| 3571963, 3571964 | diag_holdout | v9 epoch 62, epoch 149 |
| 3571598, 3571599 | exp1_depth_anchor | 2-ep fix x1, 2-ep control |
| 3572418, 3572483 | exp1_depth_anchor | 2-ep fix x5, 2-ep fix x20 |
| 3571933, 3571934, 3572482 | exp1_depth_anchor | 10-ep fix x1, 10-ep control, 10-ep fix x5 |
| 3571597 | diag_holdout | failed: v9 00062 not under `logs/anny_model/anny_teacher_v9_mixed/checkpoints/` |
| 3571601 | exp2_children | measurement only (`SKIP_TRAIN=1`): age index + batch-age measurement, natural 2.14% -> balanced 9.91% (:35, :49-51); no training |
| 3571955, 3571956 | exp2_children | sampler pilot, control |
| 3572419, 3572420, 3569949 | diag_cropK | real-photo ages: exp2 pilot, control, v5. Its "small" people are mostly crouching or seated adults (posed skeletal height); only the boy in `360_F_392252267...jpg` is a child: age 0.60, height phenotype 0.48 (3.4) |
| 3571957, 3571958 | exp3_neck_global_rot | pilot, control |
| 3571959, 3571960 | exp4_fingers | region-norm pilot; control + crop trace + hand size |
| 3573518, 3573519 | diag_neck_bones | per-bone dumps: v5, v9 e149, v10, long_pc_dist5; lineage v2/v3/v4 |
| 3573520, 3573521 | exp5_neck_fix | L1 control, geodesic |
| 3575369 | check_3dpw_conversion | 3DPW->Anny label check |
| 3579014 | exp6_bone_fix | exp6 on v5: fix on neck02, head, metacarpal1.L, finger4-1.R; zero-training FAIL (global head 19.1 vs 15.0, :120), fine-tuned PASS (:121) |
| 3579015 | exp6_bone_fix | exp6 on v10: metacarpal2.L/2.R/3.R, no training; PASS (:86) |
| 3579266 | 3dpw-labels | label rebuild into `threedpw_anny_full/labels_rebuilt`: the same counts as the old labels (:25, :39, :65, :67). Log name `logs/3dpw-labels-3579266.out` (dash, not underscore) |
| 3581312 | demo-compare | real-image demo: v5 00099 vs `exp6_v5_bonefix_00001` (the exp6 fine-tuned checkpoint; `[bone_fix] ON, 4 bone(s)` from its args, :427, :607), `DEMO_MODELS` route, 4 photos + 8 3DPW-test frames. By eye, on the 8 of these 12 images the student sent: thin necks with v5, normal width with the fix (the fixed side also had the 2-epoch fine-tune; 3.2). Log name `logs/demo-compare-3581312.out` |
| 3581320 | 3dpw-labels | second label rebuild, after the image folders of 8 sequences were re-extracted on 16 Oct. Files per folder after extraction: courtyard_arguing_00 765, backpack_00 1,262, basketball_00 468, bodyScannerMotions_00 1,257, box_00 1,041, capoeira_00 435, captureSelfies_00 480 -> 750, basketball_01 958. Into `threedpw_anny_full/labels_rebuilt`, from the old v2.2 fits. Result: train 6,437 images / 7,589 people (:25), validation 2,777 / 3,314 (:39; crosscountry still 0, its 162 frames dropped by visibility, :34), test 9,722 / 11,850 (:65); 18,936 label files, 22,753 people (:67), exactly what the fits imply; "unconverted tasks" empty (:68). Missing labels fixed; gender and missing actor not. Log name `logs/3dpw-labels-3581320.out` |
| 3585580, 3585810 | smpl2anny-all | the re-fit test (tasks 9-10) queued on the old GPUs, 16 Oct: start estimate 22 Oct for 3585580, 26 Oct for 3585810 after a cancel and resubmit (reason `Priority`). [Likely] both cancelled before they ran (no logs); an earlier submission, 3585152, too. If `squeue` still shows one, leave it: its tasks would now be skipped (same flags) |
| 3587852, 3587853 | smpl2anny-all | the re-fit test, task 9 and task 10 as two jobs with H100/H200 (and B200, RTXB6000) in the partition list. Both ran on serv-3312, an H200 NVL, and crashed: `no kernel image is available` (PyTorch 2.0.1+cu117 has sm_37-sm_86 only; `.err`). Nothing written (4f) |
| 3589270 | smpl2anny-all | the re-fit test (4e, step 1), v2.3 flags, through `PY_VENV` on serv-3341 (an H100 node). Task 9 = woman p0 (its `.err` names job 3589271), task 10 = man p1 (job 3589270). Man gender 0.0, mean / max 14.3 / 15.8 mm, 231 frames, 1,013 s; woman 1.0, 15.5 / 34.5 mm, 226 frames, 978 s; one shared frame grid; ~4.4 s per frame. Logs `logs/smpl2anny-all-3589270_9.out`, `_10.out` |

**Experiments, one line each:** exp1 depth anchor (placement much better, slope not, likely cost; rule failed);
exp2 children sampler (inconclusive at 2 epochs); exp3 global head loss (null); exp4 fingers crop + region-norm (both null); exp5
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
and the pptx in `docs/` were rebuilt from the current `build.js` on 16 Oct (~11:40 UTC, with the label rebuild
result on slides 6 and 16 and in notes 6, 16 and 17), and again on 17 Oct after notes-only edits: ~03:30 UTC, one
sentence on the re-fit test (job 3589270) in notes 6, 16 and 17; later on 17 Oct, the measured re-fit time (about 28
GPU hours at 4.4 s per frame, 3.5 to 4 hours on 8 free H100s) in place of the old 40-GPU-hour estimate in notes 16
and 17, and in note 17 ("Option 2") the final model's start (v5 with v5's bone fix, 10 epochs, with a control).
The slides were unchanged until then: every slide, layout, media and theme part of the morning 17-Oct pptx is
byte-identical to the 16-Oct one; only `notesSlide6/16/17.xml`, the timestamps and the random section IDs differ.
**Evening of 17 Oct, the children correction** (3.4): rebuilt again from `build.js`; against the morning pptx only
`slide10.xml`, `slide14.xml`, `slide16.xml`, `notesSlide8/10/14/16/17.xml`, the timestamps and the section IDs
differ [Certain, part-by-part diff]. Slide 10: the right stat tile "0.57–0.67 / predicted age of 5 small people in
real photos ..." is now "0.60 / predicted age of the one real child in our photos: the training median, an adult
body"; the left tile says "unproven after 2 epochs" instead of "no effect"; the bullets say "inconclusive" (48.5 vs
53.1 mm, baby floor 0.10), "Diagnosis: ... falls back to the training median ... Domain gap or loss: not yet
separable" and "three different scales". Slide 14, children row: "Inconclusive after 2 epochs: 10 training children
48.5 vs 53.1 mm." Slide 16, item 6: "Children: measure on Relative Human first. Fingers: check the bone-fix hands,
then larger crops." Notes: 10 rewritten (main part and "If asked"); 8 (the men: gender near the training median, the
same pull as age); 14 and 16 (children inconclusive; Relative Human first; the 10-epoch run a low-priority option);
17 ("Both options": the default-off 3DPW shape mask, a decision for the meeting). Slides 10, 14 and 16 were rendered
with LibreOffice and fit in the same space as before. Notes are generated from `build.js`; never edit the `.md`
alone.

**Before the meeting** (Monday 19 Oct): open the pptx once in PowerPoint and check slides 1, 2, 5, 6, 7, 8, 9, 10,
14, 15, 16 and 17 for text that wraps one line longer. They were only rendered with LibreOffice. Slide 8's header bars
have an 11-pt second line sized to fit the bar; slide 15's Wording card already ends at the card's bottom edge in the
LibreOffice render.

**Check a copy** (after `git pull`; the pptx itself must also be opened once):
```
cd /netscratch/najib/multi-hmr && unzip -l docs/supervisor_update_2026-10-16.pptx | grep -c 'ppt/slides/slide[0-9]*\.xml$'; grep -c "before and after the fix" docs/deck_src/build.js; grep -c runForBus docs/supervisor_talk_notes_2026-10-16.md; grep -o "check pending\|image folders will show it\|I will check, re-extract and rebuild\|check is still to do\|I have not checked that yet\|empty or short\|3581320 pending\|3581320, is pending" docs/deck_src/build.js | wc -l; grep -c "14.3 millimetres" docs/supervisor_talk_notes_2026-10-16.md; grep -c "28 GPU hours" docs/supervisor_talk_notes_2026-10-16.md; grep -c "one shared frame grid; the full re-fit" docs/supervisor_talk_notes_2026-10-16.md; grep -c "one real child" docs/supervisor_talk_notes_2026-10-16.md
```
The current deck prints `18`, `2`, `1`, `0`, `3`, `2`, `3`, `1` (18 slides; slide 8's title and its notes; the slide 8
"If asked" line on the four unseen demo images; no line that still calls the image check undone or the label rebuild
pending, where the earlier 16-Oct copy prints `7`; the re-fit test sentence in notes 6, 16 and 17, where the 16-Oct
notes print `0`; the measured re-fit time in notes 16 and 17, where older copies print `0`; the re-fit test sentence of notes 6, 16
and 17 with "one shared frame grid", where the earlier 17-Oct copy, which said "the same frames", prints `0`; the
corrected slide-10 notes, where every copy before the evening of 17 Oct prints `0` and still has the wrong "0.57–0.67,
5 small people" tile).
Anything else is an
older copy: ask the student to `git pull` again. If no
current copy can be had before the meeting, hand-edit the pptx in PowerPoint from the slide list below (slide text
and speaker notes), then port the changes into `build.js` after the meeting and rebuild, so the `.md` notes match
again.

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
   captureSelfies_00, val basketball_01 (image files missing). Fixed 16 Oct: train 4,451 → 6,437, val 2,458 →
   2,777." The notes give the 3579266 counts, say the old guess "the builder ran before the fits finished" is wrong,
   and give the re-extraction (captureSelfies_00 480 -> 750 files) and the 3581320 result (train 6,437, validation
   2,777, test 9,722; old fits, gender still reversed, not trained on before the re-fit). Since 17 Oct the main
   notes end with the re-fit test: "the man now gets the correct gender and fits at 14.3 millimetres mean joint
   error, the woman at 15.5, and both people are fitted on one shared frame grid; the full re-fit of all 87 tasks has not
   run yet."
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
     - the female-looking bodies on men in both models: not the 3DPW gender defect (v5 never saw 3DPW); since the
       evening of 17 Oct: "predicted gender near the training median, about 0.6, while Anny's male end is 0: the
       same pull to the median as for age (slide 10)", cause not settled (3.4);
     - bodies still small or misplaced in depth in both models, and a figure in a painting detected as a person;
     - the hands differ too (two hand bones turned back) and were not judged (slide 9);
     - all twelve holdout panels show the thin neck and its fix wherever a neck is visible.
9. Fingers: about half the error was two flipped bones. 17.3 deg before, 8.3 with the fix after the fine-tune (8.2
   with no training), 15.1 at rest pose before the fix. The notes give the v10 metacarpal result (PVE 74.1 -> 69.9,
   fingers 32.0 -> 17.8).
10. Children: child-like in training, adult on real photos. Corrected on the evening of 17 Oct (3.4). Tiles: "2% →
    10% / children in supervised training with the balanced sampler: unproven after 2 epochs"; "0.10 / predicted age
    of training babies; true 0.015"; "0.60 / predicted age of the one real child in our photos: the training median,
    an adult body" (was "0.57–0.67 / 5 small people", wrong: mostly crouching adults). Main notes: the median pull,
    the five were mostly crouching or seated adults, the one real child predicted as an adult, the sampler
    inconclusive, the proposal (Relative Human first, then limitation or BEV-style weak labels), the default-off 3DPW
    shape mask as a decision. "If asked": posed skeletal height, the 299-line medians, why the spread is not read,
    both synthetic tails, cause not ranked, Anny's child cue (shoulders vs head), three age scales, RH counts and
    metric, the 3DPW constants and the PVE caveat, Anny-Fit.
11. The hidden bug: head anchored, pelvis targeted.
12. Depth fix: bodies drawn much closer to true size (figure 579862).
13. Depth fix: much better size, near vs far still weak (table). Note: "No control was run at ×5 or ×20."
14. Four experiments: results (+ noise floor). Children row since the evening of 17 Oct: "Inconclusive after 2
    epochs: 10 training children 48.5 vs 53.1 mm." (was "No effect ... real photos unchanged"); main notes "Neck and
    fingers: no measurable effect. Children: inconclusive after two epochs".
15. Corrections made, and still to make. The Wording card names the README and the findings as still carrying last
    week's neck story. That is narrower than section 7, which also lists the plan's neck lines (`plan :150-152`,
    `:252`); the slide is left as it is.
16. Still open, and what I test next (7 items). Item 1, "Bone flips: confirm the pilot fix": "Neck: a longer run;
    real images by eye only (slide 8). v10's left arm: two-sided fix, not built yet." Item 3: "No result yet for the
    ×20 run at 10 epochs." Item 5: "Both v5 and the depth fix miss them (4 of 12 holdout renders): not the fix.
    Likely detection; not counted." Item 6, "Children and fingers" (since the evening of 17 Oct): "Children: measure
    on Relative Human first. Fingers: check the bone-fix hands, then larger crops." (was "10-epoch sampler run; ...";
    the notes keep that run as a low-priority option). Item 7, "3DPW labels: three defects (slide 6)": "Missing labels fixed 16 Oct
    (images re-extracted; train 4,451 → 6,437 images). Then gender (m → 0, f → 1), one shared frame grid, re-fit
    all 87 tasks, rebuild, re-check. Proposed: only then use 3DPW." Notes: "the 8 real images I have seen so
    far". Since 17 Oct the main notes add the re-fit test (man 14.3 mm with the correct gender, woman 15.5, one
    shared frame grid, full re-fit not run), and "If asked" gives the measured time: "about 28 GPU hours at the measured 4.4
    seconds per frame (the 17 October test, on an H100), so about 3.5 to 4 hours with 8 free H100s", and says the
    task script now refuses to mix old and new fits.
17. What I claim today, and what I do not (+ the question to the supervisor). Claimed: "in one pilot run, a fixed
    correction plus a 2-epoch fine-tune restores the neck and passes the fine-tune checks"; two flipped hand bones
    were half the finger error. Not claimed: "a proven neck fix (one run, no seed; head direction 14.4° vs 12.0°) or
    why the bones flipped". The question box adds: "v9 also has the neck flip (measured at epoch 149) and needs its
    own bone fix; v10 has a similar left-arm flip." Notes: "in the 8 of its 12 images I have seen, judged by eye
    only"; "about 1 to 2 hours of training", not measured with 3DPW batches; "75.0 at epoch 62, the planned start".
    So the slide and the option-1 notes describe the fine-tune as starting from v9 00062 (1-2 h, with a v9 bone fix
    estimated first), while the final model of section 5, item 7 starts from v5 00099 with v5's own bone fix and
    needs no v9 estimate; if option 2 is chosen, the student should say this at the meeting. Since 17 Oct the
    "Option 2" line of the notes adds the re-fit test (man 14.3 mm with the correct gender, both on one shared frame
    grid, full re-fit not run), the measured re-fit time (about 28 GPU hours, 3.5 to 4 hours on 8 free H100s) and the
    final model's start (v5 with v5's bone fix and the depth fix, 10 epochs, next to a control without the two
    fixes; not run yet). Since the evening of 17 Oct the "Both options" line adds the default-off 3DPW shape mask
    ("on or off is a decision for today (slide 10)") and its 3DPW-error caveat.
18. Backup: before and after, holdout image 579860.

**Results still to come: the lines they make stale.** When a result arrives, edit these lines in `build.js` and
rebuild (or hand-edit the pptx as above).

| result | slide text | speaker notes | what changes |
|---|---|---|---|
| 3DPW re-fit and the v23 check (4e; pending items 1-3) | 6 (the gender and two-person defects), 16 item 7 ("Then gender ..., re-fit all 87 tasks, rebuild, re-check."), 17 (question box) | 6 (If asked, "Gender"), 16 (main: "Then gender, frame grid and a re-fit of all 87 tasks, before any 3DPW test."; If asked: "about 28 GPU hours at the measured 4.4 seconds per frame ... about 3.5 to 4 hours with 8 free H100s"), 17 (If asked, "Option 2", with the same time) | the re-fit test (step 1) is in notes 6, 16 and 17 since 17 Oct (job 3589270), and the 40-GPU-hour estimate and the "mix without a warning" line are replaced by the measured time and "the task script now refuses to mix them". Still to come: the post-check, the full re-fit, check I1 / I3 results, the v23 label counts (expected ~16,900 images), the total runtime |
| the other four demo images (runForBus, sitOnStairs, fencing frames, one photo) | none | 8 (If asked: "I have looked at 8 of these 12 images so far ..."), 16 (main: "the 8 real images I have seen so far"; If asked: "I have looked at 8 of these 12 images so far"), 17 (If asked: "in the 8 of its 12 images I have seen") | how many of the 12 show normal necks with the fix |
| longer fine-tune with the fix (4e; section 5, item 3) | 7 ("Next:" line), 16 item 1 ("a longer run") | 7 (If asked), 16 (main: "I still need a longer run") | neck02, head, girth, global head, PVE, recall at 10 epochs against the 10-epoch control |
| `long_pc_dist20` (4d) | 13 (table note), 16 item 3 ("No result yet for the ×20 run at 10 epochs.") | 13 (main), 16 (If asked), 17 (If asked: "Five, not twenty, because ...") | root error, size, slope, PVE, recall of x20 for 10 epochs next to x5 for 10 epochs |
| the supervisor's answer (slide 17) | 16 item 7 ("Proposed: only then use 3DPW."), 17 (question box) | 16, 17 | record it in the paragraph below |
| the supervisor's children decisions (Path A or B; `MASK_3DPW_SHAPE` on or off) | 16 item 6 | 10 (main: "My proposal ..."; "Separately, a new flag ..."), 17 (If asked, "Both options") | record both in the paragraph below; if the mask is on, section 5, item 7 and note 17 |
| Relative Human evaluation (section 5, item 12, step 1) | 10 (right tile and "Diagnosis" bullet), 16 item 6 | 10 (main and If asked), 16 | per-group mean predicted age, child-vs-adult AUC, gender accuracy, per-group recall |
| the final model's start, if option 2 is chosen (section 5, item 7; pending item 4) | 17 (question box: "Fine-tune v9 ... with the fixed targets and depth ×5? Or first fix gender and frame grid, re-fit all 3DPW tasks, then fine-tune?") | 17 (If asked: "Option 1", with "75.0 at epoch 62, the planned start" and "about 1 to 2 hours"; "Option 2"; "Bone flips in v9 and v10", with a bone fix "estimated on v9 first") | note 17 "Option 2" says it since 17 Oct: v5 00099 with v5's bone fix JSON and the depth fix, 10 epochs (exp7 default), next to a control. Still to come: the review of the exp7 draft (if it changes the start or the length, change note 17), the supervisor's answer, the length if the deadline allows more than 10 epochs |

**The question on slide 17:** fine-tune v9 now (fixed targets, depth loss x5), or first fix the 3DPW labels, re-fit,
then fine-tune? The student proposes the second. Record the supervisor's answer here after the meeting.

**The children decisions (slide 10; 3.4; section 5, item 12):** (1) Path A (Relative Human measurement, then a
limitation) or Path B (A plus BEV-style weak labels, 1-2 weeks of code after exp7); the student proposes measuring
first. (2) `MASK_3DPW_SHAPE` on or off in exp7 (default off). Record both answers here after the meeting.

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
before planning anything that takes more than a day (e.g. the ~28 GPU-hour 3DPW re-fit, the final-model fine-tune
in the old-GPU queue, and writing).

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
  out real paths. It happened again on 16 Oct: a command with `--partition=PARTITION` failed. If a value is not
  known yet (a job id, a partition), ask for it first, then give the full command.
- **Never cancel or resubmit a queued job to make it start sooner.** On 16 Oct the re-fit test waited as job 3585580
  (estimate 22 Oct, reason `Priority`); it was cancelled and resubmitted as 3585810, and the estimate moved to 26
  Oct. Splitting an array into single jobs does not help either: Slurm schedules each array task as its own job. To
  change the partitions or the time limit of a waiting job, use `scontrol update` on it with the real job id
  [Likely allowed for one's own pending jobs; not tried]. Cancel only a job that would duplicate work already
  running elsewhere, or one a decision made unnecessary (e.g. the final model queued before the meeting, and the
  supervisor then picks option 1), and say so. That is a change of plan, not a cancel-and-resubmit.
- **NumPy-only steps run on the login node** (seconds to a minute, no queue): `check_refit_task.py` and
  `build_threedpw_anny_labels.py`. Their Slurm wrappers either have no partition line (`run_check_refit_task.sh`
  lands in the full CPU partition `batch`) or ask for a GPU (`run_build_labels.sh`). Anything that needs torch on a
  GPU goes through Slurm.
- **Match the GPU type to the env before adding a partition.** `multihmr` (PyTorch 2.0.1+cu117) runs only on the old
  GPUs; H100/H200 need `PY_VENV`, and only the converter (and [Likely] the conversion check) can use it (4f). On the
  login node `torch.cuda.get_arch_list()` prints `[]` (no GPU there); `torch._C._cuda_getArchFlags()` lists the
  compiled architectures [Likely].
- **Home quota.** `/home/najib` has a personal limit of about 10 GB [Likely] and was full on 16 Oct; tools that write
  to `~` (conda, pip) then fail with `Disk quota exceeded`. Keep caches on netscratch, e.g.
  `export XDG_CACHE_HOME=/netscratch/najib/.cache PIP_CACHE_DIR=/netscratch/najib/pip_cache` in the login shell
  (`slurm/env.sh` does it inside jobs).
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
