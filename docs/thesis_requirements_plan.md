# Thesis requirements (supervisor, 18 Oct 2026) and the 4-week plan

Written on Sunday 18 Oct 2026, the evening before the meeting of Monday 19 Oct, 16:00. It answers the student's
question: what of the supervisor's list is done, and what to do next for a good grade. `docs/handover.md`, section 11,
points here; where the two disagree on the plan, this file is newer.

**Update, 18 Oct afternoon.** The Relative Human evaluation of the final model has run: job 3612131
(`logs/eval_rh_3612131_0.out`, node serv-9225, 09:40; `models/multiHMR/exp7_v5_final_00009.pt`, its bone fix from the
checkpoint args; val split, 635 images, 2,130 labelled people, IoU 0.3, child threshold 0.45) [Certain, log]. Gender
accuracy 0.642 (v5 0.465), children AUC 0.719 (v5 0.606), recall 0.647 (v5 0.471); table and reading in 3.1, item 7.
Every "pending" for it in the first version of this file (the skeptic's review included) is replaced below. The
`MASK_3DPW_SHAPE=1` arm is running [Likely, the student's message of 18 Oct; no job id or log here yet]. The deck has
a requirements slide since this update (slide 20; handover section 9).

**Sources.** The supervisor's e-mail (pasted by the student on 18 Oct); `docs/handover.md` (0b, 3.4, 3.5, 3.6, 4f,
section 5 item 7); four read-only investigator reports (architecture and baseline, distillation, phenotypes,
temporal and report) and a skeptic's review of the first plan, all of 18 Oct. The skeptic's corrections are applied
here. Spot checks of the code at HEAD `b33d125` plus the working tree. Nothing was run for this file except two local
checks without a GPU: `python3 scripts/analysis/model_cost.py --no-build --all` and
`LOCAL_CHECK=1 bash slurm/experiments/exp8_distill_vitb.sh` (both passed).

**Tags** as in the handover: [Certain] = read in code or in a log line; [Likely]; [Guessing]; [Unverified] = from a
report or the task text, not in the repo or the handover. `3611336:224` = line 224 of that job's `.out` log.
`file.py:L` = that line in `main` at `a9db35c` (18 Oct) or later.

**Repo state that matters** [Certain, `git status`, 18 Oct, 09:18]. Not committed: `train.py` and
`slurm/eval/run_eval_3dpw_test.sh` (modified: `--model_class multi_hmr_original`, `EVAL_MODEL_CLASS`, `EVAL_BONE_FIX`,
`EVAL_3DPW_N`), `docs/experiments_plan.md` (an exp8 section, being written), and three new files:
`scripts/analysis/model_cost.py` (params / FLOPs / latency), `slurm/eval/run_eval_baseline_naver.sh` (the NAVER
baseline) and `slurm/experiments/exp8_distill_vitb.sh` (the ViT-B student). None of them has run on the cluster. The
They are on `main` since `a9db35c` (18 Oct), reviewed twice; the adapter crash of 3.4 item 1 is fixed in that commit
(the four attributes are exposed). Still never run on the cluster: first `CHECK_ONLY=1`, then the smoke tests.

---

## 1. The supervisor's requirements (summarised; his words where they matter)

Three tiers. The labels in brackets are used in the rest of this file.

**(1) Absolutely necessary before submission.** "I will think the thesis is not ready for submission without those
tasks even if your deadline ends." Target: 100% of (1) "within the next one month", so that the last month goes to
(2) and (3).
- **[1a-pose]** A model that predicts correct Anny parameters (pose and shape) on real images. "Pose of all body parts
  must look correct" on (a) the 3DPW test set and (b) any random in-the-wild image.
- **[1a-shape]** "Mesh size, proportions, age, muscles, and other aspects of the 3D shape must visibly change
  according to the person", as in the Anny demo (anny-demo.europe.naverlabs.com). AnnyOne has these labels, so a
  correctly trained model should predict them, "and the visible appearance should change consequently during
  inference if the predicted values are utilized properly".
- **[1b]** "The model must not be exactly the same as the Multi-HMR (or any other existing) model." Either:
  - **[1b-i]** a completely new architecture, any size, that performs better than Multi-HMR: PVE/MPJPE on **both**
    the AnnyOne test and the 3DPW test must outperform the baseline model(s); or
  - **[1b-ii]** a significantly smaller model (fewer parameters and FLOPs) that performs comparably: show how many
    fewer parameters/FLOPs, how much faster inference is, and that PVE/MPJPE are "still more than 95% of the larger
    model's corresponding metrics on both datasets".
- **[1c]** Evaluation on the real-world set with Anny parameters, not (only) SMPL-X. So Anny parameters at least for
  the 3DPW test set.

**(2) Recommended for an acceptable grade.**
- **[2a]** Convert 3DPW train and test to Anny and include the train set in training. He suspects (1) is out of reach
  without it.
- **[2b]** Temporal information: the model understands motion sequences in video, instead of predicting every frame
  on its own (Multi-HMR works frame by frame).

**(3) Nice to have for the best grade.**
- **[3-report]** A well-written report. "If it very obviously reads like an LLM-generated report, that will affect
  the grade." A reader must see that the student understands the topic, has good, well-argued reasons for the model
  and experiment choices, and that related work and analysis are not superficial.
- **[3-code]** A clean codebase and trained models he can verify independently.
- How the work is presented also counts.

[Likely] Implied deadline: one month for (1), one for (2) and (3), so submission around mid-December. The handover
still has no date (section 10). Ask on Monday.

---

## 2. Status matrix (18 Oct, evening)

Short answer to "is something done?": yes, a fair part. [1c], [2a] and the training with AnnyOne phenotype labels are
met. [1a-pose] is close on 3DPW. Not met: [1b] (the model is Multi-HMR with an Anny head), [1a-shape] on real photos
(five phenotypes sit on converter constants), [2b] (nothing exists) and [3-code] (nothing released).

| Req. | Status | Evidence (one number or fact) | Missing |
|---|---|---|---|
| 1a-pose, 3DPW test | partly | 3DPW test, v23 labels, 8,190 images: PVE 89.0 / PA-PVE 57.6 / MPJPE 88.3, recall 99, **precision 64** (3611336:224-229; handover 0b) [Certain]. By eye: 8 3DPW-test frames, necks normal (demo 3612078) | a body-joint MPJPE; a 3DPW-test overlay panel with predicted hands; a seed or a CI; depth slope 0.42 (1 = right) |
| 1a-pose, in the wild | partly, by eye only | 4 photos (3612078): necks normal; the boy is an adult body; placement offsets and ghost bodies unchanged [Certain, by eye]. **Every real-photo demo so far forces the fingers to rest pose** (`--relax_hands 1`, `slurm/demo/run_demo_compare.sh:171`) [Certain] | a demo with predicted fingers; a number on a labelled real set beyond Relative Human (recall v5 47%, final model 65%, job 3612131); offsets and ghosts as stated limitations |
| 1a-shape, visible change | partly / at risk | Men look male now, by eye [Likely]. Medians moved onto the 3DPW converter constants: muscle 0.28 -> 0.48, weight 0.59 -> 0.48, age 0.61 -> 0.56 (demo `[shape]` lines, not matched to people; handover 3.4) [Certain, log]; spreads of muscle, weight, cup size, firmness, age <= 0.05 [Unverified]. Relative Human, final model (3612131:26-32) [Certain, log]: gender accuracy 0.642 (v5 0.465; mean women 0.861, men 0.551 = still mid-range); children AUC 0.719 (v5 0.606), babies 0.438 vs adults 0.536. No correlation, slope or R² exists for any phenotype [Certain] | the `MASK_3DPW_SHAPE=1` arm (running); per-phenotype r / slope / R² on AnnyOne; a rest-pose lineup |
| 1a-shape, trained on AnnyOne labels | met | `--use_anny_shape 1 --alpha_shape 20`, L1 on all 11 values (`loss.py:319-326`; `exp7_final_model.sh:751-752`) [Certain] | nothing |
| 1b, not exactly Multi-HMR | **not met** | `model.py` is the upstream paper model with the output resized to 163x6 rotations and 11 phenotypes, plus a 0-parameter bone fix [Certain] | a route (3.2) and the supervisor's yes |
| 1b-i, new architecture | not started | no candidate. NAVER's `multiHMR_672_L_anny.pt` never evaluated here; the adapter is on `main` (`a9db35c`), fixed, never run [Certain] | the adapter fix; the baseline row |
| 1b-ii, smaller model | partly (code only) | distillation ran 4 times (v6-v8). The old ViT-S students had 1.17-1.54x their teacher's error (bar 1.05x), on the retired labels [Certain]. Cost: ViT-B student 101.4 M params / 603 GFLOPs vs 318.4 M / 1927 for our ViT-L, 3.1x / 3.2x (investigator count; backbone part reproduced by `model_cost.py --no-build`). exp8 written, untracked, never run | the exp8 run; latency in ms; v23 numbers for every student |
| 1c, 3DPW in Anny space | met, with caveats | v23 labels, test 8,190 images / 11,854 people; conversion check OVERALL WARN (old labels FAIL), body joints median 14.1 mm (3611284; handover 3.5) [Certain] | state the caveats in the thesis (3.1, item 4) |
| 2a, 3DPW train + test in Anny, trained on | met | train 5,882 images / 7,584 people, validation 2,622 / 3,311; the final model trains on 30% 3DPW-train (`exp7_final_model.sh:215,757`) [Certain] | nothing (a 50% mix ablation is optional) |
| 2b, temporal | **not started** | no temporal code in the repo (grep) [Certain]; AnnyOne is single images, multi-view, no motion [Likely] | design in section 4, weeks 5-6 |
| 3-report | at risk | draft outside the repo; its 74 mm headline is stale (the thesis model is now 73.7 V1 / 89.0 3DPW) [Certain] | chapter plan (6.1); the current numbers |
| 3-code | **not met** | README describes old labels and models; 0 checkpoints or results committed; 98 tracked files contain `/netscratch`; 79 of 89 commits authored "Claude"; exp8 package committed 18 Oct (`a9db35c`) [Certain] | list 6.2 |

---

## 3. Gaps and feasibility

### 3.1 What the supervisor will find (read before Monday)

1. **[1b] is not met, and no run so far meets it** [Certain]. The model is the paper's Multi-HMR (`model.py`) with
   Anny outputs and training-side fixes. A ViT-B student is a recipe, not an architecture: NAVER already ships
   `multiHMR_672_B` and `_S`. It fits [1b-ii] by wording; he has to say yes to it.
2. **[1a-shape] has a known mechanism** [Certain for the mechanism, Likely for the effect]. On 3DPW, 8 of the 11
   phenotypes are converter constants: age 0.55, muscle, weight, cup size, firmness and the three ethnicity values 0.5
   (`scripts/data/smpl_to_anny.py:264-268`). The final model trained on them at full weight (`MASK_3DPW_SHAPE=0`),
   through two paths: the shape L1 (x20) and the GT mesh built from them (v3d/j3d x100). On real photos the
   predictions sit on those constants. `--mask_3dpw_shape 1` removes only the L1 path (`loss.py:475`).
3. **[1a-pose] on real photos is "by eye, mostly"** [Certain]. Fixed and measured on the holdout: the neck half-turn
   (neck02 175.8 -> 4.8 deg) and the finger flips (17.4 -> 8.3 deg) (3611336:191-209). Open: depth slope 0.42; recall
   87 on the holdout (v5 90.2); placement offsets; ghost bodies; predicted fingers never shown on a real photo.
4. **[1c] caveats he will find** [Certain]:
   - MPJPE on 3DPW-Anny averages all 163 Anny joints, pelvis-centred (`train.py:704-716`), not the 14-joint H36M set.
     Not comparable with published 3DPW numbers.
   - 8 phenotypes are constants, so 3DPW tests pose (plus gender, height, proportions), not shape.
   - The converter fits only the root and the mapped body bones (`scripts/data/smpl_to_anny.py:102,594`): fingers,
     neck and head are identity in the GT, and `--mask_3dpw_hands` removes them from training. So 3DPW PVE/MPJPE
     penalise correct hand articulation. One more reason for a body-joint MPJPE.
   - ~14 mm conversion floor (body joints, median 14.1 mm).
   - Precision 64% on 3DPW test (control 62, v5 54): about a third of the detections match no label. Unlabelled
     bystanders plus ghosts; the split is unknown.
5. **Every headline number is one run, no seed** [Certain]. Two identical configs differed by 7.6 mm PVE (handover
   3.6). The AnnyOne test is 100 images (the last 100 indices, `--val_anny_n 100`). A 3.7 mm margin there is noise.
6. **Root error**: say "1402 -> 728 mm against the matched control (v5 1208)", not "1208 -> 728" (handover 0b).
7. **Relative Human of the final model: done, job 3612131** (18 Oct, 09:40; `logs/eval_rh_3612131_0.out`; handover 3.4)
   [Certain, log]. The first version of this file (and its skeptic's review) called it pending and the figures
   "unverified": that is overtaken; they are measured now.

   | group | labelled | matched | recall | age mean | age median | age < 0.45 | height | proportions | gender acc. | n_g |
   |---|---|---|---|---|---|---|---|---|---|---|
   | adult | 1,680 | 1,113 | 0.662 | 0.536 | 0.530 | 0.012 | 0.500 | 0.671 | 0.663 | 1,079 |
   | teen | 101 | 72 | 0.713 | 0.524 | 0.518 | 0.014 | 0.492 | 0.674 | 0.571 | 70 |
   | kid | 209 | 119 | 0.569 | 0.500 | 0.505 | 0.168 | 0.466 | 0.670 | 0.528 | 108 |
   | baby | 117 | 73 | 0.624 | 0.438 | 0.456 | 0.493 | 0.464 | 0.686 | 0.560 | 50 |

   Child-vs-adult AUC 0.719 (n_pos 192, n_neg 1,113); gender accuracy 0.642 (n = 1,308), mean predicted gender men
   0.551, women 0.861; overall recall 0.647 (1,379 / 2,130) (3612131:25-33). v5 (3611285_0): recall 0.471, gender
   0.465 (men 0.535, women 0.551), age adult 0.631 / kid 0.613 / baby 0.624, AUC 0.606. Reading:
   - [Certain] Better than v5 on every line: recall 47 -> 65% (babies 33 -> 62%); gender 46 -> 64% (women recognised,
     men still mid-range); children separate from adults (AUC 0.61 -> 0.72; babies 0.44 vs adults 0.54; 49% of babies
     below the child threshold, v5 3%).
   - [Likely] Part of the age shift is the 3DPW age label 0.55: every real person moved down (adults 0.63 -> 0.54),
     babies further (0.44). The AUC ranks people, so a common shift alone cannot raise it. So the
     `MASK_3DPW_SHAPE=1` arm is an experiment with two possible outcomes (3.3), not a sure improvement.
   - Caveats [Certain]: more people matched (1,305 children and adults vs 948), so the two AUCs cover different
     detected sets. On the 12 demo images muscle, weight, cup size and firmness sit at ~0.5, the converter constants
     (demo 3612078 `[shape]` lines), so "shape visibly changes per person" holds for gender and partly for age and
     height only.
   - Say it as "measured improvement, not solved" (deck slides 10, 19 and 20).

### 3.2 The route for [1b]

**Primary: [1b-ii], a ViT-B student distilled from the final model** (`slurm/experiments/exp8_distill_vitb.sh`).
- The code exists. The distillation path ran four times (v6-v8). exp8 adds the guards, the student's own bone fix,
  the benchmark and the 95% table [Certain, read].
- Half of the requirement is certain: params, FLOPs and ms are deterministic. ViT-B: 3.1x fewer params, 3.2x fewer
  FLOPs than our ViT-L; about 4x fewer params than NAVER's ~414 M [Likely, hand estimate].
- Outside evidence: NAVER's own SMPL-X ViT-B equals its ViT-L on 3DPW (PVE 94.0 vs 94.1), at 43 vs 74 ms on a V100
  (NAVER README) [Certain, fetched].
- A defensible story: Anny output on a 3x smaller backbone through projector-fed teacher heads, feature and output
  distillation, and the bone and depth fixes carried over.

**Which bar.** He wrote "comparable to the baseline model(s)" and "95% of the larger model's metrics". The baseline
is Multi-HMR, i.e. NAVER's `multiHMR_672_L_anny`. Our ViT-L is "the larger model" only if he accepts it as such.
- Report both ratios, per dataset and metric: student vs our ViT-L, and student vs NAVER-L. The stricter one is the
  bar. Ask on Monday which one he means.
- Use x1.05 (stricter than /0.95): against our ViT-L, AnnyOne holdout V1 PVE <= 77.4, PA-PVE <= 52.8; 3DPW test
  PVE <= 93.4, PA-PVE <= 60.5, MPJPE <= 92.7; recall >= 82.7 (AnnyOne) / 94 (3DPW). The AnnyOne V1 MPJPE of the
  final model is in `logs/exp7_final_model_3611336.out` (V1 row), not yet written down.
- exp8's own table uses /0.95 (targets 77.6 / 93.7) and the re-measured teacher only (`exp8_distill_vitb.sh`,
  header "THE 95% RULE"). Read its ratio column, apply x1.05 by hand and add the NAVER column by hand.

**Odds** [Guessing]. 95% on 3DPW: likely (NAVER's B = L there). 95% on AnnyOne: about 50/50; NAVER's B is 14.7% worse
than its L on synthetic BEDLAM-val (67.2 vs 58.6), and AnnyOne is synthetic.

**Fallback (decide ~10 Nov).**
- The honest fallback: the same ViT-B trained longer, with the ratio reported as measured.
- A ViT-L cut to its first 12 blocks [Guessing]: ~167 M params (only 1.9x fewer), and the teacher's heads were trained
  on block-24 tokens, so they need retraining. Not planned.
- [1b-i] through the temporal refiner (weeks 5-6) plus a per-frame comparison with NAVER, only if the NAVER evaluation
  shows the final model already beats it on both sets. The refiner cannot help on AnnyOne (single images), and he may
  not count a head on frozen features as a new architecture [Guessing].

### 3.3 [1a-shape]

- The `MASK_3DPW_SHAPE=1` arm (one exp7 run, ~2 h; running since 18 Oct [Likely, the student's message]) will [Likely]
  bring back the spread of the collapsed phenotypes. It can also cost something: the final model's children gain on
  Relative Human (AUC 0.61 -> 0.72, 3.1 item 7) came with the 0.55 age anchor on, so it may shrink or stay with the
  mask. Two outcomes; read both the spreads and the Relative Human run of the mask arm (section 4, week 1, item 6).
- Predictions, written before the run: muscle, weight, cup size, firmness and age widen back to std >= 0.06 on the
  demo and on Relative Human; their medians leave 0.5 / 0.55; adult age moves toward ~0.6. Gender, height and
  proportions: no change beyond noise. AnnyOne holdout PVE within 7.6 mm. 3DPW-test PVE equal or a few mm worse,
  because the 3DPW GT meshes use the neutral constants (`exp7_final_model.sh` header, `MASK_3DPW_SHAPE`).
- Whether the values are *right* on real people cannot be measured: no real set we have gives GT for muscle or
  weight. Evidence we can give: the rest-pose lineup (looks like the Anny demo), per-phenotype r / slope / R² on
  AnnyOne, and a small manual rating (100-200 people, build and muscularity in 3 levels, 2 raters, ~2-3 h).
- If the spread does not come back: mesh-term decoupling for 3DPW people (the 8 constants detached in the predicted
  mesh, ~0.5 day of code) [Guessing it is enough].
- Not fixable in these 4 weeks next to [1b]: children (read as short adults) and the depth slope 0.42. Both become
  limitations, with their mechanism.

### 3.4 Small code changes that block the plan

1. **NAVER adapter crash** [Certain, read; FIXED in `a9db35c` on 18 Oct, exactly as below]. `prepare_gt` reads `self.model.nearness` (`train.py:242`), `.img_size`
   (`:246`), `.patch_size` (`:256-261`) and `.person_center_idx` (`:232`, `:336`). `OriginalMultiHMREval` exposes only
   `net` and `body_model`, so `nn.Module.__getattr__` raises on the first batch. So the baseline evaluation has
   never run, whatever a task list says. Fix, in `OriginalMultiHMREval.__init__`:
   `self.nearness = True; self.img_size = net.img_size; self.patch_size = net.patch_size;
   self.person_center_idx = net.person_center_idx` (all exist on `Multi_HMR`, `multi_hmr_anny/multi_hmr.py:80,83,107`).
   GT meshes are safe: `prepare_gt` uses the Trainer's own Anny model.
2. **No masked-evaluation flag** [Certain]. There is no `--eval_mask_helper_joints`; `model.mask_helper_joints` is set
   only by the diagnostics (`model.py:156`). Without it NAVER (masked by design) meets our unmasked V0, a +3.8 mm
   handicap for ours (`README.md:320`). ~5 lines in `train.py` plus a pass-through in `run_eval_3dpw_test.sh`. To be
   written.
3. **`kd_shp` is not masked** [Certain]. `train.py:532`: L1 to the teacher's phenotypes on every person, 3DPW people
   included. With `--mask_3dpw_shape 1` the student would still copy the teacher's constants. Mask it on the 8
   constants for 3DPW people with `self.loss._shape_label_keep` (`loss.py:282-292`). ~5 lines. To be written before
   exp8 runs with `MASK_3DPW_SHAPE=1`.
4. **Latency of NAVER**: `model_cost.py --bench` skips the original class (its docstring). Add ~20 lines, or quote
   NAVER's 74 ms (V100, SMPL-X, their README) with that caveat. State the FLOP convention: 1 multiply-add = 2 FLOPs
   (ViT-L backbone 958.6 GMACs = 1917.2 GFLOPs at 672 px). `--bench` does 20 warm-up and 50 timed forwards.
5. **Body-joint MPJPE** in `evaluate()` (~30 lines), next to the 163-joint one. To be written.
6. **Demo overrides**: `run_demo_compare.sh` hard-codes `--relax_hands 1 --rest_pose 0` (`:171-172`). Add env
   overrides (`RELAX_HANDS`, `REST_POSE`, `NEUTRAL_PHENO`; ~5 lines) for the finger demo and the rest-pose lineup.
   `demo.py` already has `--rest_pose` and `--neutral_phenotypes` (`demo.py:810-818`). To be written.
7. **`scripts/diagnostics/pheno_responsiveness.py`** (CPU, 0.5-1 day): reads existing `per_person.csv` files
   (`pheno_gt_*`, `pheno_pred_*`, written by `diag_holdout_trace.py`). To be written.

### 3.5 Cluster budget: one long job per experiment, not chunks

- The cost is queue wait, paid once per job. On 16 Oct the old GPUs were estimated at 22-26 Oct (reason `Priority`);
  exp7 in fact started on 18 Oct at ~05:28 (handover 0). `sbatch --dependency=afterany` does not skip the queue.
- Time limits (`sinfo -s`, handover 4f): A100-40GB, A100-80GB, A100-PCI, RTXA6000, L40S 3 days; A100-RP, RTXA6000-AV,
  L40S-AV unlimited. H100/H200 do not run the training env (torch 2.0.1+cu117). v8 ran 48 h on A100-80GB.
- Chunking a student into 5-h jobs breaks it in five ways [Certain, read]: (a) `--init_heads_from_teacher 1` runs after
  the `--pretrained` load (`train.py:1216`) and resets all 67 head tensors every chunk; (b) `--freeze_heads_epochs`
  re-freezes them; (c) `--start_2d_epoch` compares with the per-run epoch counter (`loss.py:390`); (d) checkpoints
  restart at `00000.pt` and StepLR restarts, so the LR decay never fires; (e) Adam state is lost each time (the exp7
  smoke test after its restart: PVE 83.5 after 20 steps vs v5 70.5; handover 0a, step 6).
- Total for (1): ~45-55 GPU-h (table at the end of section 4). Small next to the queue.

### 3.6 Verdict: can (1) be 100% done in 4 weeks?

Feasible, with a real risk and no buffer [Likely]. The baseline, the shape mask and the shape evidence fit in weeks
1-2. The student is the long pole: submitted Tue 20 Oct, a few days in the queue, ~24-42 h of training, so a result
around Sun 8 Nov; the fallback decision on Tue 10 Nov; (1) closes on Sun 15 Nov. Say this on Monday.

---

## 4. The 4-week plan (19 Oct - 15 Nov), then (2) and (3)

Commands run from `/netscratch/najib/multi-hmr` on the login node, `conda activate multihmr` first. Each command
below is either runnable now (after the commit), or marked **to be written** when it depends on code that does not
exist yet.

### Monday 19 Oct, evening (after the meeting)

1. Fix the adapter (3.4, item 1). Check: `grep -c 'self.nearness = True' train.py` prints `1`.
2. Commit the six pending files in one commit (this file and the handover with them or separately), push to `main`
   (separate block, handover section 2), `git pull` on the cluster. Check there:
   `grep -c -- '--model_class' train.py` (above `0`) and `ls slurm/experiments/exp8_distill_vitb.sh`.
3. **NAVER baseline**: the checks, a 50-image smoke test, then the full run (AnnyOne test-100 + all 8,190 3DPW-test
   images; 40-min limit; the script refuses while a job of the same name is queued, so wait for the smoke test):
   ```
   cd /netscratch/najib/multi-hmr && git pull && conda activate multihmr && CHECK_ONLY=1 bash slurm/eval/run_eval_baseline_naver.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && EVAL_3DPW_N=50 bash slurm/eval/run_eval_baseline_naver.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && bash slurm/eval/run_eval_baseline_naver.sh
   ```
   Read with the `grep` in the script's header. The line `[model_class] multi_hmr_original: ... loaded N/N tensors
   (100.0%)` must be there. Until 3.4 item 2 exists, label the row: NAVER masked by design, ours unmasked (V0).
   Not known: whether NAVER trained on our last-100 AnnyOne images. If it did, its AnnyOne number is optimistic.
4. **`MASK_3DPW_SHAPE=1` arm** (the fix arm only; `bash`, not `sbatch`: the script runs its guards and sets `--time`).
   **Running since 18 Oct** [Likely, the student's message; no job id here yet]: check `squeue` and its log first and do
   not submit it a second time (handover 10). The commands, for the record:
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && MASK_3DPW_SHAPE=1 CHECK_ONLY=1 bash slurm/experiments/exp7_final_model.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && MASK_3DPW_SHAPE=1 CONTROL_RUN=results/experiments/exp7_final_model/v5_root_dist1_nofix_v23_e10 bash slurm/experiments/exp7_final_model.sh
   ```
   RUN becomes `v5_pc_dist5_bonefix_v23_e10_shpmask` (`exp7_final_model.sh:205`). `CONTROL_RUN` is needed: by default
   the table looks for a `_shpmask` control (`:225`), which will not exist, and prints "-". ~2 h + trace + 3DPW-test
   evaluation, 5-h limit. The job stops if `train.log` lacks `[loss] mask_3dpw_shape:`. Compare it with the unmasked
   fix arm `v5_pc_dist5_bonefix_v23_e10` (both `holdout_ft/summary.json` and `eval_3dpw_test.log`).
5. **Relative Human of the final model: DONE 18 Oct**, job 3612131 (3.1, item 7). The command, for the record:
   ```
   cd /netscratch/najib/multi-hmr && MODEL=/netscratch/najib/multi-hmr/models/multiHMR/exp7_v5_final_00009.pt sbatch --array=0 slurm/diagnostics/eval_relative_human.sh
   ```

### Week 1 (Mon 19 - Sun 25 Oct): baseline, shape decision, start the student

GPU: ~5 h training + ~2 h evaluation, plus the exp8 job in the queue. Code: ~2.5 days.

1. **Tue 20: code.** The `kd_shp` mask (3.4, item 3) and the masked-evaluation flag (3.4, item 2). Commit, pull.
2. **Tue 20: exp8 inputs** on the login node (URLs as exp8's refusal prints them; the B file by analogy with the S
   one [Likely]):
   ```
   cd /netscratch/najib/multi-hmr/models/multiHMR && wget -nc https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_B.pt
   ```
   ```
   mkdir -p /netscratch/najib/torch_cache/hub/checkpoints && wget -nc -P /netscratch/najib/torch_cache/hub/checkpoints https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
   ```
3. **Tue 20: submit the ViT-B student**, from the exp7 teacher, with both shape masks, as one long job, and its smoke
   test next to it (whichever starts first shows a start-up crash). Do not wait for the `MASK_3DPW_SHAPE=1` teacher:
   the student's shape comes from its own 3DPW L1 (masked by `--mask_3dpw_shape 1`) and from `kd_shp` (masked by
   3.4, item 3). Do not submit with `MASK_3DPW_SHAPE=1` before item 3 is in: through `kd_shp` the student would copy
   the teacher's constants.
   ```
   cd /netscratch/najib/multi-hmr && git pull && conda activate multihmr && MASK_3DPW_SHAPE=1 ALLOW_RECIPE_MISMATCH=1 EPOCHS=100 NB_MAX_CKPT=10 CHECK_ONLY=1 bash slurm/experiments/exp8_distill_vitb.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && MASK_3DPW_SHAPE=1 ALLOW_RECIPE_MISMATCH=1 SMOKE=1 bash slurm/experiments/exp8_distill_vitb.sh
   ```
   ```
   cd /netscratch/najib/multi-hmr && MASK_3DPW_SHAPE=1 ALLOW_RECIPE_MISMATCH=1 EPOCHS=100 NB_MAX_CKPT=10 bash slurm/experiments/exp8_distill_vitb.sh
   ```
   - `ALLOW_RECIPE_MISMATCH=1`: the exp7 teacher trained with the mask off, and exp8's teacher guard compares
     `mask_3dpw_shape` (header, GUARDS). It never relaxes the person-centre or bone-fix checks.
   - `EPOCHS=100` x 1000 iterations x batch 8 = 800,000 images; LR halves every 40 epochs (default 2/5 of EPOCHS);
     exp8 derives `--time` from EPOCHS with a 6-images/s rule (`NEED_MIN`; about 42 h here), inside the 3-day limit.
     [Guessing] ~9-10 images/s on an A100, so ~24 h. exp8's default of 30 epochs (240,000 images, a tenth of v8's
     2.4 M) is [Guessing] too short; the v6 line was still improving at 4.6 M images.
   - `NB_MAX_CKPT=10` keeps the last 10 epochs; copy the last checkpoint out when it ends.
   - Batch 8 with ViT-B + the ViT-L teacher may not fit 40 GB [Guessing]; the smoke test shows it. Then `BATCH=4`.
   - For the record, what exp8 composes is the v8 command (`slurm/train/train_distill_vits_v8_final.sh`) with these
     changes: `--backbone dinov2_vitb14`; init `multiHMR_672_B.pt` with `--pretrained_remap 1 --load_only_backbone 1`
     (v8: its own ViT-S checkpoint, remap 0); teacher = the final model (v8: v9 e62); `--init_heads_from_teacher 1
     --freeze_heads_epochs 5`; `--bone_fix <RUN>/teacher_bone_fix.json`; the final-model recipe (`--gt_center_joint
     person_center --alpha_dist 5 --alpha_transl 5 --boost_neck_weight 4 --boost_hand_weight 4 --rotmat_loss l1
     --alpha_rotmat 0.1 --mask_3dpw_hands 1 --exclude_indices_file ...`); v23 labels (v8: `threedpw_anny_full`);
     batch 8 (v8: 16). `LOCAL_CHECK=1 bash slurm/experiments/exp8_distill_vitb.sh` prints the whole command.
4. **Wed-Thu: the first honest [1b] table.** The NAVER full run (Monday, item 3). Then, once 3.4 item 2 exists,
   re-evaluate the final model and the no-fix control with helper bones masked (**to be written**: the same
   `run_eval_3dpw_test.sh` call as exp7's step 4, plus the new flag; for the final model
   `MODEL_ENTRY="/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt|0|dinov2_vitl14|final_exp7_V1"`). 3 x 15 min. Call the control "no-fix control" in every table: it has the same architecture as
   the final model and isolates the two fixes, not the architecture. A same-data NAVER-architecture control is not
   possible (no training code for `Multi_HMR`).
5. **Phenotype numbers, CPU** (3.4, item 7): v5 (`results/diagnostics/holdout_<TAG>/per_person.csv`) and the final
   model (`results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/holdout_ft/per_person.csv`): n, GT std,
   prediction std, bias, MAE vs a constant-median predictor, Pearson r, Spearman, OLS slope, R², PVE-T; the 3x4
   scatter grid. From the Relative Human `per_person.csv`: gender AUC, within-image pairs, collapse index. 0 GPU.
6. **Relative Human of the mask arm** when it is done. `TAG` is needed: for `.../train/checkpoints/00009.pt` the
   default tag is `train_00009` for every exp7 run (`eval_relative_human.sh:34-35`), so outputs would collide:
   ```
   cd /netscratch/najib/multi-hmr && MODEL=/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt TAG=exp7_final_shpmask_00009 sbatch --array=0 slurm/diagnostics/eval_relative_human.sh
   ```
7. **Per-module cost table** (build mode on the login node, CPU; needs anny and the DINOv2 hub cache):
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && mkdir -p results/final && python scripts/analysis/model_cost.py --ckpt /netscratch/najib/multi-hmr/models/multiHMR/exp7_v5_final_00009.pt --ckpt /netscratch/najib/multi-hmr/models/multiHMR/multiHMR_672_L_anny.pt --json results/final/model_cost_final_naver.json
   ```
8. **Sun 25: decision on the mask arm.** If the spreads come back (3.3) and AnnyOne / 3DPW stay within noise, the
   mask arm becomes the final ViT-L model. If not: decoupling (3.3), one more exp7-style arm in week 2.

### Week 2 (26 Oct - 1 Nov): the figures that answer [1a]; the student runs

GPU: a few GPU-minutes of demos, 2 x 15 min of evaluation, optionally one 2-h arm. Code: ~1.5 days.

1. **Rest-pose lineup** on the 12 demo images: v5, final, mask arm; predicted vs neutral phenotypes; every person in
   rest pose on one ground line, same scale. Needs 3.4, item 6. This is the figure for "visibly changes".
2. **Predicted fingers on real photos**: the same images with `--relax_hands 0` (3.4, item 6), plus a 3DPW-test
   overlay panel with hand crops. Until then, never say "hands look correct on real photos".
3. **Old students on v23** (2 x 15 min; array 2 = v6c, 4 = v8, `run_eval_3dpw_test.sh:60,62`):
   ```
   cd /netscratch/najib/multi-hmr && EVAL_LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --array=2,4 slurm/eval/run_eval_3dpw_test.sh
   ```
4. Body-joint MPJPE path (3.4, item 5).
5. If the mask arm failed: the decoupling arm (~2 h).
6. Watch exp8 without touching it: its start-up line checks are automatic; read the in-training curve every 5 epochs
   (AnnyOne holdout V0 and 300 3DPW-validation images) in its `train.log`. Never cancel and resubmit to start sooner
   (handover 10).
7. Writing: the Method and Data chapters (6.1).

### Week 3 (2 - 8 Nov): the student's result

GPU: post-steps run inside exp8. Code: ~1 day.

1. exp8 ends (~8 Nov if the queue takes a few days): `final_table.json`, `model_cost.json` (params, FLOPs, ms on the
   job's GPU, both models on the same GPU), holdout traces, 3DPW-test of student and re-measured teacher.
2. **Bone-fix gate.** exp8 re-estimates the student's own fix on 200 training images and also traces it with the
   teacher's fix. Compare `holdout_student/` with `holdout_student_trainfix/`. Keep the re-estimate only if a bone
   shows a new FLIP above 90 deg in the per-bone dump.
3. **Checkpoint, Sun 8 Nov.** The x1.05 rule (3.2) on both sets against our ViT-L; the ratio against NAVER-L next to
   it. Also compare the 3DPW numbers with the mask arm: a masked student can lose a few mm on 3DPW against neutral-body
   GT for a reason unrelated to its size.
4. Paired per-image bootstrap CI for each student-vs-teacher difference (CPU, from the `per_person.csv` files).
5. Writing: the Experiments chapter on the per-frame model (depth, neck, shape mask, Relative Human).

### Week 4 (9 - 15 Nov): close (1)

GPU: 2-4 job slots. Code: ~1.5 days.

1. **Tue 10 Nov: fallback decision.** If the student misses the bar on AnnyOne and its curve was still falling: train
   it longer from its last checkpoint (**to be written**: an exp8 resume mode with `--pretrained <ckpt>
   --pretrained_remap 0 --load_only_backbone 0`, `INIT_HEADS=0`, no head freeze, a new RUN name and a hand-set LR;
   3.5 says why a plain restart is wrong), and report the ratio as measured.
2. **The final table, one protocol**: helper bones masked for every row, detection threshold 0.2, same GT intrinsics.
   Rows: NAVER-L, no-fix control, final ViT-L (and mask arm), ViT-B student, v8 ViT-S (v23). Columns: PVE, PA-PVE,
   MPJPE (163-joint and body-joint), recall and precision on AnnyOne test-100 and 3DPW-test v23; params, GFLOPs, ms.
   Optionally NAVER with upstream's 6-phenotype filter (the local copy forwards all 11, `multi_hmr_anny/multi_hmr.py`
   local patch).
3. One more seed of the final ViT-L configuration if a slot is free (~2 h; exp7 with a new `RUN`); otherwise cite the
   7.6 mm pilot noise and the bootstrap CIs.
4. Architecture figure and module table (paper Multi-HMR / NAVER Anny / ours / student) with per-module counts from
   `model_cost.py`.
5. Freeze the (1) numbers: `results/final/summary.json` and a provenance table (number -> log:line). Start the release
   (6.2, items 1, 2 and 9).

### Weeks 5-6 (16 - 29 Nov): temporal [2b], the minimal variant

- Cache one 1024-d person token per labelled v23 person-frame from the frozen final model (P1 at the GT cells; P2
  through detection and matching): ~1 GPU-h [Likely]. `model.py` must return `token_out` (~10 lines).
- `temporal_refiner.py`: 1024 -> 256, relative-time embedding, 2-layer encoder, zero-initialised residual back to 1024,
  frozen `decpose` / `deccam` / `decshape`, attention-pooled shape per window. T = 16 (1.6 s at 10 fps); 3 seeds.
- Baselines on the same tokens: B0 per-frame; B1 shape averaging; B2 Gaussian / Savitzky-Golay smoothing; B3 both.
- Metrics: PVE, PA-PVE, acceleration error on labelled triples (10 fps units), per-track phenotype std, rest-mesh std.
  Under P1, re-evaluate B0 under P1 too: never set a P1 number against the 89.0.
- ~5-7 code-days, ~10 GPU-h. If it does not beat B3 on acceleration at equal PVE, say so; do not claim "understands
  motion".
- `demo_video.py` (1-2 days) only if the refiner wins: the strongest presentation asset.
- Cheaper fallback (3-4 days): shape consistency over a track only. It models identity, not motion; probably not
  enough for [2b] alone [Likely].

### Weeks 7-8 (30 Nov - 13 Dec): writing, clean-up, presentation

Week 7: Introduction, Related Work, Conclusion, Limitations, full revision; codebase items 3-8 and 10 (6.2). Week 8:
the presentation and **pure buffer**. (3) is where the grade is decided; do not let experiments eat week 8.

### What to drop

- The ViT-S student and the no-KD ablation, unless the student passes by 8 Nov and slots are free.
- The ViT-L 12-block cut (3.2).
- The two-sided bone fix for v10's left arm; v10's shape drift; children sampler experiments; the depth-slope
  investigation (slope 0.42 becomes a limitation, with the anchor/target mechanism).
- A masked re-run of the control; a 50% 3DPW-mix ablation; a third seed.
- End-to-end temporal training through the ViT (~12 h for 10 epochs at T = 8).
- Chains of 5-h jobs (3.5).

### Budget

| Item | GPU-h | Code-days |
|---|---|---|
| NAVER adapter fix, masked-eval flag, NAVER + 2 masked evaluations | ~1.5 | 0.5 |
| `MASK_3DPW_SHAPE=1` arm + its Relative Human (the final model's: done, 3612131) + its 3DPW test | 5-6 | 0 |
| Phenotype stats, RH CPU stats, demo overrides, lineup, finger demo | 0.3 | 2 |
| ViT-B student, one job, 100 epochs, with its post-steps | 25-45 | 0.5 (`kd_shp` mask) |
| Old students on v23, body-joint MPJPE, final table, figures | 1 | 2 |
| Extra seed (optional), fallback resume (if needed) | 2-20 | 0.5 |
| **Total for (1), 4 weeks** | **~35-75** | **~6** |
| Temporal, weeks 5-6 | ~10 | 5-7 |

---

## 5. Monday 19 Oct, 16:00: what to say and ask

1. **Done for (1c) and (2a).** 3DPW train/val/test converted to Anny with converter v2.3 (gender convention, shared
   frame grid, no missing actors; body-joint error median 14.1 mm; post-check PASS on 60 sequences, independent check
   WARN for body-model differences). The final model trains on 30% 3DPW-train and is evaluated in Anny space on all
   8,190 test images. **Caveats I must state:** MPJPE is over all 163 Anny joints; 8 of 11 phenotypes are converter
   constants; hands, fingers and neck are identity in the GT, so 3DPW PVE penalises articulated hands. **Ask:** is a
   body-joint MPJPE acceptable? Should shape on 3DPW also be fitted to the SMPL surface?
2. **The final model, one run each, noise 7.6 mm PVE.** AnnyOne-100 V1: PVE 73.7 / PA-PVE 50.3 / recall 87 (v5: 70.5 /
   53.3 / 90.2, so the best synthetic PVE is still v5's). 3DPW-test v23: PVE 89.0 / PA-PVE 57.6 / recall 99 /
   **precision 64** (v5 121.8; no-fix control 91.5). Neck and finger half-turns corrected; root error 1402 -> 728 mm
   against the matched control; depth slope still 0.42. **Ask:** is our last-100 tail "the AnnyOne test set", or is
   there an official split? Could NAVER's Anny checkpoint have trained on those images?
3. **Honest gaps.** (a) Architecturally this is Multi-HMR with an Anny head, so (1b) is open. (b) On real photos
   muscle, weight, cup size, firmness and age sit on the 3DPW converter constants, because they were trained at full
   weight; a masked run goes in this week. (c) Children and gender on Relative Human: v5 reads children as short
   adults (AUC 0.61, gender 46.5%); the final model is better on every line (AUC 0.72, babies 0.44 vs adults 0.54,
   gender 64%, recall 65%; job 3612131) but not solved: men are still mid-range, and part of the age shift is likely
   the 3DPW age anchor 0.55. **Ask:** children as a stated limitation, or weak age labels from Relative Human (1-2
   weeks)? (d) About a third of 3DPW-test detections are unmatched (unlabelled bystanders plus
   ghosts). (e) NAVER's Anny checkpoint has not been evaluated here yet; the adapter exists and runs this week, under
   the same protocol (helper bones masked for every row).
4. **Proposed route for (1b): a ViT-B student** (101 M params, 603 GFLOPs at 2 x MAC; 3.1x / 3.2x below our 318 M /
   1927; ~4x fewer params than NAVER's ~414 M, a hand estimate), distilled from the final model with feature +
   output KD and projector-fed teacher heads; reported with measured ms on one GPU and the error ratio (<= 1.05x)
   on both test sets. **Ask:** (i) is the bar against our ViT-L, against NAVER-L, or both (I will report both)?
   (ii) Does a smaller Multi-HMR-style model with this recipe count as "not exactly the same as Multi-HMR", given
   that NAVER's own ViT-B exists? **Risk stated:** NAVER's own B is 15% worse than L on synthetic BEDLAM, so
   AnnyOne may fail. One long job, queue permitting; result by ~8 Nov, so (1) finishes at the very end of the month,
   with no buffer.
5. **Temporal (2b):** a 2-layer refiner on frozen per-frame tokens, 16-frame windows, per-track shape pooling
   (VIBE / TCMR style), trained on 3DPW-train tracks, evaluated with acceleration error at 10 fps on the converted
   labels plus PVE, against smoothing baselines. **Ask:** does a refiner on frozen features count as "understands
   motion"? Is a shape-consistency window alone acceptable if time runs out? **And the submission date.**

After the meeting: write his answers into handover section 9. Then, before anything else that evening: fix the
adapter (3.4, item 1), commit the pending files, submit the baseline smoke test (the `MASK_3DPW_SHAPE=1` arm is already running)
(section 4, "Monday 19 Oct, evening").

---

## 6. Writing and codebase clean-up

### 6.1 The report ([3-report])

- **Write as you go, one chapter per week, from that week's measured results.** Week 1-2: Method (Anny vs SMPL-X;
  the architecture as it is; the losses; the depth anchor and the bone correction, each with its reason) and Data
  (AnnyOne; the 3DPW -> Anny conversion: joint-based fit, virtual joints, one shared body; the three defects and how
  they were found: rest stature 1.753 vs 1.632 m for the reversed gender, the stride-phase bug, the missing actor).
  Week 3: Experiments on the per-frame model (depth, neck, the shape collapse with the mask ablation, Relative Human).
  Week 4: Distillation (recipe, the 95% rule, params / FLOPs / ms). Weeks 5-6: Temporal, and Related Work. Week 7:
  Introduction, Conclusion, Limitations, full revision. Week 8: presentation and buffer.
- **Own voice.** Write each section first as ~10 handwritten lines: what was found, and why. Then expand. No
  [Certain] / [Likely] tags, no job IDs, no v5 / exp7 names: define 3-4 model names once (e.g. Base, Final, Student,
  Temporal). Do not carry the handover's or the deck's style into the thesis.
- **Every design choice gets a reason plus a control or a citation:** partial fine-tuning, the helper-bone mask,
  `alpha_shape`, the 30% mix, the depth anchor, the output-level bone correction, frozen-feature temporal modelling.
- **Every number gets its noise** (7.6 mm, a CI). Specific failures make it credible and show understanding: the
  anchor/target mismatch behind the depth bug; the half-turn flips of neck and hand bones; the reversed gender
  convention; the collapse onto converter constants; the 163-joint MPJPE caveat; the 14 mm conversion floor.
- **Related work must cover** HMR, SPIN, HMR 2.0; ROMP, BEV, Multi-HMR; SMPL, SMPL-X, Anny; AGORA, BEDLAM, AnnyOne;
  conversion between body models; video: HMMR (origin of the acceleration metric), VIBE, TCMR, MPS-Net, GLoT,
  SmoothNet, PHALP / 4DHumans, WHAM; distillation: Hinton et al. (KD), FitNets, DINOv2's own distillation. Read the
  papers, not summaries; one paragraph each on how it relates to a choice made here.
- **Figures from measured results only:** architecture + module table; depth before/after; neck panels; the
  phenotype responsiveness grid; the rest-pose lineup; the main results table; params / FLOPs / ms vs PVE; temporal
  acceleration vs PVE; a side-by-side video.
- **AI disclosure.** 79 of 89 commits are authored "Claude". Do not rewrite history. Follow the university's AI-use
  rules in the thesis, and be able to explain every component at the defence.

### 6.2 The codebase ([3-code]), in priority order

1. **Model release**: `models/RELEASE.md` with sha256, size, config, source job and expected metrics for the final
   model (exp7 00009), the mask arm, the no-fix control, v5 00099, the student and NAVER's base checkpoint. Hosted
   (HF Hub, Zenodo or university storage, within the NAVER / Anny non-commercial licence) with
   `scripts/release/download_models.sh`, which checks the checksums.
2. **One reproduce script** (`scripts/eval/reproduce.py` plus a Slurm wrapper): AnnyOne V1 / V0, 3DPW-v23 test,
   Relative Human, params / FLOPs / ms, the NAVER baseline; a committed `RESULTS.md` with tolerances.
3. **Data release**: the v23 label index with relative paths (or the exact rebuild command and checksums),
   `bone_fix.json`, `results/diagnostics/manifest.json`; remove the committed old v2.2 index.
4. **Paths**: `ANNYONE_ROOT`, `THREEDPW_ROOT`, `CKPT_ROOT`, `RH_ROOT` env vars; `/netscratch` only in `slurm/env.sh`.
5. **Environment**: `environment.yml` and a pip lock exported from the cluster env; pin anny, torchvision, xformers and
   the DINOv2 commit; fix `README.md:134` (CUDA 11.7, not 12.1).
6. **README rewrite** around the thesis model: contribution, quick start, results table on v23 with the NAVER row,
   limitations; drop the stale lines (`README.md:119,128,200,227,310-318,340`).
7. **Tag, then delete dead code** (tag e.g. `experiment-record-2026-10`): `archive/`, `scripts/patches/`,
   `slurm/history/`, the v3 / npz converters, `results/3dpw_to_anny_renders`, old demos, `app.py`.
8. **Hygiene**: strip the "✅ FIX" / "ADD THIS" / patch-name comments (`train.py`, `model.py:345,513,529`,
   `demo.py:640`); the final recipe in `configs/final_model.args`, read by `train.py` and the job script; a real name
   for the final model.
9. **Commit the evidence**: `results/final/` with `summary.json` and the key logs (3611336 / 3611339, 3612073-5,
   3611285, 3612131, 3611284, and the new ones), plus the provenance table (number -> file:line).
10. **Tests and a smoke script**: pytest for the 3DPW K formula, the bone-fix round trip, label loading and a
    one-image forward; `scripts/smoke_test.sh` under 5 min. Deliver as a tagged release on `main`; the current branch
    has "WIP, not for main" commits.

Items 1, 2 and 9 in weeks 4-5 (he needs them for his independent check); items 3-8 and 10 in weeks 7-8.
