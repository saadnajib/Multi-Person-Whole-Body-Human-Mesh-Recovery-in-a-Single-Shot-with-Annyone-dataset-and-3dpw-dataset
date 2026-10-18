# Thesis rewrite plan

Written on Sunday 18 October 2026 (evening) from the repository as it stands at commit `56091f3`. Nothing was run
for this plan. Every number below carries its source; where sources disagree, the handover's latest dated statement
wins (18 Oct evening). The plan covers the draft in `docs/thesis/draft_2026-10_original/` (`main.tex`, 1,379 lines,
7 chapters, 21 compiled pages, no figures) and says what to keep, what to rewrite, what to delete, and how the new
thesis is organised, illustrated, tabulated, written and split among writers.

**Source key.** `H 3.1` = `docs/handover.md`, section 3.1 (`H 0a` = "Tomorrow and the meeting", `H 0b` = "Numbers at a
glance", `H 5/7` = section 5 item 7). `P 3.2` = `docs/thesis_requirements_plan.md`, section 3.2. `D part 4` /
`D slide 20` = `docs/deck_explained_2026-10-19.md`. `E exp8` = `docs/experiments_plan.md`, exp8 section. `F` =
`docs/diagnostics_findings.md`. `3611336:191` = line 191 of that Slurm job's `.out` log (on the cluster; quoted via the
handover, which was checked against the logs). `main.tex:665` = that line of the draft. `loss.py:314` = that line in the
repo today.

**Model names used in this plan** (the thesis will rename them once, see section 5):

| plan name | what | thesis name (proposed) |
|---|---|---|
| v5 | `anny_s2_shape_v5` epoch 99; ViT-L; synthetic only; start of every fix | Base |
| v9 e62 | v5 + 30% 3DPW on the OLD labels, epoch 62; previous best on 3DPW; teacher of the ViT-S students | Base+3DPW(old labels) |
| final model | exp7 fix arm, job 3611336: v5 + depth fix + bone fix + 30% 3DPW v23, 10 epochs | Final |
| control | exp7 control, job 3611339: same training, no depth fix, no bone fix | Control |
| masked arm | exp7 with `--mask_3dpw_shape 1`, job 3612325 | Final-masked |
| partial mask | job 3614042 (queued): mask the seven 0.5 constants, keep age | Final-masked-7 |
| ViT-S students v1-v8 | the draft's distillation line (2025-26), teacher v5 / v9 | Student-S (history) |
| ViT-B student | exp8, job 3614191 (running), teacher = final model | Student |
| NAVER-L | NAVER's `multiHMR_672_L_anny.pt`, evaluated with our code, job 3614192 | Multi-HMR-Anny (NAVER) |

**Evaluation settings** (must appear in every table caption): V1 = 100-image AnnyOne holdout (indices
579,860-579,959), helper bones masked, matched people, from the trace script; V0 = same images, helper bones unmasked,
from `train.py evaluate()` (+3.8 mm PVE on the same people, README :321); 3DPW test = all 8,190 images of the v23
labels, `run_eval_3dpw_test.sh`, V0 (H 0b).

---

## 1. Verdict on the draft

Legend: **keep** = reusable with light edits; **rewrite** = the topic stays, the text and numbers change; **delete** =
remove (the content is wrong or superseded); **move** = keep the text but relocate it.

### 1.1 Known contradictions, verified, plus the ones found while reading

| # | draft says | measured / correct | source |
|---|---|---|---|
| 1 | Neck elongation is a rig property: LBS stretch over stacked neck bones (`abstract.tex`, `main.tex:1037-1045`, `:1257-1261`) | The rig is fine (all 16 single-joint perturbations pass). Two bones, neck02 and head, are turned half a turn in every v5-line checkpoint (175.6 / 177.9 deg median; the half-turns cancel so the head points right, 12.0 deg); neck skin girth 0.46 of truth. A constant output correction (exp6) restores girth 1.00; the final model keeps it (neck02 4.8 deg, girth 0.996). Two hand bones (metacarpal1.L 180, finger4-1.R 178.8 deg) were flipped too. | H 3.2; 3573518:13-15, :69; 3579014:101-114; 3611336:198-207; F "Checks 2+3" |
| 2 | Headline 74.3 mm PVE (`main.tex:71, 683, 795`) | 74.3 is V0 (helper bones unmasked). V1: v5 70.5 / PA-PVE 53.3 / recall 90.2 (3570222:40). The thesis model is now the final model: V1 73.7 / 50.3 / 87.0; 3DPW test 89.0 / 57.6 (v23) | H 1 "Headline"; H 0b; 3611336:191-229 |
| 3 | Nothing on: the depth-anchor bug, corrected 3DPW labels v23, final model + control, Relative Human, masked-shape trade-off, NAVER baseline, model cost, ViT-B student, pilot noise | All measured 14-18 Oct | H 3.1, 3.5, 5/7, 3.4, 0a, 3.6 |
| 4 | Distillation = ViT-S student (v1-v6c, 114.5 mm; v7/v8 mixed), 8.6x fewer params (`main.tex:363-453, 696-810`) | Real, stays as history; predates both fixes and the v23 labels; the answer to 1b is the ViT-B student of the final model (101.38 M params, 603.3 GFLOPs, 51.4 ms vs 318.38 M / 1,927.4 / 110 ms; 3.14x / 3.19x / 2.14x) | H 0a (smoke test 3613926); E exp8 |
| 5 | 3DPW test: teacher 119.1 -> 102.5, student 139.5 -> 123.6 (`main.tex:1190-1211`, abstract) | Old (defective) labels; retired. v23: v5 121.8, v9 95.5, control 91.5, final 89.0 (PVE); PA-PVE 72.0 / 62.9 / 59.7 / 57.6. Students not yet evaluated on v23 | H 0b; 3612073-3612075:117-124; 3611336:224-229 |
| 6 | "No published multi-person HMR system predicts Anny" (`main.tex:40`, `:197`) | NAVER ships `multiHMR_672_L_anny`; the draft itself starts training from it (`main.tex:559-560`). It is the 1b baseline: AnnyOne holdout PVE 60.3 (likely in-sample for NAVER), 3DPW test 126.9 | H 0a (job 3614192); `slurm/eval/run_eval_baseline_naver.sh` header |
| 7 | AnnyOne = 579,960 "672x672 renderings" (`main.tex:590`) | 1280x1280 renders, resized to 672 for the network | H 1 "Data"; D part 2 |
| 8 | HPH "dimension 512, 2 blocks, 16 heads" (`main.tex:252`) | `model.py:222-229`: dim 1024, depth 2, heads 8, dim_head 32, mlp_dim 1024 | model.py |
| 9 | Converted 3DPW: 16,631 images / 20,028 people; train 4,451; "no fit failed, only 162 dropped" (`main.tex:914-935`) | Old set, 31% of train labels missing (image files absent) + gender reversed + missing actor. v23: 16,694 images / 22,749 people (train 5,882 / 7,584; val 2,622 / 3,311; test 8,190 / 11,854); 163 frames dropped (crosscountry_00, GT behind the camera) | H 3.5; H 0b |
| 10 | "Recurring courtyard actors receive identical bodies (male: height 0.60, proportions 0.41; female 0.34, 0.70)" (`main.tex:909-913`) | Derived from v2.2 fits in which every man was fitted as a woman (gender 1.0 = female in Anny). Must be re-derived from the v23 `.npz` or deleted | H 3.5 defect 1; `smpl_to_anny.py` docstring v2.3 |
| 11 | "The final converter fixes gender from 3DPW's annotation" (`main.tex:525`) presented as correct | True only since v2.3 (`--gender_convention anny`); v2.2 had it reversed; re-fit of all 87 people, men 19.0 -> 14.7 mm | H 3.5 "Re-fit v2.3 result" |
| 12 | Shape re-weighting "fixed" age/height (`main.tex:1000`; abstract) | Near the mode yes (bias -0.015 age on the holdout, 3570222:75); in both tails no: GT age 0.6-0.8 predicted 0.57; in-sample babies 0.015 -> 0.10; on real photos children read as short adults (AUC 0.61), no gender signal (46.5%) | H 3.4 |
| 13 | Student v4 adds "colour jitter" (`main.tex:726`); exp8 header also says "colour jitter as v6-v8" | `grep -rn -i jitter train.py datasets/*.py distill.py` finds no reader; the diagnostics README C4 calls the flags inert for AnnyOne. Do not claim augmentation until confirmed (section 7, Q8) | `scripts/diagnostics/README.md` C4 |
| 14 | 2D losses "enabled only after epoch e_2D" (`main.tex:348-350`) | Default `--start_2d_epoch 10`; the final model and exp8 use 0 | `loss.py:450`; `exp7_final_model.sh:765` |
| 15 | Two 3DPW protocols, "(a) against original SMPL meshes, comparable to published numbers" (`main.tex:605-609`) | Never run. Only Anny-space numbers exist; MPJPE averages all 163 Anny joints, so nothing is comparable with published 3DPW numbers | P 3.1 item 4 |
| 16 | Baseline table from the paper, ViT-L cell TODO (`main.tex:638-652`) | Replace with the measured NAVER-L row (job 3614192); paper numbers only as context and re-checked against the paper (the plan quotes NAVER README PVE 94.0 B vs 94.1 L on 3DPW, the draft 73.5 / 80.1 MPJPE: different metrics, verify both) | H 0a; P 3.2 |
| 17 | Fingers not discussed; demos "show the neck elongation on all" (`main.tex:946`) | Fingers 17.3 deg; rest pose 15.1; with the bone fix 8.2 (no training) / 8.3 (final); test-time zoom 17.2, loss rebalance 17.5 vs 17.4: both null. Every real-photo demo so far forces rest-pose fingers | H 3.3 |
| 18 | Per-joint x4 boost (`main.tex:1047-1054`) "had nothing to correct" | It is x2.41 / x0.60 after renormalisation (plan :932) and could not undo a half-turn (exp5 geodesic loss did not move it either: 175.6 -> 175.0) | H 1 "x4 boost"; H 3.2 |
| 19 | "Fully unfrozen v7 run" TODO (`main.tex:1074`, `:1270`) | Unknown whether `slurm/history/train_v7_scratch.sh` / `train_v8_partial_freeze.sh` finished; v10 (fresh heads on the NAVER backbone, 30% 3DPW from the start) is the measured relative: V1 74.1, root error 1.49 m, left-arm flip | H 1 table; H 3.1; section 7 Q7 |
| 20 | Abstract: "SMPL ... research-only license" and Anny licensed Apache 2.0 | Keep, but AnnyOne / NAVER checkpoints have their own licence terms (the plan calls them non-commercial for the release); check before writing the release paragraph | P 6.2 item 1 |

Right in the draft and confirmed: detection loss = CornerNet-style focal loss (`_neg_loss`, `loss.py:17, 314`; logged as
`bce`); the j3d/v3d terms are pelvis-centred (`loss.py:367-378`); 88 helper bones masked / 75 active; 6D rotation
output with Gram-Schmidt; the converter's virtual joints and geometry-checked correspondence (`main.tex:491-517`,
appendix table); the pipeline-bug list (`main.tex:1285-1331`); the 3DPW val/test actor overlap argument (`main.tex:1169-1175`,
measured on old labels, still plausible).

### 1.2 Front matter and abstract

| part | verdict | reason / what replaces it |
|---|---|---|
| `thesis.tex` preamble (scrbook, biblatex/biber alphabetic, TikZ, booktabs) | keep | add `pgfplots` (`\pgfplotsset{compat=1.18}`), `subcaption`, `siunitx`, `\graphicspath{{figures/}}`, `\input{numbers}`; move chapters to `\input{chapters/..}`; all packages are installed here (`kpsewhich` passes) |
| title "Multi-Person Human Mesh Recovery with Anny" | keep (student's decision) | subtitle option: "from synthetic AnnyOne to real 3DPW, with a smaller distilled model" |
| `abstract.tex` | rewrite | wrong headline (74.3), wrong neck cause, old distillation and 3DPW numbers; the new abstract follows the ten-sentence story of D part 1 |
| Eigenständigkeitserklärung, title page | keep | add the AI-use statement required by RPTU if any (P 6.1 "AI disclosure") |

### 1.3 Chapter 1 Introduction (`main.tex:12-107`)

| section | verdict | reason | source |
|---|---|---|---|
| 1.1 Problem (:16-47) | keep 70% | Anny motivation is good; delete "No published multi-person HMR system predicts Anny" (:40); replace the research question with the supervisor's requirements 1a/1b/1c/2a/2b/3 | contradiction 6; P 1 |
| 1.2 Motivation (:49-64) | keep | add that the backbone and the comparison point are NAVER's own Anny checkpoint | H 1 |
| 1.3 Contributions (:66-97) | rewrite | items 1-3 carry wrong numbers and the wrong neck cause; item 4 (converter) stays but adds the three label defects; item 5 (flags, pipeline bugs) becomes a half-sentence; new items: depth-anchor bug and fix, bone correction, final model with control, Relative Human evaluation, baseline comparison, ViT-B student | H 0b, 3.1-3.5 |
| 1.4 Outline (:99-107) | rewrite | new chapter list (section 2) | |

### 1.4 Chapter 2 Related work (`main.tex:110-213`)

| section | verdict | reason | source |
|---|---|---|---|
| 2.1 Body models (:114-148) | keep | add Anny's age anchors (`linspace(-1/3, 1, 5)`: 0 baby, 0.333 child, 0.667 young, 1 old) and gender convention (0 = male; rest stature 1.753 m at 0 vs 1.632 m at 1); drop the sentence that foreshadows LBS stretch (:144-148), keep "neck = neck01-03 + head" as a fact | H 3.4 "How Anny makes a child"; H 3.5 defect 1 |
| 2.2 HMR (:150-176) | keep + extend | add SPIN, BEV (age in the model and the Relative Human dataset), Multi-HMR's Anny checkpoint and the Anny paper's own training; verify the quoted 73.5 / 80.1 MPJPE against the paper; add one paragraph each on video methods (HMMR, VIBE, TCMR, MPS-Net, GLoT, SmoothNet, PHALP/4DHumans, WHAM) for requirement 2b | P 6.1 "Related work must cover" |
| 2.3 Distillation (:178-191) | keep | add DINOv2's own distillation and FitNets already cited | |
| 2.4 Research gap (:193-213) | rewrite | gap 1 is false (NAVER predicts Anny); gap 2's "25-50% reduction" becomes the supervisor's 95% rule; gap 3 (real-image evaluation in Anny space) stays | contradiction 6; P 1 |

### 1.5 Chapter 3 Method (`main.tex:216-580`), section by section

| section | verdict | reason | source |
|---|---|---|---|
| 3.1 Problem statement (:220-233) | keep | add: each person is anchored at one joint (the head) whose image position and depth the network predicts | H 1 "Terms" |
| 3.2 Architecture (:235-292) | keep 80% | fix HPH dims (1024 / 2 / 8 / 32); the figure TODO becomes TikZ F1; add the output-level bone correction `R_b <- R_b C_b` as the last step of the pose head; say plainly that the architecture is Multi-HMR's with the output resized (requirement 1b) | model.py:222-229; `bone_fix.py` docstring; P 2 row 1b |
| 3.3 Inputs and outputs table (:294-315) | keep | add a row "anchor joint: head (2D cell + offset + depth)"; note 1280 -> 672 resize | |
| 3.4 Losses (:317-361) | rewrite | equation stays; replace the prose with the plain-words table T2 (weights default vs final; what each term sees; which see depth); add the boost renormalisation (x4 = x2.41 / x0.60), `start_2d_epoch`, the 3DPW hand mask (38 bones), the tested shape mask; state that j3d/v3d are pelvis-centred and only j2d/v2d see depth (2.5% of the loss was dist+transl before the fix) | D part 4; `loss.py:436-463`; F "Check 4" |
| 3.5 Knowledge distillation (:363-453) | move + rewrite | the mechanics (feature KL at T=4, head transfer, output L1, warm-up freeze, projector-fed heads) stay as the method; the ViT-S numbers (C_s 384, 37 M, 8.6x) move to the history subsection of ch. 4; the main text describes the ViT-B student (768 -> 1024 projector, 0.79 M; heads 14.0 M; 101.38 M total). Throughput paragraph (TF32 / bf16 teacher) stays, shortened | E exp8; H 0a |
| 3.6 SMPL-to-Anny conversion (:455-554) | move to Data chapter, keep 85% | the objective, staged optimisation, 3DPW `jointPositions` targets, geometry-checked correspondence, virtual joints, one body per person stay; add v2.3 (gender convention, shared frame grid), the drop rules (fit > 80 mm, visible < 30%, box < 40 px), and that 144 of 163 bones stay at rest (hands, neck02/03, head); fix "4,451 train images" (:914); the mixed-training paragraph (:549-554) moves to Method 4.5 | H 3.5; `smpl_to_anny.py` docstring |
| 3.7 Training schedule (:556-580) | keep + extend | the v2 -> v5 staged schedule stays as history; add the final model's fine-tune: 10 epochs x 1,000 steps x batch 4 = 40,000 images, lr 1e-5, last 4 of 24 blocks, 70/30 mix, and the matched control; keep the NaN note but with the measured rate (1-7 per 2,000 steps; 16 of 10,000 in the final model) | H 5/7; H 1 |

### 1.6 Chapter 4 Experiments (`main.tex:583-951`), section by section

| section | verdict | reason | source |
|---|---|---|---|
| 4.1 Datasets (:587-609) | rewrite | AnnyOne size (1280 px), people per image (2.73), children share (2.53% < age 0.3; holdout none < 0.513); 3DPW counts v23; delete protocol (a); add Relative Human (635 images / 2,130 people, val), the diagnostic manifest (55 images) and the demo set (4 photos + 8 3DPW test frames) | H 1; H 3.4; H 3.5; D part 3 |
| 4.2 Metrics (:611-629) | keep + extend | add root error, size ratio, depth slope, per-bone angle, girth, children AUC, gender accuracy, precision; the V0/V1 distinction; MPJPE = 163 Anny joints, pelvis-centred, not comparable with published 3DPW | H 1 "Metrics"; D slide 9 |
| 4.3 Baselines (:631-652) | rewrite | measured NAVER-L row replaces the paper table | H 0a |
| 4.4 Implementation details (:654-663) | keep | add the final model's GPU/time (A100, 4,292 s, 10.9 img/s); ViT-S sizes to history | 3611336:15 |
| 4.5 Main results AnnyOne, T:main (:665-694) | rewrite | the 145 -> 73.6 -> 74.3 history stays as a V0 table (label it V0) with the TODO cells either filled (section 7, Q5) or the columns dropped; headline moves to the final model vs control vs v5 (V1) | H 5/7 |
| 4.6 Distillation results (:696-810) | move to "ViT-S students (history, before the fixes)" | tables `tab:student_runs`, `tab:kd_epochs`, `tab:kd_eval` stay (holdout numbers are V0: say so); "A ViT-B student was not trained" (:808) deleted; "v8 final student" renamed "best ViT-S student" | |
| 4.7 Conversion results (:812-939) | move to Data, keep tables `tab:fit`, `tab:fit_versions`, `tab:fit_scenes` as converter history | `tab:labels` (:919-935) deleted and replaced by T4 (v23 counts) and T5 (label check before/after); the paragraph :904-917 rewritten (counts, the identical-body claim re-derived or dropped) | H 3.5; contradiction 9-10 |
| 4.8 Qualitative (:941-951) | rewrite | TODOs become figures F6, F7, F9-F13 | section 3 |

### 1.7 Chapter 5 Ablation and root cause (`main.tex:954-1237`), section by section

| section | verdict | reason | source |
|---|---|---|---|
| 5.1 Shape loss weight, T:shape (:963-1004) | keep | measured (v4 vs v5 on 10 holdout images: say n = 10); soften "fixed" to "near the mode"; the tail regression and the real-photo collapse go to ch. 5/6 | H 3.4 |
| 5.2 Neck root cause, T:neck (:1006-1045) | rewrite | the four eliminations are valid tests and stay as "what is not the cause"; the conclusion (LBS, rig property) is deleted; replaced by the per-bone dump, the flip, the correction and the fine-tune (new ch. 5.3 + 4.4) | H 3.2 |
| 5.3 Per-joint boosting (:1047-1054) | keep as a negative result | add the renormalisation and why a loss weight cannot undo a half-turn (exp5) | H 1; H 3.2 |
| 5.4 LR schedule for distillation (:1056-1062) | keep (history) | | |
| 5.5 Backbone transfer depth, T:ft (:1064-1081) | keep if Q7 answers, else drop the TODO row | | |
| 5.6 Distilling the heads (:1083-1120) | keep (history) | the "real-side gap is capacity" claim stays tagged as measured on old labels | |
| 5.7 Student backbone size (:1122-1129) | rewrite | the ViT-B student ran (exp8) | E exp8 |
| 5.8 Synthetic vs mixed (:1131-1221) | keep the validation-trajectory text as history; delete `tab:test3dpw` (old labels) | "4,451 converted images" wrong; "the mixed student no longer beats the synthetic teacher" was measured on old labels; re-evaluate v6c and v8 on v23 (Q4) or state "old labels, retired" | H 0b |
| 5.9 Joint correspondence ablation (:1223-1237) | keep (moves to Data) | | |

### 1.8 Chapters 6-7 and appendix

| part | verdict | reason |
|---|---|---|
| 6 Discussion, "What improved" (:1244-1253) | rewrite | new numbers; new story (depth, bones, labels, real data) |
| 6 "What did not improve" (:1255-1283) | rewrite | neck bullet deleted; firmness bullet becomes the collapse onto converter constants; PVE plateau bullet becomes the restart cost (~2.4 mm) + v5 still lowest holdout PVE; student gap bullet moves to history |
| 6 Pipeline bugs (:1285-1331) | keep + extend | add: anchor/target mismatch; reversed gender convention; per-person stride phase; missing image files (8 sequences); NAVER adapter FFN width (2048 vs 4096); `evaluate()` unmasked vs demo masked |
| 6 Limitations (:1333-1343) | rewrite | children, depth slope 0.42, one run no seed, fingers never shown on real photos, recall -2/-3.2 points, precision 64% on 3DPW, 163-joint MPJPE, ~14 mm conversion floor, 100-image holdout |
| 7 Conclusion (:1350-1374) | rewrite | |
| Appendix A scripts table | keep + extend | add exp1-exp8, diagnostics, eval scripts, job ids |
| Appendix B joint correspondence | keep | |
| `references.bib` (15 entries) | extend | add SPIN, BEV, Relative Human, HMMR, VIBE, TCMR, MPS-Net, GLoT, SmoothNet, PHALP/4DHumans, WHAM, Anny-Fit, AGORA kids if cited |

### 1.9 Where the sources disagree (and which one this plan follows)

| topic | older statement | newer statement (followed) |
|---|---|---|
| Neck cause | `docs/diagnostics_findings.md` :184-197 and README :115: weak supervision on a redundant chain; long-neck look from the depth defect | H 3.2 (15-18 Oct): two half-turned bones; depth-fix run keeps girth 0.46, so depth is not the neck cause |
| v5 size ratio | findings :77 and exp1 script "baseline(findings)": 0.65 / 0.67 | H 3.1, H 6: the trace's 0.63 |
| Depth error | "1.08 -> 0.39 m" (signed `root dz`) | H 6: absolute root error 1.21 -> 0.54 m (pilot), 1402 -> 728 mm (final vs control) |
| Children evidence | "five small people predicted as adults" (findings :222, early handover) | H 3.4 (17-18 Oct): mostly crouching adults; Relative Human is the number |
| Student plan | P 4 week 1: launch Tue 20 Oct, 100 epochs, with both shape masks | H 0a / D slide 24: launched 18 Oct as job 3614191, 30 epochs, no shape mask (teacher unmasked, decided 18 Oct 15:30) |
| NAVER-L cost | P 3.2 / P 5: "~414 M params, a hand estimate" | H 0a: pending (`model_cost.sh`); the old model_cost run built the wrong FFN width (2048, not 4096) |
| HPH size | draft `main.tex:252`: 512 / 2 blocks / 16 heads | `model.py:222-229`: 1024 / 2 / 8 heads / dim_head 32 |
| Colour jitter | draft :726 and exp8 header: used | no reader in the code; diagnostics README C4: inert; open (section 7, Q8) |
| 3DPW label counts | draft and README :227: 4,451 train, 16,631 total | H 0b (18 Oct): v23 5,882 train, 16,694 total |
| 3DPW test numbers | draft, README, `run_demo_compare.sh` header: 119.1 / 102.5 / 111.5 / 139.5 / 123.6 | H 0b (18 Oct midday): v23 121.8 / 95.5 / 91.5 / 89.0; students not yet re-run |
| Which run is "final" | draft: ViT-S student v8; README headline v5 70.5 | H 0a DECIDED 18 Oct 15:30: the unmasked exp7 fix arm (3611336) is the final model and the teacher |

---

## 2. New chapter and section outline

Target 60-80 pages with figures; estimates per chapter in section 6. Figure ids F*, table ids T* refer to sections 3
and 4. "Absorbs" names draft lines; "needs" names new material.

### 1 Introduction (5 pages)

| § | purpose | absorbs | needs | figures / tables | key numbers (source) |
|---|---|---|---|---|---|
| 1.1 The task | one photo -> every person -> Anny pose, 11 shape values, distance | :16-47 minus :40 | the four-step strip in words | F1 (small version) or none | - |
| 1.2 Why Anny | licence, phenotypes, children and old people in one model | :28-40 | age anchors, gender convention | - | 163 bones, 11 phenotypes (H 1) |
| 1.3 What the thesis had to deliver | the supervisor's six requirements, in his words, as the frame of the thesis | - | P 1 | T1 (requirements map, short form; full form in 7.1) | - |
| 1.4 Contributions | numbered, each with the chapter that holds the evidence | :66-97 (converter item) | depth bug, bone fix, labels v23, final model + control, Relative Human, baseline, student | - | root error 1402 -> 728 mm; neck02 175.8 -> 4.8 deg; 3DPW 121.8 -> 89.0 (H 0b) |
| 1.5 Outline | one line per chapter | :99-107 | - | - | - |

### 2 Background and related work (8 pages)

| § | purpose | absorbs | needs | figures / tables |
|---|---|---|---|---|
| 2.1 Parametric body models: SMPL family and Anny | definitions; LBS; Anny rig (163 bones, 88 helpers), phenotypes, age anchors, gender | :114-148 | anny 0.6.1 facts (H 3.4) | F21 (Anny age sweep, if the student sends `vis/age_sweep.jpg`) |
| 2.2 Single-person and multi-person HMR | HMR, SPIN, HMR2.0; ROMP, BEV; Multi-HMR in detail (tokens, detection, HPH, depth) | :150-176 | BEV / Relative Human; NAVER's Anny checkpoint | - |
| 2.3 Synthetic training data | AGORA, BEDLAM, AnnyOne; the synthetic-to-real question | :173 | - | - |
| 2.4 Converting between body models | smpl2soma, joints2smpl, our approach in one paragraph | :458-461 | - | - |
| 2.5 Knowledge distillation | Hinton, FitNets, DINOv2; why heads matter | :178-191 | - | - |
| 2.6 Temporal models for video (for 2b) | HMMR, VIBE, TCMR, MPS-Net, GLoT, SmoothNet, PHALP/4DHumans, WHAM; what "understands motion" means | - | P 4 weeks 5-6 | - |
| 2.7 What this thesis adds | the gap, stated against NAVER's own Anny model | :193-213 rewritten | - | - |

### 3 Data (9 pages)

| § | purpose | absorbs | needs | figures / tables | key numbers (source) |
|---|---|---|---|---|---|
| 3.1 AnnyOne | what it contains; how we split it; children share; the holdout | :589-596 | H 1, H 3.4 | T3 (datasets) | 579,960 images, 2.73 people/img (3571601:21), 2.53% age < 0.3, holdout none < 0.513 (3569948:33) |
| 3.2 3DPW | the real set; SMPL ground truth; splits | :598-604 | - | - | 60 sequences, 87 person-sequences |
| 3.3 Converting SMPL to Anny | the fit (objective, stages, targets, correspondence, virtual joints, one body per person); what is fitted vs fixed | :455-547 | drop rules; 144 bones at rest | F2 (pipeline), T6 (fitted / from metadata / fixed) | age 0.55, seven values 0.5 (`smpl_to_anny.py`) |
| 3.4 Converter development | v1 34 mm -> v2 24.6 -> v2.1 15.8 (fencing scene); age fixed in v2.2 | :812-902, :1223-1237 | - | draft tables `tab:fit`, `tab:fit_versions`, `tab:fit_scenes` (keep) | (draft, measured) |
| 3.5 Three label defects and the v2.3 re-fit | reversed gender; missing actor (stride phase); missing labels (image files); how each was found and fixed; independent check before/after | - | H 3.5 | T5 (label check before/after), F14 (label-check panel) | men 19.0 -> 14.7 mm; 407/960 -> 0/1,080; body joints 18.7 -> 14.1 mm (3575369, 3611284) |
| 3.6 What the 3DPW labels can and cannot test | 8 constant phenotypes; hands and head at rest; ~14 mm floor; 163-joint MPJPE; bystanders not labelled | - | P 3.1 item 4 | T4 (v23 counts) | 16,694 / 22,749 (H 0b) |
| 3.7 Relative Human | what it labels; how we match (IoU 0.3); child threshold 0.45; evaluation only | - | E "Relative Human" | - | 635 images, 2,130 people (H 3.4) |
| 3.8 Diagnostic manifest and demo images | 40 train + 15 holdout images excluded from training; 4 photos + 8 3DPW test frames | - | H 1 | - | 55 images |

### 4 Method (12 pages)

| § | purpose | absorbs | needs | figures / tables | key numbers (source) |
|---|---|---|---|---|---|
| 4.1 Multi-HMR for Anny: architecture, inputs, outputs | backbone, detection, HPH, Anny head (6D, helper mask, sigmoid shape, depth), body-model layer | :220-315 | HPH dims; anchor joint | F1 (architecture), T (inputs/outputs, draft) | 304 M backbone, 318.38 M total (H 0a) |
| 4.2 Losses in plain words | each term, what it compares, weight default vs final; which terms see depth | :317-361 | D part 4 | T2 | shape 20 (default 1); dist/transl 5 (1); j2d/v2d 0.1 (1) |
| 4.3 Training schedule from the public checkpoint | v2 frozen -> v3 partial -> v4 boost -> v5 shape 20; the final fine-tune recipe | :556-580 | H 5/7 | T13 (runs) | 40,000 images, lr 1e-5, last 4 blocks |
| 4.4 The depth anchor and its fix | the model anchors at the head, the targets were built at the pelvis; why the 2D terms push bodies away; the fix flag and the x5 weight | - | H 3.1; F "Root cause" | F3 (anchor sketch) | pred depth = 0.33 x true + 2.36 m; offsets -2 to -19 cells vs +/-0.5 (3571599:9-14) |
| 4.5 The bone correction | per-bone constant `R_b <- R_b C_b`; how C_b is estimated (200 training images, never the holdout); saved in the checkpoint; why a loss cannot do it | - | H 3.2; `bone_fix.py` | F4 (chain sketch) | 178.8 / 179.2 / 179.8 / 179.7 deg (3579014:12, :37) |
| 4.6 Training with real images | the 70/30 mix; the hand mask (38 bones); the shape mask as a tested option; what the 3DPW mesh terms still carry | :549-554 | H 3.4 "3DPW labels and body shape" | - | ~16% of supervised people (H 3.4) |
| 4.7 Knowledge distillation | feature KL, head transfer, output L1, warm-up freeze, projector-fed heads | :363-453 | - | F5 (teacher-student) | T=4, lambda 1.0 / 1.0 |
| 4.7.1 History: the ViT-S students | v1-v6c, v7, v8 (what each step taught) | :389-441, :705-810 | - | draft tables `tab:student_runs`, `tab:kd_epochs` | 193.4 -> 114.5 mm (V0) |
| 4.7.2 The ViT-B student of the final model | recipe; the 95% rule; cost | - | E exp8 | T11 | 101.38 M / 603.3 GFLOPs / 51.4 ms; bars 77.6 / 93.7 mm |
| 4.8 Evaluation protocol | V0 vs V1; the trace; matching; 3DPW test script; Relative Human script; demo settings (`--relax_hands 1`, box anchoring) | :611-629 | H 1 "Evaluation settings" | T7 (metrics) | +3.8 mm V0 vs V1 |

### 5 Diagnostics: how each failure was isolated (7 pages)

| § | purpose | needs | figures / tables | key numbers (source) |
|---|---|---|---|---|
| 5.1 The three visible failures | thin twisted necks, claw fingers, children as adults; what was believed (rig, resolution, no age signal) | D slide 10 | F6 (neck before), F13 (boy) | - |
| 5.2 What is not the cause: six checks | GT decode; body-model probe (age sweep 0.53 -> 1.54 m; 16 perturbations); gradients (all 75 bones, 11 shape outputs; 2 of 3 camera outputs none); prediction vs truth on holdout + manifest; shape swap; capped overfit (-36% train, +45% holdout) | F; H 1 | T (six checks: check, sample, result) | loss shares: shape 25.7%, transl 1.9%, dist 0.6% (F Check 4) |
| 5.3 The depth bug | the trace's placement table; slope 0.33, intercept 2.36 m, 100% too far, size 0.63; why PVE is blind (corr -0.03); why demos hid it (box placement) | H 3.1; F | F15 (depth scatter), T8 | 1.21 m root error (3571596) |
| 5.4 The neck: per-bone dump | the dump; neck02 175.6 / head 177.9; compensated chain; girth 0.46; same in all 8 checkpoints; v10 different flips | H 3.2 | T9 (neck table) | 3573518:13-15 |
| 5.5 The fingers | 17.3 vs rest 15.1; hands 2.25 tokens median, 41% under 2; zoom and rebalance null; two flipped bones | H 3.3 | T10 | 3571960:287-314 |
| 5.6 Children and shape on real photos | median pull in both tails; Relative Human v5 (short adults, AUC 0.61, gender 46.5%); L1 through sigmoid; rarity (2.1% per batch); how Anny makes a child (shoulder/head 2.24 vs 3.00) | H 3.4 | T12 (RH v5 row), F16 | 3611285_0:23-31 |
| 5.7 Pilot noise and restart cost | two identical runs 7.6 mm / 1.9 deg / 0.8 pts; restart ~2.4 mm; what one run can resolve | H 3.6 | T14 (noise) | 3571958:104, 3571960:320 |
| 5.8 What did not work | exp3 global-head loss (11.3 vs 11.7 / 13.6), exp4 rebalance (17.5 vs 17.4), exp5 geodesic (175.6 -> 175.0), exp2 sampler (inconclusive at 2 epochs), x4 boost | H 3.2-3.4 | T (pilots) | D slide 22 |

### 6 Experiments and results (14 pages)

| § | purpose | figures / tables | key numbers (source) |
|---|---|---|---|
| 6.1 Setup recap | one paragraph: sets, settings, noise floor, one run per arm | - | - |
| 6.2 Depth fix: pilots and final model | exp1 table (v5, 2-ep x1/x5/x20, 10-ep x1/x5, controls); final vs control | T8, F7 (depth before/after 579862), F8 (579860, appendix), F15 | root 1.21 -> 0.54 m (pilot), 1402 -> 728 mm (final vs control); slope 0.42; size 0.78; pelvis px 26.7 -> 41.1 (H 3.1) |
| 6.3 Bone correction: zero-training, fine-tuned, final | exp6 table; final vs control; by-eye panels | T9, F6, F9 (3DPW neck pairs) | girth 0.46 -> 1.00; head direction 19.1 (no training) -> 14.4 -> 13.1 (3579014:113; 3611336:200) |
| 6.4 Fingers | numbers; no real-photo render yet (or F11 if produced) | T10, F11 | 17.3 -> 8.2 / 8.3 (3579014:114; 3611336:201) |
| 6.5 Final model vs control vs Base on AnnyOne | the full V1 table; what is inside noise (PVE -3.7) and what is not (PA-PVE -6.1, root -48%); recall cost -2 / -3.2 | T15, F12 (holdout montage GT / v5 / final) | 73.7 / 50.3 / 87.0 vs 77.4 / 56.4 / 89.0 vs 70.5 / 53.3 / 90.2 (H 5/7) |
| 6.6 3DPW test on the corrected labels | every model, same script; most of the gain is training on 3DPW (control 91.5); fixes add 2.5 mm; precision 64% explained | T15 (3DPW part), F10 (3DPW overlays) | 121.8 / 95.5 / 120.2 / 91.5 / 89.0 (3612073-5; 3611339; 3611336) |
| 6.7 Real photos: Relative Human and the demo | v5 / final / masked rows; gender 46.5 -> 64.2; AUC 0.61 -> 0.72; babies 0.44 vs adults 0.54; recall 47 -> 65; by-eye reading of the 12 images | T12, F13 (final_model_real_photos), F16 (RH age distribution), F17 (rest-pose lineup) | 3611285_0, 3612131, 3613616 |
| 6.8 The shape-mask trade-off | masked arm vs unmasked: variety back (std 0.04 -> 0.11) vs gender back to 49.3% and children weaker; the decision; the partial mask (pending) | T12 (masked column), T (demo spread) | H 0a "DECIDED" |
| 6.9 Baseline comparison and model cost | NAVER-L vs final on both sets; in-sample caveat; params / FLOPs / ms table | T16, F18 (cost vs PVE) | 60.3 / 126.9 vs 73.7 / 89.0; 318.38 M vs NAVER pending (H 0a) |
| 6.10 The ViT-B student | recipe recap; the final table teacher / student / ratio / bar; verdict as measured; the 95% rule against both baselines | T11, T17, F19 (student curves) | pending (job 3614191) |
| 6.11 History: ViT-S students and the v2-v5 schedule | the draft's tables, labelled V0 and old labels, with one paragraph on what they taught | draft tables `tab:main` (V0), `tab:kd_eval`, `tab:kd_epochs`; old 3DPW table only if re-run on v23 | (draft) |

### 7 Discussion (6 pages)

| § | purpose | tables |
|---|---|---|
| 7.1 The requirements, one by one | 1a-pose, 1a-shape, 1b, 1c, 2a, 2b, 3: status word, evidence (section, table, figure), what is missing | T1 (full requirements map) |
| 7.2 Why PVE barely moved while everything else did | PVE pelvis-aligned; neck and fingers are few vertices; restart cost; PA-PVE did move; 3DPW did move | - |
| 7.3 Limitations | children (AUC 0.72, men mid-range 0.55); depth slope 0.42 (mechanism: pulled toward ~2.5-2.9 m, training mean); one run, no seed; fingers never shown on real photos; recall cost; 3DPW precision 64% (bystanders + ghosts, split unknown); labels: 8 constants, hands and head at rest, ~14 mm floor; 163-joint MPJPE; 100-image holdout; NAVER's AnnyOne number likely in-sample | - |
| 7.4 Engineering lessons | the pipeline-bug list, extended; "silent failure is the dominant failure mode" | - |

### 8 Conclusion and future work (2 pages)

Conclusion in the student's words; future work: temporal refiner on frozen per-frame tokens (16-frame windows,
acceleration error against smoothing baselines; P 4 weeks 5-6), the partial shape mask, a longer student run, a seed
and bootstrap intervals, the two-sided bone correction (v10's left arm), the depth-slope investigation (measure the
training depth distribution), weak age labels from Relative Human (Path B), AGORA kids.

### Appendices (8 pages)

| appendix | content | source |
|---|---|---|
| A Reproduction: scripts, runs and job ids | T13: every number's run, job id, log, checkpoint path, command | H 8 job index; D slide 30 |
| B SMPL-Anny joint correspondence | draft `tab:mapping` + the virtual-joint offsets | draft appendix |
| C Flags and weights | every training flag of the final model and the student, default vs used | `exp7_final_model.sh:759-771`; exp8 header |
| D Extra figures | F8, F9 (office pair), F14, F20, F21, extra holdout panels | docs/figures, cluster |
| E Pilot results in full | exp1 full table, exp2-exp5 one table each | H 3.1-3.4 |

---

## 3. Figure list

"Source" = existing file / TikZ to draw / pgfplots from a CSV / cluster command. `*` marks the 12 to make first.
Captions are the one-sentence core; the writer adds the setting and the caveats listed.

| id | caption (one sentence) | source | chapter |
|---|---|---|---|
| F1 * | The network: 672-px image -> DINOv2 ViT-L (48 x 48 tokens) -> per-token detection score and 2D offset at the head -> per-person cross-attention head (HPH) -> 163 x 6 rotations, 11 phenotypes, depth -> bone correction -> Anny body at the predicted distance; dashed teacher box for distillation. | TikZ (nodes + arrows; dims from `model.py`) | 4.1 |
| F2 * | The six steps from a 3DPW frame to an Anny label: SMPL joints -> frame grid -> per-frame fit -> one body per person -> camera frame -> label file and drop rules. | TikZ flow | 3.3 |
| F3 * | The depth-anchor bug: the model places the body by its head, the targets were built at the pelvis; the 2D terms reconcile the two by pushing the body away and drawing it smaller. | TikZ sketch: two stick figures, anchor pixel, arrows; the line "pred = 0.33 x true + 2.36 m" | 4.4 / 5.3 |
| F4 | The neck chain neck01-neck02-neck03-head: two half-turns cancel so the head points right but the skin in between is twisted thin; the correction turns both back. | TikZ: four boxes with rotation arrows; girth 0.46 -> 1.00 | 4.5 / 5.4 |
| F5 | Distillation: frozen ViT-L teacher (bf16) and ViT-B student on the same image; projector 768 -> 1024 feeds the teacher's copied heads; feature KL (T = 4) on the projected tokens and L1 on rotations, shape and depth. | TikZ | 4.7 |
| F6 * | Holdout image 579862 (and 579863): the same checkpoint without and with the bone correction, no training; necks ringed. | existing `docs/figures/neck_before_after_v5.jpg` (1254 x 1118); caveats: correction only, depth bug still in, synthetic | 5.1 / 6.3 |
| F7 * | Holdout image 579862 rendered at the model's own depth: ground truth, v5, depth-fix pilot (x5, 10 epochs); the blue mesh goes from 0.76 to 0.95 of its true height. | existing `docs/figures/depth_before_after_579862.jpg`; caveat: a good case, not typical (D notes 13) | 6.2 |
| F8 | Holdout image 579860, the same three panels: before the fix the real head sticks out above a shrunken floating mesh. | existing `docs/figures/depth_before_after_579860.jpg` | app. D |
| F9 * | Real 3DPW test frame `downtown_bar_00` 701 (and `office_phoneCall_00` 440): v5 vs the model fine-tuned 2 epochs with the correction; thin stalks become normal necks. | existing `docs/figures/neck_before_after_3dpw_bar.jpg` and `_bar_office.jpg`; caption must say "fine-tuned 2 epochs with the fix" (H 9) | 6.3 |
| F10 * | Eight 3DPW test frames, v5 next to the final model, bodies placed from the detector box. | cluster: montages already exist from demo job 3612078: `demo_montage_3dpwtest/*.jpg` and `demo_montage_photos/*.jpg`; student sends all 12 files | 6.6 |
| F11 * | Predicted fingers on the 12 demo images (first time shown): final model with `--relax_hands 0`, next to v5. | cluster, new demo (command below) | 6.4 |
| F12 * | Six holdout images: photo, ground truth, v5, final model, each at its own depth. | cluster: crop and stack the trace panels (command below); final panels exist in `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/holdout_ft/vis/`, v5 panels in the v5 trace folder (job 3570222) | 6.5 |
| F13 * | Three real photos, v5 left, final model right: men get male bodies, women stay female, the boy stays an adult body. | existing `docs/figures/final_model_real_photos.jpg` (1008 x 1000, labels baked in) | 6.7 |
| F14 | A 3DPW label-check panel: photo, 3DPW SMPL ground truth with 2D keypoints, our Anny label. | cluster: `results/diagnostics/check_3dpw_conversion_v23/vis/*.jpg` (12 panels); send 3 | 3.5 |
| F15 * | Predicted root depth against true depth for the 222 holdout people, v5 (slope 0.33) and final model (0.42) with the identity line. | pgfplots from `per_person.csv` of both traces (command below; confirm column names first) | 5.3 / 6.2 |
| F16 | Predicted Anny age per Relative Human group (adult, teen, kid, baby) for v5 and the final model, with the child threshold 0.45. | pgfplots boxplot from `results/diagnostics/relative_human_val_*/per_person.csv` | 6.7 |
| F17 * | Rest-pose line-up: every detected person of the 12 demo images in rest pose, final model, so only body shape differs. | cluster, new demo with `--rest_pose 1` (command below) | 6.7 |
| F18 | Parameters and GFLOPs against 3DPW-test PVE for NAVER-L, the final model and the student (three points). | pgfplots, numbers from T16 (NAVER cost pending) | 6.9 |
| F19 | The student's holdout and 3DPW-validation PVE per evaluated epoch next to the teacher's values. | pgfplots from `train.log` of job 3614191 (pending) | 6.10 |
| F20 | In-training 3DPW-validation PVE per epoch, final model and control. | pgfplots; final: 79.9, 75.7, 77.8, 74.4, 75.5, 75.0, 73.4, 75.4, 75.3, 74.5 (3611336:55-164); control needs lines 48-157 of its log (section 7, Q10) | 6.5 |
| F21 | Anny at the dataset-mean shape with age swept 0 to 1: skeletal height 0.53 -> 1.54 m. | cluster: `results/diagnostics/bodymodel*/vis/age_sweep.jpg` exists (diag_bodymodel.sh); send it | 2.1 / 5.2 |
| F22 | The ViT-S students' holdout PVE over epochs (v3, v4, v5, v6). | pgfplots from the draft's `tab:kd_epochs` numbers | 6.11 |
| F23 | Exp6 compare panels, two more holdout images (top without, bottom with the correction). | cluster: `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/compare/*.jpg` (12 exist) | app. D |
| F24 | Ground-truth decode overlay (check 1): AnnyOne labels rendered onto the people with the dataset camera. | cluster: `results/diagnostics/gt_decode/vis/` (check 1 ran) | 5.2 |

Optional phenotype sweeps of Anny (gender, weight, muscle at rest pose) would need a small addition to
`scripts/diagnostics/diag_bodymodel_probe.py` (it sweeps `age` only); low priority.

### Cluster commands for the new figures

All from `/netscratch/najib/multi-hmr` on the login node after `git pull`; training env `multihmr`; old GPUs only
(the demo scripts' partition lists are already right). The student sends back the named files.

**F10 (already rendered).** Send every file of `demo_montage_photos/` and `demo_montage_3dpwtest/` (job 3612078; 4 + 8
files). If the folders are gone, rebuild them on the login node with the two `make_montage.py` commands printed at the
end of `logs/demo-compare-3612078.out` (lines 815-823).

**F11, predicted fingers.** `run_demo_compare.sh` hard-codes `--relax_hands 1` (`:171`). Make a copy that turns it off
and writes to new folders (never edit the production script; the output-folder pattern is chosen so that
`make_montage.py` still pairs the images):
```
cd /netscratch/najib/multi-hmr && sed 's/--relax_hands 1/--relax_hands 0/; s/demo_out_${name}_${tag}/demo_out_${name}_hands${tag}/' slurm/demo/run_demo_compare.sh > slurm/demo/run_demo_compare_hands.sh && grep -c -- '--relax_hands 0' slurm/demo/run_demo_compare_hands.sh
```
(must print `1`), then
```
cd /netscratch/najib/multi-hmr && DEMO_MODELS="anny_s2_shape_v5|00099 exp7_v5_final|00009" sbatch slurm/demo/run_demo_compare_hands.sh
```
When the log says `ok` for all four folders (it is `logs/demo-compare-<jobid>.out`), on the login node:
```
cd /netscratch/najib/multi-hmr && python scripts/analysis/make_montage.py --out demo_montage_hands_photos --folders demo_out_anny_s2_shape_v5_00099_handsphotos demo_out_exp7_v5_final_00009_handsphotos --labels 'Base, predicted fingers' 'Final, predicted fingers' && python scripts/analysis/make_montage.py --out demo_montage_hands_3dpwtest --folders demo_out_anny_s2_shape_v5_00099_hands3dpwtest demo_out_exp7_v5_final_00009_hands3dpwtest --labels 'Base, predicted fingers' 'Final, predicted fingers'
```
Send both montage folders. Note for the caption: v5's metacarpal1.L is flipped in these renders (the fix is only in the
final model), and the index-finger base will look bent backwards on the left hand.

**F17, rest-pose line-up.** Same route with `--rest_pose 1` (the demo already has the flag, `demo.py:815`):
```
cd /netscratch/najib/multi-hmr && sed 's/--rest_pose 0/--rest_pose 1/; s/demo_out_${name}_${tag}/demo_out_${name}_rest${tag}/' slurm/demo/run_demo_compare.sh > slurm/demo/run_demo_compare_rest.sh && grep -c -- '--rest_pose 1' slurm/demo/run_demo_compare_rest.sh
```
```
cd /netscratch/najib/multi-hmr && DEMO_MODELS="anny_s2_shape_v5|00099 exp7_v5_final|00009" sbatch slurm/demo/run_demo_compare_rest.sh
```
then the two `make_montage.py` calls as above with `rest` in place of `hands`. If the masked arm should be in the
line-up too (it is the arm that restores body variety), copy its checkpoint first and add it to `DEMO_MODELS`:
```
cd /netscratch/najib/multi-hmr && cp -n results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt models/multiHMR/exp7_shpmask_00009.pt && DEMO_MODELS="anny_s2_shape_v5|00099 exp7_v5_final|00009 exp7_shpmask|00009" sbatch slurm/demo/run_demo_compare_rest.sh
```

**F12, holdout montage GT / v5 / final.** The trace panels are `[image | GT | V1 | V2]` of equal width
(`diag_holdout_trace.py:32, :904`). The command below finds the v5 trace folder itself (the job 3570222 run, named
`results/diagnostics/holdout_anny_s2_shape_v5_00099` or similar; check with `ls results/diagnostics/ | grep holdout`
first) and stacks the first three quarters of the v5 panel (image, GT, v5) with the third quarter of the final model's
panel (its V1 prediction):
```
cd /netscratch/najib/multi-hmr && conda activate multihmr && V5DIR=$(ls -d results/diagnostics/holdout_anny_s2_shape_v5_00099* | head -1) && mkdir -p thesis_fig_holdout && python - "$V5DIR" <<'PY'
import sys, os
from PIL import Image
v5dir = sys.argv[1]
fin = 'results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/holdout_ft/vis'
for idx in [579860, 579862, 579863, 579865, 579868, 579871]:
    a = Image.open(os.path.join(v5dir, 'vis', f'{idx}.jpg')); b = Image.open(os.path.join(fin, f'{idx}.jpg'))
    w = a.width // 4; h = a.height
    out = Image.new('RGB', (4 * w, h))
    out.paste(a.crop((0, 0, 3 * w, h)), (0, 0)); out.paste(b.crop((2 * w, 0, 3 * w, h)).resize((w, h)), (3 * w, 0))
    out.save(f'thesis_fig_holdout/{idx}_img_gt_v5_final.jpg', quality=92); print('wrote', idx)
PY
```
If the `ls` finds no folder, re-trace v5 first (10 min): `CKPT=/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt TAG=v5_00099_thesis sbatch slurm/diagnostics/diag_holdout.sh`, then use `results/diagnostics/holdout_v5_00099_thesis`. Send the six files.

**F15 and F16, CSVs.** Send (no GPU): `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/holdout_ft/per_person.csv`,
the same file of the control (`v5_root_dist1_nofix_v23_e10/holdout_ft/`), the v5 trace's `per_person.csv`, and
`results/diagnostics/relative_human_val_anny_s2_shape_v5_00099/per_person.csv`,
`results/diagnostics/relative_human_val_exp7_v5_final_00009/per_person.csv`, plus the masked arm's. First send
`head -1` of each so the pgfplots column names are known.

**F14, F21, F23, F24.** Send `results/diagnostics/check_3dpw_conversion_v23/vis/` (3 panels of the student's choice,
one two-person frame among them), `results/diagnostics/bodymodel*/vis/age_sweep.jpg`,
`results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/compare/579864.jpg` and `579866.jpg`, and two files
from `results/diagnostics/gt_decode/vis/`.

---

## 4. Table list

Numbers as measured; every cell traceable. "Source" gives job:line or handover section. The LaTeX writers put each table
in its own file under `docs/thesis/tables/` and every headline number in `numbers.tex` (section 6).

**T1 Requirements map** (Discussion 7.1; short form in 1.3). Columns: requirement | supervisor's wording (short) |
status | evidence (section / table / figure) | what is missing. Rows and status words from P 2 and D slide 23: 1a-pose
partly (3DPW test PVE 89.0, recall 99, precision 64, 3611336:224-229; necks and fingers fixed on the holdout; by eye on
photos); 1a-shape partly, at risk (gender 64.2%, job 3612131; muscle/weight/cup size/firmness at ~0.5, demo 3612078;
masked arm trade-off, 3613616); 1b not met (architecture is Multi-HMR; student pending, 3614191; NAVER-L 60.3 / 126.9,
3614192); 1c met with caveats (v23 test 8,190 images; check WARN, 3611284); 2a met (train 5,882 / 7,584; 30% mix;
control 91.5 vs v5 121.8); 2b not started; 3 partly.

**T2 Loss terms and weights** (Method 4.2). Columns: term (code name, `loss.py` line) | what it compares | plain meaning
| default weight | final model | control | student. Rows from D part 4: detection (`bce`, focal, :314) 10 / 10 / 10 /
10; offset 1; rotations (`rotmat`, :319-335) 0.1 with neck/hand boost x4 (= x2.41 / x0.60); shape (:336-343) 1 / 20 /
20 / 20; distance (`dist`, :344) 1 / 5 / 1 / 5; translation (`transl`, :345) 1 / 5 / 1 / 5; 3D joints 100; 3D vertices
100; 2D joints and vertices 1 / 0.1 / 0.1 / 0.1 (from epoch 0; default epoch 10); global rotation 0 (tested in exp3);
feature KD - / - / - / 1.0 (T 4); output KD - / - / - / 1.0. Footnote: 38 hand bones masked for 3DPW people in the final
model, control and student (`loss.py:231-253`); shape mask off in all three (tested in the masked arm). Sources:
`loss.py:436-463`, `exp7_final_model.sh:211-213, 762-766`, exp8 header.

**T3 Datasets** (Data 3.1). Columns: dataset | real / synthetic | labels | used for | size. Rows (D part 3): AnnyOne,
synthetic, 1280 x 1280, 2.73 people/image, full Anny labels, training 70% / holdout, 579,960 = 579,860 + 100 (H 1);
3DPW-Anny v23, real, converted, ~1.2 people/image, every third frame, training 30% / val 300 images per epoch / test,
train 5,882 / 7,584, val 2,622 / 3,311, test 8,190 / 11,854 (H 0b); Relative Human val, real, age group + gender +
boxes, evaluation only, 635 / 2,130 (H 3.4; whole set 8,741 images); diagnostic manifest, 40 train + 15 holdout,
excluded from training (H 1); demo images, 4 photos + 8 3DPW test frames (H 3.2).

**T4 3DPW-Anny label sets** (Data 3.6). Columns: label set | converter | train images / people | validation | test |
total | defects. Rows: old `labels` v2.2: 4,451 / 5,183; 2,458 / 2,995; 9,722 / 11,850; 16,631 / 20,028; gender
reversed, missing actor, missing labels (3575369:31; H 3.5). `labels_rebuilt` v2.2 after re-extraction (job 3581320):
6,437 / 7,589; 2,777 / 3,314; 9,722 / 11,850; 18,936 / 22,753; gender reversed, missing actor. v23 (v2.3, jobs 3590385 +
3602813): 5,882 / 7,584; 2,622 / 3,311; 8,190 / 11,854; 16,694 / 22,749; none of the three (H 0b). Footnote: fewer images,
same people, because both actors now share each frame; `outdoors_crosscountry_00` yields no labels (GT behind the
camera) in every set.

**T5 Label quality before and after the re-fit** (Data 3.5). Columns: check | old labels (job 3575369) | v23 (job
3611284). Rows (H 0b table): overall grade FAIL / WARN; body joints median / p95 18.7 / 24.5 -> 14.1 / 21.1 mm; surface
Anny->SMPL / SMPL->Anny 18.9 / 18.3 -> 18.9 / 16.1; 2D over raw 3DPW +1.1 -> +0.4 px; gender labels reversed (flip
better for 72 of 73 men) -> m 0.000 / f 1.000 (flip worse, men 13.9 -> 20.0 mm); two-person images missing an actor 407
of 960 -> 0 of 1,080; fitted frames and median error 22,915 / 18.2 mm -> 22,912 / 14.0 mm; head turn vs chest (labels /
SMPL) 9.5 / 20.2 -> 19.6 / 19.3 deg. Plus the re-fit post-check: men 19.0 -> 14.7 mm mean (62 people), women 15.0 ->
15.2 (25); frames with a missing actor 4,457 of 8,274 (54%) -> 0 (`logs/check_refit_all_v23.log`). Still WARN: limb
ratios (forearm 0.948, thigh 1.064), 3.2% of people with a joint > 100 mm.

**T6 What the converter fits, reads and fixes** (Data 3.3). Three columns: fitted (root position and rotation, 18 of
23 mapped bones, height, proportions, one body per person); from 3DPW metadata (gender, men 0 / women 1 since v2.3);
fixed (age 0.55; muscle, weight, cup size, firmness, african, asian, caucasian 0.5; 144 of 163 bones at rest incl.
neck02, neck03, head, jaw, all finger bones). Drop rules: fit > 80 mm, visible < 30%, box < 40 px. Source: H 3.5;
`smpl_to_anny.py`.

**T7 Metrics** (Method 4.8). Columns: metric | what it measures | perfect value | blind to. Rows (D slide 9): PVE,
PA-PVE, MPJPE (163 joints), recall, precision, root error (mm), size ratio, depth slope / intercept, bone angle error
(~180 = flipped), neck girth ratio, PVE_neck, global head direction, children AUC, gender accuracy. Footnote: V0 vs V1
(+3.8 mm, README :321); 3DPW numbers only against our labels.

**T8 Depth fix** (Results 6.2). Columns: run | epochs | depth weight | root error (m) | size | slope | intercept (m) |
PVE | recall | pelvis px. Rows exactly as H 3.1: v5 1.21 / 0.63 / 0.33 / 2.36 / 70.5 / 90.2 / 23.4 (3571596:55,64;
3570222:40); 2-ep control 1.11 / 0.66 / 0.33 / 2.28 / 72.32 / 89.84 / 23.3 (3571599); 2-ep fix x1 0.69 / 0.78 / 0.30 /
1.93 / 77.25 / 87.80 / 37.3 (3571598); 2-ep x5 0.66 / 0.80 / 0.44 / 1.62 / 76.77 / 87.80 / 38.1 (3572418); 2-ep x20
0.53 / 0.87 / 0.43 / 1.44 / 75.66 / 87.40 / 31.4 (3572483); 10-ep control 1.44 / 0.59 / 0.34 / 2.59 / 72.19 / 89.43 /
24.3 (3571934); 10-ep fix x1 0.73 / 0.78 / 0.30 / 1.98 / 75.51 / 86.99 / 38.5 (3571933); 10-ep x5 0.54 / 0.84 / 0.38 /
1.59 / 74.77 / 87.80 / 33.1 (3572482); exp7 control 1.40 / 0.59 / 0.28 / 2.66 / 77.4 / 89.0 / 26.7 (3611339:166-184);
final model 0.73 / 0.78 / 0.42 / 1.75 / 73.7 / 87.0 / 41.1 (3611336:175-193). Also v9 e62 1.39, v9 e149 1.44, v10 1.49
m (3571963/4, 3571794). Footnote: head pixel error 64.5 (v5) -> 7.6 px (final; control 67.1).

**T9 Neck** (Results 6.3). Columns: V1 holdout | v5 | correction, no training | correction + 2-epoch fine-tune |
2-epoch control | exp7 control | final model. Rows (H 3.2 tables): neck02 median deg 175.6 / 4.5 / 4.7 / 175.7 / 175.8 /
4.8; head bone 177.9 / 12.4 / 11.2 / 177.9 / 177.7 / 8.9; metacarpal1.L / finger4-1.R 180.0 / 0.0 / 0.0 / - / 180.0 /
0.0 and - / 10.2 / 10.3 / - / 180.0 / 9.6; girth 0.461 / 1.004 / 0.997 / 0.461 / 0.464 / 0.996; PVE_neck 27.3 / 10.8 /
10.7 / 27.9 / 27.7 / 11.0; PVE 70.5 / 71.3 / 68.8 / 76.9 / 77.4 / 73.7; PA-PVE 53.3 / 52.2 / 50.9 / 55.2 / 56.4 / 50.3;
recall 90.2 / 90.2 / 89.4 / 91.1 / 89.0 / 87.0; global head 12.0 / 19.1 / 14.4 / 13.8 / 13.0 / 13.1; fingers 17.3 / 8.2
/ 8.3 / 17.4 / 17.4 / 8.3. Sources 3579014:101-114; 3611339:182-198; 3611336:191-207. Footnote: constants 178.8 /
179.2 / 179.8 / 179.7 deg from 200 training images (3579014:12, :37); v10 row optional (girth 1.00, PVE_neck 10.3,
left-arm flip).

**T10 Fingers** (Results 6.4). Rows: predicted fingers v5 17.3 deg; rest pose 15.1 (3571960:290); test-time crop x1.3
17.2, body PVE 124.8 (3571960:294-299); region-normalised loss 17.5 vs control 17.4 (3571959:316; 3571960:314); bone
correction, no training 8.2 (3579014:114); + 2-epoch fine-tune 8.3; final model 8.3 vs control 17.4 (3611336:201;
3611339:192); v10 three metacarpals corrected 32.0 -> 17.8 (3579015:80). Hand size: median 2.25 tokens at 672 px, 41%
under 2 tokens (3571960:287); with a person crop 4.2 tokens / 0%; children's hands 0.7 / 80% (D slide 17).

**T11 Student recipe and the 95% bars** (Method 4.7.2 / Results 6.10). Part A recipe (exp8 header): teacher = final
model (frozen, bf16); student DINOv2 ViT-B/14 from NAVER's `multiHMR_672_B.pt` backbone, fully trained; projector 768
-> 1024 (0.79 M); heads copied from the teacher (14.0 M), frozen 5 epochs; teacher's bone fix during training, own fix
re-estimated after; targets = final model's (person_center, x5, 30% 3DPW v23, hand mask, manifest excluded); KD
feature 1.0 (T 4) + output 1.0; batch 8, lr 3e-5, 30 epochs x 1,000 = 240,000 images. Part B bars: teacher V1 PVE
73.7 -> bar 77.6 (/0.95) or 77.4 (x1.05); 3DPW test PVE 89.0 -> 93.7 / 93.4; MPJPE 88.3 -> 92.9 / 92.7; against NAVER-L:
AnnyOne 60.3 -> 63.5, 3DPW 126.9 -> 133.6 (H 0a; P 3.2). Plus fewer params AND FLOPs AND faster.

**T12 Relative Human** (Results 6.7 / 6.8). Columns: v5 (3611285_0) | final model (3612131) | masked arm (3613616).
Rows: recall 47.1 / 64.7 / 57.2%; gender accuracy 46.5 / 64.2 / 49.3% (men mean 0.535 / 0.551 / 0.717, women 0.551 /
0.861 / 0.860); child-vs-adult AUC 0.606 / 0.719 / 0.708; mean predicted age adult 0.631 / 0.536 / 0.638, teen 0.597 /
0.524 / -, kid 0.613 / 0.500 / 0.611, baby 0.624 / 0.438 / 0.524; babies below 0.45: 2.6 / 49.3 / 16.9%; height adult
/ kid / baby 0.507 / 0.451 / 0.394 vs 0.500 / 0.466 / 0.464; matched children + adults 948 vs 1,305 (different detected
sets). Plus the demo-spread rows (jobs 3612078 vs 3613617, 12 images): std of muscle 0.04 -> 0.06, weight 0.04 ->
0.11, cup size 0.05 -> 0.11, firmness 0.04 -> 0.11, age 0.03 -> 0.08 (H 0a). Footnote: detected people only; IoU 0.3;
child threshold 0.45; exp6 fine-tune = v5 within noise (AUC 0.613).

**T13 Runs, jobs and checkpoints** (Appendix A). Columns: thesis name | run name | job | what | log | checkpoint /
output. Rows from D slide 30 and H 8: v5 00099 (`logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt`; trace
3570222); v9 e62 (`models/multiHMR/anny_teacher_v9_mixed_00062.pt`); final model 3611336
(`results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt`, copy
`models/multiHMR/exp7_v5_final_00009.pt`); control 3611339; masked arm 3612325; partial mask 3614042; student 3614191
(smoke 3613926); NAVER-L 3614192 (`models/multiHMR/multiHMR_672_L_anny.pt`); RH 3611285_0/_1, 3612131, 3613616; demos
3581312, 3612078, 3613617; 3DPW tests 3612073/4/5; label check 3575369 (old), 3611284 (v23); re-fit 3589270 (test),
3590385 + 3602813; rebuilds 3579266, 3581320; exp1 3571598/3571599, 3572418, 3572483, 3571933/3571934, 3572482; exp2
3571955/3571956 (3571601 measurement); exp3 3571957, controls 3571958, 3571960; exp4 3571959; exp5 3573520/3573521;
per-bone dumps 3573518/3573519; exp6 3579014 (v5), 3579015 (v10); bone fix JSON
`results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json`; labels `threedpw_anny_v23/labels/`.

**T14 Pilot noise and restart cost** (Diagnostics 5.7). Rows: two identical 2-epoch controls (3571958 vs 3571960): PVE
76.3 vs 68.7 (7.6 mm), global head 13.6 vs 11.7 (1.9 deg), recall 0.8 points; six 2-epoch v5-recipe arms: PVE
68.7-76.3, recall 88.6-90.2, head 11.3-13.7; restart cost ~2.4 mm (controls mean 72.9 vs v5 70.5); what one run
resolves: ~10 mm, ~3 deg, ~2 recall points; root error: no measured floor, but every fix run (0.53-0.73 m) is below
every control (1.11-1.44 m). Source H 3.6; D slide 22.

**T15 Final model vs control vs Base** (Results 6.5-6.6). Part A, V1 holdout: columns v5 | long_pc_dist5 (depth
pilot) | exp6 fine-tune (neck pilot) | exp7 control | final model; rows PVE 70.5 / 74.8 / 68.8 / 77.4 / 73.7; PA-PVE
53.3 / 54.0 / 50.9 / 56.4 / 50.3; recall 90.2 / 87.8 / 89.4 / 89.0 / 87.0; root error mm 1208 / 544 / 1429 / 1402 /
728; slope 0.33 / 0.38 / 0.34 / 0.28 / 0.42; intercept 2.36 / 1.59 / 2.56 / 2.66 / 1.75; size 0.63 / 0.84 / 0.58 / 0.59
/ 0.78; girth 0.461 / 0.457 / 0.997 / 0.464 / 0.996; PVE_neck 27.3 / 27.7 / 10.7 / 27.7 / 11.0; global head 12.0 /
12.7 / 14.4 / 13.0 / 13.1; fingers 17.3 / 17.4 / 8.3 / 17.4 / 8.3; neck02 175.6 / 175.5 / 4.7 / 175.8 / 4.8; head 177.9 /
178.0 / 11.2 / 177.7 / 8.9 (H 5/7 table; 3611336:191-207; 3611339:182-198). Part B, 3DPW test v23, 8,190 images:
columns v5 | v9 e62 | exp6 | control | final | NAVER-L; PVE 121.8 / 95.5 / 120.2 / 91.5 / 89.0 / 126.9; PA-PVE 72.0 /
62.9 / 68.7 / 59.7 / 57.6 / 68.8; MPJPE 126.9 / 94.8 / 124.8 / 90.9 / 88.3 / 129.4; precision 54 / 73 / 53 / 62 / 64 /
75; recall 100 / 94 / 100 / 99 / 99 / 97; AnnyOne test-100 V0 PVE 74.3 / 78.5 / 72.7 / 81.2 / 77.5 / (60.3, masked)
(3612073-5:117-124; 3611339:215-220; 3611336:224-229; 3614192). Caption: the old-label numbers (119.1 / 102.5 / 111.5)
are retired and never appear next to these.

**T16 Baseline comparison including cost** (Results 6.9). Columns: model | params | GFLOPs (1 MAC = 2 FLOPs, 672 px,
one person) | ms/image (A40, batch 1) | AnnyOne holdout V1 PVE / PA-PVE / recall | 3DPW test PVE / PA-PVE / recall |
trained on 3DPW? Rows: NAVER-L pending / pending / - / 60.3 / 44.6 / 89 (likely in-sample) / 126.9 / 68.8 / 97 / no
(3614192); final model 318.38 M / 1,927.4 / 110 / 73.7 / 50.3 / 87 / 89.0 / 57.6 / 99 / yes (3613926 for cost); control
318.38 M / 1,927.4 / 110 / 77.4 / 56.4 / 89 / 91.5 / 59.7 / 99 / yes; v5 318.38 M / ... / 70.5 / 53.3 / 90.2 / 121.8 /
72.0 / 100 / no; student 101.38 M / 603.3 / 51.4 / pending / pending / yes (ratios 3.14x / 3.19x / 2.14x). Backbone-only
analytic row (`model_cost.py`): ViT-S 22.06 M / 196.9, ViT-B 86.58 / 589.5, ViT-L 304.37 / 1,917.2 GFLOPs. Footnote:
NAVER's ms cannot be measured by `model_cost.py --bench` (original class); quote its README 74 ms on a V100 for SMPL-X
only with that caveat (P 3.4 item 4).

**T17 Student result** (Results 6.10; pending job 3614191). Columns: metric | teacher (re-measured in the job) |
student | ratio | bar (/0.95) | met? Rows: AnnyOne V1 PVE, PA-PVE, MPJPE, recall; AnnyOne test-100 and 3DPW test PVE,
PA-PVE, MPJPE, recall; params; GFLOPs; ms; training images/s. Filled from `<RUN>/final_table.json`.

**T18 Shape head: v4 vs v5 phenotype error** (Diagnostics 5.6 / history). The draft's `tab:shape` as is (n = 10 holdout
images; errors in phenotype units), plus the tail rows: holdout GT age 0.6-0.8 predicted 0.57 (3570222:89); in-sample
babies 0.015 -> 0.10 (3569950:61); training medians gender 0.588 / age 0.601 / height 0.500 (3569948:8-18).

Also kept from the draft: `tab:fit`, `tab:fit_versions`, `tab:fit_scenes` (converter history), `tab:student_runs`,
`tab:kd_epochs`, `tab:kd_eval` (ViT-S history; mark V0), `tab:mapping` (appendix), `tab:main` as a V0 history table.

---

## 5. Writing rules for the chapter writers

**Voice and tense.** First person singular for decisions and actions ("I built the targets at the pelvis; the model
anchors at the head"); neutral present for what a method or a table is ("Table 6.3 lists"); past for what was measured
("the control reached 91.5 mm"). The supervisor grades whether the reader sees the student's own reasoning (P 1,
3-report): every design choice gets its reason and its control or citation in the same paragraph.

**Sentences.** Short. One idea per sentence. Every term explained once, where it first appears, in plain words (as in D
part 2), then used consistently. No sentence that could have been written without reading the logs.

**Model and run names.** Define once in 4.3 and use everywhere: Base (v5), Control, Final, Final-masked, Student-S
(history), Student, Multi-HMR-Anny (NAVER). Never `v5`, `exp7`, `3611336`, `long_pc_dist5` in the running text.

**Traceability without job ids in the text.** Every number in the text and tables comes from `numbers.tex`
(`\newcommand{\FinalPVE}{73.7} % 3611336:191 V1`), each macro with the source in a LaTeX comment. Each table caption
names the set, the setting (V1 / V0 / 3DPW test v23) and the run names; Appendix A (T13) maps run names to job ids,
logs and checkpoint paths, so the examiner can verify any cell in two steps. Job ids appear in Appendix A only.

**Confidence in a thesis.** No [Certain] / [Likely] / [Guessing] tags. Measured facts: "On the 100 holdout images the
root error fell from 1402 to 728 mm against the control (Table 6.1)." Inferences: "We expect that ..." / "The likely
reason is ..., which was not tested." By-eye observations: "Judged by eye on the 12 demo images, ...; no metric exists
there." Untested mechanisms are labelled as such in the same sentence (e.g. the recall drop as a cohort effect of
moving the detection cell to the head).

**One run, no seed.** State it once in 4.8 and once in 7.3, and apply it in every comparison: give the control next
to the arm; quote the noise floor (7.6 mm PVE, 1.9 deg, about 2 recall points between two identical runs); call gaps
below it "within run-to-run variation" (PVE 77.4 -> 73.7); call gaps far above it what they are (root error -48%,
neck02 175.8 -> 4.8 deg, girth 0.46 -> 1.00, PA-PVE -6.1 mm). Never read a pilot against Base alone (restart cost
~2.4 mm). For exp6: "no sign of a PVE cost", never "PVE improved". If bootstrap intervals exist by week 3, add them to
T15; otherwise say they do not.

**The NAVER AnnyOne number.** Fixed wording: "NAVER's model was trained on the AnnyOne release, which includes the
100 images we hold out, so its 60.3 mm is most likely an in-sample number; ours are held out. On 3DPW, which neither
model saw before our fine-tune, the final model is 37.9 mm better (89.0 vs 126.9)." Never present 60.3 vs 73.7 as a
fair gap without this sentence.

**3DPW wording.** "Lowest error on our 3DPW-Anny labels", never "best model" or "state of the art"; "not comparable
with published 3DPW numbers (163 Anny joints, converted labels, ~14 mm floor)"; never put old-label numbers (119.1,
102.5, 111.5, 139.5, 123.6) next to v23 numbers; if the ViT-S students are not re-evaluated on v23, their 3DPW row is
labelled "old labels, retired" or dropped.

**Other fixed rules.** Quote the absolute root error, never the signed mean depth offset (1.08 -> 0.39 m). v5's size
ratio is 0.63, not 0.67. Never convert Anny ages to years (three scales disagree; H 3.4 "Age scales"); say
"child-shaped (age below about 0.45)". Children: "a measured improvement, not solved". Fingers: "predicted fingers
have not been shown on a real photo" until F11 exists. Depth: "the fix moved the offset and the body size, not the
slope (0.42)". The bone fix "adds no parameters" and "its origin is unknown". The hand mask "removes the rotation terms
only; the mesh terms still carry the rest-pose hands".

**Banned.** Confidence tags; job ids in the text; "delve", "leverage", "robust", "novel", "seamless", "cutting-edge",
"state-of-the-art" (unless citing), "significantly" without a test, "it is worth noting", "crucially", "importantly",
"notably", "comprehensive", "holistic", "paving the way", "unlock", "showcase", "plethora"; sentence-initial
"Furthermore"/"Moreover" chains; rhetorical questions; bullet lists as prose; three-item flourishes; em-dash
asides; "measured rather than guessed" style self-praise; "solved", "fixed completely", "perfect".

**Evidence first.** Each results section opens with the figure or table, then one paragraph that reads it, then one
that interprets it, then one that says what is not shown. Every figure caption states the images, the models, the
setting (own depth vs box placement; correction only vs fine-tuned) and the caveat (synthetic / by eye / a good case).

**The requirements map (7.1).** One table (T1), then one short paragraph per requirement in the supervisor's order
(1a-pose, 1a-shape, 1b, 1c, 2a, 2b, 3), each with: his wording, our status word, the evidence by section and table
number, what is missing and where in chapter 8 the plan for it stands. Status words only: met / met with caveats /
partly / not met / not started.

**Related work.** One paragraph per paper on how it relates to a choice made here; read the papers, not summaries
(P 6.1). Verify the two Multi-HMR numbers quoted in the draft (73.5 / 80.1 MPJPE) against the paper before reuse.

**AI use.** Follow RPTU's rules; the repository history shows the assistant's commits (P 6.1); the student must be
able to explain every component at the defence.

---

## 6. Work split

### Files to create under `docs/thesis/`

```
docs/thesis/thesis.tex                 new main (copy of the draft preamble + pgfplots, subcaption, siunitx, graphicspath, \input{numbers})
docs/thesis/numbers.tex                every headline number as a macro with its source in a comment
docs/thesis/references.bib             draft's 15 entries + the additions of 1.8
docs/thesis/chapters/00_abstract.tex
docs/thesis/chapters/01_introduction.tex
docs/thesis/chapters/02_background.tex
docs/thesis/chapters/03_data.tex
docs/thesis/chapters/04_method.tex
docs/thesis/chapters/05_diagnostics.tex
docs/thesis/chapters/06_results.tex
docs/thesis/chapters/07_discussion.tex
docs/thesis/chapters/08_conclusion.tex
docs/thesis/chapters/appendix_a_reproduction.tex
docs/thesis/chapters/appendix_b_joints.tex
docs/thesis/chapters/appendix_c_flags.tex
docs/thesis/chapters/appendix_d_figures.tex
docs/thesis/chapters/appendix_e_pilots.tex
docs/thesis/tables/T01_requirements.tex ... T18_shape.tex   (one file per table of section 4; draft tables copied as tables/draft_*.tex)
docs/thesis/figures/tikz/F01_architecture.tex, F02_pipeline.tex, F03_anchor.tex, F04_neck_chain.tex, F05_distillation.tex
docs/thesis/figures/plots/F15_depth_scatter.tex, F16_rh_age.tex, F18_cost_pve.tex, F19_student.tex, F20_val_curves.tex, F22_vits_curves.tex
docs/thesis/figures/img/                copies of docs/figures/*.jpg and the files the student sends (F10-F14, F17, F21, F23, F24)
docs/thesis/data/                       the CSVs for pgfplots
```

`latexmk -pdf thesis.tex` works on this machine (pdflatex, biber, TikZ, pgfplots, booktabs, biblatex, siunitx,
subcaption all installed; no matplotlib, so every chart is pgfplots or a table). Build once with placeholder figures
before the writers start, so the skeleton compiles.

### Order and dependencies

| step | files | can start | depends on | reads first |
|---|---|---|---|---|
| 0 skeleton | `thesis.tex`, `numbers.tex`, `tables/*.tex` (numbers filled from section 4), `chapters/*.tex` with section headings and the "absorbs" lines pasted from the draft | now | - | this plan; H 0b; H 5/7 |
| 1 Data | `03_data.tex` + T3-T6, F2 (TikZ), draft converter tables | after step 0 | - | H 3.5; D part 3; `smpl_to_anny.py` docstring; draft :455-554, :812-939 |
| 1 Method | `04_method.tex` + T2, T7, T11 part A, F1, F3, F4, F5 | after step 0 | - | D part 4-5; H 1; H 3.1-3.2; E exp8; `exp7_final_model.sh` header and :759-771; `model.py:45-120, 218-232, 562-621`; `bone_fix.py`; `loss.py:300-470`; draft :216-580 |
| 1 Diagnostics | `05_diagnostics.tex` + T8-T10, T14, T18 | after step 0 | - | F (whole); H 3.1-3.4, 3.6; D slides 10-12, 15, 17, 18, 22; `scripts/diagnostics/README.md` |
| 1 Background | `02_background.tex` + bib | now | - | draft :110-213; P 6.1 reading list; the papers |
| 2 Results | `06_results.tex` + T12, T15, T16, T17, F6-F20 | after Data and Method (shares T2, T7 and the model names) | the student's figure files; pending jobs for 6.10 | H 3.1-3.4, 5/7, 0a; D slides 13-21, 23-24; draft :665-810, :1131-1221 |
| 3 Discussion | `07_discussion.tex` + T1 | after Results | 6.x numbers | P 1-3; D part 7-8; H 6 (corrections log); draft :1240-1343 |
| 3 Appendices | A, C, E | after Results | T13 | H 8 job index; D slide 30; script headers |
| 4 Intro, Conclusion, Abstract | `01_`, `08_`, `00_` | last | everything | D part 1 (ten sentences); the finished chapters |
| 5 Review pass | all | after 4 | - | section 5 rules; H 6 corrections log (nothing listed there may reappear) |

Steps marked 1 run in parallel (four writers). Each writer owns their table files; the Results writer owns
`numbers.tex` and merges additions.

### Pages

| chapter | pages | of which figures / tables (est.) |
|---|---|---|
| Abstract, front matter | 3 | - |
| 1 Introduction | 5 | T1 short |
| 2 Background | 8 | F21 |
| 3 Data | 9 | F2, F14, T3-T6, 3 draft tables |
| 4 Method | 12 | F1, F3, F4, F5, T2, T7, T11, I/O table |
| 5 Diagnostics | 7 | F15, T8-T10, T14, T18, six-checks table, pilots table |
| 6 Results | 14 | F6-F13, F16-F20, T12, T15-T17, history tables |
| 7 Discussion | 6 | T1 full |
| 8 Conclusion | 2 | - |
| Appendices | 8 | T13, flags, F8, F9b, F22-F24, pilots |
| **total** | **~74** | |

---

## 7. Open questions for the student (and the cluster)

Only he or the cluster can answer these. Commands run from `/netscratch/najib/multi-hmr` after `git pull`, in
`multihmr`. Never cancel or resubmit a queued job (H 10).

1. **Submission date and the supervisor's four answers** (meeting Mon 19 Oct, 16:00): the 1b bar (our ViT-L, NAVER-L,
   or both) and whether a smaller Multi-HMR-style student counts as "not exactly Multi-HMR"; whether a refiner on frozen
   per-frame features counts for 2b; children as a limitation or weak labels; the date. Also: is our last-100 tail
   "the AnnyOne test set" or is there an official split; could NAVER's checkpoint have trained on those 100 images;
   is a body-joint MPJPE acceptable (P 5). Chapters 6.10, 7.1 and 8 depend on the answers.
2. **Student result (job 3614191).** When the log ends:
   ```
   cd /netscratch/najib/multi-hmr && tail -n 60 logs/exp8_distill_vitb_3614191.out && cat results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/final_table.json
   ```
   Send both, plus `grep -E "epoch|pve|PVE" results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train.log | head -200` for F19, and the student's `bone_fix.json`. If it missed the bar with a still-falling curve: the longer run (P 4 week 4) and the table "as measured".
3. **NAVER-L params and GFLOPs** (T16, F18): `cd /netscratch/najib/multi-hmr && bash slurm/diagnostics/model_cost.sh` (GPU job, 30 min; the login node has no EGL). Send `logs/model_cost_<jobid>.out` (read with `grep -vE "^Using cache|Warning|warn" logs/model_cost_<jobid>.out | tail -n 30`) and `results/diagnostics/model_cost_naver_vs_final.json`.
4. **Partial mask (job 3614042)** and its Relative Human run and demo: after it ends,
   ```
   cd /netscratch/najib/multi-hmr && MODEL=/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask_noage/train/checkpoints/00009.pt TAG=exp7_shpmask_noage_00009 sbatch --array=0 slurm/diagnostics/eval_relative_human.sh
   ```
   Send `logs/eval_rh_<jobid>_0.out`, the run's `holdout_ft/summary.json` and `eval_3dpw_test.log`. It decides whether 6.8 ends with "the unmasked model stays" or with a fourth column.
5. **ViT-S students on the v23 labels** (so the history section can keep a 3DPW row): 
   ```
   cd /netscratch/najib/multi-hmr && EVAL_LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --array=2,4 slurm/eval/run_eval_3dpw_test.sh
   ```
   (array 2 = v6c, 4 = v8; 2 x 15 min). Send `logs/eval-test3dpw-<jobid>_2.out` and `_4.out`. Without them the old 3DPW student numbers are dropped from the thesis.
6. **V1 numbers for the v2-v4 checkpoints** (the `TODO` cells of the draft's `tab:main`; optional, three 10-minute jobs):
   ```
   cd /netscratch/najib/multi-hmr && CKPT=/netscratch/najib/multi-hmr/models/multiHMR/anny_s2_frozen_bb_v2_00299.pt TAG=v2_frozen_00299 sbatch slurm/diagnostics/diag_holdout.sh && CKPT=/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt TAG=v3_partialft_00054 sbatch slurm/diagnostics/diag_holdout.sh && CKPT=/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_boosted_v4/checkpoints/00299.pt TAG=v4_boosted_00299 sbatch slurm/diagnostics/diag_holdout.sh
   ```
   Send the three `logs/diag_holdout_<jobid>.out`. Otherwise the history table keeps only the V0 PVE column.
7. **Did the ViT-L "v7 scratch" and "v8 partial freeze" runs finish** (`slurm/history/train_v7_scratch.sh`, `train_v8_partial_freeze.sh`)? `ls /netscratch/najib/multi-hmr/logs/anny_model/ | grep -iE "v7|v8"` and the last eval lines of their `train.log`. If not, the draft's `tab:ft` TODO row is dropped and v10 is the only "fresh heads" relative.
8. **Colour jitter.** The draft (student v4) and the exp8 header say colour jitter is used; no Python file in the repo reads such a flag (checked here: `grep -rn -i jitter train.py datasets/*.py distill.py` is empty). Confirm on the cluster copy: `cd /netscratch/najib/multi-hmr && grep -rn -i "jitter" --include=*.py . | grep -v archive`. If empty, the thesis says no augmentation was used.
9. **Re-derive the per-person 3DPW body values from the v23 fits** (to replace the reversed-gender claim of `main.tex:909-913`):
   ```
   cd /netscratch/najib/multi-hmr && conda activate multihmr && python - <<'PY'
   import glob, numpy as np
   for f in sorted(glob.glob('threedpw_anny_v23/*/*.npz')):
       z = np.load(f, allow_pickle=True); keys = list(z.files)
       print(f, 'gender_value' in keys and float(z['gender_value']), [k for k in keys if 'shape' in k or 'pheno' in k][:4])
   PY
   ```
   Send the output; the Data writer picks the shared-body values (height, proportions) of the two recurring courtyard actors from the right key once the key names are known.
10. **Control's in-training curve** for F20: `sed -n 48,157p logs/exp7_final_model_3611339.out` and `sed -n 55,164p logs/exp7_final_model_3611336.out`. Send both outputs as text.
11. **CSV headers for F15 / F16**: `head -1` of the six `per_person.csv` files named in section 3, then the files.
12. **Figures F10-F12, F14, F17, F21, F23, F24**: the commands in section 3. F11 and F17 each cost one demo job (~30-60 min GPU); the rest are file transfers.
13. **Phenotype responsiveness on AnnyOne** (r, slope, R² per phenotype; P 3.4 item 7, "to be written"): a login-node script over `holdout_ft/per_person.csv` once the column names (item 11) are known. Without it, 1a-shape rests on Relative Human and the line-up alone; say so in 7.1.
14. **Second seed of the final model** (optional, ~2 h GPU, P 4 week 4): `cd /netscratch/najib/multi-hmr && RUN=v5_pc_dist5_bonefix_v23_e10_seed2 bash slurm/experiments/exp7_final_model.sh` (same recipe, new run name; `train.py` has no seed, so this is a repeat, not a seeded run). It would turn "within run-to-run variation" into a measured spread for T15.
15. **Licences for the release paragraph**: AnnyOne, NAVER's checkpoints, Relative Human (no licence file: evaluation only), 3DPW (licensed) — the student checks the terms before chapter 8 promises a model release.
