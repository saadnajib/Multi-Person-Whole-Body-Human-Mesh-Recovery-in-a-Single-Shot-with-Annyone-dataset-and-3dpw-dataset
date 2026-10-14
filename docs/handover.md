# Handover: state of the project on 14 October 2026

Read this first. It replaces the chat history. The other documents it points to are in this repository.

## 1. What the project is

Master thesis: Multi-HMR (ViT-L/14 DINOv2 backbone at 672 px + HPH cross-attention head) adapted to the
Anny body model (163 bones, 11 phenotypes in [0,1]: gender, age, muscle, weight, height, proportions,
cupsize, firmness, african, asian, caucasian). Trained on AnnyOne (579,960 synthetic 1280x1280 renders;
the last 100 images are the holdout, never trained on) and fine-tuned with 3DPW labels converted from SMPL.
Supervisor meeting: Friday 16 October 2026.

Repository: github.com/saadnajib/Multi-Person-Whole-Body-Human-Mesh-Recovery-in-a-Single-Shot-with-Annyone-dataset-and-3dpw-dataset
(branch main). Cluster copy: /netscratch/najib/multi-hmr (DFKI Slurm cluster, conda env `multihmr`,
enroot container, EGL for rendering). Shared Slurm setup in `slurm/env.sh`.

Models (all under /netscratch/najib/multi-hmr/logs/anny_model/<run>/checkpoints/ or models/multiHMR/):

| model | what | 3DPW-test PVE / PA-PVE |
|---|---|---|
| anny_s2_shape_v5, epoch 99 | ViT-L teacher, synthetic only; the reference for all diagnostics | 119.1 / 68.8 |
| anny_teacher_v9_mixed, epochs 62 and 149 | v5 + real 3DPW mixed in at the end; the best model | 102.5 / 62.7 |
| anny_v10_scratch_mixed, epoch 290-293 | ViT-L from scratch, real data from the start | 111.5 / 76.2 |
| anny_distill_vits_v6c / v8_final | ViT-S students | 139.5 / 88.0, 123.6 / 81.2 |

AnnyOne holdout numbers (100 images, helper bones masked = "V1", the setting the demos use):
v5 PVE 70.5 mm, PA-PVE 53.3, recall 90.2%. train.py evaluate() reports 74.3 because it does not mask the
88 helper bones (toes, breasts, face); the paired difference is +3.8 mm. Report V1 as the headline.

## 2. The three failures the thesis started from, and what the diagnostics found

Full evidence: `docs/diagnostics_findings.md`. Experiment plan: `docs/experiments_plan.md`.
Diagnostic scripts: `scripts/diagnostics/` with Slurm wrappers in `slurm/diagnostics/`.

What is NOT the cause: ground-truth parameters decode and render onto the people; all 16 single-joint
perturbations of the body model move the intended part; every active pose bone receives gradient; no
joint-mapping or rotation-convention error exists; the ImageNet-normalisation claim was false; the demo
crop-camera bug is placement-only.

| failure | diagnosed origin | key numbers (v5, holdout) |
|---|---|---|
| Neck looks wrong | weak per-bone rotation supervision on a redundant 4-bone chain; the long-neck look in overlays is mostly the depth defect below | neck length 99.3% of GT; local neck/head rotation error 63/90 deg but GLOBAL head orientation error 12 deg |
| Fingers wrong | under-trained: predicted fingers are worse than the rest pose; smallest gradients of any bone group; magnification does NOT help (measured) | finger rot error 17.3 deg vs 15.1 with fingers zeroed; crop x1.3-x3 changes it by 0.1 deg |
| Children come out as small adults | representation works in-sample (babies predicted at age 0.10 for GT 0.015) but 2.1% of supervised training people are children and 83.5% of batches contain none; 3DPW labels pin age at 0.55; no held-out child exists | age phenotype < 0.3 (about 11 y on the Anny scale) for 2.57% of people |

**The confirmed implementation bug nobody was looking for: depth.** The model attaches the mesh to the
detected pixel by its HEAD joint and reports the head as translation (model.py), but train.py prepare_gt
built the translation, distance, heatmap-cell and offset targets at joint 0, the ROOT. The 2D reprojection
losses reconcile the conflict by shrinking the body, i.e. pushing it away. Measured: predicted root depth
= 0.33 x GT + 2.36 m (GT mean 1.92 m, predicted 3.00 m, 100% of people predicted too far), body drawn ~35%
too small, head pixel error 64.5 px. Pelvis-centred PVE cannot see this, which is why the thesis numbers
looked fine while the overlays looked wrong. v10 has it too (slope 0.28, intercept 2.70 m). Fix:
`--gt_center_joint person_center` in train.py (experiment 1 below).

## 3. Experiments built this week (all flag-gated, default off, each its own commit and Slurm script)

| exp | flag(s) | script | status |
|---|---|---|---|
| exp1 depth anchor (bug fix) | `--gt_center_joint person_center`, `--debug_anchor N` | `slurm/experiments/exp1_depth_anchor.sh` | 2-epoch pilot done (below); 10-epoch runs submitted |
| exp2 children | `--age_balance 1`, `--age_balance_child_frac`, `--exclude_indices_file`, `--shape_weights` | `slurm/experiments/exp2_children.sh` | measurement done; pilot + control submitted |
| exp3 neck | `--alpha_global_rot`, `--global_rot_bones` (L1 on the post-FK global orientation of head, neck03, wrists) | `slurm/experiments/exp3_neck_global_rot.sh` | pilot + control submitted |
| exp4 fingers | trace `--person_crop 1` (inference), `--rotmat_region_norm 1`, `--mask_3dpw_hands 1` | `slurm/experiments/exp4_fingers.sh` | inference part done (below); region-norm pilot + control submitted |

Evaluation is always the same: `scripts/diagnostics/diag_holdout_trace.py` on the 100-image holdout and on
the diagnostic manifest (`CKPT=... TAG=... sbatch slurm/diagnostics/diag_holdout.sh` / `diag_manifest.sh`).
Always pass TAG, otherwise outputs overwrite each other. Throughput: 6.7 images/s at batch 4 with the last
4 backbone blocks unfrozen (1000 iterations = 10 min); one node gave 12.6 images/s.

## 4. Results measured so far (14 October)

**exp1, 2-epoch pilot, fix arm vs control arm (same schedule, flag off):**

| metric | baseline v5 | control (root targets) | fix (head targets) |
|---|---|---|---|
| root depth error | 1.08 m | 0.99 m | 0.56 m |
| depth fit intercept | 2.36 m | 2.28 m | 1.93 m |
| depth fit slope | 0.33 | 0.33 | 0.30 |
| head pixel error (mean) | 64.5 px | 58.1 px | 9.6 px |
| PVE / recall | 70.5 / 90.2 | 72.3 / 89.8 | 77.3 / 87.8 |

Reading: the fix does what the code analysis predicted (head placed correctly, depth error halved after only
8,000 images); the slope has not moved yet; PVE and recall dropped, most likely because the detection cell
moved from pelvis to head and two epochs is 1.4% of the data. The 10-epoch runs decide. The one-batch
anchor debug print confirmed the bug directly: with root targets the predicted offsets were -2 to -19 patches
against a legal range of +/-0.5.

**exp4 inference part:** at the 672-px input a holdout hand is 2.25 tokens long and 41% of hands are under 2
tokens; a x1.3 person crop makes it 4.2 tokens. Finger error stayed 17.2-17.3 deg at every magnification up
to x3 while PVE exploded (the crop camera's field of view was never seen in training). Conclusion: fingers are
a supervision/capacity problem, not a resolution problem. Children's hands are 0.66 tokens, unresolvable.

**exp2 measurement:** natural training batches: 2.1% of supervised people are children, 83.5% of batches have
none. Balanced sampler: 9.9% and 43%, with 23% of images containing a child (target 25%).

**v10 (epochs 290-293, identical):** PVE 74.1 on AnnyOne; depth slope 0.28 / intercept 2.70 m (same bug,
worse); local neck error only 5 deg (v5: 63) but global head error the same 12.9 deg (so the neck chain
decomposition is what differs, not the head); clavicle/shoulder 50 deg and arm 36 deg local error (v5: 11/8.5),
fingers 32 deg; shape biases equal to v5's. Unmasked helper bones are 110 deg off: v10 must always be run with
helper bones masked.

**Known instability:** 3-5 skipped optimiser steps per 2,000 for NaN/Inf gradients in every training run,
including old ones. Gradient clipping absorbs it; it belongs in the thesis as a known issue.

## 5. Jobs running or queued right now (14 October, afternoon)

- 3571933 exp1 long fix arm (10 epochs), 3571934 exp1 long control arm
- 3571955 exp2 pilot, 3571956 exp2 control (AGE_BALANCE=0)
- 3571957 exp3 pilot (alpha 5), 3571958 exp3 control (ALPHA_GROT=0)
- 3571959 exp4 region-norm pilot, 3571960 exp4 control (REGION_NORM=0)
- 3571963 / 3571964 holdout traces of v9 epoch 62 and epoch 149

Logs land in /netscratch/najib/multi-hmr/logs/ as <jobname>_<jobid>.out/.err; each experiment prints a
summary table and a VERDICT at the end of its .out. Read every pilot against its control arm, never only
against the baseline.

## 6. Decisions already taken

- README headline = V1 numbers; demos ship with `--relax_hands 1` (finger poses are the rest pose, not
  predictions) and say so; the failures are no longer described as rig or decoding problems.
- Nothing is claimed as an improvement until the trace of a fine-tuned checkpoint shows it.
- Children numbers are in-sample (no held-out child exists); the "hands" manifest category is confounded
  by truncation; the 51.8 mm capped-overfit figure is an optimisation check, not validation.
- The v9 lineage keeps the bug; when exp1 passes on v5, the fix goes into the mixed recipe
  (train_teacher_v9_mixed.sh + `--gt_center_joint person_center`) as a short fine-tune from v9 epoch 62,
  evaluated on both the AnnyOne holdout and 3DPW validation.

## 7. Open items

1. Read the two long exp1 arms: slope toward 1 and intercept toward 0 in the fix arm only, with PVE and
   recall recovering. If PVE does not recover by 10 epochs, try LR=5e-6 before giving up.
2. v9 traces: confirm the depth slope there.
3. exp2 / exp3 / exp4 pilots against their controls.
4. v10 shape-drift question: trace widely spaced epochs (50, 100, 200) if they exist; 290-293 are useless
   for a trend.
5. Not implemented, documented as candidates: the singular 6D pose init (fixing it output-neutrally on a
   trained checkpoint changes nothing), the boost renormalisation (x4 is really x2.41), the 3DPW identity
   fingers mask for mixed runs (flag exists, no script).
