# Reproducing the thesis numbers: scripts and commands

This file holds the scripts table and the exact commands that were Sections A.4 and A.5 of the thesis appendix
(moved here in the trimming pass of 19 Oct 2026; the long version is commit 76aa1a4). Appendix A of the thesis keeps
the model names (Table A.1), the runs, jobs, logs and checkpoints (Table A.2), the environment and the limits of
reproduction. Paths are relative to the repository root; on the cluster the clone is `/netscratch/najib/multi-hmr`.

## Scripts behind the experiments

Each Slurm script documents its recipe, its guards and its success rule in its header; the training scripts of the
L line and of Student-S lie in `slurm/history/` and `slurm/train/` (thesis Table A.1).

### Training experiments (`slurm/experiments/`)

| Script | What it produces |
|---|---|
| `exp1_depth_anchor.sh` | depth pilots: fine-tune of Base with the targets at the head joint (or at the pelvis, the control), holdout trace with the placement table |
| `exp2_children.sh` | sampler pilot: children per batch measured, fine-tune with age-balanced sampling, holdout, manifest and youngest-50 traces |
| `exp3_neck_global_rot.sh` | head-loss pilot: fine-tune with an extra loss on the camera-frame orientation of head, neck03 and wrists |
| `exp4_fingers.sh` | finger pilot: hand sizes in tokens, a test-time person-crop trace, a fine-tune with the region-normalised rotation loss |
| `exp5_neck_fix.sh` | geodesic pilot: fine-tune with the geodesic rotation loss, then a per-bone dump |
| `exp6_bone_fix.sh` | bone correction: estimates the constant per flipped bone, tests it without training and after a 2-epoch fine-tune |
| `exp7_final_model.sh` | Final (`ARM=fix`), Control (`ARM=control`), Final-masked and Final-partial-mask: label guards, training, holdout trace with per-bone dump, comparison table, 3DPW test |
| `exp8_distill_vitb.sh` | Student: guards, distillation from Final, the Student's own bone correction, holdout traces of Student and teacher, model cost, 3DPW tests, the final table `final_table.json` |

### Evaluation (`slurm/eval/`)

| Script | What it produces |
|---|---|
| `run_eval_3dpw_test.sh` | `train.py --eval_only 1`: AnnyOne test-100 (V0) and every 3DPW test image of the chosen labels; overrides `EVAL_LABELS`, `MODEL_ENTRY`, `EVAL_BONE_FIX`, `EVAL_MODEL_CLASS` |
| `run_eval_baseline_naver.sh` | the same evaluation for Multi-HMR-Anny, built with its own model class |
| `run_eval_3dpw.sh`, `run_eval_3dpw_teacher.sh` | older evaluation of one checkpoint on the AnnyOne holdout and the 3DPW validation split (history) |

### Diagnostics (`slurm/diagnostics/`)

| Script | What it produces |
|---|---|
| `diag_holdout.sh` | holdout trace (`diag_holdout_trace.py`): settings V0 to V3, placement, per-person CSV, panels |
| `diag_manifest.sh` | diagnostic manifest, ground-truth decode overlays, manifest traces |
| `diag_bodymodel.sh` | body-model probe: Anny age sweep, 16 single-joint perturbations, the 88 helper bones |
| `diag_gradients.sh` | gradient check of every head output |
| `diag_overfit.sh` | capped overfit on the 40 manifest training images |
| `diag_phenotypes.sh`, `diag_young.sh` | AnnyOne phenotype distribution; trace of the 50 youngest training people |
| `diag_crop_camera.sh` | predicted ages on the real photos with a corrected crop camera |
| `diag_neck_bones.sh` | per-bone dump of the neck chain and other bones, one trace per checkpoint |
| `check_3dpw_conversion.sh` | conversion check of a 3DPW label set against the original 3DPW ground truth |
| `eval_relative_human.sh` | Relative Human evaluation of one checkpoint (`eval_relative_human.py`) |
| `model_cost.sh` | parameters, GFLOPs and latency of one or more checkpoints on a GPU node (`model_cost.py`) |

### Demos (`slurm/demo/`)

| Script | What it produces |
|---|---|
| `run_demo_compare.sh` | two or more models on the same 4 photos and 8 3DPW test frames, bodies placed from the detector box; model list `DEMO_MODELS` |
| `run_demo.sh`, `run_demo_3dpw.sh`, `run_demo_syn.sh`, `run_demo_student.sh` | single-model demos on the photos, random 3DPW frames, AnnyOne holdout images, a student checkpoint |

### Python scripts

| Script | What it produces |
|---|---|
| `scripts/analysis/model_cost.py` | analytic backbone FLOPs, hooked head FLOPs, parameters, and with `--bench` the median latency at batch 1 |
| `scripts/diagnostics/eval_relative_human.py` | per Relative Human age group: recall, predicted age and height, child-vs-adult AUC, gender accuracy; `summary.txt` and `per_person.csv` |
| `scripts/diagnostics/compute_bone_fix.py` | the constant correction per flipped bone, as `bone_fix.json`, with the global guard |
| `scripts/data/smpl_to_anny.py` | converter v2.3: fits Anny to the SMPL joints of one 3DPW person-sequence, writes one `.npz` per person |
| `scripts/data/build_threedpw_anny_labels.py` | label files and split indices from the fits (login node) |
| `scripts/data/check_refit_task.py` | re-fit post-check: gender convention, shared frame grid, error per person (login node) |
| `slurm/data/run_smpl_to_anny_task.sh` | Slurm array over the 87 person-sequences of `configs/threedpw_tasks.txt` |

## Commands

The commands below are the ones that produced the thesis numbers, copied from the script headers and from the
project handover. Each runs from the login node in the repository clone, after `git pull` and with the conda
environment active (the submit modes run their label checks there before calling `sbatch`). The environment itself
(container, conda environment, GPU types) is described in Section A.2 of the thesis; the flags each run passes are
Appendix C.

### Final and Control

Guards only, then the two arms (5-hour limit each; one job each). The comparison table is rebuilt on the login node
once both have finished:

```bash
cd /netscratch/najib/multi-hmr && git pull && conda activate multihmr && CHECK_ONLY=1 bash slurm/experiments/exp7_final_model.sh
cd /netscratch/najib/multi-hmr && bash slurm/experiments/exp7_final_model.sh
cd /netscratch/najib/multi-hmr && ARM=control bash slurm/experiments/exp7_final_model.sh
cd /netscratch/najib/multi-hmr && TABLE_ONLY=1 bash slurm/experiments/exp7_final_model.sh
```

The smoke test, Final-masked and Final-partial-mask:

```bash
cd /netscratch/najib/multi-hmr && EPOCHS=1 N_ITERS=20 EVAL_TEST=0 VAL_3DPW_N=20 N_VIS=2 bash slurm/experiments/exp7_final_model.sh
cd /netscratch/najib/multi-hmr && MASK_3DPW_SHAPE=1 bash slurm/experiments/exp7_final_model.sh
cd /netscratch/najib/multi-hmr && MASK_3DPW_SHAPE=1 MASK_3DPW_SHAPE_NAMES=muscle,weight,cupsize,firmness,african,asian,caucasian bash slurm/experiments/exp7_final_model.sh
```

### 3DPW test of the older models

Base, Mixed-L and the neck pilot on the v23 labels, one job each:

```bash
cd /netscratch/najib/multi-hmr && for M in "logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt|teacher_v5" "models/multiHMR/anny_teacher_v9_mixed_00062.pt|teacher_v9_e62" "models/multiHMR/exp6_v5_bonefix_00001.pt|exp6_v5_bonefix"; do EVAL_LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels MODEL_ENTRY="/netscratch/najib/multi-hmr/${M%%|*}|0|dinov2_vitl14|${M##*|}" sbatch --array=0 slurm/eval/run_eval_3dpw_test.sh; done
```

Student-S/6c and Student-S/final on the same labels (array tasks 2 and 4 of the script's model list):

```bash
cd /netscratch/najib/multi-hmr && EVAL_LABELS=/netscratch/najib/multi-hmr/threedpw_anny_v23/labels sbatch --array=2,4 slurm/eval/run_eval_3dpw_test.sh
```

### Multi-HMR-Anny baseline

Checks, then one job (40-minute limit):

```bash
cd /netscratch/najib/multi-hmr && CHECK_ONLY=1 bash slurm/eval/run_eval_baseline_naver.sh && bash slurm/eval/run_eval_baseline_naver.sh
```

### Relative Human

Base and the neck pilot (array tasks 0 and 1), Final, and Final-masked; the bone correction comes from each
checkpoint's saved arguments. Final-partial-mask runs the same way with `MODEL` set to its checkpoint
(`v5_pc_dist5_bonefix_v23_e10_shpmask_noage/train/checkpoints/00009.pt`):

```bash
cd /netscratch/najib/multi-hmr && sbatch --array=0-1 slurm/diagnostics/eval_relative_human.sh
cd /netscratch/najib/multi-hmr && MODEL=/netscratch/najib/multi-hmr/models/multiHMR/exp7_v5_final_00009.pt sbatch --array=0 slurm/diagnostics/eval_relative_human.sh
cd /netscratch/najib/multi-hmr && MODEL=/netscratch/najib/multi-hmr/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt TAG=exp7_shpmask_00009 sbatch --array=0 slurm/diagnostics/eval_relative_human.sh
```

The scores can be recomputed from the cached predictions without a GPU, for example with another IoU:

```bash
cd /netscratch/najib/multi-hmr && conda activate multihmr && python scripts/diagnostics/eval_relative_human.py --score_only --out results/diagnostics/relative_human_val_exp7_v5_final_00009 --iou 0.5
```

### Model cost

Multi-HMR-Anny and Final, with latency (the script submits itself):

```bash
cd /netscratch/najib/multi-hmr && bash slurm/diagnostics/model_cost.sh
```

The analytic backbone table for ViT-S, ViT-B and ViT-L runs anywhere:

```bash
python scripts/analysis/model_cost.py --no-build --all
```

### Student

Smoke test, then the full run with a 20-hour limit:

```bash
cd /netscratch/najib/multi-hmr && conda activate multihmr && git pull && SMOKE=1 bash slurm/experiments/exp8_distill_vitb.sh
cd /netscratch/najib/multi-hmr && conda activate multihmr && git pull && TIME=20:00:00 bash slurm/experiments/exp8_distill_vitb.sh
```
