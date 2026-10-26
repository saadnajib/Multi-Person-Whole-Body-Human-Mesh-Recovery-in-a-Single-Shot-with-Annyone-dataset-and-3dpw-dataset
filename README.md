# Multi-HMR × Anny — multi-person human mesh recovery with the Anny body model, AnnyOne and 3DPW

This repository builds on **[Multi-HMR](https://github.com/naver/multi-hmr)** (Baradel et al., ECCV'24).
It studies human mesh recovery with the **[Anny](https://github.com/naver/anny)** body model, using synthetic AnnyOne images and real 3DPW images converted from SMPL labels.

The completed thesis system uses a **ViT-B student**, distilled from our **Final ViT-L teacher** with supervised losses, feature distillation and output distillation. NAVER's released **Multi-HMR-Anny** checkpoint is the original external baseline. Earlier ViT-S students and Base/v5 are historical experiments, not the current student or the external baseline.

The repository includes label conversion, training and evaluation, image comparisons, and video comparisons with optional temporal filtering and landmark-guided alignment. The network still processes individual frames; temporal filtering combines predictions across time during inference, without retraining.

**Current entry points:** [image comparisons](docs/image_comparison.md), [video comparisons](docs/video_inference.md), [model names](docs/model_names.md), [thesis](docs/thesis/thesis.tex), and [presentation with speaker scripts](docs/presentations/README.md).

<div align="center">
<img width="49%" alt="Multi-HMR illustration 1" src="assets/visu1.gif">
<img width="49%" alt="Multi-HMR illustration 2" src="assets/visu2.gif">
</div>

---

## Contents
- [Repository layout](#repository-layout)
- [Status and known gaps](#status-and-known-gaps)
  - [Known defects](#known-defects)
- [Installation](#installation)
- [Quick start: demo](#quick-start-demo)
- [Student image and video comparisons](#student-image-and-video-comparisons)
- [Pipeline 1: converting 3DPW to Anny](#pipeline-1-converting-3dpw-to-anny)
- [Pipeline 2: training the ViT-L teacher](#pipeline-2-training-the-vit-l-teacher)
- [Pipeline 3: distilling the student](#pipeline-3-distilling-the-student)
- [Evaluation](#evaluation)
- [Results recorded so far](#results-recorded-so-far)
- [Utility scripts](#utility-scripts)
- [Running on the cluster (Slurm)](#running-on-the-cluster-slurm)
- [Upstream Multi-HMR (SMPL-X) usage](#upstream-multi-hmr-smpl-x-usage)
- [License and citation](#license-and-citation)

---

## Repository layout

```
.
├── train.py                 # training / evaluation entry point (Anny + SMPL-X, distillation, mixed data)
├── demo.py                  # inference + rendering on a folder of images
├── demo_video.py            # video overlays, temporal filtering, optional landmark alignment
├── demo_image_compare.py    # input / current Student / original NAVER baseline PNG comparisons
├── model.py                 # Multi-HMR model (Anny output, optional head_dim projector for students)
├── loss.py                  # training losses
├── distill.py               # knowledge-distillation loss (temperature-softmax KL on backbone tokens)
├── app.py                   # Gradio demo (HuggingFace Space)
│
├── blocks/                  # network building blocks (DINOv2 backbone, cross-attention HPH, camera embedding, SMPL layer)
├── multi_hmr_anny/          # NAVER's reference Anny Multi-HMR (used to load the official multiHMR_672_L_anny checkpoint)
├── utils/                   # camera, rendering, image, tensor and training helpers
├── datasets/                # dataloaders: annyone.py, bedlam.py, ehf.py, threedpw.py, threedpw_anny.py
│
├── scripts/
│   ├── data/                # dataset conversion and export (SMPL→Anny fitting, label building, image export)
│   ├── analysis/            # diagnostics: fit checks, shape/phenotype analysis, checkpoint inspection, montages
│   ├── video/               # rendering checks, tracking, temporal filters, landmark extraction/fitting
│   └── patches/             # historical apply_*.py patches (already applied to train.py / model.py)
│
├── slurm/
│   ├── env.sh               # shared job setup (conda, caches, cd, DINOv2 patch) — edit paths here
│   ├── train/               # earlier teacher/student training scripts
│   ├── experiments/         # Final teacher (exp7), ViT-B student (exp8), diagnostic experiments
│   ├── eval/                # evaluation jobs (AnnyOne holdout, 3DPW val/test, dataloader test)
│   ├── demo/                # demo / visual comparison jobs
│   ├── data/                # 3DPW→Anny conversion pipeline + fit check
│   └── history/             # finished runs and one-off jobs, kept unchanged as the experiment record
│
├── configs/
│   └── threedpw_tasks.txt   # 3DPW (split, sequence, person, n_frames) task list for the conversion array job
│
├── example_data/            # real photos for the demo (default --img_folder)
├── demo_images/
│   ├── 3dpw/  3dpw_random/  # 3DPW frames for demos
│   ├── annyone/             # AnnyOne holdout images (579860–579869)
│   └── annyone_fov.json     # per-image field of view for the AnnyOne demo images
│
├── threedpw_anny_full/labels/index_{train,validation,test}.txt   # index of converted 3DPW→Anny labels
│
├── results/                 # committed qualitative outputs
│   ├── demos/               # A_base, B_neckshape, C_hands, v4_restpose ablations on example_data
│   ├── anny_fit_check/      # fit diagnostics (error histograms, overlays, phenotypes)
│   └── 3dpw_to_anny_renders/{v2,v21}/   # Anny fits rendered over 3DPW frames (two converter versions)
│
├── docs/                    # investigation notes: diagnostics findings, 3DPW adapter check, Anny shape report, cluster commands
├── archive/                 # superseded code kept for reference (train_v1–v3.py, demo_original.py)
├── assets/                  # README gifs
├── requirements.txt / conda.yaml
└── LICENSE.txt / NOTICE.txt
```

**Rule of thumb.** The core Python modules live at the repo root so that `import model`, `import utils` and `import datasets.*` keep working.
Every script under `scripts/` adds the repo root to `sys.path` itself.
Always run commands **from the repo root**, for example `python scripts/data/smpl_to_anny.py …`.

---

## Status and known gaps

Read this before trying to reproduce anything.

| Gap | Impact | Fix |
|---|---|---|
| **Cluster paths.** Conda, caches and the project root are set once in `slurm/env.sh`, but job scripts still name `/netscratch/najib/...` for checkpoints, data, logs and the container image. | The scripts only run on the original cluster as-is. | Edit `slurm/env.sh`, then search-and-replace `/netscratch/najib` in the job you want to run. |
| **Checkpoints are not versioned** (`*.pt` is git-ignored). | Final teacher/student weights must be obtained separately. | Use the checkpoint paths and correction metadata in [image/video comparisons](#student-image-and-video-comparisons). |
| `scripts/patches/apply_*.py` are historical. | All six patches are already applied to `train.py` / `model.py`. The guard checks in the training scripts pass. | Do not re-run them. They refuse to patch twice anyway. |

### Known defects

The earlier v5 depth-target mismatch and bone-rotation investigation are documented in
[diagnostics findings](docs/diagnostics_findings.md) and the [handover](docs/handover.md).
Final incorporates the head-anchor target correction and bone correction; old v5 numbers
must not be presented as current Final or Student results.

Current limitations:

- The completed 30-epoch Student reduces model cost but misses the teacher-based 95% accuracy-retention criteria on both evaluation sets.
- Children, fingers and real-photo body-shape generalization remain limitations. The synthetic holdout contains adults and cannot establish child performance.
- Converted 3DPW labels include fixed shape values and unavailable finger rotations. Label fitting does not remove differences between the SMPL and Anny body models.
- Video filtering can reduce jitter but introduces lag. Landmark alignment uses an external detector and improves image agreement, which does not establish better ground-truth 3D accuracy. Rejected observations and tracking resets can still cause visible changes.
- All model weights and cluster-only outputs must be obtained separately. Completed comparison commands and defaults are documented below.


---

## Installation

Tested with Python 3.9, CUDA 12.1 and torch 2.0.1.

```bash
# venv
python3.9 -m venv .multihmr && source .multihmr/bin/activate
pip install -r requirements.txt          # includes the `anny` package

# or conda
conda env create -f conda.yaml && conda activate multihmr
```

Extra setup:
- **DINOv2 on Python 3.9.** The torch-hub DINOv2 code uses `X | None` type hints, which Python 3.9 rejects. Every Slurm script patches it like this:
  ```bash
  python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',pretrained=False)" || true
  find $TORCH_HOME/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +
  ```
- **Headless rendering.** Set `PYOPENGL_PLATFORM=egl` (and `EGL_DEVICE_ID=0`).
- **Anny cache.** Anny writes to `~/.cache/anny`. On quota-limited homes, symlink it to scratch space (the Slurm scripts already do this).
- **SMPL models.** These are needed for 3DPW conversion and evaluation: `models/smpl/SMPL_NEUTRAL.pkl` (plus MALE/FEMALE for upstream 3DPW eval) and `models/smplx/SMPLX_NEUTRAL.npz`.

---

## Quick start: demo

This section describes the legacy `demo.py` workflow. For comparisons of the completed ViT-B Student and original NAVER baseline, use the [image/video tools below](#student-image-and-video-comparisons).

The official NAVER Anny checkpoint downloads automatically to `$HOME/models/multiHMR`:

```bash
python3.9 demo.py \
    --img_folder example_data \
    --out_folder demo_out \
    --model_name multiHMR_672_L_anny \
    --save_mesh 1
```

To run one of the checkpoints trained here, put it in the checkpoint cache as `<name>.pt` and pass `--model_name <name>`.
`slurm/demo/run_demo*.sh` copies `logs/anny_model/<run>/checkpoints/<epoch>.pt` into that cache for you.

Recommended flags depend on the input domain (see `slurm/demo/run_demo.sh` for the reasoning behind each one):

| Flag | Real photos (out of domain) | AnnyOne / synthetic (in domain) |
|---|---|---|
| `--use_person_detector` | `1` (Faster R-CNN-guided crops, `--detector_thresh 0.8 --crop_margin 0.35 --anchor_to_box 1`) | `0` |
| `--det_thresh` | `0.15` | `0.3` |
| `--fov` / `--fov_json` | `--fov 55` | `--fov_json demo_images/annyone_fov.json` (per-image FOV) |
| `--mask_helper_joints 1` | ✔ forces the 88 untrained helper bones to identity (the V1 evaluation setting) | ✔ |
| `--relax_hands 1` | ✔ sets the finger poses to the rest pose (see below) | ✔ |

**Legacy rest-hand demo renders use the rest pose for fingers.** The newer `run_image_compare.sh` and `run_video_compare.sh` retain predicted finger poses (optionally temporally filtered in video).
The legacy rest-hand demo jobs pass `--relax_hands 1`, which sets the finger and thumb phalanges to identity.
This is deliberate: the predicted fingers are worse than the rest pose (rotation error 17.3° vs 15.1° on the AnnyOne holdout; see [Known defects](#known-defects)).

Do **not** use `--neutral_phenotypes`.
Measured against AnnyOne ground truth, forcing phenotypes to 0.5 moved them *away* from the truth and made necks worse (see `docs/anny_shape_report.txt` and the comments in `run_demo.sh`).

Side-by-side comparison of several models: `slurm/demo/run_demo_compare.sh`, then `scripts/analysis/make_montage.py`.

---

## Student image and video comparisons

Run from the checkout root on the configured cluster. Both launchers use the regular
`multihmr` environment, the existing container and an EGL rendering check before
loading weights. They load one model at a time. The required defaults are:

| Model | Checkpoint / metadata |
|---|---|
| Student (ViT-B, 30 epochs) | `results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/00029.pt` |
| Original NAVER baseline | `models/multiHMR/multiHMR_672_L_anny.pt` |
| Student correction | Run-local `student_fix_arg.txt`, resolved automatically; keep it and its referenced correction JSON with the checkpoint. |

Override `STUDENT`, `BASELINE` and `STUDENT_BONE_FIX` when using different verified
paths. A copied checkpoint alone may lack the student's post-training correction.

### Compare photos

```bash
cd /netscratch/najib/multi-hmr
conda activate multihmr
mkdir -p logs
IMAGES="$PWD/example_data" sbatch slurm/demo/run_image_compare.sh
```

`IMAGES` accepts one photo or a folder of original photos. Results are labelled
**Input | Student | Original Multi-HMR-Anny** PNGs in
`results/image_compare/images_JOBID/comparison/`. All visible detections are rendered.
See [image comparison options and outputs](docs/image_comparison.md).

### Compare video without alignment

```bash
VIDEO="$PWD/man_walking.mp4" TEMPORAL=ema ALIGNMENT=off LANDMARKS="" STUDENT_ONLY=0 \
  sbatch slurm/demo/run_video_compare.sh
```

This saves input, raw Student, temporally filtered Student and original baseline
panels. Set `TEMPORAL=off` for frame-by-frame predictions only, or `STUDENT_ONLY=1`
to omit the baseline. The temporal filter blends pose rotations, body shape and
head-anchor position using previous predictions, then rebuilds the mesh through
Anny. It resets when detections are missing or tracking is discontinuous.

### Add landmark alignment

First extract landmarks from the **original video** in a separate CPU environment;
follow [landmark setup and extraction](docs/video_inference.md#improve-image-alignment-with-detected-landmarks).
Then reuse that JSON for the same source video:

```bash
VIDEO="$PWD/man_walking.mp4" TEMPORAL=ema ALIGNMENT=landmarks \
ALIGNMENT_DEPTH_TAU=0.35 LANDMARKS="$PWD/results/video_landmarks/man_walking.json" \
STUDENT_ONLY=0 sbatch slurm/demo/run_video_compare.sh
```

Alignment uses detected body landmarks to adjust camera-space position and depth,
retaining the predicted pose and shape. The final depth estimate is smoothed to
reduce apparent size fluctuations. Larger `ALIGNMENT_DEPTH_TAU` values add smoothing
and lag; `0` disables depth smoothing. Unreliable landmark fits keep the unaligned
prediction and reset alignment depth history.

Video results are in `results/video_compare/video_JOBID/`. `comparison.mp4` contains
all requested panels; `alignment_comparison.mp4` shows input, pre-alignment Student
and aligned Student. `run.json`, per-frame CSVs and summaries record settings,
tracking, corrections and timings. The original baseline remains unfiltered.
Aligned-panel FPS excludes offline landmark detection; displayed FPS is not
end-to-end video throughput. Visual fit and smoothness do not establish PVE accuracy.

Both launchers support `CHECK_ONLY=1` with `bash` instead of `sbatch` for file/option
checks without submitting. Each run requires a new output directory. Full
[video documentation](docs/video_inference.md) describes defaults, timing and limitations.

---

## Pipeline 1: converting 3DPW to Anny

3DPW provides SMPL parameters. The steps below fit Anny to each tracked person and export per-image labels in AnnyOne format, so that 3DPW can be mixed into training (`--mix_3dpw_labels`) and used for evaluation in Anny space.

| Step | Slurm job | Script | Output |
|---|---|---|---|
| 1. Extract 3DPW and list tasks | `slurm/data/run_3dpw_list.sh` | – | `data/3DPW/`, `configs/threedpw_tasks.txt` (one line per split/seq/person/n_frames) |
| 2. Fit SMPL → Anny (array job, `%8` in parallel) | `slurm/data/run_smpl_to_anny_task.sh` | `scripts/data/smpl_to_anny.py` (v2.2) | `threedpw_anny_full/<split>/<seq>_p<person>.npz` |
| 3. Build per-image labels (CPU, ~1 min) | `slurm/data/run_build_labels.sh` | `scripts/data/build_threedpw_anny_labels.py` | `threedpw_anny_full/labels/<split>/<seq>/image_NNNNN.pkl` and `labels/index_<split>.txt` |

```bash
sbatch slurm/data/run_3dpw_list.sh
N=$(wc -l < configs/threedpw_tasks.txt)
sbatch --array=0-$((N-1))%8 slurm/data/run_smpl_to_anny_task.sh
sbatch slurm/data/run_build_labels.sh
```

**How the fit works (`smpl_to_anny.py`).**
- Gradient descent fits Anny pose and shape to 3DPW `jointPositions`.
- Frames are sampled with stride 3: a cold start for the first frame, warm starts afterwards, then a refinement pass.
- One shape is shared per person. Only the `height` and `proportions` phenotypes are free; `age` is fixed at 0.55.
- Each frame also stores the mesh bbox and the visible fraction.

The camera convention `X_cam = R·X_world + t` was checked against the OpenPose 2D keypoints (8.9 px mean reprojection error); see `docs/3dpw_adapter_check.md`.

**Quality checks.**
- `scripts/analysis/check_anny_fit.py`: rotation validity, joint-error histogram, phenotype consistency, skeleton overlays and a mirror test. Job: `slurm/data/run_check_anny_fit.sh`. Output: `results/anny_fit_check/`.
- `scripts/analysis/render_anny_fit.py`: overlays the fitted meshes on 3DPW frames. Output: `results/3dpw_to_anny_renders/`.

**Earlier pilots, kept for reference.**
- `run_smpl_to_anny.sh` and `run_smpl_to_anny_scenes{,_v21}.sh`: 4-scene pilots with v2 and v2.1 of the converter.
- `scripts/data/smpl_to_anny_v3_frozen.py`: an experimental variant with a frozen-shape second pass.
- `scripts/data/npz_to_annyone.py` (+ `run_npz_to_annyone.sh`): the first single-sequence adapter, which `build_threedpw_anny_labels.py` replaced.

The earlier v2.2 paths above describe the historical conversion workflow. The evaluated **v2.3** label set contains **5,882 train / 2,622 validation / 8,190 test** images. See the [thesis data records](docs/thesis/numbers_data.tex) and [reproduction guide](docs/thesis/REPRODUCE.md) for the corrected split and conversion protocol.

---

## Pipeline 2: training the ViT-L teacher

The completed **Final teacher** is recorded by [`slurm/experiments/exp7_final_model.sh`](slurm/experiments/exp7_final_model.sh), combining head-anchor targets, bone correction and mixed-data fine-tuning. The table below is the **earlier training lineage**; v9 is not the Final teacher used by exp8.

All runs start from NAVER's `multiHMR_672_L_anny.pt` backbone (`--pretrained_remap 1 --load_only_backbone 1`) at 672 px on AnnyOne, unless noted.
Each training script is the full, self-documenting record of a run. Its header explains the reasoning and quotes the previous run's numbers.
Scripts of finished runs live in `slurm/history/` (unchanged); the runs still in use are in `slurm/train/`.

| Order | Script | Run name | Idea |
|---|---|---|---|
| 0 | `train_anny_test.sh` → `train_anny.sh` → `train_anny_medium.sh` | anny_small_run → anny_full_run_v1/v2 | Smoke test, then first full AnnyOne runs; 2D losses from epoch 20 |
| 1 | `train_step1_pretrained_bb.sh` (+ `_resume`) | anny_s1_pretrained_bb_v2 | Pretrained backbone + fresh heads, full fine-tune |
| 2 | `train_step2_frozen_bb.sh` (+ `_resume`) | anny_s2_frozen_bb_v2 | Same with the backbone frozen (epoch 299 is the step-2 result) |
| 3 | `train_s2_partialft.sh` | anny_s2_partialft_v3 | Unfreeze the last 4 blocks + mask the 88 helper joints |
| 4 | `train_s2_boosted_v4.sh` | anny_s2_boosted_v4 | ×4 loss weight on neck/hand bones (no visible gain) |
| 5 | `train_s2_shape_v5.sh` | **anny_s2_shape_v5** | `--alpha_shape 20` fixes the age/height bias. **Teacher for distillation v1–v7** (epoch 99) |
| – | `train_v7_scratch.sh`, `train_v8_partial_freeze.sh` (+ `train_v8_resume.sh`) | anny_v7_scratch, anny_partial_freeze | Clean re-runs from the base with all fixes (ablations) |
| 6 | `train_teacher_v9_mixed.sh` | **anny_teacher_v9_mixed** | v5 recipe + 30% real 3DPW-train, evaluated on 3DPW-val. **Teacher for the historical ViT-S v8 student** (epoch 62) |
| – | `train_v10_scratch_mixed.sh` | anny_v10_scratch_mixed | Ablation: mixed data from the start |

Key flags introduced along the way:
- `--mask_helper_joints_train 1`
- `--alpha_shape 20`
- `--use_anny_shape 1`
- `--person_center head`
- `--mix_3dpw_labels <dir> --mix_3dpw_ratio 0.3 --mix_3dpw_split train`
- `--val_3dpw_split validation --val_3dpw_n 300`
- `--tf32 1`

---

## Pipeline 3: distilling the student

The completed student is **ViT-B**, recorded in
[`slurm/experiments/exp8_distill_vitb.sh`](slurm/experiments/exp8_distill_vitb.sh).
It uses a 768 → 1024 feature projector, teacher-initialized heads, supervised
losses, feature KL distillation and output distillation from the frozen Final
teacher. The confirmed run trained for 30 epochs of 1,000 steps at batch size 8,
with heads frozen for the first 5 epochs and approximately 70% AnnyOne / 30% 3DPW
image sampling. See [model identities](docs/model_names.md) and
[the training recipe](docs/thesis/tables/T11_student_recipe.tex).

The following **historical ViT-S experiments** explain the earlier development.
Student: `dinov2_vits14` (384-d). Teacher: frozen ViT-L (1024-d).

**Loss (`distill.py`).**
- KL divergence between temperature-softmaxed backbone patch tokens (`--lambda_kd`, `--kd_temperature 4`, softmax over channels), scaled by T².
- A learnable projector maps the student's 384-d tokens to the teacher's 1024 dimensions.
- An optional output-level L1 term against the teacher's predictions (`--lambda_kd_out`).

**Run lineage (each run motivated by the previous run's failure mode):**

| Run | Script | Change | Recorded result |
|---|---|---|---|
| v1 | `train_distill_vits.sh` | Feature KL only, ImageNet DINOv2 init | Sanity check |
| v2 | `train_distill_vits_v2.sh` | Lower LR, slower decay | PVE 193.4 (teacher 73.6) |
| v3 | `train_distill_vits_v3.sh` | `--init_heads_from_teacher 1`, `--lambda_kd_out 1` | Worse: only the head *input* layers fail to transfer |
| v4 | `train_distill_vits_v4.sh` | Init from `multiHMR_672_S`, colour jitter | 156 / 117 |
| v5 | `train_distill_vits_v5.sh` | `--freeze_heads_epochs 5` | 159 / 113 |
| v6 | `train_distill_vits_v6.sh` | **`--head_dim 1024`**: projector before the heads, so all 67/67 head tensors transfer | 152 / 105 |
| v6b → v6c | `train_distill_vits_v6b.sh`, `_v6c.sh` | Continue training; `--teacher_bf16`, batch 16 | **114.5 / 84.6** on the AnnyOne holdout (best synthetic-only) |
| v7 | `train_distill_vits_v7_mixed.sh` | + 30% real 3DPW | **3DPW-val 107.0 / 77.1**; holdout 120.1 / 89.4 |
| **v8 (historical ViT-S final)** | `train_distill_vits_v8_final.sh` | v6c weights + v7 mixed data at batch 16, teacher v9 | 3DPW-test is evaluated once at the end |

Numbers are PVE / PA-PVE in mm, copied from the script headers.
They come from `train.py evaluate()`, which does not mask the helper bones (V0, see [Results recorded so far](#results-recorded-so-far)).

The student flags (`--init_heads_from_teacher`, `--freeze_heads_epochs`, `--head_dim`, `--teacher_bf16`, `--mix_3dpw_*`) came from the patches in `scripts/patches/`.
Those patches are already applied, so the flags exist in `train.py`.

```bash
sbatch slurm/train/train_distill_vits_v8_final.sh
```

---

## Evaluation

| Job | What it does |
|---|---|
| `slurm/eval/run_eval_3dpw.sh` | Student on 3DPW-validation (Anny labels) |
| `slurm/eval/run_eval_3dpw_teacher.sh` | Teacher on 3DPW-validation |
| `slurm/eval/run_eval_3dpw_test.sh` | **3DPW-test: run once, at the end** |
| `slurm/history/run_eval_test.sh` | Legacy AnnyOne evaluation through `archive/train_v3.py` |
| `slurm/eval/run_test.sh` | Dataloader smoke test (`scripts/analysis/test_loader.py`) |

Evaluation goes through `train.py --eval_only 1`.
The AnnyOne holdout (`--val_anny_n`) and 3DPW (`--val_3dpw_split`, `--val_3dpw_n`) are evaluated every `--eval_freq` epochs during training.

---

## Results recorded so far

Completed results for the Final teacher, 30-epoch Student and original NAVER baseline:

| Dataset | Model | PVE ↓ (mm) | PA-PVE ↓ (mm) | MPJPE ↓ (mm) | Recall ↑ |
|---|---|---:|---:|---:|---:|
| AnnyOne holdout (V1) | Teacher (Final, ViT-L) | 73.7 | 50.3 | 77.9 | 87.0% |
| AnnyOne holdout (V1) | Student (ViT-B) | 119.6 | 80.8 | 127.2 | 85.4% |
| AnnyOne holdout (V1) | Original NAVER baseline* | 60.3 | 44.6 | 63.3 | 89.0% |
| 3DPW test (v23 Anny labels) | Teacher (Final, ViT-L) | 89.0 | 57.6 | 88.3 | 99.0% |
| 3DPW test (v23 Anny labels) | Student (ViT-B) | 110.1 | 71.8 | 112.0 | 99.0% |
| 3DPW test (v23 Anny labels) | Original NAVER baseline | 126.9 | 68.8 | 129.4 | 97.0% |

*The baseline's synthetic holdout is potentially in-sample. Teacher and Student
received converted 3DPW training supervision; the baseline did not. These 3DPW
numbers use Anny labels and cannot be directly compared with published SMPL-space
scores. PVE and MPJPE are pelvis-centred; PA-PVE additionally aligns rotation,
translation and scale. Errors describe matched predictions, while recall captures
missed people. Lower errors and higher recall are better.

| Model | Parameters | Full-model compute | Median forward latency (L40S) |
|---|---:|---:|---:|
| Original NAVER baseline | 380.94 M | Not measured; backbone alone 1,917.2 GFLOPs | Not measured |
| Teacher (Final) | 318.38 M | 1,927.4 GFLOPs | 59.7 ms |
| Student (ViT-B) | 101.38 M | 603.3 GFLOPs | 28.5 ms |

Teacher/Student cost uses the paired L40S benchmark: 672-pixel input, one person,
batch one, fp32 with TF32. The Student has 3.14× fewer parameters, 3.19× less counted
computation and 2.10× faster forward time than its teacher, but higher reconstruction
error. Against the original baseline, it has 3.76× fewer parameters and lower 3DPW
PVE/MPJPE, while PA-PVE is higher. Model-forward timings exclude rendering and video I/O.

Sources: [thesis results](docs/thesis/numbers_results.tex),
[baseline handover record](docs/handover.md), and [student comparison table](docs/thesis/tables/T17_student.tex).
Earlier v5/ViT-S results and legacy qualitative ablations remain in the handover
and `results/demos/`. Images in `docs/thesis/figures/img/` mainly compare Base/v5
with Final; they must not be relabelled as current Student-versus-NAVER results.

---

## Utility scripts

**`scripts/data/`**
| Script | Purpose |
|---|---|
| `smpl_to_anny.py` | SMPL → Anny fitter for 3DPW (v2.2) |
| `smpl_to_anny_v3_frozen.py` | Experimental fitter variant (SMPL forward first, per-bone weights, frozen-shape second pass) |
| `build_threedpw_anny_labels.py` | Fits → per-image AnnyOne-format labels + split index |
| `npz_to_annyone.py` | Earlier single-sequence fit → label adapter |
| `anyone_image_export.py` | Export AnnyOne images + `annyone_fov.json` |
| `threedpw_image_export.py` | Export 3DPW images + a per-image FOV map |
| `sample_images.py` | Sample 3DPW frames across sequences into a flat demo folder |

**`scripts/analysis/`**
| Script | Purpose |
|---|---|
| `check_anny_fit.py` | Sanity report on a fit `.npz` |
| `render_anny_fit.py` | Render fitted meshes over 3DPW frames |
| `compare_gt_pred_shape.py` | GT vs predicted phenotypes on AnnyOne holdout samples (neck investigation) |
| `phenotype_stats.py` | GT phenotype distribution of AnnyOne |
| `check_mask_boost.py` | Check whether boosted neck/hand bones overlap the masked helper bones |
| `check_shape.py`, `explore_anny.py` | Inspect raw AnnyOne samples, collated batches and phenotype labels |
| `inspect_ckpt.py` | Which model class a checkpoint's keys match (catches silent `strict=False` no-loads) |
| `inspect_prefix.py` | Dump the key tree under a checkpoint prefix |
| `make_montage.py` | Side-by-side montage of demo outputs from several models |
| `test_loader.py` | AnnyOne DataLoader smoke test |

**`scripts/patches/`** (already applied; listed in the order they were applied)
1. `apply_student_fix.py`: `--init_heads_from_teacher`, `--lambda_kd_out`.
2. `apply_student_fix_v2.py`: `--freeze_heads_epochs`.
3. `apply_student_fix_v3.py`: `--head_dim` projector (model.py + train.py).
4. `apply_speed_fix.py`: `--tf32`, `--teacher_bf16`, loader pinning.
5. `apply_mixed_data_fix.py`: `--mix_3dpw_*`, `--val_3dpw_*`.
6. `apply_mixed_data_fix_v2.py`: 3DPW loader in `--eval_only`.

---

## Running on the cluster (Slurm)

Each script in `slurm/` is a standalone `sbatch` job that runs one command inside an enroot container.
The setup every job needs (conda, cache directories, the `~/.cache/anny` redirect, thread limits, `cd` to the project root, the DINOv2 Python 3.9 patch) lives in **`slurm/env.sh`**; active jobs start with `source /netscratch/najib/multi-hmr/slurm/env.sh`.
Scripts in `slurm/history/` predate `env.sh` and still carry their own copy of that setup.

Submit jobs from the repo root so that relative `logs/` paths resolve:

```bash
mkdir -p logs
sbatch slurm/train/train_teacher_v9_mixed.sh
sbatch slurm/demo/run_demo.sh
```

Legacy training checkpoints are written to `logs/anny_model/<run>/checkpoints/<epoch>.pt`, and only the 10 most recent are kept.
For exp8, use the `results/experiments/` checkpoint and correction metadata listed in the comparison section above.
**For legacy resume workflows, copy the final checkpoint to `models/multiHMR/<run>_<epoch>.pt` before launching anything that resumes from it.**
`docs/cluster_commands.txt` contains example copy commands.

---

## Upstream Multi-HMR (SMPL-X) usage

The original SMPL-X functionality is unchanged:
- BEDLAM, EHF and 3DPW preprocessing.
- `train.py --train_data BEDLAM`.
- The official checkpoints (`multiHMR_896_L`, `multiHMR_672_{L,B,S}`, `multiHMR_672_L_anny`, …).

See the [upstream README](https://github.com/naver/multi-hmr#readme) for dataset preparation, the checkpoint table and the CUFFS dataset.

---

## License and citation

Code and checkpoints follow the terms of [LICENSE](LICENSE.txt) and [NOTICE](NOTICE.txt) (NAVER Multi-HMR).
The Anny checkpoint has its own [license](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/Checkpoint_License_Anny.txt).
AnnyOne, 3DPW, SMPL and SMPL-X are each subject to their own licenses.

```bibtex
@inproceedings{multi-hmr2024,
    title={Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot},
    author={Baradel*, Fabien and Armando, Matthieu and Galaaoui, Salma and
            Br{\'e}gier, Romain and Weinzaepfel, Philippe and Rogez, Gr{\'e}gory and Lucas*, Thomas},
    booktitle={ECCV},
    year={2024}
}
```
