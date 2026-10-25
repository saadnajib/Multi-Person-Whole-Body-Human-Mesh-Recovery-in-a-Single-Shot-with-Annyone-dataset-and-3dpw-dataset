# Student versus original baseline: image comparisons

`demo_image_compare.py` compares the completed 30-epoch ViT-B student with
NAVER's original released Multi-HMR-Anny checkpoint on the same RGB images.
Each PNG shows **input | student | original baseline**, with detection counts.
All visible predicted people are drawn in a shared depth buffer for each model.
This is raw image inference: no tracking, temporal smoothing, landmark alignment,
detector crop, forced box placement, or rest-pose hands. Student helper masking
and the run's own bone correction follow the existing video predictor.

## Cluster command

Put your original photos in a folder, then run from the checkout root:

```bash
cd /netscratch/najib/multi-hmr
conda activate multihmr
mkdir -p logs
IMAGES="$PWD/my_images" sbatch slurm/demo/run_image_compare.sh
```

`IMAGES` also accepts a single photo. Directory inputs scan direct children only
(sorted JPG/JPEG/PNG/WebP/BMP/TIF/TIFF files); subfolders are not scanned. The direct
Python CLI accepts several files/folders with `--images`. EXIF orientation is
applied before full-image letterboxing. Duplicate paths are removed, and numbered
output names prevent collisions between files with the same basename.

Use `CHECK_ONLY=1 IMAGES="$PWD/my_images" bash slurm/demo/run_image_compare.sh`
to check paths and options without Python, a GPU, or submitting a job. Decoding
and checkpoint compatibility are checked inside the job. Corrupt inputs fail the
run, with details in `run.json`; they are not silently skipped.

The defaults match the video comparison:

- Student: `results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/00029.pt`.
- Baseline: `models/multiHMR/multiHMR_672_L_anny.pt`.
- Student correction: `STUDENT_BONE_FIX=auto`, reading the run's `student_fix_arg.txt`.
- FOV: `55` degrees along the longer image side; panel size: `672`.
- CUDA fp32 with TF32; one model loaded at a time.

Override `STUDENT`, `BASELINE`, `STUDENT_BONE_FIX`, `FOV`, `PANEL_SIZE`, `RUN`,
`OUTPUT`, `REPO` or `CONTAINER_IMAGE` as needed. Relative paths resolve against
`REPO`, default `/netscratch/najib/multi-hmr`. The launcher uses the existing
container/environment/EGL preflight from the video workflow. No landmark-detector
environment is needed. It does not install Python packages. See
[video setup](video_inference.md) for the regular environment dependencies.

## Output

`results/image_compare/images_JOBID/` contains:

- `comparison/`: labelled three-panel PNGs for your presentation.
- `student/` and `baseline/`: separate labelled overlays.
- `input/`: unlabelled letterboxed input panels.
- `run.json`: completion status, source paths/hashes, checkpoints, camera settings,
  correction metadata, detection counts, boxes, and individual forward timings.

Existing output directories are refused. Inspect `status: completed` before
using results. Missing detections remain visible as zero-detection panels; the
models may detect different people. This is a qualitative comparison, with no
ground-truth matching or PVE calculation. Individual image timings have no warm-up
benchmark and should not replace the paired efficiency measurements. Assumed FOV
can affect placement; the same assumption is used for both models.

Direct execution in an environment with models, OpenCV and EGL available:

```bash
PYOPENGL_PLATFORM=egl python demo_image_compare.py \
  --images my_images \
  --student-checkpoint /absolute/path/to/student/00029.pt \
  --student-bone-fix auto \
  --baseline-checkpoint models/multiHMR/multiHMR_672_L_anny.pt \
  --output results/image_compare/my_new_comparison
```
