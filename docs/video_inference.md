# Video inference with the thesis Student and Multi-HMR-Anny

`demo_video.py` applies the existing image models to a video frame by frame. The
default comparison has three panels: **input | Student (ViT-B) | NAVER's released
Multi-HMR-Anny**. Set `TEACHER` to add Final (ViT-L) as a fourth panel. This is an
inference comparison; it does not train a temporal model or change the thesis
choice of student.

The script accepts any clear single-person video; no particular clip is built
in, and none has been supplied yet. The commands below use `VIDEO`, which must
identify a real local file on the cluster. No download or sample clip is
assumed.

## Check the inputs, then submit

Run from `/netscratch/najib/multi-hmr`. Set `VIDEO` to your uploaded clip before
running these commands; `/netscratch/najib/videos/walking.mp4` is an example
filename, not a file included with this repository.

```bash
export VIDEO=/netscratch/najib/videos/walking.mp4
CHECK_ONLY=1 bash slurm/demo/run_video_compare.sh
sbatch slurm/demo/run_video_compare.sh
```

`CHECK_ONLY` checks readable input/checkpoint/correction files, numeric options,
and a fresh output path. It needs neither a GPU nor Python imports, installs
nothing, and submits nothing. It does not decode the video, load checkpoint
tensors, or establish that EGL is working. A successful check is a preflight,
not a successful inference run.

The wrapper uses the regular `multihmr` conda environment inside the existing
container, through `slurm/env.sh`. It targets the A100, L40S and RTX A6000 queues.
Unset `PY_VENV`; CUDA venv overlays and other conda environments are refused.
The wrapper sources `slurm/egl.sh`, as the working image demos do, to provision
native EGL libraries inside the job container and register NVIDIA's graphics
driver. The vendor JSON is written to a job-local directory under `/tmp`, so
registration does not require writing to `/usr/share`. `EGL_INSTALL_PYTHON=0`
skips the helper's Python package installation; the conda environment is kept
unchanged.

Before loading any checkpoint, `scripts/video/check_renderer.py` draws a tiny
test mesh and prints graphics diagnostics. A failed check stops the job with
the original exception. Model inference starts only after rendering succeeds.
`CHECK_ONLY=1` remains a login-node file check and does not perform this test.

The video decoder/writer uses OpenCV. If it is missing from the activated normal
environment, the optional video prerequisite can be installed manually once:

```bash
python -m pip install -r requirements-video.txt
```

This pins `opencv-python-headless==4.8.1.78`; it does not change the CUDA or
PyTorch environment. The Slurm job never runs that installation command.

Create the cluster's existing `logs/` directory before submitting if it is not
already present: Slurm must open its log files before the script starts.

## Which checkpoint and bone correction are used

The default Student is the confirmed 30-epoch ViT-B run:

```text
results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/00029.pt
```

The wrapper does not guess a later checkpoint or treat an unfinished run as
complete. Use `STUDENT` to select another verified checkpoint. The NAVER baseline
defaults to `models/multiHMR/multiHMR_672_L_anny.pt`; it is the original released
Anny model, rather than our Final teacher.

`STUDENT_BONE_FIX=auto` is the default. For an exp8 checkpoint under
`RUN/train/checkpoints/`, it reads `RUN/student_fix_arg.txt`. This file contains
one absolute path to the student's own estimated correction JSON, or `none` when no
correction was selected. The JSON replaces the correction stored in the
checkpoint; corrections are not stacked.

This matters because the saved training checkpoint carries the teacher's
training correction. exp8 estimates the student's own correction after training
without rewriting that checkpoint. If the sidecar or its JSON is missing, auto
mode stops. Copy the run metadata with the checkpoint when moving a run. The
following explicit overrides are available:

| Value | Meaning |
|---|---|
| `STUDENT_BONE_FIX=auto` | Use the completed run's sidecar and independently estimated student correction. |
| `STUDENT_BONE_FIX=checkpoint` | Explicitly inherit the checkpoint's saved training correction. |
| `STUDENT_BONE_FIX=none` | Explicitly disable the correction. |
| `STUDENT_BONE_FIX=/absolute/path/to/bone_fix.json` | Use that correction instead. |

`checkpoint` and `none` are diagnostic choices; record them when comparing with
reported Student results. For a copied checkpoint, use an explicit verified JSON
path if the original run layout is unavailable.

## Options

All relative paths below are resolved against `REPO` (default
`/netscratch/najib/multi-hmr`). Pass the variables to the wrapper; it forwards
arguments safely as a Bash array, including paths containing spaces.

| Variable | Default | Purpose |
|---|---|---|
| `VIDEO` | required | Existing local input video. |
| `STUDENT` | 30-epoch checkpoint above | Student checkpoint. |
| `BASELINE` | `models/multiHMR/multiHMR_672_L_anny.pt` | NAVER's original Anny checkpoint. |
| `STUDENT_BONE_FIX` | `auto` | Correction selection described above. |
| `TEACHER` | empty | Optional Final checkpoint, adding a fourth panel. |
| `RUN` | `video_<Slurm job id>` | Distinct output name. |
| `OUTPUT` | `results/video_compare/$RUN` | New output directory; an existing path is refused. |
| `FOV` | `55` | Assumed field of view along the longer image side, in degrees. |
| `MAX_FRAMES` | `0` | Limit processed frames; zero means all selected frames. |
| `STRIDE` | `1` | Process every Nth source frame. |
| `START_FRAME` | `0` | First source frame to consider. |
| `PANEL_SIZE` | `672` | Size of each square comparison panel; even and at least 128. |
| `DEVICE` | `cuda` | PyTorch device. |
| `AMP` | `0` | Set `1` to opt into mixed precision; requires CUDA. |
| `STUDENT_ONLY` | `0` | Set `1` for input and Student only; cannot combine with `TEACHER`. |
| `CHECK_ONLY` | `0` | Set `1` for login-node path/option checks. |
| `REPO` | `/netscratch/najib/multi-hmr` | Checkout root. |
| `CONTAINER_IMAGE` | `/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh` | Existing cluster container. |

For a short first run using the same `VIDEO`:

```bash
MAX_FRAMES=60 RUN=video_student_baseline_first60 sbatch slurm/demo/run_video_compare.sh
```

To add the confirmed Final teacher as the fourth panel:

```bash
TEACHER=models/multiHMR/exp7_v5_final_00009.pt RUN=video_with_final_teacher sbatch slurm/demo/run_video_compare.sh
```

Choose a new `RUN` before repeating either command. The named checkpoints and
container must exist on the cluster; they are not committed in Git.

For direct execution in an already activated normal environment:

```bash
PYOPENGL_PLATFORM=egl python demo_video.py \
  --video "$VIDEO" \
  --student-checkpoint results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/00029.pt \
  --baseline-checkpoint models/multiHMR/multiHMR_672_L_anny.pt \
  --student-bone-fix auto \
  --output results/video_compare/direct_student_baseline
```

## Outputs

Each run writes into the new `OUTPUT` directory:

| File or directory | Contents |
|---|---|
| `comparison.mp4` | Input, Student, NAVER baseline; optional Final teacher as the fourth panel. |
| `input.mp4` | The selected input frames on their own. |
| `student.mp4`, `baseline.mp4` | Each model's overlay video. |
| `teacher.mp4` | Teacher overlay, only when requested. |
| `run.json` | Checkpoints, camera/frame options, correction selection, run status and other provenance. |
| `student/frames.csv`, `baseline/frames.csv` | Selected source-frame indices/timestamps, tracking and per-frame measurements. |
| `student/predictions.npz`, `baseline/predictions.npz` | Saved prediction arrays for later analysis. |
| `student/summary.json`, `baseline/summary.json` | Per-model tracking and raw temporal summaries. |
| `teacher/` | The same CSV, NPZ and summary outputs for the optional teacher. |

In `STUDENT_ONLY=1` mode, the comparison is input and Student, and baseline and
teacher outputs are omitted. A title/status strip sits **above** each square
image: each individual video is `PANEL_SIZE` wide and `PANEL_SIZE + 64` high.
Missing or lost detections appear in the status rather than being concealed by
holding the last mesh.

Output FPS is source FPS divided by `STRIDE`. The original selected-frame
timestamps remain in the CSV, so skipping frames preserves their playback
timing for a constant-frame-rate input. Timing assumes constant frame rate;
convert a variable-frame-rate phone recording to constant frame rate first.
The videos contain no audio. Keep `run.json` and the per-model files with
the videos when sharing a result; the montage alone cannot establish the
tracking coverage or exact settings. Inspect `run.json` for completion status
before using a partially written run.

Start with `comparison.mp4`, then use the CSV timestamps to inspect moments
where the predicted body jumps, trembles or changes proportions. The summaries
and CSV provide these measurements:

| Measurement | Meaning |
|---|---|
| Root displacement and speed | How far the predicted pelvis moves between selected frames, in metres and metres/second. |
| Pelvis-relative joint speed | How quickly joints move relative to the pelvis, in metres/second; removes whole-body translation. |
| Pelvis-relative joint acceleration | How suddenly joint movement changes, in metres/second squared; spikes can help locate jitter. |
| Shape value standard deviation and absolute steps | How much each predicted body parameter varies over the clip and between adjacent frames, in its own parameter units. |
| Missing frames and track segments | How often no suitable person is matched and when tracking starts again. |

Joint summaries pool the model's skeleton joints, excluding the pelvis itself;
the joint names are recorded with the report. Temporal differences stop at
missing frames and track changes. Shape mean and standard deviation pool
selected frames across segments, so inspect identity before interpreting them
if another person enters the scene. Shape values are model parameters, not
measurements of the person's real age or body dimensions. Natural movement and
camera movement also raise speed and acceleration, so these values locate
frames to inspect; they do not automatically classify motion as an error.

## What the comparison can establish

Each model sees the same selected source frames with the same full-frame
letterbox and camera assumption. The overlays use predicted placement. There is
no detector-guided crop, detection-box placement, temporal smoothing or forced
rest-pose hands. Finger poses therefore show the models' predictions. A
single-person tracker associates detections across frames; missing detections
and track breaks must remain visible in the measurements.

Without calibrated camera information, `FOV=55` is an assumption rather than
ground truth. Single-view size and distance remain ambiguous. Person tracking
helps compare a sequence but adds no motion understanding to the image models.
Large camera motion, cuts, occlusion and multiple people can affect association.

Raw frame-to-frame changes are temporal proxies, not reconstruction accuracy.
A model can produce a very steady wrong body. Without video ground-truth meshes
and cameras, the run cannot report PVE, MPJPE or prove that either model is more
accurate. Do not compare a smoothed output against an unsmoothed baseline and
attribute the difference to the network. Report the clip, frame selection,
camera assumption, checkpoint/correction choices and any missing-track counts
alongside qualitative examples.

The confirmed 30-epoch Student is smaller and faster than Final but missed the
95% PVE/MPJPE retention bars on both existing test sets. A plausible-looking
video does not change that result. This comparison evaluates inference from the
selected checkpoint; the thesis reference model and acceptance criteria still
follow the supervisor's decision.
