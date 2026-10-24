# Video inference with the thesis Student and Multi-HMR-Anny

`demo_video.py` runs the thesis Student and optional reference models on the
same video. With `TEMPORAL=off` (the default), the comparison is **input | raw
Student (ViT-B) | NAVER's released Multi-HMR-Anny**. Set `TEACHER` to add Final
(ViT-L). With `TEMPORAL=ema`, an additional Student output combines information
across past frames to stabilize pose, shape and placement.
`ALIGNMENT=landmarks` adds a further Student output whose camera-space position
and depth are fitted to independently detected image landmarks. It can be used
with either raw or temporally filtered predictions.

This is **inference-time temporal filtering** of the existing Student. Its
image encoder still processes one frame at a time; no learned video network is
trained, and the checkpoint is unchanged. The filter gives the final output
memory of previous predictions. Both raw and filtered results are retained so
the effect can be inspected separately from network accuracy.

The script accepts any clear single-person video. For the existing
`man_walking.mp4` in the checkout root, the recommended temporal comparison is:

```bash
cd /netscratch/najib/multi-hmr
VIDEO="$PWD/man_walking.mp4" TEMPORAL=ema STUDENT_ONLY=1 \
  sbatch slurm/demo/run_video_compare.sh
```

This writes input, raw Student and temporally filtered Student panels. Omit
`STUDENT_ONLY=1` to also include the raw released baseline. Keep its raw label
when presenting the comparison: a filtering benefit is not evidence that the
Student network is more accurate than that baseline. The dedicated
`temporal_comparison.mp4` always compares the raw and filtered Student on the
same selected frames.

## Improve image alignment with detected landmarks

Alignment uses a separate CPU landmark detector before the GPU inference run.
Install its dependencies in a **separate Python 3.10 environment**, keeping the
regular `multihmr` environment used for training and mesh inference unchanged:

```bash
cd /netscratch/najib/multi-hmr
python3.10 -m venv /netscratch/najib/venvs/video-landmarks
/netscratch/najib/venvs/video-landmarks/bin/python -m pip install \
  -r requirements-video-landmarks.txt
```

The optional requirements pin MediaPipe 0.10.14, NumPy 1.26.4 and OpenCV contrib
4.11.0.86. The default MediaPipe Solutions backend uses the full pose and person
detection models included in its wheel; no separate model download is needed.
It is intended for one clear person. Run the detector on the **original source
video**, before adding mesh overlays, title strips or comparison panels:

```bash
mkdir -p results/video_landmarks
/netscratch/najib/venvs/video-landmarks/bin/python \
  scripts/video/extract_pose_landmarks.py \
  --video "$PWD/man_walking.mp4" \
  --output "$PWD/results/video_landmarks/man_walking.json"
```

The JSON contains original frame indices, image-pixel landmarks, confidence,
detector metadata and the source video's SHA-256 digest. It refuses to overwrite
an existing file; choose another output name when extracting again. Inference
checks the digest and video dimensions/timing, so use the exact same source
file for extraction and inference. A re-encoded or cropped copy is a different
input. Extract all frames needed by the inference run, including any chosen
start frame or stride.

Then submit the stabilized and aligned Student comparison:

```bash
VIDEO="$PWD/man_walking.mp4" TEMPORAL=ema ALIGNMENT=landmarks \
LANDMARKS="$PWD/results/video_landmarks/man_walking.json" STUDENT_ONLY=1 \
  sbatch slurm/demo/run_video_compare.sh
```

Add `CHECK_ONLY=1` and replace `sbatch` with `bash` for the login-node file and
option checks. This check reads no landmark JSON and runs no detector; schema
and source-video validation happen when Python starts. No package installation
or landmark extraction occurs inside the GPU job.

Inspect `alignment_comparison.mp4`: **input | before alignment | aligned
Student**. The middle panel is the temporal Student when `TEMPORAL=ema`, or the
raw Student when temporal filtering is off. The main comparison also retains
the raw and optional temporal panels.

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
| `WARMUP_FRAMES` | `10` | Exclude the first N processed frames from speed averages; keep every frame in the videos and motion measurements. |
| `TEMPORAL` | `off` | `ema` enables causal Student filtering and extra raw/filtered comparison outputs. |
| `TEMPORAL_POSE_TAU` | `0.08` | Pose smoothing time constant in seconds; `0` passes rotations through. |
| `TEMPORAL_TRANSLATION_TAU` | `0.12` | Camera-space head-anchor smoothing time constant in seconds; `0` passes placement through. |
| `TEMPORAL_SHAPE_TAU` | `1.0` | Shape smoothing time constant in seconds; `0` passes shape through. |
| `TEMPORAL_MAX_GAP` | `0.25` | Reset filter when consecutive selected timestamps are farther apart than this many seconds; must be positive. |
| `ALIGNMENT` | `off` | `landmarks` enables Student position/depth fitting to independent 2D landmarks. |
| `LANDMARKS` | empty | Existing landmark JSON from the exact original video; required with `ALIGNMENT=landmarks`. Relative paths resolve against `REPO`. |
| `ALIGNMENT_DEPTH_TAU` | `0.35` | Alignment depth smoothing time constant, seconds; `0` restores independent per-frame fits. |
| `ALIGNMENT_MIN_CONFIDENCE` | `0.5` | Minimum accepted landmark confidence, from `0` to `1` inclusive. |
| `STRIDE` | `1` | Process every Nth source frame. |
| `START_FRAME` | `0` | First source frame to consider. |
| `PANEL_SIZE` | `672` | Size of each square comparison panel; even and at least 128. |
| `DEVICE` | `cuda` | PyTorch device. |
| `AMP` | `0` | Set `1` to opt into mixed precision for raw inference; requires CUDA and `TEMPORAL=off`. |
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

## How temporal filtering works

With `TEMPORAL=ema`, use `AMP=0` (the default): temporal Anny re-decoding is
validated in fp32, with TF32 enabled on CUDA as in the raw default. Mixed
precision is rejected for temporal runs until equivalent decoding is verified.
The tracker selects the person from the raw Student predictions. The filter updates three kinds of state for that person:

- **Pose:** interpolate each bone rotation toward the new prediction using
  shortest-arc quaternion interpolation. This keeps rotations valid and avoids
  averaging a +179-degree rotation with a -179-degree rotation into zero.
- **Shape:** average the already predicted Anny phenotype values, using a
  slower time constant so the same person's proportions vary less rapidly.
- **Placement:** average the camera-space **head anchor**, which is the point
  used by the model to position the mesh. Smoothing a pelvis while placing by a
  head would introduce a mismatch.

For translation and shape, the update is
`filtered = (1 - alpha) * previous_filtered + alpha * current_prediction`, with
`alpha = 1 - exp(-dt / tau)`. Pose uses the same alpha in spherical rotation
interpolation. `dt` comes from the selected source-frame timestamps, so the
strength is expressed in seconds rather than a fixed number of frames. A zero
time constant disables smoothing of that component. Larger constants give
stronger smoothing and more lag, and may suppress genuine rapid movement.
They are initial settings to inspect on the video, not tuned performance claims.

The filtered rotations, shape and head anchor are decoded again through the
loaded Anny body model, producing one geometrically consistent mesh and joint
set. The implementation does not average mesh vertices independently. It does
not reapply sigmoid, helper masking or the checkpoint's rotation correction to
the already corrected outputs. Raw and filtered panels reuse **one Student
network forward pass** per image; the filtered branch adds filtering and Anny
decoding work.

State resets on a missing detection, a new track segment, a discontinuity in
processed frame indices, non-increasing timestamps, or a timestamp gap greater
than `TEMPORAL_MAX_GAP`. After a reset, the first valid prediction initializes
the new state. Missing detections remain missing; the previous mesh is never
held over an empty frame. If a filtered mesh fails the geometry/visibility
checks, that frame falls back to its actual raw prediction and clears the
filter history; `temporal_state` and `fallback_raw_frames` identify these
cases. A decoder-convention mismatch fails the run instead of presenting an
unverified filtered body. If `STRIDE / source_fps` exceeds the maximum gap, every
selected frame resets the filter, so choose a suitable gap when using a large
stride. Geometric tracking still cannot guarantee identity through crossings or
scene cuts. Prefer one clear person and inspect resets and identity in the clip.

Temporal filtering can reduce visible jitter, but cannot recover missing image
information, guarantee correct pose, or correct a consistently wrong body
shape. It uses past frames only and requires no future-frame look-ahead; the
smoothing itself can still produce motion lag. A learned temporal model that
extracts motion from multiple images would be a separate training project.

## How landmark alignment works

The independent detector supplies shoulders, elbows, wrists, hips, knees and
ankles in original-image pixels. The video pipeline transforms those points
into the same letterboxed image coordinates used for rendering and associates
them with the tracked Student person. The fitter then adjusts one rigid 3D
translation: horizontal position, vertical position and depth. Moving a body
farther from the camera reduces its projected size. All vertices and joints
receive the same translation, preserving the predicted body shape, limb pose,
bone lengths and inter-point distances.

The fit uses confident landmarks spanning the torso and lower body, a robust
reprojection objective and bounded translation. Missing or weak observations,
inconsistent landmarks, invalid geometry and unsuccessful fits leave that
frame's pre-alignment prediction unchanged. The per-frame diagnostics record
whether a correction was accepted, the reason for skipping, the translation
and the landmark residuals. If temporal filtering is enabled, alignment acts
on its output using the current frame's landmarks; the raw and temporal
branches are still saved separately.

The final aligned pelvis depth is now smoothed across frames (default
`ALIGNMENT_DEPTH_TAU=0.35` seconds). It uses an exponential average of log depth
and limits changes to 0.5 log-depth units per second. This reduces visible
size fluctuations while allowing gradual movement toward or away from the
camera. The pelvis keeps its current image position, and the entire body is
translated rigidly; pose and shape remain unchanged. `ALIGNMENT_DEPTH_TAU=0`
disables both smoothing and the rate limit for comparison.

Depth history resets on a missing student, rejected landmark fit, track change,
frame discontinuity, non-increasing timestamp, or gap above `TEMPORAL_MAX_GAP`
(default 0.25 seconds, also used when pose filtering is off). Unsafe smoothed
geometry falls back to the current accepted unsmoothed fit and resets history.
Resets can still cause a visible size change. Smoothing may introduce lag or
increase individual-frame landmark error. `alignment.json` records the filter
state, fitted/rendered depth, original fit translation/error, and updated
residuals for the actual rendered body. The alignment timing includes this work.

These points are detector estimates, not ground truth. Lower landmark residuals
show closer agreement with that detector in the image, not proven improvement
in 3D reconstruction accuracy. Camera calibration and body proportions remain
ambiguous. Position/depth correction can improve an oversized or shifted mesh,
but cannot repair a wrongly bent arm or leg, recover occluded body details, or
make an incorrect shape estimate anatomically correct. Inspect the before/after
video and skipped-frame diagnostics before using the result in a presentation.

## Outputs

Each run writes into the new `OUTPUT` directory:

| File or directory | Contents |
|---|---|
| `comparison.mp4` | Input, raw Student, optional temporal/aligned Student, raw NAVER baseline; optional raw Final teacher. |
| `input.mp4` | The selected input frames on their own. |
| `student.mp4`, `baseline.mp4` | Each model's raw overlay video. |
| `teacher.mp4` | Teacher overlay, only when requested. |
| `run.json` | Checkpoints, camera/frame options, correction selection, run status and other provenance. |
| `timing.csv` | Student/baseline speed comparison: inference FPS and latency, preprocessing, transfers, mesh rendering, hardware and input size. |
| `student/frames.csv`, `baseline/frames.csv` | Selected source-frame indices/timestamps, tracking and per-frame measurements. |
| `student/predictions.npz`, `baseline/predictions.npz` | Saved prediction arrays for later analysis. |
| `student/summary.json`, `baseline/summary.json` | Per-model tracking and raw temporal summaries. |
| `teacher/` | The same CSV, NPZ and summary outputs for the optional teacher. |
| `student_temporal.mp4` | Filtered Student overlay, only with `TEMPORAL=ema`. |
| `student_temporal/` | Filtered `frames.csv`, `predictions.npz`, and `summary.json`, separate from raw Student records. |
| `temporal_comparison.mp4` | Input, raw Student and filtered Student, only with `TEMPORAL=ema`. |
| `temporal_comparison.json` | Raw/filtered Student motion and shape summaries; these are stability proxies, not ground-truth accuracy. |
| `student_aligned.mp4` | Position/depth-corrected Student overlay, only with `ALIGNMENT=landmarks`. |
| `student_aligned/` | Aligned `frames.csv`, `predictions.npz`, `summary.json`, and per-frame fitting diagnostics in `alignment.json`. |
| `alignment_comparison.mp4` | Input, pre-alignment Student and aligned Student, only with `ALIGNMENT=landmarks`. |

In `STUDENT_ONLY=1` mode, baseline and teacher outputs are omitted; the
comparison is input and raw Student, plus filtered/aligned Student if requested. A title/status strip sits **above** each square
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

## Inference speed and playback speed

`comparison.mp4` shows each raw model's **final average inference FPS** on every
frame. The filtered Student panel instead shows **model + filter/decode FPS**,
including the added temporal work. The input panel separately labels
**playback FPS**. The individual
`student.mp4` and `baseline.mp4` show a running inference average, with the first
`WARMUP_FRAMES` frames marked as timing warm-up. For a clip with no frames left
after warm-up, inference FPS is `N/A` rather than an estimate from startup time.

Inference timing uses batch size one and a wall clock synchronized with the
selected CUDA device before and after the model call. It includes detection,
pose/shape prediction and Anny mesh decoding. It excludes image preprocessing,
input/output transfers outside the model, mesh rendering and video I/O. Both
models run sequentially on the same selected device and selected source frames;
each uses its checkpoint's input resolution, which is recorded along with the
GPU name, precision and PyTorch/CUDA versions in `run.json`. Report those settings
with the FPS: different model input sizes affect the comparison.

FPS is `measured frames / total inference seconds`, equivalently
`1000 / mean inference_ms`. All inference calls after warm-up count, including
missing detections. Detection count can affect runtime. Mesh-rendering latency
is measured separately, only for frames where a mesh is drawn. These are
sequential, batch-one timings from this clip; they do not measure complete
application throughput or maximum batched throughput.

`timing.csv` summarizes average, median and 95th-percentile inference latency
(the slower end of the distribution), plus the other timed stages. Each
model's `summary.json` contains the timing statistics and sample counts.
`frames.csv` records `timing_included`, `preprocess_ms`, `inference_ms`,
`output_transfer_ms` and `render_ms` for every frame, including warm-up frames.
The older `forward_ms` column retains preprocessing + forward + output-transfer
latency; use `inference_ms` for the model-speed comparison. Rendering is included
in neither column. Preprocessing timing includes the model-image resize,
normalization and input transfer; output-transfer timing includes conversion
to NumPy. Video decoding/encoding and comparison assembly are outside these
stage measurements.

For `TEMPORAL=ema`, `temporal_ms` records the added filter, Anny decode and
validation time. The temporal branch's `processing_fps` includes model
inference plus that temporal work, while the raw model `inference_fps` keeps
its original definition. Both exclude image preprocessing, rendering and video
I/O, so neither is end-to-end video throughput. Do not quote the raw model FPS
as the speed of the temporal pipeline. Inspect the saved per-frame values and
stage counts, including misses and warm-up exclusion. The completed historical
job 3642415 predates both this temporal feature and the FPS update; it cannot
provide measured temporal gains or new FPS results.

For `ALIGNMENT=landmarks`, the aligned branch's `processing_fps` includes the
Student forward pass, optional temporal work, and the alignment fit/geometry
checks. **Offline landmark extraction is excluded** from that FPS, as are
preprocessing, rendering and video I/O. The extraction JSON separately records
CPU detector time per frame and its environment metadata. The detector may run
on different hardware, so neither its earlier timing nor the refinement FPS
alone describes complete online video throughput. Keep that distinction when
comparing the raw and corrected panels.

## Movement and shape measurements

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
letterbox and camera assumption. Raw overlays use the model's predicted
placement; temporal Student overlays use the filtered head-anchor placement;
aligned Student overlays additionally use the landmark-fitted translation.
There is no detector-guided crop, detection-box placement or forced rest-pose
hands. The raw finger poses are the network predictions; the temporal branch
filters bone rotations including the fingers. A single-person tracker
associates detections across frames; missing detections and track breaks remain
visible in the measurements.

Without calibrated camera information, `FOV=55` is an assumption rather than
ground truth. Single-view size and distance remain ambiguous. Person tracking
helps compare a sequence. The optional temporal filter adds persistent output
state, but does not teach the image network to understand motion.
Large camera motion, cuts, occlusion and multiple people can affect association.

Raw and filtered frame-to-frame changes are temporal proxies, not reconstruction accuracy.
A model can produce a very steady wrong body. Without video ground-truth meshes
and cameras, the run cannot report PVE, MPJPE or prove that either model is more
accurate. Do not compare a smoothed output against an unsmoothed baseline and
attribute the difference to the network. Report the clip, frame selection,
camera assumption, checkpoint/correction choices and any missing-track counts
alongside qualitative examples. An aligned overlay also depends on an external
landmark detector; attribute any visual improvement to that refinement rather
than to a newly trained Student network.

The confirmed 30-epoch Student is smaller and faster than Final but missed the
95% PVE/MPJPE retention bars on both existing test sets. A plausible-looking
video does not change that result. This comparison evaluates inference from the
selected checkpoint; the thesis reference model and acceptance criteria still
follow the supervisor's decision.
