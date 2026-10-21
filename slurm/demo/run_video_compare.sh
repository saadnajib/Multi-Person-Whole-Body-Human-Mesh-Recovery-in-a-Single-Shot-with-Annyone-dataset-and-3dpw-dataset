#!/bin/bash
# Frame-by-frame video inference: input | ViT-B Student | NAVER Multi-HMR-Anny.
# Uses regular multihmr and the same native EGL setup as the image demos.
# Python packages are not installed or upgraded inside the job.
# CHECK_ONLY=1 validates paths/options on the login node without Python, a GPU or submission.
#SBATCH --partition=A100-40GB,A100-80GB,A100-PCI,A100-RP,L40S,L40S-AV,RTXA6000,RTXA6000-AV
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=video-compare
#SBATCH --output=/netscratch/najib/multi-hmr/logs/video-compare-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/video-compare-%j.err

set -euo pipefail

# This branch runs inside the existing container, with arguments forwarded as an array.
# It never reconstructs the Python command from interpolated shell text.
if [ "${1:-}" = --inside ]; then
  shift
  export PROJECT_ROOT="${1:?missing project root}"
  shift
  export CONDA_ENV=multihmr
  unset PY_VENV
  set +u                         # conda activation scripts can read unset variables
  source "$PROJECT_ROOT/slurm/env.sh"
  set -u
  export PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 NVIDIA_DRIVER_CAPABILITIES=all
  # The image demos provision native EGL libraries AND register the NVIDIA
  # vendor. Environment flags alone do not perform that graphics setup.
  export EGL_VENDOR_DIR
  EGL_VENDOR_DIR="$(mktemp -d /tmp/multihmr-video-egl.XXXXXX)"
  export EGL_INSTALL_PYTHON=0
  source "$PROJECT_ROOT/slurm/egl.sh"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_DIR/10_nvidia.json"
  # Fail with the actual graphics error before spending time loading weights.
  python -u "$PROJECT_ROOT/scripts/video/check_renderer.py"
  exec python -u "$PROJECT_ROOT/demo_video.py" "$@"
fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
absolute() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO" "$1" ;;
  esac
}
require_file() { [ -r "$2" ] && [ -f "$2" ] || fail "$1 is not a readable file: $2"; }
nonnegative() { [[ "$2" =~ ^[0-9]+$ ]] || fail "$1 must be a non-negative integer: $2"; }
positive() { [[ "$2" =~ ^[0-9]+$ ]] && (( 10#$2 > 0 )) || fail "$1 must be a positive integer: $2"; }
toggle() { [[ "$2" = 0 || "$2" = 1 ]] || fail "$1 must be 0 or 1: $2"; }

[ "$#" -eq 0 ] || fail "configure this wrapper with environment variables; see docs/video_inference.md"
REPO="${REPO:-/netscratch/najib/multi-hmr}"
[ -d "$REPO" ] || fail "REPO does not exist: $REPO"
REPO="$(cd "$REPO" && pwd -P)"
SELF="$REPO/slurm/demo/run_video_compare.sh"
VIDEO="${VIDEO:-}"
[ -n "$VIDEO" ] || fail "VIDEO is required: set it to an existing local video file"
VIDEO="$(absolute "$VIDEO")"
STUDENT="$(absolute "${STUDENT:-results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/00029.pt}")"
BASELINE="$(absolute "${BASELINE:-models/multiHMR/multiHMR_672_L_anny.pt}")"
TEACHER="${TEACHER:-}"
[ -z "$TEACHER" ] || TEACHER="$(absolute "$TEACHER")"
STUDENT_BONE_FIX="${STUDENT_BONE_FIX:-auto}"
FOV="${FOV:-55}"
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-1}"
START_FRAME="${START_FRAME:-0}"
PANEL_SIZE="${PANEL_SIZE:-672}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-0}"
STUDENT_ONLY="${STUDENT_ONLY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh}"
RUN="${RUN:-video_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}}"
[[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "RUN must contain letters, digits, underscores, dots or hyphens"
OUTPUT="$(absolute "${OUTPUT:-results/video_compare/$RUN}")"

require_file VIDEO "$VIDEO"
require_file STUDENT "$STUDENT"
require_file entrypoint "$REPO/demo_video.py"
require_file environment "$REPO/slurm/env.sh"
require_file "EGL setup" "$REPO/slurm/egl.sh"
require_file "renderer preflight" "$REPO/scripts/video/check_renderer.py"
nonnegative MAX_FRAMES "$MAX_FRAMES"
nonnegative START_FRAME "$START_FRAME"
positive STRIDE "$STRIDE"
positive PANEL_SIZE "$PANEL_SIZE"
(( 10#$PANEL_SIZE >= 128 && 10#$PANEL_SIZE % 2 == 0 )) || fail "PANEL_SIZE must be an even integer of at least 128"
toggle AMP "$AMP"
toggle STUDENT_ONLY "$STUDENT_ONLY"
toggle CHECK_ONLY "$CHECK_ONLY"
[[ "$FOV" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || fail "FOV must be a number between 0 and 180"
awk -v fov="$FOV" 'BEGIN { exit !(fov > 0 && fov < 180) }' || fail "FOV must be between 0 and 180"
[[ "$DEVICE" = cuda || "$DEVICE" = cpu || "$DEVICE" =~ ^cuda:[0-9]+$ ]] || fail "DEVICE must be cuda, cuda:N or cpu"
[ "$AMP" = 0 ] || [ "$DEVICE" != cpu ] || fail "AMP=1 requires a CUDA device"
[ "${CONDA_ENV:-multihmr}" = multihmr ] || fail "use the regular multihmr environment: unset CONDA_ENV or set it to multihmr"
[ -z "${PY_VENV:-}" ] || fail "unset PY_VENV: this wrapper uses regular multihmr without a CUDA venv overlay"
[ ! -e "$OUTPUT" ] || fail "OUTPUT already exists; choose a new RUN or OUTPUT: $OUTPUT"
if [ "$STUDENT_ONLY" = 0 ]; then require_file BASELINE "$BASELINE"; fi
[ -z "$TEACHER" ] || require_file TEACHER "$TEACHER"
[ "$STUDENT_ONLY" = 0 ] || [ -z "$TEACHER" ] || fail "TEACHER and STUDENT_ONLY=1 cannot be combined"

# Training checkpoints retain the teacher's training correction. exp8 writes the
# student's independently estimated correction to this sidecar after training.
case "$STUDENT_BONE_FIX" in
  auto)
    STUDENT_RUN="$(dirname "$(dirname "$(dirname "$STUDENT")")")"
    FIX_SIDECAR="$STUDENT_RUN/student_fix_arg.txt"
    require_file "student correction sidecar (use STUDENT_BONE_FIX=checkpoint only to explicitly use the saved training correction)" "$FIX_SIDECAR"
    mapfile -t FIX_LINES < "$FIX_SIDECAR"
    [ "${#FIX_LINES[@]}" -eq 1 ] || fail "student correction sidecar must contain exactly one line: $FIX_SIDECAR"
    FIX_VALUE="${FIX_LINES[0]%$'\r'}"
    FIX_VALUE="${FIX_VALUE#"${FIX_VALUE%%[![:space:]]*}"}"
    FIX_VALUE="${FIX_VALUE%"${FIX_VALUE##*[![:space:]]}"}"
    [ -n "$FIX_VALUE" ] || fail "student correction sidecar is empty: $FIX_SIDECAR"
    if [ "$FIX_VALUE" != none ]; then
      [[ "$FIX_VALUE" = /* ]] || fail "student correction sidecar must contain an absolute JSON path or none: $FIX_SIDECAR"
      require_file "student correction from sidecar" "$FIX_VALUE"
    fi
    ;;
  checkpoint|none) ;;
  *) STUDENT_BONE_FIX="$(absolute "$STUDENT_BONE_FIX")"; require_file STUDENT_BONE_FIX "$STUDENT_BONE_FIX" ;;
esac

ARGS=(--video "$VIDEO" --student-checkpoint "$STUDENT"
      --baseline-checkpoint "$BASELINE" --student-bone-fix "$STUDENT_BONE_FIX"
      --output "$OUTPUT" --fov "$FOV" --max-frames "$MAX_FRAMES"
      --stride "$STRIDE" --start-frame "$START_FRAME" --panel-size "$PANEL_SIZE"
      --device "$DEVICE")
[ -z "$TEACHER" ] || ARGS+=(--teacher-checkpoint "$TEACHER")
[ "$AMP" = 0 ] || ARGS+=(--amp)
[ "$STUDENT_ONLY" = 0 ] || ARGS+=(--student-only)
printf 'Student: %s\nVideo: %s\nOutput: %s\nStudent correction: %s\n' "$STUDENT" "$VIDEO" "$OUTPUT" "$STUDENT_BONE_FIX"
printf 'Command: '
printf '%q ' python -u "$REPO/demo_video.py" "${ARGS[@]}"
printf '\n'
if [ "$CHECK_ONLY" = 1 ]; then
  printf 'CHECK_ONLY=1: paths and options passed; no GPU, Python imports, installation or submission.\n'
  exit 0
fi

[ -n "${SLURM_JOB_ID:-}" ] || fail "submit with sbatch; use CHECK_ONLY=1 bash $SELF for login-node validation"
require_file CONTAINER_IMAGE "$CONTAINER_IMAGE"
command -v srun >/dev/null || fail "srun is unavailable"
export PROJECT_ROOT="$REPO" CONDA_ENV=multihmr
export NVIDIA_DRIVER_CAPABILITIES=all PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
unset PY_VENV
MOUNTS="/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm"
[ ! -d /dev/dri ] || MOUNTS+=",/dev/dri:/dev/dri"
exec srun -K --gpus=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir="$REPO" \
  bash "$SELF" --inside "$REPO" "${ARGS[@]}"
