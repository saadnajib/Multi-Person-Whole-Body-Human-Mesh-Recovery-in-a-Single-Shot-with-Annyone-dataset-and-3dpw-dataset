#!/bin/bash
# =============================================================================
# DEMO COMPARISON OF THE FINAL MODELS (same images, identical settings)
# =============================================================================
#   teacher v5        ViT-L, synthetic only          3DPW-test 119.1 / 68.8
#   teacher v9        ViT-L, + real data at the end   3DPW-test 102.5 / 62.7
#   v10               ViT-L, fresh, real from start   3DPW-test 111.5 / 76.2
#   student v6c       ViT-S, synthetic only           3DPW-test 139.5 / 88.0
#   student v8 final  ViT-S, + real data              3DPW-test 123.6 / 81.2
# Two image sets: real photos (example_data) and frames from 3DPW TEST
# sequences (never used in training), built automatically on the first run in
# demo_images/3dpw_test/. Each model x set goes to demo_out_<run>_<epoch>_<set>/.
# A model whose checkpoint is missing is skipped; the others still run.
# Then:  python scripts/analysis/make_montage.py ...   (see the end of the log)
# =============================================================================
#SBATCH --partition=A100-40GB,A100-80GB,RTXA6000-AV,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --job-name=demo-compare
#SBATCH --output=/netscratch/najib/multi-hmr/logs/demo-compare-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/demo-compare-%j.err

set -euo pipefail

mkdir -p /netscratch/najib/multi-hmr/logs

export NVIDIA_DRIVER_CAPABILITIES=all

DRI_MOUNT=""
[ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

COMMAND=$(cat <<'EOF'
set -euo pipefail

export PROJECT_ROOT="/netscratch/najib/multi-hmr"
export CONDA_SH="/netscratch/najib/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"

echo "======================================================"
echo "1. Installing Native EGL Graphics Routing..."
echo "======================================================"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2

mkdir -p /usr/share/glvnd/egl_vendor.d
cat << 'JSON' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}
JSON
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d

echo "======================================================"
echo "2. Activating Env..."
echo "======================================================"
conda activate multihmr
pip install --quiet --exists-action i pyglet pyopengl

# torchvision is needed for --use_person_detector 1 (Faster R-CNN).
# Usually already installed with torch; this is a no-op safeguard.
python -c "import torchvision" 2>/dev/null || pip install --quiet torchvision

export TORCH_HOME=/netscratch/najib/torch_cache

# Route anny's hardcoded ~/.cache/anny onto netscratch (avoids disk-quota error)
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "$XDG_CACHE_HOME" /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

echo "======================================================"
echo "2b. Patching DINOv2 for Python 3.9 compatibility..."
echo "======================================================"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} + || true

echo "======================================================"
echo "2c. Checkpoints + 3. Running the demo on every model"
echo "======================================================"
cd "$PROJECT_ROOT"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

grep -q "feat_proj" model.py || { echo "ERROR: model.py lacks the v3 patch (python apply_student_fix_v3.py)"; exit 1; }

# >>> EDIT HERE: models ("run_name|preferred_epoch") and image sets ("folder|tag|fov") <<<
MODELS=(
  "anny_s2_shape_v5|00099"
  "anny_teacher_v9_mixed|00062"
  "anny_v10_scratch_mixed|00299"
  "anny_distill_vits_v6c|00149"
  "anny_distill_vits_v8_final|00052"
  # "anny_distill_vits_v7_mixed|00130"
)
# 3DPW frames are 1080x1920 portrait; fy~1969 over the long side -> ~52 deg FOV.
IMG_SETS=(
  "$PROJECT_ROOT/example_data|photos|55"
  "$PROJECT_ROOT/demo_images/3dpw_test|3dpwtest|52"
)
MODELS_DIR="$PROJECT_ROOT/models/multiHMR"
mkdir -p "$MODELS_DIR"

# ---- build the 3DPW-test demo set once: the middle frame of 8 test sequences
T3="$PROJECT_ROOT/demo_images/3dpw_test"
if [ ! -d "$T3" ] || [ -z "$(ls -A "$T3" 2>/dev/null)" ]; then
  mkdir -p "$T3"
  for seq in downtown_walking_00 outdoors_fencing_01 office_phoneCall_00 flat_guitar_01 \
             downtown_bar_00 downtown_runForBus_00 downtown_weeklyMarket_00 downtown_sitOnStairs_00; do
    d="$PROJECT_ROOT/data/3DPW/imageFiles/$seq"
    [ -d "$d" ] || { echo "  (no images for $seq, skipped)"; continue; }
    n=$(ls -1 "$d"/image_*.jpg | wc -l); mid=$(ls -1 "$d"/image_*.jpg | sed -n "$(( n / 2 + 1 ))p")
    cp "$mid" "$T3/${seq}__$(basename "$mid")"
  done
  echo "built $T3: $(ls -1 "$T3" | wc -l) frames"
fi

# resolve_ckpt RUN EPOCH -> prints the model name (run_epoch) whose .pt is in $MODELS_DIR, or nothing
resolve_ckpt() {
  local run="$1" ep="$2" logs="$PROJECT_ROOT/logs/anny_model/$1/checkpoints"
  if [ -f "$MODELS_DIR/${run}_${ep}.pt" ]; then echo "${run}_${ep}"; return; fi
  if [ -f "$logs/${ep}.pt" ]; then cp "$logs/${ep}.pt" "$MODELS_DIR/${run}_${ep}.pt"; echo "${run}_${ep}"; return; fi
  local last; last=$(ls -1 "$logs"/*.pt 2>/dev/null | sort | tail -1 || true)
  if [ -n "$last" ]; then
    local e; e=$(basename "$last" .pt)
    cp "$last" "$MODELS_DIR/${run}_${e}.pt"
    echo "WARNING: ${run} epoch ${ep} not found; using the latest on disk (${e})" >&2
    echo "${run}_${e}"; return
  fi
}

OK=(); FAILED=(); NAMES=()
for entry in "${MODELS[@]}"; do
  run="${entry%%|*}"; ep="${entry##*|}"
  echo ""; echo "################ MODEL: ${run} (preferred epoch ${ep}) ################"
  name=$(resolve_ckpt "$run" "$ep")
  if [ -z "$name" ]; then echo "SKIPPED: no checkpoint for ${run}"; FAILED+=("$run (no checkpoint)"); continue; fi
  NAMES+=("$name")
  for set in "${IMG_SETS[@]}"; do
    IFS='|' read -r folder tag fov <<< "$set"
    [ -d "$folder" ] || { echo "  image folder missing: $folder"; FAILED+=("$name/$tag (no images)"); continue; }
    echo "--- $name on $tag ($folder, fov $fov)"
    if python3.9 demo.py \
        --model_name "$name" \
        --img_folder "$folder" \
        --out_folder "demo_out_${name}_${tag}" \
        --alpha 0.9 \
        --fov "$fov" \
        --use_person_detector 1 \
        --detector_thresh 0.8 \
        --crop_margin 0.35 \
        --anchor_to_box 1 \
        --max_persons 5 \
        --det_thresh 0.15 \
        --nms_kernel_size 5 \
        --iou_thresh 0.45 \
        --extra_views 0 \
        --save_rotating_video 0 \
        --person_frac 0.22 \
        --save_mesh 1 \
        --mask_helper_joints 1 \
        --relax_hands 1 \
        --rest_pose 0; then
      OK+=("${name}_${tag}")
    else
      FAILED+=("${name}_${tag} (demo.py failed)")
    fi
  done
done

echo ""
echo "======================================================"
echo "SUMMARY"
echo "======================================================"
for n in "${OK[@]}"; do echo "  ok      demo_out_${n}/"; done
for n in "${FAILED[@]}"; do echo "  FAILED  ${n}"; done
echo ""
echo "montage commands (run on the login node):"
for set in "${IMG_SETS[@]}"; do
  IFS='|' read -r folder tag fov <<< "$set"
  echo "  python scripts/analysis/make_montage.py --out demo_montage_${tag} --folders \\"
  for n in "${NAMES[@]}"; do echo "      demo_out_${n}_${tag} \\"; done
  echo "      --labels 'teacher v5' 'teacher v9 (+real)' 'v10 (real from start)' 'student v6c' 'student v8 (+real)'"
done
[ ${#FAILED[@]} -eq 0 ] || exit 1
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"