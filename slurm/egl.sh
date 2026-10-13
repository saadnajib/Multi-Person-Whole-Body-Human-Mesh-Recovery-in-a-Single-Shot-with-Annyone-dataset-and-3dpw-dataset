#!/bin/bash
# EGL setup for headless pyrender inside the container (same steps as run_demo.sh, but NON-FATAL:
# a failure only disables the mesh overlays; a warning is written to stderr).
# Source AFTER slurm/env.sh. Needs the job to run with NVIDIA_DRIVER_CAPABILITIES=all.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2 >/dev/null 2>&1 || echo "[egl.sh] WARNING: apt-get install of EGL libs failed; mesh overlays will be skipped" >&2
{
  mkdir -p /usr/share/glvnd/egl_vendor.d &&
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSON'
{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}
JSON
} 2>/dev/null || echo "[egl.sh] WARNING: could not write the EGL vendor file; mesh overlays may be skipped" >&2
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
pip install --quiet --exists-action i pyglet pyopengl >/dev/null 2>&1 || echo "[egl.sh] WARNING: pip install pyglet/pyopengl failed; mesh overlays will be skipped" >&2
