"""Check real mesh rendering before loading video-inference checkpoints.

Run inside the same container and graphics environment as demo_video.py.
No model, checkpoint or CUDA initialization is needed for this small render.
"""

import ctypes
import ctypes.util
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def describe_environment():
    for name in ('PYOPENGL_PLATFORM', 'EGL_DEVICE_ID', '__EGL_VENDOR_LIBRARY_DIRS',
                 '__EGL_VENDOR_LIBRARY_FILENAMES', 'NVIDIA_DRIVER_CAPABILITIES',
                 'CUDA_VISIBLE_DEVICES'):
        print('[renderer-check] {}={}'.format(name, os.environ.get(name, '<unset>')), flush=True)
    # Diagnose native-library visibility independently of Python package imports.
    for name, fallback in (('EGL', 'libEGL.so.1'), ('GL', 'libGL.so.1'),
                           ('GLESv2', 'libGLESv2.so.2'), ('EGL_nvidia', 'libEGL_nvidia.so.0')):
        library = ctypes.util.find_library(name) or fallback
        try:
            ctypes.CDLL(library)
            status = 'loadable'
        except OSError as error:
            status = str(error)
        print('[renderer-check] {}: {}'.format(library, status), flush=True)


def describe_graphics_context():
    # Metadata is useful but is not the success criterion; a real render is.
    try:
        from OpenGL import GL
        for name in ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION'):
            value = GL.glGetString(getattr(GL, name))
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            print('[renderer-check] {}={}'.format(name, value), flush=True)
    except Exception as error:
        print('[renderer-check] OpenGL context description unavailable: {}'.format(error), flush=True)


def main():
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    renderer = None
    try:
        describe_environment()
        import numpy as np
        from demo_video import MeshRenderer, camera_matrix

        try:
            renderer = MeshRenderer(64, camera_matrix(64, 55), np.array([[0, 1, 2, 3]]))
            describe_graphics_context()
            vertices = np.array([[-.3, -.3, 3], [.3, -.3, 3], [.3, .3, 3], [-.3, .3, 3]], np.float32)
            background = np.zeros((64, 64, 3), np.uint8)
            rendered = renderer.render(background, vertices)
            if rendered.shape != background.shape or not np.isfinite(rendered).all():
                raise RuntimeError('Renderer returned an invalid image')
            if not rendered[32, 32].any() or rendered[0, 0].any():
                raise RuntimeError('Known test mesh was not drawn at the expected position')
        finally:
            if renderer is not None:
                renderer.close()
    except Exception:
        print('[renderer-check] FAILED before checkpoint loading; original error follows:', file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    print('[renderer-check] PASS: 64x64 test mesh rendered; graphics setup is ready.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
