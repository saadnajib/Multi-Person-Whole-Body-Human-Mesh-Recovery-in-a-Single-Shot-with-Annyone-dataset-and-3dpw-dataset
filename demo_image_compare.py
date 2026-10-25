"""Compare the completed ViT-B student and original NAVER Multi-HMR-Anny on photos.

See docs/image_comparison.md. All visible detections are rendered at their raw
predicted placement. No tracking, landmark fitting, or temporal filtering.
"""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path

from demo_video import (MeshRenderer, camera_matrix, draw_title, letterbox,
                        load_predictor, require_file, resolve_student_fix,
                        visible_detections, write_json)

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', type=Path, required=True,
                        help='Image files or folders (direct children, sorted; no recursive scan)')
    parser.add_argument('--student-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, default=Path('models/multiHMR/multiHMR_672_L_anny.pt'))
    parser.add_argument('--student-bone-fix', default='auto')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--panel-size', type=int, default=672)
    parser.add_argument('--fov', type=float, default=55.)
    return parser.parse_args(argv)


def image_files(paths):
    files = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError('Missing image or directory: {}'.format(path))
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in EXTENSIONS:
                files.append(candidate.resolve())
            elif candidate == path:
                raise ValueError('Unsupported image extension: {}'.format(path))
    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError('No supported image files found')
    return files


def read_rgb(path):
    from PIL import Image, ImageOps
    import numpy as np
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def labelled(panel, title, count=None):
    import numpy as np
    import cv2
    result = np.concatenate([np.full((64, panel.shape[1], 3), 24, np.uint8), panel])
    draw_title(result, title)
    message = 'Original image' if count is None else '{} visible detection(s)'.format(count)
    cv2.putText(result, message, (8, 52), cv2.FONT_HERSHEY_SIMPLEX,
                min(.5, panel.shape[1] / 700), (255, 255, 255), 1, cv2.LINE_AA)
    return result


def run(args, predictor_factory=load_predictor, renderer_factory=MeshRenderer):
    from PIL import Image
    import numpy as np
    files = image_files(args.images)
    if not math.isfinite(args.fov) or not 0 < args.fov < 180:
        raise ValueError('--fov must be finite and between 0 and 180')
    if args.panel_size < 128:
        raise ValueError('--panel-size must be at least 128')
    student = require_file(args.student_checkpoint)
    baseline = require_file(args.baseline_checkpoint)
    fix, fix_source = resolve_student_fix(student, args.student_bone_fix)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Output exists; choose a new directory: {}'.format(output))
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', models={}, images=[], student_fix_source=fix_source,
                    fov=args.fov, panel_size=args.panel_size,
                    placement='Raw camera-space predictions; all visible detections; no forced box placement, smoothing or landmark alignment',
                    preprocessing='EXIF transpose, RGB, full-image letterbox; common assumed FOV; each model uses its saved input size',
                    timing_note='Per-image timings only; no warm-up benchmark or accuracy claims')
    try:
        for directory in ('input', 'student', 'baseline', 'comparison'):
            (output / directory).mkdir()
        for index, path in enumerate(files):
            rgb = read_rgb(path)
            name = '{:04d}_{}.png'.format(index, path.stem)
            manifest['images'].append(dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           width=rgb.shape[1], height=rgb.shape[0], output_name=name, models={}))
            # Store unlabelled lossless input panels. Both models consume the
            # same original RGB file, not a compressed render or this preview.
            Image.fromarray(letterbox(rgb, args.panel_size)).save(output / 'input' / name)
        write_json(output / 'run.json', manifest)
        camera = camera_matrix(args.panel_size, args.fov)
        for kind, checkpoint, correction in [('student', student, fix), ('baseline', baseline, 'none')]:
            predictor = predictor_factory(checkpoint, kind, correction, args.device, False)
            renderer = None
            try:
                renderer = renderer_factory(args.panel_size, camera, predictor.faces)
                manifest['models'][kind] = predictor.metadata
                for row in manifest['images']:
                    path = Path(row['source'])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                        raise ValueError('Source image changed during comparison: {}'.format(path))
                    rgb = read_rgb(path)
                    people = predictor.predict(letterbox(rgb, predictor.input_size), camera_matrix(predictor.input_size, args.fov))
                    visible = visible_detections(people, camera, args.panel_size)
                    panel = letterbox(rgb, args.panel_size)
                    if visible:
                        panel = renderer.render_many(panel, [p['vertices'] for p in visible])
                    title = 'Student (ViT-B)' if kind == 'student' else 'Original Multi-HMR-Anny (NAVER)'
                    Image.fromarray(labelled(panel, title, len(visible))).save(output / kind / row['output_name'])
                    row['models'][kind] = dict(detections=len(people), rendered=len(visible),
                        people=[dict(confidence=float(p['confidence']), bbox=p['bbox'].tolist()) for p in visible],
                        timings=getattr(predictor, 'last_timings', {}))
                    print('[{}] {}: {} visible detections'.format(kind, path.name, len(visible)), flush=True)
            finally:
                if renderer is not None:
                    renderer.close()
                del predictor
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for row in manifest['images']:
            name = row['output_name']
            panels = [labelled(read_rgb(output / 'input' / name), 'Input')]
            panels += [read_rgb(output / kind / name) for kind in ('student', 'baseline')]
            Image.fromarray(np.concatenate(panels, axis=1)).save(output / 'comparison' / name)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
        raise
    finally:
        write_json(output / 'run.json', manifest)
    print('Saved comparisons to {}'.format(output / 'comparison'), flush=True)
    return manifest


if __name__ == '__main__':
    run(parse_args())
