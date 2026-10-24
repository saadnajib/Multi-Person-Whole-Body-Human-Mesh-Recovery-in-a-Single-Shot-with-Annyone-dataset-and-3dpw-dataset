"""Student / original Multi-HMR video comparison with optional temporal filtering.

See docs/video_inference.md. Model/graphics imports are lazy so --help and file
validation work without initializing CUDA or EGL. Only load trusted checkpoints.
"""

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--student-checkpoint', required=True, type=Path)
    p.add_argument('--baseline-checkpoint', type=Path,
                   default=Path('models/multiHMR/multiHMR_672_L_anny.pt'))
    p.add_argument('--teacher-checkpoint', type=Path)
    p.add_argument('--student-bone-fix', default='auto',
                   help='auto: read RUN/student_fix_arg.txt; or checkpoint, none, or JSON path')
    p.add_argument('--student-only', action='store_true')
    p.add_argument('--output', required=True, type=Path, help='New output directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--fov', type=float, default=55, help='Field of view along the longer image side, degrees')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0, help='Maximum processed frames; 0 means all')
    p.add_argument('--warmup-frames', type=int, default=10,
                   help='Exclude the first N processed frames from timing statistics; keep them in the videos')
    p.add_argument('--panel-size', type=int, default=672)
    p.add_argument('--amp', action='store_true', help='Opt into CUDA mixed precision for all models')
    p.add_argument('--temporal', choices=('off', 'ema'), default='off',
                   help='ema: add a causal filtered student alongside raw outputs; no retraining')
    p.add_argument('--temporal-pose-tau', type=float, default=0.08, help='Rotation smoothing time constant, seconds')
    p.add_argument('--temporal-translation-tau', type=float, default=0.12, help='Head-anchor smoothing time constant, seconds')
    p.add_argument('--temporal-shape-tau', type=float, default=1.0, help='Shape smoothing time constant, seconds')
    p.add_argument('--temporal-max-gap', type=float, default=0.25, help='Reset temporal state above this time gap, seconds')
    p.add_argument('--alignment', choices=('off', 'landmarks'), default='off',
                   help='landmarks: refine Student placement/depth using independent 2D observations')
    p.add_argument('--landmarks', type=Path, help='JSON from scripts/video/extract_pose_landmarks.py for this original video')
    p.add_argument('--alignment-min-confidence', type=float, default=0.5)
    return p.parse_args(argv)


def require_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Missing file: {}'.format(path))
    return path


def resolve_student_fix(checkpoint, option):
    """Exp8 estimates its own correction after saving the training checkpoint."""
    source = option
    if option == 'auto':
        sidecar = require_file(checkpoint.parent.parent.parent / 'student_fix_arg.txt')
        lines = sidecar.read_text().splitlines()
        if len(lines) != 1 or not lines[0].strip():
            raise ValueError('{} must contain one JSON path or none'.format(sidecar))
        option = lines[0].strip()
        if option != 'none' and not Path(option).is_absolute():
            raise ValueError('{} must contain an absolute JSON path or none'.format(sidecar))
        source = str(sidecar)
    if option not in ('checkpoint', 'none'):
        option = str(require_file(option))
    return option, source


def validate_args(args):
    if not math.isfinite(args.fov) or not 0 < args.fov < 180:
        raise ValueError('--fov must be between 0 and 180 degrees')
    if args.stride < 1 or args.start_frame < 0 or args.max_frames < 0:
        raise ValueError('stride must be positive; start-frame and max-frames must be nonnegative')
    if args.warmup_frames < 0:
        raise ValueError('--warmup-frames must be nonnegative')
    for name in ('temporal_pose_tau', 'temporal_translation_tau', 'temporal_shape_tau'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError('--{} must be finite and nonnegative'.format(name.replace('_', '-')))
    if not math.isfinite(args.temporal_max_gap) or args.temporal_max_gap <= 0:
        raise ValueError('--temporal-max-gap must be finite and positive')
    if args.temporal != 'off' and args.amp:
        raise ValueError('Temporal Anny re-decoding currently requires fp32: omit --amp')
    if not math.isfinite(args.alignment_min_confidence) or not 0 <= args.alignment_min_confidence <= 1:
        raise ValueError('--alignment-min-confidence must be in [0, 1]')
    if args.alignment == 'landmarks':
        if args.landmarks is None:
            raise ValueError('--alignment landmarks requires --landmarks JSON for the original video')
        args.landmarks = require_file(args.landmarks)
    elif args.landmarks is not None:
        raise ValueError('--landmarks requires --alignment landmarks')
    if args.panel_size < 128 or args.panel_size % 2:
        raise ValueError('--panel-size must be an even integer of at least 128 (video codec requirement)')
    if args.student_only and args.teacher_checkpoint:
        raise ValueError('--student-only cannot be combined with --teacher-checkpoint')
    args.video = require_file(args.video)
    args.student_checkpoint = require_file(args.student_checkpoint)
    if not args.student_only:
        args.baseline_checkpoint = require_file(args.baseline_checkpoint)
    if args.teacher_checkpoint:
        args.teacher_checkpoint = require_file(args.teacher_checkpoint)
    args.resolved_student_fix, args.student_fix_source = resolve_student_fix(
        args.student_checkpoint, args.student_bone_fix)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new --output: {}'.format(args.output))


def write_json(path, data):
    # Atomic replacement prevents a partially written status file on interruption.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError('OpenCV cannot open video: {}'.format(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid FPS. Convert it to a constant-frame-rate MP4 first.')
        ok, first = cap.read()
        if not ok:
            raise RuntimeError('Video contains no decodable frames: {}'.format(path))
        h, w = first.shape[:2]
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return dict(fps=fps, width=w, height=h,
                    reported_frame_count=int(count) if math.isfinite(count) else 0,
                    timing='constant_frame_rate_assumed', audio='not copied')
    finally:
        cap.release()


def selected_frames(path, start, stride, limit):
    """Decode in order instead of relying on codec-dependent random seeking."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot reopen video: {}'.format(path))
    try:
        index = emitted = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index >= start and (index - start) % stride == 0:
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                emitted += 1
                if limit and emitted >= limit:
                    break
            index += 1
    finally:
        cap.release()


def letterbox(rgb, size):
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def camera_matrix(size, fov):
    import numpy as np
    f = size / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], dtype=np.float32)


def draw_title(panel, title):
    """Replace only the top title line; preserve image pixels and frame status."""
    import cv2
    panel[:32] = 24
    scale = min(0.65, panel.shape[1] / 650)
    width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale *= min(1.0, (panel.shape[1] - 16) / max(width, 1))
    cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def fps_title(title, fps, running=False):
    label = 'running inference' if running else 'inference avg'
    return '{} | {}: {}'.format(title, label, 'N/A' if fps is None else '{:.1f} FPS'.format(fps))


def annotate(rgb, title, frame, timestamp, status):
    import cv2
    import numpy as np
    header = np.full((64, rgb.shape[1], 3), 24, dtype=np.uint8)
    scale = min(0.65, rgb.shape[1] / 650)
    draw_title(header, title)
    cv2.putText(header, 'frame {} | {:.2f}s | {}'.format(frame, timestamp, status),
                (8, 49), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([header, rgb], axis=0)


class VideoWriter:
    def __init__(self, path, fps, width, height):
        import cv2
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError('Cannot create MP4 {}. Check OpenCV codec support and output permissions.'.format(path))
        self.size = (height, width, 3)

    def write(self, rgb):
        import cv2
        if rgb.shape != self.size:
            raise ValueError('Unexpected video frame dimensions: {}'.format(rgb.shape))
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()


class MeshRenderer:
    """Persistent renderer; metric camera-space geometry is never repositioned."""
    def __init__(self, size, K, faces):
        import numpy as np
        import pyrender
        self.pyrender = pyrender
        self.K = K
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.faces.shape[1] == 4:
            self.faces = np.concatenate([self.faces[:, [0, 1, 2]], self.faces[:, [0, 2, 3]]])
        if self.faces.shape[1] != 3:
            raise ValueError('Expected triangular or quad body faces')
        try:
            self.renderer = pyrender.OffscreenRenderer(size, size)
        except Exception as exc:
            raise RuntimeError(
                'Cannot initialize offscreen rendering: {}: {}. '
                'PYOPENGL_PLATFORM={}, EGL_DEVICE_ID={}, __EGL_VENDOR_LIBRARY_FILENAMES={}. '
                'Run python scripts/video/check_renderer.py inside the GPU container for diagnostics.'.format(
                    type(exc).__name__, exc, os.environ.get('PYOPENGL_PLATFORM', '<unset>'),
                    os.environ.get('EGL_DEVICE_ID', '<unset>'),
                    os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '<unset>'))) from exc

    def render(self, rgb, vertices):
        import numpy as np
        import trimesh
        p = self.pyrender
        scene = p.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        material = p.MetallicRoughnessMaterial(baseColorFactor=(0.25, 0.85, 0.95, 1),
                                               metallicFactor=0, roughnessFactor=0.7, doubleSided=True)
        scene.add(p.Mesh.from_trimesh(mesh, material=material, smooth=False))
        camera = p.IntrinsicsCamera(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2],
                                    znear=0.01, zfar=1000)
        pose = np.diag([1., -1., -1., 1.])  # CV: x right/y down/z forward -> OpenGL camera.
        scene.add(camera, pose=pose)
        scene.add(p.DirectionalLight(color=np.ones(3), intensity=2.5), pose=pose)
        color, depth = self.renderer.render(scene, flags=p.RenderFlags.RGBA)
        visible = depth > 0
        result = rgb.copy()
        result[visible] = (0.8 * color[visible, :3] + 0.2 * rgb[visible]).astype(np.uint8)
        return result

    def close(self):
        self.renderer.delete()


def as_numpy(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class TorchPredictor:
    def __init__(self, model, device, amp, metadata):
        self.model, self.device, self.amp = model, device, amp
        self.input_size = int(model.img_size)
        self.faces = as_numpy(model.body_model.faces).astype('int64')
        self.phenotype_names = list(model.body_model.phenotype_labels)
        self.joint_names = list(model.body_model.bone_labels)
        self.metadata = metadata
        self.last_timings = {}

    @property
    def anchor_index(self):
        return self.model.person_center_idx

    def decode_temporal(self, rotvec, shape, anchor):
        """Decode already-corrected parameters; never reapply sigmoid or bone fixes.

        Match the image model's Anny convention and skinning. A changed shape or
        pose changes the head location, so recenter on the newly decoded head.
        """
        import torch
        import roma
        from utils import rotation_to_homogeneous
        with torch.inference_mode():
            rotations = roma.rotvec_to_rotmat(torch.as_tensor(
                rotvec, dtype=torch.float32, device=self.device)).unsqueeze(0)
            phenotypes = torch.as_tensor(shape, dtype=torch.float32, device=self.device).unsqueeze(0)
            translation = torch.as_tensor(anchor, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
            kwargs = dict(pose_parameters=rotation_to_homogeneous(rotations),
                          phenotype_kwargs={name: phenotypes[:, i] for i, name in enumerate(self.phenotype_names)})
            convention = self.metadata.get('decode_pose_parameterization')
            if convention is not None:
                kwargs['pose_parameterization'] = convention
            decoded = self.model.body_model(**kwargs)
            joints = decoded['bone_poses'][:, :, :3, -1]
            offset = translation - joints[:, [self.anchor_index]]
            vertices = decoded['vertices'] + offset
            joints = joints + offset
            # CPU copies complete this work before the outer temporal timer ends.
            return dict(vertices=as_numpy(vertices[0]), joints=as_numpy(joints[0]),
                        shape=as_numpy(phenotypes[0]), rotvec=as_numpy(rotvec), anchor=as_numpy(anchor))

    def synchronize(self):
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def predict(self, rgb, K):
        import numpy as np
        import torch
        self.last_timings = {}
        self.synchronize()
        preprocessing_started = time.perf_counter()
        x = (rgb.astype(np.float32) / 255 - np.array([.485, .456, .406], np.float32))
        x /= np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        camera = torch.from_numpy(K).unsqueeze(0).to(self.device)
        self.synchronize()  # Input transfers finish before the model timer starts.
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.amp):
            inference_started = time.perf_counter()
            predictions = self.model(x, K=camera, is_training=False, det_thresh=0.3, nms_kernel_size=3)
            self.synchronize()  # Measure completed GPU work, not just kernel submission.
            inference_ms = (time.perf_counter() - inference_started) * 1000
        transfer_started = time.perf_counter()
        if isinstance(predictions, tuple) and len(predictions) == 2 and predictions[0] == {}:
            predictions = predictions[1]  # Original Multi_HMR's empty-detection return.
        if not isinstance(predictions, list):
            raise RuntimeError('Unexpected model inference output: {}'.format(type(predictions)))
        result = []
        for person in predictions:
            result.append(dict(vertices=as_numpy(person['v3d']), joints=as_numpy(person['j3d']),
                               shape=as_numpy(person['shape']).reshape(-1),
                               rotvec=as_numpy(person['rotvec']),
                               confidence=float(as_numpy(person.get('scores', person.get('score'))).reshape(()))))
        self.last_timings = dict(preprocess_ms=preprocessing_ms, inference_ms=inference_ms,
                                 output_transfer_ms=(time.perf_counter() - transfer_started) * 1000)
        return result


def load_predictor(checkpoint, kind, fix, device, amp):
    import torch
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Submit on a GPU node, or explicitly use --device cpu for a slow test.')
    if device.type not in ('cpu', 'cuda') or (amp and device.type != 'cuda'):
        raise ValueError('Use cpu or cuda; --amp requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = ckpt.get('args')
    kwargs = dict(vars(saved)) if hasattr(saved, '__dict__') else dict(saved or {})
    if not kwargs:
        raise ValueError('{} has no saved model arguments'.format(checkpoint))
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    if not isinstance(sd, dict):
        raise ValueError('{} has no model state dictionary'.format(checkpoint))
    original = sum(k.startswith(('encoder.backbone.', 'decoder.', 'mlp_pose.')) for k in sd)
    student = sum(k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')) for k in sd)
    if (kind == 'baseline' and original <= student) or (kind != 'baseline' and student <= original):
        raise ValueError('{} is the wrong model class for {}'.format(checkpoint, kind))
    kwargs['pretrained_backbone'] = False
    kwargs['person_center'] = 'head'
    kwargs.setdefault('simple_depth_encoding', 1)
    kwargs.pop('pretrained', None)
    correction = None
    correction_origin = 'not supported by original Multi-HMR'
    if kind == 'baseline':
        from multi_hmr_anny.multi_hmr import Multi_HMR, POSE_PARAMETERIZATION
        # Read architecture from actual tensors: released NAVER checkpoint uses 1024/2048.
        kwargs['xat_dim'] = int(sd['dec_to_token.weight'].shape[0])
        kwargs['xat_mlp_dim'] = int(sd['decoder.transformer.layers.0.2.fn.net.0.weight'].shape[0])
        model = Multi_HMR(**kwargs)
        pose_label = 'root_relative_world'
    else:
        import bone_fix
        from model import Model
        if kwargs.get('pose_param', 'local-bone') != 'local-bone':
            raise ValueError('This Model checkpoint uses an unsupported pose parameterization')
        correction, correction_origin = bone_fix.from_arg('' if fix == 'checkpoint' else fix,
                                                          kwargs.get('bone_fix_spec'))
        kwargs['bone_fix'] = 'none' if correction is None else ''
        kwargs['bone_fix_spec'] = correction
        model = Model(**kwargs)
        model.mask_helper_joints = True
        model.relax_hands = model.rest_pose = model.neutral_shape = False
        model.neutral_phenotype_names = []
        pose_label = kwargs.get('pose_param', 'local-bone')
    state = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in state and hasattr(v, 'shape') and v.shape == state[k].shape}
    missing_parameters = [k for k, _ in model.named_parameters() if k not in matched]
    coverage = len(matched) / max(1, len(state))
    if missing_parameters or coverage < 0.95:
        raise RuntimeError('Checkpoint is incomplete/incompatible: {:.1%} state coverage; missing parameters {}'.format(
            coverage, missing_parameters[:10]))
    # Fail on any shape mismatch instead of silently leaving an initialized tensor.
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns, model_class=type(model).__name__,
                    backbone=kwargs.get('backbone'), input_size=int(model.img_size),
                    device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    timing_method='perf_counter wall latency with CUDA synchronization before and after model forward' if device.type == 'cuda' else 'perf_counter wall latency on CPU',
                    loaded_state_fraction=coverage, parameter_count=sum(p.numel() for p in model.parameters()),
                    precision='amp' if amp else ('fp32_with_tf32' if device.type == 'cuda' else 'fp32'),
                    checkpoint_pose_label=pose_label,
                    body_model_default_pose_parameterization=getattr(model.body_model, 'default_pose_parameterization', None),
                    decode_pose_parameterization=POSE_PARAMETERIZATION if kind == 'baseline' else None,
                    bone_fix_source=correction_origin, bone_fix_spec=correction,
                    helper_joint_mask=True, relax_hands=False, detection_threshold=0.3)
    print('[{}] loaded {} ({:.1%} state coverage), correction: {}'.format(
        kind, checkpoint, coverage, correction_origin), flush=True)
    return TorchPredictor(model, device, amp, metadata)


def visible_detections(people, K, size):
    """Reject invalid geometry; project unchanged meshes to associate one person."""
    import numpy as np
    detections = []
    for person in people:
        v, j = person['vertices'], person['joints']
        arrays = (v, j, person['shape'], person['rotvec'])
        if not all(np.isfinite(a).all() for a in arrays) or not math.isfinite(person['confidence']):
            continue
        if v.ndim != 2 or v.shape[1] != 3 or j.ndim != 2 or j.shape[1] != 3 or not len(j):
            continue
        # Near-plane intersections cannot yield a useful whole-person box.
        if np.any(v[:, 2] <= 0.01):
            continue
        xy = v[:, :2] / v[:, 2:] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
        low, high = np.maximum(xy.min(0), 0), np.minimum(xy.max(0), size)
        if np.any(high <= low):
            continue
        detections.append(dict(person, bbox=np.r_[low, high]))
    return detections


def save_measurements(directory, records, predictor, stride, warmup_frames, playback_fps, temporal=None, alignment=None):
    import numpy as np
    from scripts.video.video_utils import summarize_stability, summarize_timing
    directory.mkdir()
    summary = summarize_stability(records, predictor.phenotype_names, expected_frame_step=stride)
    # Compare the resulting Cartesian joint motion. Raw per-bone rotations and
    # each model's correction conventions are retained for reproduction only.
    summary.pop('bone_rotation_step_degrees')
    summary.pop('bone_rotation_speed_degrees_per_s')
    summary['joint_names'] = predictor.joint_names
    summary['phenotype_names'] = predictor.phenotype_names
    summary['rotation_note'] = 'Output rotvec saved for reproduction; comparison uses Cartesian joint motion.'
    summary['processing'] = temporal or {'method': 'none', 'source': 'raw image-model predictions'}
    if temporal:
        summary['caveat'] = ('Causally filtered predictions; filtering can reduce real motion and introduce lag. '
                             'Lower motion proxies do not establish better reconstruction accuracy. '
                             'Missing frames and track boundaries break temporal differences.')
    if alignment:
        summary['processing'] = alignment
        summary['caveat'] = ('Camera placement fitted to independent detected 2D landmarks; pose and shape are retained. '
                             'Better reprojection does not establish better 3D accuracy. '
                             'Missing or unreliable observations keep the unaligned prediction.')
    summary['timing'] = summarize_timing(records, warmup_frames, playback_fps)
    write_json(directory / 'summary.json', summary)
    n, nj, ns = len(records), len(predictor.joint_names), len(predictor.phenotype_names)
    arrays = {key: np.full((n,) + shape, np.nan, np.float32)
              for key, shape in dict(shape=(ns,), root=(3,), joints=(nj, 3), rotvec=(nj, 3), bbox=(4,)).items()}
    if any('anchor' in record for record in records):
        arrays['anchor'] = np.full((n, 3), np.nan, np.float32)
    for index, record in enumerate(records):
        if record['selected_index'] is not None:
            for key in arrays:
                arrays[key][index] = record[key]
    np.savez_compressed(directory / 'predictions.npz', **arrays,
                        frame_index=np.array([r['frame_index'] for r in records], np.int64),
                        timestamp_s=np.array([r['timestamp_s'] for r in records]),
                        valid=np.array([r['selected_index'] is not None for r in records]),
                        segment_id=np.array([r['segment_id'] or -1 for r in records]),
                        joint_names=np.array(predictor.joint_names), phenotype_names=np.array(predictor.phenotype_names))
    fields = ['frame_index', 'timestamp_s', 'status', 'segment_id', 'selected_index', 'detections',
              'valid_detections', 'confidence', 'forward_ms', 'timing_included',
              'preprocess_ms', 'inference_ms', 'output_transfer_ms', 'render_ms', 'temporal_ms', 'temporal_state',
              'processing_ms', 'root_step_m', 'root_speed_m_per_s',
              'mean_joint_speed_m_per_s', 'mean_joint_acceleration_m_per_s2',
              'alignment_ms', 'alignment_status', 'alignment_accepted',
              'landmark_error_before_px', 'landmark_error_after_px', 'landmark_count', 'landmark_detector_ms']
    fields += ['shape_' + name for name in predictor.phenotype_names]
    with (directory / 'frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            row = {k: record.get(k) for k in fields}
            if record['selected_index'] is not None:
                row.update({'shape_' + name: float(value) for name, value in zip(predictor.phenotype_names, record['shape'])})
            # Use the same continuity rules for the time series and aggregate report.
            local = summarize_stability(records[max(0, index - 2):index + 1],
                                         predictor.phenotype_names, expected_frame_step=stride)
            if index and record['selected_index'] is not None:
                previous = records[index - 1]
                if (previous['selected_index'] is not None and previous['segment_id'] == record['segment_id']
                        and record['frame_index'] - previous['frame_index'] == stride):
                    dt = record['timestamp_s'] - previous['timestamp_s']
                    row['root_step_m'] = float(np.linalg.norm(record['root'] - previous['root']))
                    row['root_speed_m_per_s'] = row['root_step_m'] / dt
                    rel = record['joints'] - record['joints'][0]
                    prev_rel = previous['joints'] - previous['joints'][0]
                    row['mean_joint_speed_m_per_s'] = float(np.linalg.norm((rel - prev_rel)[1:] / dt, axis=1).mean())
                    row['mean_joint_acceleration_m_per_s2'] = local['pelvis_relative_joint_acceleration_m_per_s2']['mean']
            writer.writerow(row)
    return summary


def temporal_settings(args):
    return dict(method='causal_exponential_parameter_filter', learned=False,
                pose_tau_s=args.temporal_pose_tau, translation_tau_s=args.temporal_translation_tau,
                shape_tau_s=args.temporal_shape_tau, max_gap_s=args.temporal_max_gap,
                rotations='shortest-arc quaternion interpolation',
                placement='filtered camera-space head anchor; re-decoded Anny geometry',
                missing='no interpolation; reset on missing detections, track or time discontinuity')


def temporal_person(person, predictor, state, timestamp, segment, K, size):
    """Rebuild a valid body from filtered parameters, retaining raw provenance."""
    import numpy as np
    params = state.update(person['rotvec'], person['shape'], person['joints'][predictor.anchor_index],
                          timestamp, segment)
    decoded = predictor.decode_temporal(params['rotvec'], params['shape'], params['anchor'])
    if params['state'] != 'filtered':
        # The first body in each segment should be unchanged. This catches a
        # wrong decoder convention or accidentally applied correction early.
        for key in ('vertices', 'joints'):
            if not np.allclose(decoded[key], person[key], atol=1e-4, rtol=1e-4):
                raise RuntimeError('Temporal Anny re-decode does not reproduce raw {}. Check body convention/correction.'.format(key))
    decoded['confidence'] = person['confidence']
    valid = visible_detections([decoded], K, size)
    if not valid:
        # Do not render invalid/behind-camera interpolation. Retain a real raw
        # observation, mark the fallback, and discard filter history.
        state.reset()
        return dict(person), 'fallback_raw_invalid_geometry'
    return valid[0], params['state']


def process_model(args, kind, title, predictor, info, renderer_factory, landmark_sequence=None):
    from scripts.video.video_utils import SinglePersonTracker
    tracker = SinglePersonTracker()
    size, fps = args.panel_size, info['fps']
    K = camera_matrix(size, args.fov)
    input_K = camera_matrix(predictor.input_size, args.fov)
    renderer = writer = input_writer = temporal_writer = aligned_writer = None
    records, temporal_records, aligned_records, alignment_reports = [], [], [], []
    use_temporal = kind == 'student' and args.temporal == 'ema'
    use_alignment = kind == 'student' and args.alignment == 'landmarks'
    if use_alignment:
        from scripts.video.landmark_alignment import align_to_landmarks
        if landmark_sequence is None:
            raise ValueError('Alignment requires validated landmark observations')
    temporal_filter = None
    if use_temporal:
        from scripts.video.temporal_filter import TemporalBodyFilter
        temporal_filter = TemporalBodyFilter(pose_tau=args.temporal_pose_tau,
                                              translation_tau=args.temporal_translation_tau,
                                              shape_tau=args.temporal_shape_tau,
                                              max_gap_s=args.temporal_max_gap)
        if not callable(getattr(predictor, 'decode_temporal', None)):
            raise ValueError('Temporal processing requires an Anny parameter decoder')
    measured_count, inference_total_ms = 0, 0.0
    processing_total_ms = aligned_total_ms = 0.0
    try:
        renderer = renderer_factory(size, K, predictor.faces)
        writer = VideoWriter(args.output / (kind + '.mp4'), fps / args.stride, size, size + 64)
        if use_temporal:
            temporal_writer = VideoWriter(args.output / 'student_temporal.mp4', fps / args.stride, size, size + 64)
        if use_alignment:
            aligned_writer = VideoWriter(args.output / 'student_aligned.mp4', fps / args.stride, size, size + 64)
        if kind == 'student':
            input_writer = VideoWriter(args.output / 'input.mp4', fps / args.stride, size, size + 64)
        for frame_index, rgb in selected_frames(args.video, args.start_frame, args.stride, args.max_frames):
            timestamp = frame_index / fps
            panel = letterbox(rgb, size)
            started = time.perf_counter()
            model_image = letterbox(rgb, predictor.input_size)
            resize_ms = (time.perf_counter() - started) * 1000
            people = predictor.predict(model_image, input_K)
            forward_ms = (time.perf_counter() - started) * 1000
            stages = dict(getattr(predictor, 'last_timings', {}))
            if stages.get('preprocess_ms') is not None:
                stages['preprocess_ms'] += resize_ms
            detections = visible_detections(people, K, size)
            tracking = tracker.update(detections)
            record = dict(tracking, frame_index=frame_index, timestamp_s=timestamp,
                          detections=len(people), valid_detections=len(detections), forward_ms=forward_ms,
                          timing_included=len(records) >= args.warmup_frames, **stages)
            filtered_record = dict(record)
            filtered_panel = panel.copy() if use_temporal else None
            aligned_record = dict(record)
            aligned_panel = panel.copy() if use_alignment else None
            status = 'NO MATCH ({} detections)'.format(len(people))
            if tracking['selected_index'] is not None:
                person = detections[tracking['selected_index']]
                render_started = time.perf_counter()
                panel = renderer.render(panel, person['vertices'])
                record['render_ms'] = (time.perf_counter() - render_started) * 1000
                record.update({key: person[key] for key in ('shape', 'joints', 'rotvec', 'bbox', 'confidence')})
                record['root'] = person['joints'][0].copy()
                if hasattr(predictor, 'anchor_index'):
                    record['anchor'] = person['joints'][predictor.anchor_index].copy()
                status = '{} | track {} | {:.2f}'.format(tracking['status'], tracking['segment_id'], person['confidence'])
                alignment_source = person
                if use_temporal:
                    temporal_started = time.perf_counter()
                    if records and frame_index - records[-1]['frame_index'] != args.stride:
                        temporal_filter.reset()
                    filtered, filter_status = temporal_person(person, predictor, temporal_filter, timestamp,
                                                              tracking['segment_id'], K, size)
                    filtered_record.update(temporal_ms=(time.perf_counter() - temporal_started) * 1000,
                                           temporal_state=filter_status)
                    filtered_record.update({key: filtered[key] for key in ('shape', 'joints', 'rotvec', 'bbox', 'confidence')})
                    filtered_record['root'] = filtered['joints'][0].copy()
                    filtered_record['anchor'] = filtered['joints'][predictor.anchor_index].copy()
                    render_started = time.perf_counter()
                    filtered_panel = renderer.render(filtered_panel, filtered['vertices'])
                    filtered_record['render_ms'] = (time.perf_counter() - render_started) * 1000
                    alignment_source = filtered
                if use_alignment:
                    align_started = time.perf_counter()
                    observations, association = landmark_sequence.observation(frame_index, size, person['bbox'])
                    aligned, report = align_to_landmarks(alignment_source, predictor.joint_names,
                                                         observations, K, size, args.alignment_min_confidence)
                    aligned_record['alignment_ms'] = (time.perf_counter() - align_started) * 1000
                    aligned_record.update({key: aligned[key] for key in ('shape', 'joints', 'rotvec', 'bbox', 'confidence')})
                    aligned_record['root'] = aligned['joints'][0].copy()
                    if hasattr(predictor, 'anchor_index'):
                        aligned_record['anchor'] = aligned['joints'][predictor.anchor_index].copy()
                    report.update(frame_index=frame_index, timestamp_s=timestamp, association=association)
                    if not observations:
                        report['status'] = association['status']
                    aligned_record.update(alignment_status=report['status'], alignment_accepted=report['accepted'],
                                           landmark_error_before_px=report['before_error_px'],
                                           landmark_error_after_px=report['after_error_px'], landmark_count=report['landmark_count'],
                                           landmark_detector_ms=association.get('detector_ms'))
                    render_started = time.perf_counter()
                    aligned_panel = renderer.render(aligned_panel, aligned['vertices'])
                    aligned_record['render_ms'] = (time.perf_counter() - render_started) * 1000
                    alignment_reports.append(report)
            elif use_temporal:
                temporal_filter.reset()
                filtered_record.update(temporal_ms=0.0, temporal_state='reset_missing')
            if use_alignment and tracking['selected_index'] is None:
                aligned_record.update(alignment_ms=0.0, alignment_status='missing_student', alignment_accepted=False)
                alignment_reports.append(dict(frame_index=frame_index, timestamp_s=timestamp,
                                               accepted=False, status='missing_student'))
            latency = record.get('inference_ms')
            if use_alignment:
                aligned_record['temporal_ms'] = filtered_record.get('temporal_ms', 0.0)
                aligned_record['temporal_state'] = filtered_record.get('temporal_state')
                aligned_record['processing_ms'] = (latency + aligned_record['temporal_ms'] + aligned_record['alignment_ms']) if latency is not None else None
            if record['timing_included'] and latency is not None and math.isfinite(latency) and latency > 0:
                measured_count += 1
                inference_total_ms += latency
                if use_temporal:
                    processing_total_ms += latency + filtered_record['temporal_ms']
                if use_alignment:
                    aligned_total_ms += aligned_record['processing_ms']
            running_fps = 1000 * measured_count / inference_total_ms if measured_count else None
            timed_title = fps_title(title, running_fps, running=True)
            if not record['timing_included']:
                timed_title = title + ' | timing warm-up'
            writer.write(annotate(panel, timed_title, frame_index, timestamp, status))
            if use_temporal:
                filtered_record['processing_ms'] = (latency + filtered_record['temporal_ms']) if latency is not None else None
                processing_fps = 1000 * measured_count / processing_total_ms if measured_count else None
                filtered_title = temporal_fps_title(processing_fps) if record['timing_included'] else 'Student temporal | timing warm-up'
                temporal_writer.write(annotate(filtered_panel, filtered_title, frame_index, timestamp,
                                                 status + ' | ' + filtered_record['temporal_state']))
                temporal_records.append(filtered_record)
            if use_alignment:
                aligned_fps = 1000 * measured_count / aligned_total_ms if measured_count else None
                aligned_title = alignment_fps_title(aligned_fps) if record['timing_included'] else 'Student aligned | timing warm-up'
                aligned_writer.write(annotate(aligned_panel, aligned_title, frame_index, timestamp,
                                               aligned_record['alignment_status'] + ' | landmarks offline'))
                aligned_records.append(aligned_record)
            if input_writer:
                input_writer.write(annotate(letterbox(rgb, size), 'Input | playback: {:.2f} FPS'.format(fps / args.stride),
                                            frame_index, timestamp, 'original frame'))
            records.append(record)
            if len(records) % 25 == 0:
                print('[{}] {} frames; source frame {}'.format(kind, len(records), frame_index), flush=True)
        if not records:
            raise RuntimeError('No selected frames decoded; check --start-frame and the input video')
    finally:
        for resource in (input_writer, aligned_writer, temporal_writer, writer, renderer):
            if resource is not None:
                resource.close()
    summary = save_measurements(args.output / kind, records, predictor, args.stride, args.warmup_frames,
                                 fps / args.stride)
    if use_temporal:
        filtered_summary = save_measurements(args.output / 'student_temporal', temporal_records, predictor,
                                              args.stride, args.warmup_frames, fps / args.stride,
                                              temporal=temporal_settings(args))
        filtered_summary['fallback_raw_frames'] = sum(r['temporal_state'].startswith('fallback_raw') for r in temporal_records)
        write_json(args.output / 'student_temporal' / 'summary.json', filtered_summary)
        summary['temporal'] = filtered_summary
        save_temporal_comparison(args.output, summary, filtered_summary)
    if use_alignment:
        settings = alignment_settings(args, landmark_sequence)
        aligned_summary = save_measurements(args.output / 'student_aligned', aligned_records, predictor,
                                             args.stride, args.warmup_frames, fps / args.stride, alignment=settings)
        aligned_summary['alignment_accepted_frames'] = sum(r['accepted'] for r in alignment_reports)
        aligned_summary['alignment_skipped_frames'] = len(alignment_reports) - aligned_summary['alignment_accepted_frames']
        write_json(args.output / 'student_aligned' / 'summary.json', aligned_summary)
        write_json(args.output / 'student_aligned' / 'alignment.json', dict(settings=settings, frames=alignment_reports))
        summary['alignment'] = aligned_summary
    print('[{}] {} frames, {} with a matched person, {} missing'.format(
        kind, summary['frames'], summary['selected_frames'], summary['missing_frames']), flush=True)
    timing = summary['timing']
    print('[{}] {}; {} measured frames, {} warm-up frames excluded; render mean: {} ms'.format(
        kind, fps_title(title, timing['inference_fps']), timing['measured_frames'],
        timing['warmup_frames_excluded'], timing['render_ms']['mean']), flush=True)
    return [r['frame_index'] for r in records], summary


def temporal_fps_title(fps):
    return 'Student temporal | model+filter/decode: {}'.format('N/A' if fps is None else '{:.1f} FPS'.format(fps))


def alignment_fps_title(fps):
    return 'Student aligned | model+refine: {}'.format('N/A' if fps is None else '{:.1f} FPS'.format(fps))


def alignment_settings(args, sequence):
    return dict(method='independent_2d_landmark_translation_fit',
                source='student_temporal' if args.temporal != 'off' else 'student',
                min_confidence=args.alignment_min_confidence, landmarks=sequence.metadata,
                note='Only camera translation/depth fitted; pose/shape retained. Reprojection is not ground-truth 3D error.',
                timing='processing FPS includes forward, optional temporal filter/decode and fit; excludes offline detector, rendering and I/O')


def save_temporal_comparison(output, raw, filtered):
    keys = ('root_displacement_m', 'root_speed_m_per_s', 'pelvis_relative_joint_speed_m_per_s',
            'pelvis_relative_joint_acceleration_m_per_s2', 'phenotypes')
    write_json(output / 'temporal_comparison.json', {
        'comparison': 'same student predictions, before and after causal filtering',
        'caveat': 'Motion proxies, not ground-truth error. Smoothing can add lag and suppress true motion; inspect both videos.',
        'frames': raw['frames'], 'raw_selected_frames': raw['selected_frames'],
        'filtered_selected_frames': filtered['selected_frames'],
        'settings': filtered['processing'],
        'raw': {k: raw[k] for k in keys}, 'filtered': {k: filtered[k] for k in keys}})


def compose_comparison(output, kinds, frame_count, fps, size, benchmark_titles=None, filename='comparison.mp4'):
    import cv2
    import numpy as np
    captures = []
    writer = None
    try:
        for name in ['input'] + kinds:
            cap = cv2.VideoCapture(str(output / (name + '.mp4')))
            captures.append(cap)
            if not cap.isOpened():
                raise RuntimeError('Cannot read encoded {}.mp4'.format(name))
        writer = VideoWriter(output / filename, fps, size * len(captures), size + 64)
        for _ in range(frame_count):
            panels = [cap.read() for cap in captures]
            if not all(ok for ok, _ in panels):
                raise RuntimeError('Encoded video has missing frames; comparison is incomplete')
            # The comparison displays the final clip averages on every frame.
            # Individual overlay videos retain their running averages.
            if benchmark_titles:
                for kind, (_, panel) in zip(kinds, panels[1:]):
                    draw_title(panel, benchmark_titles[kind])
            writer.write(cv2.cvtColor(np.concatenate([p for _, p in panels], axis=1), cv2.COLOR_BGR2RGB))
        if any(cap.read()[0] for cap in captures):
            raise RuntimeError('Comparison panels contain different frame counts')
    finally:
        if writer:
            writer.close()
        for cap in captures:
            cap.release()


def save_timing_comparison(output, models):
    fields = ['model', 'device_name', 'input_size', 'precision', 'measured_frames', 'warmup_frames_excluded',
              'inference_fps', 'playback_fps', 'inference_mean_ms', 'inference_median_ms', 'inference_p95_ms',
              'preprocess_mean_ms', 'output_transfer_mean_ms', 'render_mean_ms', 'rendered_frames_measured',
              'temporal_mean_ms', 'alignment_mean_ms', 'processing_mean_ms', 'processing_fps']
    with (output / 'timing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for kind, model in models.items():
            timing = model['summary']['timing']
            row = dict(model=kind, **{k: model.get(k) for k in ('device_name', 'input_size', 'precision')})
            row.update({k: timing[k] for k in ('measured_frames', 'warmup_frames_excluded', 'inference_fps', 'playback_fps')})
            row.update(inference_mean_ms=timing['inference_ms']['mean'],
                       inference_median_ms=timing['inference_ms']['median'], inference_p95_ms=timing['inference_ms']['p95'],
                       preprocess_mean_ms=timing['preprocess_ms']['mean'],
                       output_transfer_mean_ms=timing['output_transfer_ms']['mean'], render_mean_ms=timing['render_ms']['mean'],
                       rendered_frames_measured=timing['render_ms']['count'],
                       temporal_mean_ms=timing['temporal_ms']['mean'],
                       alignment_mean_ms=timing['alignment_ms']['mean'],
                       processing_mean_ms=timing['processing_ms']['mean'], processing_fps=timing['processing_fps'])
            writer.writerow(row)


def run(args, predictor_factory=load_predictor, renderer_factory=MeshRenderer):
    validate_args(args)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    try:
        info = video_info(args.video)
    except ImportError as exc:
        raise RuntimeError('Install optional video I/O with: python -m pip install -r requirements-video.txt') from exc
    if args.start_frame >= info['reported_frame_count'] > 0:
        raise ValueError('--start-frame lies beyond the video frame count')
    landmark_sequence = None
    if args.alignment == 'landmarks':
        from scripts.video.landmark_observations import LandmarkSequence
        landmark_sequence = LandmarkSequence(args.landmarks, args.video, info)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', video=str(args.video), source=info,
                    video_bytes=args.video.stat().st_size, video_mtime_ns=args.video.stat().st_mtime_ns,
                    settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    smoothing=args.temporal != 'off',
                    temporal=temporal_settings(args) if args.temporal != 'off' else {'method': 'off'},
                    alignment=alignment_settings(args, landmark_sequence) if landmark_sequence else {'method': 'off'},
                    placement='raw camera-space prediction; optional temporal head-anchor filtering and separate landmark-fit Student; no box anchoring',
                    timing_note='Inference FPS = measured frames / summed synchronized forward seconds after warm-up; batch size 1. Model forward includes detection and Anny decoding. Preprocessing, transfers, rendering and video I/O excluded; forward_ms retains inclusive preprocessing+forward+transfer latency.',
                    tracking='One primary person per model, selected independently; inspect identity in comparison',
                    models={}, outputs=['input.mp4', 'comparison.mp4', 'timing.csv'])
    write_json(args.output / 'run.json', manifest)
    jobs = [('student', 'Student', args.student_checkpoint, args.resolved_student_fix)]
    if not args.student_only:
        jobs.append(('baseline', 'Original Multi-HMR (NAVER Anny)', args.baseline_checkpoint, 'none'))
    if args.teacher_checkpoint:
        jobs.append(('teacher', 'Final teacher', args.teacher_checkpoint, 'checkpoint'))
    reference = None
    try:
        for kind, title, checkpoint, fix in jobs:
            predictor = predictor_factory(checkpoint, kind, fix, args.device, args.amp)
            manifest['models'][kind] = predictor.metadata
            write_json(args.output / 'run.json', manifest)
            indices, summary = process_model(args, kind, title, predictor, info, renderer_factory, landmark_sequence)
            if reference is not None and indices != reference:
                raise RuntimeError('Decoded source frame sequence changed between model passes')
            reference = indices
            manifest['models'][kind]['summary'] = summary
            manifest['outputs'].extend([kind + '.mp4', kind + '/frames.csv',
                                        kind + '/predictions.npz', kind + '/summary.json'])
            if 'temporal' in summary:
                manifest['models']['student_temporal'] = dict(
                    predictor.metadata, source='student: same forward predictions',
                    summary=summary['temporal'], temporal=temporal_settings(args))
                manifest['outputs'].extend(['student_temporal.mp4', 'student_temporal/frames.csv',
                                            'student_temporal/predictions.npz', 'student_temporal/summary.json',
                                            'temporal_comparison.mp4', 'temporal_comparison.json'])
            if 'alignment' in summary:
                manifest['models']['student_aligned'] = dict(
                    predictor.metadata, source='student: same forward predictions',
                    summary=summary['alignment'], alignment=alignment_settings(args, landmark_sequence))
                manifest['outputs'].extend(['student_aligned.mp4', 'student_aligned/frames.csv',
                                            'student_aligned/predictions.npz', 'student_aligned/summary.json',
                                            'student_aligned/alignment.json', 'alignment_comparison.mp4'])
            # Only one network occupies the GPU; the next pass reads the same source video.
            del predictor
            gc.collect()
            if 'torch' in sys.modules:
                sys.modules['torch'].cuda.empty_cache()
            write_json(args.output / 'run.json', manifest)
        save_timing_comparison(args.output, manifest['models'])
        benchmark_titles = {kind: fps_title(title, manifest['models'][kind]['summary']['timing']['inference_fps'])
                            for kind, title, _, _ in jobs}
        kinds = [job[0] for job in jobs]
        if args.temporal != 'off':
            kinds.insert(1, 'student_temporal')
            benchmark_titles['student'] = fps_title('Student raw', manifest['models']['student']['summary']['timing']['inference_fps'])
            benchmark_titles['student_temporal'] = temporal_fps_title(
                manifest['models']['student_temporal']['summary']['timing']['processing_fps'])
            compose_comparison(args.output, ['student', 'student_temporal'], len(reference),
                               info['fps'] / args.stride, args.panel_size, benchmark_titles,
                               filename='temporal_comparison.mp4')
        if args.alignment != 'off':
            before_alignment = 'student_temporal' if args.temporal != 'off' else 'student'
            kinds.insert(kinds.index(before_alignment) + 1, 'student_aligned')
            benchmark_titles['student_aligned'] = alignment_fps_title(
                manifest['models']['student_aligned']['summary']['timing']['processing_fps'])
            compose_comparison(args.output, [before_alignment, 'student_aligned'], len(reference),
                               info['fps'] / args.stride, args.panel_size, benchmark_titles,
                               filename='alignment_comparison.mp4')
        compose_comparison(args.output, kinds, len(reference), info['fps'] / args.stride,
                           args.panel_size, benchmark_titles)
        manifest.update(status='completed', processed_frames=len(reference), output_fps=info['fps'] / args.stride)
    except BaseException as exc:
        manifest.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
        write_json(args.output / 'run.json', manifest)
        raise
    write_json(args.output / 'run.json', manifest)
    print('Completed: {}'.format(args.output / 'comparison.mp4'), flush=True)
    return manifest


if __name__ == '__main__':
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit('ERROR: {}'.format(error))
