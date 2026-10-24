"""Extract independent 2D body landmarks from an original video on CPU.

Use a separate environment with requirements-video-landmarks.txt. The default
Solutions backend uses its bundled full-pose model. The optional Tasks backend
requires an existing .task file. This script downloads nothing.
Coordinates refer to the original decoded image, before any model letterbox or
video title strip. Missing detections remain empty observations.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


LANDMARK_INDICES = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path, help='Original source video, not a comparison montage')
    parser.add_argument('--backend', choices=('solutions', 'tasks'), default='solutions')
    parser.add_argument('--model', type=Path, help='Existing .task file; required only with --backend tasks')
    parser.add_argument('--output', required=True, type=Path, help='New JSON output; existing files are refused')
    parser.add_argument('--max-frames', type=int, default=0, help='Extract the first N frames; 0 means all')
    parser.add_argument('--num-poses', type=int, default=1, choices=(1, 2),
                        help='Tasks supports 1 or 2; Solutions supports only 1')
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_timestamp_ms(frame_index, fps, previous_ms=None):
    """CFR timestamp in milliseconds, strictly increasing for the VIDEO API."""
    if frame_index < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Frame index must be nonnegative and FPS finite and positive')
    timestamp = int(round(1000 * frame_index / fps))
    return timestamp if previous_ms is None else max(timestamp, previous_ms + 1)


def _point(landmark, width, height, visibility_only=False):
    try:
        x, y = float(landmark.x) * width, float(landmark.y) * height
        visibility = float(landmark.visibility)
        presence = 1. if visibility_only else float(landmark.presence)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x, y, visibility, presence)):
        return None
    return [x, y, max(0., min(1., visibility, presence))]


def people_from_result(result, width, height, visibility_only=False):
    """Convert normalized MediaPipe results to finite original-image points.

    Out-of-frame landmark coordinates are retained, rather than moved onto the
    image boundary. Bboxes are clipped for association and use all confidently
    observed landmarks, including the face and feet where available.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    people = []
    poses = result.pose_landmarks
    if poses is None:
        return people
    if visibility_only:
        poses = [poses.landmark]
    for pose in poses:
        points = [_point(point, width, height, visibility_only) for point in pose]
        named = {name: points[index] for name, index in LANDMARK_INDICES.items()
                 if index < len(points) and points[index] is not None}
        reliable = [point for point in points if point is not None and point[2] >= .2]
        if not named or len(reliable) < 2:
            continue
        xmin = max(0., min(point[0] for point in reliable))
        ymin = max(0., min(point[1] for point in reliable))
        xmax = min(float(width), max(point[0] for point in reliable))
        ymax = min(float(height), max(point[1] for point in reliable))
        if xmax <= xmin or ymax <= ymin:
            continue
        people.append({'landmarks': named, 'bbox': [xmin, ymin, xmax, ymax]})
    return people


def write_json_exclusive(path, data):
    """Publish a complete strict-JSON file atomically without replacing a file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('Output already exists: {}'.format(path))
    # Serialize before creating files so nonfinite values leave no artifacts.
    encoded = json.dumps(data, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.' + path.name + '.',
                                         suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # link() is an atomic create-if-absent on the same filesystem. Unlike
        # replace(), it cannot overwrite a result created by a concurrent run.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def detector_options(mp, model, num_poses):
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=.5,
        min_pose_presence_confidence=.5,
        min_tracking_confidence=.5,
        output_segmentation_masks=False)


def bundled_solution_models(mp):
    """Require packaged assets before Pose can attempt its automatic download."""
    root = Path(mp.__file__).resolve().parent
    paths = [root / 'modules/pose_landmark/pose_landmark_full.tflite',
             root / 'modules/pose_detection/pose_detection.tflite']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('MediaPipe bundled model is missing: {}. Install the pinned wheel.'.format(path))
    return paths


def extract(args):
    if args.max_frames < 0:
        raise ValueError('--max-frames must be nonnegative')
    video = args.video.expanduser().resolve()
    model = args.model.expanduser().resolve() if args.model else None
    output = args.output.expanduser().resolve()
    if args.backend == 'tasks' and model is None:
        raise ValueError('--backend tasks requires --model with an existing .task file')
    if args.backend == 'solutions' and args.num_poses != 1:
        raise ValueError('The Solutions backend supports one person only; use --num-poses 1')
    if args.backend == 'solutions' and model is not None:
        raise ValueError('--model is only used with --backend tasks')
    for name, path in [('video', video)] + ([('model', model)] if model else []):
        if not path.is_file():
            raise FileNotFoundError('Missing {} file: {}'.format(name, path))
    if output.exists():
        raise FileExistsError('Output already exists: {}'.format(output))
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('Install requirements-video-landmarks.txt in a separate Python environment') from exc

    if args.backend == 'solutions':
        model_files = bundled_solution_models(mp)
        detector = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                         smooth_landmarks=True, enable_segmentation=False,
                                         min_detection_confidence=.5, min_tracking_confidence=.5)
    else:
        model_files = [model]
        detector = mp.tasks.vision.PoseLandmarker.create_from_options(detector_options(mp, model, args.num_poses))
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open source video: {}'.format(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video needs a finite positive FPS; convert to constant frame rate first')
        reported = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported = int(reported) if math.isfinite(reported) and reported > 0 else 0
        rows, previous_ms = [], None
        width = height = None
        reached_end = False
        # Invalid or incompatible task models fail here before any JSON is written.
        with detector:
            while not args.max_frames or len(rows) < args.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    reached_end = True
                    break
                current_height, current_width = bgr.shape[:2]
                if width is None:
                    width, height = current_width, current_height
                elif (current_width, current_height) != (width, height):
                    raise ValueError('Video dimensions changed between frames')
                index = len(rows)
                timestamp = frame_timestamp_ms(index, fps, previous_ms)
                previous_ms = timestamp
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                detector_started = time.perf_counter()
                if args.backend == 'solutions':
                    result = detector.process(rgb)
                else:
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, timestamp)
                detector_ms = (time.perf_counter() - detector_started) * 1000
                rows.append({'frame_index': index, 'timestamp_s': index / fps,
                             'detector_ms': detector_ms,
                             'people': people_from_result(result, width, height, args.backend == 'solutions')})
                if len(rows) % 100 == 0:
                    print('Extracted {} frames'.format(len(rows)), flush=True)
        if not rows:
            raise RuntimeError('Source video contains no decodable frames')
        if reached_end and reported > len(rows):
            raise RuntimeError('Video decoding stopped early: decoded {} of {} reported frames'.format(len(rows), reported))
        data = {
            'schema_version': 1,
            'coordinate_system': 'original_image_pixels',
            'video': {'path': str(video), 'width': width, 'height': height, 'fps': fps,
                      'frame_count': reported or len(rows), 'extracted_frames': len(rows),
                      'complete': reached_end or (reported > 0 and len(rows) == reported),
                      'bytes': video.stat().st_size, 'sha256': sha256_file(video),
                      'timing': 'constant_frame_rate_assumed'},
            'detector': {'name': 'MediaPipe Pose', 'version': str(mp.__version__), 'backend': args.backend,
                         'device': 'CPU', 'running_mode': 'VIDEO', 'num_poses': args.num_poses,
                         'models': [{'path': str(path), 'sha256': sha256_file(path)} for path in model_files],
                         'confidence': 'visibility' if args.backend == 'solutions' else 'min(visibility, presence)',
                         'landmark_smoothing': args.backend == 'solutions',
                         'timing': 'offline CPU detector wall time per frame; excludes video decode and JSON output',
                         'landmark_names': list(LANDMARK_INDICES)},
            'frames': rows,
        }
        write_json_exclusive(output, data)
        print('Saved {} frames to {}'.format(len(rows), output), flush=True)
        return data
    finally:
        cap.release()
        # Also release when metadata validation fails before entering its context.
        # MediaPipe close() is not guaranteed to be idempotent, so detect the
        # pre-context failure by whether the frame list was initialized.
        if 'rows' not in locals():
            detector.close()


if __name__ == '__main__':
    try:
        extract(parse_args())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit('ERROR: {}'.format(error))
