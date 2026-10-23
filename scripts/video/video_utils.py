"""Tracking and unsmoothed motion summaries for video inference.

These metrics are stability/motion proxies, not reconstruction errors: genuine
body motion and camera motion can increase them. Inputs are NumPy arrays or
ordinary Python values; this module never changes or fills model predictions.
"""

import math

import numpy as np


def _array(value, shape=None):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if shape is not None and result.shape != shape:
        return None
    return result


def _box(value):
    result = _array(value, (4,))
    if result is None or not np.isfinite(result).all():
        return None
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _overlap_distance(previous, current):
    low = np.maximum(previous[:2], current[:2])
    high = np.minimum(previous[2:], current[2:])
    overlap = float(np.prod(np.maximum(high - low, 0.0)))
    area_a = float(np.prod(previous[2:] - previous[:2]))
    area_b = float(np.prod(current[2:] - current[:2]))
    iou = overlap / (area_a + area_b - overlap)
    center_a = (previous[:2] + previous[2:]) * 0.5
    center_b = (current[:2] + current[2:]) * 0.5
    distance = float(np.linalg.norm(center_b - center_a)
                     / np.linalg.norm(previous[2:] - previous[:2]))
    return iou, distance


class SinglePersonTracker:
    """Associate one person using boxes, with explicit gaps and new segments.

    ``update`` receives dictionaries containing ``bbox`` (xyxy) and
    ``confidence``. Returned ``selected_index`` refers to the original input
    list. The highest-confidence valid detection starts a segment. Subsequent
    detections must overlap or have a nearby center; IoU, then center distance,
    then confidence determine the match. Center distance is normalized by the
    previous box's diagonal. ``max_gap`` counts consecutive *processed updates*
    without a compatible detection, rather than source-video frames.

    After more than ``max_gap`` misses a new person may start a new segment.
    This geometric association cannot guarantee identity through crossings or
    long occlusions. No boxes or poses are interpolated or smoothed.
    """

    def __init__(self, max_gap=5, min_iou=0.05, max_center_distance=1.0):
        if int(max_gap) != max_gap or max_gap < 0:
            raise ValueError("max_gap must be a nonnegative integer")
        if not 0 <= min_iou <= 1:
            raise ValueError("min_iou must be between zero and one")
        if not math.isfinite(max_center_distance) or max_center_distance <= 0:
            raise ValueError("max_center_distance must be finite and positive")
        self.max_gap = int(max_gap)
        self.min_iou = float(min_iou)
        self.max_center_distance = float(max_center_distance)
        self._previous_box = None
        self._segment_id = 0
        self._missed = 0

    def update(self, detections):
        candidates = []
        for index, detection in enumerate(detections):
            bbox = _box(detection.get("bbox"))
            try:
                confidence = float(detection.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if bbox is not None and math.isfinite(confidence):
                candidates.append((index, bbox, confidence))

        chosen = None
        iou = distance = None
        status = "missing"
        gap_before_match = self._missed
        if self._previous_box is not None:
            matches = []
            for candidate in candidates:
                overlap, center_distance = _overlap_distance(self._previous_box, candidate[1])
                if ((overlap > 0 and overlap >= self.min_iou)
                        or center_distance <= self.max_center_distance):
                    matches.append((overlap, -center_distance, candidate[2], -candidate[0], candidate))
            if matches:
                best = max(matches, key=lambda item: item[:4])
                iou, distance, chosen = float(best[0]), float(-best[1]), best[4]
                status = "matched"
            else:
                if self._missed + 1 > self.max_gap:
                    self._previous_box = None

        if chosen is None and self._previous_box is None and candidates:
            chosen = max(candidates, key=lambda item: (item[2], -item[0]))
            status = "started" if self._segment_id == 0 else "reacquired"
            self._segment_id += 1

        if chosen is not None:
            self._previous_box = chosen[1].copy()
            self._missed = 0
        else:
            self._missed += 1
        return {
            "selected_index": None if chosen is None else int(chosen[0]),
            "segment_id": self._segment_id or None,
            "status": status,
            "missed_frames": int(self._missed),
            "gap_before_match": int(gap_before_match) if chosen is not None else None,
            "match_iou": iou,
            "center_distance": distance,
        }


def _stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p95": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_timing(records, warmup_frames, playback_fps):
    """Batch-one latency statistics; FPS is frames / total inference seconds.

    Warm-up frames remain in the videos and motion metrics. All subsequent
    inference calls count, including frames with no person. Rendering statistics
    only include frames where a mesh was actually drawn.
    """
    if int(warmup_frames) != warmup_frames or warmup_frames < 0:
        raise ValueError('warmup_frames must be a nonnegative integer')
    measured = records[int(warmup_frames):]
    stages = {}
    for key in ('preprocess_ms', 'inference_ms', 'output_transfer_ms', 'render_ms', 'temporal_ms', 'processing_ms'):
        values = [float(r[key]) for r in measured
                  if r.get(key) is not None and math.isfinite(float(r[key]))
                  and (float(r[key]) >= 0 if key == 'temporal_ms' else float(r[key]) > 0)]
        stages[key] = dict(_stats(values), median=float(np.median(values)) if values else None)
    mean_ms = stages['inference_ms']['mean']
    return dict(warmup_frames_requested=int(warmup_frames),
                warmup_frames_excluded=min(int(warmup_frames), len(records)),
                measured_frames=stages['inference_ms']['count'],
                frames_after_warmup=len(measured),
                inference_fps=1000.0 / mean_ms if mean_ms is not None else None,
                processing_fps=1000.0 / stages['processing_ms']['mean'] if stages['processing_ms']['mean'] else None,
                processing_definition='Student model forward + temporal filtering/Anny re-decode/geometry checks; excludes preprocessing, rendering and video I/O; reuses raw inference',
                playback_fps=float(playback_fps), batch_size=1,
                fps_definition='1000 / mean inference_ms; includes detection and Anny decoding; excludes preprocessing, transfers, rendering and video I/O',
                render_definition='Wall time for mesh rendering/compositing, only on frames with a matched person',
                **stages)


def _selected(record):
    return record.get("selected_index") is not None and record.get("segment_id") is not None


def _pair_dt(previous, current, expected_frame_step):
    if not _selected(previous) or not _selected(current):
        return None
    if previous["segment_id"] != current["segment_id"]:
        return None
    try:
        frame_delta = current["frame_index"] - previous["frame_index"]
        dt = float(current["timestamp_s"]) - float(previous["timestamp_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_delta != expected_frame_step or not math.isfinite(dt) or dt <= 0:
        return None
    return dt


def _relative_joints(record, pelvis_index):
    joints = _array(record.get("joints"))
    if joints is None or joints.ndim != 2 or joints.shape[1] != 3:
        return None
    if not 0 <= pelvis_index < len(joints):
        return None
    if not np.isfinite(joints[pelvis_index]).all():
        return None
    joints = np.where(np.isfinite(joints), joints, np.nan)
    # Exclude the identically zero pelvis from aggregate motion statistics.
    return np.delete(joints - joints[pelvis_index], pelvis_index, axis=0)


def _finite_norms(vectors):
    return np.linalg.norm(vectors[np.isfinite(vectors).all(axis=-1)], axis=-1).tolist()


def _rotation_steps(previous, current):
    a, b = _array(previous.get("rotmat")), _array(current.get("rotmat"))
    if a is None or b is None or a.shape != b.shape or a.ndim < 2 or a.shape[-2:] != (3, 3):
        return []
    a, b = a.reshape(-1, 3, 3), b.reshape(-1, 3, 3)
    valid = np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
    a, b = a[valid], b[valid]
    if not len(a):
        return []
    # Reject malformed matrices instead of turning them into plausible angles.
    valid = ((np.max(np.abs(a @ a.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2)) < 1e-3)
             & (np.max(np.abs(b @ b.transpose(0, 2, 1) - np.eye(3)), axis=(1, 2)) < 1e-3)
             & (np.abs(np.linalg.det(a) - 1) < 1e-3)
             & (np.abs(np.linalg.det(b) - 1) < 1e-3))
    relative = b[valid] @ a[valid].transpose(0, 2, 1)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).tolist()


def summarize_stability(records, phenotype_names=None, expected_frame_step=1, pelvis_index=0):
    """Return a JSON-safe summary of unsmoothed predictions in chronological order.

    Every record supplies ``frame_index``, ``timestamp_s``, ``segment_id`` and
    ``selected_index`` (None for a miss). Optional prediction arrays are
    ``shape`` [P], ``root`` [3] in metres, ``joints`` [J, 3] in camera metres,
    and ``rotmat`` [J, 3, 3]. Times use the source video's clock. Temporal
    statistics only use adjacent input records with the same segment, positive
    dt, and exactly ``expected_frame_step`` source frames between them.

    Shape statistics use finite values per phenotype. Joint statistics pool
    finite joint/time samples, excluding the pelvis. Acceleration uses velocity
    differences divided by the time between velocity midpoints, so unequal dt
    is supported. Rotation statistics pool valid bone/time samples.
    """
    if int(expected_frame_step) != expected_frame_step or expected_frame_step <= 0:
        raise ValueError("expected_frame_step must be a positive integer")
    records = list(records)
    selected = [row for row in records if _selected(row)]
    shapes = []
    shape_size = None
    for row in selected:
        shape = _array(row.get("shape"))
        if shape is None or shape.ndim != 1:
            continue
        if shape_size is None:
            shape_size = len(shape)
        if len(shape) != shape_size:
            raise ValueError("shape dimensions changed between frames")
        shapes.append(shape)
    names = list(phenotype_names) if phenotype_names is not None else [
        "shape_{}".format(index) for index in range(shape_size or 0)]
    if shape_size is not None and len(names) != shape_size:
        raise ValueError("phenotype_names must match the shape dimension")
    if len(set(names)) != len(names):
        raise ValueError("phenotype_names must be unique")

    deltas = [[] for _ in names]
    rates = [[] for _ in names]
    root_steps, root_speeds, joint_speeds, joint_accelerations = [], [], [], []
    rotation_steps, rotation_rates = [], []
    n_pairs = n_triplets = 0
    previous_velocity = None
    previous_dt = None
    for previous, current in zip(records, records[1:]):
        dt = _pair_dt(previous, current, expected_frame_step)
        if dt is None:
            previous_velocity = previous_dt = None
            continue
        n_pairs += 1
        shape_a, shape_b = _array(previous.get("shape")), _array(current.get("shape"))
        if shape_a is not None and shape_b is not None and shape_a.shape == shape_b.shape == (len(names),):
            for index, delta in enumerate(shape_b - shape_a):
                if np.isfinite(delta):
                    deltas[index].append(float(delta))
                    rates[index].append(float(abs(delta) / dt))

        root_a, root_b = _array(previous.get("root"), (3,)), _array(current.get("root"), (3,))
        if root_a is not None and root_b is not None and np.isfinite(root_a).all() and np.isfinite(root_b).all():
            step = float(np.linalg.norm(root_b - root_a))
            root_steps.append(step)
            root_speeds.append(step / dt)

        joints_a, joints_b = _relative_joints(previous, pelvis_index), _relative_joints(current, pelvis_index)
        velocity = None
        if joints_a is not None and joints_b is not None and joints_a.shape == joints_b.shape:
            velocity = (joints_b - joints_a) / dt
            joint_speeds.extend(_finite_norms(velocity))
            if previous_velocity is not None and previous_velocity.shape == velocity.shape:
                acceleration = (velocity - previous_velocity) / (0.5 * (dt + previous_dt))
                magnitudes = _finite_norms(acceleration)
                n_triplets += int(bool(magnitudes))
                joint_accelerations.extend(magnitudes)
        previous_velocity, previous_dt = velocity, dt
        angles = _rotation_steps(previous, current)
        rotation_steps.extend(angles)
        rotation_rates.extend([angle / dt for angle in angles])

    phenotype_summary = {}
    shape_matrix = np.stack(shapes) if shapes else np.empty((0, len(names)))
    for index, name in enumerate(names):
        phenotype_summary[str(name)] = {
            "value": _stats(shape_matrix[:, index]),
            "signed_step": _stats(deltas[index]),
            "absolute_step": _stats(np.abs(deltas[index])),
            "absolute_rate_per_s": _stats(rates[index]),
        }
    return {
        "metric_kind": "motion_and_stability_proxies_not_ground_truth_error",
        "caveat": "Natural body motion and camera motion can increase these values. Predictions are unsmoothed; missing frames and track boundaries break temporal differences.",
        "frames": len(records),
        "selected_frames": len(selected),
        "missing_frames": len(records) - len(selected),
        "segments": len({row["segment_id"] for row in selected}),
        "expected_frame_step": int(expected_frame_step),
        "consecutive_selected_pairs": n_pairs,
        "joint_acceleration_triplets": n_triplets,
        "phenotypes": phenotype_summary,
        "root_displacement_m": _stats(root_steps),
        "root_speed_m_per_s": _stats(root_speeds),
        "pelvis_relative_joint_speed_m_per_s": _stats(joint_speeds),
        "pelvis_relative_joint_acceleration_m_per_s2": _stats(joint_accelerations),
        "bone_rotation_step_degrees": _stats(rotation_steps),
        "bone_rotation_speed_degrees_per_s": _stats(rotation_rates),
        "joint_aggregation": "Finite joint/time samples, with the pelvis excluded; shape mean/std pool selected frames across segments.",
    }
