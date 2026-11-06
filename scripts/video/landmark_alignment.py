"""Refine camera-space placement using independent 2D body landmarks.

Only one rigid 3D translation is fitted. Moving the existing body farther from
the camera reduces its projected size while retaining pose, shape and all
inter-point distances. This is image-space fitting, not new model supervision
or evidence of improved ground-truth 3D accuracy.

MediaPipe landmark centres and Anny bone heads are approximate anatomical
correspondences. The mapping below was checked against Anny full-body labels
and rest-pose positions: upperarm01 begins at the shoulder, lowerarm01 at the
elbow, wrist at the wrist; upperleg01/lowerleg01/foot begin at hip/knee/ankle.
"""

import math

import numpy as np
from scipy.optimize import least_squares


ANNY_LANDMARK_BONES = {
    '{}_{}'.format(side, joint): '{}.{}'.format(bone, suffix)
    for side, suffix in [('left', 'L'), ('right', 'R')]
    for joint, bone in [('shoulder', 'upperarm01'), ('elbow', 'lowerarm01'),
                        ('wrist', 'wrist'), ('hip', 'upperleg01'),
                        ('knee', 'lowerleg01'), ('ankle', 'foot')]
}


def _project(points, camera):
    homogeneous = points @ camera.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _error(points, observations, confidence, camera, delta):
    residual = _project(points + delta, camera) - observations
    return float(np.sqrt(np.sum(confidence * np.sum(residual ** 2, axis=1)) / confidence.sum()))


def _coverage(names):
    # An entire visible side is enough. Both anatomical sides are often hidden
    # behind each other in a side-on walking video.
    return (len(names) >= 4 and any(n.endswith('_shoulder') for n in names)
            and any(n.endswith('_hip') for n in names)
            and any(n.endswith(('_knee', '_ankle')) for n in names))


def _copy_person(person):
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in person.items()}


def align_to_landmarks(person, joint_names, landmarks, K, panel_size, min_confidence=.5):
    """Return ``(person_copy, report)`` after conservative image-space fitting.

    ``landmarks`` is a list of dictionaries with ``name``, ``x``, ``y`` and
    ``confidence``; coordinates refer to the square, letterboxed rendering
    panel, before its title strip. ``K`` is that panel's 3x3 camera matrix.
    The caller must associate landmarks with the same tracked person first.

    Weak/insufficient observations, poor fits or unsafe geometry preserve the
    raw body and report ``accepted=False``. Four confident points spanning a
    shoulder, hip and lower leg are required. Outliers are removed after an
    initial robust fit; the reported before/after error uses the same retained
    points and weights. All-point error and rejected landmark names are also
    recorded, so rejection cannot silently make a score look better.

    Depth is bounded to 0.5--2 times the current median joint depth, lateral
    shifts to half that depth, and all vertices stay in front of the camera.
    Arrays and parameters in ``person`` are never changed in place.
    """
    camera = np.asarray(K, dtype=np.float64)
    if (camera.shape != (3, 3) or not np.isfinite(camera).all()
            or camera[0, 0] <= 0 or camera[1, 1] <= 0
            or not np.allclose(camera[2], [0., 0., 1.])):
        raise ValueError('K must be a finite perspective camera matrix with positive focal lengths')
    if not math.isfinite(panel_size) or panel_size <= 0:
        raise ValueError('panel_size must be finite and positive')
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence must be in [0, 1]')
    output = _copy_person(person)
    report = dict(accepted=False, status='insufficient_landmarks', delta_xyz_m=[0., 0., 0.],
                  before_error_px=None, after_error_px=None, all_before_error_px=None,
                  all_after_error_px=None, landmark_count=0, inlier_count=0,
                  selected_names=[], inlier_names=[], rejected_names=[], observed_xy=[],
                  before_xy=[], after_xy=[], confidence=[])

    def rejected(status):
        report['status'] = status
        return output, report

    try:
        joints = np.asarray(person['joints'], dtype=np.float64)
        vertices = np.asarray(person['vertices'], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return rejected('invalid_geometry')
    if (joints.ndim != 2 or joints.shape != (len(joint_names), 3)
            or vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(joints).all() or not np.isfinite(vertices).all()
            or np.any(joints[:, 2] <= .01) or np.any(vertices[:, 2] <= .01)):
        return rejected('invalid_geometry')
    if len(set(joint_names)) != len(joint_names):
        raise ValueError('joint_names must be unique')
    indices = {name: index for index, name in enumerate(joint_names)}
    # Keep the strongest observation if a source accidentally repeats a name.
    observations = {}
    for item in landmarks:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if name not in ANNY_LANDMARK_BONES or ANNY_LANDMARK_BONES[name] not in indices:
            continue
        try:
            x, y, confidence = float(item['x']), float(item['y']), float(item['confidence'])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (not all(math.isfinite(v) for v in (x, y, confidence))
                or not 0 < confidence <= 1 or confidence < min_confidence
                or not 0 <= x <= panel_size or not 0 <= y <= panel_size):
            continue
        if name not in observations or confidence > observations[name][2]:
            observations[name] = (x, y, confidence)
    names = list(observations)
    report['landmark_count'] = len(names)
    report['selected_names'] = names
    if not _coverage(names):
        return rejected('insufficient_landmarks')
    points = joints[[indices[ANNY_LANDMARK_BONES[n]] for n in names]]
    target = np.array([observations[n][:2] for n in names])
    confidence = np.array([observations[n][2] for n in names])
    height = float(np.ptp(target[:, 1]))
    if height < max(30., panel_size * .06):
        return rejected('insufficient_vertical_extent')
    before = _project(points, camera)
    report.update(observed_xy=target.tolist(), before_xy=before.tolist(), after_xy=before.tolist(),
                  confidence=confidence.tolist())
    median_depth = float(np.median(joints[:, 2]))
    minimum_depth = float(min(vertices[:, 2].min(), joints[:, 2].min()))
    lateral_bound = .5 * median_depth
    lower = np.array([-lateral_bound, -lateral_bound, max(-.5 * median_depth, .011 - minimum_depth)])
    upper = np.array([lateral_bound, lateral_bound, median_depth])
    # If even the initial corresponding body is far from the observation,
    # decline to translate it onto a potentially unrelated person.
    raw_height = max(30., float(np.ptp(before[:, 1])))
    centre_gap = np.linalg.norm(np.average(before, axis=0, weights=confidence)
                                - np.average(target, axis=0, weights=confidence))
    if centre_gap > max(60., .65 * max(height, raw_height)):
        return rejected('landmarks_too_far')
    all_before = _error(points, target, confidence, camera, np.zeros(3))
    report.update(all_before_error_px=all_before, all_after_error_px=all_before)

    def fit(selected, initial):
        def residual(delta):
            xy = _project(points[selected] + delta, camera)
            return ((xy - target[selected]) * np.sqrt(confidence[selected, None])).reshape(-1)
        return least_squares(residual, initial, bounds=(lower, upper),
                             loss='soft_l1', f_scale=8., max_nfev=100)

    try:
        selected = np.ones(len(names), dtype=bool)
        result = fit(selected, np.zeros(3))
        if not result.success or not np.isfinite(result.x).all():
            return rejected('optimization_failed')
        distances = np.linalg.norm(_project(points + result.x, camera) - target, axis=1)
        selected = distances <= max(15., .12 * height)
        inlier_names = [n for n, keep in zip(names, selected) if keep]
        report.update(inlier_count=int(selected.sum()), inlier_names=inlier_names,
                      rejected_names=[n for n, keep in zip(names, selected) if not keep])
        if not _coverage(inlier_names) or float(np.ptp(target[selected, 1])) < max(30., .06 * panel_size):
            return rejected('insufficient_consistent_landmarks')
        if not selected.all():
            result = fit(selected, result.x)
        if not result.success or not np.isfinite(result.x).all():
            return rejected('optimization_failed')
        delta = result.x
        before_error = _error(points[selected], target[selected], confidence[selected], camera, np.zeros(3))
        after_error = _error(points[selected], target[selected], confidence[selected], camera, delta)
        all_after = _error(points, target, confidence, camera, delta)
        report.update(before_error_px=before_error, after_error_px=before_error)
        if before_error <= .5:
            return rejected('already_aligned')
        if after_error > max(12., .08 * height):
            return rejected('poor_fit')
        if after_error >= before_error - max(.25, .02 * before_error) or all_after > all_before + .5:
            return rejected('no_improvement')
        if np.any(delta <= lower + 1e-5) or np.any(delta >= upper - 1e-5):
            return rejected('at_search_bound')
        moved_vertices = vertices + delta
        moved_joints = joints + delta
        xy = _project(moved_vertices, camera)
        low, high = np.maximum(xy.min(0), 0), np.minimum(xy.max(0), panel_size)
        if (not np.isfinite(xy).all() or np.any(moved_vertices[:, 2] <= .01)
                or np.any(high <= low)):
            return rejected('invalid_aligned_geometry')
    except (ValueError, FloatingPointError, OverflowError):
        return rejected('optimization_failed')

    for key in ('vertices', 'joints', 'anchor', 'root'):
        if key in person:
            original = np.asarray(person[key])
            value = np.asarray(original, dtype=np.float64) + delta
            output[key] = value.astype(original.dtype if original.dtype.kind == 'f' else np.float64)
    output['bbox'] = np.r_[low, high]
    report.update(accepted=True, status='aligned', delta_xyz_m=delta.tolist(),
                  after_error_px=after_error, all_after_error_px=all_after,
                  after_xy=_project(points + delta, camera).tolist())
    return output, report
