"""Causal depth stabilization after the independent landmark fit."""
import math
import numpy as np
from scripts.video.landmark_alignment import ANNY_LANDMARK_BONES, _copy_person, _project, _error


class AlignmentDepthFilter:
    """Filter absolute pelvis depth in log space, retaining its current image position.

    A 0.5/second log-depth rate limit suppresses large single-frame scale jumps.
    Zero tau disables both smoothing and the rate limit. Rejected fits reset
    history: an old correction is never substituted for missing observations.
    """
    def __init__(self, tau=.35, max_gap=.25):
        if not math.isfinite(tau) or tau < 0 or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('Depth tau must be finite/nonnegative and max gap positive')
        self.tau, self.max_gap = tau, max_gap
        self.reset()

    def reset(self):
        self.previous = None

    def apply(self, source, fitted, report, joint_names, camera, size, timestamp, segment):
        report = dict(report)
        report['depth_filter_state'] = 'reset_rejected'
        if not report['accepted']:
            self.reset()
            return fitted, report
        depth = float(fitted['joints'][0, 2])
        target = math.log(depth)
        previous = self.previous
        dt = timestamp - previous[1] if previous else 0.
        continuous = previous and previous[2] == segment and 0 < dt <= self.max_gap
        state = 'initialized'
        value = target
        if self.tau == 0:
            state = 'disabled'
        elif continuous:
            step = -math.expm1(-dt / self.tau) * (target - previous[0])
            value = previous[0] + float(np.clip(step, -.5 * dt, .5 * dt))
            state = 'smoothed'
        # Translate the whole body, keeping the pelvis on its current camera ray.
        # This preserves all bone lengths and follows current lateral motion.
        root = np.asarray(fitted['joints'][0], dtype=float)
        adjustment = root * (math.exp(value) / depth - 1.) if state == 'smoothed' else np.zeros(3)
        output = _copy_person(fitted)
        for key in ('vertices', 'joints', 'anchor', 'root'):
            if key in output:
                output[key] = (output[key] + adjustment).astype(output[key].dtype)
        xy = _project(output['vertices'], camera)
        low, high = np.maximum(xy.min(0), 0), np.minimum(xy.max(0), size)
        total_delta = np.asarray(report['delta_xyz_m']) + adjustment
        source_depth = float(np.median(source['joints'][:, 2]))
        if (not np.isfinite(xy).all() or np.any(output['vertices'][:, 2] <= .01)
                or np.any(output['joints'][:, 2] <= .01) or np.any(high <= low)
                or np.any(np.abs(total_delta[:2]) >= .5 * source_depth)
                or not -.5 * source_depth < total_delta[2] < source_depth):
            self.reset()
            report['depth_filter_state'] = 'reset_unsafe'
            return fitted, report
        output['bbox'] = np.r_[low, high]
        self.previous = (value, timestamp, segment)
        # Scores must describe the mesh actually rendered, even if stabilization
        # trades some current-frame reprojection accuracy for less scale flicker.
        names = report['selected_names']
        indices = {name: i for i, name in enumerate(joint_names)}
        points = source['joints'][[indices[ANNY_LANDMARK_BONES[n]] for n in names]]
        observed = np.asarray(report['observed_xy'])
        confidence = np.asarray(report['confidence'])
        selected = np.array([n in report['inlier_names'] for n in names])
        report.update(depth_filter_state=state, fitted_depth_m=depth,
                      rendered_depth_m=float(output['joints'][0, 2]),
                      unsmoothed_delta_xyz_m=report['delta_xyz_m'],
                      unsmoothed_after_error_px=report['after_error_px'],
                      delta_xyz_m=total_delta.tolist(),
                      after_xy=_project(points + total_delta, camera).tolist(),
                      after_error_px=_error(points[selected], observed[selected], confidence[selected], camera, total_delta),
                      all_after_error_px=_error(points, observed, confidence, camera, total_delta))
        return output, report
