"""Causal parameter filtering for one tracked Anny body.

This is inference-time temporal stabilization, not a trained video model. It
keeps an exponentially weighted history of body parameters. Rotations are
interpolated on the unit quaternion sphere so the axis-angle discontinuity at
180 degrees cannot collapse a valid pose toward zero. The filtered parameters
must be decoded through Anny; averaging mesh vertices would deform the body.

The input pose must already include the checkpoint's rotation correction and
helper-bone masking, and shape values must already be sigmoid outputs. Anchor
coordinates are the camera-space head position in metres, not the pelvis.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation


def _finite_nonnegative(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('{} must be finite and nonnegative'.format(name)) from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError('{} must be finite and nonnegative'.format(name))
    return value


def _alpha(dt, tau):
    return 1.0 if tau == 0 else -math.expm1(-dt / tau)


def _slerp(previous, current, alpha):
    """Vectorized shortest-arc SLERP of normalized xyzw quaternions."""
    dot = np.sum(previous * current, axis=-1, keepdims=True)
    current = np.where(dot < 0, -current, current)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    omega = np.arccos(dot)
    sine = np.sin(omega)
    # Coincident rotations use normalized linear interpolation, avoiding 0/0.
    denominator = np.where(sine > 1e-8, sine, 1.0)
    spherical = (np.sin((1 - alpha) * omega) / denominator * previous
                 + np.sin(alpha * omega) / denominator * current)
    linear = (1 - alpha) * previous + alpha * current
    result = np.where(sine > 1e-8, spherical, linear)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


class TemporalBodyFilter:
    """Filter one body's pose, phenotype values and camera head anchor.

    Time constants are in seconds. Larger values provide stronger smoothing
    and more lag; zero disables filtering for that component. At dt=tau, the
    new observation has weight 1-exp(-1), approximately 0.632.

    ``update`` returns independent arrays and a ``state`` string. A new track,
    time reversal/duplicate timestamp, excessive time gap, or changed array
    dimensions starts a fresh state. Call ``reset`` on any missing detection or
    known scene cut. Invalid input raises ValueError and clears history; it is
    never replaced by an old body. Automatic scene-cut detection and identity
    recognition are not provided by this filter.
    """

    def __init__(self, pose_tau=.08, translation_tau=.12, shape_tau=1.0,
                 max_gap_s=.25):
        self.pose_tau = _finite_nonnegative('pose_tau', pose_tau)
        self.translation_tau = _finite_nonnegative('translation_tau', translation_tau)
        self.shape_tau = _finite_nonnegative('shape_tau', shape_tau)
        self.max_gap_s = _finite_nonnegative('max_gap_s', max_gap_s)
        self.reset()

    def reset(self):
        self._quaternion = None
        self._shape = None
        self._anchor = None
        self._timestamp_s = None
        self._segment_id = None

    @staticmethod
    def _validate(rotvec, shape, anchor, timestamp_s, segment_id):
        try:
            rotvec = np.asarray(rotvec, dtype=np.float64)
            shape = np.asarray(shape, dtype=np.float64)
            anchor = np.asarray(anchor, dtype=np.float64)
            timestamp_s = float(timestamp_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Temporal inputs must be finite numeric values') from exc
        if rotvec.ndim != 2 or rotvec.shape[1] != 3 or not len(rotvec):
            raise ValueError('rotvec must have shape (bones, 3), with at least one bone')
        if shape.ndim != 1 or not len(shape):
            raise ValueError('shape must be a nonempty phenotype vector')
        if anchor.shape != (3,):
            raise ValueError('anchor must have shape (3,) in camera coordinates')
        if (not all(np.isfinite(x).all() for x in (rotvec, shape, anchor))
                or not math.isfinite(timestamp_s)):
            raise ValueError('Temporal inputs must be finite numeric values')
        if np.any(shape < 0) or np.any(shape > 1):
            raise ValueError('shape must contain sigmoid phenotype values in [0, 1]')
        if anchor[2] <= 0:
            raise ValueError('anchor depth must be positive')
        if (not isinstance(segment_id, (str, int, np.integer))
                or isinstance(segment_id, (bool, np.bool_))):
            raise ValueError('segment_id must identify one track with an integer or string')
        return rotvec, shape, anchor, timestamp_s

    def update(self, rotvec, shape, anchor, timestamp_s, segment_id):
        try:
            rotvec, shape, anchor, timestamp_s = self._validate(
                rotvec, shape, anchor, timestamp_s, segment_id)
            quaternion = Rotation.from_rotvec(rotvec).as_quat()
            if not np.isfinite(quaternion).all():
                raise ValueError('rotvec cannot be represented as a finite rotation')
        except (ValueError, TypeError, OverflowError):
            self.reset()
            raise

        state = 'initialized'
        dt = None if self._timestamp_s is None else timestamp_s - self._timestamp_s
        if self._quaternion is not None:
            if segment_id != self._segment_id:
                state = 'reset_track'
            elif quaternion.shape != self._quaternion.shape or shape.shape != self._shape.shape:
                state = 'reset_dimensions'
            elif dt <= 0:
                state = 'reset_time'
            elif dt > self.max_gap_s:
                state = 'reset_gap'
            else:
                state = 'filtered'
                if self.pose_tau > 0:
                    quaternion = _slerp(self._quaternion, quaternion, _alpha(dt, self.pose_tau))
                    rotvec = Rotation.from_quat(quaternion).as_rotvec()
                if self.shape_tau > 0:
                    alpha = _alpha(dt, self.shape_tau)
                    shape = (1 - alpha) * self._shape + alpha * shape
                if self.translation_tau > 0:
                    alpha = _alpha(dt, self.translation_tau)
                    anchor = (1 - alpha) * self._anchor + alpha * anchor

        self._quaternion = quaternion.copy()
        self._shape = shape.copy()
        self._anchor = anchor.copy()
        self._timestamp_s = timestamp_s
        self._segment_id = segment_id
        return dict(rotvec=rotvec.copy(), shape=shape.copy(), anchor=anchor.copy(), state=state)
