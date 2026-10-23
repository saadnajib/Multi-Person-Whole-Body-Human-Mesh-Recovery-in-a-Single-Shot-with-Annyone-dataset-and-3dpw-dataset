"""Physical rotation and continuity checks for causal body stabilization."""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.video.temporal_filter import TemporalBodyFilter


class TemporalBodyFilterTests(unittest.TestCase):
    def observation(self, filter_, t, rotation=0., shape=.5, x=0., track=1):
        return filter_.update(np.array([[0., 0., rotation], [0., 0., 0.]]),
                              np.array([shape, .7]), np.array([x, 0., 3.]), t, track)

    def test_near_pi_rotations_take_short_arc_and_preserve_rigid_rotations(self):
        f = TemporalBodyFilter(pose_tau=.1)
        self.observation(f, 0., rotation=np.deg2rad(179))
        result = self.observation(f, .1 * math.log(2), rotation=np.deg2rad(-179))
        matrices = Rotation.from_rotvec(result['rotvec']).as_matrix()
        expected = Rotation.from_rotvec([0, 0, math.pi]).as_matrix()
        np.testing.assert_allclose(matrices[0], expected, atol=1e-10)
        np.testing.assert_allclose(matrices[1], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrices @ matrices.transpose(0, 2, 1),
                                   np.broadcast_to(np.eye(3), matrices.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(matrices), 1, atol=1e-12)

    def test_alternating_noise_is_reduced_for_pose_shape_and_placement(self):
        f = TemporalBodyFilter()
        raw, filtered = [], []
        for i in range(180):
            noise = .1 if i % 2 else -.1
            raw.append([noise, .5 + noise, noise])
            r = self.observation(f, i / 30., rotation=noise, shape=.5 + noise, x=noise)
            filtered.append([r['rotvec'][0, 2], r['shape'][0], r['anchor'][0]])
        raw, filtered = np.asarray(raw)[60:], np.asarray(filtered)[60:]
        self.assertTrue(np.all(filtered.std(0) < raw.std(0) * .3))

    def test_timestamp_weighting_is_independent_of_subdivision_for_constant_target(self):
        coarse, fine = TemporalBodyFilter(max_gap_s=1), TemporalBodyFilter(max_gap_s=1)
        for f in (coarse, fine):
            self.observation(f, 0., rotation=0., shape=.2, x=0.)
        a = self.observation(coarse, .2, rotation=.6, shape=.8, x=1.)
        for t in np.linspace(.02, .2, 10):
            b = self.observation(fine, t, rotation=.6, shape=.8, x=1.)
        for key in ('rotvec', 'shape', 'anchor'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-12)

    def test_zero_time_constants_pass_through_current_parameters(self):
        f = TemporalBodyFilter(pose_tau=0, translation_tau=0, shape_tau=0)
        self.observation(f, 0., rotation=.5, shape=.2, x=1.)
        r = self.observation(f, .03, rotation=4., shape=.8, x=-1.)
        np.testing.assert_array_equal(r['rotvec'], [[0., 0., 4.], [0., 0., 0.]])
        np.testing.assert_array_equal(r['shape'], [.8, .7])
        np.testing.assert_array_equal(r['anchor'], [-1., 0., 3.])

    def test_reset_on_track_change_does_not_mix_people(self):
        f = TemporalBodyFilter()
        self.observation(f, 0., shape=.1, x=0.)
        r = self.observation(f, .03, shape=.9, x=4., track=2)
        self.assertEqual(r['state'], 'reset_track')
        np.testing.assert_array_equal(r['shape'], [.9, .7])
        self.assertEqual(r['anchor'][0], 4.)

    def test_explicit_missing_reset_and_long_gap_start_fresh(self):
        f = TemporalBodyFilter()
        self.observation(f, 0.)
        r = self.observation(f, .3, x=2.)
        self.assertEqual(r['state'], 'reset_gap')
        self.assertEqual(r['anchor'][0], 2.)
        f.reset()
        r = self.observation(f, .33, x=4.)
        self.assertEqual(r['state'], 'initialized')
        self.assertEqual(r['anchor'][0], 4.)

    def test_duplicate_and_reverse_time_start_fresh(self):
        for second in (1., .9):
            with self.subTest(second=second):
                f = TemporalBodyFilter()
                self.observation(f, 1.)
                r = self.observation(f, second, x=4.)
                self.assertEqual(r['state'], 'reset_time')
                self.assertEqual(r['anchor'][0], 4.)

    def test_different_dimensions_start_fresh(self):
        f = TemporalBodyFilter()
        self.observation(f, 0.)
        r = f.update(np.zeros((3, 3)), np.array([.1, .2, .3]), np.array([1., 0., 3.]), .03, 1)
        self.assertEqual(r['state'], 'reset_dimensions')
        np.testing.assert_array_equal(r['shape'], [.1, .2, .3])

    def test_invalid_inputs_raise_and_clear_state(self):
        cases = [dict(rotvec=np.zeros(3)), dict(rotvec=np.full((2, 3), np.nan)),
                 dict(shape=np.array([])), dict(shape=np.array([1.01])),
                 dict(shape=np.array([-.01])), dict(anchor=np.array([0., 0., 0.])),
                 dict(anchor=np.array([0., 0., np.inf])), dict(anchor=np.zeros(2)),
                 dict(timestamp_s=np.nan), dict(timestamp_s='bad'), dict(segment_id=None)]
        for override in cases:
            with self.subTest(override=override):
                f = TemporalBodyFilter()
                self.observation(f, 0.)
                args = dict(rotvec=np.zeros((2, 3)), shape=np.array([.5, .7]),
                            anchor=np.array([0., 0., 3.]), timestamp_s=.03, segment_id=1)
                args.update(override)
                with self.assertRaises(ValueError):
                    f.update(**args)
                r = self.observation(f, .06, x=4.)
                self.assertEqual(r['state'], 'initialized')
                self.assertEqual(r['anchor'][0], 4.)

    def test_parameters_are_validated_and_zero_gap_disables_cross_frame_mix(self):
        for key in ('pose_tau', 'shape_tau', 'translation_tau', 'max_gap_s'):
            for value in (-1., float('nan'), float('inf'), 'bad'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    TemporalBodyFilter(**{key: value})
        f = TemporalBodyFilter(max_gap_s=0)
        self.observation(f, 0.)
        self.assertEqual(self.observation(f, .01)['state'], 'reset_gap')

    def test_input_and_output_arrays_do_not_alias_history(self):
        f = TemporalBodyFilter()
        rotvec, shape, anchor = np.zeros((2, 3)), np.array([.5, .7]), np.array([0., 0., 3.])
        first = f.update(rotvec, shape, anchor, 0., 1)
        first['shape'][:] = 1
        first['anchor'][:] = 100
        first['rotvec'][:] = 10
        shape[:] = .1
        anchor[:] = 20
        rotvec[:] = 20
        second = self.observation(f, .03)
        np.testing.assert_array_equal(second['shape'], [.5, .7])
        np.testing.assert_array_equal(second['anchor'], [0., 0., 3.])
        np.testing.assert_array_equal(second['rotvec'], np.zeros((2, 3)))

    def test_motion_is_not_frozen_and_shape_stays_bounded(self):
        f = TemporalBodyFilter()
        self.observation(f, 0., shape=0., x=0.)
        for i in range(1, 121):
            result = self.observation(f, i / 30, shape=1., x=2.)
            self.assertTrue(np.all((result['shape'] >= 0) & (result['shape'] <= 1)))
            self.assertGreater(result['anchor'][2], 0)
        self.assertAlmostEqual(result['anchor'][0], 2., places=8)
        self.assertGreater(result['shape'][0], .98)


if __name__ == '__main__':
    unittest.main()
