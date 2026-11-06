import json
import unittest
import numpy as np
from scripts.video.alignment_depth_filter import AlignmentDepthFilter
import test_landmark_alignment as fixtures


class DepthFilterTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.LandmarkAlignmentTests()
        self.f.setUp()
        self.state = AlignmentDepthFilter()

    def apply(self, depth, t, segment=1, state=None):
        f = self.f
        fitted, report = f.align(f.observations((.1, -.03, depth)))
        self.assertTrue(report['accepted'])
        return (state or self.state).apply(f.person, fitted, report, f.bones, f.camera, 672, t, segment)

    def test_alternating_depth_noise_is_reduced_and_geometry_preserved(self):
        raw, smooth = [], []
        for i in range(90):
            output, report = self.apply(.6 + (.12 if i % 2 else -.12), i / 30)
            raw.append(report['fitted_depth_m'])
            smooth.append(report['rendered_depth_m'])
            np.testing.assert_allclose(np.diff(output['joints'], axis=0), np.diff(self.f.joints, axis=0))
            np.testing.assert_array_equal(output['shape'], self.f.person['shape'])
            np.testing.assert_array_equal(output['rotvec'], self.f.person['rotvec'])
            np.testing.assert_allclose(output['joints'] - self.f.joints,
                                       np.broadcast_to(report['delta_xyz_m'], output['joints'].shape))
            residual = np.array(report['after_xy']) - report['observed_xy']
            expected = np.sqrt(np.average(np.sum(residual ** 2, axis=1), weights=report['confidence']))
            self.assertAlmostEqual(report['after_error_px'], expected)
            json.dumps(report, allow_nan=False)
        self.assertLess(np.std(np.diff(smooth)), .15 * np.std(np.diff(raw)))
        self.assertLessEqual(np.max(np.abs(np.diff(np.log(smooth)))), .5 / 30 + 1e-8)

    def test_gradual_depth_change_remains_possible(self):
        depths = []
        for i in range(120):
            output, report = self.apply(.4 + .2 * i / 30, i / 30)
            depths.append(report['rendered_depth_m'])
        self.assertGreater(depths[-1] - depths[0], .6)
        self.assertLess(report['fitted_depth_m'] - depths[-1], .1)

    def test_disabled_is_identical_to_original_fit(self):
        state = AlignmentDepthFilter(tau=0)
        for i, depth in enumerate((.4, .9, .6)):
            fitted, original_report = self.f.align(self.f.observations((.1, -.03, depth)))
            output, report = self.apply(depth, i / 30, state=state)
            np.testing.assert_array_equal(output['vertices'], fitted['vertices'])
            self.assertEqual(report['depth_filter_state'], 'disabled')
            self.assertAlmostEqual(report['after_error_px'], original_report['after_error_px'])

    def test_track_time_and_rejection_reset_history(self):
        for timestamp, segment in ((.5, 1), (0., 1), (.03, 2)):
            self.state.reset()
            self.apply(.4, 0.)
            _, report = self.apply(.9, timestamp, segment)
            self.assertEqual(report['depth_filter_state'], 'initialized')
            self.assertAlmostEqual(report['rendered_depth_m'], report['fitted_depth_m'])
        self.state.reset()
        self.apply(.4, 0.)
        fitted, report = self.f.align([])
        output, report = self.state.apply(self.f.person, fitted, report, self.f.bones, self.f.camera, 672, .03, 1)
        np.testing.assert_array_equal(output['vertices'], self.f.person['vertices'])
        _, report = self.apply(.9, .06)
        self.assertEqual(report['depth_filter_state'], 'initialized')

    def test_invalid_settings(self):
        for tau in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                AlignmentDepthFilter(tau=tau)
