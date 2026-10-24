"""Landmark fitting checks against independently projected known geometry."""

import json
import unittest

import numpy as np

from scripts.video.landmark_alignment import ANNY_LANDMARK_BONES, align_to_landmarks


class LandmarkAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.names = list(ANNY_LANDMARK_BONES)
        self.bones = list(ANNY_LANDMARK_BONES.values())
        self.camera = np.array([[630., 0., 336.], [0., 630., 336.], [0., 0., 1.]])
        left = [[.2, -.65, 3.9], [.35, -.25, 4.1], [.4, .05, 3.95],
                [.12, .0, 4.05], [.14, .45, 4.2], [.1, .85, 4.1]]
        right = [[-.2, -.65, 4.1], [-.35, -.25, 4.], [-.4, .05, 4.15],
                 [-.12, .0, 3.95], [-.14, .45, 3.9], [-.1, .85, 4.0]]
        self.joints = np.array(left + right)
        self.person = dict(joints=self.joints.copy(), vertices=self.joints.copy(),
                           anchor=np.array([0., -.9, 4.]), root=np.array([0., 0., 4.]),
                           shape=np.array([.4, .5]), rotvec=np.zeros((12, 3)),
                           confidence=.9, bbox=np.array([260., 230., 420., 490.]))

    def observations(self, delta=(0., 0., 0.), subset=None):
        points = self.joints + delta
        xy = points @ self.camera.T
        xy = xy[:, :2] / xy[:, 2:]
        return [dict(name=name, x=float(point[0]), y=float(point[1]), confidence=.95)
                for name, point in zip(self.names, xy) if subset is None or name in subset]

    def align(self, observations, person=None):
        return align_to_landmarks(person or self.person, self.bones, observations, self.camera, 672)

    def test_recovers_translation_and_depth_without_changing_pose_or_shape(self):
        delta = np.array([.13, -.07, .75])
        before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.person.items()}
        output, report = self.align(self.observations(delta))
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['status'], 'aligned')
        np.testing.assert_allclose(report['delta_xyz_m'], delta, atol=1e-6)
        np.testing.assert_allclose(output['joints'], self.joints + delta, atol=1e-6)
        np.testing.assert_allclose(output['vertices'][1:] - output['vertices'][:-1],
                                   self.joints[1:] - self.joints[:-1], atol=1e-10)
        for key in ('anchor', 'root'):
            np.testing.assert_allclose(output[key], before[key] + delta)
        for key in ('shape', 'rotvec'):
            np.testing.assert_array_equal(output[key], before[key])
        for key in before:
            np.testing.assert_array_equal(self.person[key], before[key])
        self.assertLess(report['after_error_px'], 1e-5)
        self.assertGreater(report['before_error_px'], 5.)
        json.dumps(report, allow_nan=False)

    def test_one_visible_side_is_sufficient(self):
        observation = self.observations((.05, .03, .4), {'left_shoulder', 'left_hip', 'left_knee', 'left_ankle'})
        output, report = self.align(observation)
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['inlier_count'], 4)
        np.testing.assert_allclose(report['delta_xyz_m'], [.05, .03, .4], atol=1e-6)

    def test_corrupted_wrist_is_rejected_without_dragging_the_body(self):
        delta = np.array([.1, -.03, .5])
        observation = self.observations(delta)
        observation[2]['x'] += 150.
        observation[2]['y'] += 80.
        output, report = self.align(observation)
        self.assertTrue(report['accepted'], report)
        self.assertIn('left_wrist', report['rejected_names'])
        np.testing.assert_allclose(report['delta_xyz_m'], delta, atol=1e-6)
        self.assertLess(report['all_after_error_px'], report['all_before_error_px'])

    def test_missing_low_confidence_and_inadequate_body_coverage_are_noops(self):
        scenarios = [[], self.observations()[:3],
                     [dict(r, confidence=.1) for r in self.observations()],
                     [r for r in self.observations() if not r['name'].endswith('_shoulder')]]
        for observation in scenarios:
            with self.subTest(observation=observation):
                output, report = self.align(observation)
                self.assertFalse(report['accepted'])
                self.assertEqual(report['status'], 'insufficient_landmarks')
                np.testing.assert_array_equal(output['vertices'], self.person['vertices'])
                output['vertices'][0, 0] = 123
                self.assertNotEqual(self.person['vertices'][0, 0], 123)

    def test_already_aligned_preserves_exact_values(self):
        output, report = self.align(self.observations())
        self.assertFalse(report['accepted'])
        self.assertEqual(report['status'], 'already_aligned')
        np.testing.assert_array_equal(output['vertices'], self.person['vertices'])

    def test_height_degeneracy_distant_person_and_bad_fit_are_rejected(self):
        flat = [dict(r, y=330. + i * .1) for i, r in enumerate(self.observations())]
        _, report = self.align(flat)
        self.assertEqual(report['status'], 'insufficient_vertical_extent')
        far = [dict(r, x=r['x'] + 250.) for r in self.observations()]
        _, report = self.align(far)
        self.assertEqual(report['status'], 'landmarks_too_far')
        inconsistent = self.observations()
        for i, row in enumerate(inconsistent):
            row['x'] += (-1 if i % 2 else 1) * 100.
        output, report = self.align(inconsistent)
        self.assertFalse(report['accepted'])
        self.assertIn(report['status'], ['poor_fit', 'insufficient_consistent_landmarks'])
        np.testing.assert_array_equal(output['vertices'], self.person['vertices'])

    def test_nonfinite_observations_are_ignored_and_report_is_strict_json(self):
        observation = self.observations((0., 0., .5))
        observation.extend([dict(name='nose', x=1., y=2., confidence=1.),
                            dict(name='left_shoulder', x=float('nan'), y=10., confidence=.9),
                            dict(name='right_ankle', x=10., y=20., confidence=float('inf'))])
        _, report = self.align(observation)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['landmark_count'], 12)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_geometry_and_behind_camera_geometry_are_rejected(self):
        for value in (float('nan'), -.1):
            person = dict(self.person, vertices=self.person['vertices'].copy())
            person['vertices'][0, 2] = value
            _, report = self.align(self.observations(), person)
            self.assertEqual(report['status'], 'invalid_geometry')
            self.assertFalse(report['accepted'])
            json.dumps(report, allow_nan=False)

    def test_extreme_scale_request_does_not_exceed_search_bound(self):
        output, report = self.align(self.observations((0., 0., 7.)))
        self.assertFalse(report['accepted'])
        np.testing.assert_array_equal(output['vertices'], self.person['vertices'])
        np.testing.assert_array_equal(report['delta_xyz_m'], np.zeros(3))

    def test_configuration_validation(self):
        for camera, size, confidence in [(np.eye(2), 672, .5), (self.camera, 0, .5),
                                          (self.camera, 672, float('nan'))]:
            with self.assertRaises(ValueError):
                align_to_landmarks(self.person, self.bones, self.observations(), camera, size, confidence)


if __name__ == '__main__':
    unittest.main()
