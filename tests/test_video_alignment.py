"""Exercise optional landmark alignment with real video I/O and known geometry.

Landmark observations are projected from an independently translated reference
body, then saved in original-image coordinates. No network weights are needed.
"""
import csv
import itertools
import json
import unittest
from unittest import mock

import numpy as np

import demo_video
from scripts.video.landmark_alignment import ANNY_LANDMARK_BONES
from scripts.video.landmark_observations import sha256_file
import test_demo_video as fixtures


class AlignmentPredictor(fixtures.FakePredictor):
    anchor_index = 1
    joint_names = ['pelvis', 'head'] + list(ANNY_LANDMARK_BONES.values())

    def __init__(self, kind, seen):
        super().__init__(kind, seen)
        self.decode_calls = 0

    @staticmethod
    def geometry(anchor):
        anchor = np.asarray(anchor, dtype=np.float32)
        relative = [[0., 1.05, 0.], [0., 0., 0.]]
        heights = dict(shoulder=.30, elbow=.70, wrist=1.1, hip=1.05, knee=1.55, ankle=2.05)
        widths = dict(shoulder=.30, elbow=.45, wrist=.5, hip=.18, knee=.18, ankle=.2)
        for name in ANNY_LANDMARK_BONES:
            side, joint = name.split('_')
            sign = -1 if side == 'left' else 1
            relative.append([sign * widths[joint], heights[joint], .03 * sign])
        joints = anchor + np.asarray(relative, np.float32)
        vertices = anchor + np.array([[-.55, -.05, 0.], [.55, -.05, 0.],
                                      [.55, 2.15, 0.], [-.55, 2.15, 0.]], np.float32)
        return dict(vertices=vertices, joints=joints,
                    shape=np.array([.4, .7], np.float32),
                    rotvec=np.zeros((len(joints), 3), np.float32), confidence=.95)

    @classmethod
    def observation(cls, processed_index):
        return cls.geometry([processed_index * .01, -1.05, 3.])

    def predict(self, rgb, camera):
        index = self.calls
        people = super().predict(rgb, camera)
        return [self.observation(index)] if people else []

    def decode_temporal(self, rotvec, shape, anchor):
        self.decode_calls += 1
        result = self.geometry(anchor)
        result.update(rotvec=np.asarray(rotvec, np.float32), shape=np.asarray(shape, np.float32))
        return result


@unittest.skipIf(fixtures.cv2 is None, 'install requirements-video.txt for video I/O')
class VideoAlignmentTests(unittest.TestCase):
    args = fixtures.VideoIntegrationTests.args
    decode = fixtures.VideoIntegrationTests.decode
    renderer_factory = fixtures.VideoIntegrationTests.renderer_factory

    def setUp(self):
        fixtures.VideoIntegrationTests.setUp(self)
        self.landmarks = self.root / 'landmarks.json'
        self.delta = np.array([.08, -.06, .7])
        camera = demo_video.camera_matrix(128, 55)
        factors = np.array([128 / 160, 77 / 96])
        origin = np.array([0, 25])

        def original_xy(points):
            projected = points @ camera.T
            return (projected[:, :2] / projected[:, 2:] - origin) / factors

        rows = []
        for processed_index, frame in enumerate([1, 3, 5, 7, 9]):
            reference = self.observation(processed_index)
            xy = original_xy(reference['joints'][2:] + self.delta)
            box_points = original_xy(reference['vertices'] + self.delta)
            bbox = np.r_[np.maximum(box_points.min(0), 0), np.minimum(box_points.max(0), [160, 96])]
            person = dict(bbox=bbox.tolist(), landmarks={name: [float(point[0]), float(point[1]), .95]
                                                        for name, point in zip(ANNY_LANDMARK_BONES, xy)})
            rows.append(dict(frame_index=frame, detector_ms=987., people=[] if frame == 7 else [person]))
        self.data = dict(schema_version=1, coordinate_system='original_image_pixels',
                         video=dict(width=160, height=96, fps=12., sha256=sha256_file(self.video)),
                         detector={'name': 'deterministic independent reference'}, frames=rows)
        self.write_observations()

    @staticmethod
    def observation(index):
        return AlignmentPredictor.observation(index)

    def write_observations(self):
        self.landmarks.write_text(json.dumps(self.data))

    def predictor_factory(self, checkpoint, kind, fix, device, amp):
        predictor = AlignmentPredictor(kind, self.seen)
        self.loaded.append(predictor)
        return predictor

    def arrays(self, kind, folder=None):
        with np.load((folder or self.output) / kind / 'predictions.npz') as source:
            return {key: source[key].copy() for key in source.files}

    def aligned_args(self, *extra):
        return self.args('--alignment', 'landmarks', '--landmarks', str(self.landmarks),
                         '--student-only', *extra)

    def test_saved_video_predictions_use_stabilized_depth(self):
        reference = self.observation(1)
        camera = demo_video.camera_matrix(128, 55)
        def project(points):
            xy = points @ camera.T
            return (xy[:, :2] / xy[:, 2:] - [0, 25]) / [128 / 160, 77 / 96]
        points = project(reference['joints'][2:] + [.08, -.06, 1.2])
        vertices = project(reference['vertices'] + [.08, -.06, 1.2])
        person = self.data['frames'][1]['people'][0]
        person['landmarks'] = {name: [*point.tolist(), .95] for name, point in zip(ANNY_LANDMARK_BONES, points)}
        person['bbox'] = np.r_[vertices.min(0), vertices.max(0)].tolist()
        self.write_observations()
        demo_video.run(self.aligned_args(), self.predictor_factory, self.renderer_factory)
        rows = json.loads((self.output / 'student_aligned/alignment.json').read_text())['frames']
        self.assertEqual(rows[1]['depth_filter_state'], 'smoothed')
        self.assertLess(rows[1]['rendered_depth_m'], rows[1]['fitted_depth_m'] - .1)
        saved = self.arrays('student_aligned')
        self.assertAlmostEqual(float(saved['joints'][1, 0, 2]), rows[1]['rendered_depth_m'], places=5)
        np.testing.assert_allclose(saved['joints'][1] - reference['joints'],
                                   np.broadcast_to(rows[1]['delta_xyz_m'], reference['joints'].shape), atol=1e-6)
        self.assertEqual(rows[4]['depth_filter_state'], 'initialized')

    def test_alignment_refines_only_translation_keeps_gaps_and_counts_added_time(self):
        with mock.patch('demo_video.time.perf_counter', side_effect=itertools.count(0., .001).__next__):
            manifest = demo_video.run(self.aligned_args(), self.predictor_factory, self.renderer_factory)
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(self.loaded[0].calls, 5)
        self.assertEqual(self.loaded[0].decode_calls, 0)
        raw, aligned = self.arrays('student'), self.arrays('student_aligned')
        for key in ('frame_index', 'timestamp_s', 'valid', 'segment_id', 'shape', 'rotvec'):
            np.testing.assert_array_equal(raw[key], aligned[key])
        for index in (0, 1, 4):
            observed_delta = aligned['joints'][index] - raw['joints'][index]
            np.testing.assert_allclose(observed_delta, np.broadcast_to(self.delta, observed_delta.shape), atol=2e-5)
            np.testing.assert_allclose(aligned['anchor'][index] - raw['anchor'][index], self.delta, atol=2e-5)
        # No student on frame 5, no landmarks on frame 7: never invent a body.
        self.assertFalse(aligned['valid'][2])
        self.assertTrue(np.isnan(aligned['joints'][2]).all())
        np.testing.assert_array_equal(aligned['joints'][3], raw['joints'][3])
        report = json.loads((self.output / 'student_aligned/alignment.json').read_text())
        self.assertEqual([r['status'] for r in report['frames']],
                         ['aligned', 'aligned', 'missing_student', 'no_landmark_person', 'aligned'])
        for row in report['frames']:
            if row['accepted']:
                self.assertGreater(row['before_error_px'], 3.)
                self.assertLess(row['after_error_px'], 1e-4)
        summary = manifest['models']['student_aligned']['summary']
        self.assertEqual(summary['alignment_accepted_frames'], 3)
        self.assertEqual(summary['alignment_skipped_frames'], 2)
        timing = summary['timing']
        self.assertAlmostEqual(timing['inference_fps'], 40.)
        self.assertAlmostEqual(timing['alignment_ms']['mean'], .75)
        self.assertAlmostEqual(timing['processing_ms']['mean'], 25.75)
        self.assertAlmostEqual(timing['processing_fps'], 1000 / 25.75)
        self.assertIn('offline landmark detector', timing['processing_definition'])
        with (self.output / 'student_aligned/frames.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(float(csv_rows[1]['landmark_detector_ms']), 987.)
        self.assertEqual(float(csv_rows[1]['processing_ms']), 11.)
        for name, columns in [('student', 1), ('student_aligned', 1), ('alignment_comparison', 3), ('comparison', 3)]:
            fps, frames = self.decode(self.output / (name + '.mp4'))
            self.assertEqual(len(frames), 5)
            self.assertAlmostEqual(fps, 6.)
            self.assertEqual(frames[0].shape, (192, 128 * columns, 3))
        self.assertTrue(all(r.closed for r in self.renderers))
        # Adding alignment does not change the original student predictions.
        aligned_folder = self.output
        self.output = self.root / 'raw control'
        demo_video.run(self.args('--student-only'), self.predictor_factory, self.renderer_factory)
        control = self.arrays('student')
        for key in control:
            np.testing.assert_array_equal(control[key], self.arrays('student', aligned_folder)[key])

    def test_alignment_can_follow_temporal_processing_without_another_forward(self):
        manifest = demo_video.run(self.aligned_args('--temporal', 'ema'),
                                  self.predictor_factory, self.renderer_factory)
        self.assertEqual(self.loaded[0].calls, 5)
        self.assertEqual(self.loaded[0].decode_calls, 4)
        self.assertEqual(manifest['models']['student_aligned']['alignment']['source'], 'student_temporal')
        raw, filtered, aligned = [self.arrays(k) for k in ('student', 'student_temporal', 'student_aligned')]
        for key in ('frame_index', 'timestamp_s', 'valid'):
            np.testing.assert_array_equal(raw[key], aligned[key])
        for key in ('shape', 'rotvec'):
            np.testing.assert_array_equal(filtered[key], aligned[key])
        np.testing.assert_array_equal(aligned['joints'][3], filtered['joints'][3])
        self.assertGreater(manifest['models']['student_aligned']['summary']['alignment_accepted_frames'], 0)
        _, comparison = self.decode(self.output / 'comparison.mp4')
        _, specific = self.decode(self.output / 'alignment_comparison.mp4')
        self.assertEqual(comparison[0].shape, (192, 512, 3))
        self.assertEqual(specific[0].shape, (192, 384, 3))

    def test_wrong_video_sha_is_rejected_before_loading_model_or_creating_output(self):
        self.data['video']['sha256'] = '0' * 64
        self.write_observations()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            demo_video.run(self.aligned_args(), self.predictor_factory, self.renderer_factory)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.loaded, [])

    def test_nonfinite_landmarks_are_rejected_before_model_execution(self):
        self.data['frames'][0]['people'][0]['landmarks']['left_hip'][0] = float('nan')
        self.write_observations()
        with self.assertRaisesRegex(ValueError, 'Invalid landmark coordinate'):
            demo_video.run(self.aligned_args(), self.predictor_factory, self.renderer_factory)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.loaded, [])


if __name__ == '__main__':
    unittest.main()
