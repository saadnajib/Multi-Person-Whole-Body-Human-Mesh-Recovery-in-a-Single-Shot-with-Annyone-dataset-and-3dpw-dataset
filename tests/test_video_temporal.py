"""Temporal video integration using real MP4 I/O and a deterministic body decoder.

The fixture's geometry depends on pose, shape and head-anchor position. It is
not an Anny accuracy test and does not require checkpoints or a GPU.
"""

import csv
import itertools
import json
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import demo_video
import test_demo_video as fixtures


class JitterPredictor(fixtures.FakePredictor):
    anchor_index = 1

    def __init__(self, kind, seen):
        super().__init__(kind, seen)
        self.decode_calls = 0

    @staticmethod
    def geometry(rotvec, shape, anchor):
        rotvec, shape, anchor = [np.asarray(x, dtype=np.float32) for x in (rotvec, shape, anchor)]
        height = .5 + .2 * shape[1]
        width = .2 + .3 * shape[0]
        root_rotation, _, hand_rotation = Rotation.from_rotvec(rotvec).as_matrix()
        vertices = np.array([[-width, 0, 0], [width, 0, 0],
                             [width, height, 0], [-width, height, 0]])
        joints = np.array([[0, height, 0], [0, 0, 0], [width, height * .5, 0]])
        joints[2] = hand_rotation @ joints[2]
        vertices = (vertices @ root_rotation.T + anchor).astype(np.float32)
        joints = (joints @ root_rotation.T + anchor).astype(np.float32)
        return dict(vertices=vertices, joints=joints, shape=shape.copy(), rotvec=rotvec.copy())

    def predict(self, rgb, camera):
        people = super().predict(rgb, camera)
        if not people:
            return people
        sign = 1 if (self.calls - 1) % 2 else -1
        rotvec = np.array([[0, 0, .25 * sign], [0, 0, 0], [0, 0, .4 * sign]])
        shape = np.array([.45 + .08 * sign, .7])
        anchor = np.array([.04 * sign, -.5, 3.])
        return [dict(self.geometry(rotvec, shape, anchor), confidence=.95)]

    def decode_temporal(self, rotvec, shape, anchor):
        self.decode_calls += 1
        return self.geometry(rotvec, shape, anchor)


@unittest.skipIf(fixtures.cv2 is None, 'install requirements-video.txt for video I/O tests')
class TemporalVideoTests(unittest.TestCase):
    # Reuse only fixture helpers: inheriting its test class would rerun every
    # unrelated plumbing test with a different geometry fixture.
    setUp = fixtures.VideoIntegrationTests.setUp
    args = fixtures.VideoIntegrationTests.args
    decode = fixtures.VideoIntegrationTests.decode
    renderer_factory = fixtures.VideoIntegrationTests.renderer_factory

    def predictor_factory(self, checkpoint, kind, fix, device, amp):
        predictor = JitterPredictor(kind, self.seen)
        self.loaded.append(predictor)
        return predictor

    def load_arrays(self, output, kind):
        with np.load(output / kind / 'predictions.npz') as source:
            return {key: value.copy() for key, value in source.items()}

    def csv_rows(self, output, kind):
        with (output / kind / 'frames.csv').open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_temporal_reuses_predictions_preserves_raw_output_and_resets_at_misses(self):
        # A deterministic clock makes filter overhead and the processing-FPS
        # denominator independently testable without sleeps or timing noise.
        with mock.patch('demo_video.time.perf_counter', side_effect=itertools.count(0., .001).__next__):
            manifest = demo_video.run(self.args('--temporal', 'ema'),
                                      self.predictor_factory, self.renderer_factory)
        on_output = self.output
        self.assertEqual(manifest['status'], 'completed')
        self.assertTrue(manifest['smoothing'])
        self.assertFalse(manifest['temporal']['learned'])
        self.assertEqual([m.calls for m in self.loaded], [5, 5])
        self.assertEqual([m.decode_calls for m in self.loaded], [4, 0])
        self.assertTrue(all(r.closed for r in self.renderers))

        raw = self.load_arrays(on_output, 'student')
        filtered = self.load_arrays(on_output, 'student_temporal')
        for key in ('frame_index', 'timestamp_s', 'valid', 'segment_id'):
            np.testing.assert_array_equal(raw[key], filtered[key])
        np.testing.assert_array_equal(raw['frame_index'], [1, 3, 5, 7, 9])
        np.testing.assert_array_equal(raw['valid'], [True, True, False, True, True])
        for key in ('rotvec', 'shape', 'anchor', 'joints'):
            self.assertTrue(np.isnan(filtered[key][2]).all())
            # The first observation and first observation after the missing
            # frame must not retain any previous filtered body.
            np.testing.assert_allclose(filtered[key][[0, 3]], raw[key][[0, 3]], atol=1e-7)
            for previous, current in ((0, 1), (3, 4)):
                raw_step = np.linalg.norm(raw[key][current] - raw[key][previous])
                filtered_step = np.linalg.norm(filtered[key][current] - filtered[key][previous])
                self.assertLess(filtered_step, raw_step, key)
        rows = self.csv_rows(on_output, 'student_temporal')
        self.assertEqual([row['temporal_state'] for row in rows],
                         ['initialized', 'filtered', 'reset_missing', 'initialized', 'filtered'])
        for index in (0, 2, 3):
            self.assertEqual(rows[index]['root_speed_m_per_s'], '')

        raw_timing = manifest['models']['student']['summary']['timing']
        filtered_timing = manifest['models']['student_temporal']['summary']['timing']
        self.assertAlmostEqual(raw_timing['inference_fps'], 40.)
        self.assertAlmostEqual(filtered_timing['inference_fps'], 40.)
        self.assertEqual(filtered_timing['temporal_ms']['count'], 4)
        self.assertAlmostEqual(filtered_timing['temporal_ms']['mean'], .75)
        self.assertAlmostEqual(filtered_timing['processing_ms']['mean'], 25.75)
        self.assertAlmostEqual(filtered_timing['processing_fps'], 1000 / 25.75)
        self.assertEqual(float(rows[2]['temporal_ms']), 0.)
        self.assertEqual(float(rows[2]['processing_ms']), 30.)
        report = json.loads((on_output / 'temporal_comparison.json').read_text())
        self.assertEqual(report['raw_selected_frames'], report['filtered_selected_frames'])
        self.assertEqual(report['raw_selected_frames'], 4)
        self.assertIn('not ground-truth error', report['caveat'])

        for name, columns in [('input', 1), ('student', 1), ('student_temporal', 1),
                              ('baseline', 1), ('temporal_comparison', 3), ('comparison', 4)]:
            fps, frames = self.decode(on_output / (name + '.mp4'))
            self.assertAlmostEqual(fps, 6.)
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].shape, (192, 128 * columns, 3))

        # Run the same fixture with filtering off and verify the added branch
        # has not changed any raw prediction or the original baseline.
        self.output = self.root / 'raw control'
        self.loaded = []
        off = demo_video.run(self.args(), self.predictor_factory, self.renderer_factory)
        self.assertFalse(off['smoothing'])
        self.assertEqual([m.calls for m in self.loaded], [5, 5])
        self.assertEqual([m.decode_calls for m in self.loaded], [0, 0])
        self.assertFalse((self.output / 'student_temporal.mp4').exists())
        for kind in ('student', 'baseline'):
            control = self.load_arrays(self.output, kind)
            observed = self.load_arrays(on_output, kind)
            self.assertEqual(set(control), set(observed))
            for key in control:
                np.testing.assert_array_equal(control[key], observed[key], err_msg=kind + '/' + key)

    def test_zero_filter_strength_is_a_noop_through_the_decoder(self):
        manifest = demo_video.run(self.args('--temporal', 'ema', '--student-only',
                                            '--temporal-pose-tau', '0', '--temporal-shape-tau', '0',
                                            '--temporal-translation-tau', '0'),
                                  self.predictor_factory, self.renderer_factory)
        self.assertEqual(list(manifest['models']), ['student', 'student_temporal'])
        raw = self.load_arrays(self.output, 'student')
        filtered = self.load_arrays(self.output, 'student_temporal')
        for key in raw:
            np.testing.assert_array_equal(raw[key], filtered[key], err_msg=key)
        _, video = self.decode(self.output / 'comparison.mp4')
        self.assertEqual(video[0].shape, (192, 384, 3))

    def test_wrong_decoder_convention_fails_first_frame_and_closes_all_resources(self):
        original_writer = demo_video.VideoWriter
        writers = []

        class WatchedWriter(original_writer):
            def __init__(self, *args):
                super().__init__(*args)
                self.closed = False
                writers.append(self)

            def close(self):
                self.closed = True
                super().close()

        def wrong_factory(*args):
            predictor = self.predictor_factory(*args)
            actual_decode = predictor.decode_temporal

            def wrong_decode(*values):
                result = actual_decode(*values)
                result['vertices'] += [1., 0., 0.]
                return result

            predictor.decode_temporal = wrong_decode
            return predictor

        with mock.patch('demo_video.VideoWriter', WatchedWriter):
            with self.assertRaisesRegex(RuntimeError, 'does not reproduce raw vertices'):
                demo_video.run(self.args('--temporal', 'ema', '--student-only'),
                               wrong_factory, self.renderer_factory)
        self.assertEqual(len(writers), 3)
        self.assertTrue(all(writer.closed for writer in writers))
        self.assertTrue(all(renderer.closed for renderer in self.renderers))
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('does not reproduce raw vertices', manifest['error'])
        self.assertFalse((self.output / 'comparison.mp4').exists())

    def test_invalid_filtered_geometry_falls_back_to_real_observation(self):
        from scripts.video.temporal_filter import TemporalBodyFilter
        predictor = JitterPredictor('student', self.seen)
        params = np.zeros((3, 3)), np.array([.4, .7]), np.array([0., -.5, 3.])
        camera = demo_video.camera_matrix(128, 55)
        person = dict(predictor.geometry(*params), confidence=.95)
        raw = demo_video.visible_detections([person], camera, 128)[0]
        state = TemporalBodyFilter()
        first, status = demo_video.temporal_person(raw, predictor, state, 0., 1, camera, 128)
        self.assertEqual(status, 'initialized')
        broken = predictor.geometry(*params)
        broken['vertices'][:, 2] = -1.
        with mock.patch.object(predictor, 'decode_temporal', return_value=broken):
            fallback, status = demo_video.temporal_person(raw, predictor, state, .03, 1, camera, 128)
        self.assertEqual(status, 'fallback_raw_invalid_geometry')
        np.testing.assert_array_equal(fallback['vertices'], raw['vertices'])
        following, status = demo_video.temporal_person(raw, predictor, state, .06, 1, camera, 128)
        self.assertEqual(status, 'initialized')
        np.testing.assert_array_equal(following['vertices'], first['vertices'])


if __name__ == '__main__':
    unittest.main()
