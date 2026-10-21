"""Video plumbing tests with real MP4 I/O and deterministic fake predictions.

No trained weights or GPU are required. Set VIDEO_TEST_EGL=1 to additionally
exercise the real offscreen mesh renderer in an environment with EGL support.
"""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import demo_video


class FakePredictor:
    phenotype_names = ['age', 'height']
    joint_names = ['pelvis', 'head', 'hand']
    faces = np.array([[0, 1, 2, 3]], dtype=np.int64)

    def __init__(self, kind, seen):
        self.kind = kind
        self.seen = seen
        self.input_size = 128 if kind == 'student' else 256
        self.metadata = {'test_predictor': kind}
        self.calls = 0

    def predict(self, rgb, camera):
        index = self.calls
        self.calls += 1
        self.seen[self.kind].append((rgb.copy(), camera.copy()))
        # Misses deliberately differ between passes. Neither may shorten video.
        if index == (2 if self.kind == 'student' else 3):
            return []
        root = np.array([index * .01, 0, 3], np.float32)
        vertices = root + np.array([[-.3, -.5, 0], [.3, -.5, 0],
                                    [.3, .5, 0], [-.3, .5, 0]], np.float32)
        joints = np.stack([root, root + [0, -.5, 0], root + [.3, 0, 0]])
        return [dict(vertices=vertices, joints=joints,
                     shape=np.array([.2 + index * .01, .7], np.float32),
                     rotvec=np.zeros((3, 3), np.float32), confidence=.95)]


class FakeRenderer:
    def __init__(self, size, camera, faces):
        self.closed = False

    def render(self, rgb, vertices):
        output = rgb.copy()
        output[55:75, 55:75] = [10, 230, 240]
        return output

    def close(self):
        self.closed = True


@unittest.skipIf(cv2 is None, 'install requirements-video.txt for video I/O tests')
class VideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='multihmr-video-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / 'input clip.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 12, (160, 96))
        if not writer.isOpened():
            self.skipTest('OpenCV build cannot encode mp4v')
        try:
            for index in range(10):
                # Different broad colors permit checking decoded frame identity
                # despite lossy MP4 encoding, padding and comparison re-encoding.
                rgb = np.full((96, 160, 3), [30 + 15 * index, 60, 90], np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        self.student = self.root / 'student run' / 'train' / 'checkpoints' / '00029.pt'
        self.student.parent.mkdir(parents=True)
        self.student.write_bytes(b'fixture; injected factory never loads weights')
        self.sidecar = self.student.parent.parent.parent / 'student_fix_arg.txt'
        self.sidecar.write_text('none\n')
        self.baseline = self.root / 'baseline.pt'
        self.baseline.write_bytes(b'fixture')
        self.output = self.root / 'new result'
        self.seen = {'student': [], 'baseline': []}
        self.loaded = []
        self.renderers = []

    def args(self, *extra):
        return demo_video.parse_args([
            '--video', str(self.video), '--student-checkpoint', str(self.student),
            '--baseline-checkpoint', str(self.baseline), '--output', str(self.output),
            '--device', 'cpu', '--panel-size', '128', '--start-frame', '1',
            '--stride', '2', '--max-frames', '5', *extra])

    def predictor_factory(self, checkpoint, kind, fix, device, amp):
        self.loaded.append((checkpoint, kind, fix, device, amp))
        return FakePredictor(kind, self.seen)

    def renderer_factory(self, size, camera, faces):
        renderer = FakeRenderer(size, camera, faces)
        self.renderers.append(renderer)
        return renderer

    def decode(self, path):
        cap = cv2.VideoCapture(str(path))
        self.assertTrue(cap.isOpened(), str(path))
        frames = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return fps, frames

    def test_both_passes_keep_timing_frames_and_explicit_detection_gaps(self):
        manifest = demo_video.run(self.args(), self.predictor_factory, self.renderer_factory)
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(manifest['processed_frames'], 5)
        self.assertEqual(manifest['output_fps'], 6)
        self.assertEqual([row[1] for row in self.loaded], ['student', 'baseline'])
        self.assertEqual([row[2] for row in self.loaded], ['none', 'none'])
        self.assertTrue(all(renderer.closed for renderer in self.renderers))
        self.assertEqual(len(self.seen['student']), 5)
        self.assertEqual(len(self.seen['baseline']), 5)
        for (student_rgb, student_k), (base_rgb, base_k) in zip(self.seen['student'], self.seen['baseline']):
            np.testing.assert_allclose(student_k[:2] / 128, base_k[:2] / 256)
            np.testing.assert_allclose(student_rgb[64, 64], base_rgb[128, 128], atol=2)
        fps, comparison = self.decode(self.output / 'comparison.mp4')
        self.assertAlmostEqual(fps, 6)
        self.assertEqual(len(comparison), 5)
        self.assertEqual(comparison[0].shape, (192, 384, 3))
        for name in ('input', 'student', 'baseline'):
            fps, frames = self.decode(self.output / (name + '.mp4'))
            self.assertAlmostEqual(fps, 6)
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].shape, (192, 128, 3))
            # A patch away from the overlay confirms source frames 1,3,5,7,9
            # survived each encoding pass in order, including missed detections.
            expected_red = [45, 75, 105, 135, 165]
            observed_red = [float(frame[100:110, 20:30, 0].mean()) for frame in frames]
            np.testing.assert_allclose(observed_red, expected_red, atol=12)
        for column, name in enumerate(('input', 'student', 'baseline')):
            _, frames = self.decode(self.output / (name + '.mp4'))
            for compared, individual in zip(comparison, frames):
                patch = compared[100:110, column * 128 + 20:column * 128 + 30]
                np.testing.assert_allclose(patch.mean((0, 1)), individual[100:110, 20:30].mean((0, 1)), atol=12)
        for kind, missing in [('student', 2), ('baseline', 3)]:
            with (self.output / kind / 'frames.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['frame_index']) for row in rows], [1, 3, 5, 7, 9])
            np.testing.assert_allclose([float(row['timestamp_s']) for row in rows], np.array([1, 3, 5, 7, 9]) / 12)
            self.assertEqual(rows[missing]['status'], 'missing')
            for index in (0, missing, missing + 1):
                for key in ('root_step_m', 'root_speed_m_per_s', 'mean_joint_speed_m_per_s'):
                    self.assertEqual(rows[index][key], '', (kind, index, key))
            self.assertAlmostEqual(float(rows[1]['root_speed_m_per_s']), .06, places=6)
            with np.load(self.output / kind / 'predictions.npz') as arrays:
                self.assertFalse(arrays['valid'][missing])
                self.assertTrue(np.isnan(arrays['joints'][missing]).all())
                self.assertEqual(arrays['joints'].shape, (5, 3, 3))
                self.assertEqual(arrays['valid'].sum(), 4)
            summary = json.loads((self.output / kind / 'summary.json').read_text())
            self.assertEqual(summary['missing_frames'], 1)
            self.assertEqual(summary['consecutive_selected_pairs'], 2)
        # A completed output must not be reused or partly overwritten.
        before = (self.output / 'run.json').read_bytes()
        with self.assertRaises(FileExistsError):
            demo_video.run(self.args(), self.predictor_factory, self.renderer_factory)
        self.assertEqual((self.output / 'run.json').read_bytes(), before)

    def test_student_only_does_not_require_baseline(self):
        self.baseline.unlink()
        result = demo_video.run(self.args('--student-only', '--max-frames', '1'),
                                self.predictor_factory, self.renderer_factory)
        self.assertEqual(list(result['models']), ['student'])
        _, frames = self.decode(self.output / 'comparison.mp4')
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (192, 256, 3))
        self.assertFalse((self.output / 'baseline.mp4').exists())

    def test_failed_prediction_marks_run_failed_and_releases_resources(self):
        def broken_factory(checkpoint, kind, fix, device, amp):
            model = FakePredictor(kind, self.seen)

            def fail(rgb, camera):
                raise RuntimeError('injected inference failure')

            model.predict = fail
            return model

        with self.assertRaisesRegex(RuntimeError, 'injected inference failure'):
            demo_video.run(self.args(), broken_factory, self.renderer_factory)
        manifest = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('injected inference failure', manifest['error'])
        self.assertTrue(all(renderer.closed for renderer in self.renderers))
        self.assertFalse((self.output / 'comparison.mp4').exists())

    def test_sidecar_selects_own_fix_and_rejects_ambiguous_provenance(self):
        correction = self.root / 'own correction.json'
        correction.write_text('{"bones": {}}\n')
        self.sidecar.write_text(str(correction) + '\n')
        choice, source = demo_video.resolve_student_fix(self.student, 'auto')
        self.assertEqual(choice, str(correction))
        self.assertEqual(source, str(self.sidecar))
        self.sidecar.write_text('relative.json\n')
        with self.assertRaisesRegex(ValueError, 'absolute'):
            demo_video.resolve_student_fix(self.student, 'auto')
        self.sidecar.write_text('none\nnone\n')
        with self.assertRaisesRegex(ValueError, 'one JSON path'):
            demo_video.resolve_student_fix(self.student, 'auto')
        self.sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            demo_video.resolve_student_fix(self.student, 'auto')
        self.assertEqual(demo_video.resolve_student_fix(self.student, 'checkpoint')[0], 'checkpoint')
        self.assertEqual(demo_video.resolve_student_fix(self.student, 'none')[0], 'none')
        self.assertEqual(demo_video.resolve_student_fix(self.student, str(correction))[0], str(correction))

    def test_invalid_frame_selection_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'beyond'):
            demo_video.run(self.args('--start-frame', '100'), self.predictor_factory, self.renderer_factory)
        self.assertEqual(self.loaded, [])
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get('VIDEO_TEST_EGL') == '1', 'opt in with VIDEO_TEST_EGL=1')
    def test_real_renderer_projects_a_quad_without_moving_geometry(self):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        vertices = np.array([[-.3, -.3, 3], [.3, -.3, 3], [.3, .3, 3], [-.3, .3, 3]], np.float32)
        renderer = demo_video.MeshRenderer(128, demo_video.camera_matrix(128, 55), np.array([[0, 1, 2, 3]]))
        try:
            rendered = renderer.render(np.zeros((128, 128, 3), np.uint8), vertices)
        finally:
            renderer.close()
        self.assertGreater(rendered[64, 64].max(), 0)
        self.assertEqual(rendered[0, 0].max(), 0)
        yy, xx = np.where(rendered.max(axis=2) > 0)
        self.assertAlmostEqual(float(xx.mean()), 63.5, delta=1)
        self.assertAlmostEqual(float(yy.mean()), 63.5, delta=1)


if __name__ == '__main__':
    unittest.main()
