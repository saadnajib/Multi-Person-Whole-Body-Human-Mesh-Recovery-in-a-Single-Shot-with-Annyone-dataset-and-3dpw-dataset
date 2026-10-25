import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
import demo_image_compare as demo
import test_video_launcher as launcher


class ImageCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.photos = self.root / 'photos with spaces'
        self.photos.mkdir()
        Image.new('RGB', (160, 90), (42, 64, 96)).save(self.photos / 'one.jpg')
        Image.new('RGB', (90, 160), (90, 60, 30)).save(self.photos / 'two.png')
        self.ckpt = self.root / 'checkpoint.pt'
        self.ckpt.write_bytes(b'fixture')
        self.output = self.root / 'output'
        self.loaded, self.rendered, self.inputs, self.closed = [], [], [], []

    def args(self):
        return demo.parse_args(['--images', str(self.photos), '--student-checkpoint', str(self.ckpt),
                                '--baseline-checkpoint', str(self.ckpt), '--student-bone-fix', 'none',
                                '--panel-size', '128', '--device', 'cpu', '--output', str(self.output)])

    def factory(self, checkpoint, kind, fix, device, amp):
        self.loaded.append((kind, fix))
        parent = self
        class Predictor:
            input_size = 128
            faces = np.array([[0, 1, 2]])
            metadata = {'kind': kind}
            calls = 0
            def predict(self, rgb, camera):
                parent.inputs.append((kind, self.calls, rgb.copy(), camera.copy()))
                self.calls += 1
                if kind == 'baseline' and self.calls == 2:
                    return []
                v = np.array([[-.2, -.5, 3.], [.2, -.5, 3.], [0., .5, 3.]])
                return [dict(vertices=v + [offset, 0, 0], joints=v.copy(), shape=np.array([.5]),
                             rotvec=np.zeros((3, 3)), confidence=.9) for offset in (-.4, .4)]
        return Predictor()

    def renderer(self, size, camera, faces):
        parent = self
        class Renderer:
            def render_many(self, rgb, bodies):
                parent.rendered.append(len(bodies))
                panel = rgb.copy()
                panel[60:68, 60:68] = 255
                return panel
            def close(self):
                parent.closed.append(True)
        return Renderer()

    def test_both_models_same_images_all_people_and_missing_detection(self):
        manifest = demo.run(self.args(), self.factory, self.renderer)
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual(self.loaded, [('student', 'none'), ('baseline', 'none')])
        self.assertEqual(self.rendered, [2, 2, 2])
        self.assertEqual(len(self.closed), 2)
        for i in range(2):
            np.testing.assert_array_equal(self.inputs[i][2], self.inputs[i + 2][2])
            np.testing.assert_array_equal(self.inputs[i][3], self.inputs[i + 2][3])
        for row in manifest['images']:
            with Image.open(self.output / 'comparison' / row['output_name']) as image:
                self.assertEqual(image.size, (384, 192))
        self.assertEqual(manifest['images'][1]['models']['baseline']['rendered'], 0)
        with self.assertRaises(FileExistsError):
            demo.run(self.args(), self.factory, self.renderer)

    def test_empty_corrupt_and_duplicate_inputs(self):
        photo = self.photos / 'one.jpg'
        self.assertEqual(demo.image_files([photo, photo]), [photo])
        empty = self.root / 'empty'; empty.mkdir()
        with self.assertRaises(ValueError):
            demo.image_files([empty])
        photo.write_bytes(b'corrupt')
        with self.assertRaises(OSError):
            demo.run(self.args(), self.factory, self.renderer)
        self.assertEqual(self.loaded, [])
        self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'failed')

    @unittest.skipUnless(os.environ.get('VIDEO_TEST_EGL') == '1', 'requires EGL')
    def test_multiple_meshes_share_depth_buffer(self):
        from demo_video import MeshRenderer, camera_matrix
        renderer = MeshRenderer(128, camera_matrix(128, 55), [[0, 1, 2]])
        try:
            rgb = np.zeros((128, 128, 3), np.uint8)
            near = np.array([[-.5, -.5, 2.], [.5, -.5, 2.], [0., .5, 2.]])
            far = near * 2
            np.testing.assert_array_equal(renderer.render_many(rgb, [near, far]),
                                          renderer.render_many(rgb, [far, near]))
            self.assertTrue(renderer.render_many(rgb, [near, far]).any())
        finally:
            renderer.close()


class ImageLauncherTests(unittest.TestCase):
    def test_paths_with_spaces_preflight_and_inside_forwarding(self):
        f = launcher.VideoLauncherTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.prepare_check_only()
        script = f.repo / 'slurm/demo/run_image_compare.sh'
        shutil.copyfile(launcher.ROOT / 'slurm/demo/run_image_compare.sh', script)
        (f.repo / 'demo_image_compare.py').write_text('# fixture')
        photos = f.repo / 'my photos'; photos.mkdir()
        f.env['IMAGES'] = str(photos)
        result = subprocess.run(['bash', str(script)], env=f.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(f.calls(), [])
        result = subprocess.run(['bash', str(script), '--inside', str(f.repo), '--images', str(photos)],
                                env=f.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(f.calls()[-1]['args'], ['-u', str(f.repo / 'demo_image_compare.py'), '--images', str(photos)])
