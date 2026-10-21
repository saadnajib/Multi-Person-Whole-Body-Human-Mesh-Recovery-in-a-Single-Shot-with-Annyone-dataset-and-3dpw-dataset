"""Renderer startup must fail visibly before an expensive model is loaded."""

import contextlib
import io
import os
import unittest
from unittest import mock

import numpy as np

from scripts.video import check_renderer


class RendererCheckTests(unittest.TestCase):
    def invoke(self, renderer_factory):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch('demo_video.MeshRenderer', renderer_factory), \
                mock.patch.object(check_renderer, 'describe_environment'), \
                mock.patch.object(check_renderer, 'describe_graphics_context'), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = check_renderer.main()
        return result, stdout.getvalue(), stderr.getvalue()

    def test_initialization_failure_preserves_original_cause(self):
        def fail(*args):
            try:
                raise RuntimeError('EGL_NOT_INITIALIZED: injected native failure')
            except RuntimeError as cause:
                raise RuntimeError('Renderer initialization failed') from cause

        result, stdout, stderr = self.invoke(fail)
        self.assertEqual(result, 1)
        self.assertNotIn('PASS', stdout)
        self.assertIn('FAILED before checkpoint loading', stderr)
        self.assertIn('EGL_NOT_INITIALIZED: injected native failure', stderr)
        self.assertIn('Renderer initialization failed', stderr)
        self.assertIn('Traceback', stderr)

    def test_blank_render_fails_and_releases_context(self):
        renderer = mock.Mock()
        renderer.render.return_value = np.zeros((64, 64, 3), np.uint8)
        result, stdout, stderr = self.invoke(mock.Mock(return_value=renderer))
        self.assertEqual(result, 1)
        self.assertNotIn('PASS', stdout)
        self.assertIn('not drawn at the expected position', stderr)
        renderer.close.assert_called_once_with()

    def test_success_is_reported_after_context_cleanup(self):
        renderer = mock.Mock()
        rendered = np.zeros((64, 64, 3), np.uint8)
        rendered[30:34, 30:34] = 127
        renderer.render.return_value = rendered
        result, stdout, stderr = self.invoke(mock.Mock(return_value=renderer))
        self.assertEqual(result, 0)
        self.assertIn('PASS', stdout)
        self.assertEqual(stderr, '')
        renderer.close.assert_called_once_with()

        # A failed context release must not produce a success message.
        renderer.close.side_effect = RuntimeError('injected cleanup failure')
        result, stdout, stderr = self.invoke(mock.Mock(return_value=renderer))
        self.assertEqual(result, 1)
        self.assertNotIn('PASS', stdout)
        self.assertIn('injected cleanup failure', stderr)

    @unittest.skipUnless(os.environ.get('VIDEO_TEST_EGL') == '1', 'opt in with VIDEO_TEST_EGL=1')
    def test_actual_egl_render(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = check_renderer.main()
        self.assertEqual(result, 0, stderr.getvalue())
        self.assertIn('PASS', stdout.getvalue())
        self.assertIn('GL_RENDERER=', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
