"""Exercise graphics setup and launch ordering without installing anything.

The real Bash wrapper and EGL helper run against a temporary repository with
mock apt-get, pip and python executables. No GPU, conda activation or native
package operation is performed by these tests.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VideoLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='video-launcher-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'project with spaces'
        self.bin = self.root / 'mock-bin'
        self.bin.mkdir()
        for name in ('slurm/demo/run_video_compare.sh', 'slurm/egl.sh',
                     'scripts/video/check_renderer.py'):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        (self.repo / 'demo_video.py').write_text('# fixture; the mock Python records arguments\n')
        (self.repo / 'slurm/env.sh').write_text(
            'export VIDEO_TEST_ENV_SOURCED=1\ncd "$PROJECT_ROOT"\n')
        self.log = self.root / 'calls.jsonl'
        self.env = os.environ.copy()
        for name in ('PY_VENV', 'CONDA_ENV', 'EGL_INSTALL_PYTHON', 'EGL_VENDOR_DIR',
                     '__EGL_VENDOR_LIBRARY_DIRS', '__EGL_VENDOR_LIBRARY_FILENAMES',
                     'STUDENT_ONLY', 'TEACHER', 'AMP', 'OUTPUT', 'RUN', 'SLURM_JOB_ID'):
            self.env.pop(name, None)
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env.get('PATH', ''),
                        VIDEO_TEST_CALL_LOG=str(self.log), VIDEO_TEST_PROBE_EXIT='0')
        mock = '''#!{python}
import json
import os
from pathlib import Path
import sys
tool = Path(sys.argv[0]).name
names = ('EGL_VENDOR_DIR', '__EGL_VENDOR_LIBRARY_DIRS', '__EGL_VENDOR_LIBRARY_FILENAMES',
         'EGL_INSTALL_PYTHON', 'PYOPENGL_PLATFORM', 'EGL_DEVICE_ID',
         'CONDA_ENV', 'PY_VENV', 'VIDEO_TEST_ENV_SOURCED', 'NVIDIA_DRIVER_CAPABILITIES')
entry = dict(tool=tool, args=sys.argv[1:], env={{name: os.environ.get(name) for name in names}})
vendor = os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')
if vendor and Path(vendor).is_file():
    entry['vendor'] = json.loads(Path(vendor).read_text())
with open(os.environ['VIDEO_TEST_CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(entry) + '\\n')
if tool == 'python' and any(arg.endswith('/check_renderer.py') for arg in sys.argv[1:]):
    raise SystemExit(int(os.environ['VIDEO_TEST_PROBE_EXIT']))
'''.format(python=sys.executable)
        for name in ('apt-get', 'pip', 'python'):
            executable = self.bin / name
            executable.write_text(mock)
            executable.chmod(0o755)
        self.addCleanup(self.cleanup_vendor_directories)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def cleanup_vendor_directories(self):
        directories = {entry['env'].get('EGL_VENDOR_DIR') for entry in self.calls()}
        for name in directories:
            if name:
                path = Path(name)
                # Only remove the private mktemp directories created by this run.
                if path.parent == Path('/tmp') and path.name.startswith('multihmr-video-egl.'):
                    shutil.rmtree(path, ignore_errors=True)

    def inside(self, *args):
        return subprocess.run(
            ['bash', str(self.repo / 'slurm/demo/run_video_compare.sh'), '--inside', str(self.repo), *args],
            env=self.env, capture_output=True, text=True, timeout=10)

    def test_inside_registers_nvidia_and_probes_before_inference_without_pip(self):
        arguments = ['--video', str(self.repo / 'man walking.mp4'), '--student-checkpoint',
                     str(self.repo / 'student run/checkpoint.pt'), '--output', str(self.repo / 'new result')]
        # The inner launch must always select regular multihmr and clear a venv overlay.
        self.env.update(CONDA_ENV='wrong-environment', PY_VENV='/do/not/activate')
        result = self.inside(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([row['tool'] for row in calls], ['apt-get', 'apt-get', 'python', 'python'])
        self.assertEqual(calls[0]['args'], ['update', '-qq'])
        self.assertEqual(calls[1]['args'], ['install', '-y', '-qq', 'libegl1', 'libglvnd0', 'libgl1', 'libgles2'])
        probe, inference = calls[-2:]
        self.assertEqual(probe['args'], ['-u', str(self.repo / 'scripts/video/check_renderer.py')])
        self.assertEqual(inference['args'], ['-u', str(self.repo / 'demo_video.py'), *arguments])
        for row in (probe, inference):
            environment = row['env']
            self.assertEqual(environment['CONDA_ENV'], 'multihmr')
            self.assertIsNone(environment['PY_VENV'])
            self.assertEqual(environment['VIDEO_TEST_ENV_SOURCED'], '1')
            self.assertEqual(environment['EGL_INSTALL_PYTHON'], '0')
            self.assertEqual(environment['PYOPENGL_PLATFORM'], 'egl')
            self.assertEqual(environment['EGL_DEVICE_ID'], '0')
            self.assertEqual(environment['NVIDIA_DRIVER_CAPABILITIES'], 'all')
            self.assertEqual(environment['__EGL_VENDOR_LIBRARY_DIRS'], environment['EGL_VENDOR_DIR'])
            self.assertEqual(environment['__EGL_VENDOR_LIBRARY_FILENAMES'],
                             str(Path(environment['EGL_VENDOR_DIR']) / '10_nvidia.json'))
            self.assertEqual(row['vendor'], {'file_format_version': '1.0.0',
                                             'ICD': {'library_path': 'libEGL_nvidia.so.0'}})

    def test_failed_probe_prevents_model_inference(self):
        self.env['VIDEO_TEST_PROBE_EXIT'] = '23'
        result = self.inside('--video', 'will not be opened.mp4')
        self.assertEqual(result.returncode, 23)
        python_calls = [row for row in self.calls() if row['tool'] == 'python']
        self.assertEqual(len(python_calls), 1)
        self.assertTrue(python_calls[0]['args'][1].endswith('/check_renderer.py'))
        self.assertNotIn('pip', [row['tool'] for row in self.calls()])

    def test_shared_egl_helper_keeps_legacy_python_install_default(self):
        # The custom vendor directory avoids all system-directory writes.
        self.env['EGL_VENDOR_DIR'] = str(self.root / 'legacy vendor directory')
        result = subprocess.run(['bash', '-c', 'source "$1"', 'test-egl-helper',
                                 str(self.repo / 'slurm/egl.sh')], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row['tool'] for row in self.calls()], ['apt-get', 'apt-get', 'pip'])
        self.assertEqual(self.calls()[-1]['args'], ['install', '--quiet', '--exists-action', 'i', 'pyglet', 'pyopengl'])
        vendor = json.loads((Path(self.env['EGL_VENDOR_DIR']) / '10_nvidia.json').read_text())
        self.assertEqual(vendor['ICD']['library_path'], 'libEGL_nvidia.so.0')

    def test_check_only_never_provisions_or_runs_python(self):
        video = self.repo / 'man walking.mp4'
        student = self.repo / 'student run/train/checkpoints/00029.pt'
        baseline = self.repo / 'baseline.pt'
        student.parent.mkdir(parents=True)
        for path in (video, student, baseline):
            path.write_text('readable fixture')
        (student.parent.parent.parent / 'student_fix_arg.txt').write_text('none\n')
        self.env.update(REPO=str(self.repo), VIDEO=str(video), STUDENT=str(student),
                        BASELINE=str(baseline), OUTPUT=str(self.repo / 'new result'),
                        CHECK_ONLY='1', CONDA_ENV='multihmr')
        result = subprocess.run(['bash', str(self.repo / 'slurm/demo/run_video_compare.sh')],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('paths and options passed', result.stdout)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.repo / 'new result').exists())


if __name__ == '__main__':
    unittest.main()
