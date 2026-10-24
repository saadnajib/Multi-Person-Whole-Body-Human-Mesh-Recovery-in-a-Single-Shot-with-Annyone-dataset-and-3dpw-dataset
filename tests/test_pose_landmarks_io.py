"""Landmark coordinates/provenance tests without installing MediaPipe."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.video.extract_pose_landmarks import (
    LANDMARK_INDICES, bundled_solution_models, detector_options, frame_timestamp_ms, parse_args,
    people_from_result, sha256_file, write_json_exclusive,
)


class LandmarkIOTests(unittest.TestCase):
    def pose(self):
        return [SimpleNamespace(x=.2 + i * .01, y=.1 + i * .02,
                                visibility=.9, presence=.7) for i in range(33)]

    def test_landmarks_use_original_pixels_and_minimum_confidence(self):
        pose = self.pose()
        pose[11].visibility, pose[11].presence = .35, .85
        result = people_from_result(SimpleNamespace(pose_landmarks=[pose]), 1920, 1080)
        self.assertEqual(len(result), 1)
        person = result[0]
        self.assertEqual(list(person['landmarks']), list(LANDMARK_INDICES))
        x, y, confidence = person['landmarks']['left_shoulder']
        self.assertAlmostEqual(x, .31 * 1920)
        self.assertAlmostEqual(y, .32 * 1080)
        self.assertEqual(confidence, .35)
        self.assertEqual(person['landmarks']['right_ankle'][2], .7)
        self.assertAlmostEqual(person['bbox'][0], .2 * 1920)
        self.assertAlmostEqual(person['bbox'][3], .74 * 1080)
        json.dumps(result, allow_nan=False)

    def test_missing_and_unreliable_people_remain_missing(self):
        self.assertEqual(people_from_result(SimpleNamespace(pose_landmarks=[]), 640, 480), [])
        pose = self.pose()
        for point in pose:
            point.visibility = 0.
        self.assertEqual(people_from_result(SimpleNamespace(pose_landmarks=[pose]), 640, 480), [])

    def test_solutions_uses_visibility_when_presence_is_unset(self):
        pose = self.pose()
        for point in pose:
            point.presence = 0.  # Proto default for an unset presence field.
        result = people_from_result(SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pose)),
                                    640, 480, visibility_only=True)
        self.assertEqual(result[0]['landmarks']['left_ankle'][2], .9)
        self.assertEqual(people_from_result(SimpleNamespace(pose_landmarks=None), 640, 480,
                                           visibility_only=True), [])

    def test_solutions_requires_both_bundled_models_before_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp = SimpleNamespace(__file__=str(root / '__init__.py'))
            with self.assertRaisesRegex(FileNotFoundError, 'bundled model'):
                bundled_solution_models(mp)
            for relative in ('pose_landmark/pose_landmark_full.tflite', 'pose_detection/pose_detection.tflite'):
                path = root / 'modules' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            self.assertEqual(len(bundled_solution_models(mp)), 2)

    def test_nonfinite_points_are_omitted_and_coordinates_are_not_fabricated(self):
        pose = self.pose()
        pose[11].x = float('nan')
        pose[12].x = -0.1
        result = people_from_result(SimpleNamespace(pose_landmarks=[pose]), 640, 480)[0]
        self.assertNotIn('left_shoulder', result['landmarks'])
        self.assertEqual(result['landmarks']['right_shoulder'][0], -64.)
        self.assertEqual(result['bbox'][0], 0.)
        json.dumps(result, allow_nan=False)

    def test_people_stay_separate_and_all_observations_are_independent(self):
        pose_a, pose_b = self.pose(), self.pose()
        for point in pose_b:
            point.x += .4
        result = people_from_result(SimpleNamespace(pose_landmarks=[pose_a, pose_b]), 100, 100)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[1]['landmarks']['left_hip'][0] - result[0]['landmarks']['left_hip'][0], 40.)
        result[0]['landmarks']['left_hip'][0] = 0.
        self.assertNotEqual(result[1]['landmarks']['left_hip'][0], 0.)

    def test_timestamps_are_source_based_and_monotonic(self):
        self.assertEqual(frame_timestamp_ms(0, 29.97), 0)
        self.assertEqual(frame_timestamp_ms(30, 29.97), 1001)
        self.assertEqual(frame_timestamp_ms(1, 2000., 0), 1)
        for fps in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                frame_timestamp_ms(0, fps)

    def test_atomic_output_refuses_overwrite_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observations.json'
            write_json_exclusive(path, {'frames': []})
            self.assertEqual(json.loads(path.read_text()), {'frames': []})
            digest = sha256_file(path)
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {'frames': [1]})
            self.assertEqual(sha256_file(path), digest)
            invalid = Path(directory) / 'invalid.json'
            with self.assertRaises(ValueError):
                write_json_exclusive(invalid, {'value': float('nan')})
            self.assertFalse(invalid.exists())
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['observations.json'])

    def test_options_select_cpu_video_and_no_segmentation(self):
        def options(**kwargs):
            return SimpleNamespace(**kwargs)
        options.Delegate = SimpleNamespace(CPU='cpu')
        mp = SimpleNamespace(tasks=SimpleNamespace(BaseOptions=options, vision=SimpleNamespace(
            PoseLandmarkerOptions=lambda **kwargs: kwargs, RunningMode=SimpleNamespace(VIDEO='video'))))
        selected = detector_options(mp, Path('/models/pose.task'), 2)
        self.assertEqual(selected['base_options'].delegate, 'cpu')
        self.assertEqual(selected['base_options'].model_asset_path, '/models/pose.task')
        self.assertEqual(selected['running_mode'], 'video')
        self.assertEqual(selected['num_poses'], 2)
        self.assertFalse(selected['output_segmentation_masks'])
        args = parse_args(['--video', 'a.mp4', '--model', 'b.task', '--output', 'c.json'])
        self.assertEqual(args.max_frames, 0)
        self.assertEqual(args.num_poses, 1)
        self.assertEqual(args.backend, 'solutions')


if __name__ == '__main__':
    unittest.main()
