"""Landmark provenance, coordinate conversion and person association."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.video.landmark_observations import LandmarkSequence, sha256_file


class LandmarkObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='landmark-observations-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'original video.mp4'
        # The loader validates provenance and metadata, not video decoding.
        self.video.write_bytes(b'exact original video bytes')
        self.path = self.root / 'landmarks.json'
        self.info = dict(width=161, height=97, fps=29.97, reported_frame_count=10)
        self.person = dict(bbox=[30., 10., 90., 90.], landmarks={
            'left_shoulder': [40., 20., .9], 'left_hip': [50., 50., .8],
            'left_knee': [60., 70., .7], 'left_ankle': [50., 88., .95]})
        self.data = dict(schema_version=1,
                         video=dict(width=161, height=97, fps=29.97, sha256=sha256_file(self.video)),
                         detector={'name': 'fixture'}, frames=[
                             dict(frame_index=0, people=[copy.deepcopy(self.person)], detector_ms=3.5),
                             dict(frame_index=1, people=[])])

    def sequence(self, data=None):
        self.path.write_text(json.dumps(self.data if data is None else data))
        return LandmarkSequence(self.path, self.video, self.info)

    def bbox(self, person=None, size=128):
        person = self.person if person is None else person
        # Letterboxing rounds the resized height, so x/y scale factors differ
        # slightly. This expectation deliberately does not copy the loader code.
        return (np.array(person['bbox']).reshape(2, 2) * [128 / 161, 77 / 97] + [0, 25]).reshape(4)

    def test_exact_letterbox_mapping_and_provenance(self):
        sequence = self.sequence()
        observations, report = sequence.observation(0, 128, self.bbox())
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['person_index'], 0)
        self.assertAlmostEqual(report['match_iou'], 1.)
        self.assertEqual(report['detector_ms'], 3.5)
        self.assertEqual(len(observations), 4)
        self.assertEqual(observations[0]['name'], 'left_shoulder')
        self.assertAlmostEqual(observations[0]['x'], 40 * 128 / 161)
        self.assertAlmostEqual(observations[0]['y'], 20 * 77 / 97 + 25)
        self.assertEqual(observations[0]['confidence'], .9)
        self.assertEqual(sequence.metadata['source_video_sha256'], sha256_file(self.video))
        self.assertEqual(sequence.metadata['sha256'], sha256_file(self.path))
        self.assertEqual(sequence.metadata['observations'], 2)
        self.assertEqual(sequence.metadata['detector'], {'name': 'fixture'})
        self.assertEqual(sequence.metadata['path'], str(self.path.resolve()))
        json.dumps(sequence.metadata, allow_nan=False)

    def test_wrong_video_and_inconsistent_camera_metadata_are_rejected(self):
        for field, value in [('sha256', 'wrong'), ('width', 162), ('height', 98), ('fps', 30.)]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data['video'][field] = value
                with self.assertRaises(ValueError):
                    self.sequence(data)
        self.path.write_text(json.dumps(self.data))
        self.video.write_bytes(b'changed video content under the same filename')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            LandmarkSequence(self.path, self.video, self.info)

    def test_missing_empty_far_and_ambiguous_frames_never_invent_observations(self):
        sequence = self.sequence()
        for index, status in [(1, 'no_landmark_person'), (8, 'missing_landmark_frame')]:
            observations, report = sequence.observation(index, 128, self.bbox())
            self.assertEqual(observations, [])
            self.assertEqual(report['status'], status)
        observations, report = sequence.observation(0, 128, [115., 1., 127., 20.])
        self.assertEqual(observations, [])
        self.assertEqual(report['status'], 'landmark_person_not_matched')
        data = copy.deepcopy(self.data)
        data['frames'][0]['people'].append(copy.deepcopy(self.person))
        observations, report = self.sequence(data).observation(0, 128, self.bbox())
        self.assertEqual(observations, [])
        self.assertEqual(report['status'], 'ambiguous_landmark_person')

    def test_association_selects_the_matching_person(self):
        data = copy.deepcopy(self.data)
        far = copy.deepcopy(self.person)
        far['bbox'] = [120., 10., 160., 90.]
        far['landmarks'] = {k: [v[0] + 90., v[1], v[2]] for k, v in far['landmarks'].items()}
        data['frames'][0]['people'].insert(0, far)
        observations, report = self.sequence(data).observation(0, 128, self.bbox())
        self.assertEqual(report['status'], 'matched')
        self.assertEqual(report['person_index'], 1)
        self.assertAlmostEqual(observations[0]['x'], 40 * 128 / 161)

    def test_duplicate_invalid_and_out_of_range_frame_ids_are_rejected(self):
        for index in [0, True, -1, 1.5, 10]:
            data = copy.deepcopy(self.data)
            data['frames'][1]['frame_index'] = index
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'frame'):
                self.sequence(data)

    def test_invalid_boxes_landmark_shapes_and_confidences_are_rejected(self):
        people = [dict(self.person, bbox=[1, 2, 3]),
                  dict(self.person, bbox=[0, 0, 0, 2]),
                  dict(self.person, bbox=[0, 0, float('nan'), 2]),
                  dict(self.person, landmarks=[]),
                  dict(self.person, landmarks={'left_hip': [1, 2]}),
                  dict(self.person, landmarks={'left_hip': [1, 2, 2]}),
                  dict(self.person, landmarks={'left_hip': [1, float('inf'), .9]})]
        for person in people:
            data = copy.deepcopy(self.data)
            data['frames'][0]['people'] = [person]
            with self.subTest(person=person), self.assertRaises(ValueError):
                self.sequence(data)

    def test_bad_schema_is_rejected(self):
        for data in [[], dict(self.data, schema_version=2), dict(self.data, frames={}),
                     dict(self.data, coordinate_system='normalized'), dict(self.data, video=[]),
                     dict(self.data, frames=[[]]), dict(self.data, frames=[dict(frame_index=0, people=[[]])])]:
            with self.assertRaises(ValueError):
                self.sequence(data)

    def test_invalid_offline_timing_is_rejected(self):
        for value in [float('nan'), float('inf'), -.1, '3']:
            data = copy.deepcopy(self.data)
            data['frames'][0]['detector_ms'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'detector_ms'):
                self.sequence(data)


if __name__ == '__main__':
    unittest.main()
