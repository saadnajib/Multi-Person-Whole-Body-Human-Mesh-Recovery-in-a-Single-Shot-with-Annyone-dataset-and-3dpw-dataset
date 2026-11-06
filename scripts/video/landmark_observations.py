"""Validate independent video landmarks and associate them with one raw detection.

This module needs only NumPy. MediaPipe runs separately so the GPU training
and inference environment does not need its newer dependencies.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class LandmarkSequence:
    def __init__(self, path, video, info):
        self.path = Path(path)
        with self.path.open() as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('frames'), list):
            raise ValueError('Landmark file must use schema_version 1 and contain frames')
        if data.get('coordinate_system', 'original_image_pixels') != 'original_image_pixels':
            raise ValueError('Landmarks must use original_image_pixels coordinates')
        source = data.get('video', {})
        if not isinstance(source, dict):
            raise ValueError('Landmark video metadata must be an object')
        for key in ('width', 'height'):
            if source.get(key) != info[key]:
                raise ValueError('Landmark {} does not match the source video'.format(key))
        try:
            matching_fps = math.isclose(float(source['fps']), info['fps'], rel_tol=1e-5, abs_tol=1e-4)
        except (KeyError, TypeError, ValueError):
            matching_fps = False
        if not matching_fps:
            raise ValueError('Landmark FPS does not match the source video')
        if source.get('sha256') != sha256_file(video):
            raise ValueError('Landmarks belong to a different video (SHA256 mismatch); use the original input video')
        self.width, self.height = info['width'], info['height']
        self.frames = {}
        for row in data['frames']:
            if not isinstance(row, dict):
                raise ValueError('Each landmark frame must be an object')
            index = row.get('frame_index')
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in self.frames:
                raise ValueError('Landmark frame indices must be unique nonnegative integers')
            if info['reported_frame_count'] > 0 and index >= info['reported_frame_count']:
                raise ValueError('Landmark frame index lies outside the source video')
            if not isinstance(row.get('people'), list):
                raise ValueError('Every landmark frame must have a people list')
            detector_ms = row.get('detector_ms')
            if detector_ms is not None and (not isinstance(detector_ms, (float, int))
                                            or not math.isfinite(detector_ms) or detector_ms < 0):
                raise ValueError('Landmark detector_ms must be finite and nonnegative')
            for person in row['people']:
                if not isinstance(person, dict):
                    raise ValueError('Each landmark person must be an object')
                bbox = np.asarray(person.get('bbox'), dtype=float)
                points = person.get('landmarks')
                if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
                    raise ValueError('Invalid landmark person bounding box')
                if not isinstance(points, dict):
                    raise ValueError('Landmarks must be a name-to-[x,y,confidence] mapping')
                for value in points.values():
                    point = np.asarray(value, dtype=float)
                    if point.shape != (3,) or not np.isfinite(point).all() or not 0 <= point[2] <= 1:
                        raise ValueError('Invalid landmark coordinate/confidence')
            self.frames[index] = row
        self.metadata = dict(path=str(self.path.resolve()), sha256=sha256_file(self.path),
                             source_video_sha256=source['sha256'], observations=len(self.frames),
                             detector=data.get('detector', {}),
                             timing='Landmarks extracted offline; detector work excluded from runtime processing FPS')

    def observation(self, frame_index, panel_size, target_bbox):
        row = self.frames.get(frame_index)
        if row is None:
            return [], {'status': 'missing_landmark_frame'}
        if not row['people']:
            return [], {'status': 'no_landmark_person'}
        scale = panel_size / max(self.width, self.height)
        nw, nh = max(1, round(self.width * scale)), max(1, round(self.height * scale))
        factors = np.array([nw / self.width, nh / self.height])
        origin = np.array([(panel_size - nw) // 2, (panel_size - nh) // 2])
        target = np.asarray(target_bbox, dtype=float)
        candidates = []
        for index, person in enumerate(row['people']):
            bbox = np.asarray(person['bbox'], dtype=float).reshape(2, 2) * factors + origin
            bbox = bbox.reshape(4)
            low, high = np.maximum(target[:2], bbox[:2]), np.minimum(target[2:], bbox[2:])
            overlap = np.prod(np.maximum(high - low, 0.))
            union = np.prod(target[2:] - target[:2]) + np.prod(bbox[2:] - bbox[:2]) - overlap
            iou = float(overlap / union) if union > 0 else 0.
            center_distance = float(np.linalg.norm((target[:2] + target[2:] - bbox[:2] - bbox[2:]) / 2)
                                    / max(np.linalg.norm(target[2:] - target[:2]), 1.))
            if iou >= .1 and center_distance < .6:
                candidates.append((iou, index, person))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return [], {'status': 'landmark_person_not_matched'}
        if len(candidates) > 1 and candidates[1][0] >= .85 * candidates[0][0]:
            return [], {'status': 'ambiguous_landmark_person'}
        iou, index, person = candidates[0]
        points = [dict(name=name, x=float(value[0] * factors[0] + origin[0]),
                       y=float(value[1] * factors[1] + origin[1]), confidence=float(value[2]))
                  for name, value in person['landmarks'].items()]
        return points, dict(status='matched', person_index=index, match_iou=iou,
                            detector_ms=row.get('detector_ms'))
