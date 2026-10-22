"""CPU-only verification of tracking identity and temporal metric semantics."""

import json
import unittest

import numpy as np

from scripts.video.video_utils import SinglePersonTracker, summarize_stability, summarize_timing


def detection(x, confidence=0.9):
    return {"bbox": [x, 0, x + 10, 20], "confidence": confidence}


def record(frame, time, shape=(0.2, 0.8), root=(0, 0, 0), relative_joint=(1, 0, 0), segment=1):
    root = np.asarray(root, dtype=float)
    return {"frame_index": frame, "timestamp_s": time, "selected_index": 0,
            "segment_id": segment, "shape": list(shape), "root": root,
            "joints": np.stack([root, root + relative_joint])}


class TimingTests(unittest.TestCase):
    def test_warmup_excluded_and_missed_detection_still_timed(self):
        rows = [dict(inference_ms=1000, render_ms=500, selected_index=0),
                dict(inference_ms=10, render_ms=8, selected_index=0),
                dict(inference_ms=30, selected_index=None)]
        result = summarize_timing(rows, 1, 24)
        self.assertEqual(result['inference_fps'], 50)  # 2 / .040, not (100 + 33.3) / 2.
        self.assertEqual(result['measured_frames'], 2)
        self.assertEqual(result['render_ms']['count'], 1)
        self.assertEqual(result['render_ms']['mean'], 8)
        self.assertEqual(result['inference_ms']['median'], 20)
        self.assertEqual(result['playback_fps'], 24)

    def test_short_clip_and_invalid_timings_do_not_invent_fps(self):
        for rows, warmup in [([dict(inference_ms=5)], 10),
                             ([dict(inference_ms=float('nan')), dict(inference_ms=0), {}], 0)]:
            result = summarize_timing(rows, warmup, 30)
            self.assertIsNone(result['inference_fps'])
            self.assertEqual(result['measured_frames'], 0)
            json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            summarize_timing([], -1, 30)


class TrackingTests(unittest.TestCase):
    def test_identity_wins_over_confidence(self):
        tracker = SinglePersonTracker()
        self.assertEqual(tracker.update([detection(0, 0.9), detection(100, 0.5)])["selected_index"], 0)
        match = tracker.update([detection(100, 0.99), detection(1, 0.1)])
        self.assertEqual(match["selected_index"], 1)
        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["segment_id"], 1)

    def test_brief_misses_keep_segment_and_expiry_starts_another(self):
        tracker = SinglePersonTracker(max_gap=2)
        tracker.update([detection(0)])
        self.assertIsNone(tracker.update([detection(100)])["selected_index"])
        match = tracker.update([detection(1)])
        self.assertEqual(match["segment_id"], 1)
        self.assertEqual(match["gap_before_match"], 1)
        tracker.update([])
        tracker.update([])
        reacquired = tracker.update([detection(100)])
        self.assertEqual(reacquired["status"], "reacquired")
        self.assertEqual(reacquired["segment_id"], 2)

    def test_invalid_boxes_are_ignored_without_renumbering(self):
        tracker = SinglePersonTracker()
        detections = [detection(float("nan")), {"bbox": [0, 0, 0, 2], "confidence": 1}, detection(10)]
        self.assertEqual(tracker.update(detections)["selected_index"], 2)
        json.dumps(tracker.update([]), allow_nan=False)

    def test_miss_count_includes_initial_and_expired_gaps(self):
        tracker = SinglePersonTracker(max_gap=0)
        self.assertEqual(tracker.update([])["missed_frames"], 1)
        self.assertEqual(tracker.update([detection(0)])["status"], "started")
        self.assertEqual(tracker.update([])["missed_frames"], 1)
        self.assertEqual(tracker.update([])["missed_frames"], 2)
        self.assertEqual(tracker.update([detection(0)])["segment_id"], 2)


class StabilityTests(unittest.TestCase):
    def test_actual_dt_shape_variance_and_translation_invariance(self):
        rows = [record(0, 0, shape=(0.2, 0.8)),
                record(1, 0.5, shape=(0.4, 0.6), root=(1, 0, 0))]
        summary = summarize_stability(rows, ["age", "height"])
        self.assertAlmostEqual(summary["root_speed_m_per_s"]["mean"], 2)
        self.assertAlmostEqual(summary["phenotypes"]["age"]["value"]["std"], 0.1)
        self.assertAlmostEqual(summary["phenotypes"]["age"]["absolute_rate_per_s"]["mean"], 0.4)
        self.assertEqual(summary["pelvis_relative_joint_speed_m_per_s"]["mean"], 0)

    def test_acceleration_uses_midpoint_time_for_unequal_dt(self):
        # Relative position x=t**2 has acceleration 2 m/s**2.
        rows = [record(0, 0, relative_joint=(0, 0, 0)),
                record(1, 1, relative_joint=(1, 0, 0)),
                record(2, 3, relative_joint=(9, 0, 0))]
        summary = summarize_stability(rows)
        self.assertAlmostEqual(summary["pelvis_relative_joint_speed_m_per_s"]["mean"], 2.5)
        self.assertAlmostEqual(summary["pelvis_relative_joint_acceleration_m_per_s2"]["mean"], 2)

    def test_missing_frame_and_new_segment_break_differences(self):
        rows = [record(0, 0),
                {"frame_index": 1, "timestamp_s": 1, "selected_index": None, "segment_id": 1},
                record(2, 2, root=(100, 0, 0)),
                record(3, 3, root=(200, 0, 0), segment=2),
                record(4, 4, root=(201, 0, 0), segment=2)]
        summary = summarize_stability(rows)
        self.assertEqual(summary["consecutive_selected_pairs"], 1)
        self.assertEqual(summary["root_speed_m_per_s"]["mean"], 1)
        self.assertEqual(summary["joint_acceleration_triplets"], 0)

    def test_omitted_frames_break_unless_the_declared_stride_matches(self):
        rows = [record(0, 0), record(2, 0.5, root=(1, 0, 0))]
        self.assertEqual(summarize_stability(rows)["root_speed_m_per_s"]["count"], 0)
        result = summarize_stability(rows, expected_frame_step=2)
        self.assertEqual(result["root_speed_m_per_s"]["mean"], 2)

    def test_nonfinite_data_excluded_and_json_is_strict(self):
        rows = [record(0, 0, shape=(0.2, float("nan"))),
                record(1, 1, shape=(0.4, 0.6), root=(float("inf"), 0, 0)),
                record(2, 2, shape=(float("nan"), 0.8))]
        with np.errstate(invalid="ignore"):
            summary = summarize_stability(rows, ["age", "height"])
        self.assertEqual(summary["phenotypes"]["age"]["value"]["count"], 2)
        self.assertEqual(summary["phenotypes"]["age"]["signed_step"]["count"], 1)
        self.assertEqual(summary["root_speed_m_per_s"]["count"], 0)
        self.assertEqual(summary["pelvis_relative_joint_acceleration_m_per_s2"]["count"], 0)
        json.dumps(summary, allow_nan=False)

    def test_rotation_steps_and_invalid_rotation_exclusion(self):
        rows = [record(0, 0), record(1, 0.5)]
        rows[0]["rotmat"] = np.stack([np.eye(3), np.zeros((3, 3))])
        rows[1]["rotmat"] = np.stack([[[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.eye(3)])
        summary = summarize_stability(rows)
        self.assertEqual(summary["bone_rotation_step_degrees"]["count"], 1)
        self.assertAlmostEqual(summary["bone_rotation_step_degrees"]["mean"], 90)
        self.assertAlmostEqual(summary["bone_rotation_speed_degrees_per_s"]["mean"], 180)

    def test_nonpositive_times_break_temporal_metrics(self):
        rows = [record(0, 1), record(1, 1, root=(1, 0, 0)), record(2, 0, root=(2, 0, 0))]
        self.assertEqual(summarize_stability(rows)["consecutive_selected_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
