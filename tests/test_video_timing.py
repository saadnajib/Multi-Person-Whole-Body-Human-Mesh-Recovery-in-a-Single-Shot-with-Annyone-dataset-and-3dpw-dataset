"""Check timing boundaries without requiring a CUDA device or trained weights."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from demo_video import TorchPredictor, camera_matrix


@unittest.skipIf(torch is None, 'PyTorch required for predictor timing tests')
class PredictorTimingTests(unittest.TestCase):
    def test_synchronized_forward_excludes_preprocessing_and_output_transfer(self):
        events = []

        class Model:
            img_size = 28
            body_model = SimpleNamespace(faces=np.array([[0, 1, 2]]),
                                         phenotype_labels=['age'], bone_labels=['root'])

            def __call__(self, image, **kwargs):
                events.append('model')
                self.image = image
                return ({}, [])  # Original Multi-HMR empty-detection path still gets timed.

        model = Model()
        predictor = TorchPredictor(model, torch.device('cpu'), False, {})
        times = iter([0, .002, .003, .013, .014, .017])

        def clock():
            events.append('clock')
            return next(times)

        with mock.patch.object(predictor, 'synchronize', side_effect=lambda: events.append('sync')), \
                mock.patch('demo_video.time.perf_counter', side_effect=clock):
            result = predictor.predict(np.zeros((28, 28, 3), np.uint8), camera_matrix(28, 55))
        self.assertEqual(result, [])
        self.assertEqual(events, ['sync', 'clock', 'sync', 'clock', 'clock', 'model', 'sync', 'clock', 'clock', 'clock'])
        self.assertAlmostEqual(predictor.last_timings['preprocess_ms'], 2)
        self.assertAlmostEqual(predictor.last_timings['inference_ms'], 10)
        self.assertAlmostEqual(predictor.last_timings['output_transfer_ms'], 3)
        self.assertEqual(tuple(model.image.shape), (1, 3, 28, 28))

    def test_cuda_synchronization_uses_the_selected_device(self):
        predictor = TorchPredictor.__new__(TorchPredictor)
        predictor.device = torch.device('cuda:2')
        with mock.patch('torch.cuda.synchronize') as synchronize:
            predictor.synchronize()
            synchronize.assert_called_once_with(torch.device('cuda:2'))
        predictor.device = torch.device('cpu')
        with mock.patch('torch.cuda.synchronize') as synchronize:
            predictor.synchronize()
            synchronize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
