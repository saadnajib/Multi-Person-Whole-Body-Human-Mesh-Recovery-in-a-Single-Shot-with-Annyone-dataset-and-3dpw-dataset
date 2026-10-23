"""Check that temporal parameters reconstruct the same anatomical body.

The fake decoder makes shape, pose and anchor errors independently observable.
A real Anny smoke is opt-in because its assets are not shipped in the repository:
VIDEO_TEST_ANNY=1 python -m unittest discover -s tests -p test_temporal_decode.py
"""

import importlib.util
import os
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
    import roma
except ImportError:
    torch = roma = None

from demo_video import TorchPredictor


@unittest.skipIf(torch is None or roma is None, 'PyTorch and roma required for decoding')
class TemporalDecodeTests(unittest.TestCase):
    def setUp(self):
        class Body:
            faces = np.array([[0, 1, 2]], np.int64)
            phenotype_labels = ['gender', 'age', 'muscle', 'weight', 'height',
                                'proportions', 'cupsize', 'firmness', 'african', 'asian', 'caucasian']
            bone_labels = ['pelvis', 'head', 'hand']
            skinning_method = 'existing_skinning_mode'

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                self.grad_enabled = torch.is_grad_enabled()
                self.inference_mode = torch.is_inference_mode_enabled()
                pose = kwargs['pose_parameters']
                shape = torch.stack([kwargs['phenotype_kwargs'][k] for k in self.phenotype_labels], dim=-1)
                # Distinct nonzero offsets make accidental pelvis placement visible.
                offsets = pose.new_tensor([[0.4, 0.2, -0.1], [-0.2, -1.0, 0.3], [0.9, -0.6, 0.1]])
                offsets = offsets[None].repeat(pose.shape[0], 1, 1)
                offsets[:, 1, 1] -= shape[:, 4]
                offsets[:, 2, 0] += shape[:, 5]
                joints = torch.einsum('bnij,bnj->bni', pose[:, :, :3, :3], offsets)
                bones = pose.clone()
                bones[:, :, :3, 3] = joints
                # Include all phenotypes in the surface, beyond merely observing kwargs.
                shape_shift = (shape * torch.arange(1, 12, device=pose.device, dtype=pose.dtype)).sum(-1)
                vertices = joints + torch.stack([shape_shift * .01, shape_shift * 0., shape_shift * .02], dim=-1)[:, None]
                self.decoded_joints = joints.detach().clone()
                self.decoded_vertices = vertices.detach().clone()
                return {'vertices': vertices, 'bone_poses': bones}

        self.body = Body()
        self.model = SimpleNamespace(img_size=672, body_model=self.body, person_center_idx=1)
        self.model._apply_bone_fix = lambda *a: self.fail('Correction was applied a second time')
        self.rotvec = np.array([[0., 0., np.pi / 2], [np.pi / 2, 0., 0.], [0., -np.pi / 2, 0.]], np.float32)
        self.shape = np.array([0.0, .12, .23, .34, .45, .56, .67, .78, .89, 1.0, .31], np.float32)
        self.anchor = np.array([.7, -.4, 4.2], np.float32)

    def predictor(self, convention=None):
        return TorchPredictor(self.model, torch.device('cpu'), False,
                              {'decode_pose_parameterization': convention})

    def test_complete_parameters_and_head_anchoring_without_recalibration(self):
        before = [x.copy() for x in (self.rotvec, self.shape, self.anchor)]
        output = self.predictor().decode_temporal(self.rotvec, self.shape, self.anchor)
        pose = self.body.kwargs['pose_parameters'].numpy()
        self.assertEqual(pose.shape, (1, 3, 4, 4))
        # Known quarter-turn matrices, independent of the implementation's converter.
        expected = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                             [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
                             [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]])
        np.testing.assert_allclose(pose[0, :, :3, :3], expected, atol=2e-7)
        np.testing.assert_array_equal(pose[:, :, :3, 3], 0.)
        np.testing.assert_array_equal(pose[0, :, 3, :], np.tile([0., 0., 0., 1.], (3, 1)))
        phenotypes = self.body.kwargs['phenotype_kwargs']
        self.assertEqual(list(phenotypes), self.body.phenotype_labels)
        for i, name in enumerate(self.body.phenotype_labels):
            np.testing.assert_array_equal(phenotypes[name].numpy(), self.shape[i:i + 1])
            self.assertEqual(phenotypes[name].dtype, torch.float32)
        np.testing.assert_array_equal(output['shape'], self.shape)
        np.testing.assert_array_equal(output['rotvec'], self.rotvec)
        np.testing.assert_allclose(output['joints'][1], self.anchor, atol=5e-7)
        self.assertGreater(np.linalg.norm(output['joints'][0] - self.anchor), .1)
        base_joints = self.body.decoded_joints[0].numpy()
        base_vertices = self.body.decoded_vertices[0].numpy()
        offset = self.anchor - base_joints[1]
        np.testing.assert_allclose(output['joints'], base_joints + offset, atol=5e-7)
        np.testing.assert_allclose(output['vertices'], base_vertices + offset, atol=5e-7)
        # Re-decoding must preserve body geometry; only a common translation follows it.
        np.testing.assert_allclose(output['vertices'] - output['joints'], base_vertices - base_joints, atol=5e-7)
        self.assertFalse(self.body.grad_enabled)
        self.assertTrue(self.body.inference_mode)
        self.assertEqual(self.body.skinning_method, 'existing_skinning_mode')
        for original, value in zip(before, (self.rotvec, self.shape, self.anchor)):
            np.testing.assert_array_equal(original, value)

    def test_student_uses_decoder_default_and_baseline_passes_explicit_convention(self):
        default = self.predictor().decode_temporal(self.rotvec, self.shape, self.anchor)
        self.assertNotIn('pose_parameterization', self.body.kwargs)
        baseline = self.predictor('root_relative_world').decode_temporal(self.rotvec, self.shape, self.anchor)
        self.assertEqual(self.body.kwargs['pose_parameterization'], 'root_relative_world')
        np.testing.assert_array_equal(default['vertices'], baseline['vertices'])
        # Forward the chosen setting, rather than replacing it with a second hardcoded default.
        self.predictor('rest_relative').decode_temporal(self.rotvec, self.shape, self.anchor)
        self.assertEqual(self.body.kwargs['pose_parameterization'], 'rest_relative')


@unittest.skipUnless(os.environ.get('VIDEO_TEST_ANNY') == '1', 'set VIDEO_TEST_ANNY=1 for cached Anny smoke')
@unittest.skipIf(torch is None or roma is None or importlib.util.find_spec('anny') is None,
                 'PyTorch, roma and Anny required')
class RealAnnyTemporalDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import anny
        cls.body = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().eval()
        cls.body.set_skinning_method('lbs')

    def test_nontrivial_parameters_match_native_anny_geometry_for_both_conventions(self):
        body = self.body
        anchor_index = list(body.bone_labels).index('head')
        model = SimpleNamespace(img_size=672, body_model=body, person_center_idx=anchor_index)
        # Keep a nontrivial small articulated pose, with a camera-facing root rotation.
        rng = np.random.default_rng(418)
        rotvec = rng.normal(0, .08, (len(body.bone_labels), 3)).astype(np.float32)
        rotvec[0] = [np.pi / 2, -.14, .08]
        shape = np.linspace(.18, .81, len(body.phenotype_labels), dtype=np.float32)
        anchor = np.array([.17, -.31, 3.7], np.float32)
        matrices = roma.rotvec_to_rotmat(torch.from_numpy(rotvec))
        homogeneous = torch.eye(4).repeat(1, len(body.bone_labels), 1, 1)
        homogeneous[0, :, :3, :3] = matrices
        shape_kwargs = {key: torch.tensor([float(shape[i])]) for i, key in enumerate(body.phenotype_labels)}
        for convention in (None, 'root_relative_world'):
            with self.subTest(convention=convention):
                kwargs = dict(pose_parameters=homogeneous, phenotype_kwargs=shape_kwargs)
                if convention is not None:
                    kwargs['pose_parameterization'] = convention
                with torch.inference_mode():
                    native = body(**kwargs)
                joints = native['bone_poses'][0, :, :3, -1].numpy()
                expected_joints = joints - joints[anchor_index] + anchor
                expected_vertices = native['vertices'][0].numpy() - joints[anchor_index] + anchor
                # Match the actual image-model interface, which exports rotations via rotvec.
                exported_rotvec = roma.rotmat_to_rotvec(matrices).numpy()
                predictor = TorchPredictor(model, torch.device('cpu'), False,
                                           {'decode_pose_parameterization': convention})
                output = predictor.decode_temporal(exported_rotvec, shape, anchor)
                self.assertTrue(np.isfinite(output['vertices']).all())
                self.assertGreater(len(output['vertices']), 1000)
                np.testing.assert_allclose(output['joints'], expected_joints, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(output['vertices'], expected_vertices, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(output['joints'][anchor_index], anchor, atol=5e-7)


if __name__ == '__main__':
    unittest.main()
