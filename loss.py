# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from argparse import ArgumentParser

def _neg_loss(pred, gt):
  '''
  Code modified from: https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/models/losses.py#L42
    Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    
  '''
  assert pred.shape == gt.shape

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  eps = 1e-7

  pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class Loss(torch.nn.Module):
    def __init__(self, parser_args, *args, **kwargs):
        super().__init__()
        self.parser_args = parser_args
        # Per-joint loss weights [1, 163, 1, 1], configured by
        # configure_joint_weights() once the model (and its bone names) exist.
        # None = original behavior, every joint weighted equally.
        self.joint_weights = None
        # --alpha_global_rot: L1 on the GLOBAL (post-FK) orientation of a few named bones, indices
        # resolved once by configure_global_rot(). None = term not computed, total loss unchanged.
        self.global_rot_idx = None
        # --rotmat_region_norm: per-bone weights [1, 163, 1, 1] that turn the sum over bones into a mean within
        # body / hands / head-neck and then across the three regions (configure_region_norm). None = the sum.
        self.region_weights = None
        # --mask_3dpw_hands: [1, 163, 1, 1] mask of the bones whose 3DPW-origin rotations are converter
        # placeholders (configure_hand_label_mask). None = every bone of every person is supervised, as before.
        self.hand_label_mask = None
        # --shape_weights: per-phenotype multipliers on the L1 shape error [1, 11].
        # None (default, or all ones) = the original unweighted sum, byte-for-byte.
        self.shape_weights = None
        sw = str(getattr(parser_args, 'shape_weights', '') or '').strip()
        if sw:
            vals = [float(v) for v in sw.split(',')]
            if len(vals) != 11:
                raise ValueError(f"--shape_weights expects 11 comma-separated floats (Anny phenotypes: gender, age, "
                                 f"muscle, weight, height, proportions, cupsize, firmness, african, asian, caucasian), "
                                 f"got {len(vals)}")
            if any(v < 0 for v in vals):
                raise ValueError("--shape_weights must be >= 0")
            if any(v != 1.0 for v in vals):
                self.shape_weights = torch.tensor(vals, dtype=torch.float32).reshape(1, 11)
                print(f"[loss] shape weights {vals} (sum {sum(vals):.1f} vs 11 unweighted: alpha_shape still "
                      f"multiplies the total, so the shape share of the loss grows by that ratio)", flush=True)

    def configure_joint_weights(self, bone_labels):
        """Build per-joint weights from bone NAMES (robust to index changes).

        --boost_neck_weight  multiplies the loss on bones whose name contains
                             'neck' (the cervical chain) and on 'head'.
        --boost_hand_weight  multiplies bones containing 'finger', 'thumb'
                             or 'wrist'.
        Weights are renormalized to mean 1 over all joints, so boosting
        REDISTRIBUTES the loss budget toward these joints instead of
        inflating the total (alpha_rotmat keeps its meaning, and loss
        values stay roughly comparable across runs).
        """
        nb = float(getattr(self.parser_args, 'boost_neck_weight', 1.0))
        hb = float(getattr(self.parser_args, 'boost_hand_weight', 1.0))
        n = len(bone_labels)
        w = torch.ones(n)
        n_neck = n_hand = 0
        for i, name in enumerate(bone_labels):
            lo = name.lower()
            if ('neck' in lo) or (lo == 'head'):
                w[i] = nb
                n_neck += 1
            elif ('finger' in lo) or ('thumb' in lo) or ('wrist' in lo):
                w[i] = hb
                n_hand += 1
        w = w * (n / w.sum())  # renormalize: mean weight = 1
        self.joint_weights = w.reshape(1, n, 1, 1)
        eff_boost = float(w.max() / w.min()) if float(w.min()) > 0 else 0.0
        print(f"[loss] joint weights configured: {n_neck} neck/head bones x{nb}, "
              f"{n_hand} hand bones x{hb}, renormalized to mean 1 "
              f"(max/min weight ratio = {eff_boost:.2f})", flush=True)

    def configure_global_rot(self, bone_labels, bone_parents=None, active_mask=None):
        """Resolve --global_rot_bones to bone indices.

        A bone that is masked, or descends from a masked bone, is refused when active_mask is given
        (--mask_helper_joints_train): the prediction's FK would carry an identity helper rotation that
        the GT FK (built from the full pose) does not, an irreducible floor the term could never close.
        """
        labels = list(bone_labels)
        names = [s.strip() for s in str(self.parser_args.global_rot_bones).split(',') if s.strip()]
        if not names:
            raise ValueError("--global_rot_bones is empty while --alpha_global_rot > 0")
        if len(set(names)) != len(names):
            raise ValueError(f"--global_rot_bones lists a bone twice: {names}")
        idx = []
        for nm in names:
            if nm not in labels:
                raise ValueError(f"--global_rot_bones: unknown bone '{nm}' (examples: head, neck03, wrist.L; "
                                 f"{len(labels)} bone names in model.body_model.bone_labels)")
            i = labels.index(nm)
            j = i
            while active_mask is not None and j is not None and j >= 0:
                if not active_mask[j]:
                    raise ValueError(f"--global_rot_bones: '{nm}' is, or descends from, the helper bone "
                                     f"'{labels[j]}' masked by --mask_helper_joints_train")
                j = bone_parents[j] if bone_parents is not None else -1
            idx.append(i)
        self.global_rot_idx = torch.tensor(idx, dtype=torch.long)
        print(f"[loss] global_rot: alpha {float(self.parser_args.alpha_global_rot)} on {len(idx)} bones {names} "
              f"(indices {idx}); mean over these bones and people of the per-bone L1 over the 9 entries of the "
              f"camera-frame rotation (the rotmat term sums the same error over all bones)", flush=True)

    def configure_region_norm(self, bone_labels, hand_mask, active_mask=None):
        """--rotmat_region_norm: weights w_b = n_active / (3 |region(b)|), so the rotmat term becomes
        n_active x mean over {body, hands, headneck} of the mean per-bone error inside each region.

        The sum over bones gives a region a share proportional to its bone count: the 30 phalanges of
        hand_mask outweigh the 4-bone neck chain 7.5:1 and the --boost_* weights only moved that ratio
        (docs/diagnostics_findings.md). The n_active factor keeps alpha_rotmat's scale: with every weight 1
        the term is the previous sum. Helper bones masked by --mask_helper_joints_train carry a zero residual
        and get weight 0 so they do not dilute the body mean. Replaces the --boost_* weights in the rotmat
        term; the j3d term keeps them.
        """
        n = len(bone_labels)
        hand = torch.as_tensor(hand_mask).reshape(-1).bool()
        active = torch.ones(n, dtype=torch.bool) if active_mask is None else torch.as_tensor(active_mask).reshape(-1).bool()
        if hand.numel() != n or active.numel() != n:
            raise ValueError(f"--rotmat_region_norm: masks of {hand.numel()} / {active.numel()} bones for {n} bone names")
        headneck = torch.tensor([('neck' in s.lower()) or (s.lower() == 'head') for s in bone_labels])
        regions = {'headneck': headneck & active, 'hands': hand & active & ~headneck,
                   'body': active & ~hand & ~headneck}
        n_active = int(active.sum())
        w = torch.zeros(n)
        for name, sel in regions.items():
            k = int(sel.sum())
            if k == 0:
                raise ValueError(f"--rotmat_region_norm: region '{name}' has no active bone")
            w[sel] = n_active / (len(regions) * k)
        self.region_weights = w.reshape(1, n, 1, 1)
        print("[loss] rotmat_region_norm: " + ", ".join(
            f"{name} {int(sel.sum())} bones x{float(w[sel][0]):.2f}" for name, sel in regions.items())
            + f" ({n_active} active bones; sum of weights {float(w.sum()):.1f} = n_active, so the term equals "
              f"n_active x the mean over the 3 regions of the per-bone mean error; --boost_* weights no longer "
              f"apply to the rotmat term)", flush=True)

    def configure_hand_label_mask(self, bone_labels):
        """--mask_3dpw_hands: the bones whose 3DPW-origin rotations are not labels.

        scripts/data/smpl_to_anny.py fits only the root and the bones carrying a mapped SMPL joint and zeroes
        every other bone's gradient, so all phalanges and metacarpals stay at exact identity (finger2-1.L/R
        carry the SMPL hand joint at their own head and get no joint gradient either); the label builder
        writes those identities into anny_pose and prepare_gt cannot tell them from AnnyOne labels. The
        per-person flag has_hand_labels (1 AnnyOne, 0 3DPW) switches these bones off in the rotmat and
        global_rot terms for the people whose labels have no fingers.
        """
        sel = [('finger' in s.lower()) or ('thumb' in s.lower()) or ('metacarpal' in s.lower()) for s in bone_labels]
        m = torch.tensor(sel, dtype=torch.float32)
        if int(m.sum()) == 0:
            raise ValueError("--mask_3dpw_hands: no finger/thumb/metacarpal bone found in the bone names")
        self.hand_label_mask = m.reshape(1, -1, 1, 1)
        print(f"[loss] mask_3dpw_hands: {int(m.sum())} finger/metacarpal bones are not supervised (rotmat, "
              f"global_rot) for people with has_hand_labels 0; people without the flag are supervised as before",
              flush=True)

    def _hand_label_keep(self, y, like):
        """[N, 163, 1, 1] factor: 0 on the hand_label_mask bones of people whose labels carry no fingers."""
        flag = y['has_hand_labels'].reshape(-1, 1, 1, 1).to(like.device).to(like.dtype)
        return 1.0 - (1.0 - flag) * self.hand_label_mask.to(like.device).to(like.dtype)

    def forward(self, y_hat, y, epoch=None, img_size=None):
        
        # Detection
        bce = _neg_loss(y_hat['scores'], (y['scores'] >= 1).to(int).unsqueeze(-1))
        reg_offset = (y_hat['offset'] - y['offset']).abs().sum(-1).mean(0)

        # SMPL-X params
        # Per-joint weighting (mean-1 normalized; None = original equal weights).
        res_rotmat = (y_hat['rotmat'] - y['rotmat']).abs()
        hand_keep = None
        if self.hand_label_mask is not None and 'has_hand_labels' in y:
            hand_keep = self._hand_label_keep(y, res_rotmat)
            res_rotmat = res_rotmat * hand_keep
        if self.region_weights is not None:
            w = self.region_weights.to(res_rotmat.device).to(res_rotmat.dtype)
            reg_rotmat = (res_rotmat * w).sum([1,2,3]).mean(0)
        elif self.joint_weights is not None:
            w = self.joint_weights.to(y_hat['rotmat'].device).to(y_hat['rotmat'].dtype)
            reg_rotmat = (res_rotmat * w).sum([1,2,3]).mean(0)
        else:
            reg_rotmat = res_rotmat.sum([1,2,3]).mean(0) # root/body/hands/jaw
        shape_dim = min([y_hat['shape'].shape[1], y['shape'].shape[1]])
        shape_err = (y_hat['shape'][:,:shape_dim] - y['shape'][:,:shape_dim]).abs()
        if self.shape_weights is not None:
            shape_err = shape_err * self.shape_weights[:, :shape_dim].to(shape_err.device).to(shape_err.dtype)
        reg_shape = shape_err.sum(-1).mean(0)
        reg_dist = (y_hat['dist_postprocessed'].squeeze() - y['dist_postprocessed']).abs().mean(0)
        reg_transl = (y_hat['transl'] - y['transl']).abs().sum(-1).mean(0)
        # Global orientation of the selected bones (--alpha_global_rot): the rotmat term's per-bone L1,
        # applied to the post-FK camera-frame rotations and averaged over bones and people. Not computed
        # when off, so the default total is numerically the previous one; only Anny batches carry the
        # target (prepare_gt builds it from the Anny FK), so SMPL-X batches of a mixed run skip it.
        reg_grot = None
        if self.global_rot_idx is not None and 'global_rotmat' in y:
            sel = self.global_rot_idx.to(y_hat['global_rotmat'].device)
            err_grot = (y_hat['global_rotmat'][:, sel] - y['global_rotmat'][:, sel]).abs().sum([2, 3])   # [N, n_sel]
            if hand_keep is not None:
                # mean over the (person, bone) entries that are labels; unselected hand bones stay untouched
                keep = hand_keep[:, sel, 0, 0]
                reg_grot = (err_grot * keep).sum() / keep.sum().clamp(min=1.0)
            else:
                reg_grot = err_grot.mean([0, 1])
            reg_grot = torch.nan_to_num(reg_grot, nan=0.0, posinf=0.0, neginf=0.0)
        # TODO expression???

        # 3D points
        # pelvis, pelvis_hat = y['j3d'][:,[0]], y_hat['j3d'][:,[0]]
        pelvis, pelvis_hat = y['transl_pelvis'].reshape(-1,1,3), y_hat['transl_pelvis'].reshape(-1,1,3)
        j3d, j3d_hat = y['j3d'] - pelvis, y_hat['j3d'] - pelvis_hat
        v3d, v3d_hat = y['v3d'] - pelvis, y_hat['v3d'] - pelvis_hat
        if self.joint_weights is not None and j3d.shape[1] == self.joint_weights.shape[1]:
            wj = self.joint_weights.reshape(1, -1, 1).to(j3d.device).to(j3d.dtype)
            # weighted mean over joints; weights have mean 1, so the scale of
            # reg_j3d is unchanged relative to the unweighted version.
            reg_j3d = ((j3d - j3d_hat).abs().sum(-1) * wj[..., 0]).mean(-1).mean(0)
        else:
            reg_j3d = (j3d - j3d_hat).abs().sum(-1).mean(-1).mean(0)
        reg_v3d = (v3d - v3d_hat).abs().sum(-1).mean(-1).mean(0)

        # 2D reprojection
        idx_v2d = torch.where(((y['v2d'] > 0).int() * (y['v2d'] < img_size).int()).sum(-1) == 2)
        reg_v2d = (y_hat['v2d'][idx_v2d[0],idx_v2d[1]] - y['v2d'][idx_v2d[0],idx_v2d[1]]).abs().sum(-1).mean(0)
        idx_j2d = torch.where(((y['j2d'] > 0).int() * (y['j2d'] < img_size).int()).sum(-1) == 2)
        reg_j2d = (y_hat['j2d'][idx_j2d[0],idx_j2d[1]] - y['j2d'][idx_j2d[0],idx_j2d[1]]).abs().sum(-1).mean(0)

        # handle nan/inf
        bce = torch.nan_to_num(bce, nan=0.0, posinf=0.0, neginf=0.0)
        reg_offset = torch.nan_to_num(reg_offset, nan=0.0, posinf=0.0, neginf=0.0)
        reg_rotmat = torch.nan_to_num(reg_rotmat, nan=0.0, posinf=0.0, neginf=0.0)
        reg_shape = torch.nan_to_num(reg_shape, nan=0.0, posinf=0.0, neginf=0.0)
        reg_dist = torch.nan_to_num(reg_dist, nan=0.0, posinf=0.0, neginf=0.0)
        reg_transl = torch.nan_to_num(reg_transl, nan=0.0, posinf=0.0, neginf=0.0)
        reg_j3d = torch.nan_to_num(reg_j3d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_v3d = torch.nan_to_num(reg_v3d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_j2d = torch.nan_to_num(reg_j2d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_v2d = torch.nan_to_num(reg_v2d, nan=0.0, posinf=0.0, neginf=0.0)

        # total
        total_loss = self.parser_args.alpha_bce * bce +\
                     self.parser_args.alpha_offset * reg_offset +\
                     self.parser_args.alpha_rotmat * reg_rotmat +\
                     self.parser_args.alpha_shape * reg_shape +\
                     self.parser_args.alpha_dist * reg_dist +\
                     self.parser_args.alpha_transl * reg_transl +\
                     self.parser_args.alpha_j3d * reg_j3d +\
                     self.parser_args.alpha_v3d * reg_v3d
        if epoch >= self.parser_args.start_2d_epoch:
            total_loss += self.parser_args.alpha_j2d * reg_j2d +\
                          self.parser_args.alpha_v2d * reg_v2d
        if reg_grot is not None:
            total_loss = total_loss + self.parser_args.alpha_global_rot * reg_grot

        # dict for tensorboard
        dict_loss = {
           'total': total_loss,
            'bce': bce,
            'offset': reg_offset,
            'rotmat': reg_rotmat,
            'shape': reg_shape,
            'dist': reg_dist,
            'transl': reg_transl,
            'j3d': reg_j3d,
            'v3d': reg_v3d,
            'j2d': reg_j2d,
            'v2d': reg_v2d,
        }
        if reg_grot is not None:
            dict_loss['global_rot'] = reg_grot

        return total_loss, dict_loss
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Detection
        parser.add_argument('--alpha_bce', type=float, default=10.0)
        parser.add_argument('--alpha_offset', type=float, default=1.0)

        # SMPL-X params
        parser.add_argument('--alpha_rotmat', type=float, default=0.1)
        parser.add_argument('--alpha_shape', type=float, default=1.0)
        parser.add_argument('--alpha_dist', type=float, default=1.0)
        parser.add_argument('--alpha_transl', type=float, default=1.0)

        # 3D
        parser.add_argument('--alpha_j3d', type=float, default=100.0)
        parser.add_argument('--alpha_v3d', type=float, default=100.0)

        # 2D
        parser.add_argument('--alpha_j2d', type=float, default=1.0)
        parser.add_argument('--alpha_v2d', type=float, default=1.0)
        parser.add_argument('--start_2d_epoch', type=int, default=10)

        # Per-joint loss redistribution (see configure_joint_weights).
        # 1.0 = off (original equal weighting). Suggested experiment: 4.0.
        parser.add_argument('--boost_neck_weight', type=float, default=1.0,
                            help='loss weight multiplier for neck-chain + head bones '
                                 '(rotmat and j3d terms); renormalized to mean 1')
        parser.add_argument('--boost_hand_weight', type=float, default=1.0,
                            help='loss weight multiplier for finger/thumb/wrist bones '
                                 '(rotmat and j3d terms); renormalized to mean 1')
        # Global (post-FK) orientation supervision of a few named bones (see configure_global_rot).
        parser.add_argument('--alpha_global_rot', type=float, default=0.0,
                            help='weight of the L1 between predicted and GT GLOBAL (camera-frame, after forward '
                                 'kinematics) 3x3 orientations of --global_rot_bones, averaged over those bones and '
                                 'people. 0 = off: the term is not computed and the total loss is unchanged. The '
                                 'per-bone rotmat term cannot see where the head points, since the 4-bone neck '
                                 'chain can realise one head pose many ways (docs/diagnostics_findings.md, Neck).')
        parser.add_argument('--global_rot_bones', type=str, default='head,neck03,wrist.L,wrist.R',
                            help='comma-separated Anny bone names supervised by --alpha_global_rot. Unknown names, '
                                 'and bones masked (or below a masked bone) by --mask_helper_joints_train, abort '
                                 'at startup.')
        # Region-normalised rotmat term (see configure_region_norm).
        parser.add_argument('--rotmat_region_norm', type=int, default=0, choices=[0, 1],
                            help='1: the rotmat L1 is averaged within the regions body / hands (model.hand_bone_mask: '
                                 'finger and thumb bones) / head-neck (neck01-03, head) and then across the three '
                                 'regions, times the number of active bones so alpha_rotmat keeps its scale. '
                                 'Replaces the --boost_* weights in the rotmat term (j3d keeps them). 0 = the sum over '
                                 'bones, unchanged.')
        # Finger labels of 3DPW-origin people are converter placeholders (see configure_hand_label_mask).
        parser.add_argument('--mask_3dpw_hands', type=int, default=0, choices=[0, 1],
                            help='1: every human is tagged has_hand_labels (1 AnnyOne, 0 3DPW-Anny) and the finger / '
                                 'metacarpal bones of people tagged 0 are left out of the rotmat and global_rot terms '
                                 '(their 3DPW rotations are the identity the SMPL->Anny converter never fits). 0 = '
                                 'unchanged: 3DPW identity fingers are supervised as ground truth in mixed runs.')
        # Per-phenotype weighting of the L1 shape term. Not renormalised: '1,3,1,1,2,1,1,1,1,1,1'
        # triples the age error and doubles the height error (sum 14 vs 11).
        parser.add_argument('--shape_weights', type=str, default='',
                            help='11 comma-separated multipliers on the per-phenotype L1 shape error, in '
                                 'phenotype order (gender, age, muscle, weight, height, proportions, cupsize, '
                                 'firmness, african, asian, caucasian), applied before the sum over phenotypes. '
                                 "Empty or all ones = unchanged loss. Example: '1,3,1,1,2,1,1,1,1,1,1'.")

        
    
        return parser