"""
CHECK 4 -- do the prediction-head parameters receive gradients, and do the
decoded meshes respond to the head outputs?

Part A: gradients, train mode, ONE replayed training step per image
-------------------------------------------------------------------
Each selected AnnyOne image is loaded exactly as training loads it
(datasets/annyone.py + bedlam.collate_fn + bulletproof_collate's 'idx'), the
target dict is built by train.py's own Trainer.prepare_gt (if the Trainer cannot
be constructed here, by a re-implementation of ONLY its Anny branch; the output
says which path was used), and the training call of Trainer.train_n_iters is
replayed up to loss.backward():

    pred = model(x, is_training=True, idx=gt['idx'], K=gt['K'])
    pred['transl_pelvis'] = pred['transl_pelvis'].reshape(-1, 3)
    pred['transl'] = pred['transl'].reshape(-1, 3)
    loss, dict_loss = Loss(args)(pred, gt, epoch=0, img_size=img_size)
    loss.backward()

args = the checkpoint's saved training args with save_dir/name -> --out,
use_anny_shape=1, mask_helper_joints_train=CLI, amp=0 (fp32); missing alpha_* /
start_2d_epoch are filled from Loss.add_specific_args defaults. No optimizer
step is taken and nothing is written to the checkpoint.

Reported over the images:
  * per module prefix: #tensors / #elements, #requires_grad, #grad None, grad L2
    norm, fraction of exactly-zero grad entries, non-finite entries; plus the
    global norm clip_grad_norm_(max_norm=1.0) would see in train.py;
  * decpose: per-bone grad norm ([163, 6, dim] rows + bias); the bones with
    EXACTLY zero gradient compared by index with the 88 bones useful_rotmat
    masks (with --mask_helper_joints_train 1 those are forced to identity before
    the body model, so zero gradient there is EXPECTED);
  * decshape: per-phenotype grad norm (bm.phenotype_labels order), logit and
    sigmoid slope (saturation), flags ~0 gradients;
  * every loss term (raw, alpha, weighted, in the total or not);
  * placement: what reg_transl compares (pred['transl'] = model.person_center
    joint vs gt['transl'] = joint 0 for Anny) and where the head/root land.

Part B: response, eval mode, torch.no_grad()
--------------------------------------------
Forward hooks add a fixed offset to ONE output of decshape (age, height,
proportions; +-2 logits) or to the 6D output of ONE bone of decpose (neck02,
head, first 'finger' bone; +0.5 on 6D components 1 and 3), and the change of the
decoded person with the highest score is measured against the unperturbed run:
phenotype (post-sigmoid, measured vs expected from the logit), skeletal height,
neck length, mean vertex displacement; bone rotation change, head-joint and
local vs distant vertex displacement (pelvis-relative frame: the model anchors
the mesh at its person_center joint, the head, so camera-frame head motion is 0
by construction). mask_helper_joints / relax_hands / rest_pose are OFF so every
bone can respond; the same pose perturbation is repeated with the demo toggles
ON (mask + relax_hands) to show that masked bones cannot respond by construction.

USAGE (repo root, multihmr env, GPU node, ~5 min):
    python scripts/diagnostics/diag_gradients.py \
        --ckpt logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt \
        --out results/diagnostics/gradients_v5
"""
import argparse
import json
import math
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402  (sets PYOPENGL_PLATFORM, sys.path)

import numpy as np   # noqa: E402
import torch         # noqa: E402
import roma          # noqa: E402

from datasets.annyone import AnnyOne    # noqa: E402
from datasets.bedlam import collate_fn  # noqa: E402
from loss import Loss                   # noqa: E402  (train.py uses the same class: `from loss import Loss`)

MODULE_GROUPS = [
    # (label, parameter-name prefixes); anything else lands in 'other'
    ('backbone.', ('backbone.',)),
    ('feat_proj.', ('feat_proj.',)),
    ('mlp_classif.', ('mlp_classif.',)),
    ('mlp_offset.', ('mlp_offset.',)),
    ('x_attention_head.transformer.', ('x_attention_head.transformer.',)),
    ('x_attention_head.decpose', ('x_attention_head.decpose.',)),
    ('x_attention_head.decshape', ('x_attention_head.decshape.',)),
    ('x_attention_head.deccam', ('x_attention_head.deccam.',)),
    ('x_attention_head.cross_*', ('x_attention_head.cross_',)),
    ('body_model.', ('body_model.',)),
    ('camera.', ('camera.',)),
]
GROUP_ORDER = [g for g, _ in MODULE_GROUPS] + ['other']
HEAD_GROUPS = ('x_attention_head.decpose', 'x_attention_head.decshape', 'x_attention_head.deccam')
LOSS_ALPHA = {'bce': 'alpha_bce', 'offset': 'alpha_offset', 'rotmat': 'alpha_rotmat', 'shape': 'alpha_shape',
              'dist': 'alpha_dist', 'transl': 'alpha_transl', 'j3d': 'alpha_j3d', 'v3d': 'alpha_v3d',
              'j2d': 'alpha_j2d', 'v2d': 'alpha_v2d'}
SHAPE_TESTS = ('age', 'height', 'proportions')
RESP_MM = 1.0          # a joint / vertex counts as "moved" above this displacement (mm)
NONZERO_MM = 0.1       # verdict: a mesh "responds" if it moves by more than this (mm); a broken path gives ~0
NONZERO_DEG = 0.1      # verdict: a bone "responds" if its decoded rotation changes by more than this (deg)
PHENO_ZERO_REL = 1e-3  # a phenotype gradient is "~0" if <= this x the largest phenotype gradient
COND_WARN = 0.05       # |sin(angle)| between the two 6D columns below this = Gram-Schmidt near-singular


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sci(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '-'
    return f"{x:.{nd}e}"


def print_table(title, headers, rows):
    print(f"\n{title}")
    rows = [[str(c) for c in r] for r in rows]
    w = [max([len(str(h))] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print('  ' + ' | '.join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print('  ' + '-+-'.join('-' * x for x in w))
    for r in rows:
        print('  ' + ' | '.join(c.ljust(w[i]) for i, c in enumerate(r)))


def mean_masked(v, m):
    return float(v[m].mean()) if bool(m.any()) else None


def mean_of(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def group_of(name):
    for label, prefixes in MODULE_GROUPS:
        if name.startswith(prefixes):
            return label
    return 'other'


def names_of(idx_list, names, cap=None):
    out = [f"{i}:{names[i]}" for i in sorted(idx_list)]
    if cap is not None and len(out) > cap:
        out = out[:cap] + [f"... (+{len(out) - cap} more)"]
    return out


# --------------------------------------------------------------------------- #
# train.py-faithful setup
# --------------------------------------------------------------------------- #
def loss_arg_defaults():
    """Defaults of every argument Loss.add_specific_args registers (alpha_*, start_2d_epoch, boosts)."""
    p = Loss.add_specific_args(argparse.ArgumentParser(add_help=False))
    return vars(p.parse_args([]))


def build_train_args(ckpt_kwargs, model, out_dir, mask_flag):
    targs = argparse.Namespace(**ckpt_kwargs)
    overrides = dict(save_dir=out_dir, name='gradient_check', use_anny_shape=1,
                     mask_helper_joints_train=int(mask_flag), amp=0, img_size=int(model.img_size))
    before = {k: getattr(targs, k, '<unset>') for k in overrides}
    for k, v in overrides.items():
        setattr(targs, k, v)
    filled = {}
    for k, v in loss_arg_defaults().items():
        if not hasattr(targs, k):
            setattr(targs, k, v)
            filled[k] = v
    return targs, before, filled


def replicate_freeze(model, targs, enabled):
    """train.py main() lines ~827-856: --freeze_backbone / --unfreeze_last_n_blocks from the ckpt args."""
    info = dict(enabled=bool(enabled),
                freeze_backbone=int(getattr(targs, 'freeze_backbone', 0) or 0),
                unfreeze_last_n_blocks=int(getattr(targs, 'unfreeze_last_n_blocks', 0) or 0))
    if not (info['enabled'] and info['freeze_backbone']):
        return info
    for p in model.backbone.parameters():
        p.requires_grad = False
    k = info['unfreeze_last_n_blocks']
    if k > 0:
        blocks = model.backbone.encoder.blocks
        for blk in blocks[len(blocks) - min(k, len(blocks)):]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.backbone.encoder.norm.parameters():
            p.requires_grad = True
    return info


def set_train_mode(model, freeze_info):
    """train.py train_n_iters lines ~399-413."""
    model.train()
    if freeze_info['enabled'] and freeze_info['freeze_backbone']:
        model.backbone.eval()
        k = freeze_info['unfreeze_last_n_blocks']
        if k > 0:
            blocks = model.backbone.encoder.blocks
            for blk in blocks[len(blocks) - min(k, len(blocks)):]:
                blk.train()
            model.backbone.encoder.norm.train()


def prepare_gt_anny_fallback(y, model, targs, device):
    """
    Re-implementation of ONLY the Anny branch of train.py Trainer.prepare_gt
    (lines ~77-311), used when the Trainer cannot be constructed here (e.g. the
    SMPL-X files under models/ are missing). Same order of operations and the
    same constants. The GT mesh uses model.body_model, which model.py builds with
    the same anny.create_fullbody_model(remove_unattached_vertices=False,
    all_phenotypes=True) + set_skinning_method('lbs') as the Trainer's own
    anny_body_model.
    """
    from utils import perspective_projection, log_depth, focal_length_normalization
    if 'anny_pose' not in y:
        return None
    bm = model.body_model
    target = {}
    bs, nhmax = y['valid_humans'].shape
    valid_h = y['valid_humans']
    idx_h = torch.where(valid_h)
    nhv = int(valid_h.sum())
    K = y['K'][idx_h[0]]

    pose_163 = y['anny_pose'][idx_h[0], idx_h[1]].to(device)[:, :163]     # [nhv,163,4,4]
    anny_rotmats = pose_163[:, :, :3, :3]
    labels = list(bm.phenotype_labels)
    shape_vec = torch.zeros((nhv, len(labels)), device=device)
    if bool(getattr(targs, 'use_anny_shape', 0)) and ('anny_shape' in y):
        raw_shape = y['anny_shape'][idx_h[0], idx_h[1]].to(device).float().reshape(nhv, -1)
        ns = min(raw_shape.shape[-1], len(labels))
        shape_vec[:, :ns] = torch.clamp(raw_shape[:, :ns], 0.0, 1.0)
    _shape = {k: shape_vec[:, l] for l, k in enumerate(labels)}
    with torch.no_grad():
        anny_out = bm(pose_parameters=pose_163, phenotype_kwargs=_shape)
    verts = anny_out['vertices']
    jts = anny_out['bone_poses'][:, :, :3, -1]

    gt_R = dc.orthonormalize(anny_rotmats.reshape(-1, 3, 3)).reshape(nhv, 163, 3, 3)   # same SVD projection
    if getattr(targs, 'mask_helper_joints_train', 0) and hasattr(model, 'useful_rotmat'):
        m = model.useful_rotmat.reshape(1, -1, 1, 1).to(gt_R.dtype).to(gt_R.device)
        eye = torch.eye(3, device=gt_R.device, dtype=gt_R.dtype).reshape(1, 1, 3, 3)
        gt_R = m * gt_R + (1.0 - m) * eye
    target['rotmat'] = gt_R
    target['rotvec'] = roma.rotmat_to_rotvec(gt_R.reshape(-1, 3, 3)).reshape(nhv, 163, 3)
    target['shape'] = shape_vec

    j2d = perspective_projection(jts, K)
    v2d = perspective_projection(verts, K)
    target['transl'] = jts[:, 0]                      # train.py: root_joint_idx = 0 for Anny batches
    target['transl_pelvis'] = jts[:, 0]
    target['dist'] = torch.clamp(jts[:, 0, -1], min=0.1)
    if model.nearness:
        d = log_depth(target['dist'])
        d = focal_length_normalization(d, K[:, 0, 0], fovn=60, img_size=model.img_size)
        target['dist_postprocessed'] = d
    target['v3d'] = verts
    target['j3d'] = jts
    target['j2d'] = j2d
    target['v2d'] = v2d

    ps = model.patch_size
    n_patch = targs.img_size // ps
    pk_loc = perspective_projection(target['transl'].unsqueeze(1), K).squeeze(1)
    pk_coarse_loc = (pk_loc // ps).int()
    pk_idx = torch.clamp(pk_coarse_loc, 0, n_patch - 1)
    pk_offset = (pk_loc - (pk_idx + 0.5) * ps) / ps
    img_size = targs.img_size
    in_front = target['transl'][:, 2] > 0.1
    in_image = (pk_loc[:, 0] >= 0) & (pk_loc[:, 0] < img_size) & \
               (pk_loc[:, 1] >= 0) & (pk_loc[:, 1] < img_size)
    is_valid = (in_front & in_image).float()
    pk_offset = torch.clamp(pk_offset, min=-1.0, max=1.0)

    scores = torch.zeros((bs, n_patch, n_patch)).to(device)
    visible_humans = is_valid.clone()
    for k in range(nhv):
        i = int(idx_h[0][k])
        j = int(idx_h[1][k])
        if visible_humans[k] == 0:
            valid_h[i, j] = 0
            continue
        _x = pk_idx[k, 1]
        _y = pk_idx[k, 0]
        if scores[i, _x, _y] == 1:
            valid_h[i, j] = 0
            visible_humans[k] = 0
        else:
            scores[i, _x, _y] = 1
    target['loc'] = pk_loc
    target['offset'] = pk_offset

    idx_vis = torch.where(visible_humans)[0]
    if idx_vis.numel() == 0:
        return None
    _target = {}
    _target['idx'] = tuple([
        idx_h[0].to(device)[idx_vis],
        pk_idx[:, 1].to(device)[idx_vis],
        pk_idx[:, 0].to(device)[idx_vis],
        torch.zeros_like(idx_h[0].to(device)[idx_vis]),
    ])
    _target['scores'] = scores
    _target['K'] = y['K']
    for k, v in target.items():
        _target[k] = v[idx_vis]
    return _target


def build_gt_builder(model, loss_fn, targs, device, mode):
    """Returns (prepare_gt callable, path description, error string or None)."""
    err = None
    if mode in ('auto', 'trainer'):
        try:
            import train as train_mod
            trainer = train_mod.Trainer(model=model, loss=loss_fn, optimizer=None, device=device,
                                        args=targs, scheduler=None)
            # prepare_gt reads the MODULE-GLOBAL `args.img_size` (train.py:239), which only exists when
            # train.py runs as __main__; provide it.
            train_mod.args = targs
            print(f"[gt] using train.Trainer.prepare_gt (log_dir {targs.log_dir})")
            return trainer.prepare_gt, 'train.Trainer.prepare_gt', None
        except Exception as e:  # noqa
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            if mode == 'trainer':
                raise
            print(f"[gt] Trainer construction failed ({err}); FALLING BACK to the Anny-branch re-implementation")
    else:
        print("[gt] --gt_path fallback: using the Anny-branch re-implementation")
    return (lambda y: prepare_gt_anny_fallback(y, model, targs, device)), \
        'fallback: re-implementation of the Anny branch of Trainer.prepare_gt', err


# --------------------------------------------------------------------------- #
# gradient statistics
# --------------------------------------------------------------------------- #
def grad_stats_by_group(model):
    stats = {}
    for name, p in model.named_parameters():
        g = group_of(name)
        s = stats.setdefault(g, dict(n_tensors=0, n_elems=0, n_requires_grad=0, n_grad_none=0, sq=0.0,
                                     n_zero=0, n_grad_elems=0, n_nonfinite=0, grad_none_examples=[],
                                     names=[]))
        s['n_tensors'] += 1
        s['n_elems'] += p.numel()
        if g == 'other' and len(s['names']) < 8:
            s['names'].append(name)
        if not p.requires_grad:
            continue
        s['n_requires_grad'] += 1
        if p.grad is None:
            s['n_grad_none'] += 1
            if len(s['grad_none_examples']) < 4:
                s['grad_none_examples'].append(name)
            continue
        gr = p.grad.detach().float()
        fin = torch.isfinite(gr)
        s['n_nonfinite'] += int((~fin).sum())
        s['sq'] += float(torch.where(fin, gr, torch.zeros_like(gr)).pow(2).sum())
        s['n_zero'] += int((gr == 0).sum())
        s['n_grad_elems'] += gr.numel()
    for s in stats.values():
        s['grad_norm'] = math.sqrt(s['sq']) if s['n_grad_elems'] else None
        s['zero_frac'] = s['n_zero'] / s['n_grad_elems'] if s['n_grad_elems'] else None
    return stats


def aggregate_groups(per_image_stats):
    agg = {}
    for g in GROUP_ORDER:
        rows = [st[g] for st in per_image_stats if g in st]
        if not rows:
            continue
        norms = [r['grad_norm'] for r in rows if r['grad_norm'] is not None]
        zf = [r['zero_frac'] for r in rows if r['zero_frac'] is not None]
        agg[g] = dict(n_tensors=rows[0]['n_tensors'], n_elems=rows[0]['n_elems'],
                      n_requires_grad=rows[0]['n_requires_grad'],
                      n_grad_none_max=max(r['n_grad_none'] for r in rows),
                      n_grad_none_min=min(r['n_grad_none'] for r in rows),
                      grad_norm_mean=float(np.mean(norms)) if norms else None,
                      grad_norm_min=float(np.min(norms)) if norms else None,
                      grad_norm_max=float(np.max(norms)) if norms else None,
                      zero_frac_mean=float(np.mean(zf)) if zf else None,
                      n_nonfinite_total=int(sum(r['n_nonfinite'] for r in rows)),
                      grad_none_examples=rows[0]['grad_none_examples'],
                      tensor_names=rows[0]['names'] if g == 'other' else [])
    return agg


def linear_row_sq(lin, n_rows, row_dim):
    """Squared grad norm per output group of a Linear: weight [n_rows*row_dim, D] (+ bias) -> [n_rows], or None."""
    if lin.weight.grad is None:
        return None
    sq = lin.weight.grad.detach().float().pow(2).reshape(n_rows, row_dim, -1).sum((1, 2))
    if lin.bias is not None and lin.bias.grad is not None:
        sq = sq + lin.bias.grad.detach().float().pow(2).reshape(n_rows, row_dim).sum(1)
    return sq.cpu()


def sixd_conditioning(raw6d, init6d):
    """
    |sin| of the angle between the two 3-vectors utils.rot6d_to_rotmat feeds to
    roma.special_gramschmidt (x.reshape(-1, 2, 3).permute -> columns x[0:3] and
    x[3:6]). ~0 = Gram-Schmidt near-singular: the decoded rotation is unstable
    and its gradient explodes. raw6d = decpose output [N, 978] BEFORE the
    init_body_pose offset that HPH.forward adds. Returns [N, 163].
    """
    x = (raw6d.float() + init6d.float()).reshape(raw6d.shape[0], -1, 2, 3)
    a1, a2 = x[..., 0, :], x[..., 1, :]
    return torch.cross(a1, a2, dim=-1).norm(dim=-1) / (a1.norm(dim=-1) * a2.norm(dim=-1)).clamp_min(1e-12)


def gradient_pass(model, loss_fn, x, gt, targs, args, freeze_info, names, head_idx):
    set_train_mode(model, freeze_info)
    model.zero_grad(set_to_none=True)
    cap = {}

    def cap_hook(key):
        def hook(module, inputs, output):
            cap[key] = output.detach()
        return hook

    hooks = [model.x_attention_head.decshape.register_forward_hook(cap_hook('shape_raw')),
             model.x_attention_head.decpose.register_forward_hook(cap_hook('pose_raw'))]
    try:
        # --- the exact call of train.py Trainer.train_n_iters (amp=0) ---
        pred = model(x, is_training=True, idx=gt['idx'], K=gt['K'])
        pred['transl_pelvis'] = pred['transl_pelvis'].reshape(-1, 3)
        pred['transl'] = pred['transl'].reshape(-1, 3)
        loss, dict_loss = loss_fn(pred, gt, epoch=args.loss_epoch, img_size=targs.img_size)
    finally:
        for h in hooks:
            h.remove()
    loss_finite = bool(torch.isfinite(loss).all())
    loss.backward()

    rec = dict(n_persons=int(gt['idx'][0].numel()), loss_finite=loss_finite,
               loss={k: float(v.detach()) for k, v in dict_loss.items()})
    rec['groups'] = grad_stats_by_group(model)
    total_sq = sum(s['sq'] for s in rec['groups'].values())
    rec['total_grad_norm'] = math.sqrt(total_sq)
    rec['clip_coef'] = min(1.0, 1.0 / (rec['total_grad_norm'] + 1e-6))   # clip_grad_norm_(max_norm=1.0)

    xh = model.x_attention_head
    rec['decpose_sq'] = linear_row_sq(xh.decpose, 163, 6)      # [163]
    rec['decshape_sq'] = linear_row_sq(xh.decshape, xh.decshape.out_features, 1)
    rec['deccam_sq'] = linear_row_sq(xh.deccam, xh.deccam.out_features, 1)

    with torch.no_grad():
        logit = cap['shape_raw'].float() + xh.init_betas.float()                    # [N, 11]
        sig = torch.sigmoid(logit)
        rec['shape_logit_mean'] = logit.mean(0).cpu()
        rec['shape_sigmoid_slope_mean'] = (sig * (1 - sig)).mean(0).cpu()
        rec['shape_pred_minus_gt_mean'] = (pred['shape'].detach().float() - gt['shape'].float()).mean(0).cpu()
        rec['sixd_cond'] = sixd_conditioning(cap['pose_raw'], xh.init_body_pose).mean(0).cpu()   # [163]
        if head_idx is not None:
            pj, gj = pred['j3d'].detach().float(), gt['j3d'].float()
            rec['placement'] = dict(
                transl_term_m=float((pred['transl'].detach() - gt['transl']).norm(dim=-1).mean()),
                pred_head_vs_gt_head_m=float((pj[:, head_idx] - gj[:, head_idx]).norm(dim=-1).mean()),
                pred_root_vs_gt_root_m=float((pj[:, 0] - gj[:, 0]).norm(dim=-1).mean()),
                gt_head_to_root_m=float((gj[:, head_idx] - gj[:, 0]).norm(dim=-1).mean()))
    model.zero_grad(set_to_none=True)
    del pred, loss, dict_loss
    return rec


# --------------------------------------------------------------------------- #
# response test
# --------------------------------------------------------------------------- #
def infer(model, x, K, args):
    with torch.no_grad():
        return model(x, is_training=False, K=K, det_thresh=args.det_thresh, nms_kernel_size=args.nms_kernel_size)


def add_to_outputs_hook(cols, delta):
    def hook(module, inputs, output):
        d = torch.zeros(output.shape[-1], device=output.device, dtype=output.dtype)
        d[list(cols)] = float(delta)
        return output + d          # returning a value replaces the module output
    return hook


def run_with_hook(module, cols, delta, model, x, K, args):
    h = module.register_forward_hook(add_to_outputs_hook(cols, delta))
    try:
        return infer(model, x, K, args)
    finally:
        h.remove()


def person_arrays(p):
    return dict(score=float(p['scores']), shape=p['shape'].float(), j3d=p['j3d'].float(),
                v3d=p['v3d'].float(),
                R=roma.rotvec_to_rotmat(p['rotvec'].float().reshape(-1, 3)).reshape(-1, 3, 3))


def shape_response(model, x, K, args, best, base, phen, names, chain):
    rows = []
    dec = model.x_attention_head.decshape
    h0 = float(dc.skeletal_height(base['j3d'][None], names)[0])
    n0 = float(dc.chain_length(base['j3d'][None], chain)[0])
    for pname in SHAPE_TESTS:
        if pname not in phen:
            print(f"  [warn] phenotype '{pname}' not in {phen}; skipped")
            continue
        j = phen.index(pname)
        p0 = float(base['shape'][j])
        pc = min(max(p0, 1e-6), 1 - 1e-6)
        logit0 = math.log(pc / (1 - pc))
        for sgn in (1.0, -1.0):
            delta = sgn * args.shape_delta
            preds = run_with_hook(dec, [j], delta, model, x, K, args)
            if len(preds) <= best:
                rows.append(dict(phenotype=pname, delta_logit=delta, error='detection set changed'))
                continue
            q = person_arrays(preds[best])
            dsh = q['shape'] - base['shape']
            others = torch.cat([dsh[:j], dsh[j + 1:]]).abs().max() if dsh.numel() > 1 else torch.zeros(())
            h1 = float(dc.skeletal_height(q['j3d'][None], names)[0])
            n1 = float(dc.chain_length(q['j3d'][None], chain)[0])
            v_cam = float((q['v3d'] - base['v3d']).norm(dim=-1).mean()) * 1000
            v_pel = float(((q['v3d'] - q['j3d'][0:1]) - (base['v3d'] - base['j3d'][0:1])).norm(dim=-1).mean()) * 1000
            rows.append(dict(phenotype=pname, delta_logit=delta, pheno_before=p0, pheno_after=float(q['shape'][j]),
                             d_pheno=float(dsh[j]), d_pheno_expected=1.0 / (1.0 + math.exp(-(logit0 + delta))) - p0,
                             d_other_pheno_max=float(others), score_change=abs(q['score'] - base['score']),
                             skel_height_before_m=h0, d_skel_height_mm=(h1 - h0) * 1000,
                             neck_before_m=n0, d_neck_mm=(n1 - n0) * 1000,
                             v_disp_cam_mm=v_cam, v_disp_pelvis_rel_mm=v_pel))
    return rows


def pose_response(model, x, K, args, best, base, demo_base, names, bones, useful, hand_mask, head_idx):
    rows = []
    dec = model.x_attention_head.decpose
    J0 = base['j3d'] - base['j3d'][0:1]                      # pelvis(root)-relative, as the 3D losses
    V0 = base['v3d'] - base['j3d'][0:1]
    nearest = torch.cdist(V0, J0).argmin(dim=-1)              # [V] nearest joint of each vertex (unperturbed)
    for b in bones:
        cols = [6 * b + c for c in args.pose_components]
        preds = run_with_hook(dec, cols, args.pose_delta, model, x, K, args)
        if len(preds) <= best:
            rows.append(dict(bone=names[b], idx=b, error='detection set changed'))
            continue
        q = person_arrays(preds[best])
        rot = dc.geodesic_deg(base['R'], q['R'])              # [163]
        J1 = q['j3d'] - q['j3d'][0:1]
        V1 = q['v3d'] - q['j3d'][0:1]
        dj = (J1 - J0).norm(dim=-1) * 1000                    # mm
        dv = (V1 - V0).norm(dim=-1) * 1000
        moved = [int(i) for i in torch.nonzero(dj > RESP_MM).flatten()]
        region = sorted(set(moved) | {b})
        near_b = nearest == b
        near_r = torch.isin(nearest, torch.tensor(region, device=nearest.device, dtype=nearest.dtype))
        top = torch.argsort(dj, descending=True)[:5]
        row = dict(bone=names[b], idx=b,
                   masked_by_useful_rotmat=not bool(useful[b]), in_hand_bone_mask=bool(hand_mask[b]),
                   bone_rot_change_deg=float(rot[b]),
                   other_bones_rot_change_max_deg=float(torch.cat([rot[:b], rot[b + 1:]]).max()),
                   head_joint_disp_pelvis_rel_mm=float(dj[head_idx]) if head_idx is not None else None,
                   head_joint_disp_cam_mm=(float((q['j3d'][head_idx] - base['j3d'][head_idx]).norm()) * 1000
                                           if head_idx is not None else None),
                   n_joints_moved=len(moved),
                   top_moved_joints_mm={names[int(i)]: float(dj[int(i)]) for i in top if float(dj[int(i)]) > RESP_MM},
                   v_disp_max_mm=float(dv.max()), v_moved_frac=float((dv > RESP_MM).float().mean()),
                   n_v_nearest_bone=int(near_b.sum()),
                   v_disp_nearest_bone_mm=mean_masked(dv, near_b), v_disp_not_nearest_bone_mm=mean_masked(dv, ~near_b),
                   n_v_region=int(near_r.sum()),
                   v_disp_region_mm=mean_masked(dv, near_r), v_disp_outside_region_mm=mean_masked(dv, ~near_r))
        if demo_base is not None:
            dc.set_inference_toggles(model, mask_helper_joints=True, relax_hands=True)
            try:
                pd = run_with_hook(dec, cols, args.pose_delta, model, x, K, args)
            finally:
                dc.set_inference_toggles(model, mask_helper_joints=False, relax_hands=False)
            if len(pd) > best:
                qd = person_arrays(pd[best])
                rotd = dc.geodesic_deg(demo_base['R'], qd['R'])
                dvd = ((qd['v3d'] - qd['j3d'][0:1]) - (demo_base['v3d'] - demo_base['j3d'][0:1])).norm(dim=-1) * 1000
                row['demo_toggles'] = dict(bone_rot_change_deg=float(rotd[b]), v_disp_max_mm=float(dvd.max()))
        rows.append(row)
    return rows


def response_pass(model, x, K, args, names, phen, chain, bones, useful, hand_mask, head_idx):
    model.eval()
    dc.set_inference_toggles(model, mask_helper_joints=False, relax_hands=False, rest_pose=False)
    cap = {}

    def hook(module, inputs, output):
        cap['pose_raw'] = output.detach()

    h = model.x_attention_head.decpose.register_forward_hook(hook)
    try:
        preds0 = infer(model, x, K, args)
    finally:
        h.remove()
    if len(preds0) == 0:
        return None
    best = int(np.argmax([float(p['scores']) for p in preds0]))
    base = person_arrays(preds0[best])
    cond = sixd_conditioning(cap['pose_raw'], model.x_attention_head.init_body_pose)[best].cpu()   # [163]
    x6 = (cap['pose_raw'][best].float() + model.x_attention_head.init_body_pose[0].float()).reshape(-1, 2, 3)
    coln = x6.norm(dim=-1).cpu()                     # [163, 2] norms of the two 6D columns (delta / norm ~ angle)

    dc.set_inference_toggles(model, mask_helper_joints=True, relax_hands=True)
    try:
        preds_demo = infer(model, x, K, args)
    finally:
        dc.set_inference_toggles(model, mask_helper_joints=False, relax_hands=False)
    demo_base = person_arrays(preds_demo[best]) if len(preds_demo) > best else None

    rec = dict(n_det=len(preds0), best=best, score=base['score'],
               sixd_cond=cond, sixd_colnorm=coln,
               shape=shape_response(model, x, K, args, best, base, phen, names, chain),
               pose=pose_response(model, x, K, args, best, base, demo_base, names, bones, useful, hand_mask, head_idx))
    return rec


def find_test_bones(names):
    lo = [n.lower() for n in names]
    out = []
    for want in ('neck02', 'head'):
        if want in lo:
            out.append(lo.index(want))
        else:
            print(f"[warn] bone '{want}' not found in the rig; skipped")
    fing = [i for i, n in enumerate(lo) if 'finger' in n]
    if fing:
        out.append(fing[0])
    else:
        print("[warn] no 'finger' bone found in the rig; skipped")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='path, models/multiHMR name, or run_name:epoch')
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--indices', default='', help='comma list of dataset indices (default: --n_images spread '
                                                  'over the holdout tail, dc.holdout_indices)')
    ap.add_argument('--n_images', type=int, default=4)
    ap.add_argument('--out', default='results/diagnostics/gradients')
    ap.add_argument('--mask_helper_joints_train', type=int, default=1, choices=[0, 1])
    ap.add_argument('--loss_epoch', type=int, default=-1,
                    help='epoch passed to Loss.forward; j2d/v2d enter the total only if >= start_2d_epoch. '
                         '-1 (default) = the checkpoint\'s own epoch (parsed from <epoch>.pt, else start_2d_epoch), '
                         'so the replayed gradient step uses the loss the checkpoint was actually training with')
    ap.add_argument('--replicate_freeze', type=int, default=1, choices=[0, 1],
                    help="apply the checkpoint's --freeze_backbone / --unfreeze_last_n_blocks like train.py main()")
    ap.add_argument('--gt_path', default='auto', choices=['auto', 'trainer', 'fallback'],
                    help='auto: train.Trainer.prepare_gt, fallback re-implementation if the Trainer cannot be built')
    ap.add_argument('--shape_delta', type=float, default=2.0, help='logit offset added to one decshape output')
    ap.add_argument('--pose_delta', type=float, default=0.5, help='offset added to the 6D output of one bone')
    ap.add_argument('--pose_components', default='1,3', help='0-based 6D components that receive --pose_delta')
    ap.add_argument('--det_thresh', type=float, default=0.2)
    ap.add_argument('--nms_kernel_size', type=int, default=3)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    args.pose_components = [int(s) for s in args.pose_components.split(',') if s.strip()]
    t0 = time.time()

    # resolve every user path BEFORE moving to the repo root (train.py's SMPL-X paths are cwd-relative)
    ckpt_path = os.path.abspath(dc.resolve_ckpt(args.ckpt))
    out_dir = os.path.abspath(args.out)
    data_folder = os.path.abspath(args.data_folder)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(os.getcwd()) != dc.ROOT:
        print(f"[cwd] {os.getcwd()} -> {dc.ROOT} (train.py / utils.constants paths are relative to the repo root)")
        os.chdir(dc.ROOT)

    device = torch.device(args.device)
    model, ckpt_kwargs = dc.load_model(ckpt_path, device)
    bm = model.body_model
    names = list(bm.bone_labels)
    phen = list(bm.phenotype_labels)
    lo = [n.lower() for n in names]
    head_idx = lo.index('head') if 'head' in lo else None
    chain = dc.neck_chain(names)
    useful = model.useful_rotmat.reshape(-1).bool().cpu()
    hand_mask = model.hand_bone_mask.reshape(-1).bool().cpu()
    masked_set = set(int(i) for i in torch.nonzero(~useful).flatten())
    n_bones = len(names)
    print(f"[rig] {n_bones} bones, {n_bones - len(masked_set)} active / {len(masked_set)} masked by useful_rotmat; "
          f"{int(hand_mask.sum())} finger/thumb bones; person_center={names[model.person_center_idx]}; "
          f"phenotypes={phen}")

    # ---- train.py-faithful args / model flags / loss ------------------------
    targs, overridden, filled = build_train_args(ckpt_kwargs, model, out_dir, args.mask_helper_joints_train)
    ckpt_mask = ckpt_kwargs.get('mask_helper_joints_train', None)
    if ckpt_mask is not None and int(ckpt_mask) != int(args.mask_helper_joints_train):
        print(f"[WARN] the checkpoint was trained with mask_helper_joints_train={ckpt_mask}; replaying with "
              f"{args.mask_helper_joints_train} (CLI)")
    model.mask_helper_joints_train = bool(args.mask_helper_joints_train)        # train.py:859-861
    freeze_info = replicate_freeze(model, targs, args.replicate_freeze)
    loss_fn = Loss(targs)
    if float(getattr(targs, 'boost_neck_weight', 1.0)) != 1.0 or float(getattr(targs, 'boost_hand_weight', 1.0)) != 1.0:
        loss_fn.configure_joint_weights(names)                                   # train.py:900-902
    if args.loss_epoch < 0:
        _m = re.search(r'(\d+)\.pt$', os.path.basename(ckpt_path))
        args.loss_epoch = int(_m.group(1)) if _m else int(targs.start_2d_epoch)
        print(f"[args] loss epoch taken from the checkpoint: {args.loss_epoch} "
              f"({'parsed from the file name' if _m else 'fallback: start_2d_epoch'})")
    two_d_in_total = args.loss_epoch >= int(targs.start_2d_epoch)
    print("[args] overrides (ckpt value -> used): "
          + ', '.join(f"{k}: {overridden[k]} -> {getattr(targs, k)}" for k in overridden))
    print(f"[args] filled from Loss.add_specific_args defaults: {filled if filled else 'none'}")
    print(f"[args] freeze replicated: {freeze_info}")
    print(f"[args] loss epoch {args.loss_epoch}, start_2d_epoch {targs.start_2d_epoch}: j2d/v2d "
          f"{'IN' if two_d_in_total else 'NOT in'} the total loss (reported either way)")
    prepare_gt, gt_path, gt_err = build_gt_builder(model, loss_fn, targs, device, args.gt_path)

    # ---- images ---------------------------------------------------------------
    ds = AnnyOne(data_folder=data_folder, img_size=model.img_size)
    if args.indices:
        candidates = [int(s) for s in args.indices.split(',') if s.strip()]
        need = len(candidates)
    else:
        hold = dc.holdout_indices(len(ds))
        step = max(1, len(hold) // max(1, args.n_images))
        first = hold[::step][:args.n_images]
        candidates = first + [i for i in hold if i not in first]     # replacements if an image has no valid GT
        need = args.n_images
    bones = find_test_bones(names)
    print(f"[data] {len(ds)} samples; need {need} image(s) with valid GT; response-test bones: "
          f"{[names[b] for b in bones]}")

    grad_recs, resp_recs, used, skipped = [], [], [], []
    for idx in candidates:
        if len(used) >= need:
            break
        img_arr, annot = ds[idx]
        x, y = collate_fn([(img_arr, annot)])
        if 'idx' not in y:
            y['idx'] = torch.arange(x.shape[0])                         # bulletproof_collate
        y = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
        x = x.to(device)
        K = y['K'].clone()
        gt = prepare_gt(y)
        if gt is None:
            skipped.append(dict(idx=idx, reason='prepare_gt returned None (no visible human)'))
            print(f"  idx {idx}: no visible human after prepare_gt; skipped")
            continue
        g = gradient_pass(model, loss_fn, x, gt, targs, args, freeze_info, names, head_idx)
        g['idx'] = idx
        grad_recs.append(g)
        r = response_pass(model, x, K, args, names, phen, chain, bones, useful, hand_mask, head_idx)
        if r is None:
            print(f"  idx {idx}: no detection at det_thresh {args.det_thresh}; response test skipped")
        else:
            r['idx'] = idx
            resp_recs.append(r)
        used.append(idx)
        dec_zero = int((g['decpose_sq'] == 0).sum()) if g['decpose_sq'] is not None else None
        print(f"  idx {idx}: {g['n_persons']} person(s), loss {g['loss'].get('total', float('nan')):.4f} "
              f"(finite={g['loss_finite']}), |grad| {g['total_grad_norm']:.3e}, decpose zero-grad bones {dec_zero}, "
              f"response {'ok' if r is not None else 'skipped'}  ({time.time() - t0:.0f}s)", flush=True)
        del gt, x, y
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if not grad_recs:
        raise SystemExit("no image with valid GT; nothing to report")

    # ======================================================================= #
    # Aggregation
    # ======================================================================= #
    modules = aggregate_groups([g['groups'] for g in grad_recs])
    total_norms = [g['total_grad_norm'] for g in grad_recs]
    clip = [g['clip_coef'] for g in grad_recs]

    # decpose per bone
    dp = [g['decpose_sq'] for g in grad_recs]
    if any(d is None for d in dp):
        bone_norm = None
        zero_set = set(range(n_bones))
        per_image_zero = [None if d is None else int((d == 0).sum()) for d in dp]
    else:
        bone_sq = torch.stack(dp).sum(0)
        bone_norm = bone_sq.sqrt()
        zero_set = set(int(i) for i in torch.nonzero(bone_sq == 0).flatten())
        per_image_zero = [int((d == 0).sum()) for d in dp]
    expected_zero = masked_set if args.mask_helper_joints_train else set()
    active_but_zero = sorted(zero_set - masked_set)
    masked_but_nonzero = sorted(masked_set - zero_set)
    unexpected_zero = sorted(zero_set - expected_zero)
    unexpected_nonzero = sorted(expected_zero - zero_set)
    zero_equals_masked = zero_set == masked_set
    groups_b = dc.bone_groups(names)
    bone_group_rows = {}
    for gname, gi in groups_b.items():
        act = [i for i in gi if bool(useful[i])]
        msk = [i for i in gi if not bool(useful[i])]
        bone_group_rows[gname] = dict(
            n_bones=len(gi), n_active=len(act), n_zero_grad=len([i for i in gi if i in zero_set]),
            grad_norm_mean_active=(float(bone_norm[act].mean()) if (bone_norm is not None and act) else None),
            grad_norm_mean_masked=(float(bone_norm[msk].mean()) if (bone_norm is not None and msk) else None))
    cond_train = torch.stack([g['sixd_cond'] for g in grad_recs]).mean(0)            # [163]
    act_idx = [i for i in range(n_bones) if bool(useful[i])]
    cond_low = sorted(act_idx, key=lambda i: float(cond_train[i]))[:6]

    # decshape per phenotype
    ds_sq = [g['decshape_sq'] for g in grad_recs]
    pheno_norm = torch.stack(ds_sq).sum(0).sqrt() if all(d is not None for d in ds_sq) else None
    logit_mean = torch.stack([g['shape_logit_mean'] for g in grad_recs]).mean(0)
    slope_mean = torch.stack([g['shape_sigmoid_slope_mean'] for g in grad_recs]).mean(0)
    bias_mean = torch.stack([g['shape_pred_minus_gt_mean'] for g in grad_recs]).mean(0)
    pheno_rows, pheno_zero = [], []
    pmax = float(pheno_norm.max()) if pheno_norm is not None else 0.0
    for i, pn in enumerate(phen):
        nrm = float(pheno_norm[i]) if pheno_norm is not None else None
        is_zero = nrm is None or nrm == 0.0 or (pmax > 0 and nrm <= PHENO_ZERO_REL * pmax)
        if is_zero:
            pheno_zero.append(pn)
        pheno_rows.append(dict(phenotype=pn, grad_norm=nrm, rel_to_max=(nrm / pmax if (nrm is not None and pmax > 0) else None),
                               logit_mean=float(logit_mean[i]), sigmoid_slope_mean=float(slope_mean[i]),
                               pred_minus_gt_mean=float(bias_mean[i]), approx_zero=is_zero))

    # deccam per output
    dcam = [g['deccam_sq'] for g in grad_recs]
    cam_norm = torch.stack(dcam).sum(0).sqrt() if all(d is not None for d in dcam) else None

    # loss
    loss_keys = list(grad_recs[0]['loss'].keys())
    loss_mean = {k: mean_of([g['loss'].get(k) for g in grad_recs]) for k in loss_keys}
    loss_rows = []
    for k in loss_keys:
        if k == 'total':
            continue
        alpha = float(getattr(targs, LOSS_ALPHA[k])) if k in LOSS_ALPHA else None
        in_total = (k not in ('j2d', 'v2d')) or two_d_in_total
        w = (alpha * loss_mean[k]) if (alpha is not None and loss_mean[k] is not None) else None
        loss_rows.append(dict(term=k, value=loss_mean[k], alpha=alpha, weighted=w, in_total=in_total))
    tot_w = sum(r['weighted'] for r in loss_rows if r['in_total'] and r['weighted'] is not None)
    for r in loss_rows:
        r['share_of_total'] = (r['weighted'] / tot_w) if (r['in_total'] and r['weighted'] is not None and tot_w > 0) else None

    placement = None
    if head_idx is not None and all('placement' in g for g in grad_recs):
        placement = {k: mean_of([g['placement'][k] for g in grad_recs]) for k in grad_recs[0]['placement']}
        placement['model_person_center'] = names[model.person_center_idx]
        placement['gt_transl_joint'] = names[0]

    # response
    shape_sum = {}
    for r in resp_recs:
        for row in r['shape']:
            if 'error' in row:
                continue
            shape_sum.setdefault((row['phenotype'], row['delta_logit']), []).append(row)
    shape_summary = []
    for (pn, dl), rows in shape_sum.items():
        keys = ['d_pheno', 'd_pheno_expected', 'd_other_pheno_max', 'd_skel_height_mm', 'd_neck_mm',
                'v_disp_cam_mm', 'v_disp_pelvis_rel_mm', 'score_change']
        s = dict(phenotype=pn, delta_logit=dl, n_images=len(rows))
        s.update({k: mean_of([rw[k] for rw in rows]) for k in keys})
        shape_summary.append(s)
    pose_sum = {}
    for r in resp_recs:
        for row in r['pose']:
            if 'error' in row:
                continue
            pose_sum.setdefault(row['idx'], []).append(row)
    pose_summary = []
    for b, rows in pose_sum.items():
        keys = ['bone_rot_change_deg', 'other_bones_rot_change_max_deg', 'head_joint_disp_pelvis_rel_mm',
                'head_joint_disp_cam_mm', 'n_joints_moved', 'v_disp_max_mm', 'v_moved_frac',
                'v_disp_nearest_bone_mm', 'v_disp_not_nearest_bone_mm', 'v_disp_region_mm', 'v_disp_outside_region_mm']
        s = dict(bone=names[b], idx=b, n_images=len(rows), masked_by_useful_rotmat=rows[0]['masked_by_useful_rotmat'],
                 in_hand_bone_mask=rows[0]['in_hand_bone_mask'],
                 sixd_cond=mean_of([float(r['sixd_cond'][b]) for r in resp_recs]),
                 sixd_col1_norm=mean_of([float(r['sixd_colnorm'][b, 0]) for r in resp_recs]),
                 sixd_col2_norm=mean_of([float(r['sixd_colnorm'][b, 1]) for r in resp_recs]))
        s.update({k: mean_of([rw[k] for rw in rows]) for k in keys})
        demo = [rw['demo_toggles'] for rw in rows if 'demo_toggles' in rw]
        s['demo_bone_rot_change_deg'] = mean_of([d['bone_rot_change_deg'] for d in demo])
        s['demo_v_disp_max_mm'] = mean_of([d['v_disp_max_mm'] for d in demo])
        s['top_moved_joints_mm_first_image'] = rows[0]['top_moved_joints_mm']
        pose_summary.append(s)

    # ======================================================================= #
    # Verdict
    # ======================================================================= #
    problems, warns = [], []
    for g in HEAD_GROUPS:
        a = modules.get(g)
        if a is None or a['n_requires_grad'] == 0:
            problems.append(f"{g}: no trainable parameters")
        elif a['n_grad_none_max'] > 0 or not a['grad_norm_min']:
            problems.append(f"{g}: no gradient in at least one image (grad None or zero norm)")
    n_nonfinite = sum(a['n_nonfinite_total'] for a in modules.values())
    if n_nonfinite:
        problems.append(f"{n_nonfinite} non-finite gradient entries (train.py's NaN guard would SKIP the optimizer step)")
    if not all(g['loss_finite'] for g in grad_recs):
        problems.append("non-finite loss on at least one image (train.py would skip the batch)")
    if unexpected_zero:
        problems.append(f"{len(unexpected_zero)} decpose bone(s) expected to learn have EXACTLY zero gradient: "
                        f"{names_of(unexpected_zero, names, 12)}")
    if args.mask_helper_joints_train and unexpected_nonzero:
        problems.append(f"{len(unexpected_nonzero)} masked bone(s) receive gradient although "
                        f"mask_helper_joints_train=1 (mask not applied in the forward?): {names_of(unexpected_nonzero, names, 12)}")
    if pheno_zero:
        problems.append(f"decshape phenotype(s) with ~0 gradient: {pheno_zero} (check sigmoid saturation column)")
    for pn in SHAPE_TESTS:
        rows = [s for s in shape_summary if s['phenotype'] == pn]
        if not rows:
            continue
        moved = max(max(s['v_disp_cam_mm'] or 0.0, s['v_disp_pelvis_rel_mm'] or 0.0) for s in rows)
        dph = max(abs(s['d_pheno'] or 0.0) for s in rows)
        if moved <= NONZERO_MM or dph <= 1e-4:
            problems.append(f"shape '{pn}' +-{args.shape_delta} logits: mesh does not respond "
                            f"(|d pheno| {dph:.5f}, mean vertex displacement {moved:.3f} mm)")
        mism = max(abs((s['d_pheno'] or 0.0) - (s['d_pheno_expected'] or 0.0)) for s in rows)
        if mism > 1e-3:
            problems.append(f"shape '{pn}': measured phenotype change differs from sigmoid(logit+delta)-sigmoid(logit) "
                            f"by {mism:.4f}: something between decshape and the output alters the phenotype")
        leak = max((s['d_other_pheno_max'] or 0.0) for s in rows)
        if leak > 1e-5:
            warns.append(f"shape '{pn}': perturbing one decshape output changed OTHER phenotypes by up to {leak:.2e}")
    for s in pose_summary:
        if (s['bone_rot_change_deg'] or 0.0) <= NONZERO_DEG or (s['v_disp_max_mm'] or 0.0) <= NONZERO_MM:
            problems.append(f"bone '{s['bone']}' (toggles OFF): decoded rotation/mesh does not respond "
                            f"(rot {fmt_or(s['bone_rot_change_deg'], 3)} deg, max vertex disp "
                            f"{fmt_or(s['v_disp_max_mm'], 3)} mm)")
        if (s['other_bones_rot_change_max_deg'] or 0.0) > NONZERO_DEG:
            warns.append(f"bone '{s['bone']}': perturbing its 6D output also rotated OTHER bones by up to "
                         f"{s['other_bones_rot_change_max_deg']:.3f} deg")
    if not resp_recs:
        problems.append("response test could not run (no detection on any image)")

    bmod = modules.get('body_model.')
    if bmod and bmod['n_requires_grad'] > 0 and (bmod['grad_norm_max'] or 0.0) > 0:
        warns.append(f"body_model has {bmod['n_requires_grad']} trainable tensor(s) receiving gradient (|g| up to "
                     f"{sci(bmod['grad_norm_max'])}): they are inside model.parameters(), so train.py's Adam "
                     f"(filter requires_grad) UPDATES THE ANNY BODY MODEL ITSELF")
    if mean_of(clip) is not None and mean_of(clip) < 0.1:
        warns.append(f"global grad norm {mean_of(total_norms):.3e} >> 1: clip_grad_norm_(max_norm=1.0) scales every "
                     f"update by ~{mean_of(clip):.2e}; the heads' effective step is lr x that factor")
    if placement is not None and placement['model_person_center'] != placement['gt_transl_joint']:
        warns.append(f"loss.py reg_transl compares pred['transl'] = joint '{placement['model_person_center']}' "
                     f"(model.py:442) with gt['transl'] = joint 0 '{placement['gt_transl_joint']}' (train.py:214-219); "
                     f"measured |pred transl - gt transl| {fmt_or(placement['transl_term_m'], 3)} m, "
                     f"|pred head - gt head| {fmt_or(placement['pred_head_vs_gt_head_m'], 3)} m, "
                     f"GT head-root distance {fmt_or(placement['gt_head_to_root_m'], 3)} m")
    if cond_low and float(cond_train[cond_low[0]]) < COND_WARN:
        warns.append(f"6D output nearly degenerate (|sin| < {COND_WARN}) for active bone(s) "
                     f"{[names[i] for i in cond_low if float(cond_train[i]) < COND_WARN]}: Gram-Schmidt is near-singular "
                     f"there. Note init_body_pose [1,0,0,1,0,0] (model.py:565) is DEGENERATE under utils.rot6d_to_rotmat's "
                     f"reshape(-1,2,3) layout (both columns (1,0,0)); it is the identity only in the reshape(-1,3,2) "
                     f"layout multi_hmr_anny/multi_hmr.py:193 uses")

    verdict = ('heads receive gradients and meshes respond -> optimisation path intact; failures are '
               'supervision/data/generalisation') if not problems else \
              ('zero gradient on active bones / no mesh response -> decoding or optimisation bug '
               '(see the problems listed above)')

    # ======================================================================= #
    # Print
    # ======================================================================= #
    print("\n" + "=" * 100)
    print(f"CHECK 4 -- gradients + response   ckpt {ckpt_path}")
    print(f"GT path: {gt_path}" + (f"   (Trainer error: {gt_err})" if gt_err else ''))
    print(f"images used {used}; skipped {[s['idx'] for s in skipped]}; mask_helper_joints_train="
          f"{args.mask_helper_joints_train}; amp=0; loss epoch {args.loss_epoch}")
    print("=" * 100)

    print_table("PER-MODULE GRADIENTS (aggregated over images; norm = L2 over the module, per image)",
                ['module', '#tensors', '#elems', '#req_grad', '#grad None (max)', '|g| mean', '|g| min',
                 'zero frac', 'non-finite'],
                [[g, a['n_tensors'], a['n_elems'], a['n_requires_grad'], a['n_grad_none_max'],
                  sci(a['grad_norm_mean']), sci(a['grad_norm_min']),
                  dc.fmt(a['zero_frac_mean'], 4) if a['zero_frac_mean'] is not None else '-',
                  a['n_nonfinite_total']] for g, a in modules.items()])
    if 'other' in modules:
        print(f"  'other' tensors: {modules['other']['tensor_names']}")
    print(f"  global grad norm (what clip_grad_norm_ sees): mean {mean_of(total_norms):.3e}; clip factor "
          f"(max_norm=1.0) mean {mean_of(clip):.3e}")
    print("  notes: cross_* rows only get gradient at the detected cells (high zero frac expected); "
          "deccam outputs 1,2 are never read (model.py:383-384 uses pred_cam[:, 0] only) -> zero by construction")

    print("\nDECPOSE per-bone gradient ([163, 6, dim] weight rows + bias, aggregated over images)")
    print(f"  bones with EXACTLY zero gradient: {len(zero_set)} (per image {per_image_zero}); "
          f"masked by useful_rotmat: {len(masked_set)}; sets identical: {zero_equals_masked}")
    print(f"  active-but-zero ({len(active_but_zero)}): {names_of(active_but_zero, names, 30)}")
    print(f"  masked-but-nonzero ({len(masked_but_nonzero)}): {names_of(masked_but_nonzero, names, 30)}")
    if args.mask_helper_joints_train:
        print("  interpretation: mask_helper_joints_train=1 replaces the masked bones' rotations by identity BEFORE the "
              "body model (model.py:360-363), so their decpose rows get exactly zero gradient BY DESIGN; any "
              "active-but-zero bone is a bug.")
    else:
        print("  interpretation: mask_helper_joints_train=0: every bone is decoded and supervised, so NO bone should "
              "have exactly zero gradient.")
    print_table("  per bone group (dc.bone_groups)", ['group', '#bones', '#active', '#zero grad', '|g| mean active',
                                                      '|g| mean masked'],
                [[k, v['n_bones'], v['n_active'], v['n_zero_grad'], sci(v['grad_norm_mean_active']),
                  sci(v['grad_norm_mean_masked'])] for k, v in bone_group_rows.items()])
    if bone_norm is not None:
        print("  tested bones: " + ', '.join(f"{names[b]} |g| {sci(float(bone_norm[b]))}" for b in bones))
    print(f"  6D conditioning |sin(col1, col2)| on the training forward (1 = orthogonal, 0 = degenerate): "
          f"active bones min {float(cond_train[act_idx].min()):.3f}, median {float(cond_train[act_idx].median()):.3f}; "
          f"lowest: {[(names[i], round(float(cond_train[i]), 3)) for i in cond_low]}")

    print_table("DECSHAPE per-phenotype gradient (aggregated over images)",
                ['phenotype', '|g|', 'rel to max', 'logit mean', 'sigmoid slope', 'pred-GT mean', '~0?'],
                [[r['phenotype'], sci(r['grad_norm']), dc.fmt(r['rel_to_max'], 3), dc.fmt(r['logit_mean'], 2),
                  dc.fmt(r['sigmoid_slope_mean'], 3), dc.fmt(r['pred_minus_gt_mean'], 3, sign=True),
                  'YES' if r['approx_zero'] else ''] for r in pheno_rows])
    if cam_norm is not None:
        print(f"\nDECCAM per output |g|: {[sci(float(v)) for v in cam_norm]}  (output 0 = dist; 1, 2 unused)")

    print_table(f"LOSS TERMS (mean over images; total {fmt_or(loss_mean.get('total'), 4)})",
                ['term', 'value', 'alpha', 'alpha x value', 'in total', 'share'],
                [[r['term'], dc.fmt(r['value'], 4), dc.fmt(r['alpha'], 2), dc.fmt(r['weighted'], 4),
                  'yes' if r['in_total'] else 'no', dc.fmt(r['share_of_total'], 3)] for r in loss_rows])

    if placement is not None:
        print(f"\nPLACEMENT (training forward, mean over persons/images): |pred transl - gt transl| "
              f"{fmt_or(placement['transl_term_m'], 3)} m  (pred = '{placement['model_person_center']}' joint, "
              f"gt = joint 0 '{placement['gt_transl_joint']}');  |pred head - gt head| "
              f"{fmt_or(placement['pred_head_vs_gt_head_m'], 3)} m;  |pred root - gt root| "
              f"{fmt_or(placement['pred_root_vs_gt_root_m'], 3)} m;  GT head-root {fmt_or(placement['gt_head_to_root_m'], 3)} m")

    print_table(f"RESPONSE: decshape logit +-{args.shape_delta} on one phenotype (eval, toggles OFF, best person; "
                f"mean over {len(resp_recs)} image(s))",
                ['phenotype', 'delta', 'd pheno', 'expected', 'd others max', 'd skel height mm', 'd neck mm',
                 'v disp cam mm', 'v disp pelvis-rel mm'],
                [[s['phenotype'], f"{s['delta_logit']:+.1f}", dc.fmt(s['d_pheno'], 4, sign=True),
                  dc.fmt(s['d_pheno_expected'], 4, sign=True), sci(s['d_other_pheno_max']),
                  dc.fmt(s['d_skel_height_mm'], 1, sign=True), dc.fmt(s['d_neck_mm'], 2, sign=True),
                  dc.fmt(s['v_disp_cam_mm'], 2), dc.fmt(s['v_disp_pelvis_rel_mm'], 2)] for s in shape_summary])
    print_table(f"RESPONSE: decpose +{args.pose_delta} on 6D components {args.pose_components} of one bone "
                f"(eval, toggles OFF; displacements pelvis-relative)",
                ['bone', 'masked', 'hand', '6D cond', '|col1|,|col2|', 'rot chg deg', 'other max deg', 'head disp mm', '#joints moved',
                 'v max mm', 'v near-bone mm', 'v other mm', 'v region mm', 'v outside mm', 'demo: rot deg / v max mm'],
                [[s['bone'], 'Y' if s['masked_by_useful_rotmat'] else '', 'Y' if s['in_hand_bone_mask'] else '',
                  dc.fmt(s['sixd_cond'], 3), f"{dc.fmt(s['sixd_col1_norm'], 2)},{dc.fmt(s['sixd_col2_norm'], 2)}",
                  dc.fmt(s['bone_rot_change_deg'], 2), dc.fmt(s['other_bones_rot_change_max_deg'], 3),
                  dc.fmt(s['head_joint_disp_pelvis_rel_mm'], 2), dc.fmt(s['n_joints_moved'], 1), dc.fmt(s['v_disp_max_mm'], 2),
                  dc.fmt(s['v_disp_nearest_bone_mm'], 2), dc.fmt(s['v_disp_not_nearest_bone_mm'], 3),
                  dc.fmt(s['v_disp_region_mm'], 2), dc.fmt(s['v_disp_outside_region_mm'], 3),
                  f"{dc.fmt(s['demo_bone_rot_change_deg'], 2)} / {dc.fmt(s['demo_v_disp_max_mm'], 2)}"]
                 for s in pose_summary])
    for s in pose_summary:
        print(f"  {s['bone']}: top moved joints (first image, mm): "
              f"{ {k: round(v, 1) for k, v in s['top_moved_joints_mm_first_image'].items()} }")
    print("  notes: toggles mask_helper_joints / relax_hands / rest_pose are OFF here so every bone CAN respond. "
          "'demo' repeats the perturbation with mask_helper_joints=1 + relax_hands=1: masked (and, with relax_hands, "
          "finger/thumb) bones are replaced by identity, so they CANNOT respond by construction (0 there is expected). "
          "The head joint is the origin of the 'head' bone, so its own rotation does not move it (children and skinned "
          "vertices move); in the camera frame the head joint never moves because the mesh is anchored at "
          "person_center='head' (model.py:430-432).")

    print("\nWARNINGS:" if warns else "\nWARNINGS: none")
    for w in warns:
        print(f"  - {w}")
    print("PROBLEMS:" if problems else "PROBLEMS: none")
    for p in problems:
        print(f"  - {p}")
    print("\nINTERPRETATION:")
    print("  heads receive gradients and meshes respond -> optimisation path intact; failures are supervision/data/generalisation")
    print("  zero gradient on active bones / no mesh response -> decoding or optimisation bug")
    print(f"VERDICT: {verdict}")

    # ======================================================================= #
    # JSON
    # ======================================================================= #
    def t2l(t):
        return None if t is None else [float(v) for v in t]

    summary = dict(
        check='CHECK 4: head gradients + decoded-mesh response',
        ckpt=ckpt_path, gt_path=gt_path, gt_path_error=gt_err,
        images_used=used, images_skipped=skipped,
        mask_helper_joints_train=args.mask_helper_joints_train, ckpt_mask_helper_joints_train=ckpt_mask,
        loss_epoch=args.loss_epoch, start_2d_epoch=int(targs.start_2d_epoch), two_d_terms_in_total=two_d_in_total,
        train_args_overrides={k: dict(ckpt=str(overridden[k]), used=getattr(targs, k)) for k in overridden},
        loss_defaults_filled=filled, freeze=freeze_info,
        modules=modules,
        total_grad_norm=dict(per_image=total_norms, mean=mean_of(total_norms), clip_coef_per_image=clip),
        decpose=dict(
            zero_grad_bones=names_of(zero_set, names), n_zero_grad=len(zero_set), per_image_n_zero=per_image_zero,
            masked_bones=names_of(masked_set, names), n_masked=len(masked_set),
            zero_set_equals_masked_set=zero_equals_masked,
            active_but_zero=names_of(active_but_zero, names), masked_but_nonzero=names_of(masked_but_nonzero, names),
            unexpected_zero=names_of(unexpected_zero, names), unexpected_nonzero=names_of(unexpected_nonzero, names),
            per_bone_grad_norm={names[i]: (float(bone_norm[i]) if bone_norm is not None else None) for i in range(n_bones)},
            per_group=bone_group_rows,
            sixd_conditioning_train={names[i]: float(cond_train[i]) for i in range(n_bones)}),
        decshape=dict(per_phenotype=pheno_rows, approx_zero=pheno_zero),
        deccam=dict(per_output_grad_norm=t2l(cam_norm)),
        loss=dict(mean=loss_mean, terms=loss_rows, per_image=[dict(idx=g['idx'], **g['loss']) for g in grad_recs]),
        placement=placement,
        placement_per_image=[dict(idx=g['idx'], **g['placement']) for g in grad_recs if 'placement' in g],
        response=dict(shape_delta=args.shape_delta, pose_delta=args.pose_delta, pose_components=args.pose_components,
                      det_thresh=args.det_thresh, shape_summary=shape_summary, pose_summary=pose_summary,
                      per_image=[dict(idx=r['idx'], n_det=r['n_det'], best=r['best'], score=r['score'],
                                      shape=r['shape'], pose=r['pose']) for r in resp_recs]),
        per_image_modules=[dict(idx=g['idx'], **{k: dict(grad_norm=v['grad_norm'], zero_frac=v['zero_frac'],
                                                         n_grad_none=v['n_grad_none'], n_nonfinite=v['n_nonfinite'])
                                                 for k, v in g['groups'].items()}) for g in grad_recs],
        warnings=warns, problems=problems, verdict=verdict,
        runtime_s=time.time() - t0,
    )
    path = os.path.join(out_dir, 'summary.json')
    with open(path, 'w') as f:
        json.dump(dc.json_safe(summary), f, indent=2, allow_nan=False)
    print(f"\n[done] {path}  ({time.time() - t0:.0f}s)")


def fmt_or(x, nd=2):
    return dc.fmt(x, nd) if x is not None else '-'


if __name__ == '__main__':
    main()
