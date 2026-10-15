"""
Shared helpers for the diagnostic scripts in scripts/diagnostics/.

Everything here is READ-ONLY with respect to the baseline: no checkpoint is
written, no training is launched. The model is loaded the same way demo.py's
load_model() does it (checkpoint args rebuild the architecture), restricted to
the Model class from model.py, which is what train.py produces.

Run from the repo root inside the multihmr conda env (GPU scripts need
PYOPENGL_PLATFORM=egl for the mesh overlays; set here before any import).
"""
import glob
import math
import os
import pickle
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

DATA_FOLDER_DEFAULT = '/netscratch/najib/anydataset/'
N_HOLDOUT = 100          # train.py --val_anny_n 100: the LAST 100 samples are never trained on
PHENOTYPES = ['gender', 'age', 'muscle', 'weight', 'height', 'proportions',
              'cupsize', 'firmness', 'african', 'asian', 'caucasian']


# --------------------------------------------------------------------------- #
# checkpoint / model
# --------------------------------------------------------------------------- #
def resolve_ckpt(spec):
    """Accept a path, a models/multiHMR name, or 'run_name:epoch'."""
    if os.path.isfile(spec):
        return spec
    cand = [os.path.join(ROOT, 'models', 'multiHMR', spec + '.pt'),
            os.path.join(ROOT, 'models', 'multiHMR', spec)]
    if ':' in spec:
        run, ep = spec.split(':', 1)
        cand.append(os.path.join(ROOT, 'logs', 'anny_model', run, 'checkpoints', f'{int(ep):05d}.pt'))
    for c in cand:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"checkpoint not found: {spec} (tried {cand})")


def load_model(ckpt_path, device, tf32=True, bone_fix=''):
    """Same construction as demo.load_model(), for model.py's Model only.
    tf32=True mirrors train.py (--tf32 default 1): evaluate()'s 'fp32' numbers used TF32 matmuls.
    bone_fix (exp6, bone_fix.py): '' = whatever the checkpoint args carry (bone_fix_spec, saved by train.py --bone_fix;
    none for older checkpoints), 'none' = forced off, a path = that JSON instead of the checkpoint's."""
    from model import Model
    import bone_fix as bone_fix_lib
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get('args') if isinstance(ckpt, dict) else None
    raw_args = dict(vars(a)) if hasattr(a, '__dict__') else dict(a or {})   # as saved, before the defaults below
    # H3 readout FIRST, before anything that can fail on an old checkpoint (Model(**kwargs), the >= 95% load check):
    # the oldest ancestors are the ones H3 is about and the ones most likely not to build under the current model.py
    rp = raw_args.get('pose_param', None)
    print(f"[load_model] {ckpt_path}: saved ckpt_args (H3, read before the model is built): "
          f"{'NO args saved; ' if not raw_args else ''}"
          f"pose_param={'absent(local-bone assumed)' if rp is None else rp} "
          + " ".join(f"{k}={raw_args.get(k)}" for k in CKPT_ARGS_REPORTED), flush=True)
    if rp is not None and rp != 'local-bone':
        print(f"[load_model] WARNING: checkpoint trained with pose_param='{rp}', but model.py evaluates every "
              f"checkpoint as local-bone (demo.py refuses this case): its rotations are read in the wrong space",
              flush=True)
    if not raw_args:                      # as before (KeyError on ckpt['args']): never rebuild from defaults
        raise KeyError(f"{ckpt_path}: no saved 'args'; cannot rebuild the model")
    kwargs = dict(raw_args)
    kwargs.setdefault('simple_depth_encoding', 1)
    kwargs.setdefault('pose_param', 'local-bone')
    kwargs['person_center'] = 'head'      # demo.py does the same override
    fix_spec, fix_origin = bone_fix_lib.from_arg(bone_fix, raw_args.get('bone_fix_spec'))
    print(f"[load_model] bone_fix: {fix_origin}" + (" (the checkpoint's own fix is overridden)"
                                                    if bone_fix and raw_args.get('bone_fix_spec') else ''), flush=True)
    kwargs['bone_fix'] = 'none' if fix_spec is None else ''
    kwargs['bone_fix_spec'] = fix_spec
    sd = ckpt['model_state_dict']
    n_model = sum(1 for k in sd if k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')))
    n_orig = sum(1 for k in sd if k.startswith(('encoder.backbone.', 'decoder.', 'mlp_pose.')))
    if n_model <= n_orig:
        raise RuntimeError("checkpoint is not a model.py Model (looks like the original Multi_HMR class); "
                           "these diagnostics target the train.py checkpoints")
    model = Model(**kwargs).to(device)
    msd = model.state_dict()
    matched = [k for k, v in sd.items() if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape]
    model.load_state_dict(sd, strict=False)
    pct = 100.0 * len(matched) / max(len(msd), 1)
    print(f"[load_model] {ckpt_path}: {len(matched)}/{len(msd)} tensors loaded ({pct:.1f}%)")
    if pct < 95.0:
        missing = [k for k in msd if k not in sd][:8]
        raise RuntimeError(f"only {pct:.1f}% of the model initialised; missing e.g. {missing}")
    model.eval()
    model.ckpt_raw_args = raw_args        # the saved args, unmodified (diag_holdout_trace --dump_bones meta)
    print(f"[load_model] img_size={model.img_size} backbone_dim={model.backbone_dim} "
          f"feat_proj={'yes' if getattr(model, 'feat_proj', None) is not None else 'no'} "
          f"ckpt_args: backbone={kwargs.get('backbone')} head_dim={kwargs.get('head_dim')} "
          f"mask_helper_joints_train={kwargs.get('mask_helper_joints_train')} "
          f"use_anny_shape={kwargs.get('use_anny_shape')} alpha_shape={kwargs.get('alpha_shape')} "
          f"pose_param={'absent(local-bone assumed)' if rp is None else rp}")
    return model, kwargs


# saved training args printed by load_model and stored in the --dump_bones meta (the H3 check of the neck audit:
# docs/experiments_plan.md, "Neck root cause"); None = the checkpoint predates the flag
CKPT_ARGS_REPORTED = ('boost_neck_weight', 'boost_hand_weight', 'alpha_rotmat', 'rotmat_loss', 'rotmat_region_norm',
                      'alpha_global_rot', 'mix_3dpw_labels', 'pretrained', 'load_only_backbone')


def set_inference_toggles(model, mask_helper_joints, relax_hands=False, rest_pose=False):
    """The flags demo.py sets from the CLI (all default OFF inside train.py evaluate())."""
    model.mask_helper_joints = bool(mask_helper_joints)
    model.relax_hands = bool(relax_hands)
    model.rest_pose = bool(rest_pose)
    model.neutral_shape = False
    model.neutral_phenotype_names = []


# --------------------------------------------------------------------------- #
# dataset access (direct pkl reads are ~100x faster than AnnyOne.__getitem__,
# which also decodes the image)
# --------------------------------------------------------------------------- #
def list_pkls(data_folder):
    files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))   # same order as AnnyOne
    if not files:
        raise FileNotFoundError(f"no .pkl annotations under {data_folder}")
    return files


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def holdout_indices(n_total, n=N_HOLDOUT):
    return list(range(max(0, n_total - n), n_total))


def image_path_for(pkl_path):
    p = pkl_path.replace('.pkl', '.jpg')
    if not os.path.exists(p):
        p = pkl_path.replace('.pkl', '_seg.png')
    return p


def shapes_from_pkl(data):
    """[n_humans, 11] float32 phenotypes, or None."""
    rows = []
    for h in data.get('humans', []):
        s = h.get('anny_shape', None)
        if s is None:
            return None
        rows.append(np.asarray(s, dtype=np.float32).reshape(-1))
    if not rows:
        return np.zeros((0, 11), dtype=np.float32)
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def orthonormalize(R):
    """Project [..., 3, 3] onto SO(3) exactly as train.py prepare_gt does."""
    shp = R.shape
    Rf = R.reshape(-1, 3, 3)
    U, _, Vh = torch.linalg.svd(Rf)
    det = torch.linalg.det(U @ Vh)
    D = torch.ones_like(U[:, :, 0])
    D[:, 2] = det
    return (U @ torch.diag_embed(D) @ Vh).reshape(shp)


def geodesic_deg(R1, R2):
    """Per-matrix rotation angle between R1 and R2 ([..., 3, 3]) in degrees."""
    M = R1.transpose(-1, -2) @ R2
    tr = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def rotmat_to_homogeneous(R):
    """[N, J, 3, 3] -> [N, J, 4, 4] with zero translation."""
    N, J = R.shape[:2]
    T = torch.eye(4, device=R.device, dtype=R.dtype).reshape(1, 1, 4, 4).repeat(N, J, 1, 1)
    T[:, :, :3, :3] = R
    return T


def identity_pose(n, n_bones, device):
    return torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(n, n_bones, 1, 1)


def swing_twist_deg(R, axis, undefined_tol=3e-4):
    """
    Swing-twist split of rotations R [..., 3, 3] about unit axes [..., 3] given in the same frame as R:
    R = R_swing @ R_twist, R_twist a rotation about `axis`, R_swing about an axis perpendicular to it.
    Returns (swing [...] in [0, 180], twist [...] signed in [-180, 180], right-hand rule about +axis), degrees.
    Closed form without a quaternion branch: with c = cos(angle) = (tr R - 1) / 2, the unit quaternion (w, v) of R
    has w^2 = (1 + tr R) / 4 and (v . axis)^2 = (axis . R axis - c) / 2, so |twist| = 2 atan2(|v . axis|, |w|), with
    the sign of axis . vee(R - R^T); swing = angle between axis and R axis (the twist leaves the axis fixed).
    The twist is UNDEFINED, and returned as NaN (not 0), when the swing is near 180 deg: w^2 + (v . axis)^2 =
    cos^2(swing / 2), so both terms vanish there and float32 round-off alone sets the atan2 (measured: 0 to 110+ deg
    for exact 180-deg swings). It degrades before that: float32 twist error, max over 400 random frames, 0.26 deg at a
    170-deg swing, 2.1 at 178, 3.2 at 179, 8.1 at 179.5. NaN is returned where w^2 + (v . axis)^2 < undefined_tol
    (3e-4: swing > ~178.0 deg), so a returned twist is within ~2.5 deg in float32 (max 2.4 at a 177.9-deg swing).
    The SIGN is also unreliable for |twist| near 180: vee(R - R^T) shrinks to 0 there, so a few degrees of residual
    flip it; average twists on the circle (circular_mean_deg), never as signed numbers.
    """
    a = axis / torch.clamp(axis.norm(dim=-1, keepdim=True), min=1e-12)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    Ra = (R @ a.unsqueeze(-1)).squeeze(-1)                                     # R axis
    a_Ra = (a * Ra).sum(-1)
    w2 = torch.clamp((1.0 + tr) / 4.0, min=0.0)
    vd2 = torch.clamp((a_Ra - (tr - 1.0) / 2.0) / 2.0, min=0.0)
    twist = 2.0 * torch.atan2(torch.sqrt(vd2), torch.sqrt(w2))
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sgn = torch.where((a * vee).sum(-1) < 0, -torch.ones_like(twist), torch.ones_like(twist))
    swing = torch.acos(torch.clamp(a_Ra, -1.0, 1.0))
    twist = torch.where(w2 + vd2 < undefined_tol, torch.full_like(twist, float('nan')), twist)
    return torch.rad2deg(swing), torch.rad2deg(twist) * sgn


def circular_mean_deg(a):
    """Circular mean of angles a (deg; NaN entries ignored) = atan2(mean sin, mean cos), in [-180, 180], plus the
    mean resultant length in [0, 1] (1 = all equal, ~0 = no preferred direction, e.g. +-180 split evenly reads ~180
    with length ~1, but +90/-90 split evenly has length 0 and no meaningful mean). Returns (None, None) when empty."""
    a = np.asarray(a, dtype=np.float64)
    a = np.deg2rad(a[np.isfinite(a)])
    if a.size == 0:
        return None, None
    s, c = float(np.sin(a).mean()), float(np.cos(a).mean())
    return math.degrees(math.atan2(s, c)), math.hypot(s, c)


def chordal_mean(R):
    """Chordal L2 mean of rotations [P, 3, 3] -> [3, 3]: the SO(3) projection of the entry-wise mean."""
    return orthonormalize(R.mean(0, keepdim=True))[0]


def rotation_spread_deg(R):
    """Mean geodesic distance (deg) of rotations [P, 3, 3] to their chordal mean: 0 = the same rotation for all P."""
    if R.shape[0] == 0:
        return float('nan')
    return float(geodesic_deg(R, chordal_mean(R).expand_as(R)).mean())


def axis_angle_deg(R):
    """Axis (unit [3]) and angle (deg) of one rotation [3, 3]. Above ~175 deg the axis comes from the symmetric part
    (R + I) / 2 = k k^T, whose sign is arbitrary; at 0 deg the axis is returned as zeros."""
    R = R.double()
    ang = float(geodesic_deg(torch.eye(3, dtype=R.dtype, device=R.device), R))
    vee = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return torch.zeros(3, dtype=torch.float32), 0.0
    if ang < 175.0:
        k = vee / vee.norm()
    else:
        S = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2.0
        j = int(torch.argmax(torch.diagonal(S)))
        k = S[:, j] / torch.clamp(S[:, j].norm(), min=1e-12)
        if float((k * vee).sum()) < 0:                                        # keep the right-hand sign when defined
            k = -k
    return k.float().cpu(), ang


def pick_twist_children(bp_rest, parents, bone_idx, min_cos=0.95):
    """
    Twist-axis convention of the per-bone dump (diag_holdout_trace.py --dump_bones). bp_rest [B, 4, 4] are the
    GLOBAL bone poses at the identity local-bone pose (the reference frames the local rotations are expressed in;
    anny's local-bone FK is pose_b = T_parent @ rest_b @ delta_b, so delta_b acts in bone b's own rest frame).
    For each bone b the axis is the rest child-offset direction R_rest_b^T (head_c - head_b) / |.| to its CONNECTED
    child c: the direct child whose offset is most aligned with the bone's local +Y, accepted when that cosine is
    >= min_cos (anny rigs follow Blender: +Y runs from a bone's head to its tail, and a connected child's head is
    that tail, e.g. neck01 -> neck02 -> neck03 -> head, spine01 -> neck01, clavicle -> shoulder01). Helper and hand
    children count too: on anny's default rig head -> special06.L, jaw -> special04 and wrist.L/R -> metacarpal1.L/R
    have their head at the parent's tail (cos 1.000), so those axes equal local +Y. A bone with no child within
    min_cos uses local +Y itself.
    Returns (child index or -1 per bone, cosine of the best child offset with +Y or nan per bone).
    """
    parents = [int(p) for p in parents]
    kids_of = {}
    for c, p in enumerate(parents):
        kids_of.setdefault(p, []).append(c)
    R, t = bp_rest[:, :3, :3], bp_rest[:, :3, 3]
    children, cosy = [], []
    for b in bone_idx:
        best, best_cos = -1, float('nan')
        for c in kids_of.get(int(b), []):
            off = R[b].transpose(0, 1) @ (t[c] - t[b])
            n = float(off.norm())
            if n < 1e-4:
                continue
            cy = float(off[1]) / n
            if not math.isfinite(best_cos) or cy > best_cos:
                best, best_cos = c, cy
        children.append(best if (best >= 0 and best_cos >= min_cos) else -1)
        cosy.append(best_cos)
    return children, cosy


def twist_axes(bp_rest, bone_idx, children):
    """Per-person twist axes [N, len(bone_idx), 3] in each bone's own rest frame (see pick_twist_children):
    the child-offset direction on that person's rest skeleton, or local +Y where children[k] == -1."""
    N = bp_rest.shape[0]
    out = torch.zeros(N, len(bone_idx), 3, device=bp_rest.device, dtype=bp_rest.dtype)
    for k, (b, c) in enumerate(zip(bone_idx, children)):
        if c < 0:
            out[:, k, 1] = 1.0
            continue
        off = (bp_rest[:, b, :3, :3].transpose(-1, -2) @ (bp_rest[:, c, :3, 3] - bp_rest[:, b, :3, 3]).unsqueeze(-1)).squeeze(-1)
        out[:, k] = off / torch.clamp(off.norm(dim=-1, keepdim=True), min=1e-12)
    return out


# --------------------------------------------------------------------------- #
# Anny body model wrappers
# --------------------------------------------------------------------------- #
def run_body_model(bm, pose44, shape11):
    """
    pose44 [N, 163, 4, 4], shape11 [N, 11] (dataset order == bm.phenotype_labels).
    Returns vertices [N, V, 3], joints [N, 163, 3] (bone heads, same frame as pose44's root).
    """
    labels = list(bm.phenotype_labels)
    shape11 = torch.clamp(shape11, 0.0, 1.0)
    kw = {k: shape11[:, i] for i, k in enumerate(labels)}
    with torch.no_grad():
        out = bm(pose_parameters=pose44, phenotype_kwargs=kw)
    return out['vertices'], out['bone_poses'][:, :, :3, -1]


def run_body_model_full(bm, pose44, shape11):
    """As run_body_model but also returns the GLOBAL bone poses [N, 163, 4, 4]."""
    labels = list(bm.phenotype_labels)
    shape11 = torch.clamp(shape11, 0.0, 1.0)
    kw = {k: shape11[:, i] for i, k in enumerate(labels)}
    with torch.no_grad():
        out = bm(pose_parameters=pose44, phenotype_kwargs=kw)
    return out['vertices'], out['bone_poses'][:, :, :3, -1], out['bone_poses']


def bone_groups(names):
    """Name-based bone groups; prints membership so the grouping can be checked."""
    groups = {k: [] for k in ['root', 'spine', 'neck', 'head', 'clavicle_shoulder', 'arm', 'wrist',
                              'fingers', 'leg', 'foot_toe', 'breast', 'face_other']}
    for i, n in enumerate(names):
        lo = n.lower()
        if i == 0 or lo == 'root':
            g = 'root'
        elif lo.startswith('neck'):
            g = 'neck'
        elif lo == 'head':
            g = 'head'
        elif 'spine' in lo:
            g = 'spine'
        elif 'clavicle' in lo or 'shoulder' in lo:
            g = 'clavicle_shoulder'
        elif 'upperarm' in lo or 'lowerarm' in lo:
            g = 'arm'
        elif 'wrist' in lo:
            g = 'wrist'
        elif 'finger' in lo or 'thumb' in lo or 'metacarpal' in lo:
            g = 'fingers'          # NB: model.relax_hands masks finger/thumb only, NOT metacarpals
        elif lo == 'jaw':
            g = 'head'
        elif 'upperleg' in lo or 'lowerleg' in lo or 'pelvis' in lo:
            g = 'leg'
        elif 'foot' in lo or 'toe' in lo:
            g = 'foot_toe'
        elif 'breast' in lo:
            g = 'breast'
        else:
            g = 'face_other'
        groups[g].append(i)
    return {k: v for k, v in groups.items() if v}


def neck_chain(names):
    """Indices [neck01, neck02, ..., head] in chain order (by name sort)."""
    necks = sorted([i for i, n in enumerate(names) if n.lower().startswith('neck')],
                   key=lambda i: names[i].lower())
    head = [i for i, n in enumerate(names) if n.lower() == 'head']
    return necks + head


def feet_indices(names):
    return [i for i, n in enumerate(names) if n.lower().startswith('foot')]


def chain_length(j3d, chain):
    """Sum of segment lengths along chain for joints [N, J, 3] -> [N]."""
    if len(chain) < 2:
        return torch.zeros(j3d.shape[0], device=j3d.device)
    pts = j3d[:, chain]                                   # [N, C, 3]
    return (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(-1)


def skeletal_height(j3d, names):
    """
    Head-joint (skull base) to mean-ankle distance, frame independent. [N]
    This is a SKELETAL span, ~25 cm short of stature (no skull above the atlas,
    no foot below the ankle); use it only to compare GT with prediction under
    one definition, never as a stature in metres.
    """
    head = [i for i, n in enumerate(names) if n.lower() == 'head']
    feet = feet_indices(names)
    if not head or not feet:
        # fallback: largest extent of the joint cloud
        return (j3d.max(1).values - j3d.min(1).values).norm(dim=-1)
    return (j3d[:, head[0]] - j3d[:, feet].mean(1)).norm(dim=-1)


body_height = skeletal_height   # backwards-compatible alias


def _neck_axis(j3d, names):
    """Unit vector neck01 -> head, [N,3], and the head joint position [N,3]."""
    head = [i for i, n in enumerate(names) if n.lower() == 'head']
    chain = neck_chain(names)
    hj = j3d[:, head[0]] if head else j3d[:, 0]
    base = j3d[:, chain[0]] if len(chain) >= 2 else j3d[:, 0]
    u = hj - base
    return u / torch.clamp(u.norm(dim=-1, keepdim=True), min=1e-6), hj


def head_neck_vertex_mask(v_rest, j_rest, names, radius=0.40, below=0.02):
    """
    Bool [N,V]: vertices of the head+neck SURFACE, selected on a REST-POSE mesh
    (A-pose, hands at the thighs), where a sphere around the head joint cannot
    contain hands or forearms. Mesh topology is fixed, so the same vertex
    indices address the head/neck on the posed mesh of the same shape.
    Rule: signed height above neck01 along the neck axis > -below, and within
    `radius` of the head joint.
    """
    chain = neck_chain(names)
    u, hj = _neck_axis(j_rest, names)
    base = j_rest[:, chain[0]] if len(chain) >= 2 else j_rest[:, 0]
    h_along = ((v_rest - base.unsqueeze(1)) * u.unsqueeze(1)).sum(-1)
    return (h_along > -below) & ((v_rest - hj.unsqueeze(1)).norm(dim=-1) < radius)


def neck_skin_vertex_mask(bm, names, min_weight=0.5):
    """
    Bool [V]: mesh-surface vertices (referenced by a face; anny keeps unattached helper geometry when built with
    remove_unattached_vertices=False) whose summed LBS weight on the neck bones neck01..neckNN is >= min_weight.
    The head and its subtree are excluded, so unlike head_neck_vertex_mask (~91% head vertices) this set sees only
    the neck skin, where a compensated local twist between adjacent neck bones shows (candy-wrapper thinning).
    Returns None when the body model exposes no vertex_bone_weights / vertex_bone_indices.
    """
    w = getattr(bm, 'vertex_bone_weights', None)
    ix = getattr(bm, 'vertex_bone_indices', None)
    if w is None or ix is None:
        return None
    necks = [i for i in neck_chain(names) if names[i].lower().startswith('neck')]
    sel = torch.zeros(len(names), dtype=torch.bool, device=w.device)
    sel[necks] = True
    wn = (w * sel[ix.long()].to(w.dtype)).sum(-1)                              # [V]
    on_face = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
    on_face[bm.faces.reshape(-1).long().to(w.device)] = True
    return (wn >= min_weight) & on_face


def neck_girth(v3d, j3d, chain, vmask):
    """Mean distance (m) of the vmask vertices [V] from the line through the chain[0] (neck01) and chain[-1] (head)
    joints, [N]: the neck's radius. pred/GT < 1 = a thinner neck skin than GT."""
    a, b = j3d[:, chain[0]], j3d[:, chain[-1]]
    u = (b - a) / torch.clamp((b - a).norm(dim=-1, keepdim=True), min=1e-9)
    rel = v3d[:, vmask] - a.unsqueeze(1)                                       # [N, M, 3]
    radial = rel - (rel * u.unsqueeze(1)).sum(-1, keepdim=True) * u.unsqueeze(1)
    return radial.norm(dim=-1).mean(-1)


def head_size(v3d, j3d, names, vmask=None):
    """
    Head height: head-bone origin -> crown along the (posed) neck axis, over
    the head/neck vertices given by `vmask` [N,V] (from head_neck_vertex_mask on
    a rest-pose mesh). Without a mask the 40 cm sphere on the posed mesh is used,
    which a hand raised to the head can contaminate. [N]
    """
    u, hj = _neck_axis(j3d, names)
    rel = v3d - hj.unsqueeze(1)                                   # [N,V,3]
    p = (rel * u.unsqueeze(1)).sum(-1)                            # signed height above head origin
    sel = vmask if vmask is not None else (rel.norm(dim=-1) < 0.40)
    p = torch.where(sel, p, torch.full_like(p, -1.0))
    return torch.clamp(p.max(1).values, min=0.0)


def visible_gt_mask(gt_j, K, img_size, patch_size):
    """
    Replicates the GT filter in train.py prepare_gt (lines ~250-277): keep a
    human only if its ROOT joint is in front of the camera (z > 0.1) and projects
    inside the image, and drop a second human whose root lands in an already
    occupied patch cell. gt_j [N,163,3], K [1,3,3]. Returns a bool mask [N].
    """
    root = gt_j[:, 0:1]                                           # [N,1,3]
    loc = project(root, K)[:, 0]                                  # [N,2]
    in_front = root[:, 0, 2] > 0.1
    in_img = (loc[:, 0] >= 0) & (loc[:, 0] < img_size) & (loc[:, 1] >= 0) & (loc[:, 1] < img_size)
    keep = (in_front & in_img).clone()
    n_patch = img_size // patch_size
    cell = torch.clamp((loc // patch_size).long(), 0, n_patch - 1)
    seen = set()
    for k in range(gt_j.shape[0]):
        if not bool(keep[k]):
            continue
        c = (int(cell[k, 0]), int(cell[k, 1]))
        if c in seen:
            keep[k] = False
        else:
            seen.add(c)
    return keep


# --------------------------------------------------------------------------- #
# age scale: read the body model's anchors instead of asserting a convention
# --------------------------------------------------------------------------- #
# anny >= 0.6 interpolates 'age' over anchors (default linspace(-1/3, 1, 5)) for the
# MakeHuman targets newborn / baby / child / young / old (0 / 1 / 11 / 25 / 90 years).
# The MakeHuman slider itself uses 0.1875 = 11 y and 0.5 = 25 y; the two scales differ.
MH_TARGET_YEARS = [0.0, 1.0, 11.0, 25.0, 90.0]
ANNY_DEFAULT_AGE_ANCHORS = [-1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]


def age_anchors(bm=None):
    """(anchor positions, source string). Reads bm.anchors['age'] when the installed anny exposes it."""
    if bm is not None:
        try:
            anc = getattr(bm, 'anchors', None)
            if anc is not None and 'age' in anc:
                vals = [float(x) for x in torch.as_tensor(anc['age']).reshape(-1).tolist()]
                if len(vals) == len(MH_TARGET_YEARS):
                    return vals, 'bm.anchors[age]'
                return vals, f'bm.anchors[age] (len {len(vals)} != 5; years not derivable)'
        except Exception:  # noqa
            pass
    return list(ANNY_DEFAULT_AGE_ANCHORS), 'ASSUMED anny default linspace(-1/3, 1, 5); verify against the installed anny'


def age_to_years(a, anchors=None):
    """Piecewise-linear MakeHuman target age for an anny 'age' value. None if the anchors are unusable."""
    anc = anchors if anchors is not None else ANNY_DEFAULT_AGE_ANCHORS
    if len(anc) != len(MH_TARGET_YEARS):
        return None
    a = float(a)
    xs, ys = list(anc), MH_TARGET_YEARS
    if a <= xs[0]:
        return ys[0]
    for i in range(len(xs) - 1):
        if a <= xs[i + 1]:
            t = (a - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def age_scale_text(bm=None):
    anc, src = age_anchors(bm)
    pts = ", ".join(f"{v:.2f}->{age_to_years(v, anc):.0f}y" for v in (0.0, 0.2, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)
                    if age_to_years(v, anc) is not None)
    return f"age scale [{src}]: {pts}"


def mean_or_none(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def json_safe(o):
    """Recursively turn NaN/inf and numpy scalars into JSON-valid values."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    return o


# --------------------------------------------------------------------------- #
# camera helpers
# --------------------------------------------------------------------------- #
def fov_from_K(K, img_size):
    """The convention anyone_image_export.py used: fov = 2 atan((2 cx) / (2 fx))."""
    fx = float(K[0, 0]); cx = float(K[0, 2])
    return math.degrees(2 * math.atan((2 * cx) / (2 * fx)))


def K_from_fov(fov_deg, img_size, device):
    """demo.get_camera_parameters(): square image, centred principal point."""
    f = img_size / (2 * math.tan(math.radians(fov_deg) / 2))
    K = torch.eye(3, device=device)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = K[1, 2] = img_size // 2
    return K.unsqueeze(0)


def project(x3d, K):
    """[N, P, 3] with K [1,3,3] or [N,3,3] -> [N, P, 2]"""
    from utils import perspective_projection
    if K.shape[0] == 1 and x3d.shape[0] != 1:
        K = K.expand(x3d.shape[0], 3, 3)
    return perspective_projection(x3d, K)


# --------------------------------------------------------------------------- #
# rendering (pyrender; falls back to None when EGL is unavailable)
# --------------------------------------------------------------------------- #
def faces_tri(bm):
    faces = bm.faces.detach().cpu().numpy().astype(np.int64)
    if faces.shape[1] == 4:
        faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], 0)
    return faces


_FIXED_FACES = None


def render_overlay(img_u8, verts_list, faces, K, colors=None, alpha=0.85):
    """Camera-frame vertices (metres, OpenCV axes) over an HxWx3 uint8 image."""
    global _FIXED_FACES
    try:
        import pyrender, trimesh
    except Exception as e:  # noqa
        print(f"[render] pyrender unavailable ({e}); overlays skipped")
        return None
    try:
        h, w = img_u8.shape[:2]
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
        if colors is None:
            colors = [(0.25, 0.65, 0.95), (0.95, 0.45, 0.25), (0.35, 0.85, 0.45), (0.85, 0.35, 0.85)]
        for i, v in enumerate(verts_list):
            v = np.asarray(v, dtype=np.float32)
            if not np.isfinite(v).all():
                continue
            if _FIXED_FACES is None:
                tm = trimesh.Trimesh(vertices=v, faces=faces, process=False)
                trimesh.repair.fix_normals(tm)
                _FIXED_FACES = tm.faces.copy()
            tm = trimesh.Trimesh(vertices=v, faces=_FIXED_FACES, process=False)
            c = colors[i % len(colors)]
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[c[0], c[1], c[2], 1.0],
                                                     metallicFactor=0.0, roughnessFactor=0.8, doubleSided=True)
            scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))
        K = np.asarray(K).reshape(3, 3)
        cam = pyrender.IntrinsicsCamera(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                                        znear=0.01, zfar=200.0)
        pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()
        a = (rgba[:, :, 3:4] / 255.0).astype(np.float32) * alpha
        out = rgba[:, :, :3].astype(np.float32) * a + img_u8.astype(np.float32) * (1 - a)
        return out.astype(np.uint8)
    except Exception as e:  # noqa
        print(f"[render] failed ({type(e).__name__}: {e}); overlays skipped")
        return None


def draw_points(img_u8, pts2d, color=(255, 0, 0), r=2):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_u8.copy())
    d = ImageDraw.Draw(im)
    for p in np.asarray(pts2d).reshape(-1, 2):
        if np.isfinite(p).all():
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    return np.asarray(im)


def label(img_u8, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_u8.copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def hstack(imgs, pad=4):
    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        if i.shape[0] != h:
            canvas = np.zeros((h, i.shape[1], 3), dtype=np.uint8)
            canvas[:i.shape[0]] = i
            i = canvas
        out.append(i)
        out.append(np.full((h, pad, 3), 255, dtype=np.uint8))
    return np.concatenate(out[:-1], 1)


def fmt(x, nd=1, sign=False):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return '   -'
        return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    except Exception:  # noqa
        return str(x)
