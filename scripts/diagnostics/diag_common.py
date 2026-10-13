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


def load_model(ckpt_path, device, tf32=True):
    """Same construction as demo.load_model(), for model.py's Model only.
    tf32=True mirrors train.py (--tf32 default 1): evaluate()'s 'fp32' numbers used TF32 matmuls."""
    from model import Model
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = dict(vars(ckpt['args']))
    kwargs.setdefault('simple_depth_encoding', 1)
    kwargs.setdefault('pose_param', 'local-bone')
    kwargs['person_center'] = 'head'      # demo.py does the same override
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
    print(f"[load_model] img_size={model.img_size} backbone_dim={model.backbone_dim} "
          f"feat_proj={'yes' if getattr(model, 'feat_proj', None) is not None else 'no'} "
          f"ckpt_args: backbone={kwargs.get('backbone')} head_dim={kwargs.get('head_dim')} "
          f"mask_helper_joints_train={kwargs.get('mask_helper_joints_train')} "
          f"use_anny_shape={kwargs.get('use_anny_shape')} alpha_shape={kwargs.get('alpha_shape')}")
    return model, kwargs


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
