"""
Body-model probe: no network, no checkpoint, no dataset images. Only the Anny
rig, built exactly as model.py builds it:
    anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float()
    .set_skinning_method('lbs')
and driven exactly as model.py drives it (bm(pose_parameters=[N,163,4,4],
phenotype_kwargs=...) with anny's DEFAULT pose parameterization, which demo.py
calls 'local-bone').

CHECK 2: canonical pose, age sweep.
    Rest pose (identity 4x4 on all 163 bones) at the dataset-mean shape
    (mean over results/diagnostics/phenotypes/all_shapes.npy if it exists, else
    all 0.5), sweeping only 'age' over 0.0 .. 1.0. Prints skeletal height, mesh
    height (stature), neck-chain length, head size and the neck/height and
    head/height ratios, and renders the 9 meshes side by side with one fixed
    synthetic camera (vis/age_sweep.jpg), so relative size is visible.

CHECK 3: single-joint perturbations.
    Rest pose at the mean shape; for each target bone (neck01..03, head, a
    proximal phalanx per side, a thumb bone, wrist.L; found by NAME) one bone's
    4x4 is set to a 30 deg rotation about the x and, separately, the y axis of
    its pose-parameter frame. Reports
      (a) vertices moved > 2 mm, attributed to their nearest REST joint
          (torch.cdist): top-5 joints and the fraction inside the expected set;
      (b) joints moved > 2 mm: in a parent-relative rig the rotated bone's own
          head is the pivot (does not move) and exactly its descendants move,
          rigidly, by one world rotation W. W is read off the bone's world frame
          (bone_poses rotation, perturbed x rest^T) and expressed in the world
          and in the bone's rest frame, which says whether the 4x4 is applied in
          the bone's local frame (local-bone) or in world-aligned axes;
      (c) rest (blue) vs perturbed (orange, translucent) renders:
          vis/perturb_<bone>_<axis>.jpg = [full body | zoom rest | zoom front | zoom side].
    Also lists the 88 helper bones masked by model.py's useful_rotmat (parsed
    from the model.py source at runtime, compared with the copy in
    scripts/analysis/check_mask_boost.py) and which neck/finger bones are
    masked vs active.

Output: --out (default results/diagnostics/bodymodel): summary.json + vis/.

USAGE (inside the multihmr env; GPU optional, renders need EGL):
    python scripts/diagnostics/diag_bodymodel_probe.py --out results/diagnostics/bodymodel
    sbatch slurm/diagnostics/diag_bodymodel.sh
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

AGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
AGE_CONVENTION = ("MakeHuman age convention, inherited by Anny (see scripts/data/smpl_to_anny.py ~l.140-150): "
                  "age 0.0 ~ 1 y, 0.1875 ~ 11 y, 0.5 = 25 y, 1.0 = 90 y (piecewise linear in between; "
                  "so 0.1 ~ 6 y, 0.2 ~ 12 y, 0.3 ~ 16 y, 0.4 ~ 20 y)")
IMG = 672                 # synthetic image, square
FOCAL = 800.0             # px
DIST = 3.5                # m, root joint straight ahead of the camera
REST_C = (0.25, 0.65, 0.95)
PERT_C = (0.95, 0.45, 0.25)
BG = 235
# Face-bone name fragments (MakeHuman default rig). Only used to FLAG members of
# dc.bone_groups()['face_other'] (a catch-all group) that do not look like face bones.
FACE_KEYS = ('jaw', 'eye', 'lid', 'oris', 'tongue', 'special', 'levator', 'orbicularis', 'oculi',
             'temporalis', 'risorius', 'brow', 'cheek', 'lip', 'nose', 'chin', 'mouth', 'teeth', 'face',
             'mandible')
HAND_KEYS = ('finger', 'thumb', 'metacarpal', 'wrist', 'hand', 'palm')

_RENDER_OK = True


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def age_to_years(a):
    a = float(a)
    if a <= 0.1875:
        return 1.0 + a / 0.1875 * 10.0
    if a <= 0.5:
        return 11.0 + (a - 0.1875) / 0.3125 * 14.0
    return 25.0 + (a - 0.5) / 0.5 * 65.0


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def side_of(name):
    lo = name.lower()
    if lo.endswith(('.l', '_l', '-l')) or '.l.' in lo or 'left' in lo:
        return 'L'
    if lo.endswith(('.r', '_r', '-r')) or '.r.' in lo or 'right' in lo:
        return 'R'
    return None


def finger_parts(name):
    """'finger2-1.L' -> ('finger2', 1, '.l'); None if the name is not <word><digits>-<digits><suffix>."""
    m = re.match(r'^([a-z]+\d+)-(\d+)(.*)$', name.lower())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def axis_rotation(axis, deg, device):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        M = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'y':
        M = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        M = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return torch.tensor(M, dtype=torch.float32, device=device)


def axis_angle(W):
    """Rotation matrix (3x3 numpy) -> (angle_deg, unit axis)."""
    c = float(np.clip((np.trace(W) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.degrees(math.acos(c))
    v = np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])
    nv = float(np.linalg.norm(v))
    axis = v / nv if nv > 1e-9 else np.full(3, np.nan)
    return ang, axis


def vec(v, nd=3):
    return [round(float(x), nd) for x in np.asarray(v).reshape(-1)]


def short_list(lst, n=25):
    lst = list(lst)
    return ", ".join(lst[:n]) + (f" ... (+{len(lst) - n} more)" if len(lst) > n else "")


def pick_device(req):
    if str(req).startswith('cuda') and not torch.cuda.is_available():
        print("[device] cuda requested but unavailable -> cpu")
        return torch.device('cpu')
    return torch.device(req)


def build_body_model(device):
    """Exactly model.py's construction (Model.__init__), then moved to the device."""
    import anny
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float()
    bm.set_skinning_method('lbs')
    bm = bm.to(device)
    if hasattr(bm, 'eval'):
        bm.eval()
    return bm


def default_pose_parameterization(bm):
    """Best effort: the default of the pose-parameterization kwarg model.py never passes."""
    try:
        import inspect
        p = inspect.signature(bm.forward).parameters
        for k in ('pose_parameterization', 'pose_param', 'parameterization'):
            if k in p and p[k].default is not inspect.Parameter.empty:
                return f"{k}={p[k].default!r} (anny default; model.py passes no such kwarg)"
        return "no pose-parameterization kwarg in bm.forward; arguments: " + ", ".join(p)
    except Exception as e:  # noqa
        return f"unknown ({type(e).__name__}: {e})"


def bone_rotations(bm, pose44, shape):
    """Fallback for older diag_common without run_body_model_full. Same call as dc.run_body_model, returning the rotation part of bone_poses [N,J,3,3] (world bone frames)."""
    labels = list(bm.phenotype_labels)
    shape = torch.clamp(shape, 0.0, 1.0)
    kw = {k: shape[:, i] for i, k in enumerate(labels)}
    try:
        with torch.no_grad():
            out = bm(pose_parameters=pose44, phenotype_kwargs=kw)
        return out['bone_poses'][:, :, :3, :3]
    except Exception as e:  # noqa
        print(f"[frames] bone_poses rotations unavailable ({type(e).__name__}: {e}); W analysis skipped")
        return None


def bone_parents(bm, n):
    """Best effort: a parent-index array if the anny object exposes one (used only as an extra check)."""
    for attr in ('bone_parents', 'parents', 'bone_parent_indices', 'parent_indices', 'bone_parent'):
        p = getattr(bm, attr, None)
        if p is None or callable(p):
            continue
        try:
            arr = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
            arr = arr.reshape(-1).astype(np.int64)
            if arr.shape[0] == n and bool(((arr >= -1) & (arr < n)).all()):
                return attr, arr
        except Exception:  # noqa
            continue
    return None, None


def descendants(parents, b):
    kids = {}
    for i, p in enumerate(parents.tolist()):
        if 0 <= p != i:
            kids.setdefault(p, []).append(i)
    out, stack = [], list(kids.get(b, []))
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(kids.get(i, []))
    return sorted(set(out))


def load_mean_shape(path, labels, device):
    """Mean phenotype vector in bm.phenotype_labels order, [1, L]. all_shapes.npy is in dataset order (dc.PHENOTYPES)."""
    vals = {k: 0.5 for k in labels}
    src = f"all 0.5 ({path} not found)"
    cands = [path] if os.path.isabs(path) else [path, os.path.join(dc.ROOT, path)]
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            P = np.asarray(np.load(p), dtype=np.float64)
            P = P.reshape(P.shape[0], -1) if P.ndim >= 1 and P.shape[0] else P
            if P.ndim != 2 or P.shape[0] == 0:
                src = f"all 0.5 ({p} is empty)"
                break
            m = P.mean(0)
            if P.shape[1] == len(dc.PHENOTYPES):
                order = dc.PHENOTYPES
            elif P.shape[1] == len(labels):
                order = labels
                print(f"[shape] WARNING: {p} has {P.shape[1]} columns != {len(dc.PHENOTYPES)}; assuming bm order")
            else:
                src = f"all 0.5 ({p} has {P.shape[1]} columns)"
                break
            for k in labels:
                if k in order:
                    vals[k] = float(m[order.index(k)])
                else:
                    print(f"[shape] WARNING: phenotype '{k}' not in {p} columns; kept 0.5")
            src = f"dataset mean over {P.shape[0]} people ({p})"
        except Exception as e:  # noqa
            src = f"all 0.5 (could not read {p}: {e})"
        break
    if list(labels) != list(dc.PHENOTYPES):
        print(f"[shape] NOTE: bm.phenotype_labels {list(labels)} differs from dataset order {dc.PHENOTYPES}; "
              f"values were mapped by NAME")
    t = torch.tensor([[vals[k] for k in labels]], dtype=torch.float32, device=device)
    return t, src, vals


# --------------------------------------------------------------------------- #
# camera + rendering
# --------------------------------------------------------------------------- #
def synthetic_K():
    return np.array([[FOCAL, 0.0, IMG / 2.0], [0.0, FOCAL, IMG / 2.0], [0.0, 0.0, 1.0]])


def upright_transform(j_rest, names):
    """
    Rotation model-frame -> OpenCV camera frame (+x right, +y down, +z forward) that
    puts the rest-pose body upright and facing the camera. Up = dominant axis of
    (head joint - root joint); left = dominant remaining axis of mean(.L joints) -
    mean(.R joints); forward = left x up. Camera images: left -> +x (the person's left
    appears on the image right when facing the camera), up -> -y, forward -> -z.
    """
    j = _np(j_rest)
    head = [i for i, n in enumerate(names) if n.lower() == 'head']
    hi = head[0] if head else int(np.argmax(np.linalg.norm(j - j[0], axis=1)))
    up_raw = j[hi] - j[0]
    ua = int(np.argmax(np.abs(up_raw)))
    u = np.zeros(3)
    u[ua] = 1.0 if up_raw[ua] >= 0 else -1.0
    L = [i for i, n in enumerate(names) if side_of(n) == 'L']
    R = [i for i, n in enumerate(names) if side_of(n) == 'R']
    info = dict(head_minus_root_m=vec(up_raw), up_axis=('+' if u[ua] > 0 else '-') + 'xyz'[ua])
    if L and R:
        lat_raw = j[L].mean(0) - j[R].mean(0)
        lat = lat_raw.copy()
        lat[ua] = 0.0
        la = int(np.argmax(np.abs(lat)))
        ls = 1.0 if lat[la] >= 0 else -1.0
        info['left_minus_right_m'] = vec(lat_raw)
        info['facing_from'] = 'mean(.L joints) - mean(.R joints)'
    else:
        la, ls = (ua + 1) % 3, 1.0
        info['left_minus_right_m'] = None
        info['facing_from'] = 'FALLBACK (no .L/.R names): arbitrary lateral axis, the body may face away'
    lft = np.zeros(3)
    lft[la] = ls
    fwd = np.cross(lft, u)
    B = np.stack([lft, u, fwd], 1)                                    # columns: left, up, forward (model frame)
    C = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # their camera images (columns)
    Rc = C @ B.T
    info['left_axis'] = ('+' if ls > 0 else '-') + 'xyz'[la]
    info['forward_axis'] = vec(fwd, 1)
    info['R_model_to_cam'] = Rc.tolist()
    return Rc, u, info


def to_cam(v, root, Rc, dist=DIST):
    """[P,3] model frame -> camera frame with the root joint at (0, 0, dist)."""
    return (_np(v) - _np(root).reshape(1, 3)) @ Rc.T + np.array([0.0, 0.0, dist])


def project_np(x, K):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    x = x[x[:, 2] > 0.01]
    if len(x) == 0:
        return np.zeros((0, 2))
    p = x @ K.T
    return p[:, :2] / p[:, 2:3]


def bg():
    return np.full((IMG, IMG, 3), BG, dtype=np.uint8)


def _render(img, verts_list, faces, K, colors, alpha):
    """dc.render_overlay, disabled after its first failure (EGL failures are persistent)."""
    global _RENDER_OK
    if not _RENDER_OK:
        return None
    out = dc.render_overlay(img, verts_list, faces, K, colors=colors, alpha=alpha)
    if out is None:
        _RENDER_OK = False
        print("[render] mesh rendering disabled for the rest of the run; panels fall back to vertex scatter")
    return out


def scatter(img, v, K, c01, max_pts=4000):
    """Fallback when pyrender/EGL is unavailable: on-screen projected vertices, subsampled."""
    col = tuple(int(255 * x) for x in c01)
    uv = project_np(v, K)
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < IMG) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG)]
    if len(uv) > max_pts:
        uv = uv[::int(math.ceil(len(uv) / max_pts))]
    return dc.draw_points(img, uv, color=col, r=1)


def add_points(img, K, red=None, green=None):
    if red is not None and len(red):
        img = dc.draw_points(img, project_np(red, K), color=(220, 0, 0), r=3)
    if green is not None and len(green):
        img = dc.draw_points(img, project_np(green, K), color=(0, 170, 0), r=5)
    return img


def mesh_panel(verts_list, faces, K, colors, title, red=None, green=None):
    img = _render(bg(), verts_list, faces, K, colors, 0.9)
    if img is None:
        img = bg()
        for v, c in zip(verts_list, colors):
            img = scatter(img, v, K, c)
    return dc.label(add_points(img, K, red, green), title)


def ghost_panel(v_rest, v_pert, faces, K, title, red=None, green=None):
    a = _render(bg(), [v_rest], faces, K, [REST_C], 0.9)
    if a is None:
        img = scatter(scatter(bg(), v_rest, K, REST_C), v_pert, K, PERT_C)
    else:
        b = _render(a, [v_pert], faces, K, [PERT_C], 0.5)
        img = b if b is not None else a
    return dc.label(add_points(img, K, red, green), title)


def save_jpg(arr, path):
    try:
        Image.fromarray(arr).save(path, quality=90)
        return True
    except Exception as e:  # noqa
        print(f"[vis] could not save {path}: {e}")
        return False


# --------------------------------------------------------------------------- #
# CHECK 2: age sweep
# --------------------------------------------------------------------------- #
def check_age_sweep(bm, names, phen, base_shape, Rc, up, faces, K, device, args):
    print("\n" + "=" * 78)
    print("CHECK 2: canonical (rest) pose, dataset-mean shape, age sweep")
    print("=" * 78)
    print(AGE_CONVENTION)
    if 'age' not in phen:
        print(f"ERROR: no 'age' phenotype in bm.phenotype_labels {phen}; age sweep skipped")
        return None
    n = len(AGES)
    shp = base_shape.repeat(n, 1).clone()
    shp[:, phen.index('age')] = torch.tensor(AGES, dtype=torch.float32, device=device)
    v, j = dc.run_body_model(bm, dc.identity_pose(n, len(names), device), shp)
    chain = dc.neck_chain(names)
    print(f"neck chain (dc.neck_chain): {[names[i] for i in chain]}")
    height = dc.skeletal_height(j, names)
    neck = dc.chain_length(j, chain)
    vmask = dc.head_neck_vertex_mask(v, j, names)
    head = dc.head_size(v, j, names, vmask)
    up_t = torch.tensor(up, dtype=v.dtype, device=v.device)
    proj = (v * up_t).sum(-1)                                           # [N,V] height along the up axis
    mesh_h = proj.max(1).values - proj.min(1).values                    # crown-to-sole stature
    print("    skel_H = head joint -> mean ankle (~25 cm short of stature); mesh_H = vertex extent along the up axis")
    print("    head = head joint -> crown along the neck axis (head+neck surface vertices)")
    print(f"{'age':>5}{'~years':>8}{'skel_H_m':>10}{'mesh_H_m':>10}{'neck_m':>9}{'head_m':>9}"
          f"{'neck/skelH':>12}{'head/skelH':>12}{'head/meshH':>12}")
    rows = []
    for i, a in enumerate(AGES):
        h, mh, nk, hd = float(height[i]), float(mesh_h[i]), float(neck[i]), float(head[i])
        r = dict(age=a, approx_years=age_to_years(a), skel_height_m=h, mesh_height_m=mh, neck_m=nk, head_m=hd,
                 neck_over_height=nk / h if h > 0 else None, head_over_height=hd / h if h > 0 else None,
                 head_over_mesh_height=hd / mh if mh > 0 else None)
        rows.append(r)
        print(f"{a:>5.2f}{r['approx_years']:>8.1f}{dc.fmt(h, 3):>10}{dc.fmt(mh, 3):>10}{dc.fmt(nk, 4):>9}"
              f"{dc.fmt(hd, 4):>9}{dc.fmt(r['neck_over_height'], 4):>12}{dc.fmt(r['head_over_height'], 4):>12}"
              f"{dc.fmt(r['head_over_mesh_height'], 4):>12}")
    print("READ: a child has a SMALLER stature and a LARGER head/height (newborn ~1/4, adult ~1/8 of stature")
    print("      chin-to-crown; 'head' here starts at the skull base, so expect ~0.6x those ratios).")
    print("      If both move monotonically as age -> 0, Anny represents children through 'age' alone.")
    vis_path = None
    if args.render:
        panels = []
        for i, a in enumerate(AGES):
            vc = to_cam(v[i], j[i, 0], Rc)
            title = f"age {a:.2f} ~{age_to_years(a):.0f}y  meshH {rows[i]['mesh_height_m']:.2f}m  " \
                    f"head/skelH {dc.fmt(rows[i]['head_over_height'], 3)}"
            panels.append(mesh_panel([vc], faces, K, [REST_C], title))
        vis_path = os.path.join(args.out, 'vis', 'age_sweep.jpg')
        if save_jpg(dc.hstack(panels), vis_path):
            print(f"[vis] wrote {vis_path} (one camera for all panels: f={FOCAL:.0f}px, {IMG}x{IMG}, "
                  f"root {DIST} m ahead)")
    return dict(convention=AGE_CONVENTION, rows=rows, vis=vis_path,
                neck_chain=[names[i] for i in chain])


# --------------------------------------------------------------------------- #
# CHECK 3: single-joint perturbations
# --------------------------------------------------------------------------- #
def find_targets(names):
    lo = [n.lower() for n in names]
    t, notes = {}, {}

    def exact_or_sub(key):
        if key in lo:
            return lo.index(key)
        sub = [i for i, n in enumerate(lo) if key in n]
        if sub:
            notes[key] = f"no exact '{key}'; used first substring match '{names[sub[0]]}'"
            return sub[0]
        return None

    for key in ('neck01', 'neck02', 'neck03', 'head'):
        t[key] = exact_or_sub(key)
    for s in ('L', 'R'):
        c = [i for i, n in enumerate(lo) if n.startswith(('finger1-1', 'finger2-1')) and side_of(names[i]) == s]
        t[f'finger_prox.{s}'] = c[0] if c else None
    th = [i for i, n in enumerate(lo) if 'thumb' in n]
    if th:
        thl = [i for i in th if side_of(names[i]) == 'L']
        t['thumb'] = (thl or th)[0]
    else:
        used = {v for v in t.values() if v is not None}
        fb = [i for i, n in enumerate(lo) if n.startswith('finger1-1') and i not in used]
        fb.sort(key=lambda i: 0 if side_of(names[i]) == 'L' else 1)
        t['thumb'] = fb[0] if fb else None
        if fb:
            notes['thumb'] = f"no bone name contains 'thumb'; fallback '{names[fb[0]]}' (MakeHuman's thumb is finger1-*)"
        else:
            notes['thumb'] = ("no bone name contains 'thumb'; MakeHuman's thumb is finger1-*, already tested "
                              "as finger_prox.L/R")
            # finger_prox picked the thumbs: add an index-finger proximal phalanx so a long finger is covered
            ix = [i for i, n in enumerate(lo) if n.startswith('finger2-1') and i not in used]
            ix.sort(key=lambda i: 0 if side_of(names[i]) == 'L' else 1)
            if ix:
                t['index_prox'] = ix[0]
                notes['index_prox'] = f"extra target: '{names[ix[0]]}' (finger_prox.* were finger1-1 = thumb)"
    if 'wrist.l' in lo:
        t['wrist.L'] = lo.index('wrist.l')
    else:
        w = [i for i, n in enumerate(lo) if 'wrist' in n and side_of(names[i]) == 'L']
        t['wrist.L'] = w[0] if w else None
        if w:
            notes['wrist.L'] = f"no exact 'wrist.L'; used '{names[w[0]]}'"
    return t, notes


def expectations(names, b, groups, chain):
    lo = names[b].lower()
    side = side_of(names[b])
    head_idx = [i for i, n in enumerate(names) if n.lower() == 'head']
    face = sorted((set(groups.get('head', [])) | set(groups.get('face_other', []))) - set(head_idx))
    neck_face = sorted(set(groups.get('neck', [])) | set(groups.get('head', [])) | set(groups.get('face_other', [])))
    if b in chain:
        k = chain.index(b)
        return dict(kind='neck', expected=neck_face, children=chain[k + 1:], face=face,
                    rule="neck chain + head + face bones = dc.bone_groups 'neck' + 'head' (head, jaw) + 'face_other'")
    if 'wrist' in lo:
        hand = [i for i, n in enumerate(names)
                if side_of(n) == side and any(s in n.lower() for s in HAND_KEYS)]
        return dict(kind='wrist', expected=hand, children=[i for i in hand if i != b], face=[],
                    rule=f"hand bones of side {side}: name contains one of {list(HAND_KEYS)}")
    fp = finger_parts(names[b])
    if fp is not None:
        same = []
        for i, n in enumerate(names):
            q = finger_parts(n)
            if q is not None and q[0] == fp[0] and q[2] == fp[2]:
                same.append(i)
        children = [i for i in same if finger_parts(names[i])[1] > fp[1]]
        return dict(kind='finger', expected=same, children=children, face=[],
                    rule=f"bones of the same finger: '{fp[0]}-<k>{fp[2]}'")
    if 'thumb' in lo:
        same = [i for i, n in enumerate(names) if 'thumb' in n.lower() and side_of(n) == side]
        return dict(kind='thumb', expected=same, children=[i for i in same if i > b], face=[],
                    rule=f"bones of side {side} whose name contains 'thumb' (children = later in bone order)")
    return dict(kind='other', expected=[b], children=[], face=[], rule='no rule for this bone')


def check_perturbations(bm, names, base_shape, Rc, faces, K, device, args, mask):
    print("\n" + "=" * 78)
    print(f"CHECK 3: single-joint perturbations ({args.angle:.0f} deg about the pose-parameter x / y axis), "
          f"rest pose, mean shape")
    print("=" * 78)
    nb = len(names)
    thr = args.thresh_mm / 1000.0
    groups = dc.bone_groups(names)
    chain = dc.neck_chain(names)
    face_other = groups.get('face_other', [])
    unverified = [names[i] for i in face_other if not any(k in names[i].lower() for k in FACE_KEYS)]
    print(f"dc.bone_groups 'face_other' ({len(face_other)} bones, used as 'face'): "
          f"{short_list([names[i] for i in face_other], 200)}")
    if unverified:
        print(f"  WARNING: face_other members not matching a face-name fragment (check!): {unverified}")
    pattr, parents = bone_parents(bm, nb)
    print(f"[rig] parent array on the anny object: {pattr if pattr else 'not exposed (name-based expectations only)'}")

    targets, notes = find_targets(names)
    print("\ntarget bones found by name:")
    tinfo = {}
    for key, b in targets.items():
        if b is None:
            print(f"  {key:<14} NOT FOUND" + (f"  ({notes[key]})" if key in notes else ""))
            tinfo[key] = dict(name=None, index=None, note=notes.get(key, 'not found'))
            continue
        ex = expectations(names, b, groups, chain)
        msk = None if mask is None or b >= len(mask) else bool(mask[b] == 0.0)
        tinfo[key] = dict(name=names[b], index=b, kind=ex['kind'], rule=ex['rule'],
                          expected=[names[i] for i in ex['expected']],
                          children=[names[i] for i in ex['children']],
                          masked_by_useful_rotmat=msk, note=notes.get(key))
        print(f"  {key:<14} [{b:3d}] '{names[b]}'  kind={ex['kind']}  "
              f"helper-mask={'MASKED' if msk else ('active' if msk is not None else '?')}"
              + (f"  ({notes[key]})" if key in notes else ""))
        print(f"      expected set: {ex['rule']}")
        print(f"      expected set names ({len(ex['expected'])}): {short_list([names[i] for i in ex['expected']], 30)}")
        print(f"      expected moving children: {[names[i] for i in ex['children']]}"
              + (f" + face bones ({len(ex['face'])})" if ex['kind'] == 'neck' else ""))

    tests = []
    seen = set()
    for key, b in targets.items():
        if b is None or b in seen:
            continue
        seen.add(b)
        for ax in ('x', 'y'):
            tests.append((key, b, ax))
    if not tests:
        print("ERROR: no target bone found; CHECK 3 skipped")
        return dict(targets=tinfo, tests=[])

    N = len(tests) + 1
    pose = dc.identity_pose(N, nb, device)
    for t, (key, b, ax) in enumerate(tests):
        pose[t + 1, b, :3, :3] = axis_rotation(ax, args.angle, device)
    shp = base_shape.repeat(N, 1)
    if hasattr(dc, 'run_body_model_full'):
        v, j, bp = dc.run_body_model_full(bm, pose, shp)          # + global bone poses [N,163,4,4]
        frames = bp[:, :, :3, :3]
    else:
        v, j = dc.run_body_model(bm, pose, shp)
        frames = bone_rotations(bm, pose, shp)
    v0, j0 = v[0], j[0]
    j0n = _np(j0)
    print(f"\n[rig] body model forward: {N} meshes ({len(tests)} perturbations + rest), {v.shape[1]} vertices")

    # pass 1: moved joints per test (needed to find each bone's direction = nearest moving descendant)
    jd_all = (j[1:] - j0.unsqueeze(0)).norm(dim=-1).cpu().numpy()          # [T, J]
    moved_all = [[i for i in range(nb) if jd_all[t, i] > thr] for t in range(len(tests))]
    moved_union = {}
    for t, (key, b, ax) in enumerate(tests):
        moved_union.setdefault(b, set()).update(i for i in moved_all[t] if i != b)

    results = []
    for t, (key, b, ax) in enumerate(tests):
        name = names[b]
        ex = expectations(names, b, groups, chain)
        exp_set = set(ex['expected'])
        vt, jt = v[t + 1], j[t + 1]
        r = dict(target=key, bone=name, index=b, axis=ax, angle_deg=args.angle, kind=ex['kind'])

        # (a) affected vertices -> nearest rest joint
        disp = (vt - v0).norm(dim=-1)
        aff = disp > thr
        n_aff = int(aff.sum())
        r['n_affected_vertices'] = n_aff
        r['max_vertex_disp_mm'] = float(disp.max()) * 1000.0
        if n_aff:
            nearest = torch.cdist(v0[aff].unsqueeze(0), j0.unsqueeze(0))[0].argmin(-1)
            counts = torch.bincount(nearest, minlength=nb).cpu().numpy()
            order = np.argsort(-counts)[:5]
            r['top5_nearest_joints'] = [[names[i], int(counts[i]), float(counts[i]) / n_aff]
                                        for i in order if counts[i] > 0]
            r['frac_affected_in_expected'] = float(counts[sorted(exp_set)].sum()) / n_aff if exp_set else None
            out_names = {}
            for i in np.nonzero(counts)[0]:
                if int(i) not in exp_set:
                    out_names[names[int(i)]] = int(counts[i])
            r['affected_outside_expected'] = dict(sorted(out_names.items(), key=lambda kv: -kv[1])[:10])
        else:
            r['top5_nearest_joints'] = []
            r['frac_affected_in_expected'] = None
            r['affected_outside_expected'] = {}

        # (b) moved joints
        moved = moved_all[t]
        r['moved_joints'] = [names[i] for i in moved]
        r['pivot_moved'] = bool(jd_all[t, b] > thr)
        r['pivot_disp_mm'] = float(jd_all[t, b]) * 1000.0
        r['leaks_outside_expected'] = [names[i] for i in moved if i not in exp_set]

        # effective world rotation W of the bone (bone_poses: perturbed x rest^T)
        W = None
        if frames is not None:
            Rr = _np(frames[0, b])
            Rp = _np(frames[t + 1, b])
            W = Rp @ Rr.T
            ang, axw = axis_angle(W)
            axl = Rr.T @ axw if np.isfinite(axw).all() else np.full(3, np.nan)
            e = np.eye(3)['xyz'.index(ax)]
            rest_dev, _ = axis_angle(Rr)
            dot_w = float(axw @ e) if np.isfinite(axw).all() else float('nan')
            dot_l = float(axl @ e) if np.isfinite(axl).all() else float('nan')
            if rest_dev < 1.0:
                interp = 'indistinguishable (bone rest frame ~ identity: local axes = world axes)'
            elif abs(dot_l) > 0.99 and abs(dot_w) > 0.99:
                interp = 'both (this axis is shared by the bone rest frame and the world; use the other axis test)'
            elif abs(dot_l) > 0.99 and abs(dot_w) < 0.99:
                interp = 'bone-local (the 4x4 acts in the bone rest frame)'
            elif abs(dot_w) > 0.99 and abs(dot_l) < 0.99:
                interp = 'world-aligned (the 4x4 acts in world axes)'
            else:
                interp = 'neither cleanly (see axes)'
            r.update(effective_angle_deg=ang, axis_world=vec(axw), axis_in_bone_rest_frame=vec(axl),
                     dot_axis_world_param=dot_w, dot_axis_local_param=dot_l,
                     bone_rest_frame_vs_identity_deg=rest_dev, frame_interpretation=interp)
            # bone direction: head -> nearest descendant that moves in either axis test
            mu = sorted(moved_union.get(b, set()))
            if mu:
                c = mu[int(np.argmin(np.linalg.norm(j0n[mu] - j0n[b], axis=1)))]
                bd = j0n[c] - j0n[b]
                bd = bd / max(np.linalg.norm(bd), 1e-9)
                r['bone_dir_world'] = vec(bd)
                r['bone_dir_child'] = names[c]
                r['twist_cos_axis_vs_bone_dir'] = float(abs(axw @ bd)) if np.isfinite(axw).all() else None
                r['cos_rest_y_vs_bone_dir'] = float(abs(Rr[:, 1] @ bd))   # Blender convention: local y along the bone
            # rigid-FK consistency: every moved joint should sit at p + W (c0 - p)
            p = j0n[b]
            jtn = _np(jt)
            pred = (j0n - p) @ W.T + p
            pred_disp = np.linalg.norm(pred - j0n, axis=1)
            chk = sorted(set(moved) | set(ex['children']))
            chk = [i for i in chk if i != b]
            r['rigid_fk_max_err_mm'] = float(np.linalg.norm(jtn[chk] - pred[chk], axis=1).max()) * 1000.0 if chk else 0.0
            r['children_on_axis_static'] = [names[i] for i in ex['children'] if pred_disp[i] < thr]
            missing = [i for i in ex['children'] if i not in moved and pred_disp[i] >= thr]
            if ex['kind'] == 'neck':
                fexp = [i for i in ex['face'] if pred_disp[i] >= thr]
                r['face_bones_moved'] = int(sum(1 for i in ex['face'] if i in moved))
                r['face_bones_expected_to_move'] = len(fexp)
                r['face_bones_missing'] = [names[i] for i in fexp if i not in moved]
                missing += [i for i in fexp if i not in moved]
        else:
            missing = [i for i in ex['children'] if i not in moved]
            if ex['kind'] == 'neck':
                r['face_bones_moved'] = int(sum(1 for i in ex['face'] if i in moved))
        r['missing_children'] = [names[i] for i in sorted(set(missing))]

        if parents is not None:
            desc = set(descendants(parents, b))
            r['hier_non_descendant_moved'] = [names[i] for i in moved if i not in desc]
            r['hier_descendants_static'] = [names[i] for i in sorted(desc) if i not in moved]

        reasons = []
        if r['pivot_moved']:
            reasons.append(f"pivot moved {r['pivot_disp_mm']:.1f} mm")
        if r['leaks_outside_expected']:
            reasons.append(f"{len(r['leaks_outside_expected'])} joints outside the expected set moved")
        if r['missing_children']:
            reasons.append(f"{len(r['missing_children'])} expected children did not move")
        if r.get('rigid_fk_max_err_mm') is not None and r['rigid_fk_max_err_mm'] > 1.0:
            reasons.append(f"non-rigid child motion ({r['rigid_fk_max_err_mm']:.1f} mm)")
        if parents is not None and r['hier_non_descendant_moved']:
            reasons.append("non-descendants moved")
        if n_aff == 0:
            reasons.append("no vertex moved")
        r['verdict'] = 'PASS' if not reasons else 'CHECK: ' + '; '.join(reasons)

        # print
        print(f"\n--- {key}: '{name}' rot_{ax} {args.angle:.0f} deg ---")
        print(f"  (a) affected vertices (> {args.thresh_mm:.0f} mm): {n_aff}  (max {r['max_vertex_disp_mm']:.1f} mm)")
        if n_aff:
            print("      top-5 nearest rest joints: " +
                  ", ".join(f"{nm} {cnt} ({fr * 100:.0f}%)" for nm, cnt, fr in r['top5_nearest_joints']))
            print(f"      fraction in expected set: {dc.fmt(r['frac_affected_in_expected'], 3)}"
                  + (f"   outside: {r['affected_outside_expected']}" if r['affected_outside_expected'] else ""))
        print(f"  (b) moved joints ({len(moved)}): {short_list(r['moved_joints'])}")
        print(f"      pivot moved: {'YES' if r['pivot_moved'] else 'no'} ({r['pivot_disp_mm']:.2f} mm); "
              f"leaks: {r['leaks_outside_expected'] or 'none'}; missing children: {r['missing_children'] or 'none'}")
        if ex['kind'] == 'neck':
            print(f"      face bones moved: {r.get('face_bones_moved')}/{len(ex['face'])}"
                  + (f" (expected to move: {r['face_bones_expected_to_move']})" if 'face_bones_expected_to_move' in r else ""))
        if W is not None:
            print(f"      effective world rotation {r['effective_angle_deg']:.2f} deg about {r['axis_world']} (world), "
                  f"{r['axis_in_bone_rest_frame']} (bone rest frame; rest frame is "
                  f"{r['bone_rest_frame_vs_identity_deg']:.1f} deg from identity) -> {r['frame_interpretation']}")
            if 'bone_dir_world' in r:
                print(f"      bone direction (-> {r['bone_dir_child']}): {r['bone_dir_world']}; "
                      f"|cos(axis, bone)| = {dc.fmt(r['twist_cos_axis_vs_bone_dir'], 3)}"
                      f"{'  = TWIST about the bone' if (r['twist_cos_axis_vs_bone_dir'] or 0) > 0.9 else ''}; "
                      f"|cos(rest y, bone)| = {r['cos_rest_y_vs_bone_dir']:.3f}")
            print(f"      rigid-FK consistency: max |joint - (p + W(c-p))| = {r['rigid_fk_max_err_mm']:.3f} mm; "
                  f"children on the axis (not expected to move): {r['children_on_axis_static'] or 'none'}")
        if parents is not None:
            print(f"      hierarchy ({pattr}): non-descendants moved {r['hier_non_descendant_moved'] or 'none'}; "
                  f"static descendants {short_list(r['hier_descendants_static'], 10) or 'none'}")
        print(f"  verdict: {r['verdict']}")

        # (c) renders
        if args.render:
            try:
                root = j0[0]
                v0c, vtc = to_cam(v0, root, Rc), to_cam(vt, root, Rc)
                j0c, jtc = to_cam(j0, root, Rc), to_cam(jt, root, Rc)
                am = aff.cpu().numpy()
                if n_aff:
                    pts = np.concatenate([v0c[am], vtc[am]], 0)
                    c = v0c[am].mean(0)
                    rad = float(np.percentile(np.linalg.norm(pts - c, axis=1), 95))
                else:
                    c, rad = j0c[b], 0.10
                rad = max(rad, 0.02)
                d = max(FOCAL * rad / (0.8 * IMG / 2.0), 0.12)
                sd = 1.0 if c[0] >= 0 else -1.0          # look from the side the region is on

                def zoom(x, side=0.0):
                    y = np.asarray(x).reshape(-1, 3) - c
                    if side:
                        th = math.radians(90.0 * side)
                        Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
                        y = y @ Ry.T
                    return y + np.array([0.0, 0.0, d])

                tag = f"{name} rot_{ax} {args.angle:.0f}deg"
                panels = [ghost_panel(v0c, vtc, faces, K, f"{tag}: rest blue / perturbed orange",
                                      red=jtc[moved], green=j0c[[b]]),
                          mesh_panel([zoom(v0c)], faces, K, [REST_C], "zoom front: rest (green = pivot)",
                                     green=zoom(j0c[[b]])),
                          ghost_panel(zoom(v0c), zoom(vtc), faces, K, "zoom front: ghost (red = moved joints)",
                                      red=zoom(jtc[moved]), green=zoom(j0c[[b]])),
                          ghost_panel(zoom(v0c, sd), zoom(vtc, sd), faces, K, "zoom side: ghost",
                                      red=zoom(jtc[moved], sd), green=zoom(j0c[[b]], sd))]
                safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
                path = os.path.join(args.out, 'vis', f'perturb_{safe}_{ax}.jpg')
                if save_jpg(dc.hstack(panels), path):
                    r['vis'] = path
            except Exception as e:  # noqa
                print(f"  [vis] failed ({type(e).__name__}: {e})")
        results.append(r)

    # compact table
    print("\n" + "-" * 78)
    print(f"{'bone':<16}{'ax':>3}{'n_aff':>7}{'frac_exp':>9}{'n_moved':>8}{'leaks':>6}{'miss':>5}{'rigid_mm':>9}  frame / verdict")
    for r in results:
        print(f"{r['bone']:<16}{r['axis']:>3}{r['n_affected_vertices']:>7}{dc.fmt(r['frac_affected_in_expected'], 3):>9}"
              f"{len(r['moved_joints']):>8}{len(r['leaks_outside_expected']):>6}{len(r['missing_children']):>5}"
              f"{dc.fmt(r.get('rigid_fk_max_err_mm'), 2):>9}  {r.get('frame_interpretation', '?').split(' (')[0]} / "
              f"{r['verdict']}")
    interp = {}
    for r in results:
        k = r.get('frame_interpretation', 'unknown').split(' (')[0]
        interp[k] = interp.get(k, 0) + 1
    print(f"pose-parameter frame across tests: {interp}")
    print("READ: PASS on every row = parent-relative FK: the rotated bone pivots on its own head, exactly its")
    print("      children follow, rigidly. A y-axis test that leaves the direct child static is a TWIST about the")
    print("      bone (child on the axis), not a failure. 'bone-local' confirms the 4x4 is applied in the bone's")
    print("      rest frame (demo.py's 'local-bone'). frac_exp << 1 for neck01 means its LBS weights reach the")
    print("      shoulders/chest: rotating the neck deforms the torso surface.")
    return dict(targets=tinfo, tests=results, frame_interpretation_counts=interp,
                parents_attr=pattr, face_other_unverified=unverified,
                thresh_mm=args.thresh_mm, angle_deg=args.angle)


# --------------------------------------------------------------------------- #
# helper-bone mask (model.py useful_rotmat), read from the source text
# --------------------------------------------------------------------------- #
_NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def read_literal(path, pattern):
    try:
        with open(path) as f:
            t = f.read()
        m = re.search(pattern, t, re.S)
        if not m:
            return None
        return [float(x) for x in re.findall(_NUM, m.group(1))]
    except Exception as e:  # noqa
        print(f"[mask] could not read {path}: {e}")
        return None


def helper_mask_report(names):
    print("\n" + "=" * 78)
    print("HELPER-BONE MASK (model.py useful_rotmat; 1 = predicted rotation kept, 0 = forced to identity)")
    print("=" * 78)
    mp = os.path.join(dc.ROOT, 'model.py')
    cp = os.path.join(dc.ROOT, 'scripts', 'analysis', 'check_mask_boost.py')
    mask = read_literal(mp, r"_useful\s*=\s*torch\.Tensor\(\s*\[(.*?)\]\s*\)")
    ref = read_literal(cp, r"USEFUL_ROTMAT\s*=\s*\[(.*?)\]")
    if mask is None:
        print(f"ERROR: useful_rotmat literal not found in {mp}")
        return None, None
    nb = len(names)
    out = dict(source=mp, length=len(mask), n_bones=nb)
    if len(mask) != nb:
        print(f"WARNING: mask has {len(mask)} entries, rig has {nb} bones; compared on the first {min(nb, len(mask))}")
    out['identical_to_check_mask_boost'] = (ref == mask) if ref is not None else None
    print(f"parsed {len(mask)} entries from {mp}; identical to {cp}: {out['identical_to_check_mask_boost']}")
    n = min(nb, len(mask))
    masked = [i for i in range(n) if mask[i] == 0.0]
    active = [i for i in range(n) if mask[i] != 0.0]
    out['n_masked'] = len(masked)
    out['n_active'] = len(active)
    out['masked'] = [names[i] for i in masked]
    out['active'] = [names[i] for i in active]
    print(f"{len(masked)} MASKED / {len(active)} active bones")
    print("masked bones by name:")
    for k in range(0, len(masked), 6):
        print("   " + "  ".join(f"[{i:3d}] {names[i]:<16}" for i in masked[k:k + 6]))

    def is_neck(nm):            # same rule as scripts/analysis/check_mask_boost.py
        lo = nm.lower()
        return ('neck' in lo) or (lo == 'head')

    def is_hand(nm):            # same rule as scripts/analysis/check_mask_boost.py
        lo = nm.lower()
        return ('finger' in lo) or ('thumb' in lo) or ('wrist' in lo)

    for gname, fn in (('neck_head', is_neck), ('hand', is_hand),
                      ('metacarpal', lambda nm: 'metacarpal' in nm.lower())):
        idx = [i for i in range(n) if fn(names[i])]
        gm = [names[i] for i in idx if mask[i] == 0.0]
        ga = [names[i] for i in idx if mask[i] != 0.0]
        out[f'{gname}_masked'] = gm
        out[f'{gname}_active'] = ga
        print(f"{gname:<11}: {len(idx)} bones, {len(gm)} MASKED {gm}")
        print(f"{'':<11}  {len(ga)} active {short_list(ga, 40)}")
    print("NB: the mask acts only when model.mask_helper_joints (inference) or mask_helper_joints_train is on;")
    print("    a MASKED bone is rendered at identity and its rotation loss is zero under the training mask.")
    return out, mask


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/diagnostics/bodymodel')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shapes', default='results/diagnostics/phenotypes/all_shapes.npy',
                    help="all_shapes.npy from diag_phenotypes.py (dataset order); mean over rows. Missing -> all 0.5")
    ap.add_argument('--angle', type=float, default=30.0)
    ap.add_argument('--thresh_mm', type=float, default=2.0)
    ap.add_argument('--render', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    t0 = time.time()

    device = pick_device(args.device)
    try:
        bm = build_body_model(device)
    except Exception as e:  # noqa
        print(f"ERROR: could not build the Anny body model ({type(e).__name__}: {e}); run inside the multihmr env")
        sys.exit(1)
    names = list(bm.bone_labels)
    phen = list(bm.phenotype_labels)
    nb = len(names)
    faces = dc.faces_tri(bm)
    pp = default_pose_parameterization(bm)
    print(f"[rig] device={device}  {nb} bones, {len(phen)} phenotypes {phen}, {faces.shape[0]} triangles")
    print(f"[rig] pose parameterization: {pp}")
    print("[rig] bone_labels:")
    for k in range(0, nb, 8):
        print("   " + " ".join(f"{i:3d}:{names[i]:<14}" for i in range(k, min(nb, k + 8))))

    base, shape_src, shape_vals = load_mean_shape(args.shapes, phen, device)
    print(f"[shape] {shape_src}")
    print("[shape] " + "  ".join(f"{k}={v:.3f}" for k, v in shape_vals.items()))

    v_rest, j_rest = dc.run_body_model(bm, dc.identity_pose(1, nb, device), base)
    Rc, up, cam_info = upright_transform(j_rest[0], names)
    K = synthetic_K()
    cam_info.update(K=K.tolist(), image_size=IMG, focal_px=FOCAL, root_distance_m=DIST)
    print(f"[camera] rest-pose up axis {cam_info['up_axis']} (head - root = {cam_info['head_minus_root_m']} m), "
          f"left axis {cam_info['left_axis']} ({cam_info['facing_from']}), forward {cam_info['forward_axis']}")
    print(f"[camera] model -> OpenCV camera rotation {np.round(Rc, 3).tolist()}; root placed {DIST} m ahead, "
          f"f={FOCAL:.0f}px, {IMG}x{IMG}")

    summary = dict(device=str(device), n_bones=nb, n_vertices=int(v_rest.shape[1]), phenotype_labels=phen,
                   bone_labels=names, pose_parameterization=pp, shape_source=shape_src, mean_shape=shape_vals,
                   camera=cam_info)

    mask_out, mask = helper_mask_report(names)
    summary['helper_mask'] = mask_out

    try:
        summary['age_sweep'] = check_age_sweep(bm, names, phen, base, Rc, up, faces, K, device, args)
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        print(f"ERROR: CHECK 2 failed ({type(e).__name__}: {e})")
        summary['age_sweep'] = dict(error=f"{type(e).__name__}: {e}")

    try:
        summary['perturbations'] = check_perturbations(bm, names, base, Rc, faces, K, device, args, mask)
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        print(f"ERROR: CHECK 3 failed ({type(e).__name__}: {e})")
        summary['perturbations'] = dict(error=f"{type(e).__name__}: {e}")

    summary['renders_ok'] = bool(_RENDER_OK) if args.render else None
    path = os.path.join(args.out, 'summary.json')
    with open(path, 'w') as f:
        json.dump(dc.json_safe(summary), f, indent=2, allow_nan=False)
    print(f"\nwrote {path} and {os.path.join(args.out, 'vis')}/  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
