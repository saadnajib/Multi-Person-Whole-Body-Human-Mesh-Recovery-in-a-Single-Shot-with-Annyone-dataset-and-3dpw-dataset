"""
Output-level per-bone rotation correction ("bone fix", exp6; docs/experiments_plan.md "exp6 bone fix").

model.py multiplies the head's final LOCAL (parent-relative, anny local-bone) rotmat of each listed bone b on the
right by one constant rotation:   R_b <- R_b @ C_b.
C_b is estimated by scripts/diagnostics/compute_bone_fix.py as the SO(3) projection of mean_i R_pred_i^T R_gt_i over
training images, so a bone whose prediction is R_gt F_b for a constant F_b (a constant half-turn, measured on v5 for
neck02/head/metacarpal1.L and on v10 for shoulder01.L/upperarm02.L/metacarpals) gets C_b = F_b^T and R_pred C_b = R_gt.

JSON file (passed as --bone_fix <path> to train.py, diag_holdout_trace.py, demo.py):
    {
      "format": "bone_fix/v1",
      "bones": {"neck02": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], ...},   # C_b, row-major
      "source": {...} or "...",            # what it was estimated from (npz, checkpoint, variant)
      "estimated_on": "...",               # the image selection (never the holdout it is evaluated on)
      ...                                  # any other keys (stats, method) are kept and ignored
    }
Bone names are resolved against the rig's bone_labels (exact, then case-insensitive); an unknown name is a hard
error. Every C_b must be a proper rotation (|C^T C - I| and |det C - 1| <= ORTHO_TOL); it is re-projected onto SO(3)
to remove JSON round-off.

numpy + json only (no torch, no anny), so compute_bone_fix.py and the tests can import it anywhere.
"""
import difflib
import json
import os

import numpy as np

FORMAT = 'bone_fix/v1'
ORTHO_TOL = 1e-4


def project_so3(M):
    """Nearest rotation (Frobenius) to a 3x3 matrix: SVD with the determinant fixed to +1."""
    M = np.asarray(M, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(M)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def angle_deg(R):
    """Rotation angle of [..., 3, 3] in degrees (numpy)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def axis_angle(R):
    """Unit axis [3] and angle (deg) of one rotation; near 180 deg the axis comes from (R + I) / 2 (sign arbitrary)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ang = float(angle_deg(R))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return np.zeros(3), 0.0
    if ang < 175.0:
        return vee / np.linalg.norm(vee), ang
    S = (R + np.eye(3)) / 2.0
    j = int(np.argmax(np.diag(S)))
    k = S[:, j] / max(np.linalg.norm(S[:, j]), 1e-12)
    if float(k @ vee) < 0:
        k = -k
    return k, ang


def check_rotation(name, C, tol=ORTHO_TOL):
    """C as a float64 [3, 3] proper rotation (re-projected), or ValueError naming the bone."""
    try:
        A = np.asarray(C, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"bone_fix: '{name}' is not a 3x3 numeric matrix ({e})")
    if A.shape != (3, 3) or not np.isfinite(A).all():
        raise ValueError(f"bone_fix: '{name}' must be a finite 3x3 matrix, got shape {A.shape}")
    ortho = float(np.abs(A.T @ A - np.eye(3)).max())
    det = float(np.linalg.det(A))
    if ortho > tol or abs(det - 1.0) > tol:
        raise ValueError(f"bone_fix: '{name}' is not a proper rotation (max |C^T C - I| = {ortho:.2e}, det = {det:.6f}; "
                         f"tolerance {tol:g})")
    return project_so3(A)


def normalize(spec, origin='<dict>'):
    """Validated copy of a bone-fix dict: bones as float64 lists of proper rotations, other keys kept."""
    if not isinstance(spec, dict) or not isinstance(spec.get('bones'), dict):
        raise ValueError(f"bone_fix {origin}: expected a JSON object with a 'bones' object "
                         "{name: 3x3 rotation}, e.g. {\"bones\": {\"neck02\": [[1,0,0],[0,1,0],[0,0,1]]}}")
    out = {k: v for k, v in spec.items() if k != 'bones'}
    out['bones'] = {str(n): check_rotation(n, C).tolist() for n, C in spec['bones'].items()}
    out.setdefault('format', FORMAT)
    return out


def load(path):
    """Read and validate a bone-fix JSON file; adds 'loaded_from' (absolute path) to the returned dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bone_fix: file not found: {path}")
    with open(path) as f:
        spec = json.load(f)
    spec = normalize(spec, origin=path)
    spec['loaded_from'] = os.path.abspath(path)
    return spec


def save(path, bones, **meta):
    """Write {format, bones, **meta}; bones = {name: 3x3}. Matrices are re-checked and written at full precision."""
    spec = normalize(dict(meta, bones=bones), origin=path)
    spec['format'] = FORMAT
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(spec, f, indent=2, allow_nan=False)
    return spec


def resolve(spec, names):
    """[(name, index, C float64 [3,3])] for every bone of spec against the rig's bone names (exact match first,
    then a unique case-insensitive one). Unknown or ambiguous names and two entries on one bone are hard errors."""
    if not spec:
        return []
    spec = normalize(spec)
    names = [str(n) for n in names]
    exact = {n: i for i, n in enumerate(names)}
    lower = {}
    for i, n in enumerate(names):
        lower.setdefault(n.lower(), []).append(i)
    out, seen = [], {}
    for n, C in spec['bones'].items():
        if n in exact:
            i = exact[n]
        elif len(lower.get(n.lower(), [])) == 1:
            i = lower[n.lower()][0]
        else:
            close = difflib.get_close_matches(n, names, n=4, cutoff=0.6)
            raise ValueError(f"bone_fix: unknown bone '{n}' (rig has {len(names)} bones; close: {close or 'none'})")
        if i in seen:
            raise ValueError(f"bone_fix: '{n}' and '{seen[i]}' both name bone {names[i]} (index {i})")
        seen[i] = n
        out.append((names[i], i, np.asarray(C, dtype=np.float64)))
    return out


def info(spec, resolved):
    """JSON-safe description of an applied fix (what model.bone_fix_info holds; None when off)."""
    if not resolved:
        return None
    bones = []
    for n, i, C in resolved:
        ax, ang = axis_angle(C)
        bones.append(dict(name=n, index=int(i), angle_deg=round(ang, 3), axis=[round(float(x), 5) for x in ax]))
    return dict(bones=bones, loaded_from=spec.get('loaded_from'), estimated_on=spec.get('estimated_on'),
                source=spec.get('source'))


def describe(spec, resolved, origin=''):
    """The one [bone_fix] line every loader prints."""
    if not resolved:
        return f"[bone_fix] off{(' (' + origin + ')') if origin else ''}"
    parts = ', '.join(f"{n} {angle_deg(C):.1f} deg" for n, _, C in resolved)
    src = origin or spec.get('loaded_from') or 'spec'
    est = spec.get('estimated_on')
    return (f"[bone_fix] ON, {len(resolved)} bone(s), R_b <- R_b @ C_b on the head's local rotmats "
            f"(from {src}{'; estimated on ' + str(est) if est else ''}): {parts}")


def from_arg(arg, ckpt_spec):
    """Resolve a --bone_fix CLI value against the spec saved in a checkpoint's args.
    '' (default) -> the checkpoint's own fix (None when it has none); 'none' -> off; anything else -> that file.
    Returns (spec or None, origin text)."""
    arg = (arg or '').strip()
    if arg.lower() == 'none':
        return None, 'forced off by --bone_fix none'
    if arg:
        return load(arg), arg
    if ckpt_spec:
        return normalize(ckpt_spec, origin='checkpoint args'), 'checkpoint args (bone_fix_spec)'
    return None, 'none in the checkpoint args'


def is_off(arg):
    return isinstance(arg, str) and arg.strip().lower() == 'none'


def angle_between_deg(R1, R2):
    """Geodesic distance (deg) between rotation stacks [..., 3, 3]."""
    return angle_deg(np.swapaxes(np.asarray(R1, dtype=np.float64), -1, -2) @ np.asarray(R2, dtype=np.float64))


def chordal_mean(Rs):
    """SO(3) projection of the entry-wise mean of rotations [P, 3, 3], plus the singular values of that mean
    (all ~1 = the rotations agree; small = spread out, the mean is ill-defined)."""
    M = np.asarray(Rs, dtype=np.float64).mean(0)
    return project_so3(M), np.linalg.svd(M, compute_uv=False)
