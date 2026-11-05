#!/usr/bin/env python
"""
render_smpl_vs_anny.py -- the SMPL and the Anny body model side by side, each in its rest pose with its joints.

WHAT IT MAKES
-------------
--mode render (default; cluster: needs smplx + models/smpl/SMPL_NEUTRAL.pkl, anny, pyrender with EGL)
    One PNG, two panels of equal width:
      left   SMPL neutral, zero pose, zero betas: the mesh (pyrender) + its 24 joints, joined along the SMPL
             kinematic tree (smpl.parents), every joint named.
      right  Anny full rig (create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True), as
             train.py / model.py build it), identity rotation on every bone, all 11 phenotypes at 0.5: the mesh +
             the heads of all bones, coloured by body region; the 24 points the 3DPW converter fits to the 24 SMPL
             joints (the bones of scripts/data/smpl_to_anny.py SMPL_TO_ANNY, spine and hips re-chosen by rest-pose
             geometry with refine_mapping_by_geometry, exactly as the converter does) drawn as large dots.
    Both bodies are turned to face the camera with smpl_to_anny._body_basis (hip midpoint, head, left/right hip),
    so the two rigs' different axes (SMPL y-up facing +z, Anny z-up facing -y) do not matter, and are drawn at the
    same scale (metres), so their rest statures can be compared.
    The presentation generator (docs/presentations/2026-10-22_supervisor/build_presentation.py) uses this PNG for
    the "Two body models" slide when it exists at the default --out path below; without it the slide uses crops of
    the thesis label-check figure and schematics.

--mode anny_skeleton (anny only, CPU, no SMPL file, no EGL)
    Writes a JSON with Anny's bone names, parents, rest-pose bone heads in the body frame, a body region per bone,
    the per-region counts and the name-based SMPL -> Anny correspondence of smpl_to_anny.py. The presentation
    generator draws the Anny skeleton schematic from it (docs/presentations/2026-10-22_supervisor/figures/
    anny_rest_skeleton.json). No rendering.

COMMANDS
--------
Render on the cluster, from login1 (repo root /netscratch/najib/multi-hmr; about 2 min on one GPU):

    cd /netscratch/najib/multi-hmr && NVIDIA_DRIVER_CAPABILITIES=all srun -K --gpus=1 --mem=16G --cpus-per-task=4 --time=00:20:00 \
      --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000 \
      --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
      --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
      --container-workdir=/netscratch/najib/multi-hmr \
      bash -c "source slurm/env.sh && source slurm/egl.sh && \
               python scripts/analysis/render_smpl_vs_anny.py --smpl_dir models --device cuda"

    It writes docs/presentations/2026-10-22_supervisor/figures/smpl_vs_anny_render.png (the path the generator
    looks for). Commit that PNG (or copy it into the same path of your checkout) and rebuild the deck:
        python docs/presentations/2026-10-22_supervisor/build_presentation.py

Export the Anny skeleton JSON (any machine with `pip install anny torch`; no GPU):

    python scripts/analysis/render_smpl_vs_anny.py --mode anny_skeleton \
        --out docs/presentations/2026-10-22_supervisor/figures/anny_rest_skeleton.json

Read-only for models and data: nothing but --out is written.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, 'scripts', 'analysis'), os.path.join(ROOT, 'scripts', 'data'), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

DEFAULT_PNG = os.path.join('docs', 'presentations', '2026-10-22_supervisor', 'figures', 'smpl_vs_anny_render.png')
DEFAULT_JSON = os.path.join('docs', 'presentations', '2026-10-22_supervisor', 'figures', 'anny_rest_skeleton.json')

# Standard SMPL kinematic tree for the 24 joints (used only if the loaded model exposes no `parents`).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Body regions for the slide, in drawing order: (key, label, RGB 0-1).
REGIONS = [
    ('root', 'root', (0.13, 0.22, 0.29)),
    ('spine', 'spine', (0.03, 0.50, 0.55)),
    ('neck_head', 'neck + head', (0.33, 0.50, 0.65)),
    ('face', 'face (jaw, lips, tongue, eyes, ...)', (0.77, 0.53, 0.21)),
    ('breast', 'breasts', (0.62, 0.45, 0.62)),
    ('arm', 'arms (clavicle to wrist)', (0.20, 0.55, 0.35)),
    ('hand', 'hands (fingers + metacarpals)', (0.85, 0.33, 0.25)),
    ('leg', 'legs (pelvis side to foot)', (0.25, 0.40, 0.80)),
    ('toe', 'toes', (0.55, 0.55, 0.55)),
]


def anny_region(i, name):
    """Body region of one Anny bone, by its MakeHuman name (list printed by --mode anny_skeleton)."""
    lo = name.lower()
    if i == 0 or lo == 'root':
        return 'root'
    if lo.startswith('spine'):
        return 'spine'
    if lo.startswith('neck') or lo == 'head':
        return 'neck_head'
    if lo.startswith('breast'):
        return 'breast'
    if lo.startswith(('clavicle', 'shoulder', 'upperarm', 'lowerarm', 'wrist')):
        return 'arm'
    if lo.startswith(('finger', 'thumb', 'metacarpal')):
        return 'hand'
    if lo.startswith(('pelvis', 'upperleg', 'lowerleg', 'foot')):
        return 'leg'
    if lo.startswith('toe'):
        return 'toe'
    return 'face'                 # jaw, special*, oris*, tongue*, levator*, eye*, orbicularis*, oculi*, ...


# --------------------------------------------------------------------------------------------------------------- #
# body models (same calls as model.py / train.py / check_3dpw_conversion.py)
# --------------------------------------------------------------------------------------------------------------- #
def load_anny(device):
    import torch
    import anny
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(device)
    bm.set_skinning_method('lbs')
    names = list(bm.bone_labels)
    pheno = list(bm.phenotype_labels)
    pose = torch.as_tensor(np.tile(np.eye(4, dtype=np.float32), (1, len(names), 1, 1)), device=device)
    kw = {k: torch.full((1,), 0.5, device=device) for k in pheno}
    with torch.no_grad():
        out = bm(pose_parameters=pose, phenotype_kwargs=kw)
    verts = out['vertices'][0].double().cpu().numpy()
    heads = out['bone_poses'][0, :, :3, 3].double().cpu().numpy()
    parents = [int(p) for p in bm.bone_parents]
    return dict(model=bm, names=names, pheno=pheno, verts=verts, heads=heads, parents=parents,
                version=getattr(anny, '__version__', '?'))


def load_smpl(smpl_dir, device):
    import torch
    import smplx
    m = smplx.create(smpl_dir, model_type='smpl', gender='neutral').to(device)
    with torch.no_grad():
        so = m(global_orient=torch.zeros(1, 3, device=device), body_pose=torch.zeros(1, 69, device=device),
               betas=torch.zeros(1, 10, device=device), transl=torch.zeros(1, 3, device=device))
    parents = getattr(m, 'parents', None)
    parents = [int(p) for p in parents[:24]] if parents is not None else list(SMPL_PARENTS)
    parents[0] = -1
    return dict(verts=so.vertices[0].double().cpu().numpy(), joints=so.joints[0, :24].double().cpu().numpy(),
                faces=np.asarray(m.faces, dtype=np.int64), parents=parents)


def body_frame(points, origin, top, lhip, rhip):
    """Points -> (lateral right->left, forward, up) coordinates, origin at the hip midpoint (smpl_to_anny._body_basis)."""
    import smpl_to_anny as S2A
    B = S2A._body_basis(origin, top, lhip, rhip)
    return (np.asarray(points) - origin) @ B


def anny_mapping(names, ja=None, js=None):
    """SMPL joint -> Anny bone pairs as the converter resolves them (name-based, then rest-geometry refined)."""
    import smpl_to_anny as S2A
    pairs = S2A.resolve_mapping(names, verbose=False)
    if ja is not None and js is not None:
        pairs = S2A.refine_mapping_by_geometry(pairs, names, ja, js, verbose=False)
    return pairs, S2A


# --------------------------------------------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------------------------------------------- #
def export_anny_skeleton(args):
    A = load_anny(args.device)
    names, heads = A['names'], A['heads']
    idx = {n: i for i, n in enumerate(names)}
    o = 0.5 * (heads[idx['upperleg01.L']] + heads[idx['upperleg01.R']])
    bf = body_frame(heads, o, heads[idx['head']], heads[idx['upperleg01.L']], heads[idx['upperleg01.R']])
    regions = [anny_region(i, n) for i, n in enumerate(names)]
    counts = {k: regions.count(k) for k, _, _ in REGIONS}
    pairs, S2A = anny_mapping(names)
    offset_default = S2A.build_parser().get_default('offset_joints') if hasattr(S2A, 'build_parser') else None
    out = {
        'source': ("anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True), lbs, identity "
                   "rotation on every bone, all phenotypes 0.5; bone heads = bone_poses[:, :3, 3]; body frame = "
                   "smpl_to_anny._body_basis(hip midpoint, head, upperleg01.L, upperleg01.R); written by "
                   "scripts/analysis/render_smpl_vs_anny.py --mode anny_skeleton"),
        'anny_version': A['version'],
        'n_bones': len(names),
        'phenotypes': A['pheno'],
        'bone_names': names,
        'bone_parents': A['parents'],
        'bone_heads_body_frame_m': np.round(bf, 5).tolist(),
        'axes': ['lateral (right -> left)', 'forward', 'up'],
        'bone_region': regions,
        'region_counts': counts,
        'regions': [[k, label] for k, label, _ in REGIONS],
        'smpl_to_anny_name_mapping': [[sn, an] for _, _, sn, an in pairs],
        'mapping_note': ("name-based SMPL_TO_ANNY of scripts/data/smpl_to_anny.py; at fit time the spine and hip "
                         "bones are re-chosen by rest-pose geometry (refine_mapping_by_geometry, needs SMPL)"),
        'virtual_joints_default': offset_default,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"anny {A['version']}: {len(names)} bones, {len(A['pheno'])} phenotypes {A['pheno']}")
    print('bones per region: ' + ', '.join(f'{k} {v}' for k, v in counts.items()) + f'  (sum {sum(counts.values())})')
    print(f"{len(pairs)} SMPL joints mapped by name: " + ', '.join(f'{sn}->{an}' for _, _, sn, an in pairs))
    print('wrote', args.out)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    import render_anny_fit as raf

    S = load_smpl(args.smpl_dir, args.device)
    A = load_anny(args.device)
    names = A['names']
    idx = {n: i for i, n in enumerate(names)}
    import smpl_to_anny as S2A
    js = S['joints']
    pairs, _ = anny_mapping(names, A['heads'], js)
    matched = {ai: sn for _, ai, sn, _ in pairs}

    sj = S2A.SMPL_IDX
    o_s = 0.5 * (js[sj['left_hip']] + js[sj['right_hip']])
    smpl_v = body_frame(S['verts'], o_s, js[sj['head']], js[sj['left_hip']], js[sj['right_hip']])
    smpl_j = body_frame(js, o_s, js[sj['head']], js[sj['left_hip']], js[sj['right_hip']])
    ha = A['heads']
    o_a = 0.5 * (ha[idx['upperleg01.L']] + ha[idx['upperleg01.R']])
    anny_v = body_frame(A['verts'], o_a, ha[idx['head']], ha[idx['upperleg01.L']], ha[idx['upperleg01.R']])
    anny_j = body_frame(ha, o_a, ha[idx['head']], ha[idx['upperleg01.L']], ha[idx['upperleg01.R']])

    W, H, f, dist = args.width, args.height, args.focal, args.distance
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])

    def to_cam(p):
        """body frame (lateral, forward, up) -> OpenCV camera: person faces the camera, hips 0.1 m below centre."""
        p = np.asarray(p, dtype=np.float64)
        return np.stack([p[:, 0], -(p[:, 2] - 0.1), dist - p[:, 1]], 1)

    def panel(verts, faces, color):
        bg = np.full((H, W, 3), 255.0, dtype=np.float32)
        return raf.render_overlay(bg, to_cam(verts), faces, K, color=color, alpha=0.55)

    left = panel(smpl_v, S['faces'], (0.95, 0.60, 0.25))
    right = panel(anny_v, raf.anny_faces(A['model']), (0.30, 0.60, 1.00))

    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
        big = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
    except OSError:
        font = big = ImageFont.load_default()
    rgb = lambda c: tuple(int(255 * x) for x in c)

    im_l = Image.fromarray(left)
    d = ImageDraw.Draw(im_l)
    uv = raf.project(to_cam(smpl_j), K)
    for c, p in enumerate(S['parents']):
        if p >= 0:
            d.line([tuple(uv[p]), tuple(uv[c])], fill=(32, 55, 75), width=3)
    for i, (u, v) in enumerate(uv):
        d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(196, 136, 53), outline=(32, 55, 75), width=2)
        name = S2A.SMPL_JOINT_NAMES[i]
        if not name.startswith('right_'):          # centre + left side named; the right side mirrors the left
            d.text((u + 8, v + (4 if name.startswith('left_') and i % 2 else -18)), name.replace('left_', ''),
                   fill=(32, 55, 75), font=font)
    d.text((16, 48), 'centre joints and the left side are named; the right side has the same 9 joints',
           fill=(82, 103, 121), font=font)
    d.text((16, 12), f"SMPL: {len(S['verts'])} vertices, 24 joints", fill=(20, 45, 69), font=big)

    im_r = Image.fromarray(right)
    d = ImageDraw.Draw(im_r)
    uv = raf.project(to_cam(anny_j), K)
    col = {k: rgb(c) for k, _, c in REGIONS}
    for c, p in enumerate(A['parents']):
        if p >= 0:
            d.line([tuple(uv[p]), tuple(uv[c])], fill=col[anny_region(c, names[c])], width=2)
    for i, (u, v) in enumerate(uv):
        r = 7 if i in matched else 3
        d.ellipse([u - r, v - r, u + r, v + r], fill=col[anny_region(i, names[i])],
                  outline=(20, 20, 20) if i in matched else None, width=2)
    counts = {}
    for i, n in enumerate(names):
        counts[anny_region(i, n)] = counts.get(anny_region(i, n), 0) + 1
    d.text((16, 12), f"Anny: {len(names)} bones ({len(matched)} matched to SMPL joints: large dots)",
           fill=(20, 45, 69), font=big)
    y = 48
    for k, label, c in REGIONS:
        d.rectangle([16, y + 3, 30, y + 17], fill=rgb(c))
        d.text((38, y), f"{label}: {counts.get(k, 0)}", fill=(32, 55, 75), font=font)
        y += 22

    out = Image.new('RGB', (2 * W, H), (255, 255, 255))
    out.paste(im_l, (0, 0))
    out.paste(im_r, (W, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out)
    print(f"SMPL {len(S['verts'])} vertices / 24 joints; anny {A['version']} {len(names)} bones, "
          f"{len(A['verts'])} vertices as loaded (remove_unattached_vertices=False)")
    print('matched:', ', '.join(f'{sn}->{names[ai]}' for ai, sn in matched.items()))
    print('wrote', args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--mode', choices=['render', 'anny_skeleton'], default='render')
    ap.add_argument('--out', default=None, help=f'default: {DEFAULT_PNG} (render) / {DEFAULT_JSON} (anny_skeleton)')
    ap.add_argument('--smpl_dir', default='models', help="smplx.create(<dir>, 'smpl', gender='neutral') folder")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--width', type=int, default=900)
    ap.add_argument('--height', type=int, default=1100)
    ap.add_argument('--focal', type=float, default=1500.0, help='px; same for both panels')
    ap.add_argument('--distance', type=float, default=3.4, help='m from the camera to the hip midpoint')
    args = ap.parse_args()
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    if args.out is None:
        args.out = DEFAULT_PNG if args.mode == 'render' else DEFAULT_JSON
    export_anny_skeleton(args) if args.mode == 'anny_skeleton' else render(args)


if __name__ == '__main__':
    main()
