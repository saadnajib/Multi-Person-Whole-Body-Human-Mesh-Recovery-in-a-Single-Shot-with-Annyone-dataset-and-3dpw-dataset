"""
E1 / E3 -- trace held-out AnnyOne samples through GT and prediction paths and
measure WHERE the neck, finger and body-shape errors come from.

For every selected image this script:
  1. loads it exactly as training does (datasets/annyone.py + bedlam.collate_fn)
     and keeps the GT people train.py's prepare_gt keeps (root in front of the
     camera, root inside the image, one person per heatmap cell);
  2. builds the GT mesh from the dataset's anny_pose / anny_shape through the
     model's own Anny body model (the train.py prepare_gt path) and projects it
     with the dataset K, so a GT overlay that does not sit on the person means
     the camera/pose convention is wrong BEFORE any network is involved;
  3. runs the checkpoint under several inference variants on the SAME tensor:
       V0  eval-faithful : fp32 (TF32 matmul, as train.py), dataset K, helper bones UNMASKED  (train.py evaluate())
       V1  V0 + helper-bone mask                                    (what every demo renders)
       V2  demo network call: fp16 autocast, fov-derived K, mask    (demo.py in-domain path,
                                                                     WITHOUT filter_humans)
       V3  V2 + --relax_hands 1                                      (flag only; not the
                                                                     detector-crop path)
       V4  --person_crop 1 only: V1 settings on a square crop of the ORIGINAL render around
           each kept GT person (--crop_factor x the GT joint bbox, kept inside the image,
           resized to the network input, crop-corrected K), one forward per person. The 3D
           prediction is in the camera frame, so every pelvis-centred, rotation and shape
           metric compares with V1; placement/depth are excluded from the paired read.
  4. matches detections to GT people (same greedy 2D matcher as train.py) and
     reports, per variant: recall/FP, PVE, PA-PVE, MPJPE, PVE on head+neck
     vertices, per-bone-group rotation error, per-phenotype shape error and bias,
     neck length, neck/skeletal-height and head-size ratios against GT, and a
     shape-vs-pose FACTORISATION of the neck length
     (GT pose + predicted shape, predicted pose + GT shape, rest pose);
  5. writes per-person CSV, a summary JSON and overlay panels
     [image | GT mesh | V1 | V2] for the first --n_vis samples.

Nothing is trained or written to the checkpoint. ~5-10 min on one GPU for 100 images.

USAGE (repo root, multihmr env, GPU node):
    python scripts/diagnostics/diag_holdout_trace.py \
        --ckpt logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt \
        --n 100 --out results/diagnostics/holdout_v5

    # E3: the youngest people in the first --scan training samples (IN-SAMPLE)
    python scripts/diagnostics/diag_holdout_trace.py --ckpt ... \
        --select youngest --scan 20000 --n 50 --out results/diagnostics/young_v5

    # CHECK 5 (manifest): the fixed (image, person) list from build_manifest.py, one split at a time;
    # adds the 'pred pose + GT shape' mesh (PVE and panel) and per-category tables (adult/child/hands/neck)
    python scripts/diagnostics/diag_holdout_trace.py --ckpt ... \
        --manifest results/diagnostics/manifest.json --manifest_split train --out results/diagnostics/manifest_trace_train_v5

    # exp4_fingers: does magnification alone fix the fingers? adds V4 and a V4-vs-V1 paired block
    python scripts/diagnostics/diag_holdout_trace.py --ckpt ... --person_crop 1 --out results/experiments/exp4_fingers/...
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402  (sets PYOPENGL_PLATFORM, sys.path)

import numpy as np   # noqa: E402
import torch         # noqa: E402
import roma          # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from datasets.annyone import AnnyOne          # noqa: E402
from datasets.bedlam import collate_fn        # noqa: E402
from utils import denormalize_rgb, normalize_rgb  # noqa: E402
from utils.training import match_2d_greedy    # noqa: E402

warnings.filterwarnings('ignore', 'Mean of empty slice')

VARIANTS = {
    # name: (autocast fp16, K source, mask_helper_joints, relax_hands, det_thresh key)
    'V0_eval_unmasked':   dict(amp=False, K='dataset', mask=False, relax=False, thr='eval'),
    'V1_eval_masked':     dict(amp=False, K='dataset', mask=True,  relax=False, thr='eval'),
    'V2_demo':            dict(amp=True,  K='fov',     mask=True,  relax=False, thr='demo'),
    'V3_demo_relaxhands': dict(amp=True,  K='fov',     mask=True,  relax=True,  thr='demo'),
}
VIS_VARIANTS = ('V1_eval_masked', 'V2_demo')
# --person_crop 1 only: V1 settings, one forward per GT person on a crop of the original render with its own K.
# Kept out of VARIANTS so the default variant list, outputs and paired table are unchanged.
CROP_VARIANT = 'V4_person_crop'
CROP_CFG = dict(amp=False, K='crop', mask=True, relax=False, thr='eval')
# anny age scale (anchors linspace(-1/3,1,5) -> baby 0, child 1/3 ~ 11 y, young 2/3 ~ 25 y, old 1 ~ 90 y)
AGE_BUCKETS = [(0.0, 0.2, '<0.2 (baby-~6y)'), (0.2, 0.4, '0.2-0.4 (~6-14y)'),
               (0.4, 0.6, '0.4-0.6 (~14-22y)'), (0.6, 0.8, '0.6-0.8 (~22-51y)'), (0.8, 1.01, '>0.8 (older)')]


def select_indices(args, pkls):
    """Returns (image indices, description, {idx: [person, ...]} or {}, {(idx, person): [tags]})."""
    n_total = len(pkls)
    if args.manifest:
        man = json.load(open(args.manifest))
        ents = [e for e in man['entries'] if e['split'] == args.manifest_split]
        if not ents:
            raise SystemExit(f"[select] manifest has no '{args.manifest_split}' entries")
        if os.path.abspath(man.get('data_folder', args.data_folder)) != os.path.abspath(args.data_folder):
            print(f"[select] WARNING: manifest built on {man.get('data_folder')} but --data_folder is {args.data_folder}")
        idx, sel, tags = [], {}, {}
        for e in ents:
            if e['idx'] not in sel:
                idx.append(e['idx']); sel[e['idx']] = []
            sel[e['idx']].append(e['person']); tags[(e['idx'], e['person'])] = list(e['tags'])
        return idx, f"manifest {args.manifest_split} ({len(ents)} people in {len(idx)} images)", sel, tags
    if args.select == 'holdout':
        pool = dc.holdout_indices(n_total)
        return pool[:args.n], 'HOLDOUT (never trained on)', {}, {}
    if args.select == 'indices':
        idx = [int(s) for s in args.indices.split(',') if s.strip()]
        return idx, 'explicit indices', {}, {}
    if args.select == 'youngest':
        scan = max(0, min(args.scan, n_total - dc.N_HOLDOUT))
        ages = []
        t0 = time.time()
        for i in range(scan):
            s = dc.shapes_from_pkl(dc.load_pkl(pkls[i]))
            if s is None or len(s) == 0:
                continue
            ages.append((float(s[:, 1].min()), i, int(np.argmin(s[:, 1]))))   # (age, image idx, person idx)
            if i % 5000 == 0 and i:
                print(f"  scanned {i}/{scan} ({time.time() - t0:.0f}s)", flush=True)
        if not ages:
            raise SystemExit(f"[select] no anny_shape found in the first {scan} annotations")
        ages.sort()
        idx = [i for _, i, _ in ages[:args.n]]
        sel_person = {i: [j] for _, i, j in ages[:args.n]}      # only THIS person is evaluated in that image
        if len(idx) < args.n:
            print(f"[select] WARNING: only {len(idx)} samples available (asked {args.n})")
        print(f"[select] youngest {len(idx)} of the first {scan} TRAINING samples "
              f"(min GT age {ages[0][0]:.3f} .. {ages[len(idx) - 1][0]:.3f}). "
              f"These were SEEN in training: this tests representation, not generalisation. "
              f"Only the youngest person of each image is evaluated.")
        return idx, f'youngest-{len(idx)} people of first {scan} training samples (IN-SAMPLE)', sel_person, {}
    raise ValueError(args.select)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='path, models/multiHMR name, or run_name:epoch')
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--select', default='holdout', choices=['holdout', 'youngest', 'indices'])
    ap.add_argument('--scan', type=int, default=20000, help='--select youngest: how many training pkls to scan')
    ap.add_argument('--indices', default='', help='--select indices: comma list of dataset indices')
    ap.add_argument('--manifest', default='', help='build_manifest.py JSON; overrides --select/--n')
    ap.add_argument('--manifest_split', default='holdout', choices=['train', 'holdout'])
    ap.add_argument('--out', default='results/diagnostics/holdout_trace')
    ap.add_argument('--n_vis', type=int, default=12)
    ap.add_argument('--variants', default=','.join(VARIANTS.keys()))
    ap.add_argument('--det_thresh_eval', type=float, default=0.2, help='train.py default')
    ap.add_argument('--det_thresh_demo', type=float, default=0.3, help='run_demo_syn.sh')
    ap.add_argument('--nms_kernel_size', type=int, default=3)
    ap.add_argument('--person_crop', type=int, default=0, choices=[0, 1],
                    help='1: add V4_person_crop (one forward per kept GT person on a square --crop_factor x GT-joint-bbox '
                         'crop of the ORIGINAL render, crop-corrected K, V1 settings otherwise) and a V4-vs-V1 paired '
                         'block. 0 = the four variants above, outputs unchanged.')
    ap.add_argument('--crop_factor', type=float, default=1.3,
                    help='--person_crop: crop side = factor x the long side of the GT joint bbox, kept inside the image')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    variants = {k: VARIANTS[k] for k in args.variants.split(',') if k in VARIANTS}
    if args.person_crop:
        variants[CROP_VARIANT] = CROP_CFG

    ckpt_path = dc.resolve_ckpt(args.ckpt)
    model, ckpt_args = dc.load_model(ckpt_path, device)
    bm = model.body_model
    names = list(bm.bone_labels)
    phen = list(bm.phenotype_labels)
    groups = dc.bone_groups(names)
    chain = dc.neck_chain(names)
    useful = model.useful_rotmat.reshape(-1).bool().cpu()        # 1 = active bone
    faces = dc.faces_tri(bm)
    S = model.img_size
    print(f"[rig] {len(names)} bones, {int(useful.sum())} active / {int((~useful).sum())} masked; "
          f"phenotypes={phen}")
    print(f"[rig] neck chain: {[names[i] for i in chain]}")
    for g, idx in groups.items():
        act = int(useful[torch.tensor(idx)].sum())
        print(f"[rig] group {g:<18s} {len(idx):3d} bones ({act} active) e.g. {[names[i] for i in idx[:4]]}")
    if phen != dc.PHENOTYPES:
        print(f"[WARN] body model phenotype order {phen} != assumed dataset order {dc.PHENOTYPES}")

    ds = AnnyOne(data_folder=args.data_folder, img_size=S)
    pkls = ds.pkl_files
    indices, sel_desc, sel_person, tagmap = select_indices(args, pkls)
    print(f"[data] {len(pkls)} samples; evaluating {len(indices)} = {sel_desc}")

    rows = []                 # one row per (variant, matched person)
    det = {k: dict(gt=0, matched=0, fp=0, img_nodet=0, nonfinite=0) for k in variants}
    cam_stats = []            # per-image camera / annotation checks
    n_gt_total = n_gt_dropped = 0
    t0 = time.time()

    for n_done, idx in enumerate(indices):
        img_arr, annot = ds[idx]
        x, y = collate_fn([(img_arr, annot)])
        x = x.to(device)
        K = y['K'].to(device)                                      # [1,3,3] dataset K
        valid = y['valid_humans'][0].bool()
        nh = int(valid.sum())
        if nh == 0:
            continue
        pose = y['anny_pose'][0][valid][:, :163].float().to(device)                    # [nh,163,4,4]
        shape_gt = y['anny_shape'][0][valid].reshape(nh, -1)[:, :len(phen)].float().to(device)  # [nh,11]

        # --- GT through the body model (train.py prepare_gt path) ------------
        gt_v, gt_j, gt_bp = dc.run_body_model_full(bm, pose, shape_gt)
        keep = dc.visible_gt_mask(gt_j, K, S, model.patch_size)
        n_gt_total += nh; n_gt_dropped += int((~keep).sum())
        if not keep.any():
            continue
        # --select youngest: the matcher still sees ALL GT people (so other people are not counted as
        # false positives), but metrics are kept only for the selected person.
        sel_gids = None
        if idx in sel_person:
            sel_gids = []
            for j in sel_person[idx]:
                if j >= nh or not bool(keep[j]):
                    print(f"  [warn] idx {idx}: selected person {j} invalid or dropped by the visibility filter; skipped")
                else:
                    sel_gids.append(int(keep[:j].sum()))          # index after the filter
            if not sel_gids:
                continue
        orig_of_gid = torch.nonzero(keep).flatten().tolist()     # filtered index -> annotation person index
        pose, shape_gt, gt_v, gt_j, gt_bp = pose[keep], shape_gt[keep], gt_v[keep], gt_j[keep], gt_bp[keep]
        nhv = int(keep.sum())
        gt_R = dc.orthonormalize(pose[:, :, :3, :3])
        gt_j2d = dc.project(gt_j, K)
        inside = ((gt_j2d >= 0) & (gt_j2d <= S)).all(-1).float().mean().item()
        gt_neck = dc.chain_length(gt_j, chain)
        gt_height = dc.skeletal_height(gt_j, names)
        I_all = dc.identity_pose(nhv, 163, device)
        v_rest_gs_all, j_rest_gs_all = dc.run_body_model(bm, I_all, shape_gt)          # rest pose, GT shapes
        hn_mask_gt = dc.head_neck_vertex_mask(v_rest_gs_all, j_rest_gs_all, names)     # [nhv,V] head+neck vertices
        gt_headsz = dc.head_size(gt_v, gt_j, names, hn_mask_gt)
        gt_helper_angle = dc.geodesic_deg(torch.eye(3, device=device).expand(nhv, int((~useful).sum()), 3, 3),
                                          gt_R[:, ~useful.to(device)]).mean(1)       # how far GT helper bones are from identity
        nonroot_t = float(pose[:, 1:, :3, 3].abs().max())        # factorisation assumes 0

        # --- camera sanity from the raw annotation ---------------------------
        raw = dc.load_pkl(pkls[idx])
        with Image.open(dc.image_path_for(pkls[idx])) as im:
            W, H = im.size
            img_full = im.convert('RGB') if args.person_crop else None
        pp = np.asarray(raw['princpt'], dtype=np.float64).reshape(-1)[:2]
        fov = dc.fov_from_K(K[0].cpu(), S)
        K_fov = dc.K_from_fov(fov, S, device)
        k_diff = float((K_fov - K).abs().max())
        K_orig = None
        if args.person_crop:
            # original-resolution intrinsics; datasets/annyone.py scales these by S / max(W, H) for the full frame
            f2 = np.asarray(raw['focal'], dtype=np.float64).reshape(-1)
            f2 = f2 if f2.size > 1 else np.repeat(f2, 2)
            K_orig = torch.tensor([[f2[0], 0.0, pp[0]], [0.0, f2[1], pp[1]], [0.0, 0.0, 1.0]],
                                  dtype=torch.float32, device=device).unsqueeze(0)
        cam_stats.append(dict(idx=idx, W=W, H=H, cx_over_W=pp[0] / W, cy_over_H=pp[1] / H,
                              fx_eq_fy=bool(abs(float(K[0, 0, 0]) - float(K[0, 1, 1])) < 1e-3),
                              fov=fov, k_fov_vs_dataset_px=k_diff, gt_j2d_inside=inside,
                              nonroot_translation=nonroot_t, n_gt=nh, n_gt_kept=nhv))

        def person_row(p, gid, vname, cfg, K_match):
            """Every per-person metric of one matched prediction p against GT person gid of this image.
            K_match is the camera the prediction is projected with for the placement block (the variant's K,
            or the dataset K for V4, whose camera-frame 3D is thereby mapped back into the full frame)."""
            v_hat = p['v3d'].float(); j_hat = p['j3d'].float()
            pelv = gt_j[gid, 0:1]; pelv_hat = j_hat[0:1]
            v_ctx, vh_ctx = gt_v[gid] - pelv, v_hat - pelv_hat
            pve = ((v_ctx - vh_ctx).norm(dim=-1) * 1000).mean().item()
            R, t, s = roma.rigid_points_registration(vh_ctx, v_ctx, compute_scaling=True)
            pa = s * (R.reshape(1, 3, 3) @ vh_ctx.reshape(-1, 3, 1)).reshape(-1, 3) + t
            pa_pve = ((v_ctx - pa).norm(dim=-1) * 1000).mean().item()
            mpjpe = (((gt_j[gid] - pelv) - (j_hat - pelv_hat)).norm(dim=-1) * 1000).mean().item()
            # PLACEMENT (not pelvis-centred): train.py builds the transl/dist/heatmap targets at joint 0
            # (root) while the model anchors the HEAD joint (person_center) at the detected location.
            # These numbers show where the predicted body actually lands relative to the GT body.
            j2d_hat = dc.project(j_hat.unsqueeze(0), K_match)[0]              # [163,2] in the variant's camera
            j2d_gt = dc.project(gt_j[gid:gid + 1], K)[0]
            head_i0 = chain[-1]
            place = dict(
                mpjpe_abs=((gt_j[gid] - j_hat).norm(dim=-1) * 1000).mean().item(),
                root_err_mm=((gt_j[gid, 0] - j_hat[0]).norm() * 1000).item(),
                head_err_mm=((gt_j[gid, head_i0] - j_hat[head_i0]).norm() * 1000).item(),
                root_depth_err_m=float(j_hat[0, 2] - gt_j[gid, 0, 2]),
                head_px_err=float((j2d_hat[head_i0] - j2d_gt[head_i0]).norm()),
                root_px_err=float((j2d_hat[0] - j2d_gt[0]).norm()),
                pred_head_vs_gt_root_px=float((j2d_hat[head_i0] - j2d_gt[0]).norm()),
                gt_head_vs_gt_root_px=float((j2d_gt[head_i0] - j2d_gt[0]).norm()),
                pred_head_vs_pred_root_px=float((j2d_hat[head_i0] - j2d_hat[0]).norm()),
                vertical_shift_px=float(j2d_hat[0, 1] - j2d_gt[0, 1]))   # + = predicted root lower in the image
            # head+neck surface vertices, selected on the REST-POSE GT mesh (hands cannot intrude): the region PVE hides
            near_neck = hn_mask_gt[gid]
            pve_headneck = ((v_ctx - vh_ctx).norm(dim=-1)[near_neck] * 1000).mean().item() if near_neck.any() else float('nan')

            # rotations
            R_hat = roma.rotvec_to_rotmat(p['rotvec'].float().reshape(-1, 3)).reshape(163, 3, 3)
            ang = dc.geodesic_deg(R_hat, gt_R[gid]).cpu()          # [163]
            grp_err = {}
            for g, gi in groups.items():
                gi_t = torch.tensor(gi)
                act = gi_t[useful[gi_t]]
                grp_err[g] = float(ang[act].mean()) if len(act) else float('nan')
            # only meaningful when the helper bones are NOT masked (V0); masked => identity by construction
            helper_pred_err = float(ang[~useful].mean()) if (not cfg['mask'] and (~useful).any()) else float('nan')

            # shape
            sh_hat = p['shape'].float().reshape(-1)
            sh_gt = shape_gt[gid]
            sh_err = (sh_hat - sh_gt).cpu().numpy()

            # factorisation: which factor moves the neck length?
            P_gt = pose[gid:gid + 1]
            P_hat = dc.rotmat_to_homogeneous(R_hat.unsqueeze(0))
            # GLOBAL orientation error of end effectors: the 4-bone neck chain is kinematically redundant,
            # so per-bone (local) errors can be large while the head still faces the right way.
            _, _, bp_hat = dc.run_body_model_full(bm, P_hat, sh_hat.unsqueeze(0))
            glob_err = dc.geodesic_deg(bp_hat[0, :, :3, :3], gt_bp[gid, :, :3, :3]).cpu()   # [163], camera-frame
            head_i = chain[-1]; wrist_i = [i for i, n in enumerate(names) if n.lower().startswith('wrist')]
            v_gp_ps, j_gp_ps = dc.run_body_model(bm, P_gt, sh_hat.unsqueeze(0))      # GT pose + pred shape
            v_pp_gs, j_pp_gs = dc.run_body_model(bm, P_hat, sh_gt.unsqueeze(0))      # pred pose + GT shape
            # CHECK 5: PVE of the swapped-shape meshes against GT, pelvis-centred (same topology, same vertex order)
            pve_pp_gs = ((v_ctx - (v_pp_gs[0] - j_pp_gs[0, 0:1])).norm(dim=-1) * 1000).mean().item()
            pve_gp_ps = ((v_ctx - (v_gp_ps[0] - j_gp_ps[0, 0:1])).norm(dim=-1) * 1000).mean().item()
            I = dc.identity_pose(1, 163, device)
            v_rest_gs, j_rest_gs = v_rest_gs_all[gid:gid + 1], j_rest_gs_all[gid:gid + 1]   # rest pose, GT shape
            v_rest_ps, j_rest_ps = dc.run_body_model(bm, I, sh_hat.unsqueeze(0))            # rest pose, pred shape
            hn_mask_pred = dc.head_neck_vertex_mask(v_rest_ps, j_rest_ps, names)

            # neck / skeletal height / head size (GT and prediction under the SAME definitions)
            neck_hat = dc.chain_length(j_hat.unsqueeze(0), chain)[0].item()
            height_hat = dc.skeletal_height(j_hat.unsqueeze(0), names)[0].item()
            headsz_hat = dc.head_size(v_hat.unsqueeze(0), j_hat.unsqueeze(0), names, hn_mask_pred)[0].item()
            g_neck, g_height, g_head = gt_neck[gid].item(), gt_height[gid].item(), gt_headsz[gid].item()

            row = dict(variant=vname, idx=idx, gid=gid, person=orig_of_gid[gid], score=float(p['scores']),
                       tags='|'.join(tagmap.get((idx, orig_of_gid[gid]), [])),
                       pve=pve, pa_pve=pa_pve, mpjpe=mpjpe, pve_headneck=pve_headneck,
                       pve_predpose_gtshape=pve_pp_gs, pve_gtpose_predshape=pve_gp_ps,
                       **place,
                       rot_global_head_deg=float(glob_err[head_i]),
                       rot_global_wrist_deg=float(glob_err[wrist_i].mean()) if wrist_i else float('nan'),
                       rot_global_root_deg=float(glob_err[0]),
                       helper_bones_pred_err_deg=helper_pred_err,
                       gt_helper_bones_angle_deg=float(gt_helper_angle[gid]),
                       gt_neck_m=g_neck, pred_neck_m=neck_hat,
                       gt_skel_height_m=g_height, pred_skel_height_m=height_hat,
                       gt_head_m=g_head, pred_head_m=headsz_hat,
                       # paired ratios computed per person (never divide two filtered columns)
                       neck_ratio=neck_hat / max(g_neck, 1e-6),
                       neck_over_height_ratio=(neck_hat / max(height_hat, 1e-6)) / max(g_neck / max(g_height, 1e-6), 1e-6),
                       head_over_height_ratio=(headsz_hat / max(height_hat, 1e-6)) / max(g_head / max(g_height, 1e-6), 1e-6),
                       neck_gtpose_predshape_m=dc.chain_length(j_gp_ps, chain)[0].item(),
                       neck_predpose_gtshape_m=dc.chain_length(j_pp_gs, chain)[0].item(),
                       neck_rest_gtshape_m=dc.chain_length(j_rest_gs, chain)[0].item(),
                       neck_rest_predshape_m=dc.chain_length(j_rest_ps, chain)[0].item(),
                       skel_height_rest_gtshape_m=dc.skeletal_height(j_rest_gs, names)[0].item(),
                       skel_height_rest_predshape_m=dc.skeletal_height(j_rest_ps, names)[0].item(),
                       head_rest_gtshape_m=dc.head_size(v_rest_gs, j_rest_gs, names, hn_mask_gt[gid:gid + 1])[0].item(),
                       head_rest_predshape_m=dc.head_size(v_rest_ps, j_rest_ps, names, hn_mask_pred)[0].item(),
                       gt_depth_m=float(gt_j[gid, 0, 2]), pred_depth_m=float(j_hat[0, 2]),
                       k_fov_vs_dataset_px=k_diff, nonroot_translation=nonroot_t)
            for g, e in grp_err.items():
                row[f'rot_{g}'] = e
            for i, nme in enumerate(phen):     # phenotype columns are namespaced: 'height' is also a metric word
                row[f'pheno_gt_{nme}'] = float(sh_gt[i]); row[f'pheno_pred_{nme}'] = float(sh_hat[i]); row[f'pheno_err_{nme}'] = float(sh_err[i])
            return row

        vis_panels = {}
        crop_panels = []
        for vname, cfg in variants.items():
            if cfg['K'] == 'crop':
                # V4: one forward per GT person on a square crop of the ORIGINAL render (side = --crop_factor x the
                # long side of the GT joint bbox, kept inside the image), preprocessed like datasets/annyone.py,
                # with the crop's own intrinsics (scripts/diagnostics/diag_crop_camera.py: f*r, (c - offset)*r,
                # r = S/side). The prediction is in the camera frame, so the pelvis-centred, camera-frame rotation
                # and shape metrics compare with V1 as they are; placement/depth come from a camera outside the
                # training fov range and are reported but kept out of the paired read.
                dc.set_inference_toggles(model, mask_helper_joints=cfg['mask'], relax_hands=cfg['relax'])
                thr = args.det_thresh_eval if cfg['thr'] == 'eval' else args.det_thresh_demo
                gt_j2d_orig = dc.project(gt_j, K_orig)                              # [nhv,163,2] original pixels
                for gid in (sel_gids if sel_gids is not None else range(nhv)):
                    det[vname]['gt'] += 1
                    pts = gt_j2d_orig[gid][gt_j[gid, :, 2] > 0.1]
                    if pts.shape[0] < 2:
                        det[vname]['img_nodet'] += 1
                        continue
                    lo, hi = pts.min(0).values, pts.max(0).values
                    side = int(round(min(max(float((hi - lo).max()) * args.crop_factor, 64.0), float(min(W, H)))))
                    left = int(round(min(max(float(lo[0] + hi[0]) / 2.0 - side / 2.0, 0.0), W - side)))
                    top = int(round(min(max(float(lo[1] + hi[1]) / 2.0 - side / 2.0, 0.0), H - side)))
                    crop = img_full.crop((left, top, left + side, top + side))
                    crop_u8 = np.asarray(ImageOps.pad(ImageOps.contain(crop, (S, S)), size=(S, S)))
                    x_c = torch.from_numpy(normalize_rgb(crop_u8, imagenet_normalization=1)).unsqueeze(0).to(device)
                    r = S / float(side)
                    K_c = torch.eye(3, device=device).unsqueeze(0)
                    K_c[0, 0, 0], K_c[0, 1, 1] = K_orig[0, 0, 0] * r, K_orig[0, 1, 1] * r
                    K_c[0, 0, 2], K_c[0, 1, 2] = (K_orig[0, 0, 2] - left) * r, (K_orig[0, 1, 2] - top) * r
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=cfg['amp']):
                            preds = model(x_c, is_training=False, K=K_c, det_thresh=thr,
                                          nms_kernel_size=args.nms_kernel_size)
                    n_raw = len(preds)
                    preds = [p for p in preds if torch.isfinite(p['j2d']).all() and torch.isfinite(p['v3d']).all()
                             and torch.isfinite(p['j3d']).all()]
                    det[vname]['nonfinite'] += n_raw - len(preds)
                    if len(preds) == 0:
                        det[vname]['img_nodet'] += 1                                # crops without a detection
                        continue
                    pj2d = np.asarray([p['j2d'].float().cpu().numpy()[:163] for p in preds])
                    gj2d_c = dc.project(gt_j, K_c).cpu().numpy()      # ALL kept people: a neighbour in the crop is not a FP
                    best, fps, _ = match_2d_greedy(pj2d, gj2d_c, np.ones_like(gj2d_c[..., 0]).astype(bool))
                    det[vname]['fp'] += len(fps)
                    hit = [pid for (pid, g) in best if g == gid]
                    if not hit:
                        continue
                    det[vname]['matched'] += 1
                    row = person_row(preds[hit[0]], gid, vname, cfg, K)    # dataset K: 3D mapped back into the full frame
                    row.update(crop_side_px=side, crop_scale_vs_input=max(W, H) / float(side),
                               crop_fov_deg=math.degrees(2 * math.atan(S / (2.0 * float(K_c[0, 0, 0])))),
                               crop_person_frac=float((hi - lo).max()) / side)
                    rows.append(row)
                    if n_done < args.n_vis and len(crop_panels) < 3:
                        ov = dc.render_overlay(crop_u8, [preds[hit[0]]['v3d'].float().cpu().numpy()], faces,
                                               K_c[0].cpu().numpy())
                        crop_panels.append(dc.label(ov if ov is not None else crop_u8,
                                                    f"{vname} x{max(W, H) / side:.1f} gid {gid}"))
                continue

            dc.set_inference_toggles(model, mask_helper_joints=cfg['mask'], relax_hands=cfg['relax'])
            K_v = K if cfg['K'] == 'dataset' else K_fov
            thr = args.det_thresh_eval if cfg['thr'] == 'eval' else args.det_thresh_demo
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg['amp']):
                    preds = model(x, is_training=False, K=K_v, det_thresh=thr,
                                  nms_kernel_size=args.nms_kernel_size)
            det[vname]['gt'] += (len(sel_gids) if sel_gids is not None else nhv)
            n_raw = len(preds)
            preds = [p for p in preds if torch.isfinite(p['j2d']).all() and torch.isfinite(p['v3d']).all()
                     and torch.isfinite(p['j3d']).all()]
            det[vname]['nonfinite'] += n_raw - len(preds)
            if len(preds) == 0:
                det[vname]['img_nodet'] += 1
                continue
            pj2d = np.asarray([p['j2d'].float().cpu().numpy()[:163] for p in preds])
            gj2d = gt_j2d.cpu().numpy()
            best, fps, misses = match_2d_greedy(pj2d, gj2d, np.ones_like(gj2d[..., 0]).astype(bool))
            det[vname]['fp'] += len(fps)
            if sel_gids is not None:
                best = [(pid, gid) for (pid, gid) in best if gid in sel_gids]
            det[vname]['matched'] += len(best)

            for (pid, gid) in best:
                rows.append(person_row(preds[pid], gid, vname, cfg, K if cfg['K'] == 'dataset' else K_fov))

            if n_done < args.n_vis and vname in VIS_VARIANTS:
                vis_panels[vname] = ([preds[pid]['v3d'].float().cpu().numpy() for pid, _ in best],
                                     (K if cfg['K'] == 'dataset' else K_fov)[0].cpu().numpy())
                if vname == 'V1_eval_masked':
                    # CHECK 5 panel: predicted pose + GT shape, placed at the predicted root
                    swapped = []
                    for pid, gid in best:
                        R_h = roma.rotvec_to_rotmat(preds[pid]['rotvec'].float().reshape(-1, 3)).reshape(1, 163, 3, 3)
                        v_s, j_s = dc.run_body_model(bm, dc.rotmat_to_homogeneous(R_h), shape_gt[gid:gid + 1])
                        root_hat = preds[pid]['j3d'].float()[0:1]
                        swapped.append((v_s[0] - j_s[0, 0:1] + root_hat).cpu().numpy())
                    vis_panels['V1 pose + GT shape'] = (swapped, K[0].cpu().numpy())

        # --- overlay panel -----------------------------------------------------
        if n_done < args.n_vis:
            img_u8 = denormalize_rgb(x[0].float().cpu().numpy())
            Kc = K[0].cpu().numpy()
            panels = [dc.label(img_u8, f"idx {idx}")]
            g_ov = dc.render_overlay(img_u8, [v.cpu().numpy() for v in gt_v], faces, Kc)
            panels.append(dc.label(g_ov if g_ov is not None else dc.draw_points(img_u8, gt_j2d.cpu().numpy()),
                                   "GT (dataset pose+shape)"))
            for vname in list(VIS_VARIANTS) + ['V1 pose + GT shape']:
                if vname in vis_panels:
                    verts, Kv = vis_panels[vname]
                    ov = dc.render_overlay(img_u8, verts, faces, Kv)        # rendered with the K the variant was placed with
                    panels.append(dc.label(ov if ov is not None else img_u8, vname))
            panels.extend(crop_panels)                                      # V4 crops (--person_crop 1 only)
            Image.fromarray(dc.hstack(panels)).save(os.path.join(args.out, 'vis', f'{idx:06d}.jpg'), quality=90)

        if n_done % 10 == 0:
            print(f"  {n_done + 1}/{len(indices)} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ report
    if not rows:
        print("no matched people; nothing to report"); return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(os.path.join(args.out, 'per_person.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def col(vn, k):
        return np.asarray([r.get(k, np.nan) for r in rows if r['variant'] == vn], dtype=np.float64)

    def pairs(vn, ka, kb):
        a = np.asarray([[r.get(ka, np.nan), r.get(kb, np.nan)] for r in rows if r['variant'] == vn], dtype=np.float64)
        a = a[np.isfinite(a).all(1)] if len(a) else a.reshape(0, 2)
        return a[:, 0], a[:, 1]

    def depth_fit(vn):
        # pred root depth = slope * GT + intercept over the matched people, plus the median pred/GT depth
        # ratio and the median projected-size ratio (pred / GT head-to-root pixel distance). A learned depth
        # gives slope ~1, intercept ~0 and ratios ~1; pelvis-centred PVE cannot see any of this.
        out = dict(depth_fit_slope=None, depth_fit_intercept_m=None, depth_ratio_median=None,
                   size_ratio_median_px=None, depth_fit_n=0)
        gd, pd_ = pairs(vn, 'gt_depth_m', 'pred_depth_m')
        out['depth_fit_n'] = int(len(gd))
        if len(gd) >= 3 and bool((gd > 0).all()) and float(np.std(gd)) > 1e-6:
            slope, intercept = np.polyfit(gd, pd_, 1)
            out.update(depth_fit_slope=float(slope), depth_fit_intercept_m=float(intercept),
                       depth_ratio_median=float(np.median(pd_ / gd)))
        gpx, ppx = pairs(vn, 'gt_head_vs_gt_root_px', 'pred_head_vs_pred_root_px')
        ok = gpx > 1e-6
        if int(ok.sum()) >= 3:
            out['size_ratio_median_px'] = float(np.median(ppx[ok] / gpx[ok]))
        return out

    m = dc.mean_or_none
    summary = dict(ckpt=ckpt_path, selection=sel_desc, n_images=len(indices), variants={})
    pp = np.asarray([[d['cx_over_W'], d['cy_over_H']] for d in cam_stats])
    kd = np.asarray([d['k_fov_vs_dataset_px'] for d in cam_stats])
    summary['camera'] = dict(
        principal_point_centred_frac=float((np.abs(pp - 0.5) < 0.01).all(1).mean()),
        cx_over_W_minmax=[float(pp[:, 0].min()), float(pp[:, 0].max())],
        cy_over_H_minmax=[float(pp[:, 1].min()), float(pp[:, 1].max())],
        fx_equals_fy_frac=float(np.mean([d['fx_eq_fy'] for d in cam_stats])),
        aspect_ratios=sorted(set(round(d['W'] / d['H'], 3) for d in cam_stats)),
        fov_minmax=[min(d['fov'] for d in cam_stats), max(d['fov'] for d in cam_stats)],
        k_fov_vs_dataset_px_max=float(kd.max()), k_fov_within_1px_frac=float((kd < 1.0).mean()),
        gt_j2d_inside_image_frac=m([d['gt_j2d_inside'] for d in cam_stats]),
        nonroot_translation_max=float(max(d['nonroot_translation'] for d in cam_stats)),
        gt_people=n_gt_total, gt_people_dropped_by_train_filter=n_gt_dropped)
    c = summary['camera']
    print("\n=== CAMERA / ANNOTATION CHECKS ===")
    print(f"principal point centred (|c/W-0.5|<1%) in {c['principal_point_centred_frac'] * 100:.0f}% of images; "
          f"cx/W in {c['cx_over_W_minmax']}, cy/H in {c['cy_over_H_minmax']}; fx==fy in {c['fx_equals_fy_frac'] * 100:.0f}%")
    print(f"aspect ratios {c['aspect_ratios']}; fov {c['fov_minmax'][0]:.0f}-{c['fov_minmax'][1]:.0f} deg; "
          f"K_fov == dataset K within 1px on {c['k_fov_within_1px_frac'] * 100:.0f}% (max diff {c['k_fov_vs_dataset_px_max']:.1f}px)")
    print(f"GT joints projected with the dataset K inside the {S}px frame: {c['gt_j2d_inside_image_frac'] * 100:.1f}% "
          f"(<95% with people in frame => K or pose convention is wrong)")
    print(f"non-root translation in GT poses: max {c['nonroot_translation_max']:.2e} "
          f"({'OK, factorisation valid' if c['nonroot_translation_max'] < 1e-4 else 'WARNING: factorisation assumption violated'})")
    print(f"GT people: {n_gt_total}, dropped by train.py's visibility filter: {n_gt_dropped}"
          + ("; selected-person mode: metrics/recall count only the selected people, FP counts use all people" if sel_person else ""))
    nf = {vn: det[vn]['nonfinite'] for vn in variants if det[vn]['nonfinite']}
    if nf:
        print(f"non-finite detections dropped before matching: {nf}")

    print("\n=== PER-VARIANT METRICS (matched people, pelvis-centred, mm; ratios pred/GT in %) ===")
    print(f"{'variant':<22}{'recall':>8}{'FP':>5}{'PVE':>8}{'PA-PVE':>8}{'MPJPE':>8}{'PVE_hn':>8}{'neck%':>7}{'nk/ht%':>8}{'hd/ht%':>8}")
    for vn in variants:
        d = det[vn]
        rec = 100.0 * d['matched'] / max(d['gt'], 1)
        s = dict(recall=rec, fp=d['fp'], images_without_detection=d['img_nodet'], nonfinite_detections_dropped=d['nonfinite'],
                 pve=m(col(vn, 'pve')), pa_pve=m(col(vn, 'pa_pve')), mpjpe=m(col(vn, 'mpjpe')),
                 pve_headneck=m(col(vn, 'pve_headneck')),
                 neck_len_pred_over_gt_pct=(m(col(vn, 'neck_ratio')) or float('nan')) * 100,
                 neck_over_height_pred_over_gt_pct=(m(col(vn, 'neck_over_height_ratio')) or float('nan')) * 100,
                 head_over_height_pred_over_gt_pct=(m(col(vn, 'head_over_height_ratio')) or float('nan')) * 100,
                 helper_bones_pred_err_deg=m(col(vn, 'helper_bones_pred_err_deg')),
                 gt_helper_bones_angle_deg=m(col(vn, 'gt_helper_bones_angle_deg')))
        s['rot_err_deg_by_group'] = {g: m(col(vn, f'rot_{g}')) for g in groups}
        s['rot_global_deg'] = dict(root=m(col(vn, 'rot_global_root_deg')), head=m(col(vn, 'rot_global_head_deg')),
                                   wrist=m(col(vn, 'rot_global_wrist_deg')))
        s['placement'] = {k: m(col(vn, k)) for k in ('mpjpe_abs', 'root_err_mm', 'head_err_mm', 'root_depth_err_m',
                                                     'head_px_err', 'root_px_err', 'pred_head_vs_gt_root_px',
                                                     'gt_head_vs_gt_root_px', 'pred_head_vs_pred_root_px',
                                                     'vertical_shift_px')}
        s['placement'].update(depth_fit(vn))
        if vn == CROP_VARIANT:
            # the crop camera's fov is outside the 51-110 deg training range and the depth prior was learned
            # full-frame: these placement numbers describe the crops and stay out of the paired read below
            med = lambda k: (float(np.nanmedian(col(vn, k))) if np.isfinite(col(vn, k)).any() else None)  # noqa: E731
            s['crop'] = dict(crop_factor=args.crop_factor, people=int(det[vn]['gt']), crops_without_detection=det[vn]['img_nodet'],
                             side_px_median=med('crop_side_px'), scale_vs_input_median=med('crop_scale_vs_input'),
                             fov_deg_median=med('crop_fov_deg'), person_frac_median=med('crop_person_frac'))
            s['placement_excluded_from_paired'] = ('crop camera: fov outside the training range, depth prior learned '
                                                   'full-frame; root/head px, depth error and depth fit are not comparable')
        s['shape_abs_err'] = {n: m(np.abs(col(vn, f'pheno_err_{n}'))) for n in phen}
        s['shape_bias'] = {n: m(col(vn, f'pheno_err_{n}')) for n in phen}
        s['neck_factorisation_m'] = dict(
            gt=m(col(vn, 'gt_neck_m')), pred=m(col(vn, 'pred_neck_m')),
            gt_pose_pred_shape=m(col(vn, 'neck_gtpose_predshape_m')),
            pred_pose_gt_shape=m(col(vn, 'neck_predpose_gtshape_m')),
            rest_gt_shape=m(col(vn, 'neck_rest_gtshape_m')),
            rest_pred_shape=m(col(vn, 'neck_rest_predshape_m')))
        s['rest_pose_m'] = dict(skel_height_gt_shape=m(col(vn, 'skel_height_rest_gtshape_m')),
                                skel_height_pred_shape=m(col(vn, 'skel_height_rest_predshape_m')),
                                head_gt_shape=m(col(vn, 'head_rest_gtshape_m')),
                                head_pred_shape=m(col(vn, 'head_rest_predshape_m')))
        ga, pa_ = pairs(vn, 'pheno_gt_age', 'pheno_pred_age')
        _, ph_ = pairs(vn, 'pheno_gt_age', 'pheno_pred_height')
        _, hr_ = pairs(vn, 'pheno_gt_age', 'skel_height_rest_predshape_m')
        _, hg_ = pairs(vn, 'pheno_gt_age', 'skel_height_rest_gtshape_m')
        buckets = {}
        for lo, hi, lab in AGE_BUCKETS:
            mk = (ga >= lo) & (ga < hi)
            if mk.any():
                buckets[lab] = dict(n=int(mk.sum()), gt_age=float(ga[mk].mean()), pred_age=float(pa_[mk].mean()),
                                    pred_height_pheno=float(ph_[mk].mean()) if len(ph_) == len(ga) else None,
                                    rest_skel_height_gt_m=float(hg_[mk].mean()) if len(hg_) == len(ga) else None,
                                    rest_skel_height_pred_m=float(hr_[mk].mean()) if len(hr_) == len(ga) else None)
        s['age_buckets'] = buckets
        summary['variants'][vn] = s
        print(f"{vn:<22}{rec:>7.1f}%{d['fp']:>5}{dc.fmt(s['pve']):>8}{dc.fmt(s['pa_pve']):>8}{dc.fmt(s['mpjpe']):>8}"
              f"{dc.fmt(s['pve_headneck']):>8}{dc.fmt(s['neck_len_pred_over_gt_pct']):>7}"
              f"{dc.fmt(s['neck_over_height_pred_over_gt_pct']):>8}{dc.fmt(s['head_over_height_pred_over_gt_pct']):>8}")

    print("\n=== ROTATION ERROR BY BONE GROUP (deg, active bones only; HELPER = the 88 masked bones, V0 only) ===")
    print(f"{'variant':<22}" + "".join(f"{g[:10]:>11}" for g in groups) + f"{'HELPER':>9}")
    for vn in variants:
        s = summary['variants'][vn]
        print(f"{vn:<22}" + "".join(f"{dc.fmt(s['rot_err_deg_by_group'][g]):>11}" for g in groups)
              + f"{dc.fmt(s['helper_bones_pred_err_deg']):>9}")
    print(f"  GT helper bones are on average {dc.fmt(summary['variants'][list(variants)[0]]['gt_helper_bones_angle_deg'])} deg "
          f"from identity: the irreducible floor the mask introduces (a dataset property, not a model error).")
    print("\n=== PLACEMENT (absolute, NOT pelvis-centred) ===")
    print(f"{'variant':<22}{'MPJPE_abs':>10}{'root mm':>9}{'head mm':>9}{'root dz m':>10}{'head px':>9}{'root px':>9}{'predHead-gtRoot px':>20}{'gtHead-gtRoot px':>18}{'vshift px':>10}")
    for vn in variants:
        pl = summary['variants'][vn]['placement']
        print(f"{vn:<22}{dc.fmt(pl['mpjpe_abs']):>10}{dc.fmt(pl['root_err_mm']):>9}{dc.fmt(pl['head_err_mm']):>9}{dc.fmt(pl['root_depth_err_m'], 2):>10}"
              f"{dc.fmt(pl['head_px_err']):>9}{dc.fmt(pl['root_px_err']):>9}{dc.fmt(pl['pred_head_vs_gt_root_px']):>20}{dc.fmt(pl['gt_head_vs_gt_root_px']):>18}{dc.fmt(pl['vertical_shift_px']):>10}")
    print("  read: train.py's transl/dist/heatmap targets are at the ROOT, the model anchors the HEAD (model.py:430,442).")
    print("        If 'predHead-gtRoot px' is small and 'head px' large, the head is placed where the pelvis is: a")
    print("        training-target bug (supervision) that pelvis-centred PVE cannot see. If 'head px' is small, the 2D")
    print("        losses compensate and the bug costs only depth/offset noise; 'vshift' > 0 = predicted body too low.")
    print(f"{'variant':<22}{'depth slope':>12}{'intercept m':>12}{'depth ratio':>12}{'size ratio':>11}{'n':>5}")
    for vn in variants:
        pl = summary['variants'][vn]['placement']
        print(f"{vn:<22}{dc.fmt(pl['depth_fit_slope'], 2):>12}{dc.fmt(pl['depth_fit_intercept_m'], 2):>12}"
              f"{dc.fmt(pl['depth_ratio_median'], 2):>12}{dc.fmt(pl['size_ratio_median_px'], 2):>11}{pl['depth_fit_n']:>5}")
    print("  read: pred root depth = slope*GT + intercept over matched people (needs >= 3); depth ratio = median pred/GT")
    print("        root depth; size ratio = median pred/GT head-to-root pixel distance. Learned depth: slope ~1, intercept ~0,")
    print("        ratios ~1. The baseline checkpoint anny_s2_shape_v5/00099 sits at slope 0.33, intercept 2.36 m, size 0.67.")
    if CROP_VARIANT in summary['variants']:
        cr = summary['variants'][CROP_VARIANT]['crop']
        print(f"  {CROP_VARIANT}: {cr['people']} crops (factor {cr['crop_factor']}, {cr['crops_without_detection']} without a detection), "
              f"median side {dc.fmt(cr['side_px_median'], 0)} px = x{dc.fmt(cr['scale_vs_input_median'], 2)} over the {S}px input, "
              f"median fov {dc.fmt(cr['fov_deg_median'], 0)} deg (training 51-110), person fills {dc.fmt((cr['person_frac_median'] or 0) * 100, 0)}% of the crop.")
        print("        Its placement/depth rows above are NOT comparable with V1 (different camera, depth prior learned full-frame)")
        print("        and are excluded from the paired comparison; only pelvis-centred, rotation and shape metrics are paired.")
    print("\n=== GLOBAL (camera-frame) ORIENTATION ERROR of end effectors, deg ===")
    for vn in variants:
        gd = summary['variants'][vn]['rot_global_deg']
        print(f"{vn:<22} root {dc.fmt(gd['root']):>6}  head {dc.fmt(gd['head']):>6}  wrist {dc.fmt(gd['wrist']):>6}")
    print("  read: local neck/head errors far above the GLOBAL head error => the neck chain is redundant (several local")
    print("        rotations give the same head placement) and the per-bone rotation loss is a weak signal there;")
    print("        a large GLOBAL head error means the face really points the wrong way.")

    print("\n=== SHAPE: bias (pred - GT) / abs error, per phenotype ===")
    print(f"{'variant':<22}" + "".join(f"{n[:9]:>13}" for n in phen))
    for vn in variants:
        s = summary['variants'][vn]
        print(f"{vn:<22}" + "".join(f"{(s['shape_bias'][n] if s['shape_bias'][n] is not None else float('nan')):>+6.3f}/"
                                    f"{dc.fmt(s['shape_abs_err'][n], 3)}" for n in phen))

    print("\n=== NECK LENGTH FACTORISATION (metres, mean over people) ===")
    for vn in variants:
        f = summary['variants'][vn]['neck_factorisation_m']; h = summary['variants'][vn]['rest_pose_m']
        print(f"{vn:<22} GT {dc.fmt(f['gt'], 4)} | pred {dc.fmt(f['pred'], 4)} | GTpose+PREDshape {dc.fmt(f['gt_pose_pred_shape'], 4)} "
              f"| PREDpose+GTshape {dc.fmt(f['pred_pose_gt_shape'], 4)} | rest: GTshape {dc.fmt(f['rest_gt_shape'], 4)} PREDshape {dc.fmt(f['rest_pred_shape'], 4)}"
              f" | rest skel-height GT {dc.fmt(h['skel_height_gt_shape'], 3)} PRED {dc.fmt(h['skel_height_pred_shape'], 3)}"
              f" | rest head GT {dc.fmt(h['head_gt_shape'], 3)} PRED {dc.fmt(h['head_pred_shape'], 3)}")
    print("  read: if GTpose+PREDshape ~= pred and PREDpose+GTshape ~= GT, the neck error is in the SHAPE head;")
    print("        if the reverse, it is in the predicted neck ROTATIONS; if pred ~= GT here but PVE_hn is high or the")
    print("        renders look long-necked, it is a SURFACE (skinning) or rendering effect, not a skeleton error.")

    print("\n=== AGE BUCKETS (GT age -> mean predicted age; rest skeletal height GT vs pred) ===")
    for vn in variants:
        b = summary['variants'][vn]['age_buckets']
        print(f"{vn:<22} " + "  ".join(f"[{k}] n={v['n']} gt={v['gt_age']:.2f} pred={v['pred_age']:.2f}"
                                        f" h={dc.fmt(v['rest_skel_height_gt_m'], 2)}->{dc.fmt(v['rest_skel_height_pred_m'], 2)}"
                                        for k, v in b.items()))

    # --- per-category tables (manifest runs) ---------------------------------------------------
    if tagmap:
        TAGM = ['pve', 'pa_pve', 'pve_headneck', 'neck_ratio', 'pheno_err_age', 'pheno_err_height',
                'pheno_err_proportions', 'pve_predpose_gtshape', 'pve_gtpose_predshape', 'rot_neck', 'rot_fingers',
                'rot_global_head_deg']
        all_tags = sorted({t for ts in tagmap.values() for t in ts})
        summary['per_tag'] = {}
        for vn in variants:
            summary['per_tag'][vn] = {}
            for t in all_tags:
                rr = [r for r in rows if r['variant'] == vn and t in r['tags'].split('|')]
                if rr:
                    d_ = {k: m([r.get(k, np.nan) for r in rr]) for k in TAGM}
                    d_['n'] = len(rr)
                    d_['shape_swap_gain_mm'] = m([r['pve'] - r['pve_predpose_gtshape'] for r in rr])
                    summary['per_tag'][vn][t] = d_
        ref_t = 'V1_eval_masked' if 'V1_eval_masked' in variants else list(variants)[0]
        print(f"\n=== PER CATEGORY ({ref_t}; mm, ratios pred/GT, phenotype errors signed) ===")
        print(f"{'tag':<8}{'n':>4}" + "".join(f"{k[:12]:>14}" for k in TAGM) + f"{'swap_gain':>11}")
        for t, d_ in summary['per_tag'][ref_t].items():
            print(f"{t:<8}{d_['n']:>4}" + "".join(f"{dc.fmt(d_[k], 3 if ('ratio' in k or 'pheno' in k) else 1):>14}" for k in TAGM)
                  + f"{dc.fmt(d_['shape_swap_gain_mm']):>11}")
        print("  swap_gain = PVE(pred) - PVE(pred pose + GT shape): how much of the error is the SHAPE head's; for")
        print("  children a large gain with pheno_err_age > 0 means the shape head makes them adults (supervision/")
        print("  generalisation cause), a small gain means the pose/decoding path is responsible.")

    # --- paired comparison: same (image, GT person) matched under EVERY variant ----------------
    # The per-variant means above are over each variant's own matched cohort; thresholds, K and the
    # mask can change who gets matched, so differences between variants are only attributable to the
    # toggle on the intersection.
    PAIRED = ['pve', 'pa_pve', 'mpjpe', 'pve_headneck', 'neck_ratio', 'neck_over_height_ratio',
              'head_over_height_ratio', 'rot_neck', 'rot_fingers', 'pheno_err_age', 'pheno_err_height',
              'pheno_err_proportions']
    if args.person_crop:      # the hand read needs the wrist and the camera-frame end-effector errors as well
        PAIRED = PAIRED + ['rot_wrist', 'rot_global_head_deg', 'rot_global_wrist_deg']
    by = {vn: {(r['idx'], r['gid']): r for r in rows if r['variant'] == vn} for vn in variants}
    common = sorted(set.intersection(*[set(d) for d in by.values()])) if by else []
    ref = 'V1_eval_masked' if 'V1_eval_masked' in by else list(variants)[0]
    paired = dict(n_common=len(common), reference=ref, per_variant={}, delta_vs_reference={})
    if common:
        for vn in variants:
            paired['per_variant'][vn] = {k: m([by[vn][c].get(k, np.nan) for c in common]) for k in PAIRED}
            paired['delta_vs_reference'][vn] = {
                k: m([by[vn][c].get(k, np.nan) - by[ref][c].get(k, np.nan) for c in common]) for k in PAIRED}
    summary['paired'] = paired
    print(f"\n=== PAIRED COMPARISON on the {len(common)} (image, person) pairs matched under ALL variants; delta vs {ref} ===")
    if common:
        print(f"{'variant':<22}" + "".join(f"{k[:11]:>13}" for k in PAIRED))
        for vn in variants:
            print(f"{vn:<22}" + "".join(f"{dc.fmt(paired['per_variant'][vn][k], 3 if 'ratio' in k or 'pheno' in k else 1):>13}" for k in PAIRED))
        for vn in variants:
            if vn != ref:
                print(f"{'  d(' + vn[:15] + ')':<22}" + "".join(f"{dc.fmt(paired['delta_vs_reference'][vn][k], 3 if 'ratio' in k or 'pheno' in k else 1, sign=True):>13}" for k in PAIRED))
        print("  read: attribute an effect to a toggle (mask, fp16/K, relax_hands) from THESE rows, not from the per-variant table.")
    else:
        print("  no (image, person) pair was matched under every variant; variant differences are not attributable.")

    # --- V4 person crop vs V1 on THEIR OWN intersection (the all-variant one above shrinks with every variant) ----
    if args.person_crop and CROP_VARIANT in by and 'V1_eval_masked' in by:
        ref_c, v4 = 'V1_eval_masked', CROP_VARIANT
        CROP_KEYS = ['rot_fingers', 'rot_wrist', 'rot_global_wrist_deg', 'rot_global_head_deg', 'pve', 'pa_pve',
                     'mpjpe', 'pve_headneck', 'pheno_err_age', 'pheno_err_height']
        cc = sorted(set(by[v4]) & set(by[ref_c]))
        d4 = det[v4]
        crop_cmp = dict(n_pairs=len(cc), reference=ref_c, crop_factor=args.crop_factor,
                        recall=100.0 * d4['matched'] / max(d4['gt'], 1), crops_without_detection=d4['img_nodet'],
                        excluded='placement and depth metrics (root/head px errors, root_depth_err_m, pred_depth_m, depth '
                                 'fit): the crop camera has a different fov and the depth prior was learned full-frame',
                        per_variant={}, delta_vs_reference={}, by_magnification={})
        vlist = [ref_c, v4] + [v for v in ('V3_demo_relaxhands',) if v in by]    # V3 = the relaxed-finger floor
        if cc:
            for vn in vlist:
                sub = [c for c in cc if c in by[vn]]
                crop_cmp['per_variant'][vn] = {k: m([by[vn][c].get(k, np.nan) for c in sub]) for k in CROP_KEYS}
                crop_cmp['per_variant'][vn]['n'] = len(sub)
            crop_cmp['delta_vs_reference'][v4] = {
                k: m([by[v4][c].get(k, np.nan) - by[ref_c][c].get(k, np.nan) for c in cc]) for k in CROP_KEYS}
            mags = np.asarray([by[v4][c].get('crop_scale_vs_input', np.nan) for c in cc], dtype=np.float64)
            for lo_, hi_, lab in ((0.0, 1.5, 'x<1.5'), (1.5, 2.5, 'x1.5-2.5'), (2.5, np.inf, 'x>=2.5')):
                sel_c = [c for c, mg in zip(cc, mags) if lo_ <= mg < hi_]
                if sel_c:
                    crop_cmp['by_magnification'][lab] = dict(
                        n=len(sel_c), scale_mean=m([by[v4][c]['crop_scale_vs_input'] for c in sel_c]),
                        rot_fingers_ref=m([by[ref_c][c].get('rot_fingers', np.nan) for c in sel_c]),
                        rot_fingers_crop=m([by[v4][c].get('rot_fingers', np.nan) for c in sel_c]),
                        pve_ref=m([by[ref_c][c]['pve'] for c in sel_c]), pve_crop=m([by[v4][c]['pve'] for c in sel_c]))
        summary['paired_crop'] = crop_cmp
        print(f"\n=== {v4} vs {ref_c} on the {len(cc)} (image, person) pairs matched under BOTH (recall of {v4}: "
              f"{crop_cmp['recall']:.1f}%, {d4['img_nodet']} crops without a detection) ===")
        if cc:
            print(f"{'variant':<22}{'n':>5}" + "".join(f"{k[:13]:>15}" for k in CROP_KEYS))
            for vn in vlist:
                pv = crop_cmp['per_variant'][vn]
                print(f"{vn:<22}{pv['n']:>5}" + "".join(f"{dc.fmt(pv[k], 3 if 'pheno' in k else 1):>15}" for k in CROP_KEYS))
            dv = crop_cmp['delta_vs_reference'][v4]
            print(f"{'  d(' + v4[:15] + ')':<27}" + "".join(f"{dc.fmt(dv[k], 3 if 'pheno' in k else 1, sign=True):>15}" for k in CROP_KEYS))
            print(f"  by magnification over the {S}px input (crop side vs full frame):")
            for lab, b in crop_cmp['by_magnification'].items():
                print(f"    {lab:<9} n={b['n']:<4} mean x{dc.fmt(b['scale_mean'], 2)}  rot_fingers {dc.fmt(b['rot_fingers_ref'])} -> "
                      f"{dc.fmt(b['rot_fingers_crop'])} deg   PVE {dc.fmt(b['pve_ref'])} -> {dc.fmt(b['pve_crop'])} mm")
        print(f"  EXCLUDED for {v4}: {crop_cmp['excluded']}.")
        print("  read: d(rot_fingers) says whether MAGNIFICATION alone helps the fingers of a head trained at ~1-3 tokens per")
        print("        hand. Fingers below the relaxed floor (V3 rot_fingers on the same pairs; 15.1 deg on the findings' holdout)")
        print("        with PVE / PA-PVE / PVE_hn not worse => resolution is the bottleneck and a crop-based second pass is the")
        print("        cheapest next step. Neutral or worse at every magnification => supervision is (docs/diagnostics_findings.md,")
        print("        Fingers: |grad| 4.7, 6D conditioning 0.25-0.30), and the training-side experiment decides.")

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(dc.json_safe(summary), f, indent=2, allow_nan=False)
    print(f"\nwrote {args.out}/summary.json, per_person.csv and {min(args.n_vis, len(indices))} panels in vis/")


if __name__ == '__main__':
    main()
