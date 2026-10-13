"""
E1 / E3 -- trace held-out AnnyOne samples through GT and prediction paths and
measure WHERE the neck, finger and body-shape errors come from.

For every selected image this script:
  1. loads it exactly as training does (datasets/annyone.py + bedlam.collate_fn);
  2. builds the GT mesh from the dataset's anny_pose / anny_shape through the
     model's own Anny body model (the train.py prepare_gt path) and projects it
     with the dataset K, so a GT overlay that does not sit on the person means
     the camera/pose convention is wrong BEFORE any network is involved;
  3. runs the checkpoint under several inference variants on the SAME tensor:
       V0  eval-faithful : fp32, dataset K, helper bones UNMASKED  (train.py evaluate())
       V1  V0 + helper-bone mask                                    (what every demo renders)
       V2  demo-faithful : fp16 autocast, fov-derived K, mask      (demo.py in-domain path)
       V3  V2 + --relax_hands 1                                      (real-photo demo setting)
  4. matches detections to GT people (same greedy 2D matcher as train.py) and
     reports, per variant: recall/FP, PVE, PA-PVE, MPJPE, per-bone-group rotation
     error, per-phenotype shape error and bias, neck length and neck/height ratio
     against GT, and a shape-vs-pose FACTORISATION of the neck length
     (GT pose + predicted shape, predicted pose + GT shape);
  5. writes per-person CSV, a summary JSON and overlay panels
     [image | GT mesh | V1 | V2] for the first --n_vis samples.

Nothing is trained or written to the checkpoint. ~5 min on one GPU for 100 images.

USAGE (repo root, multihmr env, GPU node):
    python scripts/diagnostics/diag_holdout_trace.py \
        --ckpt logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt \
        --n 100 --out results/diagnostics/holdout_v5

    # E3: the youngest people in the first --scan training samples (IN-SAMPLE)
    python scripts/diagnostics/diag_holdout_trace.py --ckpt ... \
        --select youngest --scan 20000 --n 50 --out results/diagnostics/young_v5
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402  (sets PYOPENGL_PLATFORM, sys.path)

import numpy as np   # noqa: E402
import torch         # noqa: E402
import roma          # noqa: E402
from PIL import Image  # noqa: E402

from datasets.annyone import AnnyOne          # noqa: E402
from datasets.bedlam import collate_fn        # noqa: E402
from utils import denormalize_rgb             # noqa: E402
from utils.training import match_2d_greedy    # noqa: E402

VARIANTS = {
    # name: (autocast fp16, K source, mask_helper_joints, relax_hands, det_thresh key)
    'V0_eval_unmasked': dict(amp=False, K='dataset', mask=False, relax=False, thr='eval'),
    'V1_eval_masked':   dict(amp=False, K='dataset', mask=True,  relax=False, thr='eval'),
    'V2_demo':          dict(amp=True,  K='fov',     mask=True,  relax=False, thr='demo'),
    'V3_demo_relaxhands': dict(amp=True, K='fov',    mask=True,  relax=True,  thr='demo'),
}


def select_indices(args, pkls):
    n_total = len(pkls)
    if args.select == 'holdout':
        pool = dc.holdout_indices(n_total)
        return pool[:args.n], 'HOLDOUT (never trained on)'
    if args.select == 'indices':
        idx = [int(s) for s in args.indices.split(',') if s.strip()]
        return idx, 'explicit indices'
    if args.select == 'youngest':
        scan = min(args.scan, n_total - dc.N_HOLDOUT)
        ages = []
        t0 = time.time()
        for i in range(scan):
            s = dc.shapes_from_pkl(dc.load_pkl(pkls[i]))
            if s is None or len(s) == 0:
                continue
            ages.append((float(s[:, 1].min()), i))
            if i % 5000 == 0 and i:
                print(f"  scanned {i}/{scan} ({time.time() - t0:.0f}s)", flush=True)
        ages.sort()
        idx = [i for _, i in ages[:args.n]]
        print(f"[select] youngest {len(idx)} of the first {scan} TRAINING samples "
              f"(min GT age {ages[0][0]:.3f} .. {ages[len(idx) - 1][0]:.3f}). "
              f"These were SEEN in training: this tests representation, not generalisation.")
        return idx, f'youngest-{len(idx)} of first {scan} training samples (IN-SAMPLE)'
    raise ValueError(args.select)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='path, models/multiHMR name, or run_name:epoch')
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--select', default='holdout', choices=['holdout', 'youngest', 'indices'])
    ap.add_argument('--scan', type=int, default=20000, help='--select youngest: how many training pkls to scan')
    ap.add_argument('--indices', default='', help='--select indices: comma list of dataset indices')
    ap.add_argument('--out', default='results/diagnostics/holdout_trace')
    ap.add_argument('--n_vis', type=int, default=12)
    ap.add_argument('--variants', default=','.join(VARIANTS.keys()))
    ap.add_argument('--det_thresh_eval', type=float, default=0.2, help='train.py default')
    ap.add_argument('--det_thresh_demo', type=float, default=0.3, help='run_demo_syn.sh')
    ap.add_argument('--nms_kernel_size', type=int, default=3)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    variants = {k: VARIANTS[k] for k in args.variants.split(',') if k in VARIANTS}

    ckpt_path = dc.resolve_ckpt(args.ckpt)
    model, ckpt_args = dc.load_model(ckpt_path, device)
    bm = model.body_model
    names = list(bm.bone_labels)
    phen = list(bm.phenotype_labels)
    groups = dc.bone_groups(names)
    chain = dc.neck_chain(names)
    useful = model.useful_rotmat.reshape(-1).bool().cpu()        # 1 = active bone
    faces = dc.faces_tri(bm)
    print(f"[rig] {len(names)} bones, {int(useful.sum())} active / {int((~useful).sum())} masked; "
          f"phenotypes={phen}")
    print(f"[rig] neck chain: {[names[i] for i in chain]}")
    for g, idx in groups.items():
        act = int(useful[idx].sum())
        print(f"[rig] group {g:<18s} {len(idx):3d} bones ({act} active) e.g. {[names[i] for i in idx[:4]]}")
    if phen != dc.PHENOTYPES:
        print(f"[WARN] body model phenotype order {phen} != assumed dataset order {dc.PHENOTYPES}")

    ds = AnnyOne(data_folder=args.data_folder, img_size=model.img_size)
    pkls = ds.pkl_files
    indices, sel_desc = select_indices(args, pkls)
    print(f"[data] {len(pkls)} samples; evaluating {len(indices)} = {sel_desc}")

    rows = []                 # per matched person
    det = {k: dict(gt=0, matched=0, fp=0, img_nodet=0) for k in variants}
    pp_stats = []             # principal point / aspect checks
    gt_inside = []
    t0 = time.time()

    for n_done, idx in enumerate(indices):
        img_arr, annot = ds[idx]
        x, y = collate_fn([(img_arr, annot)])
        x = x.to(device)
        K = y['K'].to(device)                                      # [1,3,3] dataset K
        valid = y['valid_humans'][0].bool()
        nhv = int(valid.sum())
        if nhv == 0:
            continue
        pose = y['anny_pose'][0][valid][:, :163].float().to(device)   # [nhv,163,4,4]
        shape_gt = y['anny_shape'][0][valid].float().to(device)       # [nhv,11]

        # --- camera sanity from the raw annotation ---------------------------
        raw = dc.load_pkl(pkls[idx])
        with Image.open(dc.image_path_for(pkls[idx])) as im:
            W, H = im.size
        pp = np.asarray(raw['princpt'], dtype=np.float64).reshape(-1)[:2]
        pp_stats.append(dict(idx=idx, W=W, H=H, cx_over_W=pp[0] / W, cy_over_H=pp[1] / H,
                             focal=float(np.asarray(raw['focal']).reshape(-1)[0]),
                             fov=dc.fov_from_K(K[0].cpu(), model.img_size)))

        # --- GT through the body model (train.py prepare_gt path) ------------
        gt_v, gt_j = dc.run_body_model(bm, pose, shape_gt)
        gt_R = dc.orthonormalize(pose[:, :, :3, :3])
        gt_j2d = dc.project(gt_j, K)
        inside = ((gt_j2d >= 0) & (gt_j2d <= model.img_size)).all(-1).float().mean().item()
        gt_inside.append(inside)
        gt_neck = dc.chain_length(gt_j, chain)
        gt_height = dc.body_height(gt_j, names)
        gt_headsz = dc.head_size(gt_v, gt_j, names)
        K_fov = dc.K_from_fov(dc.fov_from_K(K[0].cpu(), model.img_size), model.img_size, device)
        k_diff = float((K_fov - K).abs().max())

        vis_panels = {}
        for vname, cfg in variants.items():
            dc.set_inference_toggles(model, mask_helper_joints=cfg['mask'], relax_hands=cfg['relax'])
            K_v = K if cfg['K'] == 'dataset' else K_fov
            thr = args.det_thresh_eval if cfg['thr'] == 'eval' else args.det_thresh_demo
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg['amp']):
                    preds = model(x, is_training=False, K=K_v, det_thresh=thr,
                                  nms_kernel_size=args.nms_kernel_size)
            det[vname]['gt'] += nhv
            if len(preds) == 0:
                det[vname]['img_nodet'] += 1
                continue
            pj2d = np.asarray([p['j2d'].float().cpu().numpy()[:163] for p in preds])
            gj2d = gt_j2d.cpu().numpy()
            best, fps, misses = match_2d_greedy(pj2d, gj2d, np.ones_like(gj2d[..., 0]).astype(bool))
            det[vname]['fp'] += len(fps)
            det[vname]['matched'] += len(best)

            for (pid, gid) in best:
                p = preds[pid]
                v_hat = p['v3d'].float(); j_hat = p['j3d'].float()
                pelv = gt_j[gid, 0:1]; pelv_hat = j_hat[0:1]
                v_ctx, vh_ctx = gt_v[gid] - pelv, v_hat - pelv_hat
                pve = ((v_ctx - vh_ctx).norm(dim=-1) * 1000).mean().item()
                R, t, s = roma.rigid_points_registration(vh_ctx, v_ctx, compute_scaling=True)
                pa = s * (R.reshape(1, 3, 3) @ vh_ctx.reshape(-1, 3, 1)).reshape(-1, 3) + t
                pa_pve = ((v_ctx - pa).norm(dim=-1) * 1000).mean().item()
                mpjpe = (((gt_j[gid] - pelv) - (j_hat - pelv_hat)).norm(dim=-1) * 1000).mean().item()
                # per-vertex error restricted to head/neck vertices (PVE hides the neck)
                near_head = ((gt_v[gid] - gt_j[gid, chain[-1]]).norm(dim=-1) < 0.35)
                pve_headneck = ((v_ctx - vh_ctx).norm(dim=-1)[near_head] * 1000).mean().item() if near_head.any() else float('nan')

                # rotations
                R_hat = roma.rotvec_to_rotmat(p['rotvec'].float().reshape(-1, 3)).reshape(163, 3, 3)
                ang = dc.geodesic_deg(R_hat, gt_R[gid]).cpu()          # [163]
                grp_err = {}
                for g, gi in groups.items():
                    gi_t = torch.tensor(gi)
                    act = gi_t[useful[gi_t]]
                    grp_err[g] = float(ang[act].mean()) if len(act) else float('nan')
                masked_err = float(ang[~useful].mean()) if (~useful).any() else float('nan')

                # shape
                sh_hat = p['shape'].float().reshape(-1)
                sh_gt = shape_gt[gid]
                sh_err = (sh_hat - sh_gt).cpu().numpy()

                # neck / height / head size
                neck_hat = dc.chain_length(j_hat.unsqueeze(0), chain)[0].item()
                height_hat = dc.body_height(j_hat.unsqueeze(0), names)[0].item()
                headsz_hat = dc.head_size(v_hat.unsqueeze(0), j_hat.unsqueeze(0), names)[0].item()

                # factorisation: which factor moves the neck length?
                P_gt = pose[gid:gid + 1]
                P_hat = dc.rotmat_to_homogeneous(R_hat.unsqueeze(0))
                _, j_gp_ps = dc.run_body_model(bm, P_gt, sh_hat.unsqueeze(0))      # GT pose + pred shape
                _, j_pp_gs = dc.run_body_model(bm, P_hat, sh_gt.unsqueeze(0))      # pred pose + GT shape
                I = dc.identity_pose(1, 163, device)
                _, j_rest_gs = dc.run_body_model(bm, I, sh_gt.unsqueeze(0))        # rest pose, GT shape
                _, j_rest_ps = dc.run_body_model(bm, I, sh_hat.unsqueeze(0))       # rest pose, pred shape

                row = dict(variant=vname, idx=idx, gid=gid, score=float(p['scores']),
                           pve=pve, pa_pve=pa_pve, mpjpe=mpjpe, pve_headneck=pve_headneck,
                           masked_bones_err=masked_err,
                           gt_neck=gt_neck[gid].item(), pred_neck=neck_hat,
                           gt_height=gt_height[gid].item(), pred_height=height_hat,
                           gt_headsize=gt_headsz[gid].item(), pred_headsize=headsz_hat,
                           neck_gtpose_predshape=dc.chain_length(j_gp_ps, chain)[0].item(),
                           neck_predpose_gtshape=dc.chain_length(j_pp_gs, chain)[0].item(),
                           neck_rest_gtshape=dc.chain_length(j_rest_gs, chain)[0].item(),
                           neck_rest_predshape=dc.chain_length(j_rest_ps, chain)[0].item(),
                           height_rest_gtshape=dc.body_height(j_rest_gs, names)[0].item(),
                           height_rest_predshape=dc.body_height(j_rest_ps, names)[0].item(),
                           gt_depth=float(gt_j[gid, 0, 2]), pred_depth=float(j_hat[0, 2]),
                           K_fov_vs_dataset_maxdiff=k_diff)
                for g, e in grp_err.items():
                    row[f'rot_{g}'] = e
                for i, nme in enumerate(phen):
                    row[f'gt_{nme}'] = float(sh_gt[i]); row[f'pred_{nme}'] = float(sh_hat[i]); row[f'err_{nme}'] = float(sh_err[i])
                rows.append(row)

            if n_done < args.n_vis and vname in ('V1_eval_masked', 'V2_demo'):
                vis_panels[vname] = [preds[pid]['v3d'].float().cpu().numpy() for pid, _ in best]

        # --- overlay panel -----------------------------------------------------
        if n_done < args.n_vis:
            img_u8 = denormalize_rgb(x[0].float().cpu().numpy())
            Kc = K[0].cpu().numpy()
            panels = [dc.label(img_u8, f"idx {idx}")]
            g_ov = dc.render_overlay(img_u8, [v.cpu().numpy() for v in gt_v], faces, Kc)
            panels.append(dc.label(g_ov if g_ov is not None else dc.draw_points(img_u8, gt_j2d.cpu().numpy()), "GT (dataset pose+shape)"))
            for vname in ('V1_eval_masked', 'V2_demo'):
                if vname in vis_panels:
                    ov = dc.render_overlay(img_u8, vis_panels[vname], faces, Kc)
                    panels.append(dc.label(ov if ov is not None else img_u8, vname))
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
        a = np.asarray([r[k] for r in rows if r['variant'] == vn and r.get(k) is not None], dtype=np.float64)
        a = a[np.isfinite(a)]
        return a

    summary = dict(ckpt=ckpt_path, selection=sel_desc, n_images=len(indices), variants={})
    pp = np.asarray([[d['cx_over_W'], d['cy_over_H']] for d in pp_stats])
    summary['camera'] = dict(
        principal_point_centred_frac=float((np.abs(pp - 0.5) < 0.01).all(1).mean()),
        cx_over_W_minmax=[float(pp[:, 0].min()), float(pp[:, 0].max())],
        cy_over_H_minmax=[float(pp[:, 1].min()), float(pp[:, 1].max())],
        aspect_ratios=sorted(set(round(d['W'] / d['H'], 3) for d in pp_stats)),
        fov_minmax=[min(d['fov'] for d in pp_stats), max(d['fov'] for d in pp_stats)],
        gt_j2d_inside_image_frac=float(np.mean(gt_inside)))
    print("\n=== CAMERA / GT PROJECTION ===")
    print(f"principal point centred (|c/W-0.5|<1%) in {summary['camera']['principal_point_centred_frac'] * 100:.0f}% of images; "
          f"cx/W in {summary['camera']['cx_over_W_minmax']}, cy/H in {summary['camera']['cy_over_H_minmax']}")
    print(f"aspect ratios {summary['camera']['aspect_ratios']}; fov {summary['camera']['fov_minmax'][0]:.0f}-{summary['camera']['fov_minmax'][1]:.0f} deg")
    print(f"GT joints projected with the dataset K fall inside the 672px frame: {np.mean(gt_inside) * 100:.1f}% "
          f"(<95% with people in frame => K or pose convention is wrong)")

    print("\n=== PER-VARIANT METRICS (matched people, pelvis-centred, mm) ===")
    hdr = f"{'variant':<22}{'recall':>8}{'FP':>5}{'PVE':>8}{'PA-PVE':>8}{'MPJPE':>8}{'PVE_hn':>8}{'neck%':>7}{'nk/ht%':>8}{'head%':>7}"
    print(hdr)
    for vn in variants:
        d = det[vn]
        rec = 100.0 * d['matched'] / max(d['gt'], 1)
        gn, pn = col(vn, 'gt_neck'), col(vn, 'pred_neck')
        gh, ph = col(vn, 'gt_height'), col(vn, 'pred_height')
        ghs, phs = col(vn, 'gt_headsize'), col(vn, 'pred_headsize')
        neck_ratio = float(np.mean(pn / gn)) * 100 if len(gn) else float('nan')
        nh_ratio = float(np.mean((pn / ph) / (gn / gh))) * 100 if len(gn) else float('nan')
        head_ratio = float(np.mean(phs / ghs)) * 100 if len(ghs) else float('nan')
        s = dict(recall=rec, fp=d['fp'], images_without_detection=d['img_nodet'],
                 pve=float(col(vn, 'pve').mean()), pa_pve=float(col(vn, 'pa_pve').mean()),
                 mpjpe=float(col(vn, 'mpjpe').mean()), pve_headneck=float(col(vn, 'pve_headneck').mean()),
                 neck_len_pred_over_gt_pct=neck_ratio, neck_over_height_pred_over_gt_pct=nh_ratio,
                 head_size_pred_over_gt_pct=head_ratio,
                 masked_bones_rot_err_deg=float(col(vn, 'masked_bones_err').mean()))
        s['rot_err_deg_by_group'] = {g: float(col(vn, f'rot_{g}').mean()) for g in groups}
        s['shape_abs_err'] = {n: float(np.abs(col(vn, f'err_{n}')).mean()) for n in phen}
        s['shape_bias'] = {n: float(col(vn, f'err_{n}').mean()) for n in phen}
        s['neck_factorisation_m'] = dict(
            gt=float(gn.mean()), pred=float(pn.mean()),
            gt_pose_pred_shape=float(col(vn, 'neck_gtpose_predshape').mean()),
            pred_pose_gt_shape=float(col(vn, 'neck_predpose_gtshape').mean()),
            rest_gt_shape=float(col(vn, 'neck_rest_gtshape').mean()),
            rest_pred_shape=float(col(vn, 'neck_rest_predshape').mean()))
        s['height_rest_m'] = dict(gt_shape=float(col(vn, 'height_rest_gtshape').mean()),
                                  pred_shape=float(col(vn, 'height_rest_predshape').mean()))
        # age buckets
        ga, pa_ = col(vn, 'gt_age'), col(vn, 'pred_age')
        buckets = {}
        for lo, hi in [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)]:
            m = (ga >= lo) & (ga < hi)
            if m.any():
                buckets[f'{lo:.2f}-{hi:.2f}'] = dict(n=int(m.sum()), gt_age=float(ga[m].mean()),
                                                     pred_age=float(pa_[m].mean()),
                                                     pred_height_pheno=float(col(vn, 'pred_height')[m].mean()) if False else None)
        s['age_buckets'] = buckets
        summary['variants'][vn] = s
        print(f"{vn:<22}{rec:>7.1f}%{d['fp']:>5}{s['pve']:>8.1f}{s['pa_pve']:>8.1f}{s['mpjpe']:>8.1f}"
              f"{s['pve_headneck']:>8.1f}{neck_ratio:>7.1f}{nh_ratio:>8.1f}{head_ratio:>7.1f}")

    print("\n=== ROTATION ERROR BY BONE GROUP (deg, active bones only) ===")
    print(f"{'variant':<22}" + "".join(f"{g[:10]:>11}" for g in groups) + f"{'MASKED':>9}")
    for vn in variants:
        s = summary['variants'][vn]
        print(f"{vn:<22}" + "".join(f"{s['rot_err_deg_by_group'][g]:>11.1f}" for g in groups)
              + f"{s['masked_bones_rot_err_deg']:>9.1f}")

    print("\n=== SHAPE: bias (pred - GT) / abs error, per phenotype ===")
    print(f"{'variant':<22}" + "".join(f"{n[:9]:>13}" for n in phen))
    for vn in variants:
        s = summary['variants'][vn]
        print(f"{vn:<22}" + "".join(f"{s['shape_bias'][n]:>+6.3f}/{s['shape_abs_err'][n]:.3f}" for n in phen))

    print("\n=== NECK LENGTH FACTORISATION (metres, mean over people) ===")
    for vn in variants:
        f = summary['variants'][vn]['neck_factorisation_m']; h = summary['variants'][vn]['height_rest_m']
        print(f"{vn:<22} GT {f['gt']:.4f} | pred {f['pred']:.4f} | GTpose+PREDshape {f['gt_pose_pred_shape']:.4f} "
              f"| PREDpose+GTshape {f['pred_pose_gt_shape']:.4f} | rest: GTshape {f['rest_gt_shape']:.4f} PREDshape {f['rest_pred_shape']:.4f} "
              f"| rest height GT {h['gt_shape']:.3f} PRED {h['pred_shape']:.3f}")
    print("  read: if GTpose+PREDshape ~= pred and PREDpose+GTshape ~= GT, the neck error is in the SHAPE head;")
    print("        if the reverse, it is in the predicted neck ROTATIONS; if pred ~= GT here but the renders look")
    print("        long-necked, the problem is in the rendering/placement path, not the model.")

    print("\n=== AGE BUCKETS (GT age -> mean predicted age) ===")
    for vn in variants:
        b = summary['variants'][vn]['age_buckets']
        print(f"{vn:<22} " + "  ".join(f"[{k}] n={v['n']} gt={v['gt_age']:.2f} pred={v['pred_age']:.2f}" for k, v in b.items()))

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}/summary.json, per_person.csv and {min(args.n_vis, len(indices))} panels in vis/")


if __name__ == '__main__':
    main()
