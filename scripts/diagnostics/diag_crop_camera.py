"""
E4 -- does the wrong crop camera (demo.py detector path) change the predicted
BODY SHAPE, or only the placement?

demo.py detector_guided_inference() cuts a square crop of side S around each
detected person, resizes it to 672 and hands the network
    K = get_camera_parameters(672, fov=--fov)            (demo.py:366)
i.e. the FULL image's field of view. The crop actually has focal
    f_crop = f_full * 672 / S     and principal point  (W/2 - left, H/2 - top) * 672 / S.
K is not just used for placement: it enters the network through
model.embedd_camera(), so a wrong K can change pose/shape predictions. Children
are the obvious victim: the only cue separating a child from a far-away adult
is absolute scale, which comes from K and depth.

This script reproduces the demo crops (same S formula, same clamp) and runs the
checkpoint with the demo K and with the corrected K, fp32, on the same crop.
For each person it prints the predicted phenotypes (age, height, proportions),
body height, neck/height ratio and depth under both cameras, and the mean
absolute change. Optionally saves crop overlays for both.

Caveat (unverified): the corrected K has a NARROWER fov than anything in
AnnyOne (51-118 deg), so the network may be out of its camera distribution
either way; the experiment tells you whether K matters, not which K is right.

USAGE (GPU node, multihmr env):
    python scripts/diagnostics/diag_crop_camera.py \
        --ckpt logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt \
        --img_folder example_data --out results/diagnostics/crop_camera_v5
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from utils import normalize_rgb  # noqa: E402


def detect_people(img_pil, device, thresh):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    except ImportError:
        det = fasterrcnn_resnet50_fpn(pretrained=True)
    det = det.to(device).eval()
    import torchvision.transforms.functional as TF
    with torch.no_grad():
        out = det([TF.to_tensor(img_pil).to(device)])[0]
    keep = (out['labels'] == 1) & (out['scores'] >= thresh)
    return out['boxes'][keep].cpu().numpy(), out['scores'][keep].cpu().numpy()


def crop_like_demo(img_full, box, person_frac, crop_margin):
    W, H = img_full.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    S = int(round(max((y2 - y1) / max(person_frac, 0.05), max(x2 - x1, y2 - y1) * (1.0 + crop_margin))))
    S = max(S, 32)
    S = min(S, int(3 * max(W, H)))
    left, top = int(round(cx - S / 2.0)), int(round(cy - S / 2.0))
    crop = img_full.crop((left, top, left + S, top + S))
    return crop, S, left, top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_folder', default='example_data')
    ap.add_argument('--out', default='results/diagnostics/crop_camera')
    ap.add_argument('--fov', type=float, default=55.0, help='run_demo.sh --fov')
    ap.add_argument('--person_frac', type=float, default=0.22, help='run_demo.sh')
    ap.add_argument('--crop_margin', type=float, default=0.35)
    ap.add_argument('--detector_thresh', type=float, default=0.8)
    ap.add_argument('--det_thresh', type=float, default=0.15)
    ap.add_argument('--nms_kernel_size', type=int, default=5)
    ap.add_argument('--mask_helper_joints', type=int, default=1)
    ap.add_argument('--relax_hands', type=int, default=0)
    ap.add_argument('--n_vis', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)
    model, _ = dc.load_model(dc.resolve_ckpt(args.ckpt), device)
    dc.set_inference_toggles(model, mask_helper_joints=args.mask_helper_joints, relax_hands=args.relax_hands)
    bm = model.body_model; names = list(bm.bone_labels); phen = list(bm.phenotype_labels)
    chain = dc.neck_chain(names); faces = dc.faces_tri(bm); S_net = model.img_size

    imgs = sorted(f for f in os.listdir(args.img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    rows = []
    n_vis = 0
    for fname in imgs:
        img_full = Image.open(os.path.join(args.img_folder, fname)).convert('RGB')
        W, H = img_full.size
        boxes, scores = detect_people(img_full, device, args.detector_thresh)
        print(f"\n[{fname}] {W}x{H}, {len(boxes)} people detected")
        f_full = max(W, H) / (2 * math.tan(math.radians(args.fov) / 2))     # fov spans the long side (open_image pads to square)
        for bi, (box, dsc) in enumerate(zip(boxes, scores)):
            crop, S, left, top = crop_like_demo(img_full, box, args.person_frac, args.crop_margin)
            x = torch.from_numpy(normalize_rgb(np.asarray(crop.resize((S_net, S_net), Image.BILINEAR)))).unsqueeze(0).to(device)
            K_demo = dc.K_from_fov(args.fov, S_net, device)
            K_fix = torch.eye(3, device=device).unsqueeze(0)
            r = S_net / float(S)
            K_fix[0, 0, 0] = K_fix[0, 1, 1] = f_full * r
            K_fix[0, 0, 2] = (W / 2.0 - left) * r
            K_fix[0, 1, 2] = (H / 2.0 - top) * r
            fov_fix = math.degrees(2 * math.atan(S_net / (2 * float(K_fix[0, 0, 0]))))
            res = {}
            for kname, K in (('demo', K_demo), ('fixed', K_fix)):
                humans = []
                for thr in (args.det_thresh, 0.05):
                    with torch.no_grad():
                        humans = model(x, is_training=False, K=K, det_thresh=thr, nms_kernel_size=args.nms_kernel_size)
                    if humans:
                        break
                if not humans:
                    res[kname] = None; continue
                ctr = S_net / 2.0
                best = min(humans, key=lambda h: float(((h['loc'][0] - ctr) ** 2 + (h['loc'][1] - ctr) ** 2)))
                j = best['j3d'].float().unsqueeze(0); v = best['v3d'].float().unsqueeze(0)
                sh = best['shape'].float().reshape(-1)
                height = dc.body_height(j, names)[0].item(); neck = dc.chain_length(j, chain)[0].item()
                res[kname] = dict(depth=float(j[0, 0, 2]), height_m=height, neck_over_height=neck / max(height, 1e-6),
                                  head_over_height=dc.head_size(v, j, names)[0].item() / max(height, 1e-6),
                                  shape={n: float(sh[i]) for i, n in enumerate(phen)}, v3d=v[0].cpu().numpy())
            if res.get('demo') is None or res.get('fixed') is None:
                print(f"  person {bi}: no detection under {'demo' if res.get('demo') is None else 'fixed'} K; skipped")
                continue
            d, f = res['demo'], res['fixed']
            row = dict(image=fname, person=bi, det_score=float(dsc), crop_S=S, crop_frac_of_image=S / max(W, H),
                       fov_demo=args.fov, fov_fixed=fov_fix, focal_demo=float(K_demo[0, 0, 0]), focal_fixed=float(K_fix[0, 0, 0]),
                       depth_demo=d['depth'], depth_fixed=f['depth'], height_demo=d['height_m'], height_fixed=f['height_m'],
                       neck_h_demo=d['neck_over_height'], neck_h_fixed=f['neck_over_height'],
                       head_h_demo=d['head_over_height'], head_h_fixed=f['head_over_height'])
            for n in phen:
                row[f'{n}_demo'] = d['shape'][n]; row[f'{n}_fixed'] = f['shape'][n]
            rows.append(row)
            print(f"  person {bi}: crop S={S} ({S / max(W, H):.2f} of image)  fov demo {args.fov:.0f} -> fixed {fov_fix:.0f} deg")
            print(f"     age   {d['shape']['age']:.2f} -> {f['shape']['age']:.2f}   height-pheno {d['shape']['height']:.2f} -> {f['shape']['height']:.2f}"
                  f"   proportions {d['shape']['proportions']:.2f} -> {f['shape']['proportions']:.2f}")
            print(f"     body height {d['height_m']:.2f} -> {f['height_m']:.2f} m   neck/height {d['neck_over_height']:.3f} -> {f['neck_over_height']:.3f}"
                  f"   depth {d['depth']:.2f} -> {f['depth']:.2f} m")
            if n_vis < args.n_vis:
                cu8 = np.asarray(crop.resize((S_net, S_net), Image.BILINEAR)).astype(np.uint8)
                panels = [dc.label(cu8, f"{fname[:18]} p{bi}")]
                for kname, K in (('demo', K_demo), ('fixed', K_fix)):
                    ov = dc.render_overlay(cu8, [res[kname]['v3d']], faces, K[0].cpu().numpy())
                    panels.append(dc.label(ov if ov is not None else cu8, f"K_{kname} age={res[kname]['shape']['age']:.2f}"))
                Image.fromarray(dc.hstack(panels)).save(os.path.join(args.out, 'vis', f'{os.path.splitext(fname)[0]}_p{bi}.jpg'), quality=90)
                n_vis += 1

    if not rows:
        print("no people processed"); return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(os.path.join(args.out, 'per_person.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = dict(n_people=len(rows), mean_abs_change={}, mean_signed_change={})
    for n in phen + ['height', 'depth', 'neck_h', 'head_h']:
        a = np.asarray([r[f'{n}_demo'] for r in rows]); b = np.asarray([r[f'{n}_fixed'] for r in rows])
        summ['mean_abs_change'][n] = float(np.abs(b - a).mean()); summ['mean_signed_change'][n] = float((b - a).mean())
    print("\n=== MEAN CHANGE demo K -> corrected K (signed / abs) ===")
    for n, v in summ['mean_signed_change'].items():
        print(f"  {n:<12} {v:+.3f} / {summ['mean_abs_change'][n]:.3f}")
    print("READ: |age| or |height| change > ~0.05 => K is shaping the body, so the crop camera bug (C2) affects")
    print("      shape, not just placement, and fixing demo.py:366 is worth a real test. Changes ~0 => C2 is")
    print("      placement-only and anchor_to_box already compensates; look at the data (E2) for the children.")
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=2)
    print(f"wrote {args.out}/summary.json, per_person.csv, vis/")


if __name__ == '__main__':
    main()
