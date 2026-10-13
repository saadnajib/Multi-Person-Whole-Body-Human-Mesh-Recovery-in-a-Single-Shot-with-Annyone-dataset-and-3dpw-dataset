"""
CHECK 1 -- decode and render the GROUND-TRUTH parameters without any network and
overlay them with the supplied camera.

For every manifest entry: anny_pose + anny_shape -> Anny body model (the same
object model.py / train.py build) -> vertices + joints in the camera frame ->
projection with the dataset K (datasets/annyone.py convention). The panel
[image | GT mesh overlay | GT joints + tight_bbox] shows whether the data path
is right BEFORE the network is involved.

Quantitative checks per person:
  * fraction of projected GT joints inside the image,
  * fraction inside the annotation's own `tight_bbox` (the bbox format is not
    documented: both (x1,y1,x2,y2) and (x,y,w,h), in pixels or normalised, are
    tried and the best-fitting one is reported; a low score under EVERY
    interpretation means the projection, not the bbox parsing, is wrong),
  * root depth and projected body height in pixels.

A GT mesh that does not sit on the person => camera / pose-convention error
(decoding or preprocessing), and nothing downstream can be trusted.

USAGE (GPU node for the renders):
    python scripts/diagnostics/diag_gt_decode.py --manifest results/diagnostics/manifest.json \
        --out results/diagnostics/gt_decode
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from datasets.annyone import AnnyOne  # noqa: E402
from datasets.bedlam import collate_fn  # noqa: E402
from utils import denormalize_rgb  # noqa: E402


def bbox_interpretations(bb, W, H, s, pad):
    """
    Candidate (x1,y1,x2,y2) boxes in the 672 frame for a raw tight_bbox of 4 numbers.
    s = 672/max(W,H); pad = (padx, pady) offsets of the contained image inside the square.
    """
    bb = np.asarray(bb, dtype=np.float64).reshape(-1)[:4]
    out = {}
    norm = bb.max() <= 1.5
    for fmt in ('xyxy', 'xywh'):
        b = bb.copy()
        if norm:
            b = b * np.array([W, H, W, H])
        if fmt == 'xywh':
            b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        x1, y1, x2, y2 = b
        out[f"{fmt}{'_norm' if norm else '_px'}"] = np.array([x1 * s + pad[0], y1 * s + pad[1], x2 * s + pad[0], y2 * s + pad[1]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='results/diagnostics/manifest.json')
    ap.add_argument('--split', default='all', choices=['all', 'train', 'holdout'])
    ap.add_argument('--data_folder', default=None, help='default: the manifest\'s data_folder')
    ap.add_argument('--out', default='results/diagnostics/gt_decode')
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out, 'vis'), exist_ok=True)

    man = json.load(open(args.manifest))
    entries = [e for e in man['entries'] if args.split == 'all' or e['split'] == args.split]
    data_folder = args.data_folder or man['data_folder']

    import anny
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(device)
    bm.set_skinning_method('lbs')
    names = list(bm.bone_labels)
    faces = dc.faces_tri(bm)
    ds = AnnyOne(data_folder=data_folder, img_size=args.img_size)
    S = args.img_size

    rows = []
    for e in entries:
        idx, p = e['idx'], e['person']
        img_arr, annot = ds[idx]
        x, y = collate_fn([(img_arr, annot)])
        K = y['K'].to(device)
        valid = y['valid_humans'][0].bool()
        pose = y['anny_pose'][0][valid][:, :163].float().to(device)
        shape = y['anny_shape'][0][valid].reshape(int(valid.sum()), -1)[:, :11].float().to(device)
        if p >= pose.shape[0]:
            print(f"  [warn] idx {idx}: person {p} missing; skipped"); continue
        v, j = dc.run_body_model(bm, pose, shape)
        j2d = dc.project(j, K)
        raw = dc.load_pkl(ds.pkl_files[idx])
        with Image.open(dc.image_path_for(ds.pkl_files[idx])) as im:
            W, H = im.size
        s = S / max(W, H)
        pad = ((S - W * s) / 2.0, (S - H * s) / 2.0)

        jp = j2d[p]
        inside = float(((jp >= 0) & (jp < S)).all(-1).float().mean())
        bb_raw = raw['humans'][p].get('tight_bbox', None)
        best_name, best_frac, best_box = None, -1.0, None
        if bb_raw is not None:
            for name, box in bbox_interpretations(bb_raw, W, H, s, pad).items():
                inb = ((jp[:, 0] >= box[0]) & (jp[:, 0] <= box[2]) & (jp[:, 1] >= box[1]) & (jp[:, 1] <= box[3])).float().mean().item()
                if inb > best_frac:
                    best_name, best_frac, best_box = name, inb, box
        ymin, ymax = float(jp[:, 1].min()), float(jp[:, 1].max())
        rows.append(dict(split=e['split'], idx=idx, person=p, tags=e['tags'], age=e['age'],
                         joints_inside_image=inside, joints_inside_bbox=best_frac if bb_raw is not None else None,
                         bbox_format=best_name, root_depth_m=float(j[p, 0, 2]), pixel_height=ymax - ymin,
                         W=W, H=H))

        # panel: image | GT mesh | GT joints + bbox
        img_u8 = denormalize_rgb(x[0].float().cpu().numpy())
        Kc = K[0].cpu().numpy()
        ov = dc.render_overlay(img_u8, [v[p].cpu().numpy()], faces, Kc)
        pts = dc.draw_points(img_u8, jp.cpu().numpy(), color=(255, 40, 40))
        if best_box is not None:
            im = Image.fromarray(pts); d = ImageDraw.Draw(im)
            d.rectangle([float(best_box[0]), float(best_box[1]), float(best_box[2]), float(best_box[3])], outline=(40, 255, 40), width=2)
            pts = np.asarray(im)
        panels = [dc.label(img_u8, f"{e['split']} idx {idx} p{p} {','.join(e['tags'])}"),
                  dc.label(ov if ov is not None else img_u8, "GT mesh, dataset K (no network)"),
                  dc.label(pts, f"GT joints; bbox {best_name} {best_frac * 100:.0f}% in" if best_box is not None else "GT joints")]
        Image.fromarray(dc.hstack(panels)).save(os.path.join(args.out, 'vis', f"{e['split']}_{idx:06d}_p{p}.jpg"), quality=90)

    if not rows:
        print("nothing processed"); return
    m = dc.mean_or_none
    summary = dict(n=len(rows), joints_inside_image_mean=m([r['joints_inside_image'] for r in rows]),
                   joints_inside_bbox_mean=m([r['joints_inside_bbox'] for r in rows if r['joints_inside_bbox'] is not None]),
                   bbox_formats={k: sum(1 for r in rows if r['bbox_format'] == k) for k in set(r['bbox_format'] for r in rows)},
                   worst_bbox_containment=sorted([(r['joints_inside_bbox'] or 0.0, r['idx'], r['person']) for r in rows])[:5],
                   per_tag={}, rows=rows)
    for tag in ('adult', 'child', 'hands', 'neck'):
        rr = [r for r in rows if tag in r['tags']]
        if rr:
            summary['per_tag'][tag] = dict(n=len(rr), inside_image=m([r['joints_inside_image'] for r in rr]),
                                           inside_bbox=m([r['joints_inside_bbox'] for r in rr if r['joints_inside_bbox'] is not None]),
                                           pixel_height=m([r['pixel_height'] for r in rr]))
    print(f"\n=== GT DECODE (no network), {len(rows)} people ===")
    print(f"GT joints inside the {S}px image: {summary['joints_inside_image_mean'] * 100:.1f}%; "
          f"inside the annotation's tight_bbox: {dc.fmt((summary['joints_inside_bbox_mean'] or 0) * 100)}% "
          f"(bbox read as {summary['bbox_formats']})")
    for tag, v in summary['per_tag'].items():
        print(f"  {tag:<6} n={v['n']:2d} inside_image {v['inside_image'] * 100:5.1f}%  inside_bbox {dc.fmt((v['inside_bbox'] or 0) * 100):>5}%  "
              f"pixel height {v['pixel_height']:.0f}px")
    print("READ: >=95% inside the bbox under one interpretation => decoding + camera are consistent with the")
    print("      annotation; a GT overlay that still looks wrong is then a RENDERING issue. Low under every")
    print("      interpretation => the dataset K or the pose convention is wrong (decoding/preprocessing cause).")
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(dc.json_safe(summary), f, indent=2, allow_nan=False)
    print(f"wrote {args.out}/summary.json and {len(rows)} panels")


if __name__ == '__main__':
    main()
