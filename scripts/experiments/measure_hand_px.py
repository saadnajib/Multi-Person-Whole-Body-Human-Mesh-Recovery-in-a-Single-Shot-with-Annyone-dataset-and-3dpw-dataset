"""
exp4_fingers, step 1: how large are the hands in pixels, before any loss change or crop pass?

For the fixed 100-image holdout (--select holdout) or the diag manifest (--manifest, --manifest_split), per GT
person kept by train.py's visibility filter (dc.visible_gt_mask, the prepare_gt rule) and per hand (L / R):
  * hand points = wrist, metacarpals and finger bones incl. the thumbs, resolved by NAME exactly as
    scripts/diagnostics/build_manifest.hand_joint_sets does, plus the FINGERTIPS: bone heads are the only joints
    the 163-joint output carries, so the tails of the five distal phalanges are read from the body model
    (anny >= 0.6 returns them with return_bone_ends=True) or, failing that, extrapolated as head + (head - head of
    the middle phalanx); the log says which;
  * the bbox of those points at the ORIGINAL render scale (raw focal / princpt, 1280 px) and at the network input
    (datasets/annyone.py K = x img_size / max(W, H)); 'hand size' = the bbox long side, also in 14 px tokens;
  * the same under a hypothetical PERSON CROP: side = --crop_factor x the long side of the GT joint bbox, clamped
    to the image (the V4 rule of diag_holdout_trace.py --person_crop 1), so hand_px_crop = hand_px_orig x
    img_size / side;
  * median / quantiles, fractions of hands under 1, 2 and 4 tokens, hands partly outside the image, and for a
    manifest the same medians per tag (adult / child / hands / neck).
A measurement of the labels through the body model: no checkpoint, no network. Minutes on a GPU node.

    python scripts/experiments/measure_hand_px.py --select holdout --n 100 \
        --out results/experiments/exp4_fingers/hand_px_holdout.json
    python scripts/experiments/measure_hand_px.py --manifest results/diagnostics/manifest.json --manifest_split both \
        --out results/experiments/exp4_fingers/hand_px_manifest.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'diagnostics'))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

TOKEN_PX = 14            # DINOv2 ViT-L/14 patch
THRESH_TOK = (1, 2, 4)


# --------------------------------------------------------------------------- pure helpers (no torch)
def crop_side_for(person_long_side, factor, W, H, min_side=64.0):
    """Side of the square person crop diag_holdout_trace.py --person_crop uses: factor x the GT joint bbox long
    side, at least min_side, never larger than the image (a frame-filling person gets the whole frame, x1.0)."""
    return int(round(min(max(float(person_long_side) * factor, min_side), float(min(W, H)))))


def summarize_sizes(px, token_px=TOKEN_PX):
    """Quantiles of hand sizes given in pixels, plus the fraction under 1 / 2 / 4 tokens. Empty -> n 0."""
    a = np.asarray(px, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    tok = a / float(token_px)
    d = dict(n=int(a.size), mean_px=float(a.mean()), median_px=float(np.median(a)),
             p10_px=float(np.percentile(a, 10)), p25_px=float(np.percentile(a, 25)),
             p75_px=float(np.percentile(a, 75)), p90_px=float(np.percentile(a, 90)),
             median_tokens=float(np.median(tok)))
    for t in THRESH_TOK:
        d[f'frac_lt_{t}tok'] = float((tok < t).mean())
    return d


def fmt_row(label, d):
    if not d or d.get('n', 0) == 0:
        return f"{label:<44}{0:>6}" + " " * 60
    return (f"{label:<44}{d['n']:>6}{d['median_px']:>10.1f}{d['p25_px']:>8.1f}{d['p75_px']:>8.1f}"
            f"{d['median_tokens']:>10.2f}" + "".join(f"{100 * d[f'frac_lt_{t}tok']:>8.1f}" for t in THRESH_TOK))


HEADER = f"{'scale':<44}{'hands':>6}{'median px':>10}{'p25':>8}{'p75':>8}{'med tok':>10}" + \
         "".join(f"{'<' + str(t) + 'tok %':>8}" for t in THRESH_TOK)


# --------------------------------------------------------------------------- body model
def distal_pairs(names, hand_idx):
    """[(distal index, middle index)] for finger1..5 of one hand: fingertips are the distal TAILS."""
    pairs = []
    lut = {n.lower(): i for i, n in enumerate(names)}
    for i in hand_idx:
        lo = names[i].lower()
        stem, _, side = lo.partition('.')
        if stem.startswith('finger') and stem.endswith('-3'):
            mid = lut.get(stem[:-1] + '2' + ('.' + side if side else ''))
            pairs.append((i, mid))
    return pairs


def run_body_model_ends(bm, pose, shape):
    """Joint heads [N,163,3] and, when the installed anny exposes them, the bone tails [N,163,3] (else None)."""
    import torch
    labels = list(bm.phenotype_labels)
    shape = torch.clamp(shape, 0.0, 1.0)
    kw = {k: shape[:, i] for i, k in enumerate(labels)}
    with torch.no_grad():
        try:
            out = bm(pose_parameters=pose, phenotype_kwargs=kw, return_bone_ends=True)
            tails = out.get('bone_tails', None)
        except (TypeError, NotImplementedError):
            out = bm(pose_parameters=pose, phenotype_kwargs=kw)
            tails = None
    return out['bone_poses'][:, :, :3, -1], tails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_folder', default=None, help='default: diag_common.DATA_FOLDER_DEFAULT')
    ap.add_argument('--select', default='holdout', choices=['holdout'])
    ap.add_argument('--n', type=int, default=100, help='--select holdout: the last N annotations (never trained on)')
    ap.add_argument('--manifest', default='', help='build_manifest.py JSON; overrides --select')
    ap.add_argument('--manifest_split', default='both', choices=['train', 'holdout', 'both'])
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--crop_factor', type=float, default=1.3, help='diag_holdout_trace.py --crop_factor')
    ap.add_argument('--out', default='results/experiments/exp4_fingers/hand_px.json')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    import torch
    import diag_common as dc
    from build_manifest import hand_joint_sets
    from PIL import Image
    import anny

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    data_folder = args.data_folder or dc.DATA_FOLDER_DEFAULT
    S = args.img_size
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(device)
    bm.set_skinning_method('lbs')
    names = list(bm.bone_labels)
    hands = hand_joint_sets(names)
    if not hands['L'] or not hands['R']:
        raise SystemExit(f"hand joints not found by name suffix .L/.R; finger-like names: {[n for n in names if 'finger' in n.lower()][:6]}")
    tips = {side: distal_pairs(names, idx) for side, idx in hands.items()}
    print(f"[rig] hand joints L={len(hands['L'])} R={len(hands['R'])} ({[names[i] for i in hands['L'][:4]]} ...); "
          f"distal phalanges per hand: {len(tips['L'])} / {len(tips['R'])}")

    pkls = dc.list_pkls(data_folder)
    n_total = len(pkls)
    if args.manifest:
        man = json.load(open(args.manifest))
        ents = [e for e in man['entries'] if args.manifest_split == 'both' or e['split'] == args.manifest_split]
        if not ents:
            raise SystemExit(f"manifest has no '{args.manifest_split}' entries")
        sel = {}
        for e in ents:
            sel.setdefault(int(e['idx']), {})[int(e['person'])] = list(e['tags']) + [e['split']]
        indices = sorted(sel)
        desc = f"manifest {args.manifest_split} ({len(ents)} people in {len(indices)} images)"
    else:
        indices = dc.holdout_indices(n_total)[:args.n]
        sel = None
        desc = f"HOLDOUT (last {len(indices)} of {n_total}, never trained on)"
    print(f"[data] {n_total} annotations; measuring {desc}; crop factor {args.crop_factor}; token = {TOKEN_PX} input px")

    hands_rows, people_rows = [], []
    tails_seen = tails_missing = 0
    t0 = time.time()
    for k, idx in enumerate(indices):
        raw = dc.load_pkl(pkls[idx])
        hum = raw.get('humans', [])
        if not hum:
            continue
        with Image.open(dc.image_path_for(pkls[idx])) as im:
            W, H = im.size
        pose = torch.from_numpy(np.stack([np.asarray(h['anny_pose'], dtype=np.float32)[:163] for h in hum])).to(device)
        shape = torch.from_numpy(np.stack([np.asarray(h['anny_shape'], dtype=np.float32).reshape(-1)[:11] for h in hum])).to(device)
        f2 = np.asarray(raw['focal'], dtype=np.float64).reshape(-1)
        f2 = f2 if f2.size > 1 else np.repeat(f2, 2)
        pp = np.asarray(raw['princpt'], dtype=np.float64).reshape(-1)[:2]
        K_orig = torch.tensor([[f2[0], 0, pp[0]], [0, f2[1], pp[1]], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
        K_in = K_orig.clone()                                    # datasets/annyone.py: x S/max(W,H), principal point per axis
        K_in[0, 0, 0] *= S / max(W, H); K_in[0, 1, 1] *= S / max(W, H)
        K_in[0, 0, 2] *= S / W; K_in[0, 1, 2] *= S / H
        scale_in = S / float(max(W, H))

        j, tails = run_body_model_ends(bm, pose, shape)        # [nh,163,3] camera frame
        if tails is None:
            tails_missing += 1
        else:
            tails_seen += 1
        keep = dc.visible_gt_mask(j, K_in, S, TOKEN_PX)
        j2d = dc.project(j, K_orig)                              # [nh,163,2] original pixels
        tips2d = {}
        for side, pairs in tips.items():
            pts3 = []
            for d, mid in pairs:
                if tails is not None:
                    pts3.append(tails[:, d])
                elif mid is not None:
                    pts3.append(j[:, d] + (j[:, d] - j[:, mid]))   # distal phalanx about as long as the middle one
                else:
                    pts3.append(j[:, d])
            tips2d[side] = (torch.stack(pts3, 1), dc.project(torch.stack(pts3, 1), K_orig)) if pts3 else (None, None)

        for p in range(len(hum)):
            if not bool(keep[p]):
                continue
            tags = (sel.get(idx, {}).get(p) if sel is not None else None)
            if sel is not None and tags is None:
                continue
            front = j[p, :, 2] > 0.1
            pts_p = j2d[p][front]
            lo, hi = pts_p.min(0).values, pts_p.max(0).values
            person_long = float((hi - lo).max())
            side_px = crop_side_for(person_long, args.crop_factor, W, H)
            scale_crop = S / float(side_px)
            prow = dict(idx=idx, person=p, tags='|'.join(tags or []), age=float(shape[p, 1]), W=W, H=H,
                        person_px_orig=person_long, person_px_in=person_long * scale_in,
                        crop_side_px=side_px, crop_scale_vs_input=max(W, H) / float(side_px),
                        crop_person_frac=person_long / side_px, root_z=float(j[p, 0, 2]))
            people_rows.append(prow)
            for side, hidx in hands.items():
                pts = j2d[p, hidx]
                z = j[p, hidx, 2]
                if tips2d[side][1] is not None:
                    pts = torch.cat([pts, tips2d[side][1][p]], 0)
                    z = torch.cat([z, tips2d[side][0][p][:, 2]], 0)
                lo_h, hi_h = pts.min(0).values, pts.max(0).values
                wh = (hi_h - lo_h)
                size_orig = float(wh.max())
                inside = bool((pts[:, 0] >= 0).all() and (pts[:, 1] >= 0).all() and (pts[:, 0] < W).all()
                              and (pts[:, 1] < H).all() and (z > 0.1).all())
                hands_rows.append(dict(idx=idx, person=p, side=side, tags=prow['tags'], inside=inside,
                                       w_px_orig=float(wh[0]), h_px_orig=float(wh[1]), size_px_orig=size_orig,
                                       size_px_in=size_orig * scale_in, size_px_crop=size_orig * scale_crop,
                                       area_tok_in=float(wh[0] * wh[1]) * scale_in ** 2 / TOKEN_PX ** 2,
                                       crop_scale_vs_input=prow['crop_scale_vs_input'],
                                       behind_torso=bool(j[p, hidx, 2].mean() > j[p, 0, 2] + 0.15)))
        if k % 20 == 0 and k:
            print(f"  {k}/{len(indices)} ({time.time() - t0:.0f}s)", flush=True)

    if not hands_rows:
        print("no kept people; nothing to report"); return
    tip_src = ('body-model bone tails (anny return_bone_ends)' if tails_seen and not tails_missing else
               'EXTRAPOLATED distal head + (head - middle head); the installed anny did not return bone ends'
               if tails_missing and not tails_seen else f'mixed ({tails_seen} images with tails, {tails_missing} without)')
    inside_rows = [r for r in hands_rows if r['inside']]
    sets = [('original render px (median image %dx%d)' % (int(np.median([r['W'] for r in people_rows])), int(np.median([r['H'] for r in people_rows]))),
             [r['size_px_orig'] for r in hands_rows]),
            (f'network input {S} px, all hands', [r['size_px_in'] for r in hands_rows]),
            (f'network input {S} px, hand fully inside image', [r['size_px_in'] for r in inside_rows]),
            (f'person crop x{args.crop_factor} -> {S} px, all hands', [r['size_px_crop'] for r in hands_rows]),
            (f'person crop x{args.crop_factor} -> {S} px, fully inside', [r['size_px_crop'] for r in inside_rows])]
    summary = dict(selection=desc, n_images=len(indices), n_people=len(people_rows), n_hands=len(hands_rows),
                   n_hands_fully_inside=len(inside_rows), fingertips=tip_src, token_px=TOKEN_PX, img_size=S,
                   crop_factor=args.crop_factor, sizes={}, crop={}, per_tag={})
    print(f"\n=== HAND SIZE = long side of the hand bbox (wrist, metacarpals, phalanges, fingertips); fingertips from {tip_src} ===")
    print(f"{len(people_rows)} kept people in {len(indices)} images, {len(hands_rows)} hands, {len(inside_rows)} fully inside the image, "
          f"{sum(r['behind_torso'] for r in hands_rows)} behind the torso (>15 cm deeper than the root)")
    print(HEADER)
    for label, vals in sets:
        d = summarize_sizes(vals)
        summary['sizes'][label] = d
        print(fmt_row(label, d))
    area = np.asarray([r['area_tok_in'] for r in inside_rows], dtype=np.float64)
    if area.size:
        summary['sizes']['bbox_area_tokens_input_inside'] = dict(median=float(np.median(area)), p25=float(np.percentile(area, 25)),
                                                                 p75=float(np.percentile(area, 75)))
        print(f"  hand bbox AREA at the input, fully-inside hands: median {np.median(area):.1f} tokens^2 "
              f"(p25 {np.percentile(area, 25):.1f}, p75 {np.percentile(area, 75):.1f}); 1 token = {TOKEN_PX} input px"
              f" = {TOKEN_PX / (S / float(np.median([max(r['W'], r['H']) for r in people_rows]))):.1f} render px")
    cs = np.asarray([r['crop_scale_vs_input'] for r in people_rows]); cf = np.asarray([r['crop_person_frac'] for r in people_rows])
    pin = np.asarray([r['person_px_in'] for r in people_rows])
    summary['crop'] = dict(scale_vs_input_median=float(np.median(cs)), scale_vs_input_p25=float(np.percentile(cs, 25)),
                           scale_vs_input_p75=float(np.percentile(cs, 75)), frac_people_no_gain=float((cs <= 1.0 + 1e-6).mean()),
                           person_frac_of_crop_median=float(np.median(cf)), person_px_input_median=float(np.median(pin)))
    print(f"  person crop x{args.crop_factor}: magnification over the {S}px input median x{np.median(cs):.2f} "
          f"(p25 x{np.percentile(cs, 25):.2f}, p75 x{np.percentile(cs, 75):.2f}); {100 * (cs <= 1.0 + 1e-6).mean():.0f}% of people gain nothing "
          f"(crop = whole frame); person fills {100 * np.median(cf):.0f}% of the crop (training frames: ~20-30%); "
          f"median person height {np.median(pin):.0f} input px")
    if sel is not None:
        all_tags = sorted({t for r in hands_rows for t in r['tags'].split('|') if t})
        print(f"\n=== PER TAG (manifest), hand size at the {S}px input, all hands / fully inside; crop magnification ===")
        print(f"{'tag':<10}{'hands':>6}{'med px':>8}{'med tok':>8}{'<2tok %':>8}{'inside':>7}{'med px':>8}{'<2tok %':>8}{'crop x':>8}")
        for t in all_tags:
            rr = [r for r in hands_rows if t in r['tags'].split('|')]
            ri = [r for r in rr if r['inside']]
            d_all, d_in = summarize_sizes([r['size_px_in'] for r in rr]), summarize_sizes([r['size_px_in'] for r in ri])
            mag = float(np.median([r['crop_scale_vs_input'] for r in rr])) if rr else float('nan')
            summary['per_tag'][t] = dict(all=d_all, inside=d_in, crop_scale_median=mag)
            print(f"{t:<10}{d_all.get('n', 0):>6}{d_all.get('median_px', float('nan')):>8.1f}{d_all.get('median_tokens', float('nan')):>8.2f}"
                  f"{100 * d_all.get('frac_lt_2tok', float('nan')):>8.1f}{d_in.get('n', 0):>7}{d_in.get('median_px', float('nan')):>8.1f}"
                  f"{100 * d_in.get('frac_lt_2tok', float('nan')):>8.1f}{mag:>8.2f}")
    print("\nREAD: a finger is ~0.4-0.5 of the hand's long side and ~0.1 of it wide. Hands under 2 tokens cannot show")
    print("      articulation to a 14 px patch grid; if most holdout hands sit there at the input but reach 4+ tokens under")
    print("      the person crop, the V4 crop trace (step 2) is a fair test of resolution. If hands are already 4+ tokens")
    print("      at the input, resolution is not what limits the fingers and the loss-side experiment is the one to read.")
    summary['people'] = people_rows
    summary['hands'] = hands_rows
    with open(args.out, 'w') as f:
        json.dump(dc.json_safe(summary), f, indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
