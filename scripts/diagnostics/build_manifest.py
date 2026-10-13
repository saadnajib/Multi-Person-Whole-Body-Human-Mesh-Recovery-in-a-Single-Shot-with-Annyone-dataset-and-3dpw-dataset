"""
Build the FIXED sample manifest used by the diagnostic suite.

Categories (from the GT annotations alone, no network):
  adult     GT age in [0.45, 0.65] and an unremarkable neck (chain angle < 20 deg)
  child     lowest GT age (Anny/MakeHuman scale: 0.19 ~ 11 y); taken from a wide age scan
  hands     both hands of one person projected inside the image with a large pixel span
            and in front of the torso (visible, not behind the body)
  neck      largest GT neck+head rotation away from identity (hard head turns / bends)

Two splits are kept apart: TRAIN entries come from the first N-100 annotations (seen in
training), HOLDOUT entries from the last 100 (never trained on). One (image, person) pair
per entry; an entry may carry several tags. The selection is seeded and the result is a
JSON file, so every later script evaluates exactly the same people.

Side effect: `data/diag_manifest/{train,holdout}/` are created with SYMLINKS to the selected
pkl + jpg pairs, so train.py can be pointed at the subset via --anny_data_folder
(used by the capped overfit check). Nothing in the dataset is copied or modified.

USAGE (GPU node is fastest; CPU works):
    python scripts/diagnostics/build_manifest.py --out results/diagnostics/manifest.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.annyone import AnnyOne  # noqa: E402


def hand_joint_sets(names):
    """Indices of the left / right hand joints (finger, thumb, metacarpal, wrist) by name suffix."""
    sides = {'L': [], 'R': []}
    for i, n in enumerate(names):
        lo = n.lower()
        if any(k in lo for k in ('finger', 'thumb', 'metacarpal', 'wrist')):
            if lo.endswith('.l') or lo.endswith('_l') or lo.endswith('.left'):
                sides['L'].append(i)
            elif lo.endswith('.r') or lo.endswith('_r') or lo.endswith('.right'):
                sides['R'].append(i)
    return sides


def person_features(bm, names, pose, shape, K, img_size, hands, chain):
    """Per-person features for one image. pose [nh,163,4,4], shape [nh,11], K [1,3,3]."""
    v, j = dc.run_body_model(bm, pose, shape)
    keep = dc.visible_gt_mask(j, K, img_size, 14)
    R = dc.orthonormalize(pose[:, :, :3, :3])
    I = torch.eye(3, device=R.device).expand_as(R)
    ang = dc.geodesic_deg(I, R)                                   # [nh,163] GT angle from identity
    neck_deg = ang[:, chain].sum(-1) if chain else torch.zeros(pose.shape[0], device=R.device)
    j2d = dc.project(j, K)
    root_z = j[:, 0, 2]
    span = torch.zeros(pose.shape[0], device=R.device)
    both_visible = torch.ones(pose.shape[0], dtype=torch.bool, device=R.device)
    for side, idx in hands.items():
        if not idx:
            both_visible[:] = False
            continue
        p2 = j2d[:, idx]                                          # [nh,H,2]
        inside = ((p2 >= 0) & (p2 < img_size)).all(-1).all(-1)
        d = torch.cdist(p2, p2).amax(dim=(-1, -2))                # pixel span of the hand
        in_front = j[:, idx, 2].mean(-1) <= root_z + 0.15        # not behind the torso
        both_visible &= inside & in_front
        span = torch.maximum(span, torch.where(inside & in_front, d, torch.zeros_like(d)))
    return dict(keep=keep.cpu(), age=shape[:, 1].cpu(), neck_deg=neck_deg.cpu(),
                hand_span=span.cpu(), hands_visible=both_visible.cpu(), root_z=root_z.cpu())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--out', default='results/diagnostics/manifest.json')
    ap.add_argument('--link_root', default='data/diag_manifest', help='where the symlinked subsets go')
    ap.add_argument('--scan', type=int, default=6000, help='evenly spaced training images decoded for hands/neck/adults')
    ap.add_argument('--age_scan', type=int, default=60000, help='training pkls read (no decode) to find the youngest people')
    ap.add_argument('--n_train', type=int, default=10, help='per category, train split')
    ap.add_argument('--n_holdout', type=int, default=5, help='per category, holdout split')
    ap.add_argument('--child_age', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    import anny
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(device)
    bm.set_skinning_method('lbs')
    names = list(bm.bone_labels)
    chain = dc.neck_chain(names)
    hands = hand_joint_sets(names)
    print(f"[rig] neck chain {[names[i] for i in chain]}; hand joints L={len(hands['L'])} R={len(hands['R'])}")
    if not hands['L'] or not hands['R']:
        print("[WARN] hand joints not found by suffix .L/.R; hand category will be empty. Sample names:",
              [n for n in names if 'finger' in n.lower()][:6])

    ds = AnnyOne(data_folder=args.data_folder, img_size=672)
    pkls = ds.pkl_files
    n_total = len(pkls)
    hold = dc.holdout_indices(n_total)
    train_pool = n_total - dc.N_HOLDOUT

    # ---- 1. cheap age scan (pkl only) for children -------------------------------------------
    t0 = time.time()
    young = []   # (age, idx, person)
    for k, i in enumerate(np.linspace(0, train_pool - 1, min(args.age_scan, train_pool)).astype(int)):
        s = dc.shapes_from_pkl(dc.load_pkl(pkls[int(i)]))
        if s is None or len(s) == 0:
            continue
        for pidx, a in enumerate(s[:, 1]):
            if a < args.child_age:
                young.append((float(a), int(i), pidx))
        if k % 10000 == 0 and k:
            print(f"  age scan {k} ({time.time() - t0:.0f}s), {len(young)} people < {args.child_age} so far", flush=True)
    young.sort()
    print(f"[age] {len(young)} people with age < {args.child_age} among {min(args.age_scan, train_pool)} training images")

    # ---- 2. decode candidates ---------------------------------------------------------------
    def decode(indices, tag):
        feats = {}
        t0 = time.time()
        for k, i in enumerate(indices):
            i = int(i)
            raw = dc.load_pkl(pkls[i])
            with Image.open(dc.image_path_for(pkls[i])) as im:
                W, H = im.size
            hum = raw.get('humans', [])
            if not hum:
                continue
            pose = torch.from_numpy(np.stack([np.asarray(h['anny_pose'], dtype=np.float32)[:163] for h in hum])).to(device)
            shape = torch.from_numpy(np.stack([np.asarray(h['anny_shape'], dtype=np.float32).reshape(-1)[:11] for h in hum])).to(device)
            K = np.eye(3, dtype=np.float32)
            focal = np.asarray(raw['focal'], dtype=np.float32).reshape(-1)[0]
            pp = np.asarray(raw['princpt'], dtype=np.float32).reshape(-1)[:2]
            K[0, 2], K[1, 2] = 672 * pp[0] / W, 672 * pp[1] / H           # datasets/annyone.py convention
            K[0, 0] = K[1, 1] = focal / (max(W, H) / 672)
            Kt = torch.from_numpy(K).unsqueeze(0).to(device)
            with torch.no_grad():
                f = person_features(bm, names, pose, shape, Kt, 672, hands, chain)
            f['n_people'] = len(hum)
            feats[i] = f
            if k % 1000 == 0 and k:
                print(f"  decode {tag} {k}/{len(indices)} ({time.time() - t0:.0f}s)", flush=True)
        return feats

    train_idx = sorted(set(np.linspace(0, train_pool - 1, min(args.scan, train_pool)).astype(int).tolist()
                           + [i for _, i, _ in young[:4 * args.n_train]]))
    F_train = decode(train_idx, 'train')
    F_hold = decode(hold, 'holdout')

    # ---- 3. select -----------------------------------------------------------------------------
    def people(F):
        rows = []
        for i, f in F.items():
            for p in range(len(f['age'])):
                if bool(f['keep'][p]):
                    rows.append(dict(idx=i, person=p, age=float(f['age'][p]), neck_deg=float(f['neck_deg'][p]),
                                     hand_span_px=float(f['hand_span'][p]), hands_visible=bool(f['hands_visible'][p]),
                                     root_z=float(f['root_z'][p]), n_people=f['n_people']))
        return rows

    def pick(rows, n, split):
        used_imgs, chosen = set(), []
        adults = [r for r in rows if 0.45 <= r['age'] <= 0.65 and r['neck_deg'] < 20]
        rng.shuffle(adults)
        children = sorted([r for r in rows if r['age'] < args.child_age], key=lambda r: r['age'])
        hands_ = sorted([r for r in rows if r['hands_visible'] and r['age'] >= 0.3], key=lambda r: -r['hand_span_px'])
        neck_ = sorted([r for r in rows if r['neck_deg'] >= 30], key=lambda r: -r['neck_deg'])
        for tag, cand in (('child', children), ('neck', neck_), ('hands', hands_), ('adult', adults)):
            c = 0
            for r in cand:
                if c >= n:
                    break
                key = (r['idx'], r['person'])
                hit = next((e for e in chosen if (e['idx'], e['person']) == key), None)
                if hit is not None:
                    if tag not in hit['tags']:
                        hit['tags'].append(tag); c += 1
                    continue
                if r['idx'] in used_imgs:
                    continue
                e = dict(r); e['split'] = split; e['tags'] = [tag]
                e['pkl'] = pkls[r['idx']]; e['image'] = dc.image_path_for(pkls[r['idx']])
                chosen.append(e); used_imgs.add(r['idx']); c += 1
            if c < n:
                print(f"[select] {split}/{tag}: only {c} of {n} available")
        return chosen

    train_entries = pick(people(F_train), args.n_train, 'train')
    hold_entries = pick(people(F_hold), args.n_holdout, 'holdout')
    hold_ages = [r['age'] for r in people(F_hold)]
    if not any(a < args.child_age for a in hold_ages):
        print(f"[select] NOTE: the holdout has NO person with age < {args.child_age} "
              f"(min {min(hold_ages):.3f}); child results exist only in-sample (train split).")

    # ---- 4. symlinked subsets for train.py --anny_data_folder ---------------------------------
    for split, entries in (('train', train_entries), ('holdout', hold_entries)):
        d = os.path.join(args.link_root, split)
        os.makedirs(d, exist_ok=True)
        for e in entries:
            for src in (e['pkl'], e['image']):
                dst = os.path.join(d, os.path.basename(src))
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(src), dst)
        print(f"[links] {split}: {len(set(e['idx'] for e in entries))} images linked under {d}")

    manifest = dict(seed=args.seed, data_folder=args.data_folder, n_total=n_total,
                    criteria=dict(adult='0.45<=age<=0.65 and neck<20deg', child=f'age<{args.child_age}',
                                  hands='both hands inside image, span>=? (top by span), in front of torso, age>=0.3',
                                  neck='neck01..head GT angle from identity >= 30 deg (top)'),
                    counts={s: {t: sum(1 for e in E if t in e['tags']) for t in ('adult', 'child', 'hands', 'neck')}
                            for s, E in (('train', train_entries), ('holdout', hold_entries))},
                    link_root=args.link_root, entries=train_entries + hold_entries)
    with open(args.out, 'w') as f:
        json.dump(dc.json_safe(manifest), f, indent=2)
    print(f"\n[manifest] {len(train_entries)} train + {len(hold_entries)} holdout entries -> {args.out}")
    print(json.dumps(manifest['counts'], indent=1))
    for e in train_entries + hold_entries:
        print(f"  {e['split']:<7} idx {e['idx']:6d} p{e['person']} age {e['age']:.2f} neck {e['neck_deg']:5.1f}deg "
              f"hand {e['hand_span_px']:5.0f}px  {','.join(e['tags'])}")


if __name__ == '__main__':
    main()
