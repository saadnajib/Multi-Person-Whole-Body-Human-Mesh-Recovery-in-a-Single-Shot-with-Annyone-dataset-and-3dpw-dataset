"""
exp2_children, step 2: age-group representation in the REAL training batches (the measurement the plan asks for).

Rebuilds the AnnyOne training DataLoader the way train.py main() does for --train_data AnnyOne:
AnnyOne -> Subset(range(N - val_anny_n)) -> DataLoader(batch_size, shuffle unless a sampler is given, drop_last,
pin_memory, the bulletproof collate, num_workers), optionally with the --age_balance / --exclude_indices_file
sampler built by train.build_age_balance_sampler (one implementation for training and for this read).
It then iterates --n_batches batches and reports, with a [batch_ages] prefix:
  * people per batch BEFORE and AFTER the valid-human filter of Trainer.prepare_gt
  * the GT age histogram of the SUPERVISED people and the fractions with age < 0.2 / 0.3 / 0.4
  * the fraction of batches with no supervised child (age < --child_age): the batches whose age head sees none
  * the fraction of sampled images whose youngest person is a child (what --age_balance_child_frac targets)
  * loader throughput in images/s (time spent waiting on the loader only) and the total wall time

The filter is REPLICATED, not reused: prepare_gt is a Trainer method (the Trainer needs the SMPL-X files and a
Model) and it needs the GT joints from an Anny body-model forward to know where the anchor projects, so it is not
callable without a body-model forward. dc.visible_gt_mask is the same re-implementation diag_holdout_trace.py
scores with: anchor joint in front of the camera (z > 0.1), projecting inside the img_size frame, one person
per 14 px heatmap cell, per image (train.py prepare_gt, the block after `n_patch = ...`). --anchor_joint root is
the production target; head replicates exp1's --gt_center_joint person_center. --filter none skips the body
model (before-filter counts only, no anny needed). bulletproof_collate is a closure inside train.py main() and
is copied here verbatim.

    python scripts/experiments/measure_batch_ages.py --n_batches 200 --out results/experiments/exp2_children/batch_ages_baseline.json
    python scripts/experiments/measure_batch_ages.py --n_batches 200 --age_balance 1 --exclude_indices_file <manifest idx> \
        --out results/experiments/exp2_children/batch_ages_balanced.json
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'diagnostics'))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
from torch.utils.data import DataLoader, Subset      # noqa: E402

import diag_common as dc                             # noqa: E402
from datasets.annyone import AnnyOne                 # noqa: E402
from datasets.bedlam import collate_fn as raw_collate  # noqa: E402

THRESHOLDS = (0.2, 0.3, 0.4)
AGE_COL = 1                                          # anny_shape[:, 1] = age (diag_common.PHENOTYPES)


def bulletproof_collate(batch):
    """Verbatim copy of the closure train.py main() hands to its training DataLoader."""
    try:
        img, annot = raw_collate(batch)
    except Exception as e:  # noqa
        print(f"CRITICAL DATALOADER CRASH: {e}", flush=True)
        return None, None
    if annot is None:
        return None, None
    if 'idx' not in annot:
        annot['idx'] = torch.arange(img.shape[0])
    return img, annot


def build_loader(args):
    """train.py main(), --train_data AnnyOne with --train_n 0: same Subset, same DataLoader arguments."""
    full = AnnyOne(data_folder=args.data_folder, img_size=args.img_size)
    train_dataset = full
    if args.val_anny_n > 0:
        keep = len(full) - args.val_anny_n
        train_dataset = Subset(full, list(range(keep)))
        print(f"[loader] training positions 0..{keep - 1}; last {args.val_anny_n} = holdout, never drawn", flush=True)
    sampler = None
    if int(args.age_balance) or args.exclude_indices_file:
        import train as train_mod                     # the sampler lives next to the training loop
        ns = argparse.Namespace(age_balance=int(args.age_balance), age_balance_index=args.age_balance_index,
                                age_balance_child_frac=args.age_balance_child_frac,
                                age_balance_child_age=args.age_balance_child_age,
                                exclude_indices_file=args.exclude_indices_file)
        sampler = train_mod.build_age_balance_sampler(ns, train_dataset, full)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=bulletproof_collate,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    mode = 'balanced sampler' if int(args.age_balance) else ('exclusion sampler' if sampler is not None else 'shuffle (train.py default)')
    print(f"[loader] {len(train_dataset)} images, batch {args.batch_size}, {args.num_workers} workers, {mode}", flush=True)
    return full, train_dataset, loader


def load_body_model(device):
    import anny
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(device)
    bm.set_skinning_method('lbs')
    return bm


def supervised_ages(y, bm, labels, anchor_idx, img_size, patch_size, device):
    """GT ages of the people prepare_gt would keep in this batch (its visibility filter, per image)."""
    kept = []
    valid = y['valid_humans'].bool()
    for i in range(valid.shape[0]):
        v = valid[i]
        nh = int(v.sum())
        if nh == 0:
            continue
        pose = y['anny_pose'][i][v][:, :163].float().to(device)
        shape = y['anny_shape'][i][v].reshape(nh, -1)[:, :len(labels)].float().to(device)
        _, j = dc.run_body_model(bm, pose, shape)                 # clamps the phenotypes to [0,1] like prepare_gt
        keep = dc.visible_gt_mask(j[:, [anchor_idx]], y['K'][i:i + 1].to(device), img_size, patch_size)
        kept.extend(shape[keep.to(shape.device), AGE_COL].tolist())
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--val_anny_n', type=int, default=100, help='train.py --val_anny_n (last N pkls held out)')
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--n_batches', type=int, default=200)
    ap.add_argument('--age_balance', type=int, default=0, choices=[0, 1])
    ap.add_argument('--age_balance_index', default='results/experiments/exp2_children/age_index.npz')
    ap.add_argument('--age_balance_child_frac', type=float, default=0.25)
    ap.add_argument('--age_balance_child_age', type=float, default=0.3)
    ap.add_argument('--exclude_indices_file', default='')
    ap.add_argument('--child_age', type=float, default=0.3, help="'no child in the batch' threshold")
    ap.add_argument('--filter', default='bodymodel', choices=['bodymodel', 'none'],
                    help="bodymodel = replicate prepare_gt's visibility filter (needs anny); none = before-filter only")
    ap.add_argument('--anchor_joint', default='root', choices=['root', 'head'],
                    help='joint the filter tests: root = production targets, head = exp1 --gt_center_joint person_center')
    ap.add_argument('--patch_size', type=int, default=14)
    ap.add_argument('--seed', type=int, default=-1, help='>= 0 seeds torch for a repeatable read; train.py itself has no seed')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results/experiments/exp2_children/batch_ages.json')
    args = ap.parse_args()
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    full, train_dataset, loader = build_loader(args)
    bm = labels = None
    anchor_idx = 0
    if args.filter == 'bodymodel':
        bm = load_body_model(device)
        labels = list(bm.phenotype_labels)
        names = list(bm.bone_labels)
        anchor_idx = 0 if args.anchor_joint == 'root' else names.index('head')
        if labels != dc.PHENOTYPES:
            print(f"[WARN] body model phenotype order {labels} != assumed dataset order {dc.PHENOTYPES}")
        print(f"[filter] prepare_gt visibility filter replicated at joint {anchor_idx} ({names[anchor_idx]}), "
              f"{args.img_size} px frame, {args.patch_size} px cells", flush=True)
    else:
        print("[filter] none: the after-filter numbers below equal the before-filter numbers", flush=True)

    ages_before, ages_after, img_min_ages = [], [], []
    per_batch = []                                                 # (n_before, n_after, n_child_after)
    n_skipped = 0
    t_load = 0.0
    t0 = time.time()
    it = iter(loader)
    for b in range(args.n_batches):
        tb = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        t_load += time.time() - tb
        if batch is None or batch[0] is None:
            n_skipped += 1
            continue
        _, y = batch
        valid = y['valid_humans'].bool()
        a_before = y['anny_shape'][valid][:, AGE_COL].tolist()
        for i in range(valid.shape[0]):
            if bool(valid[i].any()):
                img_min_ages.append(float(y['anny_shape'][i][valid[i]][:, AGE_COL].min()))
        if bm is not None:
            with torch.no_grad():
                a_after = supervised_ages(y, bm, labels, anchor_idx, args.img_size, args.patch_size, device)
        else:
            a_after = list(a_before)
        ages_before.extend(a_before); ages_after.extend(a_after)
        per_batch.append((len(a_before), len(a_after), sum(1 for a in a_after if a < args.child_age)))
        if b % 50 == 0 and b:
            print(f"  ...{b}/{args.n_batches} batches ({time.time() - t0:.0f}s)", flush=True)
    wall = time.time() - t0
    n_b = len(per_batch)
    if n_b == 0:
        raise SystemExit("[batch_ages] no batch was read")
    n_img = n_b * args.batch_size
    ab, aa = np.asarray(ages_before), np.asarray(ages_after)
    pb = np.asarray(per_batch, dtype=np.float64)
    bins = np.linspace(0, 1, 11)
    hist_after, _ = np.histogram(aa, bins=bins)
    mode = 'balanced' if int(args.age_balance) else 'baseline'
    res = dict(mode=mode, n_batches=n_b, n_images=n_img, batch_size=args.batch_size, skipped_batches=n_skipped,
               loader_images_per_s=n_img / max(t_load, 1e-9), wall_s=wall, filter=args.filter,
               anchor_joint=args.anchor_joint, child_age=args.child_age,
               people_per_batch_before=float(pb[:, 0].mean()), people_per_batch_after=float(pb[:, 1].mean()),
               kept_frac=float(pb[:, 1].sum() / max(pb[:, 0].sum(), 1)),
               n_people_before=int(len(ab)), n_people_after=int(len(aa)),
               frac_below_before={str(t): float((ab < t).mean()) if len(ab) else None for t in THRESHOLDS},
               frac_below_after={str(t): float((aa < t).mean()) if len(aa) else None for t in THRESHOLDS},
               batches_without_child_after=float((pb[:, 2] == 0).mean()),
               images_with_child_frac=float(np.mean(np.asarray(img_min_ages) < args.age_balance_child_age)) if img_min_ages else None,
               age_hist_after={f'{bins[i]:.1f}-{bins[i + 1]:.1f}': int(hist_after[i]) for i in range(10)},
               mean_age_after=float(aa.mean()) if len(aa) else None,
               age_balance=dict(on=int(args.age_balance), index=args.age_balance_index, child_frac=args.age_balance_child_frac,
                                child_age=args.age_balance_child_age, exclude_indices_file=args.exclude_indices_file),
               seed=args.seed)

    print(f"\n[batch_ages] mode={mode} batches={n_b} images={n_img} (skipped {n_skipped}); "
          f"loader {res['loader_images_per_s']:.1f} images/s (loader wait only), wall {wall:.0f}s incl. body model")
    print(f"[batch_ages] people/batch before filter {res['people_per_batch_before']:.2f}, after {res['people_per_batch_after']:.2f} "
          f"(kept {100 * res['kept_frac']:.1f}%)")
    print("[batch_ages] supervised people with age < 0.2 / 0.3 / 0.4: " +
          " / ".join(f"{100 * res['frac_below_after'][str(t)]:.2f}%" for t in THRESHOLDS) +
          "   (before filter: " + " / ".join(f"{100 * res['frac_below_before'][str(t)]:.2f}%" for t in THRESHOLDS) + ")")
    print(f"[batch_ages] batches with NO supervised person below {args.child_age}: {100 * res['batches_without_child_after']:.1f}%")
    print(f"[batch_ages] sampled images whose youngest person is below {args.age_balance_child_age}: "
          f"{100 * (res['images_with_child_frac'] or 0):.1f}%"
          + (f" (sampler target {100 * args.age_balance_child_frac:.0f}%)" if int(args.age_balance) else " (natural rate)"))
    print("[batch_ages] age histogram of supervised people:")
    for k, v in res['age_hist_after'].items():
        print(f"  {k}: {v:6d}  {'#' * int(60 * v / max(hist_after.max(), 1))}")
    print("READ: the baseline numbers are what the shape head was trained on (2.53% of people < 0.3 in the E2 pkl scan;"
          " ~79% childless batches was arithmetic, this is the measurement). The balanced run should move the people-level"
          " fraction toward the [age_balance] prediction printed above and the childless-batch fraction toward zero; adults"
          " are still the large majority of supervised people at child_frac 0.25.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(dc.json_safe(res), f, indent=2, allow_nan=False)
    print(f"[batch_ages] wrote {args.out}")


if __name__ == '__main__':
    main()
