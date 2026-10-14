"""
exp2_children, step 1: per-image GT age index of the AnnyOne annotations (pkl reads only, no image decode).

Writes --out (default results/experiments/exp2_children/age_index.npz) with one entry per pkl in the
SAME sorted order as datasets/annyone.py (glob + sort == sorted(glob), the order train.py's Subset
indices refer to):
    min_age    [N] float32   youngest person's age phenotype (anny_shape[1]); NaN if no people
    mean_age   [N] float32   mean age phenotype over the people in the image; NaN if none
    n_people   [N] int32     len(humans)
    n_below    [N, 3] int32  people with age < thresholds (0.2, 0.3, 0.4)
    basename   [N] str       pkl basenames, so train.py --age_balance can refuse a stale index
plus thresholds, n_total, n_holdout (100 = train.py --val_anny_n, the LAST 100 pkls) and data_folder.

train.py --age_balance 1 reads min_age to build a per-image WeightedRandomSampler; this scan is the
only part of the experiment that touches all 579,960 annotations (I/O bound; --workers uses a Pool).

    python scripts/experiments/build_age_index.py --workers 16                 # full index (minutes with 16 processes)
    python scripts/experiments/build_age_index.py --limit 2000 --workers 4     # timing / smoke test (train.py rejects a partial index)

WHY SAMPLING AND NOT A SHAPE-SIDE FIX (read-only verification of the child/shape path, repo at 89db760;
quoted in docs/experiments/exp2_children.md with file:line):
  * all 11 phenotypes reach the body model and are supervised: HPH.decshape = nn.Linear(1024, 11)
    (model.py:544) plus a zero init_betas residual (model.py:566, 647); Model.forward applies sigmoid once
    (model.py:390) and forwards every phenotype by name into the Anny body model (model.py:413, 419);
    the loss is a plain L1 between that post-sigmoid prediction and the dataset's anny_shape clamped to
    [0,1] (loss.py:96-97; GT at train.py:148-159, 205), weighted by alpha_shape (20 in production).
  * no normalisation, no 0.5 prior and no per-phenotype weight exist on the loss side; the GT shape target
    is filtered by exactly the visibility filter the pose terms use (train.py:302-316, root in front of the
    camera and inside the image, one person per heatmap cell).
  * no adult-biased initialisation: zero logits give 0.5 on every phenotype (~18 y on Anny's age scale);
    the adult bias of the checkpoint is a data prior: 2.53% of training people have age < 0.3, 1.06% < 0.2,
    the holdout has none, and the 3DPW labels carry a constant age 0.55 (docs/diagnostics_findings.md).
  * the inference toggles neutral_shape / neutral_phenotype_names / relax_hands / rest_pose /
    mask_helper_joints are all gated by `not is_training` and default off (model.py:160, 393-404).
  So the lever is how often the shape head sees a child (sampling) and how much a child's age error
  costs (--shape_weights), not a missing phenotype or a wrong convention.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'diagnostics'))
try:
    import diag_common as dc                     # same constants as the diagnostic suite
    DATA_FOLDER_DEFAULT, N_HOLDOUT = dc.DATA_FOLDER_DEFAULT, dc.N_HOLDOUT
except Exception:                                # diag_common imports torch; the scan itself needs none
    DATA_FOLDER_DEFAULT, N_HOLDOUT = '/netscratch/najib/anydataset/', 100

THRESHOLDS = (0.2, 0.3, 0.4)


def list_pkls(data_folder):
    """sorted(glob) == AnnyOne.pkl_files (glob then .sort()); also diag_common.list_pkls."""
    files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))
    if not files:
        raise FileNotFoundError(f"no .pkl annotations under {data_folder}")
    return files


def scan_one(path):
    """(min_age, mean_age, n_people, n_below[3], ok). Same read as diag_common.shapes_from_pkl: age = anny_shape[1]."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ages = []
        for h in data.get('humans', []):
            s = h.get('anny_shape', None)
            if s is None:
                return (np.nan, np.nan, len(data.get('humans', [])), [0, 0, 0], False)
            ages.append(float(np.asarray(s, dtype=np.float32).reshape(-1)[1]))
        n = len(ages)
        if n == 0:
            return (np.nan, np.nan, 0, [0, 0, 0], True)
        a = np.asarray(ages)
        return (float(a.min()), float(a.mean()), n, [int((a < t).sum()) for t in THRESHOLDS], True)
    except Exception:  # noqa: a corrupt pkl must not kill the scan; it is counted and sampled like an adult image
        return (np.nan, np.nan, 0, [0, 0, 0], False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_folder', default=DATA_FOLDER_DEFAULT)
    ap.add_argument('--out', default='results/experiments/exp2_children/age_index.npz')
    ap.add_argument('--limit', type=int, default=0, help='scan only the first N pkls (smoke test / timing); 0 = all')
    ap.add_argument('--workers', type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument('--chunksize', type=int, default=256)
    args = ap.parse_args()

    t_list = time.time()
    pkls = list_pkls(args.data_folder)
    n_total = len(pkls)
    todo = pkls[:args.limit] if args.limit > 0 else pkls
    print(f"[index] {n_total} pkls under {args.data_folder} (listed in {time.time() - t_list:.1f}s); "
          f"scanning {len(todo)} with {args.workers} workers", flush=True)

    rows = []
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap(scan_one, todo, chunksize=args.chunksize)):
                rows.append(r)
                if k % 50000 == 0 and k:
                    el = time.time() - t0
                    print(f"  ...{k}/{len(todo)} ({el:.0f}s, {k / el:.0f} files/s)", flush=True)
    else:
        for k, path in enumerate(todo):
            rows.append(scan_one(path))
            if k % 50000 == 0 and k:
                el = time.time() - t0
                print(f"  ...{k}/{len(todo)} ({el:.0f}s, {k / el:.0f} files/s)", flush=True)
    el = time.time() - t0

    min_age = np.asarray([r[0] for r in rows], dtype=np.float32)
    mean_age = np.asarray([r[1] for r in rows], dtype=np.float32)
    n_people = np.asarray([r[2] for r in rows], dtype=np.int32)
    n_below = np.asarray([r[3] for r in rows], dtype=np.int32).reshape(len(rows), len(THRESHOLDS))
    ok = np.asarray([r[4] for r in rows], dtype=bool)
    basename = np.asarray([os.path.basename(p) for p in todo])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, min_age=min_age, mean_age=mean_age, n_people=n_people, n_below=n_below, basename=basename,
             thresholds=np.asarray(THRESHOLDS, dtype=np.float32), n_total=np.int64(n_total),
             n_scanned=np.int64(len(todo)), n_holdout=np.int64(N_HOLDOUT), data_folder=np.asarray(args.data_folder))

    n_scan = len(todo)
    per_10k = 1e4 * el / max(n_scan, 1)
    print(f"[index] scanned {n_scan} pkls in {el:.1f}s = {n_scan / max(el, 1e-9):.0f} files/s "
          f"-> {per_10k:.1f}s per 10k files, {n_total * el / max(n_scan, 1) / 60:.1f} min for all {n_total}")
    print(f"[index] unreadable / shape-less pkls: {int((~ok).sum())}; images without people: "
          f"{int((n_people == 0).sum())}; people per image {n_people.mean():.2f}")

    def report(mask, label):
        m = mask & np.isfinite(min_age)
        n_img = int(m.sum())
        if n_img == 0:
            print(f"[index] {label}: no images"); return {}
        d = dict(n_images=n_img, n_people=int(n_people[m].sum()))
        line = []
        for c, t in enumerate(THRESHOLDS):
            fi = float((min_age[m] < t).mean())
            fp = float(n_below[m, c].sum() / max(n_people[m].sum(), 1))
            d[f'images_min_age_lt_{t}'] = fi
            d[f'people_age_lt_{t}'] = fp
            line.append(f"<{t}: {100 * fi:.2f}% of images / {100 * fp:.2f}% of people")
        print(f"[index] {label} ({n_img} images, {d['n_people']} people): " + "; ".join(line))
        return d

    pos = np.arange(n_scan)
    summary = dict(n_total=n_total, n_scanned=n_scan, seconds=el, seconds_per_10k=per_10k, workers=args.workers,
                   all=report(pos >= 0, 'all scanned'))
    if n_scan == n_total:
        summary['train'] = report(pos < n_total - N_HOLDOUT, f'TRAIN pool (first {n_total - N_HOLDOUT})')
        summary['holdout'] = report(pos >= n_total - N_HOLDOUT, f'HOLDOUT (last {N_HOLDOUT}, never trained on)')
    else:
        print(f"[index] PARTIAL index ({n_scan} of {n_total}): train.py --age_balance will refuse it; drop --limit for the real one")
    with open(os.path.splitext(args.out)[0] + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[index] wrote {args.out} and {os.path.splitext(args.out)[0]}_summary.json")
    print("READ: 'images with min_age < 0.3' is the natural child-image fraction the sampler reweights toward "
          "--age_balance_child_frac; the people-level fraction (2.53% < 0.3 in the E2 scan of 54,784 people) is "
          "what reaches the shape head, before the visibility filter.")


if __name__ == '__main__':
    main()
