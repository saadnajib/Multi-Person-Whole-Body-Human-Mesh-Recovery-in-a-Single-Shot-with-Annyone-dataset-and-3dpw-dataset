"""
E2 -- does the training data contain children, and can the body model make one?

Part A (data, CPU only, pkl reads, ~1-3 min for 20k samples):
    GT phenotype distribution over --n evenly spaced AnnyOne samples, with an
    AGE HISTOGRAM and the fraction of people below age 0.2 / 0.3 / 0.4, plus the
    same for the 100-sample holdout. Writes summary.json, all_shapes.npy and
    age_hist.png.

Part B (body model, CPU, ~1 min, --probe_body_model 1):
    Puts Anny in rest pose at the dataset-mean shape and sweeps the 'age'
    phenotype 0.0 -> 1.0, printing body height, neck length, neck/height and
    head/height. If height and proportions change the way a child's would, the
    MODEL can represent children and any failure is data- or network-side.

USAGE (login node is fine):
    python scripts/diagnostics/diag_phenotypes.py --n 20000 --out results/diagnostics/phenotypes
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_common as dc  # noqa: E402

import numpy as np  # noqa: E402


def part_a(args):
    pkls = dc.list_pkls(args.data_folder)
    n_total = len(pkls)
    train_pool = n_total - dc.N_HOLDOUT
    idx = np.linspace(0, train_pool - 1, min(args.n, train_pool)).astype(int)
    print(f"[data] {n_total} samples; scanning {len(idx)} evenly spaced TRAINING samples + {dc.N_HOLDOUT} holdout")

    def collect(indices):
        rows, per_img = [], []
        t0 = time.time()
        for k, i in enumerate(indices):
            try:
                s = dc.shapes_from_pkl(dc.load_pkl(pkls[i]))
            except Exception as e:  # noqa
                print(f"  sample {i} failed: {e}"); continue
            if s is None:
                print("ERROR: no 'anny_shape' in annotation; keys:", list(dc.load_pkl(pkls[i])['humans'][0].keys()))
                sys.exit(1)
            rows.append(s); per_img.append(len(s))
            if k % 5000 == 0 and k:
                print(f"  ...{k} ({time.time() - t0:.0f}s)", flush=True)
        return np.concatenate(rows, 0) if rows else np.zeros((0, 11)), np.asarray(per_img)

    P, per_img = collect(idx)
    H, _ = collect(dc.holdout_indices(n_total))
    print(f"[data] {P.shape[0]} people in the training sample ({per_img.mean():.2f}/image), {H.shape[0]} in the holdout")

    names = dc.PHENOTYPES
    out = dict(n_images=int(len(idx)), n_people=int(P.shape[0]), people_per_image=float(per_img.mean()),
               train={}, holdout={})
    print(f"\n{'phenotype':<12}{'mean':>7}{'std':>7}{'min':>7}{'p1':>7}{'p5':>7}{'p50':>7}{'p95':>7}{'max':>7}   | holdout mean  std")
    for j, n in enumerate(names):
        c = P[:, j]; h = H[:, j] if len(H) else np.asarray([np.nan])
        q = np.percentile(c, [1, 5, 50, 95])
        out['train'][n] = dict(mean=float(c.mean()), std=float(c.std()), min=float(c.min()), max=float(c.max()),
                               p1=float(q[0]), p5=float(q[1]), p50=float(q[2]), p95=float(q[3]))
        out['holdout'][n] = dict(mean=float(np.nanmean(h)), std=float(np.nanstd(h)))
        print(f"{n:<12}{c.mean():>7.3f}{c.std():>7.3f}{c.min():>7.3f}{q[0]:>7.3f}{q[1]:>7.3f}{q[2]:>7.3f}{q[3]:>7.3f}{c.max():>7.3f}"
              f"   | {np.nanmean(h):>7.3f}{np.nanstd(h):>6.3f}")

    age = P[:, 1]
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(age, bins=bins)
    out['age_hist'] = {f'{bins[i]:.1f}-{bins[i + 1]:.1f}': int(hist[i]) for i in range(10)}
    out['age_frac_below'] = {str(t): float((age < t).mean()) for t in (0.2, 0.3, 0.4, 0.5)}
    print("\n=== AGE HISTOGRAM (training sample) ===")
    for k, v in out['age_hist'].items():
        print(f"  {k}: {v:7d}  {'#' * int(60 * v / max(hist.max(), 1))}")
    print("fraction of people with age < 0.2 / 0.3 / 0.4 / 0.5: " +
          " / ".join(f"{out['age_frac_below'][str(t)] * 100:.2f}%" for t in (0.2, 0.3, 0.4, 0.5)))
    # correlation age <-> height phenotype: in Anny 'height' is a separate control
    cc = np.corrcoef(P[:, 1], P[:, 4])[0, 1] if len(P) > 2 else float('nan')
    out['corr_age_height_pheno'] = float(cc)
    print(f"corr(age, height phenotype) = {cc:+.3f}")
    print("\nREAD: Anny age 0.5 ~ adult; children sit well below ~0.3 (see Part B for what each value looks like).")
    print("      If <0.3 is a fraction of a percent, the network has almost no child supervision and an")
    print("      L1 shape loss toward a 0.5 prior will regress every child to an adult: a DATA cause.")
    print("      The 3DPW labels used from v7/v9 on have age FIXED at 0.55 (scripts/data/smpl_to_anny.py),")
    print("      so mixed training adds zero child supervision as well.")

    np.save(os.path.join(args.out, 'all_shapes.npy'), P)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].hist(age, bins=50); ax[0].set_title('GT age (training sample)'); ax[0].set_xlabel('age phenotype')
        ax[1].hist(P[:, 4], bins=50); ax[1].set_title('GT height phenotype'); ax[1].set_xlabel('height')
        fig.tight_layout(); fig.savefig(os.path.join(args.out, 'age_hist.png'), dpi=120)
    except Exception as e:  # noqa
        print(f"(no plot: {e})")
    return out, P.mean(0)


def part_b(args, mean_shape):
    try:
        import torch, anny  # noqa
    except Exception as e:  # noqa
        print(f"[probe] skipped: {e} (run inside the multihmr env)"); return None
    print("\n=== BODY-MODEL PROBE: rest pose, dataset-mean shape, age sweep ===")
    bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float()
    bm.set_skinning_method('lbs')
    names = list(bm.bone_labels); phen = list(bm.phenotype_labels)
    chain = dc.neck_chain(names)
    ages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    base = torch.tensor(mean_shape, dtype=torch.float32).reshape(1, 11).repeat(len(ages), 1)
    base[:, phen.index('age')] = torch.tensor(ages)
    v, j = dc.run_body_model(bm, dc.identity_pose(len(ages), len(names), torch.device('cpu')), base)
    height = dc.body_height(j, names); neck = dc.chain_length(j, chain); head = dc.head_size(v, j, names)
    rows = []
    print(f"{'age':>5}{'height_m':>10}{'neck_m':>9}{'neck/height':>13}{'head/height':>13}")
    for a, h, n, hd in zip(ages, height.tolist(), neck.tolist(), head.tolist()):
        print(f"{a:>5.1f}{h:>10.3f}{n:>9.4f}{n / h:>13.4f}{hd / h:>13.4f}")
        rows.append(dict(age=a, height_m=h, neck_m=n, neck_over_height=n / h, head_over_height=hd / h))
    print("READ: a real child has a LARGER head/height and a SMALLER height. If the sweep shows that,")
    print("      Anny can represent children and the adult-looking outputs are a prediction (data/loss) problem.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_folder', default=dc.DATA_FOLDER_DEFAULT)
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--out', default='results/diagnostics/phenotypes')
    ap.add_argument('--probe_body_model', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out, mean_shape = part_a(args)
    if args.probe_body_model:
        out['body_model_age_sweep'] = part_b(args, mean_shape)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}/summary.json")


if __name__ == '__main__':
    main()
