"""
Detection recall on Relative Human by how crowded the image is (supervisor feedback of 12 Nov, "Multi-person
evaluation: ... particularly detection recall, occlusions, crowded scenes").

Input: one or more per_person.csv files of scripts/diagnostics/eval_relative_human.py (one row per LABELLED Relative
Human person; columns image, matched (1 = a detection was matched one-to-one to this person's box at the IoU the
evaluation ran with, default 0.3), ...). The number of rows of an image is its number of labelled people. Images are
binned by that number (default bins 1, 2, 3-4, 5-8, 9+) and per bin the script reports the images, the labelled
people and the recall = matched people / labelled people, plus the overall recall. Relative Human has no occlusion
label, so crowding (people per image) is the available proxy; precision is not in per_person.csv (unmatched
detections are not rows) and is not reported.

Each file is given as PATH or NAME=PATH; without NAME the model name is taken from the file name
(rh_per_person_<name>.csv) or from the evaluation folder (relative_human_<split>_<name>/per_person.csv).

    python scripts/analysis/rh_recall_by_crowd.py final=docs/thesis/data/rh_per_person_final.csv \
        base=docs/thesis/data/rh_per_person_base.csv --out results/analysis/rh_recall_by_crowd
    python scripts/analysis/rh_recall_by_crowd.py \
        student=results/diagnostics/relative_human_val_student_vitb_e30_00029/per_person.csv \
        final=results/diagnostics/relative_human_val_exp7_v5_final_00009/per_person.csv --out results/analysis/rh_recall_by_crowd
Outputs: <out>/rh_recall_by_crowd.csv (model, bin, images, people, recall, matched, people_per_image; the first five
columns as docs/thesis/data/rh_recall_by_crowd.csv, plus one 'All' row per model) and <out>/rh_recall_by_crowd.md.
CPU, standard library only.
"""
import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

DEFAULT_BINS = '1,2,3-4,5-8,9+'
BIN_NAMES = {'1': 'One', '2': 'Two', '3-4': 'ThreeFour', '5-8': 'FiveEight', '9+': 'NinePlus'}
COLUMNS = ['model', 'bin', 'images', 'people', 'recall', 'matched', 'people_per_image']


def parse_bins(spec):
    """'1,2,3-4,5-8,9+' -> [(label, lo, hi)] with hi None for an open bin; bins must not overlap."""
    bins = []
    for part in [p.strip() for p in spec.split(',') if p.strip()]:
        m = re.fullmatch(r'(\d+)(?:-(\d+)|(\+))?', part)
        if not m:
            raise ValueError(f'bad bin {part!r} (use N, N-M or N+)')
        lo = int(m.group(1))
        hi = None if m.group(3) else int(m.group(2) or lo)
        if lo < 1 or (hi is not None and hi < lo):
            raise ValueError(f'bad bin {part!r}')
        bins.append((part, lo, hi))
    for (_, lo1, hi1), (l2, lo2, _) in zip(bins, bins[1:]):
        if hi1 is None or lo2 <= hi1:
            raise ValueError(f'bins overlap or are not ascending at {l2!r}')
    if not bins:
        raise ValueError('no bins')
    return bins


def bin_of(n, bins):
    for label, lo, hi in bins:
        if n >= lo and (hi is None or n <= hi):
            return label
    return None


def model_name(path):
    base = os.path.basename(path)
    m = re.fullmatch(r'rh_per_person_(.+)\.csv', base)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    m = re.fullmatch(r'relative_human_[a-z]+_(.+)', parent)
    if m:
        return m.group(1)
    return parent if base == 'per_person.csv' and parent else os.path.splitext(base)[0]


def read_people(path):
    """{image: [matched 0/1 per labelled person]} in file order; raises on a file without the two columns."""
    per_image = OrderedDict()
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or 'image' not in rd.fieldnames or 'matched' not in rd.fieldnames:
            raise KeyError(f"{path}: needs the columns 'image' and 'matched' (has {rd.fieldnames})")
        for i, r in enumerate(rd):
            v = (r['matched'] or '').strip().lower()
            if v in ('1', '1.0', 'true'):
                m = 1
            elif v in ('0', '0.0', 'false'):
                m = 0
            else:
                raise ValueError(f"{path}: row {i + 2}: matched = {r['matched']!r} (expected 0 / 1)")
            per_image.setdefault(r['image'], []).append(m)
    if not per_image:
        raise ValueError(f'{path}: no rows')
    return per_image


def recall_by_crowd(per_image, bins):
    """[(bin label, images, people, matched, recall or None)] in bin order, then ('All', ...); people in images
    outside every bin are counted in All only (reported by the caller)."""
    acc = OrderedDict((label, [0, 0, 0]) for label, _, _ in bins)
    tot = [0, 0, 0]
    outside = 0
    for flags in per_image.values():
        n, k = len(flags), sum(flags)
        tot[0] += 1; tot[1] += n; tot[2] += k
        b = bin_of(n, bins)
        if b is None:
            outside += 1
            continue
        acc[b][0] += 1; acc[b][1] += n; acc[b][2] += k
    rows = [(label, im, pe, ma, (ma / pe) if pe else None) for label, (im, pe, ma) in acc.items()]
    rows.append(('All', tot[0], tot[1], tot[2], (tot[2] / tot[1]) if tot[1] else None))
    return rows, outside


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='per_person.csv files, each PATH or NAME=PATH')
    ap.add_argument('--bins', default=DEFAULT_BINS, help='labelled people per image: N, N-M or N+ (default %(default)s)')
    ap.add_argument('--out', default='results/analysis/rh_recall_by_crowd')
    a = ap.parse_args(argv)
    try:
        bins = parse_bins(a.bins)
    except ValueError as e:
        print(f'ERROR: --bins {a.bins}: {e}')
        return 1
    models = OrderedDict()
    for spec in a.files:
        name, path = spec.split('=', 1) if ('=' in spec and not os.path.isfile(spec)) else (None, spec)
        if not os.path.isfile(path):
            print(f'ERROR: {path} not found')
            return 1
        name = name or model_name(path)
        if name in models:
            print(f'ERROR: two files named {name}; give NAME=PATH')
            return 1
        try:
            models[name] = recall_by_crowd(read_people(path), bins)
        except (KeyError, ValueError) as e:
            print(f'ERROR: {e}')
            return 1
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'rh_recall_by_crowd.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for name, (rows, _) in models.items():
            for label, im, pe, ma, rec in rows:
                w.writerow([name, BIN_NAMES.get(label, label), im, pe, '' if rec is None else f'{rec:.3f}', ma, label if label != 'All' else 'all'])
    names = list(models)
    first = models[names[0]][0]
    same_gt = all([(r[0], r[1], r[2]) for r in models[n][0]] == [(r[0], r[1], r[2]) for r in first] for n in names)
    L = ['Relative Human detection recall by labelled people per image (matched people / labelled people; one-to-one '
         'box matching at the IoU of the evaluation, 0.3 by default).', '']
    if same_gt:
        L.append('| people per image | images | people | ' + ' | '.join(f'recall {n}' for n in names) + ' |')
        L.append('|---|---|---|' + '---|' * len(names))
        for i, (label, im, pe, _, _) in enumerate(first):
            recs = ['-' if models[n][0][i][4] is None else f'{models[n][0][i][4]:.3f}' for n in names]
            L.append(f'| {label} | {im} | {pe} | ' + ' | '.join(recs) + ' |')
    else:
        L.append('(the files do not hold the same images / people: one table per model)')
        for n in names:
            L += ['', f'{n}:', '| people per image | images | people | matched | recall |', '|---|---|---|---|---|']
            for label, im, pe, ma, rec in models[n][0]:
                L.append(f'| {label} | {im} | {pe} | {ma} | {"-" if rec is None else format(rec, ".3f")} |')
    for n in names:
        if models[n][1]:
            L.append(f'NOTE: {n}: {models[n][1]} image(s) outside every bin (counted in All only)')
    with open(os.path.join(a.out, 'rh_recall_by_crowd.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    for line in L:
        print(line)
    print(f'wrote {csv_path}')
    print(f'wrote {os.path.join(a.out, "rh_recall_by_crowd.md")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
