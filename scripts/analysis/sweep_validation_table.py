"""
Validation-based reporting of the KD-temperature sweep and of the training length (supervisor feedback of 12 Nov:
"Temperature sweep reports results on the 3DPW test set. Hyperparameter selection should use validation data. Test
set for final evaluation. Clarify how the final temperature and training duration were selected.").

Every exp8 run logs, at each in-training evaluation (train.py evaluate(), student with its training bone fix, V0
setting), the metrics on two VALIDATION sets that no post-step and no final table uses:
    annyone-holdout-1             the 100 AnnyOne holdout images (--val_anny_n 100)
    threedpw_anny-validation-1    300 images of the 3DPW validation split (--val_3dpw_split validation --val_3dpw_n 300)
The exp9 sweep runs have them for every epoch (EVAL_FREQ 1) together with the validation loss ('[val_loss]' lines,
VAL_LOSS 1); the reported 30- and 60-epoch runs have the metrics every 5 epochs and no validation loss. A run whose
post-steps failed after the training (the T 16 arm, job 3717598) has all of this too, so it is included.

Per run folder this script prints and saves (CSV + markdown):
    KD_T, epochs trained, the number of in-training evaluations,
    the FINAL-epoch validation PVE / PA-PVE / MPJPE / recall on 3DPW-validation and on the AnnyOne holdout,
    the final-epoch validation loss total per dataset (when the run has one),
    the epoch with the LOWEST 3DPW-validation PVE, its value, and final minus best (is the last epoch near the best?),
then, over the sweep runs (folder name contains _kdT<t>_, see --sweep_regex): which T has the lowest final-epoch
3DPW-validation PVE, the spread across T (max - min), and the same for the AnnyOne holdout; and, for the other runs
given (the reported run and the 60-epoch run), the final-epoch validation numbers by training length.
Missing pieces (no log, no validation loss, a dataset that was not evaluated) stay empty; nothing is invented.

SOURCES: scripts/analysis/plot_loss_curves.py's readers, i.e. the tensorboard events of <RUN>/train/ when the
tensorboard package imports (full float precision), else <RUN>/train.log (train.py prints one decimal); --source log
forces the log. The source used is a column of the table. KD_T comes from kd_temperature_sweep.read_temperature
(settings.txt, the folder name, or train.log's '[distill] lambda_kd=... T=...' line).

Usage (login node, CPU; no torch):
    python scripts/analysis/sweep_validation_table.py \
        "results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30_kdT*_sweep" \
        results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30 \
        results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e60 \
        --out results/experiments/exp10_kd_ablation/validation
Outputs: <out>/sweep_validation_table.csv, <out>/sweep_validation_table.md (the run folders are not written to).
"""
import argparse
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import kd_temperature_sweep as sweep      # noqa: E402  (read_temperature, expand_runs)
import plot_loss_curves as curves         # noqa: E402  (load_run: tensorboard or train.log)

VAL_3DPW = 'threedpw_anny-validation'     # dataset key prefixes as train.py names them (<name>-<split>-<subsample>)
VAL_HOLDOUT = 'annyone-holdout'
METRICS = ('pve', 'pa_pve', 'mpjpe', 'recall')
SETS = (('3dpwval', '3DPW-val'), ('holdout', 'AnnyOne holdout'))
COLUMNS = (['run', 'group', 'kd_t', 'epochs', 'n_evals', 'final_eval_epoch']
           + [f'{s}_{m}' for s, _ in SETS for m in METRICS]
           + ['3dpwval_loss_total', 'holdout_loss_total',
              'best_3dpwval_pve_epoch', 'best_3dpwval_pve', 'final_minus_best_3dpwval_pve',
              'best_holdout_pve_epoch', 'best_holdout_pve', 'source'])


def pick_dataset(d, prefix):
    """The entry of {dataset key: ...} whose key starts with `prefix` (the last one when several), or {}."""
    keys = sorted(k for k in d if k.startswith(prefix))
    return d[keys[-1]] if keys else {}


def best_epoch(series, metric='pve'):
    """(epoch, value) of the lowest `metric` in {epoch: {metric: value}}; the earliest epoch on a tie; None if empty."""
    cand = [(v[metric], e) for e, v in series.items() if v.get(metric) is not None]
    if not cand:
        return None
    v, e = min(cand)
    return e, v


def collect(run_dir, source='auto', sweep_regex=r'_kdT[0-9.eE+-]+_'):
    """One table row (dict over COLUMNS) for a run folder; whatever the run does not have stays None."""
    name = os.path.basename(os.path.normpath(run_dir))
    row = {c: None for c in COLUMNS}
    row['run'] = name
    row['group'] = 'sweep' if re.search(sweep_regex, name + '_') else 'other'
    row['kd_t'] = sweep.read_temperature(run_dir)
    data, src = curves.load_run(run_dir, source)
    row['source'] = src
    if data is None:
        return row
    val3, valh = pick_dataset(data['val'], VAL_3DPW), pick_dataset(data['val'], VAL_HOLDOUT)
    epochs = sorted(set(data['train']) | set(data['steps']) | set(val3) | set(valh))
    row['epochs'] = epochs[-1] if epochs else None
    eval_epochs = sorted(e for e in set(val3) | set(valh) if val3.get(e) or valh.get(e))
    row['n_evals'] = len(eval_epochs)
    row['final_eval_epoch'] = eval_epochs[-1] if eval_epochs else None
    for key, series in (('3dpwval', val3), ('holdout', valh)):
        last = max((e for e in series if series[e]), default=None)
        if last is None:
            continue
        for m in METRICS:
            if series[last].get(m) is not None:
                row[f'{key}_{m}'] = float(series[last][m])
    for key, prefix in (('3dpwval', VAL_3DPW), ('holdout', VAL_HOLDOUT)):
        vl = pick_dataset(data['val_loss'], prefix)
        last = max((e for e in vl if vl[e].get('total') is not None), default=None)
        if last is not None:
            row[f'{key}_loss_total'] = float(vl[last]['total'])
    b = best_epoch(val3)
    if b is not None:
        row['best_3dpwval_pve_epoch'], row['best_3dpwval_pve'] = b[0], float(b[1])
        if row['3dpwval_pve'] is not None:
            row['final_minus_best_3dpwval_pve'] = row['3dpwval_pve'] - row['best_3dpwval_pve']
    b = best_epoch(valh)
    if b is not None:
        row['best_holdout_pve_epoch'], row['best_holdout_pve'] = b[0], float(b[1])
    return row


def fmt(x, nd=1):
    if x is None:
        return '-'
    if isinstance(x, int):
        return str(x)
    return format(x, f'.{nd}f')


def tfmt(t):
    return '?' if t is None else format(t, 'g')


def markdown(rows):
    head = ['run', 'T', 'epochs', 'evals', '3DPW-val PVE', 'PA-PVE', 'MPJPE', 'recall',
            'holdout PVE', 'PA-PVE', 'MPJPE', 'recall', 'val loss 3DPW-val', 'val loss holdout',
            'best 3DPW-val PVE (epoch)', 'final - best', 'source']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        best = '-' if r['best_3dpwval_pve'] is None else f"{fmt(r['best_3dpwval_pve'])} ({r['best_3dpwval_pve_epoch']})"
        d = r['final_minus_best_3dpwval_pve']
        lines.append('| ' + ' | '.join([
            r['run'], tfmt(r['kd_t']), fmt(r['epochs']), fmt(r['n_evals']),
            fmt(r['3dpwval_pve']), fmt(r['3dpwval_pa_pve']), fmt(r['3dpwval_mpjpe']), fmt(r['3dpwval_recall']),
            fmt(r['holdout_pve']), fmt(r['holdout_pa_pve']), fmt(r['holdout_mpjpe']), fmt(r['holdout_recall']),
            fmt(r['3dpwval_loss_total'], 3), fmt(r['holdout_loss_total'], 3), best,
            '-' if d is None else format(d, '+.1f'), r['source'] or '-']) + ' |')
    return lines


def selection_lines(rows):
    """The plain statements under the table: the T with the lowest final-epoch validation PVE and the spread across
    T over the sweep runs; the final-epoch validation numbers of the other runs by training length."""
    out = []
    sw = [r for r in rows if r['group'] == 'sweep' and r['kd_t'] is not None]
    for key, label in SETS:
        have = [(r[f'{key}_pve'], r['kd_t']) for r in sw if r[f'{key}_pve'] is not None]
        if not have:
            out.append(f'sweep, {label} PVE (final epoch): no sweep run with this number')
            continue
        lo, hi = min(have), max(have)
        missing = [tfmt(r['kd_t']) for r in sw if r[f'{key}_pve'] is None]
        out.append(f'sweep, {label} PVE (final epoch): lowest at T = {tfmt(lo[1])} ({lo[0]:.1f} mm), highest at '
                   f'T = {tfmt(hi[1])} ({hi[0]:.1f} mm); spread across {len(have)} temperatures (max - min) = '
                   f'{hi[0] - lo[0]:.1f} mm' + (f'; no value for T = {", ".join(missing)}' if missing else ''))
    late = [(r['final_minus_best_3dpwval_pve'], r) for r in rows if r['final_minus_best_3dpwval_pve'] is not None]
    if late:
        worst = max(late, key=lambda x: x[0])
        out.append(f'last epoch vs best epoch, 3DPW-val PVE: final - best is at most {worst[0]:+.1f} mm over the '
                   f'{len(late)} run(s) with a curve (largest in {worst[1]["run"]}: best {worst[1]["best_3dpwval_pve"]:.1f} '
                   f'at epoch {worst[1]["best_3dpwval_pve_epoch"]}, final {worst[1]["3dpwval_pve"]:.1f})')
    others = [r for r in rows if r['group'] != 'sweep' and r['epochs'] is not None]
    for r in sorted(others, key=lambda r: (r['epochs'], r['run'])):
        out.append(f'training length, {r["run"]} ({r["epochs"]} epochs, T = {tfmt(r["kd_t"])}): final-epoch 3DPW-val PVE '
                   f'{fmt(r["3dpwval_pve"])}, AnnyOne holdout PVE {fmt(r["holdout_pve"])} mm '
                   f'({fmt(r["n_evals"])} in-training evaluations)')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run folders (paths or quoted globs): the sweep arms and any other exp8 run')
    ap.add_argument('--out', default='results/experiments/exp10_kd_ablation/validation')
    ap.add_argument('--source', choices=['auto', 'tb', 'log'], default='auto',
                    help='auto: tensorboard events when readable, else train.log (as plot_loss_curves.py)')
    ap.add_argument('--sweep_regex', default=r'_kdT[0-9.eE+-]+_',
                    help='a run whose folder name matches is a sweep arm (the T comparison uses these only)')
    a = ap.parse_args(argv)

    run_dirs = sweep.expand_runs(a.runs)
    missing = [p for p in a.runs if not sweep.expand_runs([p])]
    if not run_dirs:
        print(f'ERROR: no run folder matches {a.runs}')
        return 1
    rows = [collect(d, a.source, a.sweep_regex) for d in run_dirs]
    rows.sort(key=lambda r: (r['group'] != 'sweep', r['kd_t'] is None, r['kd_t'] or 0.0, r['epochs'] or 0, r['run']))
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'sweep_validation_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if r[c] is None else r[c]) for c in COLUMNS})
    lines = ['IN-TRAINING VALIDATION metrics (train.py evaluate(), V0 setting, student with its training bone fix): '
             '3DPW-val = 300 images of the 3DPW validation split, holdout = the 100 AnnyOne holdout images. '
             'These are not the test numbers of the final tables (3DPW test: 8,190 images; holdout V1 trace).', '']
    lines += markdown(rows)
    lines.append('')
    sel = selection_lines(rows)
    lines += ['- ' + s for s in sel]
    no_curve = [r['run'] for r in rows if r['3dpwval_pve'] is None]
    if no_curve:
        lines.append('- NO 3DPW-validation numbers (no train.log / events, or the run has not evaluated yet): ' + ', '.join(no_curve))
    if missing:
        lines.append('- NOT FOUND (no such folder): ' + ', '.join(missing))
    md_path = os.path.join(a.out, 'sweep_validation_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('=== validation-based sweep / training-length table')
    for line in lines:
        print(line)
    print(f'wrote {csv_path}')
    print(f'wrote {md_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
