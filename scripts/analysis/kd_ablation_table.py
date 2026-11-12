"""
The distillation ablation table (slurm/experiments/exp10_kd_ablation.sh; supervisor feedback of 12 Nov: "Missing
important ablation: ViT-B without distillation. Same training data and schedule.").

Rows (one exp8 run folder each; a folder that does not exist yet, or holds no number yet, is printed as "pending"):
    reported     the reported Student (KD_T 4, job 3614191)                       --reported
    sweep_t4     the temperature sweep's own T 4 arm: the same recipe trained again with EVAL_FREQ 1 / VAL_LOSS 1,
                 i.e. the like-for-like reference of the two ablation arms and, against "reported", the run-to-run
                 noise                                                              --sweep_t4
    nokd         KD_MODE=nokd: no distillation loss; teacher heads copied + frozen, teacher bone fix in training  --nokd
    supervised   KD_MODE=supervised: no teacher in training at all                  --supervised
    teacher      the ViT-L teacher as re-measured in the reported run's job (test columns only, for orientation)
Columns:
    in-training VALIDATION, final epoch (train.py evaluate(), V0 setting, the model with its training bone fix):
        3DPW-val (300 images of the 3DPW validation split) and AnnyOne holdout (100 images): PVE / PA-PVE / MPJPE / recall;
        the validation loss per dataset when the run has one: total as logged, the TASK part (total - lambda_kd x kd
        - lambda_kd_out x kd_out, comparable between arms because the distillation terms are removed), and the
        kd / kd_out values themselves (with KD_MODE=nokd they are monitors of the distance to the teacher);
    final TEST numbers (exp8 post-steps, the model with its own re-estimated bone fix):
        AnnyOne holdout V1 trace (100 images): PVE / PA-PVE / MPJPE / recall; 3DPW test (8,190 images, v23 labels):
        PVE / PA-PVE / MPJPE / recall.
Then the differences that the ablation is about, wherever both sides exist: sweep_t4 - reported (run-to-run noise),
nokd - sweep_t4 (what removing the two distillation losses costs), supervised - nokd (what the copied teacher heads
and the teacher's bone fix add), supervised - sweep_t4 (everything the teacher contributes). Nothing is invented: a
missing number is "-", a missing run "pending".

Readers: scripts/analysis/sweep_validation_table.py (validation columns; tensorboard events or train.log) and
scripts/analysis/kd_temperature_sweep.py (test columns: holdout_student/summary.json, eval_3dpw_test_student.log,
final_table.json as the fallback).

Usage (login node, CPU; no torch):
    P=results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30
    python scripts/analysis/kd_ablation_table.py --reported $P --sweep_t4 ${P}_kdT4_sweep --nokd ${P}_nokd \
        --supervised ${P}_supervised --out results/experiments/exp10_kd_ablation
Outputs: <out>/kd_ablation_table.csv, <out>/kd_ablation_table.md.
"""
import argparse
import csv
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import kd_temperature_sweep as sweep      # noqa: E402
import plot_loss_curves as curves         # noqa: E402
import sweep_validation_table as valtab   # noqa: E402

ARMS = (('reported', 'Student, reported (T 4, job 3614191)', 'full'),
        ('sweep_t4', 'Student, sweep T 4 arm (same recipe, re-run)', 'full'),
        ('nokd', 'no distillation loss (teacher heads + bone fix kept)', 'nokd'),
        ('supervised', 'no teacher in training (supervised only)', 'supervised'))
METRICS = ('pve', 'pa_pve', 'mpjpe', 'recall')
VAL_COLS = [f'val_{s}_{m}' for s in ('3dpwval', 'holdout') for m in METRICS]
LOSS_COLS = [f'val_loss_{s}_{k}' for s in ('3dpwval', 'holdout') for k in ('total', 'task', 'kd', 'kd_out')]
TEST_COLS = [f'test_{s}_{m}' for s in ('holdout', '3dpw') for m in METRICS]
COLUMNS = ['arm', 'label', 'run', 'status', 'kd_mode', 'kd_t', 'lambda_kd', 'lambda_kd_out', 'epochs',
           'final_eval_epoch'] + VAL_COLS + LOSS_COLS + TEST_COLS + ['val_source']
DIFFS = (('sweep_t4', 'reported', 'run-to-run noise (same recipe trained twice)'),
         ('nokd', 'sweep_t4', 'removing the two distillation losses (head copy, head freeze, teacher bone fix kept)'),
         ('supervised', 'nokd', 'also removing the copied teacher heads and the teacher bone fix'),
         ('supervised', 'sweep_t4', 'removing everything the teacher contributes'))
DIFF_COLS = ('val_3dpwval_pve', 'val_holdout_pve', 'test_holdout_pve', 'test_holdout_mpjpe', 'test_3dpw_pve', 'test_3dpw_mpjpe')


def read_settings(run_dir):
    """{KD_MODE, LAMBDA_KD, LAMBDA_KD_OUT} of a run, from settings.txt (KEY=value words), final_table.json's inputs
    (kd_mode) and, for a folder that predates KD_MODE, the default 'full'. Values that nothing states stay None."""
    out = dict(kd_mode=None, lambda_kd=None, lambda_kd_out=None)
    p = os.path.join(run_dir, 'settings.txt')
    if os.path.isfile(p):
        head = open(p, errors='replace').read()      # KEY=value words; the train.py command line has no KEY= word
        for key, name, conv in (('KD_MODE', 'kd_mode', str), ('LAMBDA_KD', 'lambda_kd', float),
                                ('LAMBDA_KD_OUT', 'lambda_kd_out', float)):
            m = re.search(r'(?:^|\s)%s=(\S+)' % key, head)
            if m:
                try:
                    out[name] = conv(m.group(1))
                except ValueError:
                    pass
        if out['kd_mode'] is None:
            out['kd_mode'] = 'full'          # written by an exp8 without KD_MODE: a distilled run
    try:
        km = json.load(open(os.path.join(run_dir, 'final_table.json')))['inputs'].get('kd_mode')
        if km:
            out['kd_mode'] = km
    except Exception:
        pass
    return out


def val_loss_terms(run_dir, source):
    """{'3dpwval': {term: value}, 'holdout': {...}} of the LAST epoch that has a validation loss; {} per set if none."""
    out = {'3dpwval': {}, 'holdout': {}}
    data, _ = curves.load_run(run_dir, source)
    if data is None:
        return out
    for key, prefix in (('3dpwval', valtab.VAL_3DPW), ('holdout', valtab.VAL_HOLDOUT)):
        vl = valtab.pick_dataset(data['val_loss'], prefix)
        last = max((e for e in vl if vl[e].get('total') is not None), default=None)
        if last is not None:
            out[key] = dict(vl[last])
    return out


def collect(arm, label, expect_mode, run_dir, source='auto'):
    row = {c: None for c in COLUMNS}
    row.update(arm=arm, label=label, run=os.path.basename(os.path.normpath(run_dir)) if run_dir else None, status='pending')
    if not run_dir or not os.path.isdir(run_dir):
        return row, []
    notes = []
    st = read_settings(run_dir)
    row.update(kd_mode=st['kd_mode'], lambda_kd=st['lambda_kd'], lambda_kd_out=st['lambda_kd_out'])
    if st['kd_mode'] is not None and st['kd_mode'] != expect_mode:
        notes.append(f'{arm}: {row["run"]} records KD_MODE={st["kd_mode"]}, expected {expect_mode}: wrong folder for this row?')
    v = valtab.collect(run_dir, source)
    row.update(kd_t=v['kd_t'], epochs=v['epochs'], final_eval_epoch=v['final_eval_epoch'], val_source=v['source'])
    for s in ('3dpwval', 'holdout'):
        for m in METRICS:
            row[f'val_{s}_{m}'] = v[f'{s}_{m}']
    terms = val_loss_terms(run_dir, source)
    for s in ('3dpwval', 'holdout'):
        t = terms[s]
        if 'total' not in t:
            continue
        row[f'val_loss_{s}_total'] = t['total']
        row[f'val_loss_{s}_kd'] = t.get('kd')
        row[f'val_loss_{s}_kd_out'] = t.get('kd_out')
        if 'kd' not in t and 'kd_out' not in t:
            row[f'val_loss_{s}_task'] = t['total']                 # no teacher loaded: total is the task loss
        elif st['kd_mode'] in ('nokd', 'supervised'):
            row[f'val_loss_{s}_task'] = t['total']                 # kd / kd_out are monitors outside total
        elif st['lambda_kd'] is not None and st['lambda_kd_out'] is not None:
            row[f'val_loss_{s}_task'] = t['total'] - st['lambda_kd'] * t.get('kd', 0.0) - st['lambda_kd_out'] * t.get('kd_out', 0.0)
    t = sweep.collect_run(run_dir, arm)
    for s in ('holdout', '3dpw'):
        for m in METRICS:
            row[f'test_{s}_{m}'] = t[f'{s}_{m}']
    has_val = row['val_3dpwval_pve'] is not None or row['val_holdout_pve'] is not None
    has_test = row['test_holdout_pve'] is not None and row['test_3dpw_pve'] is not None
    if has_test:
        row['status'] = 'complete'
    elif has_val or row['epochs']:
        row['status'] = 'training or post-steps pending'
    if row['epochs'] and row['final_eval_epoch'] and row['final_eval_epoch'] != row['epochs']:
        notes.append(f'{arm}: last in-training evaluation at epoch {row["final_eval_epoch"]} of {row["epochs"]} logged')
    return row, notes


def fmt(x, nd=1):
    return '-' if x is None else format(x, f'.{nd}f')


def cells(row, cols, nd=1):
    if row['status'] == 'pending':
        return ['pending'] * len(cols)
    return [fmt(row[c], nd) for c in cols]


def render(rows, teacher, notes):
    L = []
    L.append('Distillation ablation, ViT-B student (exp8 recipe: same data, schedule, architecture, backbone init in every row).')
    L.append('')
    L.append('IN-TRAINING VALIDATION, final epoch (V0 setting, training bone fix; 3DPW-val = 300 validation images, holdout = 100 AnnyOne images):')
    L.append('')
    L.append('| arm | run status | epochs | 3DPW-val PVE | PA-PVE | MPJPE | recall | holdout PVE | PA-PVE | MPJPE | recall |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append('| %s | %s | %s | %s |' % (r['label'], r['status'], '-' if r['epochs'] is None else r['epochs'],
                                           ' | '.join(cells(r, VAL_COLS))))
    L.append('')
    L.append('VALIDATION LOSS, final epoch (task = total without the distillation terms, comparable between arms; '
             'kd / kd_out = feature KL / output L1 to the teacher, monitors only in the nokd arm):')
    L.append('')
    L.append('| arm | 3DPW-val total | task | kd | kd_out | holdout total | task | kd | kd_out |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append('| %s | %s |' % (r['label'], ' | '.join(cells(r, LOSS_COLS, 3))))
    L.append('')
    L.append('FINAL TEST numbers (own bone fix; holdout = AnnyOne holdout V1 trace, 100 images; 3DPW test = 8,190 images, v23 labels):')
    L.append('')
    L.append('| arm | holdout PVE | PA-PVE | MPJPE | recall | 3DPW test PVE | PA-PVE | MPJPE | recall |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append('| %s | %s |' % (r['label'], ' | '.join(cells(r, TEST_COLS))))
    if teacher is not None and any(teacher[c] is not None for c in TEST_COLS):
        L.append('| teacher (ViT-L, re-measured in the reported run) | %s |' % ' | '.join(fmt(teacher[c]) for c in TEST_COLS))
    L.append('')
    by = {r['arm']: r for r in rows}
    L.append('Differences in mm (a - b; positive = a is worse), only where both runs have the number:')
    for a, b, what in DIFFS:
        ra, rb = by.get(a), by.get(b)
        parts = []
        for c in DIFF_COLS:
            if ra and rb and ra[c] is not None and rb[c] is not None:
                parts.append(f'{c} {ra[c] - rb[c]:+.1f}')
        L.append(f'- {a} - {b} ({what}): ' + (', '.join(parts) if parts else 'pending'))
    pend = [r['arm'] for r in rows if r['status'] != 'complete']
    if pend:
        L.append('')
        L.append('NOT COMPLETE (no final test numbers yet): ' + ', '.join(f'{r["arm"]} [{r["status"]}]' for r in rows if r['status'] != 'complete'))
    for n in notes:
        L.append('NOTE: ' + n)
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reported', default='', help='the reported Student run folder (job 3614191)')
    ap.add_argument('--sweep_t4', default='', help="the sweep's T 4 arm (..._kdT4_sweep)")
    ap.add_argument('--nokd', default='', help='the KD_MODE=nokd run folder')
    ap.add_argument('--supervised', default='', help='the KD_MODE=supervised run folder')
    ap.add_argument('--out', default='results/experiments/exp10_kd_ablation')
    ap.add_argument('--source', choices=['auto', 'tb', 'log'], default='auto',
                    help='validation columns: tensorboard events when readable, else train.log (auto)')
    a = ap.parse_args(argv)
    if not any((a.reported, a.sweep_t4, a.nokd, a.supervised)):
        print('ERROR: give at least one of --reported / --sweep_t4 / --nokd / --supervised')
        return 1
    rows, notes = [], []
    for arm, label, mode in ARMS:
        row, n = collect(arm, label, mode, getattr(a, arm).rstrip('/'), a.source)
        rows.append(row)
        notes += n
    teacher = None
    if a.reported and os.path.isdir(a.reported):
        t = sweep.collect_run(a.reported.rstrip('/'), 'teacher', who='teacher')
        teacher = {f'test_{s}_{m}': t[f'{s}_{m}'] for s in ('holdout', '3dpw') for m in METRICS}
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'kd_ablation_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if r[c] is None else r[c]) for c in COLUMNS})
        if teacher is not None and any(v is not None for v in teacher.values()):
            tr = {c: '' for c in COLUMNS}
            tr.update(arm='teacher', label='teacher (ViT-L, re-measured in the reported run)', status='reference',
                      run=os.path.basename(os.path.normpath(a.reported)))
            tr.update({c: ('' if v is None else v) for c, v in teacher.items()})
            w.writerow(tr)
    lines = render(rows, teacher, notes)
    md_path = os.path.join(a.out, 'kd_ablation_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('=== exp10 distillation ablation table')
    for line in lines:
        print(line)
    print(f'wrote {csv_path}')
    print(f'wrote {md_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
