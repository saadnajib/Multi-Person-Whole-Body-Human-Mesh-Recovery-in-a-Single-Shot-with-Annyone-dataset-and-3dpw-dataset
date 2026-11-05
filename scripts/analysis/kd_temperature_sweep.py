"""
Collect the exp9 KD-temperature sweep (slurm/experiments/exp9_kd_temperature.sh): one table and one plot of the
student's final errors against the distillation temperature, for the supervisor's request of 22 Oct ("x-axis
temperature, y-axis error, pick the value that minimises the errors").

Inputs: the exp8 run folders of the sweep (a glob), optionally the reported T 4 run (job 3614191) as a separate
point. From each folder it reads what exp8's post-steps wrote for the STUDENT:
    <RUN>/settings.txt                     KD_T=<t> (the temperature; else the _kdT<t> / _T<t> of the folder name)
    <RUN>/holdout_student/summary.json     AnnyOne holdout, 100 images, variant V1_eval_masked: pve, pa_pve, mpjpe,
                                           recall (the numbers the teacher was reported with; exp8 step 3)
    <RUN>/eval_3dpw_test_student.log       3DPW test, every image of the v23 labels, evaluate()'s
                                           '***EVAL METRICS - threedpw_anny-test-1***' block: pve, pa_pve, mpjpe,
                                           recall (exp8 step 5)
    <RUN>/final_table.json                 fallback for both (exp8 step 6 copies the same numbers into its rows)
    <RUN>/holdout_teacher/summary.json, eval_3dpw_test_teacher.log   the teacher as re-measured in that job (printed
                                           for information; the plot's reference lines are the --teacher_* values)

Outputs (--out, default results/experiments/exp9_kd_temperature):
    kd_temperature_sweep.csv   temperature, run, holdout PVE / PA-PVE / MPJPE / recall, 3DPW-test PVE / PA-PVE /
                               MPJPE / recall, source (sweep | reported)
    kd_temperature_sweep.png / .pdf   x = temperature (log2), y = error (mm): one line per metric and dataset;
                               the teacher's PVE as dashed horizontal lines (73.7 holdout, 89.0 3DPW test, exp7) and
                               the 95% bars (teacher / 0.95 = 77.6, 93.7) dotted; the reported T 4 run as a separate
                               marker when given
    stdout: the table, the temperature that minimises each error, the overall pick (the temperature with the
    smallest holdout PVE + 3DPW PVE, each divided by its bar), and, when the reported run is given, its difference
    to the sweep's own T 4 arm per metric = the run-to-run noise against which a temperature effect must be read.

Usage (login node, CPU; no torch):
    python scripts/analysis/kd_temperature_sweep.py \
        --runs "results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30_kdT*_sweep" \
        --reported results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30 \
        --out results/experiments/exp9_kd_temperature
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HOLDOUT_VARIANT = 'V1_eval_masked'
METRICS = ('pve', 'pa_pve', 'mpjpe', 'recall')
ERROR_METRICS = ('pve', 'pa_pve', 'mpjpe')      # mm, lower is better; recall stays in the CSV only
DATASETS = (('holdout', 'AnnyOne holdout'), ('3dpw', '3DPW test'))
COLUMNS = ['temperature', 'run'] + [f'{d}_{m}' for d, _ in DATASETS for m in METRICS] + ['source']


# ------------------------------------------------------------------------------------------------ readers
def read_temperature(run_dir):
    """KD_T of a run: settings.txt (KD_T=<t>), else the folder name (_kdT<t>_ or _T<t>), else train.log's
    '[distill] lambda_kd=... T=<t>' line. None when nothing says it."""
    p = os.path.join(run_dir, 'settings.txt')
    if os.path.isfile(p):
        m = re.search(r'(?:^|\s)KD_T=(\S+)', open(p, errors='replace').read())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.search(r'_kdT([0-9.]+(?:[eE][-+]?[0-9]+)?)(?:_|$)', name) or re.search(r'_T([0-9.]+(?:[eE][-+]?[0-9]+)?)(?:_|$)', name)
    if m:
        return float(m.group(1))
    p = os.path.join(run_dir, 'train.log')
    if os.path.isfile(p):
        m = re.search(r'\[distill\] lambda_kd=\S+ T=(\S+)', open(p, errors='replace').read())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def read_trace_summary(trace_dir, variant=HOLDOUT_VARIANT):
    """{metric: value} of one diag_holdout_trace.py output folder (summary.json, one variant); {} if absent."""
    try:
        s = json.load(open(os.path.join(trace_dir, 'summary.json')))
        v = s['variants'][variant]
    except Exception:
        return {}
    return {k: v.get(k) for k in METRICS if v.get(k) is not None}


def read_eval_log(path):
    """{dataset key: {metric: value}} from the '***EVAL METRICS - <name>***' blocks of a train.py evaluate() log
    (the parser of exp8's FINAL_PY)."""
    res, cur = {}, None
    try:
        lines = open(path, errors='replace').read().replace('\r', '\n').split('\n')
    except Exception:
        return res
    for line in lines:
        m = re.search(r'\*\*\*EVAL METRICS - (.+?)\*\*\*', line)
        if m:
            cur = m.group(1).strip()
            res[cur] = {}
            continue
        if cur is None:
            continue
        m = re.match(r'^\s*- (\w+): (\S+)\s*$', line)
        if m:
            try:
                res[cur][m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.strip():
            cur = None
    return res


def pick_block(res, prefix):
    keys = [k for k in res if k.startswith(prefix)]
    return res[keys[-1]] if keys else {}


def read_final_table(run_dir, who='student'):
    """{'holdout': {metric: value}, '3dpw': {...}} from exp8's final_table.json rows (fallback source)."""
    out = {'holdout': {}, '3dpw': {}}
    try:
        rows = json.load(open(os.path.join(run_dir, 'final_table.json')))['rows']
    except Exception:
        return out
    names = {'PVE': 'pve', 'PA-PVE': 'pa_pve', 'MPJPE': 'mpjpe', 'recall': 'recall'}
    for r in rows:
        sec, met = str(r.get('section', '')), names.get(r.get('metric'))
        if met is None or r.get(who) is None:
            continue
        if sec.startswith('AnnyOne holdout'):
            out['holdout'][met] = float(r[who])
        elif sec.startswith('3DPW test'):
            out['3dpw'][met] = float(r[who])
    return out


def collect_run(run_dir, source='sweep', who='student'):
    """One CSV row (dict) for a run folder; missing numbers stay None. who = student | teacher (teacher: the
    re-measured teacher of that job, holdout_teacher/ + eval_3dpw_test_teacher.log)."""
    row = {c: None for c in COLUMNS}
    row['run'] = os.path.basename(os.path.normpath(run_dir))
    row['source'] = source
    row['temperature'] = read_temperature(run_dir)
    hold = read_trace_summary(os.path.join(run_dir, f'holdout_{who}'))
    ev = pick_block(read_eval_log(os.path.join(run_dir, f'eval_3dpw_test_{who}.log')), 'threedpw_anny-test')
    fb = read_final_table(run_dir, who)
    for m in METRICS:
        v = hold.get(m, fb['holdout'].get(m))
        row[f'holdout_{m}'] = None if v is None else float(v)
        v = ev.get(m, fb['3dpw'].get(m))
        row[f'3dpw_{m}'] = None if v is None else float(v)
    return row


def expand_runs(patterns):
    """Run folders of the glob patterns / paths, sorted, each once."""
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.isdir(pat) else [])
        for h in hits:
            if os.path.isdir(h) and h not in out:
                out.append(h)
    return out


# ------------------------------------------------------------------------------------------------ analysis
def fmt(x, nd=1):
    return '-' if x is None else format(x, f'.{nd}f')


def best_temperatures(rows):
    """{column: (temperature, value)} = the sweep row with the smallest error per error column, and
    'combined' = the smallest holdout PVE / bar + 3DPW PVE / bar (bars from rows' 'bars' entry if present)."""
    best = {}
    for d, _ in DATASETS:
        for m in ERROR_METRICS:
            c = f'{d}_{m}'
            cand = [(r[c], r['temperature']) for r in rows if r.get(c) is not None and r.get('temperature') is not None]
            if cand:
                v, t = min(cand)
                best[c] = (t, v)
    return best


def combined_pick(rows, bar_holdout, bar_3dpw):
    cand = []
    for r in rows:
        if r.get('holdout_pve') is None or r.get('3dpw_pve') is None or r.get('temperature') is None:
            continue
        cand.append((r['holdout_pve'] / bar_holdout + r['3dpw_pve'] / bar_3dpw, r['temperature'], r))
    if not cand:
        return None
    cand.sort(key=lambda c: (c[0], c[1]))
    return cand[0]


def noise_vs_reported(sweep_rows, reported):
    """{column: sweep T4 - reported} for every metric both carry; None when the sweep has no T 4 arm."""
    t4 = [r for r in sweep_rows if r.get('temperature') is not None and abs(r['temperature'] - 4.0) < 1e-9]
    if not t4 or reported is None:
        return None
    r4 = t4[-1]
    return {c: (r4[c] - reported[c]) for c in COLUMNS[2:-1] if r4.get(c) is not None and reported.get(c) is not None}


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if r.get(c) is None else r[c]) for c in COLUMNS})


def plot(rows, reported, out_base, teacher_holdout_pve, teacher_3dpw_pve, rule):
    """x = temperature (log2), y = error (mm): one line per (dataset, metric); the teacher's PVE dashed, the
    95% bars dotted; the reported run as a star. Returns the written paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # fixed series order and colours (one hue per dataset x metric; never cycled)
    series = [('holdout', 'pve', 'AnnyOne holdout PVE', '#2a78d6', 'o'),
              ('holdout', 'mpjpe', 'AnnyOne holdout MPJPE', '#6250d6', 's'),
              ('holdout', 'pa_pve', 'AnnyOne holdout PA-PVE', '#1baf7a', '^'),
              ('3dpw', 'pve', '3DPW test PVE', '#eb6834', 'o'),
              ('3dpw', 'mpjpe', '3DPW test MPJPE', '#e34948', 's'),
              ('3dpw', 'pa_pve', '3DPW test PA-PVE', '#eda100', '^')]
    sweep = sorted((r for r in rows if r.get('temperature') is not None), key=lambda r: r['temperature'])
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for d, m, label, col, mk in series:
        pts = [(r['temperature'], r[f'{d}_{m}']) for r in sweep if r.get(f'{d}_{m}') is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=6, lw=1.8, label=label)
        if reported is not None and reported.get(f'{d}_{m}') is not None and reported.get('temperature') is not None:
            ax.plot([reported['temperature']], [reported[f'{d}_{m}']], color=col, marker='*', ms=13, ls='none',
                    mec='#0b0b0b', mew=0.6, label=f'{label}, reported run' if m == 'pve' else None)
    ax.axhline(teacher_holdout_pve, color='#2a78d6', ls='--', lw=1.2, label=f'teacher holdout PVE {teacher_holdout_pve:g}')
    ax.axhline(teacher_holdout_pve / rule, color='#2a78d6', ls=':', lw=1.2, label=f'95% bar holdout {teacher_holdout_pve / rule:.1f}')
    ax.axhline(teacher_3dpw_pve, color='#eb6834', ls='--', lw=1.2, label=f'teacher 3DPW PVE {teacher_3dpw_pve:g}')
    ax.axhline(teacher_3dpw_pve / rule, color='#eb6834', ls=':', lw=1.2, label=f'95% bar 3DPW {teacher_3dpw_pve / rule:.1f}')
    ax.set_xscale('log', base=2)
    temps = sorted({r['temperature'] for r in sweep} | ({reported['temperature']} if reported and reported.get('temperature') else set()))
    if temps:
        ax.set_xticks(temps)
        ax.set_xticklabels([f'{t:g}' for t in temps])
    ax.set_xlabel('KD temperature T (--kd_temperature, log2 axis)')
    ax.set_ylabel('error (mm), lower is better')
    ax.set_title('ViT-B student (exp8 recipe): final error vs distillation temperature')
    ax.grid(True, color='#dddcd8', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13))   # below the axes, off the lines
    fig.tight_layout()
    paths = []
    for ext in ('png', 'pdf'):
        p = f'{out_base}.{ext}'
        fig.savefig(p, dpi=160 if ext == 'png' else None)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='sweep run folders (globs or paths; quote the glob)')
    ap.add_argument('--reported', default='', help='the reported T 4 run folder (job 3614191), plotted as a separate point')
    ap.add_argument('--out', default='results/experiments/exp9_kd_temperature')
    ap.add_argument('--teacher_holdout_pve', type=float, default=73.7, help='exp7 teacher, AnnyOne holdout V1 (dashed line)')
    ap.add_argument('--teacher_3dpw_pve', type=float, default=89.0, help='exp7 teacher, 3DPW test v23 (dashed line)')
    ap.add_argument('--rule', type=float, default=0.95, help='the 95%% rule: bar = teacher / rule (dotted lines)')
    ap.add_argument('--no_plot', action='store_true', help='CSV and stdout only (no matplotlib)')
    a = ap.parse_args(argv)

    run_dirs = expand_runs(a.runs)
    if not run_dirs:
        print(f'ERROR: no run folder matches {a.runs}')
        return 1
    rows = [collect_run(d, 'sweep') for d in run_dirs]
    reported = None
    if a.reported:
        if not os.path.isdir(a.reported):
            print(f'ERROR: --reported {a.reported} is not a folder')
            return 1
        reported = collect_run(a.reported, 'reported')
        if reported['temperature'] is None:
            reported['temperature'] = 4.0
    rows.sort(key=lambda r: (r['temperature'] is None, r['temperature'] or 0.0))
    all_rows = rows + ([reported] if reported else [])
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'kd_temperature_sweep.csv')
    write_csv(csv_path, all_rows)

    bar_h, bar_p = a.teacher_holdout_pve / a.rule, a.teacher_3dpw_pve / a.rule
    print('=== exp9 KD temperature sweep: student final errors (holdout = AnnyOne 100 images V1 trace; 3DPW = test, v23)')
    print(f'    teacher (exp7): holdout PVE {a.teacher_holdout_pve:g} (bar <= {bar_h:.1f}), 3DPW PVE {a.teacher_3dpw_pve:g} (bar <= {bar_p:.1f})')
    print('| T | run | holdout PVE | PA-PVE | MPJPE | recall | 3DPW PVE | PA-PVE | MPJPE | recall | source |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in all_rows:
        print('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            fmt(r['temperature'], 2) if r['temperature'] is not None else '?', r['run'],
            fmt(r['holdout_pve']), fmt(r['holdout_pa_pve']), fmt(r['holdout_mpjpe']), fmt(r['holdout_recall']),
            fmt(r['3dpw_pve']), fmt(r['3dpw_pa_pve']), fmt(r['3dpw_mpjpe']), fmt(r['3dpw_recall']), r['source']))
    incomplete = [r['run'] for r in rows if any(r.get(f'{d}_pve') is None for d, _ in DATASETS)]
    if incomplete:
        print('INCOMPLETE (no holdout and/or 3DPW PVE yet; job still running or a post-step failed): ' + ', '.join(incomplete))
    for d in run_dirs:
        t_h = read_trace_summary(os.path.join(d, 'holdout_teacher'))
        t_p = pick_block(read_eval_log(os.path.join(d, 'eval_3dpw_test_teacher.log')), 'threedpw_anny-test')
        if t_h or t_p:
            print(f'    teacher re-measured in {os.path.basename(d)}: holdout PVE {fmt(t_h.get("pve"))}, 3DPW PVE {fmt(t_p.get("pve"))}')
    best = best_temperatures(rows)
    print('--- temperature with the smallest error, per metric (sweep arms only):')
    for d, dl in DATASETS:
        for m in ERROR_METRICS:
            c = f'{d}_{m}'
            if c in best:
                print(f'    {dl} {m.upper().replace("_", "-")}: T = {best[c][0]:g} ({best[c][1]:.1f} mm)')
    pick = combined_pick(rows, bar_h, bar_p)
    if pick is not None:
        print(f'PICK: T = {pick[1]:g} minimises holdout PVE / {bar_h:.1f} + 3DPW PVE / {bar_p:.1f} '
              f'(= {pick[0]:.3f}; holdout {pick[2]["holdout_pve"]:.1f}, 3DPW {pick[2]["3dpw_pve"]:.1f} mm)')
        met = [f'{dl} PVE {"met" if pick[2][f"{d}_pve"] <= b else "NOT met"}' for (d, dl), b in zip(DATASETS, (bar_h, bar_p))]
        print('      95% bars at that T: ' + '; '.join(met))
    noise = noise_vs_reported(rows, reported)
    if noise is not None:
        print('--- run-to-run noise: sweep T 4 arm minus the reported T 4 run (same recipe, seed 0; only EVAL_FREQ / VAL_LOSS differ):')
        print('    ' + ', '.join(f'{c} {v:+.1f}' for c, v in noise.items()))
        span = max((abs(v) for c, v in noise.items() if c.endswith('_pve')), default=None)
        if span is not None:
            print(f'    a temperature effect smaller than ~{span:.1f} mm PVE is within this noise')
    elif reported is not None:
        print('--- no T 4 arm in the sweep: the reported run is plotted but the run-to-run noise cannot be read')
    print(f'wrote {csv_path}')
    if not a.no_plot:
        try:
            for p in plot(all_rows if reported is None else rows, reported, os.path.join(a.out, 'kd_temperature_sweep'),
                          a.teacher_holdout_pve, a.teacher_3dpw_pve, a.rule):
                print(f'wrote {p}')
        except ImportError as e:
            print(f'no plot (matplotlib missing: {e}); the CSV has every number')
    return 0


if __name__ == '__main__':
    sys.exit(main())
