"""
Training and validation LOSS curves of train.py runs (the supervisor's request of 22 Oct: decide whether training
longer helps). For every run folder given it draws one figure with two panels sharing the epoch axis
    (1) losses: the per-epoch mean training loss (total; --terms adds single terms as thin lines) and, when the run
        was trained with --val_loss 1, the validation loss total per validation dataset;
    (2) the in-training validation PVE per dataset (evaluate(), every --eval_freq epochs)
and writes a CSV with every number, then prints a plain verdict per run with the supervisor's rule, read from the
slope over the last 20% of the epochs:
    both still falling                -> longer training may help
    train falling, val flat or rising -> over-fitting, stop
    both flat                         -> converged
"val" is the validation loss when the run has one, else the validation PVE (the old runs: jobs 3614191 and 3623342
have no validation loss, only the train loss and the holdout / 3DPW-validation PVE every 5 epochs).

SOURCES, per run: the tensorboard event files of <RUN>/train/ (train.py's SummaryWriter: loss/<term> every --log_freq
iterations, val_loss/<dataset>/<term> and <dataset>/pve at each evaluation, x = optimiser step) when the tensorboard
package imports and the files exist; otherwise, or with --source log, <RUN>/train.log, i.e. train.py's stdout:
    EPOCH=012 - i=00950/145000 -> total: 9.8765 | bce: 0.1234 | ... | kd: 1.2345 | kd_out: 0.9876
    [throughput] epoch 12: 1000 iters x batch 8 in 690s = 1.45 it/s, 11.59 images/s
    ***EVAL METRICS - annyone-holdout-1***            then '    - pve: 122.8' lines (evaluate())
    [val_loss] epoch=012 ds=annyone-holdout-1 n=100 total=8.7654 bce=0.1 ...   (--val_loss 1 only)
Both carry the same numbers: the loss values are AverageMeter running means over the epoch so far (reset every
epoch), so the per-epoch value is the LAST one logged in that epoch (the mean over ~95% of its iterations). Epochs
are counted 1-based in the CSV / plot (epoch 1 = train.py's EPOCH=000 completed); the validation points sit at the
epoch after which they were measured. The steps-to-epoch mapping of the tensorboard source comes from the
'[throughput]' lines (steps per epoch; batches skipped for NaN are not steps), so train.log is read in both cases.

Usage (login node, CPU; no torch):
    python scripts/analysis/plot_loss_curves.py \
        results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30 \
        results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e60 \
        --out results/experiments/exp9_kd_temperature/loss_curves
Outputs: <out>/<RUN>_loss_curves.png / .pdf / .csv per run (the run folders themselves are not written to).
"""
import argparse
import bisect
import csv
import glob
import math
import os
import re
import sys

RE_EPOCH_LINE = re.compile(r'^EPOCH=(\d+) - i=(\d+)/(\d+) -> (.*)$')
RE_THROUGHPUT = re.compile(r'^\[throughput\] epoch (\d+): (\d+) iters x batch (\d+) in')
RE_EVAL_HEAD = re.compile(r'\*\*\*EVAL METRICS - (.+?)\*\*\*')
RE_EVAL_ROW = re.compile(r'^\s*- (\w+): (\S+)\s*$')
RE_VAL_LOSS = re.compile(r'^\[val_loss\] epoch=(\d+) ds=(\S+) (?:n=(\d+) )?(total=.*)$')   # not the note / skipped lines
RE_KV = re.compile(r'(\w+)=(\S+)')
VAL_METRICS = ('pve', 'pa_pve', 'mpjpe', 'recall')


def _num(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


# ------------------------------------------------------------------------------------------------ train.log
def parse_train_log(text):
    """train.py stdout -> dict(train={epoch: {term: value}}, val_loss={ds: {epoch: {term: value}}},
    val={ds: {epoch: {metric: value}}}, steps={epoch: optimiser steps}); epochs 1-based. The train value of an
    epoch is the last EPOCH= line of that epoch (a running mean, see the module docstring)."""
    out = dict(train={}, val_loss={}, val={}, steps={})
    cur = None            # train.py's current_epoch (0-based) of the lines being read
    eval_ds = None
    for line in text.replace('\r', '\n').split('\n'):
        m = RE_EPOCH_LINE.match(line)
        if m:
            cur = int(m.group(1))
            vals = {}
            for part in m.group(4).split('|'):
                part = part.strip()
                if ':' not in part:
                    continue
                k, v = part.split(':', 1)
                v = _num(v.strip())
                if v is not None:
                    vals[k.strip()] = v
            if vals:
                out['train'][cur + 1] = vals
            eval_ds = None
            continue
        m = RE_THROUGHPUT.match(line)
        if m:
            cur = int(m.group(1))
            out['steps'][cur + 1] = int(m.group(2))
            eval_ds = None
            continue
        m = RE_VAL_LOSS.match(line)
        if m:
            vals = {k: _num(v) for k, v in RE_KV.findall(m.group(4))}
            vals = {k: v for k, v in vals.items() if v is not None}
            if vals:
                out['val_loss'].setdefault(m.group(2), {})[int(m.group(1)) + 1] = vals
            eval_ds = None
            continue
        m = RE_EVAL_HEAD.search(line)
        if m:
            eval_ds = m.group(1).strip()
            if cur is not None:
                out['val'].setdefault(eval_ds, {}).setdefault(cur + 1, {})
            continue
        if eval_ds is not None:
            m = RE_EVAL_ROW.match(line)
            if m and cur is not None:
                v = _num(m.group(2))
                if v is not None and m.group(1) in VAL_METRICS:
                    out['val'][eval_ds][cur + 1][m.group(1)] = v
                continue
            if line.strip():
                eval_ds = None
    return out


# ------------------------------------------------------------------------------------------------ tensorboard
def tensorboard_available():
    try:
        from tensorboard.backend.event_processing import event_accumulator  # noqa: F401
        return True
    except Exception:
        return False


def read_tensorboard(run_dir, steps_per_epoch):
    """The same structure as parse_train_log from <run>/train/events.out.tfevents.*; steps_per_epoch = {epoch:
    steps} from the log's throughput lines (or a constant for every epoch). Returns None when there is nothing."""
    from tensorboard.backend.event_processing import event_accumulator
    files = sorted(glob.glob(os.path.join(run_dir, 'train', 'events.out.tfevents.*')))
    if not files or not steps_per_epoch:
        return None
    ends, epochs = [], sorted(steps_per_epoch)
    tot = 0
    for e in epochs:
        tot += steps_per_epoch[e]
        ends.append(tot)

    def epoch_of(step):          # the epoch whose step range holds `step` (1-based); evaluation points sit on an end
        i = bisect.bisect_left(ends, step)
        return epochs[min(i, len(epochs) - 1)]

    out = dict(train={}, val_loss={}, val={}, steps=dict(steps_per_epoch))
    last_step = {}
    for f in files:
        acc = event_accumulator.EventAccumulator(f, size_guidance={event_accumulator.SCALARS: 0})
        acc.Reload()
        for tag in acc.Tags().get('scalars', []):
            if tag.startswith('loss/'):
                term = tag[len('loss/'):]
                for ev in acc.Scalars(tag):
                    e = epoch_of(ev.step + 1)
                    if ev.step >= last_step.get((e, term), -1):   # keep the last value logged in the epoch
                        last_step[(e, term)] = ev.step
                        out['train'].setdefault(e, {})[term] = float(ev.value)
            elif tag.startswith('val_loss/'):
                _, ds, term = tag.split('/', 2)
                for ev in acc.Scalars(tag):
                    out['val_loss'].setdefault(ds, {}).setdefault(epoch_of(ev.step), {})[term] = float(ev.value)
            elif '/' in tag and tag.split('/', 1)[1] in VAL_METRICS and not tag.startswith('workload/'):
                ds, met = tag.split('/', 1)
                for ev in acc.Scalars(tag):
                    out['val'].setdefault(ds, {}).setdefault(epoch_of(ev.step), {})[met] = float(ev.value)
    if not out['train'] and not out['val']:
        return None
    return out


# ------------------------------------------------------------------------------------------------ verdict
def slope_last_fraction(points, frac=0.2, n_epochs=None):
    """Least-squares slope of {epoch: value} over the last `frac` of the epochs (at least two points; the window
    widens to the last two points when fewer fall into it). Returns (slope per epoch, relative slope in % of the
    window mean per epoch, [epochs used]) or None with fewer than two points."""
    if not points or len(points) < 2:
        return None
    n_epochs = n_epochs or max(points)
    lo = n_epochs - max(1, int(math.ceil(frac * n_epochs)))
    use = sorted(e for e in points if e > lo)
    if len(use) < 2:
        use = sorted(points)[-2:]
    xs = [float(e) for e in use]
    ys = [float(points[e]) for e in use]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else 0.0
    rel = 100.0 * slope / abs(my) if my not in (0, 0.0) else 0.0
    return slope, rel, use


def trend(rel, flat_pct):
    if rel is None:
        return 'unknown'
    if rel < -flat_pct:
        return 'falling'
    if rel > flat_pct:
        return 'rising'
    return 'flat'


def verdict(train_trend, val_trend):
    """The supervisor's rule on the two trends."""
    if train_trend == 'unknown' or val_trend == 'unknown':
        return 'no verdict (fewer than two points in one of the curves)'
    if train_trend == 'rising':
        return 'train loss rising -> unstable (check the lr / NaN-skipped steps), no longer-training read'
    if train_trend == 'falling' and val_trend == 'falling':
        return 'both still falling -> longer training may help'
    if train_trend == 'falling':
        return f'train falling, val {val_trend} -> over-fitting, stop'
    if val_trend == 'flat':
        return 'both flat -> converged'
    if val_trend == 'falling':
        return 'train flat, val still falling -> within noise or a lagging validation; a few more epochs would show'
    return 'train flat, val rising -> over-fitting, stop'


# ------------------------------------------------------------------------------------------------ outputs
def _col(ds):
    return re.sub(r'[^A-Za-z0-9]+', '_', ds).strip('_')


def table_rows(data, terms):
    """CSV rows: epoch, train_total, train_<term>..., val_loss_<ds>_total, val_<ds>_<metric>."""
    epochs = sorted(set(data['train']) | {e for d in data['val_loss'].values() for e in d} | {e for d in data['val'].values() for e in d})
    cols = ['epoch', 'train_total'] + [f'train_{t}' for t in terms]
    for ds in sorted(data['val_loss']):
        cols.append(f'val_loss_{_col(ds)}_total')
    for ds in sorted(data['val']):
        cols += [f'val_{_col(ds)}_{m}' for m in VAL_METRICS]
    rows = []
    for e in epochs:
        r = {c: '' for c in cols}
        r['epoch'] = e
        tr = data['train'].get(e, {})
        if 'total' in tr:
            r['train_total'] = tr['total']
        for t in terms:
            if t in tr:
                r[f'train_{t}'] = tr[t]
        for ds, d in data['val_loss'].items():
            if 'total' in d.get(e, {}):
                r[f'val_loss_{_col(ds)}_total'] = d[e]['total']
        for ds, d in data['val'].items():
            for m in VAL_METRICS:
                if m in d.get(e, {}):
                    r[f'val_{_col(ds)}_{m}'] = d[e][m]
        rows.append(r)
    return cols, rows


def plot_run(data, terms, title, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ds_colors = ['#eb6834', '#1baf7a', '#eda100', '#e87ba4']     # per validation dataset, fixed order
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    tr = sorted(e for e in data['train'] if 'total' in data['train'][e])
    if tr:
        ax1.plot(tr, [data['train'][e]['total'] for e in tr], color='#2a78d6', lw=2, marker='o', ms=3, label='train loss (total, epoch mean)')
    for i, t in enumerate(terms):
        te = sorted(e for e in data['train'] if t in data['train'][e])
        if te:
            ax1.plot(te, [data['train'][e][t] for e in te], color='#2a78d6', lw=0.9, ls=['--', ':', '-.', (0, (6, 2, 1, 2, 1, 2))][i % 4], label=f'train {t}')
    val_ds = sorted(set(data['val_loss']) | set(data['val']))
    col = {ds: ds_colors[i % len(ds_colors)] for i, ds in enumerate(val_ds)}
    for ds in sorted(data['val_loss']):
        ve = sorted(e for e in data['val_loss'][ds] if 'total' in data['val_loss'][ds][e])
        if ve:
            ax1.plot(ve, [data['val_loss'][ds][e]['total'] for e in ve], color=col[ds], lw=2, marker='s', ms=4, label=f'val loss {ds}')
    ax1.set_yscale('log')
    ax1.set_ylabel('loss (log scale)')
    ax1.set_title(title)
    ax1.grid(True, color='#dddcd8', lw=0.6, which='both')
    ax1.legend(fontsize=8, frameon=False)
    for ds in sorted(data['val']):
        ve = sorted(e for e in data['val'][ds] if 'pve' in data['val'][ds][e])
        if ve:
            ax2.plot(ve, [data['val'][ds][e]['pve'] for e in ve], color=col[ds], lw=2, marker='o', ms=5, label=f'val PVE {ds}')
    ax2.set_ylabel('validation PVE (mm)')
    ax2.set_xlabel('epoch (completed)')
    ax2.grid(True, color='#dddcd8', lw=0.6)
    if data['val']:
        ax2.legend(fontsize=8, frameon=False)
    else:
        ax2.text(0.5, 0.5, 'no in-training validation metrics found', ha='center', va='center', transform=ax2.transAxes, color='#52514e')
    for ax in (ax1, ax2):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    paths = []
    for ext in ('png', 'pdf'):
        p = f'{out_base}.{ext}'
        fig.savefig(p, dpi=160 if ext == 'png' else None)
        paths.append(p)
    plt.close(fig)
    return paths


def report(run_name, data, frac, flat_pct):
    """Print the slopes and the verdict(s) of one run; returns [(val label, verdict)]."""
    n_epochs = max(list(data['train']) + [e for d in data['val'].values() for e in d] + [e for d in data['val_loss'].values() for e in d] or [0])
    train_pts = {e: v['total'] for e, v in data['train'].items() if 'total' in v}
    s_tr = slope_last_fraction(train_pts, frac, n_epochs)
    print(f'=== {run_name}: {n_epochs} epochs; train loss epochs {len(train_pts)}, val loss datasets '
          f'{sorted(data["val_loss"]) or "none"}, val PVE datasets {sorted(data["val"]) or "none"}')
    if s_tr is None:
        print('    train loss: fewer than two epochs')
        tr_trend = 'unknown'
    else:
        tr_trend = trend(s_tr[1], flat_pct)
        print(f'    train loss total: {train_pts[s_tr[2][0]]:.4f} -> {train_pts[s_tr[2][-1]]:.4f} over epochs {s_tr[2][0]}-{s_tr[2][-1]} '
              f'(last {int(frac * 100)}%): slope {s_tr[0]:+.4f}/epoch = {s_tr[1]:+.2f}%/epoch -> {tr_trend}')
    out = []
    val_series = []
    for ds in sorted(data['val_loss']):
        pts = {e: v['total'] for e, v in data['val_loss'][ds].items() if 'total' in v}
        val_series.append((f'val loss {ds}', pts, 'loss'))
    if not val_series:
        for ds in sorted(data['val']):
            pts = {e: v['pve'] for e, v in data['val'][ds].items() if 'pve' in v}
            val_series.append((f'val PVE {ds}', pts, 'PVE'))
        if val_series:
            print('    no validation loss in this run (trained without --val_loss 1): the validation PVE stands in for it')
    for label, pts, unit in val_series:
        s = slope_last_fraction(pts, frac, n_epochs)
        if s is None:
            print(f'    {label}: fewer than two points')
            v_trend = 'unknown'
        else:
            v_trend = trend(s[1], flat_pct)
            print(f'    {label}: {pts[s[2][0]]:.2f} -> {pts[s[2][-1]]:.2f} over epochs {s[2][0]}-{s[2][-1]}: '
                  f'slope {s[0]:+.3f} {unit}/epoch = {s[1]:+.2f}%/epoch -> {v_trend}')
        v = verdict(tr_trend, v_trend)
        print(f'    VERDICT ({label}): {v}')
        out.append((label, v))
    if not val_series:
        print('    VERDICT: no validation curve in this run')
    return out


def load_run(run_dir, source):
    """(data, source used) for one run folder."""
    log_path = os.path.join(run_dir, 'train.log')
    log_data = parse_train_log(open(log_path, errors='replace').read()) if os.path.isfile(log_path) else None
    if source in ('auto', 'tb') and tensorboard_available():
        steps = log_data['steps'] if log_data and log_data['steps'] else {}
        try:
            tb = read_tensorboard(run_dir, steps)
        except Exception as e:      # a broken event file must not hide the log
            print(f'    tensorboard read failed ({e}); using train.log')
            tb = None
        if tb is not None:
            return tb, 'tensorboard'
        if source == 'tb':
            return None, 'none'
    if log_data is None:
        return None, 'none'
    return log_data, 'train.log'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run folders (each holds train.log and/or train/events.out.tfevents.*)')
    ap.add_argument('--out', default='results/analysis/loss_curves', help='figures and CSVs go here, named after the run')
    ap.add_argument('--source', choices=['auto', 'tb', 'log'], default='auto',
                    help='auto: tensorboard events when readable, else train.log; tb / log force one')
    ap.add_argument('--terms', default='rotmat,v3d,kd,kd_out', help='training loss terms drawn as thin lines ("" = none)')
    ap.add_argument('--frac', type=float, default=0.2, help='the slope window: the last fraction of the epochs')
    ap.add_argument('--flat_pct', type=float, default=0.25, help='|relative slope| below this %% per epoch counts as flat')
    ap.add_argument('--no_plot', action='store_true', help='CSV and verdict only')
    a = ap.parse_args(argv)
    terms = [t for t in a.terms.split(',') if t]
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for run_dir in a.runs:
        run_dir = run_dir.rstrip('/')
        name = os.path.basename(run_dir)
        if not os.path.isdir(run_dir):
            print(f'=== {name}: not a folder ({run_dir})')
            rc = 1
            continue
        data, src = load_run(run_dir, a.source)
        if data is None:
            print(f'=== {name}: no train.log and no readable tensorboard events in {run_dir}')
            rc = 1
            continue
        present = [t for t in terms if any(t in v for v in data['train'].values())]
        cols, rows = table_rows(data, present)
        csv_path = os.path.join(a.out, f'{name}_loss_curves.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        report(name, data, a.frac, a.flat_pct)
        print(f'    source: {src}; wrote {csv_path}')
        if not a.no_plot:
            try:
                for p in plot_run(data, present, name, os.path.join(a.out, f'{name}_loss_curves')):
                    print(f'    wrote {p}')
            except ImportError as e:
                print(f'    no figure (matplotlib missing: {e})')
    return rc


if __name__ == '__main__':
    sys.exit(main())
