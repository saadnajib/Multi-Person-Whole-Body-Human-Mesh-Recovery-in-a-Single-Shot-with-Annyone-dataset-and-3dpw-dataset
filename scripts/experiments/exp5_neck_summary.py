"""
exp5_neck_fix, final step: compare a fine-tuned run's holdout trace and bone dump with the baseline dump and, when
given, the control arm trained on the same schedule (ROTMAT_LOSS=l1). Standard library only, so it also runs on a
login node; slurm/experiments/exp5_neck_fix.sh calls it at the end of a run, and with SUMMARY_ONLY=1 on its own.

    python3 scripts/experiments/exp5_neck_summary.py --run_dir results/experiments/exp5_neck_fix/<run> \
        --base_dump results/diagnostics/neck_bones/<tag>/bones_summary.json \
        [--control_run results/experiments/exp5_neck_fix/<l1 run>]

Three separate reads (V1_eval_masked, 100-image holdout, matched people), none hiding another:
  * neck   FIXED (neck and head group means < 15 deg, girth >= 0.9) / MOVING (every bone the baseline dump flags
           FLIP_*, PARTIAL_FLIP or CONST_OFFSET drops its MEAN local error by >= 3 deg, and by >= 3 deg more than in
           the control arm) / NO MOVEMENT / NO TARGET. The mean, not the median, so a PARTIAL_FLIP bone counts, and the
           3-deg floor comes from group MEANS (v5 arms: neck 62.5-62.8, head 89.1-90.1).
  * guards PVE <= ref + 8 mm, recall >= ref - 2 points, global head <= ref + 3 deg, with ref = the control arm when
           given, else the v5 numbers (70.5 / 90.2 / 12.0). The tolerances are the measured spread of the six 2-epoch
           v5-recipe arms (exp2-exp4 pilots and controls): PVE 68.7-76.3, recall 88.6-90.2, global head 11.3-13.7; the
           two identical-recipe control arms (exp3, exp4) differ by 7.6 mm, 0.8 points and 1.9 deg. The old +1 mm /
           -1 point guard failed all six, controls included.
  * control whether the control arm was compared at all.
Exit status 0 = summary written; 2 = --control_run given but its holdout_trace is missing (nothing is written: the
control job has not finished; submit this arm with --dependency=afterok:<control job id> or rerun with SUMMARY_ONLY=1).
"""
import argparse
import json
import os
import sys

V = 'V1_eval_masked'
BONES = ['neck01', 'neck02', 'neck03', 'head', 'jaw']
V5 = dict(pve=70.5, recall=90.2, rot_global_head_deg=12.0, pve_headneck=52.3, rot_neck=62.7, rot_head=89.5)
TOL = dict(pve=8.0, recall=2.0, rot_global_head_deg=3.0)
NOISE_DEG = 3.0
TARGET_VERDICTS = ('FLIP_TWIST', 'FLIP_SWING', 'PARTIAL_FLIP', 'CONST_OFFSET')
FIXED_DEG, FIXED_GIRTH = 15.0, 0.9


def load(p):
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:      # noqa: BLE001 - a broken file is reported, not fatal
        print('  could not read', p, e)
        return None


def dump_v1(d):
    return None if not d else (d['variants'].get(V) or next(iter(d['variants'].values())))


def trace_v1(s):
    if not s or V not in s.get('variants', {}):
        return {}
    v = s['variants'][V]
    g = v.get('rot_err_deg_by_group', {})
    return dict(pve=v.get('pve'), recall=v.get('recall'), pve_headneck=v.get('pve_headneck'),
                rot_global_head_deg=(v.get('rot_global_deg') or {}).get('head'), rot_neck=g.get('neck'),
                rot_head=g.get('head'))


def f(x, nd=1):
    return '-' if x is None else format(x, '.' + str(nd) + 'f')


def bone_stat(d, b, k):
    return None if not d or b not in d['bones'] else d['bones'][b].get(k)


def girth(d):
    return None if not d else d['neck_skin'].get('neck_girth_ratio_mean')


def evaluate(n_s, n_d, b_d, c_s, c_d):
    """The three reads; pure function of the loaded summaries (unit-tested)."""
    if not (n_s and n_d) or None in (n_s.get('pve'), n_s.get('recall'), n_s.get('rot_global_head_deg')):
        return dict(neck='INCOMPLETE', guards_ok=None), 'INCOMPLETE (missing numbers; see the trace logs)'
    use_c = bool(c_s) and None not in (c_s.get('pve'), c_s.get('recall'), c_s.get('rot_global_head_deg'))
    ref = c_s if use_c else V5
    ref_name = 'control arm' if use_c else 'v5 baseline numbers (no control arm)'
    lim = dict(pve=ref['pve'] + TOL['pve'], recall=ref['recall'] - TOL['recall'],
               rot_global_head_deg=ref['rot_global_head_deg'] + TOL['rot_global_head_deg'])
    failed = [k for k in ('pve', 'rot_global_head_deg') if n_s[k] > lim[k]] + (
        ['recall'] if n_s['recall'] < lim['recall'] else [])
    targets = [b for b in BONES if bone_stat(b_d, b, 'verdict') in TARGET_VERDICTS]
    drop = {b: bone_stat(b_d, b, 'err_mean') - bone_stat(n_d, b, 'err_mean') for b in targets
            if None not in (bone_stat(b_d, b, 'err_mean'), bone_stat(n_d, b, 'err_mean'))}
    drop_c = ({b: bone_stat(b_d, b, 'err_mean') - bone_stat(c_d, b, 'err_mean') for b in targets
               if None not in (bone_stat(b_d, b, 'err_mean'), bone_stat(c_d, b, 'err_mean'))} if c_d else {})
    moving = bool(targets) and len(drop) == len(targets) and all(
        drop[b] >= NOISE_DEG and (not c_d or (b in drop_c and drop[b] - drop_c[b] >= NOISE_DEG)) for b in targets)
    fixed = (n_s.get('rot_neck') is not None and n_s['rot_neck'] < FIXED_DEG and n_s.get('rot_head') is not None
             and n_s['rot_head'] < FIXED_DEG and (girth(n_d) or 0) >= FIXED_GIRTH)
    if fixed:
        label, neck = 'FIXED', 'FIXED: neck and head local errors < 15 deg with girth >= 0.9'
    elif moving:
        label, neck = 'MOVING', ('MOVING: the flagged bones improve by >= 3 deg' + (
            ' beyond the control arm' if c_d else ' vs the baseline (NO control arm: not separable from fine-tuning drift)')
            + ' -> run the 10-epoch arm (MAX_ITER=10000)')
    elif not targets:
        label, neck = 'NO_TARGET', ('NO TARGET: the baseline dump flags no neck/head bone (run '
                                    'slurm/diagnostics/diag_neck_bones.sh first)')
    else:
        label, neck = 'NO_MOVEMENT', ('NO MOVEMENT beyond the noise floor (does not refute the flip: see the exp5 header; '
                                      'try the decpose surgery)')
    guards = ('guards pass' if not failed else 'guards FAILED (%s)' % ', '.join(
        '%s %s vs limit %s' % (k, f(n_s[k]), f(lim[k])) for k in failed)) + ' vs the ' + ref_name
    res = dict(neck=label, fixed=fixed, moving=moving, targets=targets, drop_mean_vs_baseline=drop,
               drop_mean_control_vs_baseline=drop_c, control_compared=bool(c_d), guards_ok=not failed,
               guards_failed=failed, guard_reference=ref_name, guard_limits=lim, guard_tolerances=TOL)
    return res, neck + ' | ' + guards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True, help='exp5 run dir holding holdout_trace/{summary,bones_summary}.json')
    ap.add_argument('--base_dump', required=True, help='bones_summary.json of the starting checkpoint')
    ap.add_argument('--control_run', default='', help='the control arm\'s run dir (same schedule, ROTMAT_LOSS=l1)')
    a = ap.parse_args()

    c_dump_p = os.path.join(a.control_run, 'holdout_trace', 'bones_summary.json') if a.control_run else ''
    c_sum_p = os.path.join(a.control_run, 'holdout_trace', 'summary.json') if a.control_run else ''
    if a.control_run and not (os.path.isfile(c_dump_p) and os.path.isfile(c_sum_p)):
        print('ERROR: CONTROL_RUN=%s has no finished holdout trace (%s / %s missing). The control job has not finished,\n'
              '       failed, or the path is wrong. Submit this arm with sbatch --dependency=afterok:<control job id>, or\n'
              '       rerun only the comparison once both arms are done:\n'
              '       SUMMARY_ONLY=1 RUN=%s CONTROL_RUN=%s bash slurm/experiments/exp5_neck_fix.sh'
              % (a.control_run, c_dump_p, c_sum_p, os.path.basename(os.path.normpath(a.run_dir)), a.control_run))
        return 2

    b_d = dump_v1(load(a.base_dump))
    n_d = dump_v1(load(os.path.join(a.run_dir, 'holdout_trace', 'bones_summary.json')))
    n_s = trace_v1(load(os.path.join(a.run_dir, 'holdout_trace', 'summary.json')))
    c_d = dump_v1(load(c_dump_p)) if a.control_run else None
    c_s = trace_v1(load(c_sum_p)) if a.control_run else {}

    print()
    print('=== exp5_neck_fix SUMMARY (V1, 100-image holdout, matched people; deg / mm; bone cells = verdict median/mean) ===')
    print('run: ' + a.run_dir)
    print('baseline dump: ' + a.base_dump + ('' if b_d else ' (MISSING)'))
    print('control: ' + (a.control_run if a.control_run else 'none (moving/guards judged against the baseline only)'))
    print(format('bone', '<10') + format('baseline', '>26') + format('control', '>26') + format('this run', '>26'))

    def cell(d, b):
        if not d or b not in d['bones']:
            return '-'
        s = d['bones'][b]
        return '%s %s/%s' % (s['verdict'], f(s.get('err_median')), f(s.get('err_mean')))
    for b in BONES:
        print(format(b, '<10') + format(cell(b_d, b), '>26') + format(cell(c_d, b), '>26') + format(cell(n_d, b), '>26'))
    print(format('girth', '<10') + format(f(girth(b_d), 3), '>26') + format(f(girth(c_d), 3), '>26')
          + format(f(girth(n_d), 3), '>26'))
    print(format('metric', '<20') + format('v5', '>16') + format('control', '>26') + format('this run', '>26'))
    for k in ('rot_neck', 'rot_head', 'rot_global_head_deg', 'pve', 'pve_headneck', 'recall'):
        print(format(k, '<20') + format(f(V5.get(k)), '>16') + format(f(c_s.get(k)), '>26') + format(f(n_s.get(k)), '>26'))

    res, verdict = evaluate(n_s, n_d, b_d, c_s, c_d)
    if res.get('guards_ok') is not None:
        lim = res['guard_limits']
        print('guards vs the %s (PVE <= %s, recall >= %s, global head <= %s; tolerances +8 mm / -2 / +3 deg = the spread '
              'of six 2-epoch v5-recipe arms): %s' % (res['guard_reference'], f(lim['pve']), f(lim['recall']),
                                                     f(lim['rot_global_head_deg']),
                                                     'pass' if res['guards_ok'] else 'FAILED ' + str(res['guards_failed'])))
        print('baseline FLIP_*/PARTIAL_FLIP/CONST_OFFSET bones: %s; mean drop vs baseline: %s%s' % (
            res['targets'] or 'none', {b: round(x, 1) for b, x in res['drop_mean_vs_baseline'].items()},
            ('; control arm drop: %s' % {b: round(x, 1) for b, x in res['drop_mean_control_vs_baseline'].items()})
            if res['control_compared'] else ' (no control arm)'))
    print('VERDICT: ' + verdict)
    print('2-epoch pilot, fresh Adam state, no seed control: a go/no-go read. A guard miss is reported next to the neck read,')
    print('not instead of it.')
    out = os.path.join(a.run_dir, 'summary.json')
    with open(out, 'w') as fh:
        json.dump(dict(v5=V5, noise_floor_deg=NOISE_DEG, this_run=n_s, control=c_s, control_run=a.control_run or None,
                       baseline_dump=a.base_dump, result=res, verdict=verdict), fh, indent=2)
    print('wrote ' + out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
