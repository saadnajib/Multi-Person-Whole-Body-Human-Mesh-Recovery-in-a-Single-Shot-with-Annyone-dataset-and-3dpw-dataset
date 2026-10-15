"""
exp6_bone_fix, final step: the holdout effect of an output-level bone fix (R_b <- R_b @ C_b, bone_fix.py) next to the
baseline dump, the fine-tuned-with-fix run and the exp5 L1 control. Standard library only (login node is fine);
slurm/experiments/exp6_bone_fix.sh calls it at the end, and with SUMMARY_ONLY=1 on its own.

    python3 scripts/experiments/exp6_bone_fix_summary.py --baseline_dir results/diagnostics/neck_bones/<tag> \
        --fix_dir <run>/holdout_fix [--ft_dir <run>/holdout_ft] [--control_run results/experiments/exp5_neck_fix/<l1 run>] \
        --bone_fix <run>/bone_fix.json --out <run>/summary.json

Columns (V1_eval_masked, 100-image holdout, matched people): baseline = the checkpoint without a fix
(<baseline_dir>/{summary,bones_summary}.json, slurm/diagnostics/diag_neck_bones.sh); zero-training = the same
checkpoint with --bone_fix; fine-tuned = the exp5 control recipe (L1, alpha_rotmat 0.1, 2 epochs, lr 1e-5) trained
WITH the fix; control = the same recipe without it (exp5 pilot_l1 run; context for the fine-tuned column only).
Rows: verdict + median local error of the neck chain, the corrected bones and every bone the baseline flags
(FLIP_*, CONST_OFFSET, FIXED_OUTPUT with median > 90 deg); neck-skin girth pred/GT; PVE_neck (neck-skin PVE centred
at each mesh's own neck01, mm); PVE, PA-PVE, PVE_hn; recall; global head error; finger rotation error.

PASS CRITERIA
  zero-training (vs the baseline column):  every corrected bone median < 15 deg (v5: neck02, head, metacarpal1.L);
      girth >= 0.9; global head <= baseline + 3 deg; PVE <= baseline + 1 mm.
  fine-tuned (vs the control when it exists, else the baseline): every corrected bone median < 15 deg; girth >= 0.9;
      exp5 guards PVE <= ref + 8 mm, recall >= ref - 2, global head <= ref + 3 deg (the spread of six 2-epoch
      v5-recipe arms; docs/experiments_plan.md exp5).
Also printed: compute_bone_fix.py's GLOBAL GUARD from bone_fix.json (step a: which dumped bones the fix was predicted
to turn globally worse, i.e. a compensated chain; a FAIL there was predicted, not new), and a WARNING when the
baseline trace was made from another checkpoint than the one the fix was estimated on (TAG as exp6_bone_fix.sh).
A bone_fix_guarded.json next to bone_fix.json (ON_GLOBAL_RISE=safe) is reported as this run's first selection only
when bone_fix.json holds exactly its safe subset and is itself guard-clean; otherwise it is a leftover (WARNING).
Exit status 0 = summary written (a FAIL is a result, not an error); 2 = the zero-training trace is missing.
"""
import argparse
import json
import os
import sys

V = 'V1_eval_masked'
NECK = ['neck01', 'neck02', 'neck03', 'head', 'jaw']
FLAG_VERDICTS = ('FLIP_TWIST', 'FLIP_SWING', 'CONST_OFFSET', 'FIXED_OUTPUT')
FLAG_MIN_MEDIAN = 90.0
BONE_OK_DEG, GIRTH_MIN = 15.0, 0.9
ZT_TOL = dict(pve=1.0, rot_global_head_deg=3.0)                    # zero-training vs the baseline
FT_TOL = dict(pve=8.0, recall=2.0, rot_global_head_deg=3.0)         # fine-tuned vs the control (exp5 guards)
METRICS = (('girth', 'girth pred/GT', 3), ('pve_neck_local', 'PVE_neck (neck01-c.)', 1), ('pve', 'PVE', 1),
           ('pa_pve', 'PA-PVE', 1), ('pve_headneck', 'PVE_hn', 1), ('recall', 'recall', 1),
           ('rot_global_head_deg', 'global head', 1), ('rot_fingers', 'rot fingers', 1),
           ('rot_neck', 'rot neck (group)', 1), ('rot_head', 'rot head (group)', 1))


def load(p):
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:      # noqa: BLE001 - reported, not fatal
        print('  could not read', p, e)
        return None


def f(x, nd=1):
    return '-' if x is None else format(x, '.' + str(nd) + 'f')


def tag_of(ck):
    """exp6_bone_fix.sh / diag_neck_bones.sh TAG of a checkpoint path (run_epoch for .../<run>/checkpoints/<epoch>.pt)."""
    if not ck:
        return None
    d = os.path.dirname(ck)
    if os.path.basename(d) == 'checkpoints':
        rd = os.path.dirname(d)
        rn = os.path.basename(rd)
        if rn == 'train':
            rn = os.path.basename(os.path.dirname(rd))
        b = os.path.basename(ck)
        return rn + '_' + (b[:-3] if b.endswith('.pt') else b)
    b = os.path.basename(ck)
    return b[:-3] if b.endswith('.pt') else b


def column(trace_dir):
    """(trace summary.json, bones_summary.json) of one trace dir -> dict(metrics, bones, bone_fix, ok)."""
    s = load(os.path.join(trace_dir, 'summary.json')) if trace_dir else None
    d = load(os.path.join(trace_dir, 'bones_summary.json')) if trace_dir else None
    v = (s or {}).get('variants', {}).get(V, {})
    dv = None if not d else (d['variants'].get(V) or next(iter(d['variants'].values())))
    g = v.get('rot_err_deg_by_group', {}) or {}
    sk = (dv or {}).get('neck_skin', {}) or {}
    m = dict(pve=v.get('pve'), pa_pve=v.get('pa_pve'), pve_headneck=v.get('pve_headneck'), recall=v.get('recall'),
             rot_global_head_deg=(v.get('rot_global_deg') or {}).get('head'), rot_fingers=g.get('fingers'),
             rot_neck=g.get('neck'), rot_head=g.get('head'), girth=sk.get('neck_girth_ratio_mean'),
             pve_neck_local=sk.get('pve_neckskin_local_mm'))
    return dict(dir=trace_dir, metrics=m, bones=(dv or {}).get('bones', {}), bone_fix=(s or {}).get('bone_fix'),
                ok=bool(s and d), order=[b['name'] for b in (d or {}).get('bones', [])],
                ckpt=(s or {}).get('ckpt') or (d or {}).get('ckpt'))


def flagged(bones):
    return [n for n, b in bones.items() if b.get('verdict') in FLAG_VERDICTS and (b.get('err_median') or 0) > FLAG_MIN_MEDIAN]


def judge(col, ref, fixed, tol, guards):
    """List of failed criteria (empty = pass) and a dict of the checks."""
    fails, checks = [], {}
    for n in fixed:
        med = (col['bones'].get(n) or {}).get('err_median')
        ok = med is not None and med < BONE_OK_DEG
        checks[f'{n} median < {BONE_OK_DEG:g}'] = (med, ok)
        if not ok:
            fails.append(f'{n} median {f(med)}')
    gi = col['metrics']['girth']
    ok = gi is not None and gi >= GIRTH_MIN
    checks[f'girth >= {GIRTH_MIN}'] = (gi, ok)
    if not ok:
        fails.append(f'girth {f(gi, 3)}')
    for k in guards:
        x, r = col['metrics'].get(k), (ref or {}).get(k)
        if r is None:
            checks[f'{k} vs ref'] = (x, None)
            fails.append(f'{k}: no reference')
            continue
        lim = r - tol[k] if k == 'recall' else r + tol[k]
        ok = x is not None and (x >= lim if k == 'recall' else x <= lim)
        checks[f'{k} {">=" if k == "recall" else "<="} {f(lim)}'] = (x, ok)
        if not ok:
            fails.append(f'{k} {f(x)} vs limit {f(lim)}')
    return fails, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline_dir', required=True, help='trace dir of the checkpoint WITHOUT a fix (diag_neck_bones.sh)')
    ap.add_argument('--fix_dir', required=True, help='holdout trace dir of the same checkpoint WITH --bone_fix')
    ap.add_argument('--ft_dir', default='', help='holdout trace dir of the checkpoint fine-tuned with the fix')
    ap.add_argument('--control_run', default='', help='exp5 L1 control run dir (holdout_trace/ inside)')
    ap.add_argument('--bone_fix', default='', help='the bone_fix.json that was applied')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    base, zt = column(a.baseline_dir), column(a.fix_dir)
    ft = column(a.ft_dir) if a.ft_dir else None
    ctl = column(os.path.join(a.control_run, 'holdout_trace')) if a.control_run else None
    if ctl is not None and not ctl['ok']:
        print(f"note: control run {a.control_run} has no finished holdout_trace/: control column left empty")
        ctl = None
    if not zt['ok']:
        print(f"ERROR: zero-training trace {a.fix_dir} has no summary.json/bones_summary.json")
        return 2
    spec = load(a.bone_fix) or {}
    fixed = list((spec.get('bones') or {}).keys()) or [b['name'] for b in (zt['bone_fix'] or {}).get('bones', [])]
    flags = flagged(base['bones']) if base['ok'] else []
    order = base['order'] or zt['order']
    rows = list(dict.fromkeys([n for n in NECK if n in order or n in zt['bones']]
                              + [n for n in order if n in set(fixed) | set(flags)] + fixed))

    cols = [('baseline', base), ('zero-training', zt), ('fine-tuned', ft), ('control', ctl)]
    print()
    print('=== exp6_bone_fix SUMMARY (V1, 100-image holdout, matched people; bone cells = verdict median deg) ===')
    print('baseline:      ' + a.baseline_dir + ('' if base['ok'] else ' (MISSING: run slurm/diagnostics/diag_neck_bones.sh)'))
    print('zero-training: ' + a.fix_dir)
    print('fine-tuned:    ' + (a.ft_dir or 'not run (TRAIN=0)'))
    print('control:       ' + (a.control_run if ctl else 'none') + '   (exp5 L1 recipe without the fix; context for fine-tuned)')
    print('bone fix:      ' + (a.bone_fix or '?') + ': ' + (', '.join(
        f"{b['name']} {b['angle_deg']:.1f} deg" for b in (zt['bone_fix'] or {}).get('bones', [])) or 'NOT ACTIVE in the trace'))
    if not zt['bone_fix']:
        print('WARNING: the zero-training trace summary records no active bone fix: it ran without --bone_fix?')
    if ft and ft['ok'] and not ft['bone_fix']:
        print('WARNING: the fine-tuned trace records no active bone fix: its checkpoint args lost bone_fix_spec?')
    src_tag = tag_of((spec.get('source') or {}).get('ckpt'))
    for nm, c in (('baseline', base), ('zero-training', zt)):
        if src_tag and c['ok'] and c['ckpt'] and tag_of(c['ckpt']) != src_tag:
            print(f"WARNING: the {nm} trace is of {c['ckpt']} (TAG {tag_of(c['ckpt'])}), but the fix was estimated on "
                  f"TAG {src_tag}: this column belongs to another checkpoint")
    gg = spec.get('global_guard') or {}
    gpath = os.path.join(os.path.dirname(a.bone_fix), 'bone_fix_guarded.json') if a.bone_fix else ''
    guarded = load(gpath) if gpath else None
    if guarded and (guarded.get('global_guard') or {}).get('risen'):
        # an ON_GLOBAL_RISE=safe run re-estimates with --bones <the guarded file's safe subset>, which the guard passes.
        # Anything else (e.g. bone_fix.json = the full guarded fix of a later continue run) is a leftover of an earlier
        # run of the same RUN dir and is not reported as this run's selection
        g_b, s_b = set(guarded.get('bones') or {}), set(spec.get('bones') or {})
        if s_b and s_b == set(guarded['global_guard'].get('safe_subset') or []) and not gg.get('risen'):
            print('step a: the first selection (' + ', '.join(guarded.get('bones') or {}) + ') fired the GLOBAL GUARD; '
                  'this run tests its safe subset (ON_GLOBAL_RISE=safe; full fix in bone_fix_guarded.json)')
        else:
            print(f"WARNING: {gpath} does not belong to this fix (bones {', '.join(sorted(s_b)) or 'none'}"
                  f"{', guard risen' if gg.get('risen') else ''}; guarded {', '.join(sorted(g_b))}): a leftover of an "
                  f"earlier ON_GLOBAL_RISE=safe run, ignored")
    if gg.get('risen'):
        r = gg['risen']
        print(f"step a GLOBAL GUARD (training sample): this fix raises the predicted global median by > "
              f"{gg.get('max_rise_deg')} deg on " + ', '.join(
                  f"{n} {f(v.get('before_median'))} -> {f(v.get('after_median'))}" for n, v in r.items()))
        print('    corrected bones on those chains: ' + ', '.join(gg.get('implicated') or []) + '; safe subset: '
              + (', '.join(gg.get('safe_subset') or []) or 'none') + '. A zero-training FAIL on PVE / global head there '
              'is this predicted compensated-chain effect, not a new one.')
    elif gg.get('checked'):
        print(f"step a GLOBAL GUARD: no dumped bone's predicted global median rises by > {gg.get('max_rise_deg')} deg")
    print('baseline-flagged bones (FLIP_*/CONST_OFFSET/FIXED_OUTPUT, median > 90): ' + (', '.join(flags) or 'none')
          + (('; NOT corrected: ' + ', '.join(sorted(set(flags) - set(fixed)))) if set(flags) - set(fixed) else ''))
    W = 24
    print(format('bone', '<22') + ''.join(format(nm, '>' + str(W)) for nm, _ in cols))

    def bcell(c, n):
        if not c or not c['ok'] or n not in c['bones']:
            return '-'
        b = c['bones'][n]
        return f"{b.get('verdict')} {f(b.get('err_median'))}"
    for n in rows:
        print(format(n + (' *' if n in fixed else ''), '<22') + ''.join(format(bcell(c, n), '>' + str(W)) for _, c in cols))
    for k, lab, nd in METRICS:
        print(format(lab, '<22') + ''.join(format(f(c['metrics'].get(k), nd) if c and c['ok'] else '-', '>' + str(W))
                                         for _, c in cols))
    print('* = corrected by the bone fix')

    zt_fail, zt_checks = judge(zt, base['metrics'] if base['ok'] else None, fixed, ZT_TOL, ('pve', 'rot_global_head_deg'))
    ref_ft = ctl['metrics'] if ctl else (base['metrics'] if base['ok'] else None)
    ft_fail, ft_checks = (judge(ft, ref_ft, fixed, FT_TOL, ('pve', 'recall', 'rot_global_head_deg'))
                          if ft and ft['ok'] else (None, None))
    print(f"criteria: zero-training vs baseline = corrected bones median < {BONE_OK_DEG:g} deg, girth >= {GIRTH_MIN}, "
          f"global head <= baseline + {ZT_TOL['rot_global_head_deg']:g} deg, PVE <= baseline + {ZT_TOL['pve']:g} mm;")
    print(f"          fine-tuned vs {'the control' if ctl else 'the baseline (no control)'} = corrected bones median < "
          f"{BONE_OK_DEG:g}, girth >= {GIRTH_MIN}, PVE <= ref + {FT_TOL['pve']:g}, recall >= ref - {FT_TOL['recall']:g}, "
          f"global head <= ref + {FT_TOL['rot_global_head_deg']:g}")
    zt_v = 'PASS' if not zt_fail else 'FAIL (' + '; '.join(zt_fail) + ')'
    ft_v = ('not run' if ft_fail is None else ('PASS' if not ft_fail else 'FAIL (' + '; '.join(ft_fail) + ')'))
    print('ZERO-TRAINING FIX: ' + zt_v)
    print('FINE-TUNED FIX:    ' + ft_v)
    print('read: zero-training PASS = the constant correction alone removes the flips on unseen images (the flip is a '
          'fixed output offset of the head);')
    print('      a corrected bone still > 15 deg with relSpr ~15 in the baseline dump = a per-person compensation no constant '
          'undoes (v5 head) -> fine-tuned column; global head up = the uncompensated residual shows in the head direction.')
    print('      The fix is a constant estimated on training images (compute_bone_fix.py): it says nothing about WHY the '
          'head converged there.')
    res = dict(rows=rows, fixed=fixed, baseline_flagged=flags,
               columns={nm: (None if not c else dict(dir=c['dir'], ok=c['ok'], metrics=c['metrics'], bone_fix=c['bone_fix'],
                                                      bones={n: {k: c['bones'].get(n, {}).get(k) for k in ('verdict', 'err_median', 'err_mean')}
                                                             for n in rows}))
                        for nm, c in cols},
               zero_training=dict(verdict=zt_v, failed=zt_fail, checks={k: list(v) for k, v in zt_checks.items()}),
               fine_tuned=dict(verdict=ft_v, failed=ft_fail,
                               checks=({k: list(v) for k, v in ft_checks.items()} if ft_checks else None)),
               criteria=dict(bone_ok_deg=BONE_OK_DEG, girth_min=GIRTH_MIN, zero_training_tol=ZT_TOL, fine_tuned_tol=FT_TOL),
               step_a_global_guard=gg or None)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('wrote ' + a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
