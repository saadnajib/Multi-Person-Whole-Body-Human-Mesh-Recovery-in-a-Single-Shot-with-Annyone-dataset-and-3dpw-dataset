"""
exp6 bone fix, estimation step: one constant correction C_b per bone from a per-bone dump of TRAINING images.

    python scripts/diagnostics/compute_bone_fix.py --trace_dir <dir with bones.npz + bones_summary.json> \
        [--variant V1_eval_masked] [--bones auto | neck02,head,...] --out bone_fix.json

Input: a diag_holdout_trace.py run with --dump_bones 1 and --bone_fix none, on images the fix is NOT evaluated on
(--select train_sample; slurm/experiments/exp6_bone_fix.sh step a). The model then applies R_b <- R_b @ C_b
(model.py, --bone_fix bone_fix.json in train.py / diag_holdout_trace.py / demo.py).

Bones: --bones auto = every dumped, active bone whose verdict in bones_summary.json (--variant) is one of --verdicts
(FLIP_TWIST, FLIP_SWING, CONST_OFFSET, FIXED_OUTPUT) with a median local error > --min_median (90 deg); or an explicit
comma list of dumped bone names (unknown = error).
Estimate: C_b = SO(3) projection (SVD, det +1) of mean_i R_pred_i^T R_gt_i over the matched people (the chordal L2
mean; R_pred = R_pred_local_raw, the prediction without any fix, R_gt = the SVD-orthonormalised dataset rotation),
i.e. the constant that minimises sum_i |R_pred_i C - R_gt_i|_F^2.
Printed per bone: angle/axis of C_b; median / p90 local error before and after on the estimation set; the same
after a 2-fold split by image (C from one half applied to the other: an out-of-sample read inside the estimation set);
the singular values of the mean (all ~1 = one rotation fits everyone; small = the per-person errors disagree and no
constant fits). Then the GLOBAL (camera-frame) orientation error of every dumped bone at or below a corrected bone,
before vs after, from the dumped global poses: G_b'' = prod_a (G_a C_a G_a^T) G_b over the corrected ancestors a of b
(b included), root-most first, exact for anny's FK (pose_b = pose_parent rest_parent^-1 rest_b delta_b) because a
local right-factor C_a changes every descendant's global rotation by G_a C_a G_a^T.
Read [Likely, from the measured spreads]: a bone with relSpr ~0-5 deg in the dump (metacarpals, v5 neck02) goes to
~relSpr after the fix; a bone whose half-turn compensates another one's (v5 head, relSpr ~15) keeps ~relSpr, because a
constant cannot undo a per-person compensation; its global error is the number to watch.
GLOBAL GUARD (--max_global_rise, 5 deg): a flip that a bone further down compensates is a compensated CHAIN (v5
neck02 -> neck03 -> head; v10 shoulder01.L -> upperarm01.L -> upperarm02.L, where upperarm01.L in between is not a
flip but its GT rotation conjugated by the half-turn, local ~2x its GT angle, verdict SPREAD, so auto skips it).
Right-multiplying the chain's flipped bones by constants then leaves the link in between wrong and un-compensated:
the predicted global median of the bones below rises (synthetic v10 arm: upperarm02.L 0 -> 56 deg, elbow ~260 mm).
Any dumped bone whose predicted global median rises by more than --max_global_rise is listed with the corrected bones
on its chain, and the largest subset of the selected bones that raises no global median (dropping every corrected
bone on a risen bone's chain, repeated until none rises) is printed as the SAFE SUBSET; both go into the JSON
(global_guard). A correction that keeps such a chain intact needs a two-sided constant on the link (not implemented).

Exit status: 0 = JSON written; 4 = JSON written, but the GLOBAL GUARD fired (WARNING; exp6_bone_fix.sh acts on it per
ON_GLOBAL_RISE); 3 = no bone qualifies (nothing written; exp6 stops with NO TARGET); 2 = bad input.
numpy only (no torch / anny): runs on a login node.
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import bone_fix as bfl  # noqa: E402

AUTO_VERDICTS = ('FLIP_TWIST', 'FLIP_SWING', 'CONST_OFFSET', 'FIXED_OUTPUT')


def f(x, nd=1):
    return '-' if x is None or not np.isfinite(x) else format(float(x), '.%df' % nd)


def stats(e):
    e = np.asarray(e, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return dict(median=None, p90=None, mean=None, n=0)
    return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), mean=float(e.mean()), n=int(e.size))


def select_bones(arg, names, active, summ_v, verdicts, min_median, med_raw, relaxed=None):
    """Names to correct. auto: dumped + active + summary verdict in verdicts + median > min_median. relaxed [n_dump]:
    finger/thumb bones the variant's relax_hands forces to identity (V3): no prediction to correct there."""
    relaxed = relaxed or [False] * len(names)
    if arg.strip().lower() != 'auto':
        want = [s.strip() for s in arg.split(',') if s.strip()]
        low = {n.lower(): n for n in names}
        out = []
        for w in want:
            if w.lower() not in low:
                raise SystemExit(f"[compute_bone_fix] --bones: '{w}' is not among the {len(names)} dumped bones "
                                 f"({', '.join(names[:12])}, ...); add it to the trace's --dump_bone_names")
            n = low[w.lower()]
            if not active[names.index(n)]:
                raise SystemExit(f"[compute_bone_fix] --bones: '{n}' is a helper-masked bone (identity under V1-V3 and "
                                 f"--mask_helper_joints_train); a fix there has no effect")
            if relaxed[names.index(n)]:
                raise SystemExit(f"[compute_bone_fix] --bones: '{n}' is a finger/thumb bone that this variant's relax_hands "
                                 f"forces to identity; estimate on V1_eval_masked")
            out.append(n)
        return list(dict.fromkeys(out)), 'explicit --bones'
    if summ_v is None:
        raise SystemExit("[compute_bone_fix] --bones auto needs bones_summary.json (--summary) with this variant")
    out = []
    for k, n in enumerate(names):
        s = summ_v['bones'].get(n)
        if s is None or not active[k] or relaxed[k]:
            continue
        if s.get('verdict') in verdicts and (s.get('err_median') or 0) > min_median and med_raw[k] > min_median:
            out.append(n)
    return out, f"auto: verdict in {list(verdicts)} and median > {min_median:g} deg (bones_summary.json)"


def corrected_chain(k, bone_index, rig_parents, fix):
    """dump positions of the corrected bones on dump position k's chain (k itself and its ancestors), root-most first."""
    pos = {int(b): j for j, b in enumerate(bone_index)}
    chain, a = [], int(bone_index[k])
    while a >= 0:
        if a in pos and pos[a] in fix:
            chain.append(pos[a])
        a = int(rig_parents[a])
    return chain[::-1]


def predicted_global(G_pred, G_gt, names, bone_index, rig_parents, fix):
    """{bone: (before err [P], after err [P])} for every dumped bone with a corrected ancestor-or-self.
    G_* [P, n_dump, 3, 3]; fix = {dump position k: C [3,3]}."""
    out = {}
    for k in range(len(bone_index)):
        chain = corrected_chain(k, bone_index, rig_parents, fix)       # root-most first
        if not chain:
            continue
        M = np.broadcast_to(np.eye(3), G_pred[:, k].shape).copy()
        for ka in chain:
            Ga = G_pred[:, ka]
            M = M @ (Ga @ fix[ka] @ np.swapaxes(Ga, -1, -2))
        G_new = M @ G_pred[:, k]
        out[names[k]] = (bfl.angle_between_deg(G_pred[:, k], G_gt[:, k]), bfl.angle_between_deg(G_new, G_gt[:, k]))
    return out


def global_guard(G_pred, G_gt, names, bone_index, rig_parents, fix, max_rise):
    """The global-rise check of a fix {dump position: C}. Returns (risen, implicated, safe):
    risen = {bone: (median before, median after)} for the dumped bones whose predicted global median rises by more
    than max_rise with the full fix; implicated = the corrected bones on those bones' chains (all rounds);
    safe = the selected bones left after dropping the implicated ones, repeated until no global median rises."""
    first, implicated, cur = None, [], dict(fix)
    while cur:
        G = predicted_global(G_pred, G_gt, names, bone_index, rig_parents, cur)
        risen = {}
        for n, (b0, b1) in G.items():
            m0, m1 = stats(b0)['median'], stats(b1)['median']
            if m0 is not None and m1 is not None and m1 - m0 > max_rise:
                risen[n] = (m0, m1)
        if first is None:
            first = risen
        if not risen:
            break
        drop = []
        for n in risen:
            drop += [names[j] for j in corrected_chain(names.index(n), bone_index, rig_parents, cur)]
        drop = list(dict.fromkeys(drop))
        implicated += [n for n in drop if n not in implicated]
        cur = {k: C for k, C in cur.items() if names[k] not in drop}
    safe = [names[k] for k in fix if names[k] not in implicated]          # selection order
    return first or {}, implicated, safe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace_dir', default='', help='diag_holdout_trace.py --dump_bones 1 output dir (bones.npz, bones_summary.json)')
    ap.add_argument('--npz', default='', help='bones.npz (default <trace_dir>/bones.npz)')
    ap.add_argument('--summary', default='', help='bones_summary.json (default <trace_dir>/bones_summary.json)')
    ap.add_argument('--variant', default='V1_eval_masked')
    ap.add_argument('--bones', default='auto', help="'auto' or a comma list of dumped bone names")
    ap.add_argument('--verdicts', default=','.join(AUTO_VERDICTS), help='--bones auto: qualifying verdicts')
    ap.add_argument('--min_median', type=float, default=90.0, help='--bones auto: median local error above this (deg)')
    ap.add_argument('--allow_fixed_input', type=int, default=0, choices=[0, 1],
                    help='1: accept a dump made WITH a bone fix (uses R_pred_local_raw; no global prediction). '
                         '0 (default): refuse, rerun the trace with --bone_fix none')
    ap.add_argument('--max_global_rise', type=float, default=5.0,
                    help='GLOBAL GUARD: exit 4 (JSON still written) when the predicted global median error of any dumped '
                         'bone rises by more than this (deg); <0 = off')
    ap.add_argument('--out', required=True, help='bone-fix JSON to write')
    a = ap.parse_args()

    npz_p = a.npz or os.path.join(a.trace_dir, 'bones.npz')
    sum_p = a.summary or os.path.join(a.trace_dir, 'bones_summary.json')
    if not os.path.isfile(npz_p):
        print(f"[compute_bone_fix] ERROR: {npz_p} not found (run diag_holdout_trace.py --dump_bones 1)")
        return 2
    Z = np.load(npz_p)
    summ = json.load(open(sum_p)) if os.path.isfile(sum_p) else None
    summ_v = (summ or {}).get('variants', {}).get(a.variant)
    names = [str(n) for n in Z['bone_names']]
    bone_index = [int(b) for b in Z['bone_index']]
    sel = np.nonzero(Z['variant'].astype(str) == a.variant)[0]
    if len(sel) == 0:
        print(f"[compute_bone_fix] ERROR: no records of variant {a.variant} in {npz_p} "
              f"(has {sorted(set(Z['variant'].astype(str)))})")
        return 2
    fixed_in = 'bone_fix_on' in Z.files and bool(np.asarray(Z['bone_fix_on']).any())
    if fixed_in and not a.allow_fixed_input:
        print(f"[compute_bone_fix] ERROR: {npz_p} was dumped WITH a bone fix on "
              f"{[n for n, o in zip(names, Z['bone_fix_on']) if o]}; rerun the trace with --bone_fix none "
              f"(or --allow_fixed_input 1)")
        return 2
    if 'R_pred_local_raw' in Z.files:
        R_pred = Z['R_pred_local_raw'][sel].astype(np.float64)
    else:
        print("[compute_bone_fix] note: older bones.npz without R_pred_local_raw; using R_pred_local (= the raw "
              "prediction, since no fix existed when it was written)")
        R_pred = Z['R_pred_local'][sel].astype(np.float64)
    R_gt = Z['R_gt_local'][sel].astype(np.float64)
    active = ([bool(x) for x in Z['bone_active']] if 'bone_active' in Z.files else
              [bool(b.get('active', True)) for b in (summ or {}).get('bones', [])] or [True] * len(names))
    img = Z['idx'][sel].astype(np.int64)
    P = len(sel)
    err_raw = bfl.angle_between_deg(R_pred, R_gt)                                    # [P, n_dump]
    med_raw = np.median(err_raw, axis=0)

    verdicts = tuple(v.strip() for v in a.verdicts.split(',') if v.strip())
    # V3_demo_relaxhands forces finger/thumb bones to identity (model.hand_bone_mask; bones.npz bone_hand, older dumps:
    # the same name rule): their prediction there is no head output, so they are never corrected from that variant
    hand = ([bool(x) for x in Z['bone_hand']] if 'bone_hand' in Z.files else
            [('finger' in n.lower() or 'thumb' in n.lower()) for n in names])
    relaxed = [h and 'relaxhands' in a.variant.lower() for h in hand]
    bones, how = select_bones(a.bones, names, active, summ_v, verdicts, a.min_median, med_raw, relaxed)
    sel_desc = (summ or {}).get('selection', '?')
    print(f"=== compute_bone_fix: {npz_p} ({a.variant}, {P} matched people in {len(set(img.tolist()))} images; "
          f"selection: {sel_desc})")
    print(f"bones: {how} -> {bones or 'NONE'}")
    if summ_v is not None:
        flagged = [(n, s['verdict'], s.get('err_median')) for n, s in summ_v['bones'].items()
                   if s.get('verdict') not in ('OK', '-')]
        print("non-OK verdicts in the dump: " + (", ".join(f"{n} {v} {f(m)}" for n, v, m in flagged) or 'none'))
    if not bones:
        print("[compute_bone_fix] no bone qualifies: nothing written (NO TARGET)")
        return 3
    if 'selection' in (summ or {}) and 'HOLDOUT' in str(sel_desc).upper() and 'TRAIN_SAMPLE' not in str(sel_desc).upper():
        print("[compute_bone_fix] WARNING: this dump is a HOLDOUT trace; a fix estimated here must not be evaluated "
              "on the same holdout (use --select train_sample)")

    half = (img % 2 == 0)                                   # 2-fold split by image, so one image's people stay together
    fix_pos, out_bones, st = {}, {}, {}
    print(f"{'bone':<15}{'C angle':>8}  {'C axis':<22}{'sv(mean) min/max':>17}{'before med/p90':>17}{'after med/p90':>16}"
          f"{'2-fold med/p90':>16}{'relSpr':>8}")
    for n in bones:
        k = names.index(n)
        E = np.swapaxes(R_pred[:, k], -1, -2) @ R_gt[:, k]                           # R_pred^T R_gt, [P,3,3]
        C, sv = bfl.chordal_mean(E)
        ax, ang = bfl.axis_angle(C)
        before = err_raw[:, k]
        after = bfl.angle_between_deg(R_pred[:, k] @ C, R_gt[:, k])
        cv = np.full(P, np.nan)
        if half.any() and (~half).any():
            for m_est, m_ev in ((half, ~half), (~half, half)):
                Ch, _ = bfl.chordal_mean(E[m_est])
                cv[m_ev] = bfl.angle_between_deg(R_pred[m_ev, k] @ Ch, R_gt[m_ev, k])
        rel = float(bfl.angle_between_deg(E, np.broadcast_to(C, E.shape)).mean())
        sb, sa, sc = stats(before), stats(after), stats(cv)
        print(f"{n:<15}{f(ang):>8}  {'[' + ', '.join(f'{x:+.2f}' for x in ax) + ']':<22}"
              f"{f(sv.min(), 3) + '/' + f(sv.max(), 3):>17}{f(sb['median']) + '/' + f(sb['p90']):>17}"
              f"{f(sa['median']) + '/' + f(sa['p90']):>16}{f(sc['median']) + '/' + f(sc['p90']):>16}{f(rel):>8}")
        fix_pos[k] = C
        out_bones[n] = C
        st[n] = dict(index=int(bone_index[k]), angle_deg=ang, axis=[float(x) for x in ax],
                     mean_singular_values=[float(x) for x in sv], rel_spread_deg=rel, local_before=sb, local_after=sa,
                     local_after_2fold=sc,
                     verdict=(summ_v['bones'].get(n, {}).get('verdict') if summ_v else None))

    high = [n for n in bones if (st[n]['local_after']['median'] or 0) >= 15.0]
    if high:
        print("NOTE: corrected local median still >= 15 deg (the zero-training bar) on the estimation set: "
              + ", ".join(f"{n} {f(st[n]['local_after']['median'])} (relSpr {f(st[n]['rel_spread_deg'])})" for n in high)
              + "; a per-person error no constant removes")

    glob = {}
    guard = dict(max_rise_deg=a.max_global_rise, checked=False, risen={}, implicated=[], safe_subset=list(bones))
    have_glob = all(x in Z.files for x in ('R_pred_global', 'R_gt_global', 'rig_parents'))
    if have_glob and not fixed_in:
        Gp, Gg = Z['R_pred_global'][sel].astype(np.float64), Z['R_gt_global'][sel].astype(np.float64)
        G = predicted_global(Gp, Gg, names, bone_index, Z['rig_parents'], fix_pos)
        risen, impl, safe = ({}, [], list(bones)) if a.max_global_rise < 0 else \
            global_guard(Gp, Gg, names, bone_index, Z['rig_parents'], fix_pos, a.max_global_rise)
        guard.update(checked=a.max_global_rise >= 0, implicated=impl, safe_subset=safe,
                     risen={n: dict(before_median=m0, after_median=m1) for n, (m0, m1) in risen.items()})
        print("predicted GLOBAL (camera-frame) orientation error of the dumped bones at/below a corrected bone, "
              "estimation set (deg, median/mean):")
        for n, (b0, b1) in G.items():
            s0, s1 = stats(b0), stats(b1)
            glob[n] = dict(before=s0, after=s1)
            print(f"    {n:<15} {f(s0['median'])}/{f(s0['mean'])} -> {f(s1['median'])}/{f(s1['mean'])}"
                  + (f"   <- RISES > {a.max_global_rise:g} deg" if n in risen else '')
                  + ("   <- pass criterion: holdout global head within +3 deg of the baseline" if n == 'head' else ''))
        if 'head' not in G:
            print("    (head is not below any corrected bone: its global orientation is unchanged by this fix)")
        if risen:
            print(f"WARNING: GLOBAL GUARD: the fix raises the predicted global median by > {a.max_global_rise:g} deg on "
                  f"{len(risen)} dumped bone(s): " + ", ".join(f"{n} {f(m0)} -> {f(m1)}" for n, (m0, m1) in risen.items()))
            print(f"    corrected bones on their chains: {', '.join(impl)}. Read: a COMPENSATED CHAIN (a flip that a bone "
                  f"below undoes; the link in between")
            print("    carries the half-turn conjugated, e.g. v10 upperarm01.L SPREAD ~2x its GT angle, v5 neck03/head). One "
                  "constant per flipped bone leaves")
            print("    that link wrong and no longer compensated, so the zero-training test will fail on PVE / global head "
                  "there; keeping the chain")
            print("    needs a two-sided constant on the link (not implemented).")
            print(f"    SAFE SUBSET (no predicted global rise): {', '.join(safe) or 'NONE'}"
                  + (f"  -> BONES={','.join(safe)}" if safe else ''))
    else:
        print("global prediction skipped (" + ('dump made with a fix' if fixed_in else
                                              'older bones.npz without rig_parents') + "): GLOBAL GUARD not checked")

    meta = dict(
        source=dict(npz=os.path.abspath(npz_p), summary=os.path.abspath(sum_p) if summ else None, variant=a.variant,
                    ckpt=(summ or {}).get('ckpt'), ckpt_args=(summ or {}).get('ckpt_args'), n_people=int(P),
                    n_images=int(len(set(img.tolist()))), bones_rule=how),
        estimated_on=sel_desc,
        method='C_b = SO(3) projection (SVD, det +1) of mean_i R_pred_i^T R_gt_i (chordal L2 mean); applied as '
               'R_b <- R_b @ C_b on the head\'s final local rotmats (model.py)',
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        stats=st, predicted_global=glob, global_guard=guard)
    bfl.save(a.out, out_bones, **meta)
    print(f"wrote {a.out}: {len(out_bones)} bone(s) " + ", ".join(f"{n} {st[n]['angle_deg']:.1f} deg" for n in out_bones))
    print("  read: 'after' = the zero-training prediction on the estimation images; '2-fold' = the same with C estimated on")
    print("        the other half of the images (out-of-sample inside the training sample); the holdout trace with")
    print("        --bone_fix decides. sv(mean) min << 1 or relSpr >> 15 = the errors are not one constant rotation.")
    if guard['risen']:
        print("[compute_bone_fix] exit 4: GLOBAL GUARD fired (JSON written; global_guard in it names the safe subset)")
        return 4
    return 0


if __name__ == '__main__':
    sys.exit(main())
