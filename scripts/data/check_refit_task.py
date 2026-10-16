#!/usr/bin/env python
"""
check_refit_task.py -- post-check of smpl_to_anny.py fits for ONE 3DPW sequence (or every sequence of a fits root),
made for the v2.3 refit (--gender_convention anny --frame_grid shared). NumPy only, CPU, seconds.

For every NEW .npz it prints and checks
  * gender: the fixed Anny gender value (every frame's anny_shape, and the saved gender_value) next to the 3DPW gender
    and the convention the file says it used. Anny: 0 = male, 1 = female ('anny'); v2.2 wrote the reverse
    ('makehuman', m -> 1.0).
  * frame grid: the grid the file says it used; for 'shared' every fitted frame must satisfy frame % stride == 0, lie
    on a frame 3DPW marks valid, and (unless max_frames capped it) every valid grid frame must be fitted.
For two people of one sequence (p0 and p1) it prints the shared fitted frames and the frames where BOTH are valid in
3DPW but only ONE is fitted (count, and fraction of the fitted frames where both are valid). That is check I3's
'valid actor unlabelled, not fitted' before the label builder's filters: v2.2's per-person stride left 407 of 960
sampled two-person images with an unfitted valid actor; with the shared grid it should be ~0.
With OLD .npz files of the same people (e.g. threedpw_anny_full) it also prints the joint error mean / median before
vs after, over all fitted frames and paired on the frames both runs fitted, the old gender value and the old pair
coverage.

Validity (3DPW campose_valid + the converter's NaN filter) comes from the sequence .pkl when one is found (--pkl,
--seq_root/<split>/<seq>.pkl, or the pkl_path saved in the .npz), else from the valid_mask a v2.3 .npz saves.

EXIT CODE: 0 = every new .npz follows the expected conventions; 1 = at least one violation (listed as 'VIOLATION');
2 = bad input (no files, unreadable .npz). (--match_flags has its own codes, see USAGE.) Joint-error changes are reported, never graded (women's joint error can
rise a little with the correct gender: check 3575369's refit test gave 13.7 -> 16.3 mm for them).

USAGE
  # one sequence in the cluster layout (old fits optional):
  python scripts/data/check_refit_task.py --split train --seq courtyard_captureSelfies_00 \\
      --new_root threedpw_anny_v23 --old_root threedpw_anny_full --seq_root data/3DPW/sequenceFiles
  # every sequence of the new root:
  python scripts/data/check_refit_task.py --seq all --new_root threedpw_anny_v23 --old_root threedpw_anny_full \\
      --seq_root data/3DPW/sequenceFiles
  # explicit files: positional = new fits; --old = old fits (matched by person)
  python scripts/data/check_refit_task.py NEW_p0.npz NEW_p1.npz --old OLD_p0.npz OLD_p1.npz --pkl seq.pkl
  # two files of the SAME person and no --old: the first is the old, the second the new fit
  python scripts/data/check_refit_task.py OLD_p0.npz NEW_p0.npz
  # flags only (what slurm/data/run_smpl_to_anny_task.sh runs before it skips an existing fit outside
  # threedpw_anny_full): was FIT.npz made with the gender_convention / frame_grid these extra flags request?
  # Exit 0 = same, 1 = different, 2 = unreadable / bad flag. Keep the '=': the value starts with '--'.
  python scripts/data/check_refit_task.py --match_flags FIT.npz \\
      --refit_flags="--gender_convention anny --frame_grid shared"
"""
import argparse
import glob
import itertools
import os
import pickle
import shlex
import sys

import numpy as np

# == scripts/data/smpl_to_anny.py GENDER_CONVENTIONS (kept in sync by the tests in scratchpad/convfix)
GENDER_CONVENTIONS = {
    'makehuman': {'m': 1.0, 'f': 0.0},
    'anny':      {'m': 0.0, 'f': 1.0},
}
FRAME_GRIDS = ('per_person', 'shared')
# == the smpl_to_anny.py argparse defaults (the v2.2 behaviour); also what a file without the v2.3 keys was made with
DEFAULT_GENDER_CONVENTION = 'makehuman'
DEFAULT_FRAME_GRID = 'per_person'
GENDER_TOL = 0.01          # fixed values come out as sigmoid(logit(clamp(v, 1e-4))) = 1e-4 / 0.9999


# ---------------------------------------------------------------------------------------------------------------
# 3DPW side (same rules as smpl_to_anny.py)
# ---------------------------------------------------------------------------------------------------------------
def read_gender(seq, person):
    """'m' / 'f' / None from a 3DPW pickle dict (== smpl_to_anny.read_gender)."""
    g = seq.get('genders', seq.get('gender', None))
    if g is None:
        return None
    try:
        g = g[person]
    except (IndexError, TypeError, KeyError):
        return None
    if isinstance(g, bytes):
        g = g.decode('latin1')
    g = str(g).strip().lower()
    if not g:
        return None
    return 'm' if g[0] == 'm' else ('f' if g[0] == 'f' else None)


def valid_mask_from_seq(seq, person):
    """Per-frame validity [T] bool, the filter smpl_to_anny.find_valid_frames applies: campose_valid (when its length
    matches) AND no NaN in poses / trans / jointPositions."""
    poses = np.asarray(seq['poses'][person])
    T = poses.shape[0]
    trans = np.asarray(seq['trans'][person]) if 'trans' in seq else np.zeros((T, 3))
    valid = np.ones(T, dtype=bool)
    if 'campose_valid' in seq:
        cv = np.asarray(seq['campose_valid'][person]).astype(bool)
        if len(cv) == T:
            valid &= cv
    valid &= ~np.isnan(poses).any(axis=1)
    if trans.shape[0] == T:
        valid &= ~np.isnan(trans).any(axis=1)
    if 'jointPositions' in seq:
        jp = np.asarray(seq['jointPositions'][person])
        if jp.shape[0] == T:
            valid &= ~np.isnan(jp.reshape(T, -1)).any(axis=1)
    return valid


def load_seq(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


# ---------------------------------------------------------------------------------------------------------------
# .npz side
# ---------------------------------------------------------------------------------------------------------------
def _str(d, k):
    return str(d[k]) if k in d.files else None


def load_fit(path):
    """The fields of one smpl_to_anny.py .npz this check needs. Missing v2.3 fields come back as None."""
    d = np.load(path, allow_pickle=True)
    labels = [str(x) for x in d['phenotype_labels']] if 'phenotype_labels' in d.files else []
    shape = np.asarray(d['anny_shape'], dtype=np.float64) if 'anny_shape' in d.files else None
    g_col = None
    if shape is not None and 'gender' in labels and shape.ndim == 2:
        g_col = shape[:, labels.index('gender')]
    base = os.path.basename(path)[:-4]
    person = int(d['person']) if 'person' in d.files else (int(base.rsplit('_p', 1)[1]) if '_p' in base else -1)
    fit = dict(
        path=path, person=person,
        seq_name=_str(d, 'seq_name') or (base.rsplit('_p', 1)[0] if '_p' in base else base),
        version=_str(d, 'version') or '?',
        gender=(_str(d, 'gender') or '').strip().lower()[:1] or None,
        gender_fixed=bool(d['gender_fixed']) if 'gender_fixed' in d.files else None,
        gender_convention=_str(d, 'gender_convention'),
        frame_grid=_str(d, 'frame_grid'),
        frame_stride=int(d['frame_stride']) if 'frame_stride' in d.files else None,
        max_frames=int(d['max_frames']) if 'max_frames' in d.files else None,
        gender_value=float(d['gender_value']) if 'gender_value' in d.files else None,
        valid_mask=np.asarray(d['valid_mask'], dtype=bool) if 'valid_mask' in d.files else None,
        frames=np.asarray(d['frames']).astype(np.int64).reshape(-1),
        joint_err=np.asarray(d['joint_err_mm'], dtype=np.float64).reshape(-1) if 'joint_err_mm' in d.files else None,
        gender_col=g_col,
        pkl_path=_str(d, 'pkl_path'),
    )
    if fit['gender'] not in ('m', 'f'):
        fit['gender'] = None
    return fit


def requested_flags(refit_flags):
    """(gender_convention, frame_grid) that smpl_to_anny.py runs with when these extra flags (run_smpl_to_anny_task.sh's
    REFIT_FLAGS, appended after its own flags) are added: the last occurrence wins, an absent flag = the converter's
    default. Other converter flags are ignored. A bad value raises ValueError."""
    p = argparse.ArgumentParser(add_help=False, prog='REFIT_FLAGS')
    p.add_argument('--gender_convention', default=DEFAULT_GENDER_CONVENTION)
    p.add_argument('--frame_grid', default=DEFAULT_FRAME_GRID)
    a, _ = p.parse_known_args(shlex.split(refit_flags or ''))
    if a.gender_convention not in GENDER_CONVENTIONS:
        raise ValueError(f"--gender_convention {a.gender_convention!r} in REFIT_FLAGS "
                         f"(choices: {sorted(GENDER_CONVENTIONS)})")
    if a.frame_grid not in FRAME_GRIDS:
        raise ValueError(f"--frame_grid {a.frame_grid!r} in REFIT_FLAGS (choices: {list(FRAME_GRIDS)})")
    return a.gender_convention, a.frame_grid


def recorded_flags(path):
    """(gender_convention, frame_grid, version, recorded?) of one .npz. A file without the v2.3 keys (v2.2 or earlier)
    was made with the old behaviour, i.e. the defaults."""
    d = np.load(path, allow_pickle=True)
    rec = 'gender_convention' in d.files and 'frame_grid' in d.files
    return (_str(d, 'gender_convention') or DEFAULT_GENDER_CONVENTION, _str(d, 'frame_grid') or DEFAULT_FRAME_GRID,
            _str(d, 'version') or '?', rec)


def match_flags_main(path, refit_flags):
    """--match_flags: 0 = PATH was made with the flags REFIT_FLAGS requests, 1 = it was not, 2 = bad input."""
    try:
        want = requested_flags(refit_flags)
    except ValueError as e:
        print(f"ERROR: {e}"); return 2
    try:
        g, f, ver, rec = recorded_flags(path)
    except Exception as e:  # noqa
        print(f"ERROR: cannot read {path}: {type(e).__name__}: {e}"); return 2
    how = 'recorded' if rec else 'not recorded (v2.2 or earlier = the defaults)'
    same = (g, f) == want
    print(f"{path}: version {ver}; gender_convention '{g}', frame_grid '{f}' ({how})")
    print(f"requested by REFIT_FLAGS '{refit_flags}': gender_convention '{want[0]}', frame_grid '{want[1]}'"
          f" -> {'SAME' if same else 'DIFFERENT'}")
    return 0 if same else 1


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'))
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)))


def fmt_ms(s):
    return f"{s['mean']:.1f}/{s['median']:.1f} mm (n={s['n']})"


# ---------------------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------------------
def check_new_fit(fit, g3, valid, args, out):
    """Conventions of one new fit. g3 = 3DPW gender ('m'/'f'/None), valid = validity [T] or None.
    Appends printable lines to out['lines'] and violations to out['violations']."""
    L, V = out['lines'], out['violations']
    tag = f"p{fit['person']}"
    conv, grid = fit['gender_convention'], fit['frame_grid']
    L.append(f"  [{tag}] {os.path.basename(fit['path'])}: version {fit['version']}, {len(fit['frames'])} fitted frames, "
             f"gender_convention {conv}, frame_grid {grid}, frame_stride {fit['frame_stride']}")
    if conv is None or grid is None:
        V.append(f"{tag}: no gender_convention / frame_grid recorded (made by smpl_to_anny.py v2.2 or earlier)")
    if conv is not None and conv != args.expect_gender_convention:
        V.append(f"{tag}: gender_convention {conv!r}, expected {args.expect_gender_convention!r}")
    if grid is not None and grid != args.expect_frame_grid:
        V.append(f"{tag}: frame_grid {grid!r}, expected {args.expect_frame_grid!r}")

    # ---- gender
    g_npz = fit['gender']
    if g3 is not None and g_npz is not None and g3 != g_npz:
        V.append(f"{tag}: gender in the .npz {g_npz!r} != 3DPW gender {g3!r}")
    g = g3 if g3 is not None else g_npz
    gcol = fit['gender_col']
    gmed = float(np.median(gcol)) if gcol is not None and gcol.size else float('nan')
    line = (f"  [{tag}] gender: 3DPW {g!r}, label value median {gmed:.4f}"
            + (f" (min {gcol.min():.4f}, max {gcol.max():.4f})" if gcol is not None and gcol.size else "")
            + (f", saved gender_value {fit['gender_value']:.4f}" if fit['gender_value'] is not None else ""))
    if g in ('m', 'f'):
        want = GENDER_CONVENTIONS[args.expect_gender_convention][g]
        line += f"; expected {want:.1f} under '{args.expect_gender_convention}' (Anny: 0 = male, 1 = female)"
        if fit['gender_fixed'] is False:
            V.append(f"{tag}: gender not fixed (gender_fixed False) although 3DPW says {g!r}")
        if gcol is None or not gcol.size:
            V.append(f"{tag}: no gender column in anny_shape")
        elif np.max(np.abs(gcol - want)) > GENDER_TOL:
            n_bad = int((np.abs(gcol - want) > GENDER_TOL).sum())
            V.append(f"{tag}: gender value {gmed:.4f} on {n_bad}/{gcol.size} frames, expected {want:.1f} for 3DPW {g!r} "
                     f"under '{args.expect_gender_convention}'")
        if fit['gender_value'] is not None and np.isfinite(fit['gender_value']) \
                and abs(fit['gender_value'] - want) > GENDER_TOL:
            V.append(f"{tag}: saved gender_value {fit['gender_value']:.4f}, expected {want:.1f}")
    else:
        line += "; 3DPW gender unknown - not checked"
    L.append(line)

    # ---- frame grid
    fr = fit['frames']
    stride = args.stride if args.stride else (fit['frame_stride'] or 3)
    if len(np.unique(fr)) != len(fr) or np.any(np.diff(fr) <= 0):
        V.append(f"{tag}: fitted frames are not strictly increasing / unique")
    off = fr[fr % stride != 0]
    L.append(f"  [{tag}] frames: first {fr[:6].tolist()}..., {len(off)} of {len(fr)} off the absolute grid "
             f"frame % {stride} == 0")
    if args.expect_frame_grid == 'shared' and len(off):
        V.append(f"{tag}: {len(off)} fitted frames off the shared grid (frame % {stride} != 0), e.g. {off[:5].tolist()}")
    if valid is not None:
        T = len(valid)
        inside = fr[(fr >= 0) & (fr < T)]
        not_valid = [int(f) for f in inside if not valid[f]] + [int(f) for f in fr if f < 0 or f >= T]
        if not_valid:
            V.append(f"{tag}: {len(not_valid)} fitted frames are not valid in 3DPW, e.g. {not_valid[:5]}")
        vidx = np.where(valid)[0]
        if args.expect_frame_grid == 'shared':
            expected = vidx[vidx % stride == 0]
        else:
            expected = vidx[::stride]
        capped = fit['max_frames'] is not None and len(fr) >= fit['max_frames']
        missing = np.setdiff1d(expected, fr)
        frac = len(missing) / max(len(expected), 1)
        L.append(f"  [{tag}] validity: {int(valid.sum())}/{T} frames valid; {len(expected)} expected on the "
                 f"'{args.expect_frame_grid}' grid, {len(missing)} of them not fitted ({frac:.1%})"
                 + (" - max_frames reached, not graded" if capped else ""))
        if not capped and frac > args.max_unfitted_frac:
            V.append(f"{tag}: {len(missing)}/{len(expected)} valid grid frames not fitted ({frac:.1%} > "
                     f"{args.max_unfitted_frac:.1%}), e.g. {missing[:5].tolist()}")
    else:
        L.append(f"  [{tag}] validity unknown (no .pkl, no valid_mask in the .npz): grid completeness not checked")

    # ---- joint error (informational)
    if fit['joint_err'] is not None:
        s = stats(fit['joint_err'])
        warn = ''
        if not np.all(np.isfinite(fit['joint_err'])):
            warn = '  WARN: non-finite errors'
        elif s['median'] > 30:
            warn = '  WARN: median above the converter\'s 30 mm good-fit line'
        L.append(f"  [{tag}] joint error mean/median {fmt_ms(s)}{warn}")


def pair_coverage(fit_a, fit_b, valid_a, valid_b):
    """Fitted-frame overlap of two people of one sequence. 'one_sided' = frames where both are valid in 3DPW but only
    one is fitted, as a fraction of the fitted frames (union) where both are valid. Validity None -> those are None."""
    A, B = set(fit_a['frames'].tolist()), set(fit_b['frames'].tolist())
    union = sorted(A | B)
    res = dict(n_a=len(A), n_b=len(B), n_union=len(union), n_shared=len(A & B),
               n_both_valid=None, n_one_sided=None, frac_one_sided=None, examples=[])
    if valid_a is None or valid_b is None:
        return res
    T = min(len(valid_a), len(valid_b))
    both = [f for f in union if 0 <= f < T and valid_a[f] and valid_b[f]]
    one = [f for f in both if (f in A) != (f in B)]
    res.update(n_both_valid=len(both), n_one_sided=len(one),
               frac_one_sided=(len(one) / len(both)) if both else 0.0, examples=one[:6])
    return res


def fmt_pair(c):
    s = f"shared fitted frames {c['n_shared']} (p_a {c['n_a']}, p_b {c['n_b']}, union {c['n_union']})"
    if c['n_one_sided'] is None:
        return s + "; validity unknown: one-sided frames not measured"
    return (s + f"; both valid in 3DPW but only one fitted: {c['n_one_sided']}/{c['n_both_valid']} "
            f"({c['frac_one_sided']:.1%})" + (f", e.g. frames {c['examples']}" if c['examples'] else ""))


def compare_old_new(old, new, out):
    L = out['lines']
    tag = f"p{new['person']}"
    so = stats(old['joint_err'] if old['joint_err'] is not None else [])
    sn = stats(new['joint_err'] if new['joint_err'] is not None else [])
    common, io, inn = np.intersect1d(old['frames'], new['frames'], return_indices=True)
    L.append(f"  [{tag}] OLD {os.path.basename(old['path'])} (version {old['version']}, "
             f"gender_convention {old['gender_convention'] or 'not recorded (makehuman)'}, "
             f"frame_grid {old['frame_grid'] or 'not recorded (per_person)'}): {len(old['frames'])} frames")
    og = float(np.median(old['gender_col'])) if old['gender_col'] is not None and old['gender_col'].size else float('nan')
    ng = float(np.median(new['gender_col'])) if new['gender_col'] is not None and new['gender_col'].size else float('nan')
    L.append(f"  [{tag}] gender value old -> new: {og:.4f} -> {ng:.4f}")
    if old['joint_err'] is None or new['joint_err'] is None:
        L.append(f"  [{tag}] no joint_err_mm in one of the files: error comparison skipped")
        return
    L.append(f"  [{tag}] joint error mean/median, all fitted frames: old {fmt_ms(so)} -> new {fmt_ms(sn)}")
    if len(common):
        po, pn = stats(old['joint_err'][io]), stats(new['joint_err'][inn])
        d = new['joint_err'][inn] - old['joint_err'][io]
        L.append(f"  [{tag}] joint error mean/median, {len(common)} frames fitted by both: old {fmt_ms(po)} -> new "
                 f"{fmt_ms(pn)}; paired new - old median {float(np.median(d)):+.1f} mm, new lower on "
                 f"{float(np.mean(d < 0)):.0%} of them")
    else:
        L.append(f"  [{tag}] no frame fitted by both runs")


def check_sequence(new_fits, old_fits, seqd, args, title):
    """new_fits / old_fits: {person: fit}. seqd: 3DPW pickle dict or None. -> dict(lines, violations, summary)."""
    out = dict(lines=[f"=== {title}"], violations=[], summary={})
    valid, g3 = {}, {}
    for p, fit in sorted(new_fits.items()):
        v_pkl = valid_mask_from_seq(seqd, p) if seqd is not None and p < len(seqd['poses']) else None
        v_npz = fit['valid_mask']
        if v_pkl is not None and v_npz is not None and (len(v_pkl) != len(v_npz) or np.any(v_pkl != v_npz)):
            out['lines'].append(f"  [p{p}] WARN: valid_mask in the .npz differs from the .pkl recompute "
                                f"({int(v_npz.sum())} vs {int(v_pkl.sum())} valid); using the .pkl")
        valid[p] = v_pkl if v_pkl is not None else v_npz
        g3[p] = read_gender(seqd, p) if seqd is not None else None
    for p in old_fits:
        if p not in valid and seqd is not None and p < len(seqd['poses']):
            valid[p] = valid_mask_from_seq(seqd, p)
    src = 'sequence .pkl' if seqd is not None else ('.npz valid_mask' if any(v is not None for v in valid.values())
                                                     else 'none')
    out['lines'].append(f"  validity source: {src}; expected: gender_convention '{args.expect_gender_convention}', "
                        f"frame_grid '{args.expect_frame_grid}'")
    if seqd is not None:
        n_people = len(seqd['poses'])
        missing = [p for p in range(n_people) if p not in new_fits]
        if missing:
            out['lines'].append(f"  NOTE: the sequence has {n_people} people; no new .npz for {['p%d' % p for p in missing]}")
    for p, fit in sorted(new_fits.items()):
        check_new_fit(fit, g3.get(p), valid.get(p), args, out)
        if p in old_fits:
            compare_old_new(old_fits[p], fit, out)
    for a, b in itertools.combinations(sorted(new_fits), 2):
        c = pair_coverage(new_fits[a], new_fits[b], valid.get(a), valid.get(b))
        out['lines'].append(f"  [p{a}+p{b}] NEW: " + fmt_pair(c).replace('p_a', f'p{a}').replace('p_b', f'p{b}'))
        out['summary']['new_one_sided'] = (c['n_one_sided'], c['n_both_valid'])
        if c['frac_one_sided'] is not None and c['frac_one_sided'] > args.max_one_sided_frac:
            out['violations'].append(f"p{a}+p{b}: {c['n_one_sided']}/{c['n_both_valid']} frames with both actors "
                                     f"valid but only one fitted ({c['frac_one_sided']:.1%} > "
                                     f"{args.max_one_sided_frac:.1%})")
        if a in old_fits and b in old_fits:
            co = pair_coverage(old_fits[a], old_fits[b], valid.get(a), valid.get(b))
            out['lines'].append(f"  [p{a}+p{b}] OLD: " + fmt_pair(co).replace('p_a', f'p{a}').replace('p_b', f'p{b}'))
            out['summary']['old_one_sided'] = (co['n_one_sided'], co['n_both_valid'])
    for v in out['violations']:
        out['lines'].append(f"  VIOLATION: {v}")
    out['lines'].append(f"  -> {'FAIL' if out['violations'] else 'PASS'} ({len(out['violations'])} violation(s))")
    return out


# ---------------------------------------------------------------------------------------------------------------
# input resolution
# ---------------------------------------------------------------------------------------------------------------
def by_person(fits, what):
    d = {}
    for f in fits:
        if f['person'] in d:
            raise ValueError(f"two {what} files for person {f['person']}: {d[f['person']]['path']} and {f['path']}")
        d[f['person']] = f
    return d


def find_pkl(args, split, seq, fits):
    if args.pkl:
        return args.pkl
    if args.seq_root and split and seq:
        p = os.path.join(args.seq_root, split, f"{seq}.pkl")
        if os.path.isfile(p):
            return p
    for f in fits:
        if f.get('pkl_path') and os.path.isfile(f['pkl_path']):
            return f['pkl_path']
    return None


def run_one(new_list, old_list, args, split=None, seq=None):
    new = by_person(new_list, 'new')
    old = by_person(old_list, 'old')
    names = {f['seq_name'] for f in new_list + old_list}
    if len(names) > 1:
        raise ValueError(f"files from different sequences: {sorted(names)}")
    seq = seq or (names.pop() if names else '?')
    pkl = find_pkl(args, split, seq, new_list + old_list)
    seqd = load_seq(pkl) if pkl else None
    title = f"{split + '/' if split else ''}{seq}: {len(new)} new, {len(old)} old .npz; 3DPW pkl {pkl or 'not found'}"
    return check_sequence(new, old, seqd, args, title)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('npz', nargs='*', help='new .npz file(s) of one sequence (see USAGE for the old/new rule)')
    ap.add_argument('--old', nargs='+', default=[], help='old .npz file(s) of the same sequence, matched by person')
    ap.add_argument('--new_root', default='', help='root with <split>/<seq>_p<k>.npz of the NEW fits (with --seq)')
    ap.add_argument('--old_root', default='', help='root with <split>/<seq>_p<k>.npz of the OLD fits (optional)')
    ap.add_argument('--split', default='', help="split of --seq; '' = search --splits")
    ap.add_argument('--splits', default='train,validation,test')
    ap.add_argument('--seq', default='', help="sequence name under --new_root, or 'all'")
    ap.add_argument('--pkl', default='', help='3DPW sequenceFiles/<split>/<seq>.pkl (validity + gender)')
    ap.add_argument('--seq_root', default='', help='3DPW sequenceFiles root (looked up as <split>/<seq>.pkl)')
    ap.add_argument('--expect_gender_convention', default='anny', choices=sorted(GENDER_CONVENTIONS))
    ap.add_argument('--expect_frame_grid', default='shared', choices=list(FRAME_GRIDS))
    ap.add_argument('--stride', type=int, default=0, help="grid stride; 0 = the .npz's frame_stride (else 3)")
    ap.add_argument('--max_one_sided_frac', type=float, default=0.01,
                    help='VIOLATION above this fraction of both-valid frames with only one actor fitted')
    ap.add_argument('--max_unfitted_frac', type=float, default=0.01,
                    help='VIOLATION above this fraction of valid grid frames not fitted (NaN fits are discarded)')
    ap.add_argument('--match_flags', default='', metavar='NPZ',
                    help="only compare the gender_convention / frame_grid recorded in NPZ with those --refit_flags "
                         "requests (run_smpl_to_anny_task.sh, before it skips an existing fit); "
                         "exit 0 = same, 1 = different, 2 = unreadable / bad flag")
    ap.add_argument('--refit_flags', default='',
                    help="the extra smpl_to_anny.py flags (REFIT_FLAGS) for --match_flags; write --refit_flags='...'")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.match_flags:
        return match_flags_main(args.match_flags, args.refit_flags)
    jobs = []      # (split, seq, new paths, old paths)
    if args.seq:
        if not args.new_root:
            print("ERROR: --seq needs --new_root"); return 2
        splits = [args.split] if args.split else [s.strip() for s in args.splits.split(',') if s.strip()]
        for sp in splits:
            if args.seq == 'all':
                seqs = sorted({os.path.basename(p)[:-4].rsplit('_p', 1)[0]
                               for p in glob.glob(os.path.join(args.new_root, sp, '*_p*.npz'))})
            else:
                seqs = [args.seq]
            for sq in seqs:
                newp = sorted(glob.glob(os.path.join(args.new_root, sp, f"{sq}_p*.npz")))
                if not newp:
                    continue
                oldp = sorted(glob.glob(os.path.join(args.old_root, sp, f"{sq}_p*.npz"))) if args.old_root else []
                jobs.append((sp, sq, newp, oldp))
        if not jobs:
            print(f"ERROR: no new .npz for --seq {args.seq} under {args.new_root}/{{{','.join(splits)}}}"); return 2
    else:
        if not args.npz:
            print("ERROR: give new .npz file(s) or --seq with --new_root"); return 2
        jobs.append((args.split or None, None, list(args.npz), list(args.old)))

    n_viol, n_seq, rows = 0, 0, []
    for split, seq, newp, oldp in jobs:
        try:
            new_list = [load_fit(p) for p in newp]
            old_list = [load_fit(p) for p in oldp]
        except Exception as e:  # noqa
            print(f"ERROR: cannot read .npz: {type(e).__name__}: {e}"); return 2
        if not oldp and len(new_list) == 2 and new_list[0]['person'] == new_list[1]['person']:
            old_list, new_list = [new_list[0]], [new_list[1]]       # same person twice: first = old, second = new
        try:
            res = run_one(new_list, old_list, args, split, seq)
        except ValueError as e:
            print(f"ERROR: {e}"); return 2
        print("\n".join(res['lines']), flush=True)
        n_viol += len(res['violations'])
        n_seq += 1
        rows.append((f"{split + '/' if split else ''}{seq or new_list[0]['seq_name']}", len(res['violations']),
                     res['summary'].get('new_one_sided'), res['summary'].get('old_one_sided')))
    if n_seq > 1:
        print("\n=== SUMMARY (both valid, only one fitted: new | old)")
        for name, nv, new1, old1 in rows:
            f = lambda c: '-' if c is None or c[0] is None else f"{c[0]}/{c[1]}"   # noqa: E731
            print(f"  {name:<44} {'FAIL' if nv else 'PASS'}  {f(new1):>10} | {f(old1):>10}")
    print(f"\nRESULT: {'FAIL' if n_viol else 'PASS'} - {n_viol} violation(s) in {n_seq} sequence(s); expected "
          f"gender_convention '{args.expect_gender_convention}', frame_grid '{args.expect_frame_grid}'")
    return 1 if n_viol else 0


if __name__ == '__main__':
    sys.exit(main())
