"""
Compile the diagnostic report and the compact comparison panel from whatever
results exist under results/diagnostics/.

Reads (each optional; missing inputs are reported as missing, never guessed):
  manifest.json, gt_decode/summary.json, manifest_trace_{train,holdout}_<tag>/summary.json,
  holdout_<tag>/summary.json, young_<tag>/summary.json, phenotypes/summary.json,
  bodymodel/summary.json, gradients_<tag>/summary.json, crop_camera_<tag>/summary.json,
  overfit_<tag>/summary.json

Writes  results/diagnostics/REPORT.md  and  results/diagnostics/comparison_panel.jpg.

The verdicts are rule based and quote the numbers they rest on. Each failure
(neck, fingers, children) is assigned to one or more origins:
  decoding      body model / pose convention / camera maths produce the wrong mesh for GT parameters
  preprocessing image or K handling between the annotation and the network differs from the model's assumption
  supervision   the training signal cannot teach the behaviour (missing data, zeroed loss, wrong labels)
  optimization  the gradient path is broken or the optimiser cannot fit even a tiny set
  generalization the model fits the training data but not held-out / real images
A "tiny-subset success" (overfit) is never reported as validation performance.

USAGE (login node):  python scripts/diagnostics/diag_report.py --tag anny_s2_shape_v5_00099
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def find(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(x, nd=1):
    return '-' if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='results/diagnostics')
    ap.add_argument('--tag', default='', help='checkpoint tag, e.g. anny_s2_shape_v5_00099 (default: newest found)')
    args = ap.parse_args()
    R = args.root
    tag = args.tag or '*'

    S = dict(
        manifest=load(os.path.join(R, 'manifest.json')),
        gt=load(os.path.join(R, 'gt_decode', 'summary.json')),
        mt_train=load(find(os.path.join(R, f'manifest_trace_train_{tag}', 'summary.json')) or ''),
        mt_hold=load(find(os.path.join(R, f'manifest_trace_holdout_{tag}', 'summary.json')) or ''),
        e1=load(find(os.path.join(R, f'holdout_{tag}', 'summary.json')) or ''),
        young=load(find(os.path.join(R, f'young_{tag}', 'summary.json')) or ''),
        pheno=load(os.path.join(R, 'phenotypes', 'summary.json')),
        body=load(os.path.join(R, 'bodymodel', 'summary.json')),
        grad=load(find(os.path.join(R, f'gradients_{tag}', 'summary.json')) or ''),
        crop=load(find(os.path.join(R, f'crop_camera_{tag}', 'summary.json')) or ''),
        overfit=load(find(os.path.join(R, f'overfit_{tag}', 'summary.json')) or ''),
    )
    have = {k: v is not None for k, v in S.items()}
    L = []
    P = L.append
    P(f"# Diagnostic report\n\nInputs found: " + ", ".join(f"`{k}`" if v else f"~~{k}~~" for k, v in have.items()) + "\n")
    P("Missing inputs are struck through; verdicts that need them say so. Nothing below is inferred from a missing file.\n")

    # ------------------------------------------------------------------ data / manifest
    if S['manifest']:
        c = S['manifest']['counts']
        P("## Sample manifest\n")
        P("| split | adult | child | hands | neck |\n|---|---|---|---|---|")
        for sp in ('train', 'holdout'):
            P(f"| {sp} | {c[sp]['adult']} | {c[sp]['child']} | {c[sp]['hands']} | {c[sp]['neck']} |")
        if c['holdout']['child'] == 0:
            P("\n**The holdout has no children**: every child result below is in-sample (train split) and says what the model can *represent*, not how it generalises.\n")

    # ------------------------------------------------------------------ check 1
    P("## Check 1: GT decode without the network\n")
    if S['gt']:
        bbm = S['gt'].get('joints_inside_bbox_mean')
        P(f"GT joints inside the image: {f(100 * S['gt']['joints_inside_image_mean'])}%; inside the annotation's tight_bbox: "
          + (f"{f(100 * bbm)}% (bbox read as {S['gt']['bbox_formats']}). Worst containment: {S['gt']['worst_bbox_containment'][:3]}."
             if bbm is not None else "not measurable (no tight_bbox in the annotations)."))
        if bbm is None:
            P("\n**Verdict: bbox containment not available; judge decoding from `gt_decode/vis/` (GT mesh on the person) and the "
              f"{f(100 * S['gt']['joints_inside_image_mean'])}% in-image rate.**\n")
        else:
            ok = bbm >= 0.95
            P(f"\n**Verdict: decoding + camera {'consistent with the annotation' if ok else 'NOT consistent with the annotation -> decoding/preprocessing defect; stop here and fix before trusting anything else'}.** "
              "Look at `gt_decode/vis/` to confirm the meshes sit on the people.\n")
    else:
        P("Not run (`diag_gt_decode.py`).\n")

    # ------------------------------------------------------------------ check 2 + 3
    P("## Checks 2 and 3: body model (age sweep, single-joint perturbations)\n")
    if S['body']:
        sweep = g(S['body'], 'age_sweep', 'rows') or g(S['body'], 'body_model_age_sweep')
        if isinstance(sweep, list) and sweep:
            P("| age | skeletal height | neck | head/height |\n|---|---|---|---|")
            for r in sweep:
                P(f"| {r.get('age')} | {f(r.get('skel_height_m') or r.get('height_m'), 3)} | {f(r.get('neck_m'), 4)} | {f(r.get('head_over_height'), 3)} |")
            try:
                h0, h1 = sweep[0]['skel_height_m'], sweep[-1]['skel_height_m']
                P(f"\nHeight changes by {100 * (1 - h0 / h1):.0f}% from age {sweep[0]['age']} to {sweep[-1]['age']}: "
                  f"{'the body model CAN make a child' if h0 < 0.8 * h1 else 'age barely changes the body: child proportions are NOT reachable through age alone'}.")
            except Exception:
                pass
        tests = g(S['body'], 'perturbations', 'tests') or []
        tests = [t for t in tests if isinstance(t, dict)]
        if tests:
            P("\n| bone | axis | affected vertices in expected part | joints that moved |\n|---|---|---|---|")
            for t in tests:
                fr = t.get('frac_affected_in_expected')
                P(f"| {t.get('bone')} | {t.get('axis')} | {f(100 * fr, 0) if fr is not None else '-'}% | {', '.join((t.get('moved_joints') or [])[:6])} |")
            bad = [f"{t.get('bone')}/{t.get('axis')}" for t in tests
                   if t.get('frac_affected_in_expected') is not None and t['frac_affected_in_expected'] < 0.8]
            P(f"\n**Verdict: {'every perturbation moved the intended part' if not bad else 'perturbations with <80% of moved vertices in the intended part: ' + ', '.join(bad) + ' -> decoding (rig/skinning) defect'}.**")
        elif g(S['body'], 'perturbations', 'error'):
            P(f"\nPerturbation check failed: {g(S['body'], 'perturbations', 'error')}")
        hm = g(S['body'], 'helper_mask')
        if hm:
            P(f"\nHelper-bone mask (model.py useful_rotmat) vs neck/hand bones: `{json.dumps(hm)[:500]}`")
        P("")
    else:
        P("Not run (`diag_bodymodel_probe.py`).\n")

    # ------------------------------------------------------------------ check 4
    P("## Check 4: gradients and response of the prediction heads\n")
    if S['grad']:
        keep_keys = ('gt_path', 'images_used', 'mask_helper_joints_train', 'loss_epoch', 'two_d_terms_in_total',
                     'total_grad_norm', 'decshape', 'deccam', 'loss', 'placement', 'warnings', 'problems', 'verdict')
        P("```\n" + json.dumps({k: S['grad'].get(k) for k in keep_keys if k in S['grad']}, indent=1)[:3500] + "\n```")
        dp = S['grad'].get('decpose') or {}
        P(f"\ndecpose: {dp.get('n_zero_grad')} bones with zero gradient; masked set == zero set: {dp.get('zero_set_equals_masked_set')}; "
          f"active-but-zero: {dp.get('active_but_zero')}; masked-but-nonzero: {dp.get('masked_but_nonzero')}.")
        resp = g(S['grad'], 'response') or {}
        if resp.get('shape_summary') or resp.get('pose_summary'):
            P(f"\nResponse test: shape `{json.dumps(resp.get('shape_summary'))[:600]}`; pose `{json.dumps(resp.get('pose_summary'))[:600]}`")
        zero_active = dp.get('unexpected_zero') or dp.get('active_but_zero')
        P(f"\n**Verdict ({S['grad'].get('verdict', 'script verdict missing')}): "
          f"{'active bones with zero gradient: ' + str(zero_active) + ' -> optimisation defect' if zero_active else 'every active bone receives gradient; the masked bones have none by construction'}.** "
          f"Problems flagged by the script: {S['grad'].get('problems') or 'none'}.\n")
    else:
        P("Not run (`diag_gradients.py`).\n")

    # ------------------------------------------------------------------ check 5 / E1
    def variant(s, name='V1_eval_masked'):
        return g(s, 'variants', name) or (list(g(s, 'variants', default={}).values()) or [None])[0]

    P("## Check 5 and E1: prediction vs GT, shape swap\n")
    for key, title in (('mt_hold', 'manifest HOLDOUT (never trained on)'), ('mt_train', 'manifest TRAIN (in-sample)'), ('e1', '100-image holdout (E1)')):
        s = S[key]
        if not s:
            P(f"*{title}*: not run.\n"); continue
        v = variant(s)
        if not v:
            continue
        cam = s.get('camera', {})
        P(f"**{title}** (`{s.get('ckpt', '')}`): recall {f(v.get('recall'))}%, PVE {f(v.get('pve'))}, PA-PVE {f(v.get('pa_pve'))}, "
          f"PVE head+neck {f(v.get('pve_headneck'))} mm; neck length {f(v.get('neck_len_pred_over_gt_pct'))}% of GT, "
          f"neck/height {f(v.get('neck_over_height_pred_over_gt_pct'))}%, head/height {f(v.get('head_over_height_pred_over_gt_pct'))}%.")
        fac = v.get('neck_factorisation_m') or {}
        if fac.get('gt'):
            P(f"Neck factorisation (m): GT {f(fac.get('gt'), 4)} | pred {f(fac.get('pred'), 4)} | GT pose+pred shape {f(fac.get('gt_pose_pred_shape'), 4)} "
              f"| pred pose+GT shape {f(fac.get('pred_pose_gt_shape'), 4)}.")
        rot = v.get('rot_err_deg_by_group') or {}
        if rot:
            P("Rotation error (deg, active bones): " + ", ".join(f"{k} {f(x)}" for k, x in rot.items() if x is not None) + ".")
        gl = v.get('rot_global_deg') or {}
        if gl:
            P(f"Global orientation error: root {f(gl.get('root'))}, head {f(gl.get('head'))}, wrist {f(gl.get('wrist'))} deg "
              f"(local neck/head errors far above the global head error => redundant chain, weak rotation loss there).")
        pl = v.get('placement') or {}
        if pl:
            P(f"Placement (absolute): head {f(pl.get('head_px_err'))} px / {f(pl.get('head_err_mm'))} mm off, root {f(pl.get('root_px_err'))} px, "
              f"predicted head vs GT root {f(pl.get('pred_head_vs_gt_root_px'))} px (GT head-root distance {f(pl.get('gt_head_vs_gt_root_px'))} px), "
              f"vertical shift {f(pl.get('vertical_shift_px'))} px, root depth error {f(pl.get('root_depth_err_m'), 2)} m. "
              f"train.py supervises transl/dist/heatmap at the ROOT while the model anchors the HEAD (model.py:430,442; train.py:214-222).")
        bias = v.get('shape_bias') or {}
        if bias:
            P("Shape bias (pred-GT): " + ", ".join(f"{k} {bias[k]:+.3f}" for k in ('age', 'height', 'proportions') if bias.get(k) is not None) + ".")
        P(f"Camera checks: principal point centred on {f(100 * cam.get('principal_point_centred_frac', 0), 0)}% of images, "
          f"K_fov == dataset K on {f(100 * cam.get('k_fov_within_1px_frac', 0), 0)}%, GT joints inside image {f(100 * (cam.get('gt_j2d_inside_image_frac') or 0))}%.")
        pt = g(s, 'per_tag', 'V1_eval_masked')
        if pt:
            P("\n| category | n | PVE | PVE(pred pose + GT shape) | shape-swap gain | age bias | neck % | rot_fingers |\n|---|---|---|---|---|---|---|---|")
            for t, d in pt.items():
                P(f"| {t} | {d['n']} | {f(d.get('pve'))} | {f(d.get('pve_predpose_gtshape'))} | {f(d.get('shape_swap_gain_mm'))} | "
                  f"{(d.get('pheno_err_age') if d.get('pheno_err_age') is not None else float('nan')):+.3f} | {f((d.get('neck_ratio') or 0) * 100)} | {f(d.get('rot_fingers'))} |")
        pr = s.get('paired') or {}
        if pr.get('n_common'):
            dv = pr.get('delta_vs_reference') or {}
            P(f"\nPaired deltas vs V1 on {pr['n_common']} common people: " + "; ".join(
                f"{vn}: PVE {f(d.get('pve'))}, neck {f(d.get('neck_ratio'), 3)}, fingers {f(d.get('rot_fingers'))}" for vn, d in dv.items() if vn != pr.get('reference')) + ".")
        P("")

    # ------------------------------------------------------------------ data
    P("## Data: GT age distribution (E2)\n")
    if S['pheno']:
        fb = S['pheno'].get('age_frac_below', {})
        P(f"Training people with age < 0.2 (~child): {f(100 * fb.get('0.2', 0), 2)}%, < 0.3: {f(100 * fb.get('0.3', 0), 2)}%, < 0.5: {f(100 * fb.get('0.5', 0), 1)}%. "
          f"Holdout: {json.dumps(S['pheno'].get('holdout_age_frac_below', {}))}. 3DPW labels have age fixed at 0.55.\n")
    else:
        P("Not run (`diag_phenotypes.py`).\n")

    # ------------------------------------------------------------------ check 6
    P("## Check 6: capped overfit of the manifest train subset\n")
    if S['overfit']:
        r = S['overfit'].get('results', {})
        P("| set | PVE before | PVE after |\n|---|---|---|")
        for sp in ('train', 'holdout'):
            P(f"| {sp} | {f(g(r, f'before_{sp}', 'pve'))} | {f(g(r, f'after_{sp}', 'pve'))} |")
        td, hd = S['overfit'].get('train_pve_drop_pct'), S['overfit'].get('holdout_pve_drop_pct')
        P(f"\nTrain PVE drop {f(td)}%, holdout PVE drop {f(hd)}%. **This is an optimisation check only; the train number is not validation performance.**")
        if td is not None:
            P(f"\n**Verdict: {'the optimisation path can fit the subset' if td > 30 else 'the model cannot fit 40 images in the capped run -> optimisation (or decoding) defect, or the run is too short: see overfit_*/train.log'}; "
              f"{'the holdout barely moves, as expected (generalisation, not optimisation, limits it)' if hd is not None and hd < td / 2 else 'the holdout moved too: check that the two subsets are disjoint'}.**\n")
    else:
        P("Not run (`slurm/diagnostics/diag_overfit.sh`).\n")

    # ------------------------------------------------------------------ crop camera
    if S['crop']:
        ch = S['crop'].get('mean_abs_change', {})
        P("## E4: crop camera on real photos\n")
        P(f"Mean |change| demo K -> corrected K: age {f(ch.get('pheno_age'), 3)}, height pheno {f(ch.get('pheno_height'), 3)}, skeletal height {f(ch.get('skel_height'), 3)} m, depth {f(ch.get('depth'), 2)} m "
          f"over {S['crop'].get('n_people')} people. {'K changes the predicted body -> the crop-camera bug is a preprocessing cause for real photos.' if (ch.get('pheno_age') or 0) > 0.05 or (ch.get('pheno_height') or 0) > 0.05 else 'K barely changes the body: placement-only.'}\n")

    # ------------------------------------------------------------------ verdicts
    P("## Where each failure originates\n")
    _bbm = (S['gt'] or {}).get('joints_inside_bbox_mean')
    gt_ok = _bbm is None or _bbm >= 0.95           # None = not measurable, not a defect
    _tests = [t for t in (g(S['body'], 'perturbations', 'tests') or []) if isinstance(t, dict)]
    body_ok = all(t.get('frac_affected_in_expected') is None or t['frac_affected_in_expected'] >= 0.8 for t in _tests)
    ref = variant(S['mt_hold'] or S['e1'] or S['mt_train'] or {})
    fac = (ref or {}).get('neck_factorisation_m') or {}
    rot = (ref or {}).get('rot_err_deg_by_group') or {}
    bias = (ref or {}).get('shape_bias') or {}
    child_frac = g(S['pheno'], 'age_frac_below', '0.2')
    td = (S['overfit'] or {}).get('train_pve_drop_pct')

    def origin_line(name, lines):
        P(f"### {name}\n")
        for ln in lines:
            P(f"- {ln}")
        P("")

    neck = []
    if not gt_ok or not body_ok:
        neck.append("**decoding**: GT decode or joint perturbation failed (see checks 1/3).")
    if fac.get('gt') and fac.get('pred'):
        over = 100 * (fac['pred'] / fac['gt'] - 1)
        shp = 100 * (fac['gt_pose_pred_shape'] / fac['gt'] - 1) if fac.get('gt_pose_pred_shape') else None
        pos = 100 * (fac['pred_pose_gt_shape'] / fac['gt'] - 1) if fac.get('pred_pose_gt_shape') else None
        neck.append(f"predicted neck is {over:+.1f}% vs GT; shape alone explains {f(shp)}%, rotations alone {f(pos)}%.")
        if shp is not None and pos is not None:
            if abs(shp) > abs(pos) and abs(shp) > 2:
                neck.append("**supervision** (shape head): the age/height/proportions regression lengthens the neck; shape loss weighting / targets are the lever.")
            elif abs(pos) > 2:
                neck.append("**supervision** (neck rotations): the 4 neck/head bones carry ~2.5% of the rotation loss; neck-weighted loss or more neck supervision is the lever.")
            else:
                neck.append("skeleton matches GT: a long-looking neck is a **surface/rendering** effect (LBS skinning or demo placement), not a parameter error.")
    if rot.get('neck') is not None and rot.get('spine') is not None and rot['neck'] > 1.5 * rot['spine']:
        neck.append(f"neck rotation error {f(rot['neck'])} deg vs spine {f(rot['spine'])} deg: the neck IS harder for the network (optimisation/supervision).")
    if not neck:
        neck.append("insufficient evidence: run checks 1, 3 and the manifest trace.")
    origin_line("Neck", neck)

    fing = []
    if rot.get('fingers') is not None:
        others = [v for k, v in rot.items() if k not in ('fingers', 'wrist') and v is not None]
        fing.append(f"finger rotation error {f(rot['fingers'])} deg vs {f(float(np.mean(others)) if others else None)} deg for other groups.")
        if others and rot['fingers'] > 1.5 * float(np.mean(others)):
            fing.append("**supervision/optimisation**: fingers are undertrained (few pixels at 672 px, 38 bones); `--relax_hands 1` in the demos hides this by rendering rest hands.")
    fp = [t for t in _tests if 'finger' in str(t.get('bone', '')).lower() or 'thumb' in str(t.get('bone', '')).lower()]
    if fp and all(t.get('frac_affected_in_expected') is not None and t['frac_affected_in_expected'] >= 0.8 for t in fp):
        fing.append("decoding is fine: perturbing a finger bone moves only that finger (check 3).")
    if not fing:
        fing.append("insufficient evidence: run the manifest trace and check 3.")
    origin_line("Fingers", fing)

    kid = []
    if child_frac is not None:
        kid.append(f"{f(100 * child_frac, 2)}% of training people have age < 0.2 (~11 y).")
        if child_frac < 0.01:
            kid.append("**supervision**: almost no child labels; with an L1 shape loss toward a 0.5 prior the shape head regresses children to adults. 3DPW labels (age fixed 0.55) add none.")
    if bias.get('age') is not None:
        kid.append(f"age bias on the evaluated set {bias['age']:+.3f} (pred - GT).")
    pt_tr = g(S['mt_train'], 'per_tag', 'V1_eval_masked', 'child')
    if pt_tr:
        kid.append(f"in-sample children (n={pt_tr['n']}): PVE {f(pt_tr.get('pve'))} -> {f(pt_tr.get('pve_predpose_gtshape'))} mm with GT shape (gain {f(pt_tr.get('shape_swap_gain_mm'))} mm), age bias {pt_tr.get('pheno_err_age'):+.3f}.")
        if (pt_tr.get('shape_swap_gain_mm') or 0) > 10:
            kid.append("**the shape head is the failure** for children: substituting GT shape removes most of the error even in-sample -> supervision (and optimisation if the overfit check also fails to move age).")
    if S['crop'] and ((S['crop'].get('mean_abs_change', {}).get('pheno_age') or 0) > 0.05):
        kid.append("**preprocessing** on real photos: the crop camera changes predicted age/height (E4).")
    if td is not None and td < 30:
        kid.append("**optimisation**: the capped overfit could not fit 40 images.")
    if not kid:
        kid.append("insufficient evidence: run E2, the manifest trace (train split) and the overfit check.")
    origin_line("Children", kid)

    P("## Caveats\n")
    P("- Tiny-subset overfit success is an optimisation check, never validation performance.")
    P("- All vertex comparisons are between Anny meshes of the same topology and vertex order (GT and predictions share one body model); no cross-topology comparison is made anywhere in this suite.")
    P("- Child results come from the training split when the holdout has no children; they test representation, not generalisation.")
    P("- `V1_eval_masked` is the reference variant; `train.py evaluate()` (V0) does not mask helper bones and the demos do.")

    os.makedirs(R, exist_ok=True)
    with open(os.path.join(R, 'REPORT.md'), 'w') as fh:
        fh.write("\n".join(L))
    print("\n".join(L))

    # ------------------------------------------------------------------ panel
    try:
        from PIL import Image
        picks = []
        for pat, n in ((os.path.join(R, 'gt_decode', 'vis', '*.jpg'), 2),
                       (os.path.join(R, f'manifest_trace_train_{tag}', 'vis', '*.jpg'), 2),
                       (os.path.join(R, f'manifest_trace_holdout_{tag}', 'vis', '*.jpg'), 2),
                       (os.path.join(R, 'bodymodel', 'vis', 'age_sweep*.jpg'), 1),
                       (os.path.join(R, 'bodymodel', 'vis', 'perturb_neck*.jpg'), 1),
                       (os.path.join(R, 'bodymodel', 'vis', 'perturb_*finger*.jpg'), 1),
                       (os.path.join(R, f'crop_camera_{tag}', 'vis', '*.jpg'), 1)):
            picks += sorted(glob.glob(pat))[:n]
        if picks:
            Wt = 1600
            ims = []
            for pth in picks:
                im = Image.open(pth).convert('RGB')
                im = im.resize((Wt, max(1, int(im.height * Wt / im.width))))
                ims.append(im)
            H = sum(i.height for i in ims) + 6 * (len(ims) - 1)
            canvas = Image.new('RGB', (Wt, H), (255, 255, 255))
            yy = 0
            for im in ims:
                canvas.paste(im, (0, yy)); yy += im.height + 6
            canvas.save(os.path.join(R, 'comparison_panel.jpg'), quality=85)
            print(f"\nwrote {R}/comparison_panel.jpg from {len(ims)} panels and {R}/REPORT.md")
        else:
            print(f"\nwrote {R}/REPORT.md (no panels found for the montage)")
    except Exception as e:  # noqa
        print(f"(panel skipped: {e})")


if __name__ == '__main__':
    main()
