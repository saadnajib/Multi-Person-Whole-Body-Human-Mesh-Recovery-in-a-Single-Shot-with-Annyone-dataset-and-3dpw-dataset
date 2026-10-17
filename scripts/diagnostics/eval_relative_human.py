"""
Relative Human (RH) evaluation: predicted Anny age and gender on real photos (docs/handover.md 3.4, section 5 item 12.1).

What it measures
  Per RH age group (adult / teen / kid / baby): how many labelled people are matched by a detection (recall), the mean
  and median predicted Anny age, the fraction predicted "child-shaped" (age < --child_thr), the child-vs-adult AUC of
  predicted age (kid + baby positives, adult negatives; lower age should mean child, so the AUC is computed on -age),
  gender accuracy (pred < 0.5 = male, RH 0 = male, same direction as Anny) and the mean height / proportions phenotypes.

Inference (the same path as demo.py's plain whole-image mode):
  demo.open_image (aspect-preserving resize to img_size, zero pad, ImageNet normalisation; re-implemented here so this
  script does not import demo.py and its renderer) -> demo.get_camera_parameters (fov --fov, principal point at the
  centre) -> model(x, is_training=False, K, det_thresh, nms_kernel_size) under autocast, as demo.forward_model.
  The model is built by diag_common.load_model (bone_fix semantics: '' = the checkpoint's own fix, 'none' = off,
  path = that JSON). demo.filter_humans is NOT applied: matching is one-to-one, so duplicates are harmless.

Detection boxes
  model.py's per-person dict has no 2D box and no 'v2d'; the box is the min/max of the projected vertices
  (diag_common.project(v3d, K), the same projection as j2d) in the padded square, mapped back to original image pixels.
  This covers the whole body, so it is compared with RH 'bbox_wb' (whole-body box) when present, else 'bbox' (visible
  part). --box_source joints uses the j2d box instead. Both are cached.

RH boxes: [x1, y1, x2, y2] in image pixels (RH demo.py draws them with cv2.rectangle(box[:2], box[2:])).

Cache: <out>/pred/<image>.npz per image (boxes_verts, boxes_joints, shape [N,11], scores, phenotype labels), so
  --score_only re-scores with another --iou / --child_thr / --box_source on CPU.

  python scripts/diagnostics/eval_relative_human.py --model <ckpt> --split val --out results/diagnostics/rh_val_x
  python scripts/diagnostics/eval_relative_human.py --score_only --out results/diagnostics/rh_val_x --iou 0.5
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

AGE_NAMES = {0: 'adult', 1: 'teen', 2: 'kid', 3: 'baby', -1: 'unknown'}
GENDER_NAMES = {0: 'male', 1: 'female', -1: 'unknown'}
EXPECTED_PHEN = ['gender', 'age', 'muscle', 'weight', 'height', 'proportions',
                 'cupsize', 'firmness', 'african', 'asian', 'caucasian']


def log(msg=''):
    print(f"[rh] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def load_annots(rh_root, split):
    path = os.path.join(rh_root, f'{split}_annots.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RH annotations not found: {path}")
    z = np.load(path, allow_pickle=True)
    if list(z.files) != ['annots']:
        raise KeyError(f"{path}: expected exactly one key 'annots', got {z.files}")
    annots = z['annots'][()]
    if not isinstance(annots, dict) or not annots:
        raise TypeError(f"{path}: 'annots' is not a non-empty dict (got {type(annots)})")
    for name, people in annots.items():
        if not isinstance(people, (list, tuple)):
            raise TypeError(f"{path}: annots[{name}] is {type(people)}, expected a list of person dicts")
        for p in people:
            for k in ('bbox', 'age', 'gender'):
                if k not in p:
                    raise KeyError(f"{path}: annots[{name}] person lacks '{k}' (keys {sorted(p)})")
            if int(p['age']) not in AGE_NAMES or int(p['gender']) not in GENDER_NAMES:
                raise ValueError(f"{path}: annots[{name}] unexpected age/gender {p['age']}/{p['gender']}")
            if np.asarray(p['bbox']).reshape(-1).shape[0] != 4:
                raise ValueError(f"{path}: annots[{name}] bbox is not 4 numbers: {p['bbox']}")
    return annots


def gt_box(p):
    """bbox_wb (whole body) if present, else bbox (visible part); returns (box[4], source)."""
    if p.get('bbox_wb', None) is not None:
        b = np.asarray(p['bbox_wb'], dtype=np.float64).reshape(-1)
        if b.shape[0] != 4:
            raise ValueError(f"bbox_wb is not 4 numbers: {p['bbox_wb']}")
        return b, 'wb'
    return np.asarray(p['bbox'], dtype=np.float64).reshape(-1), 'vis'


# --------------------------------------------------------------------------- #
# matching / metrics (pure numpy, tested on CPU)
# --------------------------------------------------------------------------- #
def box_iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4); b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(gt_boxes, det_boxes, thr):
    """One-to-one greedy by descending IoU. Returns list (len gt) of (det index or -1, iou)."""
    iou = box_iou(gt_boxes, det_boxes)
    out = [(-1, 0.0)] * len(iou)
    if iou.size == 0:
        return out
    used_g, used_d = set(), set()
    order = np.argsort(-iou, axis=None, kind='stable')
    for flat in order:
        g, d = divmod(int(flat), iou.shape[1])
        v = iou[g, d]
        if v < thr:
            break
        if g in used_g or d in used_d:
            continue
        used_g.add(g); used_d.add(d); out[g] = (d, float(v))
    return out


def auc_rank(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), by average ranks (Mann-Whitney U). None if a side is empty."""
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):                       # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def run_inference(args, annots, names, pred_dir):
    import torch
    from PIL import Image, ImageOps
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import diag_common as dc
    from utils import normalize_rgb, get_focalLength_from_fieldOfView

    device = torch.device(args.device)
    ckpt = dc.resolve_ckpt(args.model)
    model, _ = dc.load_model(ckpt, device, bone_fix=args.bone_fix)
    dc.set_inference_toggles(model, mask_helper_joints=True)     # demo.py default --mask_helper_joints 1
    phen = list(model.body_model.phenotype_labels)
    if phen != EXPECTED_PHEN:
        raise RuntimeError(f"model phenotype labels {phen} != expected {EXPECTED_PHEN}")
    S = int(args.img_size or model.img_size)
    if S != model.img_size:
        log(f"WARNING: --img_size {S} differs from the model's {model.img_size}")
    K = torch.eye(3)
    f = get_focalLength_from_fieldOfView(fov=args.fov, img_size=S)
    K[0, 0] = K[1, 1] = f; K[0, 2] = K[1, 2] = S // 2           # demo.get_camera_parameters
    K = K.unsqueeze(0).to(device)
    meta = dict(ckpt=ckpt, bone_fix=args.bone_fix, img_size=S, fov=args.fov, det_thresh=args.det_thresh,
                nms_kernel_size=args.nms_kernel_size)
    n_new = 0
    for ii, name in enumerate(names):
        cpath = os.path.join(pred_dir, name.replace('/', '__') + '.npz')
        if os.path.isfile(cpath) and not args.overwrite:
            continue
        ipath = os.path.join(args.rh_root, 'images', name)
        if not os.path.isfile(ipath):
            raise FileNotFoundError(f"image listed in {args.split}_annots.npz not found: {ipath}")
        img = Image.open(ipath).convert('RGB')                     # demo.open_image (no exif transpose there)
        W, H = img.size
        img_c = ImageOps.contain(img, (S, S))
        w, h = img_c.size
        img_p = ImageOps.pad(img_c, size=(S, S))                   # centred, black padding
        offx, offy = (S - w) // 2, (S - h) // 2                    # ImageOps.pad pastes at ((S-w)//2, (S-h)//2)
        sx, sy = W / float(w), H / float(h)
        x = torch.from_numpy(normalize_rgb(np.asarray(img_p))).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            humans = model(x, is_training=False, nms_kernel_size=int(args.nms_kernel_size),
                           det_thresh=args.det_thresh, K=K)
        bv, bj, sh, sc = [], [], [], []
        for h_ in humans:
            for k in ('shape', 'v3d', 'j2d', 'scores'):
                if k not in h_:
                    raise KeyError(f"model output lacks '{k}' (keys {sorted(h_)})")
            s = h_['shape'].detach().float().reshape(-1).cpu().numpy()
            if s.shape[0] != 11 or not np.isfinite(s).all():
                raise RuntimeError(f"{name}: bad shape prediction {s}")
            v2d = dc.project(h_['v3d'].detach().float().unsqueeze(0), K.float())[0].cpu().numpy()
            j2d = h_['j2d'].detach().float().cpu().numpy()
            boxes = []
            for p2 in (v2d, j2d):
                p2 = p2[np.isfinite(p2).all(1)]
                if len(p2) == 0:
                    boxes.append([np.nan] * 4); continue
                x1, y1 = p2.min(0); x2, y2 = p2.max(0)
                boxes.append([(x1 - offx) * sx, (y1 - offy) * sy, (x2 - offx) * sx, (y2 - offy) * sy])
            bv.append(boxes[0]); bj.append(boxes[1]); sh.append(s); sc.append(float(h_['scores']))
        np.savez(cpath, image=name, W=W, H=H,
                 boxes_verts=np.asarray(bv, dtype=np.float64).reshape(-1, 4),
                 boxes_joints=np.asarray(bj, dtype=np.float64).reshape(-1, 4),
                 shape=np.asarray(sh, dtype=np.float64).reshape(-1, 11),
                 scores=np.asarray(sc, dtype=np.float64), phenotypes=np.asarray(phen), meta=np.asarray(repr(meta)))
        n_new += 1
        if ii % 50 == 0:
            log(f"{ii + 1}/{len(names)} {name}: {W}x{H}, {len(humans)} detections, {len(annots[name])} RH people")
    log(f"inference done: {n_new} new images cached in {pred_dir}")


# --------------------------------------------------------------------------- #
# scoring (CPU, from the cache)
# --------------------------------------------------------------------------- #
def load_cache(pred_dir, name):
    cpath = os.path.join(pred_dir, name.replace('/', '__') + '.npz')
    if not os.path.isfile(cpath):
        return None
    z = np.load(cpath, allow_pickle=False)
    for k in ('boxes_verts', 'boxes_joints', 'shape', 'scores', 'phenotypes'):
        if k not in z.files:
            raise KeyError(f"{cpath}: cache lacks '{k}' (has {z.files})")
    if list(z['phenotypes']) != EXPECTED_PHEN:
        raise RuntimeError(f"{cpath}: phenotypes {list(z['phenotypes'])} != {EXPECTED_PHEN}")
    return z


def score(args, annots, names, pred_dir):
    PI = {n: i for i, n in enumerate(EXPECTED_PHEN)}
    rows, n_missing_cache = [], 0
    box_key = 'boxes_verts' if args.box_source == 'verts' else 'boxes_joints'
    for name in names:
        z = load_cache(pred_dir, name)
        if z is None:
            n_missing_cache += 1; continue
        people = annots[name]
        gts = [gt_box(p) for p in people]
        dets = z[box_key]
        ok = np.isfinite(dets).all(1)
        det_idx = np.nonzero(ok)[0]
        match = greedy_match(np.array([g[0] for g in gts]).reshape(-1, 4), dets[ok], args.iou)
        for pi, (p, (d, iou)) in enumerate(zip(people, match)):
            r = dict(image=name, rh_index=pi, age_group=AGE_NAMES[int(p['age'])],
                     gender=GENDER_NAMES[int(p['gender'])], gt_box_src=gts[pi][1],
                     gt_x1=gts[pi][0][0], gt_y1=gts[pi][0][1], gt_x2=gts[pi][0][2], gt_y2=gts[pi][0][3],
                     depth_id=int(p.get('depth_id', -1)), matched=int(d >= 0), iou=iou)
            if d >= 0:
                di = int(det_idx[d])
                r['det_index'] = di; r['det_score'] = float(z['scores'][di])
                for k, n in enumerate(EXPECTED_PHEN):
                    r[f'pred_{n}'] = float(z['shape'][di, k])
                for k, c in enumerate(('x1', 'y1', 'x2', 'y2')):
                    r[f'pred_{c}'] = float(dets[di, k])
            rows.append(r)
    if n_missing_cache == len(names):
        raise RuntimeError(f"no cached predictions in {pred_dir} for any of the {len(names)} images")

    lines = []
    L = lines.append
    L(f"split={args.split} images={len(names)} (cache missing for {n_missing_cache}) iou={args.iou} "
      f"child_thr={args.child_thr} box_source={args.box_source} (det box vs RH bbox_wb if present else bbox)")
    L(f"RH people: {len(rows)}; gt box from bbox_wb: {sum(r['gt_box_src'] == 'wb' for r in rows)}")
    med = lambda a: float(np.median(a)) if len(a) else float('nan')
    mean = lambda a: float(np.mean(a)) if len(a) else float('nan')
    L(f"{'group':8s} {'n_lab':>6s} {'n_match':>7s} {'recall':>6s} {'age_mean':>8s} {'age_med':>7s} "
      f"{'<thr':>6s} {'height':>6s} {'propor':>6s} {'g_acc':>6s} {'n_g':>5s}")
    ages = {}
    for g in ('adult', 'teen', 'kid', 'baby', 'unknown'):
        R = [r for r in rows if r['age_group'] == g]
        M = [r for r in R if r['matched']]
        a = np.array([r['pred_age'] for r in M]); ages[g] = a
        G = [r for r in M if r['gender'] != 'unknown']
        gacc = mean([(r['pred_gender'] >= 0.5) == (r['gender'] == 'female') for r in G])
        L(f"{g:8s} {len(R):6d} {len(M):7d} {len(M) / max(len(R), 1):6.3f} {mean(a):8.3f} {med(a):7.3f} "
          f"{mean(a < args.child_thr):6.3f} {mean([r['pred_height'] for r in M]):6.3f} "
          f"{mean([r['pred_proportions'] for r in M]):6.3f} {gacc:6.3f} {len(G):5d}")
    pos = np.concatenate([ages['kid'], ages['baby']])
    auc = auc_rank(-pos, -ages['adult'])      # child = positive, lower predicted age = more child-like
    L(f"child-vs-adult AUC (kid+baby positive, adult negative, score = -pred_age): "
      f"{'n/a' if auc is None else f'{auc:.3f}'} (n_pos={len(pos)}, n_neg={len(ages['adult'])}; matched people only)")
    G = [r for r in rows if r['matched'] and r['gender'] != 'unknown']
    gacc = mean([(r['pred_gender'] >= 0.5) == (r['gender'] == 'female') for r in G])
    L(f"gender accuracy (pred < 0.5 = male), all matched with known gender: {gacc:.3f} (n={len(G)}); "
      f"pred gender mean male {mean([r['pred_gender'] for r in G if r['gender'] == 'male']):.3f} "
      f"female {mean([r['pred_gender'] for r in G if r['gender'] == 'female']):.3f}")
    allm = [r for r in rows if r['matched']]
    L(f"overall recall {len(allm) / max(len(rows), 1):.3f} ({len(allm)}/{len(rows)})")
    L("READ: working model = kid/baby mean age clearly below adult and AUC well above 0.5; read recall next to it "
      "(small people are missed at 672 px, so the AUC covers detected people only)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    cols = ['image', 'rh_index', 'age_group', 'gender', 'depth_id', 'gt_box_src', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2',
            'matched', 'iou', 'det_index', 'det_score'] + [f'pred_{n}' for n in EXPECTED_PHEN] + \
           ['pred_x1', 'pred_y1', 'pred_x2', 'pred_y2']
    with open(os.path.join(args.out, 'per_person.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval='')
        w.writeheader(); w.writerows(rows)
    for l in lines:
        log(l)
    log(f"wrote {os.path.join(args.out, 'summary.txt')} and per_person.csv")
    return dict(rows=rows, auc=auc, ages=ages, gender_acc=gacc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rh_root', default='/netscratch/najib/data/Relative_Human/RelativeHuman')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--model', default='', help='checkpoint (path, models/multiHMR name, or run:epoch)')
    ap.add_argument('--bone_fix', default='', help="diag_common.load_model: '' = checkpoint's own, 'none', or JSON")
    ap.add_argument('--n', type=int, default=0, help='first N images (sorted names); 0 = all')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iou', type=float, default=0.3)
    ap.add_argument('--child_thr', type=float, default=0.45)
    ap.add_argument('--img_size', type=int, default=0, help="0 = the model's img_size")
    ap.add_argument('--fov', type=float, default=55.0, help='slurm/demo/run_demo.sh --fov 55')
    ap.add_argument('--det_thresh', type=float, default=0.3, help='demo.py default')
    ap.add_argument('--nms_kernel_size', type=int, default=3, help='demo.py default')
    ap.add_argument('--box_source', default='verts', choices=['verts', 'joints'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--score_only', action='store_true', help='re-score from <out>/pred, no model')
    ap.add_argument('--overwrite', action='store_true', help='re-run images already cached')
    args = ap.parse_args(argv)

    annots = load_annots(args.rh_root, args.split)
    names = sorted(annots)
    if args.n > 0:
        names = names[:args.n]
    log(f"{args.split}: {len(annots)} images in annots, evaluating {len(names)}")
    pred_dir = os.path.join(args.out, 'pred')
    os.makedirs(pred_dir, exist_ok=True)
    if not args.score_only:
        if not args.model:
            raise SystemExit("--model is required unless --score_only")
        run_inference(args, annots, names, pred_dir)
    return score(args, annots, names, pred_dir)


if __name__ == '__main__':
    main()
