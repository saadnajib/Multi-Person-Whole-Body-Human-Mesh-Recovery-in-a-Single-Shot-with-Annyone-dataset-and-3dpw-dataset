#!/usr/bin/env python
"""
check_3dpw_conversion.py -- fresh, independent evidence for the 3DPW (SMPL) -> Anny labels.

WHAT IS COMPARED
For a fixed, seeded sample of label images from EVERY converted 3DPW sequence of every split that has an index
(threedpw_anny_full/labels/index_<split>.txt, the files datasets/threedpw_anny.py loads), every labelled person
is compared in the CAMERA frame with the ORIGINAL 3DPW ground truth, read again from
data/3DPW/sequenceFiles/<split>/<seq>.pkl:
    SMPL joints  3DPW 'jointPositions' (world) -> camera with cam_poses      (the converter's own target)
    SMPL mesh    gendered SMPL(poses, betas, trans) -> camera                (the mesh 3DPW benchmarks use)
    OpenPose 2D  3DPW 'poses2d'                                               (independent of SMPL and of the fit)
    Anny         Anny(anny_pose, anny_shape) of the label pkl, decoded as train.py prepare_gt does it
Measured per person, then per split and overall (mean / median / p95):
  1. 3D joint error Anny vs SMPL for the 24 mapped pairs, all and body only (no hand joints), for
       'fit'  = the points the converter optimised (mapped bone head + its virtual offset, from the .npz), and
       'head' = the bare Anny bone heads (what train.py uses as GT joints; torso rows include the rig gap);
     plus the error saved in the label (joint_err_mm) against the recomputed one (decode / provenance check).
  2. surface distance SMPL mesh <-> Anny skin, both directions: nearest point on the other surface (vertices +
     a fixed area-weighted point sample, scipy cKDTree), per SMPL body region.
  3. 2D: Anny joints projected with the 3DPW intrinsics vs OpenPose poses2d (13 points, confidence > --op_conf),
     side by side with the raw 3DPW joints (the adapter check measured 8.88 px mean for those); graded in mm at the
     joint depth (pixels x depth / focal, 'mm2d'), so close and far actors are judged the same.
  4. sanity: left/right test, mirror test, limb-length and stature ratios Anny/SMPL, frames with any joint error
     > 100 mm (worst 10 listed), forward-direction angles of pelvis/chest/neck/head, SMPL-forward vs jointPositions.
     Items 1-4 see only the frames the label builder KEPT (it drops joint error > 80 mm, visible < 0.3, box < 40 px).
  4b. gender (I1): (a) which end of Anny's gender value is male: rest bodies at gender 0 and 1 (stature, shoulder
     width), no data needed; (b) for --gender_refit_per_seq people per sequence, two refits of the label with
     smpl_to_anny.fit_frame on the camera-frame 3DPW joints, from the same warm start (label pose, root, shape) and
     with the same budget: label gender (control) and gender g -> 1-g, both with height/proportions free as in the
     converter. Joint error and surface distance to the gendered SMPL mesh of the two refits are compared.
  5. the training loader (datasets/threedpw_anny.py + bedlam collate) on a few items: it passes anny_pose/anny_shape
     through unchanged, keeps the human order, and its padded 672 K maps original-image pixels to the right 672 pixel.
     The Anny decode used everywhere here is a copy of train.py prepare_gt run in FP32; training runs it with TF32
     matmuls (--tf32 1), and on a GPU the TF32-vs-FP32 gap of that decode is reported (not graded).
  6. ALL converter fits (every <split>/<seq>_p<k>.npz, before the builder's filters): fitted frames, joint error of
     every fitted frame (mean / median / p95 / max, per joint), how many frames each builder filter drops, and the
     label images those fits should give per sequence against the label index (I4).
  7. ~12 overlays [photo | SMPL GT mesh | Anny label mesh + joints] in <out>/vis/ (pyrender; projected vertex dots
     when EGL is not available).
Writes <out>/summary.json (all numbers, verdicts, legend), <out>/per_person.csv, <out>/how_to_read.txt, <out>/vis/.
per_person.csv is rewritten after every sequence; summary.json and how_to_read.txt are written as soon as the
measurements are done and rewritten after the loader check and after the renders ('stage' says how far it got).
Read-only: no label, .npz or checkpoint is written or changed. The LEGEND constant below says how to read it.

USAGE (repo root on the cluster, multihmr env; GPU optional, KD-trees run on the CPU):
    python scripts/diagnostics/check_3dpw_conversion.py \
        --labels threedpw_anny_full/labels --fits threedpw_anny_full \
        --seq_root data/3DPW/sequenceFiles --smpl_dir models --out results/diagnostics/check_3dpw_conversion
"""
import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, 'scripts', 'analysis'), os.path.join(ROOT, 'scripts', 'data'), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# --------------------------------------------------------------------------------------------------------------- #
# constants (pure python: the helpers below are importable and testable without torch)
# --------------------------------------------------------------------------------------------------------------- #
SMPL_JOINT_NAMES = [            # == scripts/data/smpl_to_anny.py SMPL_JOINT_NAMES (asserted in main)
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
]
J = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
HAND_JOINTS = ('left_hand', 'right_hand')            # finger2-1 knuckles: no hand pose exists in the labels
BODY_MASK = np.array([n not in HAND_JOINTS for n in SMPL_JOINT_NAMES])
MIRROR = [J[n.replace('left_', 'right_')] if n.startswith('left_') else
          (J[n.replace('right_', 'left_')] if n.startswith('right_') else i) for i, n in enumerate(SMPL_JOINT_NAMES)]
# 3DPW poses2d = OpenPose COCO-18 (0 nose 1 neck 2 RSho 3 REl 4 RWr 5 LSho 6 LEl 7 LWr 8 RHip 9 RKnee 10 RAnk 11 LHip
# 12 LKnee 13 LAnk 14-17 eyes/ears). The 13 points with an SMPL counterpart, as in docs/3dpw_adapter_check.md:
OPENPOSE_TO_SMPL = [(1, J['neck'], 'neck'), (2, J['right_shoulder'], 'r_shoulder'), (3, J['right_elbow'], 'r_elbow'),
                    (4, J['right_wrist'], 'r_wrist'), (5, J['left_shoulder'], 'l_shoulder'),
                    (6, J['left_elbow'], 'l_elbow'), (7, J['left_wrist'], 'l_wrist'), (8, J['right_hip'], 'r_hip'),
                    (9, J['right_knee'], 'r_knee'), (10, J['right_ankle'], 'r_ankle'), (11, J['left_hip'], 'l_hip'),
                    (12, J['left_knee'], 'l_knee'), (13, J['left_ankle'], 'l_ankle')]
# SMPL body regions = argmax of SMPL's skinning weights (a vertex belongs to the part its main joint rotates)
SMPL_REGIONS = {'head': (15,), 'neck': (12,), 'torso': (0, 3, 6, 9, 13, 14), 'arms': (16, 17, 18, 19),
                'hands': (20, 21, 22, 23), 'legs': (1, 2, 4, 5), 'feet': (7, 8, 10, 11)}
# pose-independent segment lengths (left and right averaged)
SEGMENTS = {'upper_arm': (('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')),
            'forearm': (('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist')),
            'thigh': (('left_hip', 'left_knee'), ('right_hip', 'right_knee')),
            'shin': (('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'))}
LR_PAIRS = ((J['left_wrist'], J['right_wrist']), (J['left_ankle'], J['right_ankle']))
FWD_PARTS = (('pelvis', 0), ('chest', 9), ('neck', 12), ('head', 15))
NECK_BONES = ('neck01', 'neck02', 'neck03', 'head', 'jaw')     # local angles reported (the neck story)
DEFAULT_OFFSET_JOINTS = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'spine2', 'spine3', 'neck',
                         'left_collar', 'right_collar']          # smpl_to_anny.py --offset_joints default

# pass thresholds (the LEGEND explains each one)
THR = dict(selfcheck=(5.0, 15.0), fit_median=(20.0, 30.0), fit_p95=(40.0, 80.0), saved=(1.0, 5.0),
           surf=(20.0, 35.0), mm2d_delta=(15.0, 30.0), raw_mm2d_sane=45.0, lr=(0.99, 0.95), mirror=(0.99, 0.95),
           ratio=((0.95, 1.05), (0.90, 1.10)), stature=((0.95, 1.05), (0.90, 1.10)), bad_frames=(0.01, 0.05),
           loader_pose=1e-6, loader_px=0.5, loader_img=2.0, bad_frame_mm=100.0, good_fit_mm=30.0)
# scripts/data/build_threedpw_anny_labels.py defaults (slurm/data/run_build_labels.sh does not override them); the
# builder applies them in this order and counts a frame under the first filter it fails
BUILDER = dict(max_err=80.0, min_visible=0.3, min_box=40)
UNLABELLED_CAUSES = ('not_fitted', 'dropped_err', 'dropped_vis', 'dropped_box', 'kept_by_npz_but_absent', 'no_npz')

LEGEND = [
    "HOW TO READ check_3dpw_conversion (verdicts in summary.json; each threshold and why):",
    "V0 GT self-check: gendered SMPL(poses, betas, trans) joints vs 3DPW jointPositions. PASS median <= 5 mm (WARN",
    "   <= 15): the SMPL mesh used in V2 is the body the converter fitted to. Higher = the SMPL files differ from 3DPW's",
    "   (e.g. the neutral fallback is in use); then read V2 with care. 'smpl_model' in per_person.csv says which.",
    "V1 3D JOINTS vs 3DPW jointPositions, camera frame. 'fit' = the points the converter optimised (bone head + virtual",
    "   offset). PASS: body joints (no hand joints) median <= 20 mm AND p95 <= 40 mm over people; WARN <= 30 / <= 80.",
    "   Why: the converter was tuned to 16-20 mm per joint (slurm/data/run_smpl_to_anny_task.sh), 3DPW's own reference",
    "   joints are about 26 mm accurate [Likely: von Marcard et al., ECCV 2018], so 20 mm is inside the noise of the",
    "   ground truth. 'head' = bare Anny bone heads (train.py's GT joints): on torso rows it adds the fixed",
    "   rig-definition gap (several cm, not a fit error); limb rows = 'fit'.",
    "   V1-V4 describe the KEPT labels only: the sample is drawn from the label index, and the label builder already",
    "   dropped frames with joint error > 80 mm, visible < 0.3 or box < 40 px. Bad fits that were thrown away are",
    "   not in V1/V4e; I4 shows them (all fitted frames, before those filters).",
    "V1b saved vs recomputed joint error: p99 <= 1 mm PASS. The label decodes to the fit the converter reported (same",
    "   Anny version, same bone mapping, nothing changed between .npz and label pkl).",
    "V2 SURFACE, nearest point on the other surface, both directions (Anny skin -> SMPL, SMPL -> Anny skin). PASS:",
    "   median of the per-person mean <= 20 mm in BOTH directions; WARN <= 35; FAIL above. Why: 20 mm is about 1/3 of",
    "   the best 3DPW PA-PVE measured so far (62.7 mm, v9), so the label gap is a small part of the model error; 35 mm",
    "   is more than half of it. No vertex correspondence exists, so this is a LOWER bound on a per-vertex error;",
    "   accurate to ~2-3 mm (sample spacing printed). Region rows (SMPL -> Anny) say WHERE: head = head orientation,",
    "   hands = rest-pose fingers, torso = weight/belly that joints cannot see. A V2 PASS does NOT mean the gender or",
    "   body shape is right: in a synthetic test (Anny target, the real fit_frame) a label fitted with the WRONG gender",
    "   still stayed at 7-10 mm nearest-surface distance, inside the 20 mm PASS line. I1 tests the gender.",
    "V3 2D, independent of SMPL and of the fit: projected joints vs 3DPW OpenPose poses2d (13 points, conf > op_conf).",
    "   'raw' = 3DPW jointPositions (adapter check: 8.88 px mean, median 7.0 px at ~7 m depth = ~25 mm). Graded in",
    "   'mm2d' = pixel error x joint depth / focal (the sideways miss in mm at the joint), not in pixels: the same 25 mm",
    "   OpenPose-vs-SMPL gap is ~7 px at 7 m but ~20 px at 2.5 m, so a pixel rule would fail close actors for nothing.",
    "   PASS: median(Anny fit) - median(raw) <= 15 mm2d, WARN <= 30 (= the earlier 5 / 10 px at 6 m, f ~ 1960 px; the",
    "   residual V1 allows). 'raw_ok': raw median <= 45 mm2d (15 px at 6 m, ~1.8x the adapter check). Above that the",
    "   raw 3DPW joints miss OpenPose themselves (camera, 2D data or image index, not the Anny fit) and V3 is FAIL,",
    "   because the labels use the same camera and image index. Pixel values are still printed next to the mm2d ones.",
    "V4 SANITY: left/right (Anny left wrist+ankle nearer the SMPL left than the right one) >= 99% PASS; mirror test",
    "   (error as-is < error with Anny's L/R swapped) >= 99%; limb-length ratio Anny/SMPL, median per segment in",
    "   [0.95, 1.05] PASS / [0.90, 1.10] WARN (segment lengths do not depend on the pose, so they test the fitted",
    "   height/proportions); stature ratio of the rest meshes in [0.95, 1.05] (about +-8 cm); people with any joint",
    "   > 100 mm: <= 1% PASS, <= 5% WARN (worst 10 listed).",
    "V5 LOADER: datasets/threedpw_anny.py + bedlam collate pass anny_pose/anny_shape through unchanged (max abs diff",
    "   <= 1e-6), keep the human order, give the same image (mean abs <= 2 grey levels), and the loader's padded 672 K",
    "   maps original-image pixels to the right 672 pixel (<= 0.5 px). This shows the labels reach training unchanged.",
    "   It does NOT run train.py itself: the Anny decode used in this script is a COPY of train.py prepare_gt (same",
    "   calls, checked by reading), run in FP32. Training runs it with TF32 matmuls (--tf32 1, the default and the",
    "   v9/v10 setting); on a GPU 'tf32_gap' reports how far the TF32 decode moves the joints/vertices (not graded;",
    "   on a CPU TF32 does not exist and it is not measured). 'batch_vs_single_joint_mm' is rounding only.",
    "INFORMATIONAL (known issues from the code audit, no pass/fail):",
    "   I1 gender. (a) convention probe: Anny rest bodies at gender 0 and 1 (age 0.55, rest 0.5); 'male_end' = the",
    "   end that is taller AND has wider shoulders (Anny's code says 0 = male). Shown next to the mean label gender of",
    "   3DPW 'm' and 'f' people: men near the female end = reversed labels. (b) refit test, on --gender_refit_per_seq",
    "   people per sequence: the label is refitted twice to the camera-frame 3DPW joints (smpl_to_anny.fit_frame,",
    "   warm start = the label's pose, root and shape, iterations --gender_refit_iters, height/proportions free):",
    "   'same' = label gender (control: what the extra iterations alone give), 'flip' = gender g -> 1-g. Reported per",
    "   3DPW gender: joint error label / same / flip, the paired gain same - flip, and the share of people where 'flip'",
    "   has the lower joint error and the lower surface distance to the gendered SMPL mesh. Flip clearly better for",
    "   most people (share > 0.5 and a median gain of a few mm) = the label gender is wrong for the body it labels.",
    "   Synthetic test (Anny target, label fitted with the reversed gender): correct-gender refit 5-9.5 mm joint",
    "   error vs 12-18 mm for the reversed label, lower surface distance in 6/6 cases. On real 3DPW the SMPL-Anny rig",
    "   gap adds to both refits, so the gain can be smaller; a gain near 0 means 'not shown', not 'not reversed'.",
    "   Flipping g WITHOUT a refit is no test: pose, height and proportions were fitted for the labelled gender, so",
    "   the flipped body misses the joints and looks worse even when the label gender is wrong.",
    "   I2 head: forward-direction angle Anny vs SMPL of pelvis/chest/neck/head; small pelvis/chest + large head = head",
    "   orientation not transferred; 'head turn vs chest' SMPL vs Anny says the same. I3 coverage: humans per image,",
    "   people that train.py prepare_gt would drop, and two-person images where the other actor is valid in 3DPW",
    "   (campose_valid) but has no label, split by cause (from that actor's .npz): 'not_fitted' = that image is not",
    "   among the actor's fitted frames (mostly the per-person stride phase; also NaN-filtered or NaN-fit frames);",
    "   'dropped_err/vis/box' = fitted, then dropped by the builder filter (joint error > 80 mm, visible < 0.3 - e.g.",
    "   walking out of the image - or box < 40 px: expected, not a bug); 'kept_by_npz_but_absent' = the .npz says the",
    "   builder should have kept it (labels built from other .npz files); 'no_npz' = cause unknown. No projection",
    "   test: 'valid' is 3DPW's flag, not a visibility check.",
    "   I4 converter fits: every .npz under --fits, ALL fitted frames before the builder filters: joint error",
    "   mean/median/p95/max, share > 30 mm (the converter's own 'good fit' line) and > 80 mm, frames each builder",
    "   filter drops, per-joint medians, and per sequence the label images the .npz files should give against the",
    "   label index: equal = the labels were built from these .npz files with these filters (--max_err, --min_visible,",
    "   --min_box of this script; default = the builder's defaults, which run_build_labels.sh uses); 'expected' larger",
    "   = 3DPW images or label files missing on disk; smaller = the labels came from other .npz files.",
    "OVERALL = the worst V0-V5 grade; a V check that could not be measured makes OVERALL 'NOT MEASURED' (never PASS),",
    "   and 'not_measured' lists which ones.",
    "SCOPE: passing V0-V5 says the labels match 3DPW's joints, surface, camera and 2D within these tolerances. It does",
    "   NOT check bone twist, finger pose, gender (see I1), or the phenotypes joints cannot see (weight, muscle, age).",
]


# --------------------------------------------------------------------------------------------------------------- #
# pure numpy helpers
# --------------------------------------------------------------------------------------------------------------- #
def stats(x):
    """mean / median / p95 / max over the finite values of x."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def nanmean(x):
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


def world_to_cam(X, P):
    """[..., 3] world points with a 4x4 (or 3x4) world->camera matrix (3DPW cam_poses): R X + t."""
    P = np.asarray(P, dtype=np.float64)
    return np.asarray(X, dtype=np.float64) @ P[:3, :3].T + P[:3, 3]


def project(Xc, K):
    """OpenCV pinhole projection [..., 3] camera points -> [..., 2] pixels; NaN for points at or behind the camera."""
    Xc = np.asarray(Xc, dtype=np.float64)
    z = Xc[..., 2:3]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = (Xc / z) @ np.asarray(K, dtype=np.float64).T
    uv = uv[..., :2]
    return np.where(z > 1e-6, uv, np.nan)


def openpose_errors(op, j24_px, conf_min=0.0):
    """op: OpenPose COCO-18 of one person and frame, [18, 3] (x, y, conf) or 3DPW's [3, 18]; j24_px [24, 2].
    -> pixel distance for the 13 OPENPOSE_TO_SMPL pairs, NaN where conf <= conf_min or a point is missing."""
    e = np.full(len(OPENPOSE_TO_SMPL), np.nan)
    if op is None:
        return e
    op = np.asarray(op, dtype=np.float64)
    if op.shape == (3, 18):
        op = op.T
    if op.shape != (18, 3):
        return e
    j24_px = np.asarray(j24_px, dtype=np.float64)
    for k, (oi, si, _) in enumerate(OPENPOSE_TO_SMPL):
        if op[oi, 2] > conf_min and np.isfinite(op[oi, :2]).all() and np.isfinite(j24_px[si]).all():
            e[k] = float(np.linalg.norm(op[oi, :2] - j24_px[si]))
    return e


def quads_to_tris(F):
    F = np.asarray(F, dtype=np.int64)
    if F.shape[1] == 4:
        return np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]], 0)
    return F


def largest_component(F, n_verts):
    """Bool [n_verts]: vertices of the largest edge-connected component of the faces F (tris or quads). Anny's mesh
    (remove_unattached_vertices=False, as train.py builds it) has 19158 vertices: 13348 skin in one component, plus
    teeth/tongue and two eyeballs as separate components and ~5400 helper vertices on no face."""
    F = np.asarray(F, dtype=np.int64)
    e = np.concatenate([F[:, [i, (i + 1) % F.shape[1]]] for i in range(F.shape[1])], 0)
    lab = np.arange(n_verts)
    while True:                                    # min-label propagation + pointer jumping
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        new = lab.copy()
        np.minimum.at(new, e[:, 0], m)
        np.minimum.at(new, e[:, 1], m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    used = np.zeros(n_verts, dtype=bool)
    used[F.reshape(-1)] = True
    cnt = np.bincount(lab[used], minlength=n_verts)
    return used & (lab == int(np.argmax(cnt)))


def tri_areas(V, F):
    V = np.asarray(V, dtype=np.float64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


class SurfaceSampler:
    """A fixed area-weighted random sample of points on a triangle mesh, drawn ONCE on a reference pose (face index +
    barycentric coordinates) and re-used on every posed mesh of the same topology: deterministic, and the same
    sample for every person."""

    def __init__(self, V_ref, F_tri, n, seed=0):
        rng = np.random.default_rng(seed)
        self.F = np.asarray(F_tri, dtype=np.int64)
        A = tri_areas(V_ref, self.F)
        self.area = float(A.sum())
        self.fi = rng.choice(len(self.F), size=int(n), p=A / A.sum())
        r1, r2 = rng.random(int(n)), rng.random(int(n))
        s = np.sqrt(r1)
        self.bary = np.stack([1.0 - s, s * (1.0 - r2), s * r2], 1)
        self.spacing_mm = 1000.0 * math.sqrt(self.area / max(int(n), 1))

    def points(self, V):
        tri = np.asarray(V, dtype=np.float64)[self.F[self.fi]]        # [n, 3, 3]
        return (tri * self.bary[:, :, None]).sum(1)


class NN:
    """Nearest-neighbour distances to a fixed point set (scipy cKDTree, built once; brute force without scipy)."""

    def __init__(self, ref):
        self.ref = np.asarray(ref, dtype=np.float64)
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.ref)
        except ImportError:
            self.tree = None

    def __call__(self, query):
        query = np.asarray(query, dtype=np.float64)
        if self.tree is not None:
            try:
                return self.tree.query(query, k=1, workers=-1)[0]
            except TypeError:                      # scipy < 1.6
                return self.tree.query(query, k=1)[0]
        ref = self.ref
        rr = (ref * ref).sum(1)[None]
        out = np.empty(len(query))
        for i in range(0, len(query), 512):
            q = query[i:i + 512]
            d2 = (q * q).sum(1)[:, None] - 2.0 * q @ ref.T + rr
            out[i:i + 512] = np.sqrt(np.clip(d2.min(1), 0.0, None))
        return out


def nn_dist(query, ref):
    """Distance from each query point to the nearest ref point."""
    return NN(ref)(query)


def rodrigues(v):
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * Kx @ Kx


def smpl_global_rots(pose72, R_cam=None):
    """Global rotation of every SMPL joint (24 x [3, 3]) from the axis-angle pose, optionally pre-multiplied by the
    world->camera rotation. SMPL's rest global rotations are identity, so G_j maps rest directions of part j."""
    loc = [rodrigues(pose72[3 * i:3 * i + 3]) for i in range(24)]
    R0 = np.eye(3) if R_cam is None else np.asarray(R_cam, dtype=np.float64)
    G = [None] * 24
    for i in range(24):
        G[i] = R0 @ loc[0] if i == 0 else G[SMPL_PARENTS[i]] @ loc[i]
    return G


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(float(unit(a) @ unit(b)), -1.0, 1.0))))


def body_forward(left, right, bottom, top):
    """Forward direction of a rest skeleton = lateral (right -> left) x up. SMPL: (+x) x (+y) = +z (it faces +z);
    Anny: (+x) x (+z) = -y (it faces -y). Rig-independent, so both are measured the same way."""
    up = unit(np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64))
    lat = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    lat = unit(lat - (lat @ up) * up)
    return unit(np.cross(lat, up))


def extent_along(V, axis):
    p = np.asarray(V, dtype=np.float64) @ unit(axis)
    return float(p.max() - p.min())


def lr_ok(anny_pts, smpl_pts, pairs=LR_PAIRS):
    """True when each Anny left/right point is nearer the SMPL point of the same side than the opposite one."""
    for l, r in pairs:
        a_l, a_r, s_l, s_r = anny_pts[l], anny_pts[r], smpl_pts[l], smpl_pts[r]
        if not (np.isfinite(a_l).all() and np.isfinite(a_r).all()):
            return None
        if not (np.linalg.norm(a_l - s_l) < np.linalg.norm(a_l - s_r) and
                np.linalg.norm(a_r - s_r) < np.linalg.norm(a_r - s_l)):
            return False
    return True


def segment_ratios(anny_pts, smpl_pts):
    """{segment: mean over sides of |Anny segment| / |SMPL segment|} for the pose-independent SEGMENTS."""
    out = {}
    for name, sides in SEGMENTS.items():
        r = []
        for a, b in sides:
            la = np.linalg.norm(anny_pts[J[b]] - anny_pts[J[a]])
            ls = np.linalg.norm(smpl_pts[J[b]] - smpl_pts[J[a]])
            if np.isfinite(la) and np.isfinite(ls) and ls > 1e-6:
                r.append(la / ls)
        out[name] = float(np.mean(r)) if r else float('nan')
    return out


def sample_per_sequence(files_by_seq, per_seq, seed):
    """Fixed sample: per sequence, min(per_seq, n) files without replacement, from a generator seeded by
    (seed, crc32(sequence name)) so the choice does not depend on which other sequences exist. per_seq <= 0 = all."""
    out = {}
    for seq in sorted(files_by_seq):
        fs = sorted(files_by_seq[seq])
        if per_seq <= 0 or per_seq >= len(fs):
            out[seq] = fs
            continue
        rng = np.random.default_rng([int(seed), zlib.crc32(seq.encode('utf-8'))])
        idx = np.sort(rng.choice(len(fs), size=int(per_seq), replace=False))
        out[seq] = [fs[i] for i in idx]
    return out


_LABEL_RE = re.compile(r'(?P<seq>[^/\\]+)[/\\]image_(?P<fid>\d+)\.pkl$')


def parse_label_path(path):
    """.../<split>/<seq>/image_NNNNN.pkl -> (seq, frame id)."""
    m = _LABEL_RE.search(path)
    if not m:
        raise ValueError(f"not a 3DPW-Anny label path: {path}")
    return m.group('seq'), int(m.group('fid'))


def grade(value, pass_max, warn_max):
    if value is None or not np.isfinite(value):
        return 'NOT MEASURED'
    return 'PASS' if value <= pass_max else ('WARN' if value <= warn_max else 'FAIL')


def grade_min(value, pass_min, warn_min):
    if value is None or not np.isfinite(value):
        return 'NOT MEASURED'
    return 'PASS' if value >= pass_min else ('WARN' if value >= warn_min else 'FAIL')


def grade_range(value, pass_rng, warn_rng):
    if value is None or not np.isfinite(value):
        return 'NOT MEASURED'
    if pass_rng[0] <= value <= pass_rng[1]:
        return 'PASS'
    return 'WARN' if warn_rng[0] <= value <= warn_rng[1] else 'FAIL'


GRADE_ORDER = ('FAIL', 'WARN', 'NOT MEASURED', 'PASS')


def grade_word(g):
    """The grade without its explanation: 'FAIL (raw 2D check ...)' -> 'FAIL', 'NOT MEASURED' stays whole. Anything
    unknown or missing counts as 'NOT MEASURED' (never as a pass)."""
    if not isinstance(g, str) or not g.strip():
        return 'NOT MEASURED'
    g = g.strip()
    if g.startswith('NOT'):
        return 'NOT MEASURED'
    w = g.split(' ')[0]
    return w if w in GRADE_ORDER else 'NOT MEASURED'


def worst(grades):
    """Worst grade, FAIL < WARN < NOT MEASURED < PASS. Grades are normalised by grade_word first, so a check that was
    not measured can never be skipped into an overall PASS."""
    gs = [grade_word(g) for g in grades]
    return min(gs, key=GRADE_ORDER.index) if gs else 'NOT MEASURED'


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def fmt(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def read_index(labels_root, split):
    """The label files datasets/threedpw_anny.py would load for this split (index file, else glob; existing only)."""
    index = os.path.join(labels_root, f"index_{split}.txt")
    if os.path.isfile(index):
        with open(index) as f:
            files = [l.strip() for l in f if l.strip()]
        src = index
    else:
        files = sorted(glob.glob(os.path.join(labels_root, split, '*', '*.pkl')))
        src = 'glob'
    listed = len(files)
    files = [f for f in files if os.path.isfile(f)]
    return files, listed, src


def read_gender(seq, person):
    g = seq.get('genders', seq.get('gender', None))
    try:
        g = g[person]
    except Exception:  # noqa
        return None
    if isinstance(g, bytes):
        g = g.decode('latin1')
    g = str(g).strip().lower()
    return 'm' if g.startswith('m') else ('f' if g.startswith('f') else None)


# ---- converter fits (.npz) before the label builder's filters --------------------------------------------------
def read_fit_arrays(path):
    """The per-frame fields of one smpl_to_anny.py .npz that the label builder reads (same dtypes as
    build_threedpw_anny_labels.py, so its filters give the same answers here). 'usable' = has every field the
    builder filters on (v2.2 or later)."""
    d = np.load(path, allow_pickle=True)
    frames = np.asarray(d['frames']).astype(np.int64).reshape(-1)
    n = len(frames)
    err = (np.asarray(d['joint_err_mm'], dtype=np.float32).reshape(-1) if 'joint_err_mm' in d.files
           else np.full(n, np.nan, dtype=np.float32))
    vis = np.asarray(d['visible_frac'], dtype=np.float32).reshape(-1) if 'visible_frac' in d.files else None
    box = np.asarray(d['tight_bbox'], dtype=np.int32).reshape(-1, 4) if 'tight_bbox' in d.files else None
    pj = np.asarray(d['per_joint_err_mm'], dtype=np.float32) if 'per_joint_err_mm' in d.files else None
    return dict(path=path, frames=frames, err=err, vis=vis, box=box, per_joint=pj,
                version=str(d['version']) if 'version' in d.files else '?',
                usable=bool('joint_err_mm' in d.files and vis is not None and box is not None),
                frame_pos={int(f): i for i, f in enumerate(frames.tolist())})


def builder_drop_reason(err, vis, box, thr=BUILDER):
    """The build_threedpw_anny_labels.py filter a fitted frame fails first ('dropped_err' / 'dropped_vis' /
    'dropped_box'), or None when the builder keeps it. Same expressions and order as the builder."""
    if not np.isfinite(err) or err > thr['max_err']:
        return 'dropped_err'
    if vis < thr['min_visible']:
        return 'dropped_vis'
    if (box[2] - box[0]) < thr['min_box'] or (box[3] - box[1]) < thr['min_box']:
        return 'dropped_box'
    return None


def unlabelled_cause(fit, fid, thr=BUILDER):
    """Why an actor that 3DPW marks valid (campose_valid) at image fid has no label in that image. fit = that actor's
    read_fit_arrays() or None. -> one of UNLABELLED_CAUSES."""
    if fit is None or not fit.get('usable'):
        return 'no_npz'
    i = fit['frame_pos'].get(int(fid))
    if i is None:
        return 'not_fitted'                    # per-person stride phase (mostly), or a NaN-filtered / NaN-fit frame
    return builder_drop_reason(fit['err'][i], fit['vis'][i], fit['box'][i], thr) or 'kept_by_npz_but_absent'


def summarize_fits(fits, labels_per_seq=None, seqs_in_3dpw=None, thr=BUILDER, good_mm=None):
    """fits: list of (seq, person, read_fit_arrays dict) of one split (or all). ALL fitted frames, before the
    builder's filters: joint error distribution, drops per filter, per-joint medians, and per sequence the label
    images these fits should produce against the label index (labels_per_seq: {seq: label files in the index})."""
    good_mm = THR['good_fit_mm'] if good_mm is None else good_mm
    out = {'n_npz': len(fits), 'n_npz_unusable': int(sum(not f['usable'] for _, _, f in fits))}
    if not fits:
        out['sequences_in_3dpw_without_npz'] = sorted(seqs_in_3dpw or [])
        return out
    err = np.concatenate([f['err'] for _, _, f in fits]).astype(np.float64)
    fin = np.isfinite(err)
    out['fitted_frames'] = int(err.size)
    out['nonfinite_err_frames'] = int((~fin).sum())
    out['joint_err_mm'] = stats(err)
    out['frac_err_gt_good'] = float(np.mean(err[fin] > good_mm)) if fin.any() else None
    out['frac_err_gt_max_err'] = float(np.mean(~fin | (err > thr['max_err'])))
    out['good_fit_mm'], out['builder_thresholds'] = good_mm, dict(thr)
    drops = {'dropped_err': 0, 'dropped_vis': 0, 'dropped_box': 0, 'kept': 0, 'unusable': 0}
    per_seq = {}
    for seq, person, f in fits:
        ps = per_seq.setdefault(seq, dict(people=0, fitted=0, kept_person_frames=0, dropped_err=0, dropped_vis=0,
                                          dropped_box=0, unusable_npz=0, _kept=set(), _err=[]))
        ps['people'] += 1
        ps['fitted'] += int(len(f['frames']))
        ps['_err'].append(f['err'])
        if not f['usable']:
            ps['unusable_npz'] += 1
            drops['unusable'] += int(len(f['frames']))
            continue
        for i, fid in enumerate(f['frames'].tolist()):
            why = builder_drop_reason(f['err'][i], f['vis'][i], f['box'][i], thr)
            if why is None:
                drops['kept'] += 1
                ps['kept_person_frames'] += 1
                ps['_kept'].add(int(fid))
            else:
                drops[why] += 1
                ps[why] += 1
    out['builder_filter_counts'] = drops
    pj = [f['per_joint'] for _, _, f in fits if f['per_joint'] is not None and f['per_joint'].ndim == 2
          and f['per_joint'].shape[1] == len(SMPL_JOINT_NAMES)]
    if pj:
        pj = np.concatenate(pj, 0).astype(np.float64)
        out['per_joint_mm'] = {SMPL_JOINT_NAMES[j]: stats(pj[:, j]) for j in range(len(SMPL_JOINT_NAMES))}
    out['versions'] = {v: int(sum(f['version'] == v for _, _, f in fits)) for v in sorted({f['version'] for _, _, f in fits})}
    mism = []
    for seq, ps in sorted(per_seq.items()):
        e = np.concatenate(ps.pop('_err')).astype(np.float64)
        ps['err_mm'] = {k: stats(e)[k] for k in ('mean', 'median', 'p95', 'max')}
        ps['expected_label_images'] = len(ps.pop('_kept'))
        if labels_per_seq is not None:
            ps['label_images_in_index'] = int(labels_per_seq.get(seq, 0))
            if ps['label_images_in_index'] != ps['expected_label_images']:
                mism.append(seq)
    out['per_sequence'] = per_seq
    out['expected_label_images'] = int(sum(ps['expected_label_images'] for ps in per_seq.values()))
    if labels_per_seq is not None:
        out['label_images_in_index'] = int(sum(labels_per_seq.values()))
        out['sequences_expected_ne_index'] = mism
        out['sequences_in_index_without_npz'] = sorted(q for q in labels_per_seq if q not in per_seq)
    if seqs_in_3dpw is not None:
        out['sequences_in_3dpw_without_npz'] = sorted(q for q in seqs_in_3dpw if q not in per_seq)
    return out


def read_tasks(path):
    """configs/threedpw_tasks.txt (slurm/data/run_3dpw_list.sh): 'split seq person n_frames' per line."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[2].lstrip('-').isdigit():
                out.append((p[0], p[1], int(p[2])))
    return out


# --------------------------------------------------------------------------------------------------------------- #
# torch / model side
# --------------------------------------------------------------------------------------------------------------- #
class Ctx:
    """Models, caches and the mapping metadata shared by the measurement, loader and render phases."""

    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        # plain FP32 reference for every number here; the loader check measures the TF32 gap of training separately
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        import anny
        # exactly as train.py builds it (Trainer.__init__)
        bm = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).float().to(self.device)
        bm.set_skinning_method('lbs')
        self.bm = bm
        self.anny_version = getattr(anny, '__version__', '?')
        self.bones = list(bm.bone_labels)
        self.pheno = list(bm.phenotype_labels)
        self.gi = self.pheno.index('gender')
        faces = bm.faces.detach().cpu().numpy().astype(np.int64)
        self.faces_tri_all = quads_to_tris(faces)
        v_ref, _ = self.anny_forward(np.tile(np.eye(4, dtype=np.float32), (1, len(self.bones), 1, 1)),
                                     np.full((1, len(self.pheno)), 0.5, dtype=np.float32))
        self.n_verts = v_ref.shape[1]
        self.skin = largest_component(faces, self.n_verts)
        self.skin_idx = np.where(self.skin)[0]
        self.skin_faces = self.faces_tri_all[np.all(self.skin[self.faces_tri_all], 1)]
        self.anny_sampler = SurfaceSampler(v_ref[0], self.skin_faces, args.surf_samples, args.seed)
        print(f"[anny] {self.anny_version}: {len(self.bones)} bones, {self.n_verts} vertices, {len(self.skin_idx)} skin "
              f"vertices on {len(self.skin_faces)} skin triangles (largest component; eyes/teeth/helpers excluded); "
              f"surface sample {args.surf_samples} points, spacing ~{self.anny_sampler.spacing_mm:.1f} mm", flush=True)
        import smpl_to_anny as S2A
        assert list(S2A.SMPL_JOINT_NAMES) == SMPL_JOINT_NAMES, 'SMPL joint order differs from smpl_to_anny.py'
        self.S2A = S2A
        self.smpl_models = {}
        self.smpl_sampler = None
        self.smpl_region = None
        self.smpl_warned = set()
        self.meta_cache = {}
        self.fallback = None
        self.rest_cache = {}
        self.seq_cache = (None, None)
        self.fit_cache = {}
        self.builder_thr = dict(max_err=float(getattr(args, 'max_err', BUILDER['max_err'])),
                                min_visible=float(getattr(args, 'min_visible', BUILDER['min_visible'])),
                                min_box=int(getattr(args, 'min_box', BUILDER['min_box'])))

    # ---- converter fit arrays (frames, joint error, visibility, box) per person -----------------------------------
    def fit(self, split, seq, person):
        """read_fit_arrays of <fits>/<split>/<seq>_p<person>.npz, cached; None when the file is missing/unreadable."""
        key = (split, seq, int(person))
        if key not in self.fit_cache:
            npz = os.path.join(self.args.fits, split, f"{seq}_p{int(person)}.npz")
            f = None
            if os.path.isfile(npz):
                try:
                    f = read_fit_arrays(npz)
                except Exception as e:  # noqa
                    print(f"[fits] WARNING: could not read {npz} ({type(e).__name__}: {e})", flush=True)
            self.fit_cache[key] = f
        return self.fit_cache[key]

    # ---- Anny ---------------------------------------------------------------------------------------------------
    def anny_forward(self, pose44, shape11):
        """train.py prepare_gt decode: pose [N, 163, 4, 4] (camera frame), phenotypes clamped to [0, 1].
        -> vertices [N, V, 3], bone poses [N, 163, 4, 4] (float64 numpy)."""
        torch = self.torch
        P = torch.as_tensor(np.asarray(pose44, dtype=np.float32)[:, :163], device=self.device)
        S = torch.clamp(torch.as_tensor(np.asarray(shape11, dtype=np.float32), device=self.device), 0.0, 1.0)
        kw = {k: S[:, i] for i, k in enumerate(self.bm.phenotype_labels)}
        with torch.no_grad():
            out = self.bm(pose_parameters=P, phenotype_kwargs=kw)
        return (out['vertices'].double().cpu().numpy(), out['bone_poses'].double().cpu().numpy())

    # ---- SMPL ---------------------------------------------------------------------------------------------------
    def smpl(self, gender):
        """Gendered SMPL as train.py loads it: smplx.create(SMPLX_DIR, 'smpl', gender=...). Falls back to the neutral
        model (reported per person) when the gendered file is missing."""
        key = {'m': 'male', 'f': 'female'}.get(gender, 'neutral')
        if key in self.smpl_models:
            return self.smpl_models[key]
        import smplx
        try:
            m = smplx.create(self.args.smpl_dir, model_type='smpl', gender=key).to(self.device)
            entry = (m, key)
        except Exception as e:  # noqa
            if key == 'neutral':
                raise
            print(f"[smpl] WARNING: {key} SMPL not loadable from {self.args.smpl_dir} ({type(e).__name__}: {e}); "
                  f"using the NEUTRAL model for these people (V0 will show the cost)", flush=True)
            entry = self.smpl('n')
        self.smpl_models[key] = entry
        if self.smpl_region is None:
            m = entry[0]
            w = m.lbs_weights.detach().cpu().numpy()
            self.smpl_region = np.argmax(w, axis=1)
            self.smpl_faces = np.asarray(m.faces, dtype=np.int64)
            v0 = m.v_template.detach().cpu().numpy()
            self.smpl_sampler = SurfaceSampler(v0, self.smpl_faces, self.args.surf_samples, self.args.seed + 1)
            print(f"[smpl] {key}: {v0.shape[0]} vertices, {len(self.smpl_faces)} triangles; surface sample spacing "
                  f"~{self.smpl_sampler.spacing_mm:.1f} mm", flush=True)
        return entry

    def smpl_forward(self, seqd, person, fid, zero_pose=False):
        """World-frame SMPL vertices [6890, 3], joints [24, 3] and the model kind used."""
        torch = self.torch
        m, kind = self.smpl(read_gender(seqd, person))
        betas = np.zeros(10, dtype=np.float32)
        b = np.asarray(seqd['betas'][person], dtype=np.float32).reshape(-1)[:10]
        betas[:len(b)] = b
        if zero_pose:
            pose = np.zeros(72, dtype=np.float32)
            trans = np.zeros(3, dtype=np.float32)
        else:
            pose = np.asarray(seqd['poses'][person][fid], dtype=np.float32).reshape(72)
            trans = np.asarray(seqd['trans'][person][fid], dtype=np.float32).reshape(3)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)[None]  # noqa: E731
        with torch.no_grad():
            o = m(global_orient=t(pose[:3]), body_pose=t(pose[3:]), betas=t(betas), transl=t(trans))
        return (o.vertices[0].double().cpu().numpy(), o.joints[0, :24].double().cpu().numpy(), kind)

    # ---- 3DPW sequence ------------------------------------------------------------------------------------------
    def load_seq(self, split, seq):
        key = (split, seq)
        if self.seq_cache[0] == key:
            return self.seq_cache[1]
        path = os.path.join(self.args.seq_root, split, seq + '.pkl')
        if not os.path.isfile(path):
            self.seq_cache = (key, None)
            return None
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        self.seq_cache = (key, d)
        return d

    # ---- mapping metadata ---------------------------------------------------------------------------------------
    def meta(self, split, seq, person):
        """Bone mapping and virtual offsets the converter used for this person (from its .npz), else the
        smpl_to_anny.py defaults recomputed here. -> dict(smpl_idx, anny_idx, off [M, 3], source, ...)."""
        key = (split, seq, person)
        if key in self.meta_cache:
            return self.meta_cache[key]
        npz = os.path.join(self.args.fits, split, f"{seq}_p{person}.npz")
        m = None
        if os.path.isfile(npz):
            try:
                d = np.load(npz, allow_pickle=True)
                si = [int(x) for x in d['smpl_idx']]
                ai = [int(x) for x in d['anny_idx']]
                off = (np.asarray(d['joint_offsets_local'], dtype=np.float64) if 'joint_offsets_local' in d.files
                       else np.zeros((len(si), 3)))
                names = [str(x) for x in d['anny_bone_names']] if 'anny_bone_names' in d.files else None
                names_ok = None if names is None else all(self.bones[a] == n for a, n in zip(ai, names))
                m = dict(smpl_idx=si, anny_idx=ai, off=off, source='npz', npz=npz,
                         version=str(d['version']) if 'version' in d.files else '?',
                         target_mode=str(d['target_mode']) if 'target_mode' in d.files else '?',
                         free_bones=str(d['free_bones']) if 'free_bones' in d.files else '?',
                         bone_names=names, names_ok=names_ok,
                         offset_joints=[str(x) for x in d['offset_joints']] if 'offset_joints' in d.files else None)
                if names_ok is False:
                    print(f"[meta] WARNING: {npz}: anny_bone_names do not match this Anny's bone order "
                          f"(different anny version?)", flush=True)
            except Exception as e:  # noqa
                print(f"[meta] WARNING: could not read {npz} ({type(e).__name__}: {e}); using the defaults", flush=True)
                m = None
        if m is None:
            m = dict(self.fallback_mapping())
        m['anny_of'] = dict(zip(m['smpl_idx'], m['anny_idx']))
        self.meta_cache[key] = m
        return m

    def fallback_mapping(self):
        if self.fallback is not None:
            return self.fallback
        S2A = self.S2A
        pairs = S2A.resolve_mapping(self.bones, verbose=False)
        offs, source = {}, 'defaults: name mapping, no offsets (no .npz, no SMPL_NEUTRAL) -> fit points = bone heads'
        try:
            import smplx
            sm = smplx.create(self.args.smpl_dir, model_type='smpl', gender='neutral').to(self.device)
            ja = S2A.anny_rest_positions(self.bm, self.device)
            Ra = S2A.anny_rest_frames(self.bm, self.device)
            js = S2A.smpl_rest_joints(sm, self.device)
            pairs = S2A.refine_mapping_by_geometry(pairs, self.bones, ja, js, verbose=False)
            offs = S2A.compute_joint_offsets(pairs, self.bones, ja, Ra, js, DEFAULT_OFFSET_JOINTS, verbose=False)
            source = 'defaults recomputed (no .npz): smpl_to_anny auto-mapping + virtual offsets from SMPL_NEUTRAL'
        except Exception as e:  # noqa
            print(f"[meta] neutral SMPL not available for the default offsets ({type(e).__name__}: {e})", flush=True)
        si = [p[0] for p in pairs]
        ai = [p[1] for p in pairs]
        off = np.stack([np.asarray(offs.get(s, np.zeros(3)), dtype=np.float64) for s in si])
        self.fallback = dict(smpl_idx=si, anny_idx=ai, off=off, source=source, npz=None, version='-',
                             target_mode='-', free_bones='-', bone_names=[self.bones[a] for a in ai], names_ok=True,
                             offset_joints=sorted(SMPL_JOINT_NAMES[s] for s in offs))
        print(f"[meta] {source}", flush=True)
        return self.fallback

    # ---- per-person rest geometry (stature, rest forward, rest bone frames) -------------------------------------
    def rest(self, split, seq, seqd, person, shape11, meta):
        key = (split, seq, person, tuple(np.round(np.asarray(shape11, dtype=np.float64), 4)))
        if key in self.rest_cache:
            return self.rest_cache[key]
        I = np.tile(np.eye(4, dtype=np.float32), (1, len(self.bones), 1, 1))
        v, bp = self.anny_forward(I, np.asarray(shape11, dtype=np.float32)[None])
        heads = bp[0, :, :3, 3]
        a = meta['anny_of']
        hip_l, hip_r, neck = heads[a[J['left_hip']]], heads[a[J['right_hip']]], heads[a[J['neck']]]
        mid = 0.5 * (hip_l + hip_r)
        up_a = neck - mid
        vs, js, _ = self.smpl_forward(seqd, person, 0, zero_pose=True)
        mid_s = 0.5 * (js[J['left_hip']] + js[J['right_hip']])
        up_s = js[J['neck']] - mid_s
        r = dict(stature_anny=extent_along(v[0][self.skin_idx], up_a), stature_smpl=extent_along(vs, up_s),
                 fwd_anny=body_forward(hip_l, hip_r, mid, neck),
                 fwd_smpl=body_forward(js[J['left_hip']], js[J['right_hip']], mid_s, js[J['neck']]),
                 R_rest=bp[0, :, :3, :3])
        self.rest_cache[key] = r
        return r


def surface_metrics(ctx, va_full, vs, nn_smpl=None):
    """Nearest-point distances (mm): Anny skin vertices -> SMPL surface and SMPL vertices -> Anny skin surface
    (each surface = its vertices + the fixed area-weighted sample). nn_smpl: a prebuilt NN on the SMPL surface."""
    va = va_full[ctx.skin_idx]
    if nn_smpl is None:
        nn_smpl = NN(np.concatenate([vs, ctx.smpl_sampler.points(vs)], 0))
    nn_anny = NN(np.concatenate([va, ctx.anny_sampler.points(va_full)], 0))
    return nn_smpl(va) * 1000.0, nn_anny(vs) * 1000.0


def measure_image(ctx, path, split, seq, seqd):
    """All measurements for the humans of one label file. -> (person rows, per-joint arrays, image record)."""
    args = ctx.args
    lab = load_pkl(path)
    _, fid = parse_label_path(path)
    humans = lab.get('humans', [])
    n_seq_people = len(seqd['poses'])
    persons = [int(str(h['name']).rsplit('_p', 1)[1]) for h in humans]
    cv = seqd.get('campose_valid', None)
    unlabelled_valid = 0
    causes = {c: 0 for c in UNLABELLED_CAUSES}
    for q in range(n_seq_people):
        if q not in persons and cv is not None and bool(np.asarray(cv[q])[fid]):
            unlabelled_valid += 1
            causes[unlabelled_cause(ctx.fit(split, seq, q), fid, ctx.builder_thr)] += 1
    img_rec = dict(split=split, seq=seq, frame=fid, n_humans=len(humans), n_seq_people=n_seq_people,
                   unlabelled_valid=unlabelled_valid, unlabelled_causes=causes, path=path)
    if not humans:
        return [], [], img_rec
    P = np.asarray(seqd['cam_poses'][fid], dtype=np.float64)
    K = np.asarray(seqd['cam_intrinsics'], dtype=np.float64)
    k_diff = max(abs(float(lab['focal'][0]) - K[0, 0]), abs(float(lab['focal'][1]) - K[1, 1]),
                 abs(float(lab['princpt'][0]) - K[0, 2]), abs(float(lab['princpt'][1]) - K[1, 2]))
    img_path = lab.get('imagename', '')
    try:
        from PIL import Image
        W, H = Image.open(img_path).size
        img_ok = True
    except Exception:  # noqa
        W, H, img_ok = int(round(2 * K[0, 2])), int(round(2 * K[1, 2])), False
    img_rec.update(K_label_diff=k_diff, image_found=img_ok, W=W, H=H)
    s672 = args.img_size / float(max(W, H))
    K672 = K.copy()
    K672[:2] *= s672
    K672[0, 2] += (args.img_size - W * s672) / 2.0
    K672[1, 2] += (args.img_size - H * s672) / 2.0

    pose = np.stack([np.asarray(h['anny_pose'], dtype=np.float32)[:163] for h in humans])
    shape = np.stack([np.asarray(h['anny_shape'], dtype=np.float32).reshape(-1) for h in humans])
    n = len(humans)
    V, BP = ctx.anny_forward(pose, shape)
    rows, extra = [], []
    cells = set()
    for k, h in enumerate(humans):
        p = persons[k]
        meta = ctx.meta(split, seq, p)
        g3 = read_gender(seqd, p)
        jp = np.asarray(seqd['jointPositions'][p][fid], dtype=np.float64).reshape(24, 3)
        jp_c = world_to_cam(jp, P)
        vs_w, js_w, smpl_kind = ctx.smpl_forward(seqd, p, fid)
        vs_c, js_c = world_to_cam(vs_w, P), world_to_cam(js_w, P)
        bp = BP[k]
        heads = bp[:, :3, 3]
        fit_pts = np.full((24, 3), np.nan)
        head_pts = np.full((24, 3), np.nan)
        for si, ai, off in zip(meta['smpl_idx'], meta['anny_idx'], meta['off']):
            head_pts[si] = heads[ai]
            fit_pts[si] = heads[ai] + bp[ai, :3, :3] @ off
        d_fit = np.linalg.norm(fit_pts - jp_c, axis=-1) * 1000.0
        d_head = np.linalg.norm(head_pts - jp_c, axis=-1) * 1000.0
        mapped = np.isfinite(d_fit)
        jmax = int(np.nanargmax(d_fit)) if mapped.any() else -1
        # surface as labelled (V2)
        nn_smpl = NN(np.concatenate([vs_c, ctx.smpl_sampler.points(vs_c)], 0))
        a2s, s2a = surface_metrics(ctx, V[k], vs_c, nn_smpl)
        # (the gender test, I1, refits a subset of these people after the measurements: gender_refit)
        # local rotations of the label: exactly-identity bones, neck/head angles
        Rl = pose[k, 1:, :3, :3].astype(np.float64)
        ident = np.all(Rl == np.eye(3), axis=(1, 2))
        ang_local = np.degrees(np.arccos(np.clip((np.trace(Rl, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
        # 2D vs OpenPose
        op = None
        if 'poses2d' in seqd:
            try:
                op = np.asarray(seqd['poses2d'][p][fid], dtype=np.float64)
            except Exception:  # noqa
                op = None
        px = {nm: openpose_errors(op, project(X, K), args.op_conf)
              for nm, X in (('raw', jp_c), ('smplfwd', js_c), ('anny_fit', fit_pts), ('anny_head', head_pts))}
        mmpp = np.array([jp_c[si, 2] / K[0, 0] * 1000.0 for _, si, _ in OPENPOSE_TO_SMPL])
        # sanity
        e_as = nanmean(d_fit)
        e_sw = nanmean(np.linalg.norm(fit_pts[MIRROR] - jp_c, axis=-1) * 1000.0)
        rest = ctx.rest(split, seq, seqd, p, shape[k], meta)
        G = smpl_global_rots(np.asarray(seqd['poses'][p][fid], dtype=np.float64).reshape(72), P[:3, :3])
        fa, fs, ang = {}, {}, {}
        for name, sj in FWD_PARTS:
            ai = meta['anny_of'].get(sj)
            if ai is None:
                continue
            fa[name] = bp[ai, :3, :3] @ rest['R_rest'][ai].T @ rest['fwd_anny']
            fs[name] = G[sj] @ rest['fwd_smpl']
            ang[name] = angle_deg(fa[name], fs[name])
        # what train.py prepare_gt keeps: root in front, inside the 672 image, first in its 14-px cell
        root = heads[0]
        loc = project(root, K672)
        in_img = bool(root[2] > 0.1 and np.isfinite(loc).all() and 0 <= loc[0] < args.img_size
                      and 0 <= loc[1] < args.img_size)
        cell = (int(loc[0] // args.patch), int(loc[1] // args.patch)) if in_img else None
        kept = in_img and cell not in cells
        if kept:
            cells.add(cell)
        regions = {r: nanmean(s2a[np.isin(ctx.smpl_region, js)]) for r, js in SMPL_REGIONS.items()}
        sr = segment_ratios(fit_pts, jp_c)
        saved = float(h['joint_err_mm']) if 'joint_err_mm' in h else float('nan')
        fit_all = nanmean(d_fit[mapped])
        row = dict(split=split, seq=seq, frame=fid, person=p, path=path, n_humans=n,
                   gender_3dpw=g3 or '', gender_label=float(shape[k, ctx.gi]), smpl_model=smpl_kind,
                   mapping=meta['source'], version=meta['version'],
                   fit_all_mm=fit_all, fit_body_mm=nanmean(d_fit[mapped & BODY_MASK]),
                   fit_max_mm=float(d_fit[jmax]) if jmax >= 0 else float('nan'),
                   fit_max_joint=SMPL_JOINT_NAMES[jmax] if jmax >= 0 else '',
                   head_all_mm=nanmean(d_head[mapped]), head_body_mm=nanmean(d_head[mapped & BODY_MASK]),
                   saved_err_mm=saved, saved_minus_recomputed_mm=saved - fit_all,
                   smpl_selfcheck_mm=nanmean(np.linalg.norm(js_c - jp_c, axis=-1) * 1000.0),
                   a2s_mean=float(a2s.mean()), a2s_median=float(np.median(a2s)), a2s_p95=float(np.percentile(a2s, 95)),
                   s2a_mean=float(s2a.mean()), s2a_median=float(np.median(s2a)), s2a_p95=float(np.percentile(s2a, 95)),
                   age_label=float(shape[k, ctx.pheno.index('age')]) if 'age' in ctx.pheno else float('nan'),
                   height_label=float(shape[k, ctx.pheno.index('height')]) if 'height' in ctx.pheno else float('nan'),
                   proportions_label=(float(shape[k, ctx.pheno.index('proportions')]) if 'proportions' in ctx.pheno
                                      else float('nan')),
                   n_identity_bones=int(ident.sum()) + int(np.all(pose[k, 0, :3, :3] == np.eye(3))),
                   px_raw=nanmean(px['raw']), px_smplfwd=nanmean(px['smplfwd']), px_anny_fit=nanmean(px['anny_fit']),
                   px_anny_head=nanmean(px['anny_head']), n_px=int(np.isfinite(px['raw']).sum()),
                   mm2d_raw=nanmean(px['raw'] * mmpp), mm2d_anny_fit=nanmean(px['anny_fit'] * mmpp),
                   lr_ok=lr_ok(head_pts, jp_c), mirror_ok=bool(e_as < e_sw) if np.isfinite(e_sw) else None,
                   stature_anny_m=rest['stature_anny'], stature_smpl_m=rest['stature_smpl'],
                   stature_ratio=rest['stature_anny'] / max(rest['stature_smpl'], 1e-6),
                   depth_m=float(jp_c[0, 2]), train_kept=kept, K_label_diff=k_diff, image_found=img_ok)
        for r, v in regions.items():
            row[f's2a_{r}'] = v
        for s, v in sr.items():
            row[f'ratio_{s}'] = v
        for name, v in ang.items():
            row[f'ang_{name}'] = v
        for b in NECK_BONES:
            if b in ctx.bones:
                row[f'local_deg_{b}'] = float(ang_local[ctx.bones.index(b) - 1])
        if 'head' in fa and 'chest' in fa:
            row['headturn_smpl'] = angle_deg(fs['head'], fs['chest'])
            row['headturn_anny'] = angle_deg(fa['head'], fa['chest'])
        rows.append(row)
        extra.append(dict(d_fit=d_fit, d_head=d_head, px=px, mmpp=mmpp, ident=ident))
    return rows, extra, img_rec


# --------------------------------------------------------------------------------------------------------------- #
# I1 gender: convention probe + refit test
# --------------------------------------------------------------------------------------------------------------- #
def gender_convention_probe(ctx):
    """Which end of Anny's gender value is male? Rest bodies (identity pose) at gender 0 and 1, age 0.55 (the 3DPW
    label pin), every other phenotype 0.5. Deterministic, needs no data. 'male_end' = the end that is taller AND has
    the wider shoulders (None when the two disagree)."""
    try:
        b = ctx.bones.index
        I = np.tile(np.eye(4, dtype=np.float32), (2, len(ctx.bones), 1, 1))
        shp = np.full((2, len(ctx.pheno)), 0.5, dtype=np.float32)
        if 'age' in ctx.pheno:
            shp[:, ctx.pheno.index('age')] = 0.55
        shp[0, ctx.gi], shp[1, ctx.gi] = 0.0, 1.0
        V, BP = ctx.anny_forward(I, shp)
        out = {}
        for k in (0, 1):
            h = BP[k, :, :3, 3]
            mid = 0.5 * (h[b('upperleg01.L')] + h[b('upperleg01.R')])
            out[f'gender_{k}'] = dict(stature_m=extent_along(V[k][ctx.skin_idx], h[b('neck01')] - mid),
                                      shoulder_width_m=float(np.linalg.norm(h[b('upperarm01.L')] - h[b('upperarm01.R')])),
                                      hip_width_m=float(np.linalg.norm(h[b('upperleg01.L')] - h[b('upperleg01.R')])))
        taller = 0 if out['gender_0']['stature_m'] > out['gender_1']['stature_m'] else 1
        wider = 0 if out['gender_0']['shoulder_width_m'] > out['gender_1']['shoulder_width_m'] else 1
        out['male_end'] = taller if taller == wider else None
        return out
    except Exception as e:  # noqa
        return {'male_end': None, 'error': f"{type(e).__name__}: {e}"}


def pick_refit_people(rows, per_seq, seed):
    """Up to per_seq measured people per (split, sequence), seeded; one per actor first, then more frames."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r['split'], r['seq']), []).append(i)
    picks = []
    for key in sorted(by):
        idx = by[key]
        rng = np.random.default_rng([int(seed), zlib.crc32(f"{key[0]}/{key[1]}".encode())])
        order = [idx[j] for j in rng.permutation(len(idx))]
        chosen, seen = [], set()
        for i in order:
            if rows[i]['person'] not in seen:
                chosen.append(i)
                seen.add(rows[i]['person'])
        chosen += [i for i in order if i not in chosen]
        picks += chosen[:max(int(per_seq), 0)]
    return picks


def free_phenotype_mask(ctx, meta):
    """The phenotypes the converter fitted for this person (.npz 'free_phenotypes'), else height + proportions
    (smpl_to_anny.py DEFAULT_FREE_PHENOTYPES, used by run_smpl_to_anny_task.sh)."""
    if meta.get('npz'):
        try:
            d = np.load(meta['npz'], allow_pickle=True)
            fm = np.asarray(d['free_phenotypes']).astype(bool).reshape(-1)
            labels = [str(x) for x in d['phenotype_labels']] if 'phenotype_labels' in d.files else ctx.pheno
            if len(fm) == len(labels):
                free = {lb for lb, f in zip(labels, fm) if f}
                return np.array([p in free for p in ctx.pheno]), 'npz'
        except Exception:  # noqa
            pass
    return np.array([p in ('height', 'proportions') for p in ctx.pheno]), 'default height,proportions'


def gender_refit(ctx, rows, per_seq, iters, seed):
    """I1 refit test. For the picked people, refit the label to the camera-frame 3DPW joints twice with
    smpl_to_anny.fit_frame, from the same warm start (the label's 163 local rotations, root and phenotypes) and with
    the same budget: 'same' = the label gender (control: what the extra iterations alone give), 'flip' = g -> 1-g.
    Height/proportions (the converter's free phenotypes) are fitted again in both, with the converter's regularisers.
    -> one row per person: joint error and surface distance to the gendered SMPL mesh for label / same / flip."""
    S2A = ctx.S2A
    picks = pick_refit_people(rows, per_seq, seed)
    out = []
    t_start = time.time()
    print(f"\n[I1 refit] {len(picks)} people ({per_seq} per sequence), iterations {iters}, 2 refits each", flush=True)
    for n_done, i in enumerate(sorted(picks, key=lambda j: (rows[j]['split'], rows[j]['seq'], rows[j]['frame']))):
        r = rows[i]
        split, seq, fid, p = r['split'], r['seq'], int(r['frame']), int(r['person'])
        rec = dict(split=split, seq=seq, frame=fid, person=p, gender_3dpw=r['gender_3dpw'],
                   gender_label=r['gender_label'], label_joint_mm=r['fit_all_mm'], label_body_mm=r['fit_body_mm'],
                   label_surf_mm=0.5 * (r['a2s_mean'] + r['s2a_mean']), path=r['path'])
        try:
            seqd = ctx.load_seq(split, seq)
            lab = load_pkl(r['path'])
            h = next(h for h in lab['humans'] if int(str(h['name']).rsplit('_p', 1)[1]) == p)
            pose = np.asarray(h['anny_pose'], dtype=np.float32)[:163]
            shape = np.asarray(h['anny_shape'], dtype=np.float32).reshape(-1)
            P = np.asarray(seqd['cam_poses'][fid], dtype=np.float64)
            jp_c = world_to_cam(np.asarray(seqd['jointPositions'][p][fid], dtype=np.float64).reshape(24, 3), P)
            meta = ctx.meta(split, seq, p)
            si, ai = [int(x) for x in meta['smpl_idx']], [int(x) for x in meta['anny_idx']]
            offsets = {s: np.asarray(o, dtype=np.float32) for s, o in zip(si, meta['off']) if np.any(np.asarray(o) != 0)}
            free, free_src = free_phenotype_mask(ctx, meta)
            fb = meta.get('free_bones') if meta.get('free_bones') in ('mapped', 'all') else 'mapped'
            # warm start = the label itself: fit_frame's 6D layout is the first two COLUMNS of each rotation, read
            # row-wise (3x2 -> 6); the label root translation goes to transl (the root bone sits at the origin)
            init = {'rot6d': np.ascontiguousarray(pose[:, :3, :2]).reshape(len(pose), 6),
                    'transl': pose[0, :3, 3].copy()}
            vs_w, _, _ = ctx.smpl_forward(seqd, p, fid)
            vs_c = world_to_cam(vs_w, P)
            nn_smpl = NN(np.concatenate([vs_c, ctx.smpl_sampler.points(vs_c)], 0))
            ts = time.time()
            for tag, g in (('same', float(shape[ctx.gi])), ('flip', 1.0 - float(shape[ctx.gi]))):
                sh0 = shape.copy()
                sh0[ctx.gi] = g
                fr = S2A.fit_frame(ctx.bm, jp_c, ai, si, ctx.device, iters=iters, shape_init=sh0, free_shape=free,
                                   pose_reg=1e-4, shape_reg=1e-3, offsets=offsets, free_bones=fb, init=init)
                P44 = np.tile(np.eye(4, dtype=np.float32), (1, len(ctx.bones), 1, 1))
                P44[0, :, :3, :3] = fr['anny_rotmat']
                V, _ = ctx.anny_forward(P44, np.asarray(fr['anny_shape'], dtype=np.float32)[None])
                a2s, s2a = surface_metrics(ctx, V[0] + np.asarray(fr['transl'], dtype=np.float64), vs_c, nn_smpl)
                pj = np.asarray(fr['per_joint_err_mm'], dtype=np.float64)
                rec.update({f'{tag}_gender': float(fr['anny_shape'][ctx.gi]), f'{tag}_joint_mm': float(fr['joint_err_mm']),
                            f'{tag}_body_mm': nanmean(pj[BODY_MASK]),
                            f'{tag}_surf_mm': 0.5 * (float(a2s.mean()) + float(s2a.mean())),
                            f'{tag}_a2s_mm': float(a2s.mean()), f'{tag}_s2a_mm': float(s2a.mean())})
                for nm in ('height', 'proportions'):
                    if nm in ctx.pheno:
                        rec[f'{tag}_{nm}'] = float(fr['anny_shape'][ctx.pheno.index(nm)])
            rec.update(gain_joint_mm=rec['same_joint_mm'] - rec['flip_joint_mm'],
                       gain_surf_mm=rec['same_surf_mm'] - rec['flip_surf_mm'],
                       flip_better_joint=bool(rec['flip_joint_mm'] < rec['same_joint_mm']),
                       flip_better_surf=bool(rec['flip_surf_mm'] < rec['same_surf_mm']),
                       free_phenotypes=free_src, refit_s=time.time() - ts)
        except Exception as e:  # noqa
            rec['error'] = f"{type(e).__name__}: {e}"
            print(f"  [I1 refit] {split}/{seq} image {fid} p{p}: {rec['error']}", flush=True)
        out.append(rec)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(picks):
            print(f"  [I1 refit] {n_done + 1}/{len(picks)} people ({time.time() - t_start:.0f}s)", flush=True)
    return out


def summarize_refit(rr):
    """Per 3DPW gender ('m', 'f', 'all'): medians of label / same / flip joint and surface error, the paired gain
    same - flip, and the share of people where the flipped-gender refit is better."""
    ok = [r for r in rr if 'error' not in r]
    s = {'n_people': len(ok), 'n_failed': len(rr) - len(ok)}
    for g in ('m', 'f', 'all'):
        sel = [r for r in ok if g == 'all' or r['gender_3dpw'] == g]
        if not sel:
            s[g] = {'n': 0}
            continue
        med = lambda k: float(np.nanmedian(col(sel, k)))  # noqa: E731
        s[g] = dict(n=len(sel), label_gender_mean=float(np.nanmean(col(sel, 'gender_label'))),
                    joint_label_median_mm=med('label_joint_mm'), joint_same_median_mm=med('same_joint_mm'),
                    joint_flip_median_mm=med('flip_joint_mm'), gain_joint_median_mm=med('gain_joint_mm'),
                    surf_label_median_mm=med('label_surf_mm'), surf_same_median_mm=med('same_surf_mm'),
                    surf_flip_median_mm=med('flip_surf_mm'), gain_surf_median_mm=med('gain_surf_mm'),
                    flip_better_joint_frac=float(np.mean([r['flip_better_joint'] for r in sel])),
                    flip_better_surf_frac=float(np.mean([r['flip_better_surf'] for r in sel])),
                    flip_better_both_frac=float(np.mean([r['flip_better_joint'] and r['flip_better_surf']
                                                         for r in sel])))
    return s


def refit_brief(s):
    if not s or not s.get('n_people'):
        return 'not measured' if not s else f"not measured ({s.get('n_failed', 0)} refits failed)"
    keys = ('n', 'label_gender_mean', 'joint_label_median_mm', 'joint_same_median_mm', 'joint_flip_median_mm',
            'gain_joint_median_mm', 'gain_surf_median_mm', 'flip_better_joint_frac', 'flip_better_surf_frac')
    return {g: {k: s[g].get(k) for k in keys} for g in ('m', 'f') if s[g].get('n')}


def print_gender(probe, gsum):
    print("\n=== I1 GENDER ===")
    if probe.get('error'):
        print(f"convention probe failed: {probe['error']}")
    elif 'gender_0' in probe:
        g0, g1 = probe['gender_0'], probe['gender_1']
        print(f"Anny rest body, gender 0: stature {g0['stature_m']:.3f} m, shoulders {g0['shoulder_width_m']:.3f} m | "
              f"gender 1: stature {g1['stature_m']:.3f} m, shoulders {g1['shoulder_width_m']:.3f} m  -> male end = "
              f"{probe['male_end'] if probe['male_end'] is not None else 'unclear'}")
    s = (gsum or {}).get('overall')
    if not s or not s.get('n_people'):
        print("refit test: not measured" + (f" ({s.get('n_failed')} failed)" if s else ''))
        return
    print(f"refit test ({s['n_people']} people, {s['n_failed']} failed; joint = mean over the mapped fit points, "
          f"surface = mean of both directions vs the gendered SMPL mesh; medians):")
    print(f"  {'3DPW':<5}{'n':>5}{'label g':>9}{'joint label/same/flip mm':>27}{'gain':>7}{'surf same/flip mm':>20}"
          f"{'gain':>7}{'flip better: joint / surf':>27}")
    for g in ('m', 'f'):
        d = s[g]
        if not d.get('n'):
            continue
        print(f"  {g:<5}{d['n']:>5}{d['label_gender_mean']:>9.3f}"
              f"{fmt(d['joint_label_median_mm']) + '/' + fmt(d['joint_same_median_mm']) + '/' + fmt(d['joint_flip_median_mm']):>27}"
              f"{fmt(d['gain_joint_median_mm']):>7}"
              f"{fmt(d['surf_same_median_mm']) + '/' + fmt(d['surf_flip_median_mm']):>20}{fmt(d['gain_surf_median_mm']):>7}"
              f"{fmt(d['flip_better_joint_frac'], 2) + ' / ' + fmt(d['flip_better_surf_frac'], 2):>27}")


# --------------------------------------------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------------------------------------------- #
def col(rows, k):
    return np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows], dtype=np.float64)


def summarize_2d(px, mmpp):
    """V3 numbers. px: {'raw'|'smplfwd'|'anny_fit'|'anny_head': [N, 13] pixel distance to OpenPose, NaN = no point};
    mmpp: [N, 13] mm per pixel at that joint's depth (3DPW joint depth / fx). Only points where both 'raw' and
    'anny_fit' exist count, so the medians compare the same points. 'mm2d' = px x mmpp = the sideways miss in mm at
    the joint, whatever the distance to the camera (a 25 mm OpenPose-vs-SMPL gap is ~7 px at 7 m, ~20 px at 2.5 m)."""
    px = {k: np.asarray(v, dtype=np.float64) for k, v in px.items()}
    mmpp = np.asarray(mmpp, dtype=np.float64)
    both = np.isfinite(px['raw']) & np.isfinite(px['anny_fit'])
    bm = both & np.isfinite(mmpp)
    mm = {k: v * mmpp for k, v in px.items()}

    def med_diff(a, b, m):
        return float(np.median(a[m]) - np.median(b[m])) if m.any() else None
    s = {'px_points': {k: stats(v[both]) for k, v in px.items()},
         'px_points_all_valid': {k: stats(v) for k, v in px.items()},
         'px_delta_fit_minus_raw': stats(px['anny_fit'][both] - px['raw'][both]),
         'px_delta_median_of_medians': med_diff(px['anny_fit'], px['raw'], both),
         'px_delta_head_minus_raw_median': med_diff(px['anny_head'], px['raw'], both & np.isfinite(px['anny_head'])),
         'mm_per_px_median': float(np.median(mmpp[bm])) if bm.any() else None,
         'mm2d_points': {k: stats(v[bm]) for k, v in mm.items()},
         'mm2d_delta_fit_minus_raw': stats(mm['anny_fit'][bm] - mm['raw'][bm]),
         'mm2d_delta_median_of_medians': med_diff(mm['anny_fit'], mm['raw'], bm)}
    return s


def summarize(rows, extra, imgs, bones=None):
    if not rows:
        return {'n_people': 0, 'n_images': len(imgs)}
    d_fit = np.stack([e['d_fit'] for e in extra])
    d_head = np.stack([e['d_head'] for e in extra])
    px = {k: np.stack([e['px'][k] for e in extra]) for k in ('raw', 'smplfwd', 'anny_fit', 'anny_head')}
    both = np.isfinite(px['raw']) & np.isfinite(px['anny_fit'])
    s = {'n_people': len(rows), 'n_images': len(imgs), 'n_sequences': len({(r['split'], r['seq']) for r in rows})}
    for k in ('fit_all_mm', 'fit_body_mm', 'head_all_mm', 'head_body_mm', 'fit_max_mm', 'smpl_selfcheck_mm',
              'a2s_mean', 'a2s_p95', 's2a_mean', 's2a_p95', 'stature_ratio',
              'depth_m', 'px_raw', 'px_anny_fit', 'mm2d_raw', 'mm2d_anny_fit', 'age_label', 'height_label',
              'proportions_label',
              'n_identity_bones'):
        s[k] = stats(col(rows, k))
    s['saved_minus_recomputed_abs_mm'] = stats(np.abs(col(rows, 'saved_minus_recomputed_mm')))
    a = np.abs(col(rows, 'saved_minus_recomputed_mm'))
    a = a[np.isfinite(a)]
    s['saved_minus_recomputed_abs_p99_mm'] = float(np.percentile(a, 99)) if a.size else None
    s['s2a_region_mm'] = {r: stats(col(rows, f's2a_{r}')) for r in SMPL_REGIONS}
    s['segment_ratio'] = {g: stats(col(rows, f'ratio_{g}')) for g in SEGMENTS}
    s.update(summarize_2d(px, np.stack([e['mmpp'] for e in extra])))
    lr = [r['lr_ok'] for r in rows if r['lr_ok'] is not None]
    mo = [r['mirror_ok'] for r in rows if r['mirror_ok'] is not None]
    s['lr_pass_frac'] = float(np.mean(lr)) if lr else None
    s['mirror_pass_frac'] = float(np.mean(mo)) if mo else None
    fm = col(rows, 'fit_max_mm')
    s['frac_any_joint_gt100mm'] = float(np.mean(fm[np.isfinite(fm)] > THR['bad_frame_mm'])) if np.isfinite(fm).any() else None
    s['per_joint_fit_mm'] = {SMPL_JOINT_NAMES[j]: stats(d_fit[:, j]) for j in range(24)}
    s['per_joint_head_mm'] = {SMPL_JOINT_NAMES[j]: stats(d_head[:, j]) for j in range(24)}
    s['per_keypoint_px_median'] = {nm: {k: (float(np.median(v[both[:, i], i])) if both[:, i].any() else None)
                                        for k, v in px.items()} for i, (_, _, nm) in enumerate(OPENPOSE_TO_SMPL)}
    # gender (informational; the refit test is summarised separately: summarize_refit)
    g3 = np.array([r['gender_3dpw'] for r in rows])
    gl = col(rows, 'gender_label')
    s['gender'] = {g: {'n': int((g3 == g).sum()),
                       'label_gender_mean': float(gl[g3 == g].mean()) if (g3 == g).any() else None}
                   for g in ('m', 'f')}
    s['forward_angle_deg'] = {nm: stats(col(rows, f'ang_{nm}')) for nm, _ in FWD_PARTS}
    s['local_angle_deg'] = {b: stats(col(rows, f'local_deg_{b}')) for b in NECK_BONES if f'local_deg_{b}' in rows[0]}
    idm = np.stack([e['ident'] for e in extra])                     # [N, 162]: bones 1..162, exactly identity
    bones = list(bones or [])
    if len(bones) == idm.shape[1] + 1:
        s['identity_frac_by_bone'] = {bones[i + 1]: float(idm[:, i].mean()) for i in range(idm.shape[1])}
        s['bones_identity_in_every_person'] = [bones[i + 1] for i in np.where(idm.all(0))[0]]
    else:
        s['identity_frac_by_bone'], s['bones_identity_in_every_person'] = {}, []
    s['headturn_vs_chest_deg'] = {'smpl': stats(col(rows, 'headturn_smpl')), 'anny': stats(col(rows, 'headturn_anny'))}
    # coverage (sample-based)
    nh = {}
    for im in imgs:
        nh[im['n_humans']] = nh.get(im['n_humans'], 0) + 1
    two = [im for im in imgs if im['n_seq_people'] >= 2]
    s['coverage'] = {'humans_per_image': dict(sorted(nh.items())),
                     'two_person_seq_images': len(two),
                     'two_person_seq_images_with_unlabelled_valid_actor': int(sum(im['unlabelled_valid'] > 0 for im in two)),
                     'two_person_seq_images_with_unfitted_valid_actor': int(sum(
                         im.get('unlabelled_causes', {}).get('not_fitted', 0) > 0 for im in two)),
                     'two_person_seq_images_with_filter_dropped_actor': int(sum(
                         any(im.get('unlabelled_causes', {}).get(c, 0) > 0 for c in ('dropped_err', 'dropped_vis',
                                                                                    'dropped_box')) for im in two)),
                     'unlabelled_valid_actor_causes': {c: int(sum(im.get('unlabelled_causes', {}).get(c, 0)
                                                                  for im in imgs)) for c in UNLABELLED_CAUSES},
                     'train_py_would_drop_frac': float(np.mean([not r['train_kept'] for r in rows])),
                     'images_missing_on_disk': int(sum(not im.get('image_found', True) for im in imgs)),
                     'label_K_vs_3dpw_K_max_diff': float(max([im.get('K_label_diff', 0.0) for im in imgs] or [0.0])),
                     'smpl_model_kinds': {k: int(sum(r['smpl_model'] == k for r in rows))
                                          for k in sorted({r['smpl_model'] for r in rows})},
                     'mapping_sources': {k: int(sum(r['mapping'] == k for r in rows))
                                         for k in sorted({r['mapping'] for r in rows})},
                     'converter_versions': {k: int(sum(r['version'] == k for r in rows))
                                            for k in sorted({r['version'] for r in rows})}}
    return s


def fits_brief(f):
    """The I4 numbers that go into the verdicts (the full per-sequence table stays in summary.json)."""
    if not f or not f.get('n_npz'):
        return {'n_npz': 0, 'note': 'no converter .npz under --fits: fit error before the builder filters unknown',
                'sequences_in_3dpw_without_npz': (f or {}).get('sequences_in_3dpw_without_npz'),
                'tasks_without_npz': (f or {}).get('tasks_without_npz')}
    e = f.get('joint_err_mm') or {}
    pj = f.get('per_joint_mm') or {}
    return dict(n_npz=f['n_npz'], n_npz_unusable=f.get('n_npz_unusable'), fitted_frames=f.get('fitted_frames'),
                err_mean_mm=e.get('mean'), err_median_mm=e.get('median'), err_p95_mm=e.get('p95'),
                err_max_mm=e.get('max'), frac_err_gt_30mm=f.get('frac_err_gt_good'),
                frac_err_gt_80mm=f.get('frac_err_gt_max_err'), builder_filter_counts=f.get('builder_filter_counts'),
                per_joint_median_mm={k: d.get('median') for k, d in pj.items()},
                expected_label_images=f.get('expected_label_images'),
                label_images_in_index=f.get('label_images_in_index'),
                sequences_expected_ne_index=f.get('sequences_expected_ne_index'),
                sequences_in_3dpw_without_npz=f.get('sequences_in_3dpw_without_npz'),
                tasks_without_npz=f.get('tasks_without_npz'), versions=f.get('versions'))


def verdicts(s, loader, fits=None, gender=None):
    """gender: {'probe': gender_convention_probe(), 'refit': summarize_refit() of this split} (I1, informational)."""
    if not s or not s.get('n_people'):
        v = {'overall': 'NOT MEASURED', 'not_measured': ['all: no labelled person measured']}
        if fits is not None:
            v['I4_converter_fits'] = fits_brief(fits)
        return v
    v = {}
    med = lambda d: d.get('median') if d else None  # noqa: E731
    sc = med(s['smpl_selfcheck_mm'])
    v['V0_gt_selfcheck'] = dict(median_mm=sc, grade=grade(sc, *THR['selfcheck']))
    fb = s['fit_body_mm']
    g1 = worst([grade(fb['median'], *THR['fit_median']), grade(fb['p95'], *THR['fit_p95'])])
    v['V1_joint_fit_body'] = dict(median_mm=fb['median'], p95_mm=fb['p95'], mean_mm=fb['mean'], grade=g1,
                                  bone_heads_body_median_mm=med(s['head_body_mm']))
    p99 = s.get('saved_minus_recomputed_abs_p99_mm')
    v['V1b_saved_vs_recomputed'] = dict(p99_abs_mm=p99, grade=grade(p99, *THR['saved']))
    a2s, s2a = med(s['a2s_mean']), med(s['s2a_mean'])
    v['V2_surface'] = dict(anny_to_smpl_median_mm=a2s, smpl_to_anny_median_mm=s2a,
                           grade=grade(None if a2s is None or s2a is None else max(a2s, s2a), *THR['surf']),
                           worst_region=max(((r, d['median']) for r, d in s['s2a_region_mm'].items()
                                             if d['median'] is not None), key=lambda x: x[1], default=(None, None)))
    # graded in mm at the joint depth ('mm2d'), not in pixels: pixel errors grow as the actor comes closer
    m2 = s.get('mm2d_points') or {}
    raw_mm = (m2.get('raw') or {}).get('median')
    dl_mm = s.get('mm2d_delta_median_of_medians')
    raw_ok = None if raw_mm is None or not np.isfinite(raw_mm) else bool(raw_mm <= THR['raw_mm2d_sane'])
    if raw_ok is None:
        g3 = 'NOT MEASURED'
    elif raw_ok:
        g3 = grade(dl_mm, *THR['mm2d_delta'])
    else:
        g3 = 'FAIL (raw 3DPW joints miss OpenPose too: camera / 2D data / image index, not the Anny fit)'
    pxp = s.get('px_points') or {}
    v['V3_2d'] = dict(raw_median_mm2d=raw_mm, anny_fit_median_mm2d=(m2.get('anny_fit') or {}).get('median'),
                      delta_mm2d=dl_mm, raw_ok=raw_ok, raw_median_px=(pxp.get('raw') or {}).get('median'),
                      anny_fit_median_px=(pxp.get('anny_fit') or {}).get('median'),
                      anny_head_median_px=(pxp.get('anny_head') or {}).get('median'),
                      delta_px=s.get('px_delta_median_of_medians'), mm_per_px=s.get('mm_per_px_median'), grade=g3)
    v['V4a_left_right'] = dict(pass_frac=s['lr_pass_frac'], grade=grade_min(s['lr_pass_frac'], *THR['lr']))
    v['V4b_mirror'] = dict(pass_frac=s['mirror_pass_frac'], grade=grade_min(s['mirror_pass_frac'], *THR['mirror']))
    seg = {g: d['median'] for g, d in s['segment_ratio'].items()}
    v['V4c_limb_ratio'] = dict(medians=seg, grade=worst([grade_range(x, *THR['ratio']) for x in seg.values()]))
    st = med(s['stature_ratio'])
    v['V4d_stature_ratio'] = dict(median=st, grade=grade_range(st, *THR['stature']))
    fr = s['frac_any_joint_gt100mm']
    v['V4e_bad_frames'] = dict(frac_people_any_joint_gt100mm=fr, grade=grade(fr, *THR['bad_frames']))
    v['V5_loader'] = dict(loader.get('verdict_values', {}), grade=loader.get('grade', 'NOT MEASURED'))
    gender = gender or {}
    probe = gender.get('probe') or {}
    v['I1_gender'] = dict(label_gender_mean={g: s['gender'][g]['label_gender_mean'] for g in ('m', 'f')},
                          n_people={g: s['gender'][g]['n'] for g in ('m', 'f')},
                          anny_male_end=probe.get('male_end', 'not measured'),
                          refit=refit_brief(gender.get('refit')),
                          note="Anny's code says gender 0 = male, 1 = female (model_data / shape_distribution); "
                               "'anny_male_end' checks it on the rest bodies. refit = the label refitted with its own "
                               "gender ('same', control) and with g -> 1-g ('flip'), same warm start and budget: flip "
                               "lower in joint error and surface for most people = the label gender is wrong. A gain "
                               "near 0 means 'not shown', not 'not reversed'.")
    v['I2_head'] = dict(forward_angle_median_deg={k: d['median'] for k, d in s['forward_angle_deg'].items()},
                        headturn_vs_chest_median_deg={k: d['median'] for k, d in s['headturn_vs_chest_deg'].items()},
                        headturn_vs_chest_p95_deg={k: d['p95'] for k, d in s['headturn_vs_chest_deg'].items()},
                        local_angle_median_deg={k: d['median'] for k, d in s['local_angle_deg'].items()},
                        identity_frac={b: s['identity_frac_by_bone'].get(b) for b in NECK_BONES},
                        n_bones_identity_in_every_person=len(s['bones_identity_in_every_person']))
    v['I3_coverage'] = s['coverage']
    if fits is not None:
        v['I4_converter_fits'] = fits_brief(fits)
    graded = {k: x['grade'] for k, x in v.items() if k.startswith('V')}
    v['overall'] = worst(list(graded.values()))
    v['not_measured'] = [k for k, g in graded.items() if grade_word(g) == 'NOT MEASURED']
    return v


# --------------------------------------------------------------------------------------------------------------- #
# loader check (item 5)
# --------------------------------------------------------------------------------------------------------------- #
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def tf32_gap(ctx, pose, shp, Ks):
    """How far training's TF32 decode moves the GT joints/vertices from the FP32 decode used in this script (same
    poses, same shapes). Only on CUDA: TF32 is a CUDA matmul mode. Ks: loader K per person (672 px)."""
    torch = ctx.torch
    if not str(ctx.device).startswith('cuda'):
        return {'measured': False, 'why': f'device {ctx.device}: TF32 is a CUDA matmul mode, nothing to compare'}
    mm, cd = torch.backends.cuda.matmul, torch.backends.cudnn
    old = (mm.allow_tf32, cd.allow_tf32)
    try:
        mm.allow_tf32, cd.allow_tf32 = False, False
        V0, B0 = ctx.anny_forward(pose, shp)
        mm.allow_tf32, cd.allow_tf32 = True, True          # train.py main() with --tf32 1
        V1, B1 = ctx.anny_forward(pose, shp)
    finally:
        mm.allow_tf32, cd.allow_tf32 = old
    dj = np.linalg.norm(B1[:, :, :3, 3] - B0[:, :, :3, 3], axis=-1) * 1000.0
    dv = np.linalg.norm(V1 - V0, axis=-1) * 1000.0
    dpx = np.stack([np.linalg.norm(project(B1[i, :, :3, 3], Ks[i]) - project(B0[i, :, :3, 3], Ks[i]), axis=-1)
                    for i in range(len(Ks))])
    return {'measured': True, 'n_people': int(len(pose)), 'joint_mm': stats(dj), 'root_mm': stats(dj[:, 0]),
            'vert_mm': stats(dv), 'joint_px_672': stats(dpx)}


def loader_check(ctx, imgs, n_per_split):
    """Load a few sampled label files through datasets/threedpw_anny.py + datasets/bedlam.collate_fn and check what
    the loader changes: anny_pose/anny_shape must pass through unchanged, the human order must be kept, the image
    must be the contain+pad of the photo, and the loader's 672 K must map original-image pixels to the same 672
    pixel (utils.perspective_projection, as prepare_gt). The Anny decode is this script's copy of prepare_gt (FP32);
    train.py itself is not run. On CUDA the TF32-vs-FP32 gap of that decode is reported (not graded)."""
    torch = ctx.torch
    args = ctx.args
    out = {'items': [], 'max_pose_abs_diff': 0.0, 'max_shape_abs_diff': 0.0, 'batch_vs_single_joint_mm': 0.0,
           'batch_vs_single_vert_mm': 0.0, 'max_2d_diff_px': 0.0, 'max_image_mean_abs_diff': 0.0,
           'max_pad_offset_diff_px': 0.0}
    all_pose, all_shp, all_K = [], [], []
    try:
        from PIL import Image, ImageOps
        from datasets.threedpw_anny import ThreeDPWAnny
        from datasets.bedlam import collate_fn
        from utils import perspective_projection, denormalize_rgb
    except Exception as e:  # noqa
        out['error'] = f"import failed: {type(e).__name__}: {e}"
        out['grade'] = 'NOT MEASURED'
        return out
    for split in sorted({im['split'] for im in imgs}):
        recs = [im for im in imgs if im['split'] == split and im['n_humans'] > 0]
        files = [im['path'] for im in recs]
        if not files or n_per_split <= 0:
            continue
        try:
            ds = ThreeDPWAnny(args.labels, split=split, img_size=args.img_size, training=False)
        except Exception as e:  # noqa
            out.setdefault('errors', []).append(f"{split}: {type(e).__name__}: {e}")
            continue
        pos = {p: i for i, p in enumerate(ds.pkl_files)}
        cand = [f for f in files if f in pos]
        if not cand:
            continue
        pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(n_per_split, len(cand))).astype(int))]
        multi = [im['path'] for im in recs if im['n_humans'] > 1 and im['path'] in pos and im['path'] not in pick][:1]
        pick = pick + multi                       # one multi-person image when there is one (tests the collate padding)
        items = [ds[pos[f]] for f in pick]
        img, y = collate_fn(items)
        valid = y['valid_humans']
        idx = torch.where(valid > 0)
        pose = y['anny_pose'][idx[0], idx[1]][:, :163]
        shp = y['anny_shape'][idx[0], idx[1]]
        Kl = y['K'][idx[0]].to(ctx.device)
        all_pose.append(pose.numpy())
        all_shp.append(shp.numpy())
        all_K.append(y['K'][idx[0]].double().numpy())
        V_l, BP_l = ctx.anny_forward(pose.numpy(), shp.numpy())
        jts_l = torch.as_tensor(BP_l[:, :, :3, 3], dtype=torch.float32, device=ctx.device)
        j2d_l = perspective_projection(jts_l, Kl).cpu().numpy()
        k = 0
        for b, f in enumerate(pick):
            lab = load_pkl(f)
            im = Image.open(lab['imagename']).convert('RGB')
            W, H = im.size
            s = args.img_size / float(max(W, H))
            direct = np.asarray(ImageOps.pad(ImageOps.contain(im, (args.img_size, args.img_size)),
                                             size=(args.img_size, args.img_size)))
            loader_img = denormalize_rgb(img[b].numpy())
            img_diff = float(np.abs(direct.astype(np.float32) - loader_img.astype(np.float32)).mean())
            w_s = ImageOps.contain(im, (args.img_size, args.img_size)).size
            pad_px = (round((args.img_size - w_s[0]) * 0.5), round((args.img_size - w_s[1]) * 0.5))
            pad_k = ((args.img_size - w_s[0]) / 2.0, (args.img_size - w_s[1]) / 2.0)
            off = (args.img_size - W * s) / 2.0, (args.img_size - H * s) / 2.0
            K = np.array([[lab['focal'][0], 0, lab['princpt'][0]], [0, lab['focal'][1], lab['princpt'][1]], [0, 0, 1.0]])
            for hk, h in enumerate(lab['humans']):
                Vd, BPd = ctx.anny_forward(np.asarray(h['anny_pose'], dtype=np.float32)[None, :163],
                                           np.asarray(h['anny_shape'], dtype=np.float32)[None])
                jd = BPd[0, :, :3, 3]
                uv = project(jd, K)
                uv_l = np.stack([uv[:, 0] * s + off[0], uv[:, 1] * s + off[1]], -1)
                shp_lab = np.asarray(h['anny_shape'], np.float32).reshape(-1)
                item = dict(label=f, human=str(h['name']),
                            loader_name_order_ok=bool(items[b][1]['humans'][hk]['name'] == h['name']),
                            pose_abs_diff=float(np.abs(pose[k].numpy() - np.asarray(h['anny_pose'], np.float32)[:163]).max()),
                            shape_abs_diff=float(np.abs(shp[k].numpy().reshape(-1)[:len(shp_lab)] - shp_lab).max()),
                            batch_vs_single_joint_mm=float(np.abs(BP_l[k, :, :3, 3] - jd).max() * 1000.0),
                            batch_vs_single_vert_mm=float(np.abs(V_l[k] - Vd[0]).max() * 1000.0),
                            px_2d_diff=float(np.nanmax(np.abs(j2d_l[k] - uv_l))),
                            image_mean_abs_diff=img_diff,
                            pad_offset_diff_px=float(max(abs(pad_px[0] - pad_k[0]), abs(pad_px[1] - pad_k[1]))),
                            root_depth_m=float(jd[0, 2]))
                out['items'].append(item)
                out['max_pose_abs_diff'] = max(out['max_pose_abs_diff'], item['pose_abs_diff'])
                out['max_shape_abs_diff'] = max(out['max_shape_abs_diff'], item['shape_abs_diff'])
                out['batch_vs_single_joint_mm'] = max(out['batch_vs_single_joint_mm'], item['batch_vs_single_joint_mm'])
                out['batch_vs_single_vert_mm'] = max(out['batch_vs_single_vert_mm'], item['batch_vs_single_vert_mm'])
                out['max_2d_diff_px'] = max(out['max_2d_diff_px'], item['px_2d_diff'])
                out['max_image_mean_abs_diff'] = max(out['max_image_mean_abs_diff'], img_diff)
                out['max_pad_offset_diff_px'] = max(out['max_pad_offset_diff_px'], item['pad_offset_diff_px'])
                k += 1
        print(f"[loader] {split}: {len(pick)} label files / {k} people through ThreeDPWAnny + collate_fn", flush=True)
    if not out['items']:
        out['grade'] = 'NOT MEASURED'
        if 'errors' in out and 'error' not in out:
            out['error'] = '; '.join(out['errors'])
        return out
    names_ok = all(i['loader_name_order_ok'] for i in out['items'])
    ok = (out['max_pose_abs_diff'] <= THR['loader_pose'] and out['max_shape_abs_diff'] <= THR['loader_pose']
          and out['max_2d_diff_px'] <= THR['loader_px'] and out['max_image_mean_abs_diff'] <= THR['loader_img']
          and names_ok)
    out['grade'] = 'PASS' if ok else 'FAIL'
    try:
        out['tf32_gap'] = tf32_gap(ctx, np.concatenate(all_pose), np.concatenate(all_shp), np.concatenate(all_K))
    except Exception as e:  # noqa
        out['tf32_gap'] = {'measured': False, 'why': f"{type(e).__name__}: {e}"}
    tg = out['tf32_gap']
    out['verdict_values'] = {k: out[k] for k in ('max_pose_abs_diff', 'max_shape_abs_diff', 'max_2d_diff_px',
                                                  'max_image_mean_abs_diff', 'max_pad_offset_diff_px',
                                                  'batch_vs_single_joint_mm', 'batch_vs_single_vert_mm')}
    out['verdict_values']['n_people'] = len(out['items'])
    out['verdict_values']['human_order_ok'] = names_ok
    out['verdict_values']['tf32_gap_joint_max_mm'] = tg['joint_mm']['max'] if tg.get('measured') else None
    out['verdict_values']['tf32_gap_vert_max_mm'] = tg['vert_mm']['max'] if tg.get('measured') else None
    out['verdict_values']['tf32_gap_joint_max_px672'] = tg['joint_px_672']['max'] if tg.get('measured') else None
    out['verdict_values']['tf32_gap'] = 'measured' if tg.get('measured') else f"not measured ({tg.get('why')})"
    out['verdict_values']['scope'] = ('loader pass-through + K/padding; Anny decode = FP32 copy of train.py '
                                      'prepare_gt, train.py not run')
    return out


# --------------------------------------------------------------------------------------------------------------- #
# renders (item 6)
# --------------------------------------------------------------------------------------------------------------- #
_RENDER_MODE = {'mode': None}


def overlay_meshes(img_u8, verts_list, faces, K, color):
    """Mesh overlay with render_anny_fit.render_overlay (pyrender, real intrinsics); projected-vertex dots when
    pyrender/EGL fails (the first failure switches to dots for the rest of the run)."""
    out = img_u8
    if _RENDER_MODE['mode'] != 'dots':
        try:
            import render_anny_fit as raf
            for v in verts_list:
                out = raf.render_overlay(out, np.asarray(v, dtype=np.float64), faces, K, color=color, alpha=0.65)
            _RENDER_MODE['mode'] = 'pyrender'
            return out
        except Exception as e:  # noqa
            print(f"[render] pyrender overlay failed ({type(e).__name__}: {e}); drawing projected vertices instead",
                  flush=True)
            _RENDER_MODE['mode'] = 'dots'
            out = img_u8
    out = out.copy()
    h, w = out.shape[:2]
    c = np.array([int(255 * x) for x in color], dtype=np.uint8)
    on_face = np.unique(np.asarray(faces).reshape(-1))         # skip helper vertices that are on no face
    for v in verts_list:
        uv = project(np.asarray(v)[on_face][::2], K)
        ok = np.isfinite(uv).all(1)
        u, vv = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        inside = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        out[vv[inside], u[inside]] = c
    return out


def render_examples(ctx, picks, vis_dir, scale):
    import diag_common as dc
    import render_anny_fit as raf
    from PIL import Image
    os.makedirs(vis_dir, exist_ok=True)
    faces_anny = ctx.faces_tri_all
    written = []
    for n_out, (split, seq, path, why) in enumerate(picks):
        try:
            seqd = ctx.load_seq(split, seq)
            lab = load_pkl(path)
            _, fid = parse_label_path(path)
            im = Image.open(lab['imagename']).convert('RGB')
            W, H = im.size
            im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))))
            arr = np.asarray(im).copy()
            K = np.asarray(seqd['cam_intrinsics'], dtype=np.float64).copy()
            Ks = K.copy()
            Ks[:2] *= scale
            P = np.asarray(seqd['cam_poses'][fid], dtype=np.float64)
            vs_list, va_list, jp_px, fit_px, op_px, texts = [], [], [], [], [], []
            humans = lab['humans']
            pose = np.stack([np.asarray(h['anny_pose'], np.float32)[:163] for h in humans])
            shape = np.stack([np.asarray(h['anny_shape'], np.float32).reshape(-1) for h in humans])
            V, BP = ctx.anny_forward(pose, shape)
            for k, h in enumerate(humans):
                p = int(str(h['name']).rsplit('_p', 1)[1])
                meta = ctx.meta(split, seq, p)
                vs_w, _, _ = ctx.smpl_forward(seqd, p, fid)
                vs_list.append(world_to_cam(vs_w, P))
                va_list.append(V[k])
                jp_c = world_to_cam(np.asarray(seqd['jointPositions'][p][fid], np.float64).reshape(24, 3), P)
                fit = np.full((24, 3), np.nan)
                for si, ai, off in zip(meta['smpl_idx'], meta['anny_idx'], meta['off']):
                    fit[si] = BP[k, ai, :3, 3] + BP[k, ai, :3, :3] @ off
                jp_px.append(project(jp_c, Ks))
                fit_px.append(project(fit, Ks))
                d = nanmean(np.linalg.norm(fit - jp_c, axis=-1) * 1000.0)
                texts.append(f"p{p}: joint {d:.0f} mm")
                if 'poses2d' in seqd:
                    op = np.asarray(seqd['poses2d'][p][fid], dtype=np.float64)
                    op = op.T if op.shape == (3, 18) else op
                    op_px.append(op[op[:, 2] > ctx.args.op_conf, :2] * scale)
            p_smpl = overlay_meshes(arr, vs_list, ctx.smpl_faces, Ks, (0.95, 0.55, 0.2))
            p_anny = overlay_meshes(arr, va_list, faces_anny, Ks, (0.3, 0.6, 1.0))
            for o in op_px:
                p_smpl = dc.draw_points(p_smpl, o, color=(255, 255, 0), r=3)
            for g, ft in zip(jp_px, fit_px):
                p_anny = raf.draw_joints(p_anny, g, ft, radius=3)
            photo = dc.label(arr, f"{split}/{seq} image {fid} ({why})")
            p_smpl = dc.label(p_smpl, "3DPW SMPL GT mesh (orange), OpenPose 2D (yellow)")
            p_anny = dc.label(p_anny, "Anny label mesh (blue); joints red=SMPL green=Anny; " + ", ".join(texts))
            outp = os.path.join(vis_dir, f"{n_out:02d}_{split}_{seq}_{fid:05d}.jpg")
            Image.fromarray(dc.hstack([photo, p_smpl, p_anny])).save(outp, quality=88)
            written.append(outp)
        except Exception as e:  # noqa
            print(f"[render] {path}: {type(e).__name__}: {e}", flush=True)
    print(f"[render] wrote {len(written)} panels to {vis_dir} (mode: {_RENDER_MODE['mode']})", flush=True)
    return written


def pick_renders(rows, imgs, n_vis):
    """~n_vis images: the 3 worst people by max joint error, 2 two-person images, the rest evenly spread over the
    sample (one per sequence first), in a fixed order."""
    if n_vis <= 0 or not rows:
        return []
    picks, seen = [], set()

    def add(split, seq, path, why):
        if path not in seen and len(picks) < n_vis:
            seen.add(path)
            picks.append((split, seq, path, why))
    for r in sorted(rows, key=lambda r: -(r['fit_max_mm'] if np.isfinite(r['fit_max_mm']) else -1))[:3]:
        add(r['split'], r['seq'], r['path'], 'worst joint error')
    for im in [im for im in imgs if im['n_humans'] >= 2][:2]:
        add(im['split'], im['seq'], im['path'], 'two people')
    seqs = sorted({(im['split'], im['seq']) for im in imgs})
    step = max(1, len(seqs) // max(1, n_vis - len(picks)))
    for split, seq in seqs[::step]:
        cand = [im for im in imgs if im['split'] == split and im['seq'] == seq and im['n_humans'] > 0]
        if cand:
            add(split, seq, cand[len(cand) // 2]['path'], 'typical')
    return picks


# --------------------------------------------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------------------------------------------- #
def print_tables(summ):
    names = [k for k in summ if summ[k].get('n_people')]
    if not names:
        print("no people measured")
        return
    w = 22

    def line(title, getter, nd=1):
        cells = []
        for n in names:
            try:
                d = getter(summ[n])
                if isinstance(d, dict):
                    cells.append(f"{fmt(d['mean'], nd)}/{fmt(d['median'], nd)}/{fmt(d['p95'], nd)}")
                elif isinstance(d, str):
                    cells.append(d)
                else:
                    cells.append(fmt(None if d is None else float(d), nd))
            except Exception:  # noqa
                cells.append('-')
        print(f"{title:<44}" + "".join(f"{c:>{w}}" for c in cells))
    print("\n=== 3DPW -> ANNY CONVERSION CHECK (mean/median/p95; people unless 'points') ===")
    print(f"{'':<44}" + "".join(f"{n:>{w}}" for n in names))
    line('people / images / sequences', lambda s: f"{s['n_people']}/{s['n_images']}/{s['n_sequences']}")
    line('V0 SMPL fwd vs jointPositions (mm)', lambda s: s['smpl_selfcheck_mm'])
    line('V1 joint err FIT points, all 24 (mm)', lambda s: s['fit_all_mm'])
    line('V1 joint err FIT points, body (mm)', lambda s: s['fit_body_mm'])
    line('   joint err bone HEADS, body (mm)', lambda s: s['head_body_mm'])
    line('   max joint err per person (mm)', lambda s: s['fit_max_mm'])
    line('V1b |saved - recomputed| (mm)', lambda s: s['saved_minus_recomputed_abs_mm'], 2)
    line('V2 surface Anny->SMPL, mean (mm)', lambda s: s['a2s_mean'])
    line('V2 surface SMPL->Anny, mean (mm)', lambda s: s['s2a_mean'])
    line('   surface Anny->SMPL, p95 (mm)', lambda s: s['a2s_p95'])
    for r in SMPL_REGIONS:
        line(f'   SMPL->Anny region {r} (mm)', lambda s, r=r: s['s2a_region_mm'][r])
    line('V3 2D raw 3DPW joints (points, px)', lambda s: s['px_points']['raw'])
    line('V3 2D SMPL forward joints (points, px)', lambda s: s['px_points']['smplfwd'])
    line('V3 2D Anny FIT points (points, px)', lambda s: s['px_points']['anny_fit'])
    line('V3 2D Anny bone heads (points, px)', lambda s: s['px_points']['anny_head'])
    line('V3 2D Anny fit - raw, paired (px)', lambda s: s['px_delta_fit_minus_raw'])
    line('V3 2D raw 3DPW joints (points, mm2d)', lambda s: s['mm2d_points']['raw'])
    line('V3 2D Anny FIT points (points, mm2d)', lambda s: s['mm2d_points']['anny_fit'])
    line('V3 2D Anny fit - raw, paired (mm2d)', lambda s: s['mm2d_delta_fit_minus_raw'])
    line('   mm per px at the joint depth', lambda s: s['mm_per_px_median'], 2)
    line('   root depth (m)', lambda s: s['depth_m'], 2)
    line('V4a left/right pass fraction', lambda s: s['lr_pass_frac'], 3)
    line('V4b mirror pass fraction', lambda s: s['mirror_pass_frac'], 3)
    for g in SEGMENTS:
        line(f'V4c length ratio {g} Anny/SMPL', lambda s, g=g: s['segment_ratio'][g], 3)
    line('V4d stature ratio Anny/SMPL (rest)', lambda s: s['stature_ratio'], 3)
    line('V4e people with a joint > 100 mm', lambda s: s['frac_any_joint_gt100mm'], 3)
    for nm, _ in FWD_PARTS:
        line(f'I2 forward angle {nm} Anny vs SMPL (deg)', lambda s, nm=nm: s['forward_angle_deg'][nm])
    line('I2 head turn vs chest, SMPL (deg)', lambda s: s['headturn_vs_chest_deg']['smpl'])
    line('I2 head turn vs chest, Anny (deg)', lambda s: s['headturn_vs_chest_deg']['anny'])
    line("I1 label gender value, 3DPW 'm'", lambda s: s['gender']['m']['label_gender_mean'], 3)
    line("I1 label gender value, 3DPW 'f'", lambda s: s['gender']['f']['label_gender_mean'], 3)
    line('I3 train.py would drop (frac)', lambda s: s['coverage']['train_py_would_drop_frac'], 3)
    line('I3 2-person imgs: valid actor unlabelled', lambda s: f"{s['coverage']['two_person_seq_images_with_unlabelled_valid_actor']}"
                                                           f"/{s['coverage']['two_person_seq_images']}")
    line('I3   ... not fitted on it (stride phase)', lambda s: f"{s['coverage']['two_person_seq_images_with_unfitted_valid_actor']}"
                                                          f"/{s['coverage']['two_person_seq_images']}")
    line('I3   ... fitted, dropped by builder filter', lambda s: f"{s['coverage']['two_person_seq_images_with_filter_dropped_actor']}"
                                                            f"/{s['coverage']['two_person_seq_images']}")
    line('I3 unlabelled actors err/vis/box/other', lambda s: "/".join(
        str(s['coverage']['unlabelled_valid_actor_causes'][c]) for c in ('dropped_err', 'dropped_vis', 'dropped_box'))
        + "/" + str(sum(s['coverage']['unlabelled_valid_actor_causes'][c]
                        for c in ('kept_by_npz_but_absent', 'no_npz'))))
    line('label age / height / proportions', lambda s: f"{fmt(s['age_label']['median'], 2)}/"
                                                      f"{fmt(s['height_label']['median'], 2)}/"
                                                      f"{fmt(s['proportions_label']['median'], 2)}")
    line('label bones exactly identity (of 163)', lambda s: s['n_identity_bones'])
    for b in NECK_BONES:
        line(f'label local angle {b} (deg)', lambda s, b=b: s['local_angle_deg'].get(b))
        line(f'   {b} exactly identity (frac)', lambda s, b=b: s['identity_frac_by_bone'].get(b), 3)


def print_per_joint(s):
    print("\nPER JOINT (overall, mean/median/p95 mm): FIT points | bone HEADS")
    for j, nm in enumerate(SMPL_JOINT_NAMES):
        a, b = s['per_joint_fit_mm'][nm], s['per_joint_head_mm'][nm]
        if a['n'] == 0:
            print(f"  {nm:<16} unmapped")
            continue
        print(f"  {nm:<16} {fmt(a['mean']):>7}/{fmt(a['median']):>6}/{fmt(a['p95']):>6}   |"
              f" {fmt(b['mean']):>7}/{fmt(b['median']):>6}/{fmt(b['p95']):>6}")
    print("PER OPENPOSE POINT (overall median px): raw 3DPW | SMPL fwd | Anny fit | Anny head")
    for nm, d in s['per_keypoint_px_median'].items():
        print(f"  {nm:<12} {fmt(d['raw']):>7} {fmt(d['smplfwd']):>9} {fmt(d['anny_fit']):>9} {fmt(d['anny_head']):>10}")


def print_fits(fsumm):
    """I4: all converter fits, before the label builder's filters."""
    print("\n=== I4 CONVERTER FITS (every .npz, ALL fitted frames, BEFORE the label builder's filters) ===")
    for split, f in fsumm.items():
        if not f.get('n_npz'):
            print(f"  {split:<10} no .npz under --fits" + (f"; 3DPW sequences without .npz: "
                                                          f"{f.get('sequences_in_3dpw_without_npz')}"
                                                          if f.get('sequences_in_3dpw_without_npz') else '')
                  + (f"; tasks without .npz: {f.get('tasks_without_npz')}" if f.get('tasks_without_npz') else ''))
            continue
        e, d = f['joint_err_mm'], f['builder_filter_counts']
        print(f"  {split:<10} {f['n_npz']} npz ({f['n_npz_unusable']} unusable), {f['fitted_frames']} fitted frames "
              f"({f['nonfinite_err_frames']} non-finite); joint error mean/median/p95/max "
              f"{fmt(e['mean'])}/{fmt(e['median'])}/{fmt(e['p95'])}/{fmt(e['max'])} mm; > {fmt(f['good_fit_mm'], 0)} mm "
              f"{fmt(100 * f['frac_err_gt_good'] if f['frac_err_gt_good'] is not None else None)}%, > "
              f"{fmt(f['builder_thresholds']['max_err'], 0)} mm {fmt(100 * f['frac_err_gt_max_err'])}%")
        print(f"  {'':<10} builder filters: kept {d['kept']}, dropped err {d['dropped_err']} / vis {d['dropped_vis']} / "
              f"box {d['dropped_box']} (unusable npz frames {d['unusable']}); label images expected from the npz "
              f"{f['expected_label_images']} vs in index {f.get('label_images_in_index', '-')}; sequences where they "
              f"differ: {f.get('sequences_expected_ne_index') or 'none'}")
        if f.get('sequences_in_3dpw_without_npz') or f.get('tasks_without_npz'):
            print(f"  {'':<10} 3DPW sequences without any .npz: {f.get('sequences_in_3dpw_without_npz') or 'none'}; "
                  f"tasks (configs/threedpw_tasks.txt) without .npz: {f.get('tasks_without_npz') or 'none'}")
    allf = fsumm.get('overall', {})
    if allf.get('per_joint_mm'):
        print("  PER JOINT, all fitted frames (mean/median/p95 mm; the converter's own per-joint table, pooled):")
        for nm, d in allf['per_joint_mm'].items():
            if d['n']:
                print(f"    {nm:<16} {fmt(d['mean']):>7}/{fmt(d['median']):>6}/{fmt(d['p95']):>6}")
    for split, f in fsumm.items():
        if split == 'overall' or not f.get('per_sequence'):
            continue
        print(f"  {split} per sequence: people fitted kept-person-frames | err median/p95/max mm | drops err/vis/box"
              f" | label images expected/in index")
        for seq, ps in f['per_sequence'].items():
            em = ps['err_mm']
            flag = '' if ps.get('label_images_in_index', ps['expected_label_images']) == ps['expected_label_images'] \
                else '   <-- differs'
            print(f"    {seq:<34} {ps['people']} {ps['fitted']:5d} {ps['kept_person_frames']:5d} | "
                  f"{fmt(em['median'])}/{fmt(em['p95'])}/{fmt(em['max'])} | {ps['dropped_err']}/{ps['dropped_vis']}/"
                  f"{ps['dropped_box']} | {ps['expected_label_images']}/{ps.get('label_images_in_index', '-')}{flag}")


def verdict_lines(vo):
    """Human-readable verdict lines of one verdicts() dict."""
    def flat(d, pre=''):
        out = []
        for kk, vv in d.items():
            if kk in ('grade', 'note', 'humans_per_image', 'per_joint_median_mm') and not pre:
                continue
            if isinstance(vv, dict):
                out += flat(vv, f"{pre}{kk}.")
            elif isinstance(vv, list):
                if vv and len(vv) <= 12 and all(isinstance(x, (str, int, float)) for x in vv):
                    out.append(f"{pre}{kk}=[{', '.join(str(x) for x in vv)}]")
            else:
                out.append(f"{pre}{kk}={fmt(vv, 3) if isinstance(vv, float) else vv}")
        return out
    lines = []
    for k, v in vo.items():
        if k in ('overall', 'not_measured') or not isinstance(v, dict):
            continue
        lines.append(f"  {k:<26} {str(v.get('grade', '')):<14} " + ", ".join(flat(v)))
    nm = vo.get('not_measured') or []
    lines.append(f"  OVERALL (V0-V5): {vo.get('overall', 'NOT MEASURED')}" + (f"   (not measured: {', '.join(nm)})" if nm else ''))
    return lines


def _atomic_write(path, write_fn, mode='w'):
    tmp = path + '.tmp'
    with open(tmp, mode, newline='' if 'b' not in mode else None) as f:
        write_fn(f)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        _atomic_write(path, lambda f: f.write(''))       # empty file: no stale rows from an earlier run
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]

    def w(f):
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in keys})
    _atomic_write(path, w)


def write_results(out_dir, summary, rows, ver, stage):
    """summary.json + per_person.csv + how_to_read.txt; called as soon as the measurements exist and again after the
    loader check and the renders, so a crash in a later stage (an EGL segfault in pyrender, the time limit) keeps
    the measurements. 'stage' in summary.json / how_to_read.txt says how far the run got."""
    summary['stage'] = stage
    summary['verdicts'] = ver
    _atomic_write(os.path.join(out_dir, 'summary.json'), lambda f: json.dump(json_safe(summary), f, indent=1))
    write_csv(os.path.join(out_dir, 'per_person.csv'), rows)
    lines = verdict_lines(ver.get('overall', {}))
    _atomic_write(os.path.join(out_dir, 'how_to_read.txt'),
                  lambda f: f.write("\n".join(LEGEND) + f"\n\nSTAGE: {stage}\n\nRESULT (overall):\n" + "\n".join(lines)
                                    + "\n"))
    return lines


# --------------------------------------------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------------------------------------------- #
def mapping_examples(ctx):
    out = {}
    for (split, seq, p), m in ctx.meta_cache.items():
        key = '|'.join(m.get('bone_names') or [])
        if key not in out:
            out[key] = dict(first=f"{split}/{seq}_p{p}", count=0, source=m['source'],
                            bones=dict(zip([SMPL_JOINT_NAMES[s] for s in m['smpl_idx']], m.get('bone_names') or [])),
                            offset_joints=m.get('offset_joints'), names_ok=m.get('names_ok'))
        out[key]['count'] += 1
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--labels', default='threedpw_anny_full/labels', help='label root with index_<split>.txt')
    ap.add_argument('--fits', default='threedpw_anny_full', help='root of <split>/<seq>_p<k>.npz (bone mapping, offsets)')
    ap.add_argument('--seq_root', default='data/3DPW/sequenceFiles', help='3DPW sequenceFiles/ (original SMPL GT)')
    ap.add_argument('--smpl_dir', default='models', help="smplx.create(..., 'smpl', gender) folder (train.py SMPLX_DIR)")
    ap.add_argument('--tasks', default=os.path.join(ROOT, 'configs', 'threedpw_tasks.txt'),
                    help='conversion task list (split seq person n); tasks without a .npz are listed in I4')
    ap.add_argument('--max_err', type=float, default=BUILDER['max_err'],
                    help='label builder --max_err used to build --labels (default = the builder default)')
    ap.add_argument('--min_visible', type=float, default=BUILDER['min_visible'], help='label builder --min_visible')
    ap.add_argument('--min_box', type=int, default=BUILDER['min_box'], help='label builder --min_box')
    ap.add_argument('--splits', default='train,validation,test')
    ap.add_argument('--per_seq', type=int, default=40, help='label images per sequence (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--surf_samples', type=int, default=50000, help='points sampled on each surface for nearest-point')
    ap.add_argument('--op_conf', type=float, default=0.0, help='OpenPose confidence must be above this')
    ap.add_argument('--img_size', type=int, default=672, help='training image size (loader + prepare_gt filter)')
    ap.add_argument('--patch', type=int, default=14, help='patch size of the prepare_gt cell filter')
    ap.add_argument('--n_loader', type=int, default=3, help='label files per split through the training loader')
    ap.add_argument('--n_vis', type=int, default=12)
    ap.add_argument('--vis_scale', type=float, default=0.5)
    ap.add_argument('--gender_refit_per_seq', type=int, default=2,
                    help='I1: measured people per sequence refitted with the label gender and the flipped one (0 = off)')
    ap.add_argument('--gender_refit_iters', default='20,80,60',
                    help='I1 refit iterations per stage (root, +pose, +shape); default = the converter warm start')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/diagnostics/check_3dpw_conversion')
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=r'(All-NaN|Mean of empty)')

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'per_person.csv')
    summary = dict(config=vars(args), thresholds=THR, builder_thresholds=BUILDER, legend=LEGEND)
    write_results(args.out, summary, [], {}, 'started (nothing measured yet)')   # no stale result from an old run
    write_csv(csv_path, [])
    ctx = Ctx(args)
    print(f"[setup] device {ctx.device} (FP32 matmuls); labels {args.labels}; fits {args.fits}; 3DPW {args.seq_root}; "
          f"SMPL {args.smpl_dir}; per_seq {args.per_seq}; seed {args.seed}", flush=True)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    # I1 (a): which end of Anny's gender value is male (rest bodies, no data needed)
    probe = gender_convention_probe(ctx)
    gsum = {}                                         # I1 (b): refit summaries per split, filled after the measurements
    gender_for = lambda k: {'probe': probe, 'refit': gsum.get(k)}  # noqa: E731
    summary.update(gender_probe=probe)
    print(f"[I1] Anny gender convention probe: male end = {probe.get('male_end')}"
          + (f" (error: {probe['error']})" if probe.get('error') else
             f" (stature {probe['gender_0']['stature_m']:.3f} m at 0 vs {probe['gender_1']['stature_m']:.3f} m at 1)"),
          flush=True)

    # ---- 1. what the label index holds, per split ---------------------------------------------------------------
    by_split, index_info, seqs3dpw = {}, {}, {}
    for split in splits:
        files, listed, src = read_index(args.labels, split)
        all_seqs = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob(os.path.join(args.seq_root, split, '*.pkl')))
        by_seq = {}
        for f in files:
            by_seq.setdefault(parse_label_path(f)[0], []).append(f)
        by_split[split], seqs3dpw[split] = by_seq, all_seqs
        index_info[split] = dict(index=src, listed=listed, existing=len(files), sequences_with_labels=len(by_seq),
                                 sequences_in_3dpw=len(all_seqs),
                                 sequences_without_labels=[q for q in all_seqs if q not in by_seq],
                                 labels_per_sequence={q: len(v) for q, v in sorted(by_seq.items())})
        print(f"[{split}] {len(files)} label files ({listed} listed in {src}) in {len(by_seq)} sequences; "
              f"3DPW has {len(all_seqs)} sequences here; without labels: "
              f"{index_info[split]['sequences_without_labels'] or 'none'}", flush=True)

    # ---- 2. I4: every converter fit, before the label builder's filters ------------------------------------------
    tasks = read_tasks(args.tasks)
    fits_by_split, fsumm = {}, {}
    for split in splits:
        lst = []
        for p in sorted(glob.glob(os.path.join(args.fits, split, '*_p*.npz'))):
            seq, _, person = os.path.basename(p)[:-4].rpartition('_p')
            if not person.isdigit():
                continue
            f = ctx.fit(split, seq, int(person))
            if f is not None:
                lst.append((seq, int(person), f))
        fits_by_split[split] = lst
        fsumm[split] = summarize_fits(lst, index_info[split]['labels_per_sequence'], seqs3dpw[split],
                                      thr=ctx.builder_thr)
        fsumm[split]['tasks_without_npz'] = [f"{q} p{k}" for s, q, k in tasks if s == split and not
                                             os.path.isfile(os.path.join(args.fits, s, f"{q}_p{k}.npz"))]
    fo = summarize_fits([x for s in splits for x in fits_by_split[s]], thr=ctx.builder_thr)
    fo.pop('per_sequence', None)                                   # per sequence: see the splits
    fo['expected_label_images'] = int(sum(fsumm[s].get('expected_label_images', 0) for s in splits))
    fo['label_images_in_index'] = int(sum(index_info[s]['existing'] for s in splits))
    fo['sequences_expected_ne_index'] = [f"{s}/{q}" for s in splits for q in fsumm[s].get('sequences_expected_ne_index', [])]
    fo['sequences_in_3dpw_without_npz'] = [f"{s}/{q}" for s in splits
                                           for q in fsumm[s].get('sequences_in_3dpw_without_npz', [])]
    fo['tasks_without_npz'] = [f"{s}/{t}" for s in splits for t in fsumm[s]['tasks_without_npz']]
    fsumm['overall'] = fo
    print_fits(fsumm)
    summary.update(index=index_info, converter_fits=fsumm, builder_thresholds=ctx.builder_thr)

    # ---- 3. measure the sampled label files ----------------------------------------------------------------------
    rows, extra, imgs = [], [], []
    sampled_by_split, skipped = {}, []
    for split in splits:
        by_seq = by_split[split]
        if not by_seq:
            continue
        sample = sample_per_sequence(by_seq, args.per_seq, args.seed)
        sampled_by_split[split] = [f for q in sorted(sample) for f in sample[q]]
        for seq in sorted(sample):
            seqd = ctx.load_seq(split, seq)
            if seqd is None:
                print(f"  [{split}] {seq}: 3DPW sequence file missing -> skipped", flush=True)
                skipped.append(f"{split}/{seq}: sequence pkl missing")
                continue
            ts = time.time()
            n_before = len(rows)
            for path in sample[seq]:
                try:
                    r, e, im = measure_image(ctx, path, split, seq, seqd)
                except Exception as ex:  # noqa
                    print(f"  [{split}] {path}: {type(ex).__name__}: {ex}", flush=True)
                    skipped.append(f"{path}: {type(ex).__name__}: {ex}")
                    continue
                rows += r
                extra += e
                imgs.append(im)
            new = rows[n_before:]
            if new:
                print(f"  [{split}] {seq:<34} {len(sample[seq]):3d} img {len(new):3d} people  fit body "
                      f"{fmt(np.nanmedian(col(new, 'fit_body_mm')))} mm  surf A->S {fmt(np.nanmedian(col(new, 'a2s_mean')))}"
                      f" S->A {fmt(np.nanmedian(col(new, 's2a_mean')))} mm  2D raw/anny "
                      f"{fmt(np.nanmedian(col(new, 'px_raw')))}/{fmt(np.nanmedian(col(new, 'px_anny_fit')))} px "
                      f"({fmt(np.nanmedian(col(new, 'mm2d_raw')))}/{fmt(np.nanmedian(col(new, 'mm2d_anny_fit')))} mm2d, "
                      f"depth {fmt(np.nanmedian(col(new, 'depth_m')), 1)} m)  "
                      f"gender {fmt(np.nanmean(col(new, 'gender_label')), 2)}  ({time.time() - ts:.0f}s, total "
                      f"{time.time() - t0:.0f}s)", flush=True)
                write_csv(csv_path, rows)              # a crash or the time limit later keeps the rows so far

    summ = {}
    for split in sampled_by_split:
        sel = [i for i, r in enumerate(rows) if r['split'] == split]
        summ[split] = summarize([rows[i] for i in sel], [extra[i] for i in sel],
                                [im for im in imgs if im['split'] == split], ctx.bones)
    summ['overall'] = summarize(rows, extra, imgs, ctx.bones)
    print_tables(summ)
    if rows:
        print_per_joint(summ['overall'])

    per_seq = []
    for (split, seq) in sorted({(r['split'], r['seq']) for r in rows}):
        rr = [r for r in rows if r['split'] == split and r['seq'] == seq]
        ii = [im for im in imgs if im['split'] == split and im['seq'] == seq]
        per_seq.append(dict(split=split, seq=seq, people=len(rr), images=len({r['frame'] for r in rr}),
                            fit_body_median_mm=float(np.nanmedian(col(rr, 'fit_body_mm'))),
                            fit_max_mm=float(np.nanmax(col(rr, 'fit_max_mm'))),
                            a2s_median_mm=float(np.nanmedian(col(rr, 'a2s_mean'))),
                            s2a_median_mm=float(np.nanmedian(col(rr, 's2a_mean'))),
                            px_raw_median=float(np.nanmedian(col(rr, 'px_raw'))) if np.isfinite(col(rr, 'px_raw')).any() else None,
                            px_anny_median=float(np.nanmedian(col(rr, 'px_anny_fit'))) if np.isfinite(col(rr, 'px_anny_fit')).any() else None,
                            mm2d_raw_median=float(np.nanmedian(col(rr, 'mm2d_raw'))) if np.isfinite(col(rr, 'mm2d_raw')).any() else None,
                            mm2d_anny_median=float(np.nanmedian(col(rr, 'mm2d_anny_fit'))) if np.isfinite(col(rr, 'mm2d_anny_fit')).any() else None,
                            depth_median_m=float(np.nanmedian(col(rr, 'depth_m'))),
                            gender_label_mean=float(np.nanmean(col(rr, 'gender_label'))),
                            stature_ratio_median=float(np.nanmedian(col(rr, 'stature_ratio'))),
                            images_with_unlabelled_valid_actor=int(sum(im['unlabelled_valid'] > 0 for im in ii)),
                            images_with_unfitted_valid_actor=int(sum(
                                im.get('unlabelled_causes', {}).get('not_fitted', 0) > 0 for im in ii))))
    worst10 = sorted(rows, key=lambda r: -(r['fit_max_mm'] if np.isfinite(r['fit_max_mm']) else -1))[:10]
    worst10 = [{k: r[k] for k in ('split', 'seq', 'frame', 'person', 'fit_max_mm', 'fit_max_joint', 'fit_all_mm',
                                  'a2s_mean', 's2a_mean', 'px_anny_fit', 'path')} for r in worst10]
    print(f"\nWORST {len(worst10)} PEOPLE by max joint error (FIT points; kept labels only, see I4 for all fits):")
    for w_ in worst10:
        print(f"  {w_['split']:<10} {w_['seq']:<32} image {w_['frame']:5d} p{w_['person']}  max {fmt(w_['fit_max_mm'])} mm "
              f"({w_['fit_max_joint']}), mean {fmt(w_['fit_all_mm'])} mm, surf {fmt(w_['a2s_mean'])}/{fmt(w_['s2a_mean'])} mm")

    summary.update(anny_version=ctx.anny_version, device=ctx.device,
                   surface_sample_spacing_mm={'anny': ctx.anny_sampler.spacing_mm,
                                              'smpl': ctx.smpl_sampler.spacing_mm if ctx.smpl_sampler else None},
                   skipped=skipped, mappings_used=mapping_examples(ctx), splits=summ, per_sequence=per_seq,
                   worst_people=worst10, loader_check=None, render_mode=None, vis=[], runtime_s=time.time() - t0)
    # write everything measured so far BEFORE the loader check and the renders (pyrender/EGL can segfault)
    pending = {'grade': 'NOT MEASURED', 'error': 'loader check not run yet (file written before it)'}
    ver = {k: verdicts(s, pending, fsumm.get(k), gender_for(k)) for k, s in summ.items()}
    write_results(args.out, summary, rows, ver, 'measured (gender refit, loader check and renders not run yet)')
    print(f"\n[write] measurements saved to {args.out} (summary.json stage 'measured')", flush=True)

    # ---- 3b. I1 refit test on a subset (label gender vs flipped gender, same warm start and budget) ---------------
    if rows and args.gender_refit_per_seq > 0:
        try:
            iters = tuple(int(x) for x in args.gender_refit_iters.split(','))
            assert len(iters) == 3, '--gender_refit_iters needs 3 numbers'
            rr = gender_refit(ctx, rows, args.gender_refit_per_seq, iters, args.seed)
            write_csv(os.path.join(args.out, 'gender_refit.csv'), rr)
            for split in sampled_by_split:
                gsum[split] = summarize_refit([r for r in rr if r['split'] == split])
            gsum['overall'] = summarize_refit(rr)
        except Exception as e:  # noqa
            gsum['overall'] = {'n_people': 0, 'n_failed': 0, 'error': f"{type(e).__name__}: {e}"}
            print(f"[I1 refit] crashed: {gsum['overall']['error']}", flush=True)
    print_gender(probe, gsum)
    summary.update(gender_refit=gsum, runtime_s=time.time() - t0)
    ver = {k: verdicts(s, pending, fsumm.get(k), gender_for(k)) for k, s in summ.items()}
    write_results(args.out, summary, rows, ver, 'measured + gender refit (loader check and renders not run yet)')

    # ---- 4. training loader ----------------------------------------------------------------------------------------
    try:
        loader = loader_check(ctx, imgs, args.n_loader) if rows else {'grade': 'NOT MEASURED',
                                                                     'error': 'no people measured'}
    except Exception as e:  # noqa
        loader = {'grade': 'NOT MEASURED', 'error': f"loader check crashed: {type(e).__name__}: {e}"}
    print(f"\n[loader] {loader.get('grade')}: " + ", ".join(f"{k}={fmt(v, 4) if isinstance(v, float) else v}"
                                                         for k, v in loader.get('verdict_values', {}).items())
          + (f" error={loader['error']}" if 'error' in loader else ''), flush=True)
    ver = {k: verdicts(s, loader, fsumm.get(k), gender_for(k)) for k, s in summ.items()}
    summary.update(loader_check=loader, runtime_s=time.time() - t0)
    lines = write_results(args.out, summary, rows, ver, 'loader checked (renders not done yet)')

    print("\n=== VERDICTS (overall; per split in summary.json) ===")
    print("\n".join(lines))
    for split in sampled_by_split:
        print(f"  {split:<10} overall {ver[split].get('overall')}: " +
              ", ".join(f"{k.split('_')[0]}={grade_word(ver[split][k].get('grade'))}" for k in ver[split]
                        if k.startswith('V')))

    # ---- 5. renders (last: the only step that can kill the process without a Python exception) -------------------
    vis = render_examples(ctx, pick_renders(rows, imgs, args.n_vis), os.path.join(args.out, 'vis'), args.vis_scale)
    summary.update(vis=vis, render_mode=_RENDER_MODE['mode'], mappings_used=mapping_examples(ctx),
                   runtime_s=time.time() - t0)
    write_results(args.out, summary, rows, ver, 'complete')
    print("\n" + "\n".join(LEGEND))
    print(f"\nwrote {os.path.join(args.out, 'summary.json')}, per_person.csv, how_to_read.txt, vis/ "
          f"({len(rows)} people, {len(imgs)} images, {time.time() - t0:.0f} s)")
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
