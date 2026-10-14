# Diagnostic findings: neck, fingers, children, and a depth defect

Checkpoint `anny_s2_shape_v5/checkpoints/00099.pt`. All runs from `slurm/diagnostics/`.
Confidence tags: [Certain] = measured or read in code, [Likely] = strong inference, [Guessing] = unverified.

## The one-paragraph version

[Certain] The model is **not** broken in the places the thesis text suspected. Decoding is correct (GT
parameters render onto the people, every single-joint perturbation moves the intended part), the
optimisation path is intact (every active head parameter receives gradient, the mesh responds to shape and
pose changes), and in-domain neck length is within 1% of ground truth. [Certain] The largest defect found is
one nobody was looking for: **predicted depth collapses to a near-constant ~3 m** (fit: pred ≈ 0.3·GT + 2.4 m),
the whole body is projected ~35% too small, and pelvis-centred PVE/MPJPE cannot see it. [Likely] The three
visible failures are: neck → weak per-bone rotation supervision on a redundant chain; fingers → under-trained
(smallest gradients, worst 6D conditioning, worse than the rest pose); children → representable in-sample but
not generalised, because 2.5% of training people are children and 3DPW pins age to 0.55.

## What was run

| check | script | sample |
|---|---|---|
| 1 GT decode, no network | `diag_gt_decode.py` | manifest, 55 people |
| 2+3 body-model age sweep + 16 joint perturbations | `diag_bodymodel_probe.py` | none (body model only) |
| 4 gradients + mesh response | `diag_gradients.py` | 4 holdout images |
| 5 prediction vs GT, shape swap | `diag_holdout_trace.py --manifest` | train 40 img / holdout 15 img |
| E1 full holdout | `diag_holdout_trace.py --select holdout` | 100 images, 222 people kept |
| E3 youngest | `diag_holdout_trace.py --select youngest` | 50 babies, in-sample |
| E2 phenotype scan | `diag_phenotypes.py` | 54,784 people |
| E4 crop camera | `diag_crop_camera.py` | 20 people, real photos |
| 6 capped overfit | `diag_overfit.sh` | 40 train images, 400 iters, lr 1e-4, last 4 backbone blocks + heads |

Manifest: train adult/child/hands/neck = 10/10/10/10; holdout 5/0/5/5. **No children in the holdout**: every
child number below is in-sample and tests representation, not generalisation.

## Check 1: decoding is correct, the report's "defect" verdict was a false alarm

[Certain] GT joints inside the image 84.5%, inside the annotation bbox 84.9%. The two numbers move together
per category (adult 95.6/97.5, hands 63.3/63.8, neck 85.4/87.6): the bbox contains what the image contains,
and the missing joints are **truncation at the image border** (the "hands" people are 1033 px tall in a
1280 px frame). The GT overlays (`gt_decode/vis/`) sit on the people. The first compiled `REPORT.md` printed
"decoding defect" because its rule compared bbox containment to a fixed 95%; the rule is now truncation-aware.

[Certain] The bbox format is ambiguous in the annotations (29 people fit `xyxy_px` best, 26 `xywh_px`).
Nothing in training uses the bbox, so this does not affect the model, but do not use it for metrics.

## Checks 2+3: the body model can do everything asked of it

[Certain] Age sweep at the default phenotypes: skeletal height 0.528 m (age 0) → 1.374 m (0.5) → 1.553 m (0.8);
head/height 0.200 → 0.112 → 0.106. The model can make a child with child proportions.
[Certain] All 16 perturbations PASS (96–100% of displaced vertices in the expected part; FK exact; pose
parameters act in the bone's rest frame). `neck01` leaks ~4% of its displacement into shoulders/clavicles
through LBS, which is normal skinning. Helper mask: 88 masked / 75 active, identical in `model.py` and the
mask-boost patch.

## Check 4: gradients reach every head

[Certain] Zero-gradient pose bones == the 88 masked helper bones exactly; no active bone is starved.
Per-bone |grad|: root 598, spine 100, neck 25, head 15.6, **fingers 4.7** (smallest), and fingers have the
worst 6D conditioning (0.25–0.30; neck 0.99). Shape: age has the largest gradient of all phenotypes.
Loss shares at epoch 99: shape 25.7%, j3d 15.3%, rotmat 12.9%, offset 12.4%, v3d 12.1%, j2d 6.8%, v2d 6.3%,
bce 5.8%, **transl 1.9%, dist 0.6%**. Response: age logit −2 moves skeletal height by −622 mm; pose hooks
move the mesh, so decoded meshes respond to the heads.
[Certain] `deccam` has three outputs and only the first receives gradient (`per_output_grad_norm`
= [6778, 0, 0]). See the depth section.

## The depth defect (new, and the largest error in the system)

[Certain] On the 100-image holdout (222 people, V1):

| quantity | GT | predicted |
|---|---|---|
| root depth, mean (sd) | 1.92 m (0.49) | 3.00 m (0.32) |
| linear fit | | pred = 0.33·GT + 2.36 m |
| people predicted farther than GT | | 100% |
| corr(GT depth, depth error) | | −0.76 |
| skeletal height | 1.409 m | 1.405 m (r = 0.89) |
| head-to-root pixel distance (median) | 127 px | 82 px (ratio 0.67) |
| root pixel error / head pixel error (median) | | 14 px / 46 px |

The same slope (0.18–0.33) and offset (2.4–2.8 m) appear on the manifest train split, the manifest holdout
and the youngest-50 babies, so it is not a generalisation effect: **the network never learned depth**. The
body is the right metric size, placed at the wrong distance, therefore drawn too small; the root lands on
roughly the right pixel, the head does not. Pelvis-centred PVE is blind to all of this
(corr(PVE, depth error) = −0.03), which is why the thesis numbers look normal while the overlays look wrong.

[Certain] Context already established from code: the model anchors and reports the HEAD joint
(`model.py:430,442`, `person_center='head'`) while `train.py:214-222,239-245` build the transl / dist /
heatmap targets at joint 0 (root); measured |pred transl − GT root transl| = 0.90 m, |pred head − GT head|
= 0.90 m, GT head-to-root distance 0.67 m. The transl and dist terms together are 2.5% of the loss.

### Root cause (code, verified)

[Certain] Units and cameras are consistent: `dist` is decoded as `exp(raw · f/fn)` with the 672-scaled
dataset K on both sides (`model.py:255-264`, `train.py:225-229`); transl is `K⁻¹[u,v,1]·dist`, so `dist`
is the z coordinate of the anchor pixel. The one inconsistency is **which joint is anchored**:

- the model places the HEAD at the anchor pixel and reports the head as transl (`model.py:121,430-432,442`);
- `prepare_gt` builds `transl` and `dist` from joint 0 = ROOT whenever `anny_pose` is present
  (`train.py:213-222`), and derives the heatmap cell and the offset target from that root pixel
  (`train.py:239-245`);
- j2d / v2d are compared in absolute pixels after projecting with the PREDICTED transl (`model.py:496-497`,
  `loss.py:117-120`), so they are the only strong terms that see depth; j3d / v3d are pelvis-centred
  (`loss.py:104-106`) and carry no depth signal at all; dist and transl have weight 1 (0.6% and 1.9% of
  the loss).

[Likely] Mechanism. The detection cell and the offset term pull the anchor pixel toward the GT **root**, but
the mesh is attached to that pixel by its **head**. Every projected joint is therefore displaced by the
head-to-root pixel distance. The 2D terms can reduce that displacement by shrinking the projected body, and
shrinking means predicting a larger depth. Near people have a larger head-to-root pixel distance, so they are
pushed proportionally farther: slope ≈ 0.3, always farther, r = −0.76, root pixel roughly right, head pixel
wrong, body 35% too small. The weak dist/transl terms cannot resist, and the transl term itself pulls the head
onto the root's 3D position, adding noise rather than a correction. The overfit run shows the same conflict in
motion: 3D pelvis-centred terms fall, transl and 2D terms rise.

Alternatives considered: weak dist weight alone (unlikely: a consistent log-depth target would still be
learned over 99 epochs); a K / focal / image-size mismatch (ruled out by reading both sides).

**Minimal fix** (`train.py`, `prepare_gt`), then fine-tune from the current checkpoint because the heatmap
cell moves from root to head:

```diff
         if 'anny_pose' in y:
-            root_joint_idx = 0
+            root_joint_idx = self.model.person_center_idx   # anchor the target on the joint the model anchors
         else:
             root_joint_idx = JOINT_NAMES.index(self.args.person_center)
         target['transl'] = jts[:, root_joint_idx]
         target['transl_pelvis'] = jts[:, 0]
-        target['dist'] = torch.clamp(jts[:, 0, -1], min=0.1)
+        target['dist'] = torch.clamp(jts[:, root_joint_idx, -1], min=0.1)
```

Optionally raise `--alpha_dist` and `--alpha_transl` to 5–10 so depth has a direct term of its own.

**One-batch confirmation before retraining** (insert after the forward pass in the training step, around
`train.py:452`):

```python
hp = perspective_projection(gt['j3d'][:, [self.model.person_center_idx]], gt['K'][gt['idx'][0]])[:, 0]
print('offset_y', pred['offset'][:, 1].tolist(), 'loc_y', pred['loc'][:, 1].tolist(),
      'gt_root_y', gt['loc'][:, 1].tolist(), 'gt_head_y', hp[:, 1].tolist(),
      'dist pred/gt', pred['dist'].tolist(), gt['dist'].tolist())
```

Expected if the explanation holds: predicted offset_y far outside the GT range [−0.5, 0.5] patches, predicted
loc_y between the GT head and GT root rows, predicted dist above GT dist with the gap growing for near people.

## Check 6: capped overfit (optimisation check, not validation)

[Certain] 400 iterations (40 epochs × 10, batch 4) on the 40 manifest train images, evaluated with
`train.py evaluate()` (V0, unmasked, all people in the images):

| split | PVE before | PVE after | recall before → after |
|---|---|---|---|
| train (40 img) | 80.7 | 51.8 (−36%) | 93 → 90% |
| holdout (15 img) | 87.6 | 127.2 (+45%) | 91 → 76% |

The optimisation path works: the net can drive its loss down on images it is shown. The holdout gets worse,
as memorisation must; this number is a generalisation read and nothing else. The train/holdout gap at the
*real* checkpoint is small (manifest train PVE 78.3 vs holdout 56.7 in the trace; the train split is harder
because it contains the children and the truncated "hands" people), so the production model is **not**
over-fitted. Its errors are capacity/supervision errors, not memorisation.

[Certain] Per-term losses during the overfit (mean of first 5 vs last 5 logged epochs):

| term | first 5 | last 5 | moved? |
|---|---|---|---|
| shape | 1.10 | 0.32 | down |
| bce | 0.93 | 0.41 | down |
| j3d / v3d | 0.120 / 0.114 | 0.063 / 0.062 | down |
| rotmat | 44.6 | 40.8 | barely |
| offset | 6.2 | 6.5 | flat |
| dist | 0.79 | 0.54 | noisy, not converging |
| **transl** | 1.67 | 1.89 | **up** |
| **j2d / v2d** | 58 / 58 | 75 / 77 | **up** |

Even with the images memorised, the translation term does not fall and the 2D reprojection terms rise while
the pelvis-centred 3D terms fall. The network cannot satisfy the transl/dist targets and the 2D projection at
the same time: that is the signature of a target that contradicts the model's own anchor (depth section).
One NaN/Inf gradient step was skipped (epoch 33); harmless here but keep gradient clipping on.

## Neck

Origin: **supervision** (weak per-bone rotation loss on a redundant 4-bone chain), **not decoding**.

- [Certain] In-domain neck length is 99.3% of GT (holdout), neck/height 99.6%, head/height 99.5%; the
  factorisation puts the residual entirely in shape (GT pose + pred shape 0.1086 m vs GT 0.1094 m; pred pose
  + GT shape = GT). There is no "long neck" in the parameters.
- [Certain] Local rotation error: neck 62.7°, head 89.5°, spine 5.7°. Global orientation error of the head:
  12.0° (holdout), 8.6° (manifest holdout), 19.9° (manifest train, which has the hard neck poses). Four bones
  in series can distribute the same global head pose many ways; the per-bone rotmat loss penalises every
  decomposition except the annotator's, so the gradient is noisy and the net learns little. This is also why
  the v4 "neck boost" patch had no effect: it multiplied a term that carries no consistent signal.
- [Certain] The "neck" category is the hardest of the three in both splits (PVE 93.7 train / 76.3 holdout
  vs adult 34.1 / 47.4), with PVE_hn 79 mm on train.
- [Likely] The long-neck look in demos comes from the depth defect (body too small, head placed where the
  root target pulls it) plus LBS, not from neck parameters.

## Fingers

Origin: **optimisation / supervision** (under-trained), not decoding.

- [Certain] Finger rotation error 17.3° (holdout); with `--relax_hands` (fingers zeroed to rest pose) 15.1°.
  **The predicted fingers are worse than not predicting fingers.**
- [Certain] Fingers receive the smallest gradients of any bone group (4.7 vs root 598) and have the worst 6D
  conditioning. The per-bone rotation loss has 75 active bones; the 28 finger/thumb bones share a small
  slice of it, and finger vertices are a tiny fraction of the v3d/v2d terms.
- [Certain] Perturbing a finger bone moves only that finger (check 3), so the rig is fine.
- Note the "hands" manifest category is confounded: its people are frame-filling and truncated (63% of joints
  inside the image), so its PVE of 156.8 mm measures truncation and depth, not hands.

## Children

Origin: **generalisation rooted in sparse supervision**; representation is fine.

- [Certain] 2.53% of training people have age < 0.3 (child on Anny's scale, 1/3 ≈ 11 y), 1.06% < 0.2; the
  holdout has none; 3DPW labels pin age at 0.55 (≈ 20 y), so fine-tuning adds zero child signal. The GT age
  histogram is bimodal around 0.57 and 0.80 with a thin tail below 0.45.
- [Certain] In-sample babies (youngest 50, GT age 0.015): predicted age 0.10, height phenotype 0.32, rest
  skeletal height 0.59 → 0.60 m, PVE 48.8, head/height 101% of GT. The model **can** output a baby when it has
  seen it. Manifest children (n=10): PVE 47.8 → 41.6 mm with GT shape (gain 6.2 mm), age bias +0.078.
- [Certain] On real photos (E4): skeletal heights 0.62–0.99 m are produced with age 0.56–0.66 and the size is
  carried by the `height` phenotype, i.e. "small adult", with adult neck/height. That is the failure users see.
- [Certain] The crop-camera bug (C2) is placement-only: corrected K changes age by 0.004, height by 0.012,
  depth by 1.27 m. It does not explain child shape.

## What does *not* matter (measured)

| claim | result |
|---|---|
| C1a ImageNet normalisation mismatch | false in code |
| C1b fp16 in demo | +1.0 mm PVE paired |
| C2 crop K from full-image fov | placement only, shape unchanged |
| C3 `--relax_hands` | fingers 17.3° → 15.1° (better without prediction) |
| C7 eval unmasked vs masked | +3.8 mm PVE: the thesis's 74 mm is really 70.5 mm |

## Caveats

- The capped-overfit numbers (check 6) are an optimisation sanity check; the 51.8 mm train PVE is not a
  validation number and is not comparable to any holdout figure.
- All vertex comparisons are between Anny meshes with identical topology and vertex order; no cross-topology
  comparison exists in the suite.
- Children: in-sample only (no holdout children exist).
- "hands" category: truncation-confounded; rebuild the manifest with a max pixel-height criterion before
  using it for hand metrics.
- Depth measurements assume the dataset K; the camera check confirms K_fov == dataset K on 100% of images and
  the principal point is centred on 100%.

## What to do before Friday

1. Depth: apply the fix in the depth section and retrain for a few epochs from the current checkpoint; check
   that predicted root depth tracks GT (slope → 1) on the holdout trace.
2. Report V1 (masked) numbers as the headline and say why V0 differs.
3. Neck: replace or supplement the per-bone rotmat loss on the neck chain with a global head-orientation loss
   (or a loss on the head/neck joint positions, which the pelvis-centred j3d term already approximates).
4. Fingers: upweight finger bones in the rotation loss or add a hand-vertex term; until then ship demos with
   `--relax_hands` and say so in the thesis.
5. Children: oversample age < 0.4 (or weight their shape loss); do not expect 3DPW fine-tuning to help.
6. Re-run `diag_report.py` so `REPORT.md` includes check 6 and the corrected check-1 verdict.
