# Diagnostics: neck, fingers, children

Lightweight, read-only experiments on the existing checkpoints. Nothing here trains,
modifies a checkpoint or changes `train.py` / `model.py` / `demo.py`.

Scripts live in `scripts/diagnostics/`, Slurm jobs in `slurm/diagnostics/`, outputs go to
`results/diagnostics/<experiment>_<run>_<epoch>/` (`summary.json`, `per_person.csv`, `vis/`).

**Status: written and syntax-checked, not yet executed on the cluster.** The first run of each
job is itself a test; if a job fails, send the `.err` log back.

---

## What was checked in the code before writing these (file:line)

| id | claim from the analysis | verdict | evidence |
|---|---|---|---|
| C1a | demo normalises images differently from training | **false** | `utils/image.py:12` defaults `imagenet_normalization=True`; `demo.py:95,364` use the default; `datasets/annyone.py:46` passes `1`. Same normalisation. |
| C1b | demo runs fp16, eval runs fp32 | confirmed | `demo.py:464` `autocast(enabled=True)`; `train.py:570` `autocast(enabled=bool(args.amp))` with `--amp 0` in `slurm/eval/run_eval_3dpw.sh`. |
| C2 | detector-path crops get the full-image fov as K | confirmed | `demo.py:366`; K also feeds the network via `model.embedd_camera` (`model.py:333`). |
| C3 | `--relax_hands 1` forces every finger/thumb phalanx to identity (metacarpals stay predicted) | confirmed | `model.py:168-174,368-371`; on in `slurm/demo/run_demo.sh` (real photos), off in `run_demo_syn.sh`. |
| C4 | colour-jitter / flip flags are inert for AnnyOne | confirmed | only `train.py:1340-1344` argparse entries; no reader in `datasets/annyone.py`. |
| C5 | 3DPW labels have age fixed at 0.55 | confirmed (design choice) | `scripts/data/smpl_to_anny.py:229`, `slurm/data/run_smpl_to_anny_task.sh:48`. |
| C7 (new) | eval and demo do not run the same model: `train.py evaluate()` never sets `model.mask_helper_joints`, so the 74 mm PVE was measured with the 88 helper bones **unmasked**, while every demo masks them | confirmed | `train.py:859-861` sets only `mask_helper_joints_train`; `model.py:360` condition; `demo.py` default `--mask_helper_joints 1`. |
| C8 (new) | the dataset K after contain+pad skips the padding offset: the long-axis principal-point coordinate is always right, the short-axis one only for a centred principal point | confirmed; `datasets/threedpw_anny.py:16-19` implements the correct formula and names AnnyOne's omission | `datasets/annyone.py:52-56`. E1 measures `princpt/W`, `princpt/H` on every holdout image; if centred, C8 is harmless for AnnyOne. |

PVE cannot see the neck (C6 in the analysis) is arithmetic, not a code fact: head+neck vertices are a
few percent of 19k, so E1 also reports `PVE_hn` (vertices from the neck01 joint up to the crown).

Definitions used everywhere below: **skeletal height** = head joint (skull base) → mean ankle joint,
about 25 cm short of stature; **head size** = head joint → crown along the neck axis, over the head+neck
surface vertices; **head+neck vertices** (also the `PVE_hn` region) are selected once on the rest-pose
(A-pose) mesh of the same shape, so a hand raised to the head on the posed mesh cannot enter the
measurement. All are compared GT-vs-prediction under the same definition; none is a stature in metres.

---

## The experiments

| | job | script | needs | time | answers |
|---|---|---|---|---|---|
| **E1** | `slurm/diagnostics/diag_holdout.sh` | `diag_holdout_trace.py` | 1 GPU | ~10 min | Where does the neck/finger/shape error come from, and do eval and demo see the same model? |
| **E2** | `slurm/diagnostics/diag_phenotypes.sh` | `diag_phenotypes.py` | CPU | ~5 min | Does AnnyOne contain children? Can Anny represent one? |
| **E3** | `slurm/diagnostics/diag_young.sh` | `diag_holdout_trace.py --select youngest` | 1 GPU | ~15 min | Does the model predict a child when it sees one (in-sample)? |
| **E4** | `slurm/diagnostics/diag_crop_camera.sh` | `diag_crop_camera.py` | 1 GPU | ~5 min | Does the wrong crop camera change the predicted body, or only placement? |

Run E1 and E2 first; E3 only if E2 finds children; E4 is independent.

```bash
cd /netscratch/najib/multi-hmr && git pull
sbatch slurm/diagnostics/diag_holdout.sh        # E1, teacher v5 epoch 99 by default
sbatch slurm/diagnostics/diag_phenotypes.sh     # E2
sbatch slurm/diagnostics/diag_crop_camera.sh    # E4
# E3 after E2:
sbatch slurm/diagnostics/diag_young.sh

# another checkpoint (teacher v9 / final student):
CKPT=/netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt sbatch slurm/diagnostics/diag_holdout.sh
CKPT=/netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v8_final/checkpoints/00149.pt sbatch slurm/diagnostics/diag_holdout.sh
```

Logs: `logs/diag_*_<jobid>.out` (results) and `.err` (errors). Send both back.

---

## E1: holdout trace (`diag_holdout_trace.py`)

For each of the 100 held-out images (indices 579860..579959, never trained on):

1. loads the sample exactly like training (`datasets/annyone.py` + `bedlam.collate_fn`);
2. builds the **GT mesh** from the dataset's pose/shape through the same Anny body model
   (`train.py prepare_gt` path) and projects it with the dataset K. A GT overlay that does not sit
   on the person means the camera or pose convention is wrong before any network is involved;
3. runs the checkpoint on the **same tensor** under four variants:

| variant | precision | K | helper bones | hands | mirrors |
|---|---|---|---|---|---|
| `V0_eval_unmasked` | fp32 (TF32 matmul, as `train.py --tf32 1`) | dataset | unmasked | predicted | `train.py evaluate()` (the 74 mm number) |
| `V1_eval_masked` | fp32 | dataset | masked | predicted | eval numbers with the demo's masking |
| `V2_demo` | fp16 | fov-derived | masked | predicted | the `demo.py` **network call** of the in-domain path (`run_demo_syn.sh`); not its `filter_humans` / `max_persons` post-processing |
| `V3_demo_relaxhands` | fp16 | fov-derived | masked | identity (phalanges) | V2 + the `--relax_hands` flag of `run_demo.sh`; not the detector-crop path (that is E4) |

V1 vs V2 isolates fp16 only where `K_fov` equals the dataset K; `summary.json → camera.k_fov_within_1px_frac`
says on how many images that holds (it requires a centred principal point and fx = fy).

4. keeps the GT people `train.py prepare_gt` keeps (root in front of the camera, inside the image,
   one per heatmap cell), matches detections to them (same greedy matcher) and reports per variant:
   recall/FP, PVE, PA-PVE, MPJPE, `PVE_hn`, rotation error per bone group (active bones only; the
   88 helper bones' prediction error for V0 only, since V1–V3 force them to identity), per-phenotype
   shape bias and error, neck length, neck/skeletal-height and head/skeletal-height ratios vs GT
   (computed per person, then averaged), and the **neck factorisation**:

| neck length under | isolates |
|---|---|
| GT pose + GT shape | the truth |
| GT pose + predicted shape | the shape head's contribution |
| predicted pose + GT shape | the neck rotations' contribution |
| rest pose + GT / predicted shape | shape alone, no pose |

Also reported once per run: whether any non-root bone in the GT poses carries a translation
(`camera.nonroot_translation_max`; the factorisation assumes none), and how far the GT helper bones
are from identity (`gt_helper_bones_angle_deg`: the irreducible floor the mask introduces, a dataset property).

**Paired comparison.** Variants match different detections, so the per-variant means are not directly
comparable. `summary.json → paired` (and the `PAIRED COMPARISON` table) report every metric on the
(image, person) pairs matched under *all* variants, with deltas vs `V1_eval_masked`. Attribute an effect
to a toggle from those rows only.

Outputs: `summary.json`, `per_person.csv` (phenotype columns are prefixed `pheno_`), `vis/<idx>.jpg` = `[image | GT | V1 | V2]`.

### How to read E1

| observation | meaning | next step |
|---|---|---|
| GT overlay misaligned / `gt_j2d_inside_image_frac` < 95 % | camera or pose convention bug upstream of the network | fix `datasets/annyone.py` K before anything else |
| `principal_point_centred_frac` < 100 % | dataset K is wrong on those images (C8) | quantify; likely small |
| V0 PVE ≠ V1 PVE by > 2 mm | eval and demo disagree on the model; reported numbers describe a model nobody renders | re-evaluate with masking; pick one and state it in the thesis |
| V1 ≈ V2 | fp16 and fov-K are harmless; demo renders are faithful | drop C1b |
| `neck%` ≫ 100 and `GTpose+PREDshape` ≈ pred | neck error is in the **shape head** (`proportions`/`height`) | shape-loss reweighting, check `proportions` bias |
| `neck%` ≫ 100 and `PREDpose+GTshape` ≈ pred | neck error is in the **neck rotations** | neck-weighted rotation loss / more neck supervision |
| `neck%` ≈ 100, `PVE_hn` elevated, renders long-necked | the skeleton is right but the neck **surface** is not: a skinning (LBS) effect of neck01–03 | compare `--rest_pose 1` renders; this is the only route to the current "rig property" claim |
| `neck%` ≈ 100, `PVE_hn` normal, renders look long-necked | rendering/placement artefact, not the model | inspect `overlay_human_meshes` units logic (`demo.py:565-573`) |
| `rot_fingers` ≫ other groups in V1 but V3 fingers "look fine" | the fingers are undertrained and `--relax_hands` hides it (C3) | hand supervision; never report V3 hands as predictions |
| `helper_bones_pred_err_deg` (V0) ≫ `gt_helper_bones_angle_deg` | the unmasked eval mesh carries untrained helper-bone rotations | report V1 numbers, not V0, as "the model" |
| `shape_bias[age]` > 0 in the low GT-age bucket | regression to the adult prior | see E2/E3 |

---

## E2: phenotype distribution + body-model probe (`diag_phenotypes.py`)

Part A reads 20 000 evenly spaced training annotations (pkl only, no images): per-phenotype
statistics, an **age histogram**, the fraction of people below age 0.2 / 0.3 / 0.4 / 0.5, and the same
for the holdout. Anny inherits MakeHuman's age scale (`scripts/data/smpl_to_anny.py:149`): 0.0 ≈ 1 y,
0.19 ≈ 11 y, 0.3 ≈ 16 y, 0.5 = 25 y, 1.0 = 90 y, so **< 0.2 is "child", 0.2–0.3 "adolescent"**.
Part B puts Anny in rest pose at the dataset-mean shape and sweeps `age` 0 → 1, printing skeletal
height, neck/height and head/height.

| observation | meaning |
|---|---|
| age < 0.2 is a fraction of a percent | there is almost no child supervision; with an L1 shape loss and a 0.5 prior the network will regress children to adults: **a data cause**; the 3DPW labels (age fixed 0.55, C5) add none |
| several percent below 0.2 | children exist in training; run E3 |
| Part B: height drops and head/height rises as age → 0 | Anny can represent a child; the failure is network/data-side, not the body model |
| Part B: proportions barely move | `age` alone does not make a child in this rig; `height`/`proportions` must move too, which changes what "child supervision" means |

---

## E3: youngest in-sample people (`diag_holdout_trace.py --select youngest`)

Scans the first 20 000 training annotations for the lowest GT age and runs E1 on the 50 youngest
**people** (one per image; the other people in those images are still given to the matcher so they
are not counted as false positives, but their metrics are not reported). These were **seen in
training**, so this tests whether the network can *represent* a child at all, not whether it generalises.

| observation | meaning |
|---|---|
| predicted age tracks GT age (bucket table) and predicted rest height is child-sized | the model can output children; the real-photo failure is in the demo path (crop camera C2, scale ambiguity): go to E4 |
| predicted age ≈ 0.5 regardless of GT | regression to the prior even in-sample: loss-side fix (reweight rare ages / stronger shape loss on age) |

---

## E4: crop camera (`diag_crop_camera.py`)

Reproduces the demo's detector crops on `example_data` (same EXIF handling, same crop formula and
clamp) and runs the model with the demo K (full-image fov, `demo.py:366`) and with the corrected crop
K (`f·672/S`, shifted principal point), fp32, on the same crop. Reports predicted age/height/proportions
phenotypes, skeletal height, neck/height, head/height and depth under both.

| observation | meaning |
|---|---|
| mean \|Δ pheno_age\| or \|Δ pheno_height\| > ~0.05 (phenotype units, 0–1) | K shapes the body: the crop-camera bug affects shape, not just placement; fix `demo.py:366` and re-render the children |
| changes ≈ 0 | C2 is placement-only and `anchor_to_box` already compensates; the children come from the data (E2) |

Caveat: the corrected fov is `2·atan(S / 2f_full)`: narrower than AnnyOne's 51–118° when the crop is
smaller than the image, wider when `S > max(W,H)` (frame-filling people). Either way the network may be
out of its camera distribution; E4 says whether K matters, not which K is right.

---

## Suite v2: fixed manifest, six checks, one report

Builds on E1–E4 with a **fixed sample manifest** and the checks a reviewer would ask for before any
architecture change. Everything is read-only except check 6, which trains a throw-away copy for a few
minutes on 40 images (its checkpoints live under `results/diagnostics/overfit_*/`, never under
`logs/anny_model/`). The only edit to existing code is a default-preserving `--anny_data_folder` flag in
`train.py`, so the overfit run can point at the symlinked subset.

| step | job | script(s) | check |
|---|---|---|---|
| 0 | `slurm/diagnostics/diag_phenotypes.sh` | `diag_phenotypes.py` | E2 (needed by the age sweep's mean shape; optional) |
| 1 | `slurm/diagnostics/diag_manifest.sh` | `build_manifest.py` → `diag_gt_decode.py` → `diag_holdout_trace.py --manifest` (train, holdout) | **manifest**, **1** GT decode + overlay with the supplied camera, **5** GT-shape substitution with predicted pose, per-category tables |
| 2 | `slurm/diagnostics/diag_bodymodel.sh` | `diag_bodymodel_probe.py` | **2** canonical pose + age sweep, **3** single-joint perturbations (neck, fingers) |
| 3 | `slurm/diagnostics/diag_gradients.sh` | `diag_gradients.py` | **4** gradients into the heads, mesh response to head outputs |
| 4 | `slurm/diagnostics/diag_overfit.sh` | `train.py` (capped) | **6** overfit the manifest train subset; before/after PVE on train and holdout subsets |
| 5 | login node: `python scripts/diagnostics/diag_report.py --tag <run>_<epoch>` | `diag_report.py` | `REPORT.md` + `comparison_panel.jpg`, each failure assigned to decoding / preprocessing / supervision / optimization / generalization |

```bash
cd /netscratch/najib/multi-hmr && git pull
sbatch slurm/diagnostics/diag_phenotypes.sh
sbatch slurm/diagnostics/diag_manifest.sh      # builds the manifest on first run; REBUILD=1 to redo (changes the fixed set!)
sbatch slurm/diagnostics/diag_bodymodel.sh
sbatch slurm/diagnostics/diag_gradients.sh
sbatch slurm/diagnostics/diag_overfit.sh       # after diag_manifest.sh has finished (needs data/diag_manifest/)
# when all have finished:
python scripts/diagnostics/diag_report.py --tag anny_s2_shape_v5_00099
```

**The manifest** (`results/diagnostics/manifest.json`, `build_manifest.py`): one (image, person) per entry, tags
`adult` (age 0.45–0.65, quiet neck), `child` (lowest GT age, Anny/MakeHuman scale: 0.19 ≈ 11 y), `hands`
(both hands projected inside the image, large pixel span, in front of the torso), `neck` (largest GT
neck+head rotation). 10 per tag from the training pool, 5 per tag from the 100-image holdout; the
selection is seeded. `data/diag_manifest/{train,holdout}/` hold symlinks to the chosen pkl+jpg pairs.
If the holdout has no children the manifest says so, and every child number is in-sample.

**Reading the report.** `REPORT.md` quotes the numbers each verdict rests on and marks missing inputs;
`comparison_panel.jpg` stacks GT-decode, prediction-vs-GT (train and holdout), age sweep, perturbation and
crop-camera panels. Two rules it enforces: tiny-subset overfit success is an optimisation check, never a
validation result; and vertex errors are only ever computed between Anny meshes of the same topology and
vertex order (GT and predictions share one body model).

---

## Ranked causes (to revise after E1–E4)

**Confirmed from code**

| failure | cause | fix effort |
|---|---|---|
| all demo artefacts | eval and demo run different model behaviour (C7: helper-bone masking) and precision (C1b) | one flag / one line each; E1 quantifies |
| fingers | `--relax_hands 1` replaces the phalanx predictions with rest hands (C3); the real finger predictions have not been looked at since | none; E1 `rot_fingers` measures them |
| children (real photos) | crop K is the full-image camera (C2) | ~10 lines in `demo.py`; E4 says whether it matters |
| children (all) | 3DPW labels carry no child ages (C5); AnnyOne child fraction unknown (E2) | data |
| thesis text | no colour augmentation was ever applied (C4) | correct the text |

**Hypotheses (E1 decides)**

| failure | hypothesis |
|---|---|
| neck | prediction error on neck01–03/head rotations (4 bones, ~2.5 % of the rotation loss) |
| neck | `proportions`/`height` over-prediction by the shape head |
| neck | LBS rig property (current thesis claim; by elimination only, never measured directly) |
| fingers | 38 finger bones, few pixels at 672 px, regress toward the identity init |
| children | regression to the 0.5 prior under an L1 shape loss with rare young samples |
