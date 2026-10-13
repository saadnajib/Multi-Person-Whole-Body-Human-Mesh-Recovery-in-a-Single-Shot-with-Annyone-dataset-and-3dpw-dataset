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
| C3 | `--relax_hands 1` forces every finger bone to identity | confirmed | `model.py:368-371`; on in `slurm/demo/run_demo.sh` (real photos), off in `run_demo_syn.sh`. |
| C4 | colour-jitter / flip flags are inert for AnnyOne | confirmed | only `train.py:1340-1344` argparse entries; no reader in `datasets/annyone.py`. |
| C5 | 3DPW labels have age fixed at 0.55 | confirmed (design choice) | `scripts/data/smpl_to_anny.py:229`, `slurm/data/run_smpl_to_anny_task.sh:48`. |
| C7 (new) | eval and demo do not run the same model: `train.py evaluate()` never sets `model.mask_helper_joints`, so the 74 mm PVE was measured with the 88 helper bones **unmasked**, while every demo masks them | confirmed | `train.py:859-861` sets only `mask_helper_joints_train`; `model.py:360` condition; `demo.py` default `--mask_helper_joints 1`. |
| C8 (new) | the dataset K after contain+pad is only right for a centred principal point | confirmed by derivation, data unverified | `datasets/annyone.py:52-56`. E1 measures `princpt/W`, `princpt/H` on every holdout image. |

PVE cannot see the neck (C6 in the analysis) is arithmetic, not a code fact: head+neck vertices are a
few percent of 19k, so E1 also reports `PVE_headneck` (vertices within 35 cm of the head joint).

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
| `V0_eval_unmasked` | fp32 | dataset | unmasked | predicted | `train.py evaluate()` (the 74 mm number) |
| `V1_eval_masked` | fp32 | dataset | masked | predicted | eval numbers with the demo's masking |
| `V2_demo` | fp16 | fov-derived | masked | predicted | `demo.py` in-domain path (`run_demo_syn.sh`) |
| `V3_demo_relaxhands` | fp16 | fov-derived | masked | identity | `run_demo.sh` real-photo setting |

4. matches detections to GT people (same greedy matcher as `train.py`) and reports per variant:
   recall/FP, PVE, PA-PVE, MPJPE, `PVE_headneck`, rotation error per bone group (active bones
   only, masked bones separately), per-phenotype shape bias and error, neck length and
   neck/height ratio vs GT, head size vs GT, and the **neck factorisation**:

| neck length under | isolates |
|---|---|
| GT pose + GT shape | the truth |
| GT pose + predicted shape | the shape head's contribution |
| predicted pose + GT shape | the neck rotations' contribution |
| rest pose + GT / predicted shape | shape alone, no pose |

Outputs: `summary.json`, `per_person.csv`, `vis/<idx>.jpg` = `[image | GT | V1 | V2]`.

### How to read E1

| observation | meaning | next step |
|---|---|---|
| GT overlay misaligned / `gt_j2d_inside_image_frac` < 95 % | camera or pose convention bug upstream of the network | fix `datasets/annyone.py` K before anything else |
| `principal_point_centred_frac` < 100 % | dataset K is wrong on those images (C8) | quantify; likely small |
| V0 PVE ≠ V1 PVE by > 2 mm | eval and demo disagree on the model; reported numbers describe a model nobody renders | re-evaluate with masking; pick one and state it in the thesis |
| V1 ≈ V2 | fp16 and fov-K are harmless; demo renders are faithful | drop C1b |
| `neck%` ≫ 100 and `GTpose+PREDshape` ≈ pred | neck error is in the **shape head** (`proportions`/`height`) | shape-loss reweighting, check `proportions` bias |
| `neck%` ≫ 100 and `PREDpose+GTshape` ≈ pred | neck error is in the **neck rotations** | neck-weighted rotation loss / more neck supervision |
| `neck%` ≈ 100 but renders look long-necked | rendering/placement artefact, not the model | inspect `overlay_human_meshes` units logic |
| `rot_fingers` ≫ other groups in V1 but V3 fingers "look fine" | the fingers are undertrained and `--relax_hands` hides it (C3) | hand supervision; never report V3 hands as predictions |
| `shape_bias[age]` > 0 in the low GT-age bucket | regression to the adult prior | see E2/E3 |

---

## E2: phenotype distribution + body-model probe (`diag_phenotypes.py`)

Part A reads 20 000 evenly spaced training annotations (pkl only, no images): per-phenotype
statistics, an **age histogram**, the fraction of people below age 0.2 / 0.3 / 0.4, and the same for
the holdout. Part B puts Anny in rest pose at the dataset-mean shape and sweeps `age` 0 → 1,
printing body height, neck/height and head/height.

| observation | meaning |
|---|---|
| age < 0.3 is a fraction of a percent | there is almost no child supervision; with an L1 shape loss and a 0.5 prior the network will regress children to adults: **a data cause**; the 3DPW labels (age fixed 0.55, C5) add none |
| several percent below 0.3 | children exist in training; run E3 |
| Part B: height drops and head/height rises as age → 0 | Anny can represent a child; the failure is network/data-side, not the body model |
| Part B: proportions barely move | `age` alone does not make a child in this rig; `height`/`proportions` must move too, which changes what "child supervision" means |

---

## E3: youngest in-sample people (`diag_holdout_trace.py --select youngest`)

Scans the first 20 000 training annotations for the lowest GT age and runs E1 on the 50 youngest.
These were **seen in training**, so this tests whether the network can *represent* a child at all,
not whether it generalises.

| observation | meaning |
|---|---|
| predicted age tracks GT age (bucket table) and predicted rest height is child-sized | the model can output children; the real-photo failure is in the demo path (crop camera C2, scale ambiguity): go to E4 |
| predicted age ≈ 0.5 regardless of GT | regression to the prior even in-sample: loss-side fix (reweight rare ages / stronger shape loss on age) |

---

## E4: crop camera (`diag_crop_camera.py`)

Reproduces the demo's detector crops on `example_data` and runs the model with the demo K
(full-image fov, `demo.py:366`) and with the corrected crop K (`f·672/S`, shifted principal point),
fp32, on the same crop. Reports predicted age/height/proportions, body height, neck/height and depth
under both.

| observation | meaning |
|---|---|
| mean |Δage| or |Δheight| > ~0.05 | K shapes the body: the crop-camera bug affects shape, not just placement; fix `demo.py:366` and re-render the children |
| changes ≈ 0 | C2 is placement-only and `anchor_to_box` already compensates; the children come from the data (E2) |

Caveat: the corrected K has a narrower fov than anything in AnnyOne (51–118°), so the network may be
out of its camera distribution either way; E4 says whether K matters, not which K is right.

---

## Ranked causes (to revise after E1–E4)

**Confirmed from code**

| failure | cause | fix effort |
|---|---|---|
| all demo artefacts | eval and demo run different model behaviour (C7: helper-bone masking) and precision (C1b) | one flag / one line each; E1 quantifies |
| fingers | `--relax_hands 1` replaces predictions with rest hands (C3); the real finger predictions have not been looked at since | none; E1 `rot_fingers` measures them |
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
