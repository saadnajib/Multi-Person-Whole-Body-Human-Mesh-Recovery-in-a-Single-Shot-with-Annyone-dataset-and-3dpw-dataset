# exp2_children: age-balanced sampling and per-phenotype shape weights

Experiment 2 of the one-week plan (children). Everything below is read from code or quoted from
`docs/diagnostics_findings.md`; no number here comes from running the experiment. Line numbers refer to
the repository at `89db760` (before the experiment commits), as the readers verified them.

## What was verified before changing anything (read-only)

The child failure is **not** an implementation bug on the shape path. Every link was checked:

| claim | where |
|---|---|
| The shape head is `nn.Linear(1024, 11)` with PyTorch default init plus a zero `init_betas` residual; one pass, no refinement. | `model.py:544, 566, 647` |
| `Model.forward` applies `torch.sigmoid` once and forwards **all 11** phenotypes by name into the Anny body model; `out['shape']` is the post-sigmoid value, nothing is clamped, dropped or overwritten on the training path. | `model.py:390, 413, 419, 455` |
| The loss is a plain L1 between the post-sigmoid prediction and the dataset's `anny_shape` (clamped to [0,1], positional order = `phenotype_labels`), summed over phenotypes, mean over people, times `alpha_shape` (20 in production). No normalisation, no prior toward 0.5, no per-phenotype weight. | `loss.py:96-97, 138`; GT `train.py:148-159, 205` |
| The shape target goes through exactly the same visibility filter as every other target (root in front of the camera, inside the image, one person per heatmap cell), so GT mesh, GT shape and pose terms are built from the same people. | `train.py:239-316` |
| The inference toggles `neutral_shape`, `neutral_phenotype_names`, `relax_hands`, `rest_pose`, `mask_helper_joints` are all gated by `not is_training` and default off; train.py sets only `mask_helper_joints_train`. | `model.py:153-193, 360-404`; `train.py:859-862` |
| No adult-biased initialisation: zero logits give 0.5 on every phenotype (about 18 y on Anny's age scale, anchors `linspace(-1/3, 1, 5)` -> 0/1/11/25/90 y). The checkpoint's adult bias is a data prior. | `model.py:566, 647, 390`; `scripts/diagnostics/diag_common.py:339-343` |
| Data prior: 2.53% of training people have age < 0.3, 1.06% < 0.2; the holdout (last 100 images) has **no** child; 3DPW labels pin age to 0.55 (only in `--mix_3dpw_labels` runs, not in the production run). | `docs/diagnostics_findings.md`, Children; `scripts/data/smpl_to_anny.py:228-229` |
| In-sample the representation works: youngest-50 babies (GT age 0.015) are predicted at age 0.10 with the right head/height ratio; manifest children (n=10): PVE 47.8 -> 41.6 mm with GT shape (gain 6.2 mm), age bias +0.078. | `docs/diagnostics_findings.md`, Children |

Two defects were found on the way, neither affecting the production checkpoint, both left for their own
commit so the experiment code paths stay byte-identical: `Trainer.prepare_gt` reads the module-global
`args.img_size` (`train.py:239`, compare `self.args.img_size` on 250), and the argparse default
`--use_anny_shape 0` builds GT meshes and shape targets at phenotype 0 (newborn), not the neutral 0.5.

**Conclusion:** the lever is *how often the shape head sees a child* and *how much a child's age error
costs*, i.e. sampling and loss weighting. No architecture change is justified.

## The experiment (all default-off, default path unchanged)

| flag | default | effect when set |
|---|---|---|
| `--age_balance 0/1` | 0 | `WeightedRandomSampler` (with replacement) over the AnnyOne training subset (positions `0..N-val_anny_n-1`): images whose youngest person has GT age < `--age_balance_child_age` share `--age_balance_child_frac` of the draws, the others share the rest, uniform within each group (`train.py build_age_balance_sampler`). Inserted where the mixed-data sampler is built; `shuffle=(sampler is None)` already turns shuffling off. In a `--mix_3dpw_labels` run the synthetic share `1 - r` is redistributed by the same weights. |
| `--age_balance_index PATH` | `results/experiments/exp2_children/age_index.npz` | per-image `min_age` in AnnyOne's sorted-pkl order, written by `scripts/experiments/build_age_index.py` (pkl reads only). Basenames are checked against the dataset so a stale index is refused. |
| `--age_balance_child_frac` | 0.25 | target fraction of **sampled images** that contain a child (natural: a few %; the people-level share stays well below 25% because child images also contain adults). |
| `--age_balance_child_age` | 0.3 | child threshold on the Anny age phenotype (1/3 ~ 11 y). |
| `--exclude_indices_file PATH` | `''` | AnnyOne indices with sampling weight 0; the slurm script writes the diag manifest's image indices so the manifest train split (in-sample for the base model) is **not** seen by the fine-tune. Works with or without `--age_balance`. |
| `--shape_weights w0,...,w10` | `''` (= all ones) | per-phenotype multipliers on the L1 shape error before the sum (`loss.py`, `Loss.__init__` / `forward`). Not renormalised: `1,3,1,1,2,1,1,1,1,1,1` (age x3, height x2) raises the sum of weights from 11 to 14, so with `alpha_shape 20` the shape share of the loss grows by ~27%. All ones (or empty) skip the multiplication entirely. |

Scripts:

- `scripts/experiments/build_age_index.py`: one pkl-only scan of all 579,960 annotations (multiprocessing Pool),
  prints the fraction of images with `min_age < 0.2/0.3/0.4` (train pool and holdout separately) and the scan
  time per 10k files; `--limit` for a smoke test (train.py refuses a partial index).
- `scripts/experiments/measure_batch_ages.py`: rebuilds the real training `DataLoader` (same Subset, collate,
  `drop_last`, workers; optionally the new sampler via `train.build_age_balance_sampler`), iterates N batches
  and prints the age histogram of the **supervised** people (prepare_gt's visibility filter replicated with
  `diag_common.visible_gt_mask` after an Anny body-model forward; the filter is not callable without one),
  the fractions < 0.2/0.3/0.4, the fraction of batches with no child, people per batch and loader images/s.
- `slurm/experiments/exp2_children.sh`: index -> exclusion file -> measurement (baseline and balanced) ->
  2-epoch pilot from `anny_s2_shape_v5/00099.pt` -> traces -> summary.

## Pilot recipe

`train_s2_shape_v5.sh` flags (freeze pattern, lr 1e-5, alphas, boosts, batch 4, `--use_anny_shape 1`) plus
`--age_balance 1 --age_balance_child_frac 0.25 --age_balance_child_age 0.3 --exclude_indices_file <manifest idx>
--shape_weights 1,3,1,1,2,1,1,1,1,1,1 --start_2d_epoch 0 --n_iters_per_epoch 1000 --max_iter 2000 --eval_freq 1
--nb_max_ckpt 2`, checkpoints under `results/experiments/exp2_children/<run>/train/checkpoints/` (never next to
the production checkpoints: train.py keeps the `--nb_max_ckpt` highest epoch numbers of a directory).
`--start_2d_epoch 0` is mandatory: there is no resume, the epoch counter restarts at 0 and the production value
20 would switch the j2d/v2d terms off for the whole pilot. `--gt_center_joint` stays at its default (`root`), so
this experiment is independent of exp1.

```
sbatch slurm/experiments/exp2_children.sh                              # index + measurement + pilot + traces
AGE_BALANCE=0 sbatch slurm/experiments/exp2_children.sh                # control arm: same exclusion and shape weights, uniform draws
SHAPE_WEIGHTS='' sbatch slurm/experiments/exp2_children.sh             # sampler only
SKIP_TRAIN=1 sbatch slurm/experiments/exp2_children.sh                 # index + measurement only
CHILD_FRAC=0.15 sbatch slurm/experiments/exp2_children.sh              # milder oversampling
```

Runtime estimate (from the measured 1.65-1.71 it/s = ~6.7 images/s of the v5 configuration): training 2000
iterations ~ 20 min plus two 100-image V0 evaluations; the three traces (holdout 100 images, manifest train 40
images, youngest-50 with a 20k-pkl scan) ~ 20-30 min; the two 200-batch measurements a few minutes each
(loader bound; the script prints the loader's images/s); the index build is the unknown, I/O bound
(`build_age_index.py` prints seconds per 10k files, 16 processes). A 10-epoch run is `MAX_ITER=10000`
(~1.7 h of training; rescale from the pilot's `[throughput]` line).

## What decides the experiment

Evaluation is fixed: `diag_holdout_trace.py` V1 settings on the 100-image holdout, the diag manifest and the
youngest-50 selection. Baselines are `anny_s2_shape_v5/00099` from `docs/diagnostics_findings.md`.

1. **Adult regression check (holdout, 100 images, V1):** PVE 70.5 must not rise by more than 1 mm, recall 90.2
   not fall by more than 1 point, age bias -0.015 must stay within +/-0.03. The holdout has no children, so this
   is only a "did not break the adults" read.
2. **Children, in-sample reads only (no held-out child exists):**
   manifest train split, tag `child` (n=10; excluded from the fine-tune by `--exclude_indices_file`, but seen by
   the base model for 100 epochs): PVE 47.8, age bias +0.078, shape-swap gain 6.2 mm should all move toward 0;
   Secondary, in-sample read that does not decide the experiment: youngest-50 (`--select youngest`, training
   people resampled by the balanced loader): predicted age 0.10 for GT 0.015 should move toward the GT.
3. **The only out-of-domain read is real photos:** `slurm/diagnostics/diag_crop_camera.sh` (E4) with
   `CKPT=<pilot checkpoint>`: predicted age 0.56-0.66 for small people in the baseline.

Meeting 1 and 2 justifies the 10-epoch run; it does not show that the fix works on real children.

## Open questions only a cluster run can answer

- the measured, not inferred, fraction of batch-4 training batches without a supervised child (baseline), and how
  many children survive the visibility filter compared with adults;
- the wall-clock of the pkl-only scan on `/netscratch`;
- whether oversampling with replacement at lr 1e-5 regresses adult PVE or depth, and which `child_frac` is tolerable;
- the installed anny's age anchors (`bm.anchors['age']`; `diag_common.age_anchors` falls back to an assumed
  `linspace(-1/3, 1, 5)`), which fix what "child < 0.3" means in years.
