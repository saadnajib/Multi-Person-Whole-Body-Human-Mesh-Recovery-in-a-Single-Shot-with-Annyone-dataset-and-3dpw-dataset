# Statistics behind the pgfplots figures F15, F16, F20

Computed from the CSVs in this folder (see `CLUSTER_ANSWERS.md` for their origin and column meanings) with the
scratchpad script `thesis_plots/gen.py` (pandas, numpy). Cite as "data/plot_stats.md". One run per model, no seed.

## F15 depth scatter (figures/plots/F15_depth_scatter.tex)

Source: `holdout_per_person_base.csv`, `holdout_per_person_control.csv`, `holdout_per_person_final.csv`, rows with `variant == V1_eval_masked`
(AnnyOne holdout, 100 images, eval setting, masked; one row per matched person; each body at its own predicted depth).
Fit: least squares, pred = a * true + b, in metres. All points plotted (no subsampling).

| quantity | Base | Control | Final |
|---|---|---|---|
| people (matched) | 222 | 219 | 214 |
| images with at least one matched person | 92 | 92 | 89 |
| slope a | 0.333 | 0.276 | 0.419 |
| intercept b (m) | 2.360 | 2.661 | 1.746 |
| Pearson r (true vs predicted depth) | 0.511 | 0.457 | 0.558 |
| fit crosses identity at (m) | 3.54 | 3.67 | 3.01 |
| median abs(pred - true) (m) | 1.033 | 1.237 | 0.626 |
| mean abs(pred - true) (m) | 1.081 | 1.268 | 0.654 |
| share predicted too far (pred > true) | 100.0% | 99.1% | 94.4% |
| mean root error `root_err_mm` (mm) | 1208 | 1402 | 728 |
| median root error `root_err_mm` (mm) | 1102 | 1312 | 654 |
| true depth range (m) | 0.59 to 3.42 | 0.59 to 3.75 | 0.59 to 3.75 |

Handover reference slopes: Base 0.33 (intercept 2.36 m), Control 0.28, Final 0.42. Recomputed: Base 0.33
(intercept 2.36 m), Control 0.28, Final 0.42. The Control and Final mean root errors reproduce
the 1402 and 728 mm of the traces. The people count is the matched people; the 100 images hold 222 ground-truth people.

## F16 Relative Human predicted age (figures/plots/F16_rh_age.tex)

Source: `rh_per_person_base.csv`, `rh_per_person_final.csv`, `rh_per_person_masked.csv` (Relative Human validation,
2,130 labelled people each), rows with `matched` true; `pred_age` grouped by `age_group`. Box = quartiles and median,
whiskers = 5th and 95th percentile (numpy linear interpolation). "Below 0.45" = share of matched people of that group
with predicted Anny age below 0.45 (child-shaped).

Matched people: Base 1003, Final 1379, Final-masked 1219 of 2,130 (these totals
include the age group "unknown": 2 / 2 / 1 people, not plotted).

| model | group | n | 5th pct | lower quartile | median | upper quartile | 95th pct | mean | below 0.45 |
|---|---|---|---|---|---|---|---|---|---|
| Base | adult | 821 | 0.553 | 0.580 | 0.606 | 0.699 | 0.756 | 0.631 | 0.0% (0) |
| Base | teen | 53 | 0.542 | 0.558 | 0.579 | 0.605 | 0.719 | 0.597 | 0.0% (0) |
| Base | kid | 89 | 0.523 | 0.548 | 0.583 | 0.693 | 0.754 | 0.613 | 0.0% (0) |
| Base | baby | 38 | 0.494 | 0.552 | 0.608 | 0.706 | 0.767 | 0.624 | 2.6% (1) |
| Final | adult | 1113 | 0.485 | 0.511 | 0.530 | 0.562 | 0.606 | 0.536 | 1.2% (13) |
| Final | teen | 72 | 0.475 | 0.501 | 0.518 | 0.548 | 0.581 | 0.524 | 1.4% (1) |
| Final | kid | 119 | 0.343 | 0.478 | 0.505 | 0.551 | 0.593 | 0.500 | 16.8% (20) |
| Final | baby | 73 | 0.281 | 0.382 | 0.456 | 0.505 | 0.578 | 0.438 | 49.3% (36) |
| Final-masked | adult | 1002 | 0.537 | 0.560 | 0.580 | 0.772 | 0.833 | 0.638 | 0.1% (1) |
| Final-masked | teen | 61 | 0.526 | 0.552 | 0.563 | 0.619 | 0.811 | 0.616 | 0.0% (0) |
| Final-masked | kid | 96 | 0.434 | 0.538 | 0.563 | 0.715 | 0.821 | 0.611 | 6.2% (6) |
| Final-masked | baby | 59 | 0.399 | 0.472 | 0.515 | 0.550 | 0.798 | 0.524 | 16.9% (10) |

Comparison with the handover (means): the recomputed means equal the handover means to three decimals (Final adult
0.536, kid 0.500, baby 0.438; Base 0.631 / 0.613 / 0.624; Final-masked 0.638 / 0.611 / 0.524), and the baby shares
below 0.45 are Final 49.3%, Final-masked 16.9%, Base 2.6% (1 of 38; the handover rounds to 3%).

Medians that differ from the means by more than 0.03 (quote the median when describing the box plot):
- Base kid: median 0.583 vs mean 0.613 (-0.030)
- Final-masked adult: median 0.580 vs mean 0.638 (-0.059)
- Final-masked teen: median 0.563 vs mean 0.616 (-0.052)
- Final-masked kid: median 0.563 vs mean 0.611 (-0.048)

## F20 in-training curves (figures/plots/F20_val_curves.tex)

Source: `val_curves_exp7.csv` (per-epoch evaluation during training, epochs 1-10, one run each, no seed). The holdout
columns are the in-training AnnyOne holdout evaluation (setting V0), not the V1 trace numbers (73.7 / 77.4 mm).

| series | summary (PVE, mm) |
|---|---|
| 3DPW validation, Final | min 73.4 (epoch 7), last 74.5, mean over epochs 75.69 |
| 3DPW validation, Control | min 75.5 (epoch 4), last 75.9, mean over epochs 77.72 |
| AnnyOne holdout (V0, in training), Final | min 78.9 (epoch 9), last 79.3, mean over epochs 80.73 |
| AnnyOne holdout (V0, in training), Control | min 76.8 (epoch 9), last 81.2, mean over epochs 79.19 |
