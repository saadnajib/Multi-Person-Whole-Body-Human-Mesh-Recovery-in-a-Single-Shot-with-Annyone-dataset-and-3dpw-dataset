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

## Phenotype correlation on the AnnyOne holdout (19 Oct, computed from the three holdout_per_person CSVs, rows `variant == V1_eval_masked`)

Per phenotype: Pearson r between the predicted and the true value, the slope of pred on gt, and the spread (sd) of the
predictions next to the spread of the labels. Full table: `holdout_phenotype_corr.csv` (n = 222 / 219 / 214 people).

| phenotype | Base r / slope / sd_pred | Control r / slope / sd_pred | Final r / slope / sd_pred | sd of the labels |
|---|---|---|---|---|
| gender | 0.77 / 0.52 / 0.090 | 0.69 / 0.54 / 0.104 | 0.73 / 0.48 / 0.087 | 0.133 |
| age | 0.44 / 0.16 / 0.024 | 0.31 / 0.14 / 0.029 | 0.31 / 0.16 / 0.036 | 0.067 |
| height | 0.52 / 0.36 / 0.038 | 0.55 / 0.38 / 0.039 | 0.48 / 0.37 / 0.043 | 0.055 |
| proportions | 0.74 / 0.57 / 0.058 | 0.66 / 0.42 / 0.048 | 0.67 / 0.50 / 0.056 | 0.074 |
| muscle | 0.41 / 0.21 / 0.053 | 0.35 / 0.16 / 0.049 | 0.27 / 0.13 / 0.052 | 0.108 |
| weight | 0.80 / 0.67 / 0.205 | 0.80 / 0.65 / 0.200 | 0.77 / 0.61 / 0.194 | 0.247 |
| cupsize | 0.70 / 0.52 / 0.179 | 0.68 / 0.50 / 0.175 | 0.69 / 0.47 / 0.164 | 0.243 |
| firmness | 0.45 / 0.36 / 0.162 | 0.45 / 0.35 / 0.159 | 0.46 / 0.34 / 0.149 | 0.198 |
| african | 0.40 / 0.23 / 0.040 | 0.35 / 0.20 / 0.040 | 0.30 / 0.17 / 0.040 | 0.071 |
| asian | 0.65 / 0.26 / 0.036 | 0.62 / 0.26 / 0.037 | 0.62 / 0.25 / 0.036 | 0.087 |
| caucasian | 0.50 / 0.25 / 0.053 | 0.48 / 0.23 / 0.051 | 0.46 / 0.22 / 0.050 | 0.104 |

Reading: on synthetic images, where the labels vary, the final model does vary weight (r 0.77, spread 0.19 against 0.25
in the labels), cup size (0.69) and gender (0.73). Muscle is the weak one (r 0.27, slope 0.13) and it got weaker from
Base (0.41) through the 3DPW fine-tunes, as expected when 30% of the batches carry the converter constant 0.5. Every
slope is below 1: the predictions are pulled towards the middle. The flat shape on real photos (handover 3.4) is
therefore not a model that cannot express shape. The 3DPW constants explain WHERE the final model's real-photo values
land (muscle and weight at about 0.48, the converter constant); the narrow spread itself was already there in Base, which
never saw 3DPW (handover 3.4), so part of it is the synthetic-to-real gap.

## Gender on Relative Human: accuracy and AUC (19 Oct, computed from the three rh_per_person CSVs)

People matched to a labelled box with a male/female tag. Accuracy at the cut-off 0.5 (pred >= 0.5 = woman, Anny's male
end is 0); AUC rank-based (Mann-Whitney, ties averaged), women = positive, 0.5 = chance. The "best accuracy" column
scans cut-offs on Relative Human itself, so it only shows the size of the threshold effect; a real cut-off must come
from the AnnyOne holdout.

| model (Relative Human job) | n | accuracy at 0.5 | AUC | mean pred women / men | best accuracy (cut-off) |
|---|---|---|---|---|---|
| Base, v5 00099 (3611285_0) | 932 | 0.465 | 0.562 | 0.55 / 0.54 | 0.636 (0.62) |
| Final (3612131) | 1,308 | 0.642 | 0.780 | 0.86 / 0.55 | 0.723 (0.89) |
| Final-masked, full mask (3612325; RH 3613616) | 1,163 | 0.493 | 0.705 | 0.86 / 0.72 | 0.685 (0.95) |

Reading: the masked model still ranks women above men (AUC 0.71); its predictions moved up and the 0.5 cut-off calls
most men women. Its low accuracy is a threshold effect, not a lost skill. The partial mask (job 3614042, RH 3616821) is
not in the table: its per-person CSV is not in the repo yet. Report accuracy and AUC for every model.
