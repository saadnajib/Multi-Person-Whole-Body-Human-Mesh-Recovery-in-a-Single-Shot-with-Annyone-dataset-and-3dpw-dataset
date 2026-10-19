# Answers from the cluster for the thesis (18 Oct 2026, evening; pasted by the student)

- Colour jitter: `grep -rn -i "jitter" --include=*.py . | grep -v archive` on the cluster copy returns NOTHING.
  No augmentation flag exists in the code: the thesis says no data augmentation was used (plan 7, Q8).
- In-training curves of the final model (job 3611336) and its control (job 3611339), per epoch, from the job logs
  (`sed -n 55,164p logs/exp7_final_model_3611336.out`, `sed -n 48,157p logs/exp7_final_model_3611339.out`):
  `val_curves_exp7.csv` in this folder (holdout = in-training AnnyOne holdout eval, V0 setting as train.py prints it,
  NOT the V1 trace number 73.7 / 77.4; 3dpwval = 3DPW-Anny validation, 300 images, v23 labels). Throughput: final
  10.9 img/s (A100-class), control 6.6 img/s (slower node): same recipe, different GPU.
- ViT-L history runs exist under logs/anny_model/: `anny_v7_scratch`, `anny_v8_resume_anny_partial_freeze` (plan 7, Q7);
  their last eval lines are still to be requested. Also present: anny_distill_vits_v7_mixed, anny_distill_vits_v8_final,
  evaltest_student_v8_final__threedpw_anny_v23 (a v8 ViT-S student already evaluated on the v23 labels at some point).
- Holdout trace folders in results/diagnostics/: holdout_anny_s2_shape_v5_00099 (and _rerun), holdout_anny_teacher_v9_00062,
  holdout_anny_teacher_v9_00149, holdout_anny_v10_00290..00293, manifest_trace_holdout_anny_s2_shape_v5_00099.
- per_person.csv header of the holdout traces (exp7 final and control, `holdout_ft/per_person.csv`): gid, gt_depth_m,
  pred_depth_m, root_depth_err_m, root_err_mm, root_px_err, head_px_err, pve, pa_pve, mpjpe, pve_headneck,
  pve_gtpose_predshape, pve_predpose_gtshape, neck_ratio, gt_skel_height_m, pred_skel_height_m, pheno_gt_* and
  pheno_pred_* and pheno_err_* for the 11 phenotypes, rot_* per region (rot_neck, rot_head, rot_fingers, ...),
  rot_global_head_deg, score, tags, variant, idx, person. -> F15 uses gt_depth_m vs pred_depth_m (variant = V1 rows).
- per_person.csv header of the Relative Human evals (`results/diagnostics/relative_human_val_<tag>/per_person.csv`):
  image, rh_index, age_group, gender, depth_id, gt_box_src, gt_x1..gt_y2, matched, iou, det_index, det_score,
  pred_gender, pred_age, pred_muscle, pred_weight, pred_height, pred_proportions, pred_cupsize, pred_firmness,
  pred_african, pred_asian, pred_caucasian, pred_x1..pred_y2. -> F16 uses age_group vs pred_age where matched is true.
- Jobs submitted 18 Oct evening for thesis figures: 3615018 (demo, predicted fingers, `run_demo_compare_hands.sh`),
  3615019 (demo, rest pose, Base + Final + Final-masked, `run_demo_compare_rest.sh`), 3615020_[2,4] (ViT-S students
  v6c and v8 on the v23 3DPW test labels). Run 3 (3614042) was RUNNING (33 min) at that time; student 3614191 pending.

## Demo montages received (job 3612078, Base v5 | Final, box placement, 18 Oct)
Saved under docs/thesis/figures/img/ as F10_photo_{garden,tennis,stage,party}_v5_vs_final.jpg and
F10_3dpw_{guitar,office,fencing}_v5_vs_final.jpg. Each montage: photo | Base | photo | Final, labels baked in
(anny_s2_shape_v5_00099 = Base, exp7_v5_final_00009 = Final). Still missing: downtown_bar, downtown_walking,
stairs (downtown_sitOnStairs), downtown_walkBridge/walking pair, weekly market (5 of the 8 3DPW frames).

## Zip batch 1 received (18 Oct, 21:20; thesis_figs_batch1.zip, 33 images + 5 CSVs)
- figures/img/F10_3dpw_{bar,runforbus,stairs,walking,market}_v5_vs_final.jpg: the 5 missing demo montages (job 3612078).
  All 12 montages are now present (4 photos + 8 3DPW test frames).
- figures/img/F21_age_sweep.jpg (diag_bodymodel), F14_label_check_{courtyard_arguing_00024 (two-person frame),
  downtown_walking_00771, climbing_00633, basketball_00570}.jpg (check_3dpw_conversion_v23/vis; 12 panels exist, 4 copied),
  exp6_compare_579864.jpg and exp6_compare_579866.jpg (F23), F12_holdout_<idx>_img_gt_v5_final.jpg x 6
  (panels: image | GT | Base | Final, each at its own depth; Base = v5 trace job 3570222, Final = exp7 holdout_ft).
- data/holdout_per_person_{final,control}.csv (exp7 final 3611336 and control 3611339 holdout traces; 834/836 rows;
  the per-person rows carry `variant` in {V0_eval_unmasked, V1_eval_masked, V2_demo, V3_demo_relaxhands}; use
  V1_eval_masked for the headline setting; columns gt_depth_m, pred_depth_m, root_err_mm, pve, ...).
- data/rh_per_person_{base,final,masked}.csv (Relative Human val, 2,130 people each; columns age_group, gender, matched,
  pred_age, pred_gender, ...; base = v5 job 3611285_0, final = job 3612131, masked = job 3613616).
- Not yet received: the v5 (Base) holdout trace per_person.csv (results/diagnostics/holdout_anny_s2_shape_v5_00099/per_person.csv)
  for the Base line of F15; the gt_decode overlays (F24; folder name unknown); the v7/v8 ViT-L run folders' contents.

## 18 Oct, 22:40: more logs from the student
- Student-S on the v23 3DPW test labels (job 3615020, `logs/eval-test3dpw-3615020_{2,4}.out`, all 8,190 images, V0 setting: run_eval_3dpw_test.sh, helper bones unmasked, as handover 0b states):
  - Student-S/6c (`anny_distill_vits_v6c_00149.pt`): AnnyOne test-100 PVE 114.5 / PA-PVE 84.6 / MPJPE 118.4, precision 81, recall 89;
    3DPW test PVE 142.6 / PA-PVE 91.0 / MPJPE 147.5, precision 45, recall 98.
  - Student-S/final (`anny_distill_vits_v8_final_00052.pt`): AnnyOne 120.6 / 86.9 / 125.0, precision 76, recall 87;
    3DPW test PVE 121.6 / PA-PVE 83.7 / MPJPE 123.3, precision 56, recall 98.
  -> fills the "Student-S on v23" TODO in chapter 6.11 / T20; the old-label numbers (139.5 -> 123.6) stay retired.
- Student (ViT-B, job 3614191) started training 21:23 on an L40S (serv-3313); checks PASS, teacher bone fix written; result ~20 h later.
- Demo figure jobs 3615018 (predicted fingers) and 3615019 (rest pose, Base/Final/Final-masked) finished; montages made on the
  login node (demo_montage_hands_{photos,3dpwtest}, demo_montage_rest_{photos,3dpwtest}); zip batch 2 (45 MB) ready on the cluster,
  not yet uploaded here.
- v7/v8 ViT-L history runs have checkpoints and tfevents only, no train.log: dropped from the thesis history.

## Zip batch 2 received (19 Oct, 00:05; three zips)
- F11 predicted fingers (demo job 3615018, --relax_hands 0, Base | Final, 2 panels each): figures/img/F11_hands_photo_{garden,tennis,stage,party}.jpg,
  F11_hands_3dpw_{bar,guitar,office,market,walking,fencing,stairs,runforbus}.jpg. Caption note: Base's metacarpal1.L is flipped in these
  renders (the correction is only in Final), so Base's left index base looks bent backwards.
- F17 rest-pose line-up (demo job 3615019, --rest_pose 1, Base | Final | Final-masked, 3 panels): F17_rest_photo_{garden,tennis,stage,party}.jpg,
  F17_rest_3dpw_{...}.jpg (same eight frames).
- F12 proper V1 holdout montages (image | GT | Base V1 | Final V1, 672-px panels, 4-px gaps): F12_holdout_<idx>_img_gt_base_final.jpg x 6.
- F24 ground-truth decode overlays (check 1): F24_gt_decode_1..4.jpg (holdout 579888 p0, holdout 579925 p0, train 462901 p0, train 326611 p3);
  data/gt_decode_summary.json.
- Demo logs: data/demo_compare_3615018_hands.log, data/demo_compare_3615019_rest.log (the [shape] lines per detection for the spread numbers).

## 19 Oct, 00:30: Final-partial-mask (run 3, job 3614042) DONE; its Relative Human eval submitted as job 3616821
- Holdout V1 (compare table, last column, `logs/exp7_final_model_3614042.out:191-215`): PVE 73.0, PA-PVE 51.2, recall 87.4,
  root error 680 mm, depth slope 0.42 (intercept 1.69 m), size ratio 0.79, girth 1.005, neck PVE 10.5 mm, global head 13.2 deg,
  fingers 8.2 deg, neck02 5.0, head 8.9, finger4-1.R 9.2, metacarpal1.L 0.0; verdicts OK; neck kept, depth kept, cost vs control kept.
  Bone fix estimated for this run: neck02 178.8, head 179.2, metacarpal1.L 179.8, finger4-1.R 179.7 deg.
- AnnyOne test-100 V0 (run_eval_3dpw_test.sh): PVE 76.6, PA-PVE 55.2, precision 81, recall 87, MPJPE 77.9.
- 3DPW test v23, all 8,190 images: PVE 89.9, PA-PVE 58.9, precision 66, recall 99, MPJPE 90.0.
  (Final: 73.7 / 50.3 / 87.0; root 728; 3DPW 89.0 / 57.6 / 64 / 99. Final-masked: 71.7 / 50.7 / 86.2; 3DPW 91.1 / 59.0.)
- Everything is within the run-to-run noise of Final; the decision rests on Relative Human (job 3616821) and the demo spread.

## 19 Oct, 01:20: Relative Human of Final-partial-mask (job 3616821, `logs/eval_rh_3616821_0.out`) -> DECISION: unmasked Final stays
- val, 635 images, 2,130 people, iou 0.3, child_thr 0.45: overall recall 0.630 (1341/2130).
  adult n_lab 1680 matched 1086 recall 0.646 age mean 0.532 (median 0.524) <thr 0.003 height 0.527 g_acc 0.468 (n 1050);
  teen 101 / 70 / 0.693 / 0.515 / 0.513 / 0.000 / 0.520 / 0.500 (68); kid 209 / 117 / 0.560 / 0.506 / 0.505 / 0.103 / 0.497 / 0.495 (107);
  baby 117 / 67 / 0.573 / 0.465 / 0.478 / 0.343 / 0.504 / 0.556 (45).
  child-vs-adult AUC 0.727 (n_pos 184, n_neg 1086); gender accuracy 0.474 (n 1271), pred gender mean male 0.725, female 0.888.
- Rule (gender >= 60%, AUC >= 0.7, spread opened): gender FAILS (47.4%), AUC passes (0.727), spread not measured (no demo run; not needed).
  -> The unmasked Final (3611336) remains the thesis model and the Student's teacher. Final-partial-mask is the fourth column of
  the shape-mask trade-off (chapter 6.8, Table 6.2): children kept (age label kept), gender lost like the full mask.
- Reading, untested: the gender label was still supervised here and gender accuracy still collapsed (both sexes predicted
  female-side), so Final's gender gain likely depends on the seven 0.5 constants on 3DPW men (cup size, firmness, ...) as much as on
  the gender label. Not tested further.

## 19 Oct morning: STUDENT DONE (job 3614191, `logs/exp8_distill_vitb_3614191.out`) -> 95% rule FAILS
- Training: 30 epochs x 1000 x batch 8 on an L40S, 20,894 s (5.8 h), 11.6 img/s; 77 NaN/Inf steps skipped; heads frozen epochs 0-4.
  In-training curve (V0 holdout PVE / 3DPW-val PVE) at epochs 5, 10, 15, 20, 25, 30: 165.1/104.9, 170.0/101.5, 139.5/93.0,
  133.0/90.9, 130.5/88.0, 122.8/87.7 (log lines 59-143). AnnyOne still falling ~8 mm per 5 epochs; 3DPW-val flat (88.0 -> 87.7).
- Student bone fix (own estimate): neck02 178.9, head 179.1, metacarpal1.L 179.7, finger4-1.R 179.9 deg (0.2-1.8 deg from the teacher's);
  holdout with own fix 119.6 vs with the teacher's 119.7 (no difference).
- Holdout V1 (lines 195-207): student PVE 119.6, PA-PVE 80.8, MPJPE 127.2, recall 85.4, FP 64; teacher 73.7 / 50.3 / 77.9 / 87.0, FP 42.
  Region rotation errors (deg) student vs teacher: root 13.5 vs 8.7, spine 6.5/5.8, neck 5.7/5.5, head 4.5/5.0, arm 10.6/8.4,
  fingers 9.3/8.3, leg 6.4/5.1. Root error 830.7 vs 728.5 mm.
- AnnyOne test-100 V0: student 122.7 / 84.2 / 127.7, recall 85; teacher 77.5 / 54.4 / 78.5, recall 87.
- 3DPW test v23 (8,190 images): student PVE 110.1, PA-PVE 71.8, MPJPE 112.0, recall 99; teacher 89.0 / 57.6 / 88.3 / 99.
- Cost on an L40S (lines 213-221): teacher 318.38 M / 1927.4 GFLOPs / 59.7 ms full (50.7 ms backbone); student 101.38 M (backbone 86.58,
  feat_proj 0.79, heads 14.01) / 603.3 GFLOPs / 28.5 ms full (19.1 ms backbone); 3.14x fewer params, 3.19x fewer FLOPs, 2.10x faster.
- Final table (lines 259-283): bars AnnyOne PVE <= 77.6, MPJPE <= 82.0; 3DPW PVE <= 93.7, MPJPE <= 92.9. VERDICT: FAIL on all four.
- Against Multi-HMR-Anny (NAVER): 3DPW 110.1 vs 126.9 mm (student better), 101.38 vs 380.94 M params (3.76x fewer);
  AnnyOne 119.6 vs 60.3 (NAVER likely in-sample). Against NAVER's bars (63.5 / 133.6): 3DPW met, AnnyOne not met.
