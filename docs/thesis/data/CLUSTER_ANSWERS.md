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
