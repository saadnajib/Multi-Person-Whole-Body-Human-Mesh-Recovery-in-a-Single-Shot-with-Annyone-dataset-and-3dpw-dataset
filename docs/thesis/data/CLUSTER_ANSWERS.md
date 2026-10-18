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
