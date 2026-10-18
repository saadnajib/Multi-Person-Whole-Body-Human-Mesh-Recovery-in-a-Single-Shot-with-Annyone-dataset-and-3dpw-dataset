# data/

CSV files read by the pgfplots figures in `figures/plots/`. None has arrived yet. Send `head -1` of each source
first, so the column names in the plot files can be set (REWRITE_PLAN.md section 7, Q11).

| file | for | source on the cluster (`/netscratch/najib/multi-hmr`) |
|---|---|---|
| `F15_base_per_person.csv` | F15 depth scatter | v5 trace (job 3570222), `per_person.csv` |
| `F15_final_per_person.csv` | F15 | `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/holdout_ft/per_person.csv` |
| `F15_control_per_person.csv` | F15 (optional) | `results/experiments/exp7_final_model/v5_root_dist1_nofix_v23_e10/holdout_ft/per_person.csv` |
| `F16_rh_base.csv` | F16 Relative Human age | `results/diagnostics/relative_human_val_anny_s2_shape_v5_00099/per_person.csv` |
| `F16_rh_final.csv` | F16 | `results/diagnostics/relative_human_val_exp7_v5_final_00009/per_person.csv` |
| `F16_rh_masked.csv` | F16 (optional) | the masked arm's Relative Human `per_person.csv` |
| `F19_student.csv` | F19 student curves | columns `epoch,holdout_pve,val3dpw_pve`, from the Student's `train.log` (job 3614191) |
| `F20_val_curves.csv` | F20 (once the Control's curve exists) | columns `epoch,final,control`; `sed -n 48,157p logs/exp7_final_model_3611339.out` and `sed -n 55,164p logs/exp7_final_model_3611336.out` |
