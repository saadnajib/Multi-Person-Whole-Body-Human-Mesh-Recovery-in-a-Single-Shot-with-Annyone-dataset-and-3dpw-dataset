# Model names: thesis name -> old run name

The thesis, the deck and the handover use the thesis names in the first column. This file and Table A.1 of the
thesis (`docs/thesis/tables/T19_model_names.tex`, first table of Appendix A) are the only places where the old run
and checkpoint names (v2-v10, exp7, distillation v1-v8, ...) appear next to them. Paths are relative to
`/netscratch/najib/multi-hmr` on the cluster.

**How the names are built.** "L" is our ViT-L line: every model that keeps the ViT-L backbone. "L-stage 1" to
"L-stage 4" are the steps of the transfer schedule that turned NAVER's released model into ours (new heads, then a
frozen backbone, then the last four blocks trained, then the neck and hand loss weights). "Base" is the end of that
schedule and the start of every later fine-tune. "Mixed-L" and "Fresh-heads-L" are the two older ViT-L runs that
also trained on real 3DPW images. "Final" and "Control" are the two arms of the last experiment: the same 10-epoch
fine-tune of Base with and without the two fixes (depth targets at the head joint, bone correction).
"Final-masked" and "Final-partial-mask" add a shape mask for 3DPW people to Final. "Student-S" is the history of
ViT-S students (the number is the old version number); "Student" is the ViT-B model distilled from Final.
"Multi-HMR-Anny" is NAVER's released checkpoint: the starting weights of our ViT-L line and the baseline of
requirement 1b.

"Not recorded" = no script, log or document in the repository states it. Learning rates halve every 30 epochs unless
noted; "planned" epochs are the script's `--max_iter` / 1,000; the epoch kept is the one in the name. The history
scripts (`slurm/history/`) were edited after their runs, so the L-stage recipes are as recorded there.

| thesis name | old run / checkpoint name | starting weights | backbone | training data | schedule (epochs x iterations x batch, lr) | key flags | checkpoint path |
|---|---|---|---|---|---|---|---|
| Multi-HMR-Anny | `multiHMR_672_L_anny` (NAVER's release) | not recorded (trained by NAVER) | ViT-L/14, NAVER's 8-layer cross-attention decoder (76.6 M head) | AnnyOne, BEDLAM, COCO, MPII (as `train_step1_pretrained_bb.sh:12` records it); no 3DPW | not recorded | original-repo class `Multi_HMR`; evaluated with `--model_class multi_hmr_original` | `models/multiHMR/multiHMR_672_L_anny.pt` |
| L-stage 1 | `anny_s1_pretrained_bb_v2` (`slurm/history/train_step1_pretrained_bb.sh` + `_resume`) | Multi-HMR-Anny backbone, fresh heads | ViT-L/14, all blocks trained | AnnyOne | 150 planned x 1,000 x 4, lr 1e-4; resumed from its epoch 177 at 3e-5 | `--load_only_backbone 1 --freeze_backbone 0`, 2D terms from epoch 20 at weight 0.1 | not recorded (side branch, not on the path to Base) |
| L-stage 2 | `anny_s2_frozen_bb_v2` epoch 299 ("v2 frozen"; `train_step2_frozen_bb.sh` + `_resume`) | Multi-HMR-Anny backbone, fresh heads | ViT-L/14, frozen (heads only) | AnnyOne | 150 planned x 1,000 x 4, lr 1e-4; resumed from its epoch 283 at 3e-5; epoch 299 kept | `--load_only_backbone 1 --freeze_backbone 1`, 2D terms from epoch 20 | `models/multiHMR/anny_s2_frozen_bb_v2_00299.pt` |
| L-stage 3 | `anny_s2_partialft_v3` epoch 54 ("v3 partial"; `train_s2_partialft.sh`) | L-stage 2 epoch 299 | ViT-L/14, last 4 of 24 blocks trained | AnnyOne | 300 planned x 1,000 x 4, lr 1e-5; epoch 54 kept | `--unfreeze_last_n_blocks 4 --mask_helper_joints_train 1` | `logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt` |
| L-stage 4 | `anny_s2_boosted_v4` epoch 299 ("v4 boosted"; `train_s2_boosted_v4.sh`) | L-stage 3 epoch 54 | as L-stage 3 | AnnyOne | 300 x 1,000 x 4, lr 1e-5 | `--boost_neck_weight 4 --boost_hand_weight 4` | `logs/anny_model/anny_s2_boosted_v4/checkpoints/00299.pt` |
| Base | `anny_s2_shape_v5` epoch 99 ("v5"; `train_s2_shape_v5.sh`) | L-stage 4 epoch 299 | as L-stage 3 | AnnyOne (579,860 training images) | 100 x 1,000 x 4, lr 1e-5 (1.25e-6 in the last 10) | `--alpha_shape 20`, 2D terms from epoch 20 | `logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt` (copy `models/multiHMR/anny_s2_shape_v5_00099.pt`) |
| Mixed-L | `anny_teacher_v9_mixed` epoch 62 ("v9"; also epoch 149; `slurm/train/train_teacher_v9_mixed.sh`) | Base | as L-stage 3 | 70% AnnyOne + 30% 3DPW train, old labels (`threedpw_anny_full/labels`) | 150 planned x 1,000 x 4, lr 1e-5 halved every 60 | `--mix_3dpw_ratio 0.3 --start_2d_epoch 0 --tf32 1`; no hand or shape mask | `models/multiHMR/anny_teacher_v9_mixed_00062.pt` (and `_00149.pt`) |
| Fresh-heads-L | `anny_v10_scratch_mixed` epoch 290 ("v10"; `slurm/train/train_v10_scratch_mixed.sh`). Despite "scratch": NAVER's DINOv2 ViT-L backbone from Multi-HMR-Anny plus fresh heads | Multi-HMR-Anny backbone, fresh heads | as L-stage 3 | 70% AnnyOne + 30% 3DPW train, old labels, from the first step | 300 planned x 1,000 x 4, lr 1e-5 | `--load_only_backbone 1 --unfreeze_last_n_blocks 4 --alpha_shape 20 --mix_3dpw_ratio 0.3`; no neck/hand boost | `logs/anny_model/anny_v10_scratch_mixed/checkpoints/00290.pt` (copy `models/multiHMR/anny_v10_scratch_mixed_00299.pt`) |
| Control | exp7 control arm `v5_root_dist1_nofix_v23_e10` epoch 9; job 3611339 | Base | as L-stage 3 | 70% AnnyOne + 30% 3DPW train, v23 labels (`threedpw_anny_v23/labels`) | 10 x 1,000 x 4, lr 1e-5 | Base recipe + `--start_2d_epoch 0 --mask_3dpw_hands 1 --gt_center_joint root --alpha_dist 1 --alpha_transl 1 --bone_fix none` (thesis Appendix C) | `results/experiments/exp7_final_model/v5_root_dist1_nofix_v23_e10/train/checkpoints/00009.pt` |
| Final | exp7 fix arm `v5_pc_dist5_bonefix_v23_e10` epoch 9; job 3611336 | Base + Base's bone correction | as L-stage 3 | as Control | as Control | as Control but `--gt_center_joint person_center --alpha_dist 5 --alpha_transl 5 --bone_fix results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json` | `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt` (copy `models/multiHMR/exp7_v5_final_00009.pt`) |
| Final-masked | exp7 `v5_pc_dist5_bonefix_v23_e10_shpmask` epoch 9; job 3612325 | as Final | as L-stage 3 | as Control | as Control | Final + `--mask_3dpw_shape 1` (8 converter constants) | `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/train/checkpoints/00009.pt` |
| Final-partial-mask | exp7 `v5_pc_dist5_bonefix_v23_e10_shpmask_noage`; job 3614042 (running on 18 Oct) | as Final | as L-stage 3 | as Control | as Control | Final + `--mask_3dpw_shape 1 --mask_3dpw_shape_names african,asian,caucasian,cupsize,firmness,muscle,weight` (age stays supervised) | `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask_noage/train/checkpoints/00009.pt` (when finished) |
| Student-S/1 | distillation v1, `anny_distill_vits` (`slurm/history/train_distill_vits.sh`) | ImageNet DINOv2-S, random heads; teacher Base | ViT-S/14, all blocks trained | AnnyOne | 300 planned x 1,000 x 4, lr 1e-4 | feature distillation `--lambda_kd 1.0`, 2D terms from epoch 20 | not recorded |
| Student-S/2 | distillation v2, `anny_distill_vits_v2_a` (`train_distill_vits_v2.sh`) | a Student-S/1 checkpoint, recorded as `anny_distill_vits_v2` epoch 199; teacher Base | as Student-S/1 | AnnyOne | 200 planned x 1,000 x 4, lr 2e-5 halved every 60 | as Student-S/1, 2D terms from epoch 0 | not recorded |
| Student-S/3 | distillation v3, `anny_distill_vits_v3` | ImageNet DINOv2-S; 54 of 67 head tensors from Base; teacher Base | as Student-S/1 | AnnyOne | 150 planned x 1,000 x 4, lr 5e-5 halved every 60 | `--init_heads_from_teacher 1 --lambda_kd_out 1.0` | not recorded |
| Student-S/4 | distillation v4, `anny_distill_vits_v4` | NAVER's `multiHMR_672_S` backbone; heads as Student-S/3 | as Student-S/1 | AnnyOne | 150 planned x 1,000 x 8, lr 7e-5 halved every 60 | `--pretrained models/multiHMR/multiHMR_672_S.pt --load_only_backbone 1` | not recorded |
| Student-S/5 | distillation v5, `anny_distill_vits_v5` | as Student-S/4 | as Student-S/1 | AnnyOne | as Student-S/4 | `--freeze_heads_epochs 5 --start_2d_epoch 5` | not recorded |
| Student-S/6 | distillation v6, `anny_distill_vits_v6` | NAVER's `multiHMR_672_S` backbone; all 67 head tensors from Base through a 384 -> 1024 projector | ViT-S/14 + projector | AnnyOne | as Student-S/4; stopped at epoch 138 by the time limit | `--head_dim 1024` | `logs/anny_model/anny_distill_vits_v6/checkpoints/00137.pt` |
| Student-S/6b | distillation v6b, `anny_distill_vits_v6b` | Student-S/6 epoch 137; teacher Base | as Student-S/6 | AnnyOne | 150 planned x 1,000 x 8, lr 2e-5 halved every 60 | `--teacher_bf16 1` | `logs/anny_model/anny_distill_vits_v6b/checkpoints/00140.pt` |
| Student-S/6c | distillation v6c, `anny_distill_vits_v6c` | Student-S/6b epoch 140; teacher Base | as Student-S/6 | AnnyOne | 150 x 1,000 x 16, lr 3e-5 halved every 60 | batch 16 | `models/multiHMR/anny_distill_vits_v6c_00149.pt` |
| Student-S/mixed | distillation v7, `anny_distill_vits_v7_mixed` | Student-S/6b epoch 140; teacher Base | as Student-S/6 | 70% AnnyOne + 30% 3DPW train, old labels | 150 planned x 1,000 x 8, lr 2e-5 halved every 60; epoch 130 kept | `--mix_3dpw_ratio 0.3` | `models/multiHMR/anny_distill_vits_v7_mixed_00130.pt` |
| Student-S/final | distillation v8, `anny_distill_vits_v8_final` (evaluated as `evaltest_student_v8_final`; `slurm/train/train_distill_vits_v8_final.sh`) | Student-S/6c epoch 149; teacher Mixed-L | as Student-S/6 | as Student-S/mixed | 150 planned x 1,000 x 16, lr 3e-5 halved every 60; epoch 52 reported | teacher Mixed-L, `--mix_3dpw_ratio 0.3` | `models/multiHMR/anny_distill_vits_v8_final_*.pt` (newest copy) |
| Student | exp8 `student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30`; job 3614191 (running on 18 Oct) | NAVER's `multiHMR_672_B` backbone; every head tensor from Final through a 768 -> 1024 projector; teacher Final | ViT-B/14 + projector, all blocks trained | as Control | 30 x 1,000 x 8, lr 3e-5 halved every 12 | Final's targets and masks + `--head_dim 1024 --init_heads_from_teacher 1 --freeze_heads_epochs 5 --teacher_bf16 1 --lambda_kd 1.0 --lambda_kd_out 1.0 --kd_temperature 4`; teacher's bone correction during training, its own after | `results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/train/checkpoints/` (last epoch `00029.pt`, when finished) |

Notes:
- The colour-jitter flags (`--brightness 0.2`, and in the ViT-B student also `--contrast`, `--saturation`, `--hue`)
  are passed by the Student-S/4 and later scripts and by the Student script, but `train.py` only parses them
  (`train.py:1722-1725`); no code reads them. No data augmentation was used.
- Sources: `slurm/history/*.sh`, `slurm/train/*.sh`, `slurm/experiments/exp7_final_model.sh`,
  `slurm/experiments/exp8_distill_vitb.sh`, `docs/handover.md` sections 0a, 1, 3.2 and 8, the draft's tables
  `tab:main`, `tab:student_runs`, `tab:kd_epochs`, `tab:kd_eval` (`docs/thesis/draft_2026-10_original/main.tex`).
