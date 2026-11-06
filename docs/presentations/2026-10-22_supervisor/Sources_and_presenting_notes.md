# Presentation notes

Built from the seven topic slides in the supplied template. Dense architecture, training and dataset topics have been expanded into 21 main slides plus 3 backup slides. The supplied template contained no pictures or speaker notes. Following the supervisor's feedback, the body models (slide 4) and the dataset block (slides 5–8) now come before the method (slides 9–16).

## Before presenting
- Open the PPTX in PowerPoint and use Presenter View for the full speaking script.
- Copy the real video from `/netscratch/najib/multi-hmr/results/video_compare/video_3642415/comparison.mp4` onto your presentation computer. Slide 19 has a dedicated video placeholder, as requested. This delivery does not contain the video.
- On slide 19, remove the placeholder and use Insert > Video > This Device. Test video playback before the meeting.
- The older completed video job has no synchronized FPS measurement. Do not quote its coverage counts as an accuracy or temporal-stability result.
- Existing picture comparisons show Base/v5 versus Teacher/Final. They must not be described as student predictions. Real-photo legacy demos use box placement and rest fingers.
- The source PowerPoint is editable: text, tables, diagrams and connectors are native objects; equations and experimental pictures are raster images.
- Slide 4: the two mesh pictures are crops of thesis Figure F14 (3DPW `courtyard_basketball_01`, image 570): the 3DPW SMPL label mesh and our converted Anny label of the same person and frame, not model predictions. The SMPL skeleton is a schematic of the standard 24-joint tree; the Anny skeleton is drawn from the bone heads of the real 163-bone rest rig (`figures/anny_rest_skeleton.json`, written by `scripts/analysis/render_smpl_vs_anny.py --mode anny_skeleton`). If `figures/smpl_vs_anny_render.png` exists (the same script on the cluster), it replaces the two crops. The Anny vertex count is deliberately not stated: the repository gives no verified number.

## Source map
- Slide 1: Muhammad Saad Najib · Supervisor presentation · 22 October 2026
- Slide 2: Thesis revision, 22 Oct 2026
- Slide 3: Sources: Multi-HMR (Baradel et al., 2024); released NAVER Anny checkpoint; thesis §4.2
- Slide 4: Sources: thesis §2.1, §3.3, App. B; scripts/data/smpl_to_anny.py; model.py; Anny rig: figures/anny_rest_skeleton.json
- Slide 5: Sources: thesis Chapter 3; numbers_data.tex
- Slide 6: Sources: thesis §3.3–3.5; smpl_to_anny.py, converter v2.3
- Slide 7: Sources: thesis §3.4 and Appendix B
- Slide 8: Sources: thesis §3.5, Table T05; F14_label_check_downtown_walking_00771.jpg
- Slide 9: Sources: thesis §4.1, §4.8; model.py; distill.py; train.py
- Slide 10: Sources: thesis §4.2; model.py
- Slide 11: Sources: thesis §4.3; Table T02; exact formulas in backup slides
- Slide 12: Sources: thesis §4.8; distill.py; loss.py
- Slide 13: Sources: thesis Table T11; exp8_distill_vitb.sh; job 3614191
- Slide 14: Sources: thesis §4.10; Table T07
- Slide 15: Sources: thesis Tables T16–T17; jobs 3614191, 3611336, 3614192
- Slide 16: Sources: thesis Table T16; L40S job 3614191; NAVER parameter count, job 3614947
- Slide 17: Sources: thesis Figure F12; AnnyOne holdout image 579860, V1
- Slide 18: Sources: thesis Figures F10; walking and garden examples; demo rendering settings
- Slide 19: Sources: video-compare-3642415.out; thesis video evidence; demo_video.py
- Slide 20: Sources: thesis Discussion and Conclusion; video evidence
- Slide 21: Thesis revision, 22 Oct 2026
- Slide 22: Sources: thesis §4.3; Table T02; loss.py
- Slide 23: Sources: thesis §4.8; distill.py; loss.py rotation_distill_term
- Slide 24: Full bibliography and run provenance: docs/thesis/
