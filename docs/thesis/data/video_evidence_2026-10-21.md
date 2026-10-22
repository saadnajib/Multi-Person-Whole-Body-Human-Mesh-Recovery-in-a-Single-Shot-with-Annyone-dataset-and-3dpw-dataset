# Walking-video evidence

Source: author-uploaded `video-compare-3642415.out` and
`video-compare-3642415.err`, job started 21 October 2026 at 20:01:35 +0200.
This file preserves the evidence used for `numbers_video.tex`.

The output reports:

```text
[renderer-check] GL_VENDOR=NVIDIA Corporation
[renderer-check] GL_RENDERER=NVIDIA L40S/PCIe/SSE2
[renderer-check] PASS: 64x64 test mesh rendered; graphics setup is ready.
[student] 356 frames, 356 with a matched person, 0 missing
[baseline] 356 frames, 354 with a matched person, 2 missing
Completed: /netscratch/najib/multi-hmr/results/video_compare/video_3642415/comparison.mp4
```

The source video was `man_walking.mp4`. The student was the 30-epoch ViT-B
checkpoint (`00029.pt`) with its run-specific `bone_fix.json`. The released
baseline was `models/multiHMR/multiHMR_672_L_anny.pt`. Each loader reported
100.0% state coverage. Both models processed source frames 0--355.

The supplied logs contain no new synchronized inference-FPS comparison or
numerical temporal-stability report. Their matched-frame counts measure tracker
output coverage and do not establish ground-truth detection recall, pose
accuracy, shape consistency or low jitter.
