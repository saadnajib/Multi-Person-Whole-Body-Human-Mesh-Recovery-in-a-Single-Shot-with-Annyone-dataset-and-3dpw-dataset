# Report revision: 22 October 2026

This revision edits the repository's existing LaTeX draft, as confirmed by the
author. It follows the supplied supervisor meeting transcript and uses
`CS-MA-1157_Shalini_MG_thesis.pdf` as a structural example. The original report
is preserved on `main`; the revision is in the `codex/thesis-report` worktree.

## Supervisor guidance and the sample

The supervisor asks for the final implementation presented like a research
paper. Student is the main model; the larger teacher is a training reference.
The report should explain the problem, original Multi-HMR, teacher/student
diagram, training procedure, losses, 3DPW conversion, metrics and results.
Video inference should first be demonstrated without new training.

The sample illustrates the progression from motivation and problem statement
to background, related work, methodology, experiments and conclusion. Its
system overview (printed page 42), loss definition (page 45) and experimental
setup (page 53) are useful structural examples. They do not establish a required
chapter count or length. No sample prose, figures, results or method components
have been copied into this report.

## Changes

| Concern | Revision |
|---|---|
| Student as the final model | Abstract, introduction, method, results, discussion and conclusion distinguish Student, Final/teacher and released Multi-HMR-Anny. |
| Final implementation rather than debugging chronology | Diagnostics move to Appendix F; converter history to Appendix G; early training/student history to Appendix E. |
| Explain the architecture | Method begins with the teacher/student overview diagram and projector equation. |
| Define the losses | Detection, offsets, rotations, shape, placement, geometry, reprojection and distillation now have equations and defined reductions. |
| Explain real-image supervision | Data chapter focuses on final fitting, geometric correspondence, virtual joints, label construction and independent checks. |
| Present results clearly | Student reconstruction/cost and video application come first. The dense comparison table is split into readable reconstruction and cost tables. Teacher ablations are grouped and identified separately. |

## Technical corrections

- `distill.py` uses batch-mean KL: tokens/channels are summed and the image
  batch is averaged. The old equation incorrectly averaged patch tokens.
- Output rotation distillation sums bones and averages people, without the
  supervised per-bone boost or missing-hand mask.
- The student's projector remains at inference because its heads consume the
  projected features.
- The correspondence appendix no longer asserts fixed spine names, a scale
  factor or centimetre offsets that its own comments marked unverified. It
  explains the implemented construction and identifies saved mapping arrays.
- Unverified exact Anny-version/vertex-count claims were removed from the main
  description. Production configuration still needs author verification.
- Two short repeat pilots are not treated as a universal significance
  threshold for all models and datasets.
- Existing reconstruction and historical latency macros retain their values
  and provenance. Later-run results have not been guessed.

## Available video evidence

The supplied completed `video-compare-3642415.out/.err` logs establish NVIDIA
L40S rendering, full checkpoint loading, 356 processed frames per model,
356 student matches, 354 baseline matches and a completed comparison video.
These counts are in `numbers_video.tex` and Appendix A.

The actual comparison video and stability summaries were not supplied to this
editing task. The log also predates the revised FPS implementation. The report
therefore does not claim smooth pose, accurate shape, or new student/baseline
FPS. The earlier paired teacher/student benchmark is a separate protocol.

## Author review before submission

1. Read and adapt the text to your own explanation. The supervisor prefers
   your own draft and understanding; this is editing assistance.
2. Inspect the video and supply the per-model `summary.json` and new
   `timing.csv` before adding temporal-quality or new FPS conclusions.
3. Confirm whether a later student run should be used. This revision uses the
   verified 30-epoch checkpoint, without assuming the 60-epoch job finished.
4. Verify production Anny version, topology, pose convention and fit metadata.
5. Check fields still marked `% TODO verify` in `references.bib` against the
   original papers. Citation resolution does not verify bibliographic facts.
6. Approve the AI-use disclosure and title/date under the applicable university
   rules. The previous unconfirmed assertion of personal verification of every
   number was removed.

## Revised structure

1. Introduction
2. Background and Related Work
3. Datasets and Anny Label Generation
4. Methodology
5. Experiments and Results
6. Discussion
7. Conclusion and Future Work

Appendices A--G preserve reproduction details, joint correspondence, settings,
additional figures, historical pilots, diagnostics and converter validation.
Source file names remain stable: `06_results.tex` is numbered chapter 5 by LaTeX.
