# Thesis editing pass — 8 November 2026

The author requested simpler, more personal academic writing, improvements where
needed, and a length near 105 pages only if shortening helped the report. This
revision starts from the supervisor-oriented source already in the repository,
including the later video implementation on `codex/temporal-video`.

## Writing and structure

- Rewrote the abstract, introduction, discussion and conclusion in direct language.
  First person identifies the author's work and decisions; technical definitions
  remain formal. The style follows the preference for plain explanations visible
  in the conversation, without copying typing errors or claiming to reproduce
  every aspect of the author's voice.
- Simplified the background, conversion, method and results where dense wording
  made the argument harder to follow. Equations and their technical meaning remain.
- Explained the projector, copied heads, warm-up freeze, softened feature
  distributions, separate loss targets, pelvis centring and inference placement.
- Combined the repeated ViT-S history discussion and removed repeated conclusions.
  The seven main chapters and appendices A–G remain. No existing figure or result
  table was removed, and the original font size and page geometry remain.
- Kept the AI-use disclosure. Moved its red editorial review reminder out of the
  printed declaration into these notes; this does not record the author's approval
  of the declaration or verify compliance with university submission rules.

## Technical clarifications

- Distinguished the deployed Student, Final/teacher, Base and released baseline.
- Updated the video method to match the implemented EMA, quaternion SLERP,
  landmark-based translation fitting and log-depth filter. It is output filtering
  around a frame-wise model, not a trained temporal network.
- Kept the original walking-run coverage counts separate from later refined
  videos. Described the observed alignment/size problem qualitatively; added no
  new accuracy, stability or FPS measurements. No cluster inference ran during
  this editing pass.
- Clarified that the real-image hand mask removes missing supervised rotation
  targets while mesh losses and student output distillation still affect hands.
- Corrected the metrics table: centred PVE removes absolute placement, but remains
  sensitive to body size. PA-PVE also removes fitted uniform scale. Added explicit
  PVE and MPJPE definitions and distinguished them from L1 training losses.
- Removed unsupported universal noise thresholds inferred from two short pilots.
  The measured pilot results remain in their table.
- Avoided assuming that the released baseline definitely saw the selected holdout,
  that more training guarantees closing the student gap, or that a shape-mask
  experiment fully proves the cause of age/gender behaviour.
- Preserved every `numbers*.tex` file byte-for-byte. Existing result tables and
  source data retain their measured values. Added a MediaPipe implementation
  citation for the newly documented video method.

## Length and validation

The previous compiled supervisor revision had 102 pages, including appendices.
The final page count and build checks are recorded in `validation_2026-11-08.txt`.
The page reduction comes from editing and removing repetition, not smaller text
or narrower margins. The checked PDF is `thesis_revised_2026-11-08.pdf`.

## Items for the author before submission

1. Read the revised explanations and adjust any wording that does not match how
   you would describe the work. Editing cannot replace your understanding of it.
2. Confirm the title-page date, examiner/supervisor details and disclosure against
   the applicable submission requirements. No declaration was signed by this edit.
3. Check bibliographic facts still marked `% TODO verify` against the papers.
   Resolving a citation in LaTeX does not verify its metadata.
4. Verify the production Anny version, topology and fitting metadata, as requested
   in the preceding supervisor revision.
5. Supply completed timing and stability summaries if numerical claims about the
   latest video refinements are to be added. The report still uses the verified
   30-epoch student and does not assume any later training run succeeded.
