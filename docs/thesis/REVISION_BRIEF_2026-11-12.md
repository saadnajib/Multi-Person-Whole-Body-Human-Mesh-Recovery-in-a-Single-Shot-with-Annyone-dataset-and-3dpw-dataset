# Revision brief, 12 November 2026: supervisor feedback on the draft

This brief governs the revision of the thesis after the supervisor read the draft of 8 November.
Where it conflicts with `STYLE_GUIDE.md` (first-person, conversational voice), **this brief wins**.
Submission deadline: 15 November 2026.

## 1. The feedback, verbatim

**Experiments**

- *Knowledge distillation (Sections 4.5, 5.2).* Missing important ablation: ViT-B without distillation. Same
  training data and schedule. Otherwise, difficult to establish what distillation actually contributes. If no time
  for another experiment, explicitly discuss this limitation.
- *Test vs. validation data (Sections 5.3–5.4).* Temperature sweep reports results on the 3DPW test set.
  Hyperparameter selection should use validation data. Test set for final evaluation. Clarify how the final
  temperature and training duration were selected.
- *Evaluation dataset (Section 3.1).* Only 100 AnnyOne holdout images. Taken from the end of the dataset. No
  children. Too limited for strong generalization claims. Ideally, larger and more representative holdout.
- *Multi-person evaluation (Section 5.5).* Multi-person reconstruction is the main thesis topic. But video
  evaluation uses one person. Still-image demonstration uses only three. More explicit multi-person evaluation would
  strengthen the work. Particularly detection recall, occlusions, crowded scenes.
- *Student evaluation (Section 5.7).* Age/gender evaluation is only for the teacher. But the student is the main
  output of the thesis. Ideally, evaluate the student on Relative Human too. Otherwise, make this distinction very
  clear.
- *Comparison fairness (Section 5.2).* Student and released baseline trained on different data. You already
  acknowledge this. Make the limitation prominent next to the results. Lower PVE does not necessarily mean a better
  architecture.
- *Video performance (Table 5.6).* FPS excludes offline landmark extraction and I/O. Distinguish inference speed
  from end-to-end speed. Avoid implying real-time performance of the complete pipeline.

**Writing and presentation**

- *Academic writing style throughout.* Too conversational in many places. Reads more like a project report than a
  scientific thesis. Avoid phrases such as "Think of SMPL and Anny as two differently built digital mannequins."
  Also "In plain words", "A simple rule for this", etc. Prefer precise technical descriptions.
- *First-person writing.* Not necessarily wrong. But "I", "my", "I did", "I keep", etc. appear too frequently.
  Prefer discussing the method and results directly. Example: "The results indicate..." instead of "I found...".
  First person where it genuinely clarifies your own contribution.
- *Repetition.* The same limitations explained many times. Particularly dataset limitations, single-run variation,
  and baseline fairness. Important to acknowledge them. But no need to repeat the full explanation in every chapter.
  Condense and cross-reference.
- *Abstract.* Problem → method → main results → limitations. Less implementation detail.
- *Structure and figures.* Too many experiment-specific names: Base, Control, Final, Mixed-L, V0, V1, etc. Difficult
  to follow. Prefer descriptive names in the main text. Run identifiers in tables/appendices. Some comparison
  figures too small (e.g., Figure 5.6). Figure captions also unnecessarily long. Remove the red hyperlink boxes in
  the PDF.

**General point.** Try to make the thesis tell one coherent research story around four questions: (1) What was
the problem? (2) What was your solution? (3) What did you actually demonstrate? (4) What remains unresolved? The
debugging history and unsuccessful experiments can support that story, but they shouldn't dominate. Focus on the
important findings. Less repetition / more precise explanations.

## 2. Writing rules

1. **Register.** Scientific thesis prose: precise, impersonal, technical. No analogies ("digital mannequins"), no
   conversational lead-ins ("In plain words", "A simple rule", "Think of", "In other words", "This means that",
   "So", "But" as sentence openers in series), no rhetorical questions, no everyday paraphrase where a technical
   term exists. Explain a term once, technically, at first use. Keep sentences clear and of moderate length; do
   not swing to inflated academic filler either (no "delve", "leverage", "crucial", "pivotal", "comprehensive",
   "notably", "it is worth noting", "plays a key role", "not only ... but also", "furthermore/moreover" chains,
   "state-of-the-art" unless quoting, "novel").
2. **Person.** Default to the method, the model, the results or "this thesis" as the grammatical subject
   ("The student reaches ...", "The results indicate ...", "The converter fits ...", "Training used ...").
   Passive voice is acceptable where the actor is irrelevant. First person singular is allowed only where it
   marks the author's own contribution or decision and nothing else would be as clear; budget: at most about
   two uses per chapter in Chapters 1–7, none in the abstract, none in captions. Never "we".
3. **Tense.** Present for the method and for what tables/figures show; past for what was run and measured.
4. **Descriptive names in the main text** (Chapters 1–7, their tables, figures and captions). Run identifiers
   appear only in Appendix A's name table and in appendix tables.

   | Run identifier (appendix only) | Name in the main text |
   |---|---|
   | Student | the student (lower case) |
   | Student-60 | the 60-epoch student |
   | Final | the teacher |
   | Control | the teacher without corrections (short: uncorrected teacher) |
   | Base | the initial model (trained on AnnyOne only, before real-image training) |
   | Multi-HMR-Anny (NAVER checkpoint) | the released baseline (Multi-HMR-Anny at first mention) |
   | Final-masked | the teacher with full shape mask |
   | Final-partial-mask | the teacher with partial shape mask |
   | Mixed-L, Fresh-heads-L, L-stage 1–4, Student-S/* | not named in the main text; "earlier models (Appendix E)" |
   | V1 | not named; "helper bones (face, toes, breasts) set to rest pose" is stated once in the evaluation protocol |
   | V0 | not named; stated once as the protocol used on 3DPW |

   Table column heads use the descriptive names (Teacher, Student, Released baseline, Initial model, Uncorrected
   teacher). No job ids, script names or commit hashes in Chapters 1–7.
5. **One home per limitation; elsewhere one clause and a cross-reference.**
   - Evaluation-data limits (100-image AnnyOne holdout taken from the end of the dataset, no children; converted
     3DPW labels with constant phenotypes and rest-pose hands/head; Anny-space numbers not comparable with
     published 3DPW results; Relative Human has no 3D ground truth): **Section 3, label `sec:data-limits`**.
   - Single-run variation (one run per model, no fixed seed; two identical pilots differed by 7.6 mm PVE; the two
     T = 4 students by 2.5 mm): **Section 4 evaluation protocol, label `sec:method-variation`**.
   - Baseline fairness (the released baseline was not trained on converted 3DPW and may have seen the AnnyOne
     holdout; lower PVE therefore does not show a better architecture): **Section 5.2, label
     `sec:res-fairness`**, placed directly next to the comparison table.
   - Chapter 6 lists the limitations once, compactly, by reference; it does not re-explain them.
   Every other mention is a short clause with `\ref` (for example "(single runs; Section~\ref{sec:method-variation})").
6. **Captions.** At most three sentences: what is shown, on which data and protocol, and (optionally) the one
   thing to read from it. Explanations, caveats and reading instructions go into the text. No first person.
7. **Figures.** Comparison figures at full text width where panels are otherwise hard to read (in particular
   `fig:holdout-montage`, the former Figure 5.6). Prefer fewer, larger panels to many small ones.
8. **Numbers.** Only through macros in `numbers*.tex` (new ones for this revision: `numbers_rev.tex`); never typed,
   never invented. Results that are still running get a visible `\TODO{pending: ...}` and nothing else.
9. **Labels.** Never delete or rename an existing `\label`; moved content keeps its label; if a section
   disappears its label moves to the nearest remaining heading. Spelling: British, -ise.
10. **AI-use statement** in `thesis.tex` is not edited.

## 3. The story (four questions) and the chapter roles

1. *Problem* (Ch. 1): multi-person whole-body mesh recovery from one image in the Anny body model, at reduced
   inference cost, with real-image supervision that does not exist for Anny.
2. *Solution* (Ch. 3–4): (a) conversion of 3DPW's SMPL labels to Anny for real-image training and testing;
   (b) a ViT-L teacher trained on AnnyOne and converted 3DPW with two corrections; (c) a ViT-B student distilled
   from it; (d) inference-time temporal filtering and landmark alignment for video.
3. *Demonstrated* (Ch. 5): the student's accuracy–cost trade-off against its teacher and the released baseline;
   what distillation contributes (ablation); insensitivity to training length and temperature (validation-based);
   multi-person detection recall; behaviour on photos and video; the teacher analysis in condensed form.
4. *Unresolved* (Ch. 6–7): the accuracy gap to the teacher, shape on real photos (children, men), depth
   compression, the small and unrepresentative holdout, single runs, no fairly trained baseline, no learned
   temporal model.

The debugging history (how each failure of the initial model was isolated) stays in the appendices and is only
referenced. Chapter 5's teacher analysis is condensed to its findings.

## 4. Chapter 5, new section order and labels

| § | Title (sentence case) | Label | Content |
|---|---|---|---|
| 5.1 | Experimental setup | `sec:res-setup` | sets and protocol by reference to Ch. 3–4; **model-selection protocol** (`sec:res-selection`): temperature T = 4 and 30 epochs were fixed before training (T taken over from the earlier ViT-S distillation runs, 30 epochs set by the compute budget), the last checkpoint is always used, no checkpoint or hyperparameter was selected on any test set; 3DPW validation (300 images) is the in-training validation set; the AnnyOne holdout is both the in-training monitor and the reported synthetic test set, which is stated as a limitation |
| 5.2 | Student, teacher and released baseline | `sec:res-baseline`, `sec:res-student`, `sec:res-fairness` | accuracy and cost; fairness paragraph directly at the table |
| 5.3 | Contribution of distillation | `sec:res-kd-ablation` | ablation: ViT-B without distillation, same data and schedule (two arms; `tab:kd-ablation`, results pending) |
| 5.4 | Sensitivity to training length and temperature | `sec:res-training-length`, `sec:res-kd-temperature` | both judged on **validation** data (`tab:sweep-validation`, pending numbers for the sweep); test-set values shown only as the final evaluation of each run; conclusion: neither changes the result, the pre-set values were kept |
| 5.5 | Multi-person detection | `sec:res-multiperson` | people per image in each set; recall by number of labelled people per image on Relative Human (`tab:rh-crowd`; teacher available, student pending); what was not evaluated (occlusion-specific benchmarks) |
| 5.6 | Qualitative results and video | `sec:res-video`, `sec:res-image-compare` | still image; video; **inference speed vs. end-to-end speed** separated; no real-time claim |
| 5.7 | Analysis of the teacher | `sec:res-teacher-ablations`, `sec:res-final`, `sec:res-3dpw`, `sec:res-depth`, `sec:res-neck`, `sec:res-fingers`, `sec:res-rh`, `sec:res-mask` | condensed to findings; shape results are stated as **teacher** results, with the student's Relative Human evaluation pending or, failing that, the distinction stated in the first sentence |

## 5. Facts for the experiment-related feedback

- **Selection protocol (true as run).** T = 4 and the loss weights were taken over from the earlier ViT-S
  distillation recipe; 30 epochs was fixed in advance from the compute budget; every evaluation uses the last
  checkpoint; no early stopping. The 60-epoch run and the temperature sweep were run afterwards as sensitivity
  analyses and changed no setting. In-training validation: 3DPW validation, 300 images (`\SelValImages`),
  PVE at the last epoch 87.8 mm (30 epochs, `\SelThirtyValPVE`) and 86.0 mm (60 epochs, `\SelSixtyValPVE`).
  Per-temperature validation numbers of the sweep: pending from the cluster.
- **Multi-person.** AnnyOne holdout: 222 people in 100 images. 3DPW test: one or two labelled people per image.
  Relative Human validation: 2,130 labelled people in 635 images, with recall by people per image in
  `data/rh_recall_by_crowd.csv` and `numbers_rev.tex` (`\CrowdFinal...Rec`, `\CrowdBase...Rec`,
  `\Crowd...Images`, `\Crowd...People`): for the teacher 76.2 % (1 person), 78.5 % (2), 67.8 % (3–4), 56.6 %
  (5–8), 45.3 % (9 or more). The student's Relative Human run is pending.
- **Video speed.** All timings on one NVIDIA A40, batch size 1. Model inference only: student 18.3 FPS, released
  baseline 8.5 FPS. Including image preprocessing and output transfer: 64.9 ms (15.4 FPS) and 127.9 ms (7.8 FPS).
  Student with filter and alignment, including preprocessing: 87.8 ms (11.4 FPS); with rendering 9.3 FPS. Not
  timed: offline landmark extraction (MediaPipe on CPU, a separate pass over the video) and video decoding and
  encoding. Macros `\Video...Pipe...` in `numbers_rev.tex`. No end-to-end or real-time claim can be made.
- **Distillation ablation.** Two arms, same data, schedule and architecture as the student: (a) distillation
  losses off, teacher heads still copied; (b) no teacher at all (no losses, no head copy). Runs submitted
  12 November; results pending. If they do not finish, Section 5.3 states the limitation explicitly.
