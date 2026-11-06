# Style guide for the thesis text

Written 6 November 2026. Every chapter rewrite follows this guide. It describes how the author writes and explains
things, so that the report reads as their own work in their own voice.

Why this guide exists. In the meeting on 5 November the supervisor said that the presentation "was much better than
before, and if you write in a similar style in your report, it should be fine", and that the writing must "not look
obviously AI-generated". The author asked for the report to use their own writing style. The goal is clear, natural
writing that a reader can follow. The goal is **not** to hide that tools were used (see section 9).

Where the voice comes from, in order of weight:

1. The author's own messages (vocabulary, sentence shape, how they ask for and give explanations).
2. The speaker notes of the final deck of 5 November (`final_text.md` in the session scratchpad; the parts in the
   author's own phrasing, e.g. the problem statement, the camera sentence, the mannequin analogy, the conclusion).
3. The supervisor's spoken explanations in the 5 November meeting (plain, step by step, "on the y-axis you have the
   error, and on the x-axis you have the temperature").

The author's draft in `draft_2026-10_original/` was already heavily polished, so it is **not** a voice sample.
Do not copy its tone.

---

## 0. Short version

This is the compact form of the guide. Sections 1 to 10 give the details and the examples.

**Reader.** A computer-vision master's student who has not heard of SMPL, Anny, AnnyOne, 3DPW or Multi-HMR. Each term
is explained in plain words the first time it appears, then used the same way every time after that.

**Person and tense.** "I" for my own decisions and work ("I fit an Anny body to each 3DPW label"). Present tense for
how the method works ("the projector maps 768 channels to 1,024"). Past tense for what was run and measured ("the
student reached 110.1 mm"). Do not use "we" for my own work. Use the passive only when the actor really does not
matter.

**Register.** Plain, direct and professional. Write like explaining the work to a classmate at a whiteboard, with
correct grammar. Prefer everyday verbs: make, turn, find, put, keep, use, check, match, line up, follow, fix, reduce.
Prefer "use" to "utilise", "lets" or "makes it possible" to "enables", "shows" to "demonstrates/establishes", and
"so" to "therefore" (at most one "therefore" per paragraph).

**Sentences.** Short to medium (most of them 8 to 25 words), with natural variation: a 5-word sentence now and then,
and an occasional longer one for a sequence of steps. One idea per sentence and one topic per paragraph. Paragraph
length varies (2 to 6 sentences). Link ideas with simple words: because, so, this means, for this reason, in other
words, for example, then, first ... then ... finally, but, also.

**How this author explains.** (1) Say the problem first in one sentence. (2) Say what I did about it. (3) Say what
happened, with the number. (4) Say what it means, including what did not work. Concrete before abstract: give the
example or the number first, then the general statement. Use "for example" with a real number. After every equation,
add one sentence that says in plain words what it penalises or measures. When a claim needs evidence, say how I
checked it. Always say what a number is a result of: which model, which dataset, which labels, which setting. Use an
everyday analogy at most once per section (the author's own: SMPL and Anny are "two differently built digital
mannequins").

**Honesty.** State limitations plainly, in the same paragraph as the result they limit. "The student did not reach
the teacher's accuracy" is better than "an efficiency--accuracy trade-off is observed". No hedging boilerplate
("it can be argued", "to some extent", "it may potentially"), and no overclaiming ("solved", "perfect", "best model",
"state of the art").

**Keep.** Precise technical terms after they have been explained (PVE, PA-PVE, MPJPE, KL divergence, linear blend
skinning, intrinsics). Numbers with units and a thin space (`\StudentPVE\,mm`). LaTeX number macros from
`numbers*.tex`, never typed numbers. Model names as defined once in the method chapter (Student, Final as the
teacher, Base, Control, Multi-HMR-Anny). Every caption states the set, the setting and the models.

**Avoid (the AI tells).** The banned words in section 4. Rule-of-three lists everywhere. Every paragraph the same
length. Bold or colon-headed lead-ins inside prose ("Key point:"). Em-dashes as the main punctuation. Signposting such
as "In this section we will". Symmetric "While X, Y" sentences. Ending each paragraph with a moral ("This highlights
the importance of ..."). Bullet lists in running text whose items are perfectly parallel. Title-case headings with
colons; use sentence case ("Converting 3DPW labels to Anny"). Nominal style where an abstraction is the subject
("The reduction has an accuracy cost"); make a model, a person or "I" the subject.

**Disclosure.** The "Use of AI tools" section in `thesis.tex` stays truthful. It is never weakened, shortened into
vagueness or removed, and its final wording is the author's decision.

**Per paragraph, ask:** Does it say one thing? Could a reader without SMPL/Anny knowledge follow it? Is every number a
macro with a named model, set and setting? Is any banned word or structural tell present? Does a limitation sit next
to its result? Would the author say this sentence out loud in a presentation?

---

## 1. Reader and register

The reader is a computer-vision master's student or an examiner outside this exact topic. They know what a CNN, a
transformer and a loss function are. They do not know SMPL, Anny, AnnyOne, 3DPW, Multi-HMR or what PVE measures.

- Explain each term once, at first use, in one plain sentence, then use it consistently. The author's own pattern:
  "Camera information tells the model which direction each image location points into 3D space." "A loss is a
  numerical penalty for a mismatch during training." "The teacher remains frozen, meaning training does not update
  its weights."
- The supervisor asked that SMPL and Anny be introduced before anything that depends on them: SMPL has 10 shape
  parameters (as used by most methods), Anny has 11 named phenotype parameters (supervisor transcript of 5 November
  and deck slide 9; macros `\BgSmplBetas` and `\MethodNumPheno`). The reader must know early **why** 3DPW is converted: the model predicts Anny,
  the only data with Anny labels is synthetic, and 3DPW's real photos come with SMPL labels.
- The tone is professional, not chatty. No exclamation marks, no rhetorical questions, no "Let us". Contractions
  (don't, isn't) are not used in the thesis.
- Spelling: British English (colour, behaviour, centred, modelling) with -ise endings (initialise, normalise,
  optimise, visualise). Do not change code identifiers, command-line flags, labels, citation keys or quoted titles.

## 2. Person, tense and sentence shape

| what | form | example |
|---|---|---|
| my decisions and work | "I", active | "I kept the copied heads frozen for the first five epochs." |
| how the method works | present tense | "The projector maps the student's 768 channels to the 1,024 the heads expect." |
| what was run and measured | past tense | "On the 3DPW test set the student reached `\StudentThreeDPW`\,mm PVE." |
| what a table or figure shows | present tense | "Table 6.2 compares the student with its teacher." |
| an inference I did not test | "The likely reason is ..., which I did not test." | |

Sentence shape:

- Most sentences 8 to 25 words. Mix in short ones ("This did not work.") and allow an occasional longer one for a
  sequence. Do not make every sentence the same length.
- One idea per sentence. If a sentence needs a semicolon and a "rather than", split it.
- The subject is a concrete thing: a model, a dataset, a loss, a person in the image, or "I". Not "the evaluation",
  "the result", "the reduction", "this approach".
- Linking words: because, so, this means, for this reason, in other words, for example, then, first, finally, but,
  also, however (sparingly). Not "furthermore", "moreover", "additionally", "thereby", "wherein".
- Paragraphs: one topic, 2 to 6 sentences, varied in length. A one-sentence paragraph is fine once in a while.

## 3. How this author explains

These patterns come from the author's messages and slide notes. Use them.

1. **Problem first, in one sentence.** "Our problem is to turn a single photo of several people into accurate 3D body
   models while using less computation." (slide 2 notes)
2. **Then what I did, what happened, what it means.** "We evaluated a completed 30-epoch student run. It achieved
   substantial efficiency gains but did not retain the required teacher accuracy. Longer training may improve the
   synthetic results, while the real-image plateau suggests that training duration is not the only limitation."
   (slide 6 notes; in the thesis, "I" and plainer verbs.)
3. **Concrete before abstract.** Give the example or the number, then the general point.
4. **"For example" with a real number.** "For example, finding 9 out of 10 labelled people gives 90% recall." (slide 10
   notes) "A strongly concentrated distribution like [0.88, 0.12] becomes softer, around [0.62, 0.38]." (slide 4
   notes; an illustration of temperature, not a result.)
5. **One plain sentence after each equation.** The slide notes do this with "Meaning: penalize missed or false person
   detections and incorrect anchor locations." In the thesis: "In words, the loss is large when the network misses a
   person or fires on background."
6. **Steps in order.** "First ... then ... finally" for procedures, as in the video notes: the student predicts a body
   per frame, then an exponential moving average blends it with earlier frames, then MediaPipe landmarks correct
   position and depth, finally the corrected depth is smoothed.
7. **Evidence for each claim.** The author's own question was "how can we be sure or whats the evidence that the
   conversion is correct", and their answer was to render the converted mesh over the image. After a claim, say how it
   was checked: an overlay, an independent joint check (median `\DataCheckJointsNew`\,mm, slide 17 and
   `numbers_data.tex`), a control run.
8. **Name what a number is a result of.** The author asked "published 3DPW results, resuts of what?". Never write a
   number without its model, dataset, label version and setting.
9. **Admit what did not work, plainly.** "The raw student mesh sometimes extends beyond the person." (slide 18 notes)
   Say it in the same paragraph as the result, not in a separate apology section.
10. **One everyday analogy at most per section.** "Think of them as two differently built digital mannequins." "Move,
    turn and bend an Anny mannequin until its joints match the original joints." (slides 2 and 15 notes)
11. **Explain a plot by its axes.** As the supervisor did: "on the y-axis you have the error, and on the x-axis you
    have the temperature parameter. From there you can see which temperature value minimises the errors."

## 4. Banned words and phrases

Search every chapter for these before handing it back. The right-hand column gives a plain replacement or says what to
do instead.

| avoid | use instead |
|---|---|
| delve, dive into, deep dive | look at, study, describe |
| leverage, harness, utilise/utilize | use |
| crucial, crucially, pivotal, vital, paramount, key (as an adjective of praise) | important, needed, or say why it matters |
| robust, robustness (non-technical sense) | reliable, stable, or the measured fact |
| seamless(ly) | delete, or say what actually happens |
| comprehensive, holistic, thorough (as praise) | complete, or list what is covered |
| notably, importantly, interestingly, significantly (without a test) | delete; let the number speak |
| furthermore, moreover, additionally (sentence-initial chains) | also, and, a new sentence |
| it is worth noting, it should be noted, it is important to note | delete and state the fact |
| plays a key/crucial/pivotal role | name what it does |
| in the realm of, in the landscape of, in the context of (as filler) | in, for |
| landscape, tapestry, journey, ecosystem | delete |
| underscore(s), highlight(s) (as a verb about a result), This highlights | shows, or delete the sentence |
| showcase(s) | show |
| intricate, meticulous(ly), nuanced, multifaceted | detailed, careful, or the concrete property |
| a testament to | delete |
| not only ... but also | and, also |
| Overall, / In summary, / In conclusion, (at a paragraph start, outside the conclusion chapter) | delete |
| serves as, stands as, acts as (when "is" works) | is, works as, is used for |
| navigate | handle, deal with |
| novel, cutting-edge, state-of-the-art (unless citing a paper's claim), game-changing | new, recent, or delete |
| paving the way, unlock, empower, foster, elevate, streamline | say the concrete effect |
| a wide range of, a plethora of, a myriad of | many, several, or the count |
| bridges (the gap) | connects, converts, or say how |
| sheds light on | shows, explains |
| demonstrates, establishes (about my own results) | shows; "this only shows that ..." |
| enables, facilitates | lets, makes it possible to |
| thereby, wherein, whereby, in turn | so, where, then |
| essentially, ultimately, effectively (as filler) | delete |
| "rather than" more than once per paragraph | split into two sentences |
| solved, fixed completely, perfect, flawless, best model | the measured change |
| "measured rather than guessed"-type self-praise | delete |
| In this section we will / This chapter will / We now turn to | start with the content |

A reviewer can run this from `docs/thesis`:

```
grep -n -i -E "delve|leverag|harness|utili[sz]|crucial|pivotal|paramount|robust|seamless|comprehensive|holistic|notably|importantly|interestingly|furthermore|moreover|additionally|worth noting|should be noted|important to note|key role|realm|landscape|tapestry|underscor|highlight|showcas|intricate|meticulous|nuanced|multifaceted|testament|not only|^ *overall,|in summary|serves as|stands as|navigat|novel|cutting-edge|state-of-the-art|paving|unlock|plethora|myriad|bridg|sheds light|demonstrat|establish|facilitat|thereby|wherein|whereby|in turn|essentially|ultimately|perfect|in this section we" chapters/*.tex
```

"robust" is allowed in a technical sense with a definition (for example a robust loss); "state-of-the-art" is allowed
when quoting a cited paper's claim; "establish" is allowed in "establish a baseline" only if nothing plainer fits.

## 5. Structural tells to avoid

- **Rule of three everywhere.** Lists of exactly three adjectives or three clauses in sentence after sentence ("pose,
  shape and placement"; "efficiency, accuracy and stability"). Use the real number of items. Two is fine; four is fine.
- **Uniform paragraphs.** Every paragraph four sentences of similar length. Vary them.
- **Colon-headed bold lead-ins inside prose** ("**Key idea:** ..."). A LaTeX `\paragraph{...}` heading is fine when
  it is a real run-in heading of a sub-topic, but not before every paragraph of a section.
- **Em-dashes as the main punctuation.** No em-dash asides in the running text. Use a full stop, a comma or brackets.
  En-dashes in ranges and pairs (`teacher--student`, `0.5--0.6`) are fine.
- **Over-signposting.** "This chapter describes ...", "In this section we will ...", "The following paragraphs
  discuss ...". One short orientation sentence at the start of a chapter is allowed when it tells the reader **why**
  the chapter comes here ("The model is trained on two datasets, so I describe them before the method.").
- **Symmetric "While X, Y" sentences** and "X; however, Y" in every paragraph. Write two sentences.
- **Moral endings.** A last sentence that repeats the point as a lesson ("This underlines the value of ...",
  "Reporting these quantities together provides a clearer assessment"). End on the fact or on what follows from it.
- **Hedging boilerplate.** "may potentially", "it can be argued that", "to some extent", "it is possible that" piled
  up. One hedge, where the uncertainty is real, with the reason ("one run, no seed").
- **Perfectly parallel bullet lists in running text.** Explanations go in sentences. Lists are for real enumerations
  (research questions, contributions, steps of a procedure).
- **Title-case headings with colons** ("Label Conversion: From SMPL to Anny"). Use short sentence-case headings
  ("Converting 3DPW labels to Anny"). Apply one case convention to every heading.
- **Nominal style.** "The reduction has an accuracy cost." "The evaluation concerns three outcomes." Rewrite with a
  concrete subject and a verb: "The student is smaller and faster, but it is also less accurate."
- **Contrast framing everywhere.** "This is not X but Y", "rather than X" used as a reflex. Say what it is.
- **Abstract openers.** Paragraphs that start with a general claim and only reach the fact in the third sentence.
  Start with the fact.

## 6. What to keep

- Precise technical terms, once explained: PVE, PA-PVE, MPJPE, recall, KL divergence, temperature, linear blend
  skinning, camera intrinsics, Procrustes alignment, focal loss, exponential moving average.
- Numbers with units and `\,` (thin space): `\StudentPVE\,mm`, `\StudentGFLOPs\,GFLOPs`, `\StudentMsLFortyS\,ms`.
- Honest limitations stated plainly next to the result: one run per model without a seed; converted 3DPW labels are
  not comparable with published SMPL-space numbers; the AnnyOne holdout is most likely in-sample for Multi-HMR-Anny;
  the student did not reach the teacher's accuracy.
- Fixed wordings already agreed in `REWRITE_PLAN.md` section 5 (the NAVER AnnyOne sentence, "lowest error on our
  3DPW-Anny labels", "a measured improvement, not solved" for children, no conversion of Anny ages to years).
- The model roles: Student is the deployed model; Final is its teacher; Multi-HMR-Anny (NAVER) is the external
  baseline.

## 7. Numbers, macros and sources

- Never type a number into the running text or a table when a macro exists. Every new number gets a macro in the
  right `numbers_*.tex` file with its source in a LaTeX comment (log line, CSV, JSON, or `data/` file).
- Never invent a number, a rounding or a ratio. If a number is needed and no source exists, write `\TODO{...}` and
  say what must be run.
- Write the number with what it is a result of: "the student's PVE on the 3DPW test set (converted labels, 8,190
  images)" and not just "the 3DPW result".
- Ratios: "3.14 times fewer parameters" is fine; also give the two absolute numbers once.
- Job ids, checkpoint names like `v5`, `exp7` or Slurm paths never appear in the running text. They live in
  Appendix A.

## 8. Figures, tables and equations

- Each results section opens with the figure or table, then one paragraph that reads it, then one that says what it
  means, then one sentence on what it does not show.
- To describe a plot, name the axes and say what to look for: "The x-axis is the epoch and the y-axis the PVE on the
  holdout. The holdout curve is still falling at epoch 30, while the 3DPW validation curve is flat over the last five
  epochs."
- Every caption names the images, the models, the setting and the caveat (synthetic, by eye, a good case).
- After each equation: define every symbol in one sentence, then one sentence in plain words.

## 9. Use of AI tools: the disclosure stays truthful

- The "Use of AI tools" section in `thesis.tex` stays truthful and complete. It is not weakened, made vaguer,
  shortened to hide anything, or removed. Its final wording is approved by the author against the university rules;
  the existing `\TODO` there stays until the author has done that.
- This guide is about clear, natural writing. It is not a method to disguise assistance. A rewrite that makes the text
  sound like the author must keep every fact, number, caveat and citation of the original.
- No names of AI tools or models appear anywhere else in the repository files.

## 10. Before and after: twelve real sentences

Each "before" is quoted from the current chapters (6 November 2026). Each "after" keeps every fact and number; only
the wording changes. The macros keep their sources in `numbers*.tex`.

**1. Abstract, opening** (`00_abstract.tex`)

Before:
> Multi-person human mesh recovery estimates articulated body surfaces and their positions from one image. This thesis
> studies a compact Anny-based reconstruction system using a ViT-B student distilled from a larger ViT-L teacher.

After:
> The problem in this thesis is to turn one photo of several people into a 3D body model for each person, at a lower
> computational cost. For this, I trained a small ViT-B network, the student, that learns from a larger ViT-L network,
> the teacher. Both predict bodies in the Anny body model.

Why: problem first in one sentence; "I trained" instead of "this thesis studies"; concrete subject.

**2. Abstract, video** (`00_abstract.tex`)

Before:
> These counts establish execution and tracking coverage on one clip. Pose jitter and shape consistency still require
> inspection of the video and measurements.

After:
> These counts only show that both models run on this clip and how often they keep track of the person. They do not
> say whether the pose shakes or the body shape changes from frame to frame; for that I have to look at the video and
> its measurements.

Why: says plainly what the number is and is not; "establish execution" is nominal style.

**3. Introduction, motivation** (`01_introduction.tex`)

Before:
> These capabilities support applications such as motion visualization and human-centred scene understanding.

After:
> This is useful, for example, to show how people move or to understand a scene through the people in it.

Why: filler sentence with an abstract subject ("these capabilities"); shorter and concrete.

**4. Introduction, baseline** (`01_introduction.tex`)

Before:
> That released model is a baseline for this thesis; the existence of an Anny-based multi-person estimator is not
> itself a contribution.

After:
> NAVER has already released such a model for Anny, so predicting Anny bodies for several people is not new in my work.
> I use this released model as the baseline and compare against it.

Why: semicolon plus "the existence of" becomes two plain sentences with "I".

**5. Introduction, shape** (`01_introduction.tex`)

Before:
> Expressing a shape and inferring it reliably from a photograph are separate problems.

After:
> But a body model that can represent a child does not mean that a network can recognise a child in a photo. These are
> two different problems.

Why: concrete example (a child) before the abstract point.

**6. Introduction, cost** (`01_introduction.tex`)

Before:
> Computational cost is a second concern. Large image encoders provide strong features but increase parameter count
> and inference time. A compact student trained under a larger teacher offers a way to retain the single-shot pipeline
> while reducing cost.

After:
> The second problem is cost. A large image encoder gives good features, but it has many parameters and takes longer to
> run. For this reason I train a smaller student network with help from a larger teacher. The pipeline stays
> single-shot, but it becomes cheaper to run.

Why: "offers a way to retain ... while reducing" is a symmetric abstract sentence; the author says "for this reason I".

**7. Method, opening** (`04_method.tex`)

Before:
> The deployed system is the ViT-B Student. A larger ViT-L teacher, denoted Final in the experiment records, supplies
> transferred prediction heads and distillation targets.

After:
> The model I deploy is the ViT-B Student. It learns from a larger ViT-L teacher, which my experiment records call
> Final. The teacher gives the student two things: a copy of its prediction heads at the start of training, and
> targets to imitate during training.

Why: explains what "transferred heads" and "distillation targets" mean in plain words.

**8. Background, projector** (`02_background.tex`)

Before:
> The projector serves two roles: it aligns feature dimensions for distillation and preserves the representation width
> expected by the copied prediction heads.

After:
> The projector has two jobs. The student's features have \MethodWidthB{} channels, but the teacher's features and the
> copied heads use \MethodWidthL. The projector converts one width into the other, so the student's features can be
> compared with the teacher's and can be fed into the copied heads.

Why: "serves two roles" is a tell; the real numbers make the point concrete.

**9. Results, external comparison** (`06_results.tex`)

Before:
> The external 3DPW comparison includes a training-data difference. Both student and teacher use converted 3DPW
> training images; the released baseline is evaluated without that adaptation. The result demonstrates the behaviour
> of the complete trained systems, rather than isolating network architecture.

After:
> This comparison is not fully fair to the baseline. My student and teacher were trained on converted 3DPW images, and
> the released baseline was not. So the result compares the two complete trained systems. It does not show that my
> network design is better.

Why: the limitation is said plainly first; "demonstrates ... rather than isolating" is replaced by what it means.

**10. Discussion, real-image supervision** (`07_discussion.tex`)

Before:
> The large synthetic-to-real gain is therefore associated primarily with mixed training. It cannot be attributed
> entirely to the corrections or a network architecture change.

After:
> Control already reaches \ControlThreeDPW\,mm without the two corrections, against \BaseThreeDPW\,mm for Base. This
> means that most of the gain on real images comes from adding 3DPW to the training mix, and not from the corrections
> or from a change in the network.

Why: the number comes before the conclusion ("concrete before abstract"); "is associated primarily with" becomes
"most of the gain comes from". Both numbers are from the paragraph just before it in the same chapter.

**11. Discussion, compact reconstruction** (`07_discussion.tex`)

Before:
> The result demonstrates an efficiency--accuracy trade-off rather than teacher-equivalent reconstruction.

After:
> So the student is smaller and faster, but it is also less accurate. It did not reach the teacher's accuracy.

Why: admits plainly what did not work; no nominal "trade-off" phrase.

**12. Discussion, run-to-run variation** (`07_discussion.tex`)

Before:
> Two identical short pilot configurations differ by \NoisePVE\,mm PVE. This demonstrates sensitivity to training
> variation, but two pilot runs do not establish a universal significance threshold for other schedules or datasets.
> Small differences between single runs are interpreted cautiously.

After:
> I trained the same short pilot twice with identical settings, and the two runs differed by \NoisePVE\,mm PVE. In
> other words, a difference of a few millimetres between two single runs can be training noise alone. Two runs are not
> enough to give a general threshold, so I treat small differences between single runs with care.

Why: "I" did it; "in other words" explains what the number means; no "demonstrates/establish".

Also typical, not counted above: the moral ending of `07_discussion.tex` ("Reporting these quantities together
provides a clearer assessment of the compact model than a single reconstruction error or visually pleasing example.")
becomes "For this reason I report all of them, and not only one error value or one good-looking example." The chapter
opener of `03_data.tex` ("This chapter describes the images and labels behind every number in the thesis.") becomes
"Every number in this thesis comes from one of the datasets below, so I describe them before the method."

## 11. Checklist per paragraph

Run this on every paragraph of a rewritten chapter.

1. Does the paragraph say one thing, and does its first sentence say what?
2. Is the problem or the fact stated before the general claim?
3. Is every new term explained in plain words at its first use, for a reader who does not know SMPL or Anny?
4. Is "I" used for my decisions, present tense for the method and past tense for the runs?
5. Is every number a macro (or a sourced table cell) with its model, dataset, label version and setting named?
6. Is there an example with a real number where the idea is abstract?
7. If there is an equation, is there a plain-words sentence after it?
8. Is a limitation or a failure stated plainly, next to the result it limits?
9. Does the grep in section 4 find nothing in this paragraph?
10. Is there none of these: rule of three, bold lead-in, em-dash aside, "While X, Y", signposting, moral last
    sentence, stacked hedges?
11. Do sentence and paragraph lengths vary compared with the paragraphs around it?
12. Would the author say these sentences out loud when presenting the slide on this topic?
13. Are all facts, numbers, caveats and citations of the previous version still there?
