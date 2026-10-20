# The supervisor deck, explained in plain words (meeting of Monday 19 October 2026)

20 Oct: the 16 presented slides were simplified on the slide face; the speaker notes are unchanged. The other 16
(slides 4-5, 7-9, 11, 13, 16, 19, 22 and 27-32) are hidden backup slides, skipped in the slide show.

This document explains the deck for the meeting of Monday 19 October 2026, 16:00, slide by slide, in simple words.
It is written for the presenter, who must also defend the work and later reuse the text in the thesis. Part 6 follows
the final 31-slide deck (`docs/supervisor_update_2026-10-19.pptx`, speaker script
`docs/supervisor_talk_notes_2026-10-19.md`): each block starts with the slide number and the exact slide title, so the
two can be matched. This file was first written against the planned outline and reconciled with the final deck on the
evening of 18 October.

**State of the facts:** Sunday 18 October 2026, evening; updated Monday 19 October, morning, with the ViT-B student
result (job 3614191: slides 1, 2, 23-26 and 30, Part 7 1b, Part 8 Q3a); updated again on 19 October with slide 31
(predicted fingers and rest-pose line-up, demo jobs 3615018 and 3615019), the old ViT-S students on the corrected labels
(job 3615020) and the full thesis draft (92 pages): slides 2, 16, 17, 23, 25-28 and 31, Part 7 1a and 3. Nothing here was measured again for this
document. Every number comes from a file in this repository, and the source is given next to it.

## How to read this document

**Confidence tags** (as in the rest of the project):
- **[Certain]** = measured: a job log line, or read in the code.
- **[Likely]** = a strong inference from what was measured, but not tested on its own.
- **[Guessing]** = an idea or an estimate with no direct evidence.

Definitions (what a word means) get no tag. Facts and results get one.

**Source key** (how a source is written in this document):
- `H 3.1` = `docs/handover.md`, section 3.1. `H 0a` = its section 0a ("Tomorrow ... and the meeting"), which also
  holds the newest results of 18 October. `H 0b` = "Numbers at a glance".
- `P 2` = `docs/thesis_requirements_plan.md`, section 2.
- `E exp8` = `docs/experiments_plan.md`, section on exp8. `F` = `docs/diagnostics_findings.md` (older).
- `D slide N` / `D notes N` = the final 31-slide deck: the text on slide N of `docs/supervisor_update_2026-10-19.pptx`
  (built by `docs/deck_src/build.js`) / the speaker notes of slide N in `docs/supervisor_talk_notes_2026-10-19.md`.
  All slide numbers in this file are the new ones; `docs/deck_src/README.md` has the old -> new map.
- `3611336:191` = line 191 of the log of cluster job 3611336. The logs are on the cluster, not in the repo; these line
  numbers are quoted from the handover, which was checked against the logs.
- `loss.py:343` = that line of the file in this repository today.
- `C SD` = `docs/thesis/data/CLUSTER_ANSWERS.md`, last section "STUDENT DONE" (19 Oct morning): every student number
  with its line in `logs/exp8_distill_vitb_3614191.out` (written `3614191:195-207` below).

**Words used for the models** (more in Part 2):
- **v5** = the starting model (`anny_s2_shape_v5`, epoch 99), trained on synthetic images only.
- **v9** = v5 trained further with 30% real 3DPW images, on the old (defective) 3DPW labels. It was the previous best on 3DPW.
- **final model** = exp7, job 3611336: v5 plus all fixes, trained 10 more epochs. It is the thesis model and the teacher.
- **control** = exp7, job 3611339: the same training as the final model, but without the depth fix and the bone fix.
- **masked arm** = exp7, training job 3612325 (Relative Human eval 3613616, demo 3613617): the final model's recipe with the 3DPW shape loss masked (Part 5.4, slide 19).
- **student** = exp8, a smaller ViT-B model learned from the final model (slide 24).

---

## Part 1. The story in ten sentences

1. The thesis moves Multi-HMR, a network that finds every person in one image and builds a 3D body mesh for each,
   from the SMPL-X body model to the new Anny body model, and trains it on the synthetic AnnyOne images and on real
   3DPW images converted to Anny by our own fitting script [Certain, H 1].
2. Last week the model showed three visible failures: thin, twisted necks, claw-like fingers, and children who got
   adult bodies [Certain, D slide 10].
3. We first checked what is *not* the cause: the labels decode correctly, the body model reacts correctly, and every
   bone and shape output receives a training signal, so it is not a rig or decoding problem [Certain, H 1 "What is
   not the cause"].
4. We then found a hidden depth bug: the network places each body by its head, but the training targets were built
   at the pelvis, so the network learned to put people too far away and draw them about one third too small; our
   main error measure (PVE) could not see this because it removes the position of the body before comparing
   [Certain, H 3.1].
5. We found that two neck bones (neck02 and head) and two hand bones are turned half a turn in every v5-line model;
   a fixed correction at the output, with no new parameters, turns them back and gives the neck its normal thickness
   [Certain, H 3.2, H 3.3].
6. We found and fixed three defects in our 3DPW-to-Anny labels (reversed gender, a missing person in two-person
   frames, missing labels), re-fitted all 87 people and checked the new labels independently [Certain, H 3.5].
7. The final model puts all fixes together, trains 10 epochs with 30% real 3DPW images, and is compared with a
   control that has the same training without the two fixes [Certain, H 5 item 7].
8. All fixes hold: neck bone 175.8 -> 4.8 degrees, finger error 17.4 -> 8.3 degrees, position error of the body
   1402 -> 728 mm, and the 3DPW test error is 89.0 mm, the best of all our models (v9 95.5, v5 121.8), mostly
   because we trained on 3DPW at all; the fixes add 2.5 mm [Certain, 3611336:191-229, H 0b].
9. On real photos with age and gender labels (Relative Human), the final model is better than v5 on every line
   (gender 46.5 -> 64.2%, children-vs-adults separation 0.61 -> 0.72, people found 47 -> 65%), but children and men
   are not solved, and body build (muscle, weight) hardly changes from person to person [Certain, 3612131:25-33].
10. Still open: near and far people still get similar depths (depth slope 0.42, where 1 is perfect), recall costs 2
    points, every result is one run without a fixed seed, and the requirement "not exactly Multi-HMR" is not met
    yet: the smaller ViT-B student (job 3614191, done 19 Oct morning) is 3x smaller and 2x faster than the final model
    but misses the 95% bar on both test sets (AnnyOne holdout PVE 119.6 vs bar 77.6 mm, 3DPW test 110.1 vs 93.7);
    it does beat NAVER's own Anny model on 3DPW (110.1 vs 126.9) [Certain, C SD].

---

## Part 2. Glossary in plain words

**Multi-HMR.** A published network by NAVER LABS ("Multi-person Human Mesh Recovery"). It looks at the whole image
once, finds every person, and predicts a 3D body for each one. Ours uses a DINOv2 ViT-L/14 "backbone" (the part that
turns the image into features) at 672 pixels, and a cross-attention "head" (the part that turns features into body
values) called HPH (H 1). The image is cut into 48 x 48 squares of 14 pixels, called tokens (H 1 "Terms").

**Anny.** A new body model by NAVER LABS. A body model turns a few numbers into a full 3D mesh. Anny has 163 bones
(each bone has a rotation) and 11 shape values. Unlike SMPL, Anny can make children and old people from the same
model (H 1, H 3.4 "How Anny makes a child"). In our training, 88 small "helper" bones (toes, breasts, face) are kept
at rest, so 75 bones are really predicted (H 1).

**Phenotype (the 11 shape values).** The 11 numbers that set the body shape in Anny, each between 0 and 1: gender,
age, muscle, weight, height, proportions, cup size, firmness, african, asian, caucasian (H 1). Gender 0 = male end,
1 = female end, in Anny's convention (H 3.5). Age 0 = baby, about 0.33 = child, about 0.67 = young adult, 1 = old
(Anny's anchors; H 3.4). Do not turn these ages into years: the code uses three different scales (H 3.4 "Age
scales").

**AnnyOne.** A synthetic dataset: 579,960 computer-rendered images of 1280 x 1280 pixels with full Anny labels for
every person (pose, all 11 phenotypes, camera) [Certain, H 1].

**3DPW.** "3D Poses in the Wild": a public set of real outdoor videos. The ground truth is given in SMPL (another
body model), from body-worn sensors. It labels only the one or two actors, not people walking by (H 5 item 7). We
convert it to Anny ourselves (Part 3).

**Relative Human (RH).** A public set of real photos. Each person has an age group (adult, teen, kid, baby), a
gender and a box, but no 3D body (H 5 item 12). We use it only to test age and gender on real photos.

**Holdout.** The last 100 AnnyOne images (indices 579,860 to 579,959). They are never used for training, so they
test the model on unseen images [Certain, H 1]. The holdout has nobody younger than age 0.513, so no children
[Certain, H 3.4].

**V1 setting.** The standard way we evaluate on the holdout: the 88 helper bones are masked, as in the demos. Most
holdout numbers in the deck are V1. The "V0" setting leaves them unmasked and reads about 3.8 mm worse on the same
people (H 1 "Evaluation settings", "Headline").

**Control run.** A run that is identical to the experiment except for the one thing being tested. It shows what
would have happened anyway, without the change. Example: the exp7 control has the same data, steps and starting
point as the final model, but no depth fix and no bone fix (H 5 item 7).

**Fine-tune.** Continue training an already trained model, usually for a short time and with a small learning rate.
All our experiments fine-tune v5 (H 3.1).

**Pilot.** A short, cheap experiment (here: 2 epochs, or 10 epochs) to see if an idea works before a big run
(H 3.6).

**Bone fix (bone correction).** A fixed rotation of about 180 degrees, applied right after the network outputs a
flipped bone, so that the bone points the right way again. It is estimated once on 200 training images and has no
learnable parameters. In the final model it corrects four bones: neck02, head, metacarpal1.L (left index-finger base)
and finger4-1.R (a right ring-finger bone) [Certain, H 3.2 "exp6 on v5"].

**Depth anchor.** The joint that the model uses to place a person: the detection square, the 2D offset and the
distance from the camera all refer to this joint. In our model it is the head (`--person_center head`). The depth
fix makes the training targets use the same joint (`--gt_center_joint person_center`) instead of the pelvis
[Certain, H 1 "Terms"].

**Masking a loss.** Switching off one part of the training error for some people or some values, so the model is
not trained toward labels we know are not real. Example: 3DPW finger labels are just a rest pose, so we do not train
the fingers on 3DPW people (Part 4).

**Distillation; teacher and student.** A way to train a smaller model (the student) to copy a bigger trained model
(the teacher). The student learns from the normal labels and also from the teacher's features and outputs on the
same image (E exp8).

**Params and FLOPs.** Params = the number of learnable numbers in the network (size). FLOPs = the number of
arithmetic operations for one image (work). We count one multiply-add as 2 FLOPs (`scripts/analysis/model_cost.py`,
docstring). Latency = time per image in milliseconds (ms).

**PVE (per-vertex error).** The average distance, in millimetres, between the predicted and the true mesh points,
after both bodies are moved so that their pelvis is at the same place. Because of this alignment, PVE cannot see if a
body is placed too far or too near (H 1 "Metrics").

**PA-PVE.** PVE after an extra best-fit rotation, scaling and shift (Procrustes alignment). It measures only the
body's pose and shape, not its position, size or overall rotation (H 1).

**MPJPE.** Like PVE, but on joints instead of mesh points. On 3DPW-Anny our MPJPE averages all 163 Anny joints, so it
is not comparable with published 3DPW numbers (P 3.1 item 4).

**Recall and precision.** Recall = the share of real people that the model finds. Precision = the share of the
model's detections that match a labelled person (H 1; H 0b).

**Root error.** The 3D distance between the predicted and the true pelvis joint (the "root"), in millimetres or
metres. It shows depth and placement errors that PVE cannot see (H 1).

**Drawn size (size ratio).** The predicted head-to-pelvis distance in the image, divided by the true one. 1 = right
size; 0.63 = drawn at 63% of the true size (H 1).

**Depth slope.** We fit a straight line: predicted depth = slope x true depth + intercept. A perfect model has slope
1 and intercept 0. Slope 0 would mean the same depth for everyone, near or far (H 1; D slide 13).

**AUC (here: child-vs-adult AUC).** Take a random child and a random adult. AUC is the chance that the model gives
the child the lower predicted age. 0.5 = no better than a coin; 1.0 = always right (H 3.4;
`scripts/diagnostics/eval_relative_human.py` docstring).

**Pilot noise.** Our training sets no random seed, so two runs with exactly the same settings still give different
numbers. Two identical runs differed by 7.6 mm PVE, 1.9 degrees of head direction and 0.8 recall points. So one run
can only show a difference of about 10 mm PVE, 3 degrees or 2 recall points [Certain, H 3.6].

**Restart cost.** Restarting a fine-tune of v5, with nothing changed, already costs about 2.4 mm PVE. That is why each
experiment is compared with its own control, not with v5 [Certain, H 1 "Terms", H 3.6].

---

## Part 3. Datasets: what we trained on and what we evaluated on

| dataset | real or synthetic | what it contains | used for | size (source) |
|---|---|---|---|---|
| AnnyOne | synthetic | 1280 x 1280 renders, several people per image (2.73 on average), full Anny labels: 163 bone rotations, 11 phenotypes, camera | training (70% of each batch in the final model); the last 100 images are the holdout | 579,960 images: 579,860 training + 100 holdout (H 1) |
| 3DPW-Anny, v23 labels | real (outdoor video) | 3DPW's SMPL ground truth converted to Anny by our fit; about 1.2 people per image; every third frame | training (30% of each batch, train split); validation each epoch (300 images); test | train 5,882 images / 7,584 people; validation 2,622 / 3,311; test 8,190 / 11,854; total 16,694 / 22,749 (H 0b, H 3.5) |
| Relative Human | real photos | per person: age group (adult, teen, kid, baby), gender, boxes; no 3D | evaluation only (validation split) | validation 635 images / 2,130 labelled people; whole set 8,741 images (H 3.4, H 0a step 7) |
| Diagnostic manifest | synthetic | 40 training images (adults, children, visible hands, hard necks) + 15 held-out images without children | the diagnostic checks; excluded from training sampling | 55 images (H 1; exp7 header) |
| Demo images | real | 4 photos in `example_data` + 8 3DPW test frames (one middle frame of 8 test sequences) | by-eye comparison of models | 12 images (H 3.2 "Real images") |

The people-per-image numbers are [Certain, H 3.4 "3DPW labels and body shape", from 3571601:21].

### AnnyOne

AnnyOne is our main training set. It is synthetic, so every label is exact, including all 11 shape values
[Certain, H 1]. It has few children: only 2.53% of its people have age below 0.3 [Certain, H 3.4], and in normal
training batches only 2.1-2.3% of the supervised people do [Certain, H 3.4 "The sampler"]. The holdout (last 100
images) has nobody younger than 0.513 [Certain, H 3.4]. The deck also shows an "AnnyOne test-100" number from the 3DPW
test script. That script uses the last-100 AnnyOne images too, but in the V0 setting and with its own matching, so it
gives different values than the V1 holdout trace [Certain that the setting differs, H 0b; Likely that the images are
the same 100, P 3.1 item 5].

### 3DPW, and how we converted it to Anny

3DPW gives SMPL bodies, not Anny bodies. SMPL and Anny have different skeletons and different shape values, so we
wrote our own converter (`scripts/data/smpl_to_anny.py`, now version 2.3) and label builder
(`scripts/data/build_threedpw_anny_labels.py`) (H 3.5; D slide 5).

**The six steps** (D slide 5 and D notes 5; H 3.5 "How they are made"):
1. **3DPW ground truth.** Per frame: 24 SMPL joint positions, the camera, and the person's gender.
2. **Pick frames.** Keep the valid frames and take every third one. Since version 2.3 both people of a two-person
   video share the same frames (`--frame_grid shared`, frame number divisible by 3) (H 4e).
3. **Fit Anny per frame.** Gradient descent moves Anny until its joints land on the 24 SMPL joints in 3D. For nine
   torso joints, where the two skeletons differ, the fit aims at a fixed offset from the bone. Only joint positions
   are matched; SMPL's rotations, shape values and surface are not used.
4. **One body per person.** Take the median shape over all frames of the person, then fit the pose again with that
   one body.
5. **World to camera.** Move the body into the camera's view. Only the root bone changes.
6. **Labels and loader.** Write one label file per image. Drop fits with more than 80 mm joint error, people less
   than 30% visible, and boxes smaller than 40 pixels. In training, 30% of each batch comes from these labels.

**What is fitted, and what is fixed** [Certain, H 3.4 "3DPW labels and body shape", H 3.5; D slide 5]:
- Fitted (free): the root position and rotation; 18 of the 23 mapped bones; height and proportions, one value per
  person.
- From 3DPW's metadata: gender (men 0, women 1 since version 2.3).
- Fixed constants: age = 0.55; muscle, weight, cup size, firmness, african, asian, caucasian = 0.5.
- At rest by design: 144 of the 163 bones, including neck02, neck03, head, jaw and all finger bones. So in these labels
  the head turns only through neck01, and the hands are always in a rest pose.

**Why these limits.** Joint positions alone cannot tell muscle or weight, and SMPL has no finger joints in 3DPW, so
the fit cannot recover them [Likely; P 3.3 says no real set we have gives ground truth for muscle or weight].

**The three label defects, found on 15 October and fixed by 18 October** [Certain, H 3.5]:
1. **Gender reversed.** The old converter used the opposite convention (men 1, women 0), but Anny's male end is 0
   (at gender 0 the rest body is 1.753 m tall, at 1 it is 1.632 m). Flipping the gender made the men fit better in 72
   of 73 cases (3575369:225-226, :313, :316).
2. **Missing person.** In 407 of 960 checked two-person images, one valid actor had no label, because each person had
   their own every-third-frame grid, so the two people were fitted on different frames (3575369:228-229).
3. **Missing labels.** The train split had 4,451 label images where the fits gave 6,437 (validation 2,458 of 2,777),
   because the image files of 8 sequences were missing on the cluster (3575369:31; build script skips frames without
   an image file).

**How they were fixed** [Certain, H 3.5, H 0b]:
- The 8 image folders were re-extracted on 16 October; the second rebuild (job 3581320) brought all labels back.
- Converter version 2.3 adds two switches: gender by Anny's convention, and one shared frame grid. All 87 people were
  fitted again (jobs 3590385 and the rerun 3602813). The re-fit's own check passed on all 60 sequences
  (`logs/check_refit_all_v23.log`): men's mean joint error 19.0 -> 14.7 mm, women 15.0 -> 15.2 mm, two-person frames
  with a missing actor 54% (4,457 of 8,274) -> 0.
- New labels ("v23"): 16,694 images, 22,749 people. Fewer images than before (18,936) but the same people, because
  both actors now share each frame.
- An independent conversion check on the new labels (job 3611284): overall WARN (old labels: FAIL). Body joints
  median 14.1 mm (was 18.7), flipping the gender now makes the fit worse (men 13.9 -> 20.0 mm), 0 of 1,080 two-person
  images miss an actor (was 407 of 960).
- What stays WARN: limb-length ratios and the hip joint, where Anny and SMPL bodies differ; these are body-model
  limits, not label bugs [Likely, H 3.5].
- One validation sequence, `outdoors_crosscountry_00`, has no labels: with 3DPW's own camera its people sit behind the
  camera, so nothing can project into the image [Certain, H 3.5]. This is why the conversion check covers 59
  sequences and the re-fit check 60.

### Relative Human

Relative Human is only for evaluation [Certain, E "Relative Human evaluation", License]. The model runs once per
whole photo. Each detection's box comes from the projected mesh, and detections are paired one-to-one with labelled
people when the boxes overlap enough (IoU at least 0.3). A predicted age below 0.45 counts as "child-shaped". Gender
below 0.5 counts as male [Certain, `scripts/diagnostics/eval_relative_human.py` docstring; H 3.4]. Every number
covers only the people the model detected, so recall is always given next to it (E, "How to read it").

---

## Part 4. The losses: what the model learns

A loss is a number that says how wrong the prediction is. Training changes the network to make the total loss
smaller. The total is a weighted sum of the terms below (`loss.py:399-411`). The weights are set by the training
command; the final model's flags are in `slurm/experiments/exp7_final_model.sh:759-771` (and :211-225 for the depth
weights).

| loss (code name) | what it compares | plain meaning | weight in the final model (code default) |
|---|---|---|---|
| detection (`bce`, `loss.py:314`) | predicted "person here" score per image square vs the true person squares | find each person in the right square | 10 (default 10) |
| offset (`offset`, :315) | the exact position of the anchor inside its square | put the anchor point on the right pixel | 1 (default 1) |
| rotations (`rotmat`, :319-335) | every bone's 3x3 rotation matrix, L1 difference | bend each bone the right way | 0.1 (default 0.1), with neck and hand bones boosted x4 (renormalised: about x2.41 for those bones and x0.60 for the rest, H 1 "x4 boost") |
| shape (`shape`, :336-343) | the 11 phenotypes, after the sigmoid that keeps them in 0-1, L1 difference | give each person the right shape values | 20 (default 1) |
| distance (`dist`, :344) | the distance of the anchor from the camera | right depth | 5 (default 1); the control uses 1 |
| translation (`transl`, :345) | the 3D position of the anchor | right place in 3D | 5 (default 1); the control uses 1 |
| 3D joints (`j3d`, :367-377) | joint positions, both bodies moved to the same pelvis | right 3D skeleton | 100 (default 100) |
| 3D vertices (`v3d`, :378) | mesh points, both bodies moved to the same pelvis | right 3D surface | 100 (default 100) |
| 2D joints and vertices (`j2d`, `v2d`, :380-384) | projected points in the image, in pixels, using the predicted position | the mesh must cover the person in the image | 0.1 each (default 1), on from epoch 0 |
| global rotation (`global_rot`, :350-363) | camera-frame direction of a few bones (head, neck03, wrists) | the head points the right way | 0, off (default 0) |
| distillation (`kd`, `kd_out`, `train.py:512-539`) | the student's features and outputs vs the teacher's on the same image | copy the teacher | not used in the final model; 1.0 and 1.0 in the student (E exp8) |

Sources: weights [Certain, `loss.py:436-451, 463` for the defaults; `exp7_final_model.sh:211-213, 762-766` for the
final model]. The final model's recipe also trains only the last 4 of the 24 backbone blocks, with learning rate 1e-5
and batch 4 [Certain, exp7 :762, :770-771].

Three facts that matter for the slides:
- The 3D joint and vertex terms compare bodies after moving both to the same pelvis, so they carry no depth signal.
  The only strong terms that see depth are the 2D terms, because they project with the predicted position [Certain, F
  "Root cause"; `loss.py:367-384`]. Before the fix, distance and translation together were only about 2.5% of the
  total loss [Certain, F]. [Likely] that is why the depth fix also raises their weight to 5.
- The shape loss is an L1 through a sigmoid. L1 pulls toward the middle value (the median) when the image gives weak
  hints, and the sigmoid's slope is small near 0 and 1, so very young or very old values are learned slowly
  [Certain for the maths, Likely for the effect, H 3.4 "The shape head and its loss"].
- The 3D and 2D terms use a mesh that is built from the label's shape values. So whatever shape values a label has,
  they also enter training through the mesh, not only through the shape loss [Certain, H 3.4; P 3.1 item 2].

**What is masked for 3DPW people, and why.** Every person carries a flag: 1 for AnnyOne, 0 for 3DPW (`train.py`'s
HandLabelFlag; `loss.py:250-253`).
- **Hands (on in the final model).** `--mask_3dpw_hands 1` switches off the rotation and global-rotation terms for the
  38 finger, thumb and metacarpal bones of 3DPW people [Certain, `loss.py:231-253`; 38 bones in the smoke-test log,
  H 0a step 6]. Reason: the converter never fits these bones, so the 3DPW finger "labels" are just a rest pose; without
  the mask the model would learn to keep 3DPW hands flat [Certain, H 3.5]. Note: the 3D and 2D mesh terms are not
  masked, so the rest-pose hands still enter through the mesh [Certain, `loss.py:367-384` has no hand mask].
- **Shape (off in the final model).** `--mask_3dpw_shape 1` would switch off the shape loss for the 8 constant
  phenotypes of 3DPW people (age and the seven 0.5 values); gender, height and proportions stay supervised
  [Certain, `loss.py:12, 255-310`]. It was tested as the masked arm (Part 5.4, slide 19) and not adopted: it brings
  back body variety but loses the gender and children gains [Certain, H 0a "DECIDED 18 Oct"].

---

## Part 5. What we changed in training, and why (v5 -> final model)

The final model starts from v5 (epoch 99) and trains 10 epochs x 1,000 steps x batch 4 = 40,000 images, learning rate
1e-5, last 4 backbone blocks trainable [Certain, H 5 item 7 "Result (18 Oct)"; exp7 :191-192, :224, :762, :770-771].
It ran as job 3611336 on an A100 (training 4,292 s, 10.9 images/s, 16 of 10,000 steps skipped for NaN/Inf
gradients) [Certain, H 0a step 6]. Checkpoint:
`results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt` (H 0a step 6).

### 5.1 Depth anchor fix, with the depth loss x5

- **Purpose.** Remove the hidden depth bug, so bodies are placed at the right distance and drawn at the right size.
- **What it does.** The model places each person by the head joint, but `train.py` built the position, distance,
  detection-square and offset targets at joint 0, the pelvis. The two can never agree, and the 2D losses "solve" the
  conflict by pushing the body farther away, so it is drawn smaller [Certain for the mismatch; Likely for the
  mechanism, H 3.1]. The fix builds the targets at the same joint the model uses (`--gt_center_joint
  person_center`), and makes the direct depth terms 5 times stronger (`--alpha_dist 5 --alpha_transl 5`), exactly as
  in the best pilot `long_pc_dist5` [Certain, exp7 header :3-4, :211-213].
- **Result.** Against its control: root error 1402 -> 728 mm (-48%; v5 1208), drawn size 0.59 -> 0.78, depth slope
  0.28 -> 0.42, intercept 2.66 -> 1.75 m. The head now lands within 7.6 px (control 67.1). The pelvis 2D error got
  worse, 26.7 -> 41.1 px, as in every fix run [Certain, H 3.1 "Final model"; 3611336:175-193, 3611339:166-184].
  The pilot alone reached 544 mm (`long_pc_dist5`, 3572482) [Certain]; why the final model stays at 728 mm is not
  tested [Guessing: the 3DPW batches or the bone fix].
- **Not fixed.** Slope 0.42 is far from 1: near and far people still get similar depths [Certain, H 3.1].
- **Source.** H 3.1, H 0b, H 5 items 7-8.

### 5.2 Bone fix on neck02, head, metacarpal1.L and finger4-1.R

- **Purpose.** Turn back four bones that every v5-line model predicts half a turn off.
- **What it does.** One fixed rotation per bone, applied right after the network's rotation output (R <- R x C).
  The four constants are neck02 178.8, head 179.2, metacarpal1.L 179.8 and finger4-1.R 179.7 degrees, estimated on
  200 training images, never on the holdout (job 3579014:12, :37). The fix is saved in every checkpoint's settings
  and applied again when the model is loaded [Certain, H 3.2 "exp6 on v5"; exp7 header :5-6].
- **Why it is needed.** neck02 and head are each about 175-178 degrees off. The two half-turns cancel, so the head
  still points the right way and PVE never showed it, but the neck skin is twisted thin (thickness 0.46 of the true
  value) [Certain, H 3.2]. A different loss (geodesic) did not move the flip (175.6 -> 175.0) [Certain, H 3.2].
- **Result.** Against its control: neck02 175.8 -> 4.8 deg, head bone 177.7 -> 8.9 deg, neck thickness 0.464 ->
  0.996, neck-skin error 27.7 -> 11.0 mm, finger error 17.4 -> 8.3 deg, head direction 13.0 -> 13.1 deg (inside
  noise) [Certain, H 3.2 "Final model", H 3.3; 3611336:198-207, 3611339:189-198].
- **Not shown.** Why the bones flipped in the first place; the hands on a real photo [Certain, H 3.2, H 3.3].
- **Source.** H 3.2, H 3.3, H 0b.

### 5.3 30% real 3DPW in every batch (v23 labels)

- **Purpose.** Teach the model real photos, not only synthetic ones (the supervisor's requirement 2a).
- **What it does.** 70% of each batch is AnnyOne and 30% is 3DPW-train from the corrected v23 labels, as in v9; 300
  3DPW-validation images are evaluated after each epoch [Certain, exp7 header :7-10; :225, :768-769]. Because 3DPW
  images hold fewer people (about 1.2 vs 2.73), 3DPW is about 16% of the supervised people [Likely, H 3.4].
- **Result.** This is the largest single gain on 3DPW: the control, which has the 3DPW mix but neither fix, already
  gets 91.5 mm PVE on the 3DPW test, against 121.8 for v5. The two fixes add 2.5 mm PVE and 2.1 mm PA-PVE on top
  (final model 89.0 / 57.6) [Certain, H 5 item 7 "3DPW test against every model"]. The final model also finds more
  3DPW people than v9 (recall 99 vs 94) [Certain, H 0b].
- **Side effect.** With the shape mask off, the corrected 3DPW gender labels also train the shape head; on real photos
  most men now look male [Likely, by eye, demo job 3612078; H 3.4 "Final model, by eye"], and gender accuracy on
  Relative Human rises 46.5 -> 64.2% [Certain, 3612131].
- **Source.** H 3.5, H 5 item 7, H 0b.

### 5.4 Hand masking for 3DPW people (and the shape mask that was tested)

- **Purpose.** Do not train the fingers toward the fake rest-pose finger labels of 3DPW.
- **What it does.** `--mask_3dpw_hands 1` switches off the rotation terms of 38 finger, thumb and metacarpal bones for
  3DPW people only [Certain, `loss.py:231-253`; H 0a step 6].
- **Result.** The finger fix holds after 10 epochs with real photos in training: 8.3 deg, control 17.4 [Certain,
  3611336:201, 3611339:192].
- **The shape mask, tested as a separate arm (job 3612325).** Same recipe, plus `--mask_3dpw_shape 1`, which switches
  off the shape loss on the 8 constant 3DPW phenotypes [Certain, H 0a "MASK_3DPW_SHAPE=1 arm DONE"]. Results against
  the unmasked final model:
  - holdout and 3DPW: all inside noise (holdout V1 PVE 71.7 vs 73.7; 3DPW test PVE 91.1 vs 89.0) [Certain, H 0a];
  - Relative Human (job 3613616): recall 57.2% vs 64.7%, gender accuracy 49.3% vs 64.2%, children AUC 0.708 vs
    0.719, babies' predicted age 0.524 vs 0.438 [Certain, H 0a];
  - demo on 12 real images (job 3613617): the spread (standard deviation) of muscle 0.04 -> 0.06, weight 0.04 ->
    0.11, cup size 0.05 -> 0.11, firmness 0.04 -> 0.11, age 0.03 -> 0.08 [Certain, H 0a].
  - So it is a trade-off: body variety comes back, but gender returns to chance and the children gain mostly goes
    [Certain for the numbers]. [Likely] the age label 0.55 was doing useful work (pulling real-photo ages down) and
    the seven 0.5 values caused the collapse of body build. **Decision (18 Oct, 15:30): the unmasked model stays the
    final model and the teacher** [Certain, H 0a].
  - Then: mask only the seven 0.5 values and keep age (job 3614042, done 19 Oct; Relative Human 3616821): children kept
    (AUC 0.727), gender at chance (47.4%), so the unmasked model stays (slide 19) [Certain, H 0a].
- **Source.** H 0a, H 3.4, `loss.py`.

### 5.5 The control run (job 3611339)

- **Purpose.** To know which changes come from the two fixes and which come from training on 3DPW or from simply
  restarting a fine-tune.
- **What it does.** Same start (v5 00099), same data (30% 3DPW on v23, hand mask on, shape mask off), same steps and
  learning rate; but old pelvis targets with depth weight 1, and no bone fix [Certain, exp7 :215-222; H 5 item 7].
  It ran on an A40 (6,753 s, 6.7 images/s; different GPU, so the times do not compare the recipes) [Certain, H 0a
  step 6].
- **Result.** It keeps the neck flip (175.8 deg) and the old depth error (1402 mm), but it already has most of the
  3DPW gain (91.5 mm) [Certain, H 5 item 7].
- **Limit.** One run per arm, no seed: differences under about 7.6 mm PVE or 2 recall points are inside the noise
  [Certain, H 3.6].
- **Source.** H 5 item 7, H 3.6.

---

## Part 6. Slide by slide

Each block: what is on the slide, what we did, why, the result, the numbers with sources, and what is NOT claimed.
"On the slide" describes what the final deck shows; where this file gives a number that the slide does not show, it
says so ("not on the slide", "in the notes"). The deck is the reference: if the two ever differ, the deck wins.

### Slide 1. Multi-HMR with Anny: what we changed and what we measured

- **On the slide** (dark title slide; simplified 20 Oct): the title "Multi-HMR with Anny: bugs found, fixed and
  measured", three one-line results (depth bug fixed for size; neck and finger bones turned back by a fixed
  correction; 3DPW labels cleaned, Final's 3DPW test error 89.0 mm, was 121.8), a "What this means" line and the
  byline [D slide 1]. The numbers 0.63 -> 0.78 and gender 46 -> 64% are now in the notes ("If asked").
- **What we did / why:** one line per result, so the supervisor knows the story before the details. The notes add that
  the depth bug is fixed for size, not yet for near-versus-far depth [D notes 1].
- **Numbers:** size 0.63 -> 0.78 (H 3.1, final model 3611336), 3DPW test PVE 121.8 -> 89.0 mm (H 0b), Relative Human
  gender 46.5 -> 64.2% (3612131) [Certain].
- **Not claimed:** that anything is "solved".

### Slide 2. The goal and your requirements

- **On the slide** (simplified 20 Oct) [D slide 2]: title "The goal, and your requirements: two of six met"; three
  bullets (the task, the test sets, requirement 3: 92-page draft, code clean-up to do); a table of 1a (partly), 1b
  (not met), 1c + 2a (met) and 2b (not started); a "What this means" line; and a small line saying which slides are
  hidden backup. The four-step strip and the requirement details moved to the notes ("If asked").
- **Background, not on the slide** (P 1): the e-mail has three tiers. (1) necessary within one month: 1a, 1b and 1c;
  1b is either a new architecture that beats Multi-HMR on the AnnyOne and 3DPW tests, or a much smaller model (fewer
  params and FLOPs, faster) whose PVE/MPJPE stay "more than 95%" of the large model's on both. (2) recommended: 2a and
  2b. (3) best grade: a report in the student's own voice, clean code, models he can check.
- **What we did:** read the e-mail and matched each item to evidence (slide 23 shows the evidence for each dot).
- **Why:** the supervisor said the thesis is not ready without tier (1) [Certain, P 1].
- **Result:** 1c and 2a met; 1a and 3 partly; 1b not met; 2b not started (slides 2 and 23).
- **Not claimed:** a submission date; it is still unknown, and we ask on slide 26 [Certain, H 10].

### Slide 3. The model in one picture

- **On the slide** (simplified 20 Oct) [D slide 3]: title "The model: Multi-HMR's design plus our fixes"; five boxes,
  photo (672 x 672) -> DINOv2 ViT-L backbone (48 x 48 grid) -> detection (is a person's head in this cell?) ->
  per-person head (pose, 11 shape values, distance) -> Anny body; the last three boxes are teal (ours). Three bullets:
  ours (Anny output, 0-parameter bone fix), ours (head targets, 3DPW conversion), not ours (network design and
  starting weights from NAVER); a "What this means" line (so 1b is open). The 304 M parameters, the 163 / 75 / 88
  bones and SMPL-X are in the notes.
- **What it is:** the upstream Multi-HMR architecture with its output resized for Anny (163 x 6 rotation values and
  11 phenotypes) [Certain, P 2 row 1b; H 1 "Model"]. The bone fix is a fixed rotation, not a new layer [Certain,
  H 3.2].
- **Why show it:** the supervisor must see that our network is still Multi-HMR. That is why requirement 1b is open.
- **Numbers:** since 20 Oct no parameter count on the slide (the backbone's 304 M are in the notes). The whole model is 318.38 M parameters,
  1,927.4 GFLOPs and 110 ms per image on an A40 [Certain, exp8 smoke test 3613926, H 0a]; these are on slide 24, not
  here. Of the FLOPs, the ViT-L backbone alone is about 1,917 [Certain, `model_cost.py` docstring; D slide 24,
  D notes 23].
- **Not claimed:** a new architecture.

### Slide 4. The datasets

- **On the slide** [D slide 4]: a short form of the table in Part 3, with four rows (AnnyOne; 3DPW -> Anny, v23
  labels, 60 sequences; Relative Human val; demo photos, 4 photos + 8 3DPW test frames, judged by eye only), a
  "how to read" line (every number names its dataset; 3DPW numbers are against our converted labels; Relative Human
  counts detected people only) and three tiles: 100 holdout images, 8,190 3DPW test images (11,854 people; one
  sequence yields no labels), 2,130 Relative Human people (635 photos). The diagnostic manifest is not on the slide.
- **What we did:** trained on AnnyOne (70%) and 3DPW-train (30%); tested on the AnnyOne holdout, the 3DPW test split
  and Relative Human validation.
- **Why:** AnnyOne has exact labels but is synthetic; 3DPW is real but has to be converted; Relative Human is real
  photos with children and gender labels, which no other set we have offers.
- **Numbers:** AnnyOne 579,960 images, holdout 100 (H 1); 3DPW v23 train 5,882 / 7,584 people, validation 2,622 /
  3,311, test 8,190 / 11,854 (H 0b); Relative Human validation 635 images / 2,130 people (H 3.4).
- **Not claimed:** that 3DPW-Anny numbers compare with published 3DPW numbers. They are against our converted
  labels, and our MPJPE uses all 163 Anny joints [Certain, P 3.1 item 4].

### Slide 5. 3DPW to Anny: how the labels are made

- **On the slide:** the six steps and the three boxes "matched / free / fixed" (Part 3). The footnote names the code
  as `smpl_to_anny.py` v2.2 and the "fixed" box says the gender is reversed ("next slide"): this slide shows the
  conversion as it was before the v2.3 re-fit [D slide 5].
- **What we did:** wrote the converter: per frame, Anny is fitted so its joints land on 3DPW's 24 SMPL joints; one
  median body per person; then the pose again.
- **Why:** 3DPW only exists in SMPL; requirement 1c asks for Anny values on 3DPW.
- **Numbers:** fitted: root, 18 of 23 mapped bones, height and proportions; gender from metadata; age fixed at 0.55;
  the other 7 values at 0.5; 144 of 163 bones at rest; dropped: fits > 80 mm (on the slide), people < 30% visible and
  boxes < 40 px (in the notes only) [Certain, H 3.5; D slide 5, D notes 5].
- **Not claimed:** that the 3DPW labels tell the true body build (muscle, weight, age), finger pose or head turn
  beyond neck01. They do not.

### Slide 6. 3DPW labels: three defects found and fixed

- **On the slide** (simplified 20 Oct): title "3DPW labels: three defects fixed, now safe to train on"; one table of
  the three defects, before (15 Oct) and now (gender men 0 / women 1; 407 of 960 -> 0 of 1,080 two-person images
  missing a person; 8 sequences short -> re-extracted, 16,694 images and 22,749 people); three bullets (body joints
  14.1 mm, below 3DPW's own ~26 mm; grade FAIL -> WARN; remaining warnings are Anny vs SMPL differences); a "What this
  means" line and the source line naming job 3611284 [D slide 6].
- **What we did:** an independent conversion check (15 Oct, job 3575369) found three defects. We re-extracted the
  missing images (16 Oct), added two switches to the converter (gender convention, shared frame grid), re-fitted all
  87 people (jobs 3590385 + 3602813), built the v23 labels and ran the same check again (job 3611284).
- **Why:** 3DPW is 30% of each batch. With reversed gender, every man would keep learning a female body
  [Certain, D notes 26 "Option 2"].
- **Result:** the three defects are gone and the overall grade moved from FAIL to WARN.
- **Numbers (H 3.5, H 0b):** on the slide until 20 Oct (now partly in the notes): body joints median 14.1 mm (was 18.7), p95 21.1; surface 18.9 / 16.1 mm
  (was 18.9 / 18.3); 2D +0.4 px over raw 3DPW (was +1.1); gender flip now worse (men 13.9 -> 20.0 mm; was better for
  72 of 73); two-person images missing an actor 0 of 1,080 (was 407 of 960); new labels 16,694 images / 22,749 people;
  head turn vs chest 19.6 deg in the labels vs 19.3 in SMPL (was 9.5). In the notes only: re-fit check PASS on 60
  sequences, men's mean joint error 19.0 -> 14.7 mm, women "about 15". Not in the deck: the old p95 (24.5) and the
  women's exact 15.0 -> 15.2 (H 3.5).
- **Not claimed:** perfect labels. Still WARN: limb-length ratios (forearm 0.948, thigh 1.064) and 3.2% of people
  with a joint more than 100 mm off (mostly the left hip in one climbing sequence) [Certain, H 3.5]. These are
  Anny-vs-SMPL differences, not bugs [Likely].

### Slide 7. What the model learns: the losses in plain words

- **On the slide** [D slide 7]: a ten-row table (loss | what it compares | plain meaning | weight in the final model):
  detection, 2D offset, bone rotation, shape, distance, position, 3D joints, 3D mesh points, 2D joints, 2D mesh
  points; a "how to read" line (the total is the weighted sum; "default" is Multi-HMR's value where ours differs; the
  control keeps distance and position at 1; all distances are L1); and a line on what is masked for 3DPW people (the
  38 hand bones; in the masked arm, slide 19, also the 8 constant shape values). The global-rotation and distillation
  terms of Part 4 are not on the slide.
- **What we did:** read `loss.py` and the final model's flags.
- **Why:** the supervisor will ask why a change in one loss (depth x5, hand mask, shape mask) changes the result.
- **Numbers:** detection 10, offset 1, rotations 0.1 (neck and hands boosted x4), shape 20 (default 1), distance 5
  (default 1), position/translation 5 (default 1), 3D joints 100, 3D vertices 100, 2D joints 0.1 (default 1), 2D
  vertices 0.1 (default 1); global rotation 0 is not on the slide [Certain, `loss.py:436-463`,
  `exp7_final_model.sh:211-213, 762-766`].
- **Key points:** the 3D terms are pelvis-aligned, so only the 2D terms see depth (F); the shape loss is an L1
  through a sigmoid, which pulls toward the median (H 3.4); for 3DPW people the 38 hand bones are masked (Part 4).
- **Not claimed:** that these weights are tuned. They are the v5 recipe plus the depth x5 [Certain, exp7 header
  :11-12].

### Slide 8. What we changed in training: v5 → final model

- **On the slide** [D slide 8]: a six-row table (change | why | what it does, final vs control): (1) depth targets
  at the head + distance and position losses x5: drawn size 0.78 (control 0.59), root error 0.73 m (control 1.40 m);
  (2) bone fix on 4 bones at the output, 0 parameters: neck02 4.8 deg (175.8), neck thickness 1.00 (0.46), fingers
  8.3 deg (17.4); (3) 30% real 3DPW-train frames in every batch, v23 labels: 3DPW test PVE 89.0 mm, control 91.5,
  v5 121.8; (4) 3DPW hand losses masked: fingers learned from AnnyOne only; (5) the schedule, 10 epochs from v5,
  40,000 images, batch 4, lr 1e-5, last 4 backbone blocks + heads, 72 min on an A100: both arms end above v5's 70.5 mm
  on the holdout (restart cost); (6) a control with the same recipe minus rows 1 and 2. The notes call rows 1-5 "five
  changes" [D notes 8].
- **What we did:** see Part 5.
- **Why:** each change answers one measured failure; the control separates the fixes from the 3DPW training and the
  restart cost.
- **Numbers:** 10 epochs, 40,000 images, learning rate 1e-5, batch 4, start v5 epoch 99 [Certain, H 5 item 7].
- **Not claimed:** that the four changes were tested one by one in the final setting. The depth and bone fixes were
  tested alone only in pilots (H 3.1, H 3.2); in the final model they are switched together.

### Slide 9. How we measure

- **On the slide** [D slide 9]: a ten-row table (metric | what it measures | how to read it): PVE, PA-PVE,
  recall / precision, root error, drawn size ratio, depth slope, bone angle error (about 180 deg = a flipped bone),
  neck girth ratio (0.46 = half as thick), children AUC, gender accuracy (50% = chance); a "how to read" line on the
  test sets (AnnyOne: 100-image holdout, helper bones masked, matched people; 3DPW: all 8,190 test images against our
  labels); and the noise-floor line.
- **What we did:** a fixed 100-image holdout and fixed settings for every model; a trace script adds root error, size
  and slope because PVE cannot see placement [Certain, H 1 "Metrics"].
- **Why:** the depth bug stayed hidden for months because PVE removes the body position.
- **Numbers:** pilot noise 7.6 mm PVE, 1.9 deg head direction; one run resolves about 10 mm, 3 deg, 2 recall points;
  root error has no measured floor (all on the slide); restart cost about 2.4 mm (not on this slide; slide 22)
  [Certain, H 3.6].
- **Not claimed:** confidence intervals. Each model ran once, without a seed [Certain, H 3.6].

### Slide 10. Where we started: three visible failures

- **On the slide** (simplified 20 Oct): title "Three failures, each explained by an untested guess"; one table failure
  | old guess | what we found: neck (rig problem -> Base (v5): two neck bones half a turn off), fingers (too few
  pixels -> half the error is two flipped hand bones), children (age not learned -> child-like on training images,
  adult on real photos); one bullet (each guess tested on its own, slide 11) and a "What this means" line. The card
  texts are in the notes ("If asked").
- **What was believed before:** a rig or body-model problem for the neck; too few pixels for the hands; age not learned
  for children [Certain, D slide 10].
- **What we did:** tested each belief on its own before changing anything (slide 11).
- **Result in one line each** (the "this week" line of each card): neck = v5's neck02 and head bone are half a turn
  off, the neck skin half as thick, a pilot fix restores it (slides 15-16); fingers = test-time zoom and a loss
  rebalance did not help, half the error was two flipped bones (slide 17); children = child-like on training images,
  adult on real photos (slide 18) [D slide 10].
- **Not claimed:** that all three are fixed. Children are not.

### Slide 11. Checked first: what is NOT the cause

- **On the slide:** a strip of the six checks (ground truth, body model, training signal, predictions vs truth on
  the 100-image holdout + 55-image set, shape swap, overfit test on 40 images), then four results: three with a tick,
  the last with a warning, and a closing line (not rig, decoding or body-model problems; the cause had to be in
  training). The slide merges the old deck's slides 3 and 4.
- **What we did and the results** [Certain, D slide 11 and D notes 11; H 1 "What is not the cause"]:
  - ground truth decodes correctly: AnnyOne labels render onto the people; the camera matches on 100% of images;
  - the body model is fine: all 16 single-joint moves move the right body part; age 0 to 1 changes the skeletal
    height from 0.53 m to 1.54 m;
  - gradients reach all 75 active bones and all 11 shape outputs (two of the three camera outputs get none);
  - the optimiser works: on 40 training images the error falls by 36%; the summed rotation loss barely moved
    (44.6 -> 40.8).
- **Why:** to stop guessing. If the labels or the rig were wrong, no training change could help.
- **Not claimed:** that the overfit test's rising translation and 2D losses (1.67 -> 1.89; 58 -> 75) were a sign of
  the depth bug. The slide says "possibly" (the old deck said "probably"); the handover rates it a guess, because
  those are single-batch prints and the half-averages fall instead (2.10 -> 1.96, 69 -> 63) [Guessing, H 1].

### Slide 12. The hidden depth bug: anchor vs target

- **On the slide** (simplified 20 Oct): title "Depth bug: model uses the head, training the pelvis"; two stick figures
  (the model's anchor at the head; the training target at the pelvis); three bullets (Base places all 222 holdout
  people too far; drawn at 0.63 of true size, 3D height right; hidden from the pelvis-centred error) and a "What this
  means" line (a control run confirms the cause). The depth fit 0.33 x true + 2.36 m and the 64 px head error are in
  the notes.
- **What we found:** `model.py` places each mesh by the head joint and reports the head as its position, but
  `train.py prepare_gt` built the position, distance, detection-square and offset targets at the pelvis [Certain,
  H 3.1]. A one-batch debug print showed predicted offsets of -2 to -19 squares, where the legal range is +/-0.5
  [Certain, 3571599:9-14].
- **Why it stayed hidden:** PVE and PA-PVE align the pelvis first; the real-photo demos place bodies from the detection
  box. Only the synthetic demo and the trace panels use the model's own depth [Certain, H 3.1].
- **Numbers (v5, holdout):** depth slope 0.33, intercept 2.36 m; all 222 people too far; drawn size 0.63; head
  64.5 px off (64 in the notes); 3D body height right (1.405 vs 1.409 m, in the notes) [Certain, H 3.1; D notes 12].
  Not on this slide: v5's root error 1.21 m (slides 13 and 14) and the v9 / v10 root errors of 1.39-1.49 m (H 3.1);
  the deck only says that v9 and v10 have the bug and that v10 is slightly worse [Certain, H 3.1].
- **Not claimed:** that the mechanism (2D losses push the body away) is proven; it is [Likely] (H 3.1).

### Slide 13. Depth fix: before and after

- **On the slide:** holdout image 579862: ground truth | v5 | fix (x5, 10 epochs), drawn with the model's own depth.
- **What we did:** targets at the head joint, depth loss x5, a 10-epoch fine-tune of v5 (pilot `long_pc_dist5`).
- **Result:** the bodies are drawn much closer to their true size; in this image the blue mesh goes from 0.76 to 0.95
  of its true height [Certain, D slide 13].
- **Numbers (whole holdout):** root error 1.21 -> 0.54 m; drawn size 0.63 -> 0.84; slope 0.33 -> 0.38 [Certain,
  H 3.1, job 3572482]. The notes add the final model: root error 0.73 m vs 1.40 for its control and 1.21 for v5
  [D notes 13].
- **Not claimed:** a full fix. Bodies are still about 16% too small at the median in this pilot (22% in the final
  model, size 0.78), and near and far people still get similar depths. The image was picked because no ground-truth
  person is missed: a good case, not a typical one [Certain, D notes 13].

### Slide 14. Depth fix: the numbers

- **On the slide** (simplified 20 Oct): title "Depth fix: placement error almost halved"; one table Base (v5) |
  Control | Final with placement error 1.21 / 1.40 / 0.73 m, drawn size 0.63 / 0.59 / 0.78 and near/far scaling 0.33 /
  0.28 / 0.42; three bullets (Final vs Control, near/far not fixed, likely cost of 2 recall points) and a "What this
  means" line [D slide 14]. The seven-run pilot table moved to the notes ("If asked").
- **Numbers** [Certain, H 3.1 table]:
  - fix runs: root error 0.53-0.73 m; controls 1.11-1.44 m;
  - size 0.78-0.87 with the fix, 0.59-0.66 without;
  - slope 0.30 with the switch alone, 0.38-0.44 with the stronger depth loss;
  - x20 for 2 epochs (0.53 m) ties x5 for 10 epochs (0.54 m);
  - cost in the pilots: every fix run is worse than its control by +2.6 to +4.9 mm PVE and -1.6 to -2.4 recall points
    (each inside noise, all the same direction) [Likely cost];
  - final model vs its control (the slide's table since 20 Oct): root error 1.40 -> 0.73 m (v5 1.21), size 0.59 -> 0.78, slope
    0.28 -> 0.42; its PVE 77.4 -> 73.7 and recall 89.0 -> 87.0 are on slide 20, not here (the notes only say "2
    recall points, PVE inside the noise") [Certain, 3611336:175-193, 3611339:166-184].
- **Why x5** (not in the deck): before the fix, distance and translation were only about 2.5% of the loss
  [Certain, F].
- **Not claimed:** that exp1 met its own success rule. It did not: every run, controls too, misses "PVE <= 71.5 and
  recall >= 89.2" [Certain, H 3.1]. The x20 run for 10 epochs was never confirmed [Certain, H 3.1, H 4d].

### Slide 15. Neck: two half-turned bones, and a pilot fix

- **On the slide** (simplified 20 Oct) [D slide 15]: title "Neck: two bones were half a turn off; now fixed"; one
  table Base | pilot fix (2 epochs) | Final with neck02 175.6 / 4.7 / 4.8 deg, head bone 177.9 / 11.2 / 8.9 deg,
  thickness 0.46 / 1.00 / 1.00 and head direction 12.0 / 14.4 / 13.1 deg; three bullets (cause, fix, caveat) and a
  "What this means" line. The fix-without-training and control columns moved to the notes.
- **What we found:** in all 8 v5-line checkpoints neck02 is 175.6 deg off (a half-turn about its own axis) and the head
  bone 177.9 deg; the two cancel, so the head still points right (12.0 deg) and PVE never showed it; but the neck skin
  is twisted thin [Certain, H 3.2].
- **What we did:** exp6 (job 3579014): one fixed rotation per flipped bone at the output; then a 2-epoch fine-tune.
- **Numbers** [Certain, H 3.2 table; 3579014:101-114]:

  | holdout V1 | v5 | fix, no training | fix + 2-epoch fine-tune | control, no fix | final model |
  |---|---|---|---|---|---|
  | neck02, deg | 175.6 | 4.5 | 4.7 | 175.7 | 4.8 |
  | head bone, deg | 177.9 | 12.4 | 11.2 | 177.9 | 8.9 |
  | neck thickness, pred / true | 0.46 | 1.00 | 1.00 | 0.46 | 1.00 |
  | neck-skin error, mm | 27.3 | 10.8 | 10.7 | 27.9 | 11.0 |
  | body PVE, mm | 70.5 | 71.3 | 68.8 | 76.9 | (73.7, slide 20) |
  | head direction, deg | 12.0 | 19.1 | 14.4 | 13.8 | 13.1 |

  Until 20 Oct the first four columns were the slide's table; since 20 Oct the slide shows v5, the 2-epoch fine-tune and the final model. The final-model numbers are from
  3611336:198-207 (its PVE is on slide 20).
- **Not claimed:** "no training needed" (the head direction gets worse without the fine-tune, 19.1 deg); a PVE gain
  (68.8 vs 76.9 for the control is inside noise: say "no sign of a PVE cost"); why the bones flipped (origin unknown;
  the singular start value of the rotation output would give a tilt, not this twist) [Certain, H 3.2].

### Slide 16. Neck: before and after the fix

- **On the slide:** two before | after pairs, necks ringed in yellow. Left: synthetic holdout image 579862, v5 vs the
  same model with the correction alone, no training (`docs/figures/neck_before_after_v5.jpg`). Right: real 3DPW test
  frame `downtown_bar_00` 701, v5 vs the 2-epoch fine-tuned fixed model, demo job 3581312
  (`docs/figures/neck_before_after_3dpw_bar.jpg`).
- **Result:** with v5 the necks are thin stalks pinched under the head; with the fix they have normal width
  [Certain on the 12 holdout panels; Certain for the 8 real images seen, judged by eye] (H 3.2 "Visual check", "Real
  images").
- **Numbers:** thickness 0.46 -> 1.00 on the 100-image holdout (job 3579014:107) [Certain].
- **Not claimed:** a neck number on real images (there is no neck ground truth there); that the correction, and not
  the 2-epoch fine-tune, makes the real-image difference (the right pair differs by both; on the holdout the same
  fine-tune without the fix keeps thickness 0.461, so the correction is the [Likely] cause) [H 3.2]. 4 of the 12 images
  of this demo (job 3581312) were not looked at; the runForBus, sitOnStairs and fencing frames have since been seen
  only in the final model's demo montages, with no neck reading recorded [Certain, D notes 16].

### Slide 17. Fingers: about half the error was two flipped bones

- **On the slide** (simplified 20 Oct): title "Fingers: turning two flipped bones back halves the error"; one table of
  what was tried: Base 17.3 deg, test-time zoom 17.2, rebalanced loss 17.5 (control 17.4), bone fix in Final 8.3
  (control 17.4); two bullets (two flipped hand bones, same fix; by eye on photos, slide 31) and a "What this means"
  line. The hand-size table and the rest-pose 15.1 deg are in the notes.
- **What we did:** first tested the old beliefs: magnifying the person at test time (17.3 -> 17.2) and a rebalanced
  loss (17.5 vs 17.4 for its control); neither helped [Certain, H 3.3]. Then the per-bone search found two hand bones
  turned half a turn: metacarpal1.L (180 deg) and finger4-1.R (178.8 deg) [Certain, H 3.2, H 3.3].
- **Result:** correcting them halves the finger error: 17.3 -> 8.2 deg without training, 8.3 after the 2-epoch
  fine-tune, and 8.3 in the final model against 17.4 for its control [Certain, 3579014:114; 3611336:201,
  3611339:192]. On v10, correcting three flipped hand bones takes the finger error 32.0 -> 17.8 deg (in the notes,
  not on the slide) [Certain, 3579015:80; D notes 17].
- **By eye (slide 31):** with predicted fingers (demo job 3615018, `--relax_hands 0`), Base's fingers are splayed and
  stiff, one left-hand finger sticking out; Final's are closed and bent and read as hands; neither puts them on the
  real fingers [Certain by eye; REVIEW_LOG 19 Oct].
- **Not claimed:** a finger number on real photos, or fingers that match the real ones; better-looking fingers rest on
  the eye only [Certain, D notes 17].

### Slide 18. Children and gender on real photos

- **On the slide** (simplified 20 Oct) [D slide 18]: title "Real photos: gender and children improve, but not solved";
  one table Base | Final with recall 47 / 65%, gender 46.5 / 64.2%, children AUC 0.61 / 0.72 and babies below the
  child line 3 / 49%; three bullets (Base gives the average body; Final recognises women, men mid-range; kids still
  near adult age) and a "What this means" line. The masked-arm column, the mean ages and the three synthetic tiles are
  in the notes ("If asked").
- **What we did:** ran the models on 635 Relative Human photos with 2,130 labelled people (jobs 3611285_0 for v5,
  3612131 for the final model and 3613616 for the masked arm) [Certain, H 3.4, H 0a].
- **Why:** our own photos hold only one real child, so they cannot measure children; earlier "five small people" turned
  out to be mostly crouching or seated adults [Likely, by eye, H 3.4].
- **Result:** v5 reads children as short adults: their predicted age is adult, only their height value drops; and it
  has no gender signal. The final model is better on every line, but not solved.
- **Numbers** [Certain, H 3.4 tables; 3611285_0:23-31, 3612131:25-33]:

  | Relative Human val | v5 | final model | masked arm |
  |---|---|---|---|
  | recall (people found) | 47% | 65% | 57% |
  | gender accuracy | 46.5% | 64.2% | 49.3% |
  | child-vs-adult AUC | 0.61 | 0.72 | 0.71 |
  | predicted age: adult / kid / baby | 0.63 / 0.61 / 0.62 | 0.54 / 0.50 / 0.44 | 0.64 / 0.61 / 0.52 |
  | babies below the child threshold 0.45 | 3% | 49% | 17% |

  Not in the table on the slide (H 3.4): the mean predicted gender of men / women, v5 0.535 / 0.551 (in the notes of
  slide 16), final 0.551 / 0.861 (in the notes as 0.55 / 0.86); and the height value of adults / kids / babies, v5
  0.51 / 0.45 / 0.39, final 0.50 / 0.47 / 0.46.

- **Why children fail (v5):** the shape head regresses to the training median, in the synthetic tails and on real
  photos; AnnyOne has few children; an L1 through a sigmoid learns extreme values slowly [Certain for the pull; the
  cause cannot be ranked, H 3.4].
- **Not claimed:** children or gender solved (men are still mid-range, 0.55); that the two AUCs cover the same people
  (the final model matched more: 1,305 vs 948 children and adults) [Certain, H 3.4]. [Likely] part of the final
  model's age shift comes from the 3DPW age label 0.55 (all real people moved down: adults 0.63 -> 0.54). The AUC
  ranks people, so a common shift alone cannot raise it; the 0.61 -> 0.72 is real separation [Certain, H 3.4].

### Slide 19. Masked shape labels: a trade-off

- **On the slide** [D slide 19]: two boxes, "What the mask does" (the problem: 8 constant shape values in every
  3DPW label; the final model learned them, spread about 0.04 on real photos; the mask: do not learn these 8 values
  from 3DPW people, keep gender, height and proportions; training job 3612325) and "What happened (one run each)"
  (variety comes back; gender and children fall back; error unchanged within noise), and a bottom line "Result,
  19 Oct" (rebuilt 19 Oct, 01:20; it read "Reading and next step ... job 3614042, queued" before): the partial mask,
  seven 0.5 constants masked and age kept (jobs 3614042, 3616821), keeps the children (AUC 0.73, babies 0.47) but
  gender is at chance (47%); so age carries the children and gender likely leans on the constants too (untested);
  the unmasked model stays the final model and the student's teacher.
- **What we did:** one extra run with `--mask_3dpw_shape 1`, which stops the shape loss on the 8 constant 3DPW values
  (job 3612325); then Relative Human (job 3613616) and the demo (job 3613617) [Certain, H 0a].
- **Why:** on real photos the final model's muscle, weight, cup size and firmness sit at about 0.5, the converter
  constants, so shape does not visibly change per person for those values (requirement 1a) [Certain, H 3.4 "Relative
  Human, final model", item 4].
- **Numbers** [Certain, H 0a]:

  | | final model (unmasked) | masked arm |
  |---|---|---|
  | Relative Human recall | 64.7% | 57.2% |
  | gender accuracy | 64.2% | 49.3% |
  | children AUC | 0.719 | 0.708 |
  | babies' predicted age | 0.438 | 0.524 |
  | demo spread (std) of muscle / weight / cup size / firmness | 0.04 / 0.04 / 0.05 / 0.04 | 0.06 / 0.11 / 0.11 / 0.11 |
  | holdout V1 PVE / 3DPW test PVE | 73.7 / 89.0 | 71.7 / 91.1 (inside noise) |

  All of these are on the slide, as rounded values (gender 64 -> 49%, recall 65 -> 57%, babies 0.44 -> 0.52 with
  adults at 0.64, AUC 0.72 -> 0.71). The age spread (0.03 -> 0.08, H 0a) is not on the slide.

- **Result:** body variety vs gender and children. The unmasked model stays the final model and the teacher
  [Certain, H 0a "DECIDED"].
- **Partial mask, done 19 Oct (01:20)** [Certain, `logs/exp7_final_model_3614042.out:191-215`,
  `logs/eval_rh_3616821_0.out`; H 0a]: the seven 0.5 values masked, age and gender kept (job 3614042; Relative Human
  job 3616821). Holdout V1 73.0 / PA-PVE 51.2 / recall 87.4, 3DPW test 89.9 / 58.9 (inside noise); Relative Human
  recall 63.0%, gender 47.4% (men 0.725, women 0.888), AUC 0.727, ages adult 0.532 / kid 0.506 / baby 0.465, babies
  below 0.45 34.3% (final 49.3%, masked arm 16.9%). The rule (gender >= 60%, AUC >= 0.7, spread opened) fails on
  gender; its demo spread was not measured (not needed). **The unmasked model stays the final model and the teacher.**
  [Guessing, not tested] gender stayed supervised and still fell to chance, so the final model's gender gain likely
  depends on the seven constants on 3DPW men as much as on the gender label. The earlier reading (the age label carries
  both gains) holds for the children only.
- **Not claimed:** that the masked values are *right* on real people. No real set we have gives ground truth for muscle
  or weight [Certain, P 3.3]. One run per arm, noise applies.

### Slide 20. Final model: all fixes together, with a control

- **On the slide** (simplified 20 Oct) [D slide 20]: title "Final model: all fixes hold together against a control";
  one table Base | Control | Final with holdout PVE / PA-PVE, 3DPW test PVE, placement error and neck / finger bone
  error; three bullets (Final beats Multi-HMR-Anny on 3DPW, 89.0 vs 126.9; most of the gain is from 3DPW training,
  Control 91.5; cost of 2 recall points, near/far scaling 0.42), a "What this means" line and a source line. Recall,
  size ratio, neck girth, the five-model 3DPW line and the recipe line moved to the notes.
- **What we did:** exp7: fix arm 3611336 and control 3611339 (Part 5).
- **Numbers** [Certain, H 5 item 7; 3611336:191-229, 3611339:182-220; 3612073-3612075]:

  | holdout V1 (100 images) | v5 | control | final model |
  |---|---|---|---|
  | PVE, mm | 70.5 | 77.4 | 73.7 |
  | PA-PVE, mm | 53.3 | 56.4 | 50.3 |
  | recall | 90.2% | 89.0% | 87.0% |
  | root error, mm | 1208 | 1402 | 728 |
  | drawn size (1 = right) | 0.63 | 0.59 | 0.78 |
  | depth slope (1 = right; not a table row on the slide) | 0.33 | 0.28 | 0.42 |
  | neck02, deg | 175.6 | 175.8 | 4.8 |
  | neck thickness | 0.46 | 0.46 | 1.00 |
  | fingers, deg | 17.3 | 17.4 | 8.3 |
  | 3DPW test PVE, mm (v23, 8,190 images) | 121.8 | 91.5 | 89.0 |

  3DPW test, all on the same labels and script: v5 121.8, v9 (old best) 95.5, 2-epoch neck pilot 120.2, control 91.5,
  final 89.0 mm PVE; PA-PVE 72.0 / 62.9 / 68.7 / 59.7 / 57.6; final recall 99% vs v9 94% [Certain, H 0b]. The slide
  shows the PVE line and the recall; the PA-PVE values are in the notes. NAVER's own Anny model
  (`multiHMR_672_L_anny`, run with our code, labels and settings) on the same 8,190 images: PVE 126.9, PA-PVE 68.8,
  MPJPE 129.4, PA-MPJPE 66.4, precision 75%, recall 97% [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]: about v5's level and 37.9 mm behind the final
  model. It never trained on 3DPW, so this is the fairer of its two numbers (slide 24).
- **Result:** all fixes hold together; PA-PVE 50.3 is the best of every model with a V1 PA-PVE; the final model is the
  best on our 3DPW-Anny labels, mostly because of the 3DPW training (control 91.5); the fixes add 2.5 mm PVE
  [Certain, H 5 item 7].
- **Not claimed:** a PVE gain on the holdout (77.4 -> 73.7 is inside noise; v5's 70.5 is still the lowest holdout PVE,
  because every fine-tune pays the restart cost); fixes without cost (recall -2 points vs the control, -3.2 vs v5,
  stated as a real cost); correct near/far depth (slope 0.42); "best model" in general (say "best on our 3DPW-Anny
  labels"); the retired old-label numbers (v9 102.5, v5 119.1) must never be put next to these [Certain, H 0b, H 5
  item 7]. Precision on 3DPW is low for every model (final 64%), partly because 3DPW labels only the actors, not the
  bystanders [Likely, H 5 item 7].

### Slide 21. Final model on real photos: v5 vs final (by eye)

- **On the slide** (simplified 20 Oct): title "Real photos by eye: men look male, children still adult"; the figure
  `docs/figures/final_model_real_photos.jpg` (labels baked in: v5 left, final model right); three bullets (men look
  male; the boy and hard poses unchanged; gender 46.5 -> 64.2% on Relative Human), a "What this means" line and the
  source line (demo job 3612078).
- **What we did:** demo job 3612078: v5 next to the final model on 4 photos and 8 3DPW test frames; the bone fix is
  applied from the checkpoint's settings [Certain, H 3.4 "Final model, by eye"].
- **Result (by eye) [Likely]:** most men now get male bodies (flat chest, broader shoulders): 3 of 4 men on the tennis
  photo, both men in the walking frame, the man on the phone, the guitar player, the seated man; women stay female;
  bodies look a little larger (drawn size 0.63 -> 0.78 on the holdout).
- **Unchanged [Certain, by eye]:** the boy is still an adult body; some bodies sit off their person; ghost bodies on
  half-visible people; hard poses (guitar, seated) are poor in both.
- **Measured:** Relative Human gender 46.5 -> 64.2%, children AUC 0.61 -> 0.72, recall 47 -> 65% (job 3612131)
  [Certain].
- **Not claimed:** a gender number from the demo. The demo log's gender values spread out (v5 0.32-0.76, final
  0.05-0.99) but are not matched to bodies, and on the tennis photo most values are above 0.7 although the men look
  male [Certain, H 3.4]. The pictures stay a by-eye judgement.

### Slide 22. Pilots: four results and the noise floor

- **On the slide:** a table of the four 2-epoch / 10-epoch experiments (experiment | what it changes | success rule |
  result), and a box on pilot noise.
- **Results** [Certain, H 3.1-3.4, H 3.6; D slide 22]:
  - **exp1 depth anchor:** rule not met (PVE and recall miss in every run, controls too; x5 for 10 epochs 74.8 mm,
    87.8%); outside the rule clearly better: root error 1.21 -> 0.54 m, size 0.63 -> 0.84.
  - **exp2 children sampler** (5 times more synthetic children per batch, 2.1% -> 9.7-11%): inconclusive after 2
    epochs (10 training children 48.5 vs 53.1 mm; both arms with age loss x3 and height x2; baby age floor
    0.100 -> 0.102, not in the deck). The slide adds v5 on 127 real Relative Human children: adult age, AUC 0.61.
  - **exp3 neck, global-head loss:** 11.3 deg passes the < 12 rule, but two identical controls gave 11.7 and 13.6:
    inside noise. Its safety limit (PVE <= 71.5 mm, recall >= 89.2%) was just missed, 71.8 mm and 89.0%; all five
    control runs miss it too.
  - **exp4 fingers, rebalanced loss:** no effect, 17.5 vs 17.4 deg.
- **Noise box:** two identical runs differed by 7.6 mm PVE and 1.9 deg; one run resolves about 10 mm, 3 deg, 2 recall
  points; restart cost about 2.4 mm (controls average 72.9 vs 70.5 mm); root error has no measured noise floor, but
  every fix run (0.53-0.73 m) is below every control (1.11-1.44 m) [Certain, H 3.6; D slide 22].
- **Not claimed:** that the sampler cannot work (unproven at 2 epochs, not "no effect"); that exp3 improved the head.
  The later fixes (depth targets, bone fix) came from diagnostics, not from these loss changes.

### Slide 23. Thesis requirements: where we stand

- **On the slide** (simplified 20 Oct): title "Requirements: data done, pose and shape partly, 1b open"; one table
  requirement | status | evidence | next with four rows (1a pose partly; 1a shape partly, at risk; 1b not met; 1c + 2a
  met); two bullets (2b not started; 3 partly) and a "What this means" line. The longer evidence and next-step texts
  are in the notes ("If asked") and in Part 7.
- **Status words:** 1c met; 2a met; 1a pose partly; 1a shape partly, at risk; 1b not met; 2b not started; 3 partly
  [Certain, P 2; D slide 23].
- **Student in row 1b** [Certain, C SD; job 3614191]: AnnyOne holdout PVE 119.6 mm against the bar 77.6 (teacher
  73.7), 3DPW test PVE 110.1 against 93.7 (teacher 89.0): the 95% rule is missed on both sets; against NAVER's 3DPW
  number (126.9) the student is better. The status stays "not met".
- **NAVER baseline (notes of slide 23)** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]: NAVER's own Anny model gets 60.3 mm PVE on our AnnyOne holdout
  and 126.9 on the 3DPW test, against our final model's 73.7 and 89.0. [Likely] it was trained on the whole AnnyOne
  release, so the holdout images are probably in its training set (held out for us, not for it). [Certain] On real
  3DPW it is far worse than ours.
- **NAVER's size in row 1b** [Certain, job 3614947, `logs/model_cost_3614947.out`, `results/diagnostics/model_cost_naver_vs_final.json`; H 0a]: 380.9 M parameters against our 318.4 M; same ViT-L
  backbone, so the difference is the head (76.6 M vs 14.0 M). Ours is already 16% smaller, which by itself does not
  meet 1b ("significantly smaller").
- **Not claimed:** that 1b is met by the current model or by the student. The final model is still Multi-HMR
  [Certain, P 3.1 item 1]; the student misses the bar against our ViT-L [Certain, C SD]. Nor that our models beat
  NAVER's in general: they do on 3DPW, not on AnnyOne.

### Slide 24. Plan for 1b: a smaller model by distillation

- **On the slide** (simplified 20 Oct) [D slide 24]: title "Student: three times smaller, but misses the 95% bar"; one
  table Final | Student | Multi-HMR-Anny | 95% bar with parameters (318.4 / 101.4 / 380.9 M) and holdout / 3DPW test
  PVE (73.7 / 89.0, 119.6 / 110.1, 60.3* / 126.9; bars 77.6 / 93.7 missed); three bullets (3.14x smaller, 2.10x
  faster; misses both bars but beats Multi-HMR-Anny on 3DPW; bars vs Multi-HMR-Anny 63.5 / 133.6 mm, 3.76x fewer
  parameters), a "What this means" line and a source line (28.5 vs 59.7 ms on one L40S). GFLOPs, the 95% rule text,
  the recipe and the timeline moved to the notes.
- **What we did:** wrote exp8 (`slurm/experiments/exp8_distill_vitb.sh`). The student is a DINOv2 ViT-B initialised
  from NAVER's Multi-HMR ViT-B; a small projector (768 -> 1024) feeds heads copied from the teacher; it learns from the
  normal labels plus the teacher's features (weight 1.0) and outputs (weight 1.0); same targets as the teacher (depth
  fix, 30% 3DPW, hand mask, teacher's bone fix); batch 8, learning rate 3e-5, 30 epochs x 1,000 steps = 240,000 images
  [Certain, exp8 header; E exp8]. The slide's recipe bullet adds the distillation temperature, T 4.
- **Status:** smoke test 3613926 passed end to end; full run job 3614191 done on the morning of 19 Oct: 30 epochs x
  1,000 steps x batch 8 on one L40S in 20,894 s (5.8 h, 11.6 images/s); 77 NaN/Inf steps skipped; heads frozen in
  epochs 0-4 [Certain, C SD]. The 95% rule fails (below). NAVER's own Anny model on the same labels: done 18 Oct evening, job 3614192 (the first
  try, 3613928, crashed on a head-size mismatch, now fixed; that job id is not in the deck) [Certain, H 0a].
- **NAVER baseline numbers** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a] (518/518 tensors loaded, `--model_class multi_hmr_original`; Multi_HMR
  masks the helper bones by design):

  | | NAVER `multiHMR_672_L_anny` | our final model |
  |---|---|---|
  | AnnyOne holdout (same 100 images, helper bones masked): PVE / PA-PVE / MPJPE / PA-MPJPE, mm | 60.3 / 44.6 / 63.3 / 45.0 | 73.7 / 50.3 / - / - |
  | AnnyOne: precision / recall / F1 | 82.0% / 89.0% / 85.0 | - / 87.0% / - |
  | 3DPW test (v23, 8,190 images): PVE / PA-PVE / MPJPE / PA-MPJPE, mm | 126.9 / 68.8 / 129.4 / 66.4 | 89.0 / 57.6 / 88.3 / - |
  | 3DPW: precision / recall / F1 | 75.0% / 97.0% / 85.0 | 64% / 99% / - |
  | params: backbone + heads = network | 304.37 + 76.57 = 380.94 M | 304.37 + 14.01 = 318.38 M |
  | GFLOPs: backbone + heads | 1,917.2 + not counted | 1,917.2 + 10.16 = 1,927.4 |
  | ms per image, one person, median (A100, job 3614947) | not measured (class not supported by bench()) | 77.9 (backbone 60.8; peak 1.54 GB) |
  | ms per image (A40, smoke test 3613926) | - | 110 |

  Caveats: [Likely] NAVER trained on the whole AnnyOne release, so our holdout is probably in-sample for it and its
  60.3 is not a fair test number. [Certain] On 3DPW it is far worse than ours and about v5's level (121.8).
  Cost rows [Certain, job 3614947, `logs/model_cost_3614947.out`, `results/diagnostics/model_cost_naver_vs_final.json`; H 0a] (`scripts/analysis/model_cost.py` on an NVIDIA A100-PCIE-40GB, fp32 + TF32,
  batch 1, 20 warm-up + 50 timed; NAVER's weights loaded 518/518). In plain words: the backbone is the same, so the whole
  difference is the head: NAVER's 8-layer cross-attention decoder is 76.6 M parameters, ours 14.0 M. Our final model
  already has 16% fewer parameters than NAVER's (0.836 of it), which by itself is not "significantly smaller". The
  student (101.38 M) is 3.76x smaller than NAVER's model and 3.14x smaller than ours. The ms of different GPUs do not
  mix: the slide's teacher / student pair is the L40S of job 3614191 (2.10x); the A40 smoke test gave 2.14x.
- **Cost numbers** [Certain, job 3614191 `3614191:213-221`, C SD; smoke test 3613926, H 0a]:

  | | teacher ViT-L (final model) | student ViT-B | ratio |
  |---|---|---|---|
  | params | 318.38 M | 101.38 M (backbone 86.58, projector 0.79, heads 14.01) | 3.14x fewer |
  | GFLOPs per image | 1,927.4 | 603.3 | 3.19x fewer |
  | ms per image, full forward (L40S, job 3614191; on the slide) | 59.7 (backbone 50.7) | 28.5 (backbone 19.1) | 2.10x faster |
  | ms per image (A40, smoke test 3613926; no longer on the slide) | 110 | 51.4 | 2.14x faster |

  The slide rounds these to 318.4 M / 101.4 M and 1,927 / 603 GFLOPs.
- **Student result** [Certain, C SD; holdout `3614191:195-207`, final table `3614191:259-283`]:

  | | student ViT-B | teacher (final model) | 95% bar (teacher / 0.95) | verdict |
  |---|---|---|---|---|
  | AnnyOne holdout V1: PVE / PA-PVE / MPJPE, mm | 119.6 / 80.8 / 127.2 | 73.7 / 50.3 / 77.9 | PVE <= 77.6, MPJPE <= 82.0 | missed (+62% PVE) |
  | AnnyOne holdout: recall / false positives | 85.4% / 64 | 87.0% / 42 | - | recall kept |
  | AnnyOne test-100 V0: PVE / PA-PVE / MPJPE, recall | 122.7 / 84.2 / 127.7, 85 | 77.5 / 54.4 / 78.5, 87 | - | - |
  | 3DPW test v23 (8,190 images): PVE / PA-PVE / MPJPE, mm | 110.1 / 71.8 / 112.0 | 89.0 / 57.6 / 88.3 | PVE <= 93.7, MPJPE <= 92.9 | missed (+24% PVE) |
  | 3DPW test: recall | 99% | 99% | - | recall kept |

  exp8's own verdict: FAIL on all four bars (PVE and MPJPE on both sets). Rotation errors (deg), student vs teacher:
  root 13.5 vs 8.7, spine 6.5 / 5.8, neck 5.7 / 5.5, head 4.5 / 5.0, arm 10.6 / 8.4, fingers 9.3 / 8.3, leg 6.4 / 5.1;
  root position error 830.7 vs 728.5 mm. The student's own bone fix (neck02 178.9, head 179.1, metacarpal1.L 179.7,
  finger4-1.R 179.9 deg) is 0.2-1.8 deg from the teacher's, and the holdout is the same with either (119.6 vs 119.7),
  so the bone fix does not explain the gap.
- **Training curve** (in-training holdout V0 PVE / 3DPW-val PVE, `3614191:59-143`) [Certain, C SD]: epochs 5, 10, 15,
  20, 25, 30: 165.1/104.9, 170.0/101.5, 139.5/93.0, 133.0/90.9, 130.5/88.0, 122.8/87.7. AnnyOne was still falling
  (about 8 mm per 5 epochs); 3DPW-val was flat (88.0 -> 87.7). [Likely] a longer run helps AnnyOne; [Guessing] it is
  unlikely to close the 16 mm 3DPW gap, since that curve is flat.
- **Against NAVER's Multi-HMR-Anny** [Certain, C SD; job 3614192]: 3DPW 110.1 vs 126.9 mm (the student is better)
  with 101.38 vs 380.94 M params (3.76x fewer); AnnyOne 119.6 vs 60.3 (NAVER's number likely in-sample). Against
  NAVER's bars (63.5 / 133.6) the 3DPW bar is met, the AnnyOne bar is not.
- **The 95% rule:** for PVE and MPJPE on both the AnnyOne holdout and the 3DPW test, student error <= teacher error /
  0.95 (exp8's rule; targets about 77.6 mm and 93.7 mm), plus fewer params and FLOPs and faster [Certain, exp8 header
  "THE 95% RULE"]. The deck uses only this /0.95 form; the ratio is reported against both our ViT-L and NAVER's
  [Certain, P 3.2]. Against NAVER the bars would be 60.3 / 0.95 = 63.5 mm on AnnyOne, which no student of a 73.7 mm
  teacher can reach, and 126.9 / 0.95 = 133.6 mm on 3DPW, trivially met; against our ViT-L 77.6 / 93.7. The AnnyOne gap
  to NAVER is 13.4 mm and likely in-sample for NAVER; that is why question 1 on slide 26 matters.
- **Why ViT-B:** NAVER's own ViT-B equals its ViT-L on 3DPW (PVE 94.0 vs 94.1; the slide says "equal on 3DPW", the
  numbers are in the notes of slide 23) [Certain, fetched NAVER README, P 3.2]; our old ViT-S students had 1.17-1.54x
  their teacher's error (notes of slide 23) [Certain, P 2].
- **Not claimed:** that the student passes: it does not, against our ViT-L, on either set [Certain, C SD]. The plan's
  guess before the run (likely on 3DPW, about 50/50 on AnnyOne; P 3.2) was wrong on 3DPW. Nor that a longer run would
  pass. Which reference sets the bar (our ViT-L or NAVER's), and whether a smaller Multi-HMR-style student counts as
  "not exactly Multi-HMR", is the supervisor's decision (question 1, slide 26). The options already in the plan (P 3.2
  fallback): train the same ViT-B longer from its last checkpoint (epoch 29), since AnnyOne was still falling, with the
  ratio reported as measured; or stop the smaller-model route and argue 1b through the comparison with NAVER's baseline.

### Slide 25. Plan: the next four weeks

- **On the slide** (simplified 20 Oct) [D slide 25]: title "Plan: part 1 by 15 Nov, then video and writing"; one table
  weeks | dates | what happens | for with four rows (weeks 1-2, 3, 4, 5-8); two bullets (already done; dropped) and a
  "What this means" line. The full week-by-week text is in the notes ("If asked").
- **Week 1 (19-25 Oct; 1a shape, 1b):** NAVER baseline: done 18 Oct (job 3614192, slide 24); the partial-mask run (job 3614042) and its
  Relative Human run (done 19 Oct, not adopted; slide 19); the student's result (job 3614191): "done 19 Oct, failed the
  bar (119.6 / 110.1 mm vs 77.6 / 93.7; slide 24)"; code: a masked-evaluation flag and the student's shape-distillation mask;
  phenotype statistics on AnnyOne (r, slope, R^2) [D slide 25; P 4 week 1; H 0a].
- **Week 2 (26 Oct-1 Nov; 1a, 3):** "Done 19 Oct: rest-pose line-up and predicted-finger demos (slide 31); old ViT-S
  students on the new labels (121.6 / 142.6 mm)"; still to do: a body-joint MPJPE; write the Method and Data chapters
  [D slide 25; P 4 week 2]. The ViT-S numbers are 3DPW test PVE on the v23 labels, Student-S/final 121.6 and
  Student-S/6c 142.6 mm, against Final 89.0 [Certain, job 3615020; C "18 Oct, 22:40"].
- **Week 3 (2-8 Nov; 1b, 3):** "The student missed the bar; its AnnyOne curve still fell: train it longer from its
  last checkpoint, if you agree (Q1)"; the 95% table against our ViT-L and NAVER's; paired bootstrap confidence intervals; write the
  Experiments chapter (depth, neck, shape mask, Relative Human) [D slide 25; P 4 week 3].
- **Week 4 (9-15 Nov; 1b, 3):** one final table, one protocol, every model; one more seed of the final ViT-L if a slot
  is free; architecture figure and module table; freeze the numbers; start the model release with checksums
  [D slide 25; P 4 week 4].
- **After:** weeks 5-6 (16-29 Nov) the temporal refiner (2b); weeks 7-8 (30 Nov-13 Dec) introduction, related work,
  conclusion, limitations, revision, code clean-up and the presentation, week 8 as buffer [D slide 25; P 4].
- **Not on the slide:** Monday evening, write down the supervisor's answers (above all question 1, which decides what
  happens to the student); jobs 3614042 (partial mask), 3614191 (student), 3614192 and 3614947 (NAVER) are all done.
- **Not claimed:** a buffer before 15 Nov. Tier (1) closes around 15 Nov with no slack [Likely, P 3.6]. Parts of the
  plan file still say the student launches Tue 20 Oct with 100 epochs; it was in fact launched on 18 Oct with
  30 epochs and finished on 19 Oct, as the deck says.

### Slide 26. What I claim, what I do not, and my questions

- **On the slide** (dark closing slide; simplified 20 Oct): title "The fixes are measured; four questions decide what
  is next"; three bullets (claimed, measured: the bugs and label defects fixed; claimed: Final's 3DPW test PVE 89.0
  mm, Base 121.8; not claimed: seeds, children, near/far depth, a non-Multi-HMR model), a "What this means" line and a
  table of the four questions. Relative Human recall 47 -> 65% is in the notes, not on the slide; the notes also
  answer last week's
  question: labels first, then the v5 fine-tune, which is the final model [D slide 26, D notes 26].
- **Claim (measured) [Certain]:** the failures are not rig or decoding problems; a training-target bug put people too
  far away in v5, v9 and v10, and the fix draws them much closer to their true size; neck02, the head bone and two
  hand bones were half a turn off, and a fixed correction turns them back; the 3DPW labels had three defects, now
  fixed and checked; the final model keeps every fix against its own control (root error -48%, neck02 4.8 deg,
  fingers 8.3 deg, PA-PVE 50.3 vs 56.4 mm) and is the best on our 3DPW-Anny labels (89.0 vs v9 95.5, v5 121.8); on
  Relative Human it is better than v5 on every line (gender 46.5 -> 64.2%, AUC 0.61 -> 0.72, recall 47 -> 65%).
- **Not claimed:** results proven over seeds; why the bones flipped; children or gender solved; correct near/far depth
  (slope 0.42); fixes without cost (recall -2 points vs control and -3.2 vs v5, in the notes); that the fixes
  made most of the 3DPW gain (3DPW training did); "a model that is not Multi-HMR (slides 23-24)" (requirement 1b; the
  notes add that the ViT-B student misses the 95% bar against our model). The notes' fingers answer: by eye on real
  photos Final's fingers read as hands and Base's are splayed (slide 31), but only the 8.3 deg is claimed. Not in the deck's
  list but true: visibly varied body build on real photos is not claimed either (muscle, weight near 0.5; slide 19).
- **Questions for the supervisor** (D slide 26, P 5):
  1. (1b) Is the 95% bar against our ViT-L, NAVER's, or both? Does a smaller Multi-HMR-style student count as "not
     exactly Multi-HMR"? (Student: 119.6 / 110.1 mm; bars vs ours 77.6 / 93.7, vs NAVER's 63.5 / 133.6). The NAVER /
     ours numbers (60.3 likely in-sample / 126.9 vs 73.7 / 89.0) moved from the slide to the notes. Reading: against
     our ViT-L the student misses both bars; against NAVER it meets 3DPW (and beats NAVER outright there, with 3.76x
     fewer params) and misses AnnyOne [Certain, C SD; job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a].
  2. (2b) Does a temporal refiner on frozen per-frame features count as "understands motion"?
  3. Children: a stated limitation, or weak age labels from Relative Human (1-2 weeks of code)?
  4. When is the submission date?
  - Possible extra questions from P 5, not on the slide: is our last-100 AnnyOne tail "the AnnyOne test set", or is
    there an official split? Is a body-joint MPJPE acceptable on 3DPW?

### Slide 27. Still open, and what I test next

- **On the slide** (appendix) [D slide 27]: seven numbered items: (1) bone flips, fix confirmed at 10 epochs (neck02
  4.8 deg; real images by eye only; v10's left arm needs a two-sided fix, not built); (2) depth slope 0.42, cause
  unknown, next: measure training depths and the slope on training images; (3) depth-fix cost, now measured (final vs
  control: PVE within noise, recall -2 points, -3 vs v5; paired PVE not done; no x20 result at 10 epochs); (4) pilot
  noise, no seed; (5) people very close to the camera (4 of 12 holdout renders, both models); (6) children and fingers
  (AUC 0.61 -> 0.72, not solved; fingers: "by eye on slide 31; then larger crops"); (7) 3DPW labels, three defects fixed
  (re-extraction 16 Oct, re-fit and re-check 18 Oct, WARN, in use by the final model).
Background to these items, and what else is open:
- **Depth slope 0.42** (1 = right): predictions are pulled toward about 2.5-2.9 m; cause unknown; measuring the
  training depths properly needs new code [Certain, H 3.1, H 5 item 8].
- **Recall cost:** -2 points vs the control, -3.2 vs v5; a PVE comparison paired on the same people and a recall on
  people with a visible head still need code; [Likely, untested] the drop comes from moving the detection square to
  the head [H 5 item 9]; this cause is not on the slide, which gives only the measured cost (item 3).
- **No seed:** `train.py` sets none; one run per arm [Certain, H 5 item 10].
- **People very close to the camera:** both v5 and the depth-fix model miss them in 4 of 12 rendered holdout images;
  likely detection, not counted [Likely, H 3.1].
- **Children:** better, not solved (slide 18).
- **Hands on real photos:** shown with predicted fingers since 19 Oct, by eye only (slide 31); no metric.
- **Why the bones flipped:** unknown; v10's left arm needs a two-sided correction, not built [Certain, H 3.2].
- **The x20 depth run for 10 epochs:** command given on 14 Oct, never confirmed [Certain, H 3.1, H 4d].
- **3DPW precision 64% and ghost bodies** (not on this slide; slide 20 notes): about a third of the detections match
  no label; the split between bystanders and ghosts is unknown [Certain, P 3.1 item 4].
- **NAVER baseline** (not on this slide; slides 20, 23, 24, 26, 30): done, job 3614192: 60.3 mm AnnyOne holdout
  (likely in-sample for NAVER), 126.9 mm 3DPW test [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a];
  cost, job 3614947: 380.9 M params (head 76.6 M) vs our 318.4 M; head FLOPs and ms not measured for its class.

### Slide 28. Corrections to the draft and the repo

- **On the slide** (appendix) [D slide 28]: four boxes. Headline number: done, the README gives AnnyOne holdout PVE
  70.5 mm, helper bones masked; done, "the thesis draft uses 70.5 / 73.7 mm" (the old draft's 74 mm was the unmasked
  evaluation, +3.8 mm on the same people; the card's dot is now green). Demos: done, all demo scripts zero the fingers and the README says so. Wording: done, neck, fingers and
  children are no longer called rig or decoding problems, and the README's Known-defects section documents the depth
  bug; to do, the README and the findings document still carry last week's neck story. Repository: done,
  reorganised, one shared cluster setup, diagnostics and experiments documented. The notes add that the demo setting
  gives 68.5 mm at a lower recall.

The full list of corrections the thesis draft needs is longer than the slide. The draft is not in the repository;
these lines come from H 1 "Headline", H 6 and H 7, and apart from the headline number they are not on the slide
[Certain that each old claim is wrong, as stated there]:
- **Headline number (done 19 Oct: the full draft in `docs/thesis/` uses 70.5 / 73.7 mm):** the old draft said 74 mm. That is the unmasked V0 evaluation of v5. Use 70.5 mm (V1, helper bones
  masked; +3.8 mm on the same people when unmasked). The thesis model is now the final model: 73.7 mm holdout V1,
  89.0 mm 3DPW test (v23) [H 1; P 2 row 3-report].
- **Neck:** not "weak supervision on a redundant chain", not "the depth bug": two half-turned bones (slide 15).
- **Fingers:** not only "a resolution problem": about half the error was two flipped bones (slide 17).
- **Depth:** add the anchor/target mismatch; quote the absolute root error (1.21 -> 0.54 m in the pilot, 1402 -> 728
  mm vs the control in the final model), never the signed mean depth offset (1.08 -> 0.39 m), where near and far
  errors cancel [H 6].
- **Size ratio of v5:** 0.63, not 0.67 [H 6].
- **3DPW numbers:** the old-label numbers (v9 102.5, v5 119.1, v10 111.5) are retired; use only v23 numbers [H 0b].
- **Children:** the "five small people" were mostly crouching or seated adults, not children; use the Relative Human
  numbers; never convert Anny ages to years [H 6].
- **3DPW labels:** describe the three defects and the fix; state the limits (head at rest beyond neck01, 8 constant
  phenotypes, finger rest pose, 163-joint MPJPE, ~14 mm conversion floor) [P 3.1 item 4].
- **Known issue:** every fine-tune skips a few steps for NaN/Inf gradients (1-7 per 2,000; 16 of 10,000 in the final
  model); gradient clipping absorbs it [H 1].

### Slide 29. Backup: before and after, holdout image 579860

- **On the slide** (appendix): ground truth | v5 | fix (x5, 10 epochs) on holdout image 579860, with the model's own
  depth.
- **What it shows:** before the fix the boy's real head sticks out above a shrunken, floating mesh; after the fix the
  top of the head is level with the truth [Certain, D notes 29].
- **Not claimed:** an exact fit: here the fixed mesh is a little larger and wider than the truth, and it is cut off at
  the image edge, so its full height cannot be measured. The two people in front have no ground-truth mesh, so they
  do not count as misses [Certain, D slide 29].

### Slide 30. Jobs and checkpoints

**On the slide** (appendix) [D slide 30]: one table, job | what | path under `/netscratch/najib/multi-hmr`, with 13
rows (font 8): v5 epoch 99; 3611336 final model (3DPW test in its log); 3611339 control; 3612325 masked-shape arm; 3614042
partial mask, age kept (done; not adopted); 3614191 "ViT-B student: done, 119.6 / 110.1 mm (smoke 3613926)" (`logs/exp8_distill_vitb_3614191.out`); 3614192 /
3614947 "NAVER: 60.3 / 126.9 mm; 380.9 M vs our 318.4 M" (`logs/{eval-test3dpw-3614192_0,model_cost_3614947}.out`;
`models/multiHMR/multiHMR_672_L_anny.pt`; the cost job is folded into the NAVER row because a 14th row did not fit;
its json `results/diagnostics/model_cost_naver_vs_final.json` is named in the notes only); 3611285 Relative Human v5 (_0) and neck pilot (_1); 3612131 / 3613616
Relative Human final / masked; 3612078 / 3613617 demos; 3612073-75 3DPW test of v5, v9 e62 and the neck pilot;
3611284 v23 label check; 3579014 exp6 bone fix. Checkpoints are not in git (release: week 4). The lists below are
longer than the slide: the v9 checkpoint, the student's run folder, the model copy `exp7_v5_final_00009.pt` and every
job before 3611284 except 3579014 are not on it.

**Checkpoints** (relative to `/netscratch/najib/multi-hmr`; H 1, H 0a, exp7 and exp8 scripts):
- v5 epoch 99: `logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt`
- v9 epoch 62: `models/multiHMR/anny_teacher_v9_mixed_00062.pt`
- final model: `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt`
  (copy `models/multiHMR/exp7_v5_final_00009.pt`)
- control: `results/experiments/exp7_final_model/v5_root_dist1_nofix_v23_e10/`
- masked arm: `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask/`
- partial mask (done 19 Oct; not adopted): `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask_noage/train/checkpoints/00009.pt`
- student (done 19 Oct; failed the 95% bar): `results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/`
  (`train/checkpoints` keeps the last 3 epochs, NB_MAX_CKPT 3, so the last is `00029.pt`)
- bone fix used: `results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/bone_fix.json`
- 3DPW labels: `threedpw_anny_v23/labels/`

**Jobs** (all [Certain], H 0a, H 0b, H 3.x, H 8):

| job | what |
|---|---|
| 3570222 | v5 holdout trace (PVE 70.5) |
| 3571598 / 3571599, 3572418, 3572483 | exp1 2-epoch depth pilots (fix x1 / control, x5, x20) |
| 3571933 / 3571934, 3572482 | exp1 10-epoch depth runs (fix x1 / control, fix x5 `long_pc_dist5`) |
| 3571955 / 3571956 | exp2 children sampler / control |
| 3571957; 3571958, 3571960 | exp3 global-head loss; the two identical controls (pilot noise 7.6 mm) |
| 3571959 | exp4 region-normalised finger loss |
| 3573518 / 3573519 | per-bone neck dumps |
| 3573520, 3573521 | exp5 rotation-loss pilots (geodesic loss and its L1 control) |
| 3579014 / 3579015 | exp6 bone fix on v5 / on v10 |
| 3575369 | 3DPW conversion check, old labels (FAIL; the three defects) |
| 3579266, 3581320 | 3DPW label rebuilds (the second one after the image re-extraction) |
| 3581312 | real-image demo, v5 vs the 2-epoch fixed model |
| 3589270; 3590385 + 3602813 | 3DPW re-fit test; full re-fit and its rerun |
| 3611284 | 3DPW conversion check, v23 labels (WARN) |
| 3611285_0 / _1 | Relative Human, v5 / exp6 |
| 3611333; 3611336; 3611339 | exp7 smoke test; final model; control |
| 3612073 / 3612074 / 3612075 | 3DPW test on v23: v5 / v9 e62 / exp6 |
| 3612078 | demo, v5 vs final model |
| 3612131 | Relative Human, final model |
| 3612325; 3613616; 3613617 | masked arm; its Relative Human run; its demo |
| 3614042; 3616821 | partial mask (7 constants, keep age), done 19 Oct; its Relative Human run (`logs/eval_rh_3616821_0.out`): gender 47.4%, AUC 0.727, not adopted |
| 3613926; 3614191 | exp8 student smoke test (passed); full student run, done 19 Oct: holdout PVE 119.6 (bar 77.6), 3DPW test 110.1 (bar 93.7), 95% rule fails; L40S 101.38 M / 603.3 GFLOPs / 28.5 ms vs teacher 318.38 M / 1,927.4 / 59.7 |
| 3615018; 3615019 | demos: predicted fingers, Base vs Final; rest-pose line-up, Base / Final / Final-masked (slide 31) |
| 3615020_[2,4] | old ViT-S students on the v23 3DPW test: Student-S/6c 142.6, Student-S/final 121.6 mm PVE |
| 3613928; 3614192 | NAVER baseline (crashed, fixed); resubmitted, done 18 Oct evening: AnnyOne holdout PVE 60.3 (likely in-sample for NAVER), 3DPW test 126.9 |

### Slide 31. Real photos: predicted fingers and rest-pose line-up

- **On the slide** (appendix, added 19 Oct) [D slide 31]: on the left, predicted fingers: zoomed crops of the render
  panels of two demo montages, the party photo (top row) and the 3DPW guitar frame (bottom row), Base | Final
  (`docs/figures/fingers_base_vs_final.jpg`, cut from `docs/thesis/figures/img/F11_hands_photo_party.jpg` and
  `F11_hands_3dpw_guitar.jpg`). On the right, the rest-pose line-up: the three render panels Base | Final |
  Final-masked of the stage photo (three singers) and the tennis photo (five men) (`docs/figures/rest_pose_lineup_stage.jpg`
  and `rest_pose_lineup_tennis.jpg`, cut from `F17_rest_photo_stage.jpg` and `F17_rest_photo_tennis.jpg`). Below, two
  lines with the by-eye readings, ending "By eye only, demo setting (box placement); no metric on real photos."
- **What we did:** two demo jobs on the evening of 18 Oct: 3615018 runs the demo comparison with predicted fingers
  (`--relax_hands 0`) for Base (v5) and Final; 3615019 draws every detected person in the rest pose (`--rest_pose 1`)
  for Base, Final and Final-masked, so that only body shape differs [Certain, C "Zip batch 2"].
- **Why:** both were missing for requirement 1a: every earlier real-photo demo used `--relax_hands 1`, so predicted
  hands had never been shown, and no figure showed body shape changing per person (P 2 row 1a; plan week 2).
- **Result, fingers (by eye):** Base's fingers are splayed and stiff, with one finger of the left hand sticking out
  (Base's metacarpal1.L is still flipped; the correction is only in Final); Final's are closed and bent and read as
  hands (draped over an armrest, curled as if fretting a guitar, bent over a bar edge). Neither model puts the fingers
  on the real ones; in most images the hands are too small to judge [Certain by eye; `docs/thesis/REVIEW_LOG.md`
  "19 Oct: batch-2 figures placed"; C "Zip batch 2"].
- **Result, shape (by eye):** Base gives everyone a similar slim, flat body (the singers look boyish); Final gives the
  singers a fuller female shape and the man in the 3DPW bar frame a broader torso; the tennis men stay slim with a
  slight chest in all three; Final-masked is close to Final except a slimmer bar man. The bar frame is not on the
  slide (it is in the thesis, Figure 6.10, and in the notes) [Certain by eye; REVIEW_LOG 19 Oct].
- **Crops** (pixel boxes x0, y0, x1, y1 in the source files; each montage has 4 panels of equal width, or 6 for F17,
  and a label strip at the top): party Base (800, 200, 1060, 460), Final (2000, 200, 2260, 460); guitar Base
  (626, 320, 886, 580), Final (1638, 320, 1898, 580); stage panels x 436-776, 1236-1576, 2036-2376, y 12-280; tennis
  full panels x 400-800, 1200-1600, 2000-2400, y 12-280 (comment above slide 31 in `build.js`).
- **Not claimed:** any finger or shape number on real photos (there is no metric); that the predicted fingers match
  the real ones; that four crops and two line-ups stand for all twelve demo images (all are in the thesis, Figures D.6
  and D.7); anything beyond the demo setting (bodies placed from the detector box).

### Slide 32. 3DPW labels on the image: a good case and the worst case

- **On the slide** (appendix, added 19 Oct) [D slide 32]: two label-check panels side by side, each cropped to the
  3DPW SMPL mesh (orange, OpenPose 2D yellow) and our Anny label (blue; joints red = 3DPW SMPL, green = Anny):
  left `docs/figures/label_check_courtyard.jpg` (training sequence courtyard_arguing_00, image 24, two people, joint
  errors 12 and 14 mm), right `docs/figures/label_check_climbing.jpg` (outdoors_climbing_02, image 633, the worst joint
  error of the check, 57 mm). Both are cut from `docs/thesis/figures/img/F14_label_check_*.jpg` with box
  (545, 170, 1628, 830), dropping the photo panel and the label strip.
- **What it shows:** body joints of our labels match 3DPW's to 14.1 mm (median over 3,314 people), below the ~26 mm
  accuracy of 3DPW's own joints; the worst frame, picked on purpose, is 57 mm because the Anny body is thinner than the
  climber [Certain, conversion check job 3611284; slides 5-6].
- **Not claimed:** that the labels are perfect: the check grades WARN (forearm ratio 0.95, thigh 1.06; 3.2% of people
  have a joint over 100 mm). Head and hands stay at rest in every label, so these labels do not supervise them.

---

## Part 7. Thesis requirements: status and evidence

Status words as on slide 23. Sources: P 2, P 3, H 0a, H 11.

### 1a. Correct pose and visibly person-specific shape, on the 3DPW test and on any real photo

**Status: partly (pose); partly, at risk (shape).**
- **Evidence, pose [Certain]:** 3DPW test (v23, 8,190 images): PVE 89.0, PA-PVE 57.6, MPJPE 88.3, recall 99%,
  precision 64% (3611336:224-229). Neck flip fixed (neck02 4.8 deg) and finger flips fixed (8.3 deg) on the holdout
  (3611336:191-209). On real photos, by eye only: necks normal; placement offsets, ghost bodies and hard poses remain
  (demos 3581312, 3612078) [Likely / Certain by eye, H 3.2, H 3.4].
- **Evidence, shape [Certain]:** gender changes per person (Relative Human 64.2%, job 3612131); age partly (babies 0.44
  vs adults 0.54); muscle, weight, cup size and firmness sit at about 0.5 on real photos (demo 3612078). The masked arm
  restores their spread (std up to 0.11) but loses gender (49.3%) and most of the children gain (job 3613616, demo
  3613617) [H 0a].
- **Done 19 Oct, by eye only (slide 31):** a demo with predicted fingers and a rest-pose line-up (jobs 3615018,
  3615019).
- **Missing:** numbers that show
  each phenotype follows its label on AnnyOne (correlation, slope); the depth slope (0.42) as a stated limitation
  [P 2, P 3.3].
- **Planned:** partial-mask run (job 3614042) with its Relative Human run and demo, week 1; done 19 Oct, not adopted;
  lineup and finger demo: done 19 Oct (slide 31) [P 4; H 0a].

### 1b. A model that is not exactly Multi-HMR

**Status: not met.**
- **Evidence [Certain]:** `model.py` is the upstream Multi-HMR with the output resized for Anny and a 0-parameter bone
  fix (P 2).
- **Evidence, the student** [Certain, C SD; job 3614191, done 19 Oct morning]: cost on one L40S 101.38 M params /
  603.3 GFLOPs / 28.5 ms vs the teacher's 318.38 M / 1,927.4 / 59.7 ms (3.14x / 3.19x / 2.10x). Accuracy: AnnyOne
  holdout PVE 119.6 vs bar 77.6 (teacher 73.7), 3DPW test PVE 110.1 vs bar 93.7 (teacher 89.0); MPJPE 127.2 vs 82.0
  and 112.0 vs 92.9: **the 95% rule fails on all four bars**. Recall kept (85.4% vs 87.0; 99 vs 99). The AnnyOne curve
  was still falling (130.5 -> 122.8 over the last 5 epochs), the 3DPW-val curve flat (88.0 -> 87.7). Against NAVER's
  Multi-HMR-Anny: better on 3DPW (110.1 vs 126.9) with 3.76x fewer params; against NAVER's bars (63.5 / 133.6) 3DPW is
  met, AnnyOne is not (NAVER's 60.3 likely in-sample).
- **Evidence, the baseline's size** [Certain, job 3614947, `logs/model_cost_3614947.out`, `results/diagnostics/model_cost_naver_vs_final.json`; H 0a]: NAVER 380.94 M params (backbone 304.37 + head
  76.57), backbone 1,917.2 GFLOPs (head FLOPs and ms not counted for its class); our final model 318.38 M (head
  14.01), 1,927.4 GFLOPs, 77.9 ms on an A100. Ours is already 16% smaller than NAVER's, which does not meet 1b by itself
  ("significantly smaller"); the student is 3.76x smaller than NAVER's model.
- **Evidence, the baseline** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]: NAVER's own Anny model on our labels and settings: AnnyOne holdout PVE
  60.3 / PA-PVE 44.6 / recall 89%; 3DPW test PVE 126.9 / PA-PVE 68.8 / recall 97%; ours 73.7 / 50.3 / 87% and 89.0 /
  57.6 / 99%. [Likely] in-sample for NAVER on AnnyOne (trained on all of it). Consequences: against NAVER the 95% bar
  would be 63.5 mm on AnnyOne (no student of a 73.7 mm teacher can reach it) and 133.6 mm on 3DPW (trivially met);
  against our ViT-L 77.6 / 93.7. Our final model already beats NAVER on 3DPW but not on AnnyOne, so the
  new-architecture route (1b-i) would hinge on AnnyOne.
- **Two readings** [Certain for the numbers]: against our ViT-L (the reference the plan proposed, P 3.2) 1b-ii is
  not met on either set; against NAVER's model, the baseline the e-mail names, the student is better on real 3DPW
  with a quarter of the parameters, and misses only the AnnyOne bar, where NAVER's number is likely in-sample.
- **Missing:** the supervisor's choice of reference (question 1, slide 26) and his yes that a smaller student counts;
  NAVER's head FLOPs and ms (not measured: `model_cost.py` counts and times our class only) [P 3.2].
- **Next, options already in the plan** (P 3.2 fallback, P 3.6): train the same ViT-B longer from its last checkpoint
  (epoch 29), since AnnyOne was still falling, and report the ratio as measured; or stop the smaller-model route and
  argue 1b through the comparison with NAVER's baseline. Final table by 15 Nov [P 3.6, P 4]. The plan's guess before
  the run (3DPW likely, AnnyOne about 50/50) was wrong on 3DPW.

### 1c. Evaluation in Anny space on real 3DPW

**Status: met, with caveats.**
- **Evidence [Certain]:** v23 test labels, 8,190 images / 11,854 people; re-fit check PASS on 60 sequences; independent
  check WARN (old labels FAIL), body joints median 14.1 mm (job 3611284) [H 3.5].
- **Caveats to state:** MPJPE averages all 163 Anny joints, pelvis-centred, so it does not compare with published 3DPW
  numbers; 8 phenotypes are constants, so 3DPW tests pose plus gender, height and proportions, not full shape; hands,
  neck beyond neck01 and head are at rest in the ground truth; about 14 mm conversion floor [Certain, P 3.1 item 4].
- **Planned:** a body-joint MPJPE (code to write, week 2) [P 4].

### 2a. 3DPW train and test in Anny, and used in training

**Status: met.**
- **Evidence [Certain]:** train 5,882 images / 7,584 people, validation 2,622 / 3,311 (H 0b); the final model trains on
  30% 3DPW-train in every batch (exp7 :225, :768) and this gives most of its 3DPW gain (control 91.5 vs v5 121.8)
  [H 5 item 7].
- **Missing:** nothing required; a 50% mix ablation is optional [P 2].

### 2b. Temporal information (motion in video)

**Status: not started.**
- **Evidence [Certain]:** no temporal code in the repository; the model works frame by frame (P 2).
- **Planned (weeks 5-6, 16-29 Nov):** a small refiner on top of the frozen per-frame model: a 2-layer transformer over
  16-frame windows of the per-person tokens, shape pooled over each track, trained on 3DPW-train tracks, judged by
  acceleration error and PVE against simple smoothing baselines; if it does not beat smoothing, say so [P 4 weeks
  5-6]. About 5-7 code days and 10 GPU hours [Guessing, P 4].

### 3. Report and code

**Status: partly.**
- **Evidence [Certain]:** a full thesis draft of 92 pages (main text 60) in `docs/thesis/`, all results in, built
  clean, using 70.5 / 73.7 mm; 98 tracked files
  hard-code `/netscratch`; no results or checkpoints committed; 79 of 89 commits are authored "Claude" (P 2).
- **Missing:** chapter plan with current numbers; a model release with checksums; one reproduce script; clean paths and
  environment file; README rewrite [P 6.2].
- **Planned:** one chapter per week from week 1 (Method and Data in weeks 1-2, Experiments in week 3, Distillation in
  week 4); release items in weeks 4-5; clean-up in weeks 7-8. Write in the student's own voice, no confidence tags or
  job ids in the thesis; follow the university's AI-use rules [P 6.1].

---

## Part 8. Questions the supervisor may ask, and short answers

**Q1. Why is PVE not better after the fixes?**
PVE first moves both bodies to the same pelvis, so it cannot see the depth fix, which is the biggest change [Certain,
H 1]. The neck and finger fixes move few mesh points, so they barely change an average over the whole body [Likely,
H 5 item 7]. On the holdout the final model is 3.7 mm better than its control (77.4 -> 73.7), which is inside the 7.6
mm noise [Certain]. v5's 70.5 is still lower, because every fine-tune pays a restart cost of about 2.4 mm, and v5 never
had to learn real images [Certain for the restart cost, H 3.6; Likely for the rest]. PA-PVE does improve beyond noise
(56.4 -> 50.3), and on 3DPW the final model is the best (89.0) [Certain].

**Q2. Is it still Multi-HMR?**
Yes. The architecture is the upstream Multi-HMR with its output resized for Anny and a bone correction that adds no
parameters [Certain, P 2]. That is why requirement 1b is not met. Our route was a smaller ViT-B student (3.14x fewer
params, 2.10x faster on one L40S); it misses the 95% bar against our model (Q3a), and whether that route counts at all
is your decision [Certain for the numbers, C SD]. It is also not NAVER's
checkpoint under another name: NAVER's own `multiHMR_672_L_anny`, run with our code on the same labels, gets 60.3 mm on
the AnnyOne holdout (likely in-sample for it) and 126.9 on the 3DPW test, against our 73.7 and 89.0 [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]. It is also bigger: 380.9 M
parameters against our 318.4 M, all of it in its 8-layer decoder head (76.6 M vs our 14.0 M); being 16% smaller is not
"significantly smaller", so the student is still needed [Certain, job 3614947, H 0a].

**Q3. What does the 95% rule mean in practice?**
For error measures, lower is better, so we read it as: the student's PVE and MPJPE must be at most the teacher's
divided by 0.95, on both the AnnyOne holdout and the 3DPW test. With the final model's numbers that is about 77.6 mm
holdout PVE and 93.7 mm 3DPW PVE; the deck uses this /0.95 form only. It must also have fewer
params and FLOPs and be faster [Certain, exp8 header, P 3.2]. Open: is the larger model our ViT-L or NAVER's? We will
report both. **Which baseline, in numbers:** NAVER gets 60.3 mm on AnnyOne (likely in-sample: trained on all of AnnyOne)
and 126.9 on 3DPW, so its bars would be 63.5 mm (out of reach for any student of our 73.7 mm teacher) and 133.6 mm
(already met); against our ViT-L 77.6 / 93.7. The AnnyOne gap to NAVER is 13.4 mm [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]. NAVER's size is
measured too: 380.9 M params against our 318.4 M (same backbone; head 76.6 M vs 14.0 M), so the student has 3.76x fewer
params than NAVER's model; NAVER's head FLOPs and ms are not measured, because the cost script supports only our class
[Certain, job 3614947, H 0a].

**Q3a. Did the student pass?**
No, not against our ViT-L. The ViT-B student (job 3614191, done 19 Oct morning) has 119.6 mm PVE on the AnnyOne holdout
against a bar of 77.6 (teacher 73.7) and 110.1 on the 3DPW test against 93.7 (teacher 89.0), so it is 62% and 24% worse
than the teacher; MPJPE misses too (127.2 vs 82.0, 112.0 vs 92.9). It finds almost as many people (recall 85.4% vs 87.0,
99 vs 99). The cost side passes: on one L40S 101.38 M vs 318.38 M params, 603.3 vs 1,927.4 GFLOPs, 28.5 vs 59.7 ms
(3.14x / 3.19x / 2.10x) [Certain, C SD]. The AnnyOne curve was still falling at the end (130.5 -> 122.8 over the last
5 epochs), the 3DPW-val curve flat (88.0 -> 87.7) [Certain]; so a longer run may help AnnyOne [Likely] but probably not
3DPW [Guessing]. **The reference question:** against NAVER's Multi-HMR-Anny, the baseline your e-mail names, the student
is better on 3DPW (110.1 vs 126.9) with 3.76x fewer params, meets NAVER's 3DPW bar (133.6) and misses its AnnyOne bar
(63.5), where NAVER's 60.3 is likely in-sample [Certain for the numbers, C SD; Likely for in-sample]. Which reference
counts is your call (question 1, slide 26); then either the longer run from the last checkpoint (epoch 29) or we stop
the smaller-model route and argue 1b through the baseline comparison [P 3.2].

**Q4. Why do children fail?**
On real photos, v5 gives children an adult age and puts their small size into the height value instead, so they look
like short adults [Certain, 3611285_0]. The shape head pulls toward the training median; AnnyOne has few children (2.1%
of supervised people per batch); and an L1 loss through a sigmoid learns extreme values slowly [Certain for the facts;
the cause cannot be ranked between the synthetic-to-real gap and the loss, H 3.4]. In Anny a child is mainly narrow
shoulders relative to the head, which only a low age gives [Certain, H 3.4]. 3DPW has no children, so the real data adds
no child signal [Certain]. The final model is better (AUC 0.72), [Likely] partly through the 3DPW age label 0.55.

**Q5. Why mask the shape loss for 3DPW people at all?**
Our 3DPW labels set 8 of the 11 shape values to constants (age 0.55, seven values 0.5), because joint positions cannot
tell muscle or weight [Certain, H 3.4]. Training on them pulls real-photo predictions to those constants: in the final
model muscle, weight, cup size and firmness sit at about 0.5 [Certain, demo 3612078]. Masking brings their spread back
(std 0.04 -> 0.06-0.11) but costs gender and the children gain [Certain, H 0a]. So the final model stays unmasked; a
run that masks only the seven 0.5 values and keeps age (job 3614042, done 19 Oct) kept the children but also lost
gender (47.4%), so it was not adopted either (slide 19).

**Q6. What is pilot noise, and how do you deal with it?**
Training sets no random seed, so the same settings run twice gave 7.6 mm different PVE, 1.9 degrees different head
direction and 0.8 recall points [Certain, H 3.6]. So one run can only show differences of about 10 mm, 3 degrees or 2
recall points. We compare every run with its own control, we say "inside noise" for smaller gaps, and we plan bootstrap
confidence intervals and, if a slot is free, a second seed [P 4 week 3-4].

**Q7. Why does the 3DPW ground truth keep a neutral body?**
3DPW only gives SMPL bodies. Our converter matches joint positions, so it can fit only height and proportions; gender
comes from 3DPW's metadata; the other 8 values are fixed [Certain, H 3.5]. The training mesh and the 3DPW test mesh are
built from these values, so the 3DPW numbers test pose plus gender, height and proportions, not body build [Certain,
H 3.4; P 3.1 item 4]. A fit to the SMPL surface could recover more; we ask whether you want that [P 5 item 1].

**Q8. Why is the depth slope 0.42 and not 1?**
We do not know the cause yet [Certain, H 3.1]. The fix moved the offset (intercept 2.66 -> 1.75 m) and the body size
(0.59 -> 0.78), but the slope only from 0.28 to 0.42 against the control. Predictions are pulled toward about 2.5-2.9
m, close to the average training depth, while holdout people stand at about 1.9 m on average [Certain for the numbers,
H 3.1, D notes 14]. To find out, we must measure the training depth distribution properly, which needs new code
[Certain, H 5 item 8]. [Guessing] it may need more trainable backbone blocks or a stronger depth signal. For the thesis
it becomes a limitation with its mechanism [P 4 "What to drop"].

**Q9. Why does recall drop by 2-3 points?**
[Likely, untested] the depth fix moves the detection square from the pelvis to the head, so people whose head is hidden
or cut off are harder to find. Measuring recall only on people with a visible head needs new code [H 5 item 9]. Against
the control it is -2.0 points (at the noise limit); against v5 -3.2: we state it as a real cost [Certain].

**Q10. Are your 3DPW numbers comparable with published Multi-HMR numbers?**
No. They are against our own Anny conversion, our MPJPE averages 163 Anny joints, and the labels have a conversion
floor of about 14 mm [Certain, P 3.1 item 4]. We say "best on our 3DPW-Anny labels", never "best model". NAVER's own
Anny checkpoint on the same labels gets 126.9 mm PVE (ours 89.0) [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a].

**Q11. Why is 3DPW precision only 64%?**
3DPW labels only the one or two actors, not the bystanders, so a correctly detected bystander counts as a false
detection [Likely, H 5 item 7]. Ghost bodies on half-visible people also count; the split between the two is unknown
[Certain, P 3.1 item 4].

**Q12. How can a correction with no parameters fix a bone?**
The network always outputs neck02 and the head about half a turn off, in the same way for everyone. One fixed rotation
per bone, estimated once on 200 training images, turns them back [Certain, H 3.2]. Without training, the head direction
gets worse (19.1 deg), because the head's offset differs a little per person; after fine-tuning with the fix in place it
is fine (13.1 deg in the final model, control 13.0) [Certain, H 3.2].

**Q13. Did you look at why the bones flipped?**
Yes, but the origin is still unknown. The flip is there from the first training stage of the v5 line. The singular
start value of the rotation output would start a bone with a tilt, which fits the left index base but not the neck's
twist. v10, trained from the same backbone with a different recipe, has no neck flip but a left-arm flip [Certain,
H 3.2].

**Q14. Why only one run per experiment?**
The training GPUs are shared and the queue was estimated at days; one 10-epoch run takes about 1-2 hours of training
[Certain, H 4f, H 0a step 6]. We read every run against its own control and the noise floor, and we do not claim gains
smaller than the noise [Certain, H 3.6].

**Q15. Why ViT-B and not ViT-S for the student?**
NAVER's own ViT-B equals its ViT-L on 3DPW (94.0 vs 94.1 PVE) [Certain, P 3.2]. Our earlier ViT-S students had 1.17-1.54
times their teacher's error, far from the 95% bar [Certain, P 2]. ViT-B is the smallest step with a fair chance
[Likely, D notes 23].
