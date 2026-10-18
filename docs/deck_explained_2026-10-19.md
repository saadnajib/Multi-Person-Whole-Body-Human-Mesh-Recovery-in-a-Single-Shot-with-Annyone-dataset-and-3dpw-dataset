# The supervisor deck, explained in plain words (meeting of Monday 19 October 2026)

This document explains the deck for the meeting of Monday 19 October 2026, 16:00, slide by slide, in simple words.
It is written for the presenter, who must also defend the work and later reuse the text in the thesis. Part 6 follows
the final 30-slide deck (`docs/supervisor_update_2026-10-19.pptx`, speaker script
`docs/supervisor_talk_notes_2026-10-19.md`): each block starts with the slide number and the exact slide title, so the
two can be matched. This file was first written against the planned outline and reconciled with the final deck on the
evening of 18 October.

**State of the facts:** Sunday 18 October 2026, evening. Nothing here was measured again for this document. Every
number comes from a file in this repository, and the source is given next to it.

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
- `D slide N` / `D notes N` = the final 30-slide deck: the text on slide N of `docs/supervisor_update_2026-10-19.pptx`
  (built by `docs/deck_src/build.js`) / the speaker notes of slide N in `docs/supervisor_talk_notes_2026-10-19.md`.
  All slide numbers in this file are the new ones; `docs/deck_src/README.md` has the old -> new map.
- `3611336:191` = line 191 of the log of cluster job 3611336. The logs are on the cluster, not in the repo; these line
  numbers are quoted from the handover, which was checked against the logs.
- `loss.py:343` = that line of the file in this repository today.

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
    yet; a smaller student model is training for it (job 3614191, 30 epochs, submitted 18 Oct with a 20-hour limit; the result comes a day or two after the
    queue starts it) [Certain, H 0a].

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
  - Next: mask only the seven 0.5 values and keep age (job 3614042, queued; result after the meeting) [Certain, H 0a].
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

- **On the slide** (dark title slide): the title, five one-line results and the byline "Saad Najib, supervisor update,
  19 October 2026" [D slide 1]:
  1. a hidden depth bug: bodies were drawn at 0.63 of their true size; now 0.78;
  2. neck and finger bones were half a turn off; a 0-parameter fix turns them back;
  3. 3DPW converted to Anny, three label defects fixed, and 30% of every training batch;
  4. final model: best on 3DPW test, 89.0 mm (v5 121.8); real photos: gender 46 -> 64%;
  5. not yet: a model that is not Multi-HMR (ViT-B student running) and temporal modelling.
- **What we did / why:** one line per result, so the supervisor knows the story before the details. The notes add that
  the depth bug is fixed for size, not yet for near-versus-far depth [D notes 1].
- **Numbers:** size 0.63 -> 0.78 (H 3.1, final model 3611336), 3DPW test PVE 121.8 -> 89.0 mm (H 0b), Relative Human
  gender 46.5 -> 64.2% (3612131) [Certain].
- **Not claimed:** that anything is "solved".

### Slide 2. The goal and your requirements

- **On the slide** [D slide 2]: on the left, the task in one sentence (from one photo, find every person and predict
  their Anny pose, 11 shape values and distance), a four-step strip (photo, 672 px -> find people on a 48 x 48 grid ->
  predict pose, shape, depth per person -> Anny body mesh), the four test sets (AnnyOne holdout, 3DPW test, Relative
  Human, demo photos; slide 4) and the status-dot legend. On the right, the six requirements of the e-mail of
  18 October, each with a status dot: **1a** correct pose, visibly varying body shape (partly); **1b** a model that is
  not exactly Multi-HMR (not met; a ViT-B student is running, slide 24); **1c** 3DPW evaluated in Anny space (met);
  **2a** train on 3DPW converted to Anny (met); **2b** temporal modelling (not started; designed, weeks 5-6); **3**
  well-written report, clean and verifiable code (partly).
- **Background, not on the slide** (P 1): the e-mail has three tiers. (1) necessary within one month: 1a, 1b and 1c;
  1b is either a new architecture that beats Multi-HMR on the AnnyOne and 3DPW tests, or a much smaller model (fewer
  params and FLOPs, faster) whose PVE/MPJPE stay "more than 95%" of the large model's on both. (2) recommended: 2a and
  2b. (3) best grade: a report in the student's own voice, clean code, models he can check.
- **What we did:** read the e-mail and matched each item to evidence (slide 23 shows the evidence for each dot).
- **Why:** the supervisor said the thesis is not ready without tier (1) [Certain, P 1].
- **Result:** 1c and 2a met; 1a and 3 partly; 1b not met; 2b not started (slides 2 and 23).
- **Not claimed:** a submission date; it is still unknown, and we ask on slide 26 [Certain, H 10].

### Slide 3. The model in one picture

- **On the slide** [D slide 3]: five boxes, left to right: photo (672 x 672 px) -> DINOv2 ViT-L backbone (14-px
  patches, a 48 x 48 grid, 304 M parameters) -> detection (per cell: is a person's head here? plus a 2D offset) ->
  per-person cross-attention head HPH (163 bone rotations, 75 trained and 88 helper bones fixed; 11 shape values;
  distance) -> Anny body (mesh at the predicted distance). Teal tags mark what is ours: "head anchor fix" under the
  detection, "Anny output, bone fix" under the head, "3DPW -> Anny labels" under the body. A teal box lists the four
  changes (output layer sized for Anny; head-anchor fix in the training code, slide 12; the 0-parameter bone fix on
  neck02, head and two hand bones, slide 15; the 3DPW conversion and the 30% recipe, slides 5-6 and 8). Bottom line
  "Honest for 1b": still Multi-HMR's architecture, backbone weights from NAVER's Anny checkpoint; the route to 1b is a
  much smaller model (slide 24).
- **What it is:** the upstream Multi-HMR architecture with its output resized for Anny (163 x 6 rotation values and
  11 phenotypes) [Certain, P 2 row 1b; H 1 "Model"]. The bone fix is a fixed rotation, not a new layer [Certain,
  H 3.2].
- **Why show it:** the supervisor must see that our network is still Multi-HMR. That is why requirement 1b is open.
- **Numbers:** on the slide only the backbone's 304 M parameters. The whole model is 318.38 M parameters,
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

- **On the slide:** three columns: "Measured: pass", "Defects: fixed", "Limits and grades"; the source line names job
  3611284 on the re-fit labels (old labels: 3575369), 3,314 people, 2,360 images, 59 sequences [D slide 6].
- **What we did:** an independent conversion check (15 Oct, job 3575369) found three defects. We re-extracted the
  missing images (16 Oct), added two switches to the converter (gender convention, shared frame grid), re-fitted all
  87 people (jobs 3590385 + 3602813), built the v23 labels and ran the same check again (job 3611284).
- **Why:** 3DPW is 30% of each batch. With reversed gender, every man would keep learning a female body
  [Certain, D notes 26 "Option 2"].
- **Result:** the three defects are gone and the overall grade moved from FAIL to WARN.
- **Numbers (H 3.5, H 0b):** on the slide: body joints median 14.1 mm (was 18.7), p95 21.1; surface 18.9 / 16.1 mm
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

- **On the slide:** three cards: neck (too long or twisted, thin), fingers (claw-like), children (adult bodies).
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

- **On the slide:** two stick figures: the model's anchor at the head; the training target at the pelvis; the line
  "predicted depth = 0.33 x true + 2.36 m" (v5, all 222 holdout people too far); three bullets: confirmed (with
  targets at the head the bodies are drawn much closer to the right size; a same-schedule control does not improve,
  10 epochs: size 0.59, root error 1.44 m); size (3D height right, drawn at 0.63, head 64 px off); hidden from
  pelvis-centred PVE and from the box-placed demos, and v9 and v10 have it too [D slide 12].
- **What we found:** `model.py` places each mesh by the head joint and reports the head as its position, but
  `train.py prepare_gt` built the position, distance, detection-square and offset targets at the pelvis [Certain,
  H 3.1]. A one-batch debug print showed predicted offsets of -2 to -19 squares, where the legal range is +/-0.5
  [Certain, 3571599:9-14].
- **Why it stayed hidden:** PVE and PA-PVE align the pelvis first; the real-photo demos place bodies from the detection
  box. Only the synthetic demo and the trace panels use the model's own depth [Certain, H 3.1].
- **Numbers (v5, holdout):** depth slope 0.33, intercept 2.36 m; all 222 people too far; drawn size 0.63; head
  64.5 px off (64 on the slide); 3D body height right (1.405 vs 1.409 m, in the notes) [Certain, H 3.1; D notes 12].
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

- **On the slide:** the pilot table (v5, then the 10-epoch and the 2-epoch fine-tunes, each control next to its fix
  runs; rows root error, depth slope, drawn size, head and pelvis pixel error, PVE, recall), a legend line and a
  "not fixed / likely cost / final model" box [D slide 14].
- **Numbers** [Certain, H 3.1 table]:
  - fix runs: root error 0.53-0.73 m; controls 1.11-1.44 m;
  - size 0.78-0.87 with the fix, 0.59-0.66 without;
  - slope 0.30 with the switch alone, 0.38-0.44 with the stronger depth loss;
  - x20 for 2 epochs (0.53 m) ties x5 for 10 epochs (0.54 m);
  - cost in the pilots: every fix run is worse than its control by +2.6 to +4.9 mm PVE and -1.6 to -2.4 recall points
    (each inside noise, all the same direction) [Likely cost];
  - final model vs its control (box on the slide): root error 1.40 -> 0.73 m (v5 1.21), size 0.59 -> 0.78, slope
    0.28 -> 0.42; its PVE 77.4 -> 73.7 and recall 89.0 -> 87.0 are on slide 20, not here (the notes only say "2
    recall points, PVE inside the noise") [Certain, 3611336:175-193, 3611339:166-184].
- **Why x5** (not in the deck): before the fix, distance and translation were only about 2.5% of the loss
  [Certain, F].
- **Not claimed:** that exp1 met its own success rule. It did not: every run, controls too, misses "PVE <= 71.5 and
  recall >= 89.2" [Certain, H 3.1]. The x20 run for 10 epochs was never confirmed [Certain, H 3.1, H 4d].

### Slide 15. Neck: two half-turned bones, and a pilot fix

- **On the slide** [D slide 15]: one table (v5 before | fix, no training | fix + 2-epoch fine-tune | control, no fix;
  flipped values in red, the fine-tuned column in teal), a definitions line, four bullets (cause, measured: all 8
  v5-line checkpoints; the exp6 fix, one constant turn of about 179 deg per bone, estimated on 200 training images,
  also turns back two hand bones; fix with no training fails the 15-deg head-direction limit; fix + fine-tune passes
  the fine-tune checks), and three lines: caveat (one run, no seed, no PVE gain claimed), open (why the bones
  flipped), final model (neck02 4.8 deg, head 8.9 deg, thickness 1.00, neck-skin error 11.0 mm, head direction
  13.1 deg). There is no neck-chain drawing.
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

  The first four columns are the slide's table. The final-model column is a line under the table on the slide, from
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
  fine-tune without the fix keeps thickness 0.461, so the correction is the [Likely] cause) [H 3.2]. 4 of the 12 demo
  images were not looked at [Certain, D notes 16].

### Slide 17. Fingers: about half the error was two flipped bones

- **On the slide:** three numbers: 17.3 deg (predicted fingers, v5), 8.3 deg (with the bone fix, 2-epoch pilot and
  10-epoch final model; controls 17.4), 15.1 deg (fingers set to rest pose, before the fix; the index-base flip
  stays); a hand-size table (median hand size 2.3 tokens in the full 672-px image, 4.2 with a person crop at test
  time, 0.7 for children's hands; 41% / 0% / 80% under 2 tokens); bullets on the two flipped bones (not yet checked by
  eye), test-time zoom (17.2 vs 17.3; body error nearly doubles) and the loss rebalance (17.5 vs 17.4) [D slide 17].
- **What we did:** first tested the old beliefs: magnifying the person at test time (17.3 -> 17.2) and a rebalanced
  loss (17.5 vs 17.4 for its control); neither helped [Certain, H 3.3]. Then the per-bone search found two hand bones
  turned half a turn: metacarpal1.L (180 deg) and finger4-1.R (178.8 deg) [Certain, H 3.2, H 3.3].
- **Result:** correcting them halves the finger error: 17.3 -> 8.2 deg without training, 8.3 after the 2-epoch
  fine-tune, and 8.3 in the final model against 17.4 for its control [Certain, 3579014:114; 3611336:201,
  3611339:192]. On v10, correcting three flipped hand bones takes the finger error 32.0 -> 17.8 deg (in the notes,
  not on the slide) [Certain, 3579015:80; D notes 17].
- **Not claimed:** better-looking fingers on real photos. Every real-photo demo so far sets the fingers to rest pose
  (`--relax_hands 1`), so predicted hands have never been shown [Certain, P 2 row 1a; H 3.3].

### Slide 18. Children and gender on real photos

- **On the slide** [D slide 18]: a table with three columns, v5 | final model | masked arm (slide 19), rows recall,
  gender accuracy, children AUC, mean predicted age of adults / kids / babies, babies below the child line 0.45; three
  tiles from the synthetic side and the summary (2% -> 10% children per batch with a balanced sampler, unproven after
  2 epochs; 0.10 predicted age of training babies, true 0.015; children AUC 0.61 -> 0.72, measured improvement, not
  solved); and three bullets (v5 reads children as short adults, gender at chance; final: women recognised 0.86, men
  mid-range 0.55, babies 0.44 vs adults 0.54, likely pulled down by the 3DPW age label; not solved, with the
  limitation-or-labels question).
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
  slide 16), final 0.551 / 0.861 (on the slide as 0.55 / 0.86); and the height value of adults / kids / babies, v5
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
  (variety comes back; gender and children fall back; error unchanged within noise), and a bottom line "Reading and
  next step" (the age label likely does the useful work, the seven 0.5 constants cause the collapse; decided 18 Oct:
  the unmasked model stays the final model and the student's teacher; next: the partial mask, job 3614042, queued).
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
- **Next:** mask only the seven 0.5 values and keep age supervised (job 3614042, queued; result after the meeting)
  [Certain, H 0a]. [Likely] the age label 0.55 carries the gender and children gain and the 0.5 constants cause the
  collapse.
- **Not claimed:** that the masked values are *right* on real people. No real set we have gives ground truth for muscle
  or weight [Certain, P 3.3]. One run per arm, noise applies.

### Slide 20. Final model: all fixes together, with a control

- **On the slide** [D slide 20]: the table v5 | control | final model (the rows below, without the depth-slope row;
  the slide says "size ratio" and "neck girth"), a box (fixes hold together; the cost: recall -2 points vs the
  control, -3 vs v5, PVE -3.7 mm inside the noise, depth slope still 0.42), the 3DPW line, the recipe line (10 epochs,
  40,000 images, 70% AnnyOne + 30% 3DPW-train, one run each, noise about 7.6 mm / 2 recall points) and the source
  line (jobs 3611336, 3611339, 3612073-75; NAVER 3614192). The 3DPW line now starts with "NAVER 126.9".
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

- **On the slide:** `docs/figures/final_model_real_photos.jpg` (tennis photo, `downtown_walking_00`,
  `downtown_runForBus_00`; v5 left, final right) and bullets: men look male (tennis 3 of 4, walking, office, guitar);
  women stay female (stage photo, weekly market); bodies a little larger (0.63 -> 0.78); unchanged (the boy, placement
  offsets, ghost bodies, hard poses); the Relative Human line; source demo job 3612078 [D slide 21].
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

- **On the slide:** a table requirement | status | evidence (18 Oct) | next (see Part 7 for the full text), in the
  order 1c, 2a, 1a pose, 1a shape, 1b, 2b, 3, and a "how to read" line. The evidence and next cells name the partial
  mask (job 3614042), NAVER's model (row 1b: "NAVER's model: 60.3 mm AnnyOne (likely in-sample) / 126.9 mm 3DPW vs
  ours 73.7 / 89.0"; job 3614192) and the ViT-B student (job 3614191, slide 24); the report
  row says "draft exists (74 mm -> 70.5); 98 files hard-code /netscratch; no results or models committed"
  [D slide 23].
- **Status words:** 1c met; 2a met; 1a pose partly; 1a shape partly, at risk; 1b not met; 2b not started; 3 partly
  [Certain, P 2; D slide 23].
- **NAVER baseline in row 1b** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]: NAVER's own Anny model gets 60.3 mm PVE on our AnnyOne holdout
  and 126.9 on the 3DPW test, against our final model's 73.7 and 89.0. [Likely] it was trained on the whole AnnyOne
  release, so the holdout images are probably in its training set (held out for us, not for it). [Certain] On real
  3DPW it is far worse than ours.
- **Not claimed:** that 1b is met by the current model. It is still Multi-HMR [Certain, P 3.1 item 1]. Nor that our
  model beats NAVER's in general: it does on 3DPW, not on AnnyOne.

### Slide 24. Plan for 1b: a smaller model by distillation

- **On the slide** [D slide 24]: one table with four rows: our ViT-L (final model, the teacher: 318.4 M params,
  1,927 GFLOPs, 110 ms on an A40, holdout PVE 73.7 mm, 3DPW test PVE 89.0 mm); the ViT-B student (101.4 M, 603,
  51.4 ms, errors pending this week); NAVER's ViT-L Anny model (params, GFLOPs and ms "pending"; AnnyOne holdout PVE "60.3 mm*", 3DPW test PVE
  "126.9 mm"; job 3614192); and "student vs teacher"
  (3.14x fewer params, 3.19x fewer FLOPs, 2.14x faster; bars <= 77.6 mm and <= 93.7 mm). Under it: a "how to read"
  line (our cost measured in smoke test 3613926, 1 multiply-add = 2 FLOPs, 672 px, one person; NAVER's errors from job
  3614192; "*in-sample for NAVER, likely: trained on all of AnnyOne; our holdout is held out only for us"; "†same ViT-L
  backbone: 304 M params, 1,917 GFLOPs alone"), the 95% rule with both pairs of bars (ours 77.6 / 93.7 mm, NAVER's
  63.5 / 133.6 mm; "which counts: slide 26, Q1"), and three bullets: recipe, timeline, risk with the question for the
  supervisor. There is no teacher -> student drawing.
- **What we did:** wrote exp8 (`slurm/experiments/exp8_distill_vitb.sh`). The student is a DINOv2 ViT-B initialised
  from NAVER's Multi-HMR ViT-B; a small projector (768 -> 1024) feeds heads copied from the teacher; it learns from the
  normal labels plus the teacher's features (weight 1.0) and outputs (weight 1.0); same targets as the teacher (depth
  fix, 30% 3DPW, hand mask, teacher's bone fix); batch 8, learning rate 3e-5, 30 epochs x 1,000 steps = 240,000 images
  [Certain, exp8 header; E exp8]. The slide's recipe bullet adds the distillation temperature, T 4.
- **Status:** smoke test 3613926 passed end to end; full run job 3614191 launched 18 Oct (30 epochs, 20-hour limit);
  the slide says "result this week once the queue runs it"; if it misses the bar with a still-falling curve: train
  longer and report the ratio as measured. NAVER's own Anny model on the same labels: done 18 Oct evening, job 3614192 (the first
  try, 3613928, crashed on a head-size mismatch, now fixed; that job id is not in the deck) [Certain, H 0a].
- **NAVER baseline numbers** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a] (518/518 tensors loaded, `--model_class multi_hmr_original`; Multi_HMR
  masks the helper bones by design):

  | | NAVER `multiHMR_672_L_anny` | our final model |
  |---|---|---|
  | AnnyOne holdout (same 100 images, helper bones masked): PVE / PA-PVE / MPJPE / PA-MPJPE, mm | 60.3 / 44.6 / 63.3 / 45.0 | 73.7 / 50.3 / - / - |
  | AnnyOne: precision / recall / F1 | 82.0% / 89.0% / 85.0 | - / 87.0% / - |
  | 3DPW test (v23, 8,190 images): PVE / PA-PVE / MPJPE / PA-MPJPE, mm | 126.9 / 68.8 / 129.4 / 66.4 | 89.0 / 57.6 / 88.3 / - |
  | 3DPW: precision / recall / F1 | 75.0% / 97.0% / 85.0 | 64% / 99% / - |
  | params / GFLOPs / ms | pending (`model_cost.py`, H 0a) | 318.4 M / 1,927 / 110 |

  Caveats: [Likely] NAVER trained on the whole AnnyOne release, so our holdout is probably in-sample for it and its
  60.3 is not a fair test number. [Certain] On 3DPW it is far worse than ours and about v5's level (121.8).
- **Numbers** [Certain, exp8 smoke test 3613926, H 0a]:

  | | teacher ViT-L (final model) | student ViT-B | ratio |
  |---|---|---|---|
  | params | 318.38 M | 101.38 M | 3.14x fewer |
  | GFLOPs per image | 1,927.4 | 603.3 | 3.19x fewer |
  | ms per image (A40) | 110 | 51.4 | 2.14x faster |

  The slide rounds these to 318.4 M / 101.4 M and 1,927 / 603 GFLOPs. The student's 101.38 M is its ViT-B backbone
  (86.58 M) plus the teacher-sized heads (about 14.0 M) and the projector (0.79 M); this split is not in the deck
  [Certain, `model_cost.py` table in H 0a; exp8 header "heads"].
- **The 95% rule:** for PVE and MPJPE on both the AnnyOne holdout and the 3DPW test, student error <= teacher error /
  0.95 (exp8's rule; targets about 77.6 mm and 93.7 mm), plus fewer params and FLOPs and faster [Certain, exp8 header
  "THE 95% RULE"]. The deck uses only this /0.95 form; the ratio is reported against both our ViT-L and NAVER's
  [Certain, P 3.2]. Against NAVER the bars would be 60.3 / 0.95 = 63.5 mm on AnnyOne, which no student of a 73.7 mm
  teacher can reach, and 126.9 / 0.95 = 133.6 mm on 3DPW, trivially met; against our ViT-L 77.6 / 93.7. The AnnyOne gap
  to NAVER is 13.4 mm and likely in-sample for NAVER; that is why question 1 on slide 26 matters.
- **Why ViT-B:** NAVER's own ViT-B equals its ViT-L on 3DPW (PVE 94.0 vs 94.1; the slide says "equal on 3DPW", the
  numbers are in the notes of slide 23) [Certain, fetched NAVER README, P 3.2]; our old ViT-S students had 1.17-1.54x
  their teacher's error (notes of slide 23) [Certain, P 2].
- **Not claimed:** that the student will pass. [Guessing] likely on 3DPW, about 50/50 on AnnyOne, because NAVER's ViT-B
  is about 15% worse than its ViT-L on synthetic BEDLAM data (P 3.2). Whether a smaller Multi-HMR-style student counts
  as "not exactly Multi-HMR" is the supervisor's decision (question 1, slide 26).

### Slide 25. Plan: the next four weeks

- **On the slide** [D slide 25]: a table week | dates | what happens | for, with six rows (weeks 1, 2, 3, 4, 5-6 and
  7-8), and a "how to read" line: part (1) of the list closes around 15 Nov; dropped: the ViT-S student, the children
  sampler runs and the depth-slope investigation (slope 0.42 becomes a stated limitation).
- **Week 1 (19-25 Oct; 1a shape, 1b):** NAVER baseline: done 18 Oct (job 3614192, slide 24); the partial-mask run (job 3614042) and its
  Relative Human run; decide the final ViT-L on Sunday 25 Oct; the student's result (job 3614191) once the queue runs
  it: holdout, 3DPW test, the 95% table; code: a masked-evaluation flag and the student's shape-distillation mask;
  phenotype statistics on AnnyOne (r, slope, R^2) [D slide 25; P 4 week 1; H 0a].
- **Week 2 (26 Oct-1 Nov; 1a, 3):** a rest-pose line-up of predicted bodies (the "visibly changes" figure); predicted
  fingers on real photos (`--relax_hands 0`; every demo so far forces rest-pose hands); the old ViT-S students on the
  new labels; a body-joint MPJPE; write the Method and Data chapters [D slide 25; P 4 week 2].
- **Week 3 (2-8 Nov; 1b, 3):** if the student missed the bar and its curve still fell, train it longer from its last
  checkpoint; the 95% table against our ViT-L and NAVER's; paired bootstrap confidence intervals; write the
  Experiments chapter (depth, neck, shape mask, Relative Human) [D slide 25; P 4 week 3].
- **Week 4 (9-15 Nov; 1b, 3):** one final table, one protocol, every model; one more seed of the final ViT-L if a slot
  is free; architecture figure and module table; freeze the numbers; start the model release with checksums
  [D slide 25; P 4 week 4].
- **After:** weeks 5-6 (16-29 Nov) the temporal refiner (2b); weeks 7-8 (30 Nov-13 Dec) introduction, related work,
  conclusion, limitations, revision, code clean-up and the presentation, week 8 as buffer [D slide 25; P 4].
- **Not on the slide:** Monday evening, write down the supervisor's answers and check jobs 3614042 (partial mask)
  and 3614191 (student); NAVER's job 3614192 is done (its params / GFLOPs / ms still need `model_cost.py`, H 0a).
- **Not claimed:** a buffer before 15 Nov. Tier (1) closes around 15 Nov with no slack [Likely, P 3.6]. Parts of the
  plan file still say the student launches Tue 20 Oct with 100 epochs; it was in fact launched on 18 Oct with
  30 epochs, as the deck says.

### Slide 26. What I claim, what I do not, and my questions

- **On the slide** (dark closing slide): one block "Claimed, measured / Not claimed" and a box with the four
  questions. Relative Human recall 47 -> 65% is in the notes, not on the slide; the notes also answer last week's
  question: labels first, then the v5 fine-tune, which is the final model [D slide 26, D notes 26].
- **Claim (measured) [Certain]:** the failures are not rig or decoding problems; a training-target bug put people too
  far away in v5, v9 and v10, and the fix draws them much closer to their true size; neck02, the head bone and two
  hand bones were half a turn off, and a fixed correction turns them back; the 3DPW labels had three defects, now
  fixed and checked; the final model keeps every fix against its own control (root error -48%, neck02 4.8 deg,
  fingers 8.3 deg, PA-PVE 50.3 vs 56.4 mm) and is the best on our 3DPW-Anny labels (89.0 vs v9 95.5, v5 121.8); on
  Relative Human it is better than v5 on every line (gender 46.5 -> 64.2%, AUC 0.61 -> 0.72, recall 47 -> 65%).
- **Not claimed:** results proven over seeds; why the bones flipped; children or gender solved; correct near/far depth
  (slope 0.42); fixes without cost (recall -2 points vs control on the slide; -3.2 vs v5 in the notes); that the fixes
  made most of the 3DPW gain (3DPW training did); a model that is not Multi-HMR (requirement 1b). Not in the deck's
  list but true: visibly varied body build on real photos is not claimed either (muscle, weight near 0.5; slide 19).
- **Questions for the supervisor** (D slide 26, P 5):
  1. (1b) Is the 95% bar against our ViT-L, NAVER's ViT-L, or both? (NAVER: 60.3 mm on AnnyOne, likely in-sample;
     126.9 on 3DPW; ours 73.7 / 89.0) Does a smaller Multi-HMR-style student count as "not exactly Multi-HMR"?
     Against NAVER the bar would be 63.5 mm on AnnyOne (out of reach for a student of a 73.7 mm teacher) and 133.6 mm
     on 3DPW (already met); against our ViT-L 77.6 / 93.7 [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a].
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
  (AUC 0.61 -> 0.72, not solved; fingers: bone-fix hands, then larger crops); (7) 3DPW labels, three defects fixed
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
- **Hands on real photos:** never shown with predicted fingers (slide 17).
- **Why the bones flipped:** unknown; v10's left arm needs a two-sided correction, not built [Certain, H 3.2].
- **The x20 depth run for 10 epochs:** command given on 14 Oct, never confirmed [Certain, H 3.1, H 4d].
- **3DPW precision 64% and ghost bodies** (not on this slide; slide 20 notes): about a third of the detections match
  no label; the split between bystanders and ghosts is unknown [Certain, P 3.1 item 4].
- **NAVER baseline** (not on this slide; slides 20, 23, 24, 26, 30): done, job 3614192: 60.3 mm AnnyOne holdout
  (likely in-sample for NAVER), 126.9 mm 3DPW test; params / GFLOPs / ms pending [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a].

### Slide 28. Corrections to the draft and the repo

- **On the slide** (appendix) [D slide 28]: four boxes. Headline number: done, the README gives AnnyOne holdout PVE
  70.5 mm, helper bones masked; to do, replace the thesis draft's 74 mm (the unmasked evaluation, +3.8 mm on the same
  people). Demos: done, all demo scripts zero the fingers and the README says so. Wording: done, neck, fingers and
  children are no longer called rig or decoding problems, and the README's Known-defects section documents the depth
  bug; to do, the README and the findings document still carry last week's neck story. Repository: done,
  reorganised, one shared cluster setup, diagnostics and experiments documented. The notes add that the demo setting
  gives 68.5 mm at a lower recall.

The full list of corrections the thesis draft needs is longer than the slide. The draft is not in the repository;
these lines come from H 1 "Headline", H 6 and H 7, and apart from the headline number they are not on the slide
[Certain that each old claim is wrong, as stated there]:
- **Headline number:** the draft says 74 mm. That is the unmasked V0 evaluation of v5. Use 70.5 mm (V1, helper bones
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
rows: v5 epoch 99; 3611336 final model (3DPW test in its log); 3611339 control; 3612325 masked-shape arm; 3614042
partial mask (queued); 3614191 ViT-B student (smoke test 3613926; `logs/exp8_distill_vitb_3614191.out`); 3614192
NAVER's Anny model on both test sets: 60.3 / 126.9 mm (`logs/eval-test3dpw-3614192_0.out`,
`models/multiHMR/multiHMR_672_L_anny.pt`); 3611285 Relative Human v5 (_0) and neck pilot (_1); 3612131 / 3613616
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
- partial mask (queued): `results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10_shpmask_noage/`
- student (running): `results/experiments/exp8_distill_vitb/student_vitb_from_v5_pc_dist5_bonefix_v23_e10_00009_e30/`
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
| 3614042 | partial mask (7 constants, keep age), queued |
| 3613926; 3614191 | exp8 student smoke test (passed); full student run (submitted 18 Oct) |
| 3613928; 3614192 | NAVER baseline (crashed, fixed); resubmitted, done 18 Oct evening: AnnyOne holdout PVE 60.3 (likely in-sample for NAVER), 3DPW test 126.9 |

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
- **Missing:** a demo with predicted fingers; a figure that shows shape changing (rest-pose lineup); numbers that show
  each phenotype follows its label on AnnyOne (correlation, slope); the depth slope (0.42) as a stated limitation
  [P 2, P 3.3].
- **Planned:** partial-mask run (job 3614042) with its Relative Human run and demo, week 1; decide by 25 Oct; lineup
  and finger demo, week 2 [P 4; H 0a].

### 1b. A model that is not exactly Multi-HMR

**Status: not met.**
- **Evidence [Certain]:** `model.py` is the upstream Multi-HMR with the output resized for Anny and a 0-parameter bone
  fix (P 2). The ViT-B student is training (job 3614191, submitted 18 Oct; smoke test 3613926 passed). Its cost is
  measured: 101.38 M params / 603.3 GFLOPs / 51.4 ms vs the teacher's 318.38 M / 1,927.4 GFLOPs / 110 ms (A40):
  3.14x fewer params, 3.19x fewer FLOPs, 2.14x faster (H 0a).
- **Evidence, the baseline** [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]: NAVER's own Anny model on our labels and settings: AnnyOne holdout PVE
  60.3 / PA-PVE 44.6 / recall 89%; 3DPW test PVE 126.9 / PA-PVE 68.8 / recall 97%; ours 73.7 / 50.3 / 87% and 89.0 /
  57.6 / 99%. [Likely] in-sample for NAVER on AnnyOne (trained on all of it). Consequences: against NAVER the 95% bar
  would be 63.5 mm on AnnyOne (no student of a 73.7 mm teacher can reach it) and 133.6 mm on 3DPW (trivially met);
  against our ViT-L 77.6 / 93.7. Our final model already beats NAVER on 3DPW but not on AnnyOne, so the
  new-architecture route (1b-i) would hinge on AnnyOne.
- **Missing:** the student's accuracy (the 95% rule on both test sets); NAVER's params / GFLOPs / ms (pending,
  `model_cost.py`); the supervisor's yes that a smaller student counts, and which baseline sets the bar [P 3.2].
- **Planned:** student result within days of the job starting (job 3614191, 20 h); fallback decision (train longer) right after; final table by 15 Nov [P 3.6, P 4].
  [Guessing] passing on 3DPW is likely; on AnnyOne about 50/50 (P 3.2).

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
- **Evidence [Certain]:** a draft exists outside the repository, with the stale 74 mm headline; 98 tracked files
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
parameters [Certain, P 2]. That is why requirement 1b is not met. Our route is a smaller ViT-B student (3.14x fewer
params, 2.14x faster); whether that counts is your decision [Certain for the numbers, H 0a]. It is also not NAVER's
checkpoint under another name: NAVER's own `multiHMR_672_L_anny`, run with our code on the same labels, gets 60.3 mm on
the AnnyOne holdout (likely in-sample for it) and 126.9 on the 3DPW test, against our 73.7 and 89.0 [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a].

**Q3. What does the 95% rule mean in practice?**
For error measures, lower is better, so we read it as: the student's PVE and MPJPE must be at most the teacher's
divided by 0.95, on both the AnnyOne holdout and the 3DPW test. With the final model's numbers that is about 77.6 mm
holdout PVE and 93.7 mm 3DPW PVE; the deck uses this /0.95 form only. It must also have fewer
params and FLOPs and be faster [Certain, exp8 header, P 3.2]. Open: is the larger model our ViT-L or NAVER's? We will
report both. **Which baseline, in numbers:** NAVER gets 60.3 mm on AnnyOne (likely in-sample: trained on all of AnnyOne)
and 126.9 on 3DPW, so its bars would be 63.5 mm (out of reach for any student of our 73.7 mm teacher) and 133.6 mm
(already met); against our ViT-L 77.6 / 93.7. The AnnyOne gap to NAVER is 13.4 mm [Certain, job 3614192, `logs/eval-test3dpw-3614192_0.out`; H 0a]. NAVER's params /
GFLOPs / ms are pending.

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
run that masks only the seven 0.5 values and keeps age is queued (job 3614042).

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
