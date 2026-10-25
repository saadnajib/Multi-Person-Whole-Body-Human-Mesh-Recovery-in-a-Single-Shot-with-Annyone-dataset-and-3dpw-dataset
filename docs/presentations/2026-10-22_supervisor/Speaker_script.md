# Speaker script — Muhammad Saad Najib

20 main slides + 3 technical backup slides. Suggested speaking time: about 20–25 minutes, plus video and questions.

Model names: Student = deployed ViT-B; Teacher/Final = larger ViT-L; Base/v5 = earlier thesis model; released baseline = NAVER Multi-HMR-Anny.

## Slide 1 — Multi-Person Human Mesh Recovery

My thesis studies how to recover a three-dimensional body for each person visible in one photograph. A body mesh is a surface made from many connected points, like a digital mannequin. It represents body pose and shape, rather than just a few joints. The main model in this presentation is my smaller student model. It learns from labelled examples and from a larger teacher. I will explain the starting system, what the student learns, how I converted real-image labels into the Anny body format, and the measured trade-off between speed and reconstruction quality. The illustrations on this first slide show earlier Base and teacher outputs; the student results will be identified separately.

**Point / demo cue:** Introduce the task before using abbreviations.

**Transition:** First, why is this problem difficult?

**Source:** Muhammad Saad Najib · Supervisor presentation · 22 October 2026

## Slide 2 — One photograph, several difficult decisions

A photograph shows a flat image, but our output must describe a three-dimensional body. Several different bodies and distances can produce a similar picture, especially when clothing or another person hides part of the body. There are two additional challenges in this project. First, synthetic training images come with detailed labels, while real photos look different and have less complete supervision. Second, the larger model is expensive to run. My question is therefore whether a smaller student can learn enough from a stronger teacher and from both datasets to provide useful results with lower inference cost. I evaluate this as a trade-off: a faster result is valuable, but I also need to show how much reconstruction accuracy is lost.

**Point / demo cue:** Point to the three challenges from left to right.

**Transition:** The starting point is Multi-HMR.

**Source:** Thesis revision, 22 Oct 2026

## Slide 3 — Baseline: one image → multiple 3D bodies

This is the baseline pipeline. The RGB image is resized and padded to the model input size. The DINOv2 encoder turns it into image features, which can be thought of as descriptions of small image regions. A detection head identifies person anchors and their image offsets. The person decoder uses cross-attention to combine information about each person with the rest of the image. It predicts the parameters needed by a body model, and the body layer converts those parameters into a mesh. Camera information helps place the mesh in three-dimensional space. The original Multi-HMR work also describes an SMPL-X version; our external reference is specifically NAVER’s released Anny checkpoint. This distinction matters when comparing outputs and model sizes.

**Point / demo cue:** Trace the arrows. Define encoder and decoder in plain words.

**Transition:** My student preserves this general pipeline but uses a smaller encoder.

**Source:** Sources: Multi-HMR (Baradel et al., 2024); released NAVER Anny checkpoint; thesis §4.2

## Slide 4 — The student learns from labels and a frozen teacher

Both branches receive the same image and camera information. The larger teacher has a ViT-L encoder and remains frozen, meaning training does not update its weights. The student uses a smaller ViT-B encoder. Because its feature width is 768 rather than 1,024, a small trainable projector converts its features to the width expected by the transferred prediction heads. There are three sources of learning. Dataset labels provide the supervised loss. Teacher features provide the feature-distillation loss. Teacher body parameters provide the output-distillation loss. These losses update the student, while teacher targets are detached from gradient updates. The copied heads are frozen initially, then trained. At deployment the teacher is removed, but the projector stays because the student heads need its output.

**Point / demo cue:** Point to gold arrows as teacher guidance and the bottom line as label supervision.

**Transition:** Next, what exactly does the network predict?

**Source:** Sources: thesis §4.1, §4.8; model.py; distill.py; train.py

## Slide 5 — What goes in, and what comes out?

The input used at inference is an RGB image and camera intrinsics. When the camera calibration is unavailable, the demo uses an assumed field of view, which can affect recovered depth. For each detected person, the model predicts bone rotations, eleven Anny body-shape parameters, and quantities describing the head anchor’s image location and depth. Rotations describe articulation, while shape parameters control aspects such as height and body proportions. The Anny layer converts these parameters into a mesh and joints. The mesh is then translated so its head anchor reaches the predicted camera-space position. During training, labels provide target parameters and geometry. At inference those targets are absent: the model must estimate everything from the image and camera information.

**Point / demo cue:** Use pose, shape and placement as three separate ideas.

**Transition:** The losses tell the student which parts of its answer need to improve.

**Source:** Sources: thesis §4.2; model.py

## Slide 6 — Supervised losses: compare predictions with labels

A loss is a numerical penalty for a mismatch during training. Here the penalties measure several different parts of the prediction. Detection and offset losses help find people and locate their anchors. Pose and shape losses compare the predicted body parameters with their labels. Depth and translation losses teach camera-space placement. The three-dimensional joint and vertex losses compare bodies after moving each pelvis to the origin, so they measure body geometry without absolute translation. The two-dimensional losses project the body back onto the image and measure alignment there. This is why a low centred body error does not automatically mean correct placement in the photo. The student receives all these supervised terms plus the two teacher-guidance terms on the next slide.

**Point / demo cue:** Read purposes, not every formula. Full weights and equations are in the backup.

**Transition:** Distillation adds information from the teacher.

**Source:** Sources: thesis §4.3; Table T02; exact formulas in backup slides

## Slide 7 — Knowledge distillation: two kinds of teacher guidance

Knowledge distillation means using a larger trained model to guide a smaller one. The first form is feature distillation: the student’s projected image features are encouraged to resemble the teacher’s features. The implementation turns feature channels into softened distributions and compares them with KL divergence. A temperature of four controls that softening. The second form is output distillation: the student is encouraged to predict similar rotations, body-shape parameters and depth. During training the two models use the same labelled person locations, so outputs can be compared person by person. Teacher predictions are useful guidance, but they can also contain mistakes. Ground-truth supervision remains present. The exact reductions and weights are included in the backup slides for technical questions.

**Point / demo cue:** Clarify that the teacher supplies guidance; it does not create new verified ground truth.

**Transition:** Here is the training recipe used for the reported student.

**Source:** Sources: thesis §4.8; distill.py; loss.py

## Slide 8 — Training recipe for the reported student

The reported student is the completed thirty-epoch ViT-B run. Each epoch has one thousand optimizer steps, with eight images in a batch, giving two hundred and forty thousand sampled image presentations over training. These are not necessarily unique images. The sampler draws about seventy percent synthetic AnnyOne images and thirty percent converted 3DPW training images. This is an image ratio; the person ratio differs because images contain different numbers of people. The student starts with an HMR-trained ViT-B backbone and copied teacher heads. The heads stay frozen for the first five epochs, then train with the student. The learning rate starts at three times ten to the minus five and halves every twelve epochs. Student-specific output calibration is estimated after training.

**Point / demo cue:** Emphasize this is the confirmed 30-epoch checkpoint, not an unverified longer run.

**Transition:** I use several metrics because a single number does not describe the entire result.

**Source:** Sources: thesis Table T11; exp8_distill_vitb.sh; job 3614191

## Slide 9 — How reconstruction quality is measured

PVE is the average distance between corresponding mesh surface points after aligning the pelvis. A value of one hundred millimetres means an average point error of ten centimetres under that evaluation alignment. PA-PVE goes further: it optimally rotates, translates and scales the predicted body before measuring error. It is therefore useful for examining pose and shape, but can hide errors in global size and orientation. MPJPE performs a similar comparison using joint positions rather than surface vertices. Recall records how many labelled people were recovered, because a geometry score on matched people cannot describe missed detections. Finally, parameters, operation count and measured latency describe computational cost. These metrics answer different questions and should be presented together.

**Point / demo cue:** Explain millimetres with a centimetre example. Do not call PA-PVE a placement metric.

**Transition:** These are the measured reconstruction results.

**Source:** Sources: thesis §4.10; Table T07

## Slide 10 — The student loses accuracy relative to its teacher

The teacher is the strongest of our two trained models on these reconstruction measurements. On the synthetic holdout, the student’s PVE is 119.6 millimetres compared with 73.7 for the teacher. On the converted 3DPW test set, the student reaches 110.1 millimetres compared with 89.0 for the teacher. Both recover ninety-nine percent of labelled people under this evaluation. The student has lower 3DPW PVE than the released baseline, but its PA-PVE is slightly higher than that baseline’s. There are important comparison limits: the released baseline may have seen the synthetic holdout images, and it does not have the same real-image training exposure. Also, these 3DPW numbers use Anny labels, so they should not be compared directly with published SMPL-space scores.

**Point / demo cue:** Read one dataset at a time. Point out the PA-PVE reversal against the baseline.

**Transition:** The benefit of the student appears in computational cost.

**Source:** Sources: thesis Tables T16–T17; jobs 3614191, 3611336, 3614192

## Slide 11 — The student is smaller and faster than its teacher

This slide includes the original released NAVER Multi-HMR-Anny baseline, our larger teacher, and our smaller student. The original baseline has 380.94 million parameters, compared with 318.38 million for the teacher and 101.38 million for the student. The student therefore has 3.76 times fewer parameters than the original baseline. The baseline count comes from job 3614947. That benchmark counted 1,917.2 GFLOPs for its backbone only; its prediction head was not counted, so we must not present that value as full-model computation. Comparable baseline forward latency was also not measured. The teacher and student have a paired L40S benchmark at 672-pixel input, one person, batch size one, and fp32 with TF32. For this pair, the student uses roughly one third of the parameters and counted computation. Median forward time falls from 59.7 to 28.5 milliseconds, a speedup of about 2.1 times. The three large ratios refer specifically to the teacher comparison. We cannot claim that same speedup against the original baseline without measuring it under matching conditions. These are model-forward timings, not complete video throughput: preprocessing, rendering and video I/O add work. The previous accuracy slide and this efficiency slide together show the student’s measured speed–accuracy trade-off.

**Point / demo cue:** Distinguish the original NAVER baseline from our teacher. Only compare timings from the paired L40S benchmark; the baseline full-model compute and latency are unmeasured.

**Transition:** Now I will explain the data used to train these models.

**Source:** Sources: thesis Table T16; L40S job 3614191; NAVER parameter count, job 3614947

## Slide 12 — Two data sources with one common body format

AnnyOne provides synthetic images generated with the Anny body model. This gives detailed pose, shape and camera supervision in exactly the format predicted by the network. My training split uses 579,860 images, while the last one hundred images form the thesis holdout. That holdout contains adults, so it cannot answer whether the model works well for children. The real-image source is 3DPW, which consists of recorded video sequences with body annotations. However, those annotations use SMPL rather than Anny. I did not create 3DPW itself: my contribution here is converting its existing annotations into usable Anny targets while keeping the official training, validation and test separation. The corrected conversion yields the image counts shown on the right.

**Point / demo cue:** Say “converted 3DPW labels,” not “created the 3DPW dataset.”

**Transition:** The conversion is an optimization problem.

**Source:** Sources: thesis Chapter 3; numbers_data.tex

## Slide 13 — How the 3DPW–Anny labels were created

The converter begins with the source body joints and camera information from 3DPW. It searches for Anny parameters whose joints are close to those targets. The fitting proceeds in stages: root alignment first, then articulated pose, and then the shape parameters that can be supported by joint positions. A previous frame provides an initial guess for the next frame. After the first pass, the median fitted shape is assigned to the person across the sequence, and poses are fitted again with that fixed shape. This avoids treating the same actor as a different body in every frame. A label builder then transforms fits into camera coordinates and groups all labelled actors in each image. The shared sampling grid ensures both actors are represented on the same selected frames.

**Point / demo cue:** Walk through the four numbered stages.

**Transition:** The difficult part is that the two skeletons do not define every joint at the same place.

**Source:** Sources: thesis §3.3–3.5; smpl_to_anny.py, converter v2.3

## Slide 14 — Virtual joints compare equivalent body locations

SMPL and Anny can place a joint with a similar name at slightly different anatomical positions. If I simply compare the names, the optimizer may try to bend or stretch the body to remove a mismatch caused by the joint definition itself. A virtual joint gives the fitting procedure a more comparable point. I take a point attached to an Anny bone and add a fixed offset measured in the aligned rest skeletons. When the bone rotates, the offset rotates with it. In the equation, h is the bone head, delta is the local offset, and Q rotates that offset into the posed frame. The resulting point p is compared with the SMPL target. This does not change Anny’s actual rig or make the two anatomies identical.

**Point / demo cue:** The diagram is conceptual, not a measured anatomical drawing.

**Transition:** The completed annotations are checked independently of the fitting objective.

**Source:** Sources: thesis §3.4 and Appendix B

## Slide 15 — Converted labels: checked, with residual warnings

These pictures compare the real image, the source SMPL annotation, and the converted Anny annotation. They are label-conversion examples, not network predictions. An independent validation reads the completed labels and compares their joints and projections with the source annotations. The reported median body-joint discrepancy is 14.1 millimetres. The resulting diagnostic status is WARN, because the body models still differ in proportions and some joint definitions. Earlier label-integrity problems were corrected, but that does not guarantee every fit is perfect. It is especially important not to confuse this conversion discrepancy with the student’s prediction error or subtract it from PVE. The two quantities are measured on different objects and answer different questions.

**Point / demo cue:** Point to the source and converted meshes, then the discrepancy.

**Transition:** The next slides show existing qualitative reconstruction evidence.

**Source:** Sources: thesis §3.5, Table T05; F14_label_check_downtown_walking_00771.jpg

## Slide 16 — Synthetic example: inspect placement as well as pose

This synthetic image allows a visual comparison with a known ground-truth mesh. The four columns are the input image, dataset ground truth, the earlier Base or v5 model, and the larger Final teacher. This comparison is useful for looking at apparent size and placement, not just whether the body bends plausibly. Each prediction is shown at its own predicted depth in the V1 evaluation setting. In this example, the teacher’s body appears closer to the ground-truth size and location than the Base prediction. This is a selected example rather than proof of improvement on every image. It also does not show student performance. I identify the models explicitly because the word Final in older experiment filenames refers to the larger teacher, while my deployed model is the student.

**Point / demo cue:** Point to ground truth first; compare the third and fourth panels.

**Transition:** Real-photo demonstrations require additional caution about rendering settings.

**Source:** Sources: thesis Figure F12; AnnyOne holdout image 579860, V1

## Slide 17 — Real-photo examples show strengths and limits

These are existing real-image demonstrations of Base and the larger teacher. The top row is a 3DPW walking scene, while the lower row is a real-photo garden scene. They show that the system can place plausible whole-body surfaces on multiple people, but individual overlays are not a quantitative evaluation. These legacy demos use bounding-box-based placement and force fingers to a rest pose. Therefore I cannot use them to prove correct raw camera depth or accurate articulated fingers. The garden example also reminds us that people of different sizes and ages remain difficult, especially when partially hidden. The purpose of this slide is to show representative output and its limits. A fair student comparison requires student outputs rendered under the same settings.

**Point / demo cue:** Do not describe these as newly run student examples.

**Transition:** The video pipeline provides the next comparison using the student itself.

**Source:** Sources: thesis Figures F10; walking and garden examples; demo rendering settings

## Slide 18 — Video comparison: student and released baseline

The completed walking-video run processes the same 356 frames with the student and NAVER’s released baseline. The tracker matched the primary person on all 356 student frames and on 354 baseline frames. These counts show successful execution and coverage, but they do not measure pose accuracy or smoothness. For the live demo, open the comparison video and inspect whether the mesh shakes, whether limbs suddenly change pose, and whether body shape varies between frames. The pipeline operates frame by frame; it does not train or apply a temporal reconstruction model. A dedicated placeholder is left on this slide so I can insert the comparison video before presenting. The newer FPS feature is implemented, but this successful log predates it and supplies no verified video FPS result.

**Point / demo cue:** Before presenting: replace the large placeholder with comparison.mp4 using Insert > Video > This Device. Test playback. If the clip is unavailable, report the log counts only.

**Transition:** The remaining issues define what we can honestly conclude.

**Source:** Sources: video-compare-3642415.out; thesis video evidence; demo_video.py

## Slide 19 — What remains unresolved

There are three main limitations. First, the student does not retain the teacher’s reconstruction accuracy within the planned ninety-five-percent criterion. The compression succeeds computationally, but the accuracy target remains unmet. Second, body shape on real images is still difficult, particularly for children. Strong synthetic results cannot establish performance on real children, and our small synthetic holdout contains adults. Third, a successful video run does not establish temporal stability. We need to inspect the clip and report reproducible timing and motion-variation measurements. These are bounded next steps rather than claims that a large new training programme is already complete. The presentation therefore separates what was implemented, what was measured, and what still needs evidence.

**Point / demo cue:** Keep these as specific limitations, not a list of speculative fixes.

**Transition:** The final slide brings the contribution and the trade-off together.

**Source:** Sources: thesis Discussion and Conclusion; video evidence

## Slide 20 — A smaller student with a measured trade-off

The contribution is a complete and evaluated pipeline around a smaller student model. The data work makes real 3DPW supervision usable in Anny format, with validation of the converted annotations. The training work combines direct labels, transferred prediction heads and two distillation signals from a larger frozen teacher. The evaluation shows that the student reduces parameters and computation by roughly a factor of three and halves the recorded forward latency, but has higher reconstruction error than the teacher. The video-comparison implementation adds a practical way to examine deployment behaviour on identical frames. My conclusion is therefore a measured speed–accuracy trade-off. The immediate priorities are reviewing the actual walking clip, collecting its timing and stability evidence, and keeping the written conclusions aligned with those observations.

**Point / demo cue:** Pause for questions. Use the following slides only for technical details.

**Transition:** Backup: exact objectives and sources.

**Source:** Thesis revision, 22 Oct 2026

## Slide 21 — Backup: supervised objective and exact weights

This backup gives the supervised weights used by the reported student. Detection has weight ten, offsets one, rotation zero point one, and shape twenty. Depth and translation each have weight five. The centred joint and vertex geometry terms each have weight one hundred, while the two-dimensional terms each have weight zero point one. These weights cannot be interpreted in isolation because the losses use different units and reductions. In the rotation expression, the mask suppresses unavailable 3DPW hand targets, and normalized per-bone weights emphasize selected regions. The mesh terms still include the rest-pose hands from converted labels. Shape supervision in the retained final configuration includes the converter’s fixed phenotype entries. These details matter when interpreting limitations or reproducing training.

**Point / demo cue:** Use only if asked about training loss or label masks.

**Transition:** The next backup gives the two distillation formulas.

**Source:** Sources: thesis §4.3; Table T02; loss.py

## Slide 22 — Backup: distillation equations

For feature distillation, each token’s feature channels are converted into a probability distribution using temperature four. KL divergence compares the teacher distribution with the student distribution. The implementation multiplies by temperature squared and divides by batch size. It sums over tokens and channels, so describing it as an average over tokens would change the scale. Output distillation compares teacher and student rotations, phenotype values and focal-normalized log-depth for corresponding people. The weights are zero point one, twenty and five respectively. The teacher outputs are detached, so gradients update only the student. Unlike the supervised rotation loss, rotation distillation uses all bone residuals without the missing-label hand mask or the supervised per-bone weighting.

**Point / demo cue:** Explain indices only if needed.

**Transition:** The last slide lists the source papers and project evidence.

**Source:** Sources: thesis §4.8; distill.py; loss.py rotation_distill_term

## Slide 23 — Selected references and evidence

These sources provide the architecture, body model, feature encoder, dataset and distillation background. The numerical results in this presentation come from the revised repository thesis and its traceable experiment records, rather than from the sample presentation or invented estimates. The completed student evaluation is job 3614191, and the walking-video execution is job 3642415. The accompanying source map identifies the thesis sections and image files used on each slide. Existing pictures preserve the model and rendering-setting distinctions described in the thesis. Before a final thesis defence, bibliographic metadata should be checked against the official publications, as the thesis already records that author-review step.

**Point / demo cue:** Keep as a reference slide for questions, not a slide to read aloud.

**Transition:**

**Source:** Full bibliography and run provenance: docs/thesis/
