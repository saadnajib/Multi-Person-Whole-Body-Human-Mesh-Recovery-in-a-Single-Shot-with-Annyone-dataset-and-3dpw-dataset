# Speaker notes: supervisor update, 16 October 2026

## Slide 1: Title

Since last Friday I stopped guessing at the neck, finger and child problems and measured each stage of the pipeline separately. I will show what is not the cause, what the real causes are, one bug nobody was looking for, and what is running on the cluster right now. I will be careful to separate what is measured from what is still a hypothesis.

## Slide 2: Where we started

These are the three failures we discussed last time. Until this week they were explained by guesses: rig problems, resolution, or a shape head that does not learn age. None of those guesses had been tested in isolation. That is what I did first.

## Slide 3: What I did

I built a diagnostic suite with six checks, each isolating one stage. The important design choice: a fixed holdout of 100 images that were never trained on, and a fixed manifest with ten adults, ten children, ten visible-hand and ten hard-neck cases, with separate train and held-out examples. Every number from here on comes from those fixed sets with the same evaluation settings.

## Slide 4: What is not the cause

This slide is the one that changes the story. The rig, the decoding, the body model and the optimisation path are all correct. So the three failures cannot be rig or decoding problems, which is how the thesis described them until now. The one caveat on the ground-truth check: frame-filling people lose joints at the image border, which is truncation, not a decoding error.

## Slide 5: Neck

The neck length the model predicts is within one percent of the truth, and the head-neck surface error is similar to the rest of the body. What is large is the rotation error of the individual neck bones, 63 and 90 degrees, but the final orientation of the head is only 12 degrees off. That means the chain is redundant: many local rotations give the same head. The per-bone loss is a weak teacher there. So it is a supervision problem, and the visual long-neck effect is mostly the depth bug I will show in a moment.

## Slide 6: Fingers

The uncomfortable number: with predicted fingers the error is 17.3 degrees, with fingers simply left in the rest pose it is 15.1. The hand prediction is worse than no prediction, which is why the demos zero the fingers and now say so. I tested the obvious explanation, that hands are too small at 672 pixels. They are small, about two tokens, but when I crop around the person and magnify up to three times the finger error does not move at all. So the fingers are a training problem: too little gradient, badly conditioned rotation outputs, not a pixel problem.

## Slide 7: Children

I verified in the code that the shape head is complete: all eleven phenotypes go into the body model and are supervised, with no adult prior in the initialisation. The model can represent a baby: for the youngest training people it predicts age 0.10 for a true 0.015, with the right head-to-height ratio. The problem is the data. Two percent of supervised people are children and 84 percent of batches have none, and the 3DPW labels carry a fixed adult age. The honest caveat is that the holdout has no children at all, so I cannot measure generalisation to unseen children, only that the representation works.

## Slide 8: The hidden bug

This is the main result of the week. The model attaches the mesh to the detected pixel by its head joint and reports the head as its position. The training code built the position, distance and heat-map targets at the pelvis. Those two can never agree. The 2D reprojection loss resolved the conflict the only way it could: by shrinking the body, which means predicting it farther away. Measured on the holdout: predicted depth is 0.3 times the true depth plus 2.4 metres, every single person too far, bodies a third too small. Our standard metric is pelvis-centred, so it never saw this. That is why the thesis numbers looked normal while the overlays always looked slightly wrong. The bug is in training code, so v9 and v10 have it too; I measured v10 and it is slightly worse.

## Slide 9: The fix

The fix is one line of logic: build the training targets at the joint the model anchors. I ran a two-epoch pilot and, importantly, a control run with the same schedule and the flag off, so that the learning-rate restart cannot be mistaken for an effect. The control moved nothing. The fix halved the absolute depth error and brought the head pixel error from 64 to under 10 pixels. The cost is a worse pelvis-centred PVE and recall, which I expect because the detector was trained for 100 epochs to fire on pelvis cells and now has to learn head cells. Ten-epoch runs of both arms are running. I am not claiming an improvement until they are back.

## Slide 10: Experiments

All four changes are behind their own flag, default off, each in its own commit with its own cluster script, and each runs next to a control with the flag off. Experiment one is the bug fix. Two, three and four address the diagnosed causes of the three failures. Nothing new in the architecture, because the diagnostics do not justify a new branch. The pilots take about half an hour each; the long runs about two hours.

## Slide 11: Housekeeping

Some corrections are already in. The headline AnnyOne number should be 70.5 millimetres, measured with the helper bones masked as in the demos; the 74 in the draft came from the unmasked evaluation. The demos now disclose that fingers are the rest pose. The text no longer blames the rig. And the repository is reorganised with everything scripted, so any of these numbers can be regenerated.

## Slide 12: Claims

To close: what is measured is that the failures are supervision and data problems, that a training-target bug depresses depth in every model we have, and that its fix moves depth in the right direction. What is not yet shown is any improvement in the three failures, and the final effect of the fix. The runs that decide this are already on the cluster. The decision I need from you is whether to retrain the depth fix into v9, our best model, before the thesis numbers are frozen.

