const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const THEME = { name: "MeshDiag", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1B2233", lt1: "FFFFFF", dk2: "4A5568", lt2: "EEF2F7", accent1: "0E7C86", accent2: "E8792B", accent3: "3BA55D", accent4: "C0392B", accent5: "7B8794", accent6: "F4C95D", hlink: "0E7C86", folHlink: "7B8794" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Saad Najib"; pres.title = "Multi-HMR with Anny: diagnostics and fixes";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "DARK", background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.6, w: 8.8, h: 1.4, fontSize: 40, bold: true, color: C.background1, align: "left", margin: 0 } } },
            { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.1, w: 8.8, h: 1.8, fontSize: 18, color: C.background2, align: "left", margin: 0 } } }] });
// closing slide: same look, title higher; the body holds the claims, the question gets its own box below
pres.defineSlideMaster({ title: "DARK_CLOSE", background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 8.8, h: 0.8, fontSize: 36, bold: true, color: C.background1, align: "left", margin: 0 } } },
            { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.25, w: 8.8, h: 2.35, fontSize: 15, color: C.background2, align: "left", valign: "top", margin: 0 } } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: THEME.colors.accent5, margin: [0, 0, 0, 7.2], valign: "middle" } });
pres.defineSlideMaster({ title: "LIGHT", background: { color: THEME.colors.lt1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.8, fontSize: 27, bold: true, color: C.text1, align: "left", margin: 0, valign: "middle" } } },
            { text: { text: "Supervisor update, 16 Oct 2026", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: THEME.colors.accent5, margin: [0, 0, 0, 7.2], valign: "middle" } });   // insets / anchor as the footer: one baseline

const notes = [];
function N(slide, title, text) { slide.addNotes(text); notes.push({ title, text }); }

// helpers
function card(slide, x, y, w, h, head, body, color, opts = {}) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: head });
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fill: { color }, line: { color }, objectName: head + " dot" });
  slide.addText(head, { x: x + 0.75, y: y + 0.15, w: w - 0.9, h: 0.5, fontSize: opts.hs || 16, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: head + " head" });
  slide.addText(body, { x: x + 0.2, y: y + 0.75, w: w - 0.4, h: opts.bh || h - 0.9, fontSize: opts.bs || 13, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: head + " body" });
}
// big number + label; the label is top-aligned so one-line and two-line labels start on the same line
function stat(slide, x, y, w, big, label, color, o = {}) {
  const bh = o.bh || 0.9;
  slide.addText(big, { x, y, w, h: bh, fontSize: o.bfs || 40, bold: true, color, margin: 0, isTextBox: true, align: "left", objectName: "stat " + label });
  slide.addText(label, { x, y: y + bh, w, h: o.lh || 0.7, fontSize: o.lfs || 13, color: C.text2, margin: 0, isTextBox: true, align: "left", valign: "top", objectName: "label " + label });
}
function table(slide, rows, x, y, w, colW, opts = {}) {
  const data = rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: { bold: i === 0 || j === 0, color: i === 0 ? THEME.colors.lt1 : THEME.colors.dk1, fill: { color: i === 0 ? THEME.colors.dk1 : (i % 2 ? THEME.colors.lt1 : THEME.colors.lt2) }, fontSize: opts.fs || 13, align: j === 0 ? "left" : (opts.alignBody || "center"), valign: "middle" } })));
  slide.addTable(data, { x, y, w, colW, border: { type: "solid", color: "FFFFFF", pt: 1 }, rowH: opts.rowH || 0.42, margin: opts.margin === undefined ? 0.06 : opts.margin, objectName: opts.name || "table" });
}
// bulleted runs: items are strings or [boldLead, rest]
function bulletRuns(items, opts = {}) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const para = { bullet: { indent: 14 }, paraSpaceAfter: opts.space === undefined ? 4 : opts.space };
    if (Array.isArray(it)) {
      runs.push({ text: it[0], options: { ...para, bold: true } });
      runs.push({ text: it[1], options: { breakLine: !last } });
    } else {
      runs.push({ text: it, options: { ...para, breakLine: !last } });
    }
  });
  return runs;
}
// numbered item (circle + bold head + body), same motif as the "what I did" slide
function numberedItem(slide, x, y, w, num, head, body, color, bodyH) {
  slide.addShape(pres.ShapeType.ellipse, { x, y: y + 0.05, w: 0.5, h: 0.5, fill: { color }, line: { color }, objectName: "open " + num });
  slide.addText(num, { x, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "open num " + num });
  slide.addText(head, { x: x + 0.7, y, w: w - 0.7, h: 0.33, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "open head " + num });
  slide.addText(body, { x: x + 0.7, y: y + 0.36, w: w - 0.7, h: bodyH, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "open body " + num });
}

// before/after figures (pixel size measured from the files; aspect ratio preserved)
const FIG_DIR = require("path").join(__dirname, "../figures/");
const FIG_PX = { w: 2394, h: 872 };            // 3 panels of 790 px + 2 gaps of 12 px
function figureSlide(slide, file, x, y, w, captions, capH = 0.4) {
  const h = w * FIG_PX.h / FIG_PX.w;
  slide.addImage({ path: FIG_DIR + file, x, y, w, h, altText: "Ground truth, before (v5) and after (fix) meshes on a synthetic holdout image", objectName: "figure " + file });
  const s = w / FIG_PX.w;
  captions.forEach((cap, i) => {
    slide.addText(cap, { x: x + i * 802 * s, y: y + h + 0.06, w: 790 * s, h: capH, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "caption " + i });
  });
  return y + h;
}
// same left edge and width as the figure (x 1.0, w 8.0)
function footnote(slide, lead, text, y, h) {
  slide.addText([{ text: lead, options: { bold: true, color: THEME.colors.dk1 } }, { text }],
    { x: 1.0, y, w: 8.0, h, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "figure footnote" });
}

// 1 title
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Overview" });
// manual break after the colon, so "this week" is not split across the two title lines
s.addText([{ text: "Multi-HMR with Anny:", options: { breakLine: true } }, { text: "what this week found" }], { placeholder: "title" });
// byline: smaller, more space above and a muted grey, so it does not read as a fifth finding
// (9AA5B4, not the footer grey 7B8794: that one is 4.3:1 on the dark background, this one 6.4:1)
s.addText(["One hidden depth bug found and partly fixed.", "3DPW labels: joints and surface pass, three defects.", "Neck: two bones turned half a turn; a pilot fix passes (one run).", "Fingers: half the error was two flipped bones. Children: open."].map(t => ({ text: t, options: { breakLine: true } })).concat([{ text: "Saad Najib, supervisor update, 16 October 2026", options: { fontSize: 14, paraSpaceBefore: 14, color: "9AA5B4" } }]), { placeholder: "body" });
N(s, "Multi-HMR with Anny: what this week found", "Since last Friday I measured each stage of the pipeline separately. I will show what is not the cause, how the 3DPW labels are made, how well they match 3DPW and three defects I found in them, what I found in the neck and a first fix for it, what I measured about the fingers and the children, one depth bug nobody was looking for and its fix, and what is still open. I will say clearly what is measured and what is only a guess.");

// 2 where we started
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Overview" });
s.addText("Where we started: three visible failures", { placeholder: "title" });
const weekRuns = (before, week) => [{ text: before + "\n\n" }, { text: "This week: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: week }];
card(s, 0.5, 1.4, 2.9, 3.3, "Neck", weekRuns("Necks looked too long or bent wrong in the overlays.\n\nThesis assumption so far: a rig or body-model problem.", "the rig is fine. v5's neck02 and head bone are half a turn off; neck skin half as thick. A pilot fix restores it (one run)."), THEME.colors.accent1);
card(s, 3.55, 1.4, 2.9, 3.3, "Fingers", weekRuns("Fingers came out claw-like; demos already zero them.\n\nAssumption so far: a hand-resolution or hand-branch problem.", "test-time zoom and a loss rebalance did not help; half the error was two flipped bones."), THEME.colors.accent2);
card(s, 6.6, 1.4, 2.9, 3.3, "Children", weekRuns("Children reconstructed with adult proportions.\n\nAssumption so far: age is not learned.", "child-like on training images, adult on real photos."), THEME.colors.accent6);
N(s, "Where we started: three visible failures", "These are the three failures we discussed last time. Until this week they were explained by guesses: a rig problem, too few pixels on the hands, or a shape head that does not learn age. None of these guesses had been tested on its own. That is what I did first. The last line on each card is what this week showed; the details follow. For the neck, two bones in the v5 models, neck02 and the head bone, are turned half a turn, and the neck skin comes out about half as thick. A first fix, in one run, restores it; holdout renders of the correction alone, before the fine-tune, show normal necks; real photos are still to check. For the fingers, about half of the error came from two flipped hand bones.");

// 3 what we did
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Overview" });
s.addText("What I did: test each stage on its own", { placeholder: "title" });
const steps = [["1", "Ground truth", "Render the dataset's own parameters, no network. Does the label sit on the person?"],
               ["2", "Body model", "Sweep age; move one neck or finger joint at a time. Does the mesh react correctly?"],
               ["3", "Training signal", "Do all prediction heads receive gradients? Does the mesh respond to them?"],
               ["4", "Predictions vs truth", "Fixed 100-image holdout plus a fixed set of adults, children, hands, hard necks."],
               ["5", "Shape swap", "Keep the predicted pose, put in the true shape. How much error is shape?"],
               ["6", "Overfit test", "Can 40 images be memorised? Checks the optimiser only; not a test result."]];
steps.forEach((st, i) => { const col = i % 2, row = Math.floor(i / 2); const x = 0.5 + col * 4.65, y = 1.35 + row * 1.25;
  s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.12, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "step " + st[0] });
  s.addText(st[0], { x, y: y + 0.12, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "num " + st[0] });
  s.addText(st[1], { x: x + 0.7, y, w: 3.8, h: 0.35, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "head " + st[0] });
  s.addText(st[2], { x: x + 0.7, y: y + 0.36, w: 3.8, h: 0.8, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, objectName: "body " + st[0] }); });
N(s, "What I did: test each stage on its own", "I built six checks, each testing one stage. I use a fixed holdout of 100 images that were never used in training, and a fixed set of test cases: 40 training images with adults, children, visible hands and hard necks, and 15 held-out images with no children. Most numbers in this talk come from these sets with the same settings.");

// 4 what is not the cause
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Overview" });
s.addText("What is not the cause: the structural checks pass", { placeholder: "title" });
// the fourth item passes with warnings: amber "!" instead of the green tick
const oks = [["Ground truth decodes correctly", "AnnyOne labels render onto the people; camera matches on 100% of images (3DPW labels: slides 5–6).", "ok"],
             ["Body model is fine", "All 16 single-joint moves hit the intended part. Age 0 to 1: skeletal height 0.53 m to 1.54 m.", "ok"],
             ["Gradients reach every bone and shape output", "All 75 active bones and all 11 shape outputs get gradient; the mesh responds.", "ok"],
             ["Optimiser works for shape and 3D joints, with two warnings", "40 training images: error falls 36%; holdout error rises, as expected when memorising. But the translation and 2D losses rose while memorising (1.67 → 1.89; 58 → 75), probably the first sign of the depth bug (slide 11). The summed rotation loss over all bones barely moved (44.6 → 40.8).", "warn"]];
oks.forEach((o, i) => { const y = 1.3 + i * 0.9; const warn = o[2] === "warn"; const col = warn ? C.accent2 : C.accent3;
  s.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + 0.1, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col }, objectName: "ok " + i });
  s.addText(warn ? "!" : "✓", { x: 0.5, y: y + 0.1, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "tick " + i });
  s.addText(o[0], { x: 1.2, y, w: 8.3, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "okh " + i });
  s.addText(o[1], { x: 1.2, y: y + 0.38, w: 8.3, h: warn ? 0.66 : 0.45, fontSize: 13, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "okb " + i }); });
N(s, "What is not the cause: the structural checks pass", "This slide changes the story. The rig, the decoding and the body model are correct, and gradients reach every bone and every shape output. The one exception: two of the three camera outputs are unused and get no gradient. The overfit test shows the optimiser works for shape and 3D joints: on 40 training images the error falls by 36 percent. It has two warnings. First, even while the images were memorised, the translation loss and the 2D losses went up, from 1.67 to 1.89 and from 58 to 75. The network could not satisfy the position targets and the 2D projection at the same time. That was probably the first sign of the depth bug on slide 11; I have not repeated the overfit test with the fix. Second, the rotation loss barely moved, from 44.6 to 40.8. That is a sum over all 75 bones, so it does not isolate the neck; it may be related to the neck flip on slide 7, but that is a guess. So the three failures are not rig or decoding problems, which is how the thesis described them until now, but training can still get stuck. One caveat on the ground-truth check: people who fill the frame lose joints at the image border. That is truncation, not a decoding error. This check covered the AnnyOne labels only; the 3DPW labels come on the next two slides.");

// 5 3DPW conversion: pipeline, native shapes only (boxes in code order: the camera step runs after the fit)
pres.addSection({ title: "3DPW labels" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "3DPW labels" });
s.addText("3DPW to Anny: how the labels are made", { placeholder: "title" });
{
  const flow = [["1", "3DPW ground truth", "24 SMPL joints, camera, gender"],
                ["2", "Pick frames", "valid frames, then every 3rd per person"],
                ["3", "Fit Anny per frame", "match the 24 joints in 3D (9 torso joints via fixed offsets)"],
                ["4", "One body per person", "median shape, then fit the pose again"],
                ["5", "World → camera", "only the root bone changes"],
                ["6", "Label + loader", "drop fits > 80 mm; one file per image; 30% of each batch"]];
  const bw = 1.25, gap = 0.3, by = 1.3, bh = 1.1;
  flow.forEach((st, i) => {
    const x = 0.5 + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: by, w: bw, h: bh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "flow box " + st[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + bw / 2 - 0.2, y: by + 0.1, w: 0.4, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "flow dot " + st[0] });
    s.addText(st[0], { x: x + bw / 2 - 0.2, y: by + 0.1, w: 0.4, h: 0.4, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "flow num " + st[0] });
    s.addText(st[1], { x: x + 0.06, y: by + 0.55, w: bw - 0.12, h: 0.5, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "flow head " + st[0] });
    s.addText(st[2], { x: x - 0.04, y: by + bh + 0.08, w: bw + 0.08, h: 0.58, fontSize: 11, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "flow line " + st[0] });
    if (i < flow.length - 1) s.addShape(pres.ShapeType.line, { x: x + bw + 0.05, y: by + bh / 2, w: gap - 0.1, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "flow arrow " + st[0] });
  });
  // what the fit matches, what it may change, what it never changes
  const what = [["Matched", "24 joint positions per frame, from 3DPW's own male or female SMPL. SMPL rotations, body-shape values and the mesh surface are not used.", C.accent1],
                ["Free", "Root position and rotation; 18 of the 23 mapped bones (e.g. SMPL neck → neck01). The 5 end bones get no signal: head, finger2-1, toe1-1 (L/R). Height and proportions, one per person.", C.accent3],
                ["Fixed", "Gender from 3DPW (reversed: next slide); age 0.55; the other 7 shape values 0.5. By the code, 144 of 163 bones stay at rest: neck02/03, head, jaw, fingers, twist bones.", C.accent2]];
  what.forEach((w, i) => {
    const x = 0.5 + i * 3.05, y = 3.12, cw = 2.9, ch = 1.66;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "fit " + w[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.17, y: y + 0.15, w: 0.3, h: 0.3, fill: { color: w[2] }, line: { color: w[2] }, objectName: "fit dot " + w[0] });
    s.addText(w[0], { x: x + 0.6, y: y + 0.1, w: cw - 0.75, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "fit head " + w[0] });
    s.addText(w[1], { x: x + 0.17, y: y + 0.56, w: cw - 0.34, h: ch - 0.66, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "fit body " + w[0] });
  });
}
s.addText([{ text: "Code: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "scripts/data/smpl_to_anny.py v2.2 (steps 1–4), build_threedpw_anny_labels.py (5–6), datasets/threedpw_anny.py + train.py (6)." }],
  { x: 0.5, y: 4.86, w: 9.0, h: 0.26, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "flow code" });
N(s, "3DPW to Anny: how the labels are made", "One more ground-truth check, now for 3DPW. 3DPW gives, per frame, 24 SMPL joint positions, the camera and the gender. I keep the valid frames and take every third one per person. Then Anny is fitted frame by frame with gradient descent, so that its 24 joints land on the SMPL joints. The two skeletons are different, so first each SMPL joint is assigned one Anny bone; the hip and spine choice is checked on the rest pose, and for nine torso joints, where the rigs differ, the fit aims at a fixed offset from the bone, not at the bone itself. Only positions are matched: SMPL rotations, shape values and the surface are not used. Each person then gets one body, the median over the frames, and the pose is fitted again. Moving into the camera changes only the root bone. When the label files are written, fits with more than 80 millimetres joint error, people less than 30 percent visible and boxes under 40 pixels are dropped. Only the root and 18 bones move. The head, finger and toe end bones are mapped, but nothing below them is matched, so they get no signal. Age is fixed at 0.55. By the code, 144 of the 163 bones stay at rest, including neck02, neck03, the head and the fingers. The check on the next slide confirms this for every person it sampled. So in these labels the head turns only through neck01.");

// 6 3DPW conversion: measured result of slurm/diagnostics/check_3dpw_conversion.sh, cluster job 3575369
// (log: check_3dpw_conversion_3575369.out). Every number below is from that log: V1 body joints, V2 surface, V3 2D,
// V4a/b, V5 loader, I4 all fits; I1 gender refit, I3 two-person coverage, I4 expected vs indexed label images; I2 head.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "3DPW labels" });
s.addText("3DPW labels: joints and surface pass, three defects", { placeholder: "title" });
{
  const BUG = "FBE9E7";                        // light red tint: the defects card
  const cols = [["Measured: pass", C.accent3, C.background2, [
                   ["Body joints: ", "median 18.7\u00A0mm (p95 24.5), below the ~26 mm cited for 3DPW's own joints."],
                   ["Surface vs SMPL: ", "median 18.9 / 18.3\u00A0mm, nearest point, so a lower bound; worst region torso, 28 mm."],
                   ["2D vs OpenPose: ", "12.4 px median, only +1.1 px over raw 3DPW (11.3 px)."],
                   ["Left/right, mirror: ", "100% pass. The loader passes labels unchanged."],
                   ["All 87 fits: ", "22,915 frames; mean joint error per frame: median 18.2 mm, max 70.9 mm."]]],
                ["Label defects", C.accent4, BUG, [
                   ["Gender reversed: ", "Anny 0 = male, but 3DPW men get 1, women 0. Refit of 73 male samples, gender flipped: joints better for 72 (19.2 \u2192 15.3 mm), surface for 68. Female (31): mixed, surface better for 27, joints for 8."],
                   ["Two-person images: ", "in 407 of 960, one actor (valid in 3DPW) has no label: not fitted on that frame, mostly the per-person frame step (slide 5)."],
                   ["Labels missing: ", "6 train sequences, part of captureSelfies_00, val basketball_01. Rebuild: same counts; likely missing image files."]]],
                ["Limits and grades", C.accent5, C.background2, [
                   ["Head at rest: ", "neck02/03, head, jaw at rest in all 2,535 checked people; only neck01 turns."],
                   ["Head turn vs chest: ", "20\u00B0 in SMPL, 9.5\u00B0 in the labels (median), lost mainly in test."],
                   ["Overall FAIL: ", "limb-length ratios (forearm 0.879, thigh 1.119 Anny/SMPL); FAIL in test only, train PASS; cause unknown. Elbow, wrist 14\u201318\u00A0mm."],
                   ["WARN: ", "3.2% of people have a joint >\u00A0100\u00A0mm (worst: hips). Train alone: 5.2%, FAIL."]]]];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 3.05, y = 1.25, cw = 2.9, ch = 3.5;   // same card grid as slide 5
    s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: c[2] }, line: { color: c[2] }, rectRadius: 0.08, objectName: "ev " + c[0] });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.17, y: y + 0.15, w: 0.3, h: 0.3, fill: { color: c[1] }, line: { color: c[1] }, objectName: "ev dot " + c[0] });
    s.addText(c[0], { x: x + 0.6, y: y + 0.1, w: cw - 0.75, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "ev head " + c[0] });
    s.addText(bulletRuns(c[3], { space: 4 }), { x: x + 0.15, y: y + 0.55, w: cw - 0.3, h: ch - 0.62, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "ev body " + c[0] });
  });
}
s.addText([{ text: "Source: ", options: { bold: true, color: THEME.colors.dk1 } }, { text: "slurm/diagnostics/check_3dpw_conversion.sh, job 3575369; 2,535 people in 2,080 sampled images (40 per sequence, 52 sequences)" }],
  { x: 0.5, y: 4.86, w: 9.0, h: 0.26, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "check source" });
N(s, "3DPW labels: joints and surface pass, three defects", "The finished labels had never been checked end to end against 3DPW. A cluster check has now done that. Joints, surface and 2D pass. Body joints are 18.7 millimetres off at the median, below the about 26 millimetres cited for 3DPW's own joints. The surface is about 19 millimetres off; that is a nearest-point distance, so a lower bound. But the labels have three defects. First, gender is reversed: with the other gender, the joint error dropped for 72 of 73 male samples. Second, in 407 of 960 two-person images one person has no label, because that person was not fitted on that frame. Third, labels are missing for 6 train sequences, part of a seventh, and one validation sequence. The fits exist, but a rebuild gave the same counts. So the image files of those sequences are likely missing on the cluster; I have not checked that yet.\n\nIf asked:\n- Right column: by design, the head does not turn on its own. The overall FAIL comes only from the limb lengths, and they fail only in the test split; I do not know the cause yet. A few hip outliers give a warning overall, and a FAIL in the train split alone.\n- Gender: for the 31 female samples the gender result is mixed. The surface test cannot see a wrong gender; that is why there is a separate refit test. In the script's own synthetic test, a label fitted with the wrong gender still stayed at 7 to 10 millimetres from the surface. Anny's male end is 0: at gender 0 the rest body is 1.753 metres tall, at 1 it is 1.632 metres. In the checked images, 1,801 people are labelled male and 734 female; these count people in images, not different persons. All men have the label 1.000, all women 0.000. The refit used 2 people per sequence, 104 samples, so 73 and 31 are samples, not different persons. For the men, the refit with flipped gender lowered the median joint error from 19.2 to 15.3 millimetres, and the surface got closer for 68 of 73. For the women, the flip made the joints worse for most of them, 13.7 to 16.3 millimetres, but the surface closer for 27 of 31.\n- Missing labels: train has 4,451 of 6,437 expected label images, validation 2,458 of 2,777. The rebuild, job 3579266, gave the same image counts: train 4,451 (5,183 people), validation 2,458, test 9,722, 16,631 label files in total; the same six train sequences and basketball_01 at zero, captureSelfies_00 at 319. The label builder skips every frame whose image file is missing, without a warning. So the likely cause is an incomplete unzip of the 3DPW images on the cluster; counting the files in each of those image folders will show it. My earlier guess, that the builder ran before the fits finished, was wrong. All 162 frames of outdoors_crosscountry_00 are dropped by the visibility filter, under 30 percent visible; the fits are fine, and I have not yet checked why the whole sequence counts as not visible.\n- Joints: the 18.7 millimetres are measured at the points the fit aims at. The bare bone heads that train.py uses are 32.5 millimetres off, mostly a fixed torso rig gap, not a fit error. The train split alone: body joints 20.2 millimetres, just over the 20 millimetre line, so WARN; and 5.2 percent of people with a joint over 100 millimetres, so FAIL. The script marks the 26 millimetres for 3DPW as likely, from von Marcard et al., 2018; I have not checked the paper.\n- 2D and left/right: in 2D the labels are only 1 pixel worse than raw 3DPW, and left and right were correct in every checked person.\n- Limb lengths: the ratios are forearm 0.879 and thigh 1.119 overall. Train passes, with 0.965 and 1.035; only test fails. All checked test people are men labelled 1, so a link to the gender defect is possible, but not shown.\n- Head turn: train and validation keep most of it, 18.5 against 21.2 degrees and 19.1 against 19.3; the loss is mainly in test, 6.3 against 20.0.");

// 7 neck: the cause (per-bone dumps: diag_neck_bones_3573518 / _3573519, exp5_neck_fix_3573520 / _3573521; "v5 line" =
// 8 checkpoints: frozen_bb_v2, v3, v4, v5, v9, long_pc_dist5, exp5 L1, exp5 geodesic) and the exp6 pilot fix
// (exp6_bone_fix_3579014, run anny_s2_shape_v5_00099_n200_s0, final SUMMARY table: baseline | zero-training |
// fine-tuned | control = exp5 L1 recipe without the fix). All V1_eval_masked, 100-image holdout, matched people.
pres.addSection({ title: "Findings" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings" });
s.addText("Neck: two half-turned bones, and a pilot fix", { placeholder: "title" });
{
  const BAD = THEME.colors.accent4, HI = "D3ECEE";   // red = flipped or failed; teal = the fine-tuned fix (as on slide 13)
  // [label, v5 before, fix with no training, fix + 2-epoch fine-tune, control, columns shown in red]
  const rows = [["neck02", "175.6°", "4.5°", "4.7°", "175.7°", [1, 4]],
                ["head", "177.9°", "12.4°", "11.2°", "177.9°", [1, 4]],
                ["neck thickness", "0.46", "1.00", "1.00", "0.46", [1, 4]],
                ["neck-skin error", "27.3 mm", "10.8 mm", "10.7 mm", "27.9 mm", [1, 4]],
                ["body PVE", "70.5 mm", "71.3 mm", "68.8 mm", "76.9 mm", []],
                ["head direction", "12.0°", "19.1°", "14.4°", "13.8°", [2]]];
  const base = { fontSize: 12, valign: "middle", margin: [0.02, 0.06, 0.02, 0.06] };
  const head = ["holdout, 100 images", "v5 before", "fix, no training", "fix + 2-epoch fine-tune", "control (no\u00A0fix)"].map((t, c) => ({ text: t, options: { ...base, fontSize: 11, bold: true,
    color: THEME.colors.lt1, fill: { color: c === 3 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center" } }));
  const data = [head].concat(rows.map((r, i) => [0, 1, 2, 3, 4].map(c => ({ text: r[c], options: { ...base,
    bold: c === 0 || r[5].includes(c), color: r[5].includes(c) ? BAD : THEME.colors.dk1, align: c === 0 ? "left" : "center",
    fill: { color: c === 3 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) } } }))));
  s.addTable(data, { x: 0.5, y: 1.3, w: 5.3, colW: [1.45, 0.9, 0.95, 1.05, 0.95], rowH: [0.42, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27], border: { type: "solid", color: "FFFFFF", pt: 1 }, objectName: "neck table" });
  s.addText("Bones: median error vs truth, relative to the parent bone. Thickness: neck-skin radius, predicted / true. Neck-skin error: centred at neck01. Head direction: error of the whole head's direction.",
    { x: 0.5, y: 3.42, w: 5.3, h: 0.6, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "neck table note" });
  s.addText(bulletRuns([
      ["Cause, measured: ", "in all 8 v5-line checkpoints, neck02 and head sit half a turn off. They cancel, so PVE hid it, but the neck skin is half as thick."],
      ["Fix (exp6): ", "one constant turn of about 179° per flipped bone at the network output, estimated on 200 training images. It also turns back two hand bones (slide 9)."],
      ["Fix, no training: ", "head direction fails its 15° limit; likely a per-person head offset."],
      ["Fix + 2-epoch fine-tune: ", "passes the fine-tune checks, each within a tolerance of its control."]], { space: 5 }),
    { x: 6.05, y: 1.3, w: 3.45, h: 2.92, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck text" });
  // caveat, open, next: one line each, dot colour = status (amber = caveat, grey = unknown, teal = next)
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.3, w: 9.0, h: 0.82, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "neck cause box" });
  [["Caveat: ", "one run, no fixed seed (identical runs differ by up to 7.6 mm PVE), so no PVE gain is claimed.", C.accent2],
   ["Open: ", "why the bones flipped; real photos. Holdout renders of the fix with no training: next slide.", C.accent5],
   ["Next: ", "real-photo demo of the fine-tuned fixed checkpoint; a longer fine-tune with the fix.", C.accent1]].forEach((l, i) => {
    const y = 4.35 + i * 0.25;
    s.addShape(pres.ShapeType.ellipse, { x: 0.68, y: y + 0.06, w: 0.13, h: 0.13, fill: { color: l[2] }, line: { color: l[2] }, objectName: "neck cause dot " + i });
    s.addText([{ text: l[0], options: { bold: true, color: THEME.colors.dk1 } }, { text: l[1] }],
      { x: 0.92, y, w: 8.45, h: 0.25, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "neck cause " + i });
  });
}
N(s, "Neck: two half-turned bones, and a pilot fix", "In all eight v5-line checkpoints, neck02 and the head bone are about half a turn off. The two half-turns cancel, so the head still points the right way and PVE never showed it, but the neck skin comes out half as thick. I tested a fix: one constant turn per flipped bone at the network output, estimated on 200 training images. On the holdout, the fix alone turns both bones back; neck thickness goes from 0.46 to 1.00. But head direction gets worse, 19.1 degrees, likely because the head's offset differs per person. After a two-epoch fine-tune, it passes the fine-tune checks, each within a tolerance of its control; head direction is 14.4, still worse than 12.0 before. One run, no fixed seed: I claim no PVE gain, only no sign of a PVE cost, 68.8 against 76.9 for the control. The next slide shows the correction alone, before the fine-tune; real photos are not checked yet.\n\nIf asked: After the fine-tune, neck02 is 4.7 degrees and the thickness 1.00. The flip is there from the first training stage, frozen_bb_v2, and in v3, v4, v5, v9 and every pilot checked. A two-epoch geodesic loss barely moved it, 175.6 to 175.0 degrees. Its origin is not known: the singular start value of the rotation output would start a bone with a tilt, like v5's left index base, not this twist. v10 has no neck flip. The constants are 178.8 degrees for neck02 and 179.2 for the head. Before the test, the script's guard predicted the head-direction rise, 8.4 to 17.9 degrees on the training sample: a compensated chain, since the head's offset differs per person and one constant cannot undo it. Without training, the head-and-neck surface error also rose, 52.3 to 63.1 millimetres; after the fine-tune it is 51.8. The fine-tune checks, against the control: corrected bones under 15 degrees, thickness at least 0.9, PVE at most 8 millimetres higher, recall at most 2 points lower, head direction at most 3 degrees worse. Recall passed by only 0.3 points, 89.4 against a limit of 89.1. Against the control: PVE 68.8 against 76.9, PA-PVE 50.9 against 55.2, recall 89.4 against 91.1, head direction 14.4 against 13.8. Against v5 before: PVE 68.8 against 70.5, PA-PVE 50.9 against 53.3, recall 89.4 against 90.2. Identical configs differ by up to 7.6 millimetres PVE, so none of these PVE differences is certain. Last week's group errors, 63 degrees for the neck and 90 for the head, are now 5.9 and 6.7. The script wrote 12 side-by-side panels, without and with the correction alone; the next slide shows two of them. Next: a real-photo demo with the fine-tuned checkpoint, which carries the fix, and a longer fine-tune with the fix.");

// 8 neck, visual check: 2 of the 12 exp6 compare panels (results/experiments/exp6_bone_fix/anny_s2_shape_v5_00099_n200_s0/
// compare/, holdout 579860-579871; panel top row = no fix, bottom = fix, zero training, job 3579014), regrouped by the
// student into docs/figures/neck_before_after_v5.jpg (1254 x 1118 px): left column v5 as trained, right column with the
// correction; top row holdout 579862, bottom row 579863. Here the four panels are cropped out of that one file (srcRect,
// so the pptx holds the file once and unchanged) and laid out in one row, before | after per image: all four panels get
// the same height, about 1.36 times the size they had in the 2 x 2 grid. The baked-in label strip (y 0-44) is cropped
// off; slide-level header bars replace it, in the red / teal of the depth figures on slides 12 and 18. Each bar has a
// second line, "as trained" / "no training": teal marks the fine-tuned fix on slide 7, and these After panels are not it.
// Panel pixel boxes, measured from the file, inset 1 px against JPEG fringe: top row x 0-619 / 634-1253, y 50-569;
// bottom row x 64-554 / 698-1188, y 584-1103.
// Yellow rings mark the necks (centres measured from the file, left column; the right column is the same + 634 px in x):
// 579862 blue person (155, 232) and orange person (480, 335), 579863 blue person (343, 714). 579863's orange person bends
// the head forward, so the neck is foreshortened there and gets no ring.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings" });
s.addText("Neck: before and after the correction", { placeholder: "title" });
{
  const NECK_PX = { w: 1254, h: 1118 };
  const panels = [{ img: "579862", after: false, box: [1, 51, 618, 518] }, { img: "579862", after: true, box: [635, 51, 618, 518] },
                  { img: "579863", after: false, box: [65, 585, 489, 518] }, { img: "579863", after: true, box: [699, 585, 489, 518] }];
  const gIn = 0.06, gOut = 0.3, barH = 0.5, x0 = 0.5, barY = 1.2;       // gap inside a pair, gap between the pairs
  const sc = (9.0 - 2 * gIn - gOut) / panels.reduce((a, p) => a + p.box[2], 0);   // in per px: the row is 9.0 in wide
  const ph = 518 * sc, imgY = barY + barH;                               // 2.01 in tall, ends at 3.71
  const necks = { "579862": [[155, 232], [480, 335]], "579863": [[343, 714]] }, ringD = 0.26;
  let x = x0;
  const pairX = {};
  panels.forEach((p, i) => {
    const [bx, by, bw, bh] = p.box, w = bw * sc;
    if (!p.after) pairX[p.img] = x;
    s.addText([{ text: p.after ? "After (correction)" : "Before (v5)", options: { fontSize: 16, bold: true, breakLine: true } },
               { text: p.after ? "no training" : "as trained", options: { fontSize: 12 } }],
      { x, y: barY, w, h: barH, color: THEME.colors.lt1, fill: { color: p.after ? THEME.colors.accent1 : THEME.colors.accent4 }, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "neck header " + i });
    s.addImage({ path: FIG_DIR + "neck_before_after_v5.jpg", x, y: imgY, w: NECK_PX.w * sc, h: NECK_PX.h * sc,
      sizing: { type: "crop", x: bx * sc, y: by * sc, w, h: bh * sc },
      altText: "Holdout image " + p.img + (p.after ? ", with the correction on neck02 and the head, no training: normal neck width" : ", v5 as trained: thin neck"),
      objectName: "figure neck_before_after_v5.jpg " + p.img + (p.after ? " after" : " before") });
    necks[p.img].forEach(([nx, ny], k) => s.addShape(pres.ShapeType.ellipse, {
      x: x + (nx + (p.after ? 634 : 0) - bx) * sc - ringD / 2, y: imgY + (ny - by) * sc - ringD / 2, w: ringD, h: ringD,
      line: { color: THEME.colors.accent6, width: 2 }, objectName: "neck ring " + p.img + (p.after ? " after " : " before ") + k }));
    x += w + (p.after ? gOut : gIn);
  });
  const pairW = { "579862": 2 * 618 * sc + gIn, "579863": 2 * 489 * sc + gIn };
  Object.keys(pairX).forEach(id => s.addText("Synthetic holdout image " + id, { x: pairX[id], y: imgY + ph + 0.06, w: pairW[id], h: 0.22, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck caption " + id }));
  // explicit break per line, so each line ends at a sentence; left-aligned to the row's left edge, like slides 12 and 18
  const lines = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: { ...o, breakLine: i < arr.length - 1 } }));
  // the panels' correction also turns back the two hand bones (metacarpal1.L, finger4-1.R), hence "on the neck";
  // the thickness is the whole 100-image holdout (girth 0.461 -> 1.004), not these two images
  s.addText(lines(["Same model, no training. On the neck, only the output of neck02 and the head is corrected.",
                   "Neck skin thickness on the 100-image holdout: 0.46 → 1.00 of the true value (job 3579014)."]),
    { x: x0, y: imgY + ph + 0.36, w: 9.0, h: 0.4, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "neck figure caption" });
  s.addText([{ text: "Limits: ", options: { bold: true, color: THEME.colors.dk1 } },
             { text: "without fine-tuning the head points 19° off (12° before); the 2-epoch fine-tune brings it to 14°.", options: { breakLine: true } },
             { text: "Bodies are still too small here (separate depth bug). Real photos not checked yet." }],
    { x: x0, y: imgY + ph + 0.84, w: 9.0, h: 0.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "neck figure footnote" });
}
N(s, "Neck: before and after the correction", "These are two of the twelve holdout images. For each image, the left panel is v5 as trained and the right panel is the same model with the correction, no training; on the neck, only neck02 and the head are turned back. Look at the necks, ringed in yellow: on the left they are thin stalks, on the right they have normal width. This matches the thickness 0.46 to 1.00 on slide 7, measured over all 100 holdout images. All twelve images show this wherever a neck is visible. Three limits. Without fine-tuning the head points 19 degrees off, against 12 before; the fine-tune brings it to 14. The bodies are still too small; that is the depth bug. Real photos are not checked yet.\n\nIf asked: the same correction also turns back two hand bones, so the hands differ too; I have not judged them (slide 9).");

// 9 fingers. Before: 3571960 (predicted 17.3, rest pose 15.1, crop 17.2), 3571959 (region-norm 17.5 vs 17.4).
// Bone fix: exp6_bone_fix_3579014 SUMMARY, row "rot fingers": baseline 17.3, zero-training 8.2, fine-tuned 8.3
// (corrected hand bones metacarpal1.L, finger4-1.R). The group error is a per-person mean over the active hand bones.
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings" });
s.addText("Fingers: about half the error was two flipped bones", { placeholder: "title" });
stat(s, 0.5, 1.4, 2.8, "17.3°", "finger error, v5 with predicted fingers", C.accent4);
stat(s, 3.5, 1.4, 2.8, "8.3°", "with the bone fix, fine-tuned 2 epochs (8.2° with no training)", C.accent3);
stat(s, 6.6, 1.4, 2.9, "15.1°", "fingers set to rest pose, before the fix (the index-base flip stays)", C.accent5);
table(s, [["hand size at the network input", "median", "under 2 tokens"], ["full image, 672 px", "2.3 tokens", "41%"], ["person crop (box ×1.3), test time", "4.2 tokens", "0%"], ["children's hands, full image", "0.7 tokens", "80%"]], 0.5, 3.25, 5.6, [2.8, 1.4, 1.4], { fs: 12, rowH: 0.36, name: "hand table" });
s.addText("Table: median hand size (2.3 → 4.2 tokens). Per person the median gain is 1.6×; 26% of people get no magnification, a quarter 2.5× or more.", { x: 0.5, y: 4.74, w: 5.6, h: 0.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "hand table note" });
s.addText([{ text: "Two flipped bones: ", options: { bold: true, paraSpaceAfter: 6 } }, { text: "left index base and finger4-1.R, each about half a turn off. Turning them back halves the error; not yet checked by eye.", options: { breakLine: true } },
             { text: "Test-time zoom: ", options: { bold: true, paraSpaceAfter: 6 } }, { text: "17.2° vs 17.3°; body error nearly doubles.", options: { breakLine: true } },
             { text: "Loss rebalance ", options: { bold: true } }, { text: "(hands 33% of the rotation loss): 17.5° vs 17.4°." }],
  { x: 6.4, y: 3.25, w: 3.1, h: 1.9, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "finger text" });
N(s, "Fingers: about half the error was two flipped bones", "With predicted fingers the finger error is 17.3 degrees; with the fingers left at rest it is 15.1. The new result: about half of the 17.3 came from two hand bones turned half a turn, the left index base and one right finger bone. The same correction as for the neck turns them back: the error falls to 8.2 degrees with no training and 8.3 after the fine-tune. I have not yet looked at the hands in the renders. The rest-pose number still contains the index-base flip, because rest pose does not reset that bone, and I have not measured it with the fix. What I tested before did not help: magnifying the person at test time gave 17.2 against 17.3, and a rebalanced loss gave 17.5 against 17.4 for its control.\n\nIf asked: The finger error is a mean over the hand bones of each person, so two bones at half a turn raise it a lot. The left index base is 180 degrees off on the holdout. The other bone, finger4-1.R, at 178.8 degrees, was seen only on the 200 training images, because the earlier holdout dump did not cover finger bones; with the fix it is 10.2 degrees on the holdout. Hands are small: about two tokens at 672 pixels. Cropping around the person at test time, without retraining, magnifies the hands by 1.6 times on median per person, and by 2.5 times or more for a quarter; 26 percent are not magnified. The table compares median hand sizes, so it shows a slightly larger factor. For the most magnified quarter the finger error went from 17.7 to 17.5. The body error nearly doubles on these crops, because the model never saw such a camera in training. The rebalanced loss cut the hands' share of the rotation loss from about 70 percent, which includes the four-times hand boost, to 33 percent; without the boost it would be 40. So it does not test giving the fingers more weight; more weight and training on larger hand crops are untested. v10 has at least five flipped hand bones. Correcting three of them, with no training, took its finger error from 32.0 to 17.8 degrees and PVE from 74.1 to 69.9 millimetres; PA-PVE went from 55.6 to 51.1, while head direction and recall did not change. Two more, finger2-1.L and finger5-2.L, seen on training images, are not corrected yet; they may be part of why v10's fingers are still at 17.8, but that is a guess. The demos zero the fingers, but rest pose does not reset the hand bases, so the bent index base is likely visible there.");

// 10 children
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Findings" });
s.addText("Children: child-like in training, adult on real photos", { placeholder: "title" });
stat(s, 0.5, 1.4, 2.8, "2% → 10%", "children in supervised training with the balanced sampler: no effect", C.accent5);
stat(s, 3.5, 1.4, 2.8, "0.10", "predicted age of training babies; true 0.015 (child-like, but too old)", C.text2);
stat(s, 6.6, 1.4, 2.9, "0.57–0.67", "predicted age of 5 small people in real photos (sampler and control runs): same as the holdout adults", C.accent4);
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 3.15, w: 9.0, h: 1.95, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "child box" });
s.addText(bulletRuns([
    ["The model can make a child: ", "all eleven shape outputs are trained, none is clamped to an adult; training babies come out child-like, but too old. With the true shape, the 10 training children's error falls 47.8 → 41.6 mm."],
    ["More children did not help in a 2-epoch fine-tune: ", "about 5 times more in training; no change on the 10 training children or on the real photos (no age labels)."],
    ["Best guess: ", "a synthetic-to-real gap, plus an old-age bias even on training children. One 2-epoch test cannot settle it."],
    ["Caveat: ", "every synthetic child number is in-sample; the holdout has no children. Ages are Anny's 0–1 value (holdout adults: mostly 0.4–0.8). Ages are not converted to years, because Anny's calibration and the labels in our code disagree."]], { space: 3 }),
  { x: 0.7, y: 3.2, w: 8.6, h: 1.85, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "child text" });
N(s, "Children: child-like in training, adult on real photos", "I checked in the code that the shape head is complete: all eleven shape values go into the body model and are trained, with no adult default. The model can make a child: for the youngest training people it predicts age 0.10 for a true 0.015, too old but still clearly a child; the holdout adults lie mostly between 0.4 and 0.8. Only 2 percent of training people are children, so I tried a sampler that shows about 10 percent. In a 2-epoch fine-tune nothing changed, not for the training children and not on real photos: five small people still get the age of the holdout adults, 0.57 to 0.67 in the sampler and the control run. That is a short run on few people, so it shows no effect, not that there can be none. My best guess is a gap between synthetic and real images, plus an old-age bias that shows even on training children; neither is tested. The next step is a 10-epoch balanced-sampler run; if that also shows nothing, it goes into the thesis as a limitation. The holdout has no children, so every synthetic child number is in-sample.\n\nIf asked: Anny's own age calibration puts 0.10 at about 2 years and 0.67 at about 16, while the labels in our code assume 0.33 is 11 years and 0.67 is 25; I have not settled which one the dataset follows. The ten children in the training manifest also come out too old, by 0.078 on average. With their true shape, their error falls from 47.8 to 41.6 millimetres, so 6.2 millimetres of it is shape. So I compare with the holdout adults rather than give years: 0.57 to 0.67 is about 14 to 16 years on Anny's calibration and 21 to 25 on our code's labels; either way far too old for people 0.6 to 1.1 metres tall. There are no labelled real children, and the 3DPW labels fix age at one value, 0.55, for everyone (slide 5).");

// 11 the hidden bug
pres.addSection({ title: "Bug" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Bug" });
s.addText("The hidden bug: head anchored, pelvis targeted", { placeholder: "title" });
// diagram: two stick figures
function figure(slide, x, y, label, anchorY, color, note) {
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.45, y, w: 0.5, h: 0.5, fill: { color: C.background2 }, line: { color: C.text2, width: 1.5 }, objectName: label + " head" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.7, y: y + 0.5, w: 0, h: 1.3, line: { color: C.text2, width: 2 }, objectName: label + " body" });
  slide.addShape(pres.ShapeType.rect, { x: x + 0.2, y: y + 0.78, w: 1.0, h: 0.03, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: label + " arms" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.25, y: y + 1.8, w: 0.45, h: 0.9, line: { color: C.text2, width: 2 }, flipH: true, objectName: label + " legL" });
  slide.addShape(pres.ShapeType.line, { x: x + 0.7, y: y + 1.8, w: 0.45, h: 0.9, line: { color: C.text2, width: 2 }, objectName: label + " legR" });
  slide.addShape(pres.ShapeType.ellipse, { x: x + 0.58, y: anchorY - 0.12, w: 0.24, h: 0.24, fill: { color }, line: { color }, objectName: label + " anchor" });
  slide.addText(label, { x: x - 0.4, y: y + 2.8, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: label + " cap" });
  slide.addText(note, { x: x - 0.4, y: y + 3.1, w: 2.2, h: 0.5, fontSize: 11, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: label + " note" });
}
figure(s, 0.9, 1.35, "Model: anchor = head", 1.6, THEME.colors.accent1, "model.py places the mesh by its head joint");
figure(s, 3.6, 1.35, "Training target = pelvis", 3.15, THEME.colors.accent2, "train.py built transl / dist / heatmap / offset at joint 0");
s.addText([{ text: "predicted depth =", options: { fontSize: 16, breakLine: true } }, { text: "0.33 × true + 2.36 m", options: { fontSize: 24 } }],
  { x: 6.5, y: 1.4, w: 3.0, h: 0.8, bold: true, color: C.accent4, margin: 0, isTextBox: true, valign: "top", objectName: "stat depth" });
s.addText("v5: predicted vs true depth; all 222 holdout people too far", { x: 6.5, y: 2.2, w: 3.0, h: 0.6, fontSize: 13, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "label depth" });
s.addText(bulletRuns([
    ["Confirmed: ", "with targets at the head, bodies are drawn much closer to the right size; a same-schedule control does not improve (10 epochs: size 0.59, root error 1.44 m)."],
    ["Size: ", "3D height right, but drawn at 0.63 of true size; head 64 px off."],
    ["Hidden ", "from pelvis-centred PVE and from the box-placed demos. v9 and v10 have it too."]], { space: 4 }),
  { x: 6.5, y: 2.95, w: 3.0, h: 2.15, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "bug text" });
N(s, "The hidden bug: head anchored, pelvis targeted", "This is the main result of the week. The model attaches the mesh to the detected pixel by its head joint and reports the head as its position. The training code built the position, distance, heat-map and offset targets at joint 0, the root joint at the pelvis. Those two can never agree. The cause is confirmed: with targets at the head the bodies are drawn much closer to the right size, and a control run with the same schedule does not improve; after 10 epochs its drawn size is 0.59 and its root error 1.44 metres. The likely mechanism is that the 2D losses solve the conflict by pushing the body farther away, so it is drawn smaller. Measured on the holdout for v5: predicted depth is 0.33 times the true depth plus 2.36 metres, all 222 people too far, and the bodies are drawn at about two thirds of their true size in the image, a size ratio of 0.63. Their 3D height is right, 1.405 against 1.409 metres; they are only placed too far away. Our standard error is pelvis-centred, so it never saw this. That is why the thesis numbers looked normal. The real-photo demos did not show it either, because they place each body from its detection box. The synthetic demo and the trace panels use the model's own depth, and there you can see it, as on the next slide. The bug is in the training code, so v9 and v10 have it too: the same kind of depth fit and size ratio, and v10 is slightly worse.");

// 12 before / after figure (8.0 in wide so the title and the footer keep their usual spacing)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Bug" });
s.addText("Depth fix: bodies drawn much closer to true size", { placeholder: "title" });
let figBottom = figureSlide(s, "depth_before_after_579862.jpg", 1.0, 1.2, 8.0, [
  "Synthetic holdout image 579862, model's own depth (no box placement).",
  "v5, whole holdout: root error 1.21 m, median drawn size 0.63",
  "Fix + depth ×5, 10 epochs, whole holdout: root error 0.54 m, median drawn size 0.84"], 0.4);
footnote(s, "Not fully fixed: ", "near and far people still get similar depths (depth slope 0.38; 1 = perfect, 0 = same depth for everyone), and bodies are still drawn about 16% too small (median). All 12 rendered images show the same change; I picked this one because no ground-truth person is missed. After the fix the blue mesh is 0.95 of its true height (before: 0.76).", figBottom + 0.5, 0.56);
N(s, "Depth fix: bodies drawn much closer to true size", "This is the bug on a synthetic holdout image, using the model's own depth, not the box placement of the real-photo demos. Left is the truth. In the middle, the old model: the bodies are drawn too small, the real man's head sticks out above the mesh, and the mesh floats. On the right, the fix with the depth loss five times stronger, trained for 10 epochs. The blue mesh is now 95 percent of the true height; the woman's mesh is closer to her size but still narrower. I chose this image because no person with a ground-truth mesh is missed, so it is a good case, not a typical one; the arm at the bottom left has no ground-truth mesh, so it is not counted as a miss. On the whole holdout the median drawn size is 84 percent of the truth, so bodies are still about 16 percent too small, and near and far people still get similar depths: in the slope, 1 means perfect and 0 means the same depth for everyone. All 12 holdout images we rendered, in all three prediction settings, show the same change. In 4 of them both models miss people very close to the camera; since both miss them, the fix did not cause it. It is likely a detection problem, but not yet counted. Mesh colours follow detection order.");

// 13 depth fix results: all seven runs. A group band separates the 10-epoch and the 2-epoch fine-tunes;
// the column shown on slide 12 (fix + x5, 10 epochs) is tinted teal
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Bug" });
s.addText("Depth fix: much better size, near vs far still weak", { placeholder: "title" });
{
  const thin = { type: "solid", pt: 1, color: "FFFFFF" }, none = { type: "none" };
  const sepBody = { type: "solid", pt: 3, color: THEME.colors.dk2 }, sepHead = { type: "solid", pt: 3, color: "FFFFFF" };
  const HI = "D3ECEE";                         // light teal tint for the slide 12 column
  const left = c => c === 2 || c === 5, right = c => c === 1 || c === 4;
  const head = ["holdout, 100 images", "v5", "control", "fix", "fix + ×5", "control", "fix", "fix + ×5", "fix + ×20"];
  const body = [["root error, 3D (pelvis)", "1.21 m", "1.44 m", "0.73 m", "0.54 m", "1.11 m", "0.69 m", "0.66 m", "0.53 m"],
                ["depth slope (1 = perfect)", "0.33", "0.34", "0.30", "0.38", "0.33", "0.30", "0.44", "0.43"],
                ["drawn size (1 = right)", "0.63", "0.59", "0.78", "0.84", "0.66", "0.78", "0.80", "0.87"],
                ["head pixel error", "64.5 px", "64.0 px", "7.6 px", "7.2 px", "58.1 px", "9.6 px", "8.5 px", "8.8 px"],
                ["pelvis pixel error", "23.4 px", "24.3 px", "38.5 px", "33.1 px", "23.3 px", "37.3 px", "38.1 px", "31.4 px"],
                ["PVE (pelvis-centred)", "70.5 mm", "72.2 mm", "75.5 mm", "74.8 mm", "72.3 mm", "77.3 mm", "76.8 mm", "75.7 mm"],
                ["recall", "90.2%", "89.4%", "87.0%", "87.8%", "89.8%", "87.8%", "87.8%", "87.4%"]];
  const base = { fontSize: 11, valign: "middle", margin: [0.02, 0.05, 0.02, 0.05] };
  const band = (text, colspan, sepL) => ({ text, options: { ...base, colspan, bold: true, color: THEME.colors.lt1, fill: { color: THEME.colors.dk2 }, align: "center", border: [thin, sepHead, thin, sepL ? sepHead : thin] } });
  const rows = [[{ text: "", options: { ...base, colspan: 2, fill: { color: THEME.colors.lt1 }, border: [none, sepHead, thin, none] } },
                 band("10-epoch fine-tunes", 3, true), band("2-epoch fine-tunes", 4, true)],
                head.map((t, c) => ({ text: t, options: { ...base, bold: true, color: THEME.colors.lt1, fill: { color: c === 4 ? THEME.colors.accent1 : THEME.colors.dk1 }, align: c === 0 ? "left" : "center",
                                                          border: [thin, right(c) ? sepHead : thin, thin, left(c) ? sepHead : thin] } }))];
  body.forEach((r, i) => rows.push(r.map((t, c) => ({ text: t, options: { ...base, bold: c === 0, color: THEME.colors.dk1, align: c === 0 ? "left" : "center",
      fill: { color: c === 4 ? HI : (i % 2 ? THEME.colors.lt2 : THEME.colors.lt1) }, border: [thin, right(c) ? sepBody : thin, thin, left(c) ? sepBody : thin] } }))));
  const cw = 6.2 / 7;
  s.addTable(rows, { x: 0.5, y: 1.2, w: 9.0, colW: [2.0, 0.8, cw, cw, cw, cw, cw, cw, cw], rowH: [0.25, 0.27, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24], margin: [0.02, 0.05, 0.02, 0.05], objectName: "fix table" });
}
s.addText("Fix: targets at the head; ×5 / ×20 = depth loss weight. Controls: root targets, depth ×1, same schedule. No control was run at ×5 or ×20.",
  { x: 0.5, y: 3.45, w: 9.0, h: 0.22, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top", objectName: "fix table note" });
s.addText(bulletRuns([
    ["Not fixed: ", "depth slope 0.30 with the switch alone, 0.38–0.44 with a stronger depth loss; predictions are pulled toward 2.5–2.9 m, cause unknown. With ×5, 10 epochs gave a lower root error than 2 epochs (0.66 → 0.54 m) but no better slope (0.44 → 0.38); with the switch alone neither improved (0.69 → 0.73 m, slope 0.30 → 0.30)."],
    ["Likely cost: ", "every fix run is worse than its control: PVE +2.6 to +4.9 mm (inside noise, ~10 mm), recall −1.6 to −2.4 points (about the 2-point noise limit). The five runs share two controls."]], { space: 3 }),
  { x: 0.5, y: 3.78, w: 9.0, h: 1.34, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top", objectName: "fix text" });
N(s, "Depth fix: much better size, near vs far still weak", "The fix is one switch: build the training targets at the head, the joint the model anchors. Each fix run has a control with the same schedule and the switch off. The root error is the 3D distance between the predicted and the true root joint, at the pelvis. Drawn size is the predicted head-to-pelvis distance in pixels over the true one; 1 is right. With the depth loss five times stronger for 10 epochs, the root error falls from 1.21 to 0.54 metres, and bodies are drawn at 84 percent of their true size. The controls stay at 1.11 to 1.44 metres. But near and far are still not separated: the slope is 0.30 with the switch alone and 0.38 to 0.44 with the stronger depth loss. Predictions are pulled toward about 2.5 to 2.9 metres, and I do not know why. The fix likely has a cost: every fix run is worse than its control, by 2.6 to 4.9 millimetres PVE and 1.6 to 2.4 recall points. That is inside or at the edge of pilot noise, so I call it a likely cost, not a proven one. The twenty-times 10-epoch run has no result yet.\n\nIf asked: The switch is gt_center_joint person_center. The stronger-loss runs have no control at the same weight. The mean signed depth offset fell from 1.08 to 0.39 metres, but after the fix near people are too far and far people too near, and these cancel in that mean, so it flatters the fix. The head now lands within about 7 to 10 pixels, as expected, because the target moved to the head. Twenty times for 2 epochs ties with five times for 10 epochs. The pull toward 2.5 to 2.9 metres is close to the estimated training average; holdout people are nearer, 1.9 metres on average, so this pull becomes a large error here. Longer training is not clearly better: with five times, going from 2 to 10 epochs lowered the root error from 0.66 to 0.54 metres but did not improve the slope; with the switch alone neither improved, the root error went from 0.69 to 0.73 metres, and the control also got worse, from 1.11 to 1.44 metres. Root error has no measured noise floor, so I do not read a trend into this. The PVE gaps, 2.6 to 4.9 millimetres, are inside pilot noise; the recall drops, 1.6 to 2.4 points, are at the edge of what one pilot can detect; and the five fix runs share only two controls, so they are not five independent tests. The pelvis 2D error clearly got worse, from about 23 pixels to between 31 and 39 pixels; probably because the body is still drawn too small while the head is now pinned, but I have not checked that.");

// 14 experiments: results
pres.addSection({ title: "Next" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Next" });
s.addText("Four experiments: results", { placeholder: "title" });
table(s, [["experiment", "what it changes", "success rule", "result"],
          ["1  depth anchor", "targets at the head joint (+ depth loss ×5 or ×20)", "depth slope toward 1; PVE ≤\u00A071.5\u00A0mm, recall\u00A0≥\u00A089.2%", "Rule not met: PVE and recall miss in every run, controls too (×5, 10 epochs: 74.8 mm, 87.8%); slope 0.30–0.44, far from 1. Outside the rule, clearly better: root error 1.21 → 0.54 m, size 0.63 → 0.84"],
          ["2  children", "age-balanced sampler", "child error on the manifest, adults unchanged", "No effect in a 2-epoch fine-tune: training children and real photos unchanged. Pilot and control both use age loss ×3, height ×2"],
          ["3  neck", "loss on the global head orientation", "global head error below 12°", "11.3° passes, but the controls give 11.7–13.6° (inside the ~3° noise). Safety limit (PVE ≤ 71.5 mm, recall ≥ 89.2%) just missed: 71.8 mm, 89.0%. All five control runs miss it too."],
          ["4  fingers", "region-normalised rotation loss", "finger error vs rest pose (about 15°)", "No effect: 17.5° vs 17.4° for the control"]],
      0.5, 1.2, 9.0, [1.2, 1.75, 1.75, 4.3], { fs: 11, rowH: [0.3, 0.62, 0.45, 0.45, 0.45], alignBody: "left", name: "exp table" });
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 4.17, w: 9.0, h: 0.95, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "noise box" });
s.addText([{ text: "Pilot noise. ", options: { bold: true } }, { text: "No fixed random seed: two identical runs differed by 7.6 mm PVE and 1.9° head error, so one pilot only detects about 10 mm PVE, 3° head error or 2 recall points. Restarting the fine-tune alone costs about 2.4 mm (controls average 72.9 vs 70.5 mm). Root error has no measured noise floor, but every fix run (0.53–0.73 m) is below every control (1.11–1.44 m). All runs: same holdout and manifest, helper bones masked." }],
  { x: 0.7, y: 4.2, w: 8.6, h: 0.89, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle", objectName: "noise text" });
N(s, "Four experiments: results", "All four experiments are done, each next to a control with the main switch off. Depth: by its own success rule the experiment failed, because every run, the controls too, misses the limit of 71.5 millimetres PVE and 89.2 percent recall. The 10-epoch run with the stronger loss has 74.8 millimetres and 87.8 percent. The slope is 0.30 with the switch alone and 0.38 to 0.44 with the stronger depth loss, against 0.33 for v5: far from 1. What clearly improved is outside the rule: the root error went from 1.21 to 0.54 metres and the drawn size from 0.63 to 0.84. I say that openly instead of changing the rule afterwards. Children, neck and fingers: no measurable effect. The neck loss gave 11.3 degrees of head error, which passes the rule of below 12, but the two controls gave 11.7 and 13.6, so that is inside the noise. With no fixed seed, one pilot only detects about 10 millimetres PVE, 3 degrees or 2 recall points. No new architecture, because the diagnostics do not justify one.\n\nIf asked: The stronger age loss and the stronger depth loss were not separately controlled: both children arms used the same age and height weights, and the depth runs at five and twenty times have no control at the same weight. Depth PVE: 74.8 for the 10-epoch run with the stronger loss, 72.2 for the 10-epoch control, 70.5 for v5. Two identical runs differed by 7.6 millimetres PVE and 1.9 degrees of head error. The slope part of the depth rule only asks for any move toward 1 with a smaller intercept: it passed for the three stronger-loss runs and even for the 2-epoch control, and failed with the switch alone. All runs used the same 100-image holdout and manifest, with helper bones masked. The neck experiment has the same safety limit, PVE at most 71.5 millimetres and recall at least 89.2 percent. The neck run just missed it, with 71.8 millimetres and 89.0 percent, but all five control runs miss it too, mostly because of the restart cost; its PVE is even 4.5 millimetres better than its own control, which is also inside the noise. Its neck bones stayed at 62.5 degrees. For the fingers, the rest-pose floor is 15.2 degrees in this evaluation and 15.1 on slide 9; the difference is only which people are matched. Restarting the fine-tune alone costs about 2.4 millimetres. A 2-epoch pilot trains in about 12 minutes, a 10-epoch run in about one hour.");

// 15 housekeeping: each card body says "Done:" / "To do:"; dot green = all done, amber = something still to do
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Next" });
s.addText("Corrections made, and still to make", { placeholder: "title" });
const status = (...parts) => parts.flatMap(([lead, text], i) => [
  { text: lead, options: { bold: true, color: THEME.colors.dk1 } }, { text, options: { breakLine: i < parts.length - 1 } }]);
card(s, 0.5, 1.35, 4.4, 1.75, "Headline number", status(["Done: ", "the README gives AnnyOne holdout PVE 70.5 mm, helper bones masked, as in the demos."], ["To do: ", "replace the thesis draft's 74 mm, the unmasked evaluation (+3.8 mm on the same people)."]), THEME.colors.accent2, { bs: 12 });
card(s, 5.1, 1.35, 4.4, 1.75, "Demos", status(["Done: ", "all demo scripts zero the fingers and the README says so: finger renders are the rest pose, not predictions."]), THEME.colors.accent3, { bs: 12 });
card(s, 0.5, 3.28, 4.4, 1.82, "Wording", status(["Done: ", "neck, fingers, children no longer called rig or decoding problems; the README's Known-defects section documents the depth bug."], ["To do: ", "README and findings still carry last week's neck story; I will fix them next week."]), THEME.colors.accent2, { bs: 12, bh: 1.0 });
card(s, 5.1, 3.28, 4.4, 1.82, "Repository", status(["Done: ", "reorganised, one shared cluster setup, diagnostics and experiments documented so the work can be rerun."]), THEME.colors.accent3, { bs: 12 });
N(s, "Corrections made, and still to make", "Some corrections are already in. The headline AnnyOne number is 70.5 millimetres, measured with the helper bones masked, as in the demos. That is the evaluation setting V1, not the demo's own number: the demo setting gives 68.5 at a lower recall. The 74 millimetres in the thesis draft came from the unmasked evaluation; on the same people that adds 3.8 millimetres. The README already uses 70.5; the thesis draft I still have to change. The demos now say that the fingers are the rest pose. The text no longer blames the rig. The Known-defects section in the README documents the depth bug. But the README and the findings document still carry last week's neck story, that the long neck comes from weak supervision and from the depth bug, and some old depth wording, for example that the overlays look wrong and that no improvement is claimed yet. I will correct them next week. The repository is reorganised with everything scripted, so any of these numbers can be produced again.");

// 16 still open (two columns, row-major like slide 3; item 7, the 3DPW label fix, spans both columns, red like the slide 6 bug band)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Next" });
s.addText("Still open, and what I test next", { placeholder: "title" });
const open = [["1", "Bone flips: confirm the pilot fix", "Neck: real photos, a longer run (slide\u00A08: no\u00A0training). v10's left arm: two-sided fix, not built yet."],
              ["2", "Depth slope stuck at 0.3–0.4", "Cause unknown. Next: measure training depths properly, and the fix run's slope on training images."],
              ["3", "Depth fix: cost and the ×20 run", "Paired PVE on the same people; recall for visible heads. No result yet for the ×20 run at 10 epochs."],
              ["4", "Pilot noise", "No fixed seed. Fix it, or repeat the key run, before trusting differences under about 10 mm."],
              ["5", "People very close to the camera", "Both models miss them (4 of 12 rendered images), so not the fix. Likely detection; not yet counted."],
              ["6", "Children and fingers", "10-epoch sampler run; check the hands with the bone fix, then larger hand crops. Else: limitations."]];
const openY = [1.25, 2.25, 3.25];
open.forEach((o, i) => numberedItem(s, 0.5 + (i % 2) * 4.65, openY[Math.floor(i / 2)], 4.35, o[0], o[1], o[2], C.accent1, 0.6));
numberedItem(s, 0.5, 4.25, 9.0, "7", "3DPW labels: three defects (slide 6)", "Rebuild gave the same counts; the image files are likely missing (check pending): re-extract, rebuild. Then gender (m\u00A0→\u00A00,\u00A0f\u00A0→\u00A01), one shared frame grid, re-fit all 87 tasks, rebuild, re-check. Proposed: only then use 3DPW.", C.accent4, 0.55);
N(s, "Still open, and what I test next", "Seven things are still open. First, the bone flips. The neck fix passes in one run, and the holdout renders of the correction alone show normal necks; I still have to check real photos and a longer run. v10's left arm has the same kind of flip, but a bone further down the arm undoes it, so it needs a two-sided correction, not built yet. Second, the depth slope is stuck at 0.3 to 0.4, cause unknown. Third, the cost of the depth fix, paired on the same people. Fourth, pilot noise: fix the seed or repeat key runs. Fifth, people very close to the camera. Sixth, children and fingers: a longer sampler run, and a look at the hands with the bone fix. Seventh, 3DPW: the rebuild gave the same counts, so the images are likely missing; I will check, re-extract and rebuild. Then gender, frame grid and a re-fit of all 87 tasks, before any 3DPW test.\n\nIf asked: Neck: the fine-tuned checkpoint carries the fix in its saved settings, so the demo applies it on load; I copy it next to the other demo models and add it to the demo comparison script. In the 12 side-by-side holdout panels, with the correction alone and no training, the necks have normal width wherever a neck is visible (slide 8). v10: correcting three of its flipped hand bones passed without training; two more finger bones, seen on training images, are not corrected yet; shoulder01.L at 166 degrees and upperarm02.L at 180 stay flipped, because one constant per bone would break the chain that compensates them. Depth: the training depths so far come from small samples, batch means of about 2.5 to 3.2 metres. I will also measure the fix run's slope on training images; for v5 it was just as low there. Cost: I will report the root error next to PVE in the thesis. Testing the depth fix on 3DPW waits for item seven. The twenty-times 10-epoch run has no result yet. Close people: both models miss them in 4 of the 12 rendered images. Children and fingers: one 2-epoch test is not enough to close them. 3DPW labels: the rebuild, into a new folder, gave the same counts, so my old guess that the builder ran before the fits finished is wrong. The builder skips every frame without an image file. The check is one file count per image folder of the eight affected sequences; if they are empty or short, I unzip the 3DPW images again and rebuild. Gender and the frame grid are different: set men to 0 and, by Anny's convention, women to 1; the refit confirmed it only for men. And pick both people's frames on one shared grid. Both happen inside the fit, so all 87 conversion tasks must be fitted again, into a new folder, because the task script skips every person that already has a file; otherwise old and new fits would mix without a warning. That is an 87-task array job: about 40 GPU hours by the task script's own estimate of 6 seconds per frame, so 5 to 6 hours with 8 GPUs at once; not measured. Then rebuild the labels and run the check again.");

// 17 claims vs not (the question gets its own tinted box with an accent bar)
s = pres.addSlide({ masterName: "DARK_CLOSE", sectionTitle: "Next" });
s.addText("What I claim today, and what I do not", { placeholder: "title" });
s.addText([{ text: "Claimed, measured: ", options: { bold: true, color: THEME.colors.accent6 } }, { text: "the failures are not rig or decoding problems. A training-target bug puts people too far away in v5, v9 and v10; the fix draws bodies closer to their true size. In all 8 v5-line checkpoints, neck02 and the head bone sit half a turn off; in one pilot run, a fixed correction plus a 2-epoch fine-tune restores the neck and passes the fine-tune checks. Two flipped hand bones were half the finger error. 3DPW labels: joints and surface pass; three defects (slide 6).", options: { breakLine: true, paraSpaceAfter: 8 } },
           { text: "Not claimed: ", options: { bold: true, color: THEME.colors.accent6 } }, { text: "a proven neck fix (one run, no seed; head direction 14.4° vs 12.0°) or why the bones flipped; better children; correct depths for near and far people; a depth fix without cost (likely 2.6\u00A0to\u00A04.9\u00A0mm PVE, 1.6\u00A0to\u00A02.4\u00A0recall points); head turn, age or limb lengths in the 3DPW labels." }],
  { placeholder: "body" });
s.addShape(pres.ShapeType.rect, { x: 0.6, y: 3.62, w: 8.8, h: 1.5, fill: { color: "2B3550" }, line: { color: "2B3550" }, objectName: "question box" });
s.addShape(pres.ShapeType.rect, { x: 0.6, y: 3.62, w: 0.08, h: 1.5, fill: { color: THEME.colors.accent6 }, line: { color: THEME.colors.accent6 }, objectName: "question bar" });
s.addText([{ text: "Question for you", options: { bold: true, color: THEME.colors.accent6, breakLine: true, paraSpaceAfter: 4 } },
           { text: "Fine-tune v9 (best on our 3DPW-Anny labels, which have the gender defect: slide 6) with the fixed targets and depth ×5? Or first fix gender and frame grid, re-fit all 3DPW tasks, then fine-tune?", options: { bold: true, breakLine: true, paraSpaceAfter: 4 } },
           { text: "v9 also has the neck flip (measured at epoch 149) and needs its own bone fix; v10 has a similar left-arm flip." }],
  { x: 0.9, y: 3.67, w: 8.35, h: 1.4, fontSize: 13, color: THEME.colors.lt2, margin: 0, isTextBox: true, valign: "middle", objectName: "question text" });
N(s, "What I claim today, and what I do not", "What I claim, measured. The failures are not rig or decoding problems. A training-target bug puts people too far away in v5, v9 and v10; the fix draws bodies much closer to their true size. In the v5 line, neck02 and the head bone sit half a turn off. One pilot run with a fixed correction and a short fine-tune restores the neck thickness and passes the fine-tune checks. Two flipped hand bones were about half the finger error. The 3DPW labels pass on joints and surface, but have three defects. What I do not claim: that the neck fix is proven, since it is one run without a fixed seed; better children; correct depth for near and far people; a depth fix without cost. My question: fine-tune v9 now, or first fix the 3DPW labels and re-fit? I propose the second, as on slide 16, but want your view.\n\nIf asked:\n- Depth fix: it does not separate near and far people well, and it likely has a cost: every fix run is worse than its control, by 2.6 to 4.9 millimetres PVE and 1.6 to 2.4 recall points. The PVE gaps are inside pilot noise, the recall drops are at the edge of what one pilot can detect, and the five runs share two controls, but all point the same way. In v5 the depth bug affects all 222 holdout people. The ViT-S students use the same training code and are probably affected too; not measured.\n- Neck: the two half-turned bones cancel, so the head points the right way. A 2-epoch geodesic loss did not undo it. The fixed correction with a 2-epoch fine-tune did, in one run: neck02 4.7 degrees, thickness 1.00, head direction 14.4 against 13.8 for its control and 12.0 before. One run without a seed is not proof; the holdout renders of the correction without training show normal neck width, but I have not looked at the fine-tuned model's renders or at real photos yet. The origin of the flip is not known; the singular start value of the rotation output would start a bone turned about its sideways axis, a tilt like the one in v5's left index base, not the twist in neck02 and the head.\n- Fingers: two flipped hand bones were about half the finger number, 17.3 to 8.3 degrees; I have not looked at the hands, so I do not claim better-looking fingers.\n- 3DPW labels: 6 train sequences, part of a seventh, and one validation sequence have no labels; a rebuild gave the same counts, so their image files are likely missing on the cluster; the check is still to do. For the women, the gender refit is mixed. In the 3DPW labels the head, neck02 and neck03 stay at rest; only neck01 turns. Overall the labels keep about half the head turn, 9.5 against 20 degrees. Train and validation keep most of it; the loss is in the test split, and I do not know why yet. Age is fixed, and the limb-length ratios are not explained yet.\n- Option 1, v9 now: the fine-tune is about 1 to 2 hours of training; that is not measured with 3DPW batches, and the v9 script expects 8 to 10 minutes per epoch. From v5 I expect a drawn size of about 0.84 and a root error of about half a metre; on v5 that run cost 2.6 millimetres PVE and 1.6 recall points against its control. Five, not twenty, because the twenty-times 10-epoch run has no result yet. v9 is best on our 3DPW-Anny labels, but those labels have the gender and two-person defects, so that score is not a clean measure. On the AnnyOne holdout v9 is not the best: 75.0 at epoch 62, the planned start, and 77.0 at epoch 149, against 70.5.\n- Option 2, labels first: men to 0, women to 1 by Anny's convention, one shared frame grid, re-fit all 87 conversion tasks into a new folder, rebuild the labels, then fine-tune. Rebuilding the labels alone is not enough, because the gender and the frame choice are fixed inside the fit; the re-fit is an 87-task array job, about 40 GPU hours by the script's own estimate. I prefer the re-fit first: 3DPW is 30 percent of each batch, so a fine-tune before it would keep learning the wrong body for the men.\n- Bone flips in v9 and v10: a v9 fine-tune keeps the neck flip unless it carries the bone fix, estimated on v9 first; the v5 constants belong to v5. The flip was measured on v9 epoch 149; epoch 62, the planned start, was never measured per bone, so there it is likely, not shown. v10 has no neck flip, but the same kind of flip in the left shoulder and upper arm, which is likely most of why its shoulders and arms are worse; not shown.\n- Both options: either fine-tune should also mask the 3DPW finger labels, which are set to the rest pose, with mask_3dpw_hands.");

// 18 appendix: second before / after example (same geometry as slide 12)
pres.addSection({ title: "Appendix" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Appendix" });
s.addText("Backup: before and after, holdout image 579860", { placeholder: "title" });
figBottom = figureSlide(s, "depth_before_after_579860.jpg", 1.0, 1.2, 8.0, [
  "Synthetic holdout image 579860. Predictions use the model's own depth.",
  "v5: the real head sticks out above a shrunken mesh",
  "Fix + depth ×5, 10 epochs: head level with the truth; mesh cut off at the image edge"], 0.4);
footnote(s, "Not an exact fit: ", "here the fixed mesh is a little larger and wider than the ground truth. The two people in front have no ground-truth mesh, so they are not counted as misses.", figBottom + 0.5, 0.4);
N(s, "Backup: before and after, holdout image 579860", "A second holdout example, in case there is a question. Before the fix the boy's real head sticks out above a shrunken mesh that floats. After the fix the top of the head is level with the truth; the after mesh is cut off at the bottom of the image, so I cannot measure its full height. Here the fixed mesh is a little larger and wider than the ground truth, so I do not call it an exact fit. The two large people in front have no ground-truth mesh, so they are not counted as misses.");
// pptxgenjs writes one <a:pPr> per run, so a paragraph with a bold lead run gets a second pPr after the
// first run (schema-invalid; LibreOffice then drops the bullet). Keep only the first pPr of each paragraph.
async function keepFirstParaProps(file) {
  const fs = require("fs");
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  let removed = 0;
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, inner) => {
      let seen = false;
      const fixed = inner.replace(/<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>[\s\S]*?<\/a:pPr>/g, pp => { if (!seen) { seen = true; return pp; } removed++; return ""; });
      return "<a:p>" + fixed + "</a:p>";
    });
    zip.file(name, out);
  }
  // notes: pptxgenjs puts all notes text in one <a:t>, line breaks included; split it into one paragraph per line
  let split = 0;
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    const out = xml.replace(/<a:p><a:r><a:rPr lang="en-US" dirty="0"\/><a:t>([^<]*)<\/a:t><\/a:r><a:endParaRPr lang="en-US" dirty="0"\/><\/a:p>/, (m, t) => {
      const lines = t.split(/\r?\n/);
      if (lines.length > 1) split++;
      return lines.map(l => l ? `<a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t>${l}</a:t></a:r></a:p>` : `<a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p>`).join("");
    });
    zip.file(name, out);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("removed duplicate pPr:", removed, "| notes split into paragraphs:", split);
}

(async () => {
  await pres.writeFile({ fileName: "supervisor_update.pptx" });
  await applyTheme("supervisor_update.pptx", THEME);
  await keepFirstParaProps("supervisor_update.pptx");
  let md = "# Speaker notes: supervisor update, 16 October 2026\n\n";
  notes.forEach((n, i) => { md += `## Slide ${i + 1}: ${n.title}\n\n${n.text}\n\n`; });
  require("fs").writeFileSync("supervisor_talk_notes.md", md);
  console.log("written");
})();
