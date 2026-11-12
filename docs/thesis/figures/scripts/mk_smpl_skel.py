# SMPL schematic skeleton for thesis Figure 2.2(b) (fig:body-models-smpl-skel), rendered at its printed size
# (0.30 linewidth = 39 mm = 1.535 in wide), all text 7 pt on paper (second review, defect 18).
# Joint positions, parent list, colours and label placement are copied from
# docs/presentations/2026-10-22_supervisor/build_presentation.py:21 (colours) and :117-136 (SMPL_SCHEMATIC,
# SMPL_PARENTS, smpl_skeleton_png); positions are illustrative (T-pose, metres from the pelvis), not measured.
# Usage: mk_smpl_skel.py out.png
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAVY, GOLD, INK, MUTED = '#142D45', '#C48835', '#20374B', '#526779'
SMPL_SCHEMATIC = {'pelvis': (0, 0), 'left_hip': (.06, -.09), 'right_hip': (-.06, -.09), 'spine1': (0, .11),
    'left_knee': (.11, -.47), 'right_knee': (-.11, -.47), 'spine2': (0, .24), 'left_ankle': (.08, -.87),
    'right_ankle': (-.08, -.87), 'spine3': (0, .30), 'left_foot': (.12, -.94), 'right_foot': (-.12, -.94),
    'neck': (0, .51), 'left_collar': (.08, .43), 'right_collar': (-.08, .43), 'head': (0, .62),
    'left_shoulder': (.18, .45), 'right_shoulder': (-.18, .45), 'left_elbow': (.44, .44), 'right_elbow': (-.44, .44),
    'left_wrist': (.68, .44), 'right_wrist': (-.68, .44), 'left_hand': (.77, .43), 'right_hand': (-.77, .43)}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
out = sys.argv[1]
names = list(SMPL_SCHEMATIC); P = [SMPL_SCHEMATIC[n] for n in names]
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(1.535, 1.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect('equal'); ax.axis('off')
for c, par in enumerate(SMPL_PARENTS):
    if par >= 0: ax.plot([P[par][0], P[c][0]], [P[par][1], P[c][1]], color=NAVY, lw=1.1, zorder=1)
ax.add_patch(plt.Circle((0, .72), .1, fill=False, ec=MUTED, lw=0.8, ls='--'))
for x, y in P: ax.scatter(x, y, s=9, color=GOLD, edgecolor=NAVY, linewidth=0.6, zorder=3)
lab = {'head': (-.08, .66, 'right'), 'neck': (-.06, .51, 'right'), 'spine3': (-.05, .33, 'right'),
       'spine2': (-.05, .21, 'right'), 'spine1': (-.05, .09, 'right'), 'pelvis': (-.05, -.03, 'right'),
       'collar': (.07, .24, 'left'), 'shoulder': (.30, .55, 'center'), 'elbow': (.44, .34, 'center'),
       'wrist': (.92, .55, 'right'), 'hand': (.92, .34, 'right'),
       'hip': (.11, -.13, 'left'), 'knee': (.16, -.47, 'left'), 'ankle': (.14, -.82, 'left'), 'foot': (.18, -.97, 'left')}
for n, (x, y, ha) in lab.items(): ax.text(x, y, n, fontsize=7, ha=ha, va='center', color=INK)
ax.text(-.88, -.62, 'right side:\nsame\n9 joints', fontsize=7, color=MUTED, ha='left', va='center')
ax.set_xlim(-.90, .92); ax.set_ylim(-1.06, .98)
fig.savefig(out, dpi=600, facecolor='white')
