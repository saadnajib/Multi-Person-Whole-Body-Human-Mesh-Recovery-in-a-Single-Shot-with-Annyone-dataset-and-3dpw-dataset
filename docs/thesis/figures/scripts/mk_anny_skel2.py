# Anny rest-rig skeleton for thesis Figure 2.2(d), rendered at its printed size (legend 7 pt on paper).
# Usage: mk_anny_skel2.py anny_rest_skeleton.json out.png width_in height_in
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
src, out, W, Hh = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = json.load(open(src)); H = d['bone_heads_body_frame_m']; reg = d['bone_region']; par = d['bone_parents']; names = d['bone_names']
col = {'root': '#1f2d3d', 'spine': '#0b8a8f', 'neck_head': '#7b4fb0', 'face': '#d4a017', 'breast': '#c06c9b',
       'arm': '#2e8b57', 'hand': '#d9543f', 'leg': '#3d6ecc', 'toe': '#8c8c8c'}
lab = {'root': 'root', 'spine': 'spine', 'neck_head': 'neck, head', 'face': 'face', 'breast': 'breasts',
       'arm': 'arms', 'hand': 'hands', 'leg': 'legs', 'toe': 'toes'}
cnt = d['region_counts']
matched = {names.index(a) for _, a in d['smpl_to_anny_name_mapping']}
plt.rcParams.update({'font.size': 7, 'font.family': 'Liberation Sans'})
fig = plt.figure(figsize=(W, Hh))
sk_w = Hh * (1.24 / 1.82) / W           # skeleton axes width fraction for equal aspect at full height
ax = fig.add_axes([0, 0, sk_w, 1]); ax.set_aspect('equal'); ax.axis('off')
for c, p in enumerate(par):
    if p >= 0: ax.plot([H[p][0], H[c][0]], [H[p][2], H[c][2]], color=col[reg[c]], lw=0.8, zorder=1)
for i, h in enumerate(H):
    if i not in matched: ax.scatter(h[0], h[2], s=2.5, color=col[reg[i]], zorder=2, linewidth=0)
for i in matched: ax.scatter(H[i][0], H[i][2], s=16, color=col[reg[i]], edgecolor='#20374b', linewidth=0.6, zorder=3)
ax.set_xlim(-.62, .62); ax.set_ylim(-.92, .90)
order = ['root', 'spine', 'neck_head', 'face', 'breast', 'arm', 'hand', 'leg', 'toe']
hs = [Line2D([], [], color=col[k], lw=2.0) for k in order] + \
     [Line2D([], [], marker='o', ls='', mfc='white', mec='#20374b', mew=0.7, ms=4.5)]
ls = [f'{lab[k]} ({cnt[k]})' for k in order] + [f'{len(matched)} points\nfitted to SMPL']
lax = fig.add_axes([sk_w - 0.02, 0, 1 - sk_w + 0.02, 1]); lax.axis('off')
lax.legend(hs, ls, loc='center left', fontsize=7, frameon=False, handlelength=1.0, handletextpad=0.4,
           labelspacing=0.35, borderaxespad=0, borderpad=0)
fig.savefig(out, dpi=400, facecolor='white')
