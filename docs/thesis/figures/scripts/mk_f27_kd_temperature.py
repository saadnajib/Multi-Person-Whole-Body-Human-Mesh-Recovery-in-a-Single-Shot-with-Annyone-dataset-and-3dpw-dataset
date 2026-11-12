import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
sweep = [r for r in rows if r["source"] == "sweep" and r["holdout_pve"]]
rep = [r for r in rows if r["source"] == "reported"][0]
T = [float(r["temperature"]) for r in sweep]
g = lambda k: [float(r[k]) for r in sweep]
TEACHER_HOLD, TEACHER_TP = 73.7, 89.0   # \FinalPVE, \FinalThreeDPW
BAR_HOLD, BAR_TP = 77.6, 93.7           # \BarPVE, \BarThreeDPW
BLUE, GREEN = "#2A78D6", "#1BAF7A"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(5.1, 3.7))
ax.plot(T, g("holdout_pve"), "-o", color=BLUE, lw=1.5, ms=5, label="AnnyOne holdout, PVE")
ax.plot(T, g("holdout_mpjpe"), "--s", color=BLUE, lw=1.5, ms=5, label="AnnyOne holdout, MPJPE")
ax.plot(T, g("threedpw_pve"), "-o", color=GREEN, lw=1.5, ms=5, label="3DPW test, PVE")
ax.plot(T, g("threedpw_mpjpe"), "--s", color=GREEN, lw=1.5, ms=5, label="3DPW test, MPJPE")
ax.plot([4], [float(rep["holdout_pve"])], "*", color=BLUE, ms=12, label="Reported student, holdout PVE")
ax.plot([4], [float(rep["threedpw_pve"])], "*", color=GREEN, ms=12, label="Reported student, 3DPW PVE")
for y, c, ls, lab in [(TEACHER_HOLD, BLUE, "--", "teacher PVE"), (BAR_HOLD, BLUE, ":", "95 % retention threshold"),
                      (TEACHER_TP, GREEN, "--", "teacher PVE"), (BAR_TP, GREEN, ":", "95 % retention threshold")]:
    ax.axhline(y, color=c, ls=ls, lw=1.1, alpha=0.9)
    ax.text(15.5, y + 0.6, lab, color="#333333", fontsize=7, ha="right", va="bottom")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16\n(no result)"])
ax.set_xlim(0.8, 20); ax.set_ylim(70, 130)
ax.set_xlabel("distillation temperature T (logarithmic axis)"); ax.set_ylabel("error (mm)")
ax.grid(True, color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
ax.tick_params(labelsize=8, width=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=7.5, frameon=False,
          handlelength=2.6, columnspacing=1.5)
fig.tight_layout()
fig.savefig(out)
