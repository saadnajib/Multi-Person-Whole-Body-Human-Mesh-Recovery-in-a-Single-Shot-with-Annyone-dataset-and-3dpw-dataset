"""Figure 5.2: training curves of the 30- and 60-epoch students.
Source: data/loss_curves_student_e30.csv, data/loss_curves_student_e60.csv (exports of
scripts/analysis/plot_loss_curves.py on the cluster). Run from docs/thesis."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#555555"

def load(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k]) if r[k] else None
    return rows, f

for tag in ("e30", "e60"):
    rows, f = load(f"data/loss_curves_student_{tag}.csv")
    ep = [int(r["epoch"]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.9, 2.35), constrained_layout=True)
    for key, lab, col, ls in (("train_total", "total", GREY, "-"),
                              ("train_rotmat", "bone rotations", BLUE, "--"),
                              ("train_kd", "feature distillation", ORANGE, "-."),
                              ("train_kd_out", "output distillation", GREEN, ":")):
        a.plot(ep, [f(r, key) for r in rows], color=col, ls=ls, lw=1.5, label=lab)
    a.set_yscale("log"); a.set_yticks([20, 50, 100, 200]); a.set_yticklabels(["20", "50", "100", "200"])
    a.minorticks_off(); a.set_ylim(12, 1500); a.set_xlabel("epoch"); a.set_ylabel("training loss (log scale)")
    a.grid(True, color="#dddddd", lw=0.5); a.legend(frameon=False, ncol=2, loc="upper right", handlelength=2.6, columnspacing=1.0, borderaxespad=0.2)
    a.set_title("Training loss terms", loc="left")
    ev = [r for r in rows if r["val_annyone_holdout_1_pve"]]
    x = [int(r["epoch"]) for r in ev]
    b.plot(x, [f(r, "val_annyone_holdout_1_pve") for r in ev], color=BLUE, ls="-", marker="o", ms=4, lw=1.5,
           label="AnnyOne holdout (monitor)")
    b.plot(x, [f(r, "val_threedpw_anny_validation_1_pve") for r in ev], color=ORANGE, ls="--", marker="s", ms=4,
           lw=1.5, label="3DPW validation")
    b.set_xlabel("epoch"); b.set_ylabel("PVE (mm)"); b.set_ylim(70, 215); b.set_yticks([80, 100, 120, 140, 160, 180]); b.set_xlim(a.get_xlim())
    b.grid(True, color="#dddddd", lw=0.5); b.legend(frameon=False, loc="upper right")
    b.set_title("PVE during training", loc="left")
    fig.savefig(f"figures/img/F23_student_loss_curves_{tag}.pdf")
