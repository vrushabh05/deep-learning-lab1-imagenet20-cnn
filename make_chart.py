"""Draw assets/training_curves.png from results/training_log.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "training_log.csv").open()))
ep = [int(r["epoch"]) for r in rows]
tl = [float(r["train_loss"]) for r in rows]
vl = [float(r["val_loss"]) for r in rows]
ta = [float(r["train_acc"]) * 100 for r in rows]
va = [float(r["val_acc"]) * 100 for r in rows]

BG, INK, MUT, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=200, facecolor=BG)
for ax in (a1, a2):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT)
    ax.set_xlabel("Epoch", color=MUT)

a1.plot(ep, ta, color=BLUE, lw=2, label="Train")
a1.plot(ep, va, color=ORANGE, lw=2, label="Validation")
a1.axhline(60, color="#8a8984", lw=1, ls=":")
a1.text(2, 61, "course target 60%", color=MUT, fontsize=9)
a1.set_ylabel("Accuracy (%)", color=MUT)
a1.set_title("Accuracy: 67.3% validation at epoch 40", loc="left", fontsize=11, fontweight="bold", color=INK)
a1.legend(frameon=False)

a2.plot(ep, tl, color=BLUE, lw=2, label="Train")
a2.plot(ep, vl, color=ORANGE, lw=2, label="Validation")
a2.set_ylabel("Loss (label smoothing 0.1)", color=MUT)
a2.set_title("Loss", loc="left", fontsize=11, fontweight="bold", color=INK)
a2.legend(frameon=False)

fig.tight_layout()
out = root / "assets" / "training_curves.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG)
print("wrote", out)
