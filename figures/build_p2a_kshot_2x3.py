#!/usr/bin/env python3
"""Regenerate p2a_ld_vs_z_height_gemma2-9b_2x3.png from circuit/results/p2_ld.

LD distribution across shot count k (height): at k=0 the readout tracks the raw value x; from k=1 on
it tracks the context-normalized standing z_eff, sharpening from a comparator-like step toward a
graded response. (Reproduction from the committed per-prompt LD arrays; no model needed.)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LD_DIR = ROOT / "circuit" / "results" / "p2_ld" / "gemma2-9b"
OUT = ROOT / "paper" / "figures" / "p2a_ld_vs_z_height_gemma2-9b_2x3.png"
KS = [0, 1, 2, 5, 8, 15]
BLUE = "#2b6cb0"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharey=True)
    for ax, k in zip(axes.ravel(), KS):
        p = LD_DIR / f"height_k{k}.npz"
        if not p.exists():
            ax.set_visible(False)
            continue
        d = np.load(p, allow_pickle=True)
        ld = np.asarray(d["ld"], float)
        ax.axhline(0, color="0.8", lw=0.6)
        if k == 0:  # no context -> readout tracks raw x
            xv = np.asarray(d["x"], float)
            ax.scatter(xv, ld, s=7, alpha=0.5, color=BLUE, edgecolors="none")
            ax.set_title(f"k = 0   corr(LD, x) = {corr(ld, xv):+.2f}", fontsize=10)
            ax.set_xlabel("x (raw value)", fontsize=9)
        else:
            zv = np.asarray(d["z_eff"], float)
            ax.scatter(zv, ld, s=7, alpha=0.5, color=BLUE, edgecolors="none")
            ax.set_title(f"k = {k}   corr(LD, z) = {corr(ld, zv):+.2f}", fontsize=10)
            ax.set_xlabel(r"$z_{\mathrm{eff}}$", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("LD = logit(tall) − logit(short)", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
