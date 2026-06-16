#!/usr/bin/env python3
"""Regenerate p2a_ld_vs_z_height_gemma2-9b_2x3.png from circuit/results/p2_ld.

Logit difference vs context-normalized standing z at growing shot counts k: a sharp comparator-like
step at small k that smooths toward a graded response as context grows. (Reproduction from committed
per-prompt LD arrays; the original plotting script was not in git.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LD_DIR = ROOT / "circuit" / "results" / "p2_ld" / "gemma2-9b"
OUT = ROOT / "paper" / "figures" / "p2a_ld_vs_z_height_gemma2-9b_2x3.png"
KS = [0, 1, 2, 5, 15, 50]


def binned_mean(z: np.ndarray, ld: np.ndarray, nbins: int = 25):
    edges = np.linspace(-3, 3, nbins + 1)
    idx = np.digitize(z, edges) - 1
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(float(ld[m].mean()))
    return np.array(xs), np.array(ys)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.3), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), KS):
        p = LD_DIR / f"height_k{k}.npz"
        if not p.exists():
            ax.set_visible(False)
            continue
        d = np.load(p, allow_pickle=True)
        z = np.asarray(d["z_eff"] if (k > 0 and "z_eff" in d.files) else d["z"], dtype=float)
        ld = np.asarray(d["ld"], dtype=float)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.scatter(z, ld, s=4, alpha=0.12, color="#3b4cc0", edgecolors="none")
        bx, by = binned_mean(z, ld)
        if len(bx):
            ax.plot(bx, by, color="#b5371f", lw=2)
        ax.set_title(f"k = {k}", fontsize=11)
        ax.set_xlim(-3, 3)
    for ax in axes[-1]:
        ax.set_xlabel(r"$z=(x-\mu)/\sigma$", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta_{\mathrm{logit}}$(tall$-$short)", fontsize=10)
    fig.suptitle("Logit difference vs. relative standing across context length (height, Gemma-2-9B)",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
