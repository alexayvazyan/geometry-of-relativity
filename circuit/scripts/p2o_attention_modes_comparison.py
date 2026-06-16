"""Consolidated comparison: 2 positive-only cell-rankings × 3 intervention modes.

Reads:
  results/p2o_attention_modes_<short>_bycos_positive.json
  results/p2o_attention_modes_<short>_bydla_positive.json
Writes:
  figures/p2o_attention_modes_comparison_<short>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b")
    args = ap.parse_args()

    conditions = [
        ("|cos(·,d_primal)|",
         f"p2o_attention_modes_{args.short}_byresidcos_positive.json",
         "tab:olive"),
        ("σ_DLA·|ρ|",
         f"p2o_attention_modes_{args.short}_bydla_positive.json",
         "tab:purple"),
    ]
    modes = ["zero", "resample", "q_zero"]

    data = []
    for name, fname, color in conditions:
        d = json.load(open(REPO / "results" / fname))
        cells = [(c["layer"], c["head"]) for c in d["cells"]]
        data.append({
            "name": name, "color": color,
            "cells": cells,
            "baseline_r": d["baseline_r_LD_z"],
            "baseline_LD": d["baseline_LD_mean"],
            "modes": d["modes"],
        })

    base_r = data[0]["baseline_r"]
    base_LD = data[0]["baseline_LD"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    n_modes = len(modes)
    x = np.arange(n_modes)
    w = 0.35

    # Panel A: Δr
    ax = axes[0]
    for i, c in enumerate(data):
        drs = [c["modes"][m]["delta_r"] for m in modes]
        bars = ax.bar(x + (i - 0.5) * w, drs, w, color=c["color"],
                       edgecolor="black", linewidth=0.5, label=c["name"])
        for bar, dr in zip(bars, drs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     dr + (-0.03 if dr < 0 else 0.015),
                     f"{dr:+.3f}", ha="center", fontsize=16)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Δcorr(LD, z)", fontsize=18)
    ax.set_title(f"z-correlation  (baseline corr={base_r:+.2f})", fontsize=19)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=18, loc="lower right")
    ax.set_ylim(-1.05, 0.05)

    # Panel B: Δ⟨LD⟩
    ax = axes[1]
    for i, c in enumerate(data):
        dlds = [c["modes"][m]["delta_ld_mean"] for m in modes]
        bars = ax.bar(x + (i - 0.5) * w, dlds, w, color=c["color"],
                       edgecolor="black", linewidth=0.5, label=c["name"])
        for bar, dl in zip(bars, dlds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     dl + (-0.04 if dl < 0 else 0.02),
                     f"{dl:+.2f}", ha="center", fontsize=16)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Δ⟨LD⟩  (+ = more 'tall')", fontsize=18)
    ax.set_title(f"mean ⟨LD⟩  (baseline={base_LD:+.2f})", fontsize=19)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=18, loc="lower right")
    all_dlds = [c["modes"][m]["delta_ld_mean"] for c in data for m in modes]
    ax.set_ylim(min(all_dlds) - 0.10, max(0.05, max(all_dlds) + 0.05))

    fig.text(0.5, -0.02,
              "|cos(·,d_primal)|+: "
              + ", ".join(f"L{L}H{h}" for L, h in data[0]["cells"][:4])
              + " …   |   σ_DLA·|ρ|+: "
              + ", ".join(f"L{L}H{h}" for L, h in data[1]["cells"][:4]) + " …",
              ha="center", fontsize=14, color="dimgray")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_png = REPO / "figures" / f"p2o_attention_modes_comparison_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    print("\n=== consolidated comparison ===")
    print(f"{'condition':<32s}  {'cells (top 4)':<30s}  "
           f"{'zero Δr':>8s}  {'res Δr':>8s}  {'qz Δr':>8s}  {'zero Δ⟨LD⟩':>10s}")
    for c in data:
        cells_short = ",".join(f"L{L}H{h}" for L, h in c["cells"][:4])
        print(f"{c['name']:<32s}  {cells_short:<30s}  "
              f"{c['modes']['zero']['delta_r']:>+8.3f}  "
              f"{c['modes']['resample']['delta_r']:>+8.3f}  "
              f"{c['modes']['q_zero']['delta_r']:>+8.3f}  "
              f"{c['modes']['zero']['delta_ld_mean']:>+10.3f}")


if __name__ == "__main__":
    main()
