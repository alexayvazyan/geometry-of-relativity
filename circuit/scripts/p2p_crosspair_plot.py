"""Phase 2P — cross-pair comparison plot.

Reads p2o_specific_cells_<short>_<feature>_k15.json for height/weight/speed
on Gemma 2 2B and produces a grouped-bar comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SHORT = "gemma2-2b"
FEATURES = ["height", "weight", "speed"]
COLORS = {"height": "#4C72B0", "weight": "#DD8452", "speed": "#55A868"}


def main() -> None:
    data = {f: json.loads((REPO / "results" /
                            f"p2o_specific_cells_{SHORT}_{f}_k15.json").read_text())
             for f in FEATURES}
    labels = [r["label"] for r in data["height"]["runs"]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: r(LD,z) bars, grouped by cell set
    ax = axes[0]
    n_groups = len(labels) + 1
    width = 0.27
    x = np.arange(n_groups)
    for i, f in enumerate(FEATURES):
        d = data[f]
        rs = [d["baseline_r_LD_z"]] + [r["r_LD_z"] for r in d["runs"]]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, rs, width, label=f"{f} (r₀={d['baseline_r_LD_z']:+.2f})",
                       color=COLORS[f], edgecolor="black", linewidth=0.5)
        for bar, r in zip(bars, rs):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.015,
                     f"{r:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    xtick_labels = ["baseline"] + labels
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("r(LD, z) under joint resample")
    ax.set_title(f"{SHORT} — trio resample across height/weight/speed (k=15)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel B: Δr / baseline bars (relative drop)
    ax = axes[1]
    for i, f in enumerate(FEATURES):
        d = data[f]
        rels = [-r["delta_r"] / d["baseline_r_LD_z"] * 100 for r in d["runs"]]
        offset = (i - 1) * width
        bars = ax.bar(np.arange(len(labels)) + offset, rels, width,
                       label=f, color=COLORS[f], edgecolor="black", linewidth=0.5)
        for bar, rel in zip(bars, rels):
            ax.text(bar.get_x() + bar.get_width() / 2, rel + 1.0,
                     f"{rel:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("relative drop in r(LD,z) [%]")
    ax.set_title(f"Relative disruption (−Δr / r₀) by feature")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 80)

    fig.suptitle(
        "Phase 2P — cross-pair generalization of the L16H4 + L17H7 + L14H2 "
        "trio (Gemma 2 2B, last token, k=15)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = REPO / "figures" / f"p2p_crosspair_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
