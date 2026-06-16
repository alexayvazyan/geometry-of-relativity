#!/usr/bin/env python3
"""Regenerate fig_results_shared_direction_own_loo_abs_slopes_z_vs_x.png from results/v15.

For each concept, the absolute steering slope of the concept's OWN mean-difference direction vs a
leave-one-out (LOO) shared direction built from the other seven concepts, for the relative-standing
axis d_z (left) and the raw-magnitude axis d_x (right). LOO d_z retains much of own-d_z steering;
d_x transfers less. (Reproduction from committed data; the original plotting script was not in git.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "v15" / "shared_direction_loo_z_vs_x.json"
OUT = ROOT / "paper" / "figures" / "fig_results_shared_direction_own_loo_abs_slopes_z_vs_x.png"

ORDER = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
LABEL = {"height": "Height", "age": "Age", "weight": "Weight", "size": "Size",
         "speed": "Speed", "wealth": "Income", "experience": "Experience", "bmi_abs": "BMI"}


def main() -> None:
    by_pair = json.loads(SRC.read_text())["by_pair"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, comp, title in [(axes[0], "z", r"Relative standing $d_z$"),
                            (axes[1], "x", r"Raw magnitude $d_x$")]:
        own = [abs(by_pair[p][comp]["own_slope"]) for p in ORDER]
        loo = [abs(by_pair[p][comp]["loo_shared_slope"]) for p in ORDER]
        xs = np.arange(len(ORDER))
        w = 0.38
        ax.bar(xs - w / 2, own, w, label="own direction", color="#3b4cc0")
        ax.bar(xs + w / 2, loo, w, label="LOO shared", color="#e8743b")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[p] for p in ORDER], rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(r"$|$steering slope$|$", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Own vs leave-one-out shared-direction steering (Gemma-2-9B, L33)", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
