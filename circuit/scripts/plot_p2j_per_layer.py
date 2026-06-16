"""Phase 2J — plot per-layer iso sweep.

Two views:
  A. r_z and r_x vs layer (one line each), per (model, pair) panel.
     Baseline plotted as horizontal lines for reference.
  B. Δr_z and Δr_x = r_iso(L) - r_baseline, per layer — relative effect.

Highlights the layer where iso has the biggest impact on z-encoding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES_DIR = REPO / "results"
FIG_DIR = REPO / "figures"

MODELS = ["gemma2-2b", "gemma2-9b"]
PAIRS = ["height", "weight", "speed"]


def load(model, pair, k=15):
    p = RES_DIR / f"p2j_isolate_per_layer_{model}_{pair}_k{k}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def main():
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(5.5 * len(PAIRS), 4.0 * len(MODELS)),
                              squeeze=False)
    fig2, axes2 = plt.subplots(len(MODELS), len(PAIRS),
                                figsize=(5.5 * len(PAIRS), 4.0 * len(MODELS)),
                                squeeze=False)

    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            R = load(model, pair)
            ax  = axes[mi, pi]
            ax2 = axes2[mi, pi]
            if R is None:
                for a in (ax, ax2):
                    a.text(0.5, 0.5, "missing", ha="center", va="center",
                           transform=a.transAxes, color="red")
                continue
            base = R["results"]["baseline"]
            base_rz = base["r_ld_zeff"]; base_rx = base["r_ld_x"]
            layers = []
            rzs = []
            rxs = []
            for k, v in R["results"].items():
                if k == "baseline":
                    continue
                layers.append(v["layer"])
                rzs.append(v["r_ld_zeff"])
                rxs.append(v["r_ld_x"])
            order = np.argsort(layers)
            xs = np.array(layers)[order]
            rzs = np.array(rzs)[order]
            rxs = np.array(rxs)[order]

            # A: raw curves with baseline lines
            ax.axhline(base_rz, color="tab:blue", linestyle="--", alpha=0.5,
                        label=f"baseline r_z={base_rz:+.2f}")
            ax.axhline(base_rx, color="tab:orange", linestyle="--", alpha=0.5,
                        label=f"baseline r_x={base_rx:+.2f}")
            ax.plot(xs, rzs, "-o", color="tab:blue",
                     label=r"r(LD, $z_{eff}$) under iso", markersize=3.5,
                     linewidth=1.4)
            ax.plot(xs, rxs, "-o", color="tab:orange",
                     label=r"r(LD, x) under iso", markersize=3.5,
                     linewidth=1.4)
            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}",
                          fontsize=11)
            if pi == 0:
                ax.set_ylabel("Pearson r")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("layer (iso applied here only)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            ax.set_ylim(-0.1, 1.05)

            # B: deltas
            d_rz = rzs - base_rz
            d_rx = rxs - base_rx
            ax2.plot(xs, d_rz, "-o", color="tab:blue",
                      label=r"Δr(LD, $z_{eff}$)", markersize=3.5, linewidth=1.4)
            ax2.plot(xs, d_rx, "-o", color="tab:orange",
                      label=r"Δr(LD, x)", markersize=3.5, linewidth=1.4)
            ax2.axhline(0, color="black", linewidth=0.5, alpha=0.6)
            ax2.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}",
                          fontsize=11)
            if pi == 0:
                ax2.set_ylabel("Δr (iso − baseline)")
            if mi == len(MODELS) - 1:
                ax2.set_xlabel("layer (iso applied here only)")
            ax2.grid(alpha=0.25)
            ax2.legend(fontsize=7)

            # annotate strongest-effect layer (most negative Δr_z)
            i_min = int(np.argmin(d_rz))
            ax2.annotate(f"L{xs[i_min]} Δr_z={d_rz[i_min]:+.2f}",
                          (xs[i_min], d_rz[i_min]),
                          xytext=(8, -10), textcoords="offset points",
                          fontsize=8, color="tab:blue", fontweight="bold")

    fig.suptitle("Phase 2J — per-layer ctx-number isolation: absolute correlations\n"
                 "(at each L, iso_all mask applied at THAT layer only; "
                 "all other layers attend normally)",
                 y=1.005, fontsize=13)
    fig2.suptitle("Phase 2J — per-layer ctx-number isolation: Δr from baseline",
                  y=1.005, fontsize=13)
    fig.tight_layout()
    fig2.tight_layout()
    out_a = FIG_DIR / "p2j_per_layer_abs.png"
    out_b = FIG_DIR / "p2j_per_layer_delta.png"
    fig.savefig(out_a, dpi=130, bbox_inches="tight")
    fig2.savefig(out_b, dpi=130, bbox_inches="tight")
    print(f"  -> {out_a}")
    print(f"  -> {out_b}")


if __name__ == "__main__":
    main()
