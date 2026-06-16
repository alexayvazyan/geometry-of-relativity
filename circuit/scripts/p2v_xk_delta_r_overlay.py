"""Phase 2V (cross-k) — Δr(N) overlay across k=1, 5, 15.

A single-panel line plot showing Δr(LD, z) vs N for each evaluation k,
with all three curves driven by the same σ_DLA·|ρ|+ ranking from k=15.
This is the cleanest "same circuit handles all regimes" visualization.

Output: figures/p2v_xk_deltar_overlay_<short>_<feature>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_run(short: str, suffix: str):
    p = REPO / "results" / f"p2o_n_sweep_{short}_{suffix}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b")
    ap.add_argument("--feature", default="height")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 15])
    ap.add_argument("--ref-k", type=int, default=15)
    ap.add_argument("--ablation-suffix", default="positive_dla_phase")
    args = ap.parse_args()

    cmap = plt.get_cmap("plasma")
    colors = {k: cmap(0.15 + 0.7 * i / max(1, len(args.ks) - 1))
              for i, k in enumerate(args.ks)}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)

    # Track first-cell labels for annotation
    cell_labels = []

    for k in args.ks:
        suffix = (args.ablation_suffix if k == args.ref_k
                  else f"{args.ablation_suffix}_evalk{k}")
        d = load_run(args.short, suffix)
        Ns = [0] + [r["N"] for r in d["selected_runs"]]
        deltas = [0.0] + [r["delta_r"] for r in d["selected_runs"]]
        baseline_r = d["baseline_r_LD_z"]
        # Random control mean
        rand_dr = np.array([r["delta_r_mean"] for r in d["random_runs"]])

        ax.plot(Ns, deltas, "-o", color=colors[k], lw=2.4, ms=7,
                label=f"k = {k}    baseline corr$_z$ = {baseline_r:+.2f}",
                zorder=10)
        # Random control as faint scatter
        rand_Ns = [r["N"] for r in d["random_runs"]]
        ax.scatter(rand_Ns, rand_dr, marker="x", s=26,
                   color=colors[k], alpha=0.55, zorder=4,
                   label=None)
        cell_labels.append((k, d["selected_cells"][:6]))

    # Annotate top-cell entries (same across k) — cell name on the side, offset
    # away from the curve so labels don't collide with low-Δr lines.
    cells = cell_labels[0][1]
    annotate_Ns = {
        1: (f"+L{cells[0]['layer']}H{cells[0]['head']}", (12, 6)),
        2: (f"+L{cells[1]['layer']}H{cells[1]['head']}", (12, 6)),
        4: (f"+L{cells[3]['layer']}H{cells[3]['head']}", (12, 14)),
    }
    # use the k=15 curve for annotation positions
    k_anchor = args.ref_k if args.ref_k in args.ks else args.ks[-1]
    suffix = (args.ablation_suffix if k_anchor == args.ref_k
              else f"{args.ablation_suffix}_evalk{k_anchor}")
    d_anchor = load_run(args.short, suffix)
    sel = {r["N"]: r["delta_r"] for r in d_anchor["selected_runs"]}
    for N, (lbl, offset) in annotate_Ns.items():
        if N in sel:
            ax.annotate(lbl, (N, sel[N]),
                        xytext=offset, textcoords="offset points",
                        fontsize=10, color="black",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="gray",
                                  alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        lw=0.8, alpha=0.6))

    # Add a "random control" placeholder marker in the legend
    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0]
    rand_handle = Line2D([], [], marker="x", linestyle="None",
                         color="gray", alpha=0.6,
                         label="random N-cell control (one trial per N)")
    ax.legend(handles=handles + [rand_handle], fontsize=11,
              loc="upper center", bbox_to_anchor=(0.55, -0.12),
              ncol=2, framealpha=0.95)

    ax.set_xlabel("N heads ablated  (greedy σ$_{DLA}\\cdot|\\rho|$+ ranking, fixed across k)",
                  fontsize=13)
    ax.set_ylabel(r"$\Delta$corr(LD, z)  =  corr$_{ablated}$ − corr$_{baseline}$", fontsize=13)
    ax.set_title(
        f"{args.short} {args.feature} — same circuit collapses z-readout at every k\n"
        f"(heads identified at k=15; same ranking re-evaluated on k=1, 5, 15 prompts)",
        fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlim(-1, 33)
    ax.grid(alpha=0.3)

    out = REPO / "figures" / f"p2v_xk_deltar_overlay_{args.short}_{args.feature}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
