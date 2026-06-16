"""Phase 2V — steering-only phase-plane figure (no attention ablation).

Mirrors `p2v_phase_trajectory.py` but draws ONLY the manifold-informed
(formerly dual) and mean-diff (formerly primal) steering trajectories at
the canonical late layer. No interchange-ablation overlay, no DLA bars,
no per-head colour coding. Smaller figure, larger text, axis title without
the r(z, x) annotation.

Inputs:
  results/p2e_alpha_sweep_<short>_<feature>_k<k>.json   (manifold_a* + primal_a* keys)

Output:
  figures/p2v_steering_only_<short>_<feature>.png

Usage:
  python3 scripts/p2v_steering_only.py --short gemma2-9b --feature height --k 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from p2v_phase_trajectory import (  # noqa: E402
    partial_corr, load_manifold_sweep, load_primal_sweep,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-9b")
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--alpha-max", type=float, default=1.0,
                    help="upper bound on α to overlay (default 1.0)")
    ap.add_argument("--figsize", nargs=2, type=float, default=(9.0, 7.5),
                    help="(width, height) in inches")
    ap.add_argument("--out-suffix", default=None,
                    help="optional override for output filename suffix")
    args = ap.parse_args()

    m_path = (REPO / "results"
              / f"p2e_alpha_sweep_{args.short}_{args.feature}_k{args.k}.json")
    if not m_path.exists():
        raise SystemExit(f"missing α-sweep JSON: {m_path}")

    manifold = load_manifold_sweep(m_path, args.alpha_max)
    primal = load_primal_sweep(m_path, args.alpha_max)
    if manifold is None and primal is None:
        raise SystemExit(f"no manifold_a* or primal_a* keys in {m_path.name}")

    # Stimulus r(z, x) — needed for partial-correlation projection.
    stim = (REPO / "data" / "p2_shot_sweep"
            / f"{args.feature}_k{args.k}.jsonl")
    rows = [json.loads(l) for l in stim.open()]
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                      dtype=np.float32)
    x_arr = np.array([float(r["x"]) for r in rows], dtype=np.float32)
    r_zx = float(np.corrcoef(z_arr, x_arr)[0, 1])

    def _project(sweep: dict) -> None:
        pcs = [partial_corr(rz, rx, r_zx)
               for rz, rx in zip(sweep["r_zs"], sweep["r_xs"])]
        sweep["pc_xs"] = [p[0] for p in pcs]
        sweep["pc_zs"] = [p[1] for p in pcs]

    if manifold is not None:
        _project(manifold)
    if primal is not None:
        _project(primal)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    xlim = (-0.20, 1.05)
    ylim = (-0.20, 1.05)

    # Top-left corner: RELATIVISTIC label with a NORTH-pointing arrow above it
    # (more relativistic = higher corr(LD, z|x)).
    rel_x = xlim[0] + 0.04
    ax.annotate("", xy=(rel_x, ylim[1] - 0.01),
                xytext=(rel_x, ylim[1] - 0.18),
                arrowprops=dict(arrowstyle="->", color="C0", lw=2.2))
    ax.text(rel_x, ylim[1] - 0.22, "RELATIVISTIC",
             color="C0", fontsize=18, fontweight="bold",
             ha="center", va="top", rotation=90)

    # Bottom-right corner: OBJECTIVE label with an EAST-pointing arrow to its right
    # (more objective = higher corr(LD, x|z)).
    obj_y = ylim[0] + 0.04
    ax.annotate("", xy=(xlim[1] - 0.01, obj_y),
                xytext=(xlim[1] - 0.18, obj_y),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2.2))
    ax.text(xlim[1] - 0.22, obj_y, "OBJECTIVE",
             color="C2", fontsize=18, fontweight="bold",
             ha="right", va="center")

    # Per-α offsets staggered by mode to avoid overlap of the two trajectories
    # near the shared baseline cluster (top-right).
    manifold_offsets = {0.5: (12, 4), 0.75: (12, 4), 1.0: (12, 0)}
    primal_offsets = {0.5: (-12, -8), 0.75: (-12, -8), 1.0: (10, 0)}

    def _draw(sweep: dict, color: str, linestyle: str, legend_label: str,
               offsets: dict, ha: str = "left") -> None:
        ax.plot(sweep["pc_xs"], sweep["pc_zs"],
                color=color, linestyle=linestyle, lw=2.6, zorder=5,
                label=legend_label)
        ax.scatter(sweep["pc_xs"], sweep["pc_zs"],
                    facecolor="white", edgecolor=color, s=70,
                    linewidth=1.8, zorder=11)
        for a, px, pz in zip(sweep["alphas"], sweep["pc_xs"], sweep["pc_zs"]):
            if a in offsets:
                dx, dy = offsets[a]
                ax.annotate(rf"$\alpha$={a:g}", (px, pz),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=15, color=color,
                            ha=("right" if dx < 0 else "left"),
                            va="center", fontweight="bold")

    if manifold is not None:
        _draw(manifold, "tab:red", "--", "manifold-informed",
              manifold_offsets)
    if primal is not None:
        _draw(primal, "tab:purple", ":", "mean-diff",
              primal_offsets, ha="right")

    # Annotate the shared α=0 origin as "baseline".
    base_sweep = manifold if manifold is not None else primal
    if base_sweep is not None and len(base_sweep["alphas"]) > 0:
        bx, by = base_sweep["pc_xs"][0], base_sweep["pc_zs"][0]
        ax.scatter([bx], [by], facecolor="black", edgecolor="black",
                    s=85, zorder=12)
        ax.annotate("baseline", (bx, by),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=15, color="black", fontweight="bold",
                    ha="left", va="bottom")

    ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=17)

    ax.set_xlabel(r"corr(LD, $x \mid z$)", fontsize=19)
    ax.set_ylabel(r"corr(LD, $z \mid x$)", fontsize=19)
    ax.set_title(f"{args.short} — steering trajectories ({args.feature} k={args.k})",
                  fontsize=18)
    ax.legend(fontsize=16, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    suffix = args.out_suffix or f"{args.feature}"
    out_png = REPO / "figures" / f"p2v_steering_only_{args.short}_{suffix}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
