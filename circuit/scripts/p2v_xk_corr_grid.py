"""Phase 2V (cross-k) — corr-axes grid with arc-length manifold-informed flow.

Variant of `p2v_xk_grid.py` that:
  - plots raw correlations corr(LD, x) and corr(LD, z) on the axes
    (instead of partial correlations pc)
  - uses arc-length manifold flow (Phase 2F) as the "manifold-informed"
    trajectory in place of chord steering
  - renames "primal" → "mean-diff"

Inputs (per eval-k):
  results/p2o_n_sweep_<short>_positive_dla_phase[_evalk<k>].json   (ablation)
  results/p2f_arclength_alpha_sweep_<short>_<feature>_k<k>[_refk<ref>].json   (arc + mean-diff)

Output:
  figures/p2v_xk_corr_grid_<short>_<feature>.png

Usage:
  python3 scripts/p2v_xk_corr_grid.py --short gemma2-9b --feature height --ks 1 5 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from p2v_phase_trajectory import load_trajectory  # noqa: E402


def load_alpha_runs(json_path: Path, prefix: str, alpha_max: float) -> dict:
    d = json.loads(json_path.read_text())
    runs = []
    for k, v in d["results"].items():
        if not k.startswith(prefix):
            continue
        a = v.get("alpha")
        if a is None or a > alpha_max + 1e-9:
            continue
        runs.append((float(a), float(v["r_ld_zeff"]), float(v["r_ld_x"])))
    if not runs:
        return None
    runs.sort()
    return {
        "alphas": [r[0] for r in runs],
        "r_zs": [r[1] for r in runs],
        "r_xs": [r[2] for r in runs],
    }


def _draw_ablation(ax, traj, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=0, vmax=max(traj["Ns"]))
    pts = np.array(list(zip(traj["r_xs"], traj["r_zs"])))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = np.array(traj["Ns"][:-1])
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.4, zorder=4,
                         label="DLA ablation")
    lc.set_array(seg_colors)
    ax.add_collection(lc)
    sc = ax.scatter(traj["r_xs"], traj["r_zs"], c=traj["Ns"], cmap=cmap,
                     norm=norm, s=42, edgecolor="black", zorder=10,
                     linewidth=0.4)
    return sc


def _draw_steering(ax, sweep, color, linestyle, legend_label,
                    baseline_anchor=None):
    if sweep is None:
        return
    # If α=0 entry exists, ensure it sits at the baseline (anchor for visual
    # alignment with the ablation N=0 point).
    alphas = list(sweep["alphas"])
    r_zs = list(sweep["r_zs"])
    r_xs = list(sweep["r_xs"])
    if baseline_anchor is not None and alphas and abs(alphas[0]) < 1e-9:
        r_zs[0], r_xs[0] = baseline_anchor
    ax.plot(r_xs, r_zs, color=color, linestyle=linestyle, lw=2.0, zorder=5,
            label=legend_label)
    ax.scatter(r_xs, r_zs, facecolor="white", edgecolor=color, s=46,
                linewidth=1.4, zorder=11)


def _annotate_corners(ax, xlim, ylim):
    rel_x = xlim[0] + 0.04
    ax.annotate("", xy=(rel_x, ylim[1] - 0.01),
                xytext=(rel_x, ylim[1] - 0.18),
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.8))
    ax.text(rel_x, ylim[1] - 0.22, "RELATIVISTIC",
             color="C0", fontsize=14, fontweight="bold",
             ha="center", va="top", rotation=90)

    obj_y = ylim[0] + 0.04
    ax.annotate("", xy=(xlim[1] - 0.01, obj_y),
                xytext=(xlim[1] - 0.18, obj_y),
                arrowprops=dict(arrowstyle="->", color="C2", lw=1.8))
    ax.text(xlim[1] - 0.22, obj_y, "OBJECTIVE",
             color="C2", fontsize=14, fontweight="bold",
             ha="right", va="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-9b")
    ap.add_argument("--feature", default="height")
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--ref-k", type=int, default=15)
    ap.add_argument("--ablation-suffix", default="positive_dla_phase")
    ap.add_argument("--alpha-max", type=float, default=1.0)
    args = ap.parse_args()

    panels = []
    for k in args.ks:
        if k == args.ref_k:
            ab_path = (REPO / "results"
                       / f"p2o_n_sweep_{args.short}_{args.ablation_suffix}.json")
            arc_path = (REPO / "results"
                        / f"p2f_arclength_alpha_sweep_{args.short}_{args.feature}"
                          f"_k{k}.json")
        else:
            ab_path = (REPO / "results"
                       / f"p2o_n_sweep_{args.short}_{args.ablation_suffix}_evalk{k}.json")
            arc_path = (REPO / "results"
                        / f"p2f_arclength_alpha_sweep_{args.short}_{args.feature}"
                          f"_k{k}_refk{args.ref_k}.json")
        if not ab_path.exists():
            raise SystemExit(f"missing ablation: {ab_path}")
        if not arc_path.exists():
            raise SystemExit(f"missing arc-length sweep: {arc_path}")

        traj = load_trajectory(ab_path)
        arc = load_alpha_runs(arc_path, "arclength_", args.alpha_max)
        primal = load_alpha_runs(arc_path, "primal_", args.alpha_max)
        baseline = (traj["r_zs"][0], traj["r_xs"][0])
        panels.append({"k": k, "traj": traj, "arc": arc, "primal": primal,
                        "baseline": baseline})

        print(f"\n=== k={k}  baseline r_z={baseline[0]:+.3f} r_x={baseline[1]:+.3f} ===")
        print(f"ablation N=32 → r_z={traj['r_zs'][-1]:+.3f}  "
              f"r_x={traj['r_xs'][-1]:+.3f}")
        if arc:
            print(f"arc-length α=1 → r_z={arc['r_zs'][-1]:+.3f}  "
                  f"r_x={arc['r_xs'][-1]:+.3f}")
        if primal:
            print(f"mean-diff α=1 → r_z={primal['r_zs'][-1]:+.3f}  "
                  f"r_x={primal['r_xs'][-1]:+.3f}")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 7.5),
                              sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    xlim = (-0.20, 1.05)
    ylim = (-0.20, 1.05)
    last_ab_sc = None
    for i, (ax, P) in enumerate(zip(axes, panels)):
        last_ab_sc = _draw_ablation(ax, P["traj"])
        _draw_steering(ax, P["arc"], "tab:red", "--",
                        f"manifold-informed α=0→{P['arc']['alphas'][-1]:g}",
                        baseline_anchor=P["baseline"])
        _draw_steering(ax, P["primal"], "tab:purple", ":",
                        f"mean-diff α=0→{P['primal']['alphas'][-1]:g}",
                        baseline_anchor=P["baseline"])
        _annotate_corners(ax, xlim, ylim)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax.axvline(0, color="black", lw=0.4, alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"corr(LD, $x$)", fontsize=17)
        ax.set_title(
            f"k = {P['k']}\n"
            f"baseline corr(LD,z)={P['baseline'][0]:+.2f}, "
            f"corr(LD,x)={P['baseline'][1]:+.2f}",
            fontsize=14)
        annotate_Ns = {0: "N=0",
                       1: f"+L{P['traj']['cells'][0]['layer']}H{P['traj']['cells'][0]['head']}",
                       4: "N=4", 8: "N=8", 16: "N=16", 32: "N=32"}
        for j, N in enumerate(P["traj"]["Ns"]):
            if N in annotate_Ns:
                ax.annotate(annotate_Ns[N],
                            (P["traj"]["r_xs"][j], P["traj"]["r_zs"][j]),
                            xytext=(8, 4), textcoords="offset points",
                            fontsize=10, color="black",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white", edgecolor="gray",
                                      alpha=0.85))
        if i == 0:
            ax.set_ylabel(r"corr(LD, $z$)", fontsize=17)
            ax.legend(fontsize=12, loc="lower left")

    fig.suptitle(
        f"{args.short} {args.feature} — phase trajectories (heads + manifold-informed + mean-diff, "
        f"derived at k={args.ref_k}, applied at each eval-k)",
        fontsize=15)

    cbar = fig.colorbar(last_ab_sc, ax=axes, fraction=0.018, pad=0.02,
                         location="right")
    cbar.set_label("N heads ablated  (σ_DLA·|ρ|+ ranking, k=15)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    out_png = (REPO / "figures"
               / f"p2v_xk_corr_grid_{args.short}_{args.feature}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
