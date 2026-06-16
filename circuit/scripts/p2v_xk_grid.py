"""Phase 2V (cross-k) — 1×N grid of phase-diagram trajectories.

One panel per evaluation k. Each panel shows three trajectories in
partial-correlation phase space:
  - σ_DLA·|ρ| ablation N-sweep (solid viridis line, colored by N=0..32)
  - dual (manifold) steering α-sweep (dashed red)
  - primal (proj_out) steering α-sweep (dotted purple)

All three are derived from the SAME k=15 reference (head ranking +
primal direction + manifold lookup), then evaluated on each k's prompts.

Inputs (per eval-k):
  results/p2o_n_sweep_<short>_positive_dla_phase_evalk<k>.json   (or _phase if k=ref)
  results/p2e_alpha_sweep_<short>_<feature>_k<k>_refk<ref>.json   (or no _refk if k=ref)

Output:
  figures/p2v_xk_phase_grid_<short>_<feature>.png
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
from p2v_phase_trajectory import (  # noqa: E402
    partial_corr, load_trajectory, load_manifold_sweep, load_primal_sweep,
)


def _annotate_corner_labels(ax, xlim, ylim):
    ax.text(xlim[0] + 0.04, 0.97, "RELATIVISTIC",
            color="C0", fontsize=14, fontweight="bold", va="top")
    ax.text(xlim[1] - 0.04, ylim[0] + 0.04, "OBJECTIVE",
            color="C2", fontsize=14, fontweight="bold", ha="right")


def _draw_ablation(ax, traj, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=0, vmax=max(traj["Ns"]))
    pts = np.array(list(zip(traj["pc_xs"], traj["pc_zs"])))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = np.array(traj["Ns"][:-1])
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.4, zorder=4)
    lc.set_array(seg_colors)
    ax.add_collection(lc)
    sc = ax.scatter(traj["pc_xs"], traj["pc_zs"], c=traj["Ns"], cmap=cmap,
                     norm=norm, s=42, edgecolor="black", zorder=10,
                     linewidth=0.4)
    return sc


def _draw_steering(ax, sweep, color, linestyle, legend_label):
    ax.plot(sweep["pc_xs"], sweep["pc_zs"], color=color, linestyle=linestyle,
            lw=2.0, zorder=5, label=legend_label)
    ax.scatter(sweep["pc_xs"], sweep["pc_zs"], facecolor="white",
               edgecolor=color, s=46, linewidth=1.4, zorder=11)


def _anchor_to_baseline(sweep, baseline_r_z, baseline_r_x, r_zx):
    if abs(sweep["alphas"][0]) < 1e-9:
        sweep["r_zs"][0] = baseline_r_z
        sweep["r_xs"][0] = baseline_r_x
    else:
        sweep["alphas"] = [0.0] + sweep["alphas"]
        sweep["r_zs"] = [baseline_r_z] + sweep["r_zs"]
        sweep["r_xs"] = [baseline_r_x] + sweep["r_xs"]
    pcs = [partial_corr(rz, rx, r_zx)
           for rz, rx in zip(sweep["r_zs"], sweep["r_xs"])]
    sweep["pc_xs"] = [p[0] for p in pcs]
    sweep["pc_zs"] = [p[1] for p in pcs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b")
    ap.add_argument("--feature", default="height")
    ap.add_argument("--ks", type=int, nargs="+", required=True,
                    help="evaluation k values, one panel each")
    ap.add_argument("--ref-k", type=int, default=15,
                    help="reference k whose primal/manifold we transfer")
    ap.add_argument("--ablation-suffix", default="positive_dla_phase",
                    help="suffix on the in-k ablation result")
    ap.add_argument("--alpha-max", type=float, default=1.0)
    args = ap.parse_args()

    # Compute stimulus r(z, x) per panel — depends on which k is loaded.
    panels = []
    for k in args.ks:
        # Stimuli for r(z, x)
        stim = REPO / "data" / "p2_shot_sweep" / f"{args.feature}_k{k}.jsonl"
        rows = [json.loads(l) for l in stim.open()]
        z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                          dtype=np.float32)
        x_arr = np.array([float(r["x"]) for r in rows], dtype=np.float32)
        r_zx = float(np.corrcoef(z_arr, x_arr)[0, 1])

        # Ablation trajectory: in-k file uses --out-suffix evalk<k> for k!=ref.
        if k == args.ref_k:
            ab_path = (REPO / "results"
                       / f"p2o_n_sweep_{args.short}_{args.ablation_suffix}.json")
        else:
            ab_path = (REPO / "results"
                       / f"p2o_n_sweep_{args.short}_{args.ablation_suffix}_evalk{k}.json")
        if not ab_path.exists():
            raise SystemExit(f"missing {ab_path}")
        traj = load_trajectory(ab_path)
        pcs = [partial_corr(rz, rx, r_zx)
                for rz, rx in zip(traj["r_zs"], traj["r_xs"])]
        traj["pc_xs"] = [p[0] for p in pcs]
        traj["pc_zs"] = [p[1] for p in pcs]

        # Steering: in-k file is p2e_alpha_sweep_..._k<k>.json (no refk suffix);
        # cross-k file is p2e_alpha_sweep_..._k<k>_refk<ref>.json.
        if k == args.ref_k:
            sw_path = (REPO / "results"
                       / f"p2e_alpha_sweep_{args.short}_{args.feature}_k{k}.json")
        else:
            sw_path = (REPO / "results"
                       / f"p2e_alpha_sweep_{args.short}_{args.feature}"
                         f"_k{k}_refk{args.ref_k}.json")
        manifold = primal = None
        if sw_path.exists():
            manifold = load_manifold_sweep(sw_path, args.alpha_max)
            _anchor_to_baseline(manifold, traj["r_zs"][0], traj["r_xs"][0], r_zx)
            primal = load_primal_sweep(sw_path, args.alpha_max)
            if primal is not None:
                _anchor_to_baseline(primal, traj["r_zs"][0],
                                    traj["r_xs"][0], r_zx)
        else:
            print(f"[warn] no steering sweep at {sw_path}")

        panels.append({"k": k, "r_zx": r_zx, "traj": traj,
                       "manifold": manifold, "primal": primal})

        # Print quick summary
        print(f"\n=== k={k}  r(z,x)={r_zx:+.3f}  baseline r_z={traj['r_zs'][0]:+.3f} "
              f"r_x={traj['r_xs'][0]:+.3f} ===")
        print(f"ablation N=32 → r_z={traj['r_zs'][-1]:+.3f}  "
              f"r_x={traj['r_xs'][-1]:+.3f}  "
              f"pc(x|z)={traj['pc_xs'][-1]:+.3f}  pc(z|x)={traj['pc_zs'][-1]:+.3f}")
        if manifold:
            print(f"dual α={manifold['alphas'][-1]:.2f} → "
                  f"r_z={manifold['r_zs'][-1]:+.3f}  r_x={manifold['r_xs'][-1]:+.3f}  "
                  f"pc(x|z)={manifold['pc_xs'][-1]:+.3f}  "
                  f"pc(z|x)={manifold['pc_zs'][-1]:+.3f}")
        if primal:
            print(f"primal α={primal['alphas'][-1]:.2f} → "
                  f"r_z={primal['r_zs'][-1]:+.3f}  r_x={primal['r_xs'][-1]:+.3f}  "
                  f"pc(x|z)={primal['pc_xs'][-1]:+.3f}  "
                  f"pc(z|x)={primal['pc_zs'][-1]:+.3f}")

    # ---- Plot ----
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8.0 * n, 8.4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    xlim = (-0.30, 1.05)
    ylim = (-0.30, 1.05)
    last_ab_sc = None
    for i, (ax, P) in enumerate(zip(axes, panels)):
        traj = P["traj"]
        last_ab_sc = _draw_ablation(ax, traj)
        if P["manifold"] is not None:
            _draw_steering(ax, P["manifold"], "tab:red", "--",
                           f"dual α=0→{P['manifold']['alphas'][-1]:g}")
        if P["primal"] is not None:
            _draw_steering(ax, P["primal"], "tab:purple", ":",
                           f"primal α=0→{P['primal']['alphas'][-1]:g}")
        _annotate_corner_labels(ax, xlim, ylim)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax.axvline(0, color="black", lw=0.4, alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$pc$(LD, x | z)  →  objective", fontsize=15)
        ax.set_title(
            f"k = {P['k']}\n"
            f"corr(z,x) = {P['r_zx']:+.2f}     "
            f"baseline corr_z={traj['r_zs'][0]:+.2f}, corr_x={traj['r_xs'][0]:+.2f}",
            fontsize=14)
        # Annotate notable Ns inside each panel
        annotate_Ns = {0: "N=0",
                       1: f"+L{traj['cells'][0]['layer']}H{traj['cells'][0]['head']}",
                       4: "N=4", 8: "N=8", 16: "N=16", 32: "N=32"}
        for j, N in enumerate(traj["Ns"]):
            if N in annotate_Ns:
                ax.annotate(annotate_Ns[N],
                            (traj["pc_xs"][j], traj["pc_zs"][j]),
                            xytext=(8, 4), textcoords="offset points",
                            fontsize=10, color="black",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white", edgecolor="gray",
                                      alpha=0.85))
        if i == 0:
            ax.set_ylabel(r"$pc$(LD, z | x)  →  relativistic", fontsize=15)
            ax.legend(fontsize=11, loc="lower left")

    fig.suptitle(
        f"{args.short} {args.feature} — phase trajectories under fixed heads / primal / dual "
        f"(derived at k={args.ref_k}, applied at each eval-k)",
        fontsize=15)

    cbar = fig.colorbar(last_ab_sc, ax=axes, fraction=0.022, pad=0.015)
    cbar.set_label("N heads ablated  (σ_DLA·|ρ|+ ranking, k=15)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    out_png = (REPO / "figures"
               / f"p2v_xk_phase_grid_{args.short}_{args.feature}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
