"""Phase 2D — visualize partial L0 ablation.

Three-panel figure:

  A. Per-head bar chart: Δr(LD, z_eff) and Δr(LD, x) per L0 head.
     One row per model. The "outlier head(s)" pop out cleanly.

  B. Cumulative top-k vs random-k curve: Δr(LD, z_eff) and Δr(LD, x)
     vs number of L0 heads ablated. Top-k is the worst-case curve;
     random-k (with shaded band) is the average.

  C. 2D phase-space trajectory: r(LD, x) vs r(LD, z_eff) for each
     ablation. Baseline at top-right (high z, moderate x).
     l0_all at bottom-center (broken, both low). Top-k traces a path
     through this space.

We also overlay anchor points: k=0 zero-shot baseline (high r_x,
no r_z), k=4/k=15 baselines (high z, low x).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def load(model, pair="height", k=1):
    p = REPO / "results" / f"p2d_partial_l0_{model}_{pair}_k{k}.json"
    with p.open() as f:
        return json.load(f)


def get_anchor_points(model, pair="height"):
    """Return dict of (label, r_x, r_zeff) for behavioral anchors:
       k=0 baseline, k=1/4/15 baselines from p2_ld extraction."""
    out = []
    npz0 = REPO / "results" / "p2_ld" / model / f"{pair}_k0.npz"
    if npz0.exists():
        d = np.load(npz0, allow_pickle=True)
        from scipy.stats import pearsonr
        r_x = pearsonr(d["x"], d["ld"])[0]
        out.append(("k=0 (zero-shot)", float(r_x), float("nan")))
    for k in [1, 4, 15]:
        npz = REPO / "results" / "p2_ld" / model / f"{pair}_k{k}.npz"
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            from scipy.stats import pearsonr
            r_x = pearsonr(d["x"], d["ld"])[0]
            r_z = pearsonr(d["z_eff"][np.isfinite(d["z_eff"])],
                            d["ld"][np.isfinite(d["z_eff"])])[0]
            out.append((f"k={k} baseline", float(r_x), float(r_z)))
    return out


def plot_all(models, pair="height", k=1):
    fig = plt.figure(figsize=(15.5, 4.4 * len(models)))
    gs = fig.add_gridspec(len(models), 3, width_ratios=[1.5, 1.5, 1.6])

    for mi, model in enumerate(models):
        d = load(model, pair, k)
        n_heads = d["n_heads"]
        base_z = d["baseline"]["r_ld_zeff"]
        base_x = d["baseline"]["r_ld_x"]

        # Panel A: per-head bars
        axA = fig.add_subplot(gs[mi, 0])
        single = d["single"]
        delta_z = np.array([s["r_ld_zeff"] - base_z for s in single])
        delta_x = np.array([s["r_ld_x"] - base_x for s in single])
        order = np.argsort(np.abs(delta_z))[::-1]
        heads = np.arange(n_heads)
        width = 0.4
        axA.bar(heads - width/2, delta_z, width,
                label=r"$\Delta\,r$(LD, z_eff)", color="C0")
        axA.bar(heads + width/2, delta_x, width,
                label=r"$\Delta\,r$(LD, x)", color="C3")
        axA.axhline(0, color="black", linewidth=0.5)
        axA.set_xticks(heads)
        axA.set_xlabel("L0 head")
        axA.set_ylabel(r"$\Delta\,r$ (ablation − baseline)")
        axA.set_title(f"{model.replace('gemma2-', '').upper()}  "
                       f"|  per-head L0 ablation  (k={k})")
        if mi == 0:
            axA.legend(fontsize=9, loc="lower right")
        axA.grid(alpha=0.3, axis="y")

        # Panel B: cumulative top-k vs random-k
        axB = fig.add_subplot(gs[mi, 1])
        cumul = d["cumul_topk"]
        rand_k = d["random_k"]
        ks = np.arange(1, n_heads + 1)
        cumul_z = np.array([c["r_ld_zeff"] - base_z for c in cumul])
        cumul_x = np.array([c["r_ld_x"]    - base_x for c in cumul])
        rand_z_mean = np.array([
            np.mean([r["r_ld_zeff"] - base_z for r in rand_k[str(k)]])
            for k in ks
        ])
        rand_z_std = np.array([
            np.std([r["r_ld_zeff"] - base_z for r in rand_k[str(k)]])
            for k in ks
        ])
        rand_x_mean = np.array([
            np.mean([r["r_ld_x"] - base_x for r in rand_k[str(k)]])
            for k in ks
        ])
        rand_x_std = np.array([
            np.std([r["r_ld_x"] - base_x for r in rand_k[str(k)]])
            for k in ks
        ])

        axB.plot(ks, cumul_z, "o-", color="C0",
                  label=r"top-$k$ Δr(LD,z)")
        axB.fill_between(ks, rand_z_mean - rand_z_std, rand_z_mean + rand_z_std,
                          color="C0", alpha=0.15)
        axB.plot(ks, rand_z_mean, "o--", color="C0", alpha=0.6,
                  label=r"random-$k$ Δr(LD,z)")
        axB.plot(ks, cumul_x, "s-", color="C3",
                  label=r"top-$k$ Δr(LD,x)")
        axB.fill_between(ks, rand_x_mean - rand_x_std, rand_x_mean + rand_x_std,
                          color="C3", alpha=0.15)
        axB.plot(ks, rand_x_mean, "s--", color="C3", alpha=0.6,
                  label=r"random-$k$ Δr(LD,x)")
        axB.axhline(0, color="black", linewidth=0.5)
        axB.set_xlabel("# L0 heads ablated")
        axB.set_ylabel(r"$\Delta\,r$ (ablation − baseline)")
        axB.set_title(f"{model.replace('gemma2-', '').upper()}  "
                       f"|  cumulative ablation curve  (k={k})")
        if mi == 0:
            axB.legend(fontsize=8, loc="lower left")
        axB.grid(alpha=0.3)

        # Panel C: phase-space r(LD, x) vs r(LD, z_eff)
        # Three regions:
        #   bias = (low r_x, low r_z)   — origin region
        #   objective = (high r_x, low r_z) — right edge
        #   relativistic = (low r_x, high r_z) — top edge
        axC = fig.add_subplot(gs[mi, 2])

        # Region shading
        axC.axhspan(0.5, 1.05, xmin=0.0, xmax=0.5, alpha=0.07, color="C0",
                     zorder=0)   # relativistic
        axC.axvspan(0.5, 1.05, ymin=0.0, ymax=0.5, alpha=0.07, color="C2",
                     zorder=0)   # objective
        axC.add_patch(plt.Rectangle((0.0, 0.0), 0.5, 0.5,
                                     alpha=0.07, color="C3", zorder=0))   # biased
        axC.text(0.04, 0.96, "RELATIVISTIC", color="C0", fontsize=8.5,
                 fontweight="bold", va="top")
        axC.text(0.96, 0.04, "OBJECTIVE", color="C2", fontsize=8.5,
                 fontweight="bold", ha="right")
        axC.text(0.04, 0.04, "BIASED", color="C3", fontsize=8.5,
                 fontweight="bold")

        # Anchor points (other k values)
        anchors = get_anchor_points(model, pair)
        for label, ax_x, ax_z in anchors:
            color = "lightgray"
            marker = "*"
            if "zero-shot" in label:
                color = "tab:olive"; marker = "P"
            elif f"k={k} baseline" in label:
                color = "black"; marker = "o"
            axC.scatter(ax_x, ax_z, s=120, color=color, marker=marker,
                         zorder=4, edgecolor="black", linewidth=0.6)
            axC.annotate(label, (ax_x, ax_z),
                          xytext=(6, -3), textcoords="offset points",
                          fontsize=8, color="dimgray")

        # Cumulative top-k trajectory (colored by k)
        cmap = cm.viridis
        for ki, c in enumerate(cumul):
            color = cmap(ki / max(1, len(cumul) - 1))
            axC.scatter(c["r_ld_x"], c["r_ld_zeff"], s=80, color=color,
                         marker="D", zorder=3, edgecolor="black", linewidth=0.5,
                         label=f"top-{ki+1}" if ki in (0, len(cumul)//2, len(cumul)-1) else None)
        # Connect with line (path)
        axC.plot([c["r_ld_x"] for c in cumul],
                 [c["r_ld_zeff"] for c in cumul],
                 color="black", alpha=0.4, linewidth=0.8, zorder=2)

        # l0_all reference
        l0a = d["l0_all"]
        axC.scatter(l0a["r_ld_x"], l0a["r_ld_zeff"], s=180, color="C3",
                     marker="X", zorder=4, edgecolor="black", linewidth=0.8,
                     label="l0_all")
        axC.annotate("l0_all", (l0a["r_ld_x"], l0a["r_ld_zeff"]),
                      xytext=(5, 5), textcoords="offset points",
                      fontsize=8, color="C3", fontweight="bold")

        # Baseline reference
        axC.scatter(base_x, base_z, s=140, color="tab:green", marker="o",
                     zorder=5, edgecolor="black", linewidth=0.8,
                     label=f"baseline k={k}")
        axC.annotate(f"baseline k={k}", (base_x, base_z),
                      xytext=(5, -10), textcoords="offset points",
                      fontsize=8, color="tab:green", fontweight="bold")

        axC.set_xlabel(r"$r$(LD, x)   →  more absolute reading")
        axC.set_ylabel(r"$r$(LD, z_eff)   →  more relative reading")
        axC.set_xlim(-0.05, 1.05)
        axC.set_ylim(-0.05, 1.05)
        axC.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        axC.axvline(0, color="black", linewidth=0.4, alpha=0.3)
        axC.set_title(f"{model.replace('gemma2-', '').upper()}  |  "
                       f"phase-space trajectory under partial L0 ablation")
        axC.grid(alpha=0.3)
        if mi == 0:
            axC.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"Phase 2D — partial L0 ablation  |  pair={pair}  |  k={k}",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / f"p2d_partial_l0_k{k}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"])
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=1)
    args = p.parse_args()

    present = [m for m in args.models
               if (REPO / "results" / f"p2d_partial_l0_{m}_{args.pair}_k{args.k}.json").exists()]
    if not present:
        print("no p2d JSONs found")
        return
    plot_all(present, args.pair, args.k)


if __name__ == "__main__":
    main()
