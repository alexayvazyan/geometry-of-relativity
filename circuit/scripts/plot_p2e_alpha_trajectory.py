"""Phase 2E — α-sweep trajectory in partial-correlation phase space.

For each model, plots the path traced by manifold-shift α from 0 to 2.0,
with markers colored by α (viridis). Same partial-correlation axes as
plot_p2d_phase_grid_partial.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def partial_corr(r_lz, r_lx, r_zx):
    if not (np.isfinite(r_lz) and np.isfinite(r_lx) and np.isfinite(r_zx)):
        return float("nan"), float("nan")
    den_z = np.sqrt(max(0.0, (1 - r_lx**2) * (1 - r_zx**2)))
    den_x = np.sqrt(max(0.0, (1 - r_lz**2) * (1 - r_zx**2)))
    p_z = (r_lz - r_lx * r_zx) / den_z if den_z > 1e-9 else float("nan")
    p_x = (r_lx - r_lz * r_zx) / den_x if den_x > 1e-9 else float("nan")
    return max(-1.0, min(1.0, p_z)), max(-1.0, min(1.0, p_x))


def setup_axes(ax):
    quadrants = [
        ((0.0, 0.0), (0.5, 0.5), "C3", "BIASED",       "left",  "bottom"),
        ((0.5, 0.0), (1.0, 0.5), "C2", "OBJECTIVE",    "right", "bottom"),
        ((0.0, 0.5), (1.0, 1.0), "C0", "RELATIVISTIC", "left",  "top"),
        ((0.5, 0.5), (1.0, 1.0), "gold", "COMPLETE",   "right", "top"),
    ]
    for (x0, y0), (x1, y1), color, label, ha, va in quadrants:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    alpha=0.10, color=color, zorder=0))
        text_color = "darkgoldenrod" if color == "gold" else color
        tx = x1 - 0.02 if ha == "right" else x0 + 0.02
        ty = y1 - 0.02 if va == "top" else y0 + 0.02
        ax.text(tx, ty, label, color=text_color, fontsize=10,
                fontweight="bold", ha=ha, va=va)
    ax.set_xlim(-0.55, 1.05)
    ax.set_ylim(-0.85, 1.05)
    ax.axhline(0.5, color="black", linewidth=0.3, alpha=0.4)
    ax.axvline(0.5, color="black", linewidth=0.3, alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.grid(alpha=0.2)


def plot_panel(ax, model, pair, k):
    """Plot one (model, pair) α-trajectory panel; returns r_zx used."""
    setup_axes(ax)

    # Per-pair r_zx (depends on x-grid and sigma).
    rows = [json.loads(l) for l in (REPO / "data" / "p2_shot_sweep" / f"{pair}_k{k}.jsonl").open()]
    r_zx = safe_pearson(np.array([r["x"] for r in rows]),
                        np.array([r["z_eff"] for r in rows]))

    path = REPO / "results" / f"p2e_alpha_sweep_{model}_{pair}_k{k}.json"
    if not path.exists():
        ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="red")
        return r_zx
    with path.open() as f:
        D = json.load(f)

    items = []
    for name, r in D["results"].items():
        if r.get("alpha") is None:
            continue
        items.append((r["alpha"], r))
    items.sort()
    alphas = np.array([a for a, _ in items])

    cmap = cm.viridis
    norm_alpha = (alphas - alphas.min()) / max(1e-9, alphas.max() - alphas.min())

    coords = []
    for alpha, r in items:
        p_z, p_x = partial_corr(r["r_ld_zeff"], r["r_ld_x"], r_zx)
        coords.append((p_x, p_z, r))
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])

    ax.plot(xs, ys, "-", color="black", alpha=0.4, linewidth=1.2, zorder=2)

    for i, (alpha, _r) in enumerate(items):
        color = cmap(norm_alpha[i])
        size = 200 + 500 * (1 if alpha in (0.0, 0.75, 1.0, 2.0) else 0.0)
        ax.scatter(xs[i], ys[i], s=size, color=color, marker="D",
                   edgecolor="black", linewidth=0.8, zorder=4)
        if alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
            txt = f"α={alpha:.2f}"
            if alpha == 0.0:
                txt = "α=0\n(base)"
            ax.annotate(txt, (xs[i], ys[i]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=8, fontweight="bold")

    ax.set_title(f"{model.replace('gemma2-', '').upper()}  |  {pair}  "
                 f"(r(z,x)={r_zx:+.2f})", fontsize=11)
    return r_zx


def main():
    k = 15
    models = ["gemma2-2b", "gemma2-9b"]
    pairs = ["height", "weight", "speed"]

    fig, axes = plt.subplots(len(models), len(pairs),
                              figsize=(5.8 * len(pairs), 5.6 * len(models)),
                              squeeze=False)

    for mi, model in enumerate(models):
        for pi, pair in enumerate(pairs):
            ax = axes[mi, pi]
            r_zx = plot_panel(ax, model, pair, k)
            print(f"  {model}/{pair}: r(z_eff,x)={r_zx:+.3f}")
            if pi == 0:
                ax.set_ylabel(r"partial $r$(LD, z | x)  →  relativistic", fontsize=10)
            if mi == len(models) - 1:
                ax.set_xlabel(r"partial $r$(LD, x | z)  →  objective", fontsize=10)

    fig.suptitle("Phase 2E — manifold α-sweep across three features, "
                 "partial-correlation phase space\n"
                 "(α=0 is baseline; α=2.0 over-corrects past z=0 into anti-relativity)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "p2e_alpha_trajectory_multi.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
