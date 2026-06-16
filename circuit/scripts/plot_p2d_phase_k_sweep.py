"""Phase 2D — k-sweep collapsed phase plot (baseline only).

A single-panel version of `plot_p2d_phase_grid_partial.py`: instead of one
panel per k, all k values for a given model are plotted as a single
trajectory in partial-correlation phase space, and 2B and 9B are overlaid.

No interventions (no l0_all, no manifold/dual, no primal/proj_out). No
quadrant background colours. Axis labels and corner annotations match the
p2v_phase_trajectory_partial style: lower-case "objective" / "relativistic"
arrows, RELATIVISTIC top-left and OBJECTIVE bottom-right corner labels.

Output: figures/p2d_phase_k_sweep.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def partial_corr(r_lz, r_lx, r_zx):
    """Returns (p_lz_x, p_lx_z) — partial r(LD,z|x) and partial r(LD,x|z)."""
    if not (np.isfinite(r_lz) and np.isfinite(r_lx) and np.isfinite(r_zx)):
        return float("nan"), float("nan")
    den_z = np.sqrt(max(0.0, (1 - r_lx ** 2) * (1 - r_zx ** 2)))
    den_x = np.sqrt(max(0.0, (1 - r_lz ** 2) * (1 - r_zx ** 2)))
    p_z = (r_lz - r_lx * r_zx) / den_z if den_z > 1e-9 else float("nan")
    p_x = (r_lx - r_lz * r_zx) / den_x if den_x > 1e-9 else float("nan")
    p_z = max(-1.0, min(1.0, p_z))
    p_x = max(-1.0, min(1.0, p_x))
    return p_z, p_x


def baseline_partial(model: str, pair: str, k: int) -> dict | None:
    """Load per-prompt LD and compute partial correlations.

    At k=0 the relativistic variable z is undefined (no context to compute
    sample-mean μ from), so partial correlations don't apply. We place k=0
    at (r(LD, x), 0): use the raw x-correlation as the "objective"
    coordinate and zero on the relativistic axis by convention.
    """
    npz = REPO / "results" / "p2_ld" / model / f"{pair}_k{k}.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    ld = d["ld"].astype(np.float64)
    x = d["x"].astype(np.float64)
    if k == 0:
        m = np.isfinite(ld) & np.isfinite(x)
        if m.sum() < 3:
            return None
        r_lx = safe_pearson(x[m], ld[m])
        return {
            "k": 0, "n": int(m.sum()),
            "r_ld_z": float("nan"), "r_ld_x": r_lx,
            "r_zx": float("nan"),
            "p_lz_x": 0.0,           # by convention (z undefined)
            "p_lx_z": float(r_lx),   # raw x-correlation
            "k0_convention": True,
        }
    z = d["z_eff"].astype(np.float64)
    mask = np.isfinite(z) & np.isfinite(ld) & np.isfinite(x)
    if mask.sum() < 3:
        return None
    r_lz = safe_pearson(z[mask], ld[mask])
    r_lx = safe_pearson(x[mask], ld[mask])
    r_zx = safe_pearson(z[mask], x[mask])
    p_z, p_x = partial_corr(r_lz, r_lx, r_zx)
    return {
        "k": k, "n": int(mask.sum()),
        "r_ld_z": r_lz, "r_ld_x": r_lx, "r_zx": r_zx,
        "p_lz_x": p_z, "p_lx_z": p_x,
        "k0_convention": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[0, 1, 2, 4, 5, 8, 15, 30, 50])
    ap.add_argument("--pair", default="height")
    ap.add_argument("--out",
                    default="figures/p2d_phase_k_sweep.png")
    args = ap.parse_args()

    # Per-model trajectories
    traj: dict[str, list[dict]] = {}
    for model in args.models:
        rows = []
        for k in args.ks:
            r = baseline_partial(model, args.pair, k)
            if r is None:
                print(f"  [skip] {model} k={k} — no per-prompt LD data")
                continue
            rows.append(r)
        traj[model] = rows
        print(f"=== {model} ({args.pair}) ===")
        print(f"{'k':>3s}  {'r_lz':>7s}  {'r_lx':>7s}  {'r_zx':>7s}  "
               f"{'pc(x|z)':>8s}  {'pc(z|x)':>8s}")
        for r in rows:
            print(f"{r['k']:>3d}  {r['r_ld_z']:+7.3f}  {r['r_ld_x']:+7.3f}  "
                   f"{r['r_zx']:+7.3f}  {r['p_lx_z']:+8.3f}  {r['p_lz_x']:+8.3f}")

    # Plot — one panel per model, side by side
    all_ks_present = sorted({r["k"] for rows in traj.values() for r in rows})
    cmap = plt.get_cmap("viridis")
    # Log colour scale: matches the (roughly geometric) spacing of the
    # measured k values (1, 2, 4, 5, 8, 15, 30, 50) so adjacent k's are
    # visually separable. SymLog handles k=0 cleanly (linear within
    # [-linthresh, +linthresh]).
    vmax_k = max(all_ks_present)
    if 0 in all_ks_present:
        norm = mcolors.SymLogNorm(linthresh=1.0, linscale=1.0,
                                    vmin=0, vmax=vmax_k, base=10)
    else:
        norm = mcolors.LogNorm(vmin=max(1, min(all_ks_present)), vmax=vmax_k)

    model_titles = {
        "gemma2-2b": "Gemma 2 2B",
        "gemma2-9b": "Gemma 2 9B",
    }

    n_models = sum(1 for rows in traj.values() if rows)
    fig, axes = plt.subplots(1, n_models,
                               figsize=(6.0 * n_models, 6.5),
                               sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]
    axes_iter = iter(axes)

    xlim = (-0.05, 1.05)
    ylim = (-0.05, 1.05)
    sc = None

    for model, rows in traj.items():
        if not rows:
            continue
        ax = next(axes_iter)
        rows = sorted(rows, key=lambda r: r["k"])
        xs = [r["p_lx_z"] for r in rows]
        ys = [r["p_lz_x"] for r in rows]
        ks_ = [r["k"] for r in rows]

        # Corner labels per panel (smaller / lighter)
        ax.text(xlim[0] + 0.02, 1.02, "RELATIVISTIC",
                 color="C0", fontsize=10, fontweight="bold", va="top",
                 alpha=0.85)
        ax.text(xlim[1] - 0.02, 0.04, "OBJECTIVE",
                 color="C2", fontsize=10, fontweight="bold",
                 ha="right", va="bottom", alpha=0.85)

        # Connecting line through all k including k=0 anchor
        if len(xs) >= 2:
            ax.plot(xs, ys, color="0.5", linestyle="-",
                     lw=1.4, alpha=0.55, zorder=3)

        # Scatter colored by k
        sc = ax.scatter(xs, ys, c=ks_, cmap=cmap, norm=norm,
                         s=140, edgecolor="black", linewidth=1.0, zorder=5)

        # Annotate just the endpoints — the k=0 anchor on the x-axis,
        # and the largest k point — to orient the trajectory direction.
        endpoints = []
        if any(k == 0 for k in ks_):
            i = ks_.index(0)
            endpoints.append((xs[i], ys[i], "k=0", (8, -2)))
        if max(ks_) > 0:
            i = ks_.index(max(ks_))
            endpoints.append((xs[i], ys[i], f"k={max(ks_)}", (8, 6)))
        for x_, y_, lab, off in endpoints:
            ax.annotate(lab, (x_, y_),
                         xytext=off, textcoords="offset points",
                         fontsize=10, color="black", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25",
                                    facecolor="white", edgecolor="0.6",
                                    alpha=0.9))

        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax.axvline(0, color="black", lw=0.4, alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"corr(LD, $x \mid z$)  →  objective", fontsize=12)
        ax.set_title(model_titles.get(model, model), fontsize=12,
                      fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

    # Y label only on the leftmost panel
    axes[0].set_ylabel(r"corr(LD, $z \mid x$)  →  relativistic", fontsize=12)

    # k=0 convention footnote — only on first panel, low-key
    if 0 in all_ks_present:
        axes[0].text(0.02, 0.02,
                      "k=0 placed at (corr(LD, x), 0) — z undefined",
                      transform=axes[0].transAxes, ha="left", va="bottom",
                      fontsize=8, color="0.35", style="italic")

    # Single shared colorbar on the right
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02,
                             aspect=30)
        cbar.set_label("k (number of context examples)", fontsize=11)
        cbar.set_ticks(all_ks_present)
        cbar.set_ticklabels([str(k) for k in all_ks_present])

    fig.suptitle(f"Baseline phase trajectory across context-size $k$  "
                  f"(pair = {args.pair})",
                  fontsize=12, y=1.00)
    out = REPO / args.out
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
