"""Vision LD-vs-z scatter grid (mirror of p2a_ld_vs_z_height_*.png).

Reads results/<in_name>/<cell>_baseline_lds.npz produced by vphase_baseline_lds.py.
Plots a (rows = n_ref) × (cols = model) grid where each cell shows:
  - x-axis = z (or x at n_ref=1, mirroring p2a where k=0 has no z_eff)
  - y-axis = LD = logit(big) − logit(small)
  - points colored by target size x (perceptually uniform colormap)
  - title: r(LD, z), partial r(z|x), per-x slope distribution

Output: figures/vphase_ld_vs_z.png

Usage:
    python vphase_ld_vs_z.py
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def per_x_slopes(ld, z, x):
    """Returns (n_pos, n_neg, list of (x_value, slope, r))."""
    cells = defaultdict(list)
    for i in range(len(ld)):
        cells[(round(float(x[i]), 4), round(float(z[i]), 4))].append(float(ld[i]))
    cell = {k: float(np.mean(v)) for k, v in cells.items()}
    by_x = defaultdict(list)
    for (xk, zk), m in cell.items():
        by_x[xk].append((zk, m))
    pos = neg = 0
    detail = []
    for xk in sorted(by_x.keys()):
        vals = by_x[xk]
        if len(vals) < 3:
            continue
        zs = np.array([t[0] for t in vals])
        ms = np.array([t[1] for t in vals])
        slope = float(np.polyfit(zs, ms, 1)[0])
        r = float(np.corrcoef(zs, ms)[0, 1])
        detail.append((float(xk), slope, r))
        if slope > 0:
            pos += 1
        elif slope < 0:
            neg += 1
    return pos, neg, detail


def partial_r(ld, z, x):
    r_z = float(np.corrcoef(ld, z)[0, 1])
    r_x = float(np.corrcoef(ld, x)[0, 1])
    r_zx = float(np.corrcoef(z, x)[0, 1])
    denom = math.sqrt(max(1e-12, (1 - r_x ** 2) * (1 - r_zx ** 2)))
    return r_z, r_x, (r_z - r_x * r_zx) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default=str(REPO / "results" / "vphase_grid"))
    ap.add_argument("--out",
                    default=str(REPO / "figures" / "vphase_ld_vs_z.png"))
    ap.add_argument("--shorts", nargs="+",
                    default=["gemma4-e2b-it", "gemma4-e4b-it", "gemma4-31b-it"])
    ap.add_argument("--n-refs", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    nrows = len(args.n_refs)
    ncols = len(args.shorts)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.6 * ncols, 3.6 * nrows),
                              squeeze=False)

    # Find shared LD ylim across cells (for visual comparability)
    ylim_all: list[float] = []
    cell_data: dict = {}
    for short in args.shorts:
        for n_ref in args.n_refs:
            cands = sorted(in_dir.glob(f"{short}_n{n_ref}_L*_baseline_lds.npz"))
            if not cands:
                continue
            d = np.load(cands[0], allow_pickle=True)
            cell_data[(n_ref, short)] = (cands[0], d)
            ylim_all.extend([float(d["ld"].min()), float(d["ld"].max())])

    if not ylim_all:
        raise SystemExit(f"no baseline_lds NPZs found in {in_dir}")
    ylo = min(ylim_all) - 0.5
    yhi = max(ylim_all) + 0.5

    for r, n_ref in enumerate(args.n_refs):
        for c, short in enumerate(args.shorts):
            ax = axes[r, c]
            data = cell_data.get((n_ref, short))
            if data is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                ax.set_title(f"{short}  |  n_ref={n_ref}", fontsize=10)
                continue
            path, d = data
            ld = d["ld"].astype(np.float64)
            z = d["z"].astype(np.float64)
            x = d["x"].astype(np.float64)

            r_z, r_x, pc = partial_r(ld, z, x)
            pos, neg, detail = per_x_slopes(ld, z, x)

            sc = ax.scatter(z, ld, c=x, cmap="viridis", s=12, alpha=0.65,
                             edgecolors="none")
            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)

            # per-x slope lines (cell mean per (x, z) → linear fit per x)
            cells = defaultdict(list)
            for i in range(len(ld)):
                cells[(round(float(x[i]), 4), round(float(z[i]), 4))].append(
                    float(ld[i]))
            cm = {k: float(np.mean(v)) for k, v in cells.items()}
            by_x = defaultdict(list)
            for (xk, zk), m in cm.items():
                by_x[xk].append((zk, m))
            xs_uniq = sorted(by_x.keys())
            cmap = plt.get_cmap("viridis")
            x_norm_lo = min(xs_uniq) if xs_uniq else 0
            x_norm_hi = max(xs_uniq) if xs_uniq else 1
            for xk in xs_uniq:
                vals = sorted(by_x[xk], key=lambda t: t[0])
                if len(vals) < 3:
                    continue
                zs = np.array([t[0] for t in vals])
                ms = np.array([t[1] for t in vals])
                slope, intercept = np.polyfit(zs, ms, 1)
                xx = np.array([zs.min(), zs.max()])
                col = cmap((xk - x_norm_lo) /
                            max(1e-9, (x_norm_hi - x_norm_lo)))
                ax.plot(xx, slope * xx + intercept, color=col, lw=1.0,
                         alpha=0.55)

            ax.set_ylim(ylo, yhi)
            ax.set_title(
                f"{short.replace('gemma4-', '').upper()}  |  n_ref={n_ref}\n"
                f"r(LD,z)={r_z:+.2f}  r(LD,x)={r_x:+.2f}  pc(z|x)={pc:+.2f}  "
                f"+{pos}/-{neg}", fontsize=9)
            ax.grid(alpha=0.25)
            if r == nrows - 1:
                ax.set_xlabel("z")
            if c == 0:
                ax.set_ylabel("LD = logit(big) − logit(small)")
            if r == 0 and c == ncols - 1:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("target size x (px)", fontsize=8)

    fig.suptitle("Vision baseline LD vs z by n_ref × model "
                 "(line = per-x cell-mean fit; color = target size x)",
                 fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
