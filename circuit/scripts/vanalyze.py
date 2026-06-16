"""Vision baseline analysis — partial correlations and cell-mean stats.

Reads vextract_<short>_residuals.npz and reports:
  - Pearson r(LD, z), r(LD, x), r(z, x)
  - Partial correlation r(LD, z | x)
  - Cell-mean r(LD, z) (averaging over seeds per (x, z) cell)
  - Per-x slice slope of LD vs z (robust to x-z correlation)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="basename of the vextract NPZ (e.g. vextract_e4b_it)")
    args = ap.parse_args()

    npz_path = REPO / "results" / f"{args.name}_residuals.npz"
    d = np.load(npz_path)
    ld = d["ld"]
    z = d["z"].astype(np.float64)
    x = d["x"].astype(np.float64)
    seed = d["seed"]
    n = len(ld)

    print(f"=== analysis: {args.name}  (N = {n}) ===\n")

    # Pairwise pearson
    r_lz = float(np.corrcoef(ld, z)[0, 1])
    r_lx = float(np.corrcoef(ld, x)[0, 1])
    r_zx = float(np.corrcoef(z, x)[0, 1])
    print("Pairwise pearson:")
    print(f"  r(LD, z) = {r_lz:+.3f}")
    print(f"  r(LD, x) = {r_lx:+.3f}")
    print(f"  r(z, x)  = {r_zx:+.3f}   (confound on this grid)")

    # Partial correlation r(LD, z | x)
    # = (r_lz - r_lx * r_zx) / sqrt((1 - r_lx^2)(1 - r_zx^2))
    denom = np.sqrt(max(1e-12, (1 - r_lx ** 2) * (1 - r_zx ** 2)))
    pc_lz_x = (r_lz - r_lx * r_zx) / denom
    pc_lx_z = (r_lx - r_lz * r_zx) / np.sqrt(max(1e-12,
                                                    (1 - r_lz ** 2) * (1 - r_zx ** 2)))
    print(f"\nPartial corr (control for the confound):")
    print(f"  r(LD, z | x) = {pc_lz_x:+.3f}")
    print(f"  r(LD, x | z) = {pc_lx_z:+.3f}")

    # Cell-mean stats — average LD over seeds within each (x, z)
    cells: dict[tuple[float, float], list[float]] = defaultdict(list)
    for i in range(n):
        cells[(round(float(x[i]), 4), round(float(z[i]), 4))].append(float(ld[i]))
    cell_means = []
    for (xk, zk), v in cells.items():
        cell_means.append({"x": xk, "z": zk, "ld": float(np.mean(v)),
                            "n": len(v)})
    if cell_means:
        cm_x = np.array([c["x"] for c in cell_means])
        cm_z = np.array([c["z"] for c in cell_means])
        cm_ld = np.array([c["ld"] for c in cell_means])
        cm_r_lz = float(np.corrcoef(cm_ld, cm_z)[0, 1])
        cm_r_lx = float(np.corrcoef(cm_ld, cm_x)[0, 1])
        cm_r_zx = float(np.corrcoef(cm_z, cm_x)[0, 1])
        print(f"\nCell-mean (n_cells = {len(cell_means)}, "
              f"avg seeds/cell = {n/len(cell_means):.1f}):")
        print(f"  r(LD_cell, z)  = {cm_r_lz:+.3f}")
        print(f"  r(LD_cell, x)  = {cm_r_lx:+.3f}")
        print(f"  r(z, x)        = {cm_r_zx:+.3f}")
        denom = np.sqrt(max(1e-12, (1 - cm_r_lx ** 2) * (1 - cm_r_zx ** 2)))
        cm_pc = (cm_r_lz - cm_r_lx * cm_r_zx) / denom
        print(f"  r(LD_cell, z | x) = {cm_pc:+.3f}")

    # Per-x slope of LD vs z (using cells with that x)
    print(f"\nPer-x slope of cell-mean LD vs z:")
    print(f"{'x':>4} {'n_z':>4} {'r(LD,z)':>9} {'slope':>9} {'mean_LD':>9}")
    by_x = defaultdict(list)
    for c in cell_means:
        by_x[c["x"]].append((c["z"], c["ld"]))
    pos_count = 0
    neg_count = 0
    for xk in sorted(by_x):
        zs_x = np.array([t[0] for t in by_x[xk]])
        lds_x = np.array([t[1] for t in by_x[xk]])
        if len(zs_x) < 3:
            continue
        r = float(np.corrcoef(lds_x, zs_x)[0, 1])
        slope = float(np.polyfit(zs_x, lds_x, 1)[0])
        if slope > 0: pos_count += 1
        else: neg_count += 1
        print(f"{xk:>4.0f} {len(zs_x):>4} {r:>+9.3f} {slope:>+9.3f} "
              f"{float(np.mean(lds_x)):>+9.3f}")
    print(f"\nDirection: {pos_count} x-slices positive, {neg_count} negative")


if __name__ == "__main__":
    main()
