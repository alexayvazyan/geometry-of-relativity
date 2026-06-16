"""Cross-feature DLA head summary on Gemma 2 9B.

Reads p2s_dla_full_gemma2-9b{,_weight,_speed}.npz and produces:

  - LEFT  panel: heatmap (rows = union of top-K cells across features,
                  cols = {height, weight, speed}, value = σ_DLA·|ρ|).
                  Cells sorted by max σ_DLA·|ρ| across features.
  - RIGHT panel: top — Jaccard overlap of top-K sets at K=5, 10, 15;
                  bottom — Spearman rank correlation across the full
                  672-cell grid for each feature pair.

Output:
  figures/p2s_dla_xfeat_summary_gemma2-9b.png

Usage:
  python3 scripts/p2s_dla_xfeat_summary.py --short gemma2-9b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent

FEATURES = ["height", "weight", "speed"]
COLORS = {"height": "C0", "weight": "C2", "speed": "C5"}


def load_grid(short: str, feature: str) -> tuple[np.ndarray, np.ndarray]:
    tag = short if feature == "height" else f"{short}_{feature}"
    path = REPO / "results" / f"p2s_dla_full_{tag}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = np.load(path)
    return d["std_dla"], d["rho_grid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-9b")
    ap.add_argument("--top-k-display", type=int, default=15,
                    help="union of top-K cells per feature shown in heatmap")
    ap.add_argument("--positive-rho-only", action="store_true", default=True,
                    help="filter to positive ρ(DLA, z) (canonical convention)")
    ap.add_argument("--heatmap-only", action="store_true",
                    help="render only the per-cell heatmap; skip the Jaccard "
                         "and Spearman side panels")
    args = ap.parse_args()

    grids: dict[str, np.ndarray] = {}
    rhos: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    for feat in FEATURES:
        sigma, rho = load_grid(args.short, feat)
        sigmas[feat] = sigma
        rhos[feat] = rho
        score = sigma * np.abs(rho)
        if args.positive_rho_only:
            score = np.where(rho > 0, score, 0.0)
        grids[feat] = score

    n_layers, n_heads = grids[FEATURES[0]].shape
    print(f"grid: {n_layers}L × {n_heads}H = {n_layers * n_heads} cells")

    # Top-K per feature
    K = args.top_k_display
    top_cells: dict[str, set] = {}
    top_ranked: dict[str, list[tuple[int, int]]] = {}
    for feat in FEATURES:
        flat = grids[feat].flatten()
        idx = np.argsort(flat)[::-1][:K]
        cells = [(int(i // n_heads), int(i % n_heads)) for i in idx]
        top_cells[feat] = set(cells)
        top_ranked[feat] = cells
        print(f"\n{feat} top-{K}: " +
              ", ".join(f"L{L}H{h}" for L, h in cells[:10]))

    # Union of top-K cells (visualised as heatmap rows)
    union_cells = set()
    for feat in FEATURES:
        union_cells |= top_cells[feat]
    union_list = sorted(union_cells, key=lambda c: -max(
        grids[f][c[0], c[1]] for f in FEATURES))
    print(f"\nunion of top-{K} across features: {len(union_list)} cells")

    # Heatmap matrix
    H = np.zeros((len(union_list), len(FEATURES)), dtype=np.float32)
    for i, (L, h) in enumerate(union_list):
        for j, feat in enumerate(FEATURES):
            H[i, j] = grids[feat][L, h]

    # Membership grid (which feature has each cell in its top-K)
    M = np.zeros_like(H, dtype=bool)
    for i, c in enumerate(union_list):
        for j, feat in enumerate(FEATURES):
            M[i, j] = c in top_cells[feat]

    # Jaccard overlap @ K=5, 10, 15
    Ks = [5, 10, 15]
    jaccard_data = {}
    for K_eval in Ks:
        mat = np.zeros((len(FEATURES), len(FEATURES)))
        for i, fa in enumerate(FEATURES):
            for j, fb in enumerate(FEATURES):
                a = set(top_ranked[fa][:K_eval])
                b = set(top_ranked[fb][:K_eval])
                if not (a | b):
                    mat[i, j] = 1.0
                else:
                    mat[i, j] = len(a & b) / len(a | b)
        jaccard_data[K_eval] = mat

    # Full-grid Spearman rank correlation across feature pairs
    spear_mat = np.zeros((len(FEATURES), len(FEATURES)))
    for i, fa in enumerate(FEATURES):
        for j, fb in enumerate(FEATURES):
            rs, _ = spearmanr(grids[fa].flatten(), grids[fb].flatten())
            spear_mat[i, j] = float(rs)

    # ---- Plot ----
    if args.heatmap_only:
        fig = plt.figure(figsize=(7.5, max(6, 0.32 * len(union_list) + 1.5)))
        ax_hm = fig.add_subplot(1, 1, 1)
        ax_jac = ax_sp = None
    else:
        fig = plt.figure(figsize=(15, max(8, 0.32 * len(union_list) + 2)))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.4],
                                height_ratios=[1.05, 1], hspace=0.35, wspace=0.18)
        ax_hm = fig.add_subplot(gs[:, 0])
        ax_jac = fig.add_subplot(gs[0, 1])
        ax_sp = fig.add_subplot(gs[1, 1])

    # Heatmap
    vmax = float(H.max())
    im = ax_hm.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax_hm.set_xticks(range(len(FEATURES)))
    ax_hm.set_xticklabels(FEATURES, fontsize=14)
    ax_hm.set_yticks(range(len(union_list)))
    ax_hm.set_yticklabels([f"L{L}H{h}" for L, h in union_list], fontsize=10)
    ax_hm.set_title(f"Per-cell σ_DLA·|ρ|+ "
                     f"(union of top-{K} per feature, sorted by max across features)",
                     fontsize=13)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.04)
    cbar.set_label(r"σ$_{DLA}\cdot|\rho|$", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Mark with a black border the cells in this feature's top-K
    for i in range(len(union_list)):
        for j in range(len(FEATURES)):
            if M[i, j]:
                ax_hm.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False, edgecolor="white",
                                                lw=1.5))

    # Annotate each value
    for i in range(len(union_list)):
        for j in range(len(FEATURES)):
            v = H[i, j]
            if v > 0.0:
                ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < vmax * 0.55 else "black",
                            fontsize=9)

    if not args.heatmap_only:
        # Jaccard panel — three small overlap matrices stacked horizontally
        ax_jac.set_axis_off()
        sub_w = 1.0 / len(Ks)
        for ki, K_eval in enumerate(Ks):
            sub_ax = ax_jac.inset_axes([ki * sub_w, 0.0, sub_w * 0.92, 1.0])
            mat = jaccard_data[K_eval]
            sub_ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            sub_ax.set_xticks(range(len(FEATURES)))
            sub_ax.set_yticks(range(len(FEATURES)))
            if ki == 0:
                sub_ax.set_yticklabels([f[0].upper() for f in FEATURES],
                                         fontsize=11)
            else:
                sub_ax.set_yticklabels([])
            sub_ax.set_xticklabels([f[0].upper() for f in FEATURES], fontsize=11)
            sub_ax.set_title(f"K={K_eval}", fontsize=11)
            for i in range(len(FEATURES)):
                for j in range(len(FEATURES)):
                    color = "white" if mat[i, j] > 0.5 else "black"
                    sub_ax.text(j, i, f"{mat[i, j]:.2f}", ha="center",
                                 va="center", color=color, fontsize=10)
        ax_jac.set_title("Jaccard overlap of top-K sets across feature pairs",
                          fontsize=12)

        # Spearman panel — full-grid rank correlation
        sp_im = ax_sp.imshow(spear_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_sp.set_xticks(range(len(FEATURES)))
        ax_sp.set_yticks(range(len(FEATURES)))
        ax_sp.set_xticklabels(FEATURES, fontsize=11)
        ax_sp.set_yticklabels(FEATURES, fontsize=11)
        ax_sp.set_title(
            f"Spearman rank corr of σ_DLA·|ρ| across all {n_layers * n_heads} cells",
            fontsize=12)
        for i in range(len(FEATURES)):
            for j in range(len(FEATURES)):
                color = "white" if abs(spear_mat[i, j]) > 0.5 else "black"
                ax_sp.text(j, i, f"{spear_mat[i, j]:+.2f}", ha="center",
                            va="center", color=color, fontsize=11)
        fig.colorbar(sp_im, ax=ax_sp, fraction=0.05, pad=0.04,
                      label="Spearman ρ")

    short_display = {"gemma2-2b": "Gemma2-2B",
                       "gemma2-9b": "Gemma2-9B"}.get(args.short, args.short)
    fig.suptitle(
        f"{short_display} — DLA head ranking across gradable adjectives",
        fontsize=16, y=0.995)
    out = REPO / "figures" / f"p2s_dla_xfeat_summary_{args.short}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
