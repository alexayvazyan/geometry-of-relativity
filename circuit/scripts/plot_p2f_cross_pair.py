"""Phase 2F — cross-pair summary of z-tracking attention heads.

Layout: 3-row grid for each model.
  Row 1: |r(attn-to-ctx-values, z_eff)| × pairs (3 columns) — find heads that
         track z across multiple pairs.
  Row 2: signed r per pair × pair (highlighting cross-pair consistency).
  Row 3: cross-pair signed mean — if a head's r is negative across all 3 pairs,
         it's robustly z-tracking; mixed signs suggest a pair-specific artifact.

Output: figures/p2f_cross_pair_<model>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parent.parent
RES_DIR = REPO / "results"
FIG_DIR = REPO / "figures"

PRIMARY = {"gemma2-2b": (1, 6), "gemma2-9b": (1, 11)}
PAIRS = ["height", "weight", "speed"]


def load_corr(model, pair, k=15):
    p = RES_DIR / f"p2f_attn_circuit_{model}_{pair}_k{k}.json"
    with p.open() as f:
        D = json.load(f)
    return np.array(D["r_ctx_attn_z"], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(PRIMARY))
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--top-k", type=int, default=8,
                    help="how many cross-pair-consistent heads to print")
    args = p.parse_args()

    Cs = [load_corr(args.model, pair, args.k) for pair in PAIRS]
    n_layers, n_heads = Cs[0].shape

    fig, axes = plt.subplots(3, len(PAIRS),
                              figsize=(4.0 * len(PAIRS), 3.6 * 3),
                              squeeze=False)

    # Row 0: signed r per pair.
    vmax = max(np.nanmax(np.abs(C)) for C in Cs)
    for j, (pair, C) in enumerate(zip(PAIRS, Cs)):
        ax = axes[0, j]
        im = ax.imshow(C, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        ax.set_title(f"{pair}", fontsize=10)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        if j == 0:
            ax.set_ylabel("signed r\nlayer")
        ax.set_xlabel("head")
        Lp, Hp = PRIMARY[args.model]
        ax.add_patch(plt.Rectangle((Hp - 0.5, Lp - 0.5), 1, 1,
                                    fill=False, edgecolor="lime", linewidth=2.0))
        plt.colorbar(im, ax=ax, fraction=0.04)

    # Row 1: |r| per pair (consistency without sign concern).
    for j, (pair, C) in enumerate(zip(PAIRS, Cs)):
        ax = axes[1, j]
        im = ax.imshow(np.abs(C), aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{pair}  |r|", fontsize=10)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        if j == 0:
            ax.set_ylabel("|r|\nlayer")
        ax.set_xlabel("head")
        plt.colorbar(im, ax=ax, fraction=0.04)

    # Row 2: cross-pair statistics.
    Cstack = np.stack(Cs, axis=0)            # (3, L, H)
    mean_signed = Cstack.mean(0)             # average — punishes sign flips
    consistency = np.sign(Cs[0]) * np.sign(Cs[1]) * np.sign(Cs[2])  # +1 if all same sign
    min_abs = np.min(np.abs(Cstack), axis=0) * consistency  # signed min |r|

    for j, (M, title, cmap) in enumerate([
        (mean_signed, "mean signed r (across 3 pairs)", "RdBu_r"),
        (np.abs(mean_signed), "|mean signed r|", "magma"),
        (min_abs, "signed min |r| (= 0 if any sign flips)", "RdBu_r"),
    ]):
        ax = axes[2, j]
        if cmap == "RdBu_r":
            im = ax.imshow(M, aspect="auto", cmap=cmap,
                            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
        else:
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        if j == 0:
            ax.set_ylabel("cross-pair\nlayer")
        ax.set_xlabel("head")
        Lp, Hp = PRIMARY[args.model]
        ax.add_patch(plt.Rectangle((Hp - 0.5, Lp - 0.5), 1, 1,
                                    fill=False, edgecolor="lime", linewidth=2.0))
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(f"Phase 2F — z-tracking attention across pairs  |  {args.model}  k={args.k}\n"
                 f"(circled lime = primary head from Phase 2B; "
                 f"top row = signed r, mid row = |r|, bottom = cross-pair stats)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / f"p2f_cross_pair_{args.model}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  -> {out}")

    # Print top heads with consistent sign across all 3 pairs.
    flat = []
    for L in range(n_layers):
        for H in range(n_heads):
            sgn = consistency[L, H]
            if sgn == 0:  # any zero (NaN-handling), skip
                continue
            ma = float(np.min(np.abs(Cstack[:, L, H])))
            flat.append(((L, H), sgn, ma, [float(C[L, H]) for C in Cs]))
    flat.sort(key=lambda t: -t[2])
    print(f"\n[p2f] Top {args.top_k} heads consistent across 3 pairs (sorted by min |r|):")
    print(f"     L  H   sign   minabs   height   weight    speed")
    for (L, H), sgn, ma, rs in flat[:args.top_k]:
        sgn_s = "+" if sgn > 0 else "-"
        print(f"   L{L:>2}H{H:>2}    {sgn_s}    {ma:.3f}   "
              f"{rs[0]:+.3f}  {rs[1]:+.3f}  {rs[2]:+.3f}")


if __name__ == "__main__":
    main()
