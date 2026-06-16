"""Phase 2Q analysis — three-panel comparison of cos / σ / cos·σ on 2B.

Reads p2q_head_zinfo_gemma2-2b.json (sigma + R²) and p2m_alignment_gemma2-2b.json
(Phase 2M cos), produces a heatmap triplet to motivate cos·σ as the predictor
of resample Δr.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b",
                    choices=["gemma2-2b", "gemma2-9b"])
    args = ap.parse_args()
    SHORT = args.short

    zinfo = json.loads((REPO/"results"/f"p2q_head_zinfo_{SHORT}.json").read_text())
    p2m_path = REPO/"results"/f"p2m_alignment_{SHORT}_fullgrid.json"
    if not p2m_path.exists():
        p2m_path = REPO/"results"/f"p2m_alignment_{SHORT}.json"
    p2m = json.loads(p2m_path.read_text())

    r2 = np.array(zinfo["r2_grid"])
    sigma = np.array(zinfo["sigma_grid"])
    cos = np.array(p2m["cos_grid"])
    Ls = p2m["layer_idxs"]
    Hs = p2m["head_idxs"]

    n_layers, n_heads = r2.shape
    cos_full = np.full_like(r2, np.nan)
    for li, L in enumerate(Ls):
        for hi, h in enumerate(Hs):
            cos_full[L, h] = cos[li, hi]
    sig_cos = sigma * cos_full

    if SHORT == "gemma2-2b":
        known = [
            (16, 4, -0.182, "L16H4"),
            (17, 7, -0.036, "L17H7"),
            (14, 2, -0.004, "L14H2"),
            (14, 7, None, "L14H7"),
            (15, 7, None, "L15H7"),
            (12, 6, None, "L12H6 *"),
            (12, 4, None, "L12H4 *"),
            (13, 2, None, "L13H2 *"),
            (11, 2, None, "L11H2 *"),
        ]
    else:  # 9b — top cos·σ candidates, no Δr known
        known = [
            (21, 7, None, "L21H7 *"),
            (20, 9, None, "L20H9 *"),
            (18, 3, None, "L18H3 *"),
            (19, 1, None, "L19H1 *"),
            (15, 9, None, "L15H9 *"),
            (21, 13, None, "L21H13 *"),
            (17, 6, None, "L17H6 *"),
            (16, 14, None, "L16H14 *"),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 0.4 * n_layers + 1.5))

    # cos
    ax = axes[0]
    vmax = max(0.05, float(np.nanmax(np.abs(cos_full))))
    im = ax.imshow(cos_full, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Phase 2M cos(Δ_ablate, Δ_manifold)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cos")

    # sigma
    ax = axes[1]
    im = ax.imshow(sigma, aspect="auto", cmap="viridis",
                    vmin=0, vmax=float(np.nanmax(sigma)))
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_title(f"σ — last-token output spread across prompts\n(mean per-dim std)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="σ")

    # cos · σ (signed)
    ax = axes[2]
    vmax = max(0.005, float(np.nanmax(np.abs(sig_cos))))
    im = ax.imshow(sig_cos, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_title(f"cos · σ\n(predicts resample Δr)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cos · σ")
    for L, h, dr, lbl in known:
        edge = "red" if dr is None else ("yellow" if dr < -0.1 else "white")
        ax.add_patch(plt.Rectangle((h-0.5, L-0.5), 1, 1, fill=False,
                                    edgecolor=edge, lw=1.5))
        if dr is not None:
            ax.text(h, L+0.4, f"Δr={dr:+.2f}", ha="center", fontsize=6,
                     color="black")
        else:
            ax.text(h, L+0.4, "?", ha="center", fontsize=8, color="black",
                     fontweight="bold")

    fig.suptitle(f"{SHORT} — Phase 2M cos · Phase 2Q σ → cos·σ "
                  f"(red box = known causal trio, yellow box = strong, "
                  f"white box = candidate untested)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = REPO / "figures" / f"p2q_cossigma_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # Print a compact ranking table for downstream Phase 2Q-causal test
    flat = []
    for L in range(n_layers):
        for h in range(n_heads):
            if not np.isnan(cos_full[L, h]):
                flat.append((sig_cos[L, h], cos_full[L, h], sigma[L, h],
                              L, h))
    flat.sort(key=lambda r: -r[0])
    print("\n=== top 20 cells by cos·σ (positive only) ===")
    for sc, c, sg, L, h in flat[:20]:
        if sc > 0:
            print(f"  L{L:2d}H{h:1d}: cos={c:+.3f} σ={sg:.3f} cos·σ={sc:+.4f}")


if __name__ == "__main__":
    main()
