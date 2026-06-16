"""Replot p2u_n_sweep_xfeat from saved JSON (no GPU)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def partial_corr_z(r_LD_z: float, r_LD_x: float, r_zx: float) -> float:
    """pc(LD, z | x) — z-correlation of LD controlling for x."""
    den = math.sqrt(max(1e-12, (1 - r_LD_x ** 2) * (1 - r_zx ** 2)))
    return (r_LD_z - r_LD_x * r_zx) / den

REPO = Path(__file__).resolve().parent.parent

FEATURES = [
    {"name": "height", "color": "C0"},
    {"name": "weight", "color": "C2"},
    {"name": "speed", "color": "C5"},
]
NEUTRAL = {"name": "neutral_text", "color": "C7"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--cos-source", choices=["resid", "sae"], default="resid",
                    help="resid: cos to d_primal in post-attn residual space (Gemma 2B-it); "
                         "sae: cos to manifold-Δ in SAE feature space (Phase 2M, base 2B)")
    ap.add_argument("--p2m-short", default="gemma2-2b",
                    help="(sae mode) p2m alignment npz short name")
    args = ap.parse_args()

    in_path = REPO / "results" / f"p2u_n_sweep_xfeat_{args.short}.json"
    d = json.loads(in_path.read_text())
    cells = d["selected_cells"]
    sweeps = d["sweeps"]
    max_n = len(cells)

    if args.cos_source == "resid":
        rc_path = REPO / "results" / f"p2u_residual_cos_{args.short}.npz"
        rc = np.load(rc_path)
        cos_grid = rc["cos_grid"]
        n_layers = cos_grid.shape[0]
        n_heads = cos_grid.shape[1]
        layer_idxs = list(range(n_layers))
        head_idxs = list(range(n_heads))
        cos_label = r"cos($\Delta h_L$, $d_L$)"
    else:
        p2m_json = json.loads((REPO / "results" /
                               f"p2m_alignment_{args.p2m_short}.json").read_text())
        p2m_npz = np.load(REPO / "results" /
                           f"p2m_alignment_{args.p2m_short}_deltas.npz")
        cos_grid = p2m_npz["cos_grid"]
        layer_idxs = list(p2m_json["layer_idxs"])
        head_idxs = list(p2m_json["head_idxs"])
        cos_label = "cos α to SAE manifold Δ (Phase 2M, base 2B)"

    cos_per_rank = np.full(max_n, np.nan, dtype=np.float32)
    for i, c in enumerate(cells):
        L, h = c["L"], c["h"]
        if L in layer_idxs and h in head_idxs:
            cos_per_rank[i] = cos_grid[layer_idxs.index(L), head_idxs.index(h)]

    visible_n = 20

    # ---------------- TOP figure: Δcorr lines + cos bars ----------------
    fig_top, ax_top = plt.subplots(figsize=(13, 6.5))
    ax_bars = ax_top.twinx()

    bar_x = np.arange(1, max_n + 1)
    valid = ~np.isnan(cos_per_rank)
    bar_colors = ["C3" if (valid[i] and cos_per_rank[i] > 0) else "C0"
                  for i in range(max_n)]
    ax_bars.bar(bar_x[valid], cos_per_rank[valid],
                 color=[bar_colors[i] for i in range(max_n) if valid[i]],
                 alpha=0.30, width=0.8, edgecolor="none", zorder=1)
    ax_bars.axhline(0, color="black", lw=0.4, alpha=0.5, zorder=2)
    ax_bars.set_ylabel(cos_label, fontsize=20)
    ax_bars.tick_params(axis="y", labelsize=18)
    cos_visible = cos_per_rank[:visible_n]
    cmax_pos = np.nanmax(cos_visible)
    cmax_neg = np.nanmin(cos_visible)
    ax_bars.set_ylim(cmax_neg - 0.05, cmax_pos + 0.13)
    for i in np.where(~valid)[0]:
        ax_bars.text(i + 1, 0.0, "n/a", ha="center", va="center",
                      fontsize=10, color="gray", rotation=90)

    for feat in FEATURES:
        sweep = sweeps[feat["name"]]
        runs = sweep["runs"]
        ns = [r["N"] for r in runs]
        baseline_r = sweep["baseline_r"]
        baseline_r_x = sweep.get("baseline_r_x")
        r_zx = sweep.get("r_zx")
        if baseline_r_x is None or r_zx is None or any(r.get("r_LD_x") is None for r in runs):
            raise SystemExit(
                f"sweep for {feat['name']} is missing r_LD_x / r_zx — "
                f"re-run p2u_n_sweep_xfeat.py to regenerate the JSON with "
                f"the partial-correlation fields"
            )
        base_pc = partial_corr_z(baseline_r, baseline_r_x, r_zx)
        d_pcs = [partial_corr_z(r["r_LD_label"], r["r_LD_x"], r_zx) - base_pc
                  for r in runs]
        ax_top.plot(ns, d_pcs, "o-", color=feat["color"], lw=2, ms=6,
                     label=f"{feat['name']} (baseline pc={base_pc:+.2f})", zorder=3)

    ax_top.axhline(0, color="black", lw=0.5)
    ax_top.set_ylabel(r"Cumulative $\Delta$pc(LD, $z \mid x$)", fontsize=22)
    ax_top.set_xlabel("N (cells resampled, cumulative)", fontsize=22)
    short_display = {
        "gemma2-2b": "Gemma2-2B", "gemma2-9b": "Gemma2-9B",
        "gemma2-2b-it": "Gemma2-2B-it", "gemma2-9b-it": "Gemma2-9B-it",
    }.get(args.short, args.short)
    ax_top.set_title(f"{short_display} — Top-N DLA Ablation", fontsize=19)
    ax_top.legend(loc="upper right", fontsize=18)
    ax_top.tick_params(axis="both", labelsize=19)
    ax_top.grid(alpha=0.3, zorder=0)
    ax_top.set_xlim(0.5, visible_n + 0.5)
    ax_top.set_ylim(-1.15, 0.05)
    ax_top.set_xticks([1, 5, 10, 15, 20])
    fig_top.tight_layout()
    out_top = REPO / "figures" / f"p2u_n_sweep_xfeat_{args.short}_dcorr.png"
    fig_top.savefig(out_top, dpi=140, bbox_inches="tight")
    print(f"wrote {out_top}")

    # ---------------- BOTTOM figure: KL lines ----------------
    fig_bot, ax_bot = plt.subplots(figsize=(13, 6.0))
    runs = sweeps[NEUTRAL["name"]]["runs"]
    ns = [r["N"] for r in runs]
    kls = [r["kl_mean"] for r in runs]
    ax_bot.plot(ns, kls, "s-", color="C7", lw=2, ms=6,
                 label=f"neutral_text (n={sweeps[NEUTRAL['name']]['n_prompts']})")
    for feat in FEATURES:
        rkl = [r["kl_mean"] for r in sweeps[feat["name"]]["runs"]]
        ax_bot.plot(ns, rkl, "--", color=feat["color"], lw=1.2, alpha=0.6,
                     label=f"{feat['name']} (relativity)")
    ax_bot.set_xlabel("N (cells resampled, cumulative)", fontsize=22)
    ax_bot.set_ylabel("mean KL [nats / prompt]", fontsize=22)
    ax_bot.set_title("Per-prompt KL: neutral text climbs slowly, relativity saturates",
                      fontsize=19)
    ax_bot.legend(loc="upper left", fontsize=19)
    ax_bot.tick_params(axis="both", labelsize=19)
    ax_bot.grid(alpha=0.3)
    rel_kl_max = max(
        max(r["kl_mean"] for r in sweeps[feat["name"]]["runs"])
        for feat in FEATURES
    )
    ax_bot.set_ylim(0, max(0.05, rel_kl_max * 1.15, max(kls) * 1.2))
    ax_bot.set_xlim(0.5, visible_n + 0.5)
    ax_bot.set_xticks([1, 5, 10, 15, 20])
    fig_bot.tight_layout()
    out_bot = REPO / "figures" / f"p2u_n_sweep_xfeat_{args.short}_kl.png"
    fig_bot.savefig(out_bot, dpi=140, bbox_inches="tight")
    print(f"wrote {out_bot}")


if __name__ == "__main__":
    main()
