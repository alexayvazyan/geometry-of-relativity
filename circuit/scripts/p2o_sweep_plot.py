"""Phase 2O — replot N-sweep with cosine of cell-added shown as background bars.

Reads results/p2o_n_sweep_<short>.json (or p2o_n_sweep_<short>_<suffix>.json)
and plots Δr(N) line + |cos|(rank N) translucent bars (color-coded by sign).

Usage:
    python p2o_sweep_plot.py --short gemma2-2b
    python p2o_sweep_plot.py --short gemma2-2b --suffix poscos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--suffix", default=None,
                    help="optional suffix for the JSON, e.g. 'poscos'")
    ap.add_argument("--metric", default="auto",
                    choices=["auto", "cos", "cos_sigma"],
                    help="bar quantity (auto = follow score_source in JSON)")
    ap.add_argument("--out-suffix", default="bars",
                    help="suffix on output PNG filename")
    args = ap.parse_args()

    base = f"p2o_n_sweep_{args.short}"
    if args.suffix:
        base += f"_{args.suffix}"
    in_json = REPO / "results" / f"{base}.json"
    d = json.loads(in_json.read_text())

    selected = d["selected_runs"]
    cells = d["selected_cells"]
    Ns = [s["N"] for s in selected]
    drs = [s["delta_r"] for s in selected]
    kls = [s["kl_mean"] for s in selected]

    metric = args.metric
    if metric == "auto":
        metric = d.get("score_source", "cos")
    # cos of the cell added at each N (always available)
    cos_at_rank = np.array([cells[N - 1]["cos"] for N in Ns])
    if metric == "cos_sigma":
        if "cos_sigma" not in cells[0]:
            raise SystemExit("cos_sigma not in JSON; rerun n_sweep with "
                              "--score-source cos_sigma")
        bar_at_rank = np.array([cells[N - 1]["cos_sigma"] for N in Ns])
        bar_label = "cos·σ of cell added at rank N\n(red = +z, blue = −z)"
        bar_lim = max(0.005, float(np.abs(bar_at_rank).max()) * 1.15)
    else:
        bar_at_rank = cos_at_rank
        bar_label = "cos of cell added at rank N\n(red = +z, blue = −z)"
        bar_lim = 0.5
    base_r = d["baseline_r_LD_z"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: Δr line + cos bars on twin axis
    ax = axes[0]
    ax_b = ax.twinx()
    bar_colors = ["tab:red" if c > 0 else "tab:blue" for c in cos_at_rank]
    ax_b.bar(Ns, bar_at_rank, color=bar_colors, alpha=0.25, width=0.85,
              edgecolor="none", zorder=1)
    ax_b.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax_b.set_ylabel(bar_label, fontsize=11)
    ax_b.set_ylim(-bar_lim, bar_lim)

    ax.plot(Ns, drs, "-o", color="tab:purple", lw=2, ms=8, zorder=5,
             label="Δcorr(LD, z)")
    ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.5)
    ax.set_ylabel(f"Δcorr(LD, z)  (baseline corr={base_r:+.3f})", fontsize=11)
    ax.set_zorder(ax_b.get_zorder() + 1)
    ax.patch.set_visible(False)

    title = (f"{d['short']} {d['feature']} k={d['k']} — "
              f"resample N-sweep with cell-added cosine\n")
    if args.suffix:
        title += f"(suffix: {args.suffix})"
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # Annotate notable jumps
    for i in range(1, len(drs)):
        delta_step = drs[i] - drs[i - 1]
        if abs(delta_step) > 0.10:
            cell = cells[Ns[i] - 1]
            if metric == "cos_sigma":
                tag = f"cos·σ={cell.get('cos_sigma', 0):+.3f}"
            else:
                tag = f"cos={cell['cos']:+.2f}"
            txt = (f"+L{cell['layer']}H{cell['head']}\n({tag})")
            ax.annotate(txt, (Ns[i], drs[i]),
                         xytext=(0, -28 if delta_step < 0 else 18),
                         textcoords="offset points",
                         fontsize=8.5, ha="center", color="black",
                         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Bottom panel: KL
    ax = axes[1]
    ax.plot(Ns, kls, "-s", color="tab:green", lw=2, ms=6, zorder=5)
    ax.set_ylabel("mean KL\n(baseline||resample) [nats]", fontsize=10)
    ax.set_xlabel("N cells resampled", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xticks(Ns[::2] if len(Ns) > 16 else Ns)

    fig.tight_layout()
    out_png = REPO / "figures" / f"{base}_{args.out_suffix}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
