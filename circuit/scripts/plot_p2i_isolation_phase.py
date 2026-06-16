"""Phase 2I — phase-diagram transformation under context-number isolation.

Plots the same partial-correlation phase axes as Phase 2D/2E. For each
(model, pair) panel shows:
  baseline     RELATIVISTIC starting point (green circle)
  manifold α=1.0  (cyan diamond) — Phase 2E comparator
  iso_all      after full ctx-number isolation (red X)
  iso_all+tgt  after also isolating target value (gray square)

Arrows from baseline → each treatment so the trajectory is readable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
RES_DIR = REPO / "results"

sys.path.insert(0, str(REPO / "scripts"))
from plot_p2e_alpha_trajectory import setup_axes, partial_corr, safe_pearson  # noqa: E402

MODELS = ["gemma2-2b", "gemma2-9b"]
PAIRS = ["height", "weight", "speed"]


def load_iso(model, pair, k=15):
    p = RES_DIR / f"p2i_isolate_numbers_{model}_{pair}_k{k}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_alpha(model, pair, k=15):
    p = RES_DIR / f"p2e_alpha_sweep_{model}_{pair}_k{k}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def coords(r_z, r_x, r_zx):
    p_z, p_x = partial_corr(r_z, r_x, r_zx)
    return p_x, p_z


def main():
    k = 15
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(5.6 * len(PAIRS), 5.6 * len(MODELS)),
                              squeeze=False)

    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            ax = axes[mi, pi]
            setup_axes(ax)

            # r_zx from prompts (independent of model).
            rows = [json.loads(l) for l in
                    (REPO / "data" / "p2_shot_sweep" / f"{pair}_k{k}.jsonl").open()]
            r_zx = safe_pearson(np.array([r["x"] for r in rows]),
                                 np.array([r["z_eff"] for r in rows]))

            # Markers:
            iso = load_iso(model, pair, k)
            alpha = load_alpha(model, pair, k)
            if iso is None:
                ax.text(0.5, 0.5, "missing iso JSON", transform=ax.transAxes,
                        ha="center", va="center", color="red")
                continue

            # Baseline.
            br = iso["results"]["baseline"]
            base_xy = coords(br["r_ld_zeff"], br["r_ld_x"], r_zx)

            # Manifold α=1.0 from Phase 2E (if present).
            alpha_xy = None
            if alpha is not None and "manifold_a1.00" in alpha["results"]:
                ar = alpha["results"]["manifold_a1.00"]
                alpha_xy = coords(ar["r_ld_zeff"], ar["r_ld_x"], r_zx)

            iso_all_r = iso["results"]["iso_all"]
            iso_all_xy = coords(iso_all_r["r_ld_zeff"], iso_all_r["r_ld_x"], r_zx)

            iso_pt = iso["results"]["iso_all_plus_target"]
            iso_pt_xy = (None if (not np.isfinite(iso_pt["r_ld_zeff"]) or
                                    not np.isfinite(iso_pt["r_ld_x"]))
                          else coords(iso_pt["r_ld_zeff"], iso_pt["r_ld_x"], r_zx))

            # Arrows from baseline → each treatment.
            for tgt_xy, color in [(alpha_xy, "tab:cyan"),
                                    (iso_all_xy, "tab:red"),
                                    (iso_pt_xy, "tab:gray")]:
                if tgt_xy is None:
                    continue
                ax.annotate("", xy=tgt_xy, xytext=base_xy,
                            arrowprops=dict(arrowstyle="->",
                                            color=color, alpha=0.55, lw=1.4,
                                            linestyle=(0, (3, 2))),
                            zorder=3)

            # Markers + per-marker text annotations.
            def draw(xy, color, marker, size, label, txt, dx=8, dy=8,
                      alpha=1.0, lw=1.0):
                ax.scatter(*xy, s=size, color=color, marker=marker,
                            edgecolor="black", linewidth=lw, zorder=5,
                            label=label, alpha=alpha)
                ax.annotate(txt, xy, xytext=(dx, dy), textcoords="offset points",
                             fontsize=8, color=color, fontweight="bold")

            draw(base_xy, "tab:green", "o", 260, "baseline",
                  f"baseline\nr_x={br['r_ld_x']:+.2f}, r_z={br['r_ld_zeff']:+.2f}",
                  dx=-95, dy=8)
            if alpha_xy is not None:
                draw(alpha_xy, "tab:cyan", "D", 200, "manifold α=1.0",
                      f"manifold α=1\nr_x={ar['r_ld_x']:+.2f}, r_z={ar['r_ld_zeff']:+.2f}",
                      dx=8, dy=-26)
            draw(iso_all_xy, "tab:red", "X", 240, "iso_all",
                  f"iso_all\nr_x={iso_all_r['r_ld_x']:+.2f}, r_z={iso_all_r['r_ld_zeff']:+.2f}",
                  dx=8, dy=8)
            if iso_pt_xy is not None:
                draw(iso_pt_xy, "tab:gray", "s", 180, "iso_all+tgt",
                      f"iso_all+tgt\nr_x={iso_pt['r_ld_x']:+.2f}, r_z={iso_pt['r_ld_zeff']:+.2f}",
                      dx=8, dy=-26)
            else:
                ax.scatter(0.0, 0.0, s=160, color="tab:gray", marker="s",
                            edgecolor="black", linewidth=0.8, zorder=4, alpha=0.4)
                ax.annotate("iso_all+tgt\ncollapsed (std=0)", (0.0, 0.0),
                             xytext=(8, -28), textcoords="offset points",
                             fontsize=8, color="tab:gray", fontweight="bold")

            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}  "
                          f"(r(z,x)={r_zx:+.2f})", fontsize=11)
            if pi == 0:
                ax.set_ylabel(r"partial $r$(LD, z | x)  →  relativistic",
                                fontsize=10)
            if mi == len(MODELS) - 1:
                ax.set_xlabel(r"partial $r$(LD, x | z)  →  objective",
                                fontsize=10)

    fig.suptitle("Phase 2I — phase-diagram transformation under context-number isolation\n"
                 "(green ●=baseline; cyan ◆=manifold α=1.0; red ✕=ctx-numbers isolated; "
                 "gray ■=ctx+target isolated)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "p2i_isolation_phase.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
