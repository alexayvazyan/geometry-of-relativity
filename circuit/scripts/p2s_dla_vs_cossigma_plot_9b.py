"""Compare cos(Δh_L, d_primal) vs σ_DLA·ρ rankings on 9B.

Reads two n-sweep JSONs (both over the full 42×16 grid) and overlays Δr(N)
descent curves with σ_DLA·ρ bars on the right axis. The cos ranking here
is residual-space `cos(head writeout, d_primal)`, computed over all layers
— NOT the L1-SAE-restricted Phase 2M cos which only spans L0–L21 on 9B.

Inputs:
  results/p2o_n_sweep_gemma2-9b_positive_residcos.json   (cos ranking, full grid)
  results/p2o_n_sweep_gemma2-9b_positive_dla.json        (σ_DLA·ρ ranking, full grid)

Output:
  figures/p2s_dla_vs_cossigma_sweep_gemma2-9b.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

abscos_sweep = json.load(open(REPO / "results"
                                 / "p2o_n_sweep_gemma2-9b_positive_residcos.json"))
dla_sweep = json.load(open(REPO / "results"
                              / "p2o_n_sweep_gemma2-9b_positive_dla.json"))

abscos_Ns = [r["N"] for r in abscos_sweep["selected_runs"]]
abscos_drs = [r["delta_r"] for r in abscos_sweep["selected_runs"]]
abscos_kls = [r["kl_mean"] for r in abscos_sweep["selected_runs"]]

dla_Ns = [r["N"] for r in dla_sweep["selected_runs"]]
dla_drs = [r["delta_r"] for r in dla_sweep["selected_runs"]]
dla_kls = [r["kl_mean"] for r in dla_sweep["selected_runs"]]
dla_cells = dla_sweep["selected_cells"]
base_r = dla_sweep["baseline_r_LD_z"]

fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

# Top: Δr lines + σ_DLA·ρ bars on right axis
ax = axes[0]
ax_b = ax.twinx()

bar_at_rank = np.array([dla_cells[N - 1].get("cos_sigma", 0) for N in dla_Ns])
ax_b.bar(np.array(dla_Ns), bar_at_rank, color="tab:purple", alpha=0.18,
          width=0.85, edgecolor="none", zorder=1, label="σ_DLA·ρ at rank N")
ax_b.set_ylabel("σ_DLA·ρ of cell added at rank N (purple bars)",
                 fontsize=11, color="tab:purple")
ax_b.tick_params(axis="y", labelcolor="tab:purple", labelsize=11)
ax_b.set_ylim(0, max(0.005, float(np.abs(bar_at_rank).max()) * 1.15))

ax.plot(abscos_Ns, abscos_drs, "-^", color="tab:olive", lw=2, ms=6,
         alpha=0.85, zorder=5, label=r"cos($\Delta h_L$, $d_L$) ranking")
ax.plot(dla_Ns, dla_drs, "-o", color="tab:purple", lw=2.5, ms=9, zorder=6,
         label=r"$\sigma_{DLA}\cdot\rho$ ranking")
ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.5)
ax.axhline(-base_r, color="red", lw=0.6, ls=":", alpha=0.7,
            label=f"r → 0 (decorrelation, Δr=−{base_r:.2f})")

# Annotate large step-changes in the DLA curve with the entering head
for i in range(1, len(dla_drs)):
    delta_step = dla_drs[i] - dla_drs[i - 1]
    if abs(delta_step) > 0.10:
        cell = dla_cells[dla_Ns[i] - 1]
        txt = (f"+L{cell['layer']}H{cell['head']}\n"
                f"σ_DLA·ρ={cell.get('cos_sigma', 0):+.3f}")
        ax.annotate(txt, (dla_Ns[i], dla_drs[i]),
                     xytext=(0, -22 if delta_step < 0 else 18),
                     textcoords="offset points",
                     fontsize=9, ha="center", color="tab:purple",
                     arrowprops=dict(arrowstyle="-", color="purple", lw=0.5))

ax.set_ylabel(f"Δr(LD, z)  (baseline r={base_r:+.3f})", fontsize=12)
ax.set_zorder(ax_b.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_title(r"Gemma 2 9B height $k{=}15$ — N-sweep ranking comparison"
              "\n"
              r"cos($\Delta h_L$, $d_L$) (olive) vs $\sigma_{DLA}\cdot\rho$ (purple)",
              fontsize=13)
ax.legend(loc="lower left", fontsize=11)
ax.grid(alpha=0.25, axis="y")

# Crossover annotation: where σ_DLA·ρ pulls ahead of |cos|
crossover_n = None
for ni, dr in zip(dla_Ns, dla_drs):
    abscos_dr = abscos_drs[ni - 1]
    if dr < abscos_dr - 0.05:
        crossover_n = ni
        break
if crossover_n:
    ax.annotate(f"DLA pulls ahead at N={crossover_n}",
                 (crossover_n, dla_drs[crossover_n - 1]),
                 xytext=(crossover_n + 2, dla_drs[crossover_n - 1] - 0.05),
                 fontsize=11, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="purple"))

ax.text(32, abscos_drs[-1] + 0.02,
         f"cos N=32: Δr={abscos_drs[-1]:+.3f}",
         fontsize=10, color="tab:olive", ha="right")

# Bottom: KL
ax = axes[1]
ax.plot(abscos_Ns, abscos_kls, "-^", color="tab:olive", lw=1.5, ms=4,
         alpha=0.75, label=r"cos($\Delta h_L$, $d_L$)")
ax.plot(dla_Ns, dla_kls, "-o", color="tab:purple", lw=2, ms=8,
         label=r"$\sigma_{DLA}\cdot\rho$")
ax.set_ylabel("mean KL\n(baseline||resample) [nats]", fontsize=11)
ax.set_xlabel("N cells resampled", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.25)

fig.tight_layout()
out_png = REPO / "figures" / "p2s_dla_vs_cossigma_sweep_gemma2-9b.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"wrote {out_png}")
