"""Compare cos·σ N-sweep vs σ_DLA·ρ N-sweep on 2B."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# cos·σ JSON wasn't saved; reconstructed from log
cos_drs_log = {1: -0.182, 2: -0.434, 3: -0.488, 4: -0.531, 5: -0.529,
                6: -0.530, 7: -0.539, 8: -0.637, 9: -0.640, 10: -0.640,
                11: -0.642, 12: -0.642, 13: -0.642, 14: -0.642, 15: -0.659,
                16: -0.828, 17: -0.825, 18: -0.825, 19: -0.825, 20: -0.825,
                21: -0.825, 22: -0.825, 23: -0.825, 24: -0.834, 25: -0.834,
                26: -0.834, 27: -0.834, 28: -0.834, 29: -0.855, 30: -0.854,
                31: -0.853, 32: -0.854}
abscos_sweep = json.load(open(REPO/"results/p2o_n_sweep_gemma2-2b_positive.json"))
dla_sweep = json.load(open(REPO/"results/p2o_n_sweep_gemma2-2b_positive_dla.json"))

cos_sweep = {"baseline_r_LD_z": dla_sweep["baseline_r_LD_z"],
              "selected_runs": [{"N": N, "delta_r": cos_drs_log[N], "kl_mean": 0}
                                for N in range(1, 33)],
              "selected_cells": []}
abscos_drs = [r["delta_r"] for r in abscos_sweep["selected_runs"]]
abscos_Ns = [r["N"] for r in abscos_sweep["selected_runs"]]

cos_Ns = [r["N"] for r in cos_sweep["selected_runs"]]
cos_drs = [r["delta_r"] for r in cos_sweep["selected_runs"]]
dla_Ns = [r["N"] for r in dla_sweep["selected_runs"]]
dla_drs = [r["delta_r"] for r in dla_sweep["selected_runs"]]
dla_kls = [r["kl_mean"] for r in dla_sweep["selected_runs"]]
dla_cells = dla_sweep["selected_cells"]

base_r = cos_sweep["baseline_r_LD_z"]

fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax_b = ax.twinx()

bar_at_rank_dla = np.array([dla_cells[N - 1].get("cos_sigma", 0) for N in dla_Ns])
ax_b.bar(np.array(dla_Ns), bar_at_rank_dla, color="tab:purple", alpha=0.18,
          width=0.85, edgecolor="none", zorder=1, label="σ_DLA·ρ at rank N")
ax_b.set_ylabel("σ_DLA·ρ of cell added at rank N (purple bars)", fontsize=10,
                 color="tab:purple")
ax_b.tick_params(axis="y", labelcolor="tab:purple")
ax_b.set_ylim(0, max(0.005, float(np.abs(bar_at_rank_dla).max()) * 1.15))

ax.plot(abscos_Ns, abscos_drs, "-^", color="tab:olive", lw=1.5, ms=5,
         alpha=0.7, zorder=4, label="|cos| ranking")
ax.plot(cos_Ns, cos_drs, "-s", color="tab:gray", lw=2, ms=6, zorder=5,
         label="cos·σ ranking")
ax.plot(dla_Ns, dla_drs, "-o", color="tab:purple", lw=2.5, ms=9, zorder=6,
         label="σ_DLA·ρ ranking")
ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.5)
ax.axhline(-base_r, color="red", lw=0.6, ls=":", alpha=0.7,
            label=f"r → 0 (decorrelation, Δr=−{base_r:.2f})")

for i in range(1, len(dla_drs)):
    delta_step = dla_drs[i] - dla_drs[i - 1]
    if abs(delta_step) > 0.10:
        cell = dla_cells[dla_Ns[i] - 1]
        txt = (f"+L{cell['layer']}H{cell['head']}\n"
                f"σ_DLA·ρ={cell.get('cos_sigma', 0):+.3f}")
        ax.annotate(txt, (dla_Ns[i], dla_drs[i]),
                     xytext=(0, -22 if delta_step < 0 else 18),
                     textcoords="offset points",
                     fontsize=8, ha="center", color="tab:purple",
                     arrowprops=dict(arrowstyle="-", color="purple", lw=0.5))

ax.set_ylabel(f"Δr(LD, z)  (baseline r={base_r:+.3f})", fontsize=11)
ax.set_zorder(ax_b.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_title("gemma2-2b height k=15 — N-sweep ranking comparison\n"
              "cos·σ (gray squares) vs σ_DLA·ρ (purple circles, plus σ_DLA·ρ-bars)",
              fontsize=12)
ax.legend(loc="lower left", fontsize=10)
ax.grid(alpha=0.25, axis="y")

ax = axes[1]
abscos_kls = [r["kl_mean"] for r in abscos_sweep["selected_runs"]]
ax.plot(abscos_Ns, abscos_kls, "-^", color="tab:olive", lw=1.5, ms=4, alpha=0.7,
         label="|cos|")
ax.plot(dla_Ns, dla_kls, "-o", color="tab:purple", lw=2, ms=8,
         label="σ_DLA·ρ")
ax.set_ylabel("mean KL\n(baseline||resample) [nats]", fontsize=10)
ax.set_xlabel("N cells resampled", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)

ax_top = axes[0]
crossover_n = None
for ni, dr in zip(dla_Ns, dla_drs):
    cos_dr = cos_drs[ni - 1]
    if dr < cos_dr - 0.05:
        crossover_n = ni
        break
if crossover_n:
    ax_top.annotate(f"DLA pulls ahead at N={crossover_n}", (crossover_n, dla_drs[crossover_n - 1]),
                     xytext=(crossover_n + 2, dla_drs[crossover_n - 1] - 0.05),
                     fontsize=10, color="tab:purple",
                     arrowprops=dict(arrowstyle="->", color="purple"))

ax_top.text(32, cos_drs[-1] + 0.02,
             f"cos·σ N=32 ceiling: Δr={cos_drs[-1]:+.3f}",
             fontsize=9, color="tab:gray", ha="right")
ax_top.text(8, dla_drs[7] - 0.02,
             f"σ_DLA·ρ N=8 already past cos·σ ceiling: Δr={dla_drs[7]:+.3f}",
             fontsize=9, color="tab:purple", ha="left", va="top")

fig.tight_layout()
out_png = REPO / "figures" / "p2s_dla_vs_cossigma_sweep_gemma2-2b.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"wrote {out_png}")
