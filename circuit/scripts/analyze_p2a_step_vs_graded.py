"""Phase 2A — quantify the comparator/graded transition.

At k=1 the LD-vs-z_eff scatter looks like a step function. By k=4-15 it's
linear. Quantify this with three diagnostics:

  1. Slope of LD vs z_eff (OLS).
  2. Bimodality coefficient: kurtosis-based; b = (skew^2 + 1) / kurtosis.
     b > 5/9 ≈ 0.555 indicates a bimodal distribution.
  3. Slope of |LD| vs |z_eff| (gradedness): if model is doing a step, |LD|
     is roughly constant in |z_eff|; if graded, |LD| grows linearly.

Output:
  results/p2a_step_vs_graded.json
  figures/p2a_step_vs_graded.png   — slope and gradedness vs k
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "p2_ld"
FIG_DIR = REPO / "figures"


def linear_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope, intercept of y on x."""
    if x.std() < 1e-12:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def bimodality_coef(y: np.ndarray) -> float:
    """Sarle's bimodality coefficient. >5/9 ≈ 0.555 → likely bimodal."""
    if len(y) < 4:
        return float("nan")
    s = skew(y, bias=False)
    k_excess = kurtosis(y, fisher=True, bias=False)
    n = len(y)
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (s * s + 1.0) / (k_excess + correction)


def main():
    pairs = ["height", "weight", "speed"]
    models = ["gemma2-2b", "gemma2-9b"]
    ks = [1, 2, 4, 8, 15]

    summary: dict = {}
    for model in models:
        model_dir = RESULTS / model
        if not model_dir.exists():
            continue
        summary[model] = {}
        for pair in pairs:
            cells = {}
            for k in ks:
                path = model_dir / f"{pair}_k{k}.npz"
                if not path.exists():
                    continue
                d = np.load(path, allow_pickle=True)
                ld = d["ld"].astype(np.float64)
                z_eff = d["z_eff"].astype(np.float64)
                # filter out non-finite z_eff (k=0 only)
                mask = np.isfinite(z_eff) & np.isfinite(ld)
                z_eff = z_eff[mask]; ld = ld[mask]
                slope_z, intercept = linear_slope(z_eff, ld)
                slope_abs, _ = linear_slope(np.abs(z_eff), np.abs(ld))
                bm = bimodality_coef(ld)
                # Sigmoid-vs-line fit: residual after fitting tanh(beta*z) vs after fitting line
                from scipy.optimize import curve_fit
                try:
                    popt, _ = curve_fit(lambda z, A, beta: A * np.tanh(beta * z),
                                        z_eff, ld, p0=[2.0, 1.0], maxfev=2000)
                    A_tanh, beta_tanh = float(popt[0]), float(popt[1])
                    pred_tanh = A_tanh * np.tanh(beta_tanh * z_eff)
                    rss_tanh = float(((ld - pred_tanh) ** 2).sum())
                except Exception:
                    A_tanh, beta_tanh, rss_tanh = float("nan"), float("nan"), float("nan")
                pred_lin = slope_z * z_eff + intercept
                rss_lin = float(((ld - pred_lin) ** 2).sum())
                # Lower RSS = better fit. Ratio < 1 means tanh fits better than linear.
                rss_ratio = rss_tanh / max(rss_lin, 1e-12)
                cells[k] = {
                    "n": int(len(ld)),
                    "slope_z": slope_z,
                    "intercept": intercept,
                    "slope_abs": slope_abs,
                    "bimodality": bm,
                    "tanh_amp": A_tanh,
                    "tanh_beta": beta_tanh,
                    "rss_tanh": rss_tanh,
                    "rss_lin": rss_lin,
                    "rss_ratio_tanh_over_lin": rss_ratio,
                }
            summary[model][pair] = cells

    # --- print ---
    print("\n=== Phase 2A — comparator vs graded diagnostics ===")
    print("(slope_z = OLS slope LD ~ z_eff; tanh_beta high = sharper threshold;\n"
          " bimodality > 0.555 → bimodal LD distribution;\n"
          " rss_ratio < 1 → tanh fits better than line, > 1 → linear better.)")
    for model, pairs_d in summary.items():
        print(f"\n{model}")
        for pair, cells in pairs_d.items():
            print(f"  {pair}")
            print(f"    {'k':>3} {'slope_z':>9} {'tanh_β':>9} {'tanh_amp':>10} {'rss_ratio':>10} {'bimod':>7}")
            for k in sorted(cells.keys(), key=int):
                c = cells[k]
                print(f"    {k:>3} {c['slope_z']:>+9.3f} {c['tanh_beta']:>+9.3f} "
                      f"{c['tanh_amp']:>+10.3f} {c['rss_ratio_tanh_over_lin']:>10.3f} "
                      f"{c['bimodality']:>7.3f}")

    out_json = REPO / "results" / "p2a_step_vs_graded.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> {out_json}")

    # --- figure: slope_z and tanh_beta vs k ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for pi, pair in enumerate(pairs):
        ax_top = axes[0, pi]
        ax_bot = axes[1, pi]
        for model in models:
            cells = summary.get(model, {}).get(pair, {})
            ks_sorted = sorted(cells.keys(), key=int)
            slopes = [cells[k]["slope_z"] for k in ks_sorted]
            betas = [cells[k]["tanh_beta"] for k in ks_sorted]
            ratios = [cells[k]["rss_ratio_tanh_over_lin"] for k in ks_sorted]
            label_short = "9B" if "9b" in model else "2B"
            ax_top.plot(ks_sorted, slopes, "o-", label=label_short)
            ax_bot.plot(ks_sorted, ratios, "s-", label=label_short)
        ax_top.set_title(f"{pair}: linear slope LD~z_eff")
        ax_top.set_ylabel("slope")
        ax_top.set_xscale("symlog", linthresh=1.0)
        ax_top.set_xticks([1, 2, 4, 8, 15])
        ax_top.set_xticklabels(["1", "2", "4", "8", "15"])
        ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax_top.legend(fontsize=8)
        ax_bot.set_title(f"{pair}: RSS(tanh) / RSS(lin)")
        ax_bot.set_ylabel("ratio (<1: step fits better)")
        ax_bot.set_xlabel("shot count k")
        ax_bot.set_xscale("symlog", linthresh=1.0)
        ax_bot.set_xticks([1, 2, 4, 8, 15])
        ax_bot.set_xticklabels(["1", "2", "4", "8", "15"])
        ax_bot.axhline(1.0, color="black", linewidth=0.4, alpha=0.3)
    fig.suptitle("Phase 2A — comparator-to-graded transition", y=1.0)
    fig.tight_layout()
    out_fig = FIG_DIR / "p2a_step_vs_graded.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"  -> {out_fig}")


if __name__ == "__main__":
    main()
