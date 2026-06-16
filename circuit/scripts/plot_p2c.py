"""Phase 2C — visualize ablation results.

Three figures:
  p2c_r2z_trajectory.png — residual r²(z_eff) per layer, by ablation, by k.
                            (the smoking gun: where does z get rebuilt?)
  p2c_behavioral_bars.png — bar plot of r(LD, z_eff) by ablation, by k.
  p2c_ld_scatter_k1.png   — scatter of LD vs z_eff at k=1 under each ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def load(model: str, pair: str = "height"):
    p = REPO / "results" / f"p2c_ablation_{model}_{pair}.json"
    with p.open() as f:
        return json.load(f)


ABLATION_ORDER = [
    ("baseline",      "C7", "-"),
    ("primary",       "C0", "-"),
    ("primary_plus",  "C1", "-"),
    ("l1_all",        "C3", "-"),
    ("l0_all",        "C2", "-"),
    ("random_single", "C4", "--"),
    ("random_set",    "C5", "--"),
]


def plot_r2_trajectory(models, pair="height"):
    fig, axes = plt.subplots(len(models), 3, figsize=(15.5, 4.5 * len(models)),
                              squeeze=False)
    for mi, model in enumerate(models):
        data = load(model, pair)
        ks = sorted([int(k.replace("k", "")) for k in data["results"]])
        for ki, k in enumerate([1, 4, 15]):
            ax = axes[mi, ki]
            results = data["results"][f"k{k}"]
            for ab_name, color, ls in ABLATION_ORDER:
                if ab_name not in results:
                    continue
                r2 = np.asarray(results[ab_name]["r2_per_layer"])
                ax.plot(np.arange(len(r2)), r2, label=ab_name,
                        color=color, linestyle=ls, linewidth=1.6, alpha=0.9)
            ax.set_title(f"{model.replace('gemma2-', '').upper()}  |  k={k}", fontsize=11)
            ax.set_ylim(-0.05, 1.02)
            ax.set_xlim(-0.5, max(8, len(r2)*0.4))
            ax.set_xlabel("layer")
            if ki == 0:
                ax.set_ylabel("r²(z_eff)")
            ax.grid(alpha=0.3)
            if mi == 0 and ki == 2:
                ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Phase 2C — residual r²(z_eff) per layer under head ablation",
                 y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "p2c_r2z_trajectory.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def plot_behavioral_bars(models, pair="height"):
    fig, axes = plt.subplots(1, len(models), figsize=(7.0 * len(models), 4.5),
                              squeeze=False)
    ks = [1, 4, 15]
    for mi, model in enumerate(models):
        data = load(model, pair)
        ax = axes[0, mi]
        ab_names = [a[0] for a in ABLATION_ORDER if a[0] != "baseline"]
        x = np.arange(len(ab_names))
        width = 0.27
        for ki, k in enumerate(ks):
            results = data["results"][f"k{k}"]
            baseline = results["baseline"]["r_ld_zeff"]
            deltas = []
            for ab in ab_names:
                if ab in results:
                    deltas.append(results[ab]["r_ld_zeff"] - baseline)
                else:
                    deltas.append(np.nan)
            ax.bar(x + (ki - 1) * width, deltas, width, label=f"k={k}",
                   color=["C0", "C1", "C2"][ki])
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ab_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Δ r(LD, z_eff)  [ablated − baseline]")
        ax.set_title(model.replace("gemma2-", "").upper())
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Phase 2C — behavioral effect of head ablation", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "p2c_behavioral_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def plot_ld_scatter(models, pair="height", k=1):
    """Scatter LD vs z_eff at k for each ablation (subset of points)."""
    show_ablations = ["baseline", "primary_plus", "l1_all", "l0_all"]
    fig, axes = plt.subplots(len(models), len(show_ablations),
                              figsize=(2.8 * len(show_ablations), 3.0 * len(models)),
                              squeeze=False, sharex=True, sharey="row")
    for mi, model in enumerate(models):
        data = load(model, pair)
        if f"k{k}" not in data["results"]:
            continue
        results = data["results"][f"k{k}"]
        for ai, ab in enumerate(show_ablations):
            ax = axes[mi, ai]
            if ab not in results:
                ax.set_visible(False)
                continue
            ld = np.asarray(results[ab]["ld_sample"])
            z_eff = np.asarray(results[ab]["z_eff_sample"])
            ax.scatter(z_eff, ld, s=8, alpha=0.5)
            r = results[ab]["r_ld_zeff"]
            ax.set_title(f"{ab}\nr(LD, z_eff)={r:+.2f}", fontsize=9)
            if mi == len(models) - 1:
                ax.set_xlabel("z_eff")
            if ai == 0:
                ax.set_ylabel(f"{model.replace('gemma2-', '').upper()}\nLD")
            ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
            ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
    fig.suptitle(f"Phase 2C — LD vs z_eff under head ablation at k={k}", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / f"p2c_ld_scatter_k{k}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def print_summary_table(models, pair="height"):
    for model in models:
        try:
            data = load(model, pair)
        except FileNotFoundError:
            continue
        print(f"\n=== {model} | {pair} — Δ values vs baseline ===")
        for k in [1, 4, 15]:
            key = f"k{k}"
            if key not in data["results"]:
                continue
            results = data["results"][key]
            base = results["baseline"]
            print(f"\n  k={k}  baseline: r(LD,z)={base['r_ld_zeff']:+.3f}  "
                  f"L1 r²(z)={base['r2_per_layer'][1]:.2f}  "
                  f"L4 r²(z)={base['r2_per_layer'][4]:.2f}  "
                  f"⟨LD⟩={base['mean_ld']:+.2f}")
            print(f"    {'ablation':<14} {'Δr(LD,z)':>10} {'L1 r²(z)':>10} "
                   f"{'Δ⟨LD⟩':>10}")
            for ab in ["primary", "primary_plus", "l1_all", "l0_all",
                       "random_single", "random_set"]:
                if ab not in results:
                    continue
                r = results[ab]
                d_rldz = r["r_ld_zeff"] - base["r_ld_zeff"]
                d_mld = r["mean_ld"] - base["mean_ld"]
                print(f"    {ab:<14} {d_rldz:>+10.3f} {r['r2_per_layer'][1]:>10.2f} "
                       f"{d_mld:>+10.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"])
    p.add_argument("--pair", default="height")
    args = p.parse_args()

    models_present = [m for m in args.models
                      if (REPO / "results" / f"p2c_ablation_{m}_{args.pair}.json").exists()]
    if not models_present:
        print("no ablation JSONs found")
        return

    print_summary_table(models_present, args.pair)
    plot_r2_trajectory(models_present, args.pair)
    plot_behavioral_bars(models_present, args.pair)
    plot_ld_scatter(models_present, args.pair, k=1)


if __name__ == "__main__":
    main()
