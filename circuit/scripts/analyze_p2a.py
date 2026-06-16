"""Phase 2A — analyze + plot the shot-count behavioral sweep.

Reads results/p2_ld/<model>/<pair>_k<k>.npz and produces:

  results/p2_summary.json
    {model: {pair: {k: {r_z, r_zeff, r_x, mean_ld, std_ld, n,
                         r_z_lo, r_z_hi, ...  (95% bootstrap CIs)} }}}

  figures/p2a_shot_sweep.png
    Two-row figure:
      row 1: r(LD, z) vs k for each (model, pair) — the saturation curve.
      row 2: r(LD, z_eff) vs k                    — what model can see.
    (r(LD, x) curve is shown as a dashed overlay in row 1, mostly for context.)

  figures/p2a_ld_vs_z_per_k.png
    For one example (height on each model): scatter of LD vs z at each k.
    Visualizes the slope steepening as k grows.

Usage:
  python scripts/analyze_p2a.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "p2_ld"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN protection. Returns NaN if either series has zero variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs, ys = x[mask], y[mask]
    if xs.std() < 1e-12 or ys.std() < 1e-12:
        return float("nan")
    return float(pearsonr(xs, ys)[0])


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Returns (point, lo, hi) Pearson r with percentile bootstrap."""
    rng = rng if rng is not None else np.random.default_rng(0)
    point = safe_pearson(x, y)
    mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    n = len(xs)
    if n < 3 or xs.std() < 1e-12 or ys.std() < 1e-12:
        return point, float("nan"), float("nan")
    rs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[b] = safe_pearson(xs[idx], ys[idx])
    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(rs, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarize(model_dir: Path, pairs: list[str], ks: list[int]) -> dict:
    """Returns nested dict keyed by pair → k → metrics."""
    out: dict[str, dict[int, dict]] = {}
    for pair in pairs:
        out[pair] = {}
        for k in ks:
            path = model_dir / f"{pair}_k{k}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            ld = d["ld"].astype(np.float64)
            x = d["x"].astype(np.float64)
            z = d["z"].astype(np.float64)
            z_eff = d["z_eff"].astype(np.float64)

            r_z, r_z_lo, r_z_hi = bootstrap_ci(z, ld)
            r_zeff, r_zeff_lo, r_zeff_hi = bootstrap_ci(z_eff, ld)
            r_x, r_x_lo, r_x_hi = bootstrap_ci(x, ld)
            out[pair][k] = {
                "n": int(len(ld)),
                "r_z": r_z, "r_z_lo": r_z_lo, "r_z_hi": r_z_hi,
                "r_zeff": r_zeff, "r_zeff_lo": r_zeff_lo, "r_zeff_hi": r_zeff_hi,
                "r_x": r_x, "r_x_lo": r_x_lo, "r_x_hi": r_x_hi,
                "mean_ld": float(ld.mean()),
                "std_ld": float(ld.std(ddof=1)) if len(ld) > 1 else float("nan"),
            }
    return out


def plot_saturation(summary: dict, out_path: Path) -> None:
    """Saturation curves: r(LD, z), r(LD, z_eff), r(LD, x) vs k for each (model, pair)."""
    models = sorted(summary.keys())
    pairs = sorted({p for m in summary.values() for p in m.keys()})
    fig, axes = plt.subplots(len(models), len(pairs), figsize=(4.2 * len(pairs), 3.8 * len(models)),
                              sharey=True, squeeze=False)
    for mi, model in enumerate(models):
        for pi, pair in enumerate(pairs):
            ax = axes[mi, pi]
            cells = summary[model].get(pair, {})
            ks = sorted(cells.keys(), key=int)
            if not ks:
                ax.set_visible(False)
                continue
            r_z = np.array([cells[k]["r_z"] for k in ks])
            r_z_lo = np.array([cells[k]["r_z_lo"] for k in ks])
            r_z_hi = np.array([cells[k]["r_z_hi"] for k in ks])
            r_zeff = np.array([cells[k]["r_zeff"] for k in ks])
            r_zeff_lo = np.array([cells[k]["r_zeff_lo"] for k in ks])
            r_zeff_hi = np.array([cells[k]["r_zeff_hi"] for k in ks])
            r_x = np.array([cells[k]["r_x"] for k in ks])
            ks_arr = np.array(ks, dtype=float)

            ax.fill_between(ks_arr, r_z_lo, r_z_hi, alpha=0.15, color="C0")
            ax.plot(ks_arr, r_z, "o-", color="C0", label="r(LD, z) [intended]")
            ax.fill_between(ks_arr, r_zeff_lo, r_zeff_hi, alpha=0.15, color="C1")
            ax.plot(ks_arr, r_zeff, "s-", color="C1", label="r(LD, z_eff) [model-visible]")
            ax.plot(ks_arr, r_x, "x--", color="0.4", label="r(LD, x)")
            ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xticks(ks_arr)
            ax.set_xticklabels([str(int(k)) for k in ks_arr])
            ax.set_xlabel("shot count k")
            ax.set_ylim(-0.3, 1.0)
            ax.set_title(f"{model} | {pair}")
            if mi == 0 and pi == len(pairs) - 1:
                ax.legend(loc="lower right", fontsize=8)
    axes[0, 0].set_ylabel("Pearson r")
    if len(models) > 1:
        axes[1, 0].set_ylabel("Pearson r")
    fig.suptitle("Phase 2A — context-saturation of relativity (r vs shot count)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> {out_path}")


def plot_ld_scatter(model_dir: Path, model: str, pair: str, ks: list[int],
                    out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(ks), figsize=(2.8 * len(ks), 3.0),
                             sharey=True, squeeze=False)
    for ki, k in enumerate(ks):
        ax = axes[0, ki]
        path = model_dir / f"{pair}_k{k}.npz"
        if not path.exists():
            ax.set_visible(False)
            continue
        d = np.load(path, allow_pickle=True)
        ld = d["ld"].astype(np.float64)
        if k == 0:
            x = d["x"].astype(np.float64)
            ax.scatter(x, ld, s=10, alpha=0.7)
            ax.set_xlabel("x")
            ax.set_title(f"k={k}\nr(LD, x) over x grid")
        else:
            z = d["z_eff"].astype(np.float64)
            ax.scatter(z, ld, s=4, alpha=0.4)
            r = safe_pearson(z, ld)
            ax.set_xlabel("z_eff")
            ax.set_title(f"k={k}\nr(LD, z_eff) = {r:+.2f}")
        if ki == 0:
            ax.set_ylabel("LD = logit(high) − logit(low)")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    fig.suptitle(f"{model} — {pair}: LD distribution by k", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"])
    p.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"])
    p.add_argument("--k", nargs="+", type=int, default=[0, 1, 2, 4, 8, 15])
    args = p.parse_args()

    summary: dict[str, dict] = {}
    for model in args.models:
        model_dir = RESULTS / model
        if not model_dir.exists():
            print(f"[skip] {model_dir} missing")
            continue
        s = summarize(model_dir, args.pairs, args.k)
        summary[model] = s

    # --- text headline table ---
    print("\n=== Phase 2A — saturation table (Pearson r, 95% bootstrap CI in []) ===")
    for model, model_data in summary.items():
        print(f"\n{model}")
        for pair, cells in model_data.items():
            print(f"  {pair}")
            print(f"    {'k':>3} {'n':>5}  {'r(LD,z)':>16}  {'r(LD,z_eff)':>16}  {'r(LD,x)':>16}  {'⟨LD⟩':>7}")
            for k in sorted(cells.keys(), key=int):
                c = cells[k]
                rz = f"{c['r_z']:+.3f} [{c['r_z_lo']:+.2f},{c['r_z_hi']:+.2f}]"
                rze = f"{c['r_zeff']:+.3f} [{c['r_zeff_lo']:+.2f},{c['r_zeff_hi']:+.2f}]"
                rx = f"{c['r_x']:+.3f} [{c['r_x_lo']:+.2f},{c['r_x_hi']:+.2f}]"
                print(f"    {k:>3} {c['n']:>5}  {rz:>16}  {rze:>16}  {rx:>16}  {c['mean_ld']:+7.2f}")

    out_json = REPO / "results" / "p2a_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> {out_json}")

    # --- figures ---
    plot_saturation(summary, FIG_DIR / "p2a_shot_sweep.png")
    for model in args.models:
        model_dir = RESULTS / model
        if model_dir.exists():
            plot_ld_scatter(model_dir, model, "height", args.k,
                            FIG_DIR / f"p2a_ld_vs_z_height_{model}.png")


if __name__ == "__main__":
    main()
