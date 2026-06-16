"""Phase 2D — phase-space grid across shot counts.

Layout: 2 rows × 5 columns.
  Rows = models (2B, 9B).
  Cols = k ∈ {0, 1, 2, 5, 15}.
  Each panel: r(LD, x) [horizontal] vs r(LD, z_eff) [vertical], with the
              three behavioral states shaded (BIASED / OBJECTIVE / RELATIVISTIC).
  Markers per panel: baseline (green) + l0_all (red).

  At k=0, z_eff is undefined; baseline is plotted at y=0 with a "z undefined"
  annotation. Shows the model's objective vs biased anchor.

  Annotation: each marker labelled with ⟨LD⟩ and std(LD), so the difference
  between (low r, high std = lots of variance, low SNR) and (low r, low std =
  collapsed/biased) is visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def load_baseline(model, k):
    """Read baseline LD from p2_ld extraction NPZ."""
    path = REPO / "results" / "p2_ld" / model / f"height_k{k}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    ld = d["ld"].astype(np.float64)
    x = d["x"].astype(np.float64)
    z = d["z_eff"].astype(np.float64)
    mask = np.isfinite(z)
    r_x = pearsonr(x, ld)[0]
    r_z = pearsonr(z[mask], ld[mask])[0] if mask.sum() >= 3 and z[mask].std() > 1e-9 else float("nan")
    return {
        "r_ld_x":   float(r_x),
        "r_ld_zeff": float(r_z),
        "mean_ld":  float(ld.mean()),
        "std_ld":   float(ld.std(ddof=1)) if len(ld) > 1 else 0.0,
        "n":        int(len(ld)),
    }


def load_residual_interventions(model, k=15):
    """Returns dict of {mode: {r_ld_x, r_ld_zeff, mean_ld, std_ld}} for k=15
    interventions (proj_out / mean_ablate / manifold α). Empty if not available."""
    p = REPO / "results" / f"p2e_residual_interventions_{model}_height_k{k}.json"
    if not p.exists():
        return {}
    with p.open() as f:
        D = json.load(f)
    return D["results"]


def load_l0all(model, k):
    """l0_all results: prefer the dedicated p2d sweep, fall back to p2c."""
    p_d = REPO / "results" / f"p2d_l0all_per_k_{model}_height.json"
    p_c = REPO / "results" / f"p2c_ablation_{model}_height.json"
    p_p2d = REPO / "results" / f"p2d_partial_l0_{model}_height_k{k}.json"

    if p_d.exists():
        with p_d.open() as f:
            D = json.load(f)
        if f"k{k}" in D["results"]:
            return D["results"][f"k{k}"]["l0_all"]

    if p_p2d.exists():
        with p_p2d.open() as f:
            D = json.load(f)
        return D["l0_all"]

    if p_c.exists():
        with p_c.open() as f:
            D = json.load(f)
        if f"k{k}" in D["results"] and "l0_all" in D["results"][f"k{k}"]:
            r = D["results"][f"k{k}"]["l0_all"]
            return {
                "r_ld_x":    r.get("r_ld_x", float("nan")),
                "r_ld_zeff": r["r_ld_zeff"],
                "mean_ld":   r["mean_ld"],
                "std_ld":    r["std_ld"],
                "n":         r["n"],
            }
    return None


def setup_phase_axes(ax, k):
    # Region shading
    ax.axhspan(0.5, 1.05, xmin=0.0, xmax=0.5, alpha=0.07, color="C0", zorder=0)
    ax.axvspan(0.5, 1.05, ymin=0.0, ymax=0.5, alpha=0.07, color="C2", zorder=0)
    ax.add_patch(plt.Rectangle((0.0, 0.0), 0.5, 0.5,
                                alpha=0.07, color="C3", zorder=0))
    ax.text(0.04, 0.97, "RELATIVISTIC", color="C0", fontsize=8.5,
            fontweight="bold", va="top")
    ax.text(0.97, 0.04, "OBJECTIVE", color="C2", fontsize=8.5,
            fontweight="bold", ha="right")
    ax.text(0.04, 0.03, "BIASED", color="C3", fontsize=8.5,
            fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.grid(alpha=0.25)


def plot_grid(models, ks, out_path):
    # Make the k=15 column wider to fit the 4 extra intervention markers.
    width_ratios = [1.0] * len(ks)
    if 15 in ks:
        width_ratios[ks.index(15)] = 1.5
    fig, axes = plt.subplots(len(models), len(ks),
                              figsize=(4.0 * sum(width_ratios), 4.4 * len(models)),
                              squeeze=False,
                              gridspec_kw={"width_ratios": width_ratios})

    for mi, model in enumerate(models):
        for ki, k in enumerate(ks):
            ax = axes[mi, ki]
            setup_phase_axes(ax, k)

            base = load_baseline(model, k)
            abl  = load_l0all(model, k)

            # For k=0, r_z is NaN; place marker at y=0 to show on axis.
            def marker_pos(d):
                rz = d.get("r_ld_zeff", float("nan"))
                rx = d.get("r_ld_x", float("nan"))
                if np.isnan(rz):
                    return rx, 0.0, True   # third = "z undefined"
                return rx, rz, False

            if base is not None:
                rx, rz, undef = marker_pos(base)
                ax.scatter(rx, rz, s=220, color="tab:green", marker="o",
                           edgecolor="black", linewidth=1.0, zorder=4,
                           label="baseline")
                txt = (f"baseline\n⟨LD⟩={base['mean_ld']:+.2f}\n"
                       f"std={base['std_ld']:.2f}")
                if undef:
                    txt += "\nz undefined"
                ax.annotate(txt, (rx, rz),
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=7.2, color="tab:green",
                            fontweight="bold")

            if abl is not None:
                rx, rz, undef = marker_pos(abl)
                ax.scatter(rx, rz, s=220, color="C3", marker="X",
                           edgecolor="black", linewidth=1.0, zorder=5,
                           label="l0_all")
                txt = (f"l0_all\n⟨LD⟩={abl['mean_ld']:+.2f}\n"
                       f"std={abl['std_ld']:.2f}")
                if undef:
                    txt += "\nz undefined"
                ax.annotate(txt, (rx, rz),
                            xytext=(8, -28), textcoords="offset points",
                            fontsize=7.2, color="C3", fontweight="bold")

            # Optionally connect baseline → l0_all with a thin arrow.
            if base is not None and abl is not None:
                bx, by, _ = marker_pos(base)
                ax_, ay_, _ = marker_pos(abl)
                ax.annotate("", xy=(ax_, ay_), xytext=(bx, by),
                            arrowprops=dict(arrowstyle="->", color="black",
                                             alpha=0.4, lw=1.0),
                            zorder=3)

            # Add residual-stream interventions (only on the k=15 panel — that's
            # where Phase 1's primal_z and cell-mean trajectory live).
            if k == 15:
                interv = load_residual_interventions(model, k=15)
                for mode_name, color, label, dx_text, dy_text in [
                    ("proj_out",      "tab:purple", "proj_out",       10,  4),
                    ("mean_ablate",   "tab:orange", "mean_ablate",   -78, -2),
                    ("manifold_a075", "tab:cyan",   "manifold α=0.75", 10, 4),
                    ("manifold_a100", "tab:blue",   "manifold α=1.0",  10, -14),
                ]:
                    if mode_name not in interv:
                        continue
                    r = interv[mode_name]
                    rx, rz = r["r_ld_x"], r["r_ld_zeff"]
                    if not (np.isfinite(rx) and np.isfinite(rz)):
                        continue
                    ax.scatter(rx, rz, s=180, color=color, marker="D",
                               edgecolor="black", linewidth=0.7, zorder=4)
                    ax.annotate(label, (rx, rz),
                                xytext=(dx_text, dy_text), textcoords="offset points",
                                fontsize=7.5, color=color, fontweight="bold")
                    # Arrow from baseline
                    if base is not None:
                        bx, by, _ = marker_pos(base)
                        ax.annotate("", xy=(rx, rz), xytext=(bx, by),
                                    arrowprops=dict(arrowstyle="->", color=color,
                                                     alpha=0.5, lw=0.8,
                                                     linestyle=(0, (3, 2))),
                                    zorder=3)

            ax.set_title(f"{model.replace('gemma2-', '').upper()}  |  k={k}",
                         fontsize=11)
            if mi == len(models) - 1:
                ax.set_xlabel(r"$r$(LD, x)  →  objective")
            if ki == 0:
                ax.set_ylabel(r"$r$(LD, z_eff)  →  relativistic")

    fig.suptitle("Phase 2D — phase-space migration with shot count k\n"
                 "(green = baseline; red X = full L0 attention ablation)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 5, 15])
    p.add_argument("--out", default=str(FIG_DIR / "p2d_phase_grid.png"))
    args = p.parse_args()
    plot_grid(args.models, args.ks, Path(args.out))


if __name__ == "__main__":
    main()
