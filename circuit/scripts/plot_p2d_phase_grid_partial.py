"""Phase 2D — phase-space grid in PARTIAL-CORRELATION axes.

x-axis: partial r(LD, x | z_eff)   "objective component"
y-axis: partial r(LD, z_eff | x)   "relativistic component"

Both bounded in [-1, 1] independently. Top-right corner is meaningful here:
a model that reads both x and z would land there.

Marker size encodes std(LD): small marker = collapsed LD = biased.

For interventions where we have summary (r_lz, r_lx) but not per-prompt data,
we use the closed-form partial-correlation formula:
  partial_r_z = (r_lz - r_lx*r_zx) / sqrt((1-r_lx²)(1-r_zx²))
  partial_r_x = (r_lx - r_lz*r_zx) / sqrt((1-r_lz²)(1-r_zx²))
where r_zx is the (z_eff, x) correlation in the prompt set, computed once per k.

For p2c ablations that don't save r_lx, we recover it via index-replay (the
random subsampling in p2c_ablate_heads.py is deterministic given seed=0).
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


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def partial_corr(r_lz, r_lx, r_zx):
    """Returns (partial_r_z_given_x, partial_r_x_given_z)."""
    if not (np.isfinite(r_lz) and np.isfinite(r_lx) and np.isfinite(r_zx)):
        return float("nan"), float("nan")
    den_z = np.sqrt(max(0.0, (1 - r_lx**2) * (1 - r_zx**2)))
    den_x = np.sqrt(max(0.0, (1 - r_lz**2) * (1 - r_zx**2)))
    p_z = (r_lz - r_lx * r_zx) / den_z if den_z > 1e-9 else float("nan")
    p_x = (r_lx - r_lz * r_zx) / den_x if den_x > 1e-9 else float("nan")
    # Clip to [-1, 1] to handle numerical noise.
    p_z = max(-1.0, min(1.0, p_z))
    p_x = max(-1.0, min(1.0, p_x))
    return p_z, p_x


def get_r_zx_per_k(pair="height", ks=(1, 2, 4, 5, 8, 15)):
    """Compute r(z_eff, x) per k from the full prompt set."""
    out = {}
    for k in ks:
        path = REPO / "data" / "p2_shot_sweep" / f"{pair}_k{k}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        x = np.array([r["x"] for r in rows], dtype=np.float64)
        z = np.array([r["z_eff"] for r in rows], dtype=np.float64)
        out[k] = safe_pearson(x, z)
    return out


def baseline_partials(model, pair, k):
    """Use full per-prompt LD from p2_ld; compute partial directly."""
    npz = REPO / "results" / "p2_ld" / model / f"{pair}_k{k}.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    ld = d["ld"].astype(np.float64)
    x = d["x"].astype(np.float64)
    z = d["z_eff"].astype(np.float64)
    mask = np.isfinite(z)
    if mask.sum() < 3 or k == 0:
        # k=0: z undefined, return only r_ld_x
        return {
            "r_ld_x":   safe_pearson(x, ld),
            "r_ld_z":   float("nan"),
            "r_zx":     float("nan"),
            "p_lz_x":   float("nan"),
            "p_lx_z":   float("nan"),
            "mean_ld":  float(ld.mean()),
            "std_ld":   float(ld.std(ddof=1)) if len(ld) > 1 else 0.0,
        }
    r_lz = safe_pearson(z[mask], ld[mask])
    r_lx = safe_pearson(x[mask], ld[mask])
    r_zx = safe_pearson(z[mask], x[mask])
    p_z, p_x = partial_corr(r_lz, r_lx, r_zx)
    return {
        "r_ld_x":  r_lx, "r_ld_z": r_lz, "r_zx": r_zx,
        "p_lz_x": p_z, "p_lx_z": p_x,
        "mean_ld": float(ld[mask].mean()),
        "std_ld":  float(ld[mask].std(ddof=1)),
    }


def replay_p2c_indices(model_short, k, n_total_per_k):
    """Replay rng.choice from p2c_ablate_heads.py to recover prompt indices."""
    rng = np.random.default_rng(0)
    if model_short == "gemma2-2b":
        n_heads = 8
        candidate_set = {(1, 6), (1, 0), (1, 1), (1, 4)}
        n_pp = 4
    else:
        n_heads = 16
        candidate_set = {(1, 11), (1, 10), (1, 6)}
        n_pp = 3
    candidate_l1 = {h for L, h in candidate_set if L == 1}
    avail_l1 = [h for h in range(n_heads) if h not in candidate_l1]
    _ = rng.choice(avail_l1)
    avail_pool = [(L, h) for L in (0, 1) for h in range(n_heads) if (L, h) not in candidate_set]
    _ = rng.choice(len(avail_pool), size=n_pp, replace=False)
    for ki in [1, 4, 15]:
        idx = rng.choice(n_total_per_k[ki], size=min(600, n_total_per_k[ki]), replace=False)
        if ki == k:
            return idx
    return None


def l0_partials(model, pair, k, r_zx, r_zx_set=None):
    """Get l0_all (r_lz, r_lx) → partials. Try p2d_l0all_per_k, p2d_partial_l0,
    then p2c_ablation.

    For p2c we lack r_lx; recover it from the saved 200-sample LD via index
    replay, then use r_zx_set computed on the same 600-prompt subsample.
    """
    # 1) p2d_l0all_per_k
    p_l0pk = REPO / "results" / f"p2d_l0all_per_k_{model}_{pair}.json"
    if p_l0pk.exists():
        with p_l0pk.open() as f:
            D = json.load(f)
        if f"k{k}" in D["results"]:
            r = D["results"][f"k{k}"]["l0_all"]
            r_lz = r.get("r_ld_zeff", float("nan"))
            r_lx = r.get("r_ld_x", float("nan"))
            p_z, p_x = partial_corr(r_lz, r_lx, r_zx)
            return {
                "r_ld_x": r_lx, "r_ld_z": r_lz, "r_zx": r_zx,
                "p_lz_x": p_z, "p_lx_z": p_x,
                "mean_ld": r.get("mean_ld", float("nan")),
                "std_ld": r.get("std_ld", float("nan")),
            }

    # 2) p2d_partial_l0 (only at k=1)
    p_pl0 = REPO / "results" / f"p2d_partial_l0_{model}_{pair}_k{k}.json"
    if p_pl0.exists():
        with p_pl0.open() as f:
            D = json.load(f)
        r = D["l0_all"]
        r_lz = r["r_ld_zeff"]; r_lx = r["r_ld_x"]
        p_z, p_x = partial_corr(r_lz, r_lx, r_zx)
        return {
            "r_ld_x": r_lx, "r_ld_z": r_lz, "r_zx": r_zx,
            "p_lz_x": p_z, "p_lx_z": p_x,
            "mean_ld": r["mean_ld"], "std_ld": r["std_ld"],
        }

    # 3) p2c (need to recover r_lx via index replay)
    p_p2c = REPO / "results" / f"p2c_ablation_{model}_{pair}.json"
    if p_p2c.exists():
        with p_p2c.open() as f:
            D = json.load(f)
        if f"k{k}" not in D["results"]:
            return None
        r = D["results"][f"k{k}"]["l0_all"]
        ld_sample = np.asarray(r["ld_sample"])
        z_sample = np.asarray(r["z_eff_sample"])
        # Replay indices for this k
        n_total = {}
        for kk in [1, 4, 15]:
            ppath = REPO / "data" / "p2_shot_sweep" / f"{pair}_k{kk}.jsonl"
            n_total[kk] = sum(1 for _ in ppath.open())
        idx = replay_p2c_indices(model, k, n_total)
        # x for the FIRST 200 prompts of the 600-sample
        rows = [json.loads(l) for l in (REPO / "data" / "p2_shot_sweep" /
                                         f"{pair}_k{k}.jsonl").open()]
        x_full = np.array([rows[int(i)]["x"] for i in idx])
        x_200 = x_full[:200]
        r_lz = safe_pearson(z_sample, ld_sample)
        r_lx = safe_pearson(x_200, ld_sample)
        r_zx_local = safe_pearson(z_sample, x_200)
        p_z, p_x = partial_corr(r_lz, r_lx, r_zx_local)
        return {
            "r_ld_x": r_lx, "r_ld_z": r_lz, "r_zx": r_zx_local,
            "p_lz_x": p_z, "p_lx_z": p_x,
            "mean_ld": r["mean_ld"], "std_ld": r["std_ld"],
        }
    return None


def residual_intervention_partials(model, pair, k, r_zx):
    """For p2e residual-stream interventions (only at k=15)."""
    p = REPO / "results" / f"p2e_residual_interventions_{model}_{pair}_k{k}.json"
    if not p.exists():
        return {}
    with p.open() as f:
        D = json.load(f)
    out = {}
    for mode, r in D["results"].items():
        if mode == "baseline":
            continue
        r_lz = r["r_ld_zeff"]; r_lx = r["r_ld_x"]
        p_z, p_x = partial_corr(r_lz, r_lx, r_zx)
        out[mode] = {
            "r_ld_x": r_lx, "r_ld_z": r_lz, "r_zx": r_zx,
            "p_lz_x": p_z, "p_lx_z": p_x,
            "mean_ld": r["mean_ld"], "std_ld": r["std_ld"],
        }
    return out


def setup_axes(ax):
    # Four equal quadrants of the unit square [0,1]^2.
    quadrants = [
        ((0.0, 0.0), (0.5, 0.5), "C3",          "BIASED",       "left",   "bottom"),
        ((0.5, 0.0), (1.0, 0.5), "C2",          "OBJECTIVE",    "right",  "bottom"),
        ((0.0, 0.5), (1.0, 1.0), "C0",          "RELATIVISTIC", "left",   "top"),
        ((0.5, 0.5), (1.0, 1.0), "gold",        "COMPLETE",     "right",  "top"),
    ]
    for (x0, y0), (x1, y1), color, label, ha, va in quadrants:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    alpha=0.10, color=color, zorder=0))
        text_color = "darkgoldenrod" if color == "gold" else color
        tx = x1 - 0.02 if ha == "right" else x0 + 0.02
        ty = y1 - 0.02 if va == "top" else y0 + 0.02
        ax.text(tx, ty, label, color=text_color, fontsize=8.5,
                fontweight="bold", ha=ha, va=va)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="black", linewidth=0.3, alpha=0.4)
    ax.axvline(0.5, color="black", linewidth=0.3, alpha=0.4)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(alpha=0.2)


def std_to_marker_size(std_ld, scale=80, base=40):
    """Marker size scaled by std(LD): big = signal-rich, small = collapsed/biased."""
    if not np.isfinite(std_ld):
        return base
    return base + scale * std_ld   # std typically 0.2 to 2.6


def plot_partial_grid(models, ks, pair="height"):
    r_zx_per_k = get_r_zx_per_k(pair, ks=ks if 0 not in ks else [k for k in ks if k > 0])

    width_ratios = [1.0] * len(ks)
    if 15 in ks:
        width_ratios[ks.index(15)] = 1.7
    fig, axes = plt.subplots(len(models), len(ks),
                              figsize=(3.1 * sum(width_ratios), 3.4 * len(models)),
                              squeeze=False,
                              gridspec_kw={"width_ratios": width_ratios,
                                            "hspace": 0.32, "wspace": 0.30})

    for mi, model in enumerate(models):
        for ki, k in enumerate(ks):
            ax = axes[mi, ki]
            setup_axes(ax)

            # Use marker positions:
            # x-axis: partial r(LD, x | z) ("objective component")
            # y-axis: partial r(LD, z | x) ("relativistic component")
            base = baseline_partials(model, pair, k)
            r_zx = r_zx_per_k.get(k, 0.0)

            if base is not None:
                if k == 0:
                    bx = base["r_ld_x"]    # raw r since z undefined
                    by = 0.0
                else:
                    bx = base["p_lx_z"]
                    by = base["p_lz_x"]
                size = std_to_marker_size(base["std_ld"])
                ax.scatter(bx, by, s=size, color="tab:green", marker="o",
                           edgecolor="black", linewidth=1.0, zorder=4)
                txt = (f"baseline\nstd={base['std_ld']:.2f}")
                ax.annotate(txt, (bx, by),
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=7.5, color="tab:green", fontweight="bold")

            l0 = l0_partials(model, pair, k, r_zx)
            if l0 is not None:
                if k == 0:
                    lx = l0["r_ld_x"]; ly = 0.0
                else:
                    lx = l0["p_lx_z"]; ly = l0["p_lz_x"]
                size = std_to_marker_size(l0["std_ld"])
                ax.scatter(lx, ly, s=size, color="C3", marker="X",
                           edgecolor="black", linewidth=1.0, zorder=5)
                txt = (f"l0_all\nstd={l0['std_ld']:.2f}")
                ax.annotate(txt, (lx, ly),
                            xytext=(8, -22), textcoords="offset points",
                            fontsize=7.5, color="C3", fontweight="bold")
                # Arrow
                if base is not None:
                    if k == 0:
                        bx_, by_ = base["r_ld_x"], 0.0
                    else:
                        bx_, by_ = base["p_lx_z"], base["p_lz_x"]
                    ax.annotate("", xy=(lx, ly), xytext=(bx_, by_),
                                arrowprops=dict(arrowstyle="->", color="black",
                                                 alpha=0.4, lw=1.0), zorder=3)

            if k == 15:
                interv = residual_intervention_partials(model, pair, 15, r_zx)
                for mode_name, color, label, dx_t, dy_t in [
                    ("proj_out",      "tab:purple", "proj_out",      8,  6),
                    ("mean_ablate",   "tab:orange", "mean_ablate", -68, -4),
                    ("manifold_a075", "tab:cyan",   "manifold α=0.75", 8, 6),
                    ("manifold_a100", "tab:blue",   "manifold α=1.0",  8, -14),
                ]:
                    if mode_name not in interv:
                        continue
                    r = interv[mode_name]
                    px, py = r["p_lx_z"], r["p_lz_x"]
                    if not (np.isfinite(px) and np.isfinite(py)):
                        continue
                    size = std_to_marker_size(r["std_ld"])
                    ax.scatter(px, py, s=size, color=color, marker="D",
                               edgecolor="black", linewidth=0.7, zorder=4)
                    ax.annotate(label, (px, py),
                                xytext=(dx_t, dy_t), textcoords="offset points",
                                fontsize=7.5, color=color, fontweight="bold")
                    if base is not None:
                        bx_, by_ = base["p_lx_z"], base["p_lz_x"]
                        ax.annotate("", xy=(px, py), xytext=(bx_, by_),
                                    arrowprops=dict(arrowstyle="->", color=color,
                                                     alpha=0.5, lw=0.8,
                                                     linestyle=(0, (3, 2))),
                                    zorder=3)

            ax.set_title(f"{model.replace('gemma2-', '').upper()}  |  k={k}",
                         fontsize=11)
            if mi == len(models) - 1:
                ax.set_xlabel(r"partial $r$(LD, x | z)  →  objective")
            if ki == 0:
                ax.set_ylabel(r"partial $r$(LD, z | x)  →  relativistic")

    fig.suptitle("Phase 2D — partial-correlation phase grid\n"
                 "(marker size ∝ std(LD); small marker = collapsed/biased)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "p2d_phase_grid_partial.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma2-2b", "gemma2-9b"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 5, 15])
    p.add_argument("--pair", default="height")
    args = p.parse_args()
    plot_partial_grid(args.models, args.ks, args.pair)


if __name__ == "__main__":
    main()
