"""Phase 2G — context-info flow per layer.

Tracks how the target's last position pulls information from each of the 15
context value tokens, layer by layer. Reuses `attn_last` from p2_attn NPZs.

Computes per (model, pair):
  ctx_attn_per_slot[L, K] = mean over prompts (and over heads) of attention
      from target's last position to context slot k at layer L.
  ctx_attn_total_per_layer[L] = ctx_attn_per_slot summed over K.
  r_slot_layer[L, K] = Pearson(per-prompt attn-to-slot-k, z_eff)
      — where in (layer × slot) space does z get encoded?

Also computes:
  Mean attention to slot per z-tertile (separated low/mid/high z).

Output: per-pair JSON, and a 2x3 grid figure (models × pairs) for each of:
  - heatmap layer × slot (mean attention)
  - per-layer total ctx_attn split by z tertile
  - heatmap layer × slot (Pearson r with z_eff)

Usage:
  python3 scripts/p2g_info_flow.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parent.parent
RES_DIR = REPO / "results"
FIG_DIR = REPO / "figures"

MODELS = ["gemma2-2b", "gemma2-9b"]
PAIRS = ["height", "weight", "speed"]


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    xm = x[mask] - x[mask].mean(); ym = y[mask] - y[mask].mean()
    den = (xm.std() * ym.std() * len(xm))
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def per_slot_attn(attn_last, context_pos, pad_offset):
    """Returns shape (N, L, H, K) — attention from target to each ctx slot,
    summed over the slot's tokens."""
    n, n_layers, n_heads, T = attn_last.shape
    K = context_pos.shape[1]
    out = np.zeros((n, n_layers, n_heads, K), dtype=np.float32)
    for i in range(n):
        po = int(pad_offset[i])
        for k in range(K):
            s, e = int(context_pos[i, k, 0]), int(context_pos[i, k, 1])
            if s < 0:
                out[i, :, :, k] = np.nan
                continue
            out[i, :, :, k] = attn_last[i, :, :,
                                         po + s:po + e].astype(np.float32).sum(-1)
    return out


def compute_per_pair(model, pair, k=15):
    npz = RES_DIR / "p2_attn" / model / f"{pair}_k{k}.npz"
    d = np.load(npz, allow_pickle=True)
    n, n_layers, n_heads, T = d["attn_last"].shape
    z_eff = d["z_eff"].astype(np.float64)

    # Per-slot attention summed over heads → (N, L, K).
    per_slot = per_slot_attn(d["attn_last"], d["context_pos"], d["pad_offset"])
    sum_heads = per_slot.sum(axis=2)  # (N, L, K)
    K = sum_heads.shape[-1]

    # Means.
    mean_slot = np.nanmean(sum_heads, axis=0)             # (L, K)
    total_layer = np.nansum(sum_heads, axis=2)            # (N, L)
    mean_total = np.nanmean(total_layer, axis=0)          # (L,)

    # z-tertile splits.
    finite = np.isfinite(z_eff)
    z_f = z_eff[finite]
    sum_f = sum_heads[finite]
    total_f = total_layer[finite]
    qs = np.quantile(z_f, [0.33, 0.67])
    lo = z_f < qs[0]; hi = z_f >= qs[1]; mid = ~lo & ~hi
    bin_slot = {
        "lo": np.nanmean(sum_f[lo], axis=0),
        "mid": np.nanmean(sum_f[mid], axis=0),
        "hi": np.nanmean(sum_f[hi], axis=0),
    }
    bin_total = {
        "lo": np.nanmean(total_f[lo], axis=0),
        "mid": np.nanmean(total_f[mid], axis=0),
        "hi": np.nanmean(total_f[hi], axis=0),
    }

    abs_z = np.abs(z_eff)

    # r(per-prompt attn-to-slot-k at layer L, z_eff) and same with |z|.
    r_slot_layer = np.zeros((n_layers, K), dtype=np.float64)
    r_slot_layer_abs = np.zeros((n_layers, K), dtype=np.float64)
    for L in range(n_layers):
        for kk in range(K):
            r_slot_layer[L, kk]     = safe_pearson(z_eff, sum_heads[:, L, kk])
            r_slot_layer_abs[L, kk] = safe_pearson(abs_z, sum_heads[:, L, kk])

    # Total per-layer attention r with z and with |z|.
    r_total_layer = np.zeros(n_layers, dtype=np.float64)
    r_total_layer_abs = np.zeros(n_layers, dtype=np.float64)
    for L in range(n_layers):
        r_total_layer[L]     = safe_pearson(z_eff, total_layer[:, L])
        r_total_layer_abs[L] = safe_pearson(abs_z, total_layer[:, L])

    return dict(
        n_layers=n_layers, n_heads=n_heads, K=K,
        mean_slot=mean_slot,                       # (L, K)
        mean_total_layer=mean_total,               # (L,)
        bin_slot=bin_slot,                         # 3 × (L, K)
        bin_total=bin_total,                       # 3 × (L,)
        r_slot_layer=r_slot_layer,                 # (L, K)
        r_total_layer=r_total_layer,               # (L,)
        r_slot_layer_abs=r_slot_layer_abs,         # (L, K)  with |z|
        r_total_layer_abs=r_total_layer_abs,       # (L,)
    )


def plot_grid_mean(results, out_path):
    """6-panel grid: rows=models, cols=pairs.
    Each panel: heatmap of mean attention, layer × slot."""
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(4.6 * len(PAIRS), 3.6 * len(MODELS)),
                              squeeze=False)
    vmax = max(R["mean_slot"].max() for R in results.values())
    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            ax = axes[mi, pi]
            R = results[(model, pair)]
            M = R["mean_slot"]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}",
                          fontsize=10)
            if pi == 0:
                ax.set_ylabel("layer")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("context slot (1..15)")
            ax.set_xticks(range(0, R["K"], max(1, R["K"] // 8)))
            plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Phase 2G — mean attention from target → context slot, "
                 "per layer\n(sum over heads, mean over prompts)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_grid_total_by_z(results, out_path):
    """6-panel grid: per-layer TOTAL attention to ctx_values, split by z bin.
    Lines for low / mid / high z, plus mean across all prompts."""
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(4.6 * len(PAIRS), 3.6 * len(MODELS)),
                              squeeze=False)
    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            ax = axes[mi, pi]
            R = results[(model, pair)]
            xs = np.arange(R["n_layers"])
            ax.plot(xs, R["bin_total"]["lo"], "-", color="tab:blue",
                     label="z low",  linewidth=1.5)
            ax.plot(xs, R["bin_total"]["mid"], "-", color="tab:gray",
                     label="z mid",  linewidth=1.0, alpha=0.7)
            ax.plot(xs, R["bin_total"]["hi"], "-", color="tab:red",
                     label="z high", linewidth=1.5)
            ax.plot(xs, R["mean_total_layer"], "--", color="black",
                     label="all", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}",
                          fontsize=10)
            if pi == 0:
                ax.set_ylabel("∑ attn to ctx_values")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("layer")
            ax.grid(alpha=0.25)
            if mi == 0 and pi == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Phase 2G — total target → context attention by layer, "
                 "split by z_eff tertile\n"
                 "(z low = target below context; z high = target above)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_grid_r_layer_slot(results, out_path, key="r_slot_layer",
                            total_key="r_total_layer", what="z_eff"):
    """6-panel grid: heatmap of r(per-prompt attn-to-slot-k at layer L, X)
    where X is either z_eff (signed) or |z_eff| (magnitude)."""
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(4.6 * len(PAIRS), 3.6 * len(MODELS)),
                              squeeze=False)
    vmax = max(np.nanmax(np.abs(R[key])) for R in results.values())
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.5
    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            ax = axes[mi, pi]
            R = results[(model, pair)]
            M = R[key]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}  "
                          f"(layer-marg r_max={np.nanmax(np.abs(R[total_key])):.2f})",
                          fontsize=9)
            if pi == 0:
                ax.set_ylabel("layer")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("context slot (1..15)")
            ax.set_xticks(range(0, R["K"], max(1, R["K"] // 8)))
            plt.colorbar(im, ax=ax, fraction=0.04)
    label = ("z_eff (signed)" if what == "z_eff"
            else "|z_eff| (magnitude)")
    desc = ("blue = attention drops as target moves above context"
            if what == "z_eff"
            else "red = attention rises with how unusual the target is "
                 "(symmetric in z direction)")
    fig.suptitle(f"Phase 2G — r(target → ctx-slot-k attention, {label}) per layer × slot\n"
                 f"({desc})",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_grid_signed_vs_magnitude(results, out_path):
    """For each (model, pair), 2 panels side-by-side: |r(signed z)| vs r(|z|).
    A cell that is dark in signed but light in |z| is purely directional;
    a cell that is light in signed but dark in |z| is a magnitude detector
    (U-shape in z). Layout: 6 model×pair rows × 2 cols."""
    n_panels = len(MODELS) * len(PAIRS)
    fig, axes = plt.subplots(n_panels, 2, figsize=(9, 3.5 * n_panels),
                              squeeze=False)
    vmax_s = max(np.nanmax(np.abs(R["r_slot_layer"])) for R in results.values())
    vmax_a = max(np.nanmax(np.abs(R["r_slot_layer_abs"])) for R in results.values())
    vmax = max(vmax_s, vmax_a)
    row = 0
    for model in MODELS:
        for pair in PAIRS:
            R = results[(model, pair)]
            for col, (M, title, cmap, vmin) in enumerate([
                (np.abs(R["r_slot_layer"]),    "|r(z signed)|",  "magma", 0.0),
                (R["r_slot_layer_abs"],         "r(|z|)",         "RdBu_r", -vmax),
            ]):
                ax = axes[row, col]
                if cmap == "magma":
                    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(M, aspect="auto", cmap=cmap,
                                    norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
                if col == 0:
                    ax.set_ylabel(f"{model.replace('gemma2-', '').upper()} | {pair}\nlayer",
                                   fontsize=9)
                if row == n_panels - 1:
                    ax.set_xlabel("context slot")
                ax.set_title(title, fontsize=9)
                ax.set_xticks(range(0, R["K"], max(1, R["K"] // 8)))
                plt.colorbar(im, ax=ax, fraction=0.04)
            row += 1
    fig.suptitle("Phase 2G — signed-z vs |z| attention correlation\n"
                 "(left = |r| with signed z (directional); "
                 "right = r with |z| (magnitude / unusualness, red=rises with extremity))",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_grid_early_vs_late(results, out_path):
    """For each (model, pair), per-layer attention by 3 slot groups
    {early=slots 1-5, mid=6-10, late=11-15}, split by z bin (low/hi) — shows
    the budget reallocation across slot groups as z changes."""
    fig, axes = plt.subplots(len(MODELS), len(PAIRS),
                              figsize=(4.6 * len(PAIRS), 3.6 * len(MODELS)),
                              squeeze=False)
    for mi, model in enumerate(MODELS):
        for pi, pair in enumerate(PAIRS):
            ax = axes[mi, pi]
            R = results[(model, pair)]
            xs = np.arange(R["n_layers"])
            # mean_slot is (L, K). bin_slot keys (L, K).
            for label, bin_key, color in [("z low", "lo", "tab:blue"),
                                            ("z high", "hi", "tab:red")]:
                M = R["bin_slot"][bin_key]   # (L, K)
                early = M[:, :5].sum(1)
                mid   = M[:, 5:10].sum(1)
                late  = M[:, 10:].sum(1)
                ax.plot(xs, early, "-",  color=color, alpha=0.9,
                         linewidth=1.5, label=f"{label} early(1-5)" if mi==0 and pi==0 else None)
                ax.plot(xs, late,  "--", color=color, alpha=0.9,
                         linewidth=1.5, label=f"{label} late(11-15)" if mi==0 and pi==0 else None)
            ax.set_title(f"{model.replace('gemma2-', '').upper()} | {pair}",
                          fontsize=10)
            if pi == 0:
                ax.set_ylabel("∑ attn within slot group")
            if mi == len(MODELS) - 1:
                ax.set_xlabel("layer")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 2G — early-vs-late slot attention budget by z\n"
                 "(solid = early slots 1-5; dashed = late 11-15; "
                 "blue=z low, red=z high)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=15)
    args = p.parse_args()

    results = {}
    for model in MODELS:
        for pair in PAIRS:
            print(f"[p2g] {model} / {pair} ...", flush=True)
            R = compute_per_pair(model, pair, k=args.k)
            results[(model, pair)] = R
            # peak: which layer pulls most context attention
            peak_layer = int(np.argmax(R["mean_total_layer"]))
            peak_val = R["mean_total_layer"][peak_layer]
            r_peak = float(R["r_total_layer"][peak_layer])
            r_argmax = int(np.argmax(np.abs(R["r_total_layer"])))
            r_argmax_val = float(R["r_total_layer"][r_argmax])
            print(f"   peak ctx-attn layer = L{peak_layer:>2}  ({peak_val:.3f}); "
                  f"r-with-z at peak = {r_peak:+.3f}; "
                  f"|r|-argmax layer = L{r_argmax:>2}  (r={r_argmax_val:+.3f})")

    FIG_DIR.mkdir(exist_ok=True)
    plot_grid_mean(results, FIG_DIR / "p2g_info_flow_mean.png")
    plot_grid_total_by_z(results, FIG_DIR / "p2g_info_flow_total_by_z.png")
    plot_grid_r_layer_slot(results, FIG_DIR / "p2g_info_flow_r_layer_slot.png",
                            key="r_slot_layer", total_key="r_total_layer",
                            what="z_eff")
    plot_grid_r_layer_slot(results, FIG_DIR / "p2g_info_flow_r_layer_slot_absz.png",
                            key="r_slot_layer_abs", total_key="r_total_layer_abs",
                            what="abs_z")
    plot_grid_signed_vs_magnitude(results,
                                   FIG_DIR / "p2g_info_flow_signed_vs_abs.png")
    plot_grid_early_vs_late(results, FIG_DIR / "p2g_info_flow_early_vs_late.png")

    # Save numeric outputs.
    out = {}
    for (m, p_), R in results.items():
        out[f"{m}/{p_}"] = {
            "mean_total_layer": R["mean_total_layer"].tolist(),
            "r_total_layer":    R["r_total_layer"].tolist(),
            "mean_slot":        R["mean_slot"].tolist(),
            "r_slot_layer":     R["r_slot_layer"].tolist(),
        }
    out_json = RES_DIR / f"p2g_info_flow_summary_k{args.k}.json"
    with out_json.open("w") as f:
        json.dump(out, f)
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
