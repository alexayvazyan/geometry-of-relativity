"""Phase 2K — per-head value selectivity.

For each (layer, head, prompt), compute:
  attended_z[i, L, h] = Σ_k a_k * z_k  /  Σ_k a_k
where a_k = attn-to-slot-k for this (i, L, h), and z_k is the within-prompt
standardized value of slot k:  z_k = (value_k - mean_k) / std_k.

Heads with consistently HIGH attended_z (across prompts) preferentially
attend to context items whose value is HIGH within their prompt.
Heads with LOW attended_z attend to LOW within-prompt-value items.

Then test: do the high-tuned heads' outputs correlate with high z_eff
and the low-tuned heads' outputs with low z_eff? That would say "high-tuned
heads fire more in high-z prompts" — confirming dual specialization.

Outputs:
  - layer × head heatmap of mean(attended_z) — head tuning map
  - distribution histogram of head tunings, separated by layer band
  - correlation r(head's per-prompt total ctx_attn, z_eff) vs the head's
    tuning — does selectivity predict z-firing?

Usage:
  python3 scripts/p2k_head_value_selectivity.py --model gemma2-9b --pair height
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


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    xm = x[mask] - x[mask].mean(); ym = y[mask] - y[mask].mean()
    return float((xm * ym).sum() / (xm.std() * ym.std() * len(xm) + 1e-12))


def per_slot_attn_per_head(attn_last, context_pos, pad_offset):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    args = p.parse_args()

    npz = RES_DIR / "p2_attn" / args.model / f"{args.pair}_k{args.k}.npz"
    print(f"[p2k] loading {npz}")
    d = np.load(npz, allow_pickle=True)
    n, n_layers, n_heads, T = d["attn_last"].shape
    K = d["context_pos"].shape[1]
    z_eff = d["z_eff"].astype(np.float64)
    x_arr = d["x"].astype(np.float64)

    # Slot values from JSONL.
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    trials = [json.loads(l) for l in in_path.open()]
    slot_values = np.full((n, K), np.nan, dtype=np.float64)
    for i, t in enumerate(trials):
        ctx = t.get("context_values", [])
        for k in range(min(K, len(ctx))):
            slot_values[i, k] = float(ctx[k])

    # Within-prompt z of each slot.
    sv_means = np.nanmean(slot_values, axis=1, keepdims=True)
    sv_stds = np.nanstd(slot_values, axis=1, keepdims=True)
    within_z = (slot_values - sv_means) / (sv_stds + 1e-9)
    finite_z = ~np.isnan(within_z)

    # Per-slot attention per head.
    per_slot = per_slot_attn_per_head(d["attn_last"], d["context_pos"],
                                        d["pad_offset"])
    # Mask out-of-range slots.
    per_slot_safe = np.where(np.isnan(per_slot), 0.0, per_slot).astype(np.float64)
    within_z_safe = np.where(np.isnan(within_z), 0.0, within_z).astype(np.float64)

    # Weighted attended-z per (prompt, layer, head).
    num = np.einsum("nlhk,nk->nlh", per_slot_safe, within_z_safe)
    den = per_slot_safe.sum(-1)
    attended_z = num / (den + 1e-12)   # (N, L, H)
    # Where ctx_attn is essentially zero, attended_z is meaningless — set NaN.
    attended_z = np.where(den < 1e-6, np.nan, attended_z)

    # Per-head mean attended-z (over prompts).
    head_tuning = np.nanmean(attended_z, axis=0)   # (L, H)

    # Per-head correlation r(attended_z[i, L, h], z_eff[i]) — does the head's
    # selectivity move with z_eff?
    r_head_z = np.zeros((n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        for H in range(n_heads):
            r_head_z[L, H] = safe_pearson(attended_z[:, L, H], z_eff)

    # Per-head correlation r(total_ctx_attn[i, L, h], z_eff[i]) — does the
    # head's overall ctx-magnitude scale with z?
    total_ctx = den   # (N, L, H), total attention to all ctx slots
    r_total_z = np.zeros((n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        for H in range(n_heads):
            r_total_z[L, H] = safe_pearson(total_ctx[:, L, H], z_eff)

    # Save.
    out_json = RES_DIR / f"p2k_head_value_selectivity_{args.model}_{args.pair}_k{args.k}.json"
    with out_json.open("w") as f:
        json.dump({
            "model": args.model, "pair": args.pair, "k": args.k,
            "n_layers": n_layers, "n_heads": n_heads, "n_prompts": n,
            "head_tuning": head_tuning.tolist(),
            "r_attended_z_with_z_eff": r_head_z.tolist(),
            "r_total_ctx_attn_with_z_eff": r_total_z.tolist(),
        }, f)
    print(f"  -> {out_json}")

    # ----- Plotting -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)

    # (1) Head tuning heatmap.
    ax = axes[0, 0]
    vmax = float(np.nanmax(np.abs(head_tuning)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.5
    im = ax.imshow(head_tuning, aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_title(f"Head tuning: mean attended-z\n"
                  f"(red = head attends to HIGH-value items, blue = LOW)",
                  fontsize=10)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 12)))
    plt.colorbar(im, ax=ax, fraction=0.04)

    # (2) r(attended-z, z_eff) — does selectivity vary with z_eff per prompt?
    ax = axes[0, 1]
    vmax2 = float(np.nanmax(np.abs(r_head_z)))
    if not np.isfinite(vmax2) or vmax2 == 0:
        vmax2 = 0.5
    im = ax.imshow(r_head_z, aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0, vmin=-vmax2, vmax=vmax2))
    ax.set_title(f"r(attended-z, z_eff) per head\n"
                  f"(red = head's attended item shifts HIGH as z rises)",
                  fontsize=10)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 12)))
    plt.colorbar(im, ax=ax, fraction=0.04)

    # (3) Tuning histogram, split by layer band.
    ax = axes[0, 2]
    L_third = n_layers // 3
    flat_e = head_tuning[:L_third].flatten()
    flat_m = head_tuning[L_third:2*L_third].flatten()
    flat_l = head_tuning[2*L_third:].flatten()
    bins = np.linspace(-vmax, vmax, 25)
    ax.hist(flat_e, bins=bins, alpha=0.5, label=f"early L0..{L_third-1}",
             color="tab:purple")
    ax.hist(flat_m, bins=bins, alpha=0.5, label=f"mid L{L_third}..{2*L_third-1}",
             color="tab:orange")
    ax.hist(flat_l, bins=bins, alpha=0.5, label=f"late L{2*L_third}..",
             color="tab:green")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("head tuning (mean attended-z)")
    ax.set_ylabel("# heads")
    ax.set_title("Distribution of head tunings", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Phase 2K — per-head value selectivity  |  "
                 f"{args.model} / {args.pair} k={args.k}",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out_fig = FIG_DIR / f"p2k_head_value_selectivity_{args.model}_{args.pair}.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    print(f"  -> {out_fig}")

    # Print top tuned heads.
    flat = []
    for L in range(n_layers):
        for H in range(n_heads):
            flat.append(((L, H), head_tuning[L, H]))
    flat.sort(key=lambda t: t[1])
    print(f"\n  Top-10 LOW-value-tuned heads (attend to small-value items):")
    for (L, H), t in flat[:10]:
        print(f"    L{L:>2}H{H:>2}  tuning={t:+.3f}  "
              f"mean_total_attn={den[:, L, H].mean():.3f}")
    print(f"\n  Top-10 HIGH-value-tuned heads:")
    for (L, H), t in flat[-10:][::-1]:
        print(f"    L{L:>2}H{H:>2}  tuning={t:+.3f}  "
              f"mean_total_attn={den[:, L, H].mean():.3f}")


if __name__ == "__main__":
    main()
