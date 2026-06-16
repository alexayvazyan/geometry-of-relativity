"""Phase 2G — verify the differential-querying claim.

Three sanity checks on the per-slot attention vs z_eff finding:

(A) Magnitude check — print mean attention per slot at the aggregation
    layer, split by z-tertile (low / mid / high). Shows absolute change in
    attention, not just correlation.

(B) Per-prompt examples — pick 9 prompts spanning z_eff ∈ [-3, +3], plot
    attention to each of the 15 slots at the aggregation layer.

(C) BOS-sink leakage null — compute attention to BOS (token 1) and to slot 0
    per prompt, correlate each with z_eff. If they look similar, slot 0
    correlation might be inherited from BOS sink.

(D) Value-controlled correlation — for each slot, r(attn, z_eff | slot-value).
    If the partial correlation survives after controlling for the actual
    numerical value at that slot, the effect is positional, not value-driven.

Usage:
  python3 scripts/p2g_verify.py --model gemma2-9b --pair height
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES_DIR = REPO / "results"
FIG_DIR = REPO / "figures"

PEAK = {"gemma2-2b": 16, "gemma2-9b": 21}


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    xm = x[mask] - x[mask].mean(); ym = y[mask] - y[mask].mean()
    return float((xm * ym).sum() / (xm.std() * ym.std() * len(xm) + 1e-12))


def partial_pearson(x, y, z):
    """r(x, y) controlling for z."""
    rxy = safe_pearson(x, y)
    rxz = safe_pearson(x, z)
    ryz = safe_pearson(y, z)
    if not (np.isfinite(rxy) and np.isfinite(rxz) and np.isfinite(ryz)):
        return float("nan")
    den = np.sqrt(max(0, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / den) if den > 1e-9 else float("nan")


def per_slot_attn(attn_last, context_pos, pad_offset):
    """Returns (N, L, K) — per-slot attention summed over heads and over the
    slot's token span."""
    n, n_layers, n_heads, T = attn_last.shape
    K = context_pos.shape[1]
    out = np.zeros((n, n_layers, K), dtype=np.float32)
    for i in range(n):
        po = int(pad_offset[i])
        for k in range(K):
            s, e = int(context_pos[i, k, 0]), int(context_pos[i, k, 1])
            if s < 0:
                out[i, :, k] = np.nan
                continue
            out[i, :, k] = attn_last[i, :, :, po + s:po + e].astype(np.float32).sum(-1).sum(-1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(PEAK))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    args = p.parse_args()

    npz = RES_DIR / "p2_attn" / args.model / f"{args.pair}_k{args.k}.npz"
    print(f"[p2g-verify] loading {npz}")
    d = np.load(npz, allow_pickle=True)
    z_eff = d["z_eff"].astype(np.float64)
    L_peak = PEAK[args.model]
    n, n_layers, n_heads, T = d["attn_last"].shape
    K = d["context_pos"].shape[1]

    # Per-slot attention (sum over heads, sum over token span).
    per_slot = per_slot_attn(d["attn_last"], d["context_pos"], d["pad_offset"])
    A = per_slot[:, L_peak, :]   # (N, K) — attention at aggregation layer

    # ------ (A) magnitude by z-tertile ------
    finite = np.isfinite(z_eff) & np.isfinite(A).all(axis=1)
    z_f = z_eff[finite]; A_f = A[finite]
    qs = np.quantile(z_f, [0.33, 0.67])
    lo = z_f < qs[0]; hi = z_f >= qs[1]; mid = ~lo & ~hi

    print(f"\n(A) Mean attention per slot at L{L_peak} ({args.model}/{args.pair}):")
    print(f"    N: lo={lo.sum()}, mid={mid.sum()}, hi={hi.sum()}")
    print(f"    {'slot':>4}  {'z low':>8}  {'z mid':>8}  {'z high':>8}  "
          f"{'Δhi-lo':>9}  {'%change':>9}")
    for k in range(K):
        a_lo = A_f[lo, k].mean(); a_mid = A_f[mid, k].mean()
        a_hi = A_f[hi, k].mean()
        delta = a_hi - a_lo
        pct = (delta / max(a_lo, 1e-9)) * 100
        print(f"    {k:>4}  {a_lo:>8.4f}  {a_mid:>8.4f}  {a_hi:>8.4f}  "
              f"{delta:>+9.4f}  {pct:>+8.0f}%")

    # ------ (B) per-prompt examples ------
    # Pick 9 prompts spanning z_eff range with valid attention.
    finite_idx = np.where(finite)[0]
    z_sorted = np.argsort(z_eff[finite_idx])
    pick_quantiles = np.linspace(0.05, 0.95, 9)
    sample_idx = [int(finite_idx[z_sorted[int(q * len(z_sorted))]])
                  for q in pick_quantiles]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), squeeze=False)
    for ai, idx in enumerate(sample_idx):
        ax = axes[ai // 3, ai % 3]
        bars = per_slot[idx, L_peak, :]
        z_v = z_eff[idx]
        x_v = d["x"][idx]
        prompt = d["id"][idx]
        ax.bar(np.arange(K), bars, color="#3b6ea5",
                edgecolor="black", linewidth=0.5)
        ax.set_title(f"z_eff={z_v:+.2f}, x={x_v:.0f}, id={prompt}",
                      fontsize=9)
        ax.set_xlabel("context slot")
        ax.set_ylabel("attention at aggregation layer")
        ax.grid(alpha=0.25, axis="y")
        ax.set_xticks(np.arange(0, K, 2))
    fig.suptitle(f"Phase 2G verify (B) — per-prompt attention at L{L_peak}\n"
                 f"{args.model} / {args.pair} k={args.k} (9 prompts spanning z range)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out_b = FIG_DIR / f"p2g_verify_perprompt_{args.model}_{args.pair}.png"
    fig.savefig(out_b, dpi=130, bbox_inches="tight")
    print(f"\n  -> {out_b}")
    plt.close(fig)

    # ------ (C) BOS-sink null ------
    # Sum over heads, attention from target's last token to BOS-region
    # = positions [pad_offset, first ctx_value start) — the "pre_context"
    # bucket from Phase 2F. We approximate by attention to the very first
    # non-pad token, which is BOS.
    bos_attn = np.zeros(n, dtype=np.float32)
    pre_attn = np.zeros(n, dtype=np.float32)
    for i in range(n):
        po = int(d["pad_offset"][i])
        first_ctx_start = int(d["context_pos"][i, 0, 0])  # in unpadded coords
        bos_attn[i] = d["attn_last"][i, L_peak, :, po].astype(np.float32).sum()
        pre_attn[i] = d["attn_last"][i, L_peak, :,
                                       po:po + first_ctx_start].astype(np.float32).sum()

    r_bos_z = safe_pearson(bos_attn, z_eff)
    r_pre_z = safe_pearson(pre_attn, z_eff)
    r_slot0_z = safe_pearson(A[:, 0], z_eff)
    r_slot10_z = safe_pearson(A[:, 10], z_eff)
    print(f"\n(C) BOS-sink null hypothesis test at L{L_peak}:")
    print(f"    r(attn-to-BOS, z_eff)        = {r_bos_z:+.3f}")
    print(f"    r(attn-to-pre_context, z_eff) = {r_pre_z:+.3f}")
    print(f"    r(attn-to-slot 0, z_eff)     = {r_slot0_z:+.3f}")
    print(f"    r(attn-to-slot 10, z_eff)    = {r_slot10_z:+.3f}")
    # Partial: r(attn-slot0, z) | attn-BOS — does slot 0 correlation survive?
    r_slot0_z_pBOS = partial_pearson(A[:, 0], z_eff, bos_attn)
    r_slot10_z_pBOS = partial_pearson(A[:, 10], z_eff, bos_attn)
    print(f"    partial r(slot 0, z | BOS)    = {r_slot0_z_pBOS:+.3f}")
    print(f"    partial r(slot 10, z | BOS)   = {r_slot10_z_pBOS:+.3f}")

    # ------ (D) value-controlled correlation ------
    # For each slot, we need slot-k's actual value. We don't have it stored
    # per prompt; reconstruct from prompts.
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    trials = [json.loads(l) for l in in_path.open()]
    slot_values = np.full((n, K), np.nan, dtype=np.float64)
    for i, t in enumerate(trials):
        ctx = t.get("context_values", [])
        for k in range(min(K, len(ctx))):
            slot_values[i, k] = float(ctx[k])

    print(f"\n(D) Per-slot r(attn, z_eff) and partial r(attn, z_eff | slot-value):")
    print(f"    {'slot':>4}  {'r(z)':>8}  {'r(value, z)':>12}  {'r(z | value)':>14}")
    for k in range(K):
        r_az = safe_pearson(A[:, k], z_eff)
        r_vz = safe_pearson(slot_values[:, k], z_eff)
        r_az_pv = partial_pearson(A[:, k], z_eff, slot_values[:, k])
        print(f"    {k:>4}  {r_az:>+8.3f}  {r_vz:>+12.3f}  {r_az_pv:>+14.3f}")

    # Plot (D) as a 1×3 panel: raw r(attn, z), r(value, z), partial.
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), squeeze=False)
    raw = [safe_pearson(A[:, k], z_eff) for k in range(K)]
    val_z = [safe_pearson(slot_values[:, k], z_eff) for k in range(K)]
    partial = [partial_pearson(A[:, k], z_eff, slot_values[:, k])
               for k in range(K)]
    for j, (M, ttl) in enumerate([(raw, "raw r(attn-slot, z_eff)"),
                                    (val_z, "r(slot-value, z_eff)"),
                                    (partial,
                                     "partial r(attn-slot, z_eff | slot-value)")]):
        ax = axes[0, j]
        ax.bar(np.arange(K), M, color="#3b6ea5", edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_xlabel("context slot")
        ax.set_ylabel("Pearson r")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Phase 2G verify (D) — slot-value confound check at L{L_peak}\n"
                 f"{args.model} / {args.pair}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_d = FIG_DIR / f"p2g_verify_value_control_{args.model}_{args.pair}.png"
    fig.savefig(out_d, dpi=130, bbox_inches="tight")
    print(f"\n  -> {out_d}")
    plt.close(fig)


if __name__ == "__main__":
    main()
