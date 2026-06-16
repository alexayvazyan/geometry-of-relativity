"""Phase 2F — what tokens does the model attend to when producing the
relativistic answer?

Three views on the per-head attention pattern at the last query position
(decoded from `attn_last` in the existing p2_attn NPZs):

1. Bucket decomposition: split last-token attention into four roles per
   prompt — {pad, pre_context, context_values, context_scaffold,
   target_value, target_scaffold} — then average across prompts to get a
   layer × head × bucket heatmap.

2. z-tracking through attention: for each (layer, head), correlate
   attn_to_context_values with z_eff. Heatmap of layer × head shows where
   the attention pattern itself encodes z (vs. encoding via value-mix only).

3. Primary-head deep dive: for the canonical comparator heads found in
   Phase 2B (2B L1H6, 9B L1H11), plot attention to each of the 15 context
   value tokens, split into low / mid / high z_eff bins. Reveals whether
   the head reads context uniformly or selectively as z changes.

Usage:
  python3 scripts/p2f_attn_circuit.py --model gemma2-2b --pair height
  python3 scripts/p2f_attn_circuit.py --model gemma2-9b --pair height
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
RES_DIR = REPO / "results"

PRIMARY = {  # (layer, head) heads identified in Phase 2B as comparator
    "gemma2-2b": (1, 6),
    "gemma2-9b": (1, 11),
}


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    xm = x[mask] - x[mask].mean(); ym = y[mask] - y[mask].mean()
    den = (xm.std() * ym.std() * len(xm))
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def build_buckets(target_pos, context_pos, pad_offset, seq_len_unpad, max_seq):
    """For each prompt return a dict bucket_name -> 1D mask over [0, max_seq).

    Positions in target_pos/context_pos are UNPADDED token indices; attn_last
    is left-padded by pad_offset, so batched_idx = pad_offset + unpadded_idx.
    """
    n = len(seq_len_unpad)
    K = context_pos.shape[1]
    buckets = {
        "pad":             np.zeros((n, max_seq), dtype=bool),
        "pre_context":     np.zeros((n, max_seq), dtype=bool),
        "ctx_value":       np.zeros((n, max_seq), dtype=bool),
        "ctx_scaffold":    np.zeros((n, max_seq), dtype=bool),
        "tgt_value":       np.zeros((n, max_seq), dtype=bool),
        "tgt_scaffold":    np.zeros((n, max_seq), dtype=bool),
    }

    for i in range(n):
        po = int(pad_offset[i])
        L = int(seq_len_unpad[i])  # unpadded length

        # PAD: 0 .. po
        buckets["pad"][i, :po] = True

        # First context start (in unpadded coords) — anything before it is
        # "pre_context" (BOS, etc.).
        first_ctx_start_un = int(context_pos[i, 0, 0])
        first_ctx_start_b = po + first_ctx_start_un
        buckets["pre_context"][i, po:first_ctx_start_b] = True

        # Mark each context value range, in batched coords.
        last_ctx_end_un = first_ctx_start_un  # will track furthest end
        for k in range(K):
            s_un, e_un = int(context_pos[i, k, 0]), int(context_pos[i, k, 1])
            if s_un < 0:  # unused slot
                continue
            buckets["ctx_value"][i, po + s_un:po + e_un] = True
            last_ctx_end_un = max(last_ctx_end_un, e_un)

        # Target range (unpadded → batched)
        tgt_s_un, tgt_e_un = int(target_pos[i, 0]), int(target_pos[i, 1])
        buckets["tgt_value"][i, po + tgt_s_un:po + tgt_e_un] = True

        # ctx_scaffold = between first_ctx_start and tgt_s, MINUS ctx_value
        ctx_block_b_lo = po + first_ctx_start_un
        ctx_block_b_hi = po + tgt_s_un  # everything before target value
        scaffold_mask = np.zeros(max_seq, dtype=bool)
        scaffold_mask[ctx_block_b_lo:ctx_block_b_hi] = True
        scaffold_mask &= ~buckets["ctx_value"][i]
        buckets["ctx_scaffold"][i] = scaffold_mask

        # tgt_scaffold = from tgt_e to seq_end (the ". This person is" trailer)
        unpad_end_b = po + L
        buckets["tgt_scaffold"][i, po + tgt_e_un:unpad_end_b] = True

    return buckets


def attn_decomposition(attn_last, buckets):
    """Returns dict bucket -> array of shape (n, n_layers, n_heads) — fraction
    of last-token attention going to each bucket.
    """
    n, n_layers, n_heads, T = attn_last.shape
    out = {}
    for name, mask in buckets.items():
        # mask: (n, T); attn: (n, layers, heads, T). Sum over T where mask.
        m = mask[:, None, None, :]  # (n, 1, 1, T)
        s = (attn_last.astype(np.float32) * m).sum(-1)  # (n, layers, heads)
        out[name] = s
    return out


def per_context_attn(attn_last, context_pos, pad_offset):
    """Per-prompt attention to each of the K context value tokens
    (summed over the value's token span). Shape: (n, n_layers, n_heads, K).
    """
    n, n_layers, n_heads, T = attn_last.shape
    K = context_pos.shape[1]
    out = np.zeros((n, n_layers, n_heads, K), dtype=np.float32)
    for i in range(n):
        po = int(pad_offset[i])
        for k in range(K):
            s_un, e_un = int(context_pos[i, k, 0]), int(context_pos[i, k, 1])
            if s_un < 0:
                out[i, :, :, k] = np.nan
                continue
            out[i, :, :, k] = attn_last[i, :, :,
                                        po + s_un:po + e_un].astype(np.float32).sum(-1)
    return out


def plot_bucket_heatmap(per_bucket_mean, n_layers, n_heads, model, pair,
                         out_path):
    """6 small heatmaps: layer (rows) × head (cols), one per bucket, mean
    attention. Color: 0..1.
    """
    bucket_order = ["pad", "pre_context", "ctx_value", "ctx_scaffold",
                    "tgt_value", "tgt_scaffold"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    for i, name in enumerate(bucket_order):
        ax = axes[i // 3, i % 3]
        M = per_bucket_mean[name]  # (layers, heads)
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.3, float(M.max())))
        ax.set_title(f"{name}  (max={M.max():.2f})", fontsize=10)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 8)))
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Phase 2F — last-token attention bucket decomposition  "
                 f"|  {model}  |  {pair} k=15", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_z_attn_corr(corr_LH, n_layers, n_heads, primary, model, pair, out_path):
    """Heatmap of r(attn_to_ctx_values, z_eff) per (layer, head). Highlight
    primary head."""
    fig, ax = plt.subplots(figsize=(max(7, n_heads * 0.7), max(5, n_layers * 0.25)))
    M = corr_LH
    vmax = np.nanmax(np.abs(M))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.5
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 12)))
    ax.set_title(f"r(attn-to-ctx-values, z_eff) per head  |  {model}  |  {pair} k=15\n"
                 f"(red = head attends MORE to ctx when z_eff is HIGH; "
                 f"primary head circled)", fontsize=11)
    Lp, Hp = primary
    ax.add_patch(plt.Rectangle((Hp - 0.5, Lp - 0.5), 1, 1,
                                fill=False, edgecolor="lime", linewidth=2.5))
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def plot_primary_deepdive(per_ctx_attn, z_eff, primary, model, pair,
                           bucket_attn_per_prompt, out_path):
    """For the primary head only, three panels:
    A. Mean attention to each context position, overall (15 bars).
    B. Mean attention to each context position by z-tertile (3 lines).
    C. Scatter: total attn to ctx_values vs z_eff (one dot per prompt) +
       attn to tgt_value scatter for comparison.
    """
    L, H = primary
    a = per_ctx_attn[:, L, H, :]  # (n, K)
    n, K = a.shape
    z = z_eff
    finite = np.isfinite(z) & np.isfinite(a).all(axis=1)
    a_f = a[finite]; z_f = z[finite]

    # Tertile splits on z
    qs = np.quantile(z_f, [0.33, 0.67])
    lo = z_f < qs[0]; hi = z_f >= qs[1]; mid = ~lo & ~hi

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ks = np.arange(1, K + 1)

    # A. Overall mean
    ax = axes[0]
    ax.bar(ks, a_f.mean(0), color="#3b6ea5", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("context slot (1..15)"); ax.set_ylabel("mean attention")
    ax.set_title(f"A. Overall mean attention to each context value\n"
                 f"head L{L}H{H}  ({model})", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # B. By z-tertile
    ax = axes[1]
    for label, mask, color in [("z low (≤33%)", lo, "tab:blue"),
                                ("z mid",         mid, "tab:gray"),
                                ("z high (≥67%)",hi, "tab:red")]:
        ax.plot(ks, a_f[mask].mean(0), "-o", color=color,
                label=f"{label} (n={mask.sum()})", linewidth=1.6, markersize=5)
    ax.set_xlabel("context slot (1..15)"); ax.set_ylabel("mean attention")
    ax.set_title("B. Mean attention by z_eff tertile", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # C. Scatter — three quantities vs z_eff
    ax = axes[2]
    ctx_tot_per_prompt = bucket_attn_per_prompt["ctx_value"][:, L, H]
    tgt_per_prompt = bucket_attn_per_prompt["tgt_value"][:, L, H]
    finite2 = np.isfinite(z) & np.isfinite(ctx_tot_per_prompt)
    ax.scatter(z[finite2], ctx_tot_per_prompt[finite2], s=8, alpha=0.4,
               color="tab:blue", label="∑ ctx_value")
    ax.scatter(z[finite2], tgt_per_prompt[finite2], s=8, alpha=0.4,
               color="tab:orange", label="tgt_value")
    r_ctx = safe_pearson(z[finite2], ctx_tot_per_prompt[finite2])
    r_tgt = safe_pearson(z[finite2], tgt_per_prompt[finite2])
    ax.set_xlabel("z_eff"); ax.set_ylabel("attention")
    ax.set_title(f"C. Total attention vs z_eff\n"
                 f"r(ctx)={r_ctx:+.3f}  r(tgt)={r_tgt:+.3f}", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle(f"Phase 2F primary-head deep dive  |  {model}  |  {pair} k=15  "
                 f"|  L{L}H{H}", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  -> {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(PRIMARY))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--max-prompts", type=int, default=None,
                    help="optional subsample for faster runs")
    args = p.parse_args()

    npz = RES_DIR / "p2_attn" / args.model / f"{args.pair}_k{args.k}.npz"
    print(f"[p2f] loading {npz}")
    d = np.load(npz, allow_pickle=True)
    n, n_layers, n_heads, T = d["attn_last"].shape
    print(f"[p2f] {args.model}/{args.pair} k={args.k}: n={n} layers={n_layers} "
          f"heads={n_heads} T={T}")

    if args.max_prompts and n > args.max_prompts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=args.max_prompts, replace=False)
    else:
        idx = np.arange(n)

    attn = d["attn_last"][idx]                # (n, L, H, T) fp16
    target_pos = d["target_pos"][idx]
    context_pos = d["context_pos"][idx]
    pad_offset = d["pad_offset"][idx]
    seq_len_unpad = d["seq_len_unpad"][idx]
    z_eff = d["z_eff"][idx].astype(np.float64)

    # Buckets (per prompt boolean masks over T).
    buckets = build_buckets(target_pos, context_pos, pad_offset,
                             seq_len_unpad, T)
    # Sanity: rows should sum to ≈1 (modulo float).
    sum_check = sum(b.sum(-1) for b in buckets.values())
    print(f"[p2f] bucket coverage / token: min={sum_check.min()} "
          f"max={sum_check.max()} (should be == seq_len_unpad)")

    bucket_attn = attn_decomposition(attn, buckets)
    # Mean across prompts for the heatmap.
    per_bucket_mean = {k: v.mean(0) for k, v in bucket_attn.items()}

    # r(attn-to-ctx-values, z_eff) per head
    ctx_attn_per_prompt = bucket_attn["ctx_value"]   # (n, L, H)
    corr_LH = np.zeros((n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        for H in range(n_heads):
            corr_LH[L, H] = safe_pearson(z_eff, ctx_attn_per_prompt[:, L, H])

    # Per-context attention for primary head deep-dive.
    per_ctx = per_context_attn(attn, context_pos, pad_offset)  # (n, L, H, K)

    primary = PRIMARY[args.model]

    # Save numeric summary.
    out_json = RES_DIR / f"p2f_attn_circuit_{args.model}_{args.pair}_k{args.k}.json"
    summary = {
        "model": args.model, "pair": args.pair, "k": args.k,
        "n_prompts": int(len(idx)),
        "primary_head": primary,
        "bucket_mean_per_LH": {k: v.tolist() for k, v in per_bucket_mean.items()},
        "r_ctx_attn_z": corr_LH.tolist(),
    }
    with out_json.open("w") as f:
        json.dump(summary, f)
    print(f"  -> {out_json}")

    # Plots.
    FIG_DIR.mkdir(exist_ok=True)
    plot_bucket_heatmap(per_bucket_mean, n_layers, n_heads, args.model,
                         args.pair,
                         FIG_DIR / f"p2f_bucket_{args.model}_{args.pair}_k{args.k}.png")
    plot_z_attn_corr(corr_LH, n_layers, n_heads, primary, args.model, args.pair,
                      FIG_DIR / f"p2f_z_corr_{args.model}_{args.pair}_k{args.k}.png")
    plot_primary_deepdive(
        per_ctx, z_eff, primary, args.model, args.pair, bucket_attn,
        FIG_DIR / f"p2f_primary_{args.model}_{args.pair}_k{args.k}.png",
    )

    # Also print top-5 (layer, head) cells by |r(attn,z)|.
    flat = [((L, H), corr_LH[L, H]) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: abs(t[1]), reverse=True)
    print("\n[p2f] top-10 heads by |r(attn-to-ctx-values, z_eff)|:")
    for (L, H), r in flat[:10]:
        print(f"   L{L:>2}H{H:>2}  r={r:+.3f}  "
              f"mean_ctx_attn={per_bucket_mean['ctx_value'][L, H]:.3f}")


if __name__ == "__main__":
    main()
