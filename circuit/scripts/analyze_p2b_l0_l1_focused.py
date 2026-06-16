"""Phase 2B — focused L0/L1 analysis.

The residual-stream R²(z_eff) jumps from ~0.4 (L0) → ~0.92 (L1) at k=1, with
ALL of the increment at L1. This makes L0 and L1 the only layers where
attention-pattern analysis is informative for z-encoding.

For each (L ∈ {0, 1}, head h), report at every k:
  - attention mass on: BOS, last_context_value, target_value, suffix
  - Pearson r between (per-prompt attention mass on context-value vs target-value
    delta) and z_eff. A "comparator" head should attend more to the SMALLER
    side: high z_eff → context_value < target_value, head attends to context
    more (or less, depending on direction).
  - per-head DLA: < head_out, primal_z[L] @ W_O[head_slice] >.

We focus on k=1 because there's only one context anchor — attention from "is"
to either it or the target is unambiguous.

Usage:
  python scripts/analyze_p2b_l0_l1_focused.py --model gemma2-2b --pair height
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "p2_attn"
FIG_DIR = REPO / "figures"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    in_dir = RESULTS / args.model

    fig, axes = plt.subplots(len(args.layers), 4,
                              figsize=(16, 3.2 * len(args.layers)),
                              squeeze=False)

    summary = {}
    for li, L in enumerate(args.layers):
        for ki, k in enumerate([1, 4, 15]):
            path = in_dir / f"{args.pair}_k{k}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            n = d['ld'].shape[0]
            n_heads = d['attn_last'].shape[2]
            pad_off = d['pad_offset']
            seq_len = d['seq_len_unpad']
            target_pos = d['target_pos']
            context_pos = d['context_pos']
            z_eff = d['z_eff']
            ld = d['ld']

            # Aggregate per-prompt mass on:
            #   BOS = position pad_off (1 token)
            #   ctx_values = all context value tokens
            #   tgt_value = target value tokens
            #   "is" suffix = tokens after target value
            ctx_vmass = np.zeros((n, n_heads), dtype=np.float64)
            tgt_vmass = np.zeros((n, n_heads), dtype=np.float64)
            bos_mass = np.zeros((n, n_heads), dtype=np.float64)
            suffix_mass = np.zeros((n, n_heads), dtype=np.float64)
            # Per-prompt: mass on the LAST (closest-to-target) context value vs all earlier
            last_ctx_mass = np.zeros((n, n_heads), dtype=np.float64)

            A = d['attn_last'][:, L, :, :].astype(np.float64)  # (n, H, T)

            K_max = context_pos.shape[1]
            for i in range(n):
                po = int(pad_off[i])
                sl = int(seq_len[i])
                ts, te = int(target_pos[i, 0]), int(target_pos[i, 1])
                bos_mass[i] = A[i, :, po]  # all heads, single position
                # Target value tokens
                for p in range(ts, te):
                    tgt_vmass[i] += A[i, :, po + p]
                # All context values
                last_ctx_idx = -1
                for j in range(K_max):
                    cs, ce = int(context_pos[i, j, 0]), int(context_pos[i, j, 1])
                    if cs < 0:
                        continue
                    last_ctx_idx = j
                    for p in range(cs, ce):
                        ctx_vmass[i] += A[i, :, po + p]
                # Last context value separately
                if last_ctx_idx >= 0:
                    cs, ce = int(context_pos[i, last_ctx_idx, 0]), int(context_pos[i, last_ctx_idx, 1])
                    for p in range(cs, ce):
                        last_ctx_mass[i] += A[i, :, po + p]
                # Suffix
                for p in range(te, sl):
                    suffix_mass[i] += A[i, :, po + p]

            # Per-head Pearson r between (tgt - last_ctx) and z_eff:
            # if attention DIFFERENCE encodes direction of comparison, this should be nonzero.
            r_diff = np.zeros(n_heads, dtype=np.float64)
            for H in range(n_heads):
                diff = tgt_vmass[:, H] - last_ctx_mass[:, H]
                if diff.std() > 1e-9 and z_eff.std() > 1e-9 and np.isfinite(z_eff).all():
                    r_diff[H] = pearsonr(diff, z_eff)[0]

            ax = axes[li, ki]
            heads = np.arange(n_heads)
            width = 0.18
            ax.bar(heads - 1.5 * width, bos_mass.mean(0), width, label="BOS")
            ax.bar(heads - 0.5 * width, ctx_vmass.mean(0), width, label="ctx-val (all)")
            ax.bar(heads + 0.5 * width, tgt_vmass.mean(0), width, label="tgt-val")
            ax.bar(heads + 1.5 * width, suffix_mass.mean(0), width, label="suffix")
            ax.set_title(f"L{L}  |  k={k}", fontsize=10)
            ax.set_xlabel("head")
            ax.set_ylim(0, 1.0)
            if ki == 0:
                ax.set_ylabel("mean attention mass")
            if li == 0 and ki == 0:
                ax.legend(fontsize=7, loc="upper right")

            summary[f"L{L}_k{k}"] = {
                "bos": bos_mass.mean(0).tolist(),
                "ctx_val_all": ctx_vmass.mean(0).tolist(),
                "ctx_val_last": last_ctx_mass.mean(0).tolist(),
                "tgt_val": tgt_vmass.mean(0).tolist(),
                "suffix": suffix_mass.mean(0).tolist(),
                "r_attn_diff_z": r_diff.tolist(),
            }
        # 4th column: per-head Pearson r(attn_diff, z_eff) at the 3 k values
        ax = axes[li, 3]
        for k_test, color in zip([1, 4, 15], ["C0", "C1", "C2"]):
            key = f"L{L}_k{k_test}"
            if key in summary:
                ax.plot(np.arange(n_heads), summary[key]["r_attn_diff_z"],
                         "o-", label=f"k={k_test}", color=color)
        ax.axhline(0, color="black", linewidth=0.3, alpha=0.4)
        ax.set_title(f"L{L}  |  r(tgt_attn − last_ctx_attn,  z_eff)")
        ax.set_xlabel("head")
        ax.set_ylim(-1, 1)
        if li == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"P2B — L0/L1/L2 attention-mass breakdown ({args.model} | {args.pair})",
                  y=1.0)
    fig.tight_layout()
    out_fig = FIG_DIR / f"p2b_l0l1_focused_{args.model}_{args.pair}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"  -> {out_fig}")

    out_json = REPO / "results" / f"p2b_l0l1_focused_{args.model}_{args.pair}.json"
    with out_json.open("w") as f:
        json.dump(summary, f)
    print(f"  -> {out_json}")

    # Print key heads.
    print(f"\n=== L0/L1 candidate heads ({args.model} {args.pair}) ===")
    for L in args.layers:
        print(f"\n  Layer {L}:")
        for k in [1, 4, 15]:
            key = f"L{L}_k{k}"
            if key not in summary:
                continue
            s = summary[key]
            print(f"    k={k:>2}:")
            n_heads_print = len(s['bos'])
            for H in range(n_heads_print):
                ctx = s['ctx_val_all'][H]
                tgt = s['tgt_val'][H]
                bos = s['bos'][H]
                sfx = s['suffix'][H]
                r_dz = s['r_attn_diff_z'][H]
                print(f"      H{H}: bos={bos:.2f} ctx={ctx:.2f} tgt={tgt:.2f} "
                      f"sfx={sfx:.2f}   r(Δattn, z_eff)={r_dz:+.2f}")


if __name__ == "__main__":
    main()
