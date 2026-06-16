"""Phase 2B — analyze per-(layer, head) attention patterns by shot count.

For each prompt, last-token attention is partitioned into 4 buckets:
  bos        — the first real token (BOS).
  ctx_value  — the numeric value tokens within context items
               ("174" in "Person 1: 174 cm").
  ctx_struct — everything else in the context block (labels, units, "\n", etc.).
  tgt_value  — the numeric value tokens of the target line.
  suffix     — tokens after the target value (". This person is").

(Padding tokens get zero mass by design — attention mask kills them.)

Per (layer, head, k) we report:
  mean attention mass in each bucket (averaged over prompts)
  attention entropy across real tokens
  per-prompt attention "selectivity" on a single context item
    (max single-context-value mass / sum of context-value mass)

We also build the *contrast plot*: heads whose distribution shifts most between
k=1 and k=15 are the candidate aggregators or comparators.

Usage:
  python scripts/analyze_p2b_attention.py --model gemma2-2b --pair height
  python scripts/analyze_p2b_attention.py --model gemma2-9b --pair height
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "p2_attn"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def categorize_positions(d) -> dict[str, np.ndarray]:
    """Build, for each prompt, a 1D mask vector over the batched seq_len that
    assigns each token to a bucket. Returns dict: bucket -> mask array of
    shape (N, max_seq), bool.

    For each prompt we also compute, for each context value token, an
    independent 1-of-K assignment so we can measure selectivity later.
    """
    n, max_seq = d['attn_last'].shape[0], d['attn_last'].shape[-1]
    pad_off = d['pad_offset']        # (N,)
    seq_len = d['seq_len_unpad']     # (N,)
    target_pos = d['target_pos']     # (N, 2)
    context_pos = d['context_pos']   # (N, K_max, 2), -1 for unused

    bos_mask = np.zeros((n, max_seq), dtype=bool)
    ctx_value_mask = np.zeros((n, max_seq), dtype=bool)
    ctx_struct_mask = np.zeros((n, max_seq), dtype=bool)
    tgt_value_mask = np.zeros((n, max_seq), dtype=bool)
    tgt_struct_mask = np.zeros((n, max_seq), dtype=bool)
    suffix_mask = np.zeros((n, max_seq), dtype=bool)
    real_mask = np.zeros((n, max_seq), dtype=bool)

    # per-prompt list of context value position indices in the batched encoding.
    # padded so we can vectorize selectivity checks.
    K_max = context_pos.shape[1]
    ctx_value_pos_padded = np.full((n, K_max, 16), -1, dtype=np.int32)
    # Each context value can span up to 16 tokens (rare; usually 1-2).

    for i in range(n):
        po = int(pad_off[i])
        sl = int(seq_len[i])
        # All real positions
        real_mask[i, po:po + sl] = True
        # BOS = first real token
        bos_mask[i, po] = True
        # Target value tokens
        ts, te = int(target_pos[i, 0]), int(target_pos[i, 1])
        for p in range(ts, te):
            tgt_value_mask[i, po + p] = True
        # Target structural: tokens between (last context) and target_value
        # Last context end:
        last_ctx_end = 1  # right after BOS
        for j in range(K_max):
            cs, ce = int(context_pos[i, j, 0]), int(context_pos[i, j, 1])
            if cs < 0:
                continue
            for p in range(cs, ce):
                ctx_value_mask[i, po + p] = True
                # record for selectivity calc
                offset_in_value = p - cs
                if offset_in_value < 16:
                    ctx_value_pos_padded[i, j, offset_in_value] = po + p
            last_ctx_end = max(last_ctx_end, ce)
        # context structural: BOS+1 .. last_ctx_end exclusive, minus ctx_value
        for p in range(1, last_ctx_end):
            if not ctx_value_mask[i, po + p]:
                ctx_struct_mask[i, po + p] = True
        # target structural: last_ctx_end .. ts (exclusive), minus tgt_value
        for p in range(last_ctx_end, ts):
            if not tgt_value_mask[i, po + p]:
                tgt_struct_mask[i, po + p] = True
        # suffix: te .. sl
        for p in range(te, sl):
            suffix_mask[i, po + p] = True

    return {
        "bos": bos_mask,
        "ctx_value": ctx_value_mask,
        "ctx_struct": ctx_struct_mask,
        "tgt_value": tgt_value_mask,
        "tgt_struct": tgt_struct_mask,
        "suffix": suffix_mask,
        "real": real_mask,
        "ctx_value_pos_padded": ctx_value_pos_padded,
    }


def per_head_mass(d, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Returns dict: bucket -> array (N, n_layers, n_heads) of summed attention."""
    A = d['attn_last'].astype(np.float32)  # (N, L, H, T)
    out = {}
    for name, m in masks.items():
        if name in ("real", "ctx_value_pos_padded"):
            continue
        # broadcast m: (N, T) -> (N, 1, 1, T)
        out[name] = (A * m[:, None, None, :]).sum(-1)  # (N, L, H)
    return out


def per_head_entropy(d, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Entropy of attention over real tokens, per (N, L, H)."""
    A = d['attn_last'].astype(np.float32)
    real = masks['real'][:, None, None, :]  # (N, 1, 1, T)
    A_real = A * real
    Z = A_real.sum(-1, keepdims=True)
    P = A_real / np.clip(Z, 1e-9, None)
    P_log = np.log(np.clip(P, 1e-12, None))
    H = -(P * P_log).sum(-1)  # (N, L, H)
    return H


def per_head_top_ctx_concentration(d, masks: dict[str, np.ndarray]) -> np.ndarray:
    """For each prompt, measure how concentrated context-value attention is
    on a single context item: max(per-item mass) / sum(all-item mass).
    Returns (N, L, H). High means selective, low means dispersed (aggregator).
    """
    A = d['attn_last'].astype(np.float32)
    pos_padded = masks['ctx_value_pos_padded']  # (N, K_max, 16)
    n, n_layers, n_heads, max_seq = A.shape
    K_max = pos_padded.shape[1]

    # Sum attention over each context item's value tokens.
    item_mass = np.zeros((n, n_layers, n_heads, K_max), dtype=np.float32)
    for k in range(K_max):
        for tslot in range(pos_padded.shape[2]):
            pos = pos_padded[:, k, tslot]   # (N,) — int positions in batched coords, -1 if unused
            valid = pos >= 0
            if not valid.any():
                continue
            # For each prompt where this slot exists, gather A[i, :, :, pos[i]]
            idx_n = np.where(valid)[0]
            idx_p = pos[idx_n]
            item_mass[idx_n, :, :, k] += A[idx_n, :, :, idx_p]
    total = item_mass.sum(-1)  # (N, L, H)
    # Top-1 concentration:
    top1 = item_mass.max(-1)  # (N, L, H)
    conc = np.where(total > 1e-6, top1 / total, np.nan)
    return conc


def summarize_one(d, masks):
    """Mean across prompts for each head metric."""
    mass = per_head_mass(d, masks)        # bucket -> (N, L, H)
    H = per_head_entropy(d, masks)        # (N, L, H)
    conc = per_head_top_ctx_concentration(d, masks)  # (N, L, H)
    out = {b: m.mean(0) for b, m in mass.items()}    # (L, H)
    out['entropy'] = H.mean(0)
    out['concentration'] = np.nanmean(conc, axis=0)
    out['n_prompts'] = int(d['ld'].shape[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 15])
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    in_dir = RESULTS / args.model
    summaries: dict[int, dict] = {}
    for k in args.k:
        path = in_dir / f"{args.pair}_k{k}.npz"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        print(f"[load] {path}")
        d = np.load(path, allow_pickle=True)
        masks = categorize_positions(d)
        s = summarize_one(d, masks)
        summaries[k] = s

    if not summaries:
        return
    n_layers, n_heads = next(iter(summaries.values()))['bos'].shape

    # Print the headline buckets for k=1 vs k=15 — find heads whose pattern shifts.
    print(f"\n=== Per-head context vs target attention mass ({args.model} {args.pair}) ===")
    print(f"(L, H) listed when |Δ ctx_value mass| ≥ 0.10 between k=1 and k=15.\n")
    if 1 in summaries and 15 in summaries:
        s1 = summaries[1]
        s15 = summaries[15]
        delta_ctx = s15['ctx_value'] - s1['ctx_value']
        delta_tgt = s15['tgt_value'] - s1['tgt_value']
        delta_bos = s15['bos'] - s1['bos']
        delta_ent = s15['entropy'] - s1['entropy']
        rows = []
        for L in range(n_layers):
            for H in range(n_heads):
                if abs(delta_ctx[L, H]) >= 0.10:
                    rows.append((L, H, s1['ctx_value'][L, H], s15['ctx_value'][L, H],
                                 s1['tgt_value'][L, H], s15['tgt_value'][L, H],
                                 s1['bos'][L, H], s15['bos'][L, H],
                                 s1['entropy'][L, H], s15['entropy'][L, H]))
        rows.sort(key=lambda r: -abs(r[3] - r[2]))
        print(f"{'L':>3} {'H':>2}  {'ctx@1→15':>14}  {'tgt@1→15':>14}  {'bos@1→15':>14}  {'H@1→15':>14}")
        for r in rows[:24]:
            print(f"{r[0]:>3} {r[1]:>2}  {r[2]:.2f} → {r[3]:.2f}    "
                  f"{r[4]:.2f} → {r[5]:.2f}    "
                  f"{r[6]:.2f} → {r[7]:.2f}    "
                  f"{r[8]:.2f} → {r[9]:.2f}")

    # --- figure: 2x2 grid heatmap (n_layers × n_heads) of bucket masses at each k ---
    ks = sorted(summaries.keys(), key=int)
    buckets = ['ctx_value', 'tgt_value', 'bos', 'entropy']
    fig, axes = plt.subplots(len(buckets), len(ks),
                             figsize=(3.4 * len(ks), 2.6 * len(buckets)),
                             squeeze=False)
    for bi, b in enumerate(buckets):
        for ki, k in enumerate(ks):
            ax = axes[bi, ki]
            data = summaries[k][b]
            if b == 'entropy':
                vmin, vmax = 0, np.log(20.0)
                cmap = 'viridis'
            else:
                vmin, vmax = 0, 1.0
                cmap = 'plasma'
            im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{b}  |  k={k}", fontsize=9)
            if ki == 0:
                ax.set_ylabel("layer")
            if bi == len(buckets) - 1:
                ax.set_xlabel("head")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"P2B — per-head attention by k  ({args.model} | {args.pair})", y=1.0)
    fig.tight_layout()
    out_fig = FIG_DIR / f"p2b_attn_taxonomy_{args.model}_{args.pair}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"  -> {out_fig}")

    # --- figure: top-shifters as line plots over k ---
    if 1 in summaries and 15 in summaries:
        # rank heads by |Δ ctx_value|
        delta = summaries[15]['ctx_value'] - summaries[1]['ctx_value']
        flat_idx = np.argsort(-np.abs(delta).flatten())[:8]
        top_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx]
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), squeeze=False)
        for ai, (L, H) in enumerate(top_heads):
            ax = axes[ai // 4, ai % 4]
            for b, color in [('ctx_value', 'C0'), ('tgt_value', 'C1'),
                              ('bos', 'C2'), ('suffix', 'C3')]:
                vals = [summaries[k][b][L, H] for k in ks]
                ax.plot(ks, vals, "o-", color=color, label=b)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_ylim(-0.02, 1.0)
            ax.set_title(f"L{L} H{H}")
            ax.set_xlabel("k")
            if ai == 0:
                ax.set_ylabel("attention mass")
                ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"P2B — top |Δ ctx mass| heads  ({args.model} | {args.pair})", y=1.0)
        fig.tight_layout()
        out_fig2 = FIG_DIR / f"p2b_topshifters_{args.model}_{args.pair}.png"
        fig.savefig(out_fig2, dpi=140, bbox_inches="tight")
        print(f"  -> {out_fig2}")

    out_json = REPO / "results" / f"p2b_summary_{args.model}_{args.pair}.json"
    json_safe = {
        str(k): {b: v.tolist() if isinstance(v, np.ndarray) else v
                 for b, v in s.items()}
        for k, s in summaries.items()
    }
    with out_json.open("w") as f:
        json.dump(json_safe, f)
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
