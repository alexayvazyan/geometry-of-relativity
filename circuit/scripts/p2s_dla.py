"""Phase 2S — Direct Logit Attribution (DLA) on trio + cluster.

For each (L, h) in the cells of interest, compute on each prompt:

  DLA(L, h) = (attn_h @ V_h) @ W_O_h @ (W_U[tall] - W_U[short])

where attn_h @ V_h is the head's last-token output before o_proj, W_O_h is
the head's slice of o_proj.weight, and W_U is the unembedding (lm_head).

DLA is the head's *direct* contribution to logit(tall) − logit(short),
ignoring downstream nonlinearities (standard mech-interp approximation).

Also compute per-position DLA for the causal trio (L16H4, L17H7, L14H2):
how much of each head's z-write comes from each token position. This
directly tests the BOS-sink-as-z-source hypothesis.

Output:
  results/p2s_dla_gemma2-2b.json    (per-cell aggregate stats)
  results/p2s_dla_gemma2-2b.npz     (per-prompt per-cell DLA + per-position
                                      DLA for trio)
  figures/p2s_dla_gemma2-2b.png     (per-cell DLA distribution + DLA-vs-z)
  figures/p2s_dla_position_trio.png (per-position DLA for trio aggregated)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SHORT = "gemma2-2b"
MODEL = "google/gemma-2-2b"
FEATURE = "height"
K = 15

CELLS: list[tuple[int, int, str]] = [
    (16, 4, "trio (hub)"),
    (17, 7, "trio (cooperator)"),
    (14, 2, "trio (backup)"),
    (15, 7, "H7-column"),
    (14, 7, "H7-column"),
    (12, 6, "cos·σ candidate"),
    (12, 4, "cos·σ candidate"),
    (13, 2, "cos·σ candidate"),
    (11, 2, "cos·σ candidate"),
    (13, 6, "cos·σ candidate"),
    (16, 2, "high-||Δa|| control"),
    (20, 2, "high-||Δa|| control"),
]
GROUP_COLORS = {
    "trio (hub)": "#d62728",
    "trio (cooperator)": "#ff7f0e",
    "trio (backup)": "#bcbd22",
    "H7-column": "#9467bd",
    "cos·σ candidate": "#1f77b4",
    "high-||Δa|| control": "#7f7f7f",
}
TRIO_CELLS = [(16, 4), (17, 7), (14, 2)]


def find_runs(cats: list[str]) -> list[tuple[int, int]]:
    runs = []
    in_run, start = False, 0
    for i, c in enumerate(cats):
        if c == "context_num":
            if not in_run:
                start = i; in_run = True
        else:
            if in_run:
                runs.append((start, i)); in_run = False
    if in_run: runs.append((start, len(cats)))
    return runs


def main() -> None:
    # Reuse the same prompts that Phase 2R captured (from JSON's prompt_idx)
    p2r_json = REPO / "results" / f"p2r_attn_{SHORT}.json"
    p2r = json.loads(p2r_json.read_text())
    rows_all = [json.loads(l) for l in
                 (REPO / "data/p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl").open()]
    prompt_idxs = [p["prompt_idx"] for p in p2r["prompts"]]
    rows = [rows_all[i] for i in prompt_idxs]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)
    print(f"using {n} prompts (from Phase 2R)")

    # Model
    print(f"\nloading {MODEL}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # required for output_attentions
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Locate decoder layers
    layers = model.model.layers
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers} n_heads={n_heads} head_dim={head_dim} hidden={hidden}")

    # Token IDs and unembed direction tall - short
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]
    print(f"  high_id={high_id} ('{high_w}')  low_id={low_id} ('{low_w}')")

    # W_U: lm_head.weight. Gemma 2 ties weights, so it equals embedding.weight.
    W_U = model.lm_head.weight.detach()  # (vocab, hidden), bfloat16
    tall_dir = (W_U[high_id] - W_U[low_id]).float().cpu().numpy()  # (hidden,)
    print(f"  ||tall_dir||={np.linalg.norm(tall_dir):.2f}")

    # Pre-extract per-layer W_O (o_proj.weight): shape (hidden, n_heads*head_dim)
    # nn.Linear stores weight transposed: y = x @ W.T, so .weight has shape (out, in).
    W_O_layers = []
    for L in range(n_layers):
        W_O = layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
        W_O_layers.append(W_O)  # (hidden, n_heads*head_dim)
    # For head h, slice = W_O[:, h*head_dim:(h+1)*head_dim]  shape (hidden, head_dim)
    # head_contribution = (attn_h @ V_h) @ slice.T  shape (hidden,)

    # Capture o_proj input (= attn_h @ V_h concatenated across heads) at last token
    captured = {}
    def make_cap(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()  # (1, T, n_heads*head_dim)
        return hook
    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_cap(L))
                for L in range(n_layers)]

    # Also need V values + attention weights to decompose per-position DLA on trio.
    # Hook v_proj output for trio's layers (subset to keep memory small).
    trio_layers = sorted({L for L, _ in TRIO_CELLS})
    print(f"  trio layers (for per-position capture): {trio_layers}")
    captured_v = {}      # {L: (1, T, n_kv_heads * head_dim)}  -- GQA, share KV across heads
    captured_attn = {}   # {L: (1, n_heads, T, T)}
    def make_v_cap(L):
        def hook(module, args_, output):
            captured_v[L] = output.detach().float().cpu().numpy()
        return hook
    v_handles = [layers[L].self_attn.v_proj.register_forward_hook(make_v_cap(L))
                  for L in trio_layers]

    # Per-prompt per-cell DLA
    dla_grid = np.zeros((n, len(CELLS)), dtype=np.float32)
    # Per-prompt per-trio per-position DLA aggregated by category
    cat_keys = ["context_num", "target_num", "unit", "person_label",
                 "colon", "newline", "tail", "bos", "other"]
    trio_per_cat_dla = np.zeros((n, len(TRIO_CELLS), len(cat_keys)),
                                  dtype=np.float32)

    # Also baseline LD for sanity check
    ld_baseline = np.zeros(n, dtype=np.float32)

    # Use Phase 2R cats per prompt (already computed)
    cats_per_prompt = [p["cats"] for p in p2r["prompts"]]

    # Number of KV heads (Gemma 2 uses GQA)
    n_kv_heads = layers[0].self_attn.config.num_key_value_heads
    n_kv_groups = n_heads // n_kv_heads
    print(f"  GQA: n_heads={n_heads}, n_kv_heads={n_kv_heads}, group_size={n_kv_groups}")

    print(f"\n[forward + DLA] {n} prompts...")
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, output_attentions=True, use_cache=False)
            T = inp.input_ids.shape[1]
            logits = out.logits[0, -1].float()
            ld_baseline[i] = float(logits[high_id] - logits[low_id])

            # Standard DLA per cell
            for ci, (L, h, _) in enumerate(CELLS):
                head_v_last = captured[L][0, -1, h * head_dim:(h + 1) * head_dim]
                W_O_slice = W_O_layers[L][:, h * head_dim:(h + 1) * head_dim]
                contribution = head_v_last @ W_O_slice.T
                dla_grid[i, ci] = float(np.dot(contribution, tall_dir))

            # Per-position DLA for trio, aggregated by token category
            cats_arr = np.array(cats_per_prompt[i])
            for ti, (L, h) in enumerate(TRIO_CELLS):
                # attn weights from last token: shape (T,)
                attn_lh = out.attentions[L][0, h, -1, :].float().cpu().numpy()
                # V for head h: in GQA, V has n_kv_heads channels.
                # head h reads from kv_head h // n_kv_groups
                kv_h = h // n_kv_groups
                V_full = captured_v[L][0]  # (T, n_kv_heads * head_dim)
                V_h = V_full[:, kv_h * head_dim:(kv_h + 1) * head_dim]  # (T, head_dim)
                # per-position contribution to last-token residual:
                # contribution_t = attn_lh[t] * V_h[t] @ W_O_slice.T  (hidden,)
                W_O_slice = W_O_layers[L][:, h * head_dim:(h + 1) * head_dim]
                # weighted V then project, but since dot product is linear:
                # per_pos_contrib_residual[t] = attn_lh[t] * V_h[t] @ W_O_slice.T
                # per_pos_dla[t] = (per_pos_contrib_residual[t]) · tall_dir
                # Equivalently: per_pos_dla[t] = attn_lh[t] * (V_h[t] @ W_O_slice.T) · tall_dir
                # Even more efficient: project tall_dir back through W_O_slice first
                # contribution_dim = tall_dir @ W_O_slice  shape (head_dim,)
                contribution_dim = tall_dir @ W_O_slice  # (head_dim,)
                per_pos_dla = attn_lh * (V_h @ contribution_dim)  # (T,)
                # Aggregate by category
                for ki, k in enumerate(cat_keys):
                    mask = cats_arr == k
                    trio_per_cat_dla[i, ti, ki] = float(per_pos_dla[mask].sum())

            if (i + 1) % 10 == 0:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                print(f"  {i+1}/{n}  {rate:.1f} p/s", flush=True)
    for h_ in handles:
        h_.remove()
    for h_ in v_handles:
        h_.remove()

    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}")

    # === Per-cell aggregate stats ===
    print("\n=== per-cell DLA stats ===")
    print(f"{'cell':<8s} {'group':<22s}  {'mean':>8s}  {'std':>7s}  "
           f"{'corr(DLA,z)':>12s}  {'σ_z=mean+ρ·sd':>14s}")
    cell_stats = []
    for ci, (L, h, g) in enumerate(CELLS):
        m = float(dla_grid[:, ci].mean())
        s = float(dla_grid[:, ci].std(ddof=1))
        rho = float(np.corrcoef(dla_grid[:, ci], z_arr)[0, 1])
        cell_stats.append({"L": L, "h": h, "group": g, "mean_dla": m,
                            "std_dla": s, "corr_dla_z": rho})
        print(f"L{L:>2d}H{h}    {g:<22s}  {m:+8.3f}  {s:7.3f}  {rho:+12.3f}  "
              f"{m + rho * s:+14.3f}")

    # Save NPZ + JSON
    out_npz = REPO / "results" / f"p2s_dla_{SHORT}.npz"
    np.savez(out_npz, dla_grid=dla_grid, ld_baseline=ld_baseline, zs=z_arr,
              tall_dir=tall_dir, high_id=high_id, low_id=low_id,
              trio_per_cat_dla=trio_per_cat_dla,
              trio_cells=np.array([[L, h] for L, h in TRIO_CELLS]),
              cat_keys=np.array(cat_keys))
    print(f"\nwrote {out_npz}")

    out_json = REPO / "results" / f"p2s_dla_{SHORT}.json"
    out_json.write_text(json.dumps({
        "model": MODEL, "n_prompts": int(n),
        "high_word": high_w, "low_word": low_w,
        "baseline_r_LD_z": base_r,
        "cells": cell_stats,
    }, indent=2))
    print(f"wrote {out_json}")

    # === VIZ 1: per-cell DLA distribution + DLA-vs-z scatter ===
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False, sharey=False)
    for ci, (L, h, g) in enumerate(CELLS):
        ax = axes.flat[ci]
        ax.scatter(z_arr, dla_grid[:, ci], s=18, alpha=0.7,
                    color=GROUP_COLORS[g])
        # Line of best fit
        m = np.polyfit(z_arr, dla_grid[:, ci], 1)
        zline = np.linspace(z_arr.min(), z_arr.max(), 50)
        ax.plot(zline, np.polyval(m, zline), color="black", lw=1, ls="--",
                 alpha=0.5)
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        rho = float(np.corrcoef(dla_grid[:, ci], z_arr)[0, 1])
        ax.set_title(f"L{L}H{h} — {g}\nρ(DLA,z)={rho:+.3f}, mean={cell_stats[ci]['mean_dla']:+.2f}",
                      fontsize=9, color=GROUP_COLORS[g])
        ax.grid(alpha=0.3)
        if ci % 4 == 0:
            ax.set_ylabel("DLA = head→logit(tall − short)", fontsize=9)
        if ci >= 8:
            ax.set_xlabel("z", fontsize=9)
    fig.suptitle(f"{SHORT} — Direct Logit Attribution per cell\n"
                  f"(positive DLA → head pushes 'tall', negative → 'short';\n"
                  f" ρ(DLA, z) = head's z-encoding strength)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = REPO / "figures" / f"p2s_dla_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # === VIZ: per-category DLA decomposition for trio ===
    # For each trio cell, show how much of its DLA-z signal comes from each
    # token category. We use ρ(per-category DLA, z) as the metric: a category
    # contributes to z-write if its DLA-aggregate per prompt correlates with z.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ti, (L, h) in enumerate(TRIO_CELLS):
        ax = axes[ti]
        # For each category, compute corr(per-cat DLA, z) and the absolute
        # mean contribution (scale).
        cat_rhos = []
        cat_means = []
        for ki, k in enumerate(cat_keys):
            x = trio_per_cat_dla[:, ti, ki]
            if x.std() < 1e-9:
                cat_rhos.append(0.0)
            else:
                cat_rhos.append(float(np.corrcoef(x, z_arr)[0, 1]))
            cat_means.append(float(np.abs(x).mean()))
        x = np.arange(len(cat_keys))
        ax.bar(x, cat_rhos, color="#1f77b4", edgecolor="black", linewidth=0.4,
                alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_keys, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"L{L}H{h} — per-category DLA·z correlation\n"
                      f"(positive = category contributes to z-encoding)",
                      fontsize=10)
        ax.set_ylabel("ρ(per-cat DLA, z)" if ti == 0 else "")
        ax.grid(alpha=0.3, axis="y")
        # Annotate mean contribution scales
        for xi, mean in zip(x, cat_means):
            ax.text(xi, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                     f"|μ|={mean:.1f}", ha="center", fontsize=6, color="gray",
                     rotation=90)
    fig.suptitle(f"{SHORT} — trio: per-token-category DLA decomposition\n"
                  f"answers 'where in the prompt does each trio head get its "
                  f"z-write contribution from?'", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png_pos = REPO / "figures" / f"p2s_dla_trio_per_category_{SHORT}.png"
    fig.savefig(out_png_pos, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png_pos}")

    # === Summary bar plot: ρ(DLA, z) per cell ===
    fig, ax = plt.subplots(figsize=(13, 5))
    rhos = [c["corr_dla_z"] for c in cell_stats]
    means = [c["mean_dla"] for c in cell_stats]
    colors = [GROUP_COLORS[c["group"]] for c in cell_stats]
    x = np.arange(len(CELLS))
    bars = ax.bar(x, rhos, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.02 if val >= 0 else -0.04),
                 f"{val:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}H{h}" for L, h, _ in CELLS], fontsize=10)
    for tick_label, (_, _, g) in zip(ax.get_xticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_ylabel("ρ(DLA, z) — does the head's logit contribution scale with z?")
    ax.set_title(f"{SHORT} — per-head DLA-vs-z correlation\n"
                  f"a head that 'writes z to the readout' should have |ρ| close to 1")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png_b = REPO / "figures" / f"p2s_dla_corr_{SHORT}.png"
    fig.savefig(out_png_b, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png_b}")


if __name__ == "__main__":
    main()
