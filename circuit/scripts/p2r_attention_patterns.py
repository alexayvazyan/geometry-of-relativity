"""Phase 2R — attention pattern visualization for cos·σ trio + cluster.

For each (L, h) in the causal trio + H7 column + Phase 2Q candidates +
controls, capture last-token → context attention weights on a small set of
representative height k=15 prompts on Gemma 2 2B. Build visualizations:

  Fig 1: token-aligned attention heatmap on a representative prompt
         - rows = heads, columns = token positions, color = attention
         - bottom strip: token category labels (number / unit / target / etc.)
  Fig 2: per-head attention-category bar chart (aggregated over prompts)
         - bars = fraction of last-token attention going to each category
  Fig 3: attention to context-numbers vs target-number, by head, vs prompt z
         - reveals "averager" vs "comparator" heads
  Fig 4: position bias heatmap (head × position rank, mean attention)

Output:
  results/p2r_attn_gemma2-2b.npz  (per-prompt attn from last token)
  results/p2r_attn_gemma2-2b.json (per-prompt category breakdowns)
  figures/p2r_attn_overlay_gemma2-2b.png
  figures/p2r_attn_categories_gemma2-2b.png
  figures/p2r_attn_context_vs_target_gemma2-2b.png
  figures/p2r_attn_position_bias_gemma2-2b.png
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

# (L, h, group) — groups for color/legend
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


def categorize_tokens(prompt: str, tok) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    """Categorize each token in the prompt.

    Returns (token_strings, categories, offset_pairs).
    Categories: 'context_num', 'target_num', 'unit', 'person_label',
                'colon', 'newline', 'tail', 'bos', 'other'.
    """
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    token_strs = [tok.decode([i]) for i in ids]

    # Find target number position in prompt: the digits in the last "Person N: <X>"
    # Robust strategy: find the LAST run of digits in the prompt.
    digit_runs = [(m.start(), m.end()) for m in re.finditer(r"\d+", prompt)]
    target_run = digit_runs[-1] if digit_runs else (-1, -1)

    # Identify "tail" segment: from the position of "cm. This person is" onwards.
    # More robust: from the second occurrence of ".\s*This person is" onwards.
    tail_match = re.search(r"\.\s*This person is", prompt) or \
                 re.search(r"\.\s*This [A-Za-z]+ is", prompt)
    tail_start = tail_match.start() if tail_match else len(prompt)

    cats = []
    for tid, (s, e) in zip(ids, offsets):
        s_int, e_int = int(s), int(e)
        substr = prompt[s_int:e_int] if e_int > s_int else ""
        decoded = tok.decode([tid])

        if e_int <= s_int and s_int == 0:
            cats.append("bos")
            continue

        if s_int >= tail_start:
            cats.append("tail")
            continue

        # Inside target number run
        if target_run[0] <= s_int < target_run[1]:
            cats.append("target_num")
            continue

        # Other digit runs → context numbers
        if any(s_int >= ds and e_int <= de for ds, de in digit_runs):
            cats.append("context_num")
            continue
        # Even partial overlap
        if any(ds <= s_int < de or ds < e_int <= de for ds, de in digit_runs):
            cats.append("context_num")
            continue

        if "\n" in substr:
            cats.append("newline")
            continue
        if "Person" in substr or "person" in decoded.lower() and "is" not in decoded.lower():
            cats.append("person_label")
            continue
        if substr.strip() in ("cm", "kg", "km/h", "km", "m", "mm"):
            cats.append("unit")
            continue
        if substr.strip() == ":":
            cats.append("colon")
            continue
        cats.append("other")

    return token_strs, cats, offsets


def main() -> None:
    SHORT = "gemma2-2b"
    MODEL = "google/gemma-2-2b"
    FEATURE = "height"
    K = 15

    # Stimuli: pick a small set of representative prompts spanning z range
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]

    # Bin by z and sample one prompt per z bin
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows])
    z_bins = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    chosen_idx = []
    for zb in z_bins:
        i_close = int(np.argmin(np.abs(z_arr - zb)))
        chosen_idx.append(i_close)
    # Plus a chunk of random prompts for stability of category fractions
    rng = np.random.RandomState(0)
    extra = rng.choice(len(rows), size=64, replace=False).tolist()
    all_idx = list(dict.fromkeys(chosen_idx + extra))
    print(f"selected {len(all_idx)} prompts ({len(chosen_idx)} z-bin reps + 64 extra)")

    # Model
    print(f"\nloading {MODEL}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # required for output_attentions
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  n_layers={n_layers} n_heads={n_heads}")

    # Capture
    per_prompt = []  # list of dicts
    print(f"\n[capture] {len(all_idx)} prompts with output_attentions...")
    t1 = time.time()
    with torch.inference_mode():
        for i_iter, prompt_idx in enumerate(all_idx):
            row = rows[prompt_idx]
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, output_attentions=True, use_cache=False)
            T = inp.input_ids.shape[1]
            # attentions: tuple of (B, n_heads, T, T) per layer
            # Take last-token row only: shape (n_layers, n_heads, T)
            attn_last = np.zeros((n_layers, n_heads, T), dtype=np.float32)
            for L in range(n_layers):
                a = out.attentions[L]  # (1, H, T, T)
                attn_last[L] = a[0, :, -1, :].float().cpu().numpy()
            tok_strs, cats, offsets = categorize_tokens(row["prompt"], tok)
            per_prompt.append({
                "prompt_idx": int(prompt_idx),
                "z": float(row.get("z_eff", row.get("z", 0))),
                "x": float(row["x"]),
                "mu": float(row.get("mu_eff", row.get("mu", 0))),
                "sigma": float(row.get("sigma_eff", row.get("sigma", 1))),
                "T": int(T),
                "tokens": tok_strs,
                "cats": cats,
                "attn_last": attn_last,  # (n_layers, n_heads, T)
            })
            if (i_iter + 1) % 10 == 0:
                rate = (i_iter + 1) / max(1e-3, time.time() - t1)
                print(f"  {i_iter+1}/{len(all_idx)}  {rate:.2f} p/s", flush=True)

    # Save (variable T per prompt → store each prompt's attn under its own key)
    out_npz = REPO / "results" / f"p2r_attn_{SHORT}.npz"
    save_kwargs = {f"attn_p{i}": p["attn_last"].astype(np.float16)
                    for i, p in enumerate(per_prompt)}
    save_kwargs["zs"] = np.array([p["z"] for p in per_prompt])
    save_kwargs["xs"] = np.array([p["x"] for p in per_prompt])
    save_kwargs["mus"] = np.array([p["mu"] for p in per_prompt])
    save_kwargs["sigmas"] = np.array([p["sigma"] for p in per_prompt])
    save_kwargs["prompt_idxs"] = np.array([p["prompt_idx"] for p in per_prompt])
    save_kwargs["Ts"] = np.array([p["T"] for p in per_prompt])
    np.savez(out_npz, **save_kwargs)
    print(f"wrote {out_npz}")

    # JSON: tokens + cats per prompt (lightweight)
    out_json_data = {
        "model": MODEL, "short": SHORT, "feature": FEATURE, "k": K,
        "cells": [{"layer": L, "head": h, "group": g} for L, h, g in CELLS],
        "prompts": [
            {"prompt_idx": p["prompt_idx"], "z": p["z"], "x": p["x"],
             "T": p["T"], "tokens": p["tokens"], "cats": p["cats"]}
            for p in per_prompt],
    }
    out_json = REPO / "results" / f"p2r_attn_{SHORT}.json"
    out_json.write_text(json.dumps(out_json_data, indent=2))
    print(f"wrote {out_json}")

    # ============== VIZ 1: Token-aligned heatmap on z=0 representative ==============
    rep = next(p for p in per_prompt if abs(p["z"] - 0.0) < 0.5)
    print(f"\nrepresentative prompt: idx={rep['prompt_idx']}, z={rep['z']:+.2f}")
    T_rep = rep["T"]
    fig, axes = plt.subplots(2, 1, figsize=(max(14, T_rep * 0.13), 0.45 * len(CELLS) + 2.5),
                              gridspec_kw={"height_ratios": [len(CELLS), 1.5]},
                              sharex=True)
    ax = axes[0]
    cell_attn = np.zeros((len(CELLS), T_rep), dtype=np.float32)
    for ci, (L, h, _) in enumerate(CELLS):
        cell_attn[ci] = rep["attn_last"][L, h]
    # Per-row normalize (each head's max attention = 1) for visibility
    cell_attn_norm = cell_attn / np.clip(cell_attn.max(axis=1, keepdims=True), 1e-9, None)
    im = ax.imshow(cell_attn_norm, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"L{L}H{h}" for L, h, _ in CELLS], fontsize=10)
    # Color y-tick labels by group
    for tick_label, (_, _, g) in zip(ax.get_yticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_title(f"{SHORT} — per-head attention from last token, "
                  f"z={rep['z']:+.2f} prompt (idx {rep['prompt_idx']}; "
                  f"x={rep['x']:.0f}, μ={rep['mu']:.0f}, σ={rep['sigma']:.1f})\n"
                  f"row-normalized so each head's strongest target = 1",
                  fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="attn (row-normalized)")

    # Bottom: token strip with categories
    ax = axes[1]
    cat_colors = {
        "context_num": "#1f77b4", "target_num": "#d62728", "unit": "#2ca02c",
        "person_label": "#9467bd", "colon": "#7f7f7f", "newline": "#ffd700",
        "tail": "#ff7f0e", "bos": "#000000", "other": "#cccccc",
    }
    cats = rep["cats"]
    tokens = rep["tokens"]
    for t, c in enumerate(cats):
        ax.add_patch(plt.Rectangle((t - 0.5, 0), 1, 1,
                                    facecolor=cat_colors.get(c, "#cccccc"),
                                    edgecolor="white", linewidth=0.5))
    # Add token text on alternating rows for readability
    for t, tk in enumerate(tokens):
        s = tk.replace("\n", "\\n").strip()
        if not s:
            continue
        ax.text(t, 0.5 if t % 2 == 0 else 0.15,
                s[:6], ha="center", va="center",
                fontsize=6, rotation=90, color="white"
                if cats[t] in ("target_num", "tail", "person_label", "bos")
                else "black")
    ax.set_xlim(-0.5, T_rep - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("token position")
    # Legend for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c], label=c)
                for c in ["context_num", "target_num", "unit", "person_label",
                          "colon", "newline", "tail"]]
    ax.legend(handles=handles, loc="upper center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2r_attn_overlay_{SHORT}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")

    # ============== VIZ 2: aggregate category fractions per head ==============
    # For each cell, compute fraction of last-token attention going to each category,
    # averaged across all collected prompts.
    cat_keys = ["context_num", "target_num", "unit", "person_label",
                 "colon", "newline", "tail", "bos", "other"]
    frac_grid = np.zeros((len(CELLS), len(cat_keys)), dtype=np.float32)
    for p in per_prompt:
        cats_arr = np.array(p["cats"])
        for ci, (L, h, _) in enumerate(CELLS):
            a = p["attn_last"][L, h]  # (T,)
            for ki, k in enumerate(cat_keys):
                frac_grid[ci, ki] += float(a[cats_arr == k].sum())
    frac_grid /= len(per_prompt)

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(CELLS) + 1.5))
    bottoms = np.zeros(len(CELLS))
    for ki, k in enumerate(cat_keys):
        ax.barh(np.arange(len(CELLS)), frac_grid[:, ki], left=bottoms,
                color=cat_colors.get(k, "#cccccc"), label=k, edgecolor="white",
                linewidth=0.5)
        bottoms += frac_grid[:, ki]
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"L{L}H{h}" for L, h, _ in CELLS], fontsize=10)
    for tick_label, (_, _, g) in zip(ax.get_yticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_xlabel("fraction of last-token attention")
    ax.set_xlim(0, 1)
    ax.set_title(f"{SHORT} — last-token attention category breakdown "
                  f"(mean over {len(per_prompt)} prompts)\n"
                  f"red ticks = causal trio, gray = high-||Δa|| controls")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2r_attn_categories_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # ============== VIZ 3: per-head context vs target attention vs z ==============
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for ax_i, (L, h, g) in enumerate(CELLS):
        ax = axes.flat[ax_i]
        zs = []
        attn_ctx = []
        attn_tgt = []
        for p in per_prompt:
            cats_arr = np.array(p["cats"])
            a = p["attn_last"][L, h]
            zs.append(p["z"])
            attn_ctx.append(float(a[cats_arr == "context_num"].sum()))
            attn_tgt.append(float(a[cats_arr == "target_num"].sum()))
        zs = np.array(zs); attn_ctx = np.array(attn_ctx); attn_tgt = np.array(attn_tgt)
        ax.scatter(zs, attn_ctx, s=18, alpha=0.7, color="#1f77b4", label="context_num")
        ax.scatter(zs, attn_tgt, s=18, alpha=0.7, color="#d62728", label="target_num")
        ax.set_title(f"L{L}H{h} — {g}", fontsize=10,
                      color=GROUP_COLORS[g])
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        if ax_i % 4 == 0:
            ax.set_ylabel("attention fraction")
        if ax_i >= 8:
            ax.set_xlabel("z (target deviation)")
        if ax_i == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{SHORT} — last-token attention: context vs target vs z",
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = REPO / "figures" / f"p2r_attn_context_vs_target_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # ============== VIZ 4: position-bias heatmap (head × context-number-rank) ==============
    # For each prompt, find positions of context numbers (in order). For each head, average
    # attention over context-number positions ranked 1..15.
    # If the head attends to position k uniformly, this is uniform 1/15.
    # Position 1 = first reference, position 15 = last reference.
    pos_grid = np.zeros((len(CELLS), 15), dtype=np.float32)
    pos_grid_count = np.zeros(15)
    for p in per_prompt:
        cats_arr = np.array(p["cats"])
        ctx_positions = np.where(cats_arr == "context_num")[0]
        # If a number tokenizes to multiple tokens, just take the first
        # For simplicity, identify run starts: position k where cats_arr[k]=='context_num'
        # and (k==0 or cats_arr[k-1]!='context_num')
        run_starts = []
        for ki, posi in enumerate(ctx_positions):
            if posi == 0 or cats_arr[posi - 1] != "context_num":
                run_starts.append(posi)
        # For each "rank r" in 1..15, sum attention across all tokens in that number's run
        for r, start in enumerate(run_starts[:15]):
            # span: from start to next non-context_num
            end = start
            while end < len(cats_arr) and cats_arr[end] == "context_num":
                end += 1
            for ci, (L, h, _) in enumerate(CELLS):
                pos_grid[ci, r] += float(p["attn_last"][L, h, start:end].sum())
            pos_grid_count[r] += 1
    pos_grid_count = np.clip(pos_grid_count, 1, None)
    pos_grid_mean = pos_grid / pos_grid_count[None, :]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(CELLS) + 1.5))
    im = ax.imshow(pos_grid_mean, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"L{L}H{h}" for L, h, _ in CELLS], fontsize=10)
    for tick_label, (_, _, g) in zip(ax.get_yticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_xticks(range(15))
    ax.set_xticklabels([f"#{r+1}" for r in range(15)], fontsize=9)
    ax.set_xlabel("context-number position rank (Person 1..15)")
    ax.set_title(f"{SHORT} — attention to each context-number by rank "
                  f"(mean over {len(per_prompt)} prompts)\n"
                  f"uniform-attender (averager) → flat row;  "
                  f"recency/primacy → tilted row")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="mean attention")
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2r_attn_position_bias_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    print("\n[done]")


if __name__ == "__main__":
    main()
