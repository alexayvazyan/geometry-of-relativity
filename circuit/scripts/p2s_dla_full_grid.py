"""Phase 2S — full-grid DLA across all (L, h) on 2B.

Compute DLA(L, h) per prompt for ALL 26 × 8 = 208 cells, then aggregate
into σ_DLA, mean DLA, and ρ(DLA, z) grids.

Output:
  results/p2s_dla_full_gemma2-2b.npz   (per-prompt per-cell DLA grid)
  figures/p2s_dla_full_gemma2-2b.png   (3-panel: σ, mean, ρ heatmaps)
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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b",
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--feature", default="height",
                    choices=["height", "weight", "speed"])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-prompts", type=int, default=200,
                    help="cap on prompts (default 200; uses Phase 2R "
                          "subset for height, random subset otherwise)")
    args = ap.parse_args()

    SHORT = args.short
    MODEL = ("google/gemma-2-2b" if SHORT == "gemma2-2b"
              else "google/gemma-2-9b")
    FEATURE = args.feature
    K = args.k

    # For height, reuse Phase 2R prompts (so DLA matches earlier height results).
    # For weight/speed, sample a stratified subset.
    if FEATURE == "height":
        p2r_json = REPO / "results" / f"p2r_attn_{SHORT}.json"
        if p2r_json.exists():
            p2r = json.loads(p2r_json.read_text())
            rows_all = [json.loads(l) for l in
                         (REPO / "data/p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl").open()]
            prompt_idxs = [p["prompt_idx"] for p in p2r["prompts"]]
            rows = [rows_all[i] for i in prompt_idxs]
        else:
            rows_all = [json.loads(l) for l in
                         (REPO / "data/p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl").open()]
            rng = np.random.RandomState(0)
            idxs = rng.choice(len(rows_all), size=min(args.n_prompts,
                                                        len(rows_all)),
                               replace=False)
            rows = [rows_all[i] for i in idxs]
    else:
        rows_all = [json.loads(l) for l in
                     (REPO / "data/p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl").open()]
        rng = np.random.RandomState(0)
        idxs = rng.choice(len(rows_all), size=min(args.n_prompts,
                                                    len(rows_all)),
                           replace=False)
        rows = [rows_all[i] for i in idxs]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)
    print(f"using {n} prompts (from Phase 2R)")

    print(f"\nloading {MODEL}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    layers = model.model.layers
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers} n_heads={n_heads} head_dim={head_dim}")

    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]

    # Compute the head-input → tall_dir projection vector once per layer per head.
    # head_contribution_to_logit = head_input @ W_O_slice.T @ tall_dir
    #                            = head_input @ (tall_dir @ W_O_slice).T
    # So per (L, h), pre-compute proj_vec = tall_dir @ W_O[:, h*head_dim:(h+1)*head_dim]
    # shape (head_dim,). Then DLA = head_input @ proj_vec.
    W_U = model.lm_head.weight.detach()
    tall_dir = (W_U[high_id] - W_U[low_id]).float().cpu().numpy()
    proj_vecs = np.zeros((n_layers, n_heads, head_dim), dtype=np.float32)
    for L in range(n_layers):
        W_O = layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for h in range(n_heads):
            slice_ = W_O[:, h * head_dim:(h + 1) * head_dim]  # (hidden, head_dim)
            proj_vecs[L, h] = tall_dir @ slice_  # (head_dim,)

    captured = {}
    def make_cap(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook
    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_cap(L))
                for L in range(n_layers)]

    # Per-prompt per-cell DLA: shape (n, n_layers, n_heads)
    dla_grid = np.zeros((n, n_layers, n_heads), dtype=np.float32)
    ld_baseline = np.zeros(n, dtype=np.float32)

    print(f"\n[forward] {n} prompts...")
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            ld_baseline[i] = float(logits[high_id] - logits[low_id])
            for L in range(n_layers):
                # captured[L] shape (1, T, n_heads*head_dim)
                last_token = captured[L][0, -1]  # (n_heads * head_dim,)
                for h in range(n_heads):
                    head_v = last_token[h * head_dim:(h + 1) * head_dim]
                    dla_grid[i, L, h] = float(np.dot(head_v, proj_vecs[L, h]))
            if (i + 1) % 10 == 0:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                print(f"  {i+1}/{n}  {rate:.1f} p/s", flush=True)
    for h_ in handles:
        h_.remove()

    # Aggregates
    mean_dla = dla_grid.mean(axis=0)         # (n_layers, n_heads)
    std_dla = dla_grid.std(axis=0, ddof=1)   # (n_layers, n_heads)
    rho_grid = np.zeros_like(mean_dla)
    for L in range(n_layers):
        for h in range(n_heads):
            x = dla_grid[:, L, h]
            if x.std() < 1e-9:
                rho_grid[L, h] = 0.0
            else:
                rho_grid[L, h] = float(np.corrcoef(x, z_arr)[0, 1])

    # Save (feature-aware filename)
    tag = f"{SHORT}" if FEATURE == "height" else f"{SHORT}_{FEATURE}"
    out_npz = REPO / "results" / f"p2s_dla_full_{tag}.npz"
    np.savez(out_npz, dla_grid=dla_grid, mean_dla=mean_dla, std_dla=std_dla,
              rho_grid=rho_grid, zs=z_arr, ld_baseline=ld_baseline,
              high_id=high_id, low_id=low_id)
    print(f"\nwrote {out_npz}")

    # Print top σ_DLA
    flat = []
    for L in range(n_layers):
        for h in range(n_heads):
            flat.append((std_dla[L, h], mean_dla[L, h], rho_grid[L, h], L, h))
    flat.sort(key=lambda r: -r[0])
    print(f"\n=== top 25 cells by σ_DLA ===")
    print(f"{'cell':<8s}  {'σ_DLA':>7s}  {'mean DLA':>10s}  {'ρ(DLA,z)':>10s}")
    for sd, mn, rh, L, h in flat[:25]:
        print(f"L{L:>2d}H{h}     {sd:7.3f}    {mn:+8.3f}    {rh:+8.3f}")

    print(f"\n=== top 25 cells by |ρ(DLA, z)| ===")
    flat.sort(key=lambda r: -abs(r[2]))
    for sd, mn, rh, L, h in flat[:25]:
        print(f"L{L:>2d}H{h}     {sd:7.3f}    {mn:+8.3f}    {rh:+8.3f}")

    # Plot — three heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.4 * n_layers + 1.5))

    # σ_DLA — viridis
    ax = axes[0]
    im = ax.imshow(std_dla, aspect="auto", cmap="viridis",
                    vmin=0, vmax=float(std_dla.max()))
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("σ_DLA — std of DLA across prompts\n"
                  "(swap distance under interchange ablation)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="σ_DLA")
    # Mark trio with red box
    for (L, h) in [(16, 4), (17, 7), (14, 2)]:
        ax.add_patch(plt.Rectangle((h-0.5, L-0.5), 1, 1, fill=False,
                                    edgecolor="red", lw=2))

    # mean DLA — diverging
    ax = axes[1]
    vmax = max(abs(float(mean_dla.min())), abs(float(mean_dla.max())))
    im = ax.imshow(mean_dla, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_title("mean DLA — average head→logit(tall − short)\n"
                  "(red = pushes 'tall', blue = pushes 'short')")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="mean DLA")
    for (L, h) in [(16, 4), (17, 7), (14, 2)]:
        ax.add_patch(plt.Rectangle((h-0.5, L-0.5), 1, 1, fill=False,
                                    edgecolor="red", lw=2))

    # ρ(DLA, z) — diverging
    ax = axes[2]
    im = ax.imshow(rho_grid, aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=+1)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_title("ρ(DLA, z) — head's z-encoding strength\n"
                  "(red = +z writer, blue = −z / anti-z writer)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="ρ")
    for (L, h) in [(16, 4), (17, 7), (14, 2)]:
        ax.add_patch(plt.Rectangle((h-0.5, L-0.5), 1, 1, fill=False,
                                    edgecolor="red", lw=2))

    fig.suptitle(f"{SHORT} — full-grid DLA "
                  f"(red box = causal trio identified by Phase 2O)",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = REPO / "figures" / f"p2s_dla_full_{tag}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
