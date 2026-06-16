"""Phase 2Q — per-head z-information vs Phase 2M cosine.

Theory: Phase 2M cosine measures direction-alignment of zero-ablation Δ to
manifold-Δ in SAE feature space. Phase 2O resample Δr depends additionally
on how much *z-information* the head carries across prompts (high
prompt-to-prompt variance along the z-axis means resample swaps a large
signal). Two heads with similar cos can have very different resample Δr if
their z-information differs.

Operationalization: for each (L, h), regress z_arr on the head's last-token
output (D=head_dim features, n=990 prompts) using ridge with held-out R².
R²_z[L, h] = generalization-R² → "z-information per head."

Predict: among cells with cos ≥ 0.30, those with high R²_z explain the
known per-cell Δr gap (L16H4 high R², L17H7 low R², L14H7/L15H7 mid).

Single forward pass over 990 height k=15 prompts. Output:
  results/p2q_head_zinfo_<short>.npz  (pool, R²_z grid)
  results/p2q_head_zinfo_<short>.json (summary stats per cell)
  figures/p2q_head_zinfo_<short>.png  (R²_z heatmap + R²_z vs cos scatter)

Usage:
  python p2q_head_zinfo.py --short gemma2-2b --device cpu     # ~22 min
  python p2q_head_zinfo.py --short gemma2-9b --device cuda    # ~2 min
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False; break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def cv_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
          k: int = 5, seed: int = 0) -> float:
    """5-fold CV R² for ridge regression."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    yhat = np.zeros_like(y)
    for tr, te in kf.split(X):
        mdl = Ridge(alpha=alpha)
        mdl.fit(X[tr], y[tr])
        yhat[te] = mdl.predict(X[te])
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b",
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16"],
                    help="auto = float32 on cpu, bfloat16 on cuda")
    args = ap.parse_args()

    SHORT = args.short
    MODEL = ("google/gemma-2-2b" if SHORT == "gemma2-2b"
              else "google/gemma-2-9b")
    FEATURE = args.feature
    K = args.k
    DEVICE = args.device
    if args.dtype == "auto":
        DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    elif args.dtype == "float32":
        DTYPE = torch.float32
    else:
        DTYPE = torch.bfloat16

    # Stimuli
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{FEATURE}_k{K}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)
    print(f"loaded {n} prompts; z range [{z_arr.min():+.2f}, {z_arr.max():+.2f}]")

    # Model
    print(f"\nloading {MODEL} on {DEVICE} (dtype={DTYPE})...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DTYPE, device_map="auto",
            token=os.environ.get("HF_TOKEN")).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DTYPE,
            token=os.environ.get("HF_TOKEN")).eval()
        model.to(DEVICE)
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    print(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")

    # Capture pre-o_proj last token at every layer
    pool = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    captured = {}

    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
                for L in range(n_layers)]

    print(f"\n[pass] capture pre-o_proj@last token across {n_layers} layers, {n} prompts")
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            _ = model(**inp, use_cache=False)
            for L in range(n_layers):
                pool[L, i] = captured[L][0, -1]
            if (i + 1) % 50 == 0 or i == n - 1:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                eta = (n - i - 1) / rate
                print(f"  {i+1}/{n}  {rate:.2f} p/s  eta {eta/60:.1f}m",
                      flush=True)
    for h in handles:
        h.remove()

    # Free model — analysis is numpy
    del model
    print(f"\npool shape: {pool.shape}, mem: {pool.nbytes/1e6:.1f} MB")

    # Per-cell R²(z | head output) via 5-fold CV ridge
    print(f"\ncomputing CV-R²(z | head output) for {n_layers}x{n_heads} cells...")
    r2_grid = np.zeros((n_layers, n_heads), dtype=np.float32)
    sigma_grid = np.zeros((n_layers, n_heads), dtype=np.float32)  # mean per-dim std
    t1 = time.time()
    for L in range(n_layers):
        for h in range(n_heads):
            X = pool[L, :, h * head_dim:(h + 1) * head_dim]  # (n, D)
            r2_grid[L, h] = cv_r2(X, z_arr, alpha=1.0, k=5, seed=0)
            sigma_grid[L, h] = float(np.std(X, axis=0).mean())
        print(f"  L{L:2d}: max R²={r2_grid[L].max():+.3f} "
              f"max σ={sigma_grid[L].max():.2f}  "
              f"({time.time()-t1:.0f}s)", flush=True)

    # Save NPZ + JSON
    out_npz = REPO / "results" / f"p2q_head_zinfo_{SHORT}.npz"
    np.savez(out_npz, pool=pool.astype(np.float16),
             r2_grid=r2_grid, sigma_grid=sigma_grid,
             z=z_arr)
    print(f"\nwrote {out_npz}")

    # Cross with Phase 2M (prefer _fullgrid.json if present, e.g. for 9B)
    p2m_path = REPO / "results" / f"p2m_alignment_{SHORT}_fullgrid.json"
    if not p2m_path.exists():
        p2m_path = REPO / "results" / f"p2m_alignment_{SHORT}.json"
    print(f"\nusing Phase 2M cos from: {p2m_path.name}")
    p2m = json.loads(p2m_path.read_text())
    cos_grid = np.array(p2m["cos_grid"])
    norms_grid = np.array(p2m["delta_norms"])
    p2m_Ls = p2m["layer_idxs"]
    p2m_Hs = p2m["head_idxs"]

    # Build aligned cos vector for our (L, h) order
    cos_aligned = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    norms_aligned = np.full_like(cos_aligned, np.nan)
    for li, L in enumerate(p2m_Ls):
        if L >= n_layers:
            continue
        for hi, h in enumerate(p2m_Hs):
            if h >= n_heads:
                continue
            cos_aligned[L, h] = cos_grid[li, hi]
            norms_aligned[L, h] = norms_grid[li, hi]

    if SHORT == "gemma2-2b":
        known_dr = {
            (16, 4): -0.182, (17, 7): -0.036, (14, 2): -0.004,
        }
    else:
        known_dr = {}  # no per-cell Δr known on 9B yet

    # Pretty print summary table
    summary = []
    for (L, h), dr in known_dr.items():
        c = cos_aligned[L, h]
        nm = norms_aligned[L, h]
        r2 = r2_grid[L, h]
        sg = sigma_grid[L, h]
        summary.append({"L": L, "h": h, "cos": float(c), "norm": float(nm),
                         "r2": float(r2), "sigma": float(sg), "dr": dr})

    print("\n=== known causal cells ===")
    print(f"{'cell':<8s}    cos    ||Δa||    R²(z|h)   σ      Δr")
    for s in summary:
        print(f"L{s['L']:2d}H{s['h']:1d}    {s['cos']:+.3f}   {s['norm']:6.2f}   "
              f"{s['r2']:+.3f}    {s['sigma']:5.2f}  {s['dr']:+.3f}")

    # Top R²_z cells
    flat = []
    for L in range(n_layers):
        for h in range(n_heads):
            flat.append((r2_grid[L, h], cos_aligned[L, h], L, h))
    flat.sort(key=lambda r: -r[0])
    print("\n=== top 16 R²_z cells (with Phase 2M cos) ===")
    for r2, c, L, h in flat[:16]:
        cs = f"{c:+.3f}" if not np.isnan(c) else "  N/A"
        print(f"  L{L:2d}H{h:1d}: R²={r2:+.3f}  cos={cs}")

    # JSON summary
    out_json = REPO / "results" / f"p2q_head_zinfo_{SHORT}.json"
    out_json.write_text(json.dumps({
        "model": MODEL, "feature": FEATURE, "k": K,
        "n_prompts": int(n),
        "r2_grid": r2_grid.tolist(),
        "sigma_grid": sigma_grid.tolist(),
        "known_cells": summary,
    }, indent=2))
    print(f"wrote {out_json}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.4 * n_layers + 1.5))

    # R² heatmap
    ax = axes[0]
    im = ax.imshow(r2_grid, aspect="auto", cmap="viridis",
                    vmin=0, vmax=max(0.3, float(np.max(r2_grid))))
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("CV-R²(z | head output)\n(per-head z-information)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="R²")
    for s in summary:
        ax.add_patch(plt.Rectangle((s["h"]-0.5, s["L"]-0.5), 1, 1,
                                    fill=False, edgecolor="red", lw=1.5))
        ax.text(s["h"], s["L"]+0.4, f"Δr={s['dr']:+.2f}",
                 ha="center", fontsize=6, color="white")

    # Phase 2M cos heatmap (for direct visual comparison)
    ax = axes[1]
    vmax = max(0.05, float(np.nanmax(np.abs(cos_aligned))))
    im = ax.imshow(cos_aligned, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Phase 2M cos(Δ_ablate, Δ_manifold)\n(direction alignment)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cosine")

    # R²_z vs cos scatter
    ax = axes[2]
    cos_flat = cos_aligned.ravel()
    r2_flat = r2_grid.ravel()
    valid = ~np.isnan(cos_flat)
    ax.scatter(cos_flat[valid], r2_flat[valid],
                s=15, alpha=0.5, edgecolor="none", color="gray")
    for s in summary:
        ax.scatter([s["cos"]], [s["r2"]], s=80, color="red",
                    edgecolor="black", zorder=10)
        ax.annotate(f"L{s['L']}H{s['h']}\nΔr={s['dr']:+.3f}",
                     (s["cos"], s["r2"]), fontsize=8,
                     xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Phase 2M cos")
    ax.set_ylabel("CV-R²(z | head output)")
    ax.set_title("R²_z vs cos\n(known cells highlighted)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{SHORT} — per-head z-information vs Phase 2M cos "
                  f"(height k={K}, n={n}, last token)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = REPO / "figures" / f"p2q_head_zinfo_{SHORT}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
