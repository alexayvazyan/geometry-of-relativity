"""Per-(L, h) cos(head_writeout, d_primal) in post-attention residual space.

Replacement for Phase 2M's SAE-feature cos·α: instead of encoding the manifold
delta and ablation deltas through an SAE, work directly in residual space at
the post-attention point of each layer.

For each layer L and head h on Gemma 2 2B-it, height k=15:
  d_primal[L]    = mean(post_attn_resid_L | z>1) - mean(post_attn_resid_L | z<-1)
  head_dir[L, h] = W_O[L][:, h*hd:(h+1)*hd] @ (mean(attn_h | z>1) - mean(attn_h | z<-1))
  cos[L, h]      = head_dir[L, h] · d_primal[L] / (||head_dir|| · ||d_primal||)

Notes:
  - "post-attention residual" = the residual stream right after the attention
    block, before the MLP. In Gemma 2 we hook pre_feedforward_layernorm's
    forward_pre_hook for this; its input is `residual + post_attn_layernorm(attn_out)`.
  - head_dir[L, h] is exactly the per-head contribution to the attention-side
    component of d_primal[L], so heads with low cos either (a) write a small
    direction or (b) point off-manifold relative to the cumulative target.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-2-2b-it")
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--prompts",
                    default="data/p2_shot_sweep/height_k15.jsonl")
    ap.add_argument("--label-key", default="z_eff")
    ap.add_argument("--z-thresh", type=float, default=1.0,
                    help="|z| > thresh defines high vs low")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (REPO / args.prompts).open()]
    n = len(rows)
    z = np.array([float(r[args.label_key]) for r in rows], dtype=np.float32)
    high_mask = z > +args.z_thresh
    low_mask = z < -args.z_thresh
    print(f"prompts: {n} rows  high={high_mask.sum()}  low={low_mask.sum()}")

    print(f"loading {args.model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    d_model = layers[0].self_attn.o_proj.out_features
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    print(f"  layers={n_layers}  heads={n_heads}  d_model={d_model}  head_dim={head_dim}")

    head_outs = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    post_attn = np.zeros((n_layers, n, d_model), dtype=np.float32)
    captured: dict = {}

    def make_attn_pre_hook(L):
        def hook(module, args_):
            captured[("attn", L)] = args_[0].detach().float().cpu().numpy()
        return hook

    def make_post_attn_hook(L):
        def hook(module, args_):
            captured[("post", L)] = args_[0].detach().float().cpu().numpy()
        return hook

    handles = []
    for L in range(n_layers):
        handles.append(
            layers[L].self_attn.o_proj.register_forward_pre_hook(make_attn_pre_hook(L)))
        handles.append(
            layers[L].pre_feedforward_layernorm.register_forward_pre_hook(
                make_post_attn_hook(L)))

    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            _ = model(**inp, use_cache=False)
            for L in range(n_layers):
                head_outs[L, i] = captured[("attn", L)][0, -1]
                post_attn[L, i] = captured[("post", L)][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                rate = (i + 1) / max(1e-3, time.time() - t1)
                print(f"  pass {i+1}/{n}  {rate:.1f} p/s", flush=True)
    for h in handles:
        h.remove()

    W_Os = np.stack([
        layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for L in range(n_layers)
    ], axis=0)  # (n_layers, d_model, n_heads*head_dim)
    print(f"W_O stack: {W_Os.shape}")

    d_primal = (post_attn[:, high_mask].mean(axis=1)
                 - post_attn[:, low_mask].mean(axis=1))  # (n_layers, d_model)
    d_primal_norm = np.linalg.norm(d_primal, axis=1)
    print(f"||d_primal[L]|| min={d_primal_norm.min():.3f} "
           f"max={d_primal_norm.max():.3f} median={np.median(d_primal_norm):.3f}")

    attn_high = head_outs[:, high_mask].mean(axis=1)  # (n_layers, n_heads*head_dim)
    attn_low = head_outs[:, low_mask].mean(axis=1)
    attn_diff = attn_high - attn_low

    cos_grid = np.zeros((n_layers, n_heads), dtype=np.float32)
    head_dir_norms = np.zeros((n_layers, n_heads), dtype=np.float32)
    for L in range(n_layers):
        for h in range(n_heads):
            slice_W = W_Os[L][:, h * head_dim:(h + 1) * head_dim]
            head_dir = slice_W @ attn_diff[L, h * head_dim:(h + 1) * head_dim]
            nh = float(np.linalg.norm(head_dir))
            head_dir_norms[L, h] = nh
            if nh < 1e-12 or d_primal_norm[L] < 1e-12:
                cos_grid[L, h] = 0.0
            else:
                cos_grid[L, h] = float(np.dot(head_dir, d_primal[L])
                                        / (nh * d_primal_norm[L]))

    out_path = REPO / "results" / f"p2u_residual_cos_{args.short}.npz"
    np.savez(out_path,
             cos_grid=cos_grid,
             head_dir_norms=head_dir_norms,
             d_primal=d_primal.astype(np.float32),
             d_primal_norm=d_primal_norm,
             z=z, high_mask=high_mask, low_mask=low_mask)
    print(f"\nwrote {out_path}")
    print(f"\ncos_grid summary:")
    for L in range(n_layers):
        row = "  ".join(f"{cos_grid[L, h]:+.2f}" for h in range(n_heads))
        print(f"  L{L:>2d}: {row}")


if __name__ == "__main__":
    main()
