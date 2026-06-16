"""Phase 2L — Decompose L1 attention output via Gemma Scope SAEs.

Inspired by Kissane et al. 2024 ("Interpreting Attention Layer Outputs with
Sparse Autoencoders"). Phase 2F localised primary comparator heads to
  gemma2-2b: L1H6  (r=+0.71)
  gemma2-9b: L1H11 (r=+0.60)
This script asks: which Gemma Scope features at L1 attn_out carry the
z-signal, and do those features attribute back to the primary head?

Pipeline:
  1. Forward each k=15 prompt; hook L1.self_attn.o_proj to capture
        - pre-o_proj input x_full   (B, T, H*D)   — heads packed contiguously
        - post-o_proj attn_out      (B, T, hidden)
  2. SAE encode attn_out at last token (JumpReLU). Active features only.
  3. Per-feature: corr(activation, z_eff) across prompts.
  4. Per-head attribution for top z-features:
        pre_h[i] = x[h_slice] @ W_O[h_slice, :].T @ W_enc[:, i]
        (skip b_enc and threshold for attribution; sum_h pre_h ≈ pre)
  5. Save JSON + plot top features × per-head bar.

Loads SAE directly from HuggingFace (no sae_lens dep).
SAE path: google/gemma-scope-<MODEL>-pt-att/layer_<L>/width_<W>/average_l0_<L0>/params.npz

Output:
  results/p2l_attn_sae_<short>_<feature>_k<k>.json
  figures/p2l_attn_sae_<short>_<feature>_k<k>.png

Compute: ~30s on 2B, ~2 min on 9B for k=15 height.

Usage:
  python p2l_attn_sae_decompose.py --model google/gemma-2-2b --short gemma2-2b \\
      --layer 1 --primary-head 6 --sae-width 16k --sae-l0 67
  python p2l_attn_sae_decompose.py --model google/gemma-2-9b --short gemma2-9b \\
      --layer 1 --primary-head 11 --sae-width 16k --sae-l0 67
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def load_sae(model_short: str, layer: int, width: str, l0: int) -> dict:
    """Load Gemma Scope attn_out SAE from HF.

    Returns dict with W_enc (d, F), W_dec (F, d), b_enc (F,), b_dec (d,),
    threshold (F,) — all numpy fp32.
    """
    if model_short.startswith("gemma2-2b"):
        repo = "google/gemma-scope-2b-pt-att"
    elif model_short.startswith("gemma2-9b"):
        repo = "google/gemma-scope-9b-pt-att"
    else:
        raise ValueError(f"no Gemma Scope attn SAE for {model_short}")
    fname = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
    print(f"  loading SAE: {repo}/{fname}", flush=True)
    path = hf_hub_download(repo_id=repo, filename=fname,
                            token=os.environ.get("HF_TOKEN"))
    npz = np.load(path)
    sae = {k: npz[k].astype(np.float32) for k in npz.files}
    print(f"  SAE keys: {list(sae.keys())}  "
          f"W_enc={sae['W_enc'].shape}  threshold mean={sae['threshold'].mean():.3f}",
          flush=True)
    return sae


def encode_sae(x: np.ndarray, sae: dict) -> tuple[np.ndarray, np.ndarray]:
    """JumpReLU SAE encode. x: (N, d).
    Returns (pre, post) both (N, F)."""
    pre = x @ sae["W_enc"] + sae["b_enc"]
    gate = (pre > sae["threshold"]).astype(np.float32)
    post = pre * gate
    return pre, post


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers"),
                  ("model", "language_model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False
                break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def load_stim_prompts(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id, e.g. google/gemma-2-2b")
    ap.add_argument("--short", required=True,
                    help="model short name, e.g. gemma2-2b")
    ap.add_argument("--layer", type=int, default=1,
                    help="decoder layer index for attn_out hook")
    ap.add_argument("--primary-head", type=int, required=True,
                    help="head index of the Phase 2F primary comparator")
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0", type=int, default=67)
    ap.add_argument("--feature", default="height",
                    choices=["height", "weight", "speed"])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--top-n", type=int, default=20,
                    help="report top-N z-correlated features")
    ap.add_argument("--data-root", default=str(REPO / "data" / "p2_shot_sweep"))
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    out_name = (args.out_name or
                f"p2l_attn_sae_{args.short}_{args.feature}_k{args.k}")
    out_json = REPO / "results" / f"{out_name}.json"
    out_png = REPO / "figures" / f"{out_name}.png"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load stimuli
    stim_path = Path(args.data_root) / f"{args.feature}_k{args.k}.jsonl"
    rows = load_stim_prompts(stim_path)
    print(f"loaded {len(rows)} prompts from {stim_path}")

    # 2. Load model
    print(f"\nloading {args.model}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model,
                                          token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    # Resolve high/low word token ids from first stim row (per-pair words)
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]
    print(f"  LD readout: '{high_w}'(id={high_id}) - '{low_w}'(id={low_id})")
    layers = get_decoder_layers(model)
    L = args.layer
    attn = layers[L].self_attn
    o_proj = attn.o_proj
    n_heads = getattr(attn, "num_heads", None) or attn.config.num_attention_heads
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        head_dim = o_proj.in_features // n_heads
    print(f"  L{L}: n_heads={n_heads}  head_dim={head_dim}  "
          f"o_proj weight={tuple(o_proj.weight.shape)}")
    if args.primary_head >= n_heads:
        raise SystemExit(f"primary_head={args.primary_head} >= n_heads={n_heads}")

    # 3. Load SAE
    sae = load_sae(args.short, L, args.sae_width, args.sae_l0)
    d_attn = sae["W_enc"].shape[0]
    n_feats = sae["W_enc"].shape[1]
    # Gemma Scope -pt-att SAEs are on the PRE-o_proj concatenated head output
    # (z, dim = H*D), not on post-o_proj attn_out (dim = hidden_size).
    if d_attn != o_proj.in_features:
        raise SystemExit(f"SAE d_in={d_attn} != pre-o_proj d={o_proj.in_features}")

    # 4. Hook to capture pre-o_proj input + attn_out at last token
    captured = {"pre": None, "post": None}

    def pre_hook(module, args_):
        # args_[0] = pre-o_proj tensor shape (B, T, H*D)
        captured["pre"] = args_[0].detach().float().cpu().numpy()

    def post_hook(module, inputs, output):
        captured["post"] = output.detach().float().cpu().numpy()

    h_pre = o_proj.register_forward_pre_hook(pre_hook)
    h_post = o_proj.register_forward_hook(post_hook)

    # 5. Forward each prompt; collect last-token activations + LD
    n = len(rows)
    pre_arr = np.zeros((n, n_heads * head_dim), dtype=np.float32)  # (N, H*D)
    z_arr = np.zeros(n, dtype=np.float32)
    x_arr = np.zeros(n, dtype=np.float32)
    ld_arr = np.zeros(n, dtype=np.float32)
    print(f"\nforward × {n}...")
    t1 = time.time()
    try:
        with torch.inference_mode():
            for i, row in enumerate(rows):
                prompt = row.get("prompt") or row.get("text")
                if prompt is None:
                    raise SystemExit(f"row missing 'prompt' key: {row}")
                inp = tok(prompt, return_tensors="pt").to(model.device)
                out = model(**inp, use_cache=False)
                logits = out.logits[0, -1].float()
                ld_arr[i] = float(logits[high_id] - logits[low_id])
                pre_arr[i] = captured["pre"][0, -1]
                z_arr[i] = float(row.get("z_eff", row.get("z", float("nan"))))
                x_arr[i] = float(row.get("x", row.get("x_value", float("nan"))))
                if (i + 1) % 25 == 0 or i == n - 1:
                    rate = (i + 1) / max(1e-3, time.time() - t1)
                    print(f"  {i+1}/{n}  {rate:.1f} p/s", flush=True)
    finally:
        h_pre.remove()
        h_post.remove()

    # 6. SAE encode pre-o_proj z → features
    pre_feat, post_feat = encode_sae(pre_arr, sae)
    n_active = (post_feat > 0).any(axis=0).sum()
    avg_l0 = (post_feat > 0).sum(axis=1).mean()
    print(f"\nSAE: {n_active}/{n_feats} features ever-active, "
          f"avg L0/prompt={avg_l0:.1f}")

    # 7. Per-feature correlations (only active features) — vs z, vs LD
    valid = np.isfinite(z_arr) & np.isfinite(x_arr)
    z_v = z_arr[valid]
    ld_v = ld_arr[valid]
    feat_v = post_feat[valid]
    active_mask = (feat_v > 0).any(axis=0)
    feat_idx = np.where(active_mask)[0]
    z_corrs = np.zeros(len(feat_idx), dtype=np.float32)
    ld_corrs = np.zeros(len(feat_idx), dtype=np.float32)
    for j, fi in enumerate(feat_idx):
        a = feat_v[:, fi]
        if a.std() < 1e-9:
            z_corrs[j] = 0.0
            ld_corrs[j] = 0.0
        else:
            z_corrs[j] = float(np.corrcoef(a, z_v)[0, 1])
            ld_corrs[j] = float(np.corrcoef(a, ld_v)[0, 1])
    order = np.argsort(-np.abs(z_corrs))
    top_idx = feat_idx[order[:args.top_n]]
    top_corrs = z_corrs[order[:args.top_n]]
    top_ld_corrs = ld_corrs[order[:args.top_n]]
    print(f"  baseline: r(LD, z) = {float(np.corrcoef(ld_v, z_v)[0,1]):+.3f}  "
          f"<LD>={ld_v.mean():+.2f}")

    # 8. Per-head attribution for top features. Two metrics:
    #   (a) magnitude_share: |mean(z_h @ W_enc[h_slice, i])| normalized over heads
    #       — what fraction of |attribution| comes from each head on average.
    #   (b) z_corr: r(z_h @ W_enc[h_slice, i], z) across prompts
    #       — the more relevant question: does this head's *contribution* to
    #         feature i track z? (matches Phase 2F per-head correlation framing)
    # SAE input is z = concat([h0, ..., hN-1]) (N, H*D); pre[i] = z @ W_enc[:, i].
    head_attr = np.zeros((args.top_n, n_heads), dtype=np.float32)
    head_zcorr = np.zeros((args.top_n, n_heads), dtype=np.float32)
    head_contrib = {}  # feat_idx -> (n_heads, N) per-prompt contribution
    for j, fi in enumerate(top_idx):
        wenc_i = sae["W_enc"][:, fi]
        contribs = np.zeros((n_heads, len(z_v)), dtype=np.float32)
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            pre_h = pre_arr[valid][:, sl] @ wenc_i[sl]  # (N,)
            contribs[h] = pre_h
            head_attr[j, h] = float(np.mean(np.abs(pre_h)))
            if pre_h.std() > 1e-9:
                head_zcorr[j, h] = float(np.corrcoef(pre_h, z_v)[0, 1])
        head_contrib[int(fi)] = contribs

    head_attr_norm = head_attr / np.clip(head_attr.sum(axis=1, keepdims=True),
                                          1e-12, None)

    # 9. Save results
    results = {
        "model": args.model, "short": args.short, "layer": L,
        "primary_head": args.primary_head, "feature": args.feature, "k": args.k,
        "n_prompts": int(valid.sum()),
        "sae": {"width": args.sae_width, "l0": args.sae_l0,
                 "n_features": int(n_feats), "n_active": int(n_active),
                 "avg_l0_per_prompt": float(avg_l0)},
        "primary_head_attribution_in_top": [],
        "top_features": [],
    }
    results["baseline_LD_z_corr"] = float(np.corrcoef(ld_v, z_v)[0, 1])
    results["baseline_LD_mean"] = float(ld_v.mean())
    for j, (fi, rc, lc) in enumerate(zip(top_idx.tolist(),
                                            top_corrs.tolist(),
                                            top_ld_corrs.tolist())):
        ph_share = float(head_attr_norm[j, args.primary_head])
        ph_zcorr = float(head_zcorr[j, args.primary_head])
        argmax_attr_h = int(head_attr_norm[j].argmax())
        argmax_zcorr_h = int(np.argmax(np.abs(head_zcorr[j])))
        results["top_features"].append({
            "rank": j, "feat_idx": fi, "r_z": rc, "r_ld": lc,
            "primary_head_share": ph_share,
            "primary_head_zcorr": ph_zcorr,
            "argmax_attr_head": argmax_attr_h,
            "argmax_zcorr_head": argmax_zcorr_h,
            "head_attribution": head_attr_norm[j].tolist(),
            "head_zcorr": head_zcorr[j].tolist(),
        })
        if ph_share > 1.0 / n_heads:
            results["primary_head_attribution_in_top"].append(fi)

    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")

    # Save residuals NPZ for downstream analysis (ablation, k-sweep correlation)
    out_npz = REPO / "results" / f"{out_name}_residuals.npz"
    active_feats = np.where(active_mask)[0]
    post_active = post_feat[:, active_feats].astype(np.float16)
    np.savez(out_npz,
             pre=pre_arr.astype(np.float16),     # (N, H*D)
             post_active=post_active,             # (N, n_active)
             active_feats=active_feats.astype(np.int32),
             z=z_arr.astype(np.float32),
             x=x_arr.astype(np.float32),
             ld=ld_arr.astype(np.float32),
             top_feats=top_idx.astype(np.int32),
             top_corrs=top_corrs.astype(np.float32),
             top_ld_corrs=top_ld_corrs.astype(np.float32),
             head_attr=head_attr_norm.astype(np.float32),
             head_zcorr=head_zcorr.astype(np.float32),
             n_heads=np.int32(n_heads),
             head_dim=np.int32(head_dim),
             high_id=np.int32(high_id),
             low_id=np.int32(low_id))
    print(f"wrote {out_npz} ({out_npz.stat().st_size/1e6:.1f} MB)")

    # 10. Plot — 3 subplots: feature z-corrs, attribution magnitude, head z-corr
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))

    ax = axes[0]
    ax.scatter(np.arange(len(z_corrs)), z_corrs, s=8, alpha=0.4, color="C7",
                label="all active features")
    for j, (fi, rc) in enumerate(zip(top_idx, top_corrs)):
        ax.scatter([order[j]], [rc], s=80, color="C0" if rc > 0 else "C3",
                    edgecolor="black", linewidth=0.6, zorder=5)
        if abs(rc) > 0.3 or j < 5:
            ax.annotate(str(fi), (order[j], rc), fontsize=7,
                         xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("feature rank by |r(feat, z)|")
    ax.set_ylabel(f"r(feat, z_eff)  on {args.feature} k={args.k}")
    ax.set_title(f"{args.short}  L{L} attn_out SAE — z-correlation per feature\n"
                  f"{int(n_active)} active / {n_feats} total")
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(head_attr_norm.T, aspect="auto", cmap="viridis",
                    vmin=0, vmax=head_attr_norm.max())
    ax.set_xticks(np.arange(args.top_n))
    ax.set_xticklabels([f"{fi}\n({rc:+.2f})" for fi, rc in zip(top_idx, top_corrs)],
                        fontsize=7, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.axhline(args.primary_head + 0.5, color="red", lw=1.0)
    ax.axhline(args.primary_head - 0.5, color="red", lw=1.0)
    ax.set_xlabel("top z-correlated features (idx, r_z)")
    ax.set_ylabel("head")
    ax.set_title(f"|attribution| share (red lines = primary head H{args.primary_head})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="share of |attribution|")

    ax = axes[2]
    vmax = max(0.05, float(np.max(np.abs(head_zcorr))))
    im2 = ax.imshow(head_zcorr.T, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=+vmax)
    ax.set_xticks(np.arange(args.top_n))
    ax.set_xticklabels([f"{fi}\n({rc:+.2f})" for fi, rc in zip(top_idx, top_corrs)],
                        fontsize=7, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.axhline(args.primary_head + 0.5, color="red", lw=1.0)
    ax.axhline(args.primary_head - 0.5, color="red", lw=1.0)
    ax.set_xlabel("top z-correlated features")
    ax.set_title(f"r(head_h's contribution to feat, z) — signed; "
                  f"red = +z, blue = −z")
    fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.04, label="r per head")

    fig.suptitle(f"Phase 2L — {args.short} L{L} attn_out SAE on {args.feature} k={args.k}",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
