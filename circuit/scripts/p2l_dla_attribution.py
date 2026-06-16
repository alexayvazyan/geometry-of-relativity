"""Phase 2L — Direct Logit Attribution (DLA) per SAE feature.

Estimates how much each top z-feature directly contributes to LD = logit(high) − logit(low),
via the linear path:
    feature_i activation × W_dec[i] × W_O.T × (W_U[high] − W_U[low])

This is the "direct" effect — ignores downstream MLP/attention. Still useful
because the linear contribution should be the dominant part of the readout
in early layers.

Reads results/<base>_residuals.npz produced by p2l_attn_sae_decompose.py.
Re-loads model only to get W_O and W_U (lm_head). The SAE W_dec is fetched
from the same Gemma Scope checkpoint as in the original run.

Output:
  results/<base>_dla.json
  per-feature: r_z, r_LD (correlations with z and LD activations),
               dla_scalar (how much 1 unit of feature → LD),
               dla_total (mean(post_i) × dla_scalar)

Usage:
  python p2l_dla_attribution.py --short gemma2-2b --feature height --k 15
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0", type=int, default=None,
                    help="if omitted, infer from the JSON result")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    base = f"p2l_attn_sae_{args.short}_{args.feature}_k{args.k}"
    npz_path = REPO / "results" / f"{base}_residuals.npz"
    json_path = REPO / "results" / f"{base}.json"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path} — run p2l_attn_sae_decompose first")

    d = np.load(npz_path)
    j = json.loads(json_path.read_text())
    pre = d["pre"].astype(np.float32)             # (N, H*D)
    z = d["z"].astype(np.float32)
    ld = d["ld"].astype(np.float32)
    top_feats = d["top_feats"].astype(int)
    top_corrs = d["top_corrs"].astype(float)
    top_ld_corrs = d["top_ld_corrs"].astype(float) if "top_ld_corrs" in d.files else None
    high_id = int(d["high_id"])
    low_id = int(d["low_id"])
    active_feats = d["active_feats"].astype(int)
    post_active = d["post_active"].astype(np.float32)  # (N, n_active)
    print(f"loaded {pre.shape[0]} prompts, {len(active_feats)} active features")
    print(f"baseline r(LD, z) = {float(np.corrcoef(ld, z)[0,1]):+.3f}")

    # Load SAE W_dec, b_dec
    sae_l0 = args.sae_l0 or j["sae"]["l0"]
    repo = ("google/gemma-scope-2b-pt-att" if args.short.startswith("gemma2-2b")
             else "google/gemma-scope-9b-pt-att")
    fname = f"layer_{args.layer}/width_{args.sae_width}/average_l0_{sae_l0}/params.npz"
    sae_path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN"))
    sae = np.load(sae_path)
    W_dec = sae["W_dec"].astype(np.float32)        # (n_feats, H*D)
    print(f"SAE W_dec={W_dec.shape}")

    # Load model — only need W_O and W_U
    model_id = j["model"]
    print(f"\nloading {model_id} (weights only)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu",
        token=os.environ.get("HF_TOKEN")).eval()
    layers = (model.model.layers if hasattr(model.model, "layers")
               else model.model.model.layers)
    W_O = layers[args.layer].self_attn.o_proj.weight.detach().float().numpy()
    # W_O shape: (hidden, H*D)
    W_U = model.lm_head.weight.detach().float().numpy()  # (vocab, hidden)
    LD_dir = W_U[high_id] - W_U[low_id]  # (hidden,)
    del model
    print(f"W_O={W_O.shape}  LD_dir={LD_dir.shape}  (hidden={W_O.shape[0]})")

    # Direct logit attribution
    # feature i contribution to attn_out per prompt: post[:, i] * W_dec[i] @ W_O.T (in hidden space)
    # contribution to LD: post[:, i] * (W_dec[i] @ W_O.T @ LD_dir)
    # scalar_i = W_dec[i] @ W_O.T @ LD_dir
    # Compute scalar for top features only.
    rows = []
    for rank, fi in enumerate(top_feats[:args.top_n]):
        wdec_i = W_dec[fi]                    # (H*D,)
        attn_contrib_dir = wdec_i @ W_O.T     # (hidden,)  contribution direction
        dla_scalar = float(attn_contrib_dir @ LD_dir)

        # Per-prompt activation: find this feature in active_feats
        if fi not in active_feats:
            continue
        active_pos = int(np.where(active_feats == fi)[0][0])
        post_i = post_active[:, active_pos]
        if post_i.std() < 1e-9:
            r_post_z = 0.0
            r_post_ld = 0.0
        else:
            r_post_z = float(np.corrcoef(post_i, z)[0, 1])
            r_post_ld = float(np.corrcoef(post_i, ld)[0, 1])
        # DLA per prompt = post_i * dla_scalar
        dla_per_prompt = post_i * dla_scalar
        # Variance of LD explained by this feature's DLA (linear regression slope)
        ld_centered = ld - ld.mean()
        dla_centered = dla_per_prompt - dla_per_prompt.mean()
        if dla_centered.std() < 1e-9:
            slope_to_ld = 0.0
        else:
            slope_to_ld = float(np.cov(dla_per_prompt, ld)[0, 1] / dla_per_prompt.var())
        rows.append({
            "rank": rank, "feat_idx": int(fi),
            "r_post_z": r_post_z, "r_post_ld": r_post_ld,
            "dla_scalar": dla_scalar,
            "dla_mean": float(dla_per_prompt.mean()),
            "dla_std": float(dla_per_prompt.std()),
            "slope_dla_to_ld": slope_to_ld,
        })

    out = {
        "model": model_id, "short": args.short, "feature": args.feature,
        "k": args.k, "layer": args.layer,
        "n_prompts": int(len(z)),
        "baseline_r_LD_z": float(np.corrcoef(ld, z)[0, 1]),
        "baseline_LD_mean": float(ld.mean()),
        "baseline_LD_std": float(ld.std()),
        "top_features_dla": rows,
    }
    out_path = REPO / "results" / f"{base}_dla.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    print()
    print(f"{'rank':>4} {'feat':>6} {'r_z':>7} {'r_ld':>7} {'dla_scalar':>11} "
          f"{'dla_mean':>10} {'slope_to_ld':>12}")
    for r in rows[:15]:
        print(f"{r['rank']:>4} {r['feat_idx']:>6} {r['r_post_z']:>+7.3f} "
              f"{r['r_post_ld']:>+7.3f} {r['dla_scalar']:>+11.4f} "
              f"{r['dla_mean']:>+10.4f} {r['slope_dla_to_ld']:>+12.4f}")


if __name__ == "__main__":
    main()
