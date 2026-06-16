"""Phase 2L — causal feature-ablation: zero a single SAE feature in z, re-forward, measure r(LD, z) drop.

For each top z-correlated feature i:
  1. Compute per-prompt activation post[i] from saved residuals NPZ.
  2. Re-forward each prompt with a forward_pre_hook on L1.o_proj that subtracts
     post[i] * W_dec[i] from the pre-o_proj input.
  3. Measure r(LD_ablated, z) — drop from baseline = causal contribution.

Caveat: this is a clean ablation only insofar as the SAE reconstruction error
is small. For features with high activation it's a faithful linear ablation
of "this feature's z-encoding contribution".

Output: results/<base>_ablate.json — per-feature: baseline r(LD,z), r_post,
         delta_r, delta_LD_mean, n_prompts_active.

Usage:
  python p2l_feature_ablate.py --short gemma2-2b --feature height --k 15 \\
      --layer 1 --top-n 5
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


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers"),
                  ("model", "language_model", "layers")]:
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
    ap.add_argument("--short", required=True)
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=5,
                    help="ablate this many top z-features (one at a time)")
    ap.add_argument("--also-zero-head", type=int, default=None,
                    help="optional: also zero a single head h's slice in z "
                         "for each forward (uses no SAE features). When set, "
                         "ablates ALL of head h's contribution (Phase 2F-style "
                         "head zero-ablation, for comparison).")
    args = ap.parse_args()

    base = f"p2l_attn_sae_{args.short}_{args.feature}_k{args.k}"
    npz_path = REPO / "results" / f"{base}_residuals.npz"
    json_path = REPO / "results" / f"{base}.json"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path}")

    d = np.load(npz_path)
    j = json.loads(json_path.read_text())
    top_feats = d["top_feats"].astype(int)[:args.top_n]
    active_feats = d["active_feats"].astype(int)
    post_active = d["post_active"].astype(np.float32)  # (N, n_active)
    z = d["z"].astype(np.float32)
    high_id = int(d["high_id"])
    low_id = int(d["low_id"])
    n = len(z)
    print(f"loaded {n} prompts, {len(top_feats)} top features")

    # Load SAE for W_dec
    sae_l0 = args.sae_l0 or j["sae"]["l0"]
    repo = ("google/gemma-scope-2b-pt-att" if args.short.startswith("gemma2-2b")
             else "google/gemma-scope-9b-pt-att")
    fname = (f"layer_{args.layer}/width_{args.sae_width}/"
             f"average_l0_{sae_l0}/params.npz")
    sae_path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN"))
    sae = np.load(sae_path)
    W_dec = sae["W_dec"].astype(np.float32)  # (n_feats, H*D)

    # Load model
    model_id = j["model"]
    print(f"\nloading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    o_proj = layers[args.layer].self_attn.o_proj
    n_heads = layers[args.layer].self_attn.config.num_attention_heads
    head_dim = o_proj.in_features // n_heads

    # Reload stim prompts
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{args.feature}_k{args.k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    assert len(rows) == n

    # Pre-compute baseline LD for sanity check
    def fwd_LD(modify_fn=None):
        """modify_fn(pre, prompt_idx) -> modified pre tensor; or None for baseline."""
        ld = np.zeros(n, dtype=np.float32)
        state = {"i": 0}

        def pre_hook(module, args_):
            x = args_[0]
            if modify_fn is not None:
                x = modify_fn(x, state["i"])
            return (x,) + args_[1:]
        h = o_proj.register_forward_pre_hook(pre_hook, with_kwargs=False)
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    state["i"] = i
                    prompt = row["prompt"]
                    inp = tok(prompt, return_tensors="pt").to(model.device)
                    out = model(**inp, use_cache=False)
                    logits = out.logits[0, -1].float()
                    ld[i] = float(logits[high_id] - logits[low_id])
                    if (i + 1) % 100 == 0:
                        print(f"    {i+1}/{n}", flush=True)
        finally:
            h.remove()
        return ld

    # Sanity: baseline match
    print(f"\n=== baseline ===")
    t1 = time.time()
    ld_base = fwd_LD()
    base_r_z = float(np.corrcoef(ld_base, z)[0, 1])
    print(f"  baseline r(LD,z) = {base_r_z:+.3f}  <LD>={ld_base.mean():+.2f}  "
          f"({time.time()-t1:.0f}s)")

    results = {
        "model": model_id, "short": args.short, "feature": args.feature,
        "k": args.k, "layer": args.layer,
        "baseline_r_LD_z": base_r_z, "baseline_LD_mean": float(ld_base.mean()),
        "ablations": [],
    }

    # Per-feature ablation
    for feat_idx in top_feats:
        if feat_idx not in active_feats:
            continue
        active_pos = int(np.where(active_feats == feat_idx)[0][0])
        post_i = post_active[:, active_pos]  # per-prompt activation
        n_active_prompts = int((post_i > 0).sum())
        wdec_i = torch.tensor(W_dec[feat_idx], dtype=torch.bfloat16,
                                device=model.device)  # (H*D,)

        def make_ablate(post_arr):
            def fn(x, i):
                # subtract this feature's contribution at last token
                a = float(post_arr[i])
                if abs(a) < 1e-9:
                    return x
                xm = x.clone()
                xm[:, -1, :] = xm[:, -1, :] - a * wdec_i
                return xm
            return fn

        print(f"\n=== ablate feat {feat_idx} (active in {n_active_prompts}/{n}) ===")
        t1 = time.time()
        ld_abl = fwd_LD(make_ablate(post_i))
        r_z_abl = float(np.corrcoef(ld_abl, z)[0, 1])
        delta_r = r_z_abl - base_r_z
        delta_ld = float(ld_abl.mean() - ld_base.mean())
        print(f"  r(LD,z) {base_r_z:+.3f} → {r_z_abl:+.3f}  Δr={delta_r:+.3f}  "
              f"Δ<LD>={delta_ld:+.2f}  ({time.time()-t1:.0f}s)")
        results["ablations"].append({
            "feat_idx": int(feat_idx),
            "n_active_prompts": n_active_prompts,
            "r_LD_z_ablated": r_z_abl,
            "delta_r": delta_r,
            "delta_LD_mean": delta_ld,
        })

    # Optional head zero-ablation for comparison
    if args.also_zero_head is not None:
        h_idx = args.also_zero_head
        sl_lo = h_idx * head_dim
        sl_hi = (h_idx + 1) * head_dim
        def zero_head(x, i):
            xm = x.clone()
            xm[:, -1, sl_lo:sl_hi] = 0
            return xm
        print(f"\n=== zero head H{h_idx} ===")
        t1 = time.time()
        ld_zero = fwd_LD(zero_head)
        r_z_zero = float(np.corrcoef(ld_zero, z)[0, 1])
        print(f"  r(LD,z) {base_r_z:+.3f} → {r_z_zero:+.3f}  Δr={r_z_zero-base_r_z:+.3f}  "
              f"Δ<LD>={float(ld_zero.mean()-ld_base.mean()):+.2f}  "
              f"({time.time()-t1:.0f}s)")
        results["zero_head"] = {
            "head": h_idx,
            "r_LD_z": r_z_zero,
            "delta_r": r_z_zero - base_r_z,
            "delta_LD_mean": float(ld_zero.mean() - ld_base.mean()),
        }

    out_path = REPO / "results" / f"{base}_ablate.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
