"""Phase 1e — α-sweep for manifold_shift and per-pair proj_out.

For α ∈ {0.25, 0.5, 0.75, 1.0, 1.25} and each pair, run two interventions:

  manifold(α):  h ← h + α · Δ_i        (per-prompt cell-mean shift)
  proj_out(α):  h ← h - α · (h·d̂_p) d̂_p   (per-pair rank-1 partial projection)

Where Δ_i = M(x_i, z=0) - M(x_i, z_i) and d̂_p = unit(primal_z[pair]),
both computed on the train fold (cell_seed ∈ {0..4}).

Goal: identify α* per pair × method where r(LD, z) ≈ 0, and characterize
how r(LD, x) trades off across the sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from p1_text_ablation import (  # noqa: E402
    GOR_ROOT, ALL_PAIRS, MODEL_ID, MODEL_SHORT, LATE_LAYER, unit,
)
from p1d_manifold_ablation import (  # noqa: E402
    build_cell_mean_lookup, manifold_displacements,
    random_displacements_matched,
    eval_LD_with_per_prompt_shift, eval_LD_with_static_hook,
    eval_LD_no_hook, load_pair_trials,
)


# ------------------------- α-parameterized hooks -------------------------

def make_partial_proj_hook(d_t: torch.Tensor, alpha: float):
    """h ← h - α (h·d̂) d̂"""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h * d_t).sum(-1, keepdim=True)
        h = h - alpha * proj * d_t
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_alpha_shift_hook(state: dict, alpha: float):
    """h ← h + α · state['delta'] (per-prompt shift, broadcast over T)."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = state["delta"]  # (B, d)
        h = h + alpha * delta[:, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def eval_LD_with_alpha_shift(model, tok, prompts, displacements, alpha,
                              hi_id, lo_id, layer, batch_size, max_seq):
    decoder_layers = model.model.layers
    state = {"delta": None}
    handle = decoder_layers[layer].register_forward_hook(
        make_alpha_shift_hook(state, alpha))
    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            b = prompts[b0:b0 + batch_size]
            prompt_strs = [t["prompt"] for t in b]
            batch_disp = torch.tensor(displacements[b0:b0 + batch_size],
                                       dtype=torch.bfloat16,
                                       device=model.device)
            state["delta"] = batch_disp
            enc = tok(prompt_strs, return_tensors="pt", padding="max_length",
                       max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                use_cache=False).logits[:, -1, :].float()
            out[b0:b0 + len(b)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        handle.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-name", default="p1e_alpha_sweep")
    args = ap.parse_args()

    pairs = list(args.pairs)
    alphas = sorted(set(args.alphas))
    L = LATE_LAYER
    train_seeds = [0, 1, 2, 3, 4]
    test_seeds = [5, 6, 7, 8, 9]

    print(f"[p1e] loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    print(f"[p1e]   loaded in {time.time() - t0:.1f}s")

    results = {
        "model_id": MODEL_ID,
        "layer": L,
        "alphas": alphas,
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "per_pair": {},
    }

    for p in pairs:
        print(f"\n[p1e] === pair: {p} ===")
        lookup, primal_train = build_cell_mean_lookup(p, L, train_seeds)

        all_trials = load_pair_trials(p)
        seeds_arr = np.array([t["seed"] for t in all_trials])
        min_seed = int(seeds_arr.min())
        test_prompts = [t for t in all_trials
                         if (t["seed"] - min_seed) in test_seeds]
        print(f"[p1e]   held-out: {len(test_prompts)} prompts")

        first = test_prompts[0]
        hi_id = tok(first["high_word"], add_special_tokens=False).input_ids[0]
        lo_id = tok(first["low_word"], add_special_tokens=False).input_ids[0]

        Δ = manifold_displacements(test_prompts, lookup, target_z=0.0)
        zs = np.array([t["z"] for t in test_prompts], dtype=np.float64)
        xs = np.array([t["x"] for t in test_prompts], dtype=np.float64)
        d_t = torch.tensor(unit(primal_train), dtype=torch.bfloat16,
                            device=model.device)

        per_pair: dict = {"n_holdout": len(test_prompts), "alphas": alphas,
                           "manifold": {}, "proj_out": {}}

        # baseline (α=0 implicit)
        t1 = time.time()
        ld = eval_LD_no_hook(model, tok, test_prompts, hi_id, lo_id,
                              args.batch_size, args.max_seq)
        r_z = float(np.corrcoef(ld, zs)[0, 1])
        r_x = float(np.corrcoef(ld, xs)[0, 1])
        per_pair["baseline"] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1e]   baseline       r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
              flush=True)

        # α-sweep for both methods
        for alpha in alphas:
            # Manifold partial
            t1 = time.time()
            ld = eval_LD_with_alpha_shift(
                model, tok, test_prompts, Δ, alpha, hi_id, lo_id, L,
                args.batch_size, args.max_seq)
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            per_pair["manifold"][f"alpha_{alpha:.2f}"] = {
                "alpha": alpha,
                "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[p1e]   manifold α={alpha:.2f}  r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
                  f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
                  flush=True)

            # Per-pair proj_out partial
            t1 = time.time()
            ld = eval_LD_with_static_hook(
                model, tok, test_prompts,
                make_partial_proj_hook(d_t, alpha),
                hi_id, lo_id, L, args.batch_size, args.max_seq)
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            per_pair["proj_out"][f"alpha_{alpha:.2f}"] = {
                "alpha": alpha,
                "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[p1e]   proj_out α={alpha:.2f}  r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
                  f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
                  flush=True)

        results["per_pair"][p] = per_pair

    out_path = REPO / "results" / f"{args.out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[p1e] wrote {out_path}")


if __name__ == "__main__":
    main()
