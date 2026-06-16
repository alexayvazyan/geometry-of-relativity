"""Phase 2E — wider manifold α sweep on the same prompts.

Runs baseline + manifold(α) for α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
at the canonical late layer for the model. Saves r_lz, r_lx, std, ⟨LD⟩ per α.

Usage:
  python scripts/p2e_alpha_sweep.py --model gemma2-2b --pair height --k 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR = REPO.parent / "geometry-of-relativity"
sys.path.insert(0, str(GOR / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

# Reuse helpers from p2e_residual_interventions.
sys.path.insert(0, str(REPO / "scripts"))
from p2e_residual_interventions import (
    LATE_LAYER, MODEL_ID, get_decoder_layers, build_primal_z,
    build_cell_mean_lookup, manifold_delta, run_intervention, safe_pearson,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[αsweep] loading {MODEL_ID[args.model]}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID[args.model])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID[args.model], dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    L = LATE_LAYER[args.model]

    attn_npz = REPO / "results" / "p2_attn" / args.model / f"{args.pair}_k{args.k}.npz"
    d = np.load(attn_npz, allow_pickle=True)
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    low_word = all_trials[0]["low_word"]
    high_word = all_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)
    print(f"[αsweep] words: low={low_word!r}({low_id}) high={high_word!r}({high_id})",
          flush=True)

    h_at_L = d["residuals"][:, L, :].astype(np.float64)
    z_arr = np.array([t["z"] for t in all_trials], dtype=np.float64)
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)

    primal = build_primal_z(h_at_L, z_eff_arr, cs_arr)
    M, counts, x_edges, z_edges, x_marg = build_cell_mean_lookup(
        h_at_L, x_arr, z_arr, cs_arr,
    )
    deltas = manifold_delta(M, counts, x_edges, z_edges, x_marg, x_arr, z_arr)

    test_mask = cs_arr != 0
    rng = np.random.default_rng(0)
    test_idx = np.where(test_mask)[0]
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_eff_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    deltas_test = deltas[test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
    print(f"[αsweep] {args.model}/{args.pair} k={args.k}: n={n_test} max_seq={max_seq}",
          flush=True)

    out = {"model": args.model, "pair": args.pair, "k": args.k,
            "layer": L, "results": {}}

    # Build primal direction tensor for proj_out mode
    d_unit_np = primal / max(np.linalg.norm(primal), 1e-9)
    primal_t = torch.from_numpy(d_unit_np).to(args.device).to(model.dtype)

    for alpha in [None] + args.alphas:
        t1 = time.time()
        if alpha is None:
            mode_name = "baseline"
            ld = run_intervention(
                model, tok, trials, mode="baseline", layer=L,
                high_id=high_id, low_id=low_id,
                batch_size=bs, max_seq=max_seq, device=args.device,
            )
            r_z = safe_pearson(z_eff_test, ld)
            r_x = safe_pearson(x_test, ld)
            out["results"][mode_name] = {
                "alpha": alpha,
                "r_ld_zeff": r_z, "r_ld_x": r_x,
                "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
                "n": n_test,
            }
            print(f"  {mode_name:<18} α={'-':>5}  "
                  f"r_x={r_x:+.3f} r_z={r_z:+.3f} "
                  f"⟨LD⟩={ld.mean():+.2f} std={ld.std():.2f}  "
                  f"({time.time()-t1:.1f}s)", flush=True)
            continue

        # Manifold (dual)
        mode_name = f"manifold_a{alpha:.2f}"
        ld = run_intervention(
            model, tok, trials, mode="manifold", layer=L,
            deltas=deltas_test, alpha=alpha,
            high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        r_z = safe_pearson(z_eff_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][mode_name] = {
            "alpha": alpha,
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        print(f"  {mode_name:<18} α={alpha:.2f}  "
              f"r_x={r_x:+.3f} r_z={r_z:+.3f} "
              f"⟨LD⟩={ld.mean():+.2f} std={ld.std():.2f}  "
              f"({time.time()-t1:.1f}s)", flush=True)

        # Primal (proj_out)
        t2 = time.time()
        mode_name = f"primal_a{alpha:.2f}"
        ld = run_intervention(
            model, tok, trials, mode="proj_out", layer=L,
            direction=primal_t, alpha=alpha,
            high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        r_z = safe_pearson(z_eff_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][mode_name] = {
            "alpha": alpha,
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        print(f"  {mode_name:<18} α={alpha:.2f}  "
              f"r_x={r_x:+.3f} r_z={r_z:+.3f} "
              f"⟨LD⟩={ld.mean():+.2f} std={ld.std():.2f}  "
              f"({time.time()-t2:.1f}s)", flush=True)

    out_path = REPO / "results" / f"p2e_alpha_sweep_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
