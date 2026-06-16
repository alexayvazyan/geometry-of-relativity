"""Phase 2E (cross-k) — α-sweep for primal/dual steering with reference k≠eval k.

Variant of p2e_alpha_sweep.py that builds the primal direction and manifold
lookup from one stimulus distribution (--reference-k, default 15) but applies
the resulting interventions to a different one (--eval-k). This lets us ask
whether the *same* primal/dual steering trajectories that work at k=15 still
shift the LD readout when the prompt is in comparator (k=1) or early-graded
(k=5) regimes.

The reference activations come from `results/p2_attn/<short>/<pair>_k<ref>.npz`.
The eval prompts come from `data/p2_shot_sweep/<pair>_k<eval>.jsonl`.

Output JSON has the same shape as p2e_alpha_sweep so p2v_phase_trajectory.py
can ingest both manifold_a* and primal_a* keys unchanged. Filename is
`p2e_alpha_sweep_<short>_<pair>_k<eval>_refk<ref>.json` to avoid colliding
with the same-k sweep.

Usage:
  python3 scripts/p2e_xk_alpha_sweep.py --model gemma2-2b --pair height \
      --reference-k 15 --eval-k 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR = REPO.parent / "geometry-of-relativity"
sys.path.insert(0, str(GOR / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from p2e_residual_interventions import (  # noqa: E402
    LATE_LAYER, MODEL_ID, build_primal_z, build_cell_mean_lookup,
    manifold_delta, run_intervention, safe_pearson,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--reference-k", type=int, default=15,
                    help="k whose activations build primal direction + manifold "
                         "lookup (frozen across eval-k)")
    p.add_argument("--eval-k", type=int, required=True,
                    help="k of the held-out evaluation prompts")
    p.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[xk] loading {MODEL_ID[args.model]}...", flush=True)
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
    print(f"[xk] L={L}  ref_k={args.reference_k}  eval_k={args.eval_k}", flush=True)

    # ---- Build primal direction + manifold lookup from REFERENCE k ----
    ref_npz = (REPO / "results" / "p2_attn" / args.model
               / f"{args.pair}_k{args.reference_k}.npz")
    if not ref_npz.exists():
        raise SystemExit(f"missing reference activations: {ref_npz}")
    ref_d = np.load(ref_npz, allow_pickle=True)
    ref_jsonl = (REPO / "data" / "p2_shot_sweep"
                 / f"{args.pair}_k{args.reference_k}.jsonl")
    ref_trials = [json.loads(l) for l in ref_jsonl.open()]
    assert len(ref_trials) == len(ref_d["ld"])

    h_ref = ref_d["residuals"][:, L, :].astype(np.float64)
    z_ref = np.array([t["z"] for t in ref_trials], dtype=np.float64)
    z_eff_ref = np.array([t["z_eff"] for t in ref_trials], dtype=np.float64)
    x_ref = np.array([t["x"] for t in ref_trials], dtype=np.float64)
    cs_ref = np.array([t["cell_seed"] for t in ref_trials], dtype=np.int32)

    primal = build_primal_z(h_ref, z_eff_ref, cs_ref)
    d_unit_np = primal / max(np.linalg.norm(primal), 1e-9)
    M_ref, counts_ref, x_edges_ref, z_edges_ref, x_marg_ref = (
        build_cell_mean_lookup(h_ref, x_ref, z_ref, cs_ref))
    print(f"[xk] primal||={np.linalg.norm(primal):.2f}  M coverage="
          f"{(counts_ref > 0).sum()}/{counts_ref.size} cells",
          flush=True)

    # ---- Load EVAL prompts (cell_seed!=0 holdout, like in-k sweep) ----
    low_word = ref_trials[0]["low_word"]
    high_word = ref_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    eval_jsonl = (REPO / "data" / "p2_shot_sweep"
                  / f"{args.pair}_k{args.eval_k}.jsonl")
    all_trials = [json.loads(l) for l in eval_jsonl.open()]
    z_arr = np.array([t["z"] for t in all_trials], dtype=np.float64)
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)

    # Per-prompt manifold delta is computed on EVAL covariates but using
    # REFERENCE-k bin grid + cell means. So we're shifting eval prompts
    # along the trajectory the model expressed at reference k.
    deltas = manifold_delta(M_ref, counts_ref, x_edges_ref, z_edges_ref,
                            x_marg_ref, x_arr, z_arr)

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
    print(f"[xk] eval n={n_test} max_seq={max_seq}  "
          f"r(z,x)={np.corrcoef(z_eff_arr[test_idx], x_arr[test_idx])[0,1]:+.3f}",
          flush=True)

    primal_t = torch.from_numpy(d_unit_np).to(args.device).to(model.dtype)

    out = {"model": args.model, "pair": args.pair,
            "k": args.eval_k, "reference_k": args.reference_k,
            "layer": L, "results": {}}

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

        # Primal (proj_out) using REFERENCE-derived direction
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

    out_path = (REPO / "results"
                / f"p2e_alpha_sweep_{args.model}_{args.pair}"
                  f"_k{args.eval_k}_refk{args.reference_k}.json")
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
