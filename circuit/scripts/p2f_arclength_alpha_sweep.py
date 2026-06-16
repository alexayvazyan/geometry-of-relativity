"""Phase 2F — α as a flow parameter (arc-length manifold transport).

Replaces the chord interpretation of α (h ← h + α · Δ_chord) with arc-length
parameterisation: α ∈ [0, 1] maps to a position fraction along each prompt's
populated-bin path source→target. At α = 0.5 the residual is shifted to a
point ON the polyline approximation of the manifold (interpolated between
two adjacent populated cell means), not halfway off it.

Modes saved per α in the JSON:
  baseline             — no intervention (α=None)
  manifold_a{α}        — Phase 2E chord (h + α · Δ_chord)
  arclength_a{α}       — Phase 2F arc-length flow (h + Δ(α along path))
  primal_a{α}          — Phase 2E proj_out

Output JSON:
  results/p2f_arclength_alpha_sweep_<model>_<pair>_k<k>.json

Usage:
  python3 scripts/p2f_arclength_alpha_sweep.py --model gemma2-9b --pair height --k 15
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
from p2f_iterative_manifold import (  # noqa: E402
    precompute_arclength_paths, arclength_delta_at_alpha, arclength_path_summary,
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
    p.add_argument("--skip-chord", action="store_true")
    p.add_argument("--skip-primal", action="store_true")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[2F arc] loading {MODEL_ID[args.model]}...", flush=True)
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
    if not attn_npz.exists():
        raise SystemExit(f"missing reference activations: {attn_npz}")
    d_npz = np.load(attn_npz, allow_pickle=True)
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    low_word = all_trials[0]["low_word"]
    high_word = all_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    h_at_L = d_npz["residuals"][:, L, :].astype(np.float64)
    z_arr = np.array([t["z"] for t in all_trials], dtype=np.float64)
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)

    primal = build_primal_z(h_at_L, z_eff_arr, cs_arr)
    M, counts, x_edges, z_edges, x_marg = build_cell_mean_lookup(
        h_at_L, x_arr, z_arr, cs_arr,
    )
    chord_deltas = manifold_delta(M, counts, x_edges, z_edges, x_marg, x_arr, z_arr)
    paths = precompute_arclength_paths(
        M, counts, x_edges, z_edges, x_marg, x_arr, z_arr,
    )

    test_mask = cs_arr != 0
    rng = np.random.default_rng(0)
    test_idx = np.where(test_mask)[0]
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_eff_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    chord_deltas_test = chord_deltas[test_idx]
    paths_test = [paths[int(i)] for i in test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4

    arc_summary = arclength_path_summary(paths_test)
    print(f"[2F arc] {args.model}/{args.pair} k={args.k}: n={n_test} "
          f"max_seq={max_seq}", flush=True)
    print(f"[2F arc] test-set path summary: {arc_summary}", flush=True)

    out = {
        "model": args.model, "pair": args.pair, "k": args.k, "layer": L,
        "arclength_path_summary": arc_summary,
        "results": {},
    }

    primal_unit = primal / max(np.linalg.norm(primal), 1e-9)
    primal_t = torch.from_numpy(primal_unit).to(args.device).to(model.dtype)
    d_model = chord_deltas.shape[-1]

    def record(mode_name, alpha, ld, t_start, extra=None):
        r_z = safe_pearson(z_eff_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][mode_name] = {
            "alpha": alpha,
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        if extra is not None:
            out["results"][mode_name].update(extra)
        a_label = "-" if alpha is None else f"{alpha:.2f}"
        print(f"  {mode_name:<22} α={a_label:>5}  "
              f"r_x={r_x:+.3f} r_z={r_z:+.3f} "
              f"⟨LD⟩={ld.mean():+.2f} std={ld.std():.2f}  "
              f"({time.time()-t_start:.1f}s)", flush=True)

    for alpha in [None] + args.alphas:
        if alpha is None:
            t1 = time.time()
            ld = run_intervention(
                model, tok, trials, mode="baseline", layer=L,
                high_id=high_id, low_id=low_id,
                batch_size=bs, max_seq=max_seq, device=args.device,
            )
            record("baseline", None, ld, t1)
            continue

        if not args.skip_chord:
            t1 = time.time()
            ld = run_intervention(
                model, tok, trials, mode="manifold", layer=L,
                deltas=chord_deltas_test, alpha=alpha,
                high_id=high_id, low_id=low_id,
                batch_size=bs, max_seq=max_seq, device=args.device,
            )
            record(f"manifold_a{alpha:.2f}", alpha, ld, t1,
                    extra={"mean_delta_norm": float(
                        np.linalg.norm(alpha * chord_deltas_test, axis=-1).mean())})

        # Arc-length: per-α deltas. Pass α=1 to run_intervention so the hook
        # adds the delta unchanged; α scaling is baked into the delta itself.
        t1 = time.time()
        arc_deltas = arclength_delta_at_alpha(paths_test, alpha, d_model)
        ld = run_intervention(
            model, tok, trials, mode="manifold", layer=L,
            deltas=arc_deltas, alpha=1.0,
            high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        diff_to_chord = float(np.linalg.norm(
            arc_deltas - alpha * chord_deltas_test, axis=-1).mean())
        record(f"arclength_a{alpha:.2f}", alpha, ld, t1,
                extra={
                    "mean_delta_norm": float(
                        np.linalg.norm(arc_deltas, axis=-1).mean()),
                    "mean_diff_to_chord_at_alpha": diff_to_chord,
                })

        if not args.skip_primal:
            t1 = time.time()
            ld = run_intervention(
                model, tok, trials, mode="proj_out", layer=L,
                direction=primal_t, alpha=alpha,
                high_id=high_id, low_id=low_id,
                batch_size=bs, max_seq=max_seq, device=args.device,
            )
            record(f"primal_a{alpha:.2f}", alpha, ld, t1)

    out_path = (REPO / "results"
                / f"p2f_arclength_alpha_sweep_{args.model}_{args.pair}_k{args.k}.json")
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
