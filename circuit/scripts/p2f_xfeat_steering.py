"""Phase 2F (cross-feature) — height-derived steering applied to other features.

Tests whether the steering direction / manifold lookup built from one
relativity feature (e.g., height) transfers to a different feature (e.g.,
weight). Two modes:

  mean-diff (proj-out)  — single 1D direction d_L = mean(h | z>+1) - mean(h | z<-1),
                          built on the REFERENCE feature, applied as proj-out
                          on the EVAL feature's prompts.
  manifold-zonly        — 1D z-only cell-mean lookup M[b_z] = mean h conditional
                          on z bin (collapsing across x), built on the REFERENCE
                          feature. Per-prompt delta = M[target_z_bin] - M[source_z_bin]
                          where bins are defined on z-score of the within-feature
                          z distribution. Avoids the (cm vs kg) x-axis mismatch.

Usage:
  python3 scripts/p2f_xfeat_steering.py --model gemma2-9b \
      --reference-pair height --eval-pair weight --k 15
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
    LATE_LAYER, MODEL_ID, build_primal_z, run_intervention, safe_pearson,
)


def build_z_only_lookup(h_at_L: np.ndarray, z: np.ndarray, cs: np.ndarray,
                         n_bins: int = 20):
    """1D cell-mean lookup M[b_z] = mean h | z in bin b_z (train fold only)."""
    train = cs == 0
    if train.sum() < 50:
        train = np.zeros_like(cs, dtype=bool)
        train[: len(cs) // 2] = True
    z_t = z[train]; h_t = h_at_L[train]
    z_edges = np.linspace(z_t.min() - 1e-6, z_t.max() + 1e-6, n_bins + 1)
    z_idx = np.clip(np.digitize(z_t, z_edges) - 1, 0, n_bins - 1)
    d_model = h_t.shape[-1]
    M = np.full((n_bins, d_model), np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    for i in range(len(h_t)):
        b = z_idx[i]
        if counts[b] == 0:
            M[b] = h_t[i].astype(np.float64)
        else:
            M[b] = (M[b] * counts[b] + h_t[i]) / (counts[b] + 1)
        counts[b] += 1
    return M, counts, z_edges


def z_only_delta(M, counts, z_edges, z_eval, z_train_min, z_train_max):
    """Per-prompt delta = M[target_b] - M[source_b] using z-percentile match.

    Maps eval z to a relative position in the train z range, then bins.
    Falls back to nearest populated bin if source is empty. Target is the
    populated bin nearest to z=0.
    """
    n_bins = M.shape[0]
    d = M.shape[1]
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    valid = counts > 0
    if not valid.any():
        return np.zeros((len(z_eval), d), dtype=np.float32)
    valid_idx = np.where(valid)[0]
    tgt_b = int(valid_idx[np.argmin(np.abs(z_centers[valid_idx]))])

    # Map eval z (in eval-feature units) to a fraction of train range, then bin.
    z_eval_norm = (z_eval - z_train_min) / max(1e-9, z_train_max - z_train_min)
    z_eval_norm = np.clip(z_eval_norm, 0.0, 1.0)
    z_eval_in_train_units = (
        z_train_min + z_eval_norm * (z_train_max - z_train_min)
    )
    eval_b = np.clip(np.digitize(z_eval_in_train_units, z_edges) - 1,
                      0, n_bins - 1)

    deltas = np.zeros((len(z_eval), d), dtype=np.float32)
    for i, b in enumerate(eval_b):
        b_use = int(b)
        if counts[b_use] == 0:
            b_use = int(valid_idx[np.argmin(np.abs(valid_idx - b_use))])
        deltas[i] = (M[tgt_b] - M[b_use]).astype(np.float32)
    return deltas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--reference-pair", default="height",
                    help="feature whose attn extract builds d_L and M")
    p.add_argument("--eval-pair", default="weight",
                    help="feature whose held-out prompts get steered")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--include-eval-native-baseline", action="store_true",
                    help="also build d_L from EVAL prompts as positive control")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[xfeat] loading {MODEL_ID[args.model]}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID[args.model])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID[args.model], dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    L = LATE_LAYER[args.model]

    # ---- REFERENCE feature: build d_L and z-only M ----
    ref_npz = REPO / "results" / "p2_attn" / args.model / f"{args.reference_pair}_k{args.k}.npz"
    if not ref_npz.exists():
        raise SystemExit(f"missing reference activations: {ref_npz}")
    ref_d = np.load(ref_npz, allow_pickle=True)
    ref_jsonl = REPO / "data" / "p2_shot_sweep" / f"{args.reference_pair}_k{args.k}.jsonl"
    ref_trials = [json.loads(l) for l in ref_jsonl.open()]

    h_ref = ref_d["residuals"][:, L, :].astype(np.float64)
    z_eff_ref = np.array([t["z_eff"] for t in ref_trials], dtype=np.float64)
    z_ref = np.array([t["z"] for t in ref_trials], dtype=np.float64)
    cs_ref = np.array([t["cell_seed"] for t in ref_trials], dtype=np.int32)

    primal_ref = build_primal_z(h_ref, z_eff_ref, cs_ref)
    primal_ref_unit = primal_ref / max(np.linalg.norm(primal_ref), 1e-9)
    M_z, counts_z, z_edges_z = build_z_only_lookup(h_ref, z_ref, cs_ref)
    print(f"[xfeat] ref={args.reference_pair}: ||d_L||={np.linalg.norm(primal_ref):.2f}  "
          f"M_z coverage={(counts_z > 0).sum()}/{len(counts_z)} bins", flush=True)
    z_train_min, z_train_max = float(z_ref[cs_ref == 0].min()), float(z_ref[cs_ref == 0].max())

    # Optional: eval-native d_L as positive control
    primal_eval_unit = None
    if args.include_eval_native_baseline:
        eval_npz_ref = REPO / "results" / "p2_attn" / args.model / f"{args.eval_pair}_k{args.k}.npz"
        if eval_npz_ref.exists():
            eval_d_ref = np.load(eval_npz_ref, allow_pickle=True)
            eval_jsonl_ref = REPO / "data" / "p2_shot_sweep" / f"{args.eval_pair}_k{args.k}.jsonl"
            eval_trials_ref = [json.loads(l) for l in eval_jsonl_ref.open()]
            h_eval_ref = eval_d_ref["residuals"][:, L, :].astype(np.float64)
            z_eff_eval_ref = np.array([t["z_eff"] for t in eval_trials_ref], dtype=np.float64)
            cs_eval_ref = np.array([t["cell_seed"] for t in eval_trials_ref], dtype=np.int32)
            primal_eval = build_primal_z(h_eval_ref, z_eff_eval_ref, cs_eval_ref)
            primal_eval_unit = primal_eval / max(np.linalg.norm(primal_eval), 1e-9)
            print(f"[xfeat] eval-native d_L||={np.linalg.norm(primal_eval):.2f}",
                  flush=True)

    # ---- EVAL feature: held-out prompts ----
    eval_jsonl = REPO / "data" / "p2_shot_sweep" / f"{args.eval_pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in eval_jsonl.open()]
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    z_arr = np.array([t["z"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)
    low_word = all_trials[0]["low_word"]
    high_word = all_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    test_mask = cs_arr != 0
    rng = np.random.default_rng(0)
    test_idx = np.where(test_mask)[0]
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_eff_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    z_test_for_lookup = z_arr[test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
    print(f"[xfeat] eval={args.eval_pair}: n={n_test} max_seq={max_seq}  "
          f"high={high_word!r}({high_id}) low={low_word!r}({low_id})",
          flush=True)

    # Z-only manifold deltas (uses ref M_z; z_test in eval-feature units)
    manifold_deltas_test = z_only_delta(
        M_z, counts_z, z_edges_z, z_test_for_lookup, z_train_min, z_train_max,
    )
    primal_t = torch.from_numpy(primal_ref_unit).to(args.device).to(model.dtype)
    primal_eval_t = (
        torch.from_numpy(primal_eval_unit).to(args.device).to(model.dtype)
        if primal_eval_unit is not None else None
    )

    out = {
        "model": args.model,
        "reference_pair": args.reference_pair,
        "eval_pair": args.eval_pair,
        "k": args.k,
        "layer": L,
        "n_test": n_test,
        "results": {},
    }

    def record(mode_name, alpha, ld, t_start):
        r_z = safe_pearson(z_eff_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][mode_name] = {
            "alpha": alpha,
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        a_label = "-" if alpha is None else f"{alpha:.2f}"
        print(f"  {mode_name:<28} α={a_label:>5}  "
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

        # Mean-diff direction transfer (REFERENCE-derived)
        t1 = time.time()
        ld = run_intervention(
            model, tok, trials, mode="proj_out", layer=L,
            direction=primal_t, alpha=alpha,
            high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        record(f"meandiff_xfeat_a{alpha:.2f}", alpha, ld, t1)

        # Manifold (z-only) lookup transfer
        t1 = time.time()
        ld = run_intervention(
            model, tok, trials, mode="manifold", layer=L,
            deltas=manifold_deltas_test, alpha=alpha,
            high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        record(f"manifold_zonly_xfeat_a{alpha:.2f}", alpha, ld, t1)

        # Optional: eval-native d_L positive control
        if primal_eval_t is not None:
            t1 = time.time()
            ld = run_intervention(
                model, tok, trials, mode="proj_out", layer=L,
                direction=primal_eval_t, alpha=alpha,
                high_id=high_id, low_id=low_id,
                batch_size=bs, max_seq=max_seq, device=args.device,
            )
            record(f"meandiff_native_a{alpha:.2f}", alpha, ld, t1)

    out_path = (REPO / "results"
                / f"p2f_xfeat_steering_{args.model}_ref{args.reference_pair}"
                  f"_eval{args.eval_pair}_k{args.k}.json")
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
