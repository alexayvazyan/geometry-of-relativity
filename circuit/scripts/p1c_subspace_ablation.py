"""Phase 1c — rank-k subspace ablation.

Stack the 8 per-pair primal_z directions (cell_seed=0) at L20 of Gemma 2 2B.
SVD gives an orthonormal basis V (d × 8) ordered by singular value. For
k ∈ {1, 2, 4, 8}, project out (or mean-ablate) the top-k subspace.

Hooks (rank-k generalizations of Phase 1):
  proj_out_k:    h ← h − V_k V_k^T h
  mean_ablate_k: h ← h − V_k V_k^T h + V_k μ_k    (μ_k = E[V_k^T h])
  rand_proj_k:   same as proj_out_k but V_k is a random orthonormal basis

Tests the hypothesis: z is encoded in a low-rank subspace, not a single
direction. Phase 1 (k=1) suppressed r(LD,z) from 0.93 to 0.50 on
height/weight; if k=2 or k=4 brings it below 0.2, we have a clean
subspace ablation. If r_z plateaus despite increasing k, the residual
information is in the *curvature* of the manifold (z² and higher
moments), and we need on-manifold tangent ablation.

We re-use Phase 1's infra by importing the hook + eval helpers.
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
    GOR_ROOT, ALL_PAIRS, MODEL_ID, MODEL_SHORT, LATE_LAYER,
    primal_z_from_npz, unit, load_holdout_prompts, eval_LD_under_hook,
    load_wikitext_chunk, wikitext_loss_under_hook,
)


# ------------------------- subspace hook factory -------------------------

def make_subspace_hook(V_k_t: torch.Tensor, mode: str,
                        mu_k_t: torch.Tensor | None = None):
    """V_k_t: (d, k) orthonormal basis on the model device, bf16.
    mu_k_t: (k,) bias means on device; required for mean_ablate.
    """
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
        if mode == "proj_out":
            # h - V_k @ (V_k^T @ h)  in einsum form
            proj_coeffs = torch.einsum("btd,dk->btk", h, V_k_t)
            proj = torch.einsum("btk,dk->btd", proj_coeffs, V_k_t)
            h = h - proj
        elif mode == "mean_ablate":
            assert mu_k_t is not None
            proj_coeffs = torch.einsum("btd,dk->btk", h, V_k_t)
            proj = torch.einsum("btk,dk->btd", proj_coeffs, V_k_t)
            bias = torch.einsum("k,dk->d", mu_k_t, V_k_t)  # (d,)
            h = h - proj + bias
        else:
            raise ValueError(f"unknown subspace mode {mode}")
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


# ------------------------- random orthonormal -------------------------

def random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random (d, k) matrix with orthonormal columns via QR on a
    random Gaussian."""
    G = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(G)  # (d, k) when k <= d
    return Q


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"],
                    choices=ALL_PAIRS + ["all"])
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-wikitext", type=int, default=200)
    ap.add_argument("--out-name", default="p1c_subspace_ablation")
    args = ap.parse_args()

    pairs_eval = ALL_PAIRS if args.pairs == ["all"] else list(args.pairs)
    L = LATE_LAYER

    # 1. Build subspace from ALL 8 pair primals (the global z subspace), but
    # only EVALUATE on `pairs_eval`.
    print(f"[p1c] computing primal_z @ L{L} for all 8 pairs (cell_seed=0)...")
    primals: dict[str, np.ndarray] = {}
    for p in ALL_PAIRS:
        npz = (GOR_ROOT / "results" / "v11" / MODEL_SHORT / p
               / f"{MODEL_SHORT}_{p}_v11_residuals.npz")
        primals[p] = primal_z_from_npz(npz, L, cell_seed_filter=0)
    P = np.stack([primals[p] for p in ALL_PAIRS])  # (8, d)
    print(f"[p1c]   P shape: {P.shape}")

    # SVD: P = U Σ V^T where V is (d, 8)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    V = Vt.T  # (d, 8) orthonormal columns
    print(f"[p1c]   singular values: {S}")
    print(f"[p1c]   variance explained per component: "
          f"{(S**2 / (S**2).sum()).round(3)}")
    print(f"[p1c]   cumulative variance: "
          f"{(np.cumsum(S**2) / (S**2).sum()).round(3)}")

    # 2. Load model + WikiText
    print(f"[p1c] loading {MODEL_ID}...")
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
    print(f"[p1c]   loaded in {time.time() - t0:.1f}s")

    print(f"[p1c] loading {args.n_wikitext} WikiText sentences...")
    wt = load_wikitext_chunk(tok, n_sentences=args.n_wikitext)

    # 3. Per-pair token ids
    pair_ids: dict[str, tuple[int, int]] = {}
    for p in pairs_eval:
        first = json.loads(next((GOR_ROOT / "data_gen"
                                / f"v11_{p}_trials.jsonl").open()))
        hi = tok(first["high_word"], add_special_tokens=False).input_ids[0]
        lo = tok(first["low_word"], add_special_tokens=False).input_ids[0]
        pair_ids[p] = (hi, lo)

    # 4. Compute population mean projections μ_k from cell_seed=0 of the
    # eval pairs. For each k, μ_k = mean(V_k^T h_L20) over those prompts.
    print(f"[p1c] computing μ_k from {len(pairs_eval)} eval pairs...")
    Hs = []
    for p in pairs_eval:
        npz = (GOR_ROOT / "results" / "v11" / MODEL_SHORT / p
               / f"{MODEL_SHORT}_{p}_v11_residuals.npz")
        d = np.load(npz)
        Hs.append(d["activations"][:, L, :].astype(np.float64))
    H_full = np.concatenate(Hs, axis=0)  # (N_total, d)
    mu_full = (H_full @ V).mean(0)  # (8,)
    print(f"[p1c]   μ_full (per direction): {mu_full.round(3)}")

    # 5. Iterate over k
    rng = np.random.default_rng(43)
    ks_to_run = sorted(set(args.ks))
    SETTINGS_BY_K = {}
    for k in ks_to_run:
        V_k = V[:, :k]  # (d, k)
        mu_k = mu_full[:k]  # (k,)
        V_k_t = torch.tensor(V_k.copy(), dtype=torch.bfloat16, device=model.device)
        mu_k_t = torch.tensor(mu_k.copy(), dtype=torch.bfloat16, device=model.device)
        # Random orthonormal of same rank
        R_k = random_orthonormal(V.shape[0], k, rng)
        R_k_t = torch.tensor(R_k.copy(), dtype=torch.bfloat16, device=model.device)
        SETTINGS_BY_K[k] = {
            "proj_out_k":    make_subspace_hook(V_k_t, "proj_out"),
            "mean_ablate_k": make_subspace_hook(V_k_t, "mean_ablate", mu_k_t),
            "rand_proj_k":   make_subspace_hook(R_k_t, "proj_out"),
        }

    # 6. Run per-pair × per-k × per-mode
    results = {
        "model_id": MODEL_ID,
        "layer": L,
        "ks": ks_to_run,
        "pairs_eval": pairs_eval,
        "singular_values": S.tolist(),
        "variance_explained": (S**2 / (S**2).sum()).tolist(),
        "mu_full": mu_full.tolist(),
        "per_pair": {},
        "wikitext_loss": {},
    }

    print("\n[p1c] running held-out evals per pair × k × mode...")
    for p in pairs_eval:
        prompts = load_holdout_prompts(p)
        hi_id, lo_id = pair_ids[p]
        zs = np.array([t["z"] for t in prompts], dtype=np.float64)
        xs = np.array([t["x"] for t in prompts], dtype=np.float64)
        results["per_pair"][p] = {"n_holdout": len(prompts)}
        for k in ks_to_run:
            for mode_name, hook_fn in SETTINGS_BY_K[k].items():
                t1 = time.time()
                ld = eval_LD_under_hook(
                    model, tok, prompts, hi_id, lo_id, hook_fn, L,
                    batch_size=args.batch_size, max_seq=args.max_seq,
                )
                r_z = float(np.corrcoef(ld, zs)[0, 1])
                r_x = float(np.corrcoef(ld, xs)[0, 1])
                key = f"k{k}_{mode_name}"
                results["per_pair"][p][key] = {
                    "k": k,
                    "corr_LD_z": r_z,
                    "corr_LD_x": r_x,
                    "ld_mean": float(ld.mean()),
                    "elapsed_sec": round(time.time() - t1, 1),
                }
                print(f"[p1c]   {p:<10} k={k}  {mode_name:<14} "
                      f"r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
                      f"<LD>={float(ld.mean()):+.2f}  "
                      f"({time.time() - t1:.0f}s)", flush=True)

    # 7. Perplexity sweep — one run per (k, mode)
    print("\n[p1c] WikiText perplexity per setting...")
    for k in ks_to_run:
        for mode_name, hook_fn in SETTINGS_BY_K[k].items():
            t1 = time.time()
            loss = wikitext_loss_under_hook(model, tok, wt, hook_fn, L)
            ppl = float(np.exp(loss))
            key = f"k{k}_{mode_name}"
            results["wikitext_loss"][key] = {
                "k": k, "loss": loss, "ppl": ppl,
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[p1c]   k={k}  {mode_name:<14} loss={loss:.4f}  ppl={ppl:.2f} "
                  f"({time.time() - t1:.0f}s)", flush=True)

    out_path = REPO / "results" / f"{args.out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[p1c] wrote {out_path}")


if __name__ == "__main__":
    main()
