"""Phase 1d — on-manifold tangent ablation (per-prompt cell-mean shift).

For each pair, build a cell-mean lookup M(x, z) at L20 from cell_seed ∈
{0..4}. For each held-out test prompt (cell_seed ∈ {5..9}) with cell
(x_i, z_i), compute displacement

    Δ_i = M(x_i, z_target) - M(x_i, z_i)

where z_target is the cell on the same-x slice closest to 0 in absolute
value. Hook L20 with h_i ← h_i + Δ_i (broadcast across token positions).

Hypothesis: this **moves each cell along the manifold** to its z=0 same-x
neighbor, preserving within-cell variance. If the z-encoding follows a
curved manifold, this should suppress r(LD, z) more cleanly than rank-k
linear projection (which the height non-monotonicity in Phase 1c
indicated is mismatched to the geometry).

Comparators (per pair, all evaluated on the same held-out fold):
  baseline                — no hook
  per_pair_proj_out       — h ← h - (h·d̂_p) d̂_p   where d̂_p = unit(primal_z[pair])
  manifold_shift_to_z0    — h ← h + Δ_i  (per-prompt, manifold ablation)
  random_per_prompt       — h ← h + r_i  where r_i is a random unit vector
                             in d-dim scaled to ||Δ_i||  (specificity test)

Train/test split is by cell_seed: train = {0..4}, test = {5..9}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from p1_text_ablation import (  # noqa: E402
    GOR_ROOT, ALL_PAIRS, MODEL_ID, MODEL_SHORT, LATE_LAYER, unit,
)


# ------------------------- data loading -------------------------

def load_pair_npz(pair: str) -> dict:
    p = (GOR_ROOT / "results" / "v11" / MODEL_SHORT / pair
         / f"{MODEL_SHORT}_{pair}_v11_residuals.npz")
    return np.load(p, allow_pickle=True)


def load_pair_trials(pair: str) -> list[dict]:
    p = GOR_ROOT / "data_gen" / f"v11_{pair}_trials.jsonl"
    return [json.loads(line) for line in p.open()]


# ------------------------- cell-mean lookup -------------------------

def build_cell_mean_lookup(pair: str, layer: int, train_seeds: list[int]
                            ) -> tuple[dict[tuple[float, float], np.ndarray],
                                       np.ndarray]:
    """Returns (lookup, primal_z_train).

    lookup: dict mapping (x_rounded, z_rounded) -> centroid (d,) float64
    primal_z_train: mean(h | z>+1) - mean(h | z<-1) restricted to train_seeds
    """
    d = load_pair_npz(pair)
    h = d["activations"][:, layer, :].astype(np.float64)
    xs = d["x"].astype(np.float64)
    zs = d["z"].astype(np.float64)
    seeds_full = d["seed"]
    cell_seeds = seeds_full - seeds_full.min()
    train_mask = np.isin(cell_seeds, train_seeds)
    h_tr = h[train_mask]; xs_tr = xs[train_mask]; zs_tr = zs[train_mask]

    cells: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for i in range(len(h_tr)):
        key = (round(float(xs_tr[i]), 4), round(float(zs_tr[i]), 4))
        cells[key].append(h_tr[i])
    lookup = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in cells.items()}

    # primal_z from train fold
    primal_train = (h_tr[zs_tr > +1.0].mean(0) - h_tr[zs_tr < -1.0].mean(0))

    return lookup, primal_train


# ------------------------- displacement vectors -------------------------

def manifold_displacements(prompts: list[dict],
                            lookup: dict[tuple[float, float], np.ndarray],
                            target_z: float = 0.0) -> np.ndarray:
    """For each prompt, compute Δ = M(x_i, z_target) - M(x_i, z_i).

    z_target is approximated by the same-x cell whose z is closest to 0.
    If the test prompt's own (x, z) cell is missing from `lookup` (shouldn't
    happen — train+test share the same grid — but we guard), we fall back
    to the same-x cell closest to z_i.
    """
    d_dim = next(iter(lookup.values())).shape[0]
    out = np.zeros((len(prompts), d_dim), dtype=np.float64)
    # Pre-bin the lookup keys by x for fast same-x lookup
    by_x: dict[float, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for (xk, zk), v in lookup.items():
        by_x[xk].append((zk, v))

    n_missing = 0
    for i, t in enumerate(prompts):
        x = round(float(t["x"]), 4)
        z = round(float(t["z"]), 4)
        same_x = by_x.get(x, [])
        if not same_x:
            # Should not happen for v11 grid; leave zero displacement
            n_missing += 1
            continue
        # Target: same-x cell closest to z=target_z
        z_t, mu_target = min(same_x, key=lambda zk_v: abs(zk_v[0] - target_z))
        # Source: same-x cell closest to z_i (handles missing exact cell)
        z_s, mu_source = min(same_x, key=lambda zk_v: abs(zk_v[0] - z))
        out[i] = mu_target - mu_source

    if n_missing:
        print(f"[p1d]   warning: {n_missing}/{len(prompts)} prompts had no "
              f"same-x cells in lookup")
    return out


def random_displacements_matched(displacements: np.ndarray, seed: int = 7
                                  ) -> np.ndarray:
    """For each row of `displacements`, draw a random unit vector in d-dim
    and rescale it to match the row's L2 norm. Preserves the per-prompt
    magnitude profile."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(displacements)
    for i in range(displacements.shape[0]):
        norm_i = float(np.linalg.norm(displacements[i]))
        if norm_i < 1e-12:
            continue
        v = rng.standard_normal(displacements.shape[1])
        v /= np.linalg.norm(v) + 1e-12
        out[i] = norm_i * v
    return out


# ------------------------- hooks -------------------------

def make_per_prompt_shift_hook(state: dict):
    """Hook that adds state['delta'] (B, d) tensor to h, broadcast over T."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
        delta = state["delta"]  # (B, d) on device
        h = h + delta[:, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_proj_out_hook(d_t: torch.Tensor):
    """Static rank-1 project-out hook with d_t the unit direction."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # (B, T, d)
        proj = (h * d_t).sum(-1, keepdim=True)
        h = h - proj * d_t
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


# ------------------------- per-prompt eval -------------------------

def eval_LD_with_per_prompt_shift(model, tok, prompts, displacements,
                                    hi_id, lo_id, layer, batch_size, max_seq
                                    ) -> np.ndarray:
    """Forward over prompts; at L<layer> hook, add per-prompt displacement."""
    decoder_layers = model.model.layers
    state = {"delta": None}
    handle = decoder_layers[layer].register_forward_hook(
        make_per_prompt_shift_hook(state))
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


def eval_LD_with_static_hook(model, tok, prompts, hook_fn,
                              hi_id, lo_id, layer, batch_size, max_seq
                              ) -> np.ndarray:
    decoder_layers = model.model.layers
    handle = decoder_layers[layer].register_forward_hook(hook_fn)
    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            b = prompts[b0:b0 + batch_size]
            prompt_strs = [t["prompt"] for t in b]
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


def eval_LD_no_hook(model, tok, prompts, hi_id, lo_id, batch_size, max_seq
                     ) -> np.ndarray:
    out = np.zeros(len(prompts), dtype=np.float32)
    for b0 in range(0, len(prompts), batch_size):
        b = prompts[b0:b0 + batch_size]
        prompt_strs = [t["prompt"] for t in b]
        enc = tok(prompt_strs, return_tensors="pt", padding="max_length",
                   max_length=max_seq, truncation=True).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask,
                            use_cache=False).logits[:, -1, :].float()
        out[b0:b0 + len(b)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    return out


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"],
                    choices=ALL_PAIRS + ["all"])
    ap.add_argument("--target-z", type=float, default=0.0)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-name", default="p1d_manifold_ablation")
    args = ap.parse_args()

    pairs = ALL_PAIRS if args.pairs == ["all"] else list(args.pairs)
    L = LATE_LAYER
    train_seeds = [0, 1, 2, 3, 4]
    test_seeds = [5, 6, 7, 8, 9]

    # 1. Load model + tokenizer
    print(f"[p1d] loading {MODEL_ID}...")
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
    print(f"[p1d]   loaded in {time.time() - t0:.1f}s")

    # 2. Per-pair eval
    results = {
        "model_id": MODEL_ID,
        "layer": L,
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "target_z": args.target_z,
        "per_pair": {},
    }

    for p in pairs:
        print(f"\n[p1d] === pair: {p} ===")
        # Build train-fold lookup + train-fold primal_z
        t0 = time.time()
        lookup, primal_train = build_cell_mean_lookup(p, L, train_seeds)
        print(f"[p1d]   built {len(lookup)} cell-means and primal_z from "
              f"train fold ({time.time()-t0:.1f}s)")

        # Held-out prompts
        all_trials = load_pair_trials(p)
        # cell_seed = seed - min_seed for this pair
        seeds_arr = np.array([t["seed"] for t in all_trials])
        min_seed = int(seeds_arr.min())
        test_prompts = [t for t in all_trials
                         if (t["seed"] - min_seed) in test_seeds]
        print(f"[p1d]   held-out: {len(test_prompts)} prompts "
              f"(test_seeds={test_seeds})")

        # Token ids
        first = test_prompts[0]
        hi_id = tok(first["high_word"], add_special_tokens=False).input_ids[0]
        lo_id = tok(first["low_word"], add_special_tokens=False).input_ids[0]

        # Per-prompt manifold displacements
        Δ_manifold = manifold_displacements(test_prompts, lookup,
                                              target_z=args.target_z)
        # Random per-prompt of matched ||Δ||
        Δ_random = random_displacements_matched(Δ_manifold, seed=7)
        # Diagnostic: average ||Δ|| and distribution
        norms_m = np.linalg.norm(Δ_manifold, axis=1)
        norms_r = np.linalg.norm(Δ_random, axis=1)
        print(f"[p1d]   manifold ||Δ||: mean={norms_m.mean():.2f}  "
              f"median={np.median(norms_m):.2f}  max={norms_m.max():.2f}")
        print(f"[p1d]   random ||Δ||:   mean={norms_r.mean():.2f}  "
              f"(should match manifold)")

        # zs, xs for correlations
        zs = np.array([t["z"] for t in test_prompts], dtype=np.float64)
        xs = np.array([t["x"] for t in test_prompts], dtype=np.float64)

        # Direction for per-pair primal_z proj-out
        d_t = torch.tensor(unit(primal_train), dtype=torch.bfloat16,
                            device=model.device)

        results["per_pair"][p] = {
            "n_holdout": len(test_prompts),
            "n_cells_train": len(lookup),
            "manifold_norm_mean": float(norms_m.mean()),
            "manifold_norm_max": float(norms_m.max()),
        }

        # Setting 1: baseline
        t1 = time.time()
        ld = eval_LD_no_hook(model, tok, test_prompts, hi_id, lo_id,
                              args.batch_size, args.max_seq)
        r_z = float(np.corrcoef(ld, zs)[0, 1])
        r_x = float(np.corrcoef(ld, xs)[0, 1])
        results["per_pair"][p]["baseline"] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1d]   baseline       r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
              flush=True)

        # Setting 2: per-pair proj_out (rank-1, train-derived primal_z)
        t1 = time.time()
        ld = eval_LD_with_static_hook(model, tok, test_prompts,
                                       make_proj_out_hook(d_t),
                                       hi_id, lo_id, L,
                                       args.batch_size, args.max_seq)
        r_z = float(np.corrcoef(ld, zs)[0, 1])
        r_x = float(np.corrcoef(ld, xs)[0, 1])
        results["per_pair"][p]["per_pair_proj_out"] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1d]   per_pair_proj  r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
              flush=True)

        # Setting 3: manifold shift
        t1 = time.time()
        ld = eval_LD_with_per_prompt_shift(
            model, tok, test_prompts, Δ_manifold, hi_id, lo_id, L,
            args.batch_size, args.max_seq)
        r_z = float(np.corrcoef(ld, zs)[0, 1])
        r_x = float(np.corrcoef(ld, xs)[0, 1])
        results["per_pair"][p]["manifold_shift_to_z0"] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1d]   manifold_shift r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
              flush=True)

        # Setting 4: random per-prompt of matched magnitude
        t1 = time.time()
        ld = eval_LD_with_per_prompt_shift(
            model, tok, test_prompts, Δ_random, hi_id, lo_id, L,
            args.batch_size, args.max_seq)
        r_z = float(np.corrcoef(ld, zs)[0, 1])
        r_x = float(np.corrcoef(ld, xs)[0, 1])
        results["per_pair"][p]["random_per_prompt"] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[p1d]   random_pp     r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"<LD>={float(ld.mean()):+.2f}  ({time.time()-t1:.0f}s)",
              flush=True)

    out_path = REPO / "results" / f"{args.out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[p1d] wrote {out_path}")


if __name__ == "__main__":
    main()
