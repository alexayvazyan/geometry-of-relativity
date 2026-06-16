"""Cross-model runner: extract → primal_z + cell-mean lookup → α-sweep.

One self-contained pipeline for any (model_id, late_layer) pair.
Reuses Jaehoon's extract_v11_dense.py for the forward-pass extraction
(if the NPZ doesn't already exist), then runs the same α-sweep we did
on Gemma 2 2B.

Output: results/px_cross_model_<model_short>_L<layer>.json with
behavioral baselines + manifold/proj_out × {α=0.5, 0.75, 1.0} per pair.

Usage:
    python px_cross_model_run.py --model google/gemma-2-9b --model-short gemma2-9b --layer 33
    python px_cross_model_run.py --model google/gemma-2-2b-it --model-short gemma2-2b-it --layer 20
    python px_cross_model_run.py --model google/gemma-4-E4B --model-short gemma4-e4b --layer 33
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR_ROOT = Path("/home/alexander/research_projects/geometry-of-relativity")
GOR_EXTRACTOR = GOR_ROOT / "scripts" / "vast_remote" / "extract_v11_dense.py"
ALL_PAIRS_DEFAULT = ["height", "weight", "speed"]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def ensure_extraction(model_id: str, model_short: str, pair: str,
                       batch_size: int, max_seq: int) -> Path:
    """Run extract_v11_dense.py for (model, pair) if NPZ is not on disk."""
    out_dir = GOR_ROOT / "results" / "v11" / model_short / pair
    npz = out_dir / f"{model_short}_{pair}_v11_residuals.npz"
    if npz.exists():
        print(f"[xmod]   {pair}: NPZ exists → skipping extraction")
        return npz

    # Make sure the trial JSONL exists
    trials_path = GOR_ROOT / "data_gen" / f"v11_{pair}_trials.jsonl"
    if not trials_path.exists():
        gen_cmd = ["python3", "scripts/gen_v11_dense.py", "--pair", pair]
        print(f"[xmod]   {pair}: generating trials → {' '.join(gen_cmd)}")
        subprocess.run(gen_cmd, cwd=GOR_ROOT, check=True)

    cmd = [
        "python3", str(GOR_EXTRACTOR),
        "--model", model_id,
        "--pair", pair,
        "--batch-size", str(batch_size),
        "--max-seq", str(max_seq),
        "--minimal",  # residuals only — bypass GQA reshape mismatches
    ]
    print(f"[xmod]   {pair}: extracting → {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=GOR_ROOT, check=True)
    print(f"[xmod]   {pair}: extracted in {time.time() - t0:.1f}s")
    if not npz.exists():
        raise FileNotFoundError(f"Extractor finished but NPZ not at {npz}")
    return npz


def primal_z_from_npz(npz_path: Path, layer: int, cell_seeds: list[int] | None
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    """primal_z and (optionally) the train-fold cell-mean lookup at `layer`.

    cell_seeds: if given, restrict to these cell_seeds (subtracting the per-NPZ
    seed offset). If None, use all data.
    """
    d = np.load(npz_path, allow_pickle=True)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"].astype(np.float64)
    seed_full = d["seed"]
    cs = seed_full - int(seed_full.min())
    if cell_seeds is not None:
        mask = np.isin(cs, cell_seeds)
        h = h[mask]; z = z[mask]
    primal = h[z > +1.0].mean(0) - h[z < -1.0].mean(0)
    return primal


def build_cell_mean_lookup(npz_path: Path, layer: int, cell_seeds: list[int]
                            ) -> dict[tuple[float, float], np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    h = d["activations"][:, layer, :].astype(np.float64)
    xs = d["x"].astype(np.float64)
    zs = d["z"].astype(np.float64)
    seed_full = d["seed"]
    cs = seed_full - int(seed_full.min())
    mask = np.isin(cs, cell_seeds)
    h = h[mask]; xs = xs[mask]; zs = zs[mask]
    cells: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for i in range(len(h)):
        key = (round(float(xs[i]), 4), round(float(zs[i]), 4))
        cells[key].append(h[i])
    return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in cells.items()}


def manifold_displacements(prompts: list[dict],
                            lookup: dict[tuple[float, float], np.ndarray],
                            target_z: float = 0.0) -> np.ndarray:
    d_dim = next(iter(lookup.values())).shape[0]
    out = np.zeros((len(prompts), d_dim), dtype=np.float64)
    by_x: dict[float, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for (xk, zk), v in lookup.items():
        by_x[xk].append((zk, v))
    n_missing = 0
    for i, t in enumerate(prompts):
        x = round(float(t["x"]), 4)
        z = round(float(t["z"]), 4)
        same_x = by_x.get(x, [])
        if not same_x:
            n_missing += 1
            continue
        z_t, mu_target = min(same_x, key=lambda zk_v: abs(zk_v[0] - target_z))
        z_s, mu_source = min(same_x, key=lambda zk_v: abs(zk_v[0] - z))
        out[i] = mu_target - mu_source
    if n_missing:
        print(f"[xmod]   warning: {n_missing}/{len(prompts)} prompts missing same-x cell")
    return out


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError(f"could not locate decoder layers on {type(m)}")


def make_partial_proj_hook(d_t: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h * d_t).sum(-1, keepdim=True)
        h = h - alpha * proj * d_t
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_alpha_shift_hook(state: dict, alpha: float):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = state["delta"]
        h = h + alpha * delta[:, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def eval_LD(model, tok, prompts, hi_id, lo_id, batch_size, max_seq,
             hook_fn=None, layer=None, per_prompt_delta=None):
    """If per_prompt_delta is given, hook_fn is created internally as
    an alpha-shift; pass alpha=1.0 because magnitude is in the delta."""
    layers = get_decoder_layers(model)
    state = {"delta": None}
    if per_prompt_delta is not None:
        hook_fn = make_alpha_shift_hook(state, 1.0)
    handle = layers[layer].register_forward_hook(hook_fn) if hook_fn else None

    out = np.zeros(len(prompts), dtype=np.float32)
    try:
        for b0 in range(0, len(prompts), batch_size):
            b = prompts[b0:b0 + batch_size]
            prompt_strs = [t["prompt"] for t in b]
            if per_prompt_delta is not None:
                state["delta"] = torch.tensor(
                    per_prompt_delta[b0:b0 + batch_size],
                    dtype=torch.bfloat16, device=model.device)
            enc = tok(prompt_strs, return_tensors="pt", padding="max_length",
                       max_length=max_seq, truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                use_cache=False).logits[:, -1, :].float()
            out[b0:b0 + len(b)] = (logits[:, hi_id] - logits[:, lo_id]).cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--model-short", required=True,
                    help="short name for paths (e.g. gemma2-9b)")
    ap.add_argument("--layer", type=int, required=True,
                    help="late-layer index for ablation")
    ap.add_argument("--pairs", nargs="+", default=ALL_PAIRS_DEFAULT)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0])
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    out_name = args.out_name or f"px_cross_model_{args.model_short}_L{args.layer}"
    L = args.layer

    train_seeds = [0, 1, 2, 3, 4]
    test_seeds = [5, 6, 7, 8, 9]
    pairs = list(args.pairs)

    # 1. Ensure extraction for each pair
    print(f"\n[xmod] === model={args.model} short={args.model_short} L={L} ===")
    for p in pairs:
        ensure_extraction(args.model, args.model_short, p,
                           args.batch_size, args.max_seq)

    # 2. Load model for ablation phase
    print(f"\n[xmod] loading {args.model} for ablation phase...")
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    print(f"[xmod]   loaded in {time.time() - t0:.1f}s")

    # 3. Per-pair runs
    results = {
        "model_id": args.model,
        "model_short": args.model_short,
        "layer": L,
        "alphas": list(args.alphas),
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "per_pair": {},
    }

    for p in pairs:
        print(f"\n[xmod] === pair: {p} ===")
        npz_path = (GOR_ROOT / "results" / "v11" / args.model_short / p
                    / f"{args.model_short}_{p}_v11_residuals.npz")
        primal_train = primal_z_from_npz(npz_path, L, train_seeds)
        lookup = build_cell_mean_lookup(npz_path, L, train_seeds)
        print(f"[xmod]   built primal_z and {len(lookup)}-cell lookup from train fold")

        # Held-out prompts
        trials_path = GOR_ROOT / "data_gen" / f"v11_{p}_trials.jsonl"
        all_trials = [json.loads(line) for line in trials_path.open()]
        seeds_arr = np.array([t["seed"] for t in all_trials])
        min_seed = int(seeds_arr.min())
        test_prompts = [t for t in all_trials
                         if (t["seed"] - min_seed) in test_seeds]
        print(f"[xmod]   held-out: {len(test_prompts)} prompts")

        first = test_prompts[0]
        hi_id = tok(first["high_word"], add_special_tokens=False).input_ids[0]
        lo_id = tok(first["low_word"], add_special_tokens=False).input_ids[0]

        Δ = manifold_displacements(test_prompts, lookup, target_z=0.0)
        zs = np.array([t["z"] for t in test_prompts], dtype=np.float64)
        xs = np.array([t["x"] for t in test_prompts], dtype=np.float64)
        d_t = torch.tensor(unit(primal_train), dtype=torch.bfloat16,
                            device=model.device)

        per_pair = {
            "n_holdout": len(test_prompts),
            "n_cells_train": len(lookup),
            "manifold_norm_mean": float(np.linalg.norm(Δ, axis=1).mean()),
        }

        # Baseline
        t1 = time.time()
        ld = eval_LD(model, tok, test_prompts, hi_id, lo_id,
                      args.batch_size, args.max_seq)
        per_pair["baseline"] = {
            "corr_LD_z": float(np.corrcoef(ld, zs)[0, 1]),
            "corr_LD_x": float(np.corrcoef(ld, xs)[0, 1]),
            "ld_mean": float(ld.mean()),
            "elapsed_sec": round(time.time() - t1, 1),
        }
        print(f"[xmod]   baseline       r_z={per_pair['baseline']['corr_LD_z']:+.3f}  "
              f"r_x={per_pair['baseline']['corr_LD_x']:+.3f}  "
              f"<LD>={per_pair['baseline']['ld_mean']:+.2f}  "
              f"({time.time()-t1:.0f}s)", flush=True)

        # α-sweep both methods
        for alpha in args.alphas:
            # Manifold
            t1 = time.time()
            ld = eval_LD(model, tok, test_prompts, hi_id, lo_id,
                          args.batch_size, args.max_seq,
                          per_prompt_delta=alpha * Δ, layer=L)
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            per_pair[f"manifold_alpha_{alpha:.2f}"] = {
                "alpha": alpha,
                "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[xmod]   manifold α={alpha:.2f}  r_z={r_z:+.3f}  "
                  f"r_x={r_x:+.3f}  <LD>={float(ld.mean()):+.2f}  "
                  f"({time.time()-t1:.0f}s)", flush=True)

            # Per-pair proj_out
            t1 = time.time()
            ld = eval_LD(model, tok, test_prompts, hi_id, lo_id,
                          args.batch_size, args.max_seq,
                          hook_fn=make_partial_proj_hook(d_t, alpha), layer=L)
            r_z = float(np.corrcoef(ld, zs)[0, 1])
            r_x = float(np.corrcoef(ld, xs)[0, 1])
            per_pair[f"proj_out_alpha_{alpha:.2f}"] = {
                "alpha": alpha,
                "corr_LD_z": r_z, "corr_LD_x": r_x, "ld_mean": float(ld.mean()),
                "elapsed_sec": round(time.time() - t1, 1),
            }
            print(f"[xmod]   proj_out α={alpha:.2f}  r_z={r_z:+.3f}  "
                  f"r_x={r_x:+.3f}  <LD>={float(ld.mean()):+.2f}  "
                  f"({time.time()-t1:.0f}s)", flush=True)

        results["per_pair"][p] = per_pair

    out_path = REPO / "results" / f"{out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[xmod] wrote {out_path}")


if __name__ == "__main__":
    main()
