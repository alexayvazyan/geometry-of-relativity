"""Phase 2E — apply Phase 1's residual-stream interventions on the same
prompt bank we used for ablation, so the (r_x, r_z) phase-space comparison
is on a single set of axes.

Three interventions per (model, k):
  proj_out      h ← h − (h·d̂)·d̂                         d̂ = unit(primal_z)
  mean_ablate   h ← h − (h·d̂)·d̂ + μ·d̂                   μ = E[h·d̂] over baseline
  manifold(α)   h ← h + α·Δ_i,  Δ_i = M(x_i, z=0) − M(x_i, z_i)
                 (per-prompt cell-mean shift; matches Phase 1d/1e)

primal_z and the cell-mean lookup are built from cell_seed=0 (train fold);
behavioral metrics measured on cell_seed != 0 (held-out).

Usage:
  python scripts/p2e_residual_interventions.py --model gemma2-2b --pair height --k 15
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

MODEL_ID = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
}
LATE_LAYER = {"gemma2-2b": 20, "gemma2-9b": 33}


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("can't find decoder.layers")


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def build_primal_z(h_at_L, z_eff, cell_seed):
    """primal_z = mean(h | z_eff>+1, train) − mean(h | z_eff<-1, train).
    Train = cell_seed == 0 (matches Phase 1 convention)."""
    train = cell_seed == 0
    high = train & (z_eff > +1.0)
    low = train & (z_eff < -1.0)
    if high.sum() < 3 or low.sum() < 3:
        # fallback: half-split if cell_seed=0 is missing
        train = np.zeros_like(cell_seed, dtype=bool)
        train[:len(cell_seed)//2] = True
        high = train & (z_eff > +1.0)
        low = train & (z_eff < -1.0)
    return h_at_L[high].mean(0) - h_at_L[low].mean(0)


def build_cell_mean_lookup(h_at_L, x, z, cell_seed, n_x_bins=20, n_z_bins=20):
    """M[x_bin, z_bin] = mean residual at L over train-fold prompts in that cell.
    Uses cell_seed=0 only."""
    train = cell_seed == 0
    if train.sum() < 50:
        train = np.zeros_like(cell_seed, dtype=bool)
        train[:len(cell_seed)//2] = True
    x_t = x[train]; z_t = z[train]; h_t = h_at_L[train]

    x_edges = np.linspace(x_t.min() - 1e-6, x_t.max() + 1e-6, n_x_bins + 1)
    z_edges = np.linspace(z_t.min() - 1e-6, z_t.max() + 1e-6, n_z_bins + 1)
    x_idx = np.clip(np.digitize(x_t, x_edges) - 1, 0, n_x_bins - 1)
    z_idx = np.clip(np.digitize(z_t, z_edges) - 1, 0, n_z_bins - 1)

    d_model = h_t.shape[-1]
    M = np.full((n_x_bins, n_z_bins, d_model), np.nan, dtype=np.float64)
    counts = np.zeros((n_x_bins, n_z_bins), dtype=np.int32)
    for i in range(len(h_t)):
        xb, zb = x_idx[i], z_idx[i]
        if counts[xb, zb] == 0:
            M[xb, zb] = h_t[i].astype(np.float64)
        else:
            M[xb, zb] = (M[xb, zb] * counts[xb, zb] + h_t[i]) / (counts[xb, zb] + 1)
        counts[xb, zb] += 1

    # x-marginal cell mean (averaged over z, weighted by counts) — used as the
    # "z=0 reference cell" when an exact-match z=0 cell is missing on that x slice.
    x_marg = np.full((n_x_bins, d_model), np.nan, dtype=np.float64)
    for xb in range(n_x_bins):
        s = np.zeros(d_model)
        c = 0
        for zb in range(n_z_bins):
            if counts[xb, zb] > 0:
                s += M[xb, zb] * counts[xb, zb]
                c += counts[xb, zb]
        if c > 0:
            x_marg[xb] = s / c

    return M, counts, x_edges, z_edges, x_marg


def manifold_delta(M, counts, x_edges, z_edges, x_marg, x, z):
    """Per-prompt Δ = M[x_bin, z_target_bin] − M[x_bin, z_bin].
    z_target_bin is the closest-to-zero bin available on the same x slice; if
    no bin on this x has data, fall back to the marginal x-mean as M[x, *].
    """
    n_x_bins = M.shape[0]; n_z_bins = M.shape[1]; d = M.shape[-1]
    n = len(x)
    deltas = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        xb = int(np.clip(np.digitize(x[i], x_edges) - 1, 0, n_x_bins - 1))
        zb = int(np.clip(np.digitize(z[i], z_edges) - 1, 0, n_z_bins - 1))
        # Target bin: the bin whose center is closest to z=0, restricted to bins that have data on this x slice.
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        valid = counts[xb] > 0
        if valid.any():
            valid_idx = np.where(valid)[0]
            tgt_b = int(valid_idx[np.argmin(np.abs(z_centers[valid_idx]))])
            tgt = M[xb, tgt_b]
        else:
            tgt = x_marg[xb] if not np.isnan(x_marg[xb]).any() else np.zeros(d)
        cur = M[xb, zb] if counts[xb, zb] > 0 else x_marg[xb]
        if np.isnan(cur).any():
            cur = np.zeros(d)
        deltas[i] = (tgt - cur).astype(np.float32)
    return deltas


@torch.inference_mode()
def run_intervention(model, tok, trials, mode, *, layer, direction=None,
                     mu_proj=None, deltas=None, alpha=1.0,
                     high_id, low_id, batch_size, max_seq, device):
    decoder = get_decoder_layers(model)
    target = decoder[layer]

    # state for manifold mode: per-batch deltas
    state = {"batch_delta": None}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if mode == "baseline":
            return output
        if mode == "proj_out":
            d_t = direction
            proj = (h * d_t).sum(-1, keepdim=True)
            h2 = h - alpha * proj * d_t
        elif mode == "mean_ablate":
            d_t = direction
            proj = (h * d_t).sum(-1, keepdim=True)
            h2 = h - proj * d_t + mu_proj * d_t
        elif mode == "manifold":
            # state['batch_delta'] is shape (B, d), broadcast over T
            d = state["batch_delta"]
            if d is None:
                return output
            h2 = h + alpha * d.unsqueeze(1)
        else:
            raise ValueError(mode)
        if isinstance(output, tuple):
            return (h2,) + output[1:]
        return h2

    handle = target.register_forward_hook(hook)
    n = len(trials)
    ld = np.zeros(n, dtype=np.float64)
    try:
        for b0 in range(0, n, batch_size):
            batch = trials[b0:b0 + batch_size]
            enc = tok([t["prompt"] for t in batch], return_tensors="pt",
                      padding="max_length", max_length=max_seq, truncation=True
                      ).to(device)
            if mode == "manifold":
                state["batch_delta"] = torch.from_numpy(
                    deltas[b0:b0 + len(batch)]
                ).to(device).to(model.dtype)
            else:
                state["batch_delta"] = None
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask, use_cache=False)
            logits_last = out.logits[:, -1, :].float()
            ld[b0:b0+len(batch)] = (
                logits_last[:, high_id] - logits_last[:, low_id]
            ).cpu().numpy()
    finally:
        handle.remove()
    return ld


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[p2e] loading {MODEL_ID[args.model]}...", flush=True)
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
    low_id = first_token_id(tok, "short")
    high_id = first_token_id(tok, "tall")

    # Pull last-token residuals at L from existing p2_attn dump.
    attn_npz = REPO / "results" / "p2_attn" / args.model / f"{args.pair}_k{args.k}.npz"
    if not attn_npz.exists():
        print(f"missing {attn_npz}; can't build primal_z without re-extraction")
        return
    d = np.load(attn_npz, allow_pickle=True)
    # Match prompt order between NPZ and JSONL (NPZ is in same order as JSONL).
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    assert len(all_trials) == len(d["ld"]), f"length mismatch: {len(all_trials)} vs {len(d['ld'])}"

    h_at_L = d["residuals"][:, L, :].astype(np.float64)
    z_arr = np.array([t["z"] for t in all_trials], dtype=np.float64)        # intended
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)

    # Build primal_z (using z_eff as the variable Phase 1 uses too) on train fold.
    primal = build_primal_z(h_at_L, z_eff_arr, cs_arr)
    d_unit = primal / max(np.linalg.norm(primal), 1e-9)
    mu_proj = float(np.mean(h_at_L @ d_unit))
    print(f"[p2e] primal_z norm={np.linalg.norm(primal):.2f}  μ_proj={mu_proj:+.3f}",
          flush=True)

    # Cell-mean lookup uses intended z (more consistent grid for Δ computation).
    M, counts, x_edges, z_edges, x_marg = build_cell_mean_lookup(
        h_at_L, x_arr, z_arr, cs_arr,
    )
    deltas = manifold_delta(M, counts, x_edges, z_edges, x_marg, x_arr, z_arr)
    print(f"[p2e] mean ||Δ||={np.linalg.norm(deltas, axis=-1).mean():.2f}  "
          f"max ||Δ||={np.linalg.norm(deltas, axis=-1).max():.2f}", flush=True)

    # Subsample to held-out fold for evaluation.
    test_mask = cs_arr != 0
    rng = np.random.default_rng(0)
    test_idx = np.where(test_mask)[0]
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_test = z_arr[test_idx]; z_eff_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    deltas_test = deltas[test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
    print(f"[p2e] {args.model}/{args.pair} k={args.k}: n_test={n_test} max_seq={max_seq}",
          flush=True)

    direction_t = torch.from_numpy(d_unit.astype(np.float32)).to(args.device).to(model.dtype)

    out = {"model": args.model, "pair": args.pair, "k": args.k,
            "layer": L, "results": {}}

    for mode_name, kwargs in [
        ("baseline",        dict(mode="baseline")),
        ("proj_out",        dict(mode="proj_out", direction=direction_t)),
        ("mean_ablate",     dict(mode="mean_ablate", direction=direction_t,
                                  mu_proj=mu_proj)),
        ("manifold_a075",   dict(mode="manifold", deltas=deltas_test, alpha=0.75)),
        ("manifold_a100",   dict(mode="manifold", deltas=deltas_test, alpha=1.00)),
    ]:
        t1 = time.time()
        ld = run_intervention(
            model, tok, trials,
            **kwargs, layer=L, high_id=high_id, low_id=low_id,
            batch_size=bs, max_seq=max_seq, device=args.device,
        )
        r_z = safe_pearson(z_eff_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][mode_name] = {
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        print(f"  {mode_name:<14} r_x={r_x:+.3f} r_z={r_z:+.3f} "
              f"⟨LD⟩={ld.mean():+.2f} std={ld.std():.2f}  "
              f"({time.time()-t1:.1f}s)", flush=True)

    out_path = REPO / "results" / f"p2e_residual_interventions_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
