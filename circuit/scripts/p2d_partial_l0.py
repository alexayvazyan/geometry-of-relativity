"""Phase 2D — partial L0 ablation: per-head sweep + cumulative top-k.

Three sub-experiments per (model, k):
  1. SINGLE — ablate each L0 head individually (n_heads runs per model)
  2. CUMUL  — rank heads by |Δr(LD, z_eff)| from SINGLE; ablate top-1, top-2, ..., top-n
  3. RAND   — for each cardinality 1..n, ablate 3 random subsets of L0 heads

We record per ablation:
  r(LD, z_eff), r(LD, x),  ⟨LD⟩, std(LD)
  residual r²(z_eff) at L1, L4   (downstream recovery check)

Output:
  results/p2d_partial_l0_<model>_<pair>_k<k>.json

Usage:
  python scripts/p2d_partial_l0.py --model gemma2-9b --pair height --k 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
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


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("can't find decoder.layers")


def install_hooks(model, ablate_heads, head_dim):
    """ablate_heads: list of (layer, head)."""
    decoder = get_decoder_layers(model)
    by_layer: dict[int, list[int]] = {}
    for L, H in ablate_heads:
        by_layer.setdefault(L, []).append(H)
    handles = []
    for L, heads in by_layer.items():
        o_proj = decoder[L].self_attn.o_proj
        def make_hook(heads_to_zero):
            def pre_hook(module, args):
                x = args[0].clone()
                for h in heads_to_zero:
                    x[:, :, h * head_dim:(h + 1) * head_dim] = 0.0
                return (x,) + args[1:]
            return pre_hook
        handles.append(o_proj.register_forward_pre_hook(make_hook(list(heads))))
    return handles


def cv_r2(X, y, k=5):
    if y.std() < 1e-9 or X.shape[0] < k * 2:
        return 0.0
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


@torch.inference_mode()
def run_one(model, tok, trials, ablate_heads, head_dim, high_id, low_id,
            device, batch_size, max_seq, capture_layers=(1, 4)):
    """Returns dict of metrics for this ablation."""
    decoder = get_decoder_layers(model)
    n_layers = len(decoder)
    n = len(trials)

    handles_ablate = install_hooks(model, ablate_heads, head_dim)

    captured: dict[int, list[np.ndarray]] = {L: [] for L in capture_layers}
    def make_hook(L):
        def fn(_m, _ins, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L].append(h[:, -1, :].detach().float().cpu().numpy().astype(np.float16))
        return fn

    handles_resid = []
    for L in capture_layers:
        handles_resid.append(decoder[L].register_forward_hook(make_hook(L)))

    ld = np.zeros(n, dtype=np.float64)
    try:
        for b0 in range(0, n, batch_size):
            batch = trials[b0:b0 + batch_size]
            enc = tok([t["prompt"] for t in batch], return_tensors="pt",
                      padding="max_length", max_length=max_seq, truncation=True
                      ).to(device)
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask, use_cache=False)
            logits_last = out.logits[:, -1, :].float()
            ld[b0:b0+len(batch)] = (
                logits_last[:, high_id] - logits_last[:, low_id]
            ).cpu().numpy()
    finally:
        for h in handles_resid + handles_ablate:
            h.remove()

    z_eff = np.array([t["z_eff"] for t in trials], dtype=np.float64)
    x = np.array([t["x"] for t in trials], dtype=np.float64)
    out = {
        "r_ld_zeff": safe_pearson(z_eff, ld),
        "r_ld_x":    safe_pearson(x, ld),
        "mean_ld":   float(ld.mean()),
        "std_ld":    float(ld.std(ddof=1)),
        "n":         int(n),
    }
    for L in capture_layers:
        h = np.concatenate(captured[L], axis=0).astype(np.float64)
        out[f"r2_z_L{L}"] = cv_r2(h, z_eff)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-rand-draws", type=int, default=3)
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[p2d] loading {MODEL_ID[args.model]}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID[args.model])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID[args.model], dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"[p2d]   loaded in {time.time()-t0:.1f}s | n_heads={n_heads}", flush=True)

    low_word = {"height": "short", "weight": "light", "speed": "slow"}[args.pair]
    high_word = {"height": "tall", "weight": "heavy", "speed": "fast"}[args.pair]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    rng = np.random.default_rng(0)
    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    if args.n_prompts and len(all_trials) > args.n_prompts:
        idx = rng.choice(len(all_trials), size=args.n_prompts, replace=False)
        trials = [all_trials[int(i)] for i in idx]
    else:
        trials = all_trials
    n = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
    print(f"[p2d] {args.model}/{args.pair} k={args.k} n={n} max_seq={max_seq}", flush=True)

    results = {"model": args.model, "pair": args.pair, "k": args.k,
                "n_prompts": n, "n_heads": n_heads}

    # 1. Baseline + full l0_all
    print("[p2d] baseline...", flush=True)
    results["baseline"] = run_one(model, tok, trials, [], head_dim,
                                    high_id, low_id, args.device, bs, max_seq)
    print("[p2d] full L0 ablation (reference)...", flush=True)
    results["l0_all"] = run_one(model, tok, trials,
                                 [(0, h) for h in range(n_heads)], head_dim,
                                 high_id, low_id, args.device, bs, max_seq)

    # 2. Single-head sweep
    print("[p2d] SINGLE-HEAD sweep over L0...", flush=True)
    single = []
    for h in range(n_heads):
        t1 = time.time()
        r = run_one(model, tok, trials, [(0, h)], head_dim, high_id, low_id,
                    args.device, bs, max_seq)
        r["heads"] = [(0, h)]
        single.append(r)
        d_rldz = r["r_ld_zeff"] - results["baseline"]["r_ld_zeff"]
        d_rldx = r["r_ld_x"] - results["baseline"]["r_ld_x"]
        print(f"  L0H{h:>2}: Δr(LD,z)={d_rldz:+.3f} Δr(LD,x)={d_rldx:+.3f} "
              f"⟨LD⟩={r['mean_ld']:+.2f}  ({time.time()-t1:.1f}s)", flush=True)
    results["single"] = single

    # Rank heads by |Δr(LD, z_eff)| (most damaging first).
    base_rldz = results["baseline"]["r_ld_zeff"]
    ranking = sorted(range(n_heads),
                     key=lambda i: -abs(single[i]["r_ld_zeff"] - base_rldz))
    print(f"\n[p2d] ranking by |Δr(LD,z)|: {ranking}", flush=True)

    # 3. Cumulative top-k sweep
    print("\n[p2d] CUMULATIVE top-k sweep...", flush=True)
    cumul = []
    for k in range(1, n_heads + 1):
        t1 = time.time()
        heads_to_zero = [(0, ranking[j]) for j in range(k)]
        r = run_one(model, tok, trials, heads_to_zero, head_dim,
                    high_id, low_id, args.device, bs, max_seq)
        r["heads"] = heads_to_zero
        cumul.append(r)
        print(f"  top-{k:>2}: Δr(LD,z)={r['r_ld_zeff']-base_rldz:+.3f} "
              f"Δr(LD,x)={r['r_ld_x']-results['baseline']['r_ld_x']:+.3f} "
              f"⟨LD⟩={r['mean_ld']:+.2f}  ({time.time()-t1:.1f}s)", flush=True)
    results["cumul_topk"] = cumul

    # 4. Random-k sweep
    print(f"\n[p2d] RANDOM-k sweep ({args.n_rand_draws} draws per cardinality)...",
          flush=True)
    random_k = {}
    for k in range(1, n_heads + 1):
        runs = []
        for d in range(args.n_rand_draws):
            t1 = time.time()
            sel = rng.choice(n_heads, size=k, replace=False)
            heads = [(0, int(h)) for h in sel]
            r = run_one(model, tok, trials, heads, head_dim,
                        high_id, low_id, args.device, bs, max_seq)
            r["heads"] = heads
            runs.append(r)
        random_k[str(k)] = runs
        # Print summary
        rldz = np.array([r["r_ld_zeff"] for r in runs]) - base_rldz
        rldx = np.array([r["r_ld_x"] for r in runs]) - results["baseline"]["r_ld_x"]
        print(f"  rand-{k:>2}: Δr(LD,z)={rldz.mean():+.3f}±{rldz.std():.3f}  "
              f"Δr(LD,x)={rldx.mean():+.3f}±{rldx.std():.3f}",
              flush=True)
    results["random_k"] = random_k

    out_path = REPO / "results" / f"p2d_partial_l0_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
