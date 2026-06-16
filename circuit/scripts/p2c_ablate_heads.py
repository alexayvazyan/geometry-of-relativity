"""Phase 2C — causal head ablation.

Zero out specified (layer, head) contributions via a forward_pre_hook on
o_proj that masks the per-head slice of o_proj's input. Capture residuals
and LD; compute residual r²(z_eff) at every layer to localize the causal
effect.

We test:

  baseline         - no ablation (sanity)
  primary          - the primary comparator head (2B L1H6, 9B L1H11)
  primary_plus     - primary + decaying auxiliaries
  l1_all           - all heads at L1 (upper bound for L1's contribution)
  l0_all           - all heads at L0 (the bag-of-context aggregator)
  random_single    - 1 random head from L1 not in the candidate set
  random_set       - same number of random heads as `primary_plus`

Per ablation per k, save:
  residuals[N, n_layers, d_model] (subsample for storage)
  LD per prompt
  r²(z_eff) per layer
  r(LD, z_eff)

Usage:
  python scripts/p2c_ablate_heads.py --model gemma2-2b --pair height
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

# Candidate comparator heads found in Phase 2B.
COMPARATOR_HEADS = {
    "gemma2-2b": {
        "primary":      [(1, 6)],
        "primary_plus": [(1, 6), (1, 0), (1, 1), (1, 4)],
        "l1_all":       [(1, h) for h in range(8)],
        "l0_all":       [(0, h) for h in range(8)],
        # random_single / random_set are filled at runtime.
    },
    "gemma2-9b": {
        "primary":      [(1, 11)],
        "primary_plus": [(1, 11), (1, 10), (1, 6)],
        "l1_all":       [(1, h) for h in range(16)],
        "l0_all":       [(0, h) for h in range(16)],
    },
}


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("can't find decoder.layers")


def install_ablation_hooks(model, ablate_heads, head_dim, n_heads):
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
                x = args[0]   # (B, T, n_heads*head_dim)
                x = x.clone()
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
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


@torch.inference_mode()
def run_one_ablation(model, tok, trials, ablate_heads, head_dim, n_heads,
                      high_id, low_id, device, batch_size, max_seq):
    """Returns (residuals, ld) over all prompts."""
    decoder = get_decoder_layers(model)
    n_layers = len(decoder)
    n = len(trials)

    handles_ablate = install_ablation_hooks(model, ablate_heads, head_dim, n_heads)

    captured_resid: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    def make_resid_hook(L):
        def fn(_m, _ins, out):
            h = out[0] if isinstance(out, tuple) else out
            captured_resid[L].append(
                h[:, -1, :].detach().float().cpu().numpy().astype(np.float16)
            )
        return fn

    handles_resid = []
    for L in range(n_layers):
        handles_resid.append(decoder[L].register_forward_hook(make_resid_hook(L)))

    ld = np.zeros(n, dtype=np.float64)
    try:
        for b0 in range(0, n, batch_size):
            batch = trials[b0:b0 + batch_size]
            prompts = [t["prompt"] for t in batch]
            enc = tok(prompts, return_tensors="pt", padding="max_length",
                      max_length=max_seq, truncation=True).to(device)
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        use_cache=False)
            logits_last = out.logits[:, -1, :].float()
            ld[b0:b0 + len(batch)] = (
                logits_last[:, high_id] - logits_last[:, low_id]
            ).cpu().numpy()
    finally:
        for h in handles_resid + handles_ablate:
            h.remove()

    d_model = captured_resid[0][0].shape[-1]
    residuals = np.zeros((n, n_layers, d_model), dtype=np.float16)
    for L in range(n_layers):
        residuals[:, L, :] = np.concatenate(captured_resid[L], axis=0)
    return residuals, ld


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 15])
    p.add_argument("--n-prompts", type=int, default=600,
                   help="cap per (k) for r² CV speed")
    p.add_argument("--batch-size", type=int, default=None,
                   help="default 32 for 2B, 8 for 9B")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[p2c] loading {MODEL_ID[args.model]}...", flush=True)
    t0 = time.time()
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
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"[p2c]   loaded in {time.time()-t0:.1f}s | L={n_layers} H={n_heads} hd={head_dim}",
          flush=True)

    low_word = {"height": "short", "weight": "light", "speed": "slow"}[args.pair]
    high_word = {"height": "tall", "weight": "heavy", "speed": "fast"}[args.pair]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    # Build random-control configs that match sizes of primary / primary_plus.
    rng = np.random.default_rng(0)
    candidate_set = set(tuple(x) for x in COMPARATOR_HEADS[args.model]["primary_plus"])
    candidate_l1 = set(h for L, h in candidate_set if L == 1)
    # random_single: pick 1 random head at L1 not in candidates
    avail_l1 = [h for h in range(n_heads) if h not in candidate_l1]
    rs1 = (1, int(rng.choice(avail_l1)))
    # random_set: pick same count as primary_plus, randomly across L0+L1
    n_pp = len(COMPARATOR_HEADS[args.model]["primary_plus"])
    avail_pool = [(L, h) for L in (0, 1) for h in range(n_heads)
                   if (L, h) not in candidate_set]
    idxs = rng.choice(len(avail_pool), size=n_pp, replace=False)
    rs_set = [tuple(avail_pool[int(i)]) for i in idxs]

    ablations = {
        "baseline":      [],
        "primary":       COMPARATOR_HEADS[args.model]["primary"],
        "primary_plus":  COMPARATOR_HEADS[args.model]["primary_plus"],
        "l1_all":        COMPARATOR_HEADS[args.model]["l1_all"],
        "l0_all":        COMPARATOR_HEADS[args.model]["l0_all"],
        "random_single": [rs1],
        "random_set":    rs_set,
    }

    print(f"[p2c] ablation configs:")
    for name, hh in ablations.items():
        print(f"  {name:<14} = {hh}")

    out_summary = {"model": args.model, "pair": args.pair,
                    "ablations": {n: [list(h) for h in hh]
                                   for n, hh in ablations.items()},
                    "results": {}}

    for k in args.k:
        in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{k}.jsonl"
        all_trials = [json.loads(l) for l in in_path.open()]
        if args.n_prompts and len(all_trials) > args.n_prompts:
            idx = rng.choice(len(all_trials), size=args.n_prompts, replace=False)
            trials = [all_trials[int(i)] for i in idx]
        else:
            trials = all_trials
        n = len(trials)
        z_eff = np.array([t["z_eff"] for t in trials], dtype=np.float64)
        # max_seq from per-prompt tokenization
        max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4

        print(f"\n[p2c] {args.model}/{args.pair} k={k}: n={n} max_seq={max_seq}",
              flush=True)
        out_summary["results"][f"k{k}"] = {}

        for ab_name, heads in ablations.items():
            t1 = time.time()
            residuals, ld = run_one_ablation(
                model, tok, trials, heads, head_dim, n_heads,
                high_id, low_id, args.device, bs, max_seq,
            )
            r2_per_layer = []
            for L in range(n_layers):
                r2_per_layer.append(cv_r2(residuals[:, L, :].astype(np.float64), z_eff))
            r_ld_z = safe_pearson(z_eff, ld)
            mean_ld = float(ld.mean())
            std_ld = float(ld.std(ddof=1)) if len(ld) > 1 else 0.0
            print(f"  {ab_name:<14}  r(LD,z_eff)={r_ld_z:+.3f}  "
                  f"r²(z) at L1={r2_per_layer[1]:.2f}  L4={r2_per_layer[4]:.2f}  "
                  f"⟨LD⟩={mean_ld:+.2f}  ({time.time()-t1:.1f}s)", flush=True)

            out_summary["results"][f"k{k}"][ab_name] = {
                "n": int(n),
                "r2_per_layer": [float(v) for v in r2_per_layer],
                "r_ld_zeff": r_ld_z,
                "mean_ld": mean_ld,
                "std_ld": std_ld,
                # save subsampled LD + z_eff so we can replot scatter
                "ld_sample": ld[:200].tolist(),
                "z_eff_sample": z_eff[:200].tolist(),
            }

    out_path = REPO / "results" / f"p2c_ablation_{args.model}_{args.pair}.json"
    with out_path.open("w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
