"""Phase 2D — l0_all ablation across multiple k values for the phase-space plot.

Runs only baseline + l0_all for each k, both models. Cheap and focused:
just enough to anchor the phase-space-by-k figure.
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


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("can't find decoder.layers")


def install_l0_hooks(model, n_heads, head_dim):
    decoder = get_decoder_layers(model)
    o_proj = decoder[0].self_attn.o_proj
    def pre_hook(module, args):
        x = args[0].clone()
        x[:, :, :] = 0.0   # zero entire L0 attention output
        return (x,) + args[1:]
    return [o_proj.register_forward_pre_hook(pre_hook)]


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


@torch.inference_mode()
def run_one(model, tok, trials, ablate, head_dim, n_heads, high_id, low_id,
            device, batch_size, max_seq):
    handles = install_l0_hooks(model, n_heads, head_dim) if ablate else []
    n = len(trials)
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
        for h in handles:
            h.remove()

    z_eff = np.array([t["z_eff"] for t in trials], dtype=np.float64)
    x = np.array([t["x"] for t in trials], dtype=np.float64)
    return {
        "n": n,
        "r_ld_zeff": safe_pearson(z_eff, ld),
        "r_ld_x":    safe_pearson(x, ld),
        "mean_ld":   float(ld.mean()),
        "std_ld":    float(ld.std(ddof=1)) if n > 1 else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--ks", nargs="+", type=int, default=[0, 2, 5])
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[p2d-l0pk] loading {MODEL_ID[args.model]}...", flush=True)
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
    low_id = first_token_id(tok, "short")
    high_id = first_token_id(tok, "tall")

    rng = np.random.default_rng(0)
    out_path = REPO / "results" / f"p2d_l0all_per_k_{args.model}_height.json"
    out = {"model": args.model, "results": {}}
    for k in args.ks:
        in_path = REPO / "data" / "p2_shot_sweep" / f"height_k{k}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} missing")
            continue
        all_trials = [json.loads(l) for l in in_path.open()]
        if args.n_prompts and len(all_trials) > args.n_prompts:
            idx = rng.choice(len(all_trials), size=args.n_prompts, replace=False)
            trials = [all_trials[int(i)] for i in idx]
        else:
            trials = all_trials
        n = len(trials)
        max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
        print(f"\n[p2d-l0pk] {args.model}/height k={k}: n={n} max_seq={max_seq}",
              flush=True)
        b = run_one(model, tok, trials, ablate=False,
                    head_dim=head_dim, n_heads=n_heads,
                    high_id=high_id, low_id=low_id,
                    device=args.device, batch_size=bs, max_seq=max_seq)
        a = run_one(model, tok, trials, ablate=True,
                    head_dim=head_dim, n_heads=n_heads,
                    high_id=high_id, low_id=low_id,
                    device=args.device, batch_size=bs, max_seq=max_seq)
        out["results"][f"k{k}"] = {"baseline": b, "l0_all": a}
        print(f"  baseline: r_x={b['r_ld_x']:+.3f} r_z={b['r_ld_zeff']:+.3f} "
              f"⟨LD⟩={b['mean_ld']:+.2f} std={b['std_ld']:.2f}", flush=True)
        print(f"  l0_all:   r_x={a['r_ld_x']:+.3f} r_z={a['r_ld_zeff']:+.3f} "
              f"⟨LD⟩={a['mean_ld']:+.2f} std={a['std_ld']:.2f}", flush=True)

    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
