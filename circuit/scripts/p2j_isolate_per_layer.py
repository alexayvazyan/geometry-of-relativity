"""Phase 2J — per-layer context-number isolation.

Same XOR mask as Phase 2I (ctx_value positions ↔ non-numbers, bidirectional,
-1e9 added to attention scores) but applied at exactly ONE layer at a time.
Sweeps L from 0 to n_layers-1.

Each per-layer condition asks: "if context numbers cannot exchange attention
with non-numbers ONLY at this layer (and freely everywhere else), is z still
encoded?" — locating the layer where the read happens.

Saves per-layer (r_ld_zeff, r_ld_x, mean_ld, std_ld) and a baseline.

Usage:
  python3 scripts/p2j_isolate_per_layer.py --model gemma2-9b --pair height
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

sys.path.insert(0, str(REPO / "scripts"))
from p2_extract_attn import (  # noqa: E402
    PAIR_LINE_REGEX, find_value_char_ranges, char_to_token_range,
    get_decoder_layers, MODEL_ID,
)
from p2i_isolate_numbers import (  # noqa: E402
    safe_pearson, build_ctx_bool, build_xor_mask,
)


def run_one_layer(model, tok, decoder, target_layer, *, addn_mask, trials,
                  high_id, low_id, batch_size, max_seq, device):
    state = {"batch_addn": None}

    def hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        ad = state["batch_addn"]
        if ad is None:
            return args, kwargs
        if am is not None:
            kwargs["attention_mask"] = am + ad
        else:
            kwargs["attention_mask"] = ad
        return args, kwargs

    handles = []
    if target_layer is not None:
        h = decoder[target_layer].self_attn.register_forward_pre_hook(
            hook, with_kwargs=True
        )
        handles.append(h)

    n = len(trials)
    ld = np.zeros(n, dtype=np.float64)
    try:
        with torch.inference_mode():
            for b0 in range(0, n, batch_size):
                batch = trials[b0:b0 + batch_size]
                prompts = [t["prompt"] for t in batch]
                enc = tok(prompts, return_tensors="pt", padding="max_length",
                          max_length=max_seq, truncation=True).to(device)
                if target_layer is not None and addn_mask is not None:
                    state["batch_addn"] = torch.from_numpy(
                        addn_mask[b0:b0 + len(batch)]
                    ).to(device).to(model.dtype)
                else:
                    state["batch_addn"] = None
                out = model(input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask, use_cache=False)
                logits_last = out.logits[:, -1, :].float()
                ld[b0:b0 + len(batch)] = (
                    logits_last[:, high_id] - logits_last[:, low_id]
                ).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return ld


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", default="height")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer-step", type=int, default=1,
                    help="sweep every Nth layer (1 = all)")
    args = p.parse_args()
    bs = args.batch_size or (16 if args.model == "gemma2-2b" else 4)

    print(f"[p2j] loading {MODEL_ID[args.model]} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID[args.model])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID[args.model], dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    decoder = get_decoder_layers(model)
    n_layers = len(decoder)

    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    low_word = all_trials[0]["low_word"]
    high_word = all_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)

    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    rng = np.random.default_rng(0)
    test_idx = np.where(cs_arr != 0)[0]
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4
    print(f"[p2j] {args.model}/{args.pair} k={args.k}: n={n_test} max_seq={max_seq} "
          f"layers={n_layers}", flush=True)

    ctx_b = build_ctx_bool(trials, tok, args.pair, max_seq, include_target=False)
    addn_mask = build_xor_mask(ctx_b, mode="all")

    out = {"model": args.model, "pair": args.pair, "k": args.k,
           "n_layers": n_layers, "n": n_test, "results": {}}

    print(f"\n  layer  r_x       r_z       ⟨LD⟩      std        elapsed")
    # Baseline first.
    t1 = time.time()
    ld = run_one_layer(model, tok, decoder, target_layer=None,
                        addn_mask=None, trials=trials,
                        high_id=high_id, low_id=low_id,
                        batch_size=bs, max_seq=max_seq, device=args.device)
    r_z = safe_pearson(z_test, ld); r_x = safe_pearson(x_test, ld)
    out["results"]["baseline"] = {"layer": None, "r_ld_zeff": r_z, "r_ld_x": r_x,
                                    "mean_ld": float(ld.mean()),
                                    "std_ld": float(ld.std(ddof=1)), "n": n_test}
    print(f"  base   {r_x:+.3f}    {r_z:+.3f}    {ld.mean():+.2f}    {ld.std():.2f}    "
          f"({time.time()-t1:.1f}s)", flush=True)

    layers = list(range(0, n_layers, args.layer_step))
    for L in layers:
        t1 = time.time()
        ld = run_one_layer(model, tok, decoder, target_layer=L,
                            addn_mask=addn_mask, trials=trials,
                            high_id=high_id, low_id=low_id,
                            batch_size=bs, max_seq=max_seq, device=args.device)
        r_z = safe_pearson(z_test, ld); r_x = safe_pearson(x_test, ld)
        out["results"][f"L{L}"] = {"layer": L, "r_ld_zeff": r_z, "r_ld_x": r_x,
                                     "mean_ld": float(ld.mean()),
                                     "std_ld": float(ld.std(ddof=1)), "n": n_test}
        print(f"  L{L:>2}    {r_x:+.3f}    {r_z:+.3f}    {ld.mean():+.2f}    "
              f"{ld.std():.2f}    ({time.time()-t1:.1f}s)", flush=True)

    out_path = REPO / "results" / f"p2j_isolate_per_layer_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
