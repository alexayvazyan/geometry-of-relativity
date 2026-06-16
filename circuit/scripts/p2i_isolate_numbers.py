"""Phase 2I — fully isolate context-value tokens from everything else, at
every layer.

We add a -1e9 attention bias at (q, k) wherever exactly one of q, k is a
ctx_value position (XOR). Effect:
  - the 15 context-number tokens can only attend to themselves (+ each other,
    where causal allows) — they do not see scaffolding / target / BOS.
  - everything else (BOS, scaffolding, target, trailer) cannot attend to the
    context numbers.

So context numerical info has no attention-mediated path from its source
positions to anywhere else. The only remaining path for "context number i
contributed to target's residual" is via the chain of MLP outputs at the
number's own position — and since MLPs are pointwise, that doesn't help
the target's residual.

Variants tested:
  baseline       no mask
  iso_all        bidirectional isolation at every layer
  iso_keys_only  keys-only isolation (one direction): non-numbers cannot attend
                  to numbers, but numbers can still attend out (so they could
                  in principle still write something useful via residual at
                  their position, then be read elsewhere — won't happen because
                  pointwise MLP, but test it anyway)
  iso_all_plus_target  same as iso_all but TARGET value is also a "number"
                       (so trailer cannot see target value either — extreme test)

Reports r(LD, z_eff), r(LD, x), mean LD, std LD on the same 400-prompt
held-out fold used in Phase 2E/2H.

Usage:
  python3 scripts/p2i_isolate_numbers.py --model gemma2-9b --pair height
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


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def build_ctx_bool(trials, tok, pair, max_seq, include_target=False):
    """Per prompt: bool array of length max_seq; True where token is part of
    a ctx_value (or target_value if include_target). Padded positions are
    False; their attention is already -inf via the standard mask, doesn't
    matter."""
    n = len(trials)
    ctx_b = np.zeros((n, max_seq), dtype=bool)
    for i, t in enumerate(trials):
        prompt = t["prompt"]
        enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
        seq_len = len(enc.input_ids)
        ctx_ranges, tgt_range = find_value_char_ranges(prompt, pair)
        po = max_seq - seq_len  # left-pad shift
        for cr in ctx_ranges:
            s, e = char_to_token_range(enc.offset_mapping, cr)
            ctx_b[i, po + s:po + e] = True
        if include_target:
            s, e = char_to_token_range(enc.offset_mapping, tgt_range)
            ctx_b[i, po + s:po + e] = True
    return ctx_b


def build_xor_mask(ctx_b, mode):
    """Returns float32 (B, 1, T, T) mask, additive (-1e9 where masked, 0
    elsewhere).

    mode:
      'all'        bidirectional: -inf where ctx[q] XOR ctx[k]
      'keys_only'  one-way:       -inf where (NOT ctx[q]) AND ctx[k]
                    (non-numbers cannot attend to numbers; numbers can still
                     attend to non-numbers — though they're causally limited)
    """
    n, T = ctx_b.shape
    out = np.zeros((n, 1, T, T), dtype=np.float32)
    for b in range(n):
        c = ctx_b[b]
        if mode == "all":
            xor = np.bitwise_xor(c[:, None], c[None, :])  # (T, T)
            out[b, 0] = np.where(xor, -1e9, 0.0).astype(np.float32)
        elif mode == "keys_only":
            mask = (~c)[:, None] & c[None, :]
            out[b, 0] = np.where(mask, -1e9, 0.0).astype(np.float32)
        else:
            raise ValueError(mode)
    return out


def run_with_mask(model, tok, decoder, trials, *, addn_mask, high_id, low_id,
                  batch_size, max_seq, device):
    """Forward pass adding `addn_mask` (B, 1, T, T) to attention_mask at every
    layer. addn_mask=None → baseline."""
    state = {"batch_addn": None}

    def hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        ad = state["batch_addn"]
        if ad is None:
            return args, kwargs
        # ad: (B, 1, T, T) torch tensor on device
        if am is not None:
            kwargs["attention_mask"] = am + ad
        else:
            kwargs["attention_mask"] = ad
        return args, kwargs

    handles = []
    if addn_mask is not None:
        for L in range(len(decoder)):
            h = decoder[L].self_attn.register_forward_pre_hook(hook,
                                                                with_kwargs=True)
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
                if addn_mask is not None:
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
    args = p.parse_args()
    bs = args.batch_size or (16 if args.model == "gemma2-2b" else 4)

    print(f"[p2i] loading {MODEL_ID[args.model]} ...", flush=True)
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
    print(f"[p2i] words low={low_word!r}({low_id}) high={high_word!r}({high_id})",
          flush=True)

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
    print(f"[p2i] {args.model}/{args.pair} k={args.k}: n={n_test} max_seq={max_seq} "
          f"layers={n_layers}", flush=True)

    # Build ctx bool maps for both variants.
    ctx_b_default = build_ctx_bool(trials, tok, args.pair, max_seq,
                                    include_target=False)
    ctx_b_with_tgt = build_ctx_bool(trials, tok, args.pair, max_seq,
                                     include_target=True)
    # Pre-build masks (B, 1, T, T), kept in memory once.
    print(f"[p2i] building masks ...", flush=True)
    M_iso_all = build_xor_mask(ctx_b_default, mode="all")
    M_iso_keys = build_xor_mask(ctx_b_default, mode="keys_only")
    M_iso_all_plus_tgt = build_xor_mask(ctx_b_with_tgt, mode="all")
    print(f"[p2i] mask shapes: {M_iso_all.shape}", flush=True)

    out = {"model": args.model, "pair": args.pair, "k": args.k,
           "n_layers": n_layers, "n": n_test, "results": {}}

    print(f"\n  {'condition':<22}        r_x      r_z      ⟨LD⟩    std")
    for name, addn in [
        ("baseline",            None),
        ("iso_all",             M_iso_all),
        ("iso_keys_only",       M_iso_keys),
        ("iso_all_plus_target", M_iso_all_plus_tgt),
    ]:
        t1 = time.time()
        ld = run_with_mask(model, tok, decoder, trials,
                            addn_mask=addn, high_id=high_id, low_id=low_id,
                            batch_size=bs, max_seq=max_seq, device=args.device)
        r_z = safe_pearson(z_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][name] = {
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        print(f"  {name:<22}        {r_x:+.3f}   {r_z:+.3f}   "
              f"{ld.mean():+.2f}   {ld.std():.2f}    "
              f"({time.time()-t1:.1f}s)", flush=True)

    out_path = REPO / "results" / f"p2i_isolate_numbers_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
