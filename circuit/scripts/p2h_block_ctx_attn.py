"""Phase 2H — causal test of the aggregation layer.

We saw in Phase 2G that target→ctx_value attention peaks sharply at L21 in
9B (and L16 in 2B), with mass concentrated in a narrow layer band. Question:
is that aggregation layer the place where z is *causally* built, or just an
incidental info-routing event?

Intervention: at a chosen layer L, hook v_proj forward and ZERO the V vector
at every ctx_value token position. Effect: target's attention at L still
distributes normally, but ctx_value positions contribute nothing — context
numbers cannot pass information through this layer's attention (they can
still pass through other layers). Other tokens (BOS, scaffolding, target)
are untouched.

We run the same n=400 held-out fold used in Phase 2E and report
r(LD, z_eff), r(LD, x), mean(LD), std(LD) per intervention.

Conditions per (model, pair):
  baseline                     no intervention
  block_peak                   peak aggregation layer (2B L16 / 9B L21)
  block_cluster                L14-18 (2B) or L20-24 (9B) — wider band
  block_control_low            an early non-aggregation layer (2B L4 / 9B L5)
  block_control_high           a late non-aggregation layer (2B L23 / 9B L36)

Usage:
  python3 scripts/p2h_block_ctx_attn.py --model gemma2-9b --pair height
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

LATE_LAYER = {"gemma2-2b": 20, "gemma2-9b": 33}

# Phase 2G aggregation peak per model.
PEAK = {"gemma2-2b": 16, "gemma2-9b": 21}
CLUSTER = {"gemma2-2b": list(range(14, 19)),       # L14..L18
           "gemma2-9b": list(range(20, 25))}       # L20..L24
CONTROL_LOW = {"gemma2-2b": 4, "gemma2-9b": 5}
CONTROL_HIGH = {"gemma2-2b": 23, "gemma2-9b": 36}


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-12 or y[mask].std() < 1e-12:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def build_ctx_position_lists(trials, tok, pair, max_seq):
    """For each prompt build the list of token positions (in batched encoding)
    that fall inside any ctx_value range. Returns list[np.ndarray]; each entry
    is sorted batched-coords positions.

    Also computes pad_offset assuming left-padding to max_seq.
    """
    ctx_positions = []
    for t in trials:
        prompt = t["prompt"]
        enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
        seq_len = len(enc.input_ids)
        ctx_ranges, _tgt_range = find_value_char_ranges(prompt, pair)
        # Token positions in unpadded encoding.
        un_positions = []
        for cr in ctx_ranges:
            s, e = char_to_token_range(enc.offset_mapping, cr)
            un_positions.extend(range(s, e))
        # Pad offset = max_seq - seq_len (left pad).
        po = max_seq - seq_len
        b_positions = np.array([po + p for p in un_positions], dtype=np.int64)
        ctx_positions.append(b_positions)
    return ctx_positions


def run_block(model, tok, decoder, trials, *, layers_to_block, ctx_positions,
              high_id, low_id, batch_size, max_seq, device):
    """Forward pass with V-zeroing hook on layers_to_block. Returns LD vector."""
    state = {"batch_ctx": None}  # list of np.ndarray, length=batch

    def make_hook():
        def hook(_m, _ins, output):
            cb = state["batch_ctx"]
            if cb is None:
                return output
            out = output.clone()
            for b, ps in enumerate(cb):
                if len(ps) > 0:
                    out[b, ps, :] = 0.0
            return out
        return hook

    handles = []
    for L in layers_to_block:
        h = decoder[L].self_attn.v_proj.register_forward_hook(make_hook())
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
                state["batch_ctx"] = ctx_positions[b0:b0 + len(batch)]
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
    bs = args.batch_size or (32 if args.model == "gemma2-2b" else 8)

    print(f"[p2h] loading {MODEL_ID[args.model]} ...", flush=True)
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

    in_path = REPO / "data" / "p2_shot_sweep" / f"{args.pair}_k{args.k}.jsonl"
    all_trials = [json.loads(l) for l in in_path.open()]
    low_word = all_trials[0]["low_word"]
    high_word = all_trials[0]["high_word"]
    low_id = first_token_id(tok, low_word)
    high_id = first_token_id(tok, high_word)
    print(f"[p2h] words low={low_word!r}({low_id}) high={high_word!r}({high_id})",
          flush=True)

    # Held-out fold = same as Phase 2E (cell_seed != 0 + first n_prompts).
    cs_arr = np.array([t["cell_seed"] for t in all_trials], dtype=np.int32)
    z_eff_arr = np.array([t["z_eff"] for t in all_trials], dtype=np.float64)
    x_arr = np.array([t["x"] for t in all_trials], dtype=np.float64)
    test_idx = np.where(cs_arr != 0)[0]
    rng = np.random.default_rng(0)
    if args.n_prompts and len(test_idx) > args.n_prompts:
        test_idx = rng.choice(test_idx, size=args.n_prompts, replace=False)
    trials = [all_trials[int(i)] for i in test_idx]
    z_test = z_eff_arr[test_idx]; x_test = x_arr[test_idx]
    n_test = len(trials)
    max_seq = max(len(tok(t["prompt"]).input_ids) for t in trials) + 4

    # Pre-compute per-prompt ctx_value positions (in batched coords).
    ctx_positions = build_ctx_position_lists(trials, tok, args.pair, max_seq)
    print(f"[p2h] ctx-position lists built; mean #ctx-tokens/prompt = "
          f"{np.mean([len(p) for p in ctx_positions]):.1f}",
          flush=True)
    print(f"[p2h] {args.model}/{args.pair} k={args.k}: n={n_test} max_seq={max_seq}",
          flush=True)

    conditions = [
        ("baseline",          []),
        ("block_peak",        [PEAK[args.model]]),
        ("block_cluster",     CLUSTER[args.model]),
        ("block_control_low",  [CONTROL_LOW[args.model]]),
        ("block_control_high", [CONTROL_HIGH[args.model]]),
    ]

    out = {"model": args.model, "pair": args.pair, "k": args.k,
           "peak": PEAK[args.model], "cluster": CLUSTER[args.model],
           "n": n_test, "results": {}}

    print(f"\n  {'condition':<20} {'layers':<22} r_x      r_z      ⟨LD⟩    std")
    for name, layers in conditions:
        t1 = time.time()
        ld = run_block(model, tok, decoder, trials,
                        layers_to_block=layers, ctx_positions=ctx_positions,
                        high_id=high_id, low_id=low_id,
                        batch_size=bs, max_seq=max_seq, device=args.device)
        r_z = safe_pearson(z_test, ld)
        r_x = safe_pearson(x_test, ld)
        out["results"][name] = {
            "layers": layers,
            "r_ld_zeff": r_z, "r_ld_x": r_x,
            "mean_ld": float(ld.mean()), "std_ld": float(ld.std(ddof=1)),
            "n": n_test,
        }
        layer_str = (str(layers) if len(layers) <= 5 else f"{layers[0]}..{layers[-1]}")
        print(f"  {name:<20} {layer_str:<22} {r_x:+.3f}   {r_z:+.3f}   "
              f"{ld.mean():+.2f}   {ld.std():.2f}    "
              f"({time.time()-t1:.1f}s)", flush=True)

    out_path = REPO / "results" / f"p2h_block_ctx_attn_{args.model}_{args.pair}_k{args.k}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
