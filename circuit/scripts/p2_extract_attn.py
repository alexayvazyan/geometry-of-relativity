"""Phase 2B — generalized attention + per-head extractor for any k.

Captures:
  - residuals[N, n_layers, d_model]  fp16  (last-token at every block)
  - attn_last[N, n_layers, n_heads, max_seq]  fp16
        (attention probabilities at the LAST query position, every layer & head)
  - head_outs[N, n_layers, n_heads, head_dim]  fp16
        (per-head value-mix at the last position, BEFORE o_proj — the slice of
         x that passes through W_O[:, h*hd:(h+1)*hd])
  - last token logits LD = high - low + entropy + argmax.
  - position bookkeeping: seq_len_unpad[N], target_pos[N, 2],
    context_pos[N, K, 2], k[N], pad_offset[N] (left-pad shift in the batched encoding).

Designed to handle variable K per JSONL — generates K from the file, doesn't
assume 15 context items. Works on Gemma 2 dense attention. (Gemma 4's GQA is
out-of-scope here; we'll add it if 2B/9B circuits work warrants it.)

Usage:
  python scripts/p2_extract_attn.py --model gemma2-2b --pair height --k 1 4 15

Output:
  results/p2_attn/<model_short>/<pair>_k<k>.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GOR = REPO.parent / "geometry-of-relativity"
sys.path.insert(0, str(GOR / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

MODEL_ID = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-2b-it": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-9b-it": "google/gemma-2-9b-it",
}

# Per-pair line regex (matches Jaehoon's PAIR_LINE_REGEX in extract_v11_dense.py).
PAIR_LINE_REGEX = {
    "height":     r"^Person \d+: (\d+(?:\.\d+)?) cm$",
    "age":        r"^Person \d+: (\d+(?:\.\d+)?) years old$",
    "weight":     r"^Person \d+: (\d+(?:\.\d+)?) kg$",
    "size":       r"^Object \d+: (\d+(?:\.\d+)?) cm across$",
    "speed":      r"^Vehicle \d+: (\d+(?:\.\d+)?) km/h$",
    "wealth":     r"^Person \d+ earns \$(\d+(?:\.\d+)?)/year$",
    "experience": r"^Worker \d+: (\d+(?:\.\d+)?) years experience$",
    "bmi_abs":    r"^Person \d+: BMI (\d+(?:\.\d+)?)$",
}


def find_value_char_ranges(prompt: str, pair: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return (context_char_ranges_K, target_char_range). K = #context_items
    inferred from the prompt (variable across k=0/1/2/4/8/15)."""
    pattern = re.compile(PAIR_LINE_REGEX[pair])
    matches: list[tuple[int, int]] = []
    char_pos = 0
    for line in prompt.split("\n"):
        clean = line
        for sep in (". This ", ". An ", ". A "):
            if sep in clean:
                clean = clean.split(sep, 1)[0]
                break
        m = pattern.match(clean)
        if m:
            matches.append((char_pos + m.start(1), char_pos + m.end(1)))
        char_pos += len(line) + 1
    if not matches:
        raise AssertionError(f"no value matches in prompt: {prompt[:200]!r}")
    # The target is always the LAST match (it ends in ". This person/object/... is")
    ctx = matches[:-1]
    tgt = matches[-1]
    return ctx, tgt


def char_to_token_range(offsets, char_range):
    c0, c1 = char_range
    tok_start = None
    tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e > c0 and s < c1:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    assert tok_start is not None, f"no tokens cover {char_range} in offsets"
    return tok_start, tok_end


def get_decoder_layers(model):
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("can't find decoder.layers")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_ID))
    p.add_argument("--pair", required=True, choices=list(PAIR_LINE_REGEX))
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 15])
    p.add_argument("--data-dir", default=str(REPO / "data" / "p2_shot_sweep"))
    p.add_argument("--out-dir", default=str(REPO / "results" / "p2_attn"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model_id = MODEL_ID[args.model]
    out_root = Path(args.out_dir) / args.model
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"[p2b] loading {model_id} (bf16, eager attn)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = text_cfg.num_hidden_layers
    n_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim",
                       text_cfg.hidden_size // text_cfg.num_attention_heads)
    d_model = text_cfg.hidden_size
    print(f"[p2b]   loaded in {time.time()-t0:.1f}s | layers={n_layers} "
          f"heads={n_heads} head_dim={head_dim} d={d_model}", flush=True)

    decoder = get_decoder_layers(model)

    low_word_default = {"height": "short", "weight": "light", "speed": "slow",
                        "age": "young", "size": "small", "wealth": "poor",
                        "experience": "novice", "bmi_abs": "thin"}
    high_word_default = {"height": "tall", "weight": "heavy", "speed": "fast",
                         "age": "old", "size": "big", "wealth": "rich",
                         "experience": "expert", "bmi_abs": "obese"}
    low_id = first_token_id(tok, low_word_default[args.pair])
    high_id = first_token_id(tok, high_word_default[args.pair])
    print(f"[p2b] low={low_word_default[args.pair]}({low_id}) "
          f"high={high_word_default[args.pair]}({high_id})", flush=True)

    for k in args.k:
        in_path = data_dir / f"{args.pair}_k{k}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found", flush=True)
            continue
        trials = [json.loads(l) for l in in_path.open()]
        n = len(trials)

        # Pre-tokenize for position bookkeeping (k may be 0; if so, no context).
        seq_len_unpad = np.zeros(n, dtype=np.int32)
        target_pos = np.zeros((n, 2), dtype=np.int32)
        K_max = max(k, 1)
        context_pos = np.full((n, K_max, 2), -1, dtype=np.int32)
        for i, t in enumerate(trials):
            enc_one = tok(t["prompt"], return_offsets_mapping=True, add_special_tokens=True)
            seq_len_unpad[i] = len(enc_one.input_ids)
            try:
                ctx_ranges, tgt_range = find_value_char_ranges(t["prompt"], args.pair)
            except Exception as e:
                raise RuntimeError(f"prompt {i}: {e}\nprompt: {t['prompt']!r}") from e
            target_pos[i] = char_to_token_range(enc_one.offset_mapping, tgt_range)
            for j, cr in enumerate(ctx_ranges):
                context_pos[i, j] = char_to_token_range(enc_one.offset_mapping, cr)
        max_seq = int(seq_len_unpad.max()) + 4  # small buffer

        # Allocate full-layer attention dump.
        # For k=15 on 9B: 990 * 42 * 16 * ~250 * 2 bytes ≈ 330 MB. Manageable.
        attn_last = np.zeros((n, n_layers, n_heads, max_seq), dtype=np.float16)
        head_outs = np.zeros((n, n_layers, n_heads, head_dim), dtype=np.float16)
        residuals = np.zeros((n, n_layers, d_model), dtype=np.float16)
        ld = np.zeros(n, dtype=np.float32)
        pad_offset = np.zeros(n, dtype=np.int32)

        # Hooks. We register on every layer to avoid running output_attentions
        # (which dumps the full T×T matrix per layer per batch — expensive).
        # Instead, hook on attn module's o_proj input to grab per-head outputs,
        # and use a forward hook on the layer to grab residuals. For attention
        # weights we DO need output_attentions=True; pull last-row only at
        # batch granularity to avoid memory blowup.
        captured_resid: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        captured_head: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

        def make_resid_hook(L):
            def fn(_m, _ins, out):
                h = out[0] if isinstance(out, tuple) else out
                captured_resid[L].append(
                    h[:, -1, :].detach().float().cpu().numpy().astype(np.float16)
                )
            return fn

        def make_oproj_pre_hook(L):
            def fn(_m, args_):
                x = args_[0]  # (B, T, n_heads*head_dim)
                last = x[:, -1, :].detach().float().cpu().numpy()
                captured_head[L].append(
                    last.reshape(last.shape[0], n_heads, head_dim).astype(np.float16)
                )
            return fn

        handles = []
        for L in range(n_layers):
            handles.append(decoder[L].register_forward_hook(make_resid_hook(L)))
            handles.append(decoder[L].self_attn.o_proj.register_forward_pre_hook(
                make_oproj_pre_hook(L)
            ))

        print(f"\n[p2b] {args.model}/{args.pair} k={k}: {n} prompts, max_seq={max_seq}", flush=True)
        t1 = time.time()
        try:
            for b0 in range(0, n, args.batch_size):
                batch = trials[b0:b0 + args.batch_size]
                prompts = [t["prompt"] for t in batch]
                enc = tok(prompts, return_tensors="pt", padding="max_length",
                          max_length=max_seq, truncation=True).to(args.device)
                # Compute pad_offset per row (number of pad tokens on the left).
                # tokenizer-level pad_id; works for left-padded encoding.
                pad_id = tok.pad_token_id
                pad_off = (enc.input_ids == pad_id).sum(dim=1).cpu().numpy()

                with torch.no_grad():
                    out = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                output_attentions=True, use_cache=False)
                # logits at last position
                logits_last = out.logits[:, -1, :].float()
                ld_b = (logits_last[:, high_id] - logits_last[:, low_id]).cpu().numpy()
                ld[b0:b0+len(batch)] = ld_b
                pad_offset[b0:b0+len(batch)] = pad_off
                # Attention last row at every layer.
                for L in range(n_layers):
                    A = out.attentions[L][:, :, -1, :].float().cpu().numpy()  # (B, n_heads, T_full)
                    # Slice or pad to max_seq.
                    T = A.shape[-1]
                    if T <= max_seq:
                        attn_last[b0:b0+len(batch), L, :, :T] = A.astype(np.float16)
                    else:
                        attn_last[b0:b0+len(batch), L] = A[:, :, :max_seq].astype(np.float16)
                if (b0 // args.batch_size) % 10 == 0:
                    done = b0 + len(batch)
                    rate = done / max(1e-3, time.time() - t1)
                    eta = (n - done) / max(1e-3, rate)
                    print(f"[p2b]   {done}/{n}  {rate:.1f} p/s  eta {eta:.0f}s", flush=True)
        finally:
            for h in handles:
                h.remove()
        print(f"[p2b] forward pass: {time.time()-t1:.1f}s", flush=True)

        # Consolidate residual / head buffers.
        for L in range(n_layers):
            residuals[:, L, :] = np.concatenate(captured_resid[L], axis=0)
            head_outs[:, L, :, :] = np.concatenate(captured_head[L], axis=0)

        # Pack metadata.
        ids = np.array([t["id"] for t in trials], dtype=object)
        x_arr = np.array([t["x"] for t in trials], dtype=np.float64)
        z_arr = np.array([t["z"] for t in trials], dtype=np.float64)
        z_eff = np.array([t["z_eff"] for t in trials], dtype=np.float64)
        mu_arr = np.array([t["mu"] for t in trials], dtype=np.float64)
        mu_eff = np.array([t["mu_eff"] for t in trials], dtype=np.float64)
        sigma = np.array([t["sigma"] for t in trials], dtype=np.float64)
        sigma_eff = np.array([t["sigma_eff"] for t in trials], dtype=np.float64)
        cell_seed = np.array([t["cell_seed"] for t in trials], dtype=np.int32)
        k_arr = np.array([t["k"] for t in trials], dtype=np.int32)

        out_path = out_root / f"{args.pair}_k{k}.npz"
        np.savez(
            out_path,
            id=ids, x=x_arr, z=z_arr, z_eff=z_eff,
            mu=mu_arr, mu_eff=mu_eff, sigma=sigma, sigma_eff=sigma_eff,
            cell_seed=cell_seed, k=k_arr,
            seq_len_unpad=seq_len_unpad,
            target_pos=target_pos,        # (N, 2) — start/end token within unpadded prompt
            context_pos=context_pos,      # (N, K_max, 2), -1 for unused slots
            pad_offset=pad_offset,        # (N,) — left-pad in batched encoding
            ld=ld,
            residuals=residuals,
            attn_last=attn_last,
            head_outs=head_outs,
            high_id=np.array([high_id], dtype=np.int64),
            low_id=np.array([low_id], dtype=np.int64),
        )
        size_mb = out_path.stat().st_size / 1024**2
        print(f"[p2b] -> {out_path}  ({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
