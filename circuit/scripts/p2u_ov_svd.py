"""Phase 2U — OV-circuit SVD + L17H7 shuffle test.

(1) For each cell in the 7-cell feature-general core, decompose W_OV =
    W_V[kv] @ W_O[h] via SVD. Project top singular vectors onto unembed
    direction(s) and onto common-token embeddings to identify what
    tokens the head reads and writes.

(3) For L17H7, run a shuffle test: take a representative prompt, shuffle
    the context-number tokens (keeping positional frame), and check if
    the primacy bias (Phase 2R ρ_rank=−0.79) persists. If yes → positional
    primacy (RoPE-baked); if it dissolves → content/value-anchor.

Output:
  results/p2u_ov_svd_gemma2-2b.json  (per-head spectrum + projections)
  figures/p2u_ov_svd_top_directions.png (singular vectors visualized)
  figures/p2u_l17h7_shuffle_test.png (attention pattern: orig vs shuffled)
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
SHORT = "gemma2-2b"
MODEL = "google/gemma-2-2b"

# 7-cell feature-general core from Phase 2T
CORE_CELLS: list[tuple[int, int, str]] = [
    (16, 4, "trio (hub)"),
    (17, 7, "trio (cooperator)"),
    (14, 2, "trio (backup)"),
    (17, 6, "anti-z dual"),
    (22, 1, "beyond-Phase-2M"),
    (12, 4, "candidate"),
    (13, 2, "candidate"),
]

# Tokens to project onto. We'll resolve their IDs from the tokenizer.
PROBE_TOKENS = [
    " tall", " short", " big", " small", " large", " tiny", " huge",
    " heavy", " light", " fast", " slow",
    " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
    " Person", " cm", " kg",
]


def main() -> None:
    print(f"\nloading {MODEL}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    layers = model.model.layers
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    n_kv_heads = layers[0].self_attn.config.num_key_value_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    hidden = model.config.hidden_size
    group_size = n_heads // n_kv_heads
    print(f"  n_layers={n_layers} n_heads={n_heads} n_kv_heads={n_kv_heads}")
    print(f"  head_dim={head_dim} hidden={hidden} group_size={group_size}")

    # Resolve probe token IDs (last token of " <word>" since BPE often splits)
    probe_ids = {}
    for w in PROBE_TOKENS:
        ids = tok.encode(w, add_special_tokens=False)
        probe_ids[w.strip()] = ids[-1]
    # tall / short for the unembed direction
    tall_id = probe_ids["tall"]
    short_id = probe_ids["short"]
    print(f"  tall_id={tall_id}  short_id={short_id}")

    # Get unembed matrix W_U (vocab, hidden) and embedding W_E (vocab, hidden)
    # In Gemma 2 these are tied.
    W_U = model.lm_head.weight.detach().float().cpu().numpy()  # (vocab, hidden)
    W_E = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    tall_dir = W_U[tall_id] - W_U[short_id]   # (hidden,)
    tall_dir_unit = tall_dir / np.linalg.norm(tall_dir)

    # Build probe_token_dirs for visualization
    probe_unembed = {tok_str: W_U[tid] / np.linalg.norm(W_U[tid])
                      for tok_str, tid in probe_ids.items()}
    probe_embed = {tok_str: W_E[tid] / np.linalg.norm(W_E[tid])
                    for tok_str, tid in probe_ids.items()}

    # === (1) OV-SVD per cell ===
    print("\n=== OV-SVD per cell ===")
    results = []
    for L, h, role in CORE_CELLS:
        kv_h = h // group_size
        # W_V_kv is the V-projection slice for kv_head kv_h.
        # v_proj.weight shape: (n_kv_heads * head_dim, hidden)
        # nn.Linear stores weight transposed: y = x @ W.T. So .weight is (out, in).
        # W_V_kv slice: rows [kv_h*head_dim : (kv_h+1)*head_dim], shape (head_dim, hidden).
        # In matmul: V[t] = residual[t] @ W_V_full.T[:, slice] = residual[t] @ W_V_kv.T
        # So W_V_kv.T is shape (hidden, head_dim) — the projection matrix.
        v_weight = layers[L].self_attn.v_proj.weight.detach().float().cpu().numpy()
        W_V_kv = v_weight[kv_h * head_dim:(kv_h + 1) * head_dim, :]  # (head_dim, hidden)
        # o_proj.weight shape: (hidden, n_heads * head_dim)
        o_weight = layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
        W_O_h = o_weight[:, h * head_dim:(h + 1) * head_dim]  # (hidden, head_dim)

        # W_OV[h] : (hidden, hidden)
        # x @ W_V_kv.T = (head_dim,) value vector
        # value @ W_O_h.T = (hidden,) output residual contribution
        # So W_OV = W_V_kv.T @ W_O_h.T  (hidden, hidden), rank ≤ head_dim.
        # Equivalently: x @ W_OV gives the contribution.
        # SVD of (hidden, hidden) is wasteful; instead SVD W_V_kv.T (hidden, head_dim)
        # and W_O_h.T (head_dim, hidden) gives effective SVD of W_OV.
        # Easiest: directly form W_OV as W_V_kv.T @ W_O_h.T and SVD with full_matrices=False
        # (hidden, head_dim) @ (head_dim, hidden) -> (hidden, hidden), but SVD only top
        # head_dim singular values are nonzero, so use truncated SVD by exploiting the
        # factorization:
        # W_OV = (W_V_kv.T) @ (W_O_h.T) where ranks are head_dim
        # SVD of W_OV equals SVD of the head_dim composition:
        # Let A = W_V_kv.T (hidden, head_dim), B = W_O_h.T (head_dim, hidden)
        # W_OV = A @ B
        # SVD: do thin QR on A and B, then SVD the small middle matrix.
        # For simplicity at hidden=2304, compute W_OV directly and use truncated np.linalg.svd
        # full_matrices=False gives U(hidden, hidden), s(hidden,), Vt(hidden, hidden)
        # but only first head_dim are nonzero.
        # Cheaper: SVD of B @ A.T (which has shape (head_dim, head_dim)) gives σ², then back
        # to U/Vt via projections. For simplicity here, use np.linalg.svd directly on
        # the smaller representation:
        # W_OV.T = B.T @ A.T = W_O_h @ W_V_kv -> shape (hidden, hidden) again; same issue.
        # So just compute the small-rank SVD via:
        # Let A = W_V_kv.T; SVD A = U_A @ S_A @ Vt_A  with U_A(hidden, head_dim), S_A(head_dim,), Vt_A(head_dim, head_dim)
        # Then W_OV = A @ B = U_A @ S_A @ Vt_A @ B
        # Let M = S_A @ Vt_A @ B  (head_dim, hidden)
        # SVD of M: U_M, S, Vt   with U_M(head_dim, head_dim), S(head_dim,), Vt(head_dim, hidden)
        # Then W_OV = U_A @ U_M @ S @ Vt
        # Top singular values of W_OV = top singular values of M.
        # left singular vectors of W_OV = U_A @ U_M.
        # right singular vectors = Vt.
        A = W_V_kv.T  # (hidden, head_dim)
        B = W_O_h.T   # (head_dim, hidden)
        U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)
        M = (S_A[:, None] * Vt_A) @ B  # (head_dim, hidden)
        U_M, S, Vt = np.linalg.svd(M, full_matrices=False)
        U = U_A @ U_M  # (hidden, head_dim) - left singular vectors
        # Vt is (head_dim, hidden) - right singular vectors as rows
        # S is the singular values of W_OV.

        # Effective rank: number of singular values > 0.1 * top
        eff_rank = int(np.sum(S > 0.1 * S[0]))

        # Project top right singular vectors onto unembed direction
        K = 5
        write_proj = {}
        read_proj = {}
        for k in range(K):
            v_right = Vt[k]  # (hidden,) write direction
            u_left = U[:, k]  # (hidden,) read direction
            # Project onto tall_dir (signed cos similarity)
            v_right_unit = v_right / np.linalg.norm(v_right)
            u_left_unit = u_left / np.linalg.norm(u_left)
            wp = {tok_str: float(np.dot(v_right_unit, dir_))
                   for tok_str, dir_ in probe_unembed.items()}
            rp = {tok_str: float(np.dot(u_left_unit, dir_))
                   for tok_str, dir_ in probe_embed.items()}
            wp["__tall_minus_short__"] = float(np.dot(v_right_unit,
                                                       tall_dir_unit))
            write_proj[k] = wp
            read_proj[k] = rp

        # The total OV-direction's alignment with tall_dir, summed over singular values:
        # head's "static" tall-short bias direction:
        # head writes contribution = x @ W_OV; project this output direction
        # onto tall_dir, weighted by σ over basis. The expected signed alignment:
        #   sum_k σ_k * (Vt[k] · tall_dir_unit) * (norm of typical input projection on U[:,k])
        # For a dimensionless geometric measure: just report top-1's alignment.
        top1_tall_align = write_proj[0]["__tall_minus_short__"]

        results.append({
            "L": L, "h": h, "role": role, "kv_h": kv_h,
            "singular_values": [float(s) for s in S[:10]],
            "effective_rank": eff_rank,
            "top1_tall_align": top1_tall_align,
            "write_proj_top5": write_proj,
            "read_proj_top5": read_proj,
        })
        # Build a top-K summary for top-1 write direction:
        wp_top1 = write_proj[0]
        top_pos = sorted(wp_top1.items(), key=lambda kv: -kv[1])[:3]
        top_neg = sorted(wp_top1.items(), key=lambda kv: kv[1])[:3]
        rp_top1 = read_proj[0]
        rp_top = sorted(rp_top1.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"\nL{L}H{h} ({role})")
        print(f"  top-5 σ: {[f'{s:.2f}' for s in S[:5]]}")
        print(f"  effective rank @0.1: {eff_rank}/{head_dim}")
        print(f"  top-1 write direction · unembed(tall − short): "
              f"{top1_tall_align:+.3f}")
        print(f"  top-1 write promotes (positive cos with unembed):")
        for tok_str, c in top_pos:
            print(f"    +{tok_str:>10s}: {c:+.3f}")
        print(f"  top-1 write demotes (negative cos with unembed):")
        for tok_str, c in top_neg:
            print(f"    -{tok_str:>10s}: {c:+.3f}")
        print(f"  top-1 read activated by (cos with embedding):")
        for tok_str, c in rp_top:
            print(f"    {tok_str:>10s}: {c:+.3f}")

    out_json = REPO / "results" / f"p2u_ov_svd_{SHORT}.json"
    out_json.write_text(json.dumps({
        "model": MODEL,
        "core_cells": [{"L": L, "h": h, "role": r} for L, h, r in CORE_CELLS],
        "tall_dir_norm": float(np.linalg.norm(tall_dir)),
        "results": results,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # === (3) L17H7 shuffle test ===
    print("\n=== L17H7 shuffle test ===")
    # Pick a representative prompt (z=0 region from height k=15 stimuli)
    rows_all = [json.loads(l) for l in
                 (REPO / "data/p2_shot_sweep/height_k15.jsonl").open()]
    z_arr = np.array([r.get("z_eff", r.get("z", 0)) for r in rows_all])
    rep_i = int(np.argmin(np.abs(z_arr)))
    base_prompt = rows_all[rep_i]["prompt"]
    print(f"base prompt (idx {rep_i}, z={z_arr[rep_i]:+.2f}):")
    print(f"  {base_prompt[:120]}...")

    # Make a shuffled version by reordering Person N: <num> cm lines
    # Specifically, take the 15 context lines, shuffle them, keep target line.
    lines = base_prompt.split("\n")
    # Identify context lines vs target line: target has "This person is" tail
    context_lines = []
    tail_line = None
    for line in lines:
        if "This person is" in line or ". This" in line:
            tail_line = line
        elif re.match(r"Person \d+:", line):
            context_lines.append(line)
        else:
            tail_line = line if tail_line is None else tail_line  # fallback
    # The target may be on the same line as "Person 16: <x> cm. This person is"
    # In that case context_lines may include 15 entries and tail_line is
    # "Person 16: <x> cm. This person is"
    print(f"  context lines: {len(context_lines)}, tail: {tail_line[:60]}...")

    # Reorder with a fixed shuffle (so the specific ordering is reproducible)
    rng = random.Random(42)
    shuffled_lines = context_lines.copy()
    rng.shuffle(shuffled_lines)
    # But to test primacy due to *position*: KEEP positions 1..15 the same numerical
    # labels, swap which Person-N gets which value. Easiest: extract the values,
    # shuffle, reassign with original Person labels.
    # Each line is "Person N: <val> cm". Extract <val> per line.
    pat = re.compile(r"^(Person \d+:)\s*(\d+)\s*(.*)$")
    parts = []
    for line in context_lines:
        m = pat.match(line)
        if m:
            parts.append((m.group(1), m.group(2), m.group(3)))
    if len(parts) == len(context_lines):
        vals = [p[1] for p in parts]
        rng.shuffle(vals)
        new_context_lines = [f"{p[0]} {v} {p[2]}".strip()
                              for p, v in zip(parts, vals)]
    else:
        new_context_lines = shuffled_lines  # fallback
    shuffled_prompt = "\n".join(new_context_lines + [tail_line])
    print(f"shuffled prompt (first 120 chars):")
    print(f"  {shuffled_prompt[:120]}...")

    # Forward both prompts, capture L17H7 attention from last token
    def get_attn_l17h7(prompt: str) -> tuple[np.ndarray, list[str]]:
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inp, output_attentions=True, use_cache=False)
        attn_l17 = out.attentions[17][0, 7, -1, :].float().cpu().numpy()
        token_strs = [tok.decode([i]) for i in inp.input_ids[0].tolist()]
        return attn_l17, token_strs

    attn_orig, tokens_orig = get_attn_l17h7(base_prompt)
    attn_shuf, tokens_shuf = get_attn_l17h7(shuffled_prompt)
    # Compute per-context-position attention sum (rank #1..15)
    def context_position_attn(attn: np.ndarray, tokens: list[str]) -> np.ndarray:
        # Find context number runs by detecting digit-only tokens in sequence
        position_sums = []
        in_run = False
        cur_sum = 0.0
        n_runs = 0
        for i, tk in enumerate(tokens):
            decoded = tk.strip()
            if decoded.isdigit():
                cur_sum += attn[i]
                in_run = True
            else:
                if in_run:
                    position_sums.append(cur_sum)
                    cur_sum = 0.0
                    in_run = False
                    n_runs += 1
        # Skip the LAST number run (target); only return first 15
        return np.array(position_sums[:15])

    pos_orig = context_position_attn(attn_orig, tokens_orig)
    pos_shuf = context_position_attn(attn_shuf, tokens_shuf)

    print(f"\nL17H7 attention to context positions 1..15 (sums per number):")
    print(f"  ORIG : {[f'{x:.3f}' for x in pos_orig]}")
    print(f"  SHUF : {[f'{x:.3f}' for x in pos_shuf]}")
    rho_orig = np.corrcoef(np.arange(len(pos_orig)) + 1, pos_orig)[0, 1]
    rho_shuf = np.corrcoef(np.arange(len(pos_shuf)) + 1, pos_shuf)[0, 1]
    print(f"\n  Pearson(rank, attention) ORIG: {rho_orig:+.3f}")
    print(f"  Pearson(rank, attention) SHUF: {rho_shuf:+.3f}")
    print(f"\n  → If primacy is POSITIONAL, both should be ~ −0.5 to −0.8.")
    print(f"  → If primacy is CONTENT (value-anchor), shuffled should be ~ 0.")

    # Plot the two attention patterns side by side
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].bar(np.arange(1, len(pos_orig) + 1), pos_orig, color="tab:blue",
                 edgecolor="black", linewidth=0.4)
    axes[0].set_title(f"L17H7 attention to context numbers — ORIGINAL\n"
                       f"Pearson(rank, attention) = {rho_orig:+.3f}",
                       fontsize=11)
    axes[0].set_ylabel("attention sum (per context number)")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(np.arange(1, len(pos_shuf) + 1), pos_shuf, color="tab:orange",
                 edgecolor="black", linewidth=0.4)
    axes[1].set_title(f"L17H7 attention to context numbers — SHUFFLED VALUES "
                       f"(positions identical, values reassigned)\n"
                       f"Pearson(rank, attention) = {rho_shuf:+.3f}",
                       fontsize=11)
    axes[1].set_ylabel("attention sum")
    axes[1].set_xlabel("context-number position rank (Person 1..15)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2u_l17h7_shuffle_test.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
