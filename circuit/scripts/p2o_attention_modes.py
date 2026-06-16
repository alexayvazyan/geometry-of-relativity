"""Phase 2O — three attention-intervention modes on Phase 2M cluster cells.

For a chosen set of (layer, head) cells from the Phase 2M cosine grid,
run three intervention modes and measure r(LD, z):

  1. zero      : zero each cell's slice in pre-o_proj input at last token
                 (= what Phase 2N did; sanity baseline here)
  2. resample  : replace each cell's slice with the same head's slice from
                 a randomly permuted prompt — preserves "default" computation,
                 removes prompt-specific information
  3. q_zero    : zero head h's q_proj output → softmax sees q=0 → attention
                 pattern collapses to uniform → head loses positional
                 specificity but still copies a uniform mix of V's
                 (tests whether the head's *attention pattern* is necessary)

Cell selection: top-N by |cos| × ||Δ_a|| (combined importance from Phase 2M),
or by |cos| alone via --by-cos. N=8 by default.

Output:
  results/p2o_attention_modes_<short>.json
  figures/p2o_attention_modes_<short>.png

Compute: ~3 modes × 14-30s + pool capture ≈ 3-5 min on 2B / 5-8 min on 9B.

Usage:
  python p2o_attention_modes.py --short gemma2-2b --top-n 8
  python p2o_attention_modes.py --short gemma2-9b --top-n 8 --by-cos
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False; break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True,
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--top-n", type=int, default=8,
                    help="number of cells to ablate jointly (top by importance)")
    ap.add_argument("--by-cos", action="store_true",
                    help="rank by |cos| instead of |cos|×||Δ_a||")
    ap.add_argument("--score-source", choices=["phase2m", "dla", "resid_cos"],
                    default="phase2m",
                    help="phase2m = use Phase 2M cos grid (with --by-cos or "
                         "default |cos|×||Δ_a||); dla = use σ_DLA·|ρ(DLA, z)| "
                         "from Phase 2S full grid; resid_cos = use cos(head "
                         "writeout, d_primal) from p2u_residual_cos npz")
    ap.add_argument("--sign", choices=["both", "positive", "negative"],
                    default="both",
                    help="filter cells by sign of cos (phase2m) or ρ(DLA,z) (dla)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--out-suffix", default=None,
                    help="suffix for output filenames (default: derived from "
                         "ranking method)")
    args = ap.parse_args()

    # 1. Load Phase 2M cos_grid
    cos_path = REPO / "results" / f"p2m_alignment_{args.short}.json"
    cos_data = json.loads(cos_path.read_text())
    cos = np.array(cos_data["cos_grid"])
    norms = np.array(cos_data["delta_norms"])
    layer_idxs = cos_data["layer_idxs"]
    head_idxs = cos_data["head_idxs"]
    feature = cos_data["feature"]
    k = cos_data["k"]
    model_id = cos_data["model"]

    if args.score_source == "dla":
        dla_path = REPO / "results" / f"p2s_dla_full_{args.short}.npz"
        if not dla_path.exists():
            raise SystemExit(f"--score-source dla requires {dla_path.name}")
        dla = np.load(dla_path)
        std_dla_full = dla["std_dla"]
        rho_full = dla["rho_grid"]
        n_layers_full, n_heads_full = std_dla_full.shape
        score_mag = std_dla_full * np.abs(rho_full)
        sign_grid = rho_full
        layer_idxs = list(range(n_layers_full))
        head_idxs = list(range(n_heads_full))
        cos_full_grid = rho_full
        ranking_label = "σ_DLA·|ρ(DLA, z)|"
    elif args.score_source == "resid_cos":
        rc_path = REPO / "results" / f"p2u_residual_cos_{args.short}.npz"
        if not rc_path.exists():
            raise SystemExit(f"--score-source resid_cos requires {rc_path.name}")
        rc = np.load(rc_path)
        cos_full = rc["cos_grid"]
        n_layers_full, n_heads_full = cos_full.shape
        score_mag = np.abs(cos_full)
        sign_grid = cos_full
        layer_idxs = list(range(n_layers_full))
        head_idxs = list(range(n_heads_full))
        cos_full_grid = cos_full
        ranking_label = "|cos(head writeout, d_primal)|"
    elif args.by_cos:
        score_mag = np.abs(cos)
        sign_grid = cos
        cos_full_grid = cos
        ranking_label = "|cos|"
    else:
        score_mag = np.abs(cos) * norms
        sign_grid = cos
        cos_full_grid = cos
        ranking_label = "|cos|×||Δ_a||"

    # Apply sign filter
    if args.sign == "positive":
        score = np.where(sign_grid > 0, score_mag, -np.inf)
        ranking_label += " (positive only)"
    elif args.sign == "negative":
        score = np.where(sign_grid < 0, score_mag, -np.inf)
        ranking_label += " (negative only)"
    else:
        score = score_mag

    top_flat = np.argsort(score, axis=None)[::-1][:args.top_n]
    cells: list[tuple[int, int, float]] = []  # (L, H, annotation value)
    for k_flat in top_flat:
        li, hi = np.unravel_index(k_flat, cos_full_grid.shape)
        cells.append((layer_idxs[li], head_idxs[hi],
                       float(cos_full_grid[li, hi])))
    print(f"selected top-{args.top_n} cells (by {ranking_label}): ")
    if args.score_source == "dla":
        for (L, h, c) in cells:
            sd = std_dla_full[L, h]
            print(f"  L{L:>2d} H{h}: σ_DLA={sd:.3f}  ρ={c:+.3f}  "
                  f"σ·|ρ|={sd*abs(c):.4f}")
    else:
        for L, h, c in cells:
            print(f"  L{L:>2d} H{h}: cos={c:+.3f}")

    # 2. Load stimuli
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{feature}_k{k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    if args.n_prompts:
        rows = rows[:args.n_prompts]
    n = len(rows)
    print(f"\nloaded {n} prompts")

    # 3. Load model
    print(f"\nloading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]

    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)

    # 4. Baseline forward + capture pre-o_proj at each layer in cells
    layers_in_cells = sorted(set(L for L, _, _ in cells))
    print(f"\n[pass 1] baseline + capture pre-o_proj at layers {layers_in_cells}")
    pool: dict[int, np.ndarray] = {L: np.zeros((n, n_heads * head_dim), dtype=np.float32)
                                     for L in layers_in_cells}
    captured = {L: None for L in layers_in_cells}

    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook

    handles = []
    for L in layers_in_cells:
        h_cap = layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
        handles.append(h_cap)

    ld_baseline = np.zeros(n, dtype=np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            ld_baseline[i] = float(logits[high_id] - logits[low_id])
            for L in layers_in_cells:
                pool[L][i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()
    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}  <LD>={ld_baseline.mean():+.2f}")

    # 5. Forward with intervention
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def fwd_LD(intervention: str) -> np.ndarray:
        """intervention ∈ {'zero', 'resample', 'q_zero'}.
        Hooks all cells simultaneously; uses the relevant per-cell action."""
        handles = []
        state = {"i": 0}

        if intervention in ("zero", "resample"):
            # forward_pre_hook on o_proj per layer in cells
            for L in layers_in_cells:
                heads_at_L = [h for (Lc, h, _) in cells if Lc == L]
                # closure on heads_at_L, L
                if intervention == "zero":
                    def make_hook(L_=L, heads_=tuple(heads_at_L)):
                        def hook(module, args_):
                            x = args_[0].clone()
                            for h in heads_:
                                x[:, -1, h * head_dim:(h + 1) * head_dim] = 0
                            return (x,) + args_[1:]
                        return hook
                else:  # resample
                    def make_hook(L_=L, heads_=tuple(heads_at_L)):
                        def hook(module, args_):
                            x = args_[0].clone()
                            j = perm[state["i"]]
                            for h in heads_:
                                src = pool[L_][j, h * head_dim:(h + 1) * head_dim]
                                src_t = torch.tensor(src, dtype=x.dtype,
                                                       device=x.device)
                                x[:, -1, h * head_dim:(h + 1) * head_dim] = src_t
                            return (x,) + args_[1:]
                        return hook
                handles.append(layers[L].self_attn.o_proj
                                .register_forward_pre_hook(make_hook()))
        elif intervention == "q_zero":
            for L in layers_in_cells:
                heads_at_L = [h for (Lc, h, _) in cells if Lc == L]
                def make_q_hook(heads_=tuple(heads_at_L)):
                    def hook(module, inputs, output):
                        out = output.clone() if not isinstance(output, tuple) else output[0].clone()
                        for h in heads_:
                            out[:, -1, h * head_dim:(h + 1) * head_dim] = 0
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return hook
                handles.append(layers[L].self_attn.q_proj
                                .register_forward_hook(make_q_hook()))
        else:
            raise ValueError(intervention)

        ld = np.zeros(n, dtype=np.float32)
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    state["i"] = i
                    inp = tok(row["prompt"], return_tensors="pt").to(model.device)
                    out = model(**inp, use_cache=False)
                    logits = out.logits[0, -1].float()
                    ld[i] = float(logits[high_id] - logits[low_id])
        finally:
            for h in handles:
                h.remove()
        return ld

    results = {
        "model": model_id, "short": args.short,
        "feature": feature, "k": k, "n_prompts": int(n),
        "n_cells": len(cells),
        "selection_metric": ranking_label,
        "cells": [{"layer": int(L), "head": int(h),
                    "annotation": c}
                   for L, h, c in cells],
        "baseline_r_LD_z": base_r,
        "baseline_LD_mean": float(ld_baseline.mean()),
        "modes": {},
    }

    for mode in ["zero", "resample", "q_zero"]:
        print(f"\n[mode={mode}]")
        t1 = time.time()
        ld = fwd_LD(mode)
        r = float(np.corrcoef(ld, z_arr)[0, 1])
        results["modes"][mode] = {
            "r_LD_z": r, "delta_r": r - base_r,
            "ld_mean": float(ld.mean()),
            "delta_ld_mean": float(ld.mean() - ld_baseline.mean()),
        }
        print(f"  r(LD,z) = {r:+.3f}  Δr={r-base_r:+.3f}  "
              f"<LD>={ld.mean():+.2f}  Δ<LD>={ld.mean()-ld_baseline.mean():+.2f}  "
              f"({time.time()-t1:.0f}s)")

    if args.score_source == "dla":
        base_suffix = "bydla"
    elif args.score_source == "resid_cos":
        base_suffix = "byresidcos"
    elif args.by_cos:
        base_suffix = "bycos"
    else:
        base_suffix = "bynorm"
    if args.sign != "both":
        base_suffix += f"_{args.sign}"
    suffix = args.out_suffix or base_suffix
    out_json = REPO / "results" / f"p2o_attention_modes_{args.short}_{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    modes = ["baseline", "zero", "resample", "q_zero"]
    rs = [base_r] + [results["modes"][m]["r_LD_z"] for m in modes[1:]]
    lds = [float(ld_baseline.mean())] + [results["modes"][m]["ld_mean"] for m in modes[1:]]
    colors = ["black", "C3", "C0", "C2"]

    ax = axes[0]
    bars = ax.bar(modes, rs, color=colors, edgecolor="black")
    ax.axhline(base_r, color="black", lw=0.8, ls="--", alpha=0.5)
    for bar, r_ in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width() / 2, r_ + 0.005,
                 f"{r_:+.3f}", ha="center", fontsize=10)
    ax.set_ylabel("corr(LD, z)")
    ax.set_ylim(min(rs) - 0.05, max(rs) + 0.05)
    ax.set_title(f"{args.short} {feature} k={k} — corr(LD,z) under joint "
                  f"{len(cells)}-cell intervention")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(modes, lds, color=colors, edgecolor="black")
    ax.axhline(ld_baseline.mean(), color="black", lw=0.8, ls="--", alpha=0.5)
    for bar, ld_ in zip(bars, lds):
        ax.text(bar.get_x() + bar.get_width() / 2, ld_ + 0.05,
                 f"{ld_:+.2f}", ha="center", fontsize=10)
    ax.set_ylabel("⟨LD⟩")
    ax.set_title(f"⟨LD⟩ under joint {len(cells)}-cell intervention\n"
                  f"cells: {', '.join(f'L{L}H{h}' for L,h,_ in cells[:6])}"
                  f"{'...' if len(cells) > 6 else ''}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2o_attention_modes_{args.short}_{suffix}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
