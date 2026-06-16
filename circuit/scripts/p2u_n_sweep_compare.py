"""Phase 2U follow-up — N-sweep KL comparison: relativity vs neutral_text.

Cumulatively resamples the top-N cells (ranked by σ_DLA·|ρ|, positive ρ
only — i.e. the same ranking as Phase 2V's positive_dla sweep) and tracks
KL divergence on TWO prompt sets in parallel:
  1. relativity-it height k=15 (the same prompts as Phase 2U's positive control)
  2. neutral_text (raw natural-language imperatives, no task structure)

The point: as N grows, KL on relativity should grow fast; KL on neutral_text
should stay flat / random. The two diverging curves are the cleanest visual
of subspace-selective resample.

Cells are ranked from `results/p2s_dla_full_gemma2-2b.npz` (the canonical
2B base ranking from Phase 2T). Forwarded on Gemma 2 2B-it (matches Phase 2U
specificity battery).

Compute: ~5-10 min on RTX 5090 depending on N grid.

Outputs:
  results/p2u_n_sweep_compare_<short>.json
  figures/p2u_n_sweep_compare_<short>.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

DEFAULT_NS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32]


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False
                break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def select_top_cells(rank_npz_path: Path, max_n: int) -> list[tuple[int, int, float]]:
    """Top-N cells by σ_DLA · |ρ|, positive ρ only."""
    d = np.load(rank_npz_path)
    sigma = d["std_dla"]   # (n_layers, n_heads)
    rho = d["rho_grid"]
    mask = rho > 0
    score = np.where(mask, sigma * np.abs(rho), -np.inf)
    flat = np.argsort(score, axis=None)[::-1][:max_n]
    cells = []
    for k in flat:
        L, h = np.unravel_index(k, score.shape)
        cells.append((int(L), int(h), float(score[L, h])))
    return cells


def run_sweep(prompts_path: Path, label_key: str | None,
                cell_order: list[tuple[int, int]], ns: list[int],
                model, tok, layers, n_layers: int, n_heads: int, head_dim: int,
                vocab: int, high_id: int | None, low_id: int | None,
                seed: int = 42) -> dict:
    """Capture baseline + run N-sweep on a single prompt set, return per-N metrics."""
    rows = [json.loads(l) for l in prompts_path.open()]
    n = len(rows)
    has_label = label_key is not None
    if has_label:
        label_arr = np.array([float(r[label_key]) for r in rows], dtype=np.float32)
    else:
        label_arr = None
    print(f"  prompts: {prompts_path.name}  n={n}  label={label_key}")

    # Pass 1: baseline
    pool = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    captured: dict[int, np.ndarray] = {}

    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
                for L in range(n_layers)]
    baseline_logp = np.zeros((n, vocab), dtype=np.float16)
    ld_baseline = np.zeros(n, dtype=np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            if has_label:
                ld_baseline[i] = float(logits[high_id] - logits[low_id])
            baseline_logp[i] = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float16)
            for L in range(n_layers):
                pool[L, i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"    pass1 {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                       flush=True)
    for h in handles:
        h.remove()

    if has_label:
        base_r = float(np.corrcoef(ld_baseline, label_arr)[0, 1])
        print(f"    baseline r(LD,label) = {base_r:+.3f}")
    else:
        base_r = None
        print(f"    baseline (KL-only)")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    def resample_run(cells: list[tuple[int, int]]) -> dict:
        by_layer: dict = {}
        for L, h in cells:
            by_layer.setdefault(L, []).append(h)
        handles = []
        state = {"i": 0}
        for L, hs in by_layer.items():
            heads_t = tuple(hs)

            def make_hook(L_=L, heads_=heads_t):
                def hook(module, args_):
                    x = args_[0].clone()
                    j = perm[state["i"]]
                    for h in heads_:
                        src = pool[L_, j, h * head_dim:(h + 1) * head_dim]
                        src_t = torch.tensor(src, dtype=x.dtype, device=x.device)
                        x[:, -1, h * head_dim:(h + 1) * head_dim] = src_t
                    return (x,) + args_[1:]

                return hook

            handles.append(layers[L].self_attn.o_proj
                            .register_forward_pre_hook(make_hook()))
        ld = np.zeros(n, dtype=np.float32)
        kls = np.zeros(n, dtype=np.float32)
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    state["i"] = i
                    inp = tok(row["prompt"], return_tensors="pt").to(model.device)
                    out = model(**inp, use_cache=False)
                    logits = out.logits[0, -1].float()
                    log_p_r = F.log_softmax(logits, dim=-1)
                    if has_label:
                        ld[i] = float(logits[high_id] - logits[low_id])
                    log_p_b = torch.tensor(baseline_logp[i].astype(np.float32),
                                             device=model.device)
                    p_b = log_p_b.exp()
                    kls[i] = float((p_b * (log_p_b - log_p_r)).sum().item())
        finally:
            for h in handles:
                h.remove()
        if has_label:
            r = float(np.corrcoef(ld, label_arr)[0, 1])
        else:
            r = None
        return {
            "r_LD_label": r,
            "delta_r": (r - base_r) if has_label else None,
            "ld_mean": float(ld.mean()) if has_label else None,
            "kl_mean": float(kls.mean()),
            "kl_total": float(kls.sum()),
            "kl_p50": float(np.percentile(kls, 50)),
            "kl_p90": float(np.percentile(kls, 90)),
        }

    runs = []
    for N in ns:
        cells_n = cell_order[:N]
        t1 = time.time()
        m = resample_run(cells_n)
        if has_label:
            print(f"    N={N:>2d}: r={m['r_LD_label']:+.3f}  Δr={m['delta_r']:+.3f}  "
                   f"KL={m['kl_mean']:.4f}  ({time.time()-t1:.0f}s)")
        else:
            print(f"    N={N:>2d}: KL={m['kl_mean']:.4f}  KL_total={m['kl_total']:.1f}  "
                   f"({time.time()-t1:.0f}s)")
        runs.append({"N": N, **m})

    return {
        "n_prompts": int(n),
        "label_key": label_key,
        "baseline_r": base_r,
        "baseline_ld_mean": float(ld_baseline.mean()) if has_label else None,
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--model-id", default="google/gemma-2-2b-it")
    ap.add_argument("--rank-source",
                    default="results/p2s_dla_full_gemma2-2b.npz",
                    help="path to .npz with std_dla and rho_grid")
    ap.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relativity-prompts",
                    default="data/p2_shot_sweep/height_k15.jsonl")
    ap.add_argument("--neutral-prompts",
                    default="data/specificity/neutral_text.jsonl")
    args = ap.parse_args()

    rank_path = REPO / args.rank_source
    relativity_path = REPO / args.relativity_prompts
    neutral_path = REPO / args.neutral_prompts
    max_n = max(args.ns)

    cells = select_top_cells(rank_path, max_n)
    print(f"top-{max_n} cells from {rank_path.name}:")
    for L, h, s in cells:
        print(f"  L{L:>2d}H{h}  score={s:+.4f}")
    cell_order = [(L, h) for L, h, _ in cells]

    print(f"\nloading {args.model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    vocab = model.config.vocab_size

    # Tokens for relativity readout (tall/short, leading-space variant)
    rel_first_row = json.loads(relativity_path.open().readline())
    high_w = rel_first_row.get("high_word", "tall")
    low_w = rel_first_row.get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]
    print(f"  relativity tokens: high {' '+high_w!r} -> {high_id}  low {' '+low_w!r} -> {low_id}")

    print(f"\n=== relativity sweep ===")
    rel_result = run_sweep(
        relativity_path, "z_eff", cell_order, args.ns,
        model, tok, layers, n_layers, n_heads, head_dim, vocab,
        high_id, low_id, seed=args.seed,
    )

    print(f"\n=== neutral_text sweep ===")
    neu_result = run_sweep(
        neutral_path, None, cell_order, args.ns,
        model, tok, layers, n_layers, n_heads, head_dim, vocab,
        None, None, seed=args.seed,
    )

    out = {
        "model": args.model_id,
        "short": args.short,
        "rank_source": str(rank_path.name),
        "score_metric": "σ_DLA · |ρ(DLA, z)|, positive ρ only (Phase 2T)",
        "ns": list(args.ns),
        "selected_cells": [{"L": L, "h": h, "score": s} for L, h, s in cells],
        "relativity": rel_result,
        "neutral_text": neu_result,
    }
    out_path = REPO / "results" / f"p2u_n_sweep_compare_{args.short}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Plot
    import matplotlib.pyplot as plt

    rel_kl = [r["kl_mean"] for r in rel_result["runs"]]
    neu_kl = [r["kl_mean"] for r in neu_result["runs"]]
    rel_dr = [r["delta_r"] for r in rel_result["runs"]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(args.ns, rel_kl, "o-", color="C3", lw=2, ms=7,
             label=f"relativity (height k=15, n={rel_result['n_prompts']})")
    ax.plot(args.ns, neu_kl, "s-", color="C7", lw=2, ms=7,
             label=f"neutral_text (n={neu_result['n_prompts']})")
    ax.set_ylabel("mean KL(baseline || resample) [nats / prompt]")
    ax.set_title(f"σ_DLA·|ρ| top-N cumulative resample on {args.short}: "
                  f"relativity vs neutral text")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate ratio at a few key Ns
    for N_target in [3, 7, 16, 32]:
        if N_target in args.ns:
            i = args.ns.index(N_target)
            ratio = rel_kl[i] / max(neu_kl[i], 1e-6)
            ax.annotate(f"{ratio:.0f}×",
                         xy=(N_target, rel_kl[i]),
                         xytext=(N_target, rel_kl[i] + 0.03),
                         ha="center", fontsize=9,
                         color="C3")

    ax = axes[1]
    ax.plot(args.ns, rel_dr, "o-", color="C3", lw=2, ms=7,
             label="relativity Δr(LD, z)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(rel_result["baseline_r"], color="C3", lw=0.5, ls="--", alpha=0.5,
                label=f"baseline r = {rel_result['baseline_r']:+.3f}")
    ax.set_xlabel("N (number of top-σ_DLA·|ρ| cells resampled, cumulative)")
    ax.set_ylabel("Δr(LD, z)")
    ax.set_title("Relativity correlation drops as N grows (neutral_text has no analog)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2u_n_sweep_compare_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
