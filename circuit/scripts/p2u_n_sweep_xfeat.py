"""Phase 2U follow-up — combined N-sweep across relativity concepts and neutral text.

Same height-derived σ_DLA·|ρ| ranking from `p2s_dla_full_gemma2-2b.npz`,
but applied to four prompt sets in parallel:
  - height  k=15  (tall / short)
  - weight  k=15  (heavy / light)
  - speed   k=15  (fast / slow)
  - neutral_text  (raw natural-language imperatives, KL-only)

Forwarded on Gemma 2 2B-it.

Output: one combined JSON + a single figure with
  Top:    Δr(LD, z) per concept (3 lines) over top-N
          + cos·σ bars showing the signed σ_DLA·ρ of the cell entering at rank N
          (red = +z aligned, blue = -z aligned)
  Bottom: mean KL(baseline || resample) on neutral_text over the same N axis,
          measuring general output-distribution damage in lockstep.

Compute: ~15 min on RTX 5090 (4 prompt sets, 15-step N-sweep each).
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

FEATURES = [
    {"name": "height", "prompts": "data/p2_shot_sweep/height_k15.jsonl",
     "label_key": "z_eff", "high": "tall", "low": "short", "color": "C0"},
    {"name": "weight", "prompts": "data/p2_shot_sweep/weight_k15.jsonl",
     "label_key": "z_eff", "high": "heavy", "low": "light", "color": "C2"},
    {"name": "speed", "prompts": "data/p2_shot_sweep/speed_k15.jsonl",
     "label_key": "z_eff", "high": "fast", "low": "slow", "color": "C5"},
]
NEUTRAL = {"name": "neutral_text",
           "prompts": "data/specificity/neutral_text.jsonl",
           "label_key": None, "high": None, "low": None, "color": "C7"}


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


def select_top_cells(rank_npz_path: Path, max_n: int) -> list[dict]:
    d = np.load(rank_npz_path)
    sigma = d["std_dla"]
    rho = d["rho_grid"]
    mask = rho > 0
    score = np.where(mask, sigma * np.abs(rho), -np.inf)
    flat = np.argsort(score, axis=None)[::-1][:max_n]
    cells = []
    for k in flat:
        L, h = np.unravel_index(k, score.shape)
        cells.append({
            "L": int(L), "h": int(h),
            "sigma_dla": float(sigma[L, h]),
            "rho": float(rho[L, h]),
            "score": float(sigma[L, h] * abs(rho[L, h])),
            "signed_score": float(sigma[L, h] * rho[L, h]),  # used for bar color
        })
    return cells


def run_sweep(prompts_path: Path, label_key: str | None,
                cell_order: list[tuple[int, int]], ns: list[int],
                model, tok, layers, n_layers: int, n_heads: int, head_dim: int,
                vocab: int, high_id: int | None, low_id: int | None,
                seed: int = 42) -> dict:
    rows = [json.loads(l) for l in prompts_path.open()]
    n = len(rows)
    has_label = label_key is not None
    if has_label:
        label_arr = np.array([float(r[label_key]) for r in rows], dtype=np.float32)
        # Optional x covariate for partial-correlation tracking. Present in
        # the relativity prompt sets (height/weight/speed); absent in neutral.
        if "x" in rows[0]:
            x_arr = np.array([float(r["x"]) for r in rows], dtype=np.float32)
        else:
            x_arr = None
    else:
        label_arr = None
        x_arr = None
    print(f"  prompts: {prompts_path.name}  n={n}  label={label_key}  "
           f"has_x={x_arr is not None}")

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
        if x_arr is not None:
            base_r_x = float(np.corrcoef(ld_baseline, x_arr)[0, 1])
            r_zx = float(np.corrcoef(label_arr, x_arr)[0, 1])
            print(f"    baseline r(LD,label) = {base_r:+.3f}  "
                   f"r(LD,x) = {base_r_x:+.3f}  r(z,x) = {r_zx:+.3f}")
        else:
            base_r_x = None
            r_zx = None
            print(f"    baseline r(LD,label) = {base_r:+.3f}")
    else:
        base_r = None
        base_r_x = None
        r_zx = None
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
            if x_arr is not None:
                r_x = float(np.corrcoef(ld, x_arr)[0, 1])
            else:
                r_x = None
        else:
            r = None
            r_x = None
        return {
            "r_LD_label": r,
            "r_LD_x": r_x,
            "delta_r": (r - base_r) if has_label else None,
            "delta_r_x": (r_x - base_r_x) if (has_label and r_x is not None) else None,
            "ld_mean": float(ld.mean()) if has_label else None,
            "kl_mean": float(kls.mean()),
            "kl_total": float(kls.sum()),
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
        "baseline_r_x": base_r_x,
        "r_zx": r_zx,
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--model-id", default="google/gemma-2-2b-it")
    ap.add_argument("--rank-source",
                    default="results/p2s_dla_full_gemma2-2b.npz")
    ap.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank_path = REPO / args.rank_source
    max_n = max(args.ns)

    cells = select_top_cells(rank_path, max_n)
    print(f"top-{max_n} cells from {rank_path.name} (height σ_DLA·|ρ|, ρ>0):")
    for c in cells:
        print(f"  L{c['L']:>2d}H{c['h']}  σ_DLA={c['sigma_dla']:.3f}  "
               f"ρ={c['rho']:+.3f}  signed_score={c['signed_score']:+.4f}")
    cell_order = [(c["L"], c["h"]) for c in cells]

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

    sweeps = {}
    for feat in FEATURES:
        print(f"\n=== {feat['name']} ===")
        high_id = tok.encode(" " + feat["high"], add_special_tokens=False)[-1]
        low_id = tok.encode(" " + feat["low"], add_special_tokens=False)[-1]
        sweeps[feat["name"]] = run_sweep(
            REPO / feat["prompts"], feat["label_key"], cell_order, args.ns,
            model, tok, layers, n_layers, n_heads, head_dim, vocab,
            high_id, low_id, seed=args.seed,
        )

    print(f"\n=== {NEUTRAL['name']} ===")
    sweeps[NEUTRAL["name"]] = run_sweep(
        REPO / NEUTRAL["prompts"], None, cell_order, args.ns,
        model, tok, layers, n_layers, n_heads, head_dim, vocab,
        None, None, seed=args.seed,
    )

    out = {
        "model": args.model_id,
        "short": args.short,
        "rank_source": str(rank_path.name),
        "score_metric": "σ_DLA · |ρ(DLA, z)|, positive ρ only (Phase 2T, height ranking)",
        "ns": list(args.ns),
        "selected_cells": cells,
        "sweeps": sweeps,
    }
    out_path = REPO / "results" / f"p2u_n_sweep_xfeat_{args.short}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
    ax_top = axes[0]
    ax_bot = axes[1]

    # --- Top panel: Δr per relativity concept (lines, left axis) + bars (right axis) ---
    ax_bars = ax_top.twinx()

    # Bars first so lines render above
    bar_x = np.arange(1, max_n + 1)
    signed = np.array([c["signed_score"] for c in cells])
    bar_colors = ["C3" if s > 0 else "C0" for s in signed]
    ax_bars.bar(bar_x, np.abs(signed), color=bar_colors, alpha=0.25,
                 width=0.8, edgecolor="none", zorder=1)
    ax_bars.set_ylabel("σ_DLA · |ρ| of cell entering at rank N", fontsize=14)
    ax_bars.tick_params(axis="y", labelsize=13)
    ax_bars.set_ylim(0, max(np.abs(signed)) * 1.25)

    n_to_label = min(3, max_n)
    for rank0 in range(n_to_label):
        c = cells[rank0]
        lab = f"L{c['L']}H{c['h']}"
        score = abs(signed[rank0])
        ax_bars.annotate(lab, xy=(rank0 + 1, score),
                          xytext=(rank0 + 1, score + 0.02),
                          ha="center", fontsize=12, color="black",
                          fontweight="bold")

    # Lines: Δr per relativity concept
    for feat in FEATURES:
        runs = sweeps[feat["name"]]["runs"]
        ns = [r["N"] for r in runs]
        drs = [r["delta_r"] for r in runs]
        baseline_r = sweeps[feat["name"]]["baseline_r"]
        ax_top.plot(ns, drs, "o-", color=feat["color"], lw=2, ms=6,
                     label=f"{feat['name']} (baseline corr={baseline_r:+.2f})", zorder=3)

    ax_top.axhline(0, color="black", lw=0.5)
    ax_top.set_ylabel("Δcorr(LD, z)", fontsize=15)
    ax_top.set_title(f"σ_DLA·|ρ| top-N cumulative interchange ablation on {args.short}: "
                      f"all three relativity concepts collapse together",
                      fontsize=14)
    ax_top.legend(loc="lower left", fontsize=13)
    ax_top.tick_params(axis="both", labelsize=13)
    ax_top.grid(alpha=0.3, zorder=0)

    # --- Bottom panel: KL on neutral text ---
    runs = sweeps[NEUTRAL["name"]]["runs"]
    ns = [r["N"] for r in runs]
    kls = [r["kl_mean"] for r in runs]
    ax_bot.plot(ns, kls, "s-", color="C7", lw=2, ms=6,
                 label=f"neutral_text mean KL (n={sweeps[NEUTRAL['name']]['n_prompts']})")
    # Also overlay relativity-side mean-KL for reference
    for feat in FEATURES:
        rkl = [r["kl_mean"] for r in sweeps[feat["name"]]["runs"]]
        ax_bot.plot(ns, rkl, "--", color=feat["color"], lw=1.2, alpha=0.6,
                     label=f"{feat['name']} mean KL (relativity prompts)")
    ax_bot.set_xlabel("N (top-σ_DLA·|ρ| cells resampled, cumulative)", fontsize=15)
    ax_bot.set_ylabel("mean KL(baseline || resample) [nats / prompt]", fontsize=15)
    ax_bot.set_title("Per-prompt KL divergence: neutral text climbs slowly while "
                      "relativity prompts saturate sharply", fontsize=14)
    ax_bot.legend(loc="upper left", fontsize=13)
    ax_bot.tick_params(axis="both", labelsize=13)
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2u_n_sweep_xfeat_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
