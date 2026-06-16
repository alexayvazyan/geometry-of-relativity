"""Phase 2O random-resample control.

Tests whether the Phase 2O resample finding (Δr=-0.36 on top-8 |cos|-ranked
cells in 2B) is specific to those cells, vs whether ANY 8-head random
resample produces a similar effect.

Pipeline:
  1. Capture pre-o_proj at every decoder layer for all 990 prompts (1 baseline pass).
  2. For each random trial: pick 8 random (layer, head) cells uniformly from the
     full grid (n_layers × n_heads), resample those cells (replace with same
     cells' values from a permuted prompt), forward, measure r(LD, z).
  3. Compare distribution of random Δr to the Phase 2O selected Δr.

Output:
  results/p2o_random_control_<short>.json
  figures/p2o_random_control_<short>.png

Compute (2B, n=990, 5 random trials): pool capture ~14s × n_layers (one shot
covering all layers via per-layer hooks), then 5 × ~14s resample passes ≈ 5 min.

Usage:
  python p2o_random_control.py --short gemma2-2b --n-trials 5 --n-cells 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
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
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--n-cells", type=int, default=8,
                    help="cells per trial (matches Phase 2O)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=None)
    args = ap.parse_args()

    model_id = ("google/gemma-2-2b" if args.short == "gemma2-2b"
                 else "google/gemma-2-9b")

    # 1. Stimuli
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{args.feature}_k{args.k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    if args.n_prompts:
        rows = rows[:args.n_prompts]
    n = len(rows)
    print(f"loaded {n} prompts")

    # 2. Model
    print(f"\nloading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]

    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)

    # 3. Pool capture: one baseline pass, capture pre-o_proj at every layer
    print(f"\n[pass 1] capture pre-o_proj at all {n_layers} layers + baseline LD")
    pool = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    captured = {}

    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
                for L in range(n_layers)]
    ld_baseline = np.zeros(n, dtype=np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            ld_baseline[i] = float(logits[high_id] - logits[low_id])
            for L in range(n_layers):
                pool[L, i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()
    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}")

    # 4. Run random-cell resample trials
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def resample_LD(cells: list[tuple[int, int]]) -> np.ndarray:
        """cells: [(L, h), ...] — resample each at last-token using perm-mapped prompt's value."""
        # group by layer
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

    print(f"\n[pass 2+] {args.n_trials} random trials, {args.n_cells} cells each")
    trials = []
    for t in range(args.n_trials):
        # pick n_cells uniformly from (layer, head) grid
        all_cells = [(L, h) for L in range(n_layers) for h in range(n_heads)]
        idx = rng.choice(len(all_cells), size=args.n_cells, replace=False)
        cells = [all_cells[i] for i in idx]
        t1 = time.time()
        ld = resample_LD(cells)
        r = float(np.corrcoef(ld, z_arr)[0, 1])
        delta_r = r - base_r
        print(f"  trial {t+1}/{args.n_trials}: cells={cells}  r={r:+.3f}  "
              f"Δr={delta_r:+.3f}  ({time.time()-t1:.0f}s)")
        trials.append({
            "trial": t,
            "cells": [{"layer": int(L), "head": int(h)} for L, h in cells],
            "r_LD_z": r, "delta_r": delta_r,
            "ld_mean": float(ld.mean()),
        })

    # Compare to Phase 2O selected Δr (load if available)
    selected_path = REPO / "results" / f"p2o_attention_modes_{args.short}_bycos.json"
    selected_delta = None
    if selected_path.exists():
        sel = json.loads(selected_path.read_text())
        selected_delta = sel["modes"]["resample"]["delta_r"]
        print(f"\n[reference] Phase 2O |cos|-ranked top-{args.n_cells} resample: "
              f"Δr={selected_delta:+.3f}")

    rand_deltas = [t["delta_r"] for t in trials]
    rand_mean = float(np.mean(rand_deltas))
    rand_std = float(np.std(rand_deltas))
    rand_min = float(np.min(rand_deltas))
    rand_max = float(np.max(rand_deltas))
    print(f"[summary] random Δr: mean={rand_mean:+.3f} ± {rand_std:.3f}  "
          f"range [{rand_min:+.3f}, {rand_max:+.3f}]")
    if selected_delta is not None:
        print(f"          selected Δr = {selected_delta:+.3f}  "
              f"(z-score vs random: {(selected_delta - rand_mean) / max(rand_std, 1e-6):+.2f})")

    out = {
        "model": model_id, "short": args.short,
        "feature": args.feature, "k": args.k, "n_prompts": int(n),
        "baseline_r_LD_z": base_r,
        "n_trials": args.n_trials, "n_cells": args.n_cells,
        "selected_delta_r": selected_delta,
        "random_trials": trials,
        "random_delta_mean": rand_mean,
        "random_delta_std": rand_std,
    }
    out_json = REPO / "results" / f"p2o_random_control_{args.short}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(base_r, color="black", lw=0.8, ls="--",
                alpha=0.5, label=f"baseline r={base_r:+.3f}")
    rs = [t["r_LD_z"] for t in trials]
    ax.scatter(np.arange(len(rs)), rs, s=80, color="C7", edgecolor="black",
                label=f"random ({args.n_cells} cells × {args.n_trials} trials)")
    if selected_delta is not None:
        sel_r = base_r + selected_delta
        ax.axhline(sel_r, color="C3", lw=2,
                    label=f"|cos|-ranked top-{args.n_cells} resample r={sel_r:+.3f}")
    ax.set_xticks(np.arange(len(rs)))
    ax.set_xticklabels([f"trial {t+1}" for t in range(len(rs))], fontsize=9)
    ax.set_ylabel("r(LD, z) under joint resample")
    ax.set_title(f"{args.short} — random {args.n_cells}-cell resample vs |cos|-ranked\n"
                  f"random Δr mean={rand_mean:+.3f}±{rand_std:.3f}  "
                  f"selected Δr={selected_delta:+.3f}" if selected_delta else "")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2o_random_control_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
