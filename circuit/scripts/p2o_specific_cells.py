"""Phase 2O — resample ablate explicit (layer, head) cell sets.

Tests specific small cell sets (typically the cliff-causing cells from
the positive-cos sweep) to see how much each contributes alone vs
in combination.

Cell sets specified as repeated --cells L,H[,L,H,...] arguments.

Output:
  results/p2o_specific_cells_<short>.json
  figures/p2o_specific_cells_<short>.png

Usage example:
  python p2o_specific_cells.py --short gemma2-2b \\
    --label L16H4 --cells 16,4 \\
    --label L17H7 --cells 17,7 \\
    --label L14H2 --cells 14,2 \\
    --label "L16H4+L17H7" --cells 16,4,17,7 \\
    --label "L16H4+L17H7+L14H2" --cells 16,4,17,7,14,2
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


def parse_cells(s: str) -> list[tuple[int, int]]:
    """'16,4,17,7' -> [(16, 4), (17, 7)]"""
    parts = [int(x) for x in s.split(",")]
    if len(parts) % 2:
        raise ValueError(f"odd number of values in --cells {s}")
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True,
                    choices=["gemma2-2b", "gemma2-2b-it", "gemma2-9b"])
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--label", action="append", default=[],
                    help="label for each cell set (paired with --cells)")
    ap.add_argument("--cells", action="append", default=[],
                    help="comma-separated 'L,H,L,H,...' for each set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.label) != len(args.cells):
        raise SystemExit("--label and --cells must be paired")
    sets = [(label, parse_cells(cells_s))
             for label, cells_s in zip(args.label, args.cells)]
    print("cell sets:")
    for label, cells in sets:
        print(f"  {label}: {cells}")

    model_id = {
        "gemma2-2b": "google/gemma-2-2b",
        "gemma2-2b-it": "google/gemma-2-2b-it",
        "gemma2-9b": "google/gemma-2-9b",
    }[args.short]

    stim_path = REPO / "data" / "p2_shot_sweep" / f"{args.feature}_k{args.k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)

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
    vocab = model.config.vocab_size

    # Capture all-layer pre-o_proj for resample pool
    print(f"\n[pass 1] capture pre-o_proj at all {n_layers} layers + baseline LD/log-probs")
    pool = np.zeros((n_layers, n, n_heads * head_dim), dtype=np.float32)
    captured = {}
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
            ld_baseline[i] = float(logits[high_id] - logits[low_id])
            baseline_logp[i] = F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float16)
            for L in range(n_layers):
                pool[L, i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()
    base_r = float(np.corrcoef(ld_baseline, z_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}  <LD>={ld_baseline.mean():+.2f}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def resample(cells: list[tuple[int, int]]) -> tuple[float, float, float, float]:
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
                    ld[i] = float(logits[high_id] - logits[low_id])
                    log_p_b = torch.tensor(baseline_logp[i].astype(np.float32),
                                             device=model.device)
                    p_b = log_p_b.exp()
                    kls[i] = float((p_b * (log_p_b - log_p_r)).sum().item())
        finally:
            for h in handles:
                h.remove()
        r = float(np.corrcoef(ld, z_arr)[0, 1])
        return r, r - base_r, float(ld.mean()), float(kls.mean())

    print("\n[runs]")
    out = {
        "model": model_id, "short": args.short,
        "feature": args.feature, "k": args.k, "n_prompts": int(n),
        "baseline_r_LD_z": base_r,
        "baseline_LD_mean": float(ld_baseline.mean()),
        "runs": [],
    }
    for label, cells in sets:
        t1 = time.time()
        r, dr, ld_m, kl_m = resample(cells)
        print(f"  {label:<30s} r={r:+.3f}  Δr={dr:+.3f}  <LD>={ld_m:+.2f}  "
              f"KL={kl_m:.3f} nats  ({time.time()-t1:.0f}s)")
        out["runs"].append({
            "label": label, "cells": cells,
            "r_LD_z": r, "delta_r": dr,
            "ld_mean": ld_m, "kl_mean": kl_m,
        })

    tag = f"{args.short}_{args.feature}_k{args.k}"
    out_path = REPO / "results" / f"p2o_specific_cells_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["baseline"] + [r["label"] for r in out["runs"]]
    rs = [base_r] + [r["r_LD_z"] for r in out["runs"]]
    kls = [0] + [r["kl_mean"] for r in out["runs"]]

    ax = axes[0]
    bars = ax.bar(np.arange(len(labels)), rs,
                    color=["black"] + ["C0"] * len(out["runs"]), edgecolor="black")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(base_r, color="black", lw=0.5, ls="--", alpha=0.5)
    for bar, r in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                 f"{r:+.3f}", ha="center", fontsize=9)
    ax.set_ylabel("r(LD, z) under joint resample")
    ax.set_title(f"{args.short} {args.feature} k={args.k} — specific-cell resample")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(np.arange(len(labels)), kls,
                    color=["black"] + ["C2"] * len(out["runs"]), edgecolor="black")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for bar, kl in zip(bars, kls):
        ax.text(bar.get_x() + bar.get_width() / 2, kl + 0.001,
                 f"{kl:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("mean KL(baseline||resample) [nats]")
    ax.set_title("Output-distribution disruption")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2o_specific_cells_{tag}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
