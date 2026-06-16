"""Phase 2U — specificity battery for the trio + 7-cell circuit.

Tests whether resample-ablating the trio (L16H4+L17H7+L14H2) and the 7-cell
extended circuit destroys non-relativity behaviors as well as it destroys
relativity. Compares against L16H4 singleton (hub-only) and 5 random 3-cell
baselines sampled from the trio's layer band L12-L22.

Tasks supported:
  relativity-it : data/p2_shot_sweep/height_k15.jsonl  (positive control)
  arithmetic    : data/specificity/arithmetic.jsonl
  truth         : data/specificity/truth.jsonl  (cities)
  refusal       : data/specificity/refusal.jsonl  (harmful + harmless)

Per task: ~10 sweeps × 1k prompts × ~30s on 2B-it. ~5 min/task on RTX 5090.

Outputs:
  results/p2u_specificity_<task>.json
  figures/p2u_specificity_<task>.png  (per-cell-set Δr + KL bars)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

# Trio + extended 7-cell circuit (Gemma 2 2B; verified to transfer to 2B-it)
TRIO = [(16, 4), (17, 7), (14, 2)]
EXTENDED = TRIO + [(17, 6), (22, 1), (12, 4), (13, 2)]
HUB = [(16, 4)]

# Random-control sampling band (matches trio's layer span)
RANDOM_LAYER_BAND = list(range(12, 23))  # L12..L22 inclusive
N_HEADS_2B = 8
N_RANDOM_CONTROLS = 5
RANDOM_CELLS_PER_CONTROL = 3
RANDOM_SEEDS = [101, 202, 303, 404, 505]

TASK_CONFIG = {
    "relativity-it": {
        "prompts": REPO / "data" / "p2_shot_sweep" / "height_k15.jsonl",
        "label_key": "z_eff",
        "fallback_label_key": "z",
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": True,
    },
    "relativity-weight": {
        "prompts": REPO / "data" / "p2_shot_sweep" / "weight_k15.jsonl",
        "label_key": "z_eff",
        "fallback_label_key": "z",
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": True,
    },
    "relativity-speed": {
        "prompts": REPO / "data" / "p2_shot_sweep" / "speed_k15.jsonl",
        "label_key": "z_eff",
        "fallback_label_key": "z",
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": True,
    },
    "arithmetic": {
        "prompts": REPO / "data" / "specificity" / "arithmetic.jsonl",
        "label_key": "label",
        "fallback_label_key": None,
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": True,
    },
    "truth": {
        "prompts": REPO / "data" / "specificity" / "truth.jsonl",
        "label_key": "label",
        "fallback_label_key": None,
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": False,
    },
    "refusal": {
        "prompts": REPO / "data" / "specificity" / "refusal.jsonl",
        "label_key": "label",
        "fallback_label_key": None,
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": False,
    },
    "truth-base": {
        # 2-shot ICL truth task; chat-template-free so it works on base models.
        "prompts": REPO / "data" / "specificity" / "truth_base.jsonl",
        "label_key": "label",
        "fallback_label_key": None,
        "high_word_key": "high_word",
        "low_word_key": "low_word",
        "space_prefix_default": True,
    },
    "neutral_text": {
        # KL-only: varied natural-language prompts with no task structure
        # and no chat-template wrapping. Tracks KL across cell-sets to
        # measure general-output-distribution damage independent of any
        # task. r/Δr are not defined (label_key is None).
        "prompts": REPO / "data" / "specificity" / "neutral_text.jsonl",
        "label_key": None,
        "fallback_label_key": None,
        "high_word_key": None,
        "low_word_key": None,
        "space_prefix_default": False,
    },
}


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


def sample_random_controls(seeds: list[int], n_cells: int, exclude: set,
                              layer_band: list[int], n_heads: int) -> list[list[tuple[int, int]]]:
    """Sample N seeded random cell sets from the layer band, excluding `exclude`."""
    pool = [(L, h) for L in layer_band for h in range(n_heads)
             if (L, h) not in exclude]
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        out.append(sorted(rng.sample(pool, n_cells)))
    return out


def cells_label(cells: list[tuple[int, int]]) -> str:
    return "+".join(f"L{L}H{h}" for L, h in cells)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    ap.add_argument("--model-id", default="google/gemma-2-2b-it")
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, truncate prompts (for smoke tests)")
    ap.add_argument("--seed", type=int, default=42,
                    help="resample permutation seed")
    ap.add_argument("--trio", default=None,
                    help="override trio cells; format 'L,H;L,H;L,H' "
                         "(default: 2B trio L16H4;L17H7;L14H2)")
    ap.add_argument("--ext-extra", default=None,
                    help="extra cells beyond trio for the extended set; "
                         "same format as --trio (default: 2B ext-extra "
                         "L17H6;L22H1;L12H4;L13H2)")
    ap.add_argument("--hub-cell", default=None,
                    help="single hub cell 'L,H' (default: first of trio)")
    ap.add_argument("--random-band", default=None,
                    help="layer range for random control sampling 'lo,hi' "
                         "inclusive (default: 12,22 for 2B)")
    ap.add_argument("--no-extended", action="store_true",
                    help="skip the extended cell-set (useful when only the "
                         "trio is well-characterised on this model)")
    args = ap.parse_args()

    def _parse_cells(s: str) -> list[tuple[int, int]]:
        out = []
        for tok in s.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            L, h = tok.split(",")
            out.append((int(L), int(h)))
        return out

    trio_cells = _parse_cells(args.trio) if args.trio else TRIO
    ext_extra_cells = (
        _parse_cells(args.ext_extra) if args.ext_extra
        else [(17, 6), (22, 1), (12, 4), (13, 2)]
    )
    extended_cells = trio_cells + ext_extra_cells
    hub_cell = (_parse_cells(args.hub_cell)[0] if args.hub_cell
                  else trio_cells[0])
    if args.random_band:
        lo, hi = (int(x) for x in args.random_band.split(","))
        random_layer_band = list(range(lo, hi + 1))
    else:
        random_layer_band = RANDOM_LAYER_BAND

    cfg = TASK_CONFIG[args.task]
    rows = [json.loads(l) for l in cfg["prompts"].open()]
    if args.limit > 0:
        rows = rows[:args.limit]
    n = len(rows)

    label_key = cfg["label_key"]
    has_label = label_key is not None
    if has_label:
        fk = cfg["fallback_label_key"]
        label_arr = np.array(
            [float(r.get(label_key, r.get(fk, 0)) if fk else r[label_key])
              for r in rows],
            dtype=np.float32,
        )
        high_w = rows[0][cfg["high_word_key"]]
        low_w = rows[0][cfg["low_word_key"]]
        space_prefix = rows[0].get("space_prefix", cfg["space_prefix_default"])
        print(f"[task={args.task}] n={n}  label={label_key}  "
               f"high={high_w!r}  low={low_w!r}  space_prefix={space_prefix}")
    else:
        label_arr = None
        high_w = low_w = None
        space_prefix = False
        print(f"[task={args.task}] n={n}  KL-only (no label / no high-low tokens)")

    # Build cell-set list (using CLI overrides if provided)
    trio_label = f"trio ({cells_label(trio_cells)})"
    cell_sets: list[tuple[str, list[tuple[int, int]]]] = [
        (trio_label, trio_cells),
    ]
    if not args.no_extended:
        ext_label = f"{len(extended_cells)}-cell extended"
        cell_sets.append((ext_label, extended_cells))
    cell_sets.append((f"L{hub_cell[0]}H{hub_cell[1]} hub alone", [hub_cell]))
    excluded = set(extended_cells if not args.no_extended else trio_cells)
    # Random controls deferred until after model load — need n_heads.

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

    # Now that we know n_heads, sample the random control cell sets and
    # finalise the cell-set list.
    randoms = sample_random_controls(RANDOM_SEEDS, RANDOM_CELLS_PER_CONTROL,
                                       excluded, random_layer_band, n_heads)
    for i, cells in enumerate(randoms):
        cell_sets.append((f"random-3 #{i+1} ({cells_label(cells)})", cells))
    print(f"\ncell sets ({len(cell_sets)}):")
    for label, cells in cell_sets:
        print(f"  {label}")

    if has_label:
        high_token = (" " + high_w) if space_prefix else high_w
        low_token = (" " + low_w) if space_prefix else low_w
        high_id = tok.encode(high_token, add_special_tokens=False)[-1]
        low_id = tok.encode(low_token, add_special_tokens=False)[-1]
        print(f"  high token {high_token!r} -> id={high_id}  "
               f"low token {low_token!r} -> id={low_id}")
    else:
        high_id = low_id = -1  # unused

    # PASS 1: capture all-layer pre-o_proj at last token + baseline LD/log-probs
    print(f"\n[pass 1] capture pre-o_proj at all {n_layers} layers + baseline")
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
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()

    if has_label:
        base_r = float(np.corrcoef(ld_baseline, label_arr)[0, 1])
        print(f"  baseline r(LD,label) = {base_r:+.3f}  <LD>={ld_baseline.mean():+.2f}")
    else:
        base_r = float("nan")
        print(f"  baseline KL-only mode (no LD/r)")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def resample(cells: list[tuple[int, int]]) -> dict:
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
            r_field = r
            dr = r - base_r
            ld_mean = float(ld.mean())
            ld_std = float(ld.std())
        else:
            r_field = None
            dr = None
            ld_mean = None
            ld_std = None
        return {
            "r_LD_label": r_field,
            "delta_r": dr,
            "ld_mean": ld_mean,
            "ld_std": ld_std,
            "kl_mean": float(kls.mean()),
            "kl_total": float(kls.sum()),
            "kl_max": float(kls.max()),
        }

    print("\n[runs]")
    out = {
        "task": args.task,
        "model": args.model_id,
        "short": args.short,
        "n_prompts": int(n),
        "label_key": label_key,
        "high_word": high_w,
        "low_word": low_w,
        "space_prefix": space_prefix,
        "high_id": int(high_id),
        "low_id": int(low_id),
        "baseline": {
            "r_LD_label": base_r if has_label else None,
            "ld_mean": float(ld_baseline.mean()) if has_label else None,
            "ld_std": float(ld_baseline.std()) if has_label else None,
        },
        "runs": [],
    }
    for label, cells in cell_sets:
        t1 = time.time()
        m = resample(cells)
        if has_label:
            print(f"  {label[:38]:<38s}  r={m['r_LD_label']:+.3f}  "
                   f"Δr={m['delta_r']:+.3f}  <LD>={m['ld_mean']:+.2f}  "
                   f"KL={m['kl_mean']:.3f}  ({time.time()-t1:.0f}s)")
        else:
            print(f"  {label[:38]:<38s}  KL_mean={m['kl_mean']:.3f}  "
                   f"KL_total={m['kl_total']:.1f}  KL_max={m['kl_max']:.3f}  "
                   f"({time.time()-t1:.0f}s)")
        out["runs"].append({"label": label, "cells": cells, **m})

    out_path = REPO / "results" / f"p2u_specificity_{args.short}_{args.task}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Plot
    import matplotlib.pyplot as plt

    labels = ["baseline"] + [r["label"] for r in out["runs"]]
    kls_total = [0.0] + [r["kl_total"] for r in out["runs"]]
    colors = ["black", "C3", "C3", "C0"] + ["C7"] * N_RANDOM_CONTROLS

    if has_label:
        rs = [base_r] + [r["r_LD_label"] for r in out["runs"]]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        bars = ax.bar(np.arange(len(labels)), rs, color=colors, edgecolor="black")
        ax.axhline(base_r, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.3)
        for bar, r in zip(bars, rs):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02 * np.sign(r or 1),
                     f"{r:+.2f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([l[:30] for l in labels], rotation=40, ha="right",
                            fontsize=8)
        ax.set_ylabel(f"r(LD, {label_key}) under joint resample")
        ax.set_title(f"{args.short} — {args.task} specificity "
                      f"(baseline r = {base_r:+.3f})")
        ax.grid(axis="y", alpha=0.3)
        kl_ax = axes[1]
    else:
        fig, kl_ax = plt.subplots(1, 1, figsize=(8, 5))

    bars = kl_ax.bar(np.arange(len(labels)), kls_total, color=colors,
                      edgecolor="black")
    for bar, kl in zip(bars, kls_total):
        kl_ax.text(bar.get_x() + bar.get_width() / 2, kl,
                    f"{kl:.0f}", ha="center", va="bottom", fontsize=8)
    kl_ax.set_xticks(np.arange(len(labels)))
    kl_ax.set_xticklabels([l[:30] for l in labels], rotation=40, ha="right",
                           fontsize=8)
    kl_ax.set_ylabel("Σ KL(baseline || resample) [nats]")
    kl_ax.set_title(f"Total output-distribution disruption — {args.task}")
    kl_ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2u_specificity_{args.short}_{args.task}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
