"""Phase 2O — phase-space diagram showing baseline / zero / resample on (r_x, r_z) plane.

Mirrors the p2d_phase_grid style. For the top-8 |cos|-ranked cells from
Phase 2M (the same cluster used in p2o_attention_modes.py), runs four
conditions and computes (r(LD, x), r(LD, z)) plus ⟨LD⟩, std(LD), partial r(z|x):

  - baseline (no intervention)
  - zero (zero each cell at last token)
  - resample (replace each cell's slice with random-permuted prompt's value)

Plots three markers on the standard phase plane (RELATIVISTIC top-left,
OBJECTIVE bottom-right, BIASED near origin) with arrows from baseline.

Output:
  results/p2o_phase_plot_<short>.json
  figures/p2o_phase_plot_<short>.png

Usage:
  python p2o_phase_plot.py --short gemma2-2b --top-n 8 --by-cos
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


def metrics(ld: np.ndarray, z: np.ndarray, x: np.ndarray) -> dict:
    r_z = float(np.corrcoef(ld, z)[0, 1])
    r_x = float(np.corrcoef(ld, x)[0, 1])
    r_zx = float(np.corrcoef(z, x)[0, 1])
    denom = math.sqrt(max(1e-12, (1 - r_x ** 2) * (1 - r_zx ** 2)))
    pc = (r_z - r_x * r_zx) / denom
    return {
        "r_LD_z": r_z, "r_LD_x": r_x, "partial_r_z_given_x": pc,
        "ld_mean": float(ld.mean()), "ld_std": float(ld.std(ddof=1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--by-cos", action="store_true",
                    help="rank by |cos| (default: |cos|x||Δ_a||)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cos_path = REPO / "results" / f"p2m_alignment_{args.short}.json"
    cos_data = json.loads(cos_path.read_text())
    cos = np.array(cos_data["cos_grid"])
    norms = np.array(cos_data["delta_norms"])
    layer_idxs = cos_data["layer_idxs"]
    head_idxs = cos_data["head_idxs"]
    feature = cos_data["feature"]
    k = cos_data["k"]
    model_id = cos_data["model"]

    score = np.abs(cos) if args.by_cos else np.abs(cos) * norms
    top_flat = np.argsort(score, axis=None)[::-1][:args.top_n]
    cells: list[tuple[int, int, float]] = []
    for kf in top_flat:
        li, hi = np.unravel_index(kf, cos.shape)
        cells.append((layer_idxs[li], head_idxs[hi], float(cos[li, hi])))
    print(f"selected top-{args.top_n} cells (by "
          f"{'|cos|' if args.by_cos else '|cos|×||Δ||'}):")
    for L, h, c in cells:
        print(f"  L{L:>2d} H{h}: cos={c:+.3f}")

    stim_path = REPO / "data" / "p2_shot_sweep" / f"{feature}_k{k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    n = len(rows)
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)
    x_arr = np.array([float(r["x"]) for r in rows], dtype=np.float32)

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

    layers_in_cells = sorted(set(L for L, _, _ in cells))
    pool: dict[int, np.ndarray] = {L: np.zeros((n, n_heads * head_dim), dtype=np.float32)
                                    for L in layers_in_cells}
    captured = {L: None for L in layers_in_cells}

    def make_capture(L):
        def hook(module, args_):
            captured[L] = args_[0].detach().float().cpu().numpy()
        return hook

    print(f"\n[pass 1] baseline + capture pre-o_proj at {layers_in_cells}")
    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(make_capture(L))
                for L in layers_in_cells]
    ld_base = np.zeros(n, dtype=np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            inp = tok(row["prompt"], return_tensors="pt").to(model.device)
            out = model(**inp, use_cache=False)
            logits = out.logits[0, -1].float()
            ld_base[i] = float(logits[high_id] - logits[low_id])
            for L in layers_in_cells:
                pool[L][i] = captured[L][0, -1]
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {i+1}/{n}  {(i+1)/max(1e-3, time.time()-t1):.1f} p/s",
                      flush=True)
    for h in handles:
        h.remove()

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    def fwd_LD(intervention: str) -> np.ndarray:
        handles = []
        state = {"i": 0}
        if intervention in ("zero", "resample"):
            for L in layers_in_cells:
                heads_at_L = tuple(h for (Lc, h, _) in cells if Lc == L)
                if intervention == "zero":
                    def make_hook(L_=L, heads_=heads_at_L):
                        def hook(module, args_):
                            x = args_[0].clone()
                            for h in heads_:
                                x[:, -1, h * head_dim:(h + 1) * head_dim] = 0
                            return (x,) + args_[1:]
                        return hook
                else:
                    def make_hook(L_=L, heads_=heads_at_L):
                        def hook(module, args_):
                            x = args_[0].clone()
                            j = perm[state["i"]]
                            for h in heads_:
                                src = pool[L_][j, h * head_dim:(h + 1) * head_dim]
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

    print(f"\n[zero]")
    t1 = time.time()
    ld_zero = fwd_LD("zero")
    print(f"  done ({time.time()-t1:.0f}s)")
    print(f"\n[resample]")
    t1 = time.time()
    ld_res = fwd_LD("resample")
    print(f"  done ({time.time()-t1:.0f}s)")

    base_m = metrics(ld_base, z_arr, x_arr)
    zero_m = metrics(ld_zero, z_arr, x_arr)
    res_m = metrics(ld_res, z_arr, x_arr)
    print(f"\n{'mode':>10s}  {'r(LD,x)':>10s}  {'r(LD,z)':>10s}  "
          f"{'pc(z|x)':>10s}  {'<LD>':>8s}  {'std(LD)':>8s}")
    for nm, m in [("baseline", base_m), ("zero", zero_m), ("resample", res_m)]:
        print(f"{nm:>10s}  {m['r_LD_x']:>+10.3f}  {m['r_LD_z']:>+10.3f}  "
              f"{m['partial_r_z_given_x']:>+10.3f}  {m['ld_mean']:>+8.2f}  "
              f"{m['ld_std']:>8.2f}")

    out_json = {
        "model": model_id, "short": args.short,
        "feature": feature, "k": k, "n_prompts": int(n),
        "n_cells": len(cells),
        "cells": [{"layer": int(L), "head": int(h), "cos": c} for L, h, c in cells],
        "selection": "|cos|" if args.by_cos else "|cos|x||Δ||",
        "baseline": base_m, "zero": zero_m, "resample": res_m,
    }
    out_path = REPO / "results" / f"p2o_phase_plot_{args.short}.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"\nwrote {out_path}")

    # Plot — single phase diagram with three markers + arrows
    fig, ax = plt.subplots(figsize=(8, 7))
    xlim = (-0.05, 1.05); ylim = (-0.05, 1.05)
    # Region shading
    ax.add_patch(Rectangle((-0.05, 0.5), 0.55, 0.6, alpha=0.07, color="C0", zorder=0))
    ax.add_patch(Rectangle((0.5, -0.05), 0.6, 0.55, alpha=0.07, color="C2", zorder=0))
    ax.add_patch(Rectangle((-0.05, -0.05), 0.55, 0.55, alpha=0.07, color="C3", zorder=0))
    ax.text(0.04, 0.97, "RELATIVISTIC", color="C0", fontsize=11,
             fontweight="bold", va="top")
    ax.text(0.97, 0.04, "OBJECTIVE", color="C2", fontsize=11,
             fontweight="bold", ha="right")
    ax.text(0.04, 0.04, "BIASED", color="C3", fontsize=11, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    ax.grid(alpha=0.25)

    def plot_marker(m: dict, name: str, color, marker: str, dy_offset=8):
        rx, rz = m["r_LD_x"], m["r_LD_z"]
        ax.scatter(rx, rz, s=300, color=color, marker=marker,
                    edgecolor="black", linewidth=1.2, zorder=5, label=name)
        txt = (f"{name}\n⟨LD⟩={m['ld_mean']:+.2f}\nstd={m['ld_std']:.2f}\n"
                f"pc(z|x)={m['partial_r_z_given_x']:+.2f}")
        ax.annotate(txt, (rx, rz),
                     xytext=(10, dy_offset), textcoords="offset points",
                     fontsize=9, color=color, fontweight="bold")

    plot_marker(base_m, "baseline", "tab:green", "o", dy_offset=6)
    plot_marker(zero_m, "zero", "tab:red", "X", dy_offset=-50)
    plot_marker(res_m, "resample", "tab:blue", "D", dy_offset=-50)

    bx, by = base_m["r_LD_x"], base_m["r_LD_z"]
    for m, color in [(zero_m, "tab:red"), (res_m, "tab:blue")]:
        ax.annotate("", xy=(m["r_LD_x"], m["r_LD_z"]), xytext=(bx, by),
                     arrowprops=dict(arrowstyle="->", color=color,
                                      alpha=0.6, lw=1.4),
                     zorder=3)

    ax.set_xlabel(r"$r$(LD, x)  →  objective", fontsize=12)
    ax.set_ylabel(r"$r$(LD, z)  →  relativistic", fontsize=12)
    ax.set_title(f"{args.short} — Phase 2O phase-space migration\n"
                  f"top-{len(cells)} |cos|-ranked cluster, "
                  f"{feature} k={k}, {n} prompts", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    out_png = REPO / "figures" / f"p2o_phase_plot_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
